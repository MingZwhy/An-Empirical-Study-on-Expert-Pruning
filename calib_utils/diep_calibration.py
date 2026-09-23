"""Offline DiEP calibration for Transformers MoE models."""

from __future__ import annotations

import argparse
import json
import types
from collections import defaultdict
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from calib_utils.data_utils import get_loaders
from calib_utils.transformers_collector import _extract_input_ids


def _infer_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _layer_name_to_index(layer_name: str) -> int:
    import re

    match = re.search(r"layers\.(\d+)", layer_name)
    if match is None:
        raise ValueError(f"Cannot parse layer index from {layer_name}")
    return int(match.group(1))


def _default_model_name(model_path: str) -> str:
    return Path(model_path.rstrip("/")).name


def _resolve_output_prefix(args) -> Path:
    if args.output_prefix:
        return Path(args.output_prefix)
    model_name = args.model_name or _default_model_name(args.model_path)
    return Path(args.output_dir) / model_name / f"{args.dataset}_diep"


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Feature-space linear CKA, equivalent to linear-kernel Gram CKA."""
    if x.size(0) < 2 or y.size(0) < 2:
        return float("nan")
    n = min(x.size(0), y.size(0))
    x = x[:n].float()
    y = y[:n].float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    hsic_xy = xy.square().sum()
    hsic_xx = xx.square().sum()
    hsic_yy = yy.square().sum()
    denom = torch.sqrt(hsic_xx * hsic_yy).clamp_min(1e-20)
    return float((hsic_xy / denom).clamp(0, 1).item())


class DiEPCalibrationCollector(AbstractContextManager):
    """Collect expert output samples and routing ratios for DiEP."""

    def __init__(
        self,
        model: torch.nn.Module,
        max_outputs_per_expert: int = 2048,
        min_tokens_for_cka: int = 16,
        collect_marginal_values: bool = False,
        marginal_base_ks: tuple[int, ...] = (3, 5),
        marginal_forward_base_k: int | None = None,
        collect_exact_full_teacher: bool = False,
    ) -> None:
        self.model = model
        self.max_outputs_per_expert = max_outputs_per_expert
        self.min_tokens_for_cka = min_tokens_for_cka
        self.collect_marginal_values = collect_marginal_values
        self.marginal_base_ks = tuple(sorted(set(marginal_base_ks)))
        self.marginal_forward_base_k = marginal_forward_base_k
        self.collect_exact_full_teacher = collect_exact_full_teacher
        if any(base_k < 1 for base_k in self.marginal_base_ks):
            raise ValueError("marginal_base_ks must contain positive integers")
        if (marginal_forward_base_k is not None
                and marginal_forward_base_k not in self.marginal_base_ks):
            raise ValueError(
                "marginal_forward_base_k must be included in "
                "marginal_base_ks")
        self._original_forwards: list[tuple[torch.nn.Module, Any]] = []
        self._module_names = {module: name for name, module in model.named_modules()}
        self.layer_num_experts: dict[str, int] = {}
        self.layer_topk: dict[str, int] = {}
        self.expert_outputs: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list))
        self.expert_seen: dict[str, dict[int, int]] = defaultdict(
            lambda: defaultdict(int))
        self.gamma1_ratios: dict[str, list[float]] = defaultdict(list)
        self.marginal_value_stats: dict[
            str,
            dict[int, dict[str, torch.Tensor]],
        ] = defaultdict(dict)

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()
        return False

    def install(self) -> None:
        if self._original_forwards:
            return
        for module in self.model.modules():
            if self._is_qwen3_moe_block(module):
                original_forward = module.forward
                self._original_forwards.append((module, original_forward))
                module.forward = types.MethodType(
                    self._make_qwen3_moe_forward(module),
                    module,
                )
            elif self._is_qwen3_next_moe_block(module):
                original_forward = module.forward
                self._original_forwards.append((module, original_forward))
                module.forward = types.MethodType(
                    self._make_qwen3_next_moe_forward(module),
                    module,
                )
            elif self._is_gpt_oss_mlp(module):
                original_forward = module.forward
                self._original_forwards.append((module, original_forward))
                module.forward = types.MethodType(
                    self._make_gpt_oss_mlp_forward(module),
                    module,
                )
            elif self._is_bailing_moe_block(module):
                original_forward = module.forward
                self._original_forwards.append((module, original_forward))
                module.forward = types.MethodType(
                    self._make_bailing_moe_forward(module),
                    module,
                )
        if not self._original_forwards:
            model_type = getattr(getattr(self.model, "config", None),
                                 "model_type", "unknown")
            raise RuntimeError(
                "DiEP calibration did not find supported MoE modules to "
                f"patch (model_type={model_type}). Supported Transformers "
                "modules currently include Qwen3MoeSparseMoeBlock, "
                "Qwen3NextSparseMoeBlock, GptOssMLP, and "
                "BailingMoeSparseMoeBlock.")

    def remove(self) -> None:
        for module, original_forward in self._original_forwards:
            module.forward = original_forward
        self._original_forwards.clear()

    def _record_outputs(self, layer_name: str, expert_idx: int,
                        outputs: torch.Tensor) -> None:
        self.expert_seen[layer_name][expert_idx] += int(outputs.size(0))
        bucket = self.expert_outputs[layer_name][expert_idx]
        remaining = self.max_outputs_per_expert - len(bucket)
        if remaining <= 0:
            return
        outputs = outputs[:remaining].detach().float().cpu()
        bucket.extend(row.clone() for row in outputs)

    def _record_marginal_values(
        self,
        layer_name: str,
        base_k: int,
        expert_ids: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        num_experts: int,
    ) -> None:
        """Aggregate token-level marginal values by the added expert ID."""
        bucket = self.marginal_value_stats[layer_name].get(base_k)
        if bucket is None:
            bucket = {
                "count": torch.zeros(num_experts, dtype=torch.long),
            }
            for metric_name in metrics:
                bucket[f"{metric_name}_sum"] = torch.zeros(
                    num_experts, dtype=torch.float64)
                bucket[f"{metric_name}_sumsq"] = torch.zeros(
                    num_experts, dtype=torch.float64)
            self.marginal_value_stats[layer_name][base_k] = bucket

        expert_ids = expert_ids.detach().long()
        counts = torch.bincount(expert_ids, minlength=num_experts).cpu()
        bucket["count"] += counts
        for metric_name, values in metrics.items():
            values = values.detach().float()
            sums = torch.zeros(
                num_experts, dtype=torch.float32, device=values.device)
            sums.scatter_add_(0, expert_ids, values)
            sumsq = torch.zeros_like(sums)
            sumsq.scatter_add_(0, expert_ids, values.square())
            bucket[f"{metric_name}_sum"] += sums.double().cpu()
            bucket[f"{metric_name}_sumsq"] += sumsq.double().cpu()

    def _record_qwen3_marginal_values(
        self,
        layer_name: str,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        full_output: torch.Tensor,
        prefix_sums: dict[int, torch.Tensor],
        next_sums: dict[int, torch.Tensor],
        num_experts: int,
        exact_prefix_outputs: dict[int, torch.Tensor] | None = None,
        exact_expanded_outputs: dict[int, torch.Tensor] | None = None,
        exact_full_output: torch.Tensor | None = None,
    ) -> None:
        del exact_prefix_outputs, exact_expanded_outputs, exact_full_output
        full_output_float = full_output.float()
        full_norm = full_output_float.norm(dim=-1).clamp_min(1e-12)
        for base_k in prefix_sums:
            prefix_mass = routing_weights[:, :base_k].float().sum(dim=-1)
            next_weight = routing_weights[:, base_k].float()
            expanded_mass = (prefix_mass + next_weight).clamp_min(1e-12)
            prefix_output = (
                prefix_sums[base_k].float() /
                prefix_mass.clamp_min(1e-12).unsqueeze(-1)
            )
            expanded_output = (
                (prefix_sums[base_k] + next_sums[base_k]).float() /
                expanded_mass.unsqueeze(-1)
            )
            delta = expanded_output - prefix_output
            next_output = (
                next_sums[base_k].float() /
                next_weight.clamp_min(1e-12).unsqueeze(-1)
            )
            output_disagreement = next_output - prefix_output
            prefix_norm = prefix_output.norm(dim=-1)
            expanded_norm = expanded_output.norm(dim=-1)
            next_output_norm = next_output.norm(dim=-1)
            metrics = {
                "relative_delta_norm": delta.norm(dim=-1) / full_norm,
                "symmetric_delta_distance": (
                    delta.norm(dim=-1) /
                    (prefix_norm + expanded_norm).clamp_min(1e-12)
                ),
                "cosine_change": (
                    1.0 - F.cosine_similarity(
                        prefix_output, expanded_output, dim=-1, eps=1e-12)
                ).clamp_min(0.0),
                "relative_output_disagreement": (
                    output_disagreement.norm(dim=-1) / full_norm
                ),
                "symmetric_output_distance": (
                    output_disagreement.norm(dim=-1) /
                    (prefix_norm + next_output_norm).clamp_min(1e-12)
                ),
                "output_cosine_disagreement": (
                    1.0 - F.cosine_similarity(
                        prefix_output, next_output, dim=-1, eps=1e-12)
                ).clamp_min(0.0),
                "next_weight_fraction": next_weight / expanded_mass,
                "next_weight": next_weight,
                "prefix_mass": prefix_mass,
            }
            self._record_marginal_values(
                layer_name=layer_name,
                base_k=base_k,
                expert_ids=selected_experts[:, base_k],
                metrics=metrics,
                num_experts=num_experts,
            )

    @staticmethod
    def _is_qwen3_moe_block(module: torch.nn.Module) -> bool:
        return (module.__class__.__name__ == "Qwen3MoeSparseMoeBlock"
                and hasattr(module, "gate")
                and hasattr(module, "experts")
                and hasattr(module, "top_k"))

    @staticmethod
    def _is_qwen3_next_moe_block(module: torch.nn.Module) -> bool:
        return (module.__class__.__name__ == "Qwen3NextSparseMoeBlock"
                and hasattr(module, "gate")
                and hasattr(module, "experts")
                and hasattr(module, "top_k"))

    @staticmethod
    def _is_gpt_oss_mlp(module: torch.nn.Module) -> bool:
        return (module.__class__.__name__ == "GptOssMLP"
                and hasattr(module, "router")
                and hasattr(module, "experts"))

    @staticmethod
    def _is_bailing_moe_block(module: torch.nn.Module) -> bool:
        return (module.__class__.__name__ == "BailingMoeSparseMoeBlock"
                and hasattr(module, "gate")
                and hasattr(module, "experts")
                and hasattr(module, "num_experts_per_tok"))

    def _make_qwen3_moe_forward(self, patched_module: torch.nn.Module):
        collector = self
        layer_name = self._module_names.get(patched_module,
                                            f"moe_{id(patched_module)}")

        def forward(module_self, hidden_states: torch.Tensor):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flat_hidden_states = hidden_states.view(-1, hidden_dim)
            router_logits = module_self.gate(flat_hidden_states)

            routing_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
            raw_routing_weights, selected_experts = torch.topk(
                routing_probs,
                module_self.top_k,
                dim=-1,
            )
            final_routing_weights_float = raw_routing_weights
            if module_self.norm_topk_prob:
                final_routing_weights_float = (
                    final_routing_weights_float /
                    final_routing_weights_float.sum(dim=-1, keepdim=True))
            final_routing_weights = final_routing_weights_float.to(
                flat_hidden_states.dtype)

            collector.layer_num_experts[layer_name] = int(module_self.num_experts)
            collector.layer_topk[layer_name] = int(module_self.top_k)
            if raw_routing_weights.size(1) >= 2:
                ratios = (raw_routing_weights[:, 1] /
                          raw_routing_weights[:, 0].clamp_min(1e-20))
                collector.gamma1_ratios[layer_name].extend(
                    ratios.detach().float().cpu().tolist())

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=flat_hidden_states.dtype,
                device=flat_hidden_states.device,
            )
            active_base_ks = (
                tuple(
                    base_k for base_k in collector.marginal_base_ks
                    if base_k < module_self.top_k
                )
                if collector.collect_marginal_values else ()
            )
            marginal_prefix_sums = {
                base_k: torch.zeros_like(final_hidden_states)
                for base_k in active_base_ks
            }
            marginal_next_sums = {
                base_k: torch.zeros_like(final_hidden_states)
                for base_k in active_base_ks
            }
            exact_prefix_outputs = {
                base_k: torch.zeros_like(final_hidden_states)
                for base_k in active_base_ks
            }
            exact_expanded_outputs = {
                base_k: torch.zeros_like(final_hidden_states)
                for base_k in active_base_ks
            }
            exact_prefix_weights = {}
            exact_expanded_weights = {}
            for base_k in active_base_ks:
                prefix_weights = final_routing_weights_float[:, :base_k]
                prefix_weights = prefix_weights / prefix_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                expanded_weights = final_routing_weights_float[:, :base_k + 1]
                expanded_weights = expanded_weights / expanded_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                exact_prefix_weights[base_k] = prefix_weights.to(
                    flat_hidden_states.dtype
                )
                exact_expanded_weights[base_k] = expanded_weights.to(
                    flat_hidden_states.dtype
                )
            execution_top_k = module_self.top_k
            if collector.marginal_forward_base_k is not None:
                execution_top_k = max(active_base_ks) + 1
            execution_selected_experts = selected_experts[:, :execution_top_k]
            expert_mask = F.one_hot(
                execution_selected_experts,
                num_classes=module_self.num_experts,
            ).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)),
                                       0).nonzero()
            for expert_idx_tensor in expert_hit:
                expert_idx = int(expert_idx_tensor.item())
                expert_layer = module_self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                current_state = flat_hidden_states[None,
                                                   top_x].reshape(-1,
                                                                  hidden_dim)
                raw_output = expert_layer(current_state)
                collector._record_outputs(layer_name, expert_idx, raw_output)
                weighted_output = (
                    raw_output * final_routing_weights[top_x, idx, None])
                final_hidden_states.index_add_(
                    0,
                    top_x,
                    weighted_output.to(flat_hidden_states.dtype),
                )
                for base_k in active_base_ks:
                    prefix_mask = idx < base_k
                    marginal_prefix_sums[base_k].index_add_(
                        0,
                        top_x[prefix_mask],
                        weighted_output[prefix_mask].to(
                            flat_hidden_states.dtype),
                    )
                    next_mask = idx == base_k
                    marginal_next_sums[base_k].index_add_(
                        0,
                        top_x[next_mask],
                        weighted_output[next_mask].to(
                            flat_hidden_states.dtype),
                    )
                    prefix_mask = idx < base_k
                    exact_prefix_outputs[base_k].index_add_(
                        0,
                        top_x[prefix_mask],
                        (
                            raw_output[prefix_mask]
                            * exact_prefix_weights[base_k][
                                top_x[prefix_mask], idx[prefix_mask], None
                            ]
                        ).to(flat_hidden_states.dtype),
                    )
                    expanded_mask = idx <= base_k
                    exact_expanded_outputs[base_k].index_add_(
                        0,
                        top_x[expanded_mask],
                        (
                            raw_output[expanded_mask]
                            * exact_expanded_weights[base_k][
                                top_x[expanded_mask], idx[expanded_mask], None
                            ]
                        ).to(flat_hidden_states.dtype),
                    )

            exact_full_output = None
            if collector.collect_exact_full_teacher and active_base_ks:
                if execution_top_k == module_self.top_k:
                    exact_full_output = final_hidden_states
                else:
                    exact_full_output = torch.zeros_like(final_hidden_states)
                    full_expert_mask = F.one_hot(
                        selected_experts,
                        num_classes=module_self.num_experts,
                    ).permute(2, 1, 0)
                    full_expert_hit = torch.greater(
                        full_expert_mask.sum(dim=(-1, -2)), 0
                    ).nonzero()
                    for expert_idx_tensor in full_expert_hit:
                        expert_idx = int(expert_idx_tensor.item())
                        expert_layer = module_self.experts[expert_idx]
                        idx, top_x = torch.where(
                            full_expert_mask[expert_idx].squeeze(0)
                        )
                        current_state = flat_hidden_states[None, top_x].reshape(
                            -1, hidden_dim
                        )
                        raw_output = expert_layer(current_state)
                        weighted_output = (
                            raw_output
                            * final_routing_weights[top_x, idx, None]
                        )
                        exact_full_output.index_add_(
                            0,
                            top_x,
                            weighted_output.to(flat_hidden_states.dtype),
                        )

            forward_hidden_states = final_hidden_states
            forward_base_k = collector.marginal_forward_base_k
            if forward_base_k is not None:
                if forward_base_k not in marginal_prefix_sums:
                    raise ValueError(
                        f"marginal_forward_base_k={forward_base_k} must be "
                        f"smaller than model top_k={module_self.top_k}")
                forward_hidden_states = exact_prefix_outputs[forward_base_k]

            if active_base_ks:
                collector._record_qwen3_marginal_values(
                    layer_name=layer_name,
                    selected_experts=selected_experts,
                    routing_weights=final_routing_weights_float,
                    full_output=forward_hidden_states,
                    prefix_sums=marginal_prefix_sums,
                    next_sums=marginal_next_sums,
                    num_experts=int(module_self.num_experts),
                    exact_prefix_outputs=exact_prefix_outputs,
                    exact_expanded_outputs=exact_expanded_outputs,
                    exact_full_output=exact_full_output,
                )

            forward_hidden_states = forward_hidden_states.reshape(
                batch_size,
                sequence_length,
                hidden_dim,
            )
            return forward_hidden_states, router_logits

        return forward

    def _make_qwen3_next_moe_forward(self, patched_module: torch.nn.Module):
        collector = self
        layer_name = self._module_names.get(patched_module,
                                            f"qwen3_next_moe_{id(patched_module)}")

        def forward(module_self, hidden_states: torch.Tensor):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flat_hidden_states = hidden_states.view(-1, hidden_dim)
            router_logits = module_self.gate(flat_hidden_states)

            routing_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
            raw_routing_weights, selected_experts = torch.topk(
                routing_probs,
                module_self.top_k,
                dim=-1,
            )
            final_routing_weights = raw_routing_weights
            if module_self.norm_topk_prob:
                final_routing_weights = (
                    final_routing_weights /
                    final_routing_weights.sum(dim=-1, keepdim=True))
            final_routing_weights = final_routing_weights.to(
                flat_hidden_states.dtype)

            collector.layer_num_experts[layer_name] = int(module_self.num_experts)
            collector.layer_topk[layer_name] = int(module_self.top_k)
            if raw_routing_weights.size(1) >= 2:
                ratios = (raw_routing_weights[:, 1] /
                          raw_routing_weights[:, 0].clamp_min(1e-20))
                collector.gamma1_ratios[layer_name].extend(
                    ratios.detach().float().cpu().tolist())

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=flat_hidden_states.dtype,
                device=flat_hidden_states.device,
            )
            expert_mask = F.one_hot(
                selected_experts,
                num_classes=module_self.num_experts,
            ).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)),
                                       0).nonzero()
            for expert_idx_tensor in expert_hit:
                expert_idx = int(expert_idx_tensor.item())
                expert_layer = module_self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                current_state = flat_hidden_states[None,
                                                   top_x].reshape(-1,
                                                                  hidden_dim)
                raw_output = expert_layer(current_state)
                collector._record_outputs(layer_name, expert_idx, raw_output)
                weighted_output = (
                    raw_output * final_routing_weights[top_x, idx, None])
                final_hidden_states.index_add_(
                    0,
                    top_x,
                    weighted_output.to(flat_hidden_states.dtype),
                )

            shared_expert_output = module_self.shared_expert(flat_hidden_states)
            shared_expert_output = (
                F.sigmoid(module_self.shared_expert_gate(flat_hidden_states)) *
                shared_expert_output)
            final_hidden_states = final_hidden_states + shared_expert_output
            final_hidden_states = final_hidden_states.reshape(
                batch_size,
                sequence_length,
                hidden_dim,
            )
            return final_hidden_states, router_logits

        return forward

    def _make_bailing_moe_forward(self, patched_module: torch.nn.Module):
        collector = self
        layer_name = self._module_names.get(patched_module,
                                            f"bailing_moe_{id(patched_module)}")

        def forward(module_self, hidden_states: torch.Tensor):
            identity = hidden_states
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            topk_idx, topk_weight, router_logits = module_self.gate(
                hidden_states,
                sort=True,
            )
            flat_hidden_states = hidden_states.reshape(-1, hidden_dim)

            num_experts = len(module_self.experts)
            collector.layer_num_experts[layer_name] = int(num_experts)
            collector.layer_topk[layer_name] = int(module_self.num_experts_per_tok)
            if topk_weight.size(1) >= 2:
                ratios = (topk_weight[:, 1] /
                          topk_weight[:, 0].clamp_min(1e-20))
                collector.gamma1_ratios[layer_name].extend(
                    ratios.detach().float().cpu().tolist())

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=flat_hidden_states.dtype,
                device=flat_hidden_states.device,
            )
            expert_mask = F.one_hot(
                topk_idx,
                num_classes=num_experts,
            ).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)),
                                       0).nonzero()
            for expert_idx_tensor in expert_hit:
                expert_idx = int(expert_idx_tensor.item())
                expert_layer = module_self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                current_state = flat_hidden_states[top_x]
                raw_output = expert_layer(current_state)
                collector._record_outputs(layer_name, expert_idx, raw_output)
                weighted_output = raw_output * topk_weight[top_x, idx, None]
                final_hidden_states.index_add_(
                    0,
                    top_x,
                    weighted_output.to(flat_hidden_states.dtype),
                )

            final_hidden_states = final_hidden_states.reshape(
                batch_size,
                sequence_length,
                hidden_dim,
            )
            if getattr(module_self.config, "num_shared_experts", None) is not None:
                final_hidden_states = (
                    final_hidden_states + module_self.shared_experts(identity))
            return final_hidden_states, (
                router_logits.view(batch_size, sequence_length, -1),
                topk_idx.view(batch_size, sequence_length, -1),
            )

        return forward

    def _make_gpt_oss_mlp_forward(self, patched_module: torch.nn.Module):
        collector = self
        layer_name = self._module_names.get(patched_module,
                                            f"gpt_oss_mlp_{id(patched_module)}")

        def forward(module_self, hidden_states: torch.Tensor):
            orig_shape = hidden_states.shape
            hidden_dim = orig_shape[-1]
            flat_hidden_states = hidden_states.reshape(-1, hidden_dim)

            router_scores, router_indices = module_self.router(hidden_states)
            experts = module_self.experts
            num_experts = int(experts.num_experts)
            top_k = int(module_self.router.top_k)

            collector.layer_num_experts[layer_name] = num_experts
            collector.layer_topk[layer_name] = top_k

            selected_weights = router_scores.gather(
                1,
                router_indices.clamp_min(0),
            )
            if selected_weights.size(1) >= 2:
                ratios = (selected_weights[:, 1] /
                          selected_weights[:, 0].clamp_min(1e-20))
                collector.gamma1_ratios[layer_name].extend(
                    ratios.detach().float().cpu().tolist())

            final_hidden_states = torch.zeros_like(flat_hidden_states)
            expert_mask = F.one_hot(
                router_indices,
                num_classes=num_experts,
            ).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)),
                                       0).nonzero()
            for expert_idx_tensor in expert_hit:
                expert_idx = int(expert_idx_tensor.item())
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                current_state = flat_hidden_states[top_x]

                gate_up = (
                    current_state @ experts.gate_up_proj[expert_idx] +
                    experts.gate_up_proj_bias[expert_idx])
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=experts.limit)
                up = up.clamp(min=-experts.limit, max=experts.limit)
                glu = gate * torch.sigmoid(gate * experts.alpha)
                raw_output = (
                    (up + 1) * glu
                ) @ experts.down_proj[expert_idx] + experts.down_proj_bias[
                    expert_idx]

                collector._record_outputs(layer_name, expert_idx, raw_output)
                weighted_output = raw_output * router_scores[top_x,
                                                             expert_idx, None]
                final_hidden_states.index_add_(
                    0,
                    top_x,
                    weighted_output.to(flat_hidden_states.dtype),
                )

            return final_hidden_states.reshape(orig_shape), router_scores

        return forward

    def compute_artifact(self,
                         cka_device: torch.device | str | None = None,
                         show_progress: bool = True,
                         ) -> dict[str, Any]:
        if not self.layer_num_experts:
            raise RuntimeError("No MoE layer statistics were collected.")

        if cka_device is None:
            cka_device = "cuda" if torch.cuda.is_available() else "cpu"
        cka_device = torch.device(cka_device)

        max_layer = max(_layer_name_to_index(name)
                        for name in self.layer_num_experts)
        num_layers = max_layer + 1
        num_experts = max(self.layer_num_experts.values())
        sim_matrix = torch.eye(num_experts).repeat(num_layers, 1, 1)
        mean_sim = torch.ones(num_layers)
        gamma_1 = torch.ones(num_layers)
        layer_names = [""] * num_layers
        token_counts = torch.zeros(num_layers, num_experts, dtype=torch.long)

        sorted_layer_names = sorted(self.layer_num_experts,
                                    key=_layer_name_to_index)
        layer_iterator = sorted_layer_names
        tqdm = None
        if show_progress:
            from tqdm.auto import tqdm

            layer_iterator = tqdm(
                sorted_layer_names,
                desc="DiEP CKA layers",
                dynamic_ncols=True,
            )

        for layer_name in layer_iterator:
            layer_idx = _layer_name_to_index(layer_name)
            layer_names[layer_idx] = layer_name
            cur_num_experts = self.layer_num_experts[layer_name]
            layer_sim = torch.full((num_experts, num_experts), float("nan"))
            layer_sim.fill_diagonal_(1.0)

            outputs = {}
            for expert_idx in range(cur_num_experts):
                bucket = self.expert_outputs[layer_name][expert_idx]
                token_counts[layer_idx, expert_idx] = len(bucket)
                if len(bucket) >= self.min_tokens_for_cka:
                    outputs[expert_idx] = torch.stack(
                        bucket,
                        dim=0,
                    ).to(cka_device, non_blocking=True)

            valid_values = []
            pair_total = cur_num_experts * (cur_num_experts - 1) // 2
            pair_pbar = None
            if tqdm is not None:
                pair_pbar = tqdm(
                    total=pair_total,
                    desc=f"Layer {layer_idx} CKA pairs",
                    leave=False,
                    dynamic_ncols=True,
                )
            for i in range(cur_num_experts):
                for j in range(i + 1, cur_num_experts):
                    if i in outputs and j in outputs:
                        value = linear_cka(outputs[i], outputs[j])
                    else:
                        value = float("nan")
                    if value == value:
                        valid_values.append(value)
                        layer_sim[i, j] = value
                        layer_sim[j, i] = value
                    if pair_pbar is not None:
                        pair_pbar.update(1)
            if pair_pbar is not None:
                pair_pbar.close()

            layer_mean = (sum(valid_values) / len(valid_values)
                          if valid_values else 1.0)
            for i in range(cur_num_experts):
                for j in range(cur_num_experts):
                    if i != j and torch.isnan(layer_sim[i, j]):
                        layer_sim[i, j] = layer_mean
            layer_sim = torch.nan_to_num(layer_sim, nan=layer_mean)

            ratios = self.gamma1_ratios[layer_name]
            gamma_1[layer_idx] = (float(torch.tensor(ratios).median().item())
                                  if ratios else 1.0)
            sim_matrix[layer_idx] = layer_sim
            mean_sim[layer_idx] = layer_mean
            if hasattr(layer_iterator, "set_postfix"):
                layer_iterator.set_postfix(
                    device=str(cka_device),
                    valid_experts=len(outputs),
                    layer_mean=f"{layer_mean:.4f}",
                )
            del outputs
            if cka_device.type == "cuda":
                torch.cuda.empty_cache()

        return {
            "format": "diep_calibration_v1",
            "layer_names": layer_names,
            "sim_matrix": sim_matrix,
            "mean_sim": mean_sim,
            "gamma_1": gamma_1,
            "token_counts": token_counts,
            "min_tokens_for_cka": self.min_tokens_for_cka,
            "max_outputs_per_expert": self.max_outputs_per_expert,
        }

    def compute_marginal_value_artifact(self) -> dict[str, Any]:
        """Build compact layer/expert priors for adding expert k+1 to top-k."""
        if not self.marginal_value_stats:
            raise RuntimeError(
                "No marginal expert values were collected. This calibration "
                "currently supports Qwen3MoeSparseMoeBlock and requires "
                "collect_marginal_values=True."
            )

        max_layer = max(
            _layer_name_to_index(name) for name in self.marginal_value_stats)
        num_layers = max_layer + 1
        num_experts = max(self.layer_num_experts.values())
        base_ks = sorted({
            base_k
            for layer_stats in self.marginal_value_stats.values()
            for base_k in layer_stats
        })
        first_bucket = next(
            bucket
            for layer_stats in self.marginal_value_stats.values()
            for bucket in layer_stats.values()
        )
        metric_names = sorted(
            key[:-4]
            for key in first_bucket
            if key.endswith("_sum") and not key.endswith("_sumsq")
        )
        layer_names = [""] * num_layers
        count = torch.zeros(
            len(base_ks), num_layers, num_experts, dtype=torch.long)
        means = {
            metric_name: torch.full(
                (len(base_ks), num_layers, num_experts),
                float("nan"),
                dtype=torch.float32,
            )
            for metric_name in metric_names
        }
        stds = {
            metric_name: torch.full_like(means[metric_name], float("nan"))
            for metric_name in metric_names
        }
        base_k_to_index = {base_k: idx for idx, base_k in enumerate(base_ks)}

        for layer_name, layer_stats in self.marginal_value_stats.items():
            layer_idx = _layer_name_to_index(layer_name)
            layer_names[layer_idx] = layer_name
            for base_k, bucket in layer_stats.items():
                base_idx = base_k_to_index[base_k]
                cur_count = bucket["count"]
                cur_num_experts = cur_count.numel()
                count[base_idx, layer_idx, :cur_num_experts] = cur_count
                valid = cur_count > 0
                denominator = cur_count[valid].double()
                for metric_name in metric_names:
                    cur_mean = (
                        bucket[f"{metric_name}_sum"][valid] / denominator)
                    cur_second_moment = (
                        bucket[f"{metric_name}_sumsq"][valid] / denominator)
                    cur_variance = (
                        cur_second_moment - cur_mean.square()).clamp_min(0.0)
                    means[metric_name][
                        base_idx, layer_idx, :cur_num_experts
                    ][valid] = cur_mean.float()
                    stds[metric_name][
                        base_idx, layer_idx, :cur_num_experts
                    ][valid] = cur_variance.sqrt().float()

        return {
            "format": "moe_marginal_value_calibration_v1",
            "model_type": getattr(
                getattr(self.model, "config", None), "model_type", "unknown"),
            "layer_names": layer_names,
            "base_ks": base_ks,
            "metric_names": metric_names,
            "count": count,
            "mean": means,
            "std": stds,
            "definition": {
                "base_output": "renormalized weighted output of ranks [0, k)",
                "expanded_output": (
                    "renormalized weighted output of ranks [0, k+1)"
                ),
                "relative_delta_norm": (
                    "norm(expanded_output-base_output)/norm(full_topk_output)"
                ),
                "relative_output_disagreement": (
                    "norm(added_expert_output-base_output)/norm(full_topk_output)"
                ),
                "symmetric_delta_distance": (
                    "norm(expanded_output-base_output)/(norm(expanded_output)"
                    "+norm(base_output))"
                ),
                "symmetric_output_distance": (
                    "norm(added_expert_output-base_output)/(norm(added_expert_"
                    "output)+norm(base_output))"
                ),
                "next_weight_fraction": "w[k]/sum(w[:k+1])",
            },
        }


def run_diep_calibration(
    model: torch.nn.Module,
    trainloader: Any,
    max_samples: int | None = None,
    device: torch.device | str | None = None,
    max_outputs_per_expert: int = 2048,
    min_tokens_for_cka: int = 16,
    collect_marginal_values: bool = False,
    marginal_base_ks: tuple[int, ...] = (3, 5),
    use_cache: bool = False,
    show_progress: bool = True,
) -> DiEPCalibrationCollector:
    model.eval()
    input_device = torch.device(device) if device is not None else \
        _infer_input_device(model)
    collector = DiEPCalibrationCollector(
        model,
        max_outputs_per_expert=max_outputs_per_expert,
        min_tokens_for_cka=min_tokens_for_cka,
        collect_marginal_values=collect_marginal_values,
        marginal_base_ks=marginal_base_ks,
    )
    total = len(trainloader) if hasattr(trainloader, "__len__") else None
    if max_samples is not None:
        total = min(total, max_samples) if total is not None else max_samples
    iterator = trainloader
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(trainloader,
                        total=total,
                        desc="DiEP calibration",
                        dynamic_ncols=True)

    with collector, torch.no_grad():
        for sample_idx, batch in enumerate(iterator):
            if max_samples is not None and sample_idx >= max_samples:
                break
            input_ids = _extract_input_ids(batch).to(input_device)
            attention_mask = torch.ones_like(input_ids, device=input_device)
            model(input_ids=input_ids,
                  attention_mask=attention_mask,
                  use_cache=use_cache)
    return collector


def save_diep_artifact(collector: DiEPCalibrationCollector,
                       output_prefix: str | Path,
                       cka_device: torch.device | str | None = None,
                       show_progress: bool = True,
                       ) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    artifact = collector.compute_artifact(
        cka_device=cka_device,
        show_progress=show_progress,
    )
    pt_path = prefix.with_suffix(".pt")
    json_path = prefix.with_suffix(".json")
    torch.save(artifact, pt_path)
    summary = {
        "format": artifact["format"],
        "layer_names": artifact["layer_names"],
        "mean_sim": artifact["mean_sim"].tolist(),
        "gamma_1": artifact["gamma_1"].tolist(),
        "token_counts_min": artifact["token_counts"].min(dim=1).values.tolist(),
        "token_counts_max": artifact["token_counts"].max(dim=1).values.tolist(),
        "min_tokens_for_cka": artifact["min_tokens_for_cka"],
        "max_outputs_per_expert": artifact["max_outputs_per_expert"],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return pt_path, json_path


def save_marginal_value_artifact(
    collector: DiEPCalibrationCollector,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    artifact = collector.compute_marginal_value_artifact()
    pt_path = prefix.with_suffix(".pt")
    json_path = prefix.with_suffix(".json")
    torch.save(artifact, pt_path)

    count = artifact["count"]
    summary: dict[str, Any] = {
        "format": artifact["format"],
        "model_type": artifact["model_type"],
        "layer_names": artifact["layer_names"],
        "base_ks": artifact["base_ks"],
        "metric_names": artifact["metric_names"],
        "definition": artifact["definition"],
        "by_base_k": {},
    }
    for base_idx, base_k in enumerate(artifact["base_ks"]):
        cur_count = count[base_idx]
        layer_token_count = cur_count.sum(dim=-1)
        base_summary: dict[str, Any] = {
            "layer_token_count": layer_token_count.tolist(),
            "layer_valid_experts": (cur_count > 0).sum(dim=-1).tolist(),
            "layer_weighted_mean": {},
        }
        for metric_name in artifact["metric_names"]:
            values = artifact["mean"][metric_name][base_idx]
            weighted_sum = torch.nan_to_num(values) * cur_count
            layer_mean = weighted_sum.sum(dim=-1) / layer_token_count.clamp_min(1)
            base_summary["layer_weighted_mean"][metric_name] = (
                layer_mean.tolist())
        summary["by_base_k"][str(base_k)] = base_summary

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return pt_path, json_path


def _load_transformers_model(args):
    from transformers import AutoModelForCausalLM

    dtype = "auto" if args.torch_dtype == "auto" else getattr(
        torch, args.torch_dtype)
    return AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Collect DiEP calibration artifacts.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", default="c4", choices=["wikitext2", "c4", "ptb"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_data_path", default=None)
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Full output prefix. If omitted, uses "
        "<output_dir>/<model_name>/<dataset>_diep.",
    )
    parser.add_argument("--output_dir", default="calib_utils/results")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--torch_dtype",
                        default="bfloat16",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max_outputs_per_expert", type=int, default=2048)
    parser.add_argument("--min_tokens_for_cka", type=int, default=16)
    parser.add_argument(
        "--collect_marginal_values",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Collect Qwen3 layer/expert priors for the output change caused "
            "by adding expert k+1 to a top-k route."
        ),
    )
    parser.add_argument(
        "--marginal_base_ks",
        default="3,5",
        help="Comma-separated base expert counts k (default: 3,5).",
    )
    parser.add_argument(
        "--skip_cka",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save only the marginal-value artifact and skip CKA.",
    )
    parser.add_argument(
        "--cka_device",
        default=None,
        help="Device used for CKA matrix multiplications. Defaults to cuda "
        "when available, otherwise cpu. Use cpu to force the old behavior.",
    )
    parser.add_argument("--local_files_only",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--progress",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        marginal_base_ks = tuple(
            int(value.strip())
            for value in args.marginal_base_ks.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError(
            "--marginal_base_ks must be a comma-separated list of integers"
        ) from exc
    if not marginal_base_ks:
        raise ValueError("--marginal_base_ks cannot be empty")
    if args.skip_cka and not args.collect_marginal_values:
        raise ValueError("--skip_cka requires --collect_marginal_values")
    print(f"[diep_calibration] dataset={args.dataset} "
          f"nsamples={args.nsamples} seqlen={args.seqlen}",
          flush=True)
    trainloader = get_loaders(args.dataset,
                              nsamples=args.nsamples,
                              seed=args.seed,
                              seqlen=args.seqlen,
                              model=args.model_path,
                              local_data_path=args.local_data_path)
    print("[diep_calibration] loading model...", flush=True)
    model = _load_transformers_model(args)
    print("[diep_calibration] collecting expert outputs...", flush=True)
    collector = run_diep_calibration(
        model,
        trainloader,
        max_samples=args.max_samples,
        device=args.device,
        max_outputs_per_expert=(
            0 if args.skip_cka else args.max_outputs_per_expert),
        min_tokens_for_cka=args.min_tokens_for_cka,
        collect_marginal_values=args.collect_marginal_values,
        marginal_base_ks=marginal_base_ks,
        show_progress=args.progress,
    )
    output_prefix = _resolve_output_prefix(args)
    if args.skip_cka:
        print(
            "[diep_calibration] saving marginal-value artifact...",
            flush=True,
        )
        pt_path, json_path = save_marginal_value_artifact(
            collector, output_prefix)
        print(f"Marginal-value artifact saved: {pt_path}")
        print(f"Marginal-value summary saved: {json_path}")
    else:
        print(
            "[diep_calibration] computing CKA and saving artifact...",
            flush=True,
        )
        pt_path, json_path = save_diep_artifact(
            collector,
            output_prefix,
            cka_device=args.cka_device,
            show_progress=args.progress,
        )
        print(f"DiEP artifact saved: {pt_path}")
        print(f"DiEP summary saved: {json_path}")
        if args.collect_marginal_values:
            marginal_prefix = output_prefix.with_name(
                f"{output_prefix.name}_marginal_values")
            marginal_pt, marginal_json = save_marginal_value_artifact(
                collector, marginal_prefix)
            print(f"Marginal-value artifact saved: {marginal_pt}")
            print(f"Marginal-value summary saved: {marginal_json}")


if __name__ == "__main__":
    main()
