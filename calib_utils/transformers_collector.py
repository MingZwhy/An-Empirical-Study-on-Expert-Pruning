"""MoE calibration collectors for Hugging Face Transformers models.

The collector is intentionally runtime-only: it monkey-patches MoE block
``forward`` methods on a loaded model and restores them afterwards.  This keeps
Transformers and the model files untouched while giving calibration code access
to router choices and per-expert outputs.
"""

from __future__ import annotations

import types
import json
from collections import defaultdict
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _zeros(num_experts: int) -> list[float]:
    return [0.0 for _ in range(num_experts)]


def _long_zeros(num_experts: int) -> list[int]:
    return [0 for _ in range(num_experts)]


class Qwen3MoeCalibrationCollector(AbstractContextManager):
    """Collect router and expert-output statistics from Qwen3-MoE models.

    Collected statistics are aggregated per MoE layer and per expert.  By
    default this stores scalar summaries only.  Set ``capture_tensor_samples``
    to keep a small CPU copy of selected router/expert tensors for inspection.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        capture_tensor_samples: bool = False,
        max_tensor_tokens: int = 16,
    ) -> None:
        self.model = model
        self.capture_tensor_samples = capture_tensor_samples
        self.max_tensor_tokens = max_tensor_tokens
        self._original_forwards: list[tuple[torch.nn.Module, Any]] = []
        self._module_names = {module: name for name, module in model.named_modules()}
        self.layer_stats: dict[str, dict[str, Any]] = {}
        self.tensor_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()
        return False

    def install(self) -> None:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeSparseMoeBlock,
        )

        if self._original_forwards:
            return

        for module in self.model.modules():
            if isinstance(module, Qwen3MoeSparseMoeBlock):
                original_forward = module.forward
                self._original_forwards.append((module, original_forward))
                module.forward = types.MethodType(
                    self._make_qwen3_moe_forward(module),
                    module,
                )

    def remove(self) -> None:
        for module, original_forward in self._original_forwards:
            module.forward = original_forward
        self._original_forwards.clear()

    def reset(self) -> None:
        self.layer_stats.clear()
        self.tensor_samples.clear()

    def _get_layer_stats(self, module: torch.nn.Module) -> dict[str, Any]:
        layer_name = self._module_names.get(module, f"moe_{id(module)}")
        if layer_name not in self.layer_stats:
            num_experts = int(module.num_experts)
            self.layer_stats[layer_name] = {
                "num_experts": num_experts,
                "top_k": int(module.top_k),
                "num_tokens": 0,
                "num_forwards": 0,
                "router_logits_sum": _zeros(num_experts),
                "router_logits_sq_sum": _zeros(num_experts),
                "router_prob_sum": _zeros(num_experts),
                "router_entropy_sum": 0.0,
                "selected_count": _long_zeros(num_experts),
                "selected_raw_weight_sum": _zeros(num_experts),
                "selected_final_weight_sum": _zeros(num_experts),
                "expert_input_l2_sum": _zeros(num_experts),
                "expert_raw_output_l2_sum": _zeros(num_experts),
                "expert_weighted_output_l2_sum": _zeros(num_experts),
                "expert_token_count": _long_zeros(num_experts),
                "final_output_l2_sum": 0.0,
            }
        return self.layer_stats[layer_name]

    @staticmethod
    def _add_list_values(dst: list[float], values: torch.Tensor) -> None:
        cpu_values = values.detach().float().cpu().tolist()
        for idx, value in enumerate(cpu_values):
            dst[idx] += float(value)

    @staticmethod
    def _add_count_values(dst: list[int], values: torch.Tensor) -> None:
        cpu_values = values.detach().cpu().tolist()
        for idx, value in enumerate(cpu_values):
            dst[idx] += int(value)

    def _record_router(
        self,
        module: torch.nn.Module,
        router_logits: torch.Tensor,
        routing_probs: torch.Tensor,
        selected_experts: torch.Tensor,
        raw_routing_weights: torch.Tensor,
        final_routing_weights: torch.Tensor,
    ) -> None:
        stats = self._get_layer_stats(module)
        num_tokens = int(router_logits.size(0))
        stats["num_tokens"] += num_tokens
        stats["num_forwards"] += 1
        self._add_list_values(stats["router_logits_sum"],
                              router_logits.detach().sum(dim=0))
        self._add_list_values(stats["router_logits_sq_sum"],
                              router_logits.detach().float().square().sum(dim=0))
        self._add_list_values(stats["router_prob_sum"],
                              routing_probs.detach().sum(dim=0))
        entropy = -(routing_probs.detach().float() *
                    routing_probs.detach().float().clamp_min(1e-20).log())
        stats["router_entropy_sum"] += float(entropy.sum(dim=-1).sum().item())

        selected_counts = torch.bincount(
            selected_experts.reshape(-1),
            minlength=int(module.num_experts),
        )
        self._add_count_values(stats["selected_count"], selected_counts)

        flat_experts = selected_experts.reshape(-1)
        flat_raw_weights = raw_routing_weights.reshape(-1).detach().float()
        flat_final_weights = final_routing_weights.reshape(-1).detach().float()
        raw_weight_sum = torch.zeros(
            int(module.num_experts),
            dtype=torch.float32,
            device=router_logits.device,
        )
        final_weight_sum = torch.zeros_like(raw_weight_sum)
        raw_weight_sum.scatter_add_(0, flat_experts, flat_raw_weights)
        final_weight_sum.scatter_add_(0, flat_experts, flat_final_weights)
        self._add_list_values(stats["selected_raw_weight_sum"], raw_weight_sum)
        self._add_list_values(stats["selected_final_weight_sum"],
                              final_weight_sum)

    def _record_expert(
        self,
        module: torch.nn.Module,
        expert_idx: int,
        current_state: torch.Tensor,
        raw_output: torch.Tensor,
        weighted_output: torch.Tensor,
    ) -> None:
        stats = self._get_layer_stats(module)
        token_count = int(current_state.size(0))
        stats["expert_token_count"][expert_idx] += token_count
        stats["expert_input_l2_sum"][expert_idx] += float(
            current_state.detach().float().norm(dim=-1).sum().item())
        stats["expert_raw_output_l2_sum"][expert_idx] += float(
            raw_output.detach().float().norm(dim=-1).sum().item())
        stats["expert_weighted_output_l2_sum"][expert_idx] += float(
            weighted_output.detach().float().norm(dim=-1).sum().item())

    def _maybe_capture_sample(
        self,
        module: torch.nn.Module,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        final_routing_weights: torch.Tensor,
        final_hidden_states: torch.Tensor,
    ) -> None:
        if not self.capture_tensor_samples:
            return
        layer_name = self._module_names.get(module, f"moe_{id(module)}")
        if self.max_tensor_tokens <= 0:
            return
        token_slice = slice(0, min(self.max_tensor_tokens,
                                  router_logits.size(0)))
        self.tensor_samples[layer_name].append({
            "router_logits": router_logits[token_slice].detach().float().cpu(),
            "selected_experts": selected_experts[token_slice].detach().cpu(),
            "routing_weights": final_routing_weights[token_slice].detach().float().cpu(),
            "final_hidden_states": final_hidden_states[token_slice].detach().float().cpu(),
        })

    def _make_qwen3_moe_forward(self, patched_module: torch.nn.Module):
        collector = self

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

            collector._record_router(
                patched_module,
                router_logits=router_logits,
                routing_probs=routing_probs,
                selected_experts=selected_experts,
                raw_routing_weights=raw_routing_weights,
                final_routing_weights=final_routing_weights,
            )

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
                weighted_output = (
                    raw_output * final_routing_weights[top_x, idx, None])
                collector._record_expert(
                    patched_module,
                    expert_idx=expert_idx,
                    current_state=current_state,
                    raw_output=raw_output,
                    weighted_output=weighted_output,
                )
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
            collector._get_layer_stats(patched_module)[
                "final_output_l2_sum"] += float(
                    final_hidden_states.detach().float().norm(dim=-1).sum().item())
            collector._maybe_capture_sample(
                patched_module,
                router_logits=router_logits,
                selected_experts=selected_experts,
                final_routing_weights=final_routing_weights,
                final_hidden_states=final_hidden_states.reshape(
                    batch_size * sequence_length,
                    hidden_dim,
                ),
            )
            return final_hidden_states, router_logits

        return forward

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for layer_name, stats in self.layer_stats.items():
            num_tokens = max(1, int(stats["num_tokens"]))
            num_experts = int(stats["num_experts"])
            layer_summary: dict[str, Any] = dict(stats)
            layer_summary["router_logits_mean"] = [
                value / num_tokens for value in stats["router_logits_sum"]
            ]
            layer_summary["router_prob_mean"] = [
                value / num_tokens for value in stats["router_prob_sum"]
            ]
            layer_summary["router_entropy_mean"] = (
                stats["router_entropy_sum"] / num_tokens)
            layer_summary["final_output_l2_mean"] = (
                stats["final_output_l2_sum"] / num_tokens)

            expert_token_count = stats["expert_token_count"]
            for field in (
                    "expert_input_l2_sum",
                    "expert_raw_output_l2_sum",
                    "expert_weighted_output_l2_sum",
            ):
                mean_field = field.replace("_sum", "_mean")
                layer_summary[mean_field] = []
                for expert_idx in range(num_experts):
                    denom = max(1, expert_token_count[expert_idx])
                    layer_summary[mean_field].append(
                        stats[field][expert_idx] / denom)
            result[layer_name] = layer_summary
        return result


def _infer_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _extract_input_ids(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        if "input_ids" not in batch:
            raise KeyError("Calibration batch dict must contain `input_ids`.")
        return batch["input_ids"]
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise TypeError(
        "Calibration batch must be a tensor, dict with input_ids, or tuple/list."
    )


def run_qwen3_moe_calibration(
    model: torch.nn.Module,
    trainloader: Any,
    max_samples: int | None = None,
    device: torch.device | str | None = None,
    capture_tensor_samples: bool = False,
    max_tensor_tokens: int = 16,
    use_cache: bool = False,
) -> Qwen3MoeCalibrationCollector:
    """Run ``trainloader`` through a Qwen3-MoE model and collect MoE stats."""
    model.eval()
    input_device = torch.device(device) if device is not None else \
        _infer_input_device(model)
    collector = Qwen3MoeCalibrationCollector(
        model,
        capture_tensor_samples=capture_tensor_samples,
        max_tensor_tokens=max_tensor_tokens,
    )
    with collector, torch.no_grad():
        for sample_idx, batch in enumerate(trainloader):
            if max_samples is not None and sample_idx >= max_samples:
                break
            input_ids = _extract_input_ids(batch).to(input_device)
            attention_mask = torch.ones_like(input_ids, device=input_device)
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
    return collector


def save_calibration_outputs(
    collector: Qwen3MoeCalibrationCollector,
    output_prefix: str | Path,
    save_tensor_samples: bool = True,
) -> tuple[Path, Path | None]:
    """Save scalar summary as JSON and optional tensor samples as ``.pt``."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(collector.summary(), f, indent=2, ensure_ascii=False)

    tensor_path = None
    if save_tensor_samples and collector.tensor_samples:
        tensor_path = prefix.with_suffix(".pt")
        torch.save(dict(collector.tensor_samples), tensor_path)
    return summary_path, tensor_path
