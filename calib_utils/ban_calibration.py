"""Offline Ban calibration for Transformers MoE models."""

from __future__ import annotations

import argparse
import json
import os
import re
import types
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from calib_utils.data_utils import get_loaders
from calib_utils.transformers_collector import _extract_input_ids


def _default_model_name(model_path: str) -> str:
    return Path(model_path.rstrip("/")).name


def _resolve_output_prefix(args) -> Path:
    if args.output_prefix:
        return Path(args.output_prefix)
    model_name = args.model_name or _default_model_name(args.model_path)
    return Path(args.output_dir) / model_name / f"{args.dataset}_ban"


def _infer_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _layer_name_to_index(layer_name: str) -> int:
    match = re.search(r"layers\.(\d+)", layer_name)
    if match is None:
        raise ValueError(f"Cannot parse layer index from {layer_name}")
    return int(match.group(1))


class BanMoePatch(AbstractContextManager):
    """Patch Transformers MoE blocks for Ban calibration.

    ``pruned_layer_index=None`` keeps the model equivalent to the original
    top-k route and optionally records Top3/TopK ratios.  If a layer index is
    set, only that layer uses ``k_pruned`` active experts.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pruned_layer_index: Optional[int] = None,
        k_pruned: int = 3,
        collect_ratios: bool = False,
    ) -> None:
        self.model = model
        self.pruned_layer_index = pruned_layer_index
        self.k_pruned = k_pruned
        self.collect_ratios = collect_ratios
        self._original_forwards: list[tuple[torch.nn.Module, Any]] = []
        self._module_names = {module: name for name, module in model.named_modules()}
        self.layer_names: dict[int, str] = {}
        self.layer_topk: dict[int, int] = {}
        self.r_min = float("inf")
        self.r_max = float("-inf")

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
                "Ban calibration did not find supported MoE modules to patch "
                f"(model_type={model_type}). Supported Transformers modules "
                "currently include Qwen3MoeSparseMoeBlock, "
                "Qwen3NextSparseMoeBlock, GptOssMLP, and "
                "BailingMoeSparseMoeBlock.")

    def remove(self) -> None:
        for module, original_forward in self._original_forwards:
            module.forward = original_forward
        self._original_forwards.clear()

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
        patch = self
        layer_name = self._module_names.get(patched_module,
                                            f"moe_{id(patched_module)}")
        layer_index = _layer_name_to_index(layer_name)
        self.layer_names[layer_index] = layer_name
        self.layer_topk[layer_index] = int(patched_module.top_k)

        def forward(module_self, hidden_states: torch.Tensor):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flat_hidden_states = hidden_states.view(-1, hidden_dim)
            router_logits = module_self.gate(flat_hidden_states)
            routing_probs = F.softmax(router_logits, dim=1, dtype=torch.float)

            base_topk = int(module_self.top_k)
            raw_routing_weights, selected_experts = torch.topk(
                routing_probs,
                base_topk,
                dim=-1,
                sorted=True,
            )
            if patch.collect_ratios and base_topk >= 3:
                ratios = (raw_routing_weights[:, :3].sum(dim=-1) /
                          raw_routing_weights.sum(dim=-1).clamp_min(1e-20))
                patch.r_min = min(patch.r_min, float(ratios.min().item()))
                patch.r_max = max(patch.r_max, float(ratios.max().item()))

            active_topk = base_topk
            if patch.pruned_layer_index == layer_index:
                active_topk = min(max(1, patch.k_pruned), base_topk)
            raw_routing_weights = raw_routing_weights[:, :active_topk]
            selected_experts = selected_experts[:, :active_topk]
            final_routing_weights = raw_routing_weights
            if module_self.norm_topk_prob:
                final_routing_weights = (
                    final_routing_weights /
                    final_routing_weights.sum(dim=-1, keepdim=True))
            final_routing_weights = final_routing_weights.to(
                flat_hidden_states.dtype)

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
            return final_hidden_states, router_logits

        return forward

    def _make_qwen3_next_moe_forward(self, patched_module: torch.nn.Module):
        patch = self
        layer_name = self._module_names.get(patched_module,
                                            f"qwen3_next_moe_{id(patched_module)}")
        layer_index = _layer_name_to_index(layer_name)
        self.layer_names[layer_index] = layer_name
        self.layer_topk[layer_index] = int(patched_module.top_k)

        def forward(module_self, hidden_states: torch.Tensor):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flat_hidden_states = hidden_states.view(-1, hidden_dim)
            router_logits = module_self.gate(flat_hidden_states)
            routing_probs = F.softmax(router_logits, dim=1, dtype=torch.float)

            base_topk = int(module_self.top_k)
            raw_routing_weights, selected_experts = torch.topk(
                routing_probs,
                base_topk,
                dim=-1,
                sorted=True,
            )
            if patch.collect_ratios and base_topk >= 3:
                ratios = (raw_routing_weights[:, :3].sum(dim=-1) /
                          raw_routing_weights.sum(dim=-1).clamp_min(1e-20))
                patch.r_min = min(patch.r_min, float(ratios.min().item()))
                patch.r_max = max(patch.r_max, float(ratios.max().item()))

            active_topk = base_topk
            if patch.pruned_layer_index == layer_index:
                active_topk = min(max(1, patch.k_pruned), base_topk)
            raw_routing_weights = raw_routing_weights[:, :active_topk]
            selected_experts = selected_experts[:, :active_topk]
            final_routing_weights = raw_routing_weights
            if module_self.norm_topk_prob:
                final_routing_weights = (
                    final_routing_weights /
                    final_routing_weights.sum(dim=-1, keepdim=True))
            final_routing_weights = final_routing_weights.to(
                flat_hidden_states.dtype)

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
        patch = self
        layer_name = self._module_names.get(patched_module,
                                            f"bailing_moe_{id(patched_module)}")
        layer_index = _layer_name_to_index(layer_name)
        base_topk = int(patched_module.num_experts_per_tok)
        num_experts = len(patched_module.experts)
        self.layer_names[layer_index] = layer_name
        self.layer_topk[layer_index] = base_topk

        def forward(module_self, hidden_states: torch.Tensor):
            identity = hidden_states
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flat_hidden_states = hidden_states.reshape(-1, hidden_dim)

            router_logits = F.linear(flat_hidden_states,
                                     module_self.gate.weight,
                                     None)
            routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
            raw_routing_weights, selected_experts = torch.topk(
                routing_probs,
                base_topk,
                dim=-1,
                sorted=True,
            )
            if patch.collect_ratios and base_topk >= 3:
                ratios = (raw_routing_weights[:, :3].sum(dim=-1) /
                          raw_routing_weights.sum(dim=-1).clamp_min(1e-20))
                patch.r_min = min(patch.r_min, float(ratios.min().item()))
                patch.r_max = max(patch.r_max, float(ratios.max().item()))

            active_topk = base_topk
            if patch.pruned_layer_index == layer_index:
                active_topk = min(max(1, patch.k_pruned), base_topk)
            raw_routing_weights = raw_routing_weights[:, :active_topk]
            selected_experts = selected_experts[:, :active_topk]
            final_routing_weights = raw_routing_weights
            if module_self.gate.norm_topk_prob:
                final_routing_weights = (
                    final_routing_weights /
                    final_routing_weights.sum(dim=-1, keepdim=True))
            final_routing_weights = final_routing_weights.to(
                flat_hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=flat_hidden_states.dtype,
                device=flat_hidden_states.device,
            )
            expert_mask = F.one_hot(
                selected_experts,
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
                weighted_output = raw_output * final_routing_weights[top_x,
                                                                      idx, None]
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
                selected_experts.view(batch_size, sequence_length, -1),
            )

        return forward

    def _make_gpt_oss_mlp_forward(self, patched_module: torch.nn.Module):
        patch = self
        layer_name = self._module_names.get(patched_module,
                                            f"gpt_oss_mlp_{id(patched_module)}")
        layer_index = _layer_name_to_index(layer_name)
        base_topk = int(patched_module.router.top_k)
        num_experts = int(patched_module.experts.num_experts)
        self.layer_names[layer_index] = layer_name
        self.layer_topk[layer_index] = base_topk

        def forward(module_self, hidden_states: torch.Tensor):
            orig_shape = hidden_states.shape
            hidden_dim = orig_shape[-1]
            flat_hidden_states = hidden_states.reshape(-1, hidden_dim)

            router = module_self.router
            router_logits = F.linear(flat_hidden_states, router.weight,
                                     router.bias)
            raw_routing_logits, selected_experts = torch.topk(
                router_logits,
                base_topk,
                dim=-1,
                sorted=True,
            )
            raw_routing_weights = F.softmax(
                raw_routing_logits,
                dim=-1,
                dtype=raw_routing_logits.dtype,
            )
            if patch.collect_ratios and base_topk >= 3:
                ratios = (raw_routing_weights[:, :3].sum(dim=-1) /
                          raw_routing_weights.sum(dim=-1).clamp_min(1e-20))
                patch.r_min = min(patch.r_min, float(ratios.min().item()))
                patch.r_max = max(patch.r_max, float(ratios.max().item()))

            active_topk = base_topk
            if patch.pruned_layer_index == layer_index:
                active_topk = min(max(1, patch.k_pruned), base_topk)
            active_logits = raw_routing_logits[:, :active_topk]
            selected_experts = selected_experts[:, :active_topk]
            final_routing_weights = F.softmax(
                active_logits,
                dim=-1,
                dtype=active_logits.dtype,
            ).to(flat_hidden_states.dtype)

            experts = module_self.experts
            final_hidden_states = torch.zeros_like(flat_hidden_states)
            expert_mask = F.one_hot(
                selected_experts,
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
                weighted_output = raw_output * final_routing_weights[top_x,
                                                                      idx, None]
                final_hidden_states.index_add_(
                    0,
                    top_x,
                    weighted_output.to(flat_hidden_states.dtype),
                )

            router_scores = torch.zeros_like(router_logits).scatter_(
                1,
                selected_experts,
                final_routing_weights.to(router_logits.dtype),
            )
            return final_hidden_states.reshape(orig_shape), router_scores

        return forward


def _select_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "last":
        return logits[:, -1, :].reshape(-1, logits.size(-1))
    if mode == "all":
        return logits.reshape(-1, logits.size(-1))
    raise ValueError("--kl_positions must be last or all")


def _reference_topk(logits: torch.Tensor,
                    topn: int) -> tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor]:
    top_logits, top_indices = torch.topk(logits.float(), k=topn, dim=-1)
    p_log_probs = torch.log_softmax(top_logits, dim=-1)
    return top_indices, p_log_probs.exp(), p_log_probs


def _topk_kl(
    pruned_logits: torch.Tensor,
    top_indices: torch.Tensor,
    p_probs: torch.Tensor,
    p_log_probs: torch.Tensor,
) -> float:
    q_top_logits = torch.gather(pruned_logits.float(), dim=-1, index=top_indices)
    q_log_probs = torch.log_softmax(q_top_logits, dim=-1)
    kl = (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1)
    return float(kl.mean().item())


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


def run_ban_calibration(
    model: torch.nn.Module,
    trainloader: Any,
    k_pruned: int = 3,
    topn: int = 1000,
    kl_positions: str = "last",
    max_samples: int | None = None,
    device: torch.device | str | None = None,
    use_cache: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    model.eval()
    input_device = torch.device(device) if device is not None else \
        _infer_input_device(model)

    total = len(trainloader) if hasattr(trainloader, "__len__") else None
    if max_samples is not None:
        total = min(total, max_samples) if total is not None else max_samples

    sample_iterator = trainloader
    if show_progress:
        from tqdm.auto import tqdm

        sample_iterator = tqdm(
            trainloader,
            total=total,
            desc="Ban calibration samples",
            dynamic_ncols=True,
        )

    layer_kl_sum: dict[int, float] = {}
    layer_kl_count: dict[int, int] = {}
    layer_names: dict[int, str] = {}
    layer_topk: dict[int, int] = {}
    r_min = float("inf")
    r_max = float("-inf")
    num_samples = 0

    with torch.no_grad():
        for sample_idx, batch in enumerate(sample_iterator):
            if max_samples is not None and sample_idx >= max_samples:
                break
            input_ids = _extract_input_ids(batch).to(input_device)
            attention_mask = torch.ones_like(input_ids, device=input_device)

            with BanMoePatch(model, collect_ratios=True) as full_patch:
                full_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                )
            r_min = min(r_min, full_patch.r_min)
            r_max = max(r_max, full_patch.r_max)
            layer_names.update(full_patch.layer_names)
            layer_topk.update(full_patch.layer_topk)

            ref_logits = _select_logits(full_outputs.logits, kl_positions)
            top_indices, p_probs, p_log_probs = _reference_topk(
                ref_logits,
                topn=min(topn, ref_logits.size(-1)),
            )
            del full_outputs, ref_logits

            sorted_layers = sorted(layer_names)
            layer_iterator = sorted_layers
            if show_progress:
                from tqdm.auto import tqdm

                layer_iterator = tqdm(
                    sorted_layers,
                    desc=f"sample {sample_idx} Ban layers",
                    leave=False,
                    dynamic_ncols=True,
                )
            for layer_idx in layer_iterator:
                with BanMoePatch(
                    model,
                    pruned_layer_index=layer_idx,
                    k_pruned=k_pruned,
                    collect_ratios=False,
                ):
                    pruned_outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                    )
                q_logits = _select_logits(pruned_outputs.logits, kl_positions)
                value = _topk_kl(q_logits, top_indices, p_probs, p_log_probs)
                layer_kl_sum[layer_idx] = layer_kl_sum.get(layer_idx, 0.0) + value
                layer_kl_count[layer_idx] = layer_kl_count.get(layer_idx, 0) + 1
                del pruned_outputs, q_logits
            num_samples += 1

    if not layer_kl_sum:
        raise RuntimeError("No Ban layer KL values were collected.")
    max_layer = max(layer_kl_sum)
    raw_kl = torch.zeros(max_layer + 1, dtype=torch.float32)
    layer_name_list = ["" for _ in range(max_layer + 1)]
    topk_list = [0 for _ in range(max_layer + 1)]
    for layer_idx in range(max_layer + 1):
        count = max(1, layer_kl_count.get(layer_idx, 0))
        raw_kl[layer_idx] = layer_kl_sum.get(layer_idx, 0.0) / count
        layer_name_list[layer_idx] = layer_names.get(layer_idx, "")
        topk_list[layer_idx] = layer_topk.get(layer_idx, 0)

    w_min = raw_kl.min()
    w_max = raw_kl.max()
    denom = (w_max - w_min).clamp_min(1e-20)
    layer_sensitivity = (raw_kl - w_min) / denom
    if not (r_min < float("inf") and r_max > float("-inf")):
        r_min, r_max = 0.0, 1.0
    if r_max <= r_min:
        r_max = r_min + 1e-6

    return {
        "format": "ban_calibration_v1",
        "layer_names": layer_name_list,
        "layer_sensitivity": layer_sensitivity,
        "raw_layer_kl": raw_kl,
        "r_min": float(r_min),
        "r_max": float(r_max),
        "k_base": max(topk_list) if topk_list else 0,
        "k_pruned": int(k_pruned),
        "topn": int(topn),
        "kl_positions": kl_positions,
        "num_samples": int(num_samples),
    }


def plot_ban_layer_sensitivity(artifact: dict[str, Any],
                               output_path: str | Path) -> Path:
    """Save a simple bar chart for normalized Ban layer sensitivity."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = path.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_config_dir / "cache"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sensitivity = artifact["layer_sensitivity"].detach().float().cpu()
    layers = list(range(int(sensitivity.numel())))
    values = sensitivity.tolist()

    width = max(8.0, 0.28 * max(1, len(layers)))
    fig, ax = plt.subplots(figsize=(width, 4.5), dpi=160)
    ax.bar(layers, values, color="#3b82f6", edgecolor="#1e3a8a", linewidth=0.4)
    ax.set_title("Ban Layer Sensitivity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized sensitivity")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    if len(layers) <= 80:
        step = max(1, len(layers) // 24)
        ax.set_xticks(layers[::step])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_ban_artifact(artifact: dict[str, Any],
                      output_prefix: str | Path) -> tuple[Path, Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pt_path = prefix.with_suffix(".pt")
    json_path = prefix.with_suffix(".json")
    plot_path = prefix.with_name(f"{prefix.name}_layer_sensitivity.png")
    torch.save(artifact, pt_path)
    summary = {
        "format": artifact["format"],
        "layer_names": artifact["layer_names"],
        "layer_sensitivity": artifact["layer_sensitivity"].tolist(),
        "raw_layer_kl": artifact["raw_layer_kl"].tolist(),
        "r_min": artifact["r_min"],
        "r_max": artifact["r_max"],
        "k_base": artifact["k_base"],
        "k_pruned": artifact["k_pruned"],
        "topn": artifact["topn"],
        "kl_positions": artifact["kl_positions"],
        "num_samples": artifact["num_samples"],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # The plot is a convenience; a missing plotting backend must not discard hours of
    # calibration that is already safely on disk.
    try:
        plot_ban_layer_sensitivity(artifact, plot_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ban_calibration] layer-sensitivity plot skipped: {exc}")
        plot_path = None
    return pt_path, json_path, plot_path


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Ban calibration artifact.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", default="c4", choices=["wikitext2", "c4", "ptb"])
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_data_path", default=None)
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--output_dir", default="calib_utils/results")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--torch_dtype",
                        default="bfloat16",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--k_pruned", type=int, default=3)
    parser.add_argument("--topn", type=int, default=1000)
    parser.add_argument("--kl_positions",
                        choices=["last", "all"],
                        default="last")
    parser.add_argument("--local_files_only",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--progress",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[ban_calibration] dataset={args.dataset} nsamples={args.nsamples} "
          f"seqlen={args.seqlen} k_pruned={args.k_pruned} "
          f"kl_positions={args.kl_positions}",
          flush=True)
    trainloader = get_loaders(args.dataset,
                              nsamples=args.nsamples,
                              seed=args.seed,
                              seqlen=args.seqlen,
                              model=args.model_path,
                              local_data_path=args.local_data_path)
    print("[ban_calibration] loading model...", flush=True)
    model = _load_transformers_model(args)
    print("[ban_calibration] running layer sensitivity calibration...",
          flush=True)
    artifact = run_ban_calibration(
        model,
        trainloader,
        k_pruned=args.k_pruned,
        topn=args.topn,
        kl_positions=args.kl_positions,
        max_samples=args.max_samples,
        device=args.device,
        show_progress=args.progress,
    )
    output_prefix = _resolve_output_prefix(args)
    pt_path, json_path, plot_path = save_ban_artifact(artifact, output_prefix)
    print(f"Ban artifact saved: {pt_path}")
    print(f"Ban summary saved: {json_path}")
    print(f"Ban layer sensitivity plot saved: {plot_path}")


if __name__ == "__main__":
    main()
