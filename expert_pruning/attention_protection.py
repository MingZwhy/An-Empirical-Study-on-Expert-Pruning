"""Attention-sink protection helpers for vLLM Qwen3-MoE.

The module observes online Q/K tensors inside ``Qwen3MoeAttention.forward`` and
builds protected-token masks for MC-MoE-style routing.  Expert pruning itself
is still handled by ``expert_pruning.routing``.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from typing import Any, Optional

import torch

_ENABLED = False
_DEBUG_MAX_PRINTS = 8
_MAX_QUERY_TOKENS = 2048
_MAX_KEY_TOKENS = 2048
_DEBUG_TOPK = 4
_PROTECTION_RATIO = 0.02
_PRINT_COUNT = 0
_SKIP_DEPTH = 0
_LAST_SUMMARY: Optional[dict[str, Any]] = None
_POSITION_CONTEXT_ENABLED = False
_LAST_POSITION_SUMMARY: Optional[dict[str, Any]] = None


def configure_attention_sink_probe(
    enabled: bool = False,
    debug_max_prints: int = 8,
    max_query_tokens: int = 2048,
    max_key_tokens: int = 2048,
    protect_topk: int = 4,
    protection_ratio: float = 0.02,
) -> None:
    global _ENABLED, _DEBUG_MAX_PRINTS, _MAX_QUERY_TOKENS, _MAX_KEY_TOKENS
    global _DEBUG_TOPK, _PROTECTION_RATIO, _PRINT_COUNT, _LAST_SUMMARY

    if debug_max_prints < 0:
        raise ValueError("--attention_sink_probe_max_prints must be >= 0")
    if max_query_tokens < 1:
        raise ValueError("--attention_sink_probe_max_query_tokens must be >= 1")
    if max_key_tokens < 1:
        raise ValueError("--attention_sink_probe_max_key_tokens must be >= 1")
    if protect_topk < 1:
        raise ValueError("--attention_sink_probe_topk must be >= 1")
    if not 0 < protection_ratio <= 1:
        raise ValueError("--mc_moe_protection_ratio must be in (0, 1]")

    _ENABLED = enabled
    _DEBUG_MAX_PRINTS = debug_max_prints
    _MAX_QUERY_TOKENS = max_query_tokens
    _MAX_KEY_TOKENS = max_key_tokens
    _DEBUG_TOPK = protect_topk
    _PROTECTION_RATIO = protection_ratio
    _PRINT_COUNT = 0
    _LAST_SUMMARY = None


def configure_attention_position_context(enabled: bool = False) -> None:
    """Enable lightweight position tracking for prefill-only routing methods."""
    global _POSITION_CONTEXT_ENABLED, _LAST_POSITION_SUMMARY
    _POSITION_CONTEXT_ENABLED = enabled
    _LAST_POSITION_SUMMARY = None


@contextmanager
def skip_attention_sink_probe():
    global _SKIP_DEPTH
    _SKIP_DEPTH += 1
    try:
        yield
    finally:
        _SKIP_DEPTH -= 1


def _layer_name_to_index(layer_name: str | None) -> Optional[int]:
    if layer_name is None:
        return None
    match = re.search(r"layers\.(\d+)", layer_name)
    if match is None:
        return None
    return int(match.group(1))


def _is_cuda_graph_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is None or not hasattr(compiler, "is_compiling"):
        return False
    return bool(compiler.is_compiling())


def _should_probe() -> bool:
    return (_ENABLED and _SKIP_DEPTH == 0 and not _is_cuda_graph_capturing()
            and not _is_torch_compiling())


def _should_record_positions() -> bool:
    return (_POSITION_CONTEXT_ENABLED and _SKIP_DEPTH == 0
            and not _is_cuda_graph_capturing() and not _is_torch_compiling())


def _sequence_segments(positions: torch.Tensor,
                       num_tokens: int) -> list[tuple[int, int]]:
    if num_tokens <= 0:
        return []
    pos_cpu = positions[:num_tokens].detach().to("cpu")
    segments = []
    start = 0
    previous = int(pos_cpu[0].item())
    for idx in range(1, num_tokens):
        current = int(pos_cpu[idx].item())
        if current != previous + 1:
            segments.append((start, idx))
            start = idx
        previous = current
    segments.append((start, num_tokens))
    return segments


def record_attention_positions(layer_name: str | None,
                               positions: torch.Tensor) -> None:
    """Record token segments for the following MoE layer without Q/K probing."""
    global _LAST_POSITION_SUMMARY

    if not _should_record_positions() or positions.numel() == 0:
        return

    with torch.no_grad():
        num_tokens = int(positions.numel())
        pos = positions[:num_tokens].detach()
        _LAST_POSITION_SUMMARY = {
            "layer_name": layer_name,
            "layer_index": _layer_name_to_index(layer_name),
            "num_tokens": num_tokens,
            "positions": pos,
            "segments": _sequence_segments(pos, num_tokens),
        }


def get_moe_token_segments(
        layer_name: str | None,
        num_tokens: int) -> Optional[list[tuple[int, int]]]:
    """Return position-derived token segments for the following MoE layer."""
    if not _should_record_positions() or _LAST_POSITION_SUMMARY is None:
        return None

    moe_layer_index = _layer_name_to_index(layer_name)
    if moe_layer_index is None or moe_layer_index != _LAST_POSITION_SUMMARY[
            "layer_index"]:
        return None
    if num_tokens != _LAST_POSITION_SUMMARY["num_tokens"]:
        return None
    return _LAST_POSITION_SUMMARY["segments"]


def observe_attention_qk(
    layer_name: str | None,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scaling: float,
) -> None:
    """Store a Q/K-derived attention-received score for current layer tokens."""
    global _LAST_SUMMARY, _PRINT_COUNT

    if not _should_probe():
        return
    if q.ndim != 2 or k.ndim != 2 or q.size(0) == 0 or k.size(0) == 0:
        return
    if q.size(-1) != num_heads * head_dim:
        return
    if k.size(-1) != num_kv_heads * head_dim:
        return

    with torch.no_grad():
        num_tokens = min(q.size(0), k.size(0), positions.numel())
        if num_tokens <= 0:
            return

        q_view = q[:num_tokens].view(num_tokens, num_heads, head_dim).float()
        k_view = k[:num_tokens].view(num_tokens, num_kv_heads, head_dim).float()
        pos = positions[:num_tokens].detach()
        segments = _sequence_segments(pos, num_tokens)
        attention_score = torch.zeros(
            num_tokens, dtype=torch.float32, device=q.device)

        max_query_count = 0
        max_key_count = 0
        for start, end in segments:
            segment_len = end - start
            if segment_len <= 0:
                continue
            query_count = min(_MAX_QUERY_TOKENS, segment_len)
            key_count = min(_MAX_KEY_TOKENS, segment_len)
            query_start = end - query_count
            key_end = start + key_count

            # Online Q/K-derived attention map for the current flattened chunk.
            # This does not inspect historical KV-cache entries during decode.
            q_probe = q_view[query_start:end]
            k_probe = k_view[start:key_end]
            if num_heads != num_kv_heads:
                if num_heads % num_kv_heads != 0:
                    continue
                repeat = num_heads // num_kv_heads
                k_probe = k_probe.repeat_interleave(repeat, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q_probe, k_probe) * scaling

            query_pos = pos[query_start:end].view(-1, 1)
            key_pos = pos[start:key_end].view(1, -1)
            scores = scores.masked_fill(key_pos > query_pos, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attention_score[start:key_end] = attn.nan_to_num().mean(dim=(0, 1))
            max_query_count = max(max_query_count, query_count)
            max_key_count = max(max_key_count, key_count)

        _LAST_SUMMARY = {
            "layer_name": layer_name,
            "layer_index": _layer_name_to_index(layer_name),
            "num_tokens": num_tokens,
            "query_count": max_query_count,
            "key_count": max_key_count,
            "positions": pos,
            "attention_score": attention_score,
            "segments": segments,
        }

        if _PRINT_COUNT >= _DEBUG_MAX_PRINTS:
            return

        debug_topk = min(_DEBUG_TOPK, num_tokens)
        top_scores, top_local_ids = torch.topk(
            attention_score, k=debug_topk, sorted=True)
        print(
            "[expert_pruning][AttentionProbe] "
            f"layer={layer_name} tokens={num_tokens} "
            f"segments={len(segments)} "
            f"max_query_tokens={max_query_count} "
            f"max_key_tokens={max_key_count} "
            f"q_heads={num_heads} kv_heads={num_kv_heads} "
            f"top_attention_indices={top_local_ids.detach().cpu().tolist()} "
            "top_attention_positions="
            f"{pos[top_local_ids].detach().cpu().tolist()} "
            f"attention_scores={top_scores.detach().cpu().tolist()}",
            flush=True,
        )
        _PRINT_COUNT += 1


def get_moe_protected_mask(
        layer_name: str | None,
        hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
    """Compute top-ratio protected tokens for the following MoE layer."""
    global _PRINT_COUNT

    if not _should_probe() or _LAST_SUMMARY is None:
        return None

    moe_layer_index = _layer_name_to_index(layer_name)
    if moe_layer_index is None or moe_layer_index != _LAST_SUMMARY[
            "layer_index"]:
        return None
    if hidden_states.size(0) != _LAST_SUMMARY["num_tokens"]:
        return None

    with torch.no_grad():
        attention_score = _LAST_SUMMARY["attention_score"].to(
            device=hidden_states.device)
        feature_magnitude = hidden_states.detach().float().abs().sum(dim=-1)
        importance = feature_magnitude * attention_score
        protected_mask = torch.zeros(
            hidden_states.size(0),
            dtype=torch.bool,
            device=hidden_states.device,
        )

        protected_count = 0
        for start, end in _LAST_SUMMARY["segments"]:
            segment_len = end - start
            # vLLM decode commonly arrives as many single-token segments. The
            # paper's top-ratio protection is defined over a sequence attention
            # map, so applying ceil(ratio * 1) here would protect every decode
            # token and inflate the protection ratio far beyond tau_h.
            if segment_len <= 1:
                continue
            k = max(1, math.ceil(_PROTECTION_RATIO * segment_len))
            k = min(k, segment_len)
            _, local_ids = torch.topk(importance[start:end], k=k, sorted=True)
            protected_mask[start + local_ids] = True
            protected_count += int(k)

    if _PRINT_COUNT >= _DEBUG_MAX_PRINTS:
        return protected_mask

    protected_ids = torch.nonzero(protected_mask, as_tuple=False).view(-1)
    sample_count = min(8, protected_ids.numel())
    sample_ids = protected_ids[:sample_count]
    positions = _LAST_SUMMARY["positions"].to(device=hidden_states.device)
    print(
        "[expert_pruning][AttentionProbe->MoE] "
        f"layer={layer_name} protected_count={protected_count} "
        f"protection_ratio={_PROTECTION_RATIO} tokens={hidden_states.size(0)} "
        f"sample_indices={sample_ids.detach().cpu().tolist()} "
        f"sample_positions={positions[sample_ids].detach().cpu().tolist()}",
        flush=True,
    )
    _PRINT_COUNT += 1
    return protected_mask
