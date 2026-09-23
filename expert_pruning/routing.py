"""Expert routing implementations owned by this repository.

The first implementation intentionally matches vLLM's default top-k routing
semantics closely, but uses plain torch ops so pruning policies can be added
here without editing ``3rdparty/vllm``.
"""

from __future__ import annotations

import atexit
import json
import os
import re
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import torch

_ROUTING_METHOD = "none"
_NAEE_BETA = 0.3
_NAEE_K_MIN = 2
_DYNAMIC_ROUTING_THRESHOLD = 0.8
_DYNAMIC_ROUTING_SCORE_SOURCE = "raw"
_DYNAMIC_ROUTING_NEXT_RANK_PENALTY = 1.0
_LAYERWISE_DYNAMIC_BASE_THRESHOLD = 0.55
_LAYERWISE_DYNAMIC_LAYER_ALPHA = 0.15
_LAYERWISE_DYNAMIC_NUM_LAYERS = 48
_LAYERWISE_DYNAMIC_K_MIN = 1
_BUDGET_DYNAMIC_EASY_K = 4
_BUDGET_DYNAMIC_BASE_K = 5
_BUDGET_DYNAMIC_HARD_K = 6
_BUDGET_DYNAMIC_EASY_THRESHOLD = 0.90
_BUDGET_DYNAMIC_HARD_THRESHOLD = 0.75
_BUDGET_DYNAMIC_SCORE_TOPN = 3
_LAYER_BUDGET_DYNAMIC_EASY_LAYER_ALPHA = 0.08
_LAYER_BUDGET_DYNAMIC_HARD_LAYER_ALPHA = 0.08
_LAYER_BUDGET_DYNAMIC_NUM_LAYERS = 48
_BAND_LAYER_BUDGET_MIDDLE_START = 19
_BAND_LAYER_BUDGET_MIDDLE_END = 28
_BAND_LAYER_BUDGET_PER_LAYER_K: tuple[int, ...] = ()
_BAND_LAYER_BUDGET_EASY_LAYERS: tuple[int, ...] = ()
_BAND_LAYER_BUDGET_HARD_LAYERS: tuple[int, ...] = ()
_BAND_LAYER_BUDGET_TOKEN_GATE_RATIO = -1.0
_BAND_LAYER_BUDGET_EXTRA_MIDDLE_START = -1
_BAND_LAYER_BUDGET_EXTRA_MIDDLE_END = -1
_BAND_LAYER_BUDGET_MIDDLE_PHASE = "all"
_BAND_LAYER_BUDGET_EASY_START = 0
_BAND_LAYER_BUDGET_EASY_END = 47
_BAND_LAYER_BUDGET_EASY_PHASE = "all"
_BAND_LAYER_BUDGET_HARD_PHASE = "all"
_BAND_LAYER_BUDGET_EASY_TAIL_THRESHOLD = 1.0
_BAND_LAYER_BUDGET_HARD_MIN_RANK_WEIGHT = 0.0
_BAND_LAYER_BUDGET_HARD_MAX_RANK_WEIGHT = 1.0
_BAND_LAYER_BUDGET_HARD_DECODE_MIN_RANK_WEIGHT = -1.0
_BAND_LAYER_BUDGET_MIXED_RESCUE_MIN_RANK_WEIGHT = -1.0
_BAND_LAYER_BUDGET_MIXED_RESCUE_MAX_CONCENTRATION = -1.0
_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_TOKENS = -1
_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_TOKENS = -1
_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_DENSITY = -1.0
_BAND_LAYER_BUDGET_HARD_PREFILL_MIN_DENSITY = -1.0
_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_HARD_RATIO = -1.0
_BAND_LAYER_BUDGET_HARD_PREFILL_SEGMENT_CAP_SCORE = "concentration"
_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_SEGMENT_HARD_RATIO = -1.0
_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_DENSITY = -1.0
_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_SEQ_LEN = -1
_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_IDS: tuple[int, ...] = ()
_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_NGRAMS: tuple[tuple[int, ...], ...] = ()
_BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_SEGMENT_HARD_RATIO = -1.0
_BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_IDS: tuple[int, ...] = ()
_BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_NGRAMS: tuple[tuple[int, ...], ...] = ()
_BAND_LAYER_BUDGET_HARD_PREFILL_EXCLUDE_TOKEN_IDS: tuple[int, ...] = ()
_BAND_LAYER_BUDGET_HARD_PREFILL_MIN_RELATIVE_POS = -1.0
_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_RELATIVE_POS = -1.0
_BAND_LAYER_BUDGET_HARD_PREFILL_START = -1
_BAND_LAYER_BUDGET_HARD_PREFILL_END = -1
_BAND_LAYER_BUDGET_HARD_DECODE_MIN_SEQ_LEN = -1
_BAND_LAYER_BUDGET_HARD_DECODE_MAX_SEQ_LEN = -1
_BAND_LAYER_BUDGET_HARD_DECODE_SEQ_LEN_SCOPE = "all"
_BAND_LAYER_BUDGET_HARD_DECODE_MIN_OFFSET = -1
_BAND_LAYER_BUDGET_HARD_DECODE_MAX_OFFSET = -1
_BAND_LAYER_BUDGET_HARD_DECODE_START = -1
_BAND_LAYER_BUDGET_HARD_DECODE_END = -1
_BAND_LAYER_BUDGET_HARD_DECODE_TOKEN_IDS: tuple[int, ...] = ()
_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_START = -1
_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_END = -1
_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_START = -1
_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_END = -1
_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_ALPHA = 0.0
_BAND_LAYER_BUDGET_HARD_LAYER_SENSITIVITY_ALPHA = 0.0
_BAND_LAYER_BUDGET_LATE_CONCENTRATION_START = -1
_BAND_LAYER_BUDGET_LATE_CONCENTRATION_THRESHOLD = -1.0
_BAND_LAYER_BUDGET_LATE_RESCUE_START = -1
_BAND_LAYER_BUDGET_LATE_RESCUE_CONCENTRATION_THRESHOLD = -1.0
_BAND_LAYER_BUDGET_LATE_RESCUE_MIXED_MAX = -1.0
_BAND_LAYER_BUDGET_PROMPT_PROFILE1_TOKEN_NGRAMS: tuple[tuple[int, ...], ...] = ()
_BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_THRESHOLD = -1.0
_BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_MIN_RANK_WEIGHT = -1.0
_BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_START = -1
_BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_END = -1
_BAND_LAYER_BUDGET_PROMPT_PROFILE1_PREFILL_SEGMENT_HARD_RATIO = -1.0
_BAND_LAYER_BUDGET_PROMPT_PROFILE2_TOKEN_NGRAMS: tuple[tuple[int, ...], ...] = ()
_BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_THRESHOLD = -1.0
_BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_MIN_RANK_WEIGHT = -1.0
_BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_START = -1
_BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_END = -1
_BAND_LAYER_BUDGET_PROMPT_PROFILE2_PREFILL_SEGMENT_HARD_RATIO = -1.0
_DIEP_PRUNING_MODE = "independent"
_EAC_ALPHA = 0.5
_MC_MOE_PROTECTION_RATIO = 0.02
_BIASED_RENORM_KEEP_TOPN = 3
_ROUTER_VALUE_ARTIFACT_PATH: Optional[str] = None
_ROUTER_VALUE_BASE_K = 3
_ROUTER_VALUE_QUOTA_RATIO = 0.10
_ROUTER_VALUE_SELECTION_MODE = "threshold"
_ROUTER_VALUE_SWAP_DIRECTION = "both"
_ROUTER_VALUE_ARTIFACT_FORMAT = ""
_ROUTER_VALUE_INTERCEPT = 0.0
_ROUTER_VALUE_COEFFICIENTS: tuple[float, ...] = ()
_ROUTER_VALUE_LOWER_COEFFICIENTS: tuple[float, ...] = ()
_ROUTER_VALUE_LAYER_COEFFICIENTS: tuple[float, ...] = ()
_ROUTER_VALUE_LAYER_THRESHOLDS: tuple[float, ...] = ()
_ROUTER_VALUE_DEMOTION_THRESHOLDS: tuple[float, ...] = ()
_ROUTER_VALUE_DEVICE_CACHE: dict[str, torch.Tensor] = {}
_ROUTER_VALUE_LOWER_DEVICE_CACHE: dict[str, torch.Tensor] = {}
_DIEP_ARTIFACT_PATH: Optional[str] = None
_DIEP_SIM_MATRIX: Optional[torch.Tensor] = None
_DIEP_MEAN_SIM: Optional[torch.Tensor] = None
_DIEP_GAMMA1: Optional[torch.Tensor] = None
_DIEP_DEVICE_CACHE: dict[
    str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_DIEP_USE_GAMMA1 = False
# gamma_2 is raised to this power before it scales the threshold. 1.0 is the plain
# generalisation of the paper's rule; 0.0 removes the similarity signal entirely and leaves
# NAEE. Anything in between damps how far gamma_2 moves the bar, which is the axis that matters,
# because gamma_2's spread is what costs DiEP its accuracy rather than its tail.
_DIEP_GAMMA_ALPHA = 1.0
# Upper bound on the threshold as a fraction of the top-1 gate weight. Has to be below 1 to do
# anything: at exactly 1 the threshold equals w_e0 and every lower-ranked expert still fails
# `w_ei < threshold`, so the decision is the same as for any larger value.
_DIEP_THRESHOLD_CAP: Optional[float] = None
_BAN_ARTIFACT_PATH: Optional[str] = None
_BAN_LAYER_SENSITIVITY: Optional[torch.Tensor] = None
_BAN_R_MIN = 0.0
_BAN_R_MAX = 1.0
_BAN_LAMBDA = 0.7
_BAN_K_MIN = 3
_BAN_DEVICE_CACHE: dict[str, torch.Tensor] = {}
_CURRENT_MOE_LAYER_INDEX: Optional[int] = None
_CURRENT_PROTECTED_TOKEN_MASK: Optional[torch.Tensor] = None
_CURRENT_EAC_PREFILL_MASK: Optional[torch.Tensor] = None
_CURRENT_TOKEN_SEGMENTS: Optional[list[tuple[int, int]]] = None
_CURRENT_TOKEN_SEGMENT_SEQ_LENS: Optional[list[tuple[int, int, int]]] = None
_CURRENT_TOKEN_SEQ_LENS: Optional[torch.Tensor] = None
_CURRENT_TOKEN_DECODE_OFFSETS: Optional[torch.Tensor] = None
_CURRENT_INPUT_TOKEN_IDS: Optional[torch.Tensor] = None
_CURRENT_INPUT_POSITIONS: Optional[torch.Tensor] = None
_CURRENT_PROMPT_PROFILE_IDS: Optional[torch.Tensor] = None
_CURRENT_PROMPT_PROFILE = 0
_LAST_PROMPT_PROFILE = 0
_DEBUG_ROUTING = False
_DEBUG_MAX_PRINTS = 8
_DEBUG_PRINT_COUNT = 0
_STATS_DIR: Optional[Path] = None
# Statistics are accumulated on the device and only read back every this many routing
# calls. Reading back per call costs a device synchronisation and a snapshot write on
# every MoE layer of every decode step, which cut generation throughput by ~14x.
_STATS_FLUSH_INTERVAL_DEFAULT = 512
_STATS_FLUSH_INTERVAL = _STATS_FLUSH_INTERVAL_DEFAULT
_STATS_TOTAL_SELECTED = 0.0
_STATS_TOTAL_TOKENS = 0
_STATS_TOTAL_CALLS = 0
_STATS_TOTAL_PROTECTED_TOKENS = 0
_STATS_DEVICE_ACCUM: Optional[dict] = None
_STATS_LAST_TOPK: Optional[int] = None
_STATS_EAC_PREFILL_TOKENS = 0
_STATS_EAC_SEGMENTS = 0
_STATS_EAC_CALLS_WITH_SEGMENTS = 0
_STATS_EAC_PRUNED_SELECTIONS = 0
_STATS_EAC_DISABLED_EXPERTS = 0
_STATS_BAN_TOTAL_K = 0.0
_STATS_BAN_TOKENS = 0
_STATS_BUDGET_EASY_TOKENS = 0
_STATS_BUDGET_BASE_TOKENS = 0
_STATS_BUDGET_HARD_TOKENS = 0
_STATS_BUDGET_LAYER_COUNTS: dict[int, list[int]] = {}
_STATS_BUDGET_PROFILE_COUNTS: dict[int, list[int]] = {}
_STATS_BUDGET_STAGE_COUNTS: dict[str, list[int]] = {
    "prefill": [0, 0, 0],
    "decode": [0, 0, 0],
    "unknown": [0, 0, 0],
}
_STATS_BUDGET_HARD_RANK_WEIGHT_BOUNDS = [
    0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 1.0
]


def _stats_hist_size(bounds: list) -> int:
    return len(bounds) + 1


_STATS_BUDGET_HARD_RANK_WEIGHT_CANDIDATE = {
    "count": 0,
    "sum": 0.0,
    "min": None,
    "max": None,
    "hist": [0] * _stats_hist_size(_STATS_BUDGET_HARD_RANK_WEIGHT_BOUNDS),
}
_STATS_BUDGET_HARD_RANK_WEIGHT_SELECTED = {
    "count": 0,
    "sum": 0.0,
    "min": None,
    "max": None,
    "hist": [0] * _stats_hist_size(_STATS_BUDGET_HARD_RANK_WEIGHT_BOUNDS),
}
_STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BOUNDS = [
    0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0
]
_STATS_BUDGET_PREFILL_SEGMENT_DENSITY_CANDIDATE = {
    "count": 0,
    "sum": 0.0,
    "min": None,
    "max": None,
    "hist": [0] * _stats_hist_size(
        _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BOUNDS),
    "hard_tokens": 0,
    "segment_tokens": 0,
}
_STATS_BUDGET_PREFILL_SEGMENT_DENSITY_SELECTED = {
    "count": 0,
    "sum": 0.0,
    "min": None,
    "max": None,
    "hist": [0] * _stats_hist_size(
        _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BOUNDS),
    "hard_tokens": 0,
    "segment_tokens": 0,
}
_STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BY_LAYER: dict[int, dict[str, dict]] = {}
_STATS_BUDGET_HARD_SEQ_LEN_BOUNDS = [
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
]
_STATS_BUDGET_HARD_SEQ_LEN_CANDIDATE = {
    "count": 0,
    "sum": 0.0,
    "min": None,
    "max": None,
    "hist": [0] * _stats_hist_size(_STATS_BUDGET_HARD_SEQ_LEN_BOUNDS),
}
_STATS_BUDGET_HARD_SEQ_LEN_SELECTED = {
    "count": 0,
    "sum": 0.0,
    "min": None,
    "max": None,
    "hist": [0] * _stats_hist_size(_STATS_BUDGET_HARD_SEQ_LEN_BOUNDS),
}
_STATS_BUDGET_HARD_DECODE_OFFSET_BOUNDS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 220, 260, 320, 512, 1024, 2048, 4096
]
_STATS_BUDGET_HARD_DECODE_OFFSET_CANDIDATE = {
    "count": 0,
    "sum": 0.0,
    "min": None,
    "max": None,
    "hist": [0] * _stats_hist_size(
        _STATS_BUDGET_HARD_DECODE_OFFSET_BOUNDS),
}
_STATS_BUDGET_HARD_DECODE_OFFSET_SELECTED = {
    "count": 0,
    "sum": 0.0,
    "min": None,
    "max": None,
    "hist": [0] * _stats_hist_size(
        _STATS_BUDGET_HARD_DECODE_OFFSET_BOUNDS),
}
_SKIP_STATS_DEPTH = 0


def current_routing_method() -> str:
    """Return the currently configured expert pruning method."""
    return _ROUTING_METHOD


def is_expert_routing_stats_skipped() -> bool:
    """Return whether routing is inside a known dummy/profile run."""
    return _SKIP_STATS_DEPTH > 0


def current_moe_layer_index() -> Optional[int]:
    """Return the MoE layer index set by the FusedMoE forward wrapper."""
    return _CURRENT_MOE_LAYER_INDEX


def current_eac_prefill_mask_tensor() -> Optional[torch.Tensor]:
    """Return the current prefill mask for routed tokens, if any."""
    return _CURRENT_EAC_PREFILL_MASK


@contextmanager
def skip_expert_routing_stats():
    """Temporarily disable routing statistics in known dummy/profile runs."""
    global _SKIP_STATS_DEPTH
    _SKIP_STATS_DEPTH += 1
    try:
        yield
    finally:
        _SKIP_STATS_DEPTH -= 1


def _layer_name_to_index(layer_name: str | None) -> Optional[int]:
    if layer_name is None:
        return None
    match = re.search(r"layers\.(\d+)", layer_name)
    if match is None:
        return None
    return int(match.group(1))


@contextmanager
def set_current_moe_layer(layer_name: str | None):
    """Set current MoE layer so routing can use layer-specific artifacts."""
    global _CURRENT_MOE_LAYER_INDEX
    previous = _CURRENT_MOE_LAYER_INDEX
    layer_index = _layer_name_to_index(layer_name)
    if layer_index is not None:
        _CURRENT_MOE_LAYER_INDEX = layer_index
    try:
        yield
    finally:
        _CURRENT_MOE_LAYER_INDEX = previous


@contextmanager
def set_current_protected_token_mask(mask: Optional[torch.Tensor]):
    """Set token mask whose selected experts must not be pruned."""
    global _CURRENT_PROTECTED_TOKEN_MASK
    previous = _CURRENT_PROTECTED_TOKEN_MASK
    _CURRENT_PROTECTED_TOKEN_MASK = mask
    try:
        yield
    finally:
        _CURRENT_PROTECTED_TOKEN_MASK = previous


@contextmanager
def set_current_eac_prefill_mask(mask: Optional[torch.Tensor]):
    """Set a tensor mask for tokens that EAC_MoE may prune."""
    global _CURRENT_EAC_PREFILL_MASK
    previous = _CURRENT_EAC_PREFILL_MASK
    _CURRENT_EAC_PREFILL_MASK = mask
    try:
        yield
    finally:
        _CURRENT_EAC_PREFILL_MASK = previous


@contextmanager
def set_current_token_segments(
        segments: Optional[list[tuple[int, int]]]):
    """Set position-derived token segments for prefill-only routing."""
    global _CURRENT_TOKEN_SEGMENTS
    previous = _CURRENT_TOKEN_SEGMENTS
    _CURRENT_TOKEN_SEGMENTS = segments
    try:
        yield
    finally:
        _CURRENT_TOKEN_SEGMENTS = previous


@contextmanager
def set_current_token_segment_seq_lens(
        segments: Optional[list[tuple[int, int, int]]]):
    """Set prefill token segments with request-level sequence lengths."""
    global _CURRENT_TOKEN_SEGMENT_SEQ_LENS
    previous = _CURRENT_TOKEN_SEGMENT_SEQ_LENS
    _CURRENT_TOKEN_SEGMENT_SEQ_LENS = segments
    try:
        yield
    finally:
        _CURRENT_TOKEN_SEGMENT_SEQ_LENS = previous


@contextmanager
def set_current_token_seq_lens(seq_lens: Optional[torch.Tensor]):
    """Set per-token sequence lengths when vLLM metadata exposes them."""
    global _CURRENT_TOKEN_SEQ_LENS
    previous = _CURRENT_TOKEN_SEQ_LENS
    _CURRENT_TOKEN_SEQ_LENS = seq_lens
    try:
        yield
    finally:
        _CURRENT_TOKEN_SEQ_LENS = previous


@contextmanager
def set_current_token_decode_offsets(offsets: Optional[torch.Tensor]):
    """Set per-token decode offsets; prefill tokens should use -1."""
    global _CURRENT_TOKEN_DECODE_OFFSETS
    previous = _CURRENT_TOKEN_DECODE_OFFSETS
    _CURRENT_TOKEN_DECODE_OFFSETS = offsets
    try:
        yield
    finally:
        _CURRENT_TOKEN_DECODE_OFFSETS = previous


@contextmanager
def set_current_input_token_ids(
        input_ids: Optional[torch.Tensor],
        positions: Optional[torch.Tensor] = None):
    """Set current request token ids for prompt-marker routing diagnostics."""
    global _CURRENT_INPUT_TOKEN_IDS, _CURRENT_INPUT_POSITIONS
    global _CURRENT_PROMPT_PROFILE_IDS, _CURRENT_PROMPT_PROFILE
    global _LAST_PROMPT_PROFILE
    previous = _CURRENT_INPUT_TOKEN_IDS
    previous_positions = _CURRENT_INPUT_POSITIONS
    previous_profile_ids = _CURRENT_PROMPT_PROFILE_IDS
    previous_profile = _CURRENT_PROMPT_PROFILE
    if isinstance(input_ids, torch.Tensor):
        _CURRENT_INPUT_TOKEN_IDS = input_ids.reshape(-1)
        _CURRENT_INPUT_POSITIONS = (
            positions.reshape(-1) if isinstance(positions, torch.Tensor)
            else None)
        _CURRENT_PROMPT_PROFILE_IDS = _detect_prompt_profile_ids(
            _CURRENT_INPUT_TOKEN_IDS,
            _CURRENT_INPUT_POSITIONS,
        )
        if _CURRENT_PROMPT_PROFILE_IDS is not None:
            nonzero_profiles = _CURRENT_PROMPT_PROFILE_IDS[
                _CURRENT_PROMPT_PROFILE_IDS > 0]
            if int(nonzero_profiles.numel()) > 0:
                _CURRENT_PROMPT_PROFILE = int(nonzero_profiles[-1].item())
                _LAST_PROMPT_PROFILE = _CURRENT_PROMPT_PROFILE
            else:
                _CURRENT_PROMPT_PROFILE = 0
        else:
            _CURRENT_PROMPT_PROFILE = _detect_prompt_profile(
                _CURRENT_INPUT_TOKEN_IDS)
            _LAST_PROMPT_PROFILE = _CURRENT_PROMPT_PROFILE
    else:
        _CURRENT_INPUT_TOKEN_IDS = None
        _CURRENT_INPUT_POSITIONS = None
        _CURRENT_PROMPT_PROFILE_IDS = None
        _CURRENT_PROMPT_PROFILE = 0
    try:
        yield
    finally:
        _CURRENT_INPUT_TOKEN_IDS = previous
        _CURRENT_INPUT_POSITIONS = previous_positions
        _CURRENT_PROMPT_PROFILE_IDS = previous_profile_ids
        _CURRENT_PROMPT_PROFILE = previous_profile


def _is_cuda_graph_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is None or not hasattr(compiler, "is_compiling"):
        return False
    return bool(compiler.is_compiling())


def _device_cache_key(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return f"cuda:{index}"
    return str(device)


def _parse_token_id_list(token_ids: str | None) -> tuple[int, ...]:
    if token_ids is None:
        return ()
    values = []
    for item in re.split(r"[,\s]+", str(token_ids).strip()):
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError(
                "--band_layer_budget_hard_prefill_conditional_token_ids "
                "must contain non-negative token ids")
        values.append(value)
    return tuple(dict.fromkeys(values))


def _parse_positive_int_list(
    raw_values: str | None,
    option_name: str,
) -> tuple[int, ...]:
    if raw_values is None:
        return ()
    values = []
    for item in re.split(r"[,\s]+", str(raw_values).strip()):
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError(f"{option_name} must contain positive integers")
        values.append(value)
    return tuple(values)


def _parse_nonnegative_int_list(
    raw_values: str | None,
    option_name: str,
) -> tuple[int, ...]:
    if raw_values is None:
        return ()
    values = []
    for item in re.split(r"[,\s]+", str(raw_values).strip()):
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError(
                f"{option_name} must contain non-negative integers")
        values.append(value)
    return tuple(dict.fromkeys(values))


def _extreme_fraction_mask(
    scores: torch.Tensor,
    ratio: float,
    *,
    largest: bool,
) -> torch.Tensor:
    """Select a fixed floor(ratio * N) quota without absolute thresholds."""
    count = int(float(ratio) * int(scores.numel()))
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if count <= 0:
        return mask
    count = min(count, int(scores.numel()))
    selected = torch.topk(
        scores,
        k=count,
        largest=largest,
        sorted=False,
    ).indices
    mask[selected] = True
    return mask


def _parse_token_ngram_list(
        token_ngrams: str | None) -> tuple[tuple[int, ...], ...]:
    if token_ngrams is None:
        return ()
    ngrams = []
    seen = set()
    for group in str(token_ngrams).strip().split(";"):
        group = group.strip()
        if not group:
            continue
        values = []
        for item in re.split(r"[,\s]+", group):
            if not item:
                continue
            value = int(item)
            if value < 0:
                raise ValueError(
                    "prompt profile token n-grams must contain "
                    "non-negative token ids")
            values.append(value)
        if not values:
            continue
        ngram = tuple(values)
        if ngram not in seen:
            seen.add(ngram)
            ngrams.append(ngram)
    return tuple(ngrams)


def _contains_token_ngram(
        token_ids: torch.Tensor,
        token_ngrams: tuple[tuple[int, ...], ...]) -> bool:
    if not token_ngrams:
        return False
    min_len = min(len(ngram) for ngram in token_ngrams)
    num_tokens = int(token_ids.numel())
    if num_tokens < min_len:
        return False
    ids = token_ids.detach().to(device="cpu").tolist()
    for ngram in token_ngrams:
        n = len(ngram)
        if n == 0 or num_tokens < n:
            continue
        first = ngram[0]
        for start, value in enumerate(ids[:num_tokens - n + 1]):
            if value == first and tuple(ids[start:start + n]) == ngram:
                return True
    return False


def _detect_prompt_profile(token_ids: torch.Tensor) -> int:
    all_ngrams = (
        _BAND_LAYER_BUDGET_PROMPT_PROFILE1_TOKEN_NGRAMS
        + _BAND_LAYER_BUDGET_PROMPT_PROFILE2_TOKEN_NGRAMS)
    if not all_ngrams:
        return 0
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return _LAST_PROMPT_PROFILE
    min_len = min(len(ngram) for ngram in all_ngrams)
    if int(token_ids.numel()) < min_len:
        return _LAST_PROMPT_PROFILE
    if _contains_token_ngram(
            token_ids, _BAND_LAYER_BUDGET_PROMPT_PROFILE1_TOKEN_NGRAMS):
        return 1
    if _contains_token_ngram(
            token_ids, _BAND_LAYER_BUDGET_PROMPT_PROFILE2_TOKEN_NGRAMS):
        return 2
    return 0


def _match_prompt_profile_no_fallback(token_ids: torch.Tensor) -> int:
    if _contains_token_ngram(
            token_ids, _BAND_LAYER_BUDGET_PROMPT_PROFILE1_TOKEN_NGRAMS):
        return 1
    if _contains_token_ngram(
            token_ids, _BAND_LAYER_BUDGET_PROMPT_PROFILE2_TOKEN_NGRAMS):
        return 2
    return 0


def _segments_from_positions(
        num_tokens: int,
        positions: Optional[torch.Tensor]) -> list[tuple[int, int]]:
    if positions is None or int(positions.numel()) != int(num_tokens):
        return [(0, num_tokens)]
    if num_tokens <= 0:
        return []
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return [(0, num_tokens)]

    pos_values = positions.detach().to(device="cpu").tolist()
    segments = []
    start = 0
    previous_pos = int(pos_values[0])
    for index in range(1, num_tokens):
        current_pos = int(pos_values[index])
        if current_pos <= previous_pos:
            segments.append((start, index))
            start = index
        previous_pos = current_pos
    segments.append((start, num_tokens))
    return segments


def _detect_prompt_profile_ids(
        token_ids: torch.Tensor,
        positions: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    all_ngrams = (
        _BAND_LAYER_BUDGET_PROMPT_PROFILE1_TOKEN_NGRAMS
        + _BAND_LAYER_BUDGET_PROMPT_PROFILE2_TOKEN_NGRAMS)
    if not all_ngrams:
        return None
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None

    flat_ids = token_ids.reshape(-1)
    num_tokens = int(flat_ids.numel())
    profile_ids = torch.zeros(num_tokens, dtype=torch.long)
    min_len = min(len(ngram) for ngram in all_ngrams)
    for start, end in _segments_from_positions(num_tokens, positions):
        if end <= start:
            continue
        segment_ids = flat_ids[start:end]
        if int(segment_ids.numel()) < min_len:
            profile = _LAST_PROMPT_PROFILE
        else:
            profile = _match_prompt_profile_no_fallback(segment_ids)
        if profile > 0:
            profile_ids[start:end] = int(profile)
    return profile_ids


def prepare_diep_tensors_for_device(device: torch.device) -> None:
    """Move DiEP calibration tensors to ``device`` before CUDA graph capture."""
    global _DIEP_DEVICE_CACHE

    if _DIEP_SIM_MATRIX is None or _DIEP_MEAN_SIM is None:
        return

    cache_key = _device_cache_key(device)
    if cache_key in _DIEP_DEVICE_CACHE:
        return

    if _is_cuda_graph_capturing():
        raise RuntimeError(
            "DiEP artifact tensors were first needed during CUDA graph "
            "capture. They must be moved to GPU before capture starts. "
            "This should happen automatically in the vLLM patch; if this "
            "still appears, run with --enforce_eager as a temporary "
            "workaround.")

    gamma1 = _DIEP_GAMMA1
    if gamma1 is None:
        gamma1 = torch.ones_like(_DIEP_MEAN_SIM)
    _DIEP_DEVICE_CACHE[cache_key] = (
        _DIEP_SIM_MATRIX.to(device=device, non_blocking=True),
        _DIEP_MEAN_SIM.to(device=device, non_blocking=True),
        gamma1.to(device=device, non_blocking=True),
    )


def prepare_ban_tensors_for_device(device: torch.device) -> None:
    """Move Ban calibration tensors to ``device`` before CUDA graph capture."""
    global _BAN_DEVICE_CACHE

    if _BAN_LAYER_SENSITIVITY is None:
        return

    cache_key = _device_cache_key(device)
    if cache_key in _BAN_DEVICE_CACHE:
        return

    if _is_cuda_graph_capturing():
        raise RuntimeError(
            "Ban artifact tensors were first needed during CUDA graph "
            "capture. They must be moved to GPU before capture starts. "
            "This should happen automatically in the vLLM patch; if this "
            "still appears, run with --enforce_eager as a temporary "
            "workaround.")

    _BAN_DEVICE_CACHE[cache_key] = _BAN_LAYER_SENSITIVITY.to(
        device=device,
        non_blocking=True,
    )


def prepare_router_value_tensors_for_device(device: torch.device) -> None:
    """Move router-value coefficients before CUDA graph capture."""
    global _ROUTER_VALUE_DEVICE_CACHE, _ROUTER_VALUE_LOWER_DEVICE_CACHE

    if not _ROUTER_VALUE_COEFFICIENTS:
        return

    cache_key = _device_cache_key(device)
    if cache_key in _ROUTER_VALUE_DEVICE_CACHE:
        return

    if _is_cuda_graph_capturing():
        raise RuntimeError(
            "Router-value coefficients were first needed during CUDA graph "
            "capture. They must be moved to GPU before capture starts.")

    _ROUTER_VALUE_DEVICE_CACHE[cache_key] = torch.tensor(
        _ROUTER_VALUE_COEFFICIENTS,
        dtype=torch.float32,
        device=device,
    )
    if _ROUTER_VALUE_LOWER_COEFFICIENTS:
        _ROUTER_VALUE_LOWER_DEVICE_CACHE[cache_key] = torch.tensor(
            _ROUTER_VALUE_LOWER_COEFFICIENTS,
            dtype=torch.float32,
            device=device,
        )


def _get_diep_tensors_for_device(
        device: torch.device) -> tuple[Optional[torch.Tensor],
                                      Optional[torch.Tensor],
                                      Optional[torch.Tensor]]:
    prepare_diep_tensors_for_device(device)
    if _DIEP_SIM_MATRIX is None or _DIEP_MEAN_SIM is None:
        return None, None, None
    return _DIEP_DEVICE_CACHE[_device_cache_key(device)]


def _get_ban_layer_sensitivity_for_device(
        device: torch.device) -> Optional[torch.Tensor]:
    prepare_ban_tensors_for_device(device)
    if _BAN_LAYER_SENSITIVITY is None:
        return None
    return _BAN_DEVICE_CACHE[_device_cache_key(device)]


def _get_router_value_coefficients_for_device(
        device: torch.device) -> torch.Tensor:
    prepare_router_value_tensors_for_device(device)
    return _ROUTER_VALUE_DEVICE_CACHE[_device_cache_key(device)]


def _get_router_value_lower_coefficients_for_device(
        device: torch.device) -> torch.Tensor:
    prepare_router_value_tensors_for_device(device)
    return _ROUTER_VALUE_LOWER_DEVICE_CACHE[_device_cache_key(device)]


def _load_diep_artifact(artifact_path: str | None) -> None:
    global _DIEP_ARTIFACT_PATH, _DIEP_SIM_MATRIX, _DIEP_MEAN_SIM, _DIEP_GAMMA1
    global _DIEP_DEVICE_CACHE

    _DIEP_DEVICE_CACHE = {}
    if not artifact_path:
        _DIEP_ARTIFACT_PATH = None
        _DIEP_SIM_MATRIX = None
        _DIEP_MEAN_SIM = None
        _DIEP_GAMMA1 = None
        return
    if _DIEP_ARTIFACT_PATH == artifact_path and _DIEP_SIM_MATRIX is not None:
        return

    payload = torch.load(artifact_path, map_location="cpu")
    _DIEP_SIM_MATRIX = payload["sim_matrix"].float()
    _DIEP_MEAN_SIM = payload["mean_sim"].float()
    _DIEP_GAMMA1 = payload["gamma_1"].float()
    _DIEP_ARTIFACT_PATH = artifact_path


def _load_ban_artifact(artifact_path: str | None) -> None:
    global _BAN_ARTIFACT_PATH, _BAN_LAYER_SENSITIVITY
    global _BAN_R_MIN, _BAN_R_MAX, _BAN_DEVICE_CACHE

    _BAN_DEVICE_CACHE = {}
    if not artifact_path:
        _BAN_ARTIFACT_PATH = None
        _BAN_LAYER_SENSITIVITY = None
        _BAN_R_MIN = 0.0
        _BAN_R_MAX = 1.0
        return
    if (_BAN_ARTIFACT_PATH == artifact_path
            and _BAN_LAYER_SENSITIVITY is not None):
        return

    payload = torch.load(artifact_path, map_location="cpu")
    if "layer_sensitivity" not in payload:
        raise KeyError("Ban artifact must contain `layer_sensitivity`.")
    _BAN_LAYER_SENSITIVITY = payload["layer_sensitivity"].float()
    _BAN_R_MIN = float(payload.get("r_min", 0.0))
    _BAN_R_MAX = float(payload.get("r_max", 1.0))
    if _BAN_R_MAX <= _BAN_R_MIN:
        _BAN_R_MAX = _BAN_R_MIN + 1e-6
    _BAN_ARTIFACT_PATH = artifact_path


def _load_router_value_artifact(
    artifact_path: str | None,
    base_k: int,
    quota_ratio: float,
) -> None:
    global _ROUTER_VALUE_ARTIFACT_PATH, _ROUTER_VALUE_INTERCEPT
    global _ROUTER_VALUE_ARTIFACT_FORMAT
    global _ROUTER_VALUE_COEFFICIENTS, _ROUTER_VALUE_LAYER_COEFFICIENTS
    global _ROUTER_VALUE_LOWER_COEFFICIENTS
    global _ROUTER_VALUE_LAYER_THRESHOLDS, _ROUTER_VALUE_DEMOTION_THRESHOLDS
    global _ROUTER_VALUE_DEVICE_CACHE, _ROUTER_VALUE_LOWER_DEVICE_CACHE
    _ROUTER_VALUE_DEVICE_CACHE = {}
    _ROUTER_VALUE_LOWER_DEVICE_CACHE = {}
    if not artifact_path:
        raise ValueError(
            "RouterValue_Dynamic_Routing requires "
            "--router_value_artifact_path")
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Router-value artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact_format = payload.get("format")
    if artifact_format not in {
            "router_value_logistic_v1",
            "router_value_fixed_swap_v1",
    }:
        raise ValueError(
            f"Unsupported router-value artifact format: {artifact_format}")
    lower_coefficients: tuple[float, ...] = ()
    demotion_thresholds: tuple[float, ...] = ()
    if artifact_format == "router_value_logistic_v1":
        predictor = payload.get("base_ks", {}).get(str(base_k))
        if predictor is None:
            raise ValueError(
                f"Router-value artifact does not contain base_k={base_k}")
        coefficients = tuple(
            float(value) for value in predictor["router_coefficients"])
        layer_coefficients = tuple(
            float(value) for value in predictor["layer_coefficients"])
        layer_thresholds = tuple(
            float(value)
            for value in predictor["layer_router_score_thresholds"])
        calibrated_ratio = float(predictor["positive_fraction"])
        if abs(calibrated_ratio - quota_ratio) > 1e-9:
            raise ValueError(
                "--router_value_quota_ratio must match the predictor's "
                f"calibrated positive_fraction={calibrated_ratio}")
        intercept = float(predictor["intercept"])
    else:
        artifact_base_k = int(payload["base_k"])
        if artifact_base_k != base_k:
            raise ValueError(
                "--router_value_base_k must match the fixed-swap artifact's "
                f"base_k={artifact_base_k}")
        profile_key = f"{quota_ratio:g}"
        profile = payload.get("profiles", {}).get(profile_key)
        if profile is None:
            raise ValueError(
                "Router-value fixed-swap artifact does not contain profile "
                f"{profile_key}")
        coefficients = tuple(
            float(value)
            for value in payload["upper_router_coefficients"])
        lower_coefficients = tuple(
            float(value)
            for value in payload["lower_router_coefficients"])
        layer_thresholds = tuple(
            float(value) for value in profile["promotion_thresholds"])
        demotion_thresholds = tuple(
            float(value) for value in profile["demotion_thresholds"])
        layer_coefficients = (0.0,) * len(layer_thresholds)
        intercept = 0.0
    if len(coefficients) != 31:
        raise ValueError(
            "Router-value predictor must contain 31 top-8 router features")
    if lower_coefficients and len(lower_coefficients) != 31:
        raise ValueError(
            "Router-value lower predictor must contain 31 top-8 features")
    if not layer_coefficients:
        raise ValueError("Router-value predictor has no layer coefficients")
    if len(layer_thresholds) != len(layer_coefficients):
        raise ValueError(
            "Router-value predictor must contain one threshold per layer")
    if demotion_thresholds and len(demotion_thresholds) != len(
            layer_thresholds):
        raise ValueError(
            "Router-value fixed-swap artifact must contain one promotion and "
            "demotion threshold per layer")
    _ROUTER_VALUE_ARTIFACT_PATH = str(path)
    _ROUTER_VALUE_ARTIFACT_FORMAT = artifact_format
    _ROUTER_VALUE_INTERCEPT = intercept
    _ROUTER_VALUE_COEFFICIENTS = coefficients
    _ROUTER_VALUE_LOWER_COEFFICIENTS = lower_coefficients
    _ROUTER_VALUE_LAYER_COEFFICIENTS = layer_coefficients
    _ROUTER_VALUE_LAYER_THRESHOLDS = layer_thresholds
    _ROUTER_VALUE_DEMOTION_THRESHOLDS = demotion_thresholds


def configure_expert_router(
    method: str = "none",
    naee_beta: float = 0.3,
    naee_k_min: int = 2,
    dynamic_routing_threshold: float = 0.8,
    dynamic_routing_score_source: str = "raw",
    dynamic_routing_next_rank_penalty: float = 1.0,
    layerwise_dynamic_base_threshold: float = 0.55,
    layerwise_dynamic_layer_alpha: float = 0.15,
    layerwise_dynamic_num_layers: int = 48,
    layerwise_dynamic_k_min: int = 1,
    budget_dynamic_easy_k: int = 4,
    budget_dynamic_base_k: int = 5,
    budget_dynamic_hard_k: int = 6,
    budget_dynamic_easy_threshold: float = 0.90,
    budget_dynamic_hard_threshold: float = 0.75,
    budget_dynamic_score_topn: int = 3,
    layer_budget_dynamic_easy_layer_alpha: float = 0.08,
    layer_budget_dynamic_hard_layer_alpha: float = 0.08,
    layer_budget_dynamic_num_layers: int = 48,
    band_layer_budget_middle_start: int = 19,
    band_layer_budget_middle_end: int = 28,
    band_layer_budget_per_layer_k: str | None = None,
    band_layer_budget_easy_layers: str | None = None,
    band_layer_budget_hard_layers: str | None = None,
    band_layer_budget_token_gate_ratio: float = -1.0,
    band_layer_budget_extra_middle_start: int = -1,
    band_layer_budget_extra_middle_end: int = -1,
    band_layer_budget_middle_phase: str = "all",
    band_layer_budget_easy_start: int = 0,
    band_layer_budget_easy_end: int = 47,
    band_layer_budget_easy_phase: str = "all",
    band_layer_budget_hard_phase: str = "all",
    band_layer_budget_easy_tail_threshold: float = 1.0,
    band_layer_budget_hard_min_rank_weight: float = 0.0,
    band_layer_budget_hard_max_rank_weight: float = 1.0,
    band_layer_budget_hard_decode_min_rank_weight: float = -1.0,
    band_layer_budget_mixed_rescue_min_rank_weight: float = -1.0,
    band_layer_budget_mixed_rescue_max_concentration: float = -1.0,
    band_layer_budget_hard_prefill_max_tokens: int = -1,
    band_layer_budget_hard_prefill_max_segment_tokens: int = -1,
    band_layer_budget_hard_prefill_max_density: float = -1.0,
    band_layer_budget_hard_prefill_min_density: float = -1.0,
    band_layer_budget_hard_prefill_max_segment_hard_ratio: float = -1.0,
    band_layer_budget_hard_prefill_segment_cap_score: str = "concentration",
    band_layer_budget_hard_prefill_conditional_segment_hard_ratio:
    float = -1.0,
    band_layer_budget_hard_prefill_conditional_min_density: float = -1.0,
    band_layer_budget_hard_prefill_conditional_min_seq_len: int = -1,
    band_layer_budget_hard_prefill_conditional_token_ids: str | None = None,
    band_layer_budget_hard_prefill_conditional_token_ngrams:
    str | None = None,
    band_layer_budget_hard_prefill_marker_segment_hard_ratio:
    float = -1.0,
    band_layer_budget_hard_prefill_marker_token_ids: str | None = None,
    band_layer_budget_hard_prefill_marker_token_ngrams:
    str | None = None,
    band_layer_budget_hard_prefill_exclude_token_ids: str | None = None,
    band_layer_budget_hard_prefill_min_relative_pos: float = -1.0,
    band_layer_budget_hard_prefill_max_relative_pos: float = -1.0,
    band_layer_budget_hard_prefill_start: int = -1,
    band_layer_budget_hard_prefill_end: int = -1,
    band_layer_budget_hard_decode_min_seq_len: int = -1,
    band_layer_budget_hard_decode_max_seq_len: int = -1,
    band_layer_budget_hard_decode_seq_len_scope: str = "all",
    band_layer_budget_hard_decode_min_offset: int = -1,
    band_layer_budget_hard_decode_max_offset: int = -1,
    band_layer_budget_hard_decode_start: int = -1,
    band_layer_budget_hard_decode_end: int = -1,
    band_layer_budget_hard_decode_token_ids: str | None = None,
    band_layer_budget_hard_layer_prior_start: int = -1,
    band_layer_budget_hard_layer_prior_end: int = -1,
    band_layer_budget_hard_layer_prior_extra_start: int = -1,
    band_layer_budget_hard_layer_prior_extra_end: int = -1,
    band_layer_budget_hard_layer_prior_alpha: float = 0.0,
    band_layer_budget_hard_layer_sensitivity_alpha: float = 0.0,
    band_layer_budget_late_concentration_start: int = -1,
    band_layer_budget_late_concentration_threshold: float = -1.0,
    band_layer_budget_late_rescue_start: int = -1,
    band_layer_budget_late_rescue_concentration_threshold: float = -1.0,
    band_layer_budget_late_rescue_mixed_max: float = -1.0,
    band_layer_budget_prompt_profile1_token_ngrams: str | None = None,
    band_layer_budget_prompt_profile1_hard_threshold: float = -1.0,
    band_layer_budget_prompt_profile1_hard_min_rank_weight: float = -1.0,
    band_layer_budget_prompt_profile1_middle_start: int = -1,
    band_layer_budget_prompt_profile1_middle_end: int = -1,
    band_layer_budget_prompt_profile1_prefill_segment_hard_ratio:
    float = -1.0,
    band_layer_budget_prompt_profile2_token_ngrams: str | None = None,
    band_layer_budget_prompt_profile2_hard_threshold: float = -1.0,
    band_layer_budget_prompt_profile2_hard_min_rank_weight: float = -1.0,
    band_layer_budget_prompt_profile2_middle_start: int = -1,
    band_layer_budget_prompt_profile2_middle_end: int = -1,
    band_layer_budget_prompt_profile2_prefill_segment_hard_ratio:
    float = -1.0,
    diep_artifact_path: str | None = None,
    diep_pruning_mode: str = "independent",
    diep_use_gamma1: bool = False,
    diep_gamma_alpha: float = 1.0,
    diep_threshold_cap: float | None = None,
    ban_artifact_path: str | None = None,
    ban_lambda: float = 0.7,
    ban_k_min: int = 3,
    eac_alpha: float = 0.5,
    mc_moe_protection_ratio: float = 0.02,
    biased_renorm_keep_topn: int = 3,
    router_value_artifact_path: str | None = None,
    router_value_base_k: int = 3,
    router_value_quota_ratio: float = 0.10,
    router_value_selection_mode: str = "threshold",
    router_value_swap_direction: str = "both",
    debug: bool = False,
    debug_max_prints: int = 8,
) -> None:
    """Configure the local expert router.

    ``method="none"`` keeps the local router equivalent to ordinary top-k.
    ``method="NAEE"`` enables dynamic expert pruning by thresholding selected
    expert weights against ``top1_weight * beta`` after ``k_min`` experts.
    ``method="Dynamic_Routing"`` keeps the smallest sorted expert prefix whose
    cumulative softmax score reaches ``dynamic_routing_threshold``.
    ``method="DiEP"`` extends NAEE with a layer/expert-pair factor loaded from
    a calibration artifact.
    """
    global _ROUTING_METHOD, _NAEE_BETA, _NAEE_K_MIN
    global _DYNAMIC_ROUTING_THRESHOLD, _DYNAMIC_ROUTING_SCORE_SOURCE
    global _DYNAMIC_ROUTING_NEXT_RANK_PENALTY
    global _LAYERWISE_DYNAMIC_BASE_THRESHOLD
    global _LAYERWISE_DYNAMIC_LAYER_ALPHA, _LAYERWISE_DYNAMIC_NUM_LAYERS
    global _LAYERWISE_DYNAMIC_K_MIN
    global _BUDGET_DYNAMIC_EASY_K, _BUDGET_DYNAMIC_BASE_K
    global _BUDGET_DYNAMIC_HARD_K, _BUDGET_DYNAMIC_EASY_THRESHOLD
    global _BUDGET_DYNAMIC_HARD_THRESHOLD, _BUDGET_DYNAMIC_SCORE_TOPN
    global _LAYER_BUDGET_DYNAMIC_EASY_LAYER_ALPHA
    global _LAYER_BUDGET_DYNAMIC_HARD_LAYER_ALPHA
    global _LAYER_BUDGET_DYNAMIC_NUM_LAYERS
    global _BAND_LAYER_BUDGET_MIDDLE_START, _BAND_LAYER_BUDGET_MIDDLE_END
    global _BAND_LAYER_BUDGET_PER_LAYER_K
    global _BAND_LAYER_BUDGET_EASY_LAYERS, _BAND_LAYER_BUDGET_HARD_LAYERS
    global _BAND_LAYER_BUDGET_TOKEN_GATE_RATIO
    global _BAND_LAYER_BUDGET_EXTRA_MIDDLE_START
    global _BAND_LAYER_BUDGET_EXTRA_MIDDLE_END
    global _BAND_LAYER_BUDGET_MIDDLE_PHASE
    global _BAND_LAYER_BUDGET_EASY_START, _BAND_LAYER_BUDGET_EASY_END
    global _BAND_LAYER_BUDGET_EASY_PHASE, _BAND_LAYER_BUDGET_HARD_PHASE
    global _BAND_LAYER_BUDGET_EASY_TAIL_THRESHOLD
    global _BAND_LAYER_BUDGET_HARD_MIN_RANK_WEIGHT
    global _BAND_LAYER_BUDGET_HARD_MAX_RANK_WEIGHT
    global _BAND_LAYER_BUDGET_HARD_DECODE_MIN_RANK_WEIGHT
    global _BAND_LAYER_BUDGET_MIXED_RESCUE_MIN_RANK_WEIGHT
    global _BAND_LAYER_BUDGET_MIXED_RESCUE_MAX_CONCENTRATION
    global _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_TOKENS
    global _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_TOKENS
    global _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_DENSITY
    global _BAND_LAYER_BUDGET_HARD_PREFILL_MIN_DENSITY
    global _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_HARD_RATIO
    global _BAND_LAYER_BUDGET_HARD_PREFILL_SEGMENT_CAP_SCORE
    global _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_SEGMENT_HARD_RATIO
    global _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_DENSITY
    global _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_SEQ_LEN
    global _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_IDS
    global _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_NGRAMS
    global _BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_SEGMENT_HARD_RATIO
    global _BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_IDS
    global _BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_NGRAMS
    global _BAND_LAYER_BUDGET_HARD_PREFILL_EXCLUDE_TOKEN_IDS
    global _BAND_LAYER_BUDGET_HARD_PREFILL_MIN_RELATIVE_POS
    global _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_RELATIVE_POS
    global _BAND_LAYER_BUDGET_HARD_PREFILL_START
    global _BAND_LAYER_BUDGET_HARD_PREFILL_END
    global _BAND_LAYER_BUDGET_HARD_DECODE_MIN_SEQ_LEN
    global _BAND_LAYER_BUDGET_HARD_DECODE_MAX_SEQ_LEN
    global _BAND_LAYER_BUDGET_HARD_DECODE_SEQ_LEN_SCOPE
    global _BAND_LAYER_BUDGET_HARD_DECODE_MIN_OFFSET
    global _BAND_LAYER_BUDGET_HARD_DECODE_MAX_OFFSET
    global _BAND_LAYER_BUDGET_HARD_DECODE_START
    global _BAND_LAYER_BUDGET_HARD_DECODE_END
    global _BAND_LAYER_BUDGET_HARD_DECODE_TOKEN_IDS
    global _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_START
    global _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_END
    global _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_START
    global _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_END
    global _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_ALPHA
    global _BAND_LAYER_BUDGET_HARD_LAYER_SENSITIVITY_ALPHA
    global _BAND_LAYER_BUDGET_LATE_CONCENTRATION_START
    global _BAND_LAYER_BUDGET_LATE_CONCENTRATION_THRESHOLD
    global _BAND_LAYER_BUDGET_LATE_RESCUE_START
    global _BAND_LAYER_BUDGET_LATE_RESCUE_CONCENTRATION_THRESHOLD
    global _BAND_LAYER_BUDGET_LATE_RESCUE_MIXED_MAX
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE1_TOKEN_NGRAMS
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_THRESHOLD
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_MIN_RANK_WEIGHT
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_START
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_END
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE1_PREFILL_SEGMENT_HARD_RATIO
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE2_TOKEN_NGRAMS
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_THRESHOLD
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_MIN_RANK_WEIGHT
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_START
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_END
    global _BAND_LAYER_BUDGET_PROMPT_PROFILE2_PREFILL_SEGMENT_HARD_RATIO
    global _DIEP_PRUNING_MODE, _DIEP_USE_GAMMA1, _BAN_LAMBDA, _BAN_K_MIN
    global _DIEP_GAMMA_ALPHA, _DIEP_THRESHOLD_CAP
    global _EAC_ALPHA, _MC_MOE_PROTECTION_RATIO, _BIASED_RENORM_KEEP_TOPN
    global _ROUTER_VALUE_BASE_K, _ROUTER_VALUE_QUOTA_RATIO
    global _ROUTER_VALUE_SELECTION_MODE, _ROUTER_VALUE_SWAP_DIRECTION
    global _DEBUG_ROUTING, _DEBUG_MAX_PRINTS, _DEBUG_PRINT_COUNT
    global _STATS_DIR, _STATS_FLUSH_INTERVAL
    global _STATS_TOTAL_SELECTED, _STATS_TOTAL_TOKENS, _STATS_TOTAL_CALLS
    global _STATS_TOTAL_PROTECTED_TOKENS
    global _STATS_DEVICE_ACCUM, _STATS_LAST_TOPK
    global _STATS_EAC_PREFILL_TOKENS, _STATS_EAC_SEGMENTS
    global _STATS_EAC_CALLS_WITH_SEGMENTS, _STATS_EAC_PRUNED_SELECTIONS
    global _STATS_EAC_DISABLED_EXPERTS
    global _STATS_BAN_TOTAL_K, _STATS_BAN_TOKENS
    global _STATS_BUDGET_EASY_TOKENS, _STATS_BUDGET_BASE_TOKENS
    global _STATS_BUDGET_HARD_TOKENS
    global _STATS_BUDGET_LAYER_COUNTS, _STATS_BUDGET_PROFILE_COUNTS
    global _STATS_BUDGET_STAGE_COUNTS
    global _STATS_BUDGET_HARD_RANK_WEIGHT_CANDIDATE
    global _STATS_BUDGET_HARD_RANK_WEIGHT_SELECTED
    global _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_CANDIDATE
    global _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_SELECTED
    global _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BY_LAYER
    global _STATS_BUDGET_HARD_SEQ_LEN_CANDIDATE
    global _STATS_BUDGET_HARD_SEQ_LEN_SELECTED
    global _STATS_BUDGET_HARD_DECODE_OFFSET_CANDIDATE
    global _STATS_BUDGET_HARD_DECODE_OFFSET_SELECTED

    method = method.lower()
    if method not in {
            "none", "naee", "dynamic_routing", "diep", "mc_moe", "eac_moe",
            "ban", "topk_biased_renorm", "layerwise_dynamic_routing",
            "budget_dynamic_routing", "layerbudget_dynamic_routing",
            "bandedlayerbudget_dynamic_routing",
            "bansensitivelayerbudget_dynamic_routing",
            "routervalue_dynamic_routing"
    }:
        raise ValueError(f"Unsupported expert pruning method: {method}")
    if naee_beta < 0:
        raise ValueError("--naee_beta must be non-negative")
    if naee_k_min < 1:
        raise ValueError("--naee_k_min must be >= 1")
    if not 0 < dynamic_routing_threshold <= 1:
        raise ValueError("--dynamic_routing_threshold must be in (0, 1]")
    if dynamic_routing_score_source not in {
            "raw",
            "renormalized",
            "next_rank_weight",
            "dropped_mass",
            "renorm_minus_next_rank",
            "renorm_minus_scaled_next_rank",
    }:
        raise ValueError(
            "--dynamic_routing_score_source must be raw, renormalized, "
            "next_rank_weight, dropped_mass, renorm_minus_next_rank, or "
            "renorm_minus_scaled_next_rank")
    if dynamic_routing_next_rank_penalty < 0:
        raise ValueError("--dynamic_routing_next_rank_penalty must be >= 0")
    if not 0 < layerwise_dynamic_base_threshold <= 1:
        raise ValueError(
            "--layerwise_dynamic_base_threshold must be in (0, 1]")
    if layerwise_dynamic_num_layers < 1:
        raise ValueError("--layerwise_dynamic_num_layers must be >= 1")
    if layerwise_dynamic_k_min < 1:
        raise ValueError("--layerwise_dynamic_k_min must be >= 1")
    if budget_dynamic_easy_k < 1:
        raise ValueError("--budget_dynamic_easy_k must be >= 1")
    if budget_dynamic_base_k < 1:
        raise ValueError("--budget_dynamic_base_k must be >= 1")
    if budget_dynamic_hard_k < 1:
        raise ValueError("--budget_dynamic_hard_k must be >= 1")
    if not 0 <= budget_dynamic_hard_threshold <= 1:
        raise ValueError("--budget_dynamic_hard_threshold must be in [0, 1]")
    if not 0 <= budget_dynamic_easy_threshold <= 1:
        raise ValueError("--budget_dynamic_easy_threshold must be in [0, 1]")
    if budget_dynamic_hard_threshold > budget_dynamic_easy_threshold:
        raise ValueError(
            "--budget_dynamic_hard_threshold must be <= "
            "--budget_dynamic_easy_threshold")
    if budget_dynamic_score_topn < 1:
        raise ValueError("--budget_dynamic_score_topn must be >= 1")
    if layer_budget_dynamic_num_layers < 1:
        raise ValueError("--layer_budget_dynamic_num_layers must be >= 1")
    if band_layer_budget_middle_start < 0:
        raise ValueError("--band_layer_budget_middle_start must be >= 0")
    if band_layer_budget_middle_end < band_layer_budget_middle_start:
        raise ValueError(
            "--band_layer_budget_middle_end must be >= "
            "--band_layer_budget_middle_start")
    per_layer_k = _parse_positive_int_list(
        band_layer_budget_per_layer_k,
        "--band_layer_budget_per_layer_k",
    )
    easy_layers = _parse_nonnegative_int_list(
        band_layer_budget_easy_layers,
        "--band_layer_budget_easy_layers",
    )
    hard_layers = _parse_nonnegative_int_list(
        band_layer_budget_hard_layers,
        "--band_layer_budget_hard_layers",
    )
    if (band_layer_budget_token_gate_ratio < 0
            and band_layer_budget_token_gate_ratio != -1):
        raise ValueError(
            "--band_layer_budget_token_gate_ratio must be in [0, 1] or -1")
    if band_layer_budget_token_gate_ratio > 1:
        raise ValueError(
            "--band_layer_budget_token_gate_ratio must be in [0, 1] or -1")
    if band_layer_budget_token_gate_ratio >= 0 and (
            not easy_layers or not hard_layers):
        raise ValueError(
            "--band_layer_budget_token_gate_ratio requires both explicit "
            "easy and hard layer lists")
    if band_layer_budget_extra_middle_start < -1:
        raise ValueError(
            "--band_layer_budget_extra_middle_start must be >= -1")
    if band_layer_budget_extra_middle_end < -1:
        raise ValueError(
            "--band_layer_budget_extra_middle_end must be >= -1")
    if ((band_layer_budget_extra_middle_start >= 0
            or band_layer_budget_extra_middle_end >= 0)
            and band_layer_budget_extra_middle_end
            < band_layer_budget_extra_middle_start):
        raise ValueError(
            "--band_layer_budget_extra_middle_end must be >= "
            "--band_layer_budget_extra_middle_start")
    if band_layer_budget_middle_phase not in {"all", "prefill", "decode"}:
        raise ValueError(
            "--band_layer_budget_middle_phase must be all, prefill, or "
            "decode")
    if band_layer_budget_easy_start < 0:
        raise ValueError("--band_layer_budget_easy_start must be >= 0")
    if band_layer_budget_easy_end < band_layer_budget_easy_start:
        raise ValueError(
            "--band_layer_budget_easy_end must be >= "
            "--band_layer_budget_easy_start")
    if band_layer_budget_easy_phase not in {"all", "prefill", "decode"}:
        raise ValueError(
            "--band_layer_budget_easy_phase must be all, prefill, or decode")
    if band_layer_budget_hard_phase not in {"all", "prefill", "decode"}:
        raise ValueError(
            "--band_layer_budget_hard_phase must be all, prefill, or decode")
    if not 0 <= band_layer_budget_easy_tail_threshold <= 1:
        raise ValueError(
            "--band_layer_budget_easy_tail_threshold must be in [0, 1]")
    if not 0 <= band_layer_budget_hard_min_rank_weight <= 1:
        raise ValueError(
            "--band_layer_budget_hard_min_rank_weight must be in [0, 1]")
    if not 0 <= band_layer_budget_hard_max_rank_weight <= 1:
        raise ValueError(
            "--band_layer_budget_hard_max_rank_weight must be in [0, 1]")
    if (band_layer_budget_hard_decode_min_rank_weight < 0
            and band_layer_budget_hard_decode_min_rank_weight != -1):
        raise ValueError(
            "--band_layer_budget_hard_decode_min_rank_weight must be in "
            "[0, 1] or -1")
    if band_layer_budget_hard_decode_min_rank_weight > 1:
        raise ValueError(
            "--band_layer_budget_hard_decode_min_rank_weight must be in "
            "[0, 1] or -1")
    if (band_layer_budget_mixed_rescue_min_rank_weight < 0
            and band_layer_budget_mixed_rescue_min_rank_weight != -1):
        raise ValueError(
            "--band_layer_budget_mixed_rescue_min_rank_weight must be in "
            "[0, 1] or -1")
    if band_layer_budget_mixed_rescue_min_rank_weight > 1:
        raise ValueError(
            "--band_layer_budget_mixed_rescue_min_rank_weight must be in "
            "[0, 1] or -1")
    if (band_layer_budget_mixed_rescue_max_concentration < 0
            and band_layer_budget_mixed_rescue_max_concentration != -1):
        raise ValueError(
            "--band_layer_budget_mixed_rescue_max_concentration must be in "
            "[0, 1] or -1")
    if band_layer_budget_mixed_rescue_max_concentration > 1:
        raise ValueError(
            "--band_layer_budget_mixed_rescue_max_concentration must be in "
            "[0, 1] or -1")
    if band_layer_budget_hard_prefill_max_tokens < -1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_tokens must be >= -1")
    if band_layer_budget_hard_prefill_max_segment_tokens < -1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_segment_tokens must be "
            ">= -1")
    if (band_layer_budget_hard_prefill_max_density < 0
            and band_layer_budget_hard_prefill_max_density != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_density must be in "
            "[0, 1] or -1")
    if band_layer_budget_hard_prefill_max_density > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_density must be in "
            "[0, 1] or -1")
    if (band_layer_budget_hard_prefill_min_density < 0
            and band_layer_budget_hard_prefill_min_density != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_min_density must be in "
            "[0, 1] or -1")
    if band_layer_budget_hard_prefill_min_density > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_min_density must be in "
            "[0, 1] or -1")
    if (band_layer_budget_hard_prefill_max_segment_hard_ratio < 0
            and band_layer_budget_hard_prefill_max_segment_hard_ratio != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_segment_hard_ratio must be "
            "in [0, 1] or -1")
    if band_layer_budget_hard_prefill_max_segment_hard_ratio > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_segment_hard_ratio must be "
            "in [0, 1] or -1")
    if band_layer_budget_hard_prefill_segment_cap_score not in {
            "concentration", "rank4", "rank4_over_topn"}:
        raise ValueError(
            "--band_layer_budget_hard_prefill_segment_cap_score must be "
            "concentration, rank4, or rank4_over_topn")
    if (band_layer_budget_hard_prefill_conditional_segment_hard_ratio < 0
            and
            band_layer_budget_hard_prefill_conditional_segment_hard_ratio
            != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_conditional_segment_hard_ratio "
            "must be in [0, 1] or -1")
    if band_layer_budget_hard_prefill_conditional_segment_hard_ratio > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_conditional_segment_hard_ratio "
            "must be in [0, 1] or -1")
    if (band_layer_budget_hard_prefill_conditional_min_density < 0
            and band_layer_budget_hard_prefill_conditional_min_density
            != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_conditional_min_density must "
            "be in [0, 1] or -1")
    if band_layer_budget_hard_prefill_conditional_min_density > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_conditional_min_density must "
            "be in [0, 1] or -1")
    if band_layer_budget_hard_prefill_conditional_min_seq_len < -1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_conditional_min_seq_len must "
            "be >= -1")
    conditional_token_ids = _parse_token_id_list(
        band_layer_budget_hard_prefill_conditional_token_ids)
    conditional_token_ngrams = _parse_token_ngram_list(
        band_layer_budget_hard_prefill_conditional_token_ngrams)
    if (band_layer_budget_hard_prefill_marker_segment_hard_ratio < 0
            and band_layer_budget_hard_prefill_marker_segment_hard_ratio
            != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_marker_segment_hard_ratio "
            "must be in [0, 1] or -1")
    if band_layer_budget_hard_prefill_marker_segment_hard_ratio > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_marker_segment_hard_ratio "
            "must be in [0, 1] or -1")
    marker_token_ids = _parse_token_id_list(
        band_layer_budget_hard_prefill_marker_token_ids)
    marker_token_ngrams = _parse_token_ngram_list(
        band_layer_budget_hard_prefill_marker_token_ngrams)
    exclude_token_ids = _parse_token_id_list(
        band_layer_budget_hard_prefill_exclude_token_ids)
    decode_token_ids = _parse_token_id_list(
        band_layer_budget_hard_decode_token_ids)
    if (band_layer_budget_hard_prefill_min_relative_pos < 0
            and band_layer_budget_hard_prefill_min_relative_pos != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_min_relative_pos must be "
            "in [0, 1] or -1")
    if band_layer_budget_hard_prefill_min_relative_pos > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_min_relative_pos must be "
            "in [0, 1] or -1")
    if (band_layer_budget_hard_prefill_max_relative_pos < 0
            and band_layer_budget_hard_prefill_max_relative_pos != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_relative_pos must be "
            "in [0, 1] or -1")
    if band_layer_budget_hard_prefill_max_relative_pos > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_relative_pos must be "
            "in [0, 1] or -1")
    if (band_layer_budget_hard_prefill_min_relative_pos >= 0
            and band_layer_budget_hard_prefill_max_relative_pos >= 0
            and band_layer_budget_hard_prefill_max_relative_pos
            < band_layer_budget_hard_prefill_min_relative_pos):
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_relative_pos must be >= "
            "--band_layer_budget_hard_prefill_min_relative_pos")
    if band_layer_budget_hard_prefill_start < -1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_start must be >= -1")
    if band_layer_budget_hard_prefill_end < -1:
        raise ValueError("--band_layer_budget_hard_prefill_end must be >= -1")
    if ((band_layer_budget_hard_prefill_start >= 0
         or band_layer_budget_hard_prefill_end >= 0)
            and band_layer_budget_hard_prefill_end
            < band_layer_budget_hard_prefill_start):
        raise ValueError(
            "--band_layer_budget_hard_prefill_end must be >= "
            "--band_layer_budget_hard_prefill_start")
    if band_layer_budget_hard_decode_min_seq_len < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_min_seq_len must be >= -1")
    if band_layer_budget_hard_decode_max_seq_len < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_max_seq_len must be >= -1")
    if (band_layer_budget_hard_decode_min_seq_len >= 0
            and band_layer_budget_hard_decode_max_seq_len >= 0
            and band_layer_budget_hard_decode_max_seq_len
            < band_layer_budget_hard_decode_min_seq_len):
        raise ValueError(
            "--band_layer_budget_hard_decode_max_seq_len must be >= "
            "--band_layer_budget_hard_decode_min_seq_len")
    if band_layer_budget_hard_decode_seq_len_scope not in {"all", "extra"}:
        raise ValueError(
            "--band_layer_budget_hard_decode_seq_len_scope must be all or "
            "extra")
    if band_layer_budget_hard_decode_min_offset < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_min_offset must be >= -1")
    if band_layer_budget_hard_decode_max_offset < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_max_offset must be >= -1")
    if (band_layer_budget_hard_decode_min_offset >= 0
            and band_layer_budget_hard_decode_max_offset >= 0
            and band_layer_budget_hard_decode_max_offset
            < band_layer_budget_hard_decode_min_offset):
        raise ValueError(
            "--band_layer_budget_hard_decode_max_offset must be >= "
            "--band_layer_budget_hard_decode_min_offset")
    if band_layer_budget_hard_decode_start < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_start must be >= -1")
    if band_layer_budget_hard_decode_end < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_end must be >= -1")
    if ((band_layer_budget_hard_decode_start >= 0
         or band_layer_budget_hard_decode_end >= 0)
            and band_layer_budget_hard_decode_end
            < band_layer_budget_hard_decode_start):
        raise ValueError(
            "--band_layer_budget_hard_decode_end must be >= "
            "--band_layer_budget_hard_decode_start")
    if band_layer_budget_hard_layer_prior_start < -1:
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_start must be >= -1")
    if band_layer_budget_hard_layer_prior_end < -1:
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_end must be >= -1")
    if ((band_layer_budget_hard_layer_prior_start >= 0
         or band_layer_budget_hard_layer_prior_end >= 0)
            and band_layer_budget_hard_layer_prior_end
            < band_layer_budget_hard_layer_prior_start):
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_end must be >= "
            "--band_layer_budget_hard_layer_prior_start")
    if band_layer_budget_hard_layer_prior_extra_start < -1:
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_extra_start must be >= -1")
    if band_layer_budget_hard_layer_prior_extra_end < -1:
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_extra_end must be >= -1")
    if ((band_layer_budget_hard_layer_prior_extra_start >= 0
         or band_layer_budget_hard_layer_prior_extra_end >= 0)
            and band_layer_budget_hard_layer_prior_extra_end
            < band_layer_budget_hard_layer_prior_extra_start):
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_extra_end must be >= "
            "--band_layer_budget_hard_layer_prior_extra_start")
    if band_layer_budget_hard_layer_prior_alpha < 0:
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_alpha must be >= 0")
    if band_layer_budget_hard_layer_sensitivity_alpha < 0:
        raise ValueError(
            "--band_layer_budget_hard_layer_sensitivity_alpha must be >= 0")
    if band_layer_budget_late_concentration_start < -1:
        raise ValueError(
            "--band_layer_budget_late_concentration_start must be >= -1")
    if (band_layer_budget_late_concentration_threshold < 0
            and band_layer_budget_late_concentration_threshold != -1):
        raise ValueError(
            "--band_layer_budget_late_concentration_threshold must be in "
            "[0, 1] or -1")
    if band_layer_budget_late_concentration_threshold > 1:
        raise ValueError(
            "--band_layer_budget_late_concentration_threshold must be in "
            "[0, 1] or -1")
    if band_layer_budget_late_rescue_start < -1:
        raise ValueError(
            "--band_layer_budget_late_rescue_start must be >= -1")
    if (band_layer_budget_late_rescue_concentration_threshold < 0
            and band_layer_budget_late_rescue_concentration_threshold != -1):
        raise ValueError(
            "--band_layer_budget_late_rescue_concentration_threshold must be "
            "in [0, 1] or -1")
    if band_layer_budget_late_rescue_concentration_threshold > 1:
        raise ValueError(
            "--band_layer_budget_late_rescue_concentration_threshold must be "
            "in [0, 1] or -1")
    if (band_layer_budget_late_rescue_mixed_max < 0
            and band_layer_budget_late_rescue_mixed_max != -1):
        raise ValueError(
            "--band_layer_budget_late_rescue_mixed_max must be in [0, 1] "
            "or -1")
    if band_layer_budget_late_rescue_mixed_max > 1:
        raise ValueError(
            "--band_layer_budget_late_rescue_mixed_max must be in [0, 1] "
            "or -1")
    profile1_ngrams = _parse_token_ngram_list(
        band_layer_budget_prompt_profile1_token_ngrams)
    profile2_ngrams = _parse_token_ngram_list(
        band_layer_budget_prompt_profile2_token_ngrams)
    for (
            profile_id,
            hard_threshold,
            hard_min_rank_weight,
            middle_start,
            middle_end,
            segment_ratio,
    ) in (
            (
                1,
                band_layer_budget_prompt_profile1_hard_threshold,
                band_layer_budget_prompt_profile1_hard_min_rank_weight,
                band_layer_budget_prompt_profile1_middle_start,
                band_layer_budget_prompt_profile1_middle_end,
                band_layer_budget_prompt_profile1_prefill_segment_hard_ratio,
            ),
            (
                2,
                band_layer_budget_prompt_profile2_hard_threshold,
                band_layer_budget_prompt_profile2_hard_min_rank_weight,
                band_layer_budget_prompt_profile2_middle_start,
                band_layer_budget_prompt_profile2_middle_end,
                band_layer_budget_prompt_profile2_prefill_segment_hard_ratio,
            ),
    ):
        if hard_threshold < 0 and hard_threshold != -1:
            raise ValueError(
                f"--band_layer_budget_prompt_profile{profile_id}_hard_threshold "
                "must be in [0, 1] or -1")
        if hard_threshold > 1:
            raise ValueError(
                f"--band_layer_budget_prompt_profile{profile_id}_hard_threshold "
                "must be in [0, 1] or -1")
        if hard_min_rank_weight < 0 and hard_min_rank_weight != -1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_hard_min_rank_weight must be in [0, 1] "
                "or -1")
        if hard_min_rank_weight > 1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_hard_min_rank_weight must be in [0, 1] "
                "or -1")
        if middle_start < -1:
            raise ValueError(
                f"--band_layer_budget_prompt_profile{profile_id}_middle_start "
                "must be >= -1")
        if middle_end < -1:
            raise ValueError(
                f"--band_layer_budget_prompt_profile{profile_id}_middle_end "
                "must be >= -1")
        if ((middle_start >= 0 or middle_end >= 0)
                and middle_end < middle_start):
            raise ValueError(
                f"--band_layer_budget_prompt_profile{profile_id}_middle_end "
                "must be >= the corresponding middle_start")
        if segment_ratio < 0 and segment_ratio != -1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_prefill_segment_hard_ratio must be in "
                "[0, 1] or -1")
        if segment_ratio > 1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_prefill_segment_hard_ratio must be in "
                "[0, 1] or -1")
    if diep_pruning_mode not in {"independent", "prefix"}:
        raise ValueError("--diep_pruning_mode must be independent or prefix")
    if not 0 <= ban_lambda <= 1:
        raise ValueError("--ban_lambda must be in [0, 1]")
    if ban_k_min < 1:
        raise ValueError("--ban_k_min must be >= 1")
    if not 0 < eac_alpha < 1:
        raise ValueError("--eac_alpha must be in (0, 1)")
    if not 0 < mc_moe_protection_ratio <= 1:
        raise ValueError("--mc_moe_protection_ratio must be in (0, 1]")
    if biased_renorm_keep_topn < 1:
        raise ValueError("--biased_renorm_keep_topn must be >= 1")
    if router_value_base_k < 1:
        raise ValueError("--router_value_base_k must be >= 1")
    if not 0 <= router_value_quota_ratio <= 1:
        raise ValueError("--router_value_quota_ratio must be in [0, 1]")
    if router_value_selection_mode not in {"threshold", "quota"}:
        raise ValueError(
            "--router_value_selection_mode must be threshold or quota")
    if router_value_swap_direction not in {
            "both", "promote_only", "demote_only"}:
        raise ValueError(
            "--router_value_swap_direction must be both, promote_only, "
            "or demote_only")
    if debug_max_prints < 0:
        raise ValueError("--expert_pruning_debug_max_prints must be >= 0")

    _ROUTING_METHOD = method
    _NAEE_BETA = naee_beta
    _NAEE_K_MIN = naee_k_min
    _DYNAMIC_ROUTING_THRESHOLD = dynamic_routing_threshold
    _DYNAMIC_ROUTING_SCORE_SOURCE = dynamic_routing_score_source
    _DYNAMIC_ROUTING_NEXT_RANK_PENALTY = dynamic_routing_next_rank_penalty
    _LAYERWISE_DYNAMIC_BASE_THRESHOLD = layerwise_dynamic_base_threshold
    _LAYERWISE_DYNAMIC_LAYER_ALPHA = layerwise_dynamic_layer_alpha
    _LAYERWISE_DYNAMIC_NUM_LAYERS = layerwise_dynamic_num_layers
    _LAYERWISE_DYNAMIC_K_MIN = layerwise_dynamic_k_min
    _BUDGET_DYNAMIC_EASY_K = budget_dynamic_easy_k
    _BUDGET_DYNAMIC_BASE_K = budget_dynamic_base_k
    _BUDGET_DYNAMIC_HARD_K = budget_dynamic_hard_k
    _BUDGET_DYNAMIC_EASY_THRESHOLD = budget_dynamic_easy_threshold
    _BUDGET_DYNAMIC_HARD_THRESHOLD = budget_dynamic_hard_threshold
    _BUDGET_DYNAMIC_SCORE_TOPN = budget_dynamic_score_topn
    _LAYER_BUDGET_DYNAMIC_EASY_LAYER_ALPHA = (
        layer_budget_dynamic_easy_layer_alpha)
    _LAYER_BUDGET_DYNAMIC_HARD_LAYER_ALPHA = (
        layer_budget_dynamic_hard_layer_alpha)
    _LAYER_BUDGET_DYNAMIC_NUM_LAYERS = layer_budget_dynamic_num_layers
    _BAND_LAYER_BUDGET_MIDDLE_START = band_layer_budget_middle_start
    _BAND_LAYER_BUDGET_MIDDLE_END = band_layer_budget_middle_end
    _BAND_LAYER_BUDGET_PER_LAYER_K = per_layer_k
    _BAND_LAYER_BUDGET_EASY_LAYERS = easy_layers
    _BAND_LAYER_BUDGET_HARD_LAYERS = hard_layers
    _BAND_LAYER_BUDGET_TOKEN_GATE_RATIO = (
        band_layer_budget_token_gate_ratio)
    _BAND_LAYER_BUDGET_EXTRA_MIDDLE_START = (
        band_layer_budget_extra_middle_start)
    _BAND_LAYER_BUDGET_EXTRA_MIDDLE_END = (
        band_layer_budget_extra_middle_end)
    _BAND_LAYER_BUDGET_MIDDLE_PHASE = band_layer_budget_middle_phase
    _BAND_LAYER_BUDGET_EASY_START = band_layer_budget_easy_start
    _BAND_LAYER_BUDGET_EASY_END = band_layer_budget_easy_end
    _BAND_LAYER_BUDGET_EASY_PHASE = band_layer_budget_easy_phase
    _BAND_LAYER_BUDGET_HARD_PHASE = band_layer_budget_hard_phase
    _BAND_LAYER_BUDGET_EASY_TAIL_THRESHOLD = (
        band_layer_budget_easy_tail_threshold)
    _BAND_LAYER_BUDGET_HARD_MIN_RANK_WEIGHT = (
        band_layer_budget_hard_min_rank_weight)
    _BAND_LAYER_BUDGET_HARD_MAX_RANK_WEIGHT = (
        band_layer_budget_hard_max_rank_weight)
    _BAND_LAYER_BUDGET_HARD_DECODE_MIN_RANK_WEIGHT = (
        band_layer_budget_hard_decode_min_rank_weight)
    _BAND_LAYER_BUDGET_MIXED_RESCUE_MIN_RANK_WEIGHT = (
        band_layer_budget_mixed_rescue_min_rank_weight)
    _BAND_LAYER_BUDGET_MIXED_RESCUE_MAX_CONCENTRATION = (
        band_layer_budget_mixed_rescue_max_concentration)
    _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_TOKENS = (
        band_layer_budget_hard_prefill_max_tokens)
    _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_TOKENS = (
        band_layer_budget_hard_prefill_max_segment_tokens)
    _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_DENSITY = (
        band_layer_budget_hard_prefill_max_density)
    _BAND_LAYER_BUDGET_HARD_PREFILL_MIN_DENSITY = (
        band_layer_budget_hard_prefill_min_density)
    _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_HARD_RATIO = (
        band_layer_budget_hard_prefill_max_segment_hard_ratio)
    _BAND_LAYER_BUDGET_HARD_PREFILL_SEGMENT_CAP_SCORE = (
        band_layer_budget_hard_prefill_segment_cap_score)
    _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_SEGMENT_HARD_RATIO = (
        band_layer_budget_hard_prefill_conditional_segment_hard_ratio)
    _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_DENSITY = (
        band_layer_budget_hard_prefill_conditional_min_density)
    _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_SEQ_LEN = (
        band_layer_budget_hard_prefill_conditional_min_seq_len)
    _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_IDS = (
        conditional_token_ids)
    _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_NGRAMS = (
        conditional_token_ngrams)
    _BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_SEGMENT_HARD_RATIO = (
        band_layer_budget_hard_prefill_marker_segment_hard_ratio)
    _BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_IDS = marker_token_ids
    _BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_NGRAMS = marker_token_ngrams
    _BAND_LAYER_BUDGET_HARD_PREFILL_EXCLUDE_TOKEN_IDS = exclude_token_ids
    _BAND_LAYER_BUDGET_HARD_PREFILL_MIN_RELATIVE_POS = (
        band_layer_budget_hard_prefill_min_relative_pos)
    _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_RELATIVE_POS = (
        band_layer_budget_hard_prefill_max_relative_pos)
    _BAND_LAYER_BUDGET_HARD_PREFILL_START = (
        band_layer_budget_hard_prefill_start)
    _BAND_LAYER_BUDGET_HARD_PREFILL_END = (
        band_layer_budget_hard_prefill_end)
    _BAND_LAYER_BUDGET_HARD_DECODE_MIN_SEQ_LEN = (
        band_layer_budget_hard_decode_min_seq_len)
    _BAND_LAYER_BUDGET_HARD_DECODE_MAX_SEQ_LEN = (
        band_layer_budget_hard_decode_max_seq_len)
    _BAND_LAYER_BUDGET_HARD_DECODE_SEQ_LEN_SCOPE = (
        band_layer_budget_hard_decode_seq_len_scope)
    _BAND_LAYER_BUDGET_HARD_DECODE_MIN_OFFSET = (
        band_layer_budget_hard_decode_min_offset)
    _BAND_LAYER_BUDGET_HARD_DECODE_MAX_OFFSET = (
        band_layer_budget_hard_decode_max_offset)
    _BAND_LAYER_BUDGET_HARD_DECODE_START = (
        band_layer_budget_hard_decode_start)
    _BAND_LAYER_BUDGET_HARD_DECODE_END = band_layer_budget_hard_decode_end
    _BAND_LAYER_BUDGET_HARD_DECODE_TOKEN_IDS = decode_token_ids
    _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_START = (
        band_layer_budget_hard_layer_prior_start)
    _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_END = (
        band_layer_budget_hard_layer_prior_end)
    _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_START = (
        band_layer_budget_hard_layer_prior_extra_start)
    _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_END = (
        band_layer_budget_hard_layer_prior_extra_end)
    _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_ALPHA = (
        band_layer_budget_hard_layer_prior_alpha)
    _BAND_LAYER_BUDGET_HARD_LAYER_SENSITIVITY_ALPHA = (
        band_layer_budget_hard_layer_sensitivity_alpha)
    _BAND_LAYER_BUDGET_LATE_CONCENTRATION_START = (
        band_layer_budget_late_concentration_start)
    _BAND_LAYER_BUDGET_LATE_CONCENTRATION_THRESHOLD = (
        band_layer_budget_late_concentration_threshold)
    _BAND_LAYER_BUDGET_LATE_RESCUE_START = (
        band_layer_budget_late_rescue_start)
    _BAND_LAYER_BUDGET_LATE_RESCUE_CONCENTRATION_THRESHOLD = (
        band_layer_budget_late_rescue_concentration_threshold)
    _BAND_LAYER_BUDGET_LATE_RESCUE_MIXED_MAX = (
        band_layer_budget_late_rescue_mixed_max)
    _BAND_LAYER_BUDGET_PROMPT_PROFILE1_TOKEN_NGRAMS = profile1_ngrams
    _BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_THRESHOLD = (
        band_layer_budget_prompt_profile1_hard_threshold)
    _BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_MIN_RANK_WEIGHT = (
        band_layer_budget_prompt_profile1_hard_min_rank_weight)
    _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_START = (
        band_layer_budget_prompt_profile1_middle_start)
    _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_END = (
        band_layer_budget_prompt_profile1_middle_end)
    _BAND_LAYER_BUDGET_PROMPT_PROFILE1_PREFILL_SEGMENT_HARD_RATIO = (
        band_layer_budget_prompt_profile1_prefill_segment_hard_ratio)
    _BAND_LAYER_BUDGET_PROMPT_PROFILE2_TOKEN_NGRAMS = profile2_ngrams
    _BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_THRESHOLD = (
        band_layer_budget_prompt_profile2_hard_threshold)
    _BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_MIN_RANK_WEIGHT = (
        band_layer_budget_prompt_profile2_hard_min_rank_weight)
    _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_START = (
        band_layer_budget_prompt_profile2_middle_start)
    _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_END = (
        band_layer_budget_prompt_profile2_middle_end)
    _BAND_LAYER_BUDGET_PROMPT_PROFILE2_PREFILL_SEGMENT_HARD_RATIO = (
        band_layer_budget_prompt_profile2_prefill_segment_hard_ratio)
    _DIEP_PRUNING_MODE = diep_pruning_mode
    _DIEP_USE_GAMMA1 = bool(diep_use_gamma1)
    _DIEP_GAMMA_ALPHA = float(diep_gamma_alpha)
    _DIEP_THRESHOLD_CAP = (None if diep_threshold_cap is None else
                           float(diep_threshold_cap))
    _BAN_LAMBDA = ban_lambda
    _BAN_K_MIN = ban_k_min
    _EAC_ALPHA = eac_alpha
    _MC_MOE_PROTECTION_RATIO = mc_moe_protection_ratio
    _BIASED_RENORM_KEEP_TOPN = biased_renorm_keep_topn
    _ROUTER_VALUE_BASE_K = router_value_base_k
    _ROUTER_VALUE_QUOTA_RATIO = router_value_quota_ratio
    _ROUTER_VALUE_SELECTION_MODE = router_value_selection_mode
    _ROUTER_VALUE_SWAP_DIRECTION = router_value_swap_direction
    if method == "routervalue_dynamic_routing":
        _load_router_value_artifact(
            router_value_artifact_path,
            router_value_base_k,
            router_value_quota_ratio,
        )
    _DEBUG_ROUTING = debug
    _DEBUG_MAX_PRINTS = debug_max_prints
    _DEBUG_PRINT_COUNT = 0
    stats_dir = os.environ.get("EXPERT_PRUNING_STATS_DIR")
    _STATS_DIR = Path(stats_dir) if stats_dir else None
    _STATS_FLUSH_INTERVAL = max(
        1,
        int(os.environ.get("EXPERT_PRUNING_STATS_FLUSH_INTERVAL",
                           str(_STATS_FLUSH_INTERVAL_DEFAULT))))
    _STATS_TOTAL_SELECTED = 0.0
    _STATS_TOTAL_TOKENS = 0
    _STATS_TOTAL_CALLS = 0
    _STATS_TOTAL_PROTECTED_TOKENS = 0
    _STATS_DEVICE_ACCUM = None
    _STATS_LAST_TOPK = None
    _STATS_EAC_PREFILL_TOKENS = 0
    _STATS_EAC_SEGMENTS = 0
    _STATS_EAC_CALLS_WITH_SEGMENTS = 0
    _STATS_EAC_PRUNED_SELECTIONS = 0
    _STATS_EAC_DISABLED_EXPERTS = 0
    _STATS_BAN_TOTAL_K = 0.0
    _STATS_BAN_TOKENS = 0
    _STATS_BUDGET_EASY_TOKENS = 0
    _STATS_BUDGET_BASE_TOKENS = 0
    _STATS_BUDGET_HARD_TOKENS = 0
    _STATS_BUDGET_LAYER_COUNTS = {}
    _STATS_BUDGET_PROFILE_COUNTS = {}
    _STATS_BUDGET_STAGE_COUNTS = {
        "prefill": [0, 0, 0],
        "decode": [0, 0, 0],
        "unknown": [0, 0, 0],
    }
    _STATS_BUDGET_HARD_RANK_WEIGHT_CANDIDATE = {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
        "hist": [0] * _stats_hist_size(
            _STATS_BUDGET_HARD_RANK_WEIGHT_BOUNDS),
    }
    _STATS_BUDGET_HARD_RANK_WEIGHT_SELECTED = {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
        "hist": [0] * _stats_hist_size(
            _STATS_BUDGET_HARD_RANK_WEIGHT_BOUNDS),
    }
    _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_CANDIDATE = {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
        "hist": [0] * _stats_hist_size(
            _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BOUNDS),
        "hard_tokens": 0,
        "segment_tokens": 0,
    }
    _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_SELECTED = {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
        "hist": [0] * _stats_hist_size(
            _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BOUNDS),
        "hard_tokens": 0,
        "segment_tokens": 0,
    }
    _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BY_LAYER = {}
    _STATS_BUDGET_HARD_SEQ_LEN_CANDIDATE = {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
        "hist": [0] * _stats_hist_size(_STATS_BUDGET_HARD_SEQ_LEN_BOUNDS),
    }
    _STATS_BUDGET_HARD_SEQ_LEN_SELECTED = {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
        "hist": [0] * _stats_hist_size(_STATS_BUDGET_HARD_SEQ_LEN_BOUNDS),
    }
    _STATS_BUDGET_HARD_DECODE_OFFSET_CANDIDATE = {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
        "hist": [0] * _stats_hist_size(
            _STATS_BUDGET_HARD_DECODE_OFFSET_BOUNDS),
    }
    _STATS_BUDGET_HARD_DECODE_OFFSET_SELECTED = {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
        "hist": [0] * _stats_hist_size(
            _STATS_BUDGET_HARD_DECODE_OFFSET_BOUNDS),
    }
    if method == "diep" and not diep_artifact_path:
        raise ValueError("--diep_artifact_path is required for DiEP")
    if method == "ban" and not ban_artifact_path:
        raise ValueError("--ban_artifact_path is required for Ban")
    _load_diep_artifact(diep_artifact_path)
    _load_ban_artifact(ban_artifact_path)


def _token_expert_indices(
    num_tokens: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.arange(
        0,
        num_tokens * topk,
        dtype=torch.int32,
        device=device,
    ).reshape(num_tokens, topk)


def _topk_softmax(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    sorted: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.size(0) != gating_output.size(0):
        raise AssertionError("Number of tokens mismatch")

    scores = torch.softmax(gating_output.float(), dim=-1)
    return torch.topk(scores, k=topk, dim=-1, sorted=sorted)


def _renormalize_topk_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    return topk_weights / topk_weights.sum(dim=-1, keepdim=True)


def _biased_renormalize_topk_weights(
    topk_weights: torch.Tensor,
    keep_topn: int,
) -> torch.Tensor:
    """Keep tail ranks unchanged and assign missing mass to leading ranks."""
    topk = topk_weights.size(1)
    keep_topn = min(max(1, keep_topn), topk)
    if keep_topn >= topk:
        return _renormalize_topk_weights(topk_weights)

    head = topk_weights[:, :keep_topn]
    tail = topk_weights[:, keep_topn:]
    tail_sum = tail.sum(dim=-1, keepdim=True)
    head_sum = head.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    head_scale = (1.0 - tail_sum).clamp_min(0.0) / head_sum
    return torch.cat((head * head_scale, tail), dim=-1)


def _naee_keep_mask(topk_weights: torch.Tensor) -> torch.Tensor:
    topk = topk_weights.size(1)
    ranks = torch.arange(topk, device=topk_weights.device).view(1, topk)
    k_min = min(_NAEE_K_MIN, topk)
    keep_by_min = ranks < k_min
    threshold = topk_weights[:, :1] * _NAEE_BETA
    prune_candidate = (ranks >= k_min) & (topk_weights < threshold)
    has_previous_prune = prune_candidate.int().cumsum(dim=-1) > 0
    return keep_by_min | ~has_previous_prune


def _current_protected_mask_for_tokens(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if _CURRENT_PROTECTED_TOKEN_MASK is None:
        return None
    if _CURRENT_PROTECTED_TOKEN_MASK.numel() != num_tokens:
        return None
    return _CURRENT_PROTECTED_TOKEN_MASK.to(device=device, dtype=torch.bool)


def _current_eac_prefill_mask_for_tokens(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if _CURRENT_EAC_PREFILL_MASK is None:
        return None
    if _CURRENT_EAC_PREFILL_MASK.numel() != num_tokens:
        return None
    return _CURRENT_EAC_PREFILL_MASK.to(device=device, dtype=torch.bool)


def _current_prefill_mask_for_tokens(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    prefill_mask = _current_eac_prefill_mask_for_tokens(num_tokens, device)
    if prefill_mask is not None:
        return prefill_mask
    if _CURRENT_TOKEN_SEGMENTS is None:
        return None

    mask = torch.zeros(num_tokens, device=device, dtype=torch.bool)
    for start, end in _CURRENT_TOKEN_SEGMENTS:
        start = max(0, min(int(start), num_tokens))
        end = max(0, min(int(end), num_tokens))
        if end > start:
            mask[start:end] = True
    return mask


def _band_layer_budget_phase_mask(
        phase: str,
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if phase == "all":
        return None

    prefill_mask = _current_prefill_mask_for_tokens(num_tokens, device)
    if prefill_mask is None:
        if phase == "decode":
            return torch.ones(num_tokens, device=device, dtype=torch.bool)
        return torch.zeros(num_tokens, device=device, dtype=torch.bool)

    if phase == "prefill":
        return prefill_mask
    return ~prefill_mask


def _apply_prefill_only_mask(
        mask: torch.Tensor,
        prefill_filter_mask: Optional[torch.Tensor],
        num_tokens: int,
        device: torch.device) -> torch.Tensor:
    if prefill_filter_mask is None:
        return mask
    prefill_phase_mask = _band_layer_budget_phase_mask(
        "prefill",
        num_tokens,
        device,
    )
    if prefill_phase_mask is None:
        return mask & prefill_filter_mask
    return mask & ((~prefill_phase_mask) | prefill_filter_mask)


def _band_layer_budget_prefill_length_mask(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_TOKENS < 0:
        return None
    if _CURRENT_TOKEN_SEGMENT_SEQ_LENS is not None:
        mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
        for start, end, seq_len in _CURRENT_TOKEN_SEGMENT_SEQ_LENS:
            start = max(0, min(int(start), num_tokens))
            end = max(0, min(int(end), num_tokens))
            if end <= start:
                continue
            if int(seq_len) > _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_TOKENS:
                mask[start:end] = False
        return mask
    if _CURRENT_TOKEN_SEGMENTS is None:
        return None

    mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
    for start, end in _CURRENT_TOKEN_SEGMENTS:
        start = max(0, min(int(start), num_tokens))
        end = max(0, min(int(end), num_tokens))
        if end <= start:
            continue
        if end - start > _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_TOKENS:
            mask[start:end] = False
    return mask


def _band_layer_budget_prefill_density_mask(
        candidate_mask: torch.Tensor,
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if (_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_DENSITY < 0
            and _BAND_LAYER_BUDGET_HARD_PREFILL_MIN_DENSITY < 0):
        return None
    if _CURRENT_TOKEN_SEGMENTS is None:
        return None

    max_threshold = float(_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_DENSITY)
    min_threshold = float(_BAND_LAYER_BUDGET_HARD_PREFILL_MIN_DENSITY)
    mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
    for start, end in _CURRENT_TOKEN_SEGMENTS:
        start = max(0, min(int(start), num_tokens))
        end = max(0, min(int(end), num_tokens))
        if end <= start:
            continue
        segment = slice(start, end)
        segment_tokens = float(end - start)
        hard_tokens = candidate_mask[segment].sum()
        allow_segment = torch.ones((), device=device, dtype=torch.bool)
        if max_threshold >= 0:
            allow_segment = allow_segment & (
                hard_tokens <= max_threshold * segment_tokens)
        if min_threshold >= 0:
            allow_segment = allow_segment & (
                hard_tokens >= min_threshold * segment_tokens)
        mask[segment] = allow_segment
    return mask


def _band_layer_budget_prefill_relative_position_mask(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    min_pos = float(_BAND_LAYER_BUDGET_HARD_PREFILL_MIN_RELATIVE_POS)
    max_pos = float(_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_RELATIVE_POS)
    if min_pos < 0 and max_pos < 0:
        return None
    if _CURRENT_TOKEN_SEGMENTS is None:
        return None

    mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
    for start, end in _CURRENT_TOKEN_SEGMENTS:
        start = max(0, min(int(start), num_tokens))
        end = max(0, min(int(end), num_tokens))
        if end <= start:
            continue
        segment_len = end - start
        if segment_len <= 1:
            rel_pos = torch.zeros(segment_len, device=device)
        else:
            rel_pos = (
                torch.arange(segment_len, device=device, dtype=torch.float32)
                / float(segment_len - 1))
        segment_mask = torch.ones(segment_len, device=device, dtype=torch.bool)
        if min_pos >= 0:
            segment_mask = segment_mask & (rel_pos >= min_pos)
        if max_pos >= 0:
            segment_mask = segment_mask & (rel_pos <= max_pos)
        mask[start:end] = segment_mask
    return mask


def _band_layer_budget_prefill_exclude_token_id_mask(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if not _BAND_LAYER_BUDGET_HARD_PREFILL_EXCLUDE_TOKEN_IDS:
        return None
    if _CURRENT_TOKEN_SEGMENTS is None or _CURRENT_INPUT_TOKEN_IDS is None:
        return None

    token_ids = _CURRENT_INPUT_TOKEN_IDS
    token_count = int(token_ids.numel())
    if token_count < int(num_tokens):
        return None
    if token_count > int(num_tokens):
        token_ids = token_ids[-int(num_tokens):]

    mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
    exclude_tensor = torch.tensor(
        _BAND_LAYER_BUDGET_HARD_PREFILL_EXCLUDE_TOKEN_IDS,
        device=device,
        dtype=token_ids.dtype,
    )
    token_ids = token_ids.to(device=device, non_blocking=True)
    excluded = torch.isin(token_ids, exclude_tensor)
    if not bool(excluded.any().item()):
        return mask

    for start, end in _CURRENT_TOKEN_SEGMENTS:
        start = max(0, min(int(start), num_tokens))
        end = max(0, min(int(end), num_tokens))
        if end <= start:
            continue
        mask[start:end] = ~excluded[start:end]
    return mask


def _band_layer_budget_decode_token_id_mask(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if not _BAND_LAYER_BUDGET_HARD_DECODE_TOKEN_IDS:
        return None
    if _CURRENT_INPUT_TOKEN_IDS is None:
        return None

    token_ids = _CURRENT_INPUT_TOKEN_IDS
    token_count = int(token_ids.numel())
    if token_count < int(num_tokens):
        return None
    if token_count > int(num_tokens):
        token_ids = token_ids[-int(num_tokens):]

    token_ids = token_ids.to(device=device, non_blocking=True)
    allowed = torch.zeros_like(token_ids, dtype=torch.bool, device=device)
    for token_id in _BAND_LAYER_BUDGET_HARD_DECODE_TOKEN_IDS:
        allowed = allowed | (token_ids == int(token_id))
    decode_phase_mask = _band_layer_budget_phase_mask(
        "decode",
        num_tokens,
        device,
    )
    if decode_phase_mask is None:
        return allowed
    return (~decode_phase_mask) | allowed


def _band_layer_budget_segment_has_conditional_token_marker(
        start: int,
        end: int,
        num_tokens: int,
        device: torch.device,
        marker_ids: Optional[tuple[int, ...]] = None,
        marker_ngrams: Optional[tuple[tuple[int, ...], ...]] = None) -> bool:
    use_conditional_defaults = marker_ids is None and marker_ngrams is None
    if use_conditional_defaults:
        marker_ids = _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_IDS
        marker_ngrams = (
            _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_NGRAMS)
    else:
        marker_ids = marker_ids or ()
        marker_ngrams = marker_ngrams or ()
    if not marker_ids and not marker_ngrams:
        return use_conditional_defaults
    if _CURRENT_INPUT_TOKEN_IDS is None:
        return False
    token_ids = _CURRENT_INPUT_TOKEN_IDS
    if int(token_ids.numel()) < int(num_tokens):
        return False
    segment_token_ids = token_ids[start:end]
    if segment_token_ids.numel() == 0:
        return False
    segment_token_ids = segment_token_ids.to(device=device, non_blocking=True)
    if marker_ids:
        marker_tensor = torch.tensor(
            marker_ids,
            device=device,
            dtype=segment_token_ids.dtype,
        )
        if bool(torch.isin(segment_token_ids, marker_tensor).any().item()):
            return True
    if not marker_ngrams:
        return False

    segment_tokens = int(segment_token_ids.numel())
    for ngram in marker_ngrams:
        ngram_len = len(ngram)
        if ngram_len == 0 or segment_tokens < ngram_len:
            continue
        ngram_tensor = torch.tensor(
            ngram,
            device=device,
            dtype=segment_token_ids.dtype,
        )
        windows = segment_token_ids.unfold(0, ngram_len, 1)
        if bool((windows == ngram_tensor).all(dim=1).any().item()):
            return True
    return False


def _band_layer_budget_prefill_segment_hard_ratio_mask(
        candidate_mask: torch.Tensor,
        difficulty_score: torch.Tensor,
        num_tokens: int,
        device: torch.device,
        base_max_ratio_override: float | torch.Tensor = -1.0,
        largest: bool = False,
) -> Optional[torch.Tensor]:
    ratio_override_tensor = (
        base_max_ratio_override
        if isinstance(base_max_ratio_override, torch.Tensor) else None)
    if ratio_override_tensor is not None:
        base_max_ratio = -1.0
    elif base_max_ratio_override >= 0:
        base_max_ratio = float(base_max_ratio_override)
    else:
        base_max_ratio = float(
            _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_HARD_RATIO)
    conditional_max_ratio = float(
        _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_SEGMENT_HARD_RATIO)
    marker_max_ratio = float(
        _BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_SEGMENT_HARD_RATIO)
    if base_max_ratio < 0 and conditional_max_ratio < 0 and marker_max_ratio < 0:
        return None
    if _CURRENT_TOKEN_SEGMENTS is None:
        return None

    segment_seq_lens: dict[tuple[int, int], int] = {}
    if _CURRENT_TOKEN_SEGMENT_SEQ_LENS is not None:
        for seq_start, seq_end, seq_len in _CURRENT_TOKEN_SEGMENT_SEQ_LENS:
            segment_seq_lens[(int(seq_start), int(seq_end))] = int(seq_len)

    mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
    for start, end in _CURRENT_TOKEN_SEGMENTS:
        start = max(0, min(int(start), num_tokens))
        end = max(0, min(int(end), num_tokens))
        if end <= start:
            continue
        segment = slice(start, end)
        candidate_indices = torch.nonzero(
            candidate_mask[segment],
            as_tuple=False,
        ).flatten()
        candidate_count = int(candidate_indices.numel())
        if candidate_count == 0:
            continue
        segment_tokens = end - start
        max_ratio = base_max_ratio
        if ratio_override_tensor is not None:
            segment_ratios = ratio_override_tensor[segment]
            if int(segment_ratios.numel()) > 0:
                valid_ratios = segment_ratios[segment_ratios >= 0]
                if int(valid_ratios.numel()) > 0:
                    max_ratio = float(valid_ratios.max().item())
        if conditional_max_ratio >= 0:
            use_conditional = True
            min_density = float(
                _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_DENSITY)
            if min_density >= 0:
                density = candidate_count / float(segment_tokens)
                use_conditional = use_conditional and density >= min_density
            min_seq_len = int(
                _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_SEQ_LEN)
            if min_seq_len >= 0:
                seq_len = segment_seq_lens.get((start, end), segment_tokens)
                use_conditional = use_conditional and seq_len >= min_seq_len
            use_conditional = use_conditional and (
                _band_layer_budget_segment_has_conditional_token_marker(
                    start,
                    end,
                    num_tokens,
                    device,
                ))
            if use_conditional:
                max_ratio = conditional_max_ratio
        if (marker_max_ratio >= 0
                and _band_layer_budget_segment_has_conditional_token_marker(
                    start,
                    end,
                    num_tokens,
                    device,
                    _BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_IDS,
                    _BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_NGRAMS,
                )):
            max_ratio = marker_max_ratio
        if max_ratio < 0:
            continue
        if max_ratio == 0:
            mask[segment] = False
            continue

        max_hard = int(max_ratio * segment_tokens + 0.999999)
        max_hard = max(1, min(max_hard, segment_tokens))
        if candidate_count <= max_hard:
            continue

        segment_mask = torch.zeros(segment_tokens,
                                   device=device,
                                   dtype=torch.bool)
        candidate_scores = difficulty_score[segment][candidate_indices]
        selected_offsets = torch.topk(
            candidate_scores,
            k=max_hard,
            largest=largest,
            sorted=False,
        ).indices
        segment_mask[candidate_indices[selected_offsets]] = True
        mask[segment] = segment_mask
    return mask


def _band_layer_budget_active_profile_overrides(
) -> tuple[int, int, float, float]:
    middle_start = _BAND_LAYER_BUDGET_MIDDLE_START
    middle_end = _BAND_LAYER_BUDGET_MIDDLE_END
    hard_threshold = _BUDGET_DYNAMIC_HARD_THRESHOLD
    prefill_segment_hard_ratio = -1.0

    if _CURRENT_PROMPT_PROFILE == 1:
        if _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_START >= 0:
            middle_start = _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_START
        if _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_END >= 0:
            middle_end = _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_END
        if _BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_THRESHOLD >= 0:
            hard_threshold = (
                _BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_THRESHOLD)
        if (_BAND_LAYER_BUDGET_PROMPT_PROFILE1_PREFILL_SEGMENT_HARD_RATIO
                >= 0):
            prefill_segment_hard_ratio = (
                _BAND_LAYER_BUDGET_PROMPT_PROFILE1_PREFILL_SEGMENT_HARD_RATIO)
    elif _CURRENT_PROMPT_PROFILE == 2:
        if _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_START >= 0:
            middle_start = _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_START
        if _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_END >= 0:
            middle_end = _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_END
        if _BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_THRESHOLD >= 0:
            hard_threshold = (
                _BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_THRESHOLD)
        if (_BAND_LAYER_BUDGET_PROMPT_PROFILE2_PREFILL_SEGMENT_HARD_RATIO
                >= 0):
            prefill_segment_hard_ratio = (
                _BAND_LAYER_BUDGET_PROMPT_PROFILE2_PREFILL_SEGMENT_HARD_RATIO)

    return middle_start, middle_end, hard_threshold, prefill_segment_hard_ratio


def _current_prompt_profile_ids_for_tokens(
        num_tokens: int,
        device: torch.device) -> torch.Tensor:
    if (_CURRENT_PROMPT_PROFILE_IDS is not None
            and int(_CURRENT_PROMPT_PROFILE_IDS.numel()) == int(num_tokens)):
        return _CURRENT_PROMPT_PROFILE_IDS.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
    return torch.full(
        (num_tokens, ),
        int(_CURRENT_PROMPT_PROFILE),
        device=device,
        dtype=torch.long,
    )


def _profile_layer_band_contains(
        profile_id: int,
        layer_index: Optional[int],
        default_start: int,
        default_end: int,
) -> bool:
    if layer_index is None:
        return False
    start = default_start
    end = default_end
    if profile_id == 1:
        if _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_START >= 0:
            start = _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_START
        if _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_END >= 0:
            end = _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_END
    elif profile_id == 2:
        if _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_START >= 0:
            start = _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_START
        if _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_END >= 0:
            end = _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_END
    return start <= layer_index <= end


def _band_layer_budget_token_middle_mask(
        profile_ids: torch.Tensor,
        layer_index: Optional[int],
        num_tokens: int,
        device: torch.device) -> torch.Tensor:
    default_in_band = _profile_layer_band_contains(
        0,
        layer_index,
        _BAND_LAYER_BUDGET_MIDDLE_START,
        _BAND_LAYER_BUDGET_MIDDLE_END,
    )
    if (layer_index is not None
            and _BAND_LAYER_BUDGET_EXTRA_MIDDLE_START >= 0
            and _BAND_LAYER_BUDGET_EXTRA_MIDDLE_START <= layer_index
            <= _BAND_LAYER_BUDGET_EXTRA_MIDDLE_END):
        default_in_band = True
    mask = torch.full(
        (num_tokens, ),
        default_in_band,
        device=device,
        dtype=torch.bool,
    )
    if _BAND_LAYER_BUDGET_PROMPT_PROFILE1_TOKEN_NGRAMS:
        profile1_in_band = _profile_layer_band_contains(
            1,
            layer_index,
            _BAND_LAYER_BUDGET_MIDDLE_START,
            _BAND_LAYER_BUDGET_MIDDLE_END,
        )
        mask = torch.where(
            profile_ids == 1,
            torch.full_like(mask, profile1_in_band),
            mask,
        )
    if _BAND_LAYER_BUDGET_PROMPT_PROFILE2_TOKEN_NGRAMS:
        profile2_in_band = _profile_layer_band_contains(
            2,
            layer_index,
            _BAND_LAYER_BUDGET_MIDDLE_START,
            _BAND_LAYER_BUDGET_MIDDLE_END,
        )
        mask = torch.where(
            profile_ids == 2,
            torch.full_like(mask, profile2_in_band),
            mask,
        )
    phase_mask = _band_layer_budget_phase_mask(
        _BAND_LAYER_BUDGET_MIDDLE_PHASE,
        num_tokens,
        device,
    )
    if phase_mask is not None:
        mask = mask & phase_mask
    return mask


def _band_layer_budget_token_hard_thresholds(
        profile_ids: torch.Tensor,
        base_threshold: float,
        dtype: torch.dtype,
        device: torch.device) -> torch.Tensor:
    thresholds = torch.full(
        profile_ids.shape,
        float(base_threshold),
        device=device,
        dtype=dtype,
    )
    if _BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_THRESHOLD >= 0:
        thresholds = torch.where(
            profile_ids == 1,
            torch.full_like(
                thresholds,
                float(_BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_THRESHOLD),
            ),
            thresholds,
        )
    if _BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_THRESHOLD >= 0:
        thresholds = torch.where(
            profile_ids == 2,
            torch.full_like(
                thresholds,
                float(_BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_THRESHOLD),
            ),
            thresholds,
        )
    return thresholds


def _band_layer_budget_token_hard_min_rank_weights(
        profile_ids: torch.Tensor,
        base_min_rank_weight: float,
        dtype: torch.dtype,
        device: torch.device) -> torch.Tensor:
    min_rank_weights = torch.full(
        profile_ids.shape,
        float(base_min_rank_weight),
        device=device,
        dtype=dtype,
    )
    if _BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_MIN_RANK_WEIGHT >= 0:
        min_rank_weights = torch.where(
            profile_ids == 1,
            torch.full_like(
                min_rank_weights,
                float(
                    _BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_MIN_RANK_WEIGHT),
            ),
            min_rank_weights,
        )
    if _BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_MIN_RANK_WEIGHT >= 0:
        min_rank_weights = torch.where(
            profile_ids == 2,
            torch.full_like(
                min_rank_weights,
                float(
                    _BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_MIN_RANK_WEIGHT),
            ),
            min_rank_weights,
        )
    return min_rank_weights


def _band_layer_budget_token_prefill_segment_hard_ratios(
        profile_ids: torch.Tensor,
        device: torch.device) -> Optional[torch.Tensor]:
    if (_BAND_LAYER_BUDGET_PROMPT_PROFILE1_PREFILL_SEGMENT_HARD_RATIO < 0
            and _BAND_LAYER_BUDGET_PROMPT_PROFILE2_PREFILL_SEGMENT_HARD_RATIO
            < 0):
        return None
    ratios = torch.full(
        profile_ids.shape,
        -1.0,
        device=device,
        dtype=torch.float32,
    )
    if _BAND_LAYER_BUDGET_PROMPT_PROFILE1_PREFILL_SEGMENT_HARD_RATIO >= 0:
        ratios = torch.where(
            profile_ids == 1,
            torch.full_like(
                ratios,
                float(
                    _BAND_LAYER_BUDGET_PROMPT_PROFILE1_PREFILL_SEGMENT_HARD_RATIO),
            ),
            ratios,
        )
    if _BAND_LAYER_BUDGET_PROMPT_PROFILE2_PREFILL_SEGMENT_HARD_RATIO >= 0:
        ratios = torch.where(
            profile_ids == 2,
            torch.full_like(
                ratios,
                float(
                    _BAND_LAYER_BUDGET_PROMPT_PROFILE2_PREFILL_SEGMENT_HARD_RATIO),
            ),
            ratios,
        )
    return ratios


def _band_layer_budget_prefill_segment_length_mask(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_TOKENS < 0:
        return None
    if _CURRENT_TOKEN_SEGMENTS is None:
        return None

    max_tokens = int(_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_TOKENS)
    mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
    for start, end in _CURRENT_TOKEN_SEGMENTS:
        start = max(0, min(int(start), num_tokens))
        end = max(0, min(int(end), num_tokens))
        if end <= start:
            continue
        if end - start > max_tokens:
            mask[start:end] = False
    return mask


def _current_token_seq_lens_for_tokens(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if _CURRENT_TOKEN_SEQ_LENS is None:
        return None
    if int(_CURRENT_TOKEN_SEQ_LENS.numel()) != int(num_tokens):
        return None
    return _CURRENT_TOKEN_SEQ_LENS.to(device=device, dtype=torch.long)


def _current_token_decode_offsets_for_tokens(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if _CURRENT_TOKEN_DECODE_OFFSETS is None:
        return None
    if int(_CURRENT_TOKEN_DECODE_OFFSETS.numel()) != int(num_tokens):
        return None
    return _CURRENT_TOKEN_DECODE_OFFSETS.to(device=device, dtype=torch.long)


def _band_layer_budget_decode_seq_len_mask(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if (_BAND_LAYER_BUDGET_HARD_DECODE_MIN_SEQ_LEN < 0
            and _BAND_LAYER_BUDGET_HARD_DECODE_MAX_SEQ_LEN < 0):
        return None
    seq_lens = _current_token_seq_lens_for_tokens(num_tokens, device)
    if seq_lens is None:
        return None
    mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
    if _BAND_LAYER_BUDGET_HARD_DECODE_MIN_SEQ_LEN >= 0:
        mask = mask & (seq_lens >= _BAND_LAYER_BUDGET_HARD_DECODE_MIN_SEQ_LEN)
    if _BAND_LAYER_BUDGET_HARD_DECODE_MAX_SEQ_LEN >= 0:
        mask = mask & (seq_lens <= _BAND_LAYER_BUDGET_HARD_DECODE_MAX_SEQ_LEN)
    return mask


def _band_layer_budget_decode_offset_mask(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if (_BAND_LAYER_BUDGET_HARD_DECODE_MIN_OFFSET < 0
            and _BAND_LAYER_BUDGET_HARD_DECODE_MAX_OFFSET < 0):
        return None
    offsets = _current_token_decode_offsets_for_tokens(num_tokens, device)
    if offsets is None:
        return None
    mask = offsets >= 0
    if _BAND_LAYER_BUDGET_HARD_DECODE_MIN_OFFSET >= 0:
        mask = (
            mask
            & (offsets >= _BAND_LAYER_BUDGET_HARD_DECODE_MIN_OFFSET))
    if _BAND_LAYER_BUDGET_HARD_DECODE_MAX_OFFSET >= 0:
        mask = (
            mask
            & (offsets <= _BAND_LAYER_BUDGET_HARD_DECODE_MAX_OFFSET))
    return mask


def _band_layer_budget_easy_phase_mask(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    return _band_layer_budget_phase_mask(
        _BAND_LAYER_BUDGET_EASY_PHASE,
        num_tokens,
        device,
    )


def _band_layer_budget_hard_layer_prior_boost(
        layer_index: Optional[int]) -> float:
    if layer_index is None:
        return 0.0
    alpha = float(_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_ALPHA)
    if alpha <= 0:
        return 0.0

    in_prior_band = (
        _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_START >= 0
        and _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_START <= layer_index
        <= _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_END)
    in_extra_prior_band = (
        _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_START >= 0
        and _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_START <= layer_index
        <= _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_END)
    if in_prior_band or in_extra_prior_band:
        return alpha
    return 0.0


def _maybe_print_naee_debug(
    topk_weights_before_pruning: torch.Tensor,
    topk_weights_after_pruning: torch.Tensor,
    keep_mask: torch.Tensor,
) -> None:
    global _DEBUG_PRINT_COUNT

    if not _DEBUG_ROUTING or _DEBUG_PRINT_COUNT >= _DEBUG_MAX_PRINTS:
        return

    with torch.no_grad():
        keep_counts = keep_mask.sum(dim=-1).float()
        sample_size = min(3, topk_weights_after_pruning.size(0))
        before = topk_weights_before_pruning[:sample_size].detach().cpu()
        after = topk_weights_after_pruning[:sample_size].detach().cpu()
        counts = keep_counts[:sample_size].detach().cpu()
        print(
            "[expert_pruning][NAEE] "
            f"beta={_NAEE_BETA} k_min={_NAEE_K_MIN} "
            f"topk={topk_weights_after_pruning.size(1)} "
            f"avg_keep={keep_counts.mean().item():.4f} "
            f"min_keep={keep_counts.min().item():.0f} "
            f"max_keep={keep_counts.max().item():.0f} "
            f"sample_keep={counts.tolist()} "
            f"sample_before={before.tolist()} "
            f"sample_after={after.tolist()}",
            flush=True,
        )
    _DEBUG_PRINT_COUNT += 1


def _budget_rank_weight_payload(stats: dict) -> dict:
    count = int(stats["count"])
    return {
        "count": count,
        "sum": float(stats["sum"]),
        "mean": float(stats["sum"]) / count if count else 0.0,
        "min": stats["min"],
        "max": stats["max"],
        "hist": [int(v) for v in stats["hist"]],
    }


def _budget_seq_len_payload(stats: dict) -> dict:
    count = int(stats["count"])
    return {
        "count": count,
        "sum": float(stats["sum"]),
        "mean": float(stats["sum"]) / count if count else 0.0,
        "min": stats["min"],
        "max": stats["max"],
        "hist": [int(v) for v in stats["hist"]],
    }


def _budget_segment_density_payload(stats: dict) -> dict:
    count = int(stats["count"])
    segment_tokens = int(stats.get("segment_tokens", 0))
    hard_tokens = int(stats.get("hard_tokens", 0))
    return {
        "count": count,
        "sum": float(stats["sum"]),
        "mean": float(stats["sum"]) / count if count else 0.0,
        "token_weighted_mean": (
            hard_tokens / segment_tokens if segment_tokens else 0.0),
        "min": stats["min"],
        "max": stats["max"],
        "hist": [int(v) for v in stats["hist"]],
        "hard_tokens": hard_tokens,
        "segment_tokens": segment_tokens,
    }


def _device_stats_accumulators(device: torch.device) -> dict:
    """Device-resident counters, so recording a call needs no host synchronisation."""
    global _STATS_DEVICE_ACCUM

    accum = _STATS_DEVICE_ACCUM
    if accum is not None:
        if accum["device"] == device:
            return accum
        # A worker should only ever see one device, but if that changes, fold the old
        # counters into the Python totals before switching.
        _drain_device_stats()
    _STATS_DEVICE_ACCUM = {
        "device": device,
        "selected": torch.zeros((), dtype=torch.float64, device=device),
        "protected": torch.zeros((), dtype=torch.float64, device=device),
    }
    return _STATS_DEVICE_ACCUM


def _drain_device_stats() -> None:
    """Fold the device counters into the Python totals, costing one synchronisation."""
    global _STATS_DEVICE_ACCUM, _STATS_TOTAL_SELECTED
    global _STATS_TOTAL_PROTECTED_TOKENS

    accum = _STATS_DEVICE_ACCUM
    if accum is None:
        return
    _STATS_DEVICE_ACCUM = None
    _STATS_TOTAL_SELECTED += float(accum["selected"].item())
    _STATS_TOTAL_PROTECTED_TOKENS += int(accum["protected"].item())


def _record_routing_stats(
    keep_counts: torch.Tensor,
    topk: int,
    protected_mask: Optional[torch.Tensor] = None,
) -> None:
    global _STATS_TOTAL_SELECTED, _STATS_TOTAL_TOKENS, _STATS_TOTAL_CALLS
    global _STATS_TOTAL_PROTECTED_TOKENS

    global _STATS_LAST_TOPK

    if _STATS_DIR is None or _SKIP_STATS_DEPTH > 0:
        return
    if _is_cuda_graph_capturing():
        return

    with torch.no_grad():
        # Summing on the device and leaving the result there keeps the CUDA stream from
        # draining; the totals are read back once per flush instead of once per layer.
        accum = _device_stats_accumulators(keep_counts.device)
        accum["selected"] += keep_counts.sum().to(torch.float64)
        _STATS_TOTAL_TOKENS += int(keep_counts.numel())
        _STATS_TOTAL_CALLS += 1
        if protected_mask is not None:
            accum["protected"] += protected_mask.sum().to(torch.float64)

    _STATS_LAST_TOPK = topk
    if _STATS_TOTAL_CALLS % _STATS_FLUSH_INTERVAL != 0:
        return
    _write_routing_stats_snapshot(topk)


def _write_routing_stats_snapshot(topk: int) -> None:
    """Write this worker's cumulative routing statistics to the stats directory."""
    if _STATS_DIR is None:
        return
    _drain_device_stats()

    _STATS_DIR.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    stats_path = _STATS_DIR / f"routing_stats_{pid}.json"
    tmp_path = _STATS_DIR / f".routing_stats_{pid}.json.tmp"
    payload = {
        "pid": pid,
        "method": _ROUTING_METHOD,
        "naee_beta": _NAEE_BETA,
        "naee_k_min": _NAEE_K_MIN,
        "dynamic_routing_threshold": _DYNAMIC_ROUTING_THRESHOLD,
        "dynamic_routing_score_source": _DYNAMIC_ROUTING_SCORE_SOURCE,
        "dynamic_routing_next_rank_penalty": (
            _DYNAMIC_ROUTING_NEXT_RANK_PENALTY),
        "layerwise_dynamic_base_threshold": (
            _LAYERWISE_DYNAMIC_BASE_THRESHOLD),
        "layerwise_dynamic_layer_alpha": _LAYERWISE_DYNAMIC_LAYER_ALPHA,
        "layerwise_dynamic_num_layers": _LAYERWISE_DYNAMIC_NUM_LAYERS,
        "layerwise_dynamic_k_min": _LAYERWISE_DYNAMIC_K_MIN,
        "budget_dynamic_easy_k": _BUDGET_DYNAMIC_EASY_K,
        "budget_dynamic_base_k": _BUDGET_DYNAMIC_BASE_K,
        "budget_dynamic_hard_k": _BUDGET_DYNAMIC_HARD_K,
        "budget_dynamic_easy_threshold": _BUDGET_DYNAMIC_EASY_THRESHOLD,
        "budget_dynamic_hard_threshold": _BUDGET_DYNAMIC_HARD_THRESHOLD,
        "budget_dynamic_score_topn": _BUDGET_DYNAMIC_SCORE_TOPN,
        "layer_budget_dynamic_easy_layer_alpha": (
            _LAYER_BUDGET_DYNAMIC_EASY_LAYER_ALPHA),
        "layer_budget_dynamic_hard_layer_alpha": (
            _LAYER_BUDGET_DYNAMIC_HARD_LAYER_ALPHA),
        "layer_budget_dynamic_num_layers": (
            _LAYER_BUDGET_DYNAMIC_NUM_LAYERS),
        "band_layer_budget_middle_start": _BAND_LAYER_BUDGET_MIDDLE_START,
        "band_layer_budget_middle_end": _BAND_LAYER_BUDGET_MIDDLE_END,
        "band_layer_budget_per_layer_k": list(
            _BAND_LAYER_BUDGET_PER_LAYER_K),
        "band_layer_budget_easy_layers": list(
            _BAND_LAYER_BUDGET_EASY_LAYERS),
        "band_layer_budget_hard_layers": list(
            _BAND_LAYER_BUDGET_HARD_LAYERS),
        "band_layer_budget_token_gate_ratio": (
            _BAND_LAYER_BUDGET_TOKEN_GATE_RATIO),
        "band_layer_budget_extra_middle_start": (
            _BAND_LAYER_BUDGET_EXTRA_MIDDLE_START),
        "band_layer_budget_extra_middle_end": (
            _BAND_LAYER_BUDGET_EXTRA_MIDDLE_END),
        "band_layer_budget_middle_phase": _BAND_LAYER_BUDGET_MIDDLE_PHASE,
        "band_layer_budget_easy_start": _BAND_LAYER_BUDGET_EASY_START,
        "band_layer_budget_easy_end": _BAND_LAYER_BUDGET_EASY_END,
        "band_layer_budget_easy_phase": _BAND_LAYER_BUDGET_EASY_PHASE,
        "band_layer_budget_hard_phase": _BAND_LAYER_BUDGET_HARD_PHASE,
        "band_layer_budget_easy_tail_threshold": (
            _BAND_LAYER_BUDGET_EASY_TAIL_THRESHOLD),
        "band_layer_budget_hard_min_rank_weight": (
            _BAND_LAYER_BUDGET_HARD_MIN_RANK_WEIGHT),
        "band_layer_budget_hard_max_rank_weight": (
            _BAND_LAYER_BUDGET_HARD_MAX_RANK_WEIGHT),
        "band_layer_budget_hard_decode_min_rank_weight": (
            _BAND_LAYER_BUDGET_HARD_DECODE_MIN_RANK_WEIGHT),
        "band_layer_budget_mixed_rescue_min_rank_weight": (
            _BAND_LAYER_BUDGET_MIXED_RESCUE_MIN_RANK_WEIGHT),
        "band_layer_budget_mixed_rescue_max_concentration": (
            _BAND_LAYER_BUDGET_MIXED_RESCUE_MAX_CONCENTRATION),
        "band_layer_budget_hard_prefill_max_tokens": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_TOKENS),
        "band_layer_budget_hard_prefill_max_segment_tokens": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_TOKENS),
        "band_layer_budget_hard_prefill_max_density": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_DENSITY),
        "band_layer_budget_hard_prefill_min_density": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_MIN_DENSITY),
        "band_layer_budget_hard_prefill_max_segment_hard_ratio": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_HARD_RATIO),
        "band_layer_budget_hard_prefill_segment_cap_score": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_SEGMENT_CAP_SCORE),
        "band_layer_budget_hard_prefill_conditional_segment_hard_ratio": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_SEGMENT_HARD_RATIO),
        "band_layer_budget_hard_prefill_conditional_min_density": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_DENSITY),
        "band_layer_budget_hard_prefill_conditional_min_seq_len": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_SEQ_LEN),
        "band_layer_budget_hard_prefill_conditional_token_ids": (
            list(_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_IDS)),
        "band_layer_budget_hard_prefill_conditional_token_ngrams": (
            [list(ngram) for ngram in
             _BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_NGRAMS]),
        "band_layer_budget_hard_prefill_marker_segment_hard_ratio": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_SEGMENT_HARD_RATIO),
        "band_layer_budget_hard_prefill_marker_token_ids": (
            list(_BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_IDS)),
        "band_layer_budget_hard_prefill_marker_token_ngrams": (
            [list(ngram) for ngram in
             _BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_NGRAMS]),
        "band_layer_budget_hard_prefill_exclude_token_ids": (
            list(_BAND_LAYER_BUDGET_HARD_PREFILL_EXCLUDE_TOKEN_IDS)),
        "band_layer_budget_hard_prefill_min_relative_pos": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_MIN_RELATIVE_POS),
        "band_layer_budget_hard_prefill_max_relative_pos": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_MAX_RELATIVE_POS),
        "band_layer_budget_hard_prefill_start": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_START),
        "band_layer_budget_hard_prefill_end": (
            _BAND_LAYER_BUDGET_HARD_PREFILL_END),
        "band_layer_budget_hard_decode_min_seq_len": (
            _BAND_LAYER_BUDGET_HARD_DECODE_MIN_SEQ_LEN),
        "band_layer_budget_hard_decode_max_seq_len": (
            _BAND_LAYER_BUDGET_HARD_DECODE_MAX_SEQ_LEN),
        "band_layer_budget_hard_decode_seq_len_scope": (
            _BAND_LAYER_BUDGET_HARD_DECODE_SEQ_LEN_SCOPE),
        "band_layer_budget_hard_decode_min_offset": (
            _BAND_LAYER_BUDGET_HARD_DECODE_MIN_OFFSET),
        "band_layer_budget_hard_decode_max_offset": (
            _BAND_LAYER_BUDGET_HARD_DECODE_MAX_OFFSET),
        "band_layer_budget_hard_decode_start": (
            _BAND_LAYER_BUDGET_HARD_DECODE_START),
        "band_layer_budget_hard_decode_end": (
            _BAND_LAYER_BUDGET_HARD_DECODE_END),
        "band_layer_budget_hard_decode_token_ids": (
            list(_BAND_LAYER_BUDGET_HARD_DECODE_TOKEN_IDS)),
        "band_layer_budget_hard_layer_prior_start": (
            _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_START),
        "band_layer_budget_hard_layer_prior_end": (
            _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_END),
        "band_layer_budget_hard_layer_prior_extra_start": (
            _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_START),
        "band_layer_budget_hard_layer_prior_extra_end": (
            _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_END),
        "band_layer_budget_hard_layer_prior_alpha": (
            _BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_ALPHA),
        "band_layer_budget_hard_layer_sensitivity_alpha": (
            _BAND_LAYER_BUDGET_HARD_LAYER_SENSITIVITY_ALPHA),
        "band_layer_budget_late_concentration_start": (
            _BAND_LAYER_BUDGET_LATE_CONCENTRATION_START),
        "band_layer_budget_late_concentration_threshold": (
            _BAND_LAYER_BUDGET_LATE_CONCENTRATION_THRESHOLD),
        "band_layer_budget_late_rescue_start": (
            _BAND_LAYER_BUDGET_LATE_RESCUE_START),
        "band_layer_budget_late_rescue_concentration_threshold": (
            _BAND_LAYER_BUDGET_LATE_RESCUE_CONCENTRATION_THRESHOLD),
        "band_layer_budget_late_rescue_mixed_max": (
            _BAND_LAYER_BUDGET_LATE_RESCUE_MIXED_MAX),
        "band_layer_budget_prompt_profile1_token_ngrams": (
            [list(ngram)
             for ngram in _BAND_LAYER_BUDGET_PROMPT_PROFILE1_TOKEN_NGRAMS]),
        "band_layer_budget_prompt_profile1_hard_threshold": (
            _BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_THRESHOLD),
        "band_layer_budget_prompt_profile1_hard_min_rank_weight": (
            _BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_MIN_RANK_WEIGHT),
        "band_layer_budget_prompt_profile1_middle_start": (
            _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_START),
        "band_layer_budget_prompt_profile1_middle_end": (
            _BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_END),
        "band_layer_budget_prompt_profile1_prefill_segment_hard_ratio": (
            _BAND_LAYER_BUDGET_PROMPT_PROFILE1_PREFILL_SEGMENT_HARD_RATIO),
        "band_layer_budget_prompt_profile2_token_ngrams": (
            [list(ngram)
             for ngram in _BAND_LAYER_BUDGET_PROMPT_PROFILE2_TOKEN_NGRAMS]),
        "band_layer_budget_prompt_profile2_hard_threshold": (
            _BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_THRESHOLD),
        "band_layer_budget_prompt_profile2_hard_min_rank_weight": (
            _BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_MIN_RANK_WEIGHT),
        "band_layer_budget_prompt_profile2_middle_start": (
            _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_START),
        "band_layer_budget_prompt_profile2_middle_end": (
            _BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_END),
        "band_layer_budget_prompt_profile2_prefill_segment_hard_ratio": (
            _BAND_LAYER_BUDGET_PROMPT_PROFILE2_PREFILL_SEGMENT_HARD_RATIO),
        "diep_pruning_mode": _DIEP_PRUNING_MODE,
        "diep_use_gamma1": _DIEP_USE_GAMMA1,
        "diep_gamma_alpha": _DIEP_GAMMA_ALPHA,
        "diep_threshold_cap": _DIEP_THRESHOLD_CAP,
        "diep_artifact_path": _DIEP_ARTIFACT_PATH,
        "ban_artifact_path": _BAN_ARTIFACT_PATH,
        "ban_lambda": _BAN_LAMBDA,
        "ban_k_min": _BAN_K_MIN,
        "eac_alpha": _EAC_ALPHA,
        "mc_moe_protection_ratio": _MC_MOE_PROTECTION_RATIO,
        "biased_renorm_keep_topn": _BIASED_RENORM_KEEP_TOPN,
        "router_value_artifact_path": _ROUTER_VALUE_ARTIFACT_PATH,
        "router_value_base_k": _ROUTER_VALUE_BASE_K,
        "router_value_quota_ratio": _ROUTER_VALUE_QUOTA_RATIO,
        "router_value_selection_mode": _ROUTER_VALUE_SELECTION_MODE,
        "router_value_swap_direction": _ROUTER_VALUE_SWAP_DIRECTION,
        "topk": topk,
        "total_selected_experts": _STATS_TOTAL_SELECTED,
        "total_routed_tokens": _STATS_TOTAL_TOKENS,
        "total_routing_calls": _STATS_TOTAL_CALLS,
        "total_protected_tokens": _STATS_TOTAL_PROTECTED_TOKENS,
        "eac_prefill_tokens": _STATS_EAC_PREFILL_TOKENS,
        "eac_segments": _STATS_EAC_SEGMENTS,
        "eac_calls_with_segments": _STATS_EAC_CALLS_WITH_SEGMENTS,
        "eac_pruned_selections": _STATS_EAC_PRUNED_SELECTIONS,
        "eac_disabled_experts": _STATS_EAC_DISABLED_EXPERTS,
        "eac_pruned_selection_ratio": (
            _STATS_EAC_PRUNED_SELECTIONS /
            (_STATS_EAC_PREFILL_TOKENS * topk)
            if _STATS_EAC_PREFILL_TOKENS else 0.0),
        "ban_average_dynamic_k": (
            _STATS_BAN_TOTAL_K / _STATS_BAN_TOKENS
            if _STATS_BAN_TOKENS else 0.0),
        "ban_total_tokens": _STATS_BAN_TOKENS,
        "budget_easy_tokens": _STATS_BUDGET_EASY_TOKENS,
        "budget_base_tokens": _STATS_BUDGET_BASE_TOKENS,
        "budget_hard_tokens": _STATS_BUDGET_HARD_TOKENS,
        "budget_layer_counts": {
            str(layer): counts
            for layer, counts in sorted(_STATS_BUDGET_LAYER_COUNTS.items())
        },
        "budget_prompt_profile_counts": {
            str(profile): counts
            for profile, counts in sorted(
                _STATS_BUDGET_PROFILE_COUNTS.items())
        },
        "budget_stage_counts": {
            stage: counts
            for stage, counts in sorted(_STATS_BUDGET_STAGE_COUNTS.items())
        },
        "budget_hard_rank_weight": {
            "bounds": _STATS_BUDGET_HARD_RANK_WEIGHT_BOUNDS,
            "candidate": _budget_rank_weight_payload(
                _STATS_BUDGET_HARD_RANK_WEIGHT_CANDIDATE),
            "selected": _budget_rank_weight_payload(
                _STATS_BUDGET_HARD_RANK_WEIGHT_SELECTED),
        },
        "budget_prefill_segment_hard_density": {
            "bounds": _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BOUNDS,
            "candidate": _budget_segment_density_payload(
                _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_CANDIDATE),
            "selected": _budget_segment_density_payload(
                _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_SELECTED),
            "by_layer": {
                str(layer): {
                    "candidate": _budget_segment_density_payload(
                        stats["candidate"]),
                    "selected": _budget_segment_density_payload(
                        stats["selected"]),
                }
                for layer, stats in sorted(
                    _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BY_LAYER.items())
            },
        },
        "budget_hard_seq_len": {
            "bounds": _STATS_BUDGET_HARD_SEQ_LEN_BOUNDS,
            "candidate": _budget_seq_len_payload(
                _STATS_BUDGET_HARD_SEQ_LEN_CANDIDATE),
            "selected": _budget_seq_len_payload(
                _STATS_BUDGET_HARD_SEQ_LEN_SELECTED),
        },
        "budget_hard_decode_offset": {
            "bounds": _STATS_BUDGET_HARD_DECODE_OFFSET_BOUNDS,
            "candidate": _budget_seq_len_payload(
                _STATS_BUDGET_HARD_DECODE_OFFSET_CANDIDATE),
            "selected": _budget_seq_len_payload(
                _STATS_BUDGET_HARD_DECODE_OFFSET_SELECTED),
        },
        "budget_easy_ratio": (
            _STATS_BUDGET_EASY_TOKENS / _STATS_TOTAL_TOKENS
            if _STATS_TOTAL_TOKENS else 0.0),
        "budget_base_ratio": (
            _STATS_BUDGET_BASE_TOKENS / _STATS_TOTAL_TOKENS
            if _STATS_TOTAL_TOKENS else 0.0),
        "budget_hard_ratio": (
            _STATS_BUDGET_HARD_TOKENS / _STATS_TOTAL_TOKENS
            if _STATS_TOTAL_TOKENS else 0.0),
        "protected_token_ratio": (
            _STATS_TOTAL_PROTECTED_TOKENS / _STATS_TOTAL_TOKENS
            if _STATS_TOTAL_TOKENS else 0.0),
        "average_selected_experts": (
            _STATS_TOTAL_SELECTED / _STATS_TOTAL_TOKENS
            if _STATS_TOTAL_TOKENS else 0.0),
    }
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, stats_path)


@atexit.register
def _flush_routing_stats_at_exit() -> None:
    """Snapshots are periodic, so the calls since the last one still need writing."""
    if _STATS_LAST_TOPK is None or _STATS_TOTAL_CALLS == 0:
        return
    try:
        _write_routing_stats_snapshot(_STATS_LAST_TOPK)
    except Exception as exc:  # noqa: BLE001
        print(f"[expert_pruning] final routing-stats flush failed: {exc}")


def _naee_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    keep_mask = _naee_keep_mask(topk_weights)

    topk_weights_before_pruning = topk_weights
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _maybe_print_naee_debug(
        topk_weights_before_pruning=topk_weights_before_pruning,
        topk_weights_after_pruning=topk_weights,
        keep_mask=keep_mask,
    )
    _record_routing_stats(keep_counts=keep_mask.sum(dim=-1), topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )

    return topk_weights, topk_ids, token_expert_indices


def _mc_moe_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    keep_mask = _naee_keep_mask(topk_weights)
    protected_mask = _current_protected_mask_for_tokens(
        num_tokens=num_tokens,
        device=hidden_states.device,
    )
    if protected_mask is not None:
        keep_mask = keep_mask | protected_mask.view(-1, 1)

    topk_weights_before_pruning = topk_weights
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _maybe_print_naee_debug(
        topk_weights_before_pruning=topk_weights_before_pruning,
        topk_weights_after_pruning=topk_weights,
        keep_mask=keep_mask,
    )
    _record_routing_stats(
        keep_counts=keep_mask.sum(dim=-1),
        topk=topk,
        protected_mask=protected_mask,
    )

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )

    return topk_weights, topk_ids, token_expert_indices


def _eac_moe_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _STATS_EAC_PREFILL_TOKENS, _STATS_EAC_SEGMENTS
    global _STATS_EAC_CALLS_WITH_SEGMENTS, _STATS_EAC_PRUNED_SELECTIONS
    global _STATS_EAC_DISABLED_EXPERTS

    num_tokens = hidden_states.size(0)
    num_experts = gating_output.size(-1)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    keep_mask = torch.ones_like(topk_weights, dtype=torch.bool)
    saw_segment = False
    prefill_mask = _current_eac_prefill_mask_for_tokens(
        num_tokens=num_tokens,
        device=hidden_states.device,
    )
    if prefill_mask is not None:
        token_is_prefill = prefill_mask.view(num_tokens, 1)
        expert_ids = torch.arange(
            num_experts, device=hidden_states.device).view(1, 1, num_experts)
        selected_experts = topk_ids.long().unsqueeze(-1)
        counts = ((selected_experts == expert_ids)
                  & token_is_prefill.view(num_tokens, 1, 1)).sum(
                      dim=(0, 1)).to(dtype=torch.float32)
        segment_len = prefill_mask.sum().to(dtype=torch.float32)
        threshold = (segment_len * topk / num_experts) * _EAC_ALPHA
        disabled_experts = counts < threshold
        eac_keep = ~disabled_experts[topk_ids.long()]
        keep_mask = torch.where(token_is_prefill, eac_keep, keep_mask)

        empty_rows = token_is_prefill.squeeze(-1) & ~keep_mask.any(dim=-1)
        if empty_rows.any():
            keep_mask[empty_rows, 0] = True

        if (_SKIP_STATS_DEPTH == 0 and not _is_cuda_graph_capturing()
                and not _is_torch_compiling()):
            prefill_tokens = int(prefill_mask.sum().item())
            if prefill_tokens > 0:
                saw_segment = True
                _STATS_EAC_PREFILL_TOKENS += prefill_tokens
                _STATS_EAC_SEGMENTS += 1
                _STATS_EAC_PRUNED_SELECTIONS += int(
                    ((~keep_mask) & token_is_prefill).sum().item())
                _STATS_EAC_DISABLED_EXPERTS += int(
                    disabled_experts.sum().item())
    elif _CURRENT_TOKEN_SEGMENTS is not None:
        for start, end in _CURRENT_TOKEN_SEGMENTS:
            segment_len = end - start
            if segment_len <= 1:
                continue
            saw_segment = True

            segment_ids = topk_ids[start:end].long()
            counts = torch.bincount(
                segment_ids.reshape(-1),
                minlength=num_experts,
            ).to(dtype=torch.float32)
            threshold = (segment_len * topk / num_experts) * _EAC_ALPHA
            disabled_experts = counts < threshold
            segment_keep = ~disabled_experts[segment_ids]

            # Avoid producing an all-zero expert set for a token. In this rare
            # fallback, keep its top-1 expert so vLLM can still produce output.
            empty_rows = ~segment_keep.any(dim=-1)
            if empty_rows.any():
                segment_keep[empty_rows, 0] = True
            keep_mask[start:end] = segment_keep

            if _SKIP_STATS_DEPTH == 0 and not _is_cuda_graph_capturing():
                _STATS_EAC_PREFILL_TOKENS += int(segment_len)
                _STATS_EAC_SEGMENTS += 1
                _STATS_EAC_PRUNED_SELECTIONS += int(
                    (~segment_keep).sum().item())
                _STATS_EAC_DISABLED_EXPERTS += int(
                    disabled_experts.sum().item())

    if (saw_segment and _SKIP_STATS_DEPTH == 0
            and not _is_cuda_graph_capturing()):
        _STATS_EAC_CALLS_WITH_SEGMENTS += 1

    topk_weights_before_pruning = topk_weights
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _maybe_print_naee_debug(
        topk_weights_before_pruning=topk_weights_before_pruning,
        topk_weights_after_pruning=topk_weights,
        keep_mask=keep_mask,
    )
    _record_routing_stats(keep_counts=keep_mask.sum(dim=-1), topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )

    return topk_weights, topk_ids, token_expert_indices


def _diep_gamma_for_topk(topk_ids: torch.Tensor,
                         topk_weights: torch.Tensor) -> torch.Tensor:
    if (_DIEP_SIM_MATRIX is None or _DIEP_MEAN_SIM is None
            or _CURRENT_MOE_LAYER_INDEX is None):
        return torch.ones_like(topk_weights)

    layer_index = _CURRENT_MOE_LAYER_INDEX
    if layer_index >= _DIEP_SIM_MATRIX.size(0):
        return torch.ones_like(topk_weights)

    sim_matrix_all, mean_sim_all, gamma1_all = _get_diep_tensors_for_device(
        topk_ids.device)
    if sim_matrix_all is None or mean_sim_all is None:
        return torch.ones_like(topk_weights)

    sim_matrix = sim_matrix_all[layer_index]
    mean_sim = mean_sim_all[layer_index]
    mean_sim = mean_sim.clamp_min(1e-6)

    top1_ids = topk_ids[:, :1].long()
    pair_ids = topk_ids.long()
    pair_sim = sim_matrix[top1_ids.expand_as(pair_ids), pair_ids]
    gamma = pair_sim / mean_sim
    if _DIEP_GAMMA_ALPHA != 1.0:
        gamma = gamma.clamp_min(0.0).pow(_DIEP_GAMMA_ALPHA)
    # By default only gamma_2 is applied and the swept beta stands in for gamma_1, which is a
    # fixed constant per layer and would pin the method to one operating point. Multiplying
    # gamma_1 back in reproduces the paper's eq. 12 exactly, which is the one setting the
    # paper actually specifies; combine it with --naee_beta 1.0.
    if _DIEP_USE_GAMMA1 and gamma1_all is not None:
        gamma = gamma * gamma1_all[layer_index]
    gamma[:, 0] = 1.0
    return gamma.to(dtype=topk_weights.dtype)


def _ban_layer_sensitivity(
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if (_BAN_LAYER_SENSITIVITY is None or _CURRENT_MOE_LAYER_INDEX is None):
        return torch.tensor(0.0, device=device, dtype=dtype)
    layer_index = _CURRENT_MOE_LAYER_INDEX
    if layer_index >= _BAN_LAYER_SENSITIVITY.size(0):
        return torch.tensor(0.0, device=device, dtype=dtype)
    layer_sensitivity_all = _get_ban_layer_sensitivity_for_device(device)
    if layer_sensitivity_all is None:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return layer_sensitivity_all[layer_index].to(dtype=dtype)


def _ban_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _STATS_BAN_TOTAL_K, _STATS_BAN_TOKENS

    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    top3 = min(3, topk)
    k_min = min(_BAN_K_MIN, topk)
    ratio_numer = topk_weights[:, :top3].sum(dim=-1)
    ratio_denom = topk_weights.sum(dim=-1).clamp_min(1e-20)
    ratio = ratio_numer / ratio_denom
    ratio_range = max(_BAN_R_MAX - _BAN_R_MIN, 1e-6)
    token_sensitivity = ((_BAN_R_MAX - ratio) / ratio_range).clamp(0.0, 1.0)
    layer_sensitivity = _ban_layer_sensitivity(
        device=hidden_states.device,
        dtype=topk_weights.dtype,
    ).clamp(0.0, 1.0)
    combined_sensitivity = (
        _BAN_LAMBDA * 0.5 * (layer_sensitivity + token_sensitivity)
    ).clamp(0.0, 1.0)
    dynamic_k_float = k_min + (topk - k_min) * combined_sensitivity
    dynamic_k = torch.round(dynamic_k_float).to(dtype=torch.long).clamp(
        min=k_min,
        max=topk,
    )
    ranks = torch.arange(topk, device=hidden_states.device).view(1, topk)
    keep_mask = ranks < dynamic_k.view(-1, 1)

    topk_weights_before_pruning = topk_weights
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _maybe_print_naee_debug(
        topk_weights_before_pruning=topk_weights_before_pruning,
        topk_weights_after_pruning=topk_weights,
        keep_mask=keep_mask,
    )
    _record_routing_stats(keep_counts=dynamic_k, topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )
    return topk_weights, topk_ids, token_expert_indices


def _diep_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    ranks = torch.arange(topk, device=hidden_states.device).view(1, topk)
    k_min = min(_NAEE_K_MIN, topk)
    keep_by_min = ranks < k_min
    gamma = _diep_gamma_for_topk(topk_ids=topk_ids, topk_weights=topk_weights)
    bar = _NAEE_BETA * gamma
    if _DIEP_THRESHOLD_CAP is not None:
        bar = bar.clamp_max(_DIEP_THRESHOLD_CAP)
    threshold = topk_weights[:, :1] * bar
    prune_candidate = (ranks >= k_min) & (topk_weights < threshold)
    if _DIEP_PRUNING_MODE == "prefix":
        has_previous_prune = prune_candidate.int().cumsum(dim=-1) > 0
        keep_mask = keep_by_min | ~has_previous_prune
    else:
        keep_mask = keep_by_min | ~prune_candidate

    topk_weights_before_pruning = topk_weights
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _maybe_print_naee_debug(
        topk_weights_before_pruning=topk_weights_before_pruning,
        topk_weights_after_pruning=topk_weights,
        keep_mask=keep_mask,
    )
    _record_routing_stats(keep_counts=keep_mask.sum(dim=-1), topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )
    return topk_weights, topk_ids, token_expert_indices


def _dynamic_routing_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    threshold_weights = topk_weights
    if _DYNAMIC_ROUTING_SCORE_SOURCE == "renormalized":
        threshold_weights = _renormalize_topk_weights(topk_weights)

    cumulative_weights = threshold_weights.cumsum(dim=-1)
    reaches_threshold = cumulative_weights >= _DYNAMIC_ROUTING_THRESHOLD
    first_reach = reaches_threshold.to(dtype=torch.int64).argmax(dim=-1) + 1
    keep_counts = torch.where(
        reaches_threshold.any(dim=-1),
        first_reach,
        torch.full_like(first_reach, topk),
    )
    ranks = torch.arange(topk, device=hidden_states.device).view(1, topk)
    keep_mask = ranks < keep_counts.view(-1, 1)

    topk_weights_before_pruning = topk_weights
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _maybe_print_dynamic_routing_debug(
        topk_weights_before_pruning=topk_weights_before_pruning,
        threshold_weights=threshold_weights,
        topk_weights_after_pruning=topk_weights,
        keep_mask=keep_mask,
    )
    _record_routing_stats(keep_counts=keep_counts, topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )

    return topk_weights, topk_ids, token_expert_indices


def _layerwise_dynamic_threshold() -> float:
    layer_index = _CURRENT_MOE_LAYER_INDEX
    if layer_index is None:
        return _LAYERWISE_DYNAMIC_BASE_THRESHOLD
    hardness = _layer_position_hardness(
        layer_index=layer_index,
        num_layers=_LAYERWISE_DYNAMIC_NUM_LAYERS,
    )
    threshold = (_LAYERWISE_DYNAMIC_BASE_THRESHOLD +
                 _LAYERWISE_DYNAMIC_LAYER_ALPHA * hardness)
    return max(1e-6, min(1.0, threshold))


def _layer_position_hardness(
    layer_index: Optional[int],
    num_layers: int,
) -> float:
    if layer_index is None:
        return 0.0
    if num_layers <= 1:
        return 1.0
    position = max(0.0, min(1.0, float(layer_index) / float(num_layers - 1)))
    return 1.0 - abs(2.0 * position - 1.0)


def _layerwise_dynamic_routing_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    threshold_weights = topk_weights
    if _DYNAMIC_ROUTING_SCORE_SOURCE == "renormalized":
        threshold_weights = _renormalize_topk_weights(topk_weights)

    threshold = _layerwise_dynamic_threshold()
    cumulative_weights = threshold_weights.cumsum(dim=-1)
    reaches_threshold = cumulative_weights >= threshold
    first_reach = reaches_threshold.to(dtype=torch.int64).argmax(dim=-1) + 1
    keep_counts = torch.where(
        reaches_threshold.any(dim=-1),
        first_reach,
        torch.full_like(first_reach, topk),
    )
    k_min = min(_LAYERWISE_DYNAMIC_K_MIN, topk)
    keep_counts = torch.clamp(keep_counts, min=k_min, max=topk)
    ranks = torch.arange(topk, device=hidden_states.device).view(1, topk)
    keep_mask = ranks < keep_counts.view(-1, 1)

    topk_weights_before_pruning = topk_weights
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _maybe_print_dynamic_routing_debug(
        topk_weights_before_pruning=topk_weights_before_pruning,
        threshold_weights=threshold_weights,
        topk_weights_after_pruning=topk_weights,
        keep_mask=keep_mask,
    )
    _record_routing_stats(keep_counts=keep_counts, topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )

    return topk_weights, topk_ids, token_expert_indices


def _record_budget_class_counts(class_ids: torch.Tensor) -> None:
    global _STATS_BUDGET_EASY_TOKENS, _STATS_BUDGET_BASE_TOKENS
    global _STATS_BUDGET_HARD_TOKENS
    global _STATS_BUDGET_LAYER_COUNTS, _STATS_BUDGET_PROFILE_COUNTS
    global _STATS_BUDGET_STAGE_COUNTS

    if _STATS_DIR is None or _SKIP_STATS_DEPTH > 0:
        return
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return

    def add_stage_counts(stage: str, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        counts = _STATS_BUDGET_STAGE_COUNTS.setdefault(stage, [0, 0, 0])
        counts[0] += int((values == 0).sum().item())
        counts[1] += int((values == 1).sum().item())
        counts[2] += int((values == 2).sum().item())

    with torch.no_grad():
        easy = int((class_ids == 0).sum().item())
        base = int((class_ids == 1).sum().item())
        hard = int((class_ids == 2).sum().item())
        _STATS_BUDGET_EASY_TOKENS += easy
        _STATS_BUDGET_BASE_TOKENS += base
        _STATS_BUDGET_HARD_TOKENS += hard
        if _CURRENT_MOE_LAYER_INDEX is not None:
            counts = _STATS_BUDGET_LAYER_COUNTS.setdefault(
                _CURRENT_MOE_LAYER_INDEX, [0, 0, 0])
            counts[0] += easy
            counts[1] += base
            counts[2] += hard
        profile_ids = _current_prompt_profile_ids_for_tokens(
            int(class_ids.numel()),
            class_ids.device,
        )
        for profile_id in (0, 1, 2):
            profile_mask = profile_ids == profile_id
            if not bool(profile_mask.any().item()):
                continue
            profile_values = class_ids[profile_mask]
            profile_counts = _STATS_BUDGET_PROFILE_COUNTS.setdefault(
                int(profile_id),
                [0, 0, 0],
            )
            profile_counts[0] += int((profile_values == 0).sum().item())
            profile_counts[1] += int((profile_values == 1).sum().item())
            profile_counts[2] += int((profile_values == 2).sum().item())
        prefill_mask = _current_prefill_mask_for_tokens(
            int(class_ids.numel()),
            class_ids.device,
        )
        if prefill_mask is None:
            add_stage_counts("unknown", class_ids)
        else:
            add_stage_counts("prefill", class_ids[prefill_mask])
            add_stage_counts("decode", class_ids[~prefill_mask])


def _update_rank_weight_stats(stats: dict, values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    values = values.detach()
    count = int(values.numel())
    value_sum = float(values.sum().item())
    value_min = float(values.min().item())
    value_max = float(values.max().item())
    bounds = torch.tensor(
        _STATS_BUDGET_HARD_RANK_WEIGHT_BOUNDS,
        device=values.device,
        dtype=values.dtype,
    )
    bucket_ids = torch.bucketize(values, bounds)
    hist = torch.bincount(
        bucket_ids,
        minlength=_stats_hist_size(_STATS_BUDGET_HARD_RANK_WEIGHT_BOUNDS),
    ).detach().cpu()

    stats["count"] += count
    stats["sum"] += value_sum
    stats["min"] = (
        value_min if stats["min"] is None else min(stats["min"], value_min))
    stats["max"] = (
        value_max if stats["max"] is None else max(stats["max"], value_max))
    for idx, value in enumerate(hist.tolist()):
        stats["hist"][idx] += int(value)


def _update_seq_len_stats(stats: dict, values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    values = values.detach().to(dtype=torch.float32)
    count = int(values.numel())
    value_sum = float(values.sum().item())
    value_min = int(values.min().item())
    value_max = int(values.max().item())
    bounds = torch.tensor(
        _STATS_BUDGET_HARD_SEQ_LEN_BOUNDS,
        device=values.device,
        dtype=values.dtype,
    )
    bucket_ids = torch.bucketize(values, bounds)
    hist = torch.bincount(
        bucket_ids,
        minlength=_stats_hist_size(_STATS_BUDGET_HARD_SEQ_LEN_BOUNDS),
    ).detach().cpu()

    stats["count"] += count
    stats["sum"] += value_sum
    stats["min"] = (
        value_min if stats["min"] is None else min(stats["min"], value_min))
    stats["max"] = (
        value_max if stats["max"] is None else max(stats["max"], value_max))
    for idx, value in enumerate(hist.tolist()):
        stats["hist"][idx] += int(value)


def _update_decode_offset_stats(stats: dict, values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    values = values.detach().to(dtype=torch.float32)
    values = values[values >= 0]
    if values.numel() == 0:
        return
    count = int(values.numel())
    value_sum = float(values.sum().item())
    value_min = int(values.min().item())
    value_max = int(values.max().item())
    bounds = torch.tensor(
        _STATS_BUDGET_HARD_DECODE_OFFSET_BOUNDS,
        device=values.device,
        dtype=values.dtype,
    )
    bucket_ids = torch.bucketize(values, bounds)
    hist = torch.bincount(
        bucket_ids,
        minlength=_stats_hist_size(_STATS_BUDGET_HARD_DECODE_OFFSET_BOUNDS),
    ).detach().cpu()

    stats["count"] += count
    stats["sum"] += value_sum
    stats["min"] = (
        value_min if stats["min"] is None else min(stats["min"], value_min))
    stats["max"] = (
        value_max if stats["max"] is None else max(stats["max"], value_max))
    for idx, value in enumerate(hist.tolist()):
        stats["hist"][idx] += int(value)


def _new_budget_segment_density_stats() -> dict:
    return {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
        "hist": [0] * _stats_hist_size(
            _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BOUNDS),
        "hard_tokens": 0,
        "segment_tokens": 0,
    }


def _update_segment_density_stats(
    stats: dict,
    densities: torch.Tensor,
    hard_counts: torch.Tensor,
    segment_lengths: torch.Tensor,
) -> None:
    if densities.numel() == 0:
        return
    densities = densities.detach()
    hard_counts = hard_counts.detach()
    segment_lengths = segment_lengths.detach()
    count = int(densities.numel())
    value_sum = float(densities.sum().item())
    value_min = float(densities.min().item())
    value_max = float(densities.max().item())
    hard_tokens = int(hard_counts.sum().item())
    segment_tokens = int(segment_lengths.sum().item())
    bounds = torch.tensor(
        _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BOUNDS,
        device=densities.device,
        dtype=densities.dtype,
    )
    bucket_ids = torch.bucketize(densities, bounds)
    hist = torch.bincount(
        bucket_ids,
        minlength=_stats_hist_size(
            _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BOUNDS),
    ).detach().cpu()

    stats["count"] += count
    stats["sum"] += value_sum
    stats["min"] = (
        value_min if stats["min"] is None else min(stats["min"], value_min))
    stats["max"] = (
        value_max if stats["max"] is None else max(stats["max"], value_max))
    stats["hard_tokens"] += hard_tokens
    stats["segment_tokens"] += segment_tokens
    for idx, value in enumerate(hist.tolist()):
        stats["hist"][idx] += int(value)


def _record_budget_prefill_segment_density_stats(
    candidate_mask: torch.Tensor,
    selected_mask: torch.Tensor,
) -> None:
    if _CURRENT_TOKEN_SEGMENTS is None:
        return
    if _STATS_DIR is None or _SKIP_STATS_DEPTH > 0:
        return
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return

    num_tokens = int(candidate_mask.numel())
    candidate_counts = []
    selected_counts = []
    segment_lengths = []
    with torch.no_grad():
        for start, end in _CURRENT_TOKEN_SEGMENTS:
            start = max(0, min(int(start), num_tokens))
            end = max(0, min(int(end), num_tokens))
            if end <= start:
                continue
            segment = slice(start, end)
            length = end - start
            candidate_counts.append(candidate_mask[segment].sum())
            selected_counts.append(selected_mask[segment].sum())
            segment_lengths.append(length)
        if not segment_lengths:
            return
        lengths = torch.tensor(
            segment_lengths,
            device=candidate_mask.device,
            dtype=torch.float32,
        )
        candidate_counts_tensor = torch.stack(candidate_counts).float()
        selected_counts_tensor = torch.stack(selected_counts).float()
        candidate_density = candidate_counts_tensor / lengths
        selected_density = selected_counts_tensor / lengths
        _update_segment_density_stats(
            _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_CANDIDATE,
            candidate_density,
            candidate_counts_tensor,
            lengths,
        )
        _update_segment_density_stats(
            _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_SELECTED,
            selected_density,
            selected_counts_tensor,
            lengths,
        )
        if _CURRENT_MOE_LAYER_INDEX is not None:
            layer_stats = (
                _STATS_BUDGET_PREFILL_SEGMENT_DENSITY_BY_LAYER.setdefault(
                    _CURRENT_MOE_LAYER_INDEX,
                    {
                        "candidate": _new_budget_segment_density_stats(),
                        "selected": _new_budget_segment_density_stats(),
                    },
                ))
            _update_segment_density_stats(
                layer_stats["candidate"],
                candidate_density,
                candidate_counts_tensor,
                lengths,
            )
            _update_segment_density_stats(
                layer_stats["selected"],
                selected_density,
                selected_counts_tensor,
                lengths,
            )


def _record_budget_hard_seq_len_stats(
    candidate_mask: torch.Tensor,
    selected_mask: torch.Tensor,
) -> None:
    if _STATS_DIR is None or _SKIP_STATS_DEPTH > 0:
        return
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return
    seq_lens = _current_token_seq_lens_for_tokens(
        int(candidate_mask.numel()),
        candidate_mask.device,
    )
    if seq_lens is None:
        return
    with torch.no_grad():
        _update_seq_len_stats(
            _STATS_BUDGET_HARD_SEQ_LEN_CANDIDATE,
            seq_lens[candidate_mask],
        )
        _update_seq_len_stats(
            _STATS_BUDGET_HARD_SEQ_LEN_SELECTED,
            seq_lens[selected_mask],
        )


def _record_budget_hard_decode_offset_stats(
    candidate_mask: torch.Tensor,
    selected_mask: torch.Tensor,
) -> None:
    if _STATS_DIR is None or _SKIP_STATS_DEPTH > 0:
        return
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return
    offsets = _current_token_decode_offsets_for_tokens(
        int(candidate_mask.numel()),
        candidate_mask.device,
    )
    if offsets is None:
        return
    with torch.no_grad():
        _update_decode_offset_stats(
            _STATS_BUDGET_HARD_DECODE_OFFSET_CANDIDATE,
            offsets[candidate_mask],
        )
        _update_decode_offset_stats(
            _STATS_BUDGET_HARD_DECODE_OFFSET_SELECTED,
            offsets[selected_mask],
        )


def _record_budget_hard_rank_weight_stats(
    rank_weight: Optional[torch.Tensor],
    candidate_mask: torch.Tensor,
    selected_mask: torch.Tensor,
) -> None:
    if rank_weight is None:
        return
    if _STATS_DIR is None or _SKIP_STATS_DEPTH > 0:
        return
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return
    with torch.no_grad():
        _update_rank_weight_stats(
            _STATS_BUDGET_HARD_RANK_WEIGHT_CANDIDATE,
            rank_weight[candidate_mask],
        )
        _update_rank_weight_stats(
            _STATS_BUDGET_HARD_RANK_WEIGHT_SELECTED,
            rank_weight[selected_mask],
        )


def _budget_score_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    if _DYNAMIC_ROUTING_SCORE_SOURCE in {
            "renormalized",
            "next_rank_weight",
            "dropped_mass",
            "renorm_minus_next_rank",
            "renorm_minus_scaled_next_rank",
    }:
        return _renormalize_topk_weights(topk_weights)
    return topk_weights


def _budget_hard_score(
    score_weights: torch.Tensor,
    topk: int,
    base_k: int,
) -> tuple[torch.Tensor, bool]:
    if _DYNAMIC_ROUTING_SCORE_SOURCE == "next_rank_weight":
        if base_k < topk:
            return score_weights[:, base_k], True
        return torch.zeros(
            score_weights.size(0),
            device=score_weights.device,
            dtype=score_weights.dtype,
        ), True
    if _DYNAMIC_ROUTING_SCORE_SOURCE == "dropped_mass":
        if base_k < topk:
            return score_weights[:, base_k:topk].sum(dim=-1), True
        return torch.zeros(
            score_weights.size(0),
            device=score_weights.device,
            dtype=score_weights.dtype,
        ), True

    score_topn = min(_BUDGET_DYNAMIC_SCORE_TOPN, topk)
    concentration = score_weights[:, :score_topn].sum(dim=-1)
    if _DYNAMIC_ROUTING_SCORE_SOURCE == "renorm_minus_next_rank":
        if base_k < topk:
            return concentration - score_weights[:, base_k], False
        return concentration, False
    if _DYNAMIC_ROUTING_SCORE_SOURCE == "renorm_minus_scaled_next_rank":
        if base_k < topk:
            return (
                concentration
                - _DYNAMIC_ROUTING_NEXT_RANK_PENALTY
                * score_weights[:, base_k]
            ), False
        return concentration, False
    return concentration, False


def _budget_dynamic_keep_counts(
    hard_score: torch.Tensor,
    topk: int,
    easy_threshold: float,
    hard_threshold: float,
    hard_score_high_is_hard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    easy_k = min(_BUDGET_DYNAMIC_EASY_K, topk)
    base_k = min(_BUDGET_DYNAMIC_BASE_K, topk)
    hard_k = min(_BUDGET_DYNAMIC_HARD_K, topk)
    keep_counts = torch.full(
        (hard_score.size(0), ),
        base_k,
        dtype=torch.long,
        device=hard_score.device,
    )
    class_ids = torch.ones_like(keep_counts)
    if hard_score_high_is_hard:
        easy_mask = torch.zeros_like(keep_counts, dtype=torch.bool)
        hard_mask = hard_score >= hard_threshold
    else:
        easy_mask = hard_score >= easy_threshold
        hard_mask = hard_score <= hard_threshold
    keep_counts = torch.where(easy_mask, torch.full_like(keep_counts, easy_k),
                              keep_counts)
    class_ids = torch.where(easy_mask, torch.zeros_like(class_ids), class_ids)
    keep_counts = torch.where(hard_mask, torch.full_like(keep_counts, hard_k),
                              keep_counts)
    class_ids = torch.where(hard_mask, torch.full_like(class_ids, 2),
                            class_ids)
    keep_counts = keep_counts.clamp(min=1, max=topk)
    return keep_counts, class_ids


def _budget_dynamic_routing_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    score_weights = _budget_score_weights(topk_weights)
    base_k = min(_BUDGET_DYNAMIC_BASE_K, topk)
    hard_score, hard_score_high_is_hard = _budget_hard_score(
        score_weights,
        topk,
        base_k,
    )

    keep_counts, class_ids = _budget_dynamic_keep_counts(
        hard_score=hard_score,
        topk=topk,
        easy_threshold=_BUDGET_DYNAMIC_EASY_THRESHOLD,
        hard_threshold=_BUDGET_DYNAMIC_HARD_THRESHOLD,
        hard_score_high_is_hard=hard_score_high_is_hard,
    )
    _record_budget_class_counts(class_ids)

    ranks = torch.arange(topk, device=hidden_states.device).view(1, topk)
    keep_mask = ranks < keep_counts.view(-1, 1)

    topk_weights_before_pruning = topk_weights
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _maybe_print_dynamic_routing_debug(
        topk_weights_before_pruning=topk_weights_before_pruning,
        threshold_weights=score_weights,
        topk_weights_after_pruning=topk_weights,
        keep_mask=keep_mask,
    )
    _record_routing_stats(keep_counts=keep_counts, topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )

    return topk_weights, topk_ids, token_expert_indices


def _layerbudget_dynamic_routing_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    score_weights = _budget_score_weights(topk_weights)
    base_k = min(_BUDGET_DYNAMIC_BASE_K, topk)
    hard_score, hard_score_high_is_hard = _budget_hard_score(
        score_weights,
        topk,
        base_k,
    )

    hardness = _layer_position_hardness(
        layer_index=_CURRENT_MOE_LAYER_INDEX,
        num_layers=_LAYER_BUDGET_DYNAMIC_NUM_LAYERS,
    )
    easy_threshold = min(
        1.0,
        _BUDGET_DYNAMIC_EASY_THRESHOLD +
        _LAYER_BUDGET_DYNAMIC_EASY_LAYER_ALPHA * hardness,
    )
    hard_threshold = min(
        easy_threshold,
        _BUDGET_DYNAMIC_HARD_THRESHOLD +
        _LAYER_BUDGET_DYNAMIC_HARD_LAYER_ALPHA * hardness,
    )
    keep_counts, class_ids = _budget_dynamic_keep_counts(
        hard_score=hard_score,
        topk=topk,
        easy_threshold=easy_threshold,
        hard_threshold=hard_threshold,
        hard_score_high_is_hard=hard_score_high_is_hard,
    )
    _record_budget_class_counts(class_ids)

    ranks = torch.arange(topk, device=hidden_states.device).view(1, topk)
    keep_mask = ranks < keep_counts.view(-1, 1)

    topk_weights_before_pruning = topk_weights
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _maybe_print_dynamic_routing_debug(
        topk_weights_before_pruning=topk_weights_before_pruning,
        threshold_weights=score_weights,
        topk_weights_after_pruning=topk_weights,
        keep_mask=keep_mask,
    )
    _record_routing_stats(keep_counts=keep_counts, topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )

    return topk_weights, topk_ids, token_expert_indices


def _banded_layerbudget_dynamic_routing_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer_index = _CURRENT_MOE_LAYER_INDEX
    if (_BAND_LAYER_BUDGET_PER_LAYER_K
            and layer_index is not None
            and layer_index < len(_BAND_LAYER_BUDGET_PER_LAYER_K)):
        return _fixed_layer_k_routing_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            keep_k=_BAND_LAYER_BUDGET_PER_LAYER_K[layer_index],
            renormalize=renormalize,
            indices_type=indices_type,
        )

    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    score_weights = _budget_score_weights(topk_weights)
    score_topn = min(_BUDGET_DYNAMIC_SCORE_TOPN, topk)
    concentration = score_weights[:, :score_topn].sum(dim=-1)

    easy_k = min(_BUDGET_DYNAMIC_EASY_K, topk)
    base_k = min(_BUDGET_DYNAMIC_BASE_K, topk)
    hard_k = min(_BUDGET_DYNAMIC_HARD_K, topk)
    hard_score, hard_score_high_is_hard = _budget_hard_score(
        score_weights,
        topk,
        base_k,
    )
    keep_counts = torch.full(
        (num_tokens, ),
        base_k,
        dtype=torch.long,
        device=hidden_states.device,
    )
    class_ids = torch.ones_like(keep_counts)
    profile_ids = _current_prompt_profile_ids_for_tokens(
        num_tokens,
        hidden_states.device,
    )
    middle_token_mask = _band_layer_budget_token_middle_mask(
        profile_ids,
        layer_index,
        num_tokens,
        hidden_states.device,
    )
    in_explicit_hard_layer = (
        layer_index is not None
        and bool(_BAND_LAYER_BUDGET_HARD_LAYERS)
        and layer_index in _BAND_LAYER_BUDGET_HARD_LAYERS
    )
    use_default_hard_layers = not _BAND_LAYER_BUDGET_HARD_LAYERS
    in_any_middle_band = in_explicit_hard_layer or (
        use_default_hard_layers
        and (
            _profile_layer_band_contains(
                0,
                layer_index,
                _BAND_LAYER_BUDGET_MIDDLE_START,
                _BAND_LAYER_BUDGET_MIDDLE_END,
            )
            or _profile_layer_band_contains(
                1,
                layer_index,
                _BAND_LAYER_BUDGET_MIDDLE_START,
                _BAND_LAYER_BUDGET_MIDDLE_END,
            )
            or _profile_layer_band_contains(
                2,
                layer_index,
                _BAND_LAYER_BUDGET_MIDDLE_START,
                _BAND_LAYER_BUDGET_MIDDLE_END,
            )
            or (
                layer_index is not None
                and _BAND_LAYER_BUDGET_EXTRA_MIDDLE_START >= 0
                and _BAND_LAYER_BUDGET_EXTRA_MIDDLE_START <= layer_index
                <= _BAND_LAYER_BUDGET_EXTRA_MIDDLE_END
            )
        )
    )
    hard_token_band_mask = middle_token_mask
    if in_explicit_hard_layer:
        hard_token_band_mask = torch.ones_like(middle_token_mask)
    decode_seq_len_target_mask = None
    in_any_hard_band = in_any_middle_band
    if (layer_index is not None
            and _BAND_LAYER_BUDGET_HARD_DECODE_START >= 0
            and _BAND_LAYER_BUDGET_HARD_DECODE_START <= layer_index
            <= _BAND_LAYER_BUDGET_HARD_DECODE_END):
        decode_band_mask = _band_layer_budget_phase_mask(
            "decode",
            num_tokens,
            hidden_states.device,
        )
        if decode_band_mask is not None:
            hard_token_band_mask = hard_token_band_mask | decode_band_mask
            decode_seq_len_target_mask = decode_band_mask & (
                ~middle_token_mask)
            in_any_hard_band = True
    prefill_segment_hard_ratio_override = (
        _band_layer_budget_token_prefill_segment_hard_ratios(
            profile_ids,
            hidden_states.device,
        ))
    in_easy_band = (
        layer_index is not None
        and (
            (bool(_BAND_LAYER_BUDGET_EASY_LAYERS)
             and layer_index in _BAND_LAYER_BUDGET_EASY_LAYERS)
            or (not _BAND_LAYER_BUDGET_EASY_LAYERS
                and _BAND_LAYER_BUDGET_EASY_START <= layer_index
                <= _BAND_LAYER_BUDGET_EASY_END)
        ))

    if in_any_hard_band:
        use_late_concentration = (
            _BAND_LAYER_BUDGET_LATE_CONCENTRATION_START >= 0
            and layer_index is not None
            and layer_index >= _BAND_LAYER_BUDGET_LATE_CONCENTRATION_START)
        if use_late_concentration:
            hard_score = concentration
            hard_score_high_is_hard = False
        hard_threshold_base = (
            _BAND_LAYER_BUDGET_LATE_CONCENTRATION_THRESHOLD
            if use_late_concentration
            and _BAND_LAYER_BUDGET_LATE_CONCENTRATION_THRESHOLD >= 0 else
            _BUDGET_DYNAMIC_HARD_THRESHOLD)
        hard_threshold = _band_layer_budget_token_hard_thresholds(
            profile_ids,
            float(hard_threshold_base),
            hard_score.dtype,
            hidden_states.device,
        )
        prior_boost = _band_layer_budget_hard_layer_prior_boost(layer_index)
        if prior_boost > 0:
            hard_threshold = hard_threshold + prior_boost
        if _BAND_LAYER_BUDGET_HARD_LAYER_SENSITIVITY_ALPHA > 0:
            layer_sensitivity = _ban_layer_sensitivity(
                device=hidden_states.device,
                dtype=concentration.dtype,
            ).clamp(0.0, 1.0)
            hard_threshold = (
                hard_threshold
                + _BAND_LAYER_BUDGET_HARD_LAYER_SENSITIVITY_ALPHA
                * layer_sensitivity).clamp(0.0, 1.0)
        else:
            hard_threshold = hard_threshold.clamp(0.0, 1.0)
        use_token_gate_quota = (
            _BAND_LAYER_BUDGET_TOKEN_GATE_RATIO >= 0
            and in_explicit_hard_layer
        )
        if use_token_gate_quota:
            hard_threshold_mask = _extreme_fraction_mask(
                concentration,
                _BAND_LAYER_BUDGET_TOKEN_GATE_RATIO,
                largest=False,
            )
        elif hard_score_high_is_hard:
            hard_threshold_mask = hard_score >= hard_threshold
        else:
            hard_threshold_mask = hard_score <= hard_threshold
        first_added_rank_weight = None
        if hard_k > base_k and base_k < topk:
            first_added_rank_weight = score_weights[:, base_k]
        use_mixed_rescue = (
            _BAND_LAYER_BUDGET_MIXED_RESCUE_MIN_RANK_WEIGHT >= 0
            and _BAND_LAYER_BUDGET_MIXED_RESCUE_MAX_CONCENTRATION >= 0
            and first_added_rank_weight is not None)
        if use_mixed_rescue:
            mixed_rescue_mask = (
                (~hard_threshold_mask)
                & (concentration <=
                   _BAND_LAYER_BUDGET_MIXED_RESCUE_MAX_CONCENTRATION)
                & (first_added_rank_weight >=
                   _BAND_LAYER_BUDGET_MIXED_RESCUE_MIN_RANK_WEIGHT))
            hard_threshold_mask = hard_threshold_mask | mixed_rescue_mask
        use_late_rescue = (
            not use_late_concentration
            and _BAND_LAYER_BUDGET_LATE_RESCUE_START >= 0
            and layer_index is not None
            and layer_index >= _BAND_LAYER_BUDGET_LATE_RESCUE_START
            and _BAND_LAYER_BUDGET_LATE_RESCUE_CONCENTRATION_THRESHOLD >= 0)
        if use_late_rescue:
            rescue_mask = (
                concentration
                <= _BAND_LAYER_BUDGET_LATE_RESCUE_CONCENTRATION_THRESHOLD)
            if _BAND_LAYER_BUDGET_LATE_RESCUE_MIXED_MAX >= 0:
                if hard_score_high_is_hard:
                    rescue_mask = (
                        rescue_mask
                        & (hard_score >=
                           _BAND_LAYER_BUDGET_LATE_RESCUE_MIXED_MAX))
                else:
                    rescue_mask = (
                        rescue_mask
                        & (hard_score <=
                           _BAND_LAYER_BUDGET_LATE_RESCUE_MIXED_MAX))
            hard_threshold_mask = hard_threshold_mask | rescue_mask
        hard_mask = hard_threshold_mask & hard_token_band_mask
        hard_phase_mask = _band_layer_budget_phase_mask(
            _BAND_LAYER_BUDGET_HARD_PHASE,
            num_tokens,
            hidden_states.device,
        )
        if hard_phase_mask is not None:
            effective_hard_phase_mask = hard_phase_mask
            if (_BAND_LAYER_BUDGET_HARD_PREFILL_START >= 0
                    and _BAND_LAYER_BUDGET_HARD_PREFILL_START <= layer_index
                    <= _BAND_LAYER_BUDGET_HARD_PREFILL_END):
                prefill_phase_mask = _band_layer_budget_phase_mask(
                    "prefill",
                    num_tokens,
                    hidden_states.device,
                )
                if prefill_phase_mask is not None:
                    effective_hard_phase_mask = (
                        effective_hard_phase_mask | prefill_phase_mask)
            hard_mask = hard_mask & effective_hard_phase_mask
        prefill_length_mask = _band_layer_budget_prefill_length_mask(
            num_tokens,
            hidden_states.device,
        )
        hard_mask = _apply_prefill_only_mask(
            hard_mask,
            prefill_length_mask,
            num_tokens,
            hidden_states.device,
        )
        prefill_segment_length_mask = (
            _band_layer_budget_prefill_segment_length_mask(
                num_tokens,
                hidden_states.device,
            ))
        hard_mask = _apply_prefill_only_mask(
            hard_mask,
            prefill_segment_length_mask,
            num_tokens,
            hidden_states.device,
        )
        prefill_density_mask = _band_layer_budget_prefill_density_mask(
            hard_threshold_mask,
            num_tokens,
            hidden_states.device,
        )
        hard_mask = _apply_prefill_only_mask(
            hard_mask,
            prefill_density_mask,
            num_tokens,
            hidden_states.device,
        )
        prefill_relative_position_mask = (
            _band_layer_budget_prefill_relative_position_mask(
                num_tokens,
                hidden_states.device,
            ))
        hard_mask = _apply_prefill_only_mask(
            hard_mask,
            prefill_relative_position_mask,
            num_tokens,
            hidden_states.device,
        )
        prefill_exclude_token_id_mask = (
            _band_layer_budget_prefill_exclude_token_id_mask(
                num_tokens,
                hidden_states.device,
            ))
        hard_mask = _apply_prefill_only_mask(
            hard_mask,
            prefill_exclude_token_id_mask,
            num_tokens,
            hidden_states.device,
        )
        hard_candidate_mask = hard_mask
        segment_cap_score = (
            -hard_score if hard_score_high_is_hard else hard_score)
        segment_cap_largest = False
        if (_BAND_LAYER_BUDGET_HARD_PREFILL_SEGMENT_CAP_SCORE == "rank4"
                and first_added_rank_weight is not None):
            segment_cap_score = first_added_rank_weight
            segment_cap_largest = True
        elif (_BAND_LAYER_BUDGET_HARD_PREFILL_SEGMENT_CAP_SCORE
              == "rank4_over_topn"
              and first_added_rank_weight is not None):
            segment_cap_score = (
                first_added_rank_weight / concentration.clamp_min(1e-12))
            segment_cap_largest = True
        prefill_segment_hard_ratio_mask = (
            _band_layer_budget_prefill_segment_hard_ratio_mask(
                hard_mask,
                segment_cap_score,
                num_tokens,
                hidden_states.device,
                (prefill_segment_hard_ratio_override
                 if prefill_segment_hard_ratio_override is not None else
                 -1.0),
                largest=segment_cap_largest,
            ))
        if prefill_segment_hard_ratio_mask is not None:
            hard_mask = _apply_prefill_only_mask(
                hard_mask,
                prefill_segment_hard_ratio_mask,
                num_tokens,
                hidden_states.device,
            )
        decode_seq_len_mask = _band_layer_budget_decode_seq_len_mask(
            num_tokens,
            hidden_states.device,
        )
        if decode_seq_len_mask is not None:
            if (_BAND_LAYER_BUDGET_HARD_DECODE_SEQ_LEN_SCOPE == "extra"
                    and decode_seq_len_target_mask is not None):
                hard_mask = (
                    hard_mask
                    & ((~decode_seq_len_target_mask) | decode_seq_len_mask))
            else:
                decode_phase_mask = _band_layer_budget_phase_mask(
                    "decode",
                    num_tokens,
                    hidden_states.device,
                )
                if decode_phase_mask is None:
                    hard_mask = hard_mask & decode_seq_len_mask
                else:
                    hard_mask = (
                        hard_mask
                        & ((~decode_phase_mask) | decode_seq_len_mask))
        decode_offset_mask = _band_layer_budget_decode_offset_mask(
            num_tokens,
            hidden_states.device,
        )
        if decode_offset_mask is not None:
            if (_BAND_LAYER_BUDGET_HARD_DECODE_SEQ_LEN_SCOPE == "extra"
                    and decode_seq_len_target_mask is not None):
                hard_mask = (
                    hard_mask
                    & ((~decode_seq_len_target_mask) | decode_offset_mask))
            else:
                decode_phase_mask = _band_layer_budget_phase_mask(
                    "decode",
                    num_tokens,
                    hidden_states.device,
                )
                if decode_phase_mask is None:
                    hard_mask = hard_mask & decode_offset_mask
                else:
                    hard_mask = (
                        hard_mask
                        & ((~decode_phase_mask) | decode_offset_mask))
        decode_token_id_mask = _band_layer_budget_decode_token_id_mask(
            num_tokens,
            hidden_states.device,
        )
        if decode_token_id_mask is not None:
            hard_mask = hard_mask & decode_token_id_mask
        hard_min_rank_weights = None
        if first_added_rank_weight is not None:
            hard_min_rank_weights = (
                _band_layer_budget_token_hard_min_rank_weights(
                    profile_ids,
                    _BAND_LAYER_BUDGET_HARD_MIN_RANK_WEIGHT,
                    first_added_rank_weight.dtype,
                    hidden_states.device,
                ))
        if hard_min_rank_weights is not None:
            hard_mask = (
                hard_mask
                & ((hard_min_rank_weights <= 0)
                   | (first_added_rank_weight >= hard_min_rank_weights)))
        if (_BAND_LAYER_BUDGET_HARD_MAX_RANK_WEIGHT < 1
                and first_added_rank_weight is not None):
            hard_mask = (
                hard_mask
                & (first_added_rank_weight <=
                   _BAND_LAYER_BUDGET_HARD_MAX_RANK_WEIGHT))
        if (_BAND_LAYER_BUDGET_HARD_DECODE_MIN_RANK_WEIGHT >= 0
                and first_added_rank_weight is not None):
            decode_phase_mask = _band_layer_budget_phase_mask(
                "decode",
                num_tokens,
                hidden_states.device,
            )
            if decode_phase_mask is not None:
                hard_mask = (
                    hard_mask
                    & ((~decode_phase_mask)
                       | (first_added_rank_weight >=
                          _BAND_LAYER_BUDGET_HARD_DECODE_MIN_RANK_WEIGHT)))
        _record_budget_hard_rank_weight_stats(
            first_added_rank_weight,
            hard_candidate_mask,
            hard_mask,
        )
        _record_budget_hard_seq_len_stats(
            hard_candidate_mask,
            hard_mask,
        )
        _record_budget_hard_decode_offset_stats(
            hard_candidate_mask,
            hard_mask,
        )
        _record_budget_prefill_segment_density_stats(
            hard_threshold_mask,
            hard_mask,
        )
        keep_counts = torch.where(
            hard_mask,
            torch.full_like(keep_counts, hard_k),
            keep_counts,
        )
        class_ids = torch.where(
            hard_mask,
            torch.full_like(class_ids, 2),
            class_ids,
        )
    if in_easy_band:
        if (_BAND_LAYER_BUDGET_TOKEN_GATE_RATIO >= 0
                and _BAND_LAYER_BUDGET_EASY_LAYERS):
            easy_mask = _extreme_fraction_mask(
                concentration,
                _BAND_LAYER_BUDGET_TOKEN_GATE_RATIO,
                largest=True,
            )
        else:
            easy_mask = concentration >= _BUDGET_DYNAMIC_EASY_THRESHOLD
        if in_any_middle_band:
            easy_mask = easy_mask & (~middle_token_mask)
        easy_phase_mask = _band_layer_budget_easy_phase_mask(
            num_tokens,
            hidden_states.device,
        )
        if easy_phase_mask is not None:
            easy_mask = easy_mask & easy_phase_mask
        if _BAND_LAYER_BUDGET_EASY_TAIL_THRESHOLD < 1.0:
            dropped_start = min(easy_k, topk)
            dropped_end = min(max(base_k, dropped_start), topk)
            if dropped_end > dropped_start:
                dropped_tail_mass = score_weights[:,
                                                  dropped_start:dropped_end].sum(
                                                      dim=-1)
            else:
                dropped_tail_mass = torch.zeros_like(concentration)
            easy_mask = (
                easy_mask
                & (dropped_tail_mass <=
                   _BAND_LAYER_BUDGET_EASY_TAIL_THRESHOLD))
        keep_counts = torch.where(
            easy_mask,
            torch.full_like(keep_counts, easy_k),
            keep_counts,
        )
        class_ids = torch.where(
            easy_mask,
            torch.zeros_like(class_ids),
            class_ids,
        )
    keep_counts = keep_counts.clamp(min=1, max=topk)
    _record_budget_class_counts(class_ids)

    ranks = torch.arange(topk, device=hidden_states.device).view(1, topk)
    keep_mask = ranks < keep_counts.view(-1, 1)

    topk_weights_before_pruning = topk_weights
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _maybe_print_dynamic_routing_debug(
        topk_weights_before_pruning=topk_weights_before_pruning,
        threshold_weights=score_weights,
        topk_weights_after_pruning=topk_weights,
        keep_mask=keep_mask,
    )
    _record_routing_stats(keep_counts=keep_counts, topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )

    return topk_weights, topk_ids, token_expert_indices


def _fixed_layer_k_routing_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    keep_k: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )
    keep_k = min(max(int(keep_k), 1), topk)
    keep_counts = torch.full(
        (num_tokens,),
        keep_k,
        dtype=torch.long,
        device=hidden_states.device,
    )
    base_k = min(_BUDGET_DYNAMIC_BASE_K, topk)
    if keep_k < base_k:
        class_id = 0
    elif keep_k > base_k:
        class_id = 2
    else:
        class_id = 1
    _record_budget_class_counts(torch.full_like(keep_counts, class_id))

    ranks = torch.arange(topk, device=hidden_states.device).view(1, topk)
    keep_mask = ranks < keep_k
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _record_routing_stats(keep_counts=keep_counts, topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )
    return topk_weights, topk_ids, token_expert_indices


def _ban_sensitive_layerbudget_dynamic_routing_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    base_k = min(_BUDGET_DYNAMIC_BASE_K, topk)
    hard_k = min(_BUDGET_DYNAMIC_HARD_K, topk)
    keep_counts = torch.full(
        (num_tokens, ),
        base_k,
        dtype=torch.long,
        device=hidden_states.device,
    )
    class_ids = torch.ones_like(keep_counts)

    layer_sensitivity = _ban_layer_sensitivity(
        device=hidden_states.device,
        dtype=topk_weights.dtype,
    ).clamp(0.0, 1.0)
    hard_mask = (
        layer_sensitivity >= _BUDGET_DYNAMIC_HARD_THRESHOLD).expand(
            num_tokens)
    keep_counts = torch.where(
        hard_mask,
        torch.full_like(keep_counts, hard_k),
        keep_counts,
    )
    class_ids = torch.where(
        hard_mask,
        torch.full_like(class_ids, 2),
        class_ids,
    )
    keep_counts = keep_counts.clamp(min=1, max=topk)
    _record_budget_class_counts(class_ids)

    ranks = torch.arange(topk, device=hidden_states.device).view(1, topk)
    keep_mask = ranks < keep_counts.view(-1, 1)

    topk_weights_before_pruning = topk_weights
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _maybe_print_dynamic_routing_debug(
        topk_weights_before_pruning=topk_weights_before_pruning,
        threshold_weights=topk_weights_before_pruning,
        topk_weights_after_pruning=topk_weights,
        keep_mask=keep_mask,
    )
    _record_routing_stats(keep_counts=keep_counts, topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )

    return topk_weights, topk_ids, token_expert_indices


def _maybe_print_dynamic_routing_debug(
    topk_weights_before_pruning: torch.Tensor,
    threshold_weights: torch.Tensor,
    topk_weights_after_pruning: torch.Tensor,
    keep_mask: torch.Tensor,
) -> None:
    global _DEBUG_PRINT_COUNT

    if not _DEBUG_ROUTING or _DEBUG_PRINT_COUNT >= _DEBUG_MAX_PRINTS:
        return

    with torch.no_grad():
        keep_counts = keep_mask.sum(dim=-1).float()
        sample_size = min(3, topk_weights_after_pruning.size(0))
        before = topk_weights_before_pruning[:sample_size].detach().cpu()
        threshold_scores = threshold_weights[:sample_size].detach().cpu()
        after = topk_weights_after_pruning[:sample_size].detach().cpu()
        counts = keep_counts[:sample_size].detach().cpu()
        print(
            "[expert_pruning][Dynamic_Routing] "
            f"threshold={_DYNAMIC_ROUTING_THRESHOLD} "
            f"score_source={_DYNAMIC_ROUTING_SCORE_SOURCE} "
            f"topk={topk_weights_after_pruning.size(1)} "
            f"avg_keep={keep_counts.mean().item():.4f} "
            f"min_keep={keep_counts.min().item():.0f} "
            f"max_keep={keep_counts.max().item():.0f} "
            f"sample_keep={counts.tolist()} "
            f"sample_before={before.tolist()} "
            f"sample_threshold_scores={threshold_scores.tolist()} "
            f"sample_after={after.tolist()}",
            flush=True,
        )
    _DEBUG_PRINT_COUNT += 1


def _topk_biased_renorm_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )

    if renormalize:
        topk_weights = _biased_renormalize_topk_weights(
            topk_weights,
            keep_topn=_BIASED_RENORM_KEEP_TOPN,
        )
    _record_routing_stats(
        keep_counts=torch.full((num_tokens, ),
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device),
        topk=topk,
    )

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )

    return topk_weights, topk_ids, token_expert_indices


def _router_value_features(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float()
    adjacent_ratio = weights[:, 1:] / weights[:, :-1].clamp_min(1e-12)
    adjacent_gap = weights[:, :-1] - weights[:, 1:]
    cumulative = weights.cumsum(dim=-1)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(
        dim=-1, keepdim=True)
    return torch.cat(
        (weights, adjacent_ratio, adjacent_gap, cumulative, entropy),
        dim=-1,
    )


def _router_value_dynamic_routing_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if topk != 8:
        raise ValueError(
            "RouterValue_Dynamic_Routing currently requires model topk=8")
    base_k = min(_ROUTER_VALUE_BASE_K, topk)
    if base_k >= topk:
        raise ValueError("router_value_base_k must be smaller than model topk")
    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=True,
    )
    feature_weights = _renormalize_topk_weights(topk_weights)
    coefficients = _get_router_value_coefficients_for_device(
        feature_weights.device)
    scores = _router_value_features(feature_weights) @ coefficients
    layer_index = _CURRENT_MOE_LAYER_INDEX
    if layer_index is None or not 0 <= layer_index < len(
            _ROUTER_VALUE_LAYER_THRESHOLDS):
        raise RuntimeError(
            "RouterValue_Dynamic_Routing requires a valid current MoE layer")
    allow_promotion = _ROUTER_VALUE_SWAP_DIRECTION != "demote_only"
    allow_demotion = _ROUTER_VALUE_SWAP_DIRECTION != "promote_only"
    quota_count = min(
        int(_ROUTER_VALUE_QUOTA_RATIO * num_tokens),
        num_tokens // 2,
    )
    selected_mask = torch.zeros_like(scores, dtype=torch.bool)
    if allow_promotion:
        if _ROUTER_VALUE_SELECTION_MODE == "quota":
            if quota_count > 0:
                promoted_indices = torch.topk(
                    scores, quota_count, largest=True, sorted=False
                ).indices
                selected_mask.scatter_(0, promoted_indices, True)
        else:
            threshold = _ROUTER_VALUE_LAYER_THRESHOLDS[layer_index]
            selected_mask = scores >= threshold
    if _ROUTER_VALUE_ARTIFACT_FORMAT == "router_value_fixed_swap_v1":
        if base_k <= 1:
            raise ValueError("Fixed-swap RouterValue requires base_k >= 2")
        lower_coefficients = (
            _get_router_value_lower_coefficients_for_device(
                feature_weights.device))
        lower_scores = (
            _router_value_features(feature_weights) @ lower_coefficients)
        demoted_mask = torch.zeros_like(selected_mask)
        if allow_demotion:
            if _ROUTER_VALUE_SELECTION_MODE == "quota":
                if quota_count > 0:
                    demotion_scores = lower_scores.masked_fill(
                        selected_mask, float("inf"))
                    demoted_indices = torch.topk(
                        demotion_scores,
                        quota_count,
                        largest=False,
                        sorted=False,
                    ).indices
                    demoted_mask.scatter_(0, demoted_indices, True)
            else:
                demotion_threshold = (
                    _ROUTER_VALUE_DEMOTION_THRESHOLDS[layer_index])
                demoted_mask = (
                    ~selected_mask & (lower_scores <= demotion_threshold))
        keep_counts = torch.full_like(
            selected_mask, base_k, dtype=torch.long)
        keep_counts = torch.where(
            selected_mask,
            torch.full_like(keep_counts, base_k + 1),
            keep_counts,
        )
        keep_counts = torch.where(
            demoted_mask,
            torch.full_like(keep_counts, base_k - 1),
            keep_counts,
        )
        class_ids = torch.where(
            selected_mask,
            torch.full_like(keep_counts, 2),
            torch.where(
                demoted_mask,
                torch.zeros_like(keep_counts),
                torch.ones_like(keep_counts),
            ),
        )
    else:
        keep_counts = torch.where(
            selected_mask,
            torch.full_like(selected_mask, base_k + 1, dtype=torch.long),
            torch.full_like(selected_mask, base_k, dtype=torch.long),
        )
        class_ids = torch.where(
            selected_mask,
            torch.full_like(keep_counts, 2),
            torch.ones_like(keep_counts),
        )
    _record_budget_class_counts(class_ids)

    ranks = torch.arange(topk, device=hidden_states.device).view(1, topk)
    keep_mask = ranks < keep_counts.view(-1, 1)
    topk_weights = topk_weights.masked_fill(~keep_mask, 0.0)
    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _record_routing_stats(keep_counts=keep_counts, topk=topk)

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )
    return topk_weights, topk_ids, token_expert_indices


def torch_fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype] = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for vLLM's ``fused_topk``.

    Args mirror ``vllm.model_executor.layers.fused_moe.fused_moe.fused_topk``.
    It returns ``topk_weights``, ``topk_ids`` and ``token_expert_indices``.
    """
    if scoring_func != "softmax":
        raise ValueError(
            "Local expert routing currently supports only scoring_func='softmax', "
            f"got {scoring_func!r}")
    if _ROUTING_METHOD == "naee":
        return _naee_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "mc_moe":
        return _mc_moe_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "eac_moe":
        return _eac_moe_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "ban":
        return _ban_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "diep":
        return _diep_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "routervalue_dynamic_routing":
        return _router_value_dynamic_routing_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "dynamic_routing":
        return _dynamic_routing_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "layerwise_dynamic_routing":
        return _layerwise_dynamic_routing_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "budget_dynamic_routing":
        return _budget_dynamic_routing_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "layerbudget_dynamic_routing":
        return _layerbudget_dynamic_routing_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "bandedlayerbudget_dynamic_routing":
        return _banded_layerbudget_dynamic_routing_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "bansensitivelayerbudget_dynamic_routing":
        return _ban_sensitive_layerbudget_dynamic_routing_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )
    if _ROUTING_METHOD == "topk_biased_renorm":
        return _topk_biased_renorm_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=indices_type,
        )

    num_tokens = hidden_states.size(0)
    topk_weights, topk_ids = _topk_softmax(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        sorted=False,
    )

    if renormalize:
        topk_weights = _renormalize_topk_weights(topk_weights)
    _record_routing_stats(
        keep_counts=torch.full((num_tokens, ),
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device),
        topk=topk,
    )

    topk_ids_dtype = torch.int32 if indices_type is None else indices_type
    topk_ids = topk_ids.to(dtype=topk_ids_dtype)
    topk_weights = topk_weights.to(dtype=torch.float32)
    token_expert_indices = _token_expert_indices(
        num_tokens=num_tokens,
        topk=topk,
        device=hidden_states.device,
    )

    return topk_weights, topk_ids, token_expert_indices
