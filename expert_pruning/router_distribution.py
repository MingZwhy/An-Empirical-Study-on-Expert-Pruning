"""Native MoE router distribution observer for vLLM serving runs.

Observes ``gating_output`` (router logits) at the original ``fused_topk`` entry
point, delegates routing to the native vLLM implementation unchanged, and
accumulates storage-efficient online statistics plus a bounded reservoir of
sorted softmax vectors.
"""

from __future__ import annotations

import atexit
import inspect
import json
import os
import struct
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from expert_pruning.routing import (
    current_eac_prefill_mask_tensor,
    current_moe_layer_index,
)

STAGES = ("prefill", "decode", "unknown")
CUM_RANKS = (1, 2, 4, 8, 16, 32, 64)
CONCENTRATION_BOUNDS = [
    0.0,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    1.0,
]

_ENABLED = False
_SKIP_DEPTH = 0
_OUTPUT_DIR: Optional[Path] = None
_SAMPLE_RATE = 0.005
_SAMPLE_CAP = 2048
_HISTOGRAM_BINS = len(CONCENTRATION_BOUNDS)
_FLUSH_INTERVAL = 512
_DATASET: str = ""
_MODEL_PATH: str = ""
_HARNESS: str = ""
_TASK: str = ""
_SEED = 0
_NUM_EXPERTS: Optional[int] = None

_INVOCATION_COUNT = 0
_ROUTING_CALLS = 0
_FLUSH_CALLS = 0

# layer -> stage -> accumulator
_ACCUMULATORS: dict[int, dict[str, "_StageAccumulator"]] = {}
_RESERVOIRS: dict[int, dict[str, "_Reservoir"]] = {}


def _is_cuda_graph_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is None or not hasattr(compiler, "is_compiling"):
        return False
    return bool(compiler.is_compiling())


def _is_tp_rank_zero() -> bool:
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank()) == 0
    except Exception:
        return True


def is_router_distribution_enabled() -> bool:
    return _ENABLED


def is_router_distribution_skipped() -> bool:
    return _SKIP_DEPTH > 0


@contextmanager
def skip_router_distribution():
    """Disable observation inside vLLM dummy/profile runs."""
    global _SKIP_DEPTH
    _SKIP_DEPTH += 1
    try:
        yield
    finally:
        _SKIP_DEPTH -= 1


def configure_router_distribution(
    *,
    enabled: bool,
    output_dir: str | None = None,
    sample_rate: float = 0.005,
    sample_cap: int = 2048,
    histogram_bins: int | None = None,
    flush_interval: int = 512,
    dataset: str = "",
    model_path: str = "",
    harness: str = "",
    task: str = "",
    seed: int = 0,
) -> None:
    global _ENABLED, _OUTPUT_DIR, _SAMPLE_RATE, _SAMPLE_CAP
    global _HISTOGRAM_BINS, _FLUSH_INTERVAL, _DATASET, _MODEL_PATH
    global _HARNESS, _TASK, _SEED, _NUM_EXPERTS
    global _INVOCATION_COUNT, _ROUTING_CALLS, _FLUSH_CALLS
    global _ACCUMULATORS, _RESERVOIRS

    if not enabled:
        _ENABLED = False
        _OUTPUT_DIR = None
        return

    if not 0 < sample_rate <= 1:
        raise ValueError("router_distribution sample_rate must be in (0, 1]")
    if sample_cap < 1:
        raise ValueError("router_distribution sample_cap must be >= 1")
    if flush_interval < 1:
        raise ValueError("router_distribution flush_interval must be >= 1")
    if output_dir is None:
        raise ValueError("router_distribution output_dir is required when enabled")

    _ENABLED = True
    _OUTPUT_DIR = Path(output_dir)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _SAMPLE_RATE = float(sample_rate)
    _SAMPLE_CAP = int(sample_cap)
    _HISTOGRAM_BINS = int(histogram_bins or len(CONCENTRATION_BOUNDS))
    _FLUSH_INTERVAL = int(flush_interval)
    _DATASET = dataset
    _MODEL_PATH = model_path
    _HARNESS = harness
    _TASK = task
    _SEED = int(seed)
    _NUM_EXPERTS = None
    _INVOCATION_COUNT = 0
    _ROUTING_CALLS = 0
    _FLUSH_CALLS = 0
    _ACCUMULATORS = {}
    _RESERVOIRS = {}


def _stable_hash64(*parts: Any) -> int:
    ints: list[int] = []
    for part in parts:
        if isinstance(part, str):
            ints.append(zlib.adler32(part.encode("utf-8")) & 0xFFFFFFFF)
        else:
            ints.append(int(part))
    payload = struct.pack("<" + "q" * len(ints), *ints)
    return zlib.adler32(payload) & 0xFFFFFFFF


@dataclass
class _StageAccumulator:
    num_experts: int
    device: torch.device
    token_count: int = 0
    sorted_prob_sum: torch.Tensor = field(init=False)
    sorted_prob_sq_sum: torch.Tensor = field(init=False)
    cum_mass_sum: dict[int, float] = field(default_factory=dict)
    entropy_sum: float = 0.0
    entropy_sq_sum: float = 0.0
    hhi_sum: float = 0.0
    hhi_sq_sum: float = 0.0
    margin_sum: float = 0.0
    margin_sq_sum: float = 0.0
    top1_hist: list[int] = field(default_factory=list)
    entropy_hist: list[int] = field(default_factory=list)
    margin_hist: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sorted_prob_sum = torch.zeros(
            self.num_experts,
            dtype=torch.float64,
            device=self.device,
        )
        self.sorted_prob_sq_sum = torch.zeros(
            self.num_experts,
            dtype=torch.float64,
            device=self.device,
        )
        for rank in CUM_RANKS:
            self.cum_mass_sum[rank] = 0.0
        self.top1_hist = [0] * _HISTOGRAM_BINS
        self.entropy_hist = [0] * _HISTOGRAM_BINS
        self.margin_hist = [0] * _HISTOGRAM_BINS

    def add_batch(
        self,
        sorted_probs: torch.Tensor,
        probs: torch.Tensor,
    ) -> None:
        count = int(sorted_probs.size(0))
        if count <= 0:
            return
        self.token_count += count
        self.sorted_prob_sum += sorted_probs.sum(dim=0, dtype=torch.float64)
        self.sorted_prob_sq_sum += sorted_probs.square().sum(
            dim=0, dtype=torch.float64)
        for rank in CUM_RANKS:
            k = min(rank, sorted_probs.size(1))
            self.cum_mass_sum[rank] += float(
                sorted_probs[:, :k].sum(dtype=torch.float64).item())
        log_probs = probs.clamp_min(1e-20).log()
        entropy = -(probs * log_probs).sum(dim=-1)
        hhi = probs.square().sum(dim=-1)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        self.entropy_sum += float(entropy.sum(dtype=torch.float64).item())
        self.entropy_sq_sum += float(
            entropy.square().sum(dtype=torch.float64).item())
        self.hhi_sum += float(hhi.sum(dtype=torch.float64).item())
        self.hhi_sq_sum += float(hhi.square().sum(dtype=torch.float64).item())
        self.margin_sum += float(margin.sum(dtype=torch.float64).item())
        self.margin_sq_sum += float(
            margin.square().sum(dtype=torch.float64).item())
        self._add_histogram(self.top1_hist, sorted_probs[:, 0])
        self._add_histogram(self.entropy_hist, entropy)
        self._add_histogram(self.margin_hist, margin)

    @staticmethod
    def _add_histogram(hist: list[int], values: torch.Tensor) -> None:
        cpu = values.detach().float().cpu().numpy()
        for value in cpu:
            idx = int(np.searchsorted(CONCENTRATION_BOUNDS, float(value),
                                      side="right") - 1)
            idx = max(0, min(idx, len(hist) - 1))
            hist[idx] += 1


@dataclass
class _Reservoir:
    cap: int
    sample_rate: float
    seed: int
    layer: int
    stage: str
    seen: int = 0
    vectors: list[np.ndarray] = field(default_factory=list)

    def maybe_add(self, sorted_probs: torch.Tensor) -> None:
        for row in sorted_probs.detach().float().cpu().numpy():
            self.seen += 1
            token_key = _stable_hash64(self.seed, self.layer, self.stage,
                                       self.seen)
            if (token_key % 100_000) >= int(self.sample_rate * 100_000):
                continue
            vec = np.sort(row)[::-1].astype(np.float16)
            if len(self.vectors) < self.cap:
                self.vectors.append(vec)
                continue
            replace_idx = _stable_hash64(token_key, 17) % self.cap
            replace_gate = _stable_hash64(token_key, 23) % self.seen
            if replace_gate < self.cap:
                self.vectors[replace_idx] = vec


def _get_accumulator(layer: int, stage: str,
                     num_experts: int,
                     device: torch.device) -> _StageAccumulator:
    layer_map = _ACCUMULATORS.setdefault(layer, {})
    accum = layer_map.get(stage)
    if accum is None:
        accum = _StageAccumulator(num_experts=num_experts, device=device)
        layer_map[stage] = accum
    return accum


def _get_reservoir(layer: int, stage: str) -> _Reservoir:
    layer_map = _RESERVOIRS.setdefault(layer, {})
    reservoir = layer_map.get(stage)
    if reservoir is None:
        reservoir = _Reservoir(
            cap=_SAMPLE_CAP,
            sample_rate=_SAMPLE_RATE,
            seed=_SEED,
            layer=layer,
            stage=stage,
        )
        layer_map[stage] = reservoir
    return reservoir


def _stage_masks(num_tokens: int,
                 device: torch.device) -> dict[str, torch.Tensor]:
    prefill_mask = current_eac_prefill_mask_tensor()
    if prefill_mask is None or prefill_mask.numel() != num_tokens:
        return {"unknown": torch.ones(num_tokens, dtype=torch.bool,
                                       device=device)}
    prefill_mask = prefill_mask.to(device=device, dtype=torch.bool)
    decode_mask = ~prefill_mask
    masks: dict[str, torch.Tensor] = {}
    if bool(prefill_mask.any()):
        masks["prefill"] = prefill_mask
    if bool(decode_mask.any()):
        masks["decode"] = decode_mask
    if not masks:
        masks["unknown"] = torch.ones(num_tokens, dtype=torch.bool,
                                      device=device)
    return masks


def observe_gating_output(gating_output: torch.Tensor) -> None:
    global _INVOCATION_COUNT, _ROUTING_CALLS, _NUM_EXPERTS

    if (not _ENABLED or _SKIP_DEPTH > 0 or _OUTPUT_DIR is None
            or _is_cuda_graph_capturing() or _is_torch_compiling()
            or not _is_tp_rank_zero()):
        return

    _INVOCATION_COUNT += 1
    _ROUTING_CALLS += 1

    layer = current_moe_layer_index()
    if layer is None:
        layer = -1

    num_tokens, num_experts = gating_output.shape
    if _NUM_EXPERTS is None:
        _NUM_EXPERTS = int(num_experts)
    device = gating_output.device
    probs = torch.softmax(gating_output.float(), dim=-1)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)

    for stage, mask in _stage_masks(num_tokens, device).items():
        if not bool(mask.any()):
            continue
        stage_sorted = sorted_probs[mask]
        stage_probs = probs[mask]
        accum = _get_accumulator(layer, stage, num_experts, device)
        accum.add_batch(stage_sorted, stage_probs)
        _get_reservoir(layer, stage).maybe_add(stage_sorted)

    if _ROUTING_CALLS % _FLUSH_INTERVAL == 0:
        flush_snapshot()


def _drain_accumulator(accum: _StageAccumulator) -> dict[str, Any]:
    count = accum.token_count
    sorted_mean = (
        (accum.sorted_prob_sum / count).detach().cpu().tolist()
        if count else [0.0] * accum.num_experts)
    sorted_var = []
    if count:
        mean_t = accum.sorted_prob_sum / count
        var_t = accum.sorted_prob_sq_sum / count - mean_t.square()
        sorted_var = var_t.clamp_min(0.0).detach().cpu().tolist()
    else:
        sorted_var = [0.0] * accum.num_experts

    def _mean_std(sum_val: float, sq_sum: float) -> tuple[float, float]:
        if count <= 0:
            return 0.0, 0.0
        mean = sum_val / count
        var = max(0.0, sq_sum / count - mean * mean)
        return mean, var**0.5

    entropy_mean, entropy_std = _mean_std(accum.entropy_sum,
                                          accum.entropy_sq_sum)
    hhi_mean, hhi_std = _mean_std(accum.hhi_sum, accum.hhi_sq_sum)
    margin_mean, margin_std = _mean_std(accum.margin_sum, accum.margin_sq_sum)
    effective_experts = float(np.exp(entropy_mean)) if count else 0.0

    return {
        "token_count": count,
        "num_experts": accum.num_experts,
        "sorted_prob_mean": sorted_mean,
        "sorted_prob_var": sorted_var,
        "cum_mass_mean": {
            str(rank): (accum.cum_mass_sum[rank] / count if count else 0.0)
            for rank in CUM_RANKS
        },
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "effective_experts_mean": effective_experts,
        "hhi_mean": hhi_mean,
        "hhi_std": hhi_std,
        "top1_top2_margin_mean": margin_mean,
        "top1_top2_margin_std": margin_std,
        "histogram_bounds": CONCENTRATION_BOUNDS,
        "top1_hist": accum.top1_hist,
        "entropy_hist": accum.entropy_hist,
        "margin_hist": accum.margin_hist,
    }


def build_snapshot_payload() -> dict[str, Any]:
    layers: dict[str, dict[str, Any]] = {}
    for layer, stage_map in sorted(_ACCUMULATORS.items()):
        layers[str(layer)] = {
            stage: _drain_accumulator(accum)
            for stage, accum in sorted(stage_map.items())
        }
    reservoirs: dict[str, dict[str, Any]] = {}
    for layer, stage_map in sorted(_RESERVOIRS.items()):
        reservoirs[str(layer)] = {
            stage: {
                "seen": reservoir.seen,
                "stored": len(reservoir.vectors),
                "cap": reservoir.cap,
                "sample_rate": reservoir.sample_rate,
            }
            for stage, reservoir in sorted(stage_map.items())
        }
    return {
        "schema_version": 1,
        "pid": os.getpid(),
        "dataset": _DATASET,
        "task": _TASK,
        "harness": _HARNESS,
        "model_path": _MODEL_PATH,
        "num_experts": _NUM_EXPERTS,
        "sample_rate": _SAMPLE_RATE,
        "sample_cap": _SAMPLE_CAP,
        "histogram_bins": _HISTOGRAM_BINS,
        "flush_interval": _FLUSH_INTERVAL,
        "seed": _SEED,
        "invocation_count": _INVOCATION_COUNT,
        "routing_calls": _ROUTING_CALLS,
        "layers": layers,
        "reservoirs": reservoirs,
    }


def _write_npz_snapshot(base_path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for layer, stage_map in _ACCUMULATORS.items():
        for stage, accum in stage_map.items():
            prefix = f"layer{layer}_{stage}"
            arrays[f"{prefix}_sorted_prob_mean"] = np.array(
                _drain_accumulator(accum)["sorted_prob_mean"],
                dtype=np.float32)
            arrays[f"{prefix}_token_count"] = np.array([accum.token_count],
                                                       dtype=np.int64)
    for layer, stage_map in _RESERVOIRS.items():
        for stage, reservoir in stage_map.items():
            prefix = f"layer{layer}_{stage}"
            if reservoir.vectors:
                arrays[f"{prefix}_reservoir"] = np.stack(reservoir.vectors)
            arrays[f"{prefix}_reservoir_seen"] = np.array([reservoir.seen],
                                                          dtype=np.int64)
    if arrays:
        npz_path = Path(f"{base_path}.npz")
        # np.savez_compressed appends ".npz" when absent; avoid ".npz.tmp" -> ".npz.tmp.npz".
        tmp_npz = Path(f"{base_path}.tmp.npz")
        np.savez_compressed(Path(f"{base_path}.tmp"), **arrays)
        os.replace(tmp_npz, npz_path)


def flush_snapshot() -> None:
    global _FLUSH_CALLS
    if not _ENABLED or _OUTPUT_DIR is None:
        return
    _FLUSH_CALLS += 1
    pid = os.getpid()
    json_path = _OUTPUT_DIR / f"router_distribution_{pid}.json"
    tmp_json = _OUTPUT_DIR / f".router_distribution_{pid}.json.tmp"
    payload = build_snapshot_payload()
    with tmp_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(tmp_json, json_path)
    _write_npz_snapshot(_OUTPUT_DIR / f"router_distribution_{pid}")


def _should_strict_assert_observation() -> bool:
    """Only MoE worker processes on TP rank 0 must observe routing."""
    if not _is_tp_rank_zero():
        return False
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        if int(get_tensor_model_parallel_world_size()) <= 1:
            return False
        return int(get_tensor_model_parallel_rank()) == 0
    except Exception:
        return False


def _assert_observation_ran() -> None:
    if not _ENABLED or _INVOCATION_COUNT > 0:
        return
    if not _should_strict_assert_observation():
        return
    raise RuntimeError(
        "router distribution observer was enabled but never invoked; "
        "native fused_topk wrapper did not observe any routing calls")


@atexit.register
def _router_distribution_atexit() -> None:
    if not _ENABLED:
        return
    try:
        flush_snapshot()
        _assert_observation_ran()
    except Exception as exc:  # noqa: BLE001
        print(f"[router_distribution] final flush/check failed: {exc}")


def wrap_fused_topk_observer(
    original: Callable,
) -> Callable:
    """Return a wrapper that observes logits then delegates to ``original``."""
    signature = inspect.signature(original)

    def observed(*args: Any, **kwargs: Any):
        bound = signature.bind_partial(*args, **kwargs)
        gating_output = bound.arguments.get("gating_output")
        if gating_output is None and len(args) >= 2:
            gating_output = args[1]
        if isinstance(gating_output, torch.Tensor):
            observe_gating_output(gating_output)
        return original(*args, **kwargs)

    observed.__name__ = getattr(original, "__name__", "fused_topk")
    observed.__doc__ = getattr(original, "__doc__", None)
    observed._router_distribution_wrapper = True  # type: ignore[attr-defined]
    observed._router_distribution_original = original  # type: ignore[attr-defined]
    return observed


def merge_worker_snapshots(paths: list[Path]) -> dict[str, Any]:
    """Merge per-worker JSON snapshots without double-counting tokens."""
    merged_layers: dict[str, dict[str, dict[str, Any]]] = {}
    total_invocations = 0
    meta: dict[str, Any] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            item = json.load(handle)
        total_invocations += int(item.get("invocation_count", 0))
        meta.setdefault("dataset", item.get("dataset"))
        meta.setdefault("task", item.get("task"))
        meta.setdefault("harness", item.get("harness"))
        meta.setdefault("model_path", item.get("model_path"))
        meta.setdefault("num_experts", item.get("num_experts"))
        for layer, stage_map in item.get("layers", {}).items():
            layer_out = merged_layers.setdefault(layer, {})
            for stage, stats in stage_map.items():
                dst = layer_out.setdefault(stage, {
                    "token_count": 0,
                    "sorted_prob_mean": None,
                    "sorted_prob_m2": None,
                    "cum_mass_mean": {str(r): 0.0 for r in CUM_RANKS},
                    "entropy_mean": 0.0,
                    "entropy_m2": 0.0,
                    "hhi_mean": 0.0,
                    "hhi_m2": 0.0,
                    "margin_mean": 0.0,
                    "margin_m2": 0.0,
                    "top1_hist": [0] * _HISTOGRAM_BINS,
                    "entropy_hist": [0] * _HISTOGRAM_BINS,
                    "margin_hist": [0] * _HISTOGRAM_BINS,
                })
                n_old = int(dst["token_count"])
                n_new = int(stats.get("token_count", 0))
                n_total = n_old + n_new
                dst["token_count"] = n_total
                if n_new <= 0:
                    continue
                mean_new = np.array(stats["sorted_prob_mean"], dtype=np.float64)
                if dst["sorted_prob_mean"] is None:
                    dst["sorted_prob_mean"] = mean_new * n_new
                    dst["sorted_prob_m2"] = np.array(stats["sorted_prob_var"],
                                                     dtype=np.float64) * n_new
                else:
                    mean_old = np.array(dst["sorted_prob_mean"],
                                        dtype=np.float64) / max(n_old, 1)
                    dst["sorted_prob_mean"] = (mean_old * n_old +
                                               mean_new * n_new)
                    dst["sorted_prob_m2"] = (
                        np.array(dst["sorted_prob_m2"], dtype=np.float64) +
                        np.array(stats["sorted_prob_var"], dtype=np.float64) *
                        n_new)
                for rank in CUM_RANKS:
                    key = str(rank)
                    dst["cum_mass_mean"][key] += float(
                        stats.get("cum_mass_mean", {}).get(key, 0.0) * n_new)
                for src_key, dst_key in (
                        ("entropy_mean", "entropy_mean"),
                        ("hhi_mean", "hhi_mean"),
                        ("top1_top2_margin_mean", "margin_mean"),
                ):
                    old_mean = float(dst[dst_key]) / max(n_old, 1)
                    new_mean = float(stats.get(src_key, 0.0))
                    dst[dst_key] = old_mean * n_old + new_mean * n_new
                for hist_key in ("top1_hist", "entropy_hist", "margin_hist"):
                    src_hist = stats.get(hist_key, [])
                    for idx, value in enumerate(src_hist):
                        dst[hist_key][idx] += int(value)
    # finalize means
    for layer_map in merged_layers.values():
        for dst in layer_map.values():
            n_total = int(dst["token_count"])
            if n_total <= 0:
                continue
            dst["sorted_prob_mean"] = (
                np.array(dst["sorted_prob_mean"]) / n_total).tolist()
            dst["sorted_prob_var"] = (
                np.array(dst["sorted_prob_m2"]) / n_total).tolist()
            dst["cum_mass_mean"] = {
                k: v / n_total for k, v in dst["cum_mass_mean"].items()
            }
            dst["entropy_mean"] /= n_total
            dst["effective_experts_mean"] = float(np.exp(dst["entropy_mean"]))
            dst["hhi_mean"] /= n_total
            dst["margin_mean"] /= n_total
            del dst["sorted_prob_m2"]
            dst["entropy_m2"] = 0.0
            dst["hhi_m2"] = 0.0
            dst["margin_m2"] = 0.0
    return {
        "schema_version": 1,
        "num_worker_files": len(paths),
        "invocation_count": total_invocations,
        **meta,
        "layers": merged_layers,
    }
