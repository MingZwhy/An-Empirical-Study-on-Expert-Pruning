"""Router hyper-parameter sweeps on calibration data.

This module estimates the average number of selected experts for pruning
methods without changing model outputs.  It attaches forward hooks to
Transformers Qwen3-MoE blocks, reads ``router_logits``, and evaluates many
NAEE/Dynamic_Routing settings in one calibration pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from calib_utils.data_utils import get_loaders
from calib_utils.transformers_collector import _extract_input_ids


def default_naee_betas() -> list[float]:
    return [round(i * 0.05, 2) for i in range(21)]


def default_dynamic_thresholds() -> list[float]:
    return [round(i * 0.05, 2) for i in range(1, 21)]


def parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if value is None or value.strip() == "":
        return default
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(float(item))
    if not result:
        raise ValueError("Float list must contain at least one value.")
    return result


def _default_model_name(model_path: str) -> str:
    return Path(model_path.rstrip("/")).name


def _resolve_output_prefix(args) -> Path:
    if args.output_prefix:
        return Path(args.output_prefix)
    model_name = args.model_name or _default_model_name(args.model_path)
    return Path(args.output_dir) / model_name / f"{args.dataset}_router_sweep"


def _infer_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _safe_renormalize(weights: torch.Tensor) -> torch.Tensor:
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)


def _naee_keep_counts(
    topk_weights: torch.Tensor,
    betas: torch.Tensor,
    k_min: int = 1,
) -> torch.Tensor:
    """Return keep-count sums for every beta.

    Shape:
        topk_weights: [tokens, topk]
        betas: [num_betas]
        return: [num_betas], sum of selected experts across tokens
    """
    topk = topk_weights.size(-1)
    ranks = torch.arange(topk, device=topk_weights.device).view(1, 1, topk)
    k_min = min(k_min, topk)
    keep_by_min = ranks < k_min
    threshold = topk_weights[None, :, :1] * betas.view(-1, 1, 1)
    prune_candidate = (ranks >= k_min) & (topk_weights[None, :, :] <
                                          threshold)
    has_previous_prune = prune_candidate.int().cumsum(dim=-1) > 0
    keep_mask = keep_by_min | ~has_previous_prune
    return keep_mask.sum(dim=-1).sum(dim=-1).to(dtype=torch.float64)


def _dynamic_keep_counts(
    topk_weights: torch.Tensor,
    thresholds: torch.Tensor,
    score_source: str,
) -> torch.Tensor:
    """Return keep-count sums for every Dynamic_Routing threshold."""
    topk = topk_weights.size(-1)
    threshold_weights = topk_weights
    if score_source == "renormalized":
        threshold_weights = _safe_renormalize(topk_weights)
    elif score_source != "raw":
        raise ValueError("score_source must be raw or renormalized")

    cumulative = threshold_weights.cumsum(dim=-1)
    reaches = cumulative[None, :, :] >= thresholds.view(-1, 1, 1)
    first_reach = reaches.to(dtype=torch.int64).argmax(dim=-1) + 1
    keep_counts = torch.where(
        reaches.any(dim=-1),
        first_reach,
        torch.full_like(first_reach, topk),
    )
    return keep_counts.sum(dim=-1).to(dtype=torch.float64)


class RouterSweepCollector(AbstractContextManager):
    """Collect average selected experts for NAEE and Dynamic_Routing grids."""

    def __init__(
        self,
        model: torch.nn.Module,
        naee_betas: list[float] | None = None,
        dynamic_thresholds: list[float] | None = None,
        naee_k_min: int = 1,
    ) -> None:
        self.model = model
        self.naee_betas = default_naee_betas(
        ) if naee_betas is None else naee_betas
        self.dynamic_thresholds = default_dynamic_thresholds(
        ) if dynamic_thresholds is None else dynamic_thresholds
        self.naee_k_min = naee_k_min
        self._hooks: list[Any] = []
        self._module_names = {module: name for name, module in model.named_modules()}
        self.total_token_routes = 0
        self.total_forwards = 0
        self.global_sums: dict[str, list[float]] = {
            "naee": [0.0 for _ in self.naee_betas],
            "dynamic_raw": [0.0 for _ in self.dynamic_thresholds],
            "dynamic_renormalized": [
                0.0 for _ in self.dynamic_thresholds
            ],
        }
        self.layer_sums: dict[str, dict[str, Any]] = defaultdict(dict)

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

        if self._hooks:
            return
        for module in self.model.modules():
            if isinstance(module, Qwen3MoeSparseMoeBlock):
                self._hooks.append(module.register_forward_hook(self._hook))

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _init_layer(self, layer_name: str, topk: int) -> dict[str, Any]:
        if "total_token_routes" not in self.layer_sums[layer_name]:
            self.layer_sums[layer_name] = {
                "topk": int(topk),
                "total_token_routes": 0,
                "num_forwards": 0,
                "naee": [0.0 for _ in self.naee_betas],
                "dynamic_raw": [0.0 for _ in self.dynamic_thresholds],
                "dynamic_renormalized": [
                    0.0 for _ in self.dynamic_thresholds
                ],
            }
        return self.layer_sums[layer_name]

    def _hook(self, module: torch.nn.Module, inputs: Any, output: Any) -> None:
        if not isinstance(output, tuple) or len(output) < 2:
            return
        router_logits = output[1]
        if not isinstance(router_logits, torch.Tensor):
            return

        with torch.no_grad():
            topk = int(module.top_k)
            scores = F.softmax(router_logits.float(), dim=-1)
            topk_weights = torch.topk(scores, k=topk, dim=-1,
                                      sorted=True).values

            token_routes = int(topk_weights.size(0))
            self.total_token_routes += token_routes
            self.total_forwards += 1
            layer_name = self._module_names.get(module, f"moe_{id(module)}")
            layer_stats = self._init_layer(layer_name, topk)
            layer_stats["total_token_routes"] += token_routes
            layer_stats["num_forwards"] += 1

            beta_tensor = torch.tensor(
                self.naee_betas,
                dtype=torch.float32,
                device=topk_weights.device,
            )
            threshold_tensor = torch.tensor(
                self.dynamic_thresholds,
                dtype=torch.float32,
                device=topk_weights.device,
            )
            naee_counts = _naee_keep_counts(
                topk_weights,
                beta_tensor,
                k_min=self.naee_k_min,
            )
            dynamic_raw_counts = _dynamic_keep_counts(
                topk_weights,
                threshold_tensor,
                score_source="raw",
            )
            dynamic_renorm_counts = _dynamic_keep_counts(
                topk_weights,
                threshold_tensor,
                score_source="renormalized",
            )
            self._add_counts("naee", naee_counts, layer_stats)
            self._add_counts("dynamic_raw", dynamic_raw_counts, layer_stats)
            self._add_counts("dynamic_renormalized", dynamic_renorm_counts,
                             layer_stats)

    def _add_counts(self, key: str, counts: torch.Tensor,
                    layer_stats: dict[str, Any]) -> None:
        values = counts.detach().cpu().tolist()
        for idx, value in enumerate(values):
            self.global_sums[key][idx] += float(value)
            layer_stats[key][idx] += float(value)

    def rows(self) -> list[dict[str, Any]]:
        denom = max(1, self.total_token_routes)
        rows: list[dict[str, Any]] = []
        for idx, beta in enumerate(self.naee_betas):
            rows.append({
                "method": "NAEE",
                "parameter": beta,
                "score_source": "",
                "average_selected_experts":
                self.global_sums["naee"][idx] / denom,
                "total_selected_experts": self.global_sums["naee"][idx],
                "total_token_routes": self.total_token_routes,
            })
        for key, score_source in (("dynamic_raw", "raw"),
                                  ("dynamic_renormalized",
                                   "renormalized")):
            for idx, threshold in enumerate(self.dynamic_thresholds):
                rows.append({
                    "method": "Dynamic_Routing",
                    "parameter": threshold,
                    "score_source": score_source,
                    "average_selected_experts":
                    self.global_sums[key][idx] / denom,
                    "total_selected_experts": self.global_sums[key][idx],
                    "total_token_routes": self.total_token_routes,
                })
        return rows

    def summary(self) -> dict[str, Any]:
        layer_summary: dict[str, Any] = {}
        for layer_name, stats in self.layer_sums.items():
            denom = max(1, int(stats["total_token_routes"]))
            layer_summary[layer_name] = {
                "topk": stats["topk"],
                "total_token_routes": stats["total_token_routes"],
                "num_forwards": stats["num_forwards"],
                "naee": {
                    str(beta): stats["naee"][idx] / denom
                    for idx, beta in enumerate(self.naee_betas)
                },
                "dynamic_raw": {
                    str(threshold): stats["dynamic_raw"][idx] / denom
                    for idx, threshold in enumerate(self.dynamic_thresholds)
                },
                "dynamic_renormalized": {
                    str(threshold): stats["dynamic_renormalized"][idx] / denom
                    for idx, threshold in enumerate(self.dynamic_thresholds)
                },
            }
        return {
            "naee_betas": self.naee_betas,
            "dynamic_thresholds": self.dynamic_thresholds,
            "naee_k_min": self.naee_k_min,
            "total_token_routes": self.total_token_routes,
            "total_forwards": self.total_forwards,
            "rows": self.rows(),
            "layers": layer_summary,
        }


def run_router_sweep(
    model: torch.nn.Module,
    trainloader: Any,
    naee_betas: list[float] | None = None,
    dynamic_thresholds: list[float] | None = None,
    naee_k_min: int = 1,
    max_samples: int | None = None,
    device: torch.device | str | None = None,
    use_cache: bool = False,
    show_progress: bool = True,
    progress_desc: str = "router sweep",
) -> RouterSweepCollector:
    model.eval()
    input_device = torch.device(device) if device is not None else \
        _infer_input_device(model)
    collector = RouterSweepCollector(
        model,
        naee_betas=naee_betas,
        dynamic_thresholds=dynamic_thresholds,
        naee_k_min=naee_k_min,
    )
    total = len(trainloader) if hasattr(trainloader, "__len__") else None
    if max_samples is not None:
        total = min(total, max_samples) if total is not None else max_samples
    iterator = trainloader
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(trainloader,
                            total=total,
                            desc=progress_desc,
                            dynamic_ncols=True)
        except ImportError:
            print(f"{progress_desc}: tqdm is not installed; "
                  "printing every 10 samples.",
                  flush=True)

    with collector, torch.no_grad():
        for sample_idx, batch in enumerate(iterator):
            if max_samples is not None and sample_idx >= max_samples:
                break
            if show_progress and "tqdm" not in type(iterator).__module__:
                if sample_idx == 0 or (sample_idx + 1) % 10 == 0:
                    suffix = f"/{total}" if total is not None else ""
                    print(f"{progress_desc}: {sample_idx + 1}{suffix}",
                          flush=True)
            input_ids = _extract_input_ids(batch).to(input_device)
            attention_mask = torch.ones_like(input_ids, device=input_device)
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
    return collector


def write_router_sweep_outputs(
    collector: RouterSweepCollector,
    output_prefix: str | Path,
) -> tuple[Path, Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")

    summary = collector.summary()
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    rows = collector.rows()
    fieldnames = [
        "method",
        "parameter",
        "score_source",
        "average_selected_experts",
        "total_selected_experts",
        "total_token_routes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| method | parameter | score_source | avg_selected_experts |\n")
        f.write("|---|---:|---|---:|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['parameter']} | "
                f"{row['score_source']} | "
                f"{row['average_selected_experts']:.6f} |\n")
    return json_path, csv_path, md_path


def _load_transformers_model(args):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.torch_dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)
    else:
        kwargs["torch_dtype"] = "auto"
    if args.device_map:
        kwargs["device_map"] = args.device_map
    return AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep router pruning hyper-parameters on calibration data."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset",
                        default="wikitext2",
                        choices=["wikitext2", "c4", "ptb"])
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_data_path", default=None)
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Full output prefix. If omitted, uses "
        "<output_dir>/<model_name>/<dataset>_router_sweep.",
    )
    parser.add_argument("--output_dir", default="calib_utils/results")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--naee_betas", default=None,
                        help="Comma-separated beta list. Default: 0..1 step 0.05")
    parser.add_argument(
        "--dynamic_thresholds",
        default=None,
        help="Comma-separated threshold list. Default: 0.05..1 step 0.05",
    )
    parser.add_argument("--naee_k_min", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--torch_dtype",
                        default="bfloat16",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--local_files_only",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--progress",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[router_sweep] dataset={args.dataset} nsamples={args.nsamples} "
          f"seqlen={args.seqlen}",
          flush=True)
    print("[router_sweep] preparing calibration data...", flush=True)
    naee_betas = parse_float_list(args.naee_betas, default_naee_betas())
    dynamic_thresholds = parse_float_list(args.dynamic_thresholds,
                                          default_dynamic_thresholds())
    trainloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        model=args.model_path,
        local_data_path=args.local_data_path,
    )
    print(f"[router_sweep] prepared {len(trainloader)} calibration samples",
          flush=True)
    print("[router_sweep] loading Transformers model...", flush=True)
    model = _load_transformers_model(args)
    print("[router_sweep] model loaded; starting forward passes...",
          flush=True)
    collector = run_router_sweep(
        model,
        trainloader,
        naee_betas=naee_betas,
        dynamic_thresholds=dynamic_thresholds,
        naee_k_min=args.naee_k_min,
        max_samples=args.max_samples,
        device=args.device,
        show_progress=args.progress,
        progress_desc=f"{args.dataset} router sweep",
    )
    print("[router_sweep] writing outputs...", flush=True)
    output_prefix = _resolve_output_prefix(args)
    json_path, csv_path, md_path = write_router_sweep_outputs(
        collector,
        output_prefix,
    )
    print(f"Router sweep saved: {json_path}")
    print(f"CSV table saved: {csv_path}")
    print(f"Markdown table saved: {md_path}")


if __name__ == "__main__":
    main()
