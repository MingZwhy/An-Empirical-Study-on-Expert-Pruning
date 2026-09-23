#!/usr/bin/env python3
"""Compare native-router distributions: Qwen3-30B text QA vs Qwen3-VL multimodal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CUM_RANKS = (1, 2, 4, 8, 16, 32, 64)
NATIVE_K = 8
LAYER_BANDS = {"early": (0, 15), "mid": (16, 31), "late": (32, 47)}

TEXT_TASKS = ("arc_challenge", "arc_easy", "winogrande", "openbookqa")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_best_snapshot(directory: Path) -> Path:
    best: Path | None = None
    best_inv = -1
    for path in sorted(directory.glob("router_distribution_*.json")):
        data = load_json(path)
        inv = int(data.get("invocation_count", 0))
        if inv > best_inv:
            best_inv = inv
            best = path
    if best is None:
        raise FileNotFoundError(f"No observer snapshots in {directory}")
    return best


def entropy_from_sorted(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def hhi_from_sorted(probs: np.ndarray) -> float:
    return float(np.sum(probs * probs))


def metrics_from_sorted_mean(sorted_mean: np.ndarray) -> dict[str, float]:
    cum = np.cumsum(sorted_mean)
    out: dict[str, float] = {
        "top1": float(sorted_mean[0]),
        "top2": float(sorted_mean[1]) if len(sorted_mean) > 1 else 0.0,
        "entropy": entropy_from_sorted(sorted_mean),
        "hhi": hhi_from_sorted(sorted_mean),
        "margin_top1_top2": float(sorted_mean[0] - sorted_mean[1])
        if len(sorted_mean) > 1 else float(sorted_mean[0]),
        "mass_outside_top8": float(1.0 - cum[NATIVE_K - 1]),
    }
    out["effective_experts"] = float(math.exp(out["entropy"]))
    for rank in CUM_RANKS:
        out[f"cum_mass_{rank}"] = float(cum[rank - 1])
    return out


def metrics_from_snapshot_layer(
    stats: dict[str, Any],
) -> dict[str, float]:
    sorted_mean = np.asarray(stats["sorted_prob_mean"], dtype=np.float64)
    m = metrics_from_sorted_mean(sorted_mean)
    n = max(int(stats.get("token_count", 0)), 1)
    if "sorted_prob_var" in stats:
        var = np.asarray(stats["sorted_prob_var"], dtype=np.float64)
        m["sorted_prob_std_mean_rank"] = float(np.mean(np.sqrt(np.maximum(var, 0.0))))
    for src, dst in (
        ("entropy_std", "entropy_std"),
        ("hhi_std", "hhi_std"),
        ("top1_top2_margin_std", "margin_std"),
    ):
        if src in stats:
            m[dst] = float(stats[src])
    m["token_count"] = int(stats.get("token_count", 0))
    m["_weight"] = n
    return m


def iter_layer_stage(snapshot: dict[str, Any], stage: str) -> Iterable[tuple[int, dict[str, Any]]]:
    for layer_str, stage_map in snapshot.get("layers", {}).items():
        if stage not in stage_map:
            continue
        yield int(layer_str), stage_map[stage]


def aggregate_token_weighted(items: list[dict[str, float]], key: str) -> float:
    weights = [float(x.get("_weight", x.get("token_count", 0))) for x in items]
    total = sum(weights)
    if total <= 0:
        return float("nan")
    return sum(float(x.get(key, 0.0)) * w for x, w in zip(items, weights)) / total


def aggregate_equal_dataset(dataset_values: dict[str, float]) -> float:
    vals = list(dataset_values.values())
    return float(np.mean(vals)) if vals else float("nan")


def weighted_sorted_curve(layer_stats: list[dict[str, Any]]) -> np.ndarray:
    total = sum(int(s.get("token_count", 0)) for s in layer_stats)
    if total <= 0:
        return np.zeros(128)
    acc = np.zeros(128, dtype=np.float64)
    for stats in layer_stats:
        w = int(stats.get("token_count", 0))
        acc += np.asarray(stats["sorted_prob_mean"], dtype=np.float64) * w
    return acc / total


def load_text_datasets(text_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task in TEXT_TASKS:
        obs_dir = text_root / "tasks" / task / "observer"
        snap_path = pick_best_snapshot(obs_dir)
        npz_path = snap_path.with_suffix(".npz")
        out[task] = {
            "snapshot": snap_path,
            "npz": npz_path if npz_path.is_file() else None,
            "data": load_json(snap_path),
        }
    return out


def load_vl_datasets(vl_root: Path, manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json(manifest_path)
    out: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("source_manifests", []):
        task = entry["task_spec"]["task"]
        obs = entry["observer"]
        snap_path = Path(obs["snapshot"])
        npz_path = Path(obs["npz"])
        out[task] = {
            "snapshot": snap_path,
            "npz": npz_path if npz_path.is_file() else None,
            "data": load_json(snap_path),
            "manifest": entry,
        }
    return out


def validate_inputs(
    text_root: Path,
    vl_manifest: Path,
    vl_merged: Path,
    text_datasets: dict[str, dict[str, Any]],
    vl_datasets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "text_root": str(text_root),
        "vl_manifest": str(vl_manifest),
        "vl_merged": str(vl_merged),
        "checks": [],
        "counts": {},
    }

    text_manifest_path = text_root / "MANIFEST.sha256"
    if text_manifest_path.is_file():
        lines = [ln for ln in text_manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        report["text_manifest_lines"] = len(lines)
        verified = 0
        failed = 0
        for line in text_manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split(maxsplit=1)
            path = text_root / rel
            got = sha256_file(path)
            ok = got == digest
            verified += 1
            if not ok:
                failed += 1
                report["checks"].append({"path": rel, "ok": False})
        report["text_manifest_verified"] = verified
        report["text_manifest_failed"] = failed

    vl_val = load_json(vl_manifest)
    for entry in vl_val.get("source_manifests", []):
        task = entry["task_spec"]["task"]
        obs = entry["observer"]
        snap = Path(obs["snapshot"])
        got = sha256_file(snap)
        ok = got == obs["snapshot_sha256"]
        report["checks"].append({"task": task, "snapshot_ok": ok, "path": str(snap)})
        if obs.get("npz") and Path(obs["npz"]).is_file():
            npz_ok = sha256_file(Path(obs["npz"])) == obs["npz_sha256"]
            report["checks"].append({"task": task, "npz_ok": npz_ok, "path": obs["npz"]})

    def count_snapshot(data: dict[str, Any], stage: str) -> dict[str, int]:
        routed = 0
        layer_obs = 0
        reservoir = 0
        unknown = 0
        for layer, stage_map in data.get("layers", {}).items():
            for st, stats in stage_map.items():
                if st == "unknown":
                    unknown += int(stats.get("token_count", 0))
                if st != stage:
                    continue
                routed += int(stats.get("token_count", 0))
                layer_obs += int(stats.get("token_count", 0))
        for layer, stage_map in data.get("reservoirs", {}).items():
            if stage in stage_map:
                reservoir += int(stage_map[stage].get("stored", 0))
        return {
            "layer_token_observations": layer_obs,
            "routed_tokens_proxy": routed,
            "reservoir_stored": reservoir,
            "unknown_token_count": unknown,
        }

    report["counts"]["text_prefill"] = {}
    for task, item in text_datasets.items():
        report["counts"]["text_prefill"][task] = count_snapshot(item["data"], "prefill")
        report["counts"]["text_prefill"][task]["invocation_count"] = int(
            item["data"].get("invocation_count", 0))

    for stage in ("prefill", "decode"):
        report["counts"].setdefault(f"vl_{stage}", {})
        for task, item in vl_datasets.items():
            report["counts"][f"vl_{stage}"][task] = count_snapshot(item["data"], stage)
            report["counts"][f"vl_{stage}"][task]["invocation_count"] = int(
                item["data"].get("invocation_count", 0))

    report["vl_validation"] = {
        "total_samples": vl_val.get("total_samples"),
        "task_count": vl_val.get("task_count"),
        "merged_observer_sha256": vl_val.get("merged_observer_sha256"),
    }
    return report


def dataset_layer_metrics(
    datasets: dict[str, dict[str, Any]],
    stage: str,
) -> tuple[dict[str, dict[int, dict[str, float]]], list[dict[str, Any]]]:
    per_dataset: dict[str, dict[int, dict[str, float]]] = {}
    pooled_layer_stats: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for name, item in datasets.items():
        per_dataset[name] = {}
        for layer, stats in iter_layer_stage(item["data"], stage):
            per_dataset[name][layer] = metrics_from_snapshot_layer(stats)
            pooled_layer_stats[layer].append(stats)
    return per_dataset, [
        {"layer": layer, "stats": {"sorted_prob_mean": weighted_sorted_curve(stats_list).tolist(),
                                    "token_count": sum(int(s.get("token_count", 0)) for s in stats_list)}}
        for layer, stats_list in sorted(pooled_layer_stats.items())
    ]


def global_metrics_from_layers(layer_rows: list[dict[str, Any]]) -> dict[str, float]:
    stats_list = [row["stats"] for row in layer_rows]
    curve = weighted_sorted_curve(stats_list)
    m = metrics_from_sorted_mean(curve)
    metric_items = [metrics_from_snapshot_layer(s) for s in stats_list]
    for key in ("entropy", "hhi", "effective_experts", "margin_top1_top2", "mass_outside_top8"):
        m[f"{key}_token_weighted_layers"] = aggregate_token_weighted(metric_items, key)
    for rank in CUM_RANKS:
        m[f"cum_mass_{rank}"] = float(np.sum(curve[:rank]))
    m["total_layer_token_obs"] = int(sum(int(s.get("token_count", 0)) for s in stats_list))
    return m


def layerwise_series(layer_rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    layers = sorted(row["layer"] for row in layer_rows)
    series: dict[str, list[float]] = {
        "layer": [float(x) for x in layers],
        "entropy": [],
        "effective_experts": [],
        "hhi": [],
        "cum_mass_8": [],
        "mass_outside_top8": [],
        "top1": [],
    }
    for row in layer_rows:
        m = metrics_from_snapshot_layer(row["stats"])
        series["entropy"].append(m["entropy"])
        series["effective_experts"].append(m["effective_experts"])
        series["hhi"].append(m["hhi"])
        series["cum_mass_8"].append(m["cum_mass_8"])
        series["mass_outside_top8"].append(m["mass_outside_top8"])
        series["top1"].append(m["top1"])
    return series


def band_summary(layer_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for band, (lo, hi) in LAYER_BANDS.items():
        subset = [row for row in layer_rows if lo <= row["layer"] <= hi]
        out[band] = global_metrics_from_layers(subset)
    return out


def concentration_score(metrics: dict[str, float]) -> float:
    """Higher = more concentrated on top experts (lower entropy, higher top1/cum8)."""
    return float(metrics["cum_mass_8"] - 0.01 * metrics["entropy"])


def bootstrap_diff(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    stage_left: str,
    stage_right: str,
    *,
    seed: int = 42,
    n_boot: int = 1000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    left_strata: list[tuple[str, int, np.ndarray]] = []
    right_strata: list[tuple[str, int, np.ndarray]] = []

    def collect(datasets: dict[str, dict[str, Any]], stage: str, bucket: list) -> None:
        for ds_name, item in datasets.items():
            if item.get("npz") is None:
                continue
            with np.load(item["npz"]) as z:
                for layer, _ in iter_layer_stage(item["data"], stage):
                    key = f"layer{layer}_{stage}_reservoir"
                    if key not in z.files:
                        continue
                    bucket.append((ds_name, layer, z[key]))

    collect(left, stage_left, left_strata)
    collect(right, stage_right, right_strata)
    if not left_strata or not right_strata:
        return {"available": False, "reason": "missing_npz_reservoirs"}

    def draw(strata: list[tuple[str, int, np.ndarray]]) -> dict[str, float]:
        # Resample datasets with replacement, then layers within dataset equally.
        ds_names = sorted({s[0] for s in strata})
        chosen_ds = rng.choice(ds_names, size=len(ds_names), replace=True)
        vecs: list[np.ndarray] = []
        for ds in chosen_ds:
            layers = [s for s in strata if s[0] == ds]
            if not layers:
                continue
            layer_pick = layers[rng.integers(0, len(layers))]
            arr = layer_pick[2]
            if len(arr) == 0:
                continue
            idx = rng.integers(0, len(arr), size=min(len(arr), 64))
            vecs.append(arr[idx])
        if not vecs:
            return metrics_from_sorted_mean(np.ones(128) / 128)
        pooled = np.vstack(vecs)
        mean_curve = pooled.mean(axis=0)
        return metrics_from_sorted_mean(mean_curve)

    deltas = defaultdict(list)
    signs = defaultdict(list)
    for _ in range(n_boot):
        ml = draw(left_strata)
        mr = draw(right_strata)
        for key in ("entropy", "effective_experts", "hhi", "cum_mass_8", "mass_outside_top8", "top1"):
            deltas[key].append(mr[key] - ml[key])
            signs[key].append(np.sign(mr[key] - ml[key]))

    summary: dict[str, Any] = {"available": True, "n_boot": n_boot, "seed": seed, "metrics": {}}
    for key, vals in deltas.items():
        arr = np.asarray(vals, dtype=np.float64)
        sign_arr = np.asarray(signs[key])
        summary["metrics"][key] = {
            "delta_mean": float(arr.mean()),
            "delta_median": float(np.median(arr)),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "p_positive": float((sign_arr > 0).mean()),
            "robust_sign_consistency": float(max((sign_arr > 0).mean(), (sign_arr < 0).mean())),
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_sorted_curves(curves: dict[str, np.ndarray], out_base: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ranks = np.arange(1, 129)
    for label, curve in curves.items():
        ax.plot(ranks, curve, label=label, linewidth=1.6)
    ax.set_xlabel("Sorted expert rank")
    ax.set_ylabel("Mean softmax probability")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=160)
    plt.close(fig)


def plot_cum_mass(curves: dict[str, np.ndarray], out_base: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ranks = np.arange(1, 129)
    for label, curve in curves.items():
        ax.plot(ranks, np.cumsum(curve), label=label, linewidth=1.6)
    for k in CUM_RANKS:
        ax.axvline(k, color="gray", linestyle=":", alpha=0.35)
    ax.set_xlabel("Top-k sorted experts")
    ax.set_ylabel("Cumulative softmax mass")
    ax.set_title(title)
    ax.set_xlim(1, 128)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=160)
    plt.close(fig)


def plot_layer_series(series_map: dict[str, dict[str, list[float]]], metric: str,
                      out_base: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, series in series_map.items():
        ax.plot(series["layer"], series[metric], label=label, linewidth=1.4)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=160)
    plt.close(fig)


def plot_dataset_concentration(rows: list[dict[str, Any]], out_base: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [r["dataset"] for r in rows]
    vals = [r["concentration_score"] for r in rows]
    colors = ["#1f77b4" if r["model_group"] == "text_qa" else "#ff7f0e" for r in rows]
    ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Concentration score (cum@8 - 0.01*entropy)")
    ax.set_title(title)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=160)
    plt.close(fig)


def largest_diff_layers(text_rows: list[dict[str, Any]], vl_rows: list[dict[str, Any]],
                        metric: str = "entropy") -> list[dict[str, Any]]:
    vl_by_layer = {row["layer"]: metrics_from_snapshot_layer(row["stats"]) for row in vl_rows}
    diffs = []
    for row in text_rows:
        layer = row["layer"]
        tm = metrics_from_snapshot_layer(row["stats"])
        vm = vl_by_layer.get(layer)
        if vm is None:
            continue
        diffs.append({
            "layer": layer,
            "metric": metric,
            "text": tm[metric],
            "vl_prefill": vm[metric],
            "delta_vl_minus_text": vm[metric] - tm[metric],
        })
    diffs.sort(key=lambda x: abs(x["delta_vl_minus_text"]), reverse=True)
    return diffs[:10]


def build_markdown(result: dict[str, Any]) -> str:
    p = result["primary_prefill_comparison"]
    lines = [
        "# Qwen3 text QA vs Qwen3-VL multimodal router-distribution comparison",
        "",
        f"- Generated: `{result['generated_at']}`",
        f"- Analysis script: `{result['analysis_script']}`",
        "",
        "## Validation",
        "",
        f"- Text manifest failures: {result['validation'].get('text_manifest_failed', 'n/a')}",
        f"- VL tasks validated: {result['validation']['vl_validation']['task_count']}",
        "",
        "## Primary fair comparison (prefill only)",
        "",
        "**Label:** Text lm_eval loglikelihood prefill vs VL multimodal prefill (image+text tokens untagged).",
        "This is **not** architecture-only causation; confounded by model, domain, and modality.",
        "",
        "| Metric | Text QA prefill | VL prefill | Δ (VL−Text) |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("top1", "Top-1 sorted prob"),
        ("cum_mass_8", "Cumulative mass @8"),
        ("mass_outside_top8", "Mass outside native top-8"),
        ("entropy", "Entropy (nats)"),
        ("effective_experts", "Effective experts exp(H)"),
        ("hhi", "HHI"),
        ("margin_top1_top2", "Top1−Top2 margin"),
    ):
        t = p["text_global"][key]
        v = p["vl_global"][key]
        lines.append(f"| {label} | {t:.4f} | {v:.4f} | {v - t:+.4f} |")

    boot = p.get("bootstrap", {})
    if boot.get("available"):
        lines.extend(["", "### Bootstrap (stratified by dataset×layer reservoir)", ""])
        for key in ("entropy", "cum_mass_8", "effective_experts"):
            b = boot["metrics"][key]
            lines.append(
                f"- **{key}** Δ mean={b['delta_mean']:+.4f}, 95% CI [{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}], "
                f"robust sign consistency={b['robust_sign_consistency']:.3f}")

    lines.extend([
        "",
        "## Core question",
        "",
        result["conclusion"]["headline"],
        "",
        result["conclusion"]["detail"],
        "",
        "## VL decode vs VL prefill (secondary)",
        "",
    ])
    sec = result["vl_stage_comparison"]
    for key in ("entropy", "cum_mass_8", "effective_experts"):
        lines.append(
            f"- {key}: prefill={sec['prefill_global'][key]:.4f}, decode={sec['decode_global'][key]:.4f}, "
            f"Δ decode−prefill={sec['decode_global'][key] - sec['prefill_global'][key]:+.4f}")

    lines.extend(["", "## Dataset concentration extremes", ""])
    for side in ("most_concentrated", "least_concentrated"):
        row = result["dataset_extremes"][side]
        lines.append(f"- **{side.replace('_', ' ').title()}:** {row['model_group']} / {row['dataset']} "
                     f"(score={row['concentration_score']:.4f}, entropy={row['entropy']:.3f}, cum@8={row['cum_mass_8']:.3f})")

    lines.extend(["", "## Caveats", ""])
    for c in result["conclusion"]["caveats"]:
        lines.append(f"- {c}")
    return "\n".join(lines) + "\n"


def run_analysis(repo_root: Path, output_dir: Path, seed: int = 42) -> dict[str, Any]:
    text_root = repo_root / "results/router_distribution/Qwen3-30B-A3B-Instruct-2507/qa_20260817_151339"
    vl_root = repo_root / "results/router_distribution/Qwen3-VL-30B-A3B-Instruct"
    vl_manifest = vl_root / "multimodal_merged/validation_manifest.json"
    vl_merged = vl_root / "multimodal_merged/qwen3vl_mm_summary.json"

    text_datasets = load_text_datasets(text_root)
    vl_datasets = load_vl_datasets(vl_root, vl_manifest)
    validation = validate_inputs(text_root, vl_manifest, vl_merged, text_datasets, vl_datasets)

    text_per_ds, text_layer_rows = dataset_layer_metrics(text_datasets, "prefill")
    vl_prefill_per_ds, vl_prefill_rows = dataset_layer_metrics(vl_datasets, "prefill")
    vl_decode_per_ds, vl_decode_rows = dataset_layer_metrics(vl_datasets, "decode")

    text_global = global_metrics_from_layers(text_layer_rows)
    vl_prefill_global = global_metrics_from_layers(vl_prefill_rows)
    vl_decode_global = global_metrics_from_layers(vl_decode_rows)

    text_series = layerwise_series(text_layer_rows)
    vl_prefill_series = layerwise_series(vl_prefill_rows)
    vl_decode_series = layerwise_series(vl_decode_rows)

    dataset_rows: list[dict[str, Any]] = []
    for task in TEXT_TASKS:
        layer_rows = [{"layer": layer, "stats": stats}
                      for layer, stats in iter_layer_stage(text_datasets[task]["data"], "prefill")]
        m = global_metrics_from_layers(layer_rows)
        m.update({"model_group": "text_qa", "dataset": task, "stage": "prefill",
                  "concentration_score": concentration_score(m)})
        dataset_rows.append(m)
    for task in vl_datasets:
        layer_rows = [{"layer": layer, "stats": stats}
                      for layer, stats in iter_layer_stage(vl_datasets[task]["data"], "prefill")]
        m = global_metrics_from_layers(layer_rows)
        m.update({"model_group": "vl_multimodal", "dataset": task, "stage": "prefill",
                  "concentration_score": concentration_score(m)})
        dataset_rows.append(m)

    dataset_rows_sorted = sorted(dataset_rows, key=lambda r: r["concentration_score"], reverse=True)

    bootstrap = bootstrap_diff(
        text_datasets, vl_datasets, "prefill", "prefill", seed=seed, n_boot=1000)

    diff_layers = largest_diff_layers(text_layer_rows, vl_prefill_rows, "entropy")

    # Conclusion logic
    delta_entropy = vl_prefill_global["entropy"] - text_global["entropy"]
    delta_cum8 = vl_prefill_global["cum_mass_8"] - text_global["cum_mass_8"]
    if delta_entropy > 0.05 and delta_cum8 < -0.02:
        headline = (
            "VL multimodal **prefill routing is more uniform** (higher entropy, lower top-8 mass) "
            "than text QA prefill under native top-8 routing."
        )
    elif delta_entropy < -0.05 and delta_cum8 > 0.02:
        headline = (
            "VL multimodal **prefill routing is more concentrated** on high-weight experts than text QA prefill."
        )
    else:
        headline = (
            "Prefill routing differences are present but modest relative to layer/dataset heterogeneity; "
            "do not over-interpret as pure architecture effects."
        )

    detail = (
        f"Token-weighted global prefill: text entropy={text_global['entropy']:.3f}, "
        f"VL entropy={vl_prefill_global['entropy']:.3f} (Δ={delta_entropy:+.3f}); "
        f"text cum@8={text_global['cum_mass_8']:.3f}, VL cum@8={vl_prefill_global['cum_mass_8']:.3f} "
        f"(Δ={delta_cum8:+.3f}); text effective experts={text_global['effective_experts']:.1f}, "
        f"VL={vl_prefill_global['effective_experts']:.1f}. "
        f"VL decode is much more concentrated than VL prefill (decode cum@8={vl_decode_global['cum_mass_8']:.3f})."
    )

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_script": str(Path(__file__).resolve()),
        "seed": seed,
        "validation": validation,
        "primary_prefill_comparison": {
            "label": "text_qa_prefill_vs_vl_multimodal_prefill_confounded",
            "text_global": text_global,
            "vl_global": vl_prefill_global,
            "delta_vl_minus_text": {k: vl_prefill_global[k] - text_global[k]
                                    for k in text_global if isinstance(text_global[k], (int, float))},
            "bootstrap": bootstrap,
            "text_band_summary": band_summary(text_layer_rows),
            "vl_band_summary": band_summary(vl_prefill_rows),
            "largest_entropy_diff_layers": diff_layers,
            "layer_sign_consistency_entropy": float(np.mean(
                np.sign(np.array(vl_prefill_series["entropy"]) - np.array(text_series["entropy"])) != 0)),
        },
        "vl_stage_comparison": {
            "prefill_global": vl_prefill_global,
            "decode_global": vl_decode_global,
            "delta_decode_minus_prefill": {
                k: vl_decode_global[k] - vl_prefill_global[k]
                for k in vl_prefill_global if isinstance(vl_prefill_global[k], (int, float))
            },
        },
        "dataset_extremes": {
            "most_concentrated": dataset_rows_sorted[0],
            "least_concentrated": dataset_rows_sorted[-1],
        },
        "dataset_rows": dataset_rows_sorted,
        "conclusion": {
            "headline": headline,
            "detail": detail,
            "caveats": [
                "Text QA uses lm_eval loglikelihood (almost all prefill); do not compare to VL decode directly.",
                "VL prefill commingles image and text tokens; modality not tagged in observer.",
                "Models differ (Qwen3-30B text vs Qwen3-VL-30B); domain/task sets differ (4 text vs 9 VL).",
                "Bootstrap uses reservoir subsamples (~0.5%); tokens are not independent.",
                "Equal-dataset and token-weighted aggregates can disagree when token counts are skewed.",
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)

    text_curve = weighted_sorted_curve([r["stats"] for r in text_layer_rows])
    vl_pref_curve = weighted_sorted_curve([r["stats"] for r in vl_prefill_rows])
    vl_dec_curve = weighted_sorted_curve([r["stats"] for r in vl_decode_rows])

    plot_sorted_curves(
        {
            "Text QA prefill": text_curve,
            "VL prefill": vl_pref_curve,
            "VL decode (secondary)": vl_dec_curve,
        },
        plots / "mean_sorted_curve",
        "Mean sorted 128-expert softmax curve (token-weighted over layers)",
    )
    plot_cum_mass(
        {
            "Text QA prefill": text_curve,
            "VL prefill": vl_pref_curve,
            "VL decode (secondary)": vl_dec_curve,
        },
        plots / "cumulative_topk_mass",
        "Cumulative mass over sorted experts",
    )
    plot_layer_series(
        {
            "Text QA prefill": text_series,
            "VL prefill": vl_prefill_series,
            "VL decode": vl_decode_series,
        },
        "entropy",
        plots / "layerwise_entropy",
        "Layerwise mean entropy",
    )
    plot_layer_series(
        {
            "Text QA prefill": text_series,
            "VL prefill": vl_prefill_series,
            "VL decode": vl_decode_series,
        },
        "effective_experts",
        plots / "layerwise_effective_experts",
        "Layerwise effective experts exp(H)",
    )
    plot_dataset_concentration(
        dataset_rows_sorted,
        plots / "dataset_concentration",
        "Dataset-level concentration (prefill)",
    )

    write_csv(output_dir / "aggregates_by_dataset.csv", dataset_rows_sorted, [
        "model_group", "dataset", "stage", "concentration_score", "entropy", "effective_experts",
        "hhi", "top1", "cum_mass_8", "mass_outside_top8", "total_layer_token_obs",
    ])
    layer_csv_rows = []
    for label, rows in (
        ("text_qa_prefill", text_layer_rows),
        ("vl_prefill", vl_prefill_rows),
        ("vl_decode", vl_decode_rows),
    ):
        for row in rows:
            m = metrics_from_snapshot_layer(row["stats"])
            layer_csv_rows.append({
                "group": label,
                "layer": row["layer"],
                "entropy": m["entropy"],
                "effective_experts": m["effective_experts"],
                "hhi": m["hhi"],
                "top1": m["top1"],
                "cum_mass_8": m["cum_mass_8"],
                "mass_outside_top8": m["mass_outside_top8"],
                "token_count": m["token_count"],
            })
    write_csv(output_dir / "aggregates_by_layer.csv", layer_csv_rows,
              ["group", "layer", "entropy", "effective_experts", "hhi", "top1",
               "cum_mass_8", "mass_outside_top8", "token_count"])

    global_rows = [
        {"comparison": "text_qa_prefill", **{k: v for k, v in text_global.items() if isinstance(v, (int, float))}},
        {"comparison": "vl_prefill", **{k: v for k, v in vl_prefill_global.items() if isinstance(v, (int, float))}},
        {"comparison": "vl_decode", **{k: v for k, v in vl_decode_global.items() if isinstance(v, (int, float))}},
    ]
    write_csv(output_dir / "aggregates_global.csv", global_rows, [
        "comparison", "entropy", "effective_experts", "hhi", "top1", "cum_mass_8",
        "mass_outside_top8", "total_layer_token_obs",
    ])

    if bootstrap.get("available"):
        boot_rows = []
        for metric, vals in bootstrap["metrics"].items():
            boot_rows.append({"metric": metric, **vals})
        write_csv(output_dir / "bootstrap_prefill_text_vs_vl.csv", boot_rows,
                  ["metric", "delta_mean", "delta_median", "ci95_low", "ci95_high",
                   "p_positive", "robust_sign_consistency"])

    (output_dir / "comparison_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "analysis_report.md").write_text(build_markdown(result), encoding="utf-8")

    script_copy = output_dir / "run_analysis.py"
    shutil.copy2(Path(__file__), script_copy)

    file_hashes = {
        str(p.relative_to(output_dir)): sha256_file(p)
        for p in sorted(output_dir.rglob("*"))
        if p.is_file() and p.name != "reproducibility_check.json"
    }
    (output_dir / "reproducibility_check.json").write_text(
        json.dumps({"file_sha256": file_hashes}, indent=2), encoding="utf-8")
    result["output_hashes"] = file_hashes
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path,
                        default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-twice", action="store_true",
                        help="Run twice and compare output hashes")
    args = parser.parse_args()
    out = args.output_dir or (
        args.repo_root / "results/router_distribution/comparison_qwen3_text_vs_vl")
    first = run_analysis(args.repo_root, out, seed=args.seed)
    if args.verify_twice:
        tmp = out.with_name(out.name + "_verify_tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        second = run_analysis(args.repo_root, tmp, seed=args.seed)
        keys = sorted(set(first["output_hashes"]) | set(second["output_hashes"]))
        volatile = {
            "comparison_summary.json", "analysis_report.md", "reproducibility_check.json",
        }
        mism = [k for k in keys
                if k not in volatile
                and not k.endswith(".pdf")
                and first["output_hashes"].get(k) != second["output_hashes"].get(k)]
        check = {
            "match": len(mism) == 0,
            "mismatched_files": mism,
            "first": {k: first["output_hashes"].get(k) for k in keys},
            "second": {k: second["output_hashes"].get(k) for k in keys},
        }
        (out / "reproducibility_check.json").write_text(
            json.dumps(check, indent=2), encoding="utf-8")
        shutil.rmtree(tmp)
        if mism:
            raise SystemExit(f"Reproducibility check failed: {mism}")
    print(f"Wrote analysis to {out}")


if __name__ == "__main__":
    main()
