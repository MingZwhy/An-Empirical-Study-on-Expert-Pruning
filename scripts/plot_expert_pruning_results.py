#!/usr/bin/env python3
"""Plot per-model expert pruning curves (3x3, excluding MuSR).

This reads the raw per-run artifacts a sweep writes, `results/<model>/<k>experts/`, and
those are gigabytes, so they are not in the repository: `results/<model>/` carries only
the summary table. Run a sweep first (`scripts/run.sh <model> fixedk`) and this plots
whatever it finds, skipping any model with no artifacts yet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

# (display_name, results_rel_path, default num_experts_per_tok)
MODEL_DIRS = [
    ("DeepSeek-V2-Lite-Chat", "results/DeepSeek-V2-Lite-Chat", 6),
    ("gpt-oss-20b", "results/gpt-oss-20b", 4),
    ("Ling-lite-1.5-2507", "results/Ling-lite-1.5-2507", 6),
    ("Qwen3-30B-A3B-Instruct-2507", "results/Qwen3-30B-A3B-Instruct-2507", 8),
    ("Qwen3-Next-80B-A3B-Instruct", "results/Qwen3-Next-80B-A3B-Instruct", 10),
    ("DeepSeek-V4-Flash-0731", "results/DeepSeek-V4-Flash-0731", 8),
    ("Hy3", "results/Hy3", 8),
    ("MiniMax-M2.7", "results/MiniMax-M2.7", 8),
]

BENCHMARKS = [
    ("MMLU-Pro", [("mmlu_pro|0", "extractive_match")]),
    ("GPQA", [("gpqa:diamond|0", "gpqa_pass@k:k=1")]),
    ("MATH-500", [("math_500|0", "pass@k:k=1&n=1")]),
    # The sample count is part of the metric key, and it changed: runs from 2026-08-12 sample each AIME
    # question 16 times instead of 64, so both keys have to be looked for or the newer runs read as
    # missing data. They estimate the same quantity and are plotted as one series.
    ("AIME24", [
        ("aime24_avg|0", "avg@n:n=64"),
        ("aime24_avg|0", "avg@n:n=16"),
        ("aime24|0", "avg@n:n=64"),
        ("aime24|0", "avg@n:n=16"),
        ("aime24|0", "pass@k:k=1"),
        ("aime24|0", "avg@n:n=1"),
    ]),
    ("AIME25", [
        ("aime25_avg|0", "avg@n:n=64"),
        ("aime25_avg|0", "avg@n:n=16"),
        ("aime25|0", "avg@n:n=64"),
        ("aime25|0", "avg@n:n=16"),
        ("aime25|0", "pass@k:k=1&n=1"),
        ("aime25|0", "pass@k:k=1"),
        ("aime25|0", "avg@n:n=1"),
    ]),
    ("LCB", [("lcb:codegeneration_v6|0", "codegen_pass@1:16")]),
    ("IFEval", [("ifeval|0", "prompt_level_strict_acc")]),
    ("HellaSwag", [("hellaswag_fixed|0", "acc")]),
    ("WinoGrande", [("winogrande|0", "acc")]),
]


def find_k_dirs(base: Path) -> list[int]:
    ks: list[int] = []
    if not base.is_dir():
        return ks
    for name in base.iterdir():
        m = re.match(r"^(\d+)experts$", name.name)
        if m and int(m.group(1)) >= 2:
            ks.append(int(m.group(1)))
    return sorted(ks)


def find_results_files(k_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, _, fnames in os.walk(k_dir):
        if Path(root).name == "results":
            for fname in fnames:
                if fname.startswith("results_") and fname.endswith(".json"):
                    files.append(Path(root) / fname)
    return sorted(files)


def merge_results(files: list[Path]) -> dict:
    merged: dict = {}
    for fp in files:
        with fp.open() as f:
            data = json.load(f)
        for task, metrics in data.get("results", {}).items():
            if task != "all":
                merged[task] = metrics
    return merged


def get_val(merged: dict, candidates: list[tuple[str, str]]) -> float | None:
    for task, metric in candidates:
        if task in merged and metric in merged[task]:
            return merged[task][metric]
    return None


def load_model_data(base: Path) -> tuple[list[int], dict[str, dict[int, float | None]]]:
    ks = find_k_dirs(base)
    model_data: dict[str, dict[int, float | None]] = {name: {} for name, _ in BENCHMARKS}
    for k in ks:
        merged = merge_results(find_results_files(base / f"{k}experts"))
        for bench_name, candidates in BENCHMARKS:
            model_data[bench_name][k] = get_val(merged, candidates)
    return ks, model_data


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def plot_model(
    model_name: str,
    ks: list[int],
    model_data: dict[str, dict[int, float | None]],
    default_k: int,
    out_dir: Path,
) -> Path:
    if not ks:
        raise ValueError(f"No k>=2 results for {model_name}")

    half_k = default_k // 2 + 1
    x_desc = list(reversed(ks))
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(model_name, fontsize=16, fontweight="bold")

    for ax, (bench_name, _) in zip(axes.flat, BENCHMARKS):
        ys = [model_data[bench_name].get(k) for k in x_desc]
        valid_x = [k for k, y in zip(x_desc, ys) if y is not None]
        valid_y = [y * 100 for y in ys if y is not None]

        ax.plot(valid_x, valid_y, marker="o", linewidth=2, markersize=5, color="#2563eb")
        if default_k in model_data[bench_name] and model_data[bench_name][default_k] is not None:
            ax.plot(
                default_k,
                model_data[bench_name][default_k] * 100,
                marker="*",
                markersize=12,
                color="#dc2626",
                zorder=5,
            )
        if min(ks) <= half_k <= max(ks):
            ax.axvline(half_k, color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)
        ax.set_title(bench_name, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Activated Experts (k)", fontsize=9)
        ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.set_xticks(valid_x)
        ax.set_ylim(0, 100)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=8)

    out_path = out_dir / f"{slugify(model_name)}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the repository root",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for figures",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = (args.out_dir or repo_root / "docs" / "figures" / "expert_pruning").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for model_name, rel_path, default_k in MODEL_DIRS:
        base = (repo_root / rel_path).resolve()
        ks, model_data = load_model_data(base)
        if not ks:
            print(f"Skipped: {model_name} (no <k>experts directories under {rel_path})")
            continue
        out_path = plot_model(model_name, ks, model_data, default_k, out_dir)
        generated.append(out_path)
        print(f"Saved: {out_path}")

    print(f"\nGenerated {len(generated)} figures in {out_dir}")


if __name__ == "__main__":
    main()
