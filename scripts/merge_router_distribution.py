#!/usr/bin/env python3
"""Merge router-distribution worker snapshots into dataset/model summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from expert_pruning.router_distribution import CUM_RANKS, merge_worker_snapshots


def _collect_snapshot_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in inputs:
        if root.is_file() and root.name.startswith("router_distribution_"):
            paths.append(root)
            continue
        if not root.is_dir():
            continue
        paths.extend(sorted(root.glob("router_distribution_*.json")))
        for child in sorted(root.iterdir()):
            if child.is_dir():
                paths.extend(sorted(child.glob("router_distribution_*.json")))
    deduped = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _load_reservoir_npz(npz_paths: list[Path]) -> dict[str, np.ndarray]:
    merged: dict[str, list[np.ndarray]] = {}
    for path in npz_paths:
        if not path.is_file():
            continue
        with np.load(path) as arrays:
            for key in arrays.files:
                if not key.endswith("_reservoir"):
                    continue
                merged.setdefault(key, []).append(arrays[key])
    return {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in merged.items()
        if chunks
    }


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Router distribution summary",
        "",
        f"- model: `{payload.get('model_path', '')}`",
        f"- dataset/task: `{payload.get('dataset', '')}` / `{payload.get('task', '')}`",
        f"- harness: `{payload.get('harness', '')}`",
        f"- worker files merged: {payload.get('num_worker_files', 0)}",
        f"- invocations: {payload.get('invocation_count', 0)}",
        "",
        "## Per-layer / stage",
        "",
        "| layer | stage | tokens | entropy | eff_experts | hhi | margin | cum@8 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, stage_map in sorted(payload.get("layers", {}).items(),
                                   key=lambda x: int(x[0])):
        for stage, stats in sorted(stage_map.items()):
            lines.append(
                f"| {layer} | {stage} | {stats.get('token_count', 0)} | "
                f"{stats.get('entropy_mean', 0.0):.4f} | "
                f"{stats.get('effective_experts_mean', 0.0):.2f} | "
                f"{stats.get('hhi_mean', 0.0):.4f} | "
                f"{stats.get('margin_mean', 0.0):.4f} | "
                f"{stats.get('cum_mass_mean', {}).get('8', 0.0):.4f} |")
    lines.append("")
    return "\n".join(lines)


def _csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for layer, stage_map in sorted(payload.get("layers", {}).items(),
                                   key=lambda x: int(x[0])):
        for stage, stats in sorted(stage_map.items()):
            row = {
                "layer": layer,
                "stage": stage,
                "token_count": stats.get("token_count", 0),
                "entropy_mean": stats.get("entropy_mean", 0.0),
                "effective_experts_mean": stats.get("effective_experts_mean",
                                                     0.0),
                "hhi_mean": stats.get("hhi_mean", 0.0),
                "margin_mean": stats.get("margin_mean", 0.0),
            }
            for rank in CUM_RANKS:
                row[f"cum_mass_{rank}"] = stats.get("cum_mass_mean",
                                                     {}).get(str(rank), 0.0)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Task run dirs or router_distribution_*.json files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default="merged")
    args = parser.parse_args()

    json_paths = _collect_snapshot_paths(args.inputs)
    if not json_paths:
        raise SystemExit("No router_distribution_*.json snapshots found")

    merged = merge_worker_snapshots(json_paths)
    merged["merge_label"] = args.label
    merged["source_paths"] = [str(p) for p in json_paths]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.output_dir / f"{args.label}_summary.json"
    tmp_json = args.output_dir / f".{args.label}_summary.json.tmp"
    with tmp_json.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)
    tmp_json.replace(json_out)

    md_out = args.output_dir / f"{args.label}_summary.md"
    md_out.write_text(_markdown_summary(merged), encoding="utf-8")

    csv_out = args.output_dir / f"{args.label}_summary.csv"
    rows = _csv_rows(merged)
    if rows:
        with csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    npz_paths = []
    for root in args.inputs:
        if root.is_dir():
            npz_paths.extend(root.glob("router_distribution_*.npz"))
            for child in root.iterdir():
                if child.is_dir():
                    npz_paths.extend(child.glob("router_distribution_*.npz"))
    reservoirs = _load_reservoir_npz(sorted(set(npz_paths)))
    if reservoirs:
        npz_out = args.output_dir / f"{args.label}_reservoirs.npz"
        tmp_stem = args.output_dir / f".{args.label}_reservoirs.tmp"
        tmp_npz = args.output_dir / f".{args.label}_reservoirs.tmp.npz"
        np.savez_compressed(tmp_stem, **reservoirs)
        tmp_npz.replace(npz_out)

    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    if rows:
        print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
