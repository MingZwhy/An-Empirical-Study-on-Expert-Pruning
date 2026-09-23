#!/usr/bin/env python3
"""Export the user-selected model/k speed cells as readable Markdown."""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DETAILS = REPO / "results" / "speed_details"
OUTPUT = REPO / "results" / "SPEED_SELECTED_K_DETAILS.md"

SELECTIONS = [
    ("Qwen3-30B-A3B-Instruct-2507", 5),
    ("Qwen3-Next-80B-A3B-Instruct", 6),
    ("Ling-lite-1.5-2507", 4),
    ("gpt-oss-20b", 3),
]


def read_rows(model: str, k: int) -> list[dict]:
    with (DETAILS / f"{model}.csv").open(newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if int(row["k"]) == k
        ]


def number(value: str, digits: int = 2) -> str:
    if value in ("", None):
        return "—"
    return f"{float(value):.{digits}f}"


def table(rows: list[dict]) -> list[str]:
    lines = [
        "| bsz | input | output | metric | Baseline tok/s | Fixed-K tok/s | speedup | wall s | trial 1 s | trial 2 s |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            int(row["bsz"]),
            int(row["input_tokens"]),
            int(row["output_tokens"]),
        ),
    )
    for row in rows:
        metric = "prefill" if row["speedup_metric"] == "prefill_tok_s" else "decode"
        fixed = row["prefill_tok_s"] if metric == "prefill" else row["decode_tok_s"]
        lines.append(
            "| {bsz} | {inp} | {out} | {metric} | {base} | {fixed} | {speedup}× | {wall} | {t1} | {t2} |".format(
                bsz=row["bsz"],
                inp=row["input_tokens"],
                out=row["output_tokens"],
                metric=metric,
                base=number(row["native_same_shape_tok_s"]),
                fixed=number(fixed),
                speedup=number(row["speedup_vs_native_same_shape"], 3),
                wall=number(row["wall_s"], 3),
                t1=number(row["trial_1_s"], 3),
                t2=number(row["trial_2_s"], 3),
            )
        )
    return lines


def main() -> None:
    lines = [
        "# Selected Fixed-K speed details",
        "",
        "Every row below is one measured shape; no geometric averaging is used.",
        "`Baseline tok/s` is the native-k result at the identical backend and",
        "shape. For output=1 the metric is prefill. For output>1 the metric is",
        "decode throughput after subtracting the matching output=1 wall time.",
        "",
        "Each wall time is the median of the two listed trials.",
        "",
    ]
    for model, k in SELECTIONS:
        rows = read_rows(model, k)
        if len(rows) != 76:
            raise RuntimeError(f"Expected 76 rows for {model} k={k}, got {len(rows)}")
        lines += [f"## {model} — Fixed-K k={k}", ""]
        groups = [
            ("Expanded — vLLM", "expanded", "vllm"),
            ("Expanded — HF/Transformers", "expanded", "hf"),
            ("Long decode — vLLM (input=1024)", "long_decode", "vllm"),
        ]
        for title, suite, backend in groups:
            selected = [
                row
                for row in rows
                if row["suite"] == suite and row["backend"] == backend
            ]
            lines += [f"### {title}", "", *table(selected), ""]

    OUTPUT.write_text("\n".join(lines))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
