#!/usr/bin/env python3
"""Export every fixed-shape speed cell to filterable CSV files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
EXPANDED = RESULTS / "_speed_fixed_expanded"
LONG = RESULTS / "_speed_long_decode"
OUT = RESULTS / "speed_details"

MODELS = {
    "Qwen3-Next-80B-A3B-Instruct": 10,
    "Qwen3-30B-A3B-Instruct-2507": 8,
    "Ling-lite-1.5-2507": 6,
    "gpt-oss-20b": 4,
}

FIELDS = [
    "suite",
    "model",
    "backend",
    "k",
    "native_k",
    "bsz",
    "input_tokens",
    "output_tokens",
    "wall_s",
    "trial_1_s",
    "trial_2_s",
    "prompt_tokens",
    "generated_tokens",
    "prefill_tok_s",
    "decode_tok_s",
    "total_tok_s",
    "speedup_vs_native_same_shape",
    "speedup_metric",
    "native_same_shape_tok_s",
    "load_s",
    "parallelism",
    "host",
    "cards",
    "peak_allocated_mib",
    "source_json",
    "error",
]


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("complete") is not True:
        raise RuntimeError(f"Incomplete benchmark file: {path}")
    return data


def cell_key(cell: dict) -> tuple[int, int, int]:
    return cell["bsz"], cell["in_len"], cell["out_len"]


def source_specs(model: str, native_k: int):
    for backend in ("vllm", "hf"):
        base_path = EXPANDED / model / f"{backend}_k{native_k}.json"
        base = load(base_path)
        yield "expanded", backend, EXPANDED / model, base
    base_path = LONG / model / f"vllm_k{native_k}.json"
    base = load(base_path)
    yield "long_decode", "vllm", LONG / model, base


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for model, native_k in MODELS.items():
        for suite, backend, directory, baseline in source_specs(model, native_k):
            baseline_cells = {
                cell_key(cell): cell
                for cell in baseline["cells"]
                if "error" not in cell
            }
            for k in range(native_k, 0, -1):
                path = directory / f"{backend}_k{k}.json"
                data = load(path)
                for cell in data["cells"]:
                    key = cell_key(cell)
                    base_cell = baseline_cells.get(key, {})
                    is_prefill = cell["out_len"] == 1
                    metric = "prefill_tok_s" if is_prefill else "decode_tok_s"
                    value = cell.get(metric)
                    base_value = base_cell.get(metric)
                    speedup = (
                        value / base_value
                        if value is not None and base_value not in (None, 0)
                        else None
                    )
                    trials = cell.get("trials_s", [])
                    rows.append(
                        {
                            "suite": suite,
                            "model": model,
                            "backend": backend,
                            "k": k,
                            "native_k": native_k,
                            "bsz": cell["bsz"],
                            "input_tokens": cell["in_len"],
                            "output_tokens": cell["out_len"],
                            "wall_s": cell.get("wall_s"),
                            "trial_1_s": trials[0] if len(trials) > 0 else None,
                            "trial_2_s": trials[1] if len(trials) > 1 else None,
                            "prompt_tokens": cell.get("prompt_tok"),
                            "generated_tokens": cell.get("gen_tok"),
                            "prefill_tok_s": cell.get("prefill_tok_s"),
                            "decode_tok_s": cell.get("decode_tok_s"),
                            "total_tok_s": cell.get("total_tok_s"),
                            "speedup_vs_native_same_shape": speedup,
                            "speedup_metric": metric,
                            "native_same_shape_tok_s": base_value,
                            "load_s": data.get("load_s"),
                            "parallelism": data.get("parallelism"),
                            "host": data.get("host"),
                            "cards": data.get("cards"),
                            "peak_allocated_mib": json.dumps(
                                cell.get("peak_allocated_mib"),
                                separators=(",", ":"),
                            )
                            if cell.get("peak_allocated_mib") is not None
                            else None,
                            "source_json": str(path.relative_to(RESULTS)),
                            "error": cell.get("error"),
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(OUT / "all_speed_cells.csv", rows)
    counts = {}
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        write_csv(OUT / f"{model}.csv", model_rows)
        counts[model] = len(model_rows)

    readme = [
        "# Per-shape speed results",
        "",
        "Each CSV row is one unaggregated model/backend/k/bsz/input/output cell.",
        "`speedup_vs_native_same_shape` compares that row with the native-k",
        "Baseline at exactly the same backend and shape.",
        "",
        f"- [All models ({len(rows):,} rows)](./all_speed_cells.csv)",
    ]
    for model, count in counts.items():
        readme.append(f"- [{model} ({count:,} rows)](./{model}.csv)")
    readme += [
        "",
        "Important columns:",
        "",
        "- `prefill_tok_s`: populated for output=1 cells.",
        "- `decode_tok_s`: output throughput after subtracting the matching",
        "  output=1 prefill wall time.",
        "- `trial_1_s`, `trial_2_s`: the two raw measured wall times.",
        "- `speedup_vs_native_same_shape`: no geometric averaging.",
        "- `suite=expanded`: the 34-shape vLLM/HF grid.",
        "- `suite=long_decode`: vLLM input=1024, bsz=1/4, output=2k/4k/8k.",
        "",
        "Generated by `scripts/export_speed_details.py`.",
    ]
    (OUT / "README.md").write_text("\n".join(readme) + "\n")
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
