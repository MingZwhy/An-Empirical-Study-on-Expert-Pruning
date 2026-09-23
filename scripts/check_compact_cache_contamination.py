#!/usr/bin/env python3
"""Show which cells of the old compact tables were damaged by the cache-reuse bug.

Those tables were produced before the lighteval sample cache was keyed on the expert
pruning configuration, so different methods and hyperparameters silently replayed one
another's cached samples. That leaves a fingerprint: byte-identical scores across
configurations that should have behaved differently. This quantifies the damage per
dataset, which is what justifies re-running rather than comparing against those numbers.

    python scripts/check_compact_cache_contamination.py Qwen3-30B-A3B-Instruct-2507
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COLUMNS = ["mmlu_pro", "gpqa", "math_500", "aime24", "aime25", "lcb", "gsm8k",
           "hellaswag", "livebench"]


def load_configs(path: Path) -> list[dict]:
    """Distinct (method, hyperparams) rows; one configuration spans several sections."""
    uniq: dict[tuple[str, str], dict] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 + len(COLUMNS) or cells[0] in ("method", "---"):
            continue
        try:
            avg = float(cells[1])
        except ValueError:
            continue
        vals = {}
        for key, cell in zip(COLUMNS, cells[5:5 + len(COLUMNS)]):
            try:
                vals[key] = float(cell)
            except ValueError:
                pass
        method = "Baseline" if cells[0].startswith("Baseline") else cells[0]
        uniq.setdefault((method, cells[2]),
                        {"method": method, "hyper": cells[2], "avg": avg, "vals": vals})
    return list(uniq.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*",
                    default=["Qwen3-30B-A3B-Instruct-2507", "gpt-oss-20b",
                             "Ling-lite-1.5-2507", "Qwen3-Next-80B-A3B-Instruct"])
    args = ap.parse_args()

    lines = ["# Cache-reuse damage in the compact tables", "",
             "The compact tables predate keying the lighteval sample cache on the expert "
             "pruning configuration, so runs replayed each other's samples. Identical "
             "scores across configurations that should differ are the fingerprint of "
             "that. Only the mapping from hyperparameters to average expert count "
             "survives; the scores do not.", ""]

    for model in args.models:
        path = REPO / "compact" / f"{model}.md"
        if not path.is_file():
            continue
        configs = load_configs(path)
        lines += [f"## {model}", "",
                  f"{len(configs)} distinct configurations.", "",
                  "| dataset | duplicated cells | most repeated value | shared by |",
                  "| --- | --- | --- | --- |"]
        tainted = total = 0
        for col in COLUMNS:
            groups = defaultdict(list)
            for cfg in configs:
                if col in cfg["vals"]:
                    groups[cfg["vals"][col]].append(cfg)
            dupes = {v: rs for v, rs in groups.items() if len(rs) > 1}
            n_col = sum(1 for c in configs if col in c["vals"])
            n_bad = sum(len(rs) for rs in dupes.values())
            tainted += n_bad
            total += n_col
            if dupes:
                worst_v, worst = max(dupes.items(), key=lambda kv: len(kv[1]))
                worst_txt = f"{worst_v:.2f}"
                shared = f"{len(worst)} configurations"
            else:
                worst_txt, shared = "—", "—"
            lines.append(f"| {col} | {n_bad}/{n_col} | {worst_txt} | {shared} |")
        pct = 100 * tainted / total if total else 0
        lines += ["",
                  f"Overall **{tainted}/{total} ({pct:.0f}%)** of the score cells are "
                  f"exact duplicates of another configuration's.", ""]

        # The clearest single illustration: one value spread over many configurations.
        worst_col, worst_val, worst_cfgs = None, None, []
        for col in COLUMNS:
            groups = defaultdict(list)
            for cfg in configs:
                if col in cfg["vals"]:
                    groups[cfg["vals"][col]].append(cfg)
            for v, rs in groups.items():
                if len(rs) > len(worst_cfgs):
                    worst_col, worst_val, worst_cfgs = col, v, rs
        if len(worst_cfgs) > 2:
            spread = f"{min(c['avg'] for c in worst_cfgs):.2f}–" \
                     f"{max(c['avg'] for c in worst_cfgs):.2f}"
            lines += [f"Worst single case: `{worst_col}` = {worst_val:.2f} appears for "
                      f"{len(worst_cfgs)} configurations whose average expert counts span "
                      f"{spread}:", ""]
            for cfg in sorted(worst_cfgs, key=lambda c: -c["avg"]):
                lines.append(f"- {cfg['method']} ({cfg['hyper']}) — avg {cfg['avg']:.2f}")
            lines.append("")

    dest = REPO / "docs" / "compact_cache_contamination.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
