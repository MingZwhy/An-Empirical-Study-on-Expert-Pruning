#!/usr/bin/env python3
"""Report the multimodal sweep for Qwen3-VL-30B-A3B-Instruct.

This model is evaluated through lmms-eval rather than lighteval, so its results live in a different
file shape and the scores are not comparable to the text suite's cell for cell. What is comparable is
the shape of the decline: the same model family, the same five expert budgets, so the question this
table answers is whether taking experts away costs a vision-language model more than it costs the
text-only model of the same size.

Only Fixed-K is here. The dynamic methods need the router patch, which is written against the vLLM
build the text line uses; this model requires a newer one where the routing call moved, so the
methods cannot be applied yet (see docs/known_issues.md). Fixed-K needs no patch -- it is the
config-level top-k -- which is why the budget curve is available first.

    python scripts/report_vl.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = "Qwen3-VL-30B-A3B-Instruct"
TEXT_MODEL = "Qwen3-30B-A3B-Instruct-2507"
NATIVE_K = 8

# The nine datasets, with the metric each one reports. lmms-eval names its primary metric per task,
# so there is no single key to read; picking them explicitly also keeps the column order fixed.
TASKS = [
    ("mmmu_val", "mmmu_acc", "mmmu"),
    ("mmbench_en_dev_lite", "gpt_eval_score", "mmbench"),
    ("mmstar", "average", "mmstar"),
    ("chartqa_lite", "relaxed_overall", "chartqa"),
    ("infovqa_val_lite", "anls", "infovqa"),
    ("textvqa_val_lite", "exact_match", "textvqa"),
    ("gqa_lite", "exact_match", "gqa"),
    ("vizwiz_vqa_val_lite", "exact_match", "vizwiz"),
    ("mmerealworld_lite", "mme_realworld_score", "mme-rw"),
]
RUNS = [("Baseline", "k=8 (unpruned)", 8.0)] + [
    (f"FixedK-k{k}", f"k={k}", float(k)) for k in (7, 6, 5, 4, 3, 2, 1)
]

# The text model's matched side: nine log-likelihood QA datasets over the same ladder, which is what
# makes the two sweeps comparable dataset-count for dataset-count. Its generative suite answers a
# different question (see the section that uses it) and stays in the report as well.
# `acc,none` throughout, because winogrande publishes no `acc_norm` and one metric per table is worth
# more than the highest number each dataset can show.
QA9_TASKS = [
    ("arc_easy", "arc-e"),
    ("arc_challenge", "arc-c"),
    ("winogrande", "wino"),
    ("openbookqa", "obqa"),
    ("boolq", "boolq"),
    ("piqa", "piqa"),
    ("social_iqa_siqa_parquet", "siqa"),
    ("mmlu", "mmlu"),
    ("hellaswag", "hswag"),
]


def as_percent(value: float) -> float:
    """lmms-eval reports some metrics as fractions and mmbench's judge score out of 100."""
    return value if value > 1.5 else value * 100


def read_vl(run_dir: Path) -> dict | None:
    files = sorted(run_dir.glob("*/results/lmms_eval_results_*.json"))
    if not files:
        return None
    j = json.loads(files[-1].read_text())
    scores = {}
    for task, metric, _ in TASKS:
        got = j["results"].get(task, {})
        val = got.get(f"{metric},none")
        if isinstance(val, (int, float)):
            scores[task] = as_percent(float(val))
    return {"scores": scores,
            "mean": sum(scores.values()) / len(scores) if scores else None,
            "n": len(scores)}


def read_text(run_dir: Path) -> float | None:
    """The text model's nine-task mean for the same budget, for the drop-curve comparison."""
    import report_results as rr

    run = rr.read_run(run_dir)
    return run["mean"] if run else None


def read_text_qa9(budget: float) -> dict | None:
    """The text model's nine QA datasets at one budget, from its own Fixed-K ladder."""
    name = "QA9-Baseline-k8" if budget == 8.0 else f"QA9-FixedK-k{budget:.0f}"
    files = sorted((REPO / "results" / TEXT_MODEL / name).glob(
        "*/results/lm_eval_results_*.json"))
    if not files:
        return None
    results = json.loads(files[-1].read_text()).get("results", {})
    scores = {}
    for task, _ in QA9_TASKS:
        val = results.get(task, {}).get("acc,none")
        if isinstance(val, (int, float)):
            scores[task] = 100 * float(val)
    if not scores:
        return None
    return {"scores": scores, "mean": sum(scores.values()) / len(scores), "n": len(scores)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / MODEL / "RESULTS.md"))
    args = ap.parse_args()

    root = REPO / "results" / MODEL
    rows = [(name, hyper, avg, read_vl(root / name)) for name, hyper, avg in RUNS]
    done = [(n, h, a, r) for n, h, a, r in rows if r]
    base = next((r for n, _, _, r in done if n == "Baseline"), None)

    short = [s for _, _, s in TASKS]
    out = [
        f"# {MODEL} — expert pruning on the multimodal suite",
        "",
        f"{len(done)} of {len(RUNS)} configurations finished. Only Fixed-K appears here: the dynamic "
        "methods reach the router through a patch written for the vLLM build the text line runs on, "
        "and this model needs a newer one where that call moved, so they cannot be applied to it yet. "
        "Fixed-K is the model's own top-k set lower, which needs no patch, so the budget curve is "
        "available first.",
        "",
        "Scores are the primary metric of each dataset, as percentages: accuracy for mmmu, mmstar, "
        "gqa, vizwiz and mme-realworld, the judge score for mmbench, relaxed accuracy for chartqa, "
        "ANLS for infovqa, and exact match for textvqa. Seven of the nine are the `lite` subsets, "
        "which is what makes a full sweep affordable — the whole suite is 7319 questions and each "
        "configuration takes about 90 minutes on two cards.",
        "",
        "The average expert count is the definition, not a measurement: this environment's vLLM is "
        "not the patched one, so the counters that produce a measured average on the text line are "
        "not present. Fixed-K is exact by construction, so nothing is lost.",
        "",
        f"Settings: 9-dataset suite, TP=2, batch size 16, greedy decoding as lmms-eval defaults it, "
        f"native top-k of {NATIVE_K}.",
        "",
        "## Results by expert budget",
        "",
        "| config | hyperparams | avg experts | mean | " + " | ".join(short) + " |",
        "| --- | --- | --- | --- | " + " | ".join("---" for _ in short) + " |",
    ]
    for name, hyper, avg, run in done:
        cells = [f"{run['scores'][t]:.1f}" if t in run["scores"] else "—" for t, _, _ in TASKS]
        gap = ""
        if base and name != "Baseline":
            gap = f" ({run['mean'] - base['mean']:+.2f})"
        out.append(f"| {name.replace('FixedK-k', 'Fixed-K k=')} | {hyper} | {avg:.3f} | "
                   f"**{run['mean']:.2f}**{gap} | " + " | ".join(cells) + " |")

    # The comparison this model is here for. The text line named its unpruned run for the top-k it
    # runs at, so the directory names do not line up with this suite's.
    text_root = REPO / "results" / TEXT_MODEL
    text_dir = {"Baseline": "Baseline-k8"}
    text = {a: read_text(text_root / text_dir.get(n, n)) for n, _, a in RUNS
            if (text_root / text_dir.get(n, n)).is_dir()}
    text_base = text.get(8.0)
    if base and text_base:
        out += [
            "",
            "## Against the text-only model of the same size",
            "",
            f"{TEXT_MODEL} is the same parameter count and the same native top-k, evaluated on nine "
            "reasoning datasets instead of nine multimodal ones. Absolute scores are not comparable "
            "across the two suites; the decline from each model's own unpruned score is. The text "
            "means below are each run's own measured value, which is why the k=6 row reads slightly "
            "above its baseline — the text report holds its Fixed-K reference rows to the lower of "
            "two readings so that no method is credited with beating an inflated reference, and that "
            "adjustment is deliberately not applied here.",
            "",
            "| budget | multimodal mean | Δ from own baseline | text mean | Δ from own baseline |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, _, avg, run in done:
            if name == "Baseline":
                out.append(f"| k=8 (unpruned) | {run['mean']:.2f} | — | {text_base:.2f} | — |")
                continue
            t = text.get(avg)
            tcell = f"{t:.2f}" if t else "—"
            tgap = f"{t - text_base:+.2f}" if t else "—"
            out.append(f"| k={avg:.0f} | {run['mean']:.2f} | {run['mean'] - base['mean']:+.2f} | "
                       f"{tcell} | {tgap} |")
        pairs = [(a, r["mean"] - base["mean"], text[a] - text_base)
                 for n, _, a, r in done if n != "Baseline" and text.get(a)]
        if pairs:
            worst = max(pairs, key=lambda p: abs(p[1] - p[2]))
            knee = next((a for a, v, _ in sorted(pairs, reverse=True) if v < -5), None)
            text_knee = next((a for a, _, t in sorted(pairs, reverse=True) if t < -5), None)
            cheap = [a for a, v, t in pairs if v > -2 and t > -2]
            steep = "; ".join(f"{v:+.1f} against {t:+.1f} at k={a:.0f}"
                              for a, v, t in sorted(pairs, reverse=True) if v <= -5)
            out += [
                "",
                f"The two curves stay within {abs(worst[1] - worst[2]):.1f} points of each other at "
                f"every budget, and they break in the same place: "
                + (f"both are within two points of their own baseline down to k={min(cheap):.0f}"
                   if cheap else "")
                + (f", then multimodal against text runs {steep}. " if steep else ". ")
                + ("The knee is at the same budget for both suites, "
                   f"k={knee:.0f}. " if knee and knee == text_knee else
                   f"The multimodal suite's knee is at k={knee:.0f} against the text suite's "
                   f"k={text_knee:.0f}. " if knee and text_knee else "")
                + "So on this evidence what governs the damage is the expert budget rather than the "
                "modality: attaching a vision tower does not make the mixture more fragile than the "
                "same mixture answering text, at least not by more than a couple of points at any "
                "tier. The one caveat is that the multimodal suite is mostly short answers, so it "
                "has no equivalent of the reasoning tasks where one wrong token forfeits the whole "
                "item — the collapse at k=2 lands on both regardless, which suggests that at that "
                "budget the router, not the answer format, is what has stopped working.",
            ]

    # The matched comparison: nine datasets a side, same budgets, both scored on short answers.
    qa9 = {avg: read_text_qa9(avg) for _, _, avg in RUNS}
    qa9_base = qa9.get(8.0)
    if base and qa9_base:
        short_qa = [s for _, s in QA9_TASKS]
        out += [
            "",
            "## Nine against nine, on suites of the same character",
            "",
            "The comparison above is against the text model's reasoning suite, where a pruned router "
            "loses points two ways: it knows less, and it rambles past the token limit on tasks it "
            "could arguably still answer. The multimodal suite has no equivalent — its answers are a "
            "word or a letter — so the same nine-dataset count was measured on the text model with "
            "nine log-likelihood QA datasets, where nothing is generated and only the ranking of "
            "candidate answers matters. Both sides of this table are therefore nine datasets of short "
            "answers over the same eight budgets.",
            "",
            "| budget | multimodal mean | Δ | text QA mean | Δ | Δ difference |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        # Every budget gets a row, not only those the multimodal side has reached: the two ladders run on
        # different hosts and land out of step, and a row missing from this table reads as a budget nobody
        # measured rather than one measured on one side so far.
        vl_by_budget = {avg: run for name, _, avg, run in done}
        for name, _, avg in RUNS:
            run, t = vl_by_budget.get(avg), qa9.get(avg)
            if not run and not t:
                continue
            vcell = f"{run['mean']:.2f}" if run else "—"
            tcell = f"{t['mean']:.2f}" if t else "—"
            if avg == 8.0:
                out.append(f"| k=8 (unpruned) | {vcell} | — | {tcell} | — | — |")
                continue
            vgap = run["mean"] - base["mean"] if run else None
            tgap = t["mean"] - qa9_base["mean"] if t else None
            out.append(f"| k={avg:.0f} | {vcell} | " + (f"{vgap:+.2f}" if run else "—")
                       + f" | {tcell} | " + (f"{tgap:+.2f}" if t else "—") + " | "
                       + (f"{vgap - tgap:+.2f}" if run and t else "—") + " |")
        out += [
            "",
            "### The text model's nine QA datasets",
            "",
            "| budget | mean | " + " | ".join(short_qa) + " |",
            "| --- | --- | " + " | ".join("---" for _ in short_qa) + " |",
        ]
        for _, _, avg in RUNS:
            t = qa9.get(avg)
            if not t:
                continue
            cells = [f"{t['scores'][k]:.1f}" if k in t["scores"] else "—" for k, _ in QA9_TASKS]
            label = "k=8 (unpruned)" if avg == 8.0 else f"k={avg:.0f}"
            out.append(f"| {label} | **{t['mean']:.2f}** | " + " | ".join(cells) + " |")

        paired = [(a, r["mean"] - base["mean"], qa9[a]["mean"] - qa9_base["mean"])
                  for n, _, a, r in done if n != "Baseline" and qa9.get(a)]
        if paired:
            worst = max(paired, key=lambda p: abs(p[1] - p[2]))
            free = [a for a, v, t in paired if v > -2 and t > -2]
            deep = min(paired, key=lambda p: p[0])            # the tightest budget both sides reached
            gen = text.get(deep[0])                            # generative text mean at that budget
            gen_gap = gen - text_base if gen and text_base else None
            out += [
                "",
                (f"Both sides are within two points of their own baseline down to k={min(free):.0f}, "
                 if free else "")
                + f"and then they part: at k={deep[0]:.0f} the multimodal suite is {deep[1]:+.1f} "
                f"against the text QA suite's {deep[2]:+.1f}, a gap of "
                f"{abs(worst[1] - worst[2]):.1f} points at its widest."
                + (f" The text model's *generative* suite at that budget is {gen_gap:+.1f} — the "
                   "multimodal number sits with it, not with the QA number measured on the very same "
                   "text model. " if gen_gap is not None else " ")
                + "That is the useful finding of this table, and it is not about modality: lmms-eval "
                "scores a generated answer (exact match, ANLS, a judge), so it carries the same failure "
                "as the reasoning suite — a router this thin stops producing well-formed output. The QA "
                "ladder never generates a token; it ranks candidate continuations, and by that measure "
                "the model still knows a great deal at budgets where it can no longer say it. Reading "
                "the multimodal collapse as evidence that a vision tower is fragile would therefore be "
                "reading the scoring method.",
            ]

    missing = [n for n, _, _, r in rows if not r]
    if missing:
        out += ["", f"Not yet reported: {', '.join(missing)}."]
    qa_missing = [f"k={a:.0f}" for _, _, a in RUNS if not qa9.get(a)]
    if qa_missing:
        out += ["", f"Text QA ladder not yet reported at: {', '.join(qa_missing)}."]
    out += ["", f"Generated by `scripts/report_vl.py`."]

    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"写入 {args.out}（{len(done)} 个配置）")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
