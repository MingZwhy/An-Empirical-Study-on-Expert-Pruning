#!/usr/bin/env python3
"""Report the corrected step-4 results for one model.

The compact tables' scores are unusable — they were produced before the lighteval sample
cache was keyed on the expert pruning configuration, so configurations replayed each
other's samples (see docs/compact_cache_contamination.md). The one thing they still give
us is which hyperparameters put a method near a given average expert count, so these runs
reuse that mapping and measure the scores afresh.

The old table's average expert count is therefore shown as a check that the hyperparameters
were applied as intended; its scores are shown only where the analysis calls for them.

    python scripts/report_results.py Qwen3-30B-A3B-Instruct-2507
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Per model: the router's native top-k, the tensor parallelism the sweep ran at, and the
# number of always-on shared experts, which our statistics do not count.
MODELS = {
    "Qwen3-30B-A3B-Instruct-2507": dict(native_k=8, tp=2, configs=None),
    # Same architecture as the Instruct model above, down to the layer count and the router, so it
    # is here to answer one question: whether a model that reasons in its own output is more or less
    # sensitive to having experts taken away than the same model that answers directly.
    "Qwen3-30B-A3B-Thinking-2507": dict(native_k=8, tp=2,
                                       configs=None,
                                       mean_excludes=("gsm8k",),
                                       compare_to="Qwen3-30B-A3B-Instruct-2507"),
    "Ling-lite-1.5-2507": dict(native_k=6, tp=1, shared_experts=2,
                               configs=None),
    "gpt-oss-20b": dict(native_k=4, tp=1, configs=None),
    "Qwen3-Next-80B-A3B-Instruct": dict(native_k=10, tp=4, shared_experts=1,
                                        configs=None),
    # The one sigmoid-scored router in the study: every model above scores its experts with softmax and
    # renormalizes the selected weights, and no shipped checkpoint scores with sigmoid *without*
    # renormalizing, so this isolates the scoring function alone. It reasons in its output like the
    # Thinking model above, hence the same two exclusions. Budget rows only — the router patch does not
    # reach the engine this checkpoint needs.
    "GLM-4.7-Flash": dict(native_k=4, tp=2, shared_experts=1,
                          configs=None,
                          mean_excludes=("gsm8k",)),
    # Softmax without renormalization, the cell nothing else in the study occupies. Native top-6 plus
    # two always-on shared experts, same shape as Ling-lite; AIME is omitted from the sweep because
    # the model does not solve those items at any budget.
    "DeepSeek-V2-Lite-Chat": dict(native_k=6, tp=2, shared_experts=2,
                                  configs=None,
                                  omit_tasks=("aime24_avg", "aime25_avg")),
}

# The suite's generation limit. Runs at the most aggressive budgets use a shorter one, which is
# read back from each run and marked in its row rather than assumed here.
DEFAULT_CAP = 32768

COLUMNS = [
    ("mmlu_pro", "mmlu_pro"),
    ("gpqa:diamond", "gpqa"),
    ("math_500", "math500"),
    ("aime24_avg", "aime24"),
    ("aime25_avg", "aime25"),
    ("lcb:codegeneration_v6", "lcb"),
    ("gsm8k", "gsm8k"),
]
KEYS = [k for k, _ in COLUMNS]

# The 0-shot QA sweep, run by lm-eval over the same configurations. `acc` is used for all four
# because it is the only metric all four report: winogrande has no `acc_norm`.
QA_COLUMNS = [
    ("arc_challenge", "arc-c"),
    ("arc_easy", "arc-e"),
    ("winogrande", "winogrande"),
    ("openbookqa", "obqa"),
]
QA_KEYS = [k for k, _ in QA_COLUMNS]


def parse_compact_reference(path: Path) -> dict:
    """The old table's Baseline and Fixed-K rows, keyed by ("Baseline", 8) / ("Fixed-K", k).

    These stand in until the corresponding run finishes here. They are the rows the old
    table is most likely to have got right, since Fixed-K needs no calibration, but the
    cache bug did touch some of their cells, so they are shown as provisional.
    """
    refs: dict[tuple[str, int], dict] = {}
    if not path.is_file():
        return refs
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 + len(KEYS) or cells[0] in ("method", "---"):
            continue
        base = cells[0].startswith("Baseline")
        if not base and cells[0] != "Fixed-K":
            continue
        m = re.search(r"k=(\d+)", cells[2])
        if not m:
            continue
        try:
            avg = float(cells[1])
        except ValueError:
            continue
        cols = {}
        for key, cell in zip(KEYS, cells[5:5 + len(KEYS)]):
            try:
                cols[key] = float(cell)
            except ValueError:
                pass
        vals = list(cols.values())
        key = ("Baseline" if base else "Fixed-K", int(m.group(1)))
        refs.setdefault(key, {"cols": cols, "avg": avg,
                              "mean": sum(vals) / len(vals) if vals else None})
    return refs


def read_run(run_dir: Path, mean_excludes: tuple[str, ...] = ()) -> dict | None:
    """Scores and routing statistics from a finished run, or None if unfinished.

    `mean_excludes` names tasks that are shown but kept out of the mean. This exists for the
    reasoning model, where two of the nine tasks cap generation far below what a chain of thought
    needs — gsm8k at 256 tokens, livebench at 4096 — so the answer never arrives and the column
    reads near zero whether or not the model was pruned. Averaging that in would report a model as
    robust to pruning on the strength of two tasks it was never allowed to finish.
    """
    found = sorted(glob.glob(str(run_dir / "*" / "results" / "results_*.json")))
    if not found:
        return None
    data = json.loads(Path(found[-1]).read_text())
    scores = {}
    for task, metrics in data["results"].items():
        name = task.split("|")[0]
        if name == "livebench_20241125:_average":
            name = "livebench_20241125"
        elif name not in KEYS:
            continue
        vals = [v for k, v in metrics.items()
                if not k.endswith("_stderr") and isinstance(v, (int, float))]
        if vals:
            scores[name] = 100 * sum(vals) / len(vals)
    ep = data.get("expert_pruning") or {}
    # The generation limit is read back from the run rather than from its configuration, because
    # the two can disagree: a configuration edited after a run finished would otherwise relabel a
    # result that was produced under the old limit.
    gen = (data["config_general"].get("model_config") or {}).get(
        "generation_parameters") or {}
    counted = {k: v for k, v in scores.items() if k not in mean_excludes}
    return {
        "cols": scores,
        "mean": sum(counted.values()) / len(counted) if counted else None,
        "avg": ep.get("average_selected_experts"),
        "cap": gen.get("max_new_tokens"),
        "router_invoked": ep.get("router_invoked"),
        "cache_fingerprint": ep.get("expert_pruning_cache_fingerprint"),
    }


def read_qa(run_dir: Path) -> dict | None:
    """The 0-shot QA scores for a configuration, or None if its QA run has not landed.

    These sit in the same directory as the generative results under their own filename prefix, and
    they are a separate measurement of the same configuration rather than more columns of the same
    one: multiple choice scored by likelihood, no generation at all. That also makes the expert
    average worth reading again here — the router reacts to its input, and a one-sentence question
    is not a chain of reasoning.
    """
    found = sorted(glob.glob(str(run_dir / "*" / "results" / "lm_eval_results_*.json")))
    if not found:
        return None
    data = json.loads(Path(found[-1]).read_text())
    scores, variances = {}, []
    for task, metrics in data.get("results", {}).items():
        if task in QA_KEYS and "acc,none" in metrics:
            scores[task] = 100 * metrics["acc,none"]
            se = metrics.get("acc_stderr,none")
            if se is not None:
                variances.append((100 * se) ** 2)
    ep = data.get("expert_pruning") or {}
    # The standard error of the four-task mean, so a two-point difference between rows can be told
    # from a two-point difference in what the sets happened to ask.
    se_mean = (sum(variances) ** 0.5 / len(variances)
               if len(variances) == len(scores) and variances else None)
    # Read the concurrency ceiling from the run rather than from the configuration list: some rows
    # were capped by editing the queue directly, so the list does not always carry the flag while the
    # result always records what the engine was actually given.
    engine = (data.get("config_general") or {}).get("vllm_model_args") or {}
    return {
        "cols": scores,
        "mean": sum(scores.values()) / len(scores) if scores else None,
        "se": se_mean,
        "avg": ep.get("average_selected_experts"),
        "n_tasks": len(scores),
        "max_num_seqs": engine.get("max_num_seqs"),
    }


def generation_lengths(run_dir: Path, tasks: tuple[str, ...],
                       cap: int = DEFAULT_CAP) -> dict[str, dict]:
    """Output-token length stats per task, from the run's cached generations.

    A pruned model that rambles can run into the context window, and a cut-off answer
    scores zero however good the routing was, so this separates truncation from quality.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return {}
    stats = {}
    for task in tasks:
        lens: list[int] = []
        files = sorted(glob.glob(
            str(run_dir / ".lighteval_cache" / "*" / "*" / "*" / f"{task}|0"
                / "*" / "*.parquet")))
        # A run that was restarted under a different generation limit leaves its earlier samples
        # behind in a second cache namespace, since the limit is part of lighteval's model hash.
        # Mixing the two would make these lengths describe neither run, so only the namespace
        # written most recently — the one that produced the scores above — is read.
        if files:
            newest = max(files, key=lambda f: Path(f).stat().st_mtime)
            hash_dir = Path(newest).parents[2]
            files = [f for f in files if Path(f).parents[2] == hash_dir]
        for f in files:
            for sample in pd.read_parquet(f)["sample"]:
                out = sample.get("output_tokens")
                if out is None:
                    continue
                if isinstance(out, (list, np.ndarray)) and len(out) and isinstance(
                        out[0], (list, np.ndarray)):
                    lens.extend(len(x) for x in out)
                else:
                    lens.append(len(out))
        if lens:
            s = pd.Series(lens)
            stats[task] = {"n": len(s), "median": s.median(), "cap": cap,
                           "p90": s.quantile(0.9),
                           "at_cap": 100 * (s >= cap - 768).mean()}
    return stats


def num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def duplicate_share(model: str) -> str:
    """The model's line from the contamination analysis, so the figure has one source."""
    path = REPO / "docs" / "compact_cache_contamination.md"
    if not path.is_file():
        return "most"
    section = False
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            section = line[3:].strip() == model
        elif section:
            m = re.search(r"Overall \*\*\d+/\d+ \((\d+)%\)\*\*", line)
            if m:
                return f"{m.group(1)}% of"
    return "most"


def pick_qa_rows(rows: list[tuple], target: float) -> list[tuple]:
    """One row per method for a tier's QA table, keeping the reference rows whole.

    A method can have two settings in a tier here: the one calibrated on generation and the one
    re-solved for these questions. They are not two measurements of one configuration, they are two
    operating points, and the table is about the tier's budget -- so the row shown is the cheapest one
    that still spends at least what Fixed-K does. Where no setting reaches the tier yet, the closest
    one is shown and marked, since the alternative is an empty cell that hides the gap.
    """
    keep: dict[str, tuple] = {}
    out = []
    for cfg, got, run in rows:
        if cfg["method"] in ("Baseline", "Fixed-K"):
            out.append((cfg, got, run))
            continue
        cur = keep.get(cfg["method"])
        if cur is None:
            keep[cfg["method"]] = (cfg, got, run)
            continue
        new_avg, old_avg = got["avg"], cur[1]["avg"]
        if new_avg is None or old_avg is None:
            continue
        new_ok, old_ok = new_avg >= target, old_avg >= target
        if (new_ok and not old_ok) or (new_ok == old_ok
                                      and (new_avg < old_avg if new_ok else new_avg > old_avg)):
            keep[cfg["method"]] = (cfg, got, run)
    out += list(keep.values())
    return sorted(out, key=lambda t: row_order(t[0]))


def row_order(cfg: dict) -> tuple:
    """Reference rows first: the unpruned model, then the Fixed-K point for that target."""
    rank = {"Baseline": 0, "Fixed-K": 1}.get(cfg["method"], 2)
    return (rank, cfg["method"], cfg["hyper"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-30B-A3B-Instruct-2507",
                    choices=sorted(MODELS))
    ap.add_argument("--configs")
    ap.add_argument("--lengths", action="store_true",
                    help="also summarise generation lengths (reads the cached outputs)")
    args = ap.parse_args()

    spec = MODELS[args.model]
    native_k = spec["native_k"]
    configs_path = args.configs or spec.get("configs")
    if not configs_path or not Path(configs_path).exists():
        raise SystemExit(
            f"[report_results] no configuration manifest for {args.model}.\n"
            "  This script regenerates results/<model>/RESULTS.md from raw run\n"
            "  directories, so it needs both your own sweep output under\n"
            "  results/<model>/ and a manifest describing the configurations:\n"
            "    python scripts/report_results.py <model> --configs my.json\n"
            "  The manifest is a JSON list of {dirname, method, hyper, targets,\n"
            "  table_avg} entries, one per configuration. The published tables in\n"
            "  results/*/RESULTS.md were produced this way; the raw runs behind\n"
            "  them are tens of GB and are not shipped.")
    configs = json.loads(Path(configs_path).read_text())
    # A configuration that was decided against is not pending work, and counting it as such makes
    # a finished sweep look unfinished for as long as the decision stands.
    skipped = [c for c in configs if c.get("skipped")]
    configs = [c for c in configs if not c.get("skipped")]
    # A probe is not a result of its own: it is a short run whose only job is to measure the average
    # expert count for a configuration that could not measure its own (see the ᵍ footnote).
    probes = {c["probe_for"]: c for c in configs if c.get("probe_for")}
    configs = [c for c in configs if not c.get("probe_for")]
    # A QA-recalibrated configuration is the same method at a threshold solved for the QA prompts
    # rather than for generation, because the generative settings spend fewer experts on short
    # questions than the tier they were calibrated for. It has no generative run and is not waiting
    # for one, so the tables above and their "N of M finished" count leave it out.
    qa_only = [c for c in configs if c.get("qa_only")]
    configs = [c for c in configs if not c.get("qa_only")]
    results_root = REPO / "results" / args.model
    refs = parse_compact_reference(REPO / "compact" / f"{args.model}.md")
    done = []
    qa: dict[str, dict] = {}
    excludes = tuple(spec.get("mean_excludes", ()))
    for cfg in configs:
        run = read_run(results_root / cfg["dirname"], excludes)
        if run is None:
            continue
        got = read_qa(results_root / cfg["dirname"])
        if got and got["mean"] is not None:
            if got["avg"] is None and cfg["method"] in ("Baseline", "Fixed-K"):
                m = re.search(r"k=(\d+)", cfg["hyper"])
                if m:
                    got["avg"] = float(m.group(1))
            qa[cfg["dirname"]] = got
        # Neither the unpruned run nor Fixed-K installs a router, so nothing counts experts
        # for them; their average is the k they were given, by definition.
        if run["avg"] is None and cfg["method"] in ("Baseline", "Fixed-K"):
            m = re.search(r"k=(\d+)", cfg["hyper"])
            if m:
                run["avg"] = float(m.group(1))
        # A run that kept CUDA graphs cannot count its own experts, so the average comes from the
        # probe alongside it. Without this the row would have no place on the expert-count axis.
        if run["avg"] is None and cfg["dirname"] in probes:
            probe = read_run(results_root / probes[cfg["dirname"]]["dirname"])
            if probe and probe["avg"] is not None:
                run["avg"], run["avg_from_probe"] = probe["avg"], True
        done.append((cfg, run))
    done.sort(key=lambda t: (-max(t[0]["targets"]),) + row_order(t[0]))
    for cfg in qa_only:
        got = read_qa(results_root / cfg["dirname"])
        if got and got["mean"] is not None:
            qa[cfg["dirname"]] = got

    omit = tuple(spec.get("omit_tasks", ()))
    columns = [(k, s) for k, s in COLUMNS if k not in omit]
    keys = [k for k, _ in columns]
    excl_short = {s for k, s in columns if k in tuple(spec.get("mean_excludes", ()))}
    short = [s + (" ˣ" if s in excl_short else "") for _, s in columns]
    out = [
        f"# {args.model} — corrected expert-pruning results",
        "",
        f"{len(done)} of {len(configs)} configurations finished."
        + (f" {len(skipped)} further configurations are set aside and not counted here: "
           + "; ".join(sorted({c["skipped"] for c in skipped})) + "."
           if skipped else ""),
        "",
    ]
    # A model the old study never swept has no compact table, and saying its scores are unusable
    # would describe a document that does not exist. Only the four original models have one.
    if refs:
        out += [
            f"The compact table's scores cannot be used as a reference: "
            f"{duplicate_share(args.model)} its score cells "
            "are exact duplicates of another configuration's, because it predates keying the "
            "lighteval sample cache on the expert pruning configuration "
            "(see `docs/compact_cache_contamination.md`). What it still provides, and what "
            "these runs take from it, is which hyperparameters land a method near a given "
            "average expert count.",
            "",
        ]
    out += [
        "Every table opens with two reference rows: the unpruned model (its native top-k "
        f"of {native_k}) and the Fixed-K point for that target. Fixed-K is what a method "
        "has to beat, since it reaches the same average expert count by simply keeping "
        "fewer experts per token. The bracketed number after each mean is the gap to the "
        "unpruned model.",
        "",
    ]
    if refs:
        out += [
            "Rows marked ᵗ are still the old table's values, kept as a placeholder so the "
            "comparison is readable now; they are being re-run and will be replaced. Fixed-K "
            "needs no calibration, so it is the part of the old table most likely to be right, "
            "but some of its cells do carry the duplicate-score fingerprint too, so the "
            "re-runs are worth waiting for.",
            "",
            "The reference rows are held to the lower of the two readings: where the re-run came "
            "out above the old table it keeps the old value and is marked ᵒ, with what it "
            "actually scored given in the footnote. This only ever moves the reference down, so "
            "no pruning method is credited with beating an inflated Fixed-K.",
            "",
        ]
    suite_note = ("five generative datasets (AIME24/25 omitted — this model does not solve "
                  "them), " if omit else "the seven generative datasets, ")
    out += [
        f"Settings: {suite_note}temperature 0.7 / top_p 0.8 / top_k 20, "
        f"max_model_len and max_new_tokens {DEFAULT_CAP} unless a row is marked ᵏ, "
        f"TP={spec['tp']}. Every run has its own cache namespace keyed on the "
        "pruning configuration, so no two rows here can share samples.",
        "",
        "## Results by target average expert count",
        "",
    ]
    if excludes:
        out[-2:-2] = [
            "This model reasons in its output, and gsm8k caps generation below what that "
            "needs: 256 tokens, against the "
            f"{DEFAULT_CAP} the rest of the suite gets. A chain of thought does not finish inside "
            "256 tokens, so that column reads near zero for the unpruned model as well, and "
            "it is shown below but left out of the mean — averaging them in would credit "
            "pruning with preserving a score the model never got to produce. The caps are "
            "deliberately left as they are, because the point of this model being here is "
            f"comparison against {spec.get('compare_to', 'the other models in this study')}, "
            "which ran under exactly these "
            "limits; raising them for one side would buy a fairer number at the cost of the "
            "comparison. Tasks kept out of the mean are marked ˣ in the header — not ᵏ, which "
            "is a row-level marker for a run that used a shorter limit than the suite's.",
            "",
        ]

    # A configuration listed under several targets belongs in each of those tables, so the
    # unpruned baseline shows up as the reference row everywhere.
    diep_gaps: list[float] = []
    gen_under: list[tuple[str, float]] = []
    off_target = probed = False
    capped: dict[int, int] = {}
    held: list[tuple[str, int, float, float]] = []
    by_target: dict[float, list] = {}
    for cfg in configs:                      # so a target's table exists, with its reference
        for target in cfg["targets"]:        # rows, before any of its runs has finished
            by_target.setdefault(target, [])
    for cfg, run in done:
        for target in cfg["targets"]:
            by_target[target].append((cfg, run))
    for target in sorted(by_target, reverse=True):
        avg_mark = " ᵃ" if spec.get("shared_experts") else ""
        out += [f"### Target avg ≈ {target:.1f}", "",
                f"| method | hyperparams | avg experts{avg_mark} | mean | "
                + " | ".join(short) + " |",
                "| --- | --- | --- | --- | "
                + " | ".join("---" for _ in short) + " |"]
        rows = []
        for cfg, run in sorted(by_target[target], key=lambda t: row_order(t[0])):
            mark = ""
            if cfg["method"] in ("Baseline", "Fixed-K"):
                k = native_k if cfg["method"] == "Baseline" else int(target)
                ref = refs.get((cfg["method"], k))
                # A reference row is held to the lower of the two readings. Fixed-K needs no
                # calibration, so the old table had little room to get it wrong, and a re-run
                # that lands above it is taken as measurement spread rather than as a better
                # reference — crediting it would flatter every pruning method in the table.
                if ref and run["mean"] and ref["mean"] and run["mean"] > ref["mean"]:
                    held.append((cfg["method"], k, run["mean"], ref["mean"]))
                    run, mark = ref, "ᵒ"
            rows.append((cfg["method"], cfg["hyper"], run, mark))

        # Reference rows come from the old table until the run here finishes and replaces
        # them, so the comparison is readable from the first day of the sweep.
        for method, k in (("Baseline", native_k), ("Fixed-K", int(target))):
            if any(r[0] == method for r in rows):
                continue
            ref = refs.get((method, k))
            if ref:
                hyper = f"k={native_k} (unpruned)" if method == "Baseline" else f"k={k}"
                rows.insert(0 if method == "Baseline" else 1,
                            (method, hyper, ref, "ᵗ"))

        base = next((r[2] for r in rows if r[0] == "Baseline"), None)
        base_mean = base["mean"] if base else None
        # Only compare against Fixed-K where the budgets actually match. A method that never
        # reached the target still appears in its table at the closest setting it has, and
        # calling that a gap would credit it for the experts it did not give up.
        fixed = next((r[2] for r in rows if r[0] == "Fixed-K"), None)
        if fixed and fixed["mean"] and fixed["avg"]:
            diep_gaps += [fixed["mean"] - r[2]["mean"] for r in rows
                          if r[0] == "DiEP" and r[2]["mean"] and r[2]["avg"]
                          and abs(r[2]["avg"] - fixed["avg"]) <= 0.5]
        for method, hyper, run, mark in rows:
            mean = num(run["mean"])
            if base_mean and method != "Baseline" and run["mean"]:
                mean = f"**{mean}** ({run['mean'] - base_mean:+.2f})"
            else:
                mean = f"**{mean}**"
            label = f"{method} {mark}" if mark else method
            if method == "DiEP":
                label += " ᵈ"
            # A method with no setting near this target is shown at its closest one, which
            # spends more experts than the table's other rows; say so rather than let the
            # mean be read as a like-for-like comparison.
            if (method != "Baseline" and run["avg"]
                    and abs(run["avg"] - target) > 0.5):
                label += " ᵛ"
                off_target = True
            # The other way to miss the target, which the mark above does not distinguish and which
            # matters more: fewer experts than Fixed-K spends at this tier. Then a lower mean can be
            # explained by the smaller budget, so the row is not the comparison it looks like.
            if (method not in ("Baseline", "Fixed-K") and run["avg"]
                    and run["avg"] < target - 0.05):
                label += " ᵘ"
                gen_under.append((f"{method} at {hyper}", run["avg"] - target))
            if run.get("cap") and run["cap"] != DEFAULT_CAP:
                label += " ᵏ"
                capped[run["cap"]] = capped.get(run["cap"], 0) + 1
            if run.get("avg_from_probe"):
                label += " ᵍ"
                probed = True
            out.append(
                f"| {label} | {hyper} | {num(run['avg'], 3)} | {mean} | "
                + " | ".join(num(run["cols"].get(k), 1) for k in keys) + " |")
        out.append("")
    gap = ""
    if diep_gaps:
        lo, hi = min(diep_gaps), max(diep_gaps)
        gap = (f", leaving it {lo:.0f} to {hi:.0f} points below Fixed-K at the same budget"
               if hi - lo >= 1 else f", leaving it {hi:.0f} points below Fixed-K")
    under_note = []
    if gen_under:
        worst = min(gen_under, key=lambda u: u[1])
        under_note = [
            f"ᵘ spends fewer experts than Fixed-K does at that tier, by up to "
            f"{abs(worst[1]):.2f} ({worst[0]}). {len(gen_under)} "
            + ("rows are" if len(gen_under) > 1 else "row is")
            + " in that position, and a mean below Fixed-K's there can be explained by the smaller "
            "budget rather than by the method. These are not being re-run: a generative run of this "
            "model costs a day, and the threshold that would fix it is a different operating point "
            "rather than a correction to this one. The 0-shot QA tables below were re-solved instead, "
            "since a run there is minutes.",
            "",
        ]
    out += under_note
    # Both of these explain a marker, so they only belong in a report that carries it: a model added
    # after the compact tables were written has no ᵗ rows, and one whose sweep is budgets-only has no
    # DiEP rows to qualify.
    if refs:
        out += [
            "ᵗ still the old table's numbers, waiting on the run here to replace them.",
            "",
        ]
    if any("DiEP" in cfg["method"] for cfg in configs):
        out += [
            "ᵈ DiEP's skipping rule is specified for k=2 and our generalisation of it to a "
            f"top-{native_k} router prunes the whole tail{gap}. Only the paper's auxiliary "
            "skipping mechanism is implemented, not its differentiable pruning. Read these rows "
            "as a result about that generalisation, not as DiEP's achievable quality — see "
            "`docs/known_issues.md`.",
            "",
        ]
    if held:
        measured = ", ".join(f"k={k} scored {new:.2f} against {old:.2f}"
                             for _, k, new, old in sorted(held, key=lambda t: -t[1]))
        out += [
            "ᵒ the re-run of this reference row came out above the old table, so the old "
            "table's value is kept. Fixed-K needs no calibration and the old table had little "
            "room to get it wrong, so a higher reading here is treated as measurement spread "
            "rather than as a better reference; taking it would flatter every pruning method "
            f"measured against it. What the re-runs actually scored: {measured}.",
            "",
        ]
    if off_target:
        out += [
            "ᵛ this method has no setting that reaches this target, so its closest one is "
            "shown. It is spending more experts than the other rows and its mean is not "
            "comparable to them.",
            "",
        ]
    if probed:
        out += [
            "ᵍ this model is quantized, so its expert selection is inlined into the compiled "
            "region and a run that keeps CUDA graphs cannot count its own experts. Forcing "
            "eager mode to get the count costs a factor of 6.7 in generation throughput "
            "(166 against 1138 tok/s measured here), which is the difference between a run "
            "that takes a day and one that takes two weeks, so the score comes from a graph "
            "run and the average from a separate eager probe of the same configuration over "
            "40 prompts each of mmlu_pro, math_500 and gsm8k — a few million routing "
            "decisions, far more than an average needs, and spread over three input "
            "distributions because the router reacts to its input. "
            "The probe is also the more correct place to measure it: under CUDA graphs vLLM "
            "pads each batch to a captured size, and those padding rows would be counted as "
            "tokens.",
            "",
        ]
    if capped:
        limits = ", ".join(f"{cap // 1024}k on {n} row{'s' if n > 1 else ''}"
                           for cap, n in sorted(capped.items()))
        out += [
            f"ᵏ generated under a shorter limit than the {DEFAULT_CAP // 1024}k used "
            f"elsewhere ({limits}). At two or three experts the model rambles and runs almost "
            "every answer into the limit, and an answer that is not right by 16k is very "
            "rarely right at 32k, so the shorter window buys a large amount of time at little "
            "cost in accuracy. It is not free, though: a row marked here could in principle "
            "lose a point to truncation that the unpruned reference did not pay. "
            + ("`Ban-kmin2-lambda0.3-cap16k` measures exactly that, being the same "
               "configuration as `Ban-kmin2-lambda0.3` under the two limits."
               if any("cap16k" in c["dirname"] for c in configs) else ""),
            "",
        ]
    if spec.get("shared_experts"):
        out += [
            f"ᵃ this model keeps {spec['shared_experts']} shared experts active for every "
            "token on top of the routed ones. The average counts routed experts only, "
            "matching the old table, so an average of 3 here costs "
            f"{3 + spec['shared_experts']} experts of compute and is not comparable to an "
            "average of 3 on a model without shared experts.",
            "",
        ]

    if qa:
        qa_short = [s for _, s in QA_COLUMNS]
        out += [
            "## 0-shot QA (lm-eval)",
            "",
            "Four multiple-choice sets, scored by likelihood over the options with nothing "
            "generated, so they answer a different question from the tables above: how much of the "
            "model's knowledge survives pruning, as opposed to how much of its reasoning does. A "
            "budget that ruins the generative scores can leave these nearly intact, and the gap "
            "between the two is the interesting part.",
            "",
            "The thresholds below are not the ones in the tables above, and that is the point. A "
            "threshold is not a budget: it admits however many experts the router's own confidence "
            "lets it, and the router is more decided about a one-sentence question than about the "
            "middle of a derivation. The generative settings therefore spend fewer experts here "
            "than the tier they were calibrated for, and a row that is both cheaper than Fixed-K "
            "and worse than it answers nothing. Each method was re-solved against this distribution "
            "(`scripts/solve_qa_hyper.py`) so that its average sits at or just above Fixed-K's.",
            "",
            "`acc` is used for all four columns because it is the only metric all four report — "
            "winogrande has no `acc_norm`. The result files carry `acc_norm` for the other three, "
            "and taking it where available moves rows by up to three places, so a comparison "
            "against a published table should be read off the JSON with that table's own metric.",
            "",
        ]
        # Where a dynamic method comes out above Fixed-K without spending more experts than it. The
        # generative tables have almost none of these, so the margins are worth stating next to the
        # standard error of the mean they are read off, rather than reported as wins.
        beats: list[float] = []
        under: list[tuple[str, float]] = []
        capped_qa: set[str] = set()
        qa_by_target: dict[float, list] = {}
        for cfg, run in done:
            if cfg["dirname"] in qa:
                for t in cfg["targets"]:
                    qa_by_target.setdefault(t, []).append((cfg, qa[cfg["dirname"]], run))
        for cfg in qa_only:
            if cfg["dirname"] in qa:
                for t in cfg["targets"]:
                    qa_by_target.setdefault(t, []).append((cfg, qa[cfg["dirname"]], {"avg": None}))
        for target in sorted(qa_by_target, reverse=True):
            rows = pick_qa_rows(qa_by_target[target], target)
            if not rows:
                continue
            ref = next((q for c, q, _ in rows if c["method"] == "Fixed-K"), None)
            if ref and ref["mean"] and ref["avg"]:
                beats += [q["mean"] - ref["mean"] for c, q, _ in rows
                          if c["method"] not in ("Baseline", "Fixed-K") and q["mean"]
                          and q["avg"] and q["avg"] <= ref["avg"] + 0.15
                          and q["mean"] > ref["mean"]]
            out += [
                f"### Target avg ≈ {target:.1f}",
                "",
                "| method | hyperparams | avg experts (QA) | avg (gen) | mean | "
                + " | ".join(qa_short) + " |",
                "| --- | --- | --- | --- | --- | "
                + " | ".join("---" for _ in qa_short) + " |",
            ]
            base = next((q["mean"] for c, q, _ in rows if c["method"] == "Baseline"), None)
            for cfg, got, run in rows:
                mean = num(got["mean"])
                if base and cfg["method"] != "Baseline":
                    mean = f"**{mean}** ({got['mean'] - base:+.2f})"
                else:
                    mean = f"**{mean}**"
                mark = ""
                if (cfg["method"] not in ("Baseline", "Fixed-K") and got["avg"] is not None
                        and got["avg"] < target - 0.05):
                    mark = " ᵘ"
                    under.append((cfg["dirname"], got["avg"] - target))
                if got.get("max_num_seqs"):
                    mark += " ᶜ"
                    capped_qa.add(cfg["dirname"])
                out.append(
                    f"| {cfg['method']} | {cfg['hyper']} | {num(got['avg'], 3)}{mark} | "
                    f"{num(run['avg'], 3)} | {mean} | "
                    + " | ".join(num(got["cols"].get(k), 1) for k in QA_KEYS) + " |")
            out.append("")
        # Both averages are measured over the same router with the same threshold, so where they
        # disagree it is the input distribution talking, and that is worth naming rather than
        # leaving as two columns a reader has to diff by eye.
        if under:
            worst = min(under, key=lambda u: u[1])
            out += [
                f"ᵘ spends fewer experts than Fixed-K does at that tier, by up to "
                f"{abs(worst[1]):.2f} ({worst[0]}). {len(under)} "
                + ("rows are" if len(under) > 1 else "row is")
                + " in that position, which makes them unreadable as comparisons rather than merely "
                "imprecise: a lower score can be explained by the smaller budget. Their re-solved "
                "settings are queued, and each replaces the row above when it lands.",
                "",
            ]
        if capped_qa:
            out += [
                f"ᶜ run with vLLM's concurrent sequences capped at 64 instead of the default 256, "
                f"which {len(capped_qa)} "
                + ("rows needed" if len(capped_qa) > 1 else "row needed")
                + ": this model at TP=4 has under 2 GiB left on a 47 GiB card once weights, "
                "activation peak, CUDA graphs and KV cache are placed, and the sampler warm-up "
                "allocates for all 256 sequences at once, which DiEP and Ban tip over while NAEE and "
                "Dynamic Routing do not. Worth knowing what the cap does and does not touch, since "
                "one configuration was measured both ways: expert selection is unaffected (6.0674 "
                "experts at 64 against 6.0674 at 256), and the scores move by 0.92 points on average "
                "and 2.20 at most, in both directions. That is not run-to-run noise — repeating a run "
                "at unchanged settings reproduced all seven metrics exactly — it is a real dependence "
                "on batch shape, flipping the multiple-choice calls that were near ties. So a "
                "difference of about a point between a marked row and an unmarked one is not a "
                "difference between the methods.",
                "",
            ]
        ses = [q["se"] for q in qa.values() if q.get("se")]
        if beats and ses:
            out += [
                f"Fixed-K does not win here the way it does above: {len(beats)} "
                + ("rows come" if len(beats) > 1 else "row comes")
                + f" out ahead of it without spending more experts, by up to "
                  f"{max(beats):.2f} points. "
                f"The standard error of this four-task mean is about {sum(ses) / len(ses):.2f} "
                "points, though, so those are ties rather than wins — which is itself the finding, "
                "because in the generative tables the same methods lose to Fixed-K by margins far "
                "outside that.",
                "",
            ]
        spread = [(qa[c["dirname"]]["avg"] - r["avg"], c["dirname"]) for c, r in done
                  if c["dirname"] in qa and qa[c["dirname"]]["avg"] is not None
                  and r["avg"] is not None and c["method"] not in ("Baseline", "Fixed-K")]
        if not any(q["avg"] is not None for c, q in qa.items()
                   if c not in {x["dirname"] for x in configs
                                if x["method"] in ("Baseline", "Fixed-K")}):
            out += [
                "The QA average expert count is blank for this model for the same reason its "
                "generative rows take theirs from a probe: expert selection is inlined into the "
                "compiled region, and a run that keeps CUDA graphs cannot count it. The generative "
                "probe's value is not substituted here, because the number below would then "
                "describe a different input distribution from the scores next to it.",
                "",
            ]
        if spread:
            lo, hi = min(spread), max(spread)
            out += [
                f"The two averages are the same configuration measured under different inputs, and "
                f"they differ by between {lo[0]:+.2f} and {hi[0]:+.2f} experts "
                f"({lo[1]} and {hi[1]} are the extremes). A dynamic method reads the router's own "
                "confidence, and short factual questions do not distribute it the way a long "
                "derivation does, so a budget calibrated on generation does not transfer here "
                "exactly. The QA column is the one that describes these scores.",
                "",
            ]

    out += [
        "## Average expert count landed where it was expected to",
        "",
        "The hyperparameter-to-average mapping is the only part of the compact table that "
        "survived the cache bug, so this confirms the re-runs are sitting at the intended "
        "operating points.",
        "",
        "| config | expected avg | this run | Δ |",
        "| --- | --- | --- | --- |",
    ]
    predicted = unexpected = False
    for cfg, run in done:
        mark = ""
        if cfg.get("avg_source") == "predicted":
            mark, predicted = " ᵖ", True
        # A configuration that exists only to isolate one hyperparameter has no expected
        # average: it is not in the old table and nothing was solved for it, so its average is
        # a measurement rather than a prediction under test.
        if cfg["table_avg"] is None:
            expected, delta, unexpected = "— ⁿ", "—", True
        else:
            expected = f"{cfg['table_avg']:.4f}{mark}"
            delta = f"{(run['avg'] or 0) - cfg['table_avg']:+.2f}"
        out.append(f"| {cfg['dirname']} | {expected} | {num(run['avg'], 3)} | {delta} |")
    if predicted:
        out += [
            "",
            "ᵖ not in the old table, either because it never swept this method that far or "
            "because the configuration did not exist then. The expected value comes from "
            "solving the hyperparameter offline against averages that were measured "
            "(`scripts/calibrate_ban_lambda.py` for Ban, `scripts/solve_diep_beta.py` for "
            "DiEP), so it is a prediction being tested here rather than a target that was "
            "already hit.",
        ]
    if unexpected:
        out += [
            "",
            "ⁿ run to isolate one hyperparameter rather than to hit an average, so there was "
            "nothing to predict: its average is simply what the setting produced. It used to "
            "share the ᵘ of the tables above, where that marker means the opposite kind of thing "
            "— a row whose budget is too low to compare — which is worth avoiding in a document "
            "read by searching for a superscript.",
        ]

    if args.lengths:
        tasks = ("aime24_avg", "lcb:codegeneration_v6", "math_500", "mmlu_pro")
        rows = []
        for cfg, run in done:
            for task, st in generation_lengths(
                    results_root / cfg["dirname"], tasks,
                    run.get("cap") or DEFAULT_CAP).items():
                rows.append((cfg["dirname"], task, st))
        if rows:
            out += [
                "",
                "## Generation length",
                "",
                "`at cap` is the share of generations reaching that run's own generation "
                "limit, which is given next to it because the most aggressive budgets ran "
                "under a shorter one. A truncated answer scores zero regardless of how good "
                "the routing was, so a high value here means the score is limited by rambling "
                "rather than by the pruning decision itself.",
                "",
                "| config | task | generations | median tokens | p90 | limit | at cap |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for name, task, st in sorted(rows, key=lambda r: (r[1], r[0])):
                out.append(f"| {name} | {task} | {st['n']} | {st['median']:.0f} | "
                           f"{st['p90']:.0f} | {st['cap'] // 1024}k | "
                           f"{st['at_cap']:.1f}% |")

    dest = results_root / "RESULTS.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {dest}  ({len(done)}/{len(configs)} configurations)")


if __name__ == "__main__":
    main()
