#!/usr/bin/env python3
"""Re-solve each method's threshold so its 0-shot QA average is not below Fixed-K's.

The thresholds in the generative tables were calibrated on the generative prompts, and a threshold is
not a budget: it is a rule about the router's own output, so the number of experts it admits depends on
what the model is reading. QA prompts are short multiple-choice questions, the router is more decided
about them than about the middle of a derivation, and every threshold method therefore spends fewer
experts on QA than the tier it was calibrated for -- up to 1.15 fewer on Qwen3-Next-80B. That makes the
QA table an unfair comparison in the direction that flatters nobody: a row scoring below Fixed-K might
simply have been given less to work with.

So the QA table needs its own calibration. This solves for it from the QA averages already measured:
one monotone curve per method from its settings across the tiers, inverted at each tier's k plus a
small margin, since spending slightly more than Fixed-K is honest and spending less is not.

    python scripts/solve_qa_hyper.py                     # proposals for every model
    python scripts/solve_qa_hyper.py --model Ling-lite-1.5-2507
    python scripts/solve_qa_hyper.py --json /tmp/qa.json # for queue_qa_recal.py

Where a curve has to be extrapolated past the end of its measured range the prediction is a guess with
a slope attached, and where the knob runs into its own bound (Ban's lambda cannot exceed 1) no setting
reaches the tier at that k_min and the row is reported as needing a higher floor instead. Both cases
are marked; both are settled by running the thing, which takes minutes per configuration.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_results as rr    # noqa: E402  (same directory, shares the readers and the model table)

# knob name, the flag that carries it, and whether raising it admits more experts.
KNOBS = {
    "Ban": ("lambda", "--ban_lambda", True, (0.0, 1.0)),
    "NAEE": ("beta", "--naee_beta", False, (0.01, 8.0)),
    "DiEP": ("beta", "--naee_beta", False, (0.01, 8.0)),
    "DiEP-damped": ("beta", "--naee_beta", False, (0.01, 8.0)),
    "Dynamic Routing": ("threshold", "--dynamic_routing_threshold", True, (0.0, 1.0)),
}
# DiEP-paper reproduces a published operating point rather than hitting a budget, and Fixed-K and the
# unpruned model have no threshold to solve for.
SKIP_METHODS = {"DiEP-paper", "Fixed-K", "Baseline", "MC-MoE"}


def knob_of(cfg: dict) -> float | None:
    """The value of this configuration's threshold, read off its flags."""
    spec = KNOBS.get(cfg["method"])
    if spec is None:
        return None
    flag = spec[1]
    flags = cfg.get("flags", [])
    if flag not in flags:
        return None
    return float(flags[flags.index(flag) + 1])


def group_of(cfg: dict) -> tuple:
    """Configurations whose averages lie on one curve.

    The damping parameters change the shape of the threshold and k_min changes where the floor is, so
    an alpha=0.25 setting says nothing about where an alpha=0.5 one lands and a k_min=1 setting says
    nothing about a k_min=2 one -- Ling's DiEP has beta=0.8 twice, at 1.900 and 2.391, differing only
    in the floor.
    """
    hyper = cfg["hyper"]
    alpha = re.search(r"(?:gamma_)?alpha=([\d.]+)", hyper)
    cap = re.search(r"cap=([\d.]+)", hyper)
    k_min = re.search(r"k_min=(\d+)", hyper)
    return variant_of(cfg) + (k_min.group(1) if k_min else None,)


def variant_of(cfg: dict) -> tuple:
    """One method's distinct operating points, floors aside.

    DiEP-damped at alpha=0.5 and at alpha=0.25 are two different rules sharing a name, and both are in
    the tables on purpose -- the second is the damped fix, the first is what it is compared against. So
    they are re-solved separately and neither displaces the other. k_min is excluded because the floor
    is one of the things re-solving is allowed to move.
    """
    hyper = cfg["hyper"]
    alpha = re.search(r"(?:gamma_)?alpha=([\d.]+)", hyper)
    cap = re.search(r"cap=([\d.]+)", hyper)
    return (cfg["method"], alpha.group(1) if alpha else None, cap.group(1) if cap else None)


def slope_near(points: list[tuple[float, float]], avg: float,
               want: float | None = None) -> float | None:
    """d(average)/d(knob) over the stretch this correction has to cross.

    Taking the segment nearest the current average is the wrong reading when the correction is upward
    and the nearest segment lies below it: these curves steepen as the threshold falls, so a slope
    borrowed from the flat end sends beta far past where it needs to go -- 0.636 to 0.314 for four
    tenths of an expert, in the case that made this argument necessary. The segment that brackets the
    target is the one that describes getting there.
    """
    pts = sorted(points, key=lambda p: p[0])
    if len(pts) < 2:
        return None
    spans = [(min(pts[i][1], pts[i + 1][1]), max(pts[i][1], pts[i + 1][1]), i)
             for i in range(len(pts) - 1)]
    aim = avg if want is None else want
    bracketing = [s for s in spans if s[0] <= aim <= s[1]]
    if bracketing:
        best = bracketing[0][2]
    else:
        best = min(spans, key=lambda s: min(abs(s[0] - aim), abs(s[1] - aim)))[2]
    (k0, a0), (k1, a1) = pts[best], pts[best + 1]
    return None if k1 == k0 or a1 == a0 else (a1 - a0) / (k1 - k0)


def solve_ban(avg: float, lam: float, k_min: int, topk: int,
              want: float) -> tuple[float, int] | None:
    """Ban's own rule, inverted: the (lambda, k_min) that spends `want` experts per token.

    Ban does not threshold a weight, it picks a count: k = k_min + (topk - k_min) * clamp(lambda * s),
    with s the mean of a layer and a token sensitivity. So one measured average pins down the mean
    sensitivity this model shows on these questions, and every other setting follows in closed form --
    which matters here because lambda is bounded at 1, and above some tier no lambda reaches the budget
    at the floor the generative row used. The floor is then what has to move, and this finds the lowest
    one that works rather than guessing at it.

    Returns None when even k_min = topk - 1 cannot get there, i.e. when nothing short of fixed top-k
    spends that much.
    """
    span = topk - k_min
    if span <= 0 or lam <= 0:
        return None
    s_bar = (avg - k_min) / (span * lam)
    if s_bar <= 0:
        return None
    for floor in range(k_min, topk):
        lam_new = (want - floor) / ((topk - floor) * s_bar)
        if 0 < lam_new <= 1.0:
            return lam_new, floor
    return None


def solve(anchor: tuple[float, float], want: float, slope: float) -> float:
    """The knob that should reach `want`, stepped from a configuration's own measurement.

    Anchoring on the row's own average rather than on a fit through the whole curve keeps the one
    reliable number in play: this configuration was measured, on these questions, and the correction
    needed is usually a fraction of an expert. The slope comes from neighbouring settings and the
    curves are not straight, so a large correction is a first guess -- which is what the re-run
    measures.
    """
    knob, avg = anchor
    return knob + (want - avg) / slope


def collect(model: str) -> tuple[dict, dict]:
    """{group: [(knob, qa_avg)]} and {(method, tier): [(config, qa_avg)]} for one model."""
    spec = rr.MODELS[model]
    cfgs = [c for c in json.loads(Path(spec["configs"]).read_text())
            if not c.get("skipped") and not c.get("probe_for")
            and c["method"] not in SKIP_METHODS]
    root = rr.REPO / "results" / model
    curves: dict[tuple, list[tuple[float, float]]] = {}
    tiers: dict[tuple[str, float], list[tuple[dict, float | None]]] = {}
    for cfg in cfgs:
        knob = knob_of(cfg)
        if knob is None:
            continue
        avg = (rr.read_qa(root / cfg["dirname"]) or {}).get("avg")
        if avg is not None:
            curves.setdefault(group_of(cfg), []).append((knob, avg))
        for t in cfg["targets"]:
            tiers.setdefault((variant_of(cfg), float(t)), []).append((cfg, avg))
    return curves, tiers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", help="default: all four")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="aim this far above the tier, so noise cannot put a row under it")
    ap.add_argument("--over-tolerance", type=float, default=0.35,
                    help="how far above the tier a row may sit before it is re-solved downwards")
    ap.add_argument("--json", type=Path, help="write the proposals here for the queue script")
    args = ap.parse_args()

    models = args.model or list(rr.MODELS)
    out = []
    for model in models:
        curves, tiers = collect(model)
        if not curves:
            print(f"\n=== {model}: no QA averages measured yet, nothing to solve from")
            continue
        print(f"\n=== {model}")
        print(f"  {'tier':>4}  {'method':<16} {'now':>20} {'QA avg':>7}   {'proposed':>20}  note")
        for (variant, tier), rows in sorted(tiers.items(),
                                           key=lambda kv: (-kv[0][1], kv[0][0])):
            method = variant[0]
            label = method + (f" a{variant[1]}" if variant[1] else "")
            name, flag, rising, (lo, hi) = KNOBS[method]
            measured = [(c, a) for c, a in rows if a is not None]
            if not measured:
                print(f"  {tier:>4.0f}  {label:<19} {'':>20} {'—':>7}   {'(wait)':>20}  "
                      "its QA run has not landed yet")
                continue
            # A tier can hold two configurations of one method, e.g. a k_min=1 and a k_min=2 variant.
            # If either already spends at least what Fixed-K does, the tier is satisfied by that one.
            over = [(c, a) for c, a in measured if a >= tier]
            if over:
                cfg, avg = min(over, key=lambda ca: ca[1])
                # Slack in the other direction is not a broken comparison, but it is not free either:
                # a row that spends half an expert more than the column asks for is being credited for
                # experts Fixed-K did not get. Worth one more run when the setting was solved for this
                # tier; not worth it when the row is a higher tier's setting shown here because the
                # method has nothing that reaches this one, which the tables mark separately.
                slack = avg - tier
                if slack <= args.over_tolerance or len(cfg.get("targets", [])) > 1:
                    print(f"  {tier:>4.0f}  {label:<19} {name+'='+format(knob_of(cfg), 'g'):>20} "
                          f"{avg:7.3f}   {'(keep)':>20}  "
                          + ("at or above Fixed-K already" if slack <= args.over_tolerance else
                             f"{slack:+.2f} over, but this is a higher tier's setting"))
                    continue
                measured = [(cfg, avg)]      # tighten from the row that overshot
            cfg, avg_now = max(measured, key=lambda ca: ca[1])
            knob_now = knob_of(cfg)
            group = group_of(cfg)
            note = []
            k_min = re.search(r"k_min=(\d+)", cfg["hyper"])
            k_min_now = int(k_min.group(1)) if k_min else None
            want = tier + args.margin

            if method == "Ban" and k_min_now is not None:
                got = solve_ban(avg_now, knob_now, k_min_now, rr.MODELS[model]["native_k"], want)
                if got is None:
                    print(f"  {tier:>4.0f}  {label:<19} {'lambda='+format(knob_now, 'g'):>20} "
                          f"{avg_now:7.3f}   {'(unreachable)':>20}  no lambda below its bound of 1 "
                          "reaches this tier at any floor short of fixed top-k")
                    continue
                lam_new, floor = got
                rec = {
                    "model": model, "method": method, "tier": tier,
                    "from_dirname": cfg["dirname"], "knob": "lambda", "flag": "--ban_lambda",
                    "value_now": knob_now, "qa_avg_now": round(avg_now, 4),
                    "value_new": round(lam_new, 4), "k_min_now": k_min_now,
                    "k_min_new": floor if floor != k_min_now else None,
                    "note": ("solved from Ban's own rule"
                             + (f"; lambda alone cannot reach it, floor {k_min_now}->{floor}"
                                if floor != k_min_now else "")),
                }
                out.append(rec)
                shown = f"lambda={lam_new:.3g}" + (f", k_min={floor}" if floor != k_min_now else "")
                print(f"  {tier:>4.0f}  {label:<19} {'lambda='+format(knob_now, 'g'):>20} "
                      f"{avg_now:7.3f}   {shown:>20}  {rec['note']}")
                continue

            slope = slope_near(curves.get(group, []), avg_now, want)
            if slope is None:
                # One measured setting at this k_min, so borrow the sensitivity from the same method's
                # other floors. The floor shifts the curve; it does not much change how fast the
                # average moves with the threshold well above it.
                others = [s for g, pts in curves.items() if g[0] == method and g != group
                          and (s := slope_near(pts, avg_now, want)) is not None]
                if not others:
                    print(f"  {tier:>4.0f}  {label:<19} {name+'='+format(knob_now, 'g'):>20} "
                          f"{avg_now:7.3f}   {'(manual)':>20}  no neighbouring setting to take a "
                          "slope from")
                    continue
                slope = sum(others) / len(others)
                note.append("slope borrowed from another k_min")
            solved = solve((knob_now, avg_now), want, slope)
            if want - avg_now > 0.6:
                note.append(f"{want - avg_now:+.2f} experts is a long step, expect one correction")
            clamped = min(max(solved, lo), hi)
            k_min_new = None
            if abs(clamped - solved) > 1e-9:
                # The threshold cannot go where the tier is: Ban's lambda is already near its bound of
                # 1, where it prunes least. The floor has to rise instead, since k_min forces every
                # token to keep at least that many experts.
                k_min_new = max(int(tier) - 1, (k_min_now or 1) + 1)
                note.append(f"{name} bound {clamped:g} reached, so k_min {k_min_now}->{k_min_new}")
            rec = {
                "model": model, "method": method, "tier": tier,
                "from_dirname": cfg["dirname"], "knob": name, "flag": flag,
                "value_now": knob_now, "qa_avg_now": round(avg_now, 4),
                "value_new": round(clamped, 4),
                "k_min_now": k_min_now, "k_min_new": k_min_new,
                "note": "; ".join(note),
            }
            out.append(rec)
            shown = f"{name}={clamped:g}" + (f", k_min={k_min_new}" if k_min_new else "")
            print(f"  {tier:>4.0f}  {label:<19} {name+'='+format(knob_now, 'g'):>20} "
                  f"{avg_now:7.3f}   {shown:>20}  {rec['note']}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}  ({len(out)} configurations to re-run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
