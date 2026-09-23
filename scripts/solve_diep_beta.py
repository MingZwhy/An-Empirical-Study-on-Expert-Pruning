"""Solve DiEP's beta for a target average expert count, offline.

Trying betas on a GPU costs a lane per attempt, so this predicts the average instead, the way
scripts/calibrate_ban_lambda.py does for Ban's lambda.

How it works. All of these rules prune rank i when `w_ei < bar * w_e0`, and they differ only in
what sets `bar`:

    NAEE          bar = beta                       (flat, same for every rank)
    DiEP          bar = beta * gamma_2             (gamma_2 = sim(top1, e_i) / mean_sim)
    corrected     bar = min(beta * gamma_2**alpha, cap)

Write G(b) for the expected number of experts past k_min whose weight ratio w_ei/w_e0 clears a
flat bar b. Then, treating gamma_2 as independent of the token's weight profile:

    avg_NAEE(beta) = k_min + G(beta)
    avg_rule(beta) = k_min + E_gamma[ G(min(beta * gamma**alpha, cap)) ]

So the NAEE runs measure G directly, and the calibration artifact gives the distribution of
gamma_2. G is pinned at both ends without any fitting: weight ratios lie in (0, 1], so G(0) is
topk - k_min and G(b) is 0 for every b >= 1. A two-parameter Beta survival function is fitted to
the measured NAEE points between those ends.

The independence assumption is the weak part, so it is checked rather than trusted: the script
predicts the average for every DiEP run that has already been measured (alpha=1) and prints the
error. Those runs are not used in the fit, so agreement there is evidence the predictions for
other alphas are usable.

usage: python scripts/solve_diep_beta.py <model> [--targets 6,5,4,3,2] [--alpha 0.5] [--cap 0.8]
"""

import argparse
import glob
import json
import math
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]

MODELS = {
    "Qwen3-30B-A3B-Instruct-2507": {"topk": 8, "artifact": "c4_diep.pt"},
    "Ling-lite-1.5-2507": {"topk": 6, "artifact": "c4_diep.pt"},
    "Qwen3-Next-80B-A3B-Instruct": {"topk": 10, "artifact": "c4_diep.pt"},
    "gpt-oss-20b": {"topk": 4, "artifact": "c4_diep.pt"},
}


def gamma_distribution(model: str, bins: int = 4096):
    """Discretised distribution of gamma_2, weighted by how often experts are routed to.

    The artifact has no record of which pairs actually co-occur, so pair (a, b) is weighted by
    p(a) * p(b) from the calibration token counts. That is exact for the marginals and only
    approximates the correlation between the two slots.
    """
    path = REPO / "calib_utils/results" / model / MODELS[model]["artifact"]
    d = torch.load(path, map_location="cpu", weights_only=False)
    sim, mean_sim = d["sim_matrix"].float(), d["mean_sim"].float()
    counts = d["token_counts"].float()
    layers, experts, _ = sim.shape

    p = counts / counts.sum(dim=1, keepdim=True).clamp_min(1)
    g = (sim / mean_sim.view(layers, 1, 1).clamp_min(1e-6)).clamp_min(0.0)
    w = p.unsqueeze(2) * p.unsqueeze(1)
    w = w.masked_fill(torch.eye(experts, dtype=torch.bool).expand(layers, -1, -1), 0.0)
    w = w / w.sum(dim=(1, 2), keepdim=True) / layers

    return g.reshape(-1).double(), w.reshape(-1).double()


def measured_runs(model: str, method: str, k_min: int):
    """(parameter, measured average) for finished runs of one method at one k_min."""
    out = []
    for run in sorted((REPO / "results" / model).glob(f"{method}-kmin{k_min}-beta*")):
        found = sorted(glob.glob(str(run / "*" / "results" / "results_*.json")))
        if not found:
            continue
        ep = json.loads(Path(found[-1]).read_text()).get("expert_pruning") or {}
        avg = ep.get("average_selected_experts")
        if avg is None:
            continue
        # An `-avgprobe` run is a short run whose whole purpose is to read this average back, so it is
        # exactly the kind of point wanted here — only its directory name needs the suffix removed.
        beta = run.name.split("beta")[1].removesuffix("-avgprobe")
        out.append((float(beta), float(avg)))
    return sorted(out)


def beta_survival(a: float, b: float, x: torch.Tensor) -> torch.Tensor:
    """1 - BetaCDF(x; a, b), by the regularised incomplete beta function."""
    x = x.clamp(0.0, 1.0)
    dist = torch.distributions.Beta(torch.tensor(a, dtype=torch.float64),
                                    torch.tensor(b, dtype=torch.float64))
    # torch has no incomplete beta, so integrate the density on a fixed grid instead. The grid
    # is fine enough that the error is well below the run-to-run spread of a measured average.
    grid = torch.linspace(1e-6, 1 - 1e-6, 20001, dtype=torch.float64)
    pdf = dist.log_prob(grid).exp()
    cdf = torch.cumulative_trapezoid(pdf, grid)
    cdf = torch.cat([torch.zeros(1, dtype=torch.float64), cdf]) / cdf[-1]
    idx = torch.searchsorted(grid, x.flatten().clamp(grid[0], grid[-1]))
    return (1 - cdf[idx.clamp(max=len(cdf) - 1)]).reshape(x.shape)


def fit_g(points, topk: int, k_min: int):
    """Fit G(b) = (topk - k_min) * (1 - BetaCDF(b; a, b)) to the measured NAEE averages."""
    span = topk - k_min
    best, best_err = None, float("inf")
    for a in [round(0.1 * i, 2) for i in range(1, 121)]:
        for bb in [round(0.1 * i, 2) for i in range(1, 121)]:
            xs = torch.tensor([p for p, _ in points], dtype=torch.float64)
            pred = k_min + span * beta_survival(a, bb, xs)
            err = sum((float(pred[i]) - avg) ** 2 for i, (_, avg) in enumerate(points))
            if err < best_err:
                best, best_err = (a, bb), err
    return best, math.sqrt(best_err / max(1, len(points)))


def predict(beta: float, alpha: float, cap: float | None, fit, topk: int, k_min: int,
            gamma, weights) -> float:
    a, b = fit
    bar = beta * gamma.pow(alpha) if alpha != 1.0 else beta * gamma
    if cap is not None:
        bar = bar.clamp_max(cap)
    return k_min + (topk - k_min) * float((beta_survival(a, b, bar) * weights).sum())


def bias(residuals, avg: float, alpha: float) -> float:
    """Predicted minus measured, interpolated over the average axis and scaled by alpha.

    The independence assumption is the only reason the prediction is off, and its effect scales
    with how far gamma_2 moves the bar: at alpha=0 the bar is flat and the prediction reduces to
    the NAEE fit, which is exact by construction. Scaling the alpha=1 residual linearly is the
    simplest thing consistent with both ends.
    """
    if not residuals:
        return 0.0
    pts = sorted(residuals)                      # (predicted avg, predicted - measured)
    if avg <= pts[0][0]:
        r = pts[0][1]
    elif avg >= pts[-1][0]:
        r = pts[-1][1]
    else:
        r = 0.0
        for (x0, r0), (x1, r1) in zip(pts, pts[1:]):
            if x0 <= avg <= x1:
                t = (avg - x0) / (x1 - x0) if x1 > x0 else 0.0
                r = r0 + t * (r1 - r0)
                break
    return alpha * r


def solve(target: float, alpha: float, cap: float | None, fit, topk: int, k_min: int,
          gamma, weights, residuals=()) -> tuple[float, float]:
    """Beta whose corrected prediction hits the target, and that corrected prediction."""
    lo, hi = 1e-3, 20.0
    for _ in range(60):                        # average falls as beta rises
        mid = (lo + hi) / 2
        raw = predict(mid, alpha, cap, fit, topk, k_min, gamma, weights)
        if raw - bias(residuals, raw, alpha) > target:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2
    raw = predict(beta, alpha, cap, fit, topk, k_min, gamma, weights)
    return beta, raw - bias(residuals, raw, alpha)


def saturating_beta(alpha: float, cap: float | None, fit, topk: int, k_min: int, gamma, weights,
                    residuals, floor: float, tol: float = 0.02) -> float:
    """Smallest beta whose average is within `tol` of the floor: raising it further buys nothing."""
    lo, hi = 1e-3, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        raw = predict(mid, alpha, cap, fit, topk, k_min, gamma, weights)
        if raw - bias(residuals, raw, alpha) - floor > tol:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--targets", default="6,5,4,3,2")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=0.8)
    ap.add_argument("--k_min", type=int, default=2)
    args = ap.parse_args()

    topk = MODELS[args.model]["topk"]
    gamma, weights = gamma_distribution(args.model)
    naee = measured_runs(args.model, "NAEE", args.k_min)
    diep = measured_runs(args.model, "DiEP", args.k_min)
    if not naee:
        print(f"no finished NAEE runs at k_min={args.k_min} for {args.model}; "
              "G cannot be pinned yet")
        return 1

    print(f"=== {args.model} (top-{topk}, k_min={args.k_min}) ===")
    print("NAEE points used to pin G(b):")
    for beta, avg in naee:
        print(f"  beta={beta:<5} avg={avg:.3f}")
    fit, rms = fit_g(naee, topk, args.k_min)
    print(f"fitted Beta(a={fit[0]}, b={fit[1]}), rms on those points {rms:.3f}")

    residuals = []
    if diep:
        print("\ncheck against the DiEP runs already measured (alpha=1, no cap, not in the fit):")
        print(f"  {'beta':<6} {'measured':>9} {'predicted':>10} {'error':>7}")
        for beta, avg in diep:
            pred = predict(beta, 1.0, None, fit, topk, args.k_min, gamma, weights)
            residuals.append((pred, pred - avg))
            print(f"  {beta:<6} {avg:9.3f} {pred:10.3f} {pred - avg:+7.3f}")
        print("  those errors are corrected for below, scaled by alpha")

    floor = predict(20.0, args.alpha, args.cap, fit, topk, args.k_min, gamma, weights)
    floor -= bias(residuals, floor, args.alpha)
    print(f"\nbeta for each target with alpha={args.alpha}, cap={args.cap} "
          f"(the average bottoms out at {floor:.2f}):")
    print(f"  {'target':<7} {'beta':>7} {'expected avg':>13}")
    for t in [float(x) for x in args.targets.split(",")]:
        beta, got = solve(t, args.alpha, args.cap, fit, topk, args.k_min, gamma, weights,
                          residuals)
        note = ""
        if beta > 19:
            # Hitting the search ceiling does not mean there is no usable setting: with a cap the
            # average approaches a floor rather than reaching the target exactly, and if that floor
            # is near the target then the beta where it settles *is* the operating point. Reporting
            # only "unreachable" cost this project a row: the avg-2 tier of Qwen3-Next-80B was
            # written off on that word, and a probe later measured 2.075 at beta=1.2.
            beta = saturating_beta(args.alpha, args.cap, fit, topk, args.k_min, gamma, weights,
                                   residuals, floor)
            got = floor
            gap = abs(floor - t)
            note = ("  bottoms out here; within %.2f of the target, so this is the operating point"
                    % gap) if gap <= 0.15 else "  bottoms out %.2f above the target" % (floor - t)
        print(f"  {t:<7.1f} {beta:7.3f} {got:13.3f}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
