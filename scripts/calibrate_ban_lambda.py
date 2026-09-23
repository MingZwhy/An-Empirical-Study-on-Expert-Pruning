"""Pick Ban's lambda for a target average expert count, without touching a GPU.

Ban selects

    k = round(k_min + (topk - k_min) * clamp(lambda * m, 0, 1)),
    m = 0.5 * (layer_sensitivity + token_sensitivity)

so lambda only scales a quantity that does not itself depend on lambda. layer_sensitivity
comes straight out of the Ban calibration artifact, which leaves the distribution of
token_sensitivity as the one unknown; this fits it against (lambda, k_min) -> average pairs
that have already been measured, then inverts the relation for the target.

Fitting rather than measuring is what makes this cheap: it needs no forward pass, so it can
run while every GPU is busy with the sweep. On Qwen3-30B-A3B-Instruct-2507 it reproduces the
six measured averages to within 0.04 experts under leave-one-out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq, minimize
from scipy.stats import beta as beta_dist

REPO = Path(__file__).resolve().parent.parent

# Averages the original sweep measured, from compact/<model>.md.
MEASURED: dict[str, list[tuple[float, int, float]]] = {
    "Qwen3-30B-A3B-Instruct-2507": [
        (0.50, 2, 3.1305),
        (0.60, 2, 3.5253),
        (0.60, 3, 4.1264),
        (0.70, 3, 4.4071),
        (0.90, 3, 4.9585),
        (0.95, 3, 5.0484),
    ],
}

# Quadrature grid over token_sensitivity, shared by every layer.
GRID = (np.arange(512) + 0.5) / 512


def average_experts(lam: float, k_min: int, topk: int, layer_sens: np.ndarray,
                    a: float, b: float) -> float:
    weights = beta_dist.pdf(GRID, a, b)
    weights /= weights.sum()
    m = 0.5 * (layer_sens[:, None] + GRID[None, :])
    k = np.clip(np.rint(k_min + (topk - k_min) * np.clip(lam * m, 0.0, 1.0)), k_min, topk)
    return float((k * weights[None, :]).sum(axis=1).mean())


def keep_distribution(lam: float, k_min: int, topk: int, layer_sens: np.ndarray,
                      a: float, b: float) -> dict[int, float]:
    weights = beta_dist.pdf(GRID, a, b)
    weights /= weights.sum()
    m = 0.5 * (layer_sens[:, None] + GRID[None, :])
    k = np.clip(np.rint(k_min + (topk - k_min) * np.clip(lam * m, 0.0, 1.0)),
                k_min, topk).astype(int)
    shares = {}
    for kk in range(1, topk + 1):
        share = float(((k == kk) * weights[None, :]).sum() / len(layer_sens))
        if share > 5e-4:
            shares[kk] = share
    return shares


def fit(observed, topk: int, layer_sens: np.ndarray) -> tuple[float, float]:
    def loss(params):
        a, b = np.exp(params)
        return sum((average_experts(lam, k, topk, layer_sens, a, b) - target) ** 2
                   for lam, k, target in observed)

    best = min((minimize(loss, np.log([a0, b0]), method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 4000})
                for a0 in (0.5, 1.0, 2.0, 5.0) for b0 in (0.5, 1.0, 2.0, 5.0)),
               key=lambda r: r.fun)
    return tuple(np.exp(best.x))


def solve_lambda(target: float, k_min: int, topk: int, layer_sens: np.ndarray,
                 a: float, b: float) -> float | None:
    f = lambda lam: average_experts(lam, k_min, topk, layer_sens, a, b) - target
    if f(1e-4) > 0 or f(1.0) < 0:
        return None
    return brentq(f, 1e-4, 1.0, xtol=1e-5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", default="Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--dataset", default="c4")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--targets", default="2.0,2.5,3.0")
    ap.add_argument("--k-mins", default="1,2,3")
    ap.add_argument("--loo", action="store_true",
                    help="also report leave-one-out error on the measured points")
    args = ap.parse_args()

    if args.model not in MEASURED:
        raise SystemExit(f"no measured (lambda, k_min) -> avg pairs recorded for {args.model}; "
                         f"add them to MEASURED from compact/{args.model}.md")
    observed = MEASURED[args.model]

    artifact = (REPO / "calib_utils" / "results" / args.model /
                f"{args.dataset}_ban.pt")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    layer_sens = np.clip(payload["layer_sensitivity"].double().numpy(), 0.0, 1.0)

    a, b = fit(observed, args.topk, layer_sens)
    print(f"layer_sensitivity over {len(layer_sens)} layers: {layer_sens.min():.3f} .. "
          f"{layer_sens.max():.3f}, mean {layer_sens.mean():.3f}, "
          f"{100 * (layer_sens < 0.1).mean():.0f}% below 0.1")
    print(f"fitted token_sensitivity ~ Beta({a:.3f}, {b:.3f}), mean {a / (a + b):.3f}\n")

    print("fit against the measured points")
    print(f"{'lambda':>7} {'k_min':>6} {'measured':>9} {'fitted':>8} {'err':>7}")
    for lam, k, target in observed:
        got = average_experts(lam, k, args.topk, layer_sens, a, b)
        print(f"{lam:>7.2f} {k:>6d} {target:>9.3f} {got:>8.3f} {got - target:>+7.3f}")

    if args.loo:
        print("\nleave-one-out")
        for i, (lam, k, target) in enumerate(observed):
            sub = [o for j, o in enumerate(observed) if j != i]
            aa, bb = fit(sub, args.topk, layer_sens)
            got = average_experts(lam, k, args.topk, layer_sens, aa, bb)
            print(f"{lam:>7.2f} {k:>6d} {target:>9.3f} {got:>8.3f} {got - target:>+7.3f}")

    k_mins = [int(v) for v in args.k_mins.split(",")]
    print("\nlambda for each target average")
    print(f"{'target':>7} {'k_min':>6} {'lambda':>8}  distribution of selected experts")
    for target in (float(v) for v in args.targets.split(",")):
        for k_min in k_mins:
            lam = solve_lambda(target, k_min, args.topk, layer_sens, a, b)
            if lam is None:
                floor = average_experts(1e-4, k_min, args.topk, layer_sens, a, b)
                print(f"{target:>7.1f} {k_min:>6d} {'—':>8}  unreachable, floor is {floor:.2f}")
                continue
            dist = keep_distribution(lam, k_min, args.topk, layer_sens, a, b)
            spread = "  ".join(f"k={kk}:{v * 100:4.1f}%" for kk, v in dist.items())
            print(f"{target:>7.1f} {k_min:>6d} {lam:>8.3f}  {spread}")


if __name__ == "__main__":
    main()
