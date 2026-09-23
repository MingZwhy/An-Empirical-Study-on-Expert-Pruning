#!/usr/bin/env python3
"""Where does the paper's own gamma land on our average-experts axis?

DiEP's skipping rule (section 4.3, eq. 12) has no free hyperparameter: gamma = gamma_1 *
gamma_2 is fixed by calibration. On a top-2 router it can therefore remove at most one expert
per token, which is the regime the paper validates — 1.04x to 1.07x speedup for 0.5 to 1
accuracy point. This estimates what the same rule does to a fine-grained router, where it is
free to remove most of the activated experts.

Estimate only, to decide whether the configuration is worth a lane. Rank i survives when
w_ei/w_e0 >= gamma_1 * gamma_2(e0, ei). The per-rank weight ratios are not in the artifact,
but gamma_1 is by definition the median of w_e1/w_e0, so taking the ratios to decay
geometrically at that rate turns the test into gamma_2 <= gamma_1^(i-1).

    python scripts/estimate_diep_paper_point.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
NATIVE_K = {"Qwen3-30B-A3B-Instruct-2507": 8, "Ling-lite-1.5-2507": 6,
            "Qwen3-Next-80B-A3B-Instruct": 10, "gpt-oss-20b": 4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=sorted(NATIVE_K))
    ap.add_argument("--k-min", type=int, nargs="*", default=[1, 2])
    args = ap.parse_args()

    for model in args.models:
        path = REPO / "calib_utils" / "results" / model / "c4_diep.pt"
        if not path.is_file():
            print(f"\n{model}: no DiEP artifact")
            continue
        payload = torch.load(path, map_location="cpu")
        sim = payload["sim_matrix"].float()
        mean_sim = payload["mean_sim"].float()
        gamma1 = payload["gamma_1"].float()
        num_layers, num_experts, _ = sim.shape
        topk = NATIVE_K[model]
        off_diagonal = ~torch.eye(num_experts, dtype=torch.bool)

        gamma2 = [(sim[l] / mean_sim[l].clamp_min(1e-6))[off_diagonal]
                  for l in range(num_layers)]
        unconditional = sum(float((gamma1[l] * gamma2[l] >= 1).float().mean())
                            for l in range(num_layers)) / num_layers

        print(f"\n{model}  (top-{topk}, {num_layers} layers, {num_experts} experts)")
        print(f"  gamma_1 mean {gamma1.mean():.4f}, range {gamma1.min():.4f} to "
              f"{gamma1.max():.4f}")
        print("  expert pairs dropped regardless of gate weight, at the paper's gamma: "
              f"{unconditional * 100:.1f}%")
        for k_min in args.k_min:
            per_layer = torch.tensor([
                k_min + sum(float((gamma2[l] <= gamma1[l] ** (rank - 1)).float().mean())
                            for rank in range(k_min, topk))
                for l in range(num_layers)])
            print(f"  k_min={k_min}: estimated average {per_layer.mean():.2f} experts "
                  f"of {topk} (per-layer {per_layer.min():.2f} to {per_layer.max():.2f})")


if __name__ == "__main__":
    main()
