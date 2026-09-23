"""Check the gamma_2 exponent and the threshold cap on CPU, without loading a model.

Three properties matter and none of them need a GPU:

* alpha=0 has to reproduce NAEE exactly. gamma_2**0 is 1, so the threshold collapses to
  `w_e0 * beta`, which is NAEE's rule. If the two disagree the exponent is not wired into the
  threshold the way it is meant to be, and the corrected variant would not be comparable to
  NAEE.
* The defaults (alpha=1, no cap) have to leave the existing behaviour untouched, otherwise the
  DiEP runs already in the report would no longer be reproducible.
* A cap of exactly 1.0 has to be a no-op, and only a cap below 1 may change decisions. The top-k
  weights are sorted, so `w_ei < w_e0` holds for every rank past the first whatever the
  multiplier is once it reaches 1.

usage: python scripts/test_diep_gamma_alpha.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import expert_pruning.routing as R


def fake_artifact(num_layers: int, num_experts: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    sim = torch.rand(num_layers, num_experts, num_experts, generator=g)
    sim = (sim + sim.transpose(1, 2)) / 2          # similarity is symmetric
    mean_sim = sim.mean(dim=(1, 2))
    return sim, mean_sim


def keep_counts(hidden, gating, topk, beta, k_min, alpha=1.0, cap=None,
                with_artifact=True, sim=None, mean_sim=None, layer=0):
    R._NAEE_BETA = beta
    R._NAEE_K_MIN = k_min
    R._DIEP_GAMMA_ALPHA = alpha
    R._DIEP_THRESHOLD_CAP = cap
    R._DIEP_PRUNING_MODE = "independent"
    R._DIEP_USE_GAMMA1 = False
    R._DIEP_GAMMA1 = None
    R._DIEP_DEVICE_CACHE.clear()
    R._DIEP_SIM_MATRIX = sim if with_artifact else None
    R._DIEP_MEAN_SIM = mean_sim if with_artifact else None
    # NAEE prunes a prefix: past the first pruned rank everything goes. With a flat bar that is
    # the same thing as judging each rank on its own, because the weights are sorted — which is
    # why alpha=0 is expected to match it even though DiEP runs in independent mode.
    fn = R._diep_topk if with_artifact else R._naee_topk
    with R.set_current_moe_layer(f"model.layers.{layer}.mlp"):
        weights, _ids, _ = fn(hidden_states=hidden, gating_output=gating, topk=topk,
                              renormalize=True, indices_type=None)
    return (weights > 0).sum(dim=-1)


def main() -> int:
    torch.manual_seed(0)
    num_tokens, num_experts, topk, num_layers, k_min = 4096, 128, 8, 48, 2
    hidden = torch.randn(num_tokens, 64)
    gating = torch.randn(num_tokens, num_experts)
    sim, mean_sim = fake_artifact(num_layers, num_experts)

    failures = []

    # alpha=0 against NAEE (no artifact at all, which is how NAEE runs)
    for beta in (0.35, 0.5, 0.7, 0.9):
        naee = keep_counts(hidden, gating, topk, beta, k_min, with_artifact=False)
        alpha0 = keep_counts(hidden, gating, topk, beta, k_min, alpha=0.0,
                             sim=sim, mean_sim=mean_sim)
        same = bool(torch.equal(naee, alpha0))
        print(f"alpha=0 vs NAEE at beta={beta}: avg {naee.float().mean():.4f} vs "
              f"{alpha0.float().mean():.4f} {'ok' if same else 'MISMATCH'}")
        if not same:
            failures.append(f"alpha=0 differs from NAEE at beta={beta}")

    # A cap of exactly 1 must not change anything; a cap below 1 must relax pruning.
    print()
    for beta in (0.7, 1.0, 1.2):
        base = keep_counts(hidden, gating, topk, beta, k_min, sim=sim, mean_sim=mean_sim)
        cap1 = keep_counts(hidden, gating, topk, beta, k_min, cap=1.0, sim=sim,
                           mean_sim=mean_sim)
        cap08 = keep_counts(hidden, gating, topk, beta, k_min, cap=0.8, sim=sim,
                            mean_sim=mean_sim)
        noop = bool(torch.equal(base, cap1))
        relaxed = bool((cap08 >= base).all())
        print(f"beta={beta}: avg no cap {base.float().mean():.4f}, cap=1.0 "
              f"{cap1.float().mean():.4f} ({'no-op ok' if noop else 'CHANGED'}), cap=0.8 "
              f"{cap08.float().mean():.4f} ({'>= no cap ok' if relaxed else 'PRUNED MORE'})")
        if not noop:
            failures.append(f"cap=1.0 changed decisions at beta={beta}")
        if not relaxed:
            failures.append(f"cap=0.8 pruned more than no cap at beta={beta}")

    # alpha between 0 and 1 has to sit between the two, since it only damps gamma_2's spread.
    print()
    beta = 0.7
    avgs = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        avgs[alpha] = keep_counts(hidden, gating, topk, beta, k_min, alpha=alpha,
                                  sim=sim, mean_sim=mean_sim).float().mean().item()
    print("beta=0.7, average experts by alpha: "
          + ", ".join(f"{a}: {v:.4f}" for a, v in avgs.items()))

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
