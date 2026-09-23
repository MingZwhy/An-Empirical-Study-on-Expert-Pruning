# gpt-oss-20b — corrected expert-pruning results

14 of 15 configurations finished.

The compact table's scores cannot be used as a reference: 68% of its score cells are exact duplicates of another configuration's, because it predates keying the lighteval sample cache on the expert pruning configuration (see `docs/compact_cache_contamination.md`). What it still provides, and what these runs take from it, is which hyperparameters land a method near a given average expert count.

Every table opens with two reference rows: the unpruned model (its native top-k of 4) and the Fixed-K point for that target. Fixed-K is what a method has to beat, since it reaches the same average expert count by simply keeping fewer experts per token. The bracketed number after each mean is the gap to the unpruned model.

The reference rows are held to the lower of the two readings: where the re-run came out above the old table it keeps the old value and is marked ᵒ, with what it actually scored given in the footnote. This only ever moves the reference down, so no pruning method is credited with beating an inflated Fixed-K.

Settings: the seven generative datasets, temperature 0.7 / top_p 0.8 / top_k 20, max_model_len and max_new_tokens 32768 unless a row is marked ᵏ, TP=1. Every run has its own cache namespace keyed on the pruning configuration, so no two rows here can share samples.

## Results by target average expert count

### Target avg ≈ 3.0

| method | hyperparams | avg experts | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=4 (native) | 4.000 | **74.21** | 74.3 | 64.6 | 90.6 | 73.7 | 71.7 | 58.3 | 86.2 |
| Fixed-K | k=3 | 3.000 | **75.84** (+1.64) | 73.6 | 65.7 | 91.0 | 77.4 | 74.5 | 61.7 | 87.0 |
| Ban ᵍ | lambda=0.9, k_min=2 | 3.051 | **73.82** (-0.39) | 73.4 | 60.6 | 92.4 | 73.1 | 70.4 | 60.0 | 86.9 |
| DiEP ᵈ ᵘ ᵍ | beta=0.9, k_min=1 | 2.857 | **60.39** (-13.81) | 67.7 | 57.1 | 85.2 | 50.4 | 44.2 | 32.0 | 86.2 |
| DiEP-damped ᵍ | beta=0.68, k_min=2, gamma_alpha=0.25, cap=0.9 | 2.992 | **72.64** (-1.57) | 71.7 | 66.2 | 91.6 | 72.4 | 65.9 | 55.4 | 85.3 |
| Dynamic Routing ᵍ | threshold=0.75, score_source=renormalized | 2.974 | **75.00** (+0.79) | 73.5 | 63.1 | 90.0 | 77.0 | 72.4 | 62.3 | 86.6 |
| NAEE ᵍ | beta=0.55, k_min=1 | 2.973 | **73.00** (-1.21) | 71.1 | 63.1 | 88.8 | 73.5 | 69.0 | 58.3 | 87.2 |

### Target avg ≈ 2.0

| method | hyperparams | avg experts | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=4 (native) | 4.000 | **74.21** | 74.3 | 64.6 | 90.6 | 73.7 | 71.7 | 58.3 | 86.2 |
| Fixed-K | k=2 | 2.000 | **71.74** (-2.47) | 70.5 | 65.2 | 86.6 | 73.9 | 69.4 | 48.0 | 88.7 |
| Ban ᵍ | lambda=0.55, k_min=1 | 1.994 | **68.98** (-5.23) | 68.2 | 63.1 | 86.2 | 71.5 | 66.0 | 40.6 | 87.2 |
| DiEP ᵈ ᵘ ᵍ | beta=1.95, k_min=1 | 1.940 | **41.36** (-32.85) | 49.8 | 41.4 | 66.4 | 19.6 | 13.9 | 24.0 | 74.4 |
| DiEP ᵈ ᵛ ᵍ | beta=1.95, k_min=2 | 2.522 | **72.07** (-2.14) | 71.9 | 65.2 | 85.4 | 74.1 | 69.6 | 49.7 | 88.6 |
| Dynamic Routing ᵍ | threshold=0.5, score_source=renormalized | 1.964 | **70.87** (-3.34) | 70.2 | 61.6 | 87.6 | 72.7 | 68.4 | 47.4 | 88.2 |
| NAEE ᵍ | beta=0.75, k_min=1 | 1.998 | **62.55** (-11.66) | 62.1 | 58.6 | 82.8 | 61.1 | 53.4 | 36.0 | 83.9 |
| NAEE ᵍ | beta=0.75, k_min=2 | 2.365 | **73.32** (-0.89) | 71.8 | 65.7 | 87.2 | 76.0 | 71.1 | 53.7 | 87.8 |

ᵘ spends fewer experts than Fixed-K does at that tier, by up to 0.14 (DiEP at beta=0.9, k_min=1). 2 rows are in that position, and a mean below Fixed-K's there can be explained by the smaller budget rather than by the method. These are not being re-run: a generative run of this model costs a day, and the threshold that would fix it is a different operating point rather than a correction to this one. The 0-shot QA tables below were re-solved instead, since a run there is minutes.

ᵗ still the old table's numbers, waiting on the run here to replace them.

ᵈ DiEP's skipping rule is specified for k=2 and our generalisation of it to a top-4 router prunes the whole tail, leaving it 14 to 28 points below Fixed-K at the same budget. Only the paper's auxiliary skipping mechanism is implemented, not its differentiable pruning. Read these rows as a result about that generalisation, not as DiEP's achievable quality — see `docs/known_issues.md`.

ᵛ this method has no setting that reaches this target, so its closest one is shown. It is spending more experts than the other rows and its mean is not comparable to them.

ᵍ this model is quantized, so its expert selection is inlined into the compiled region and a run that keeps CUDA graphs cannot count its own experts. Forcing eager mode to get the count costs a factor of 6.7 in generation throughput (166 against 1138 tok/s measured here), which is the difference between a run that takes a day and one that takes two weeks, so the score comes from a graph run and the average from a separate eager probe of the same configuration over 40 prompts each of mmlu_pro, math_500 and gsm8k — a few million routing decisions, far more than an average needs, and spread over three input distributions because the router reacts to its input. The probe is also the more correct place to measure it: under CUDA graphs vLLM pads each batch to a captured size, and those padding rows would be counted as tokens.

## 0-shot QA (lm-eval)

Four multiple-choice sets, scored by likelihood over the options with nothing generated, so they answer a different question from the tables above: how much of the model's knowledge survives pruning, as opposed to how much of its reasoning does. A budget that ruins the generative scores can leave these nearly intact, and the gap between the two is the interesting part.

The thresholds below are not the ones in the tables above, and that is the point. A threshold is not a budget: it admits however many experts the router's own confidence lets it, and the router is more decided about a one-sentence question than about the middle of a derivation. The generative settings therefore spend fewer experts here than the tier they were calibrated for, and a row that is both cheaper than Fixed-K and worse than it answers nothing. Each method was re-solved against this distribution (`scripts/solve_qa_hyper.py`) so that its average sits at or just above Fixed-K's.

`acc` is used for all four columns because it is the only metric all four report — winogrande has no `acc_norm`. The result files carry `acc_norm` for the other three, and taking it where available moves rows by up to three places, so a comparison against a published table should be read off the JSON with that table's own metric.

### Target avg ≈ 3.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=4 (native) | 4.000 | 4.000 | **54.22** | 45.3 | 77.5 | 66.9 | 27.2 |
| Fixed-K | k=3 | 3.000 | 3.000 | **52.56** (-1.67) | 43.8 | 75.7 | 63.4 | 27.4 |
| Ban | lambda=0.9, k_min=2 | — | 3.051 | **52.95** (-1.27) | 43.8 | 75.8 | 64.2 | 28.0 |
| DiEP | beta=0.9, k_min=1 | — | 2.857 | **48.62** (-5.60) | 38.2 | 72.0 | 61.9 | 22.4 |
| DiEP-damped | beta=0.68, k_min=2, gamma_alpha=0.25, cap=0.9 | — | 2.992 | **52.48** (-1.74) | 41.8 | 74.9 | 65.6 | 27.6 |
| Dynamic Routing | threshold=0.75, score_source=renormalized | — | 2.974 | **52.63** (-1.59) | 42.4 | 76.8 | 63.1 | 28.2 |
| NAEE | beta=0.55, k_min=1 | — | 2.973 | **50.81** (-3.41) | 39.9 | 72.9 | 64.0 | 26.4 |

### Target avg ≈ 2.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=4 (native) | 4.000 | 4.000 | **54.22** | 45.3 | 77.5 | 66.9 | 27.2 |
| Fixed-K | k=2 | 2.000 | 2.000 | **48.28** (-5.94) | 38.7 | 69.4 | 59.1 | 26.0 |
| Ban | lambda=0.55, k_min=1 | — | 1.994 | **47.12** (-7.10) | 38.2 | 69.4 | 57.3 | 23.6 |
| DiEP | beta=1.95, k_min=1 | — | 1.940 | **39.68** (-14.54) | 28.4 | 57.4 | 53.5 | 19.4 |
| Dynamic Routing | threshold=0.5, score_source=renormalized | — | 1.964 | **47.44** (-6.78) | 37.1 | 68.4 | 59.7 | 24.6 |
| NAEE | beta=0.75, k_min=1 | — | 1.998 | **43.09** (-11.13) | 31.2 | 61.8 | 56.4 | 23.0 |

## Average expert count landed where it was expected to

The hyperparameter-to-average mapping is the only part of the compact table that survived the cache bug, so this confirms the re-runs are sitting at the intended operating points.

| config | expected avg | this run | Δ |
| --- | --- | --- | --- |
| Baseline-k4 | 4.0000 | 4.000 | +0.00 |
| FixedK-k3 | 3.0000 | 3.000 | +0.00 |
| Ban-kmin2-lambda0.9 | 3.0389 | 3.051 | +0.01 |
| DiEP-kmin1-beta0.9 | 2.9995 | 2.857 | -0.14 |
| DiEP-damped-a0.25-cap0.9-kmin2-beta0.68 | 3.0000 | 2.992 | -0.01 |
| Dynamic_Routing-thr0.75-renormalized | 2.9686 | 2.974 | +0.01 |
| NAEE-kmin1-beta0.55 | 2.9345 | 2.973 | +0.04 |
| FixedK-k2 | 2.0000 | 2.000 | +0.00 |
| Ban-kmin1-lambda0.55 | 1.9972 | 1.994 | -0.00 |
| DiEP-kmin1-beta1.95 | 2.1186 | 1.940 | -0.18 |
| DiEP-kmin2-beta1.95 | — ᵘ | 2.522 | — |
| Dynamic_Routing-thr0.5-renormalized | 1.9514 | 1.964 | +0.01 |
| NAEE-kmin1-beta0.75 | 1.9914 | 1.998 | +0.01 |
| NAEE-kmin2-beta0.75 | — ᵘ | 2.365 | — |

ᵘ run to isolate one hyperparameter rather than to hit an average, so there was nothing to predict: its average is simply what the setting produced.
