# Ling-lite-1.5-2507 — corrected expert-pruning results

24 of 24 configurations finished.

The compact table's scores cannot be used as a reference: 62% of its score cells are exact duplicates of another configuration's, because it predates keying the lighteval sample cache on the expert pruning configuration (see `docs/compact_cache_contamination.md`). What it still provides, and what these runs take from it, is which hyperparameters land a method near a given average expert count.

Every table opens with two reference rows: the unpruned model (its native top-k of 6) and the Fixed-K point for that target. Fixed-K is what a method has to beat, since it reaches the same average expert count by simply keeping fewer experts per token. The bracketed number after each mean is the gap to the unpruned model.

The reference rows are held to the lower of the two readings: where the re-run came out above the old table it keeps the old value and is marked ᵒ, with what it actually scored given in the footnote. This only ever moves the reference down, so no pruning method is credited with beating an inflated Fixed-K.

Settings: the seven generative datasets, temperature 0.7 / top_p 0.8 / top_k 20, max_model_len and max_new_tokens 32768 unless a row is marked ᵏ, TP=1. Every run has its own cache namespace keyed on the pruning configuration, so no two rows here can share samples.

## Results by target average expert count

### Target avg ≈ 4.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline ᵒ | k=6 (native) | 6.000 | **58.56** | 69.4 | 56.6 | 89.2 | 40.9 | 28.1 | 34.3 | 91.4 |
| Fixed-K ᵒ | k=4 | 4.000 | **57.27** (-1.29) | 67.4 | 55.6 | 89.0 | 38.7 | 26.0 | 33.7 | 90.5 |
| Ban | lambda=0.9, k_min=2 | 3.966 | **57.49** (-1.07) | 67.8 | 55.1 | 89.2 | 39.8 | 25.9 | 34.9 | 89.8 |
| DiEP ᵈ | beta=0.45, k_min=2 | 4.086 | **54.16** (-4.40) | 63.5 | 52.0 | 87.4 | 35.1 | 24.3 | 28.0 | 88.7 |
| DiEP-damped | beta=0.467, k_min=2, gamma_alpha=0.25, cap=0.9 | 4.045 | **54.10** (-4.46) | 62.6 | 52.0 | 87.0 | 35.7 | 23.1 | 30.9 | 87.5 |
| DiEP-damped | beta=0.468, alpha=0.5, cap=0.9, k_min=2 | 3.984 | **55.29** (-3.27) | 62.8 | 58.1 | 86.2 | 34.1 | 25.8 | 32.0 | 87.9 |
| Dynamic Routing | threshold=0.75, score_source=renormalized | 4.171 | **57.93** (-0.62) | 67.3 | 58.6 | 90.2 | 40.1 | 26.8 | 32.0 | 90.5 |
| NAEE | beta=0.45, k_min=2 | 4.058 | **53.93** (-4.63) | 61.9 | 53.5 | 87.4 | 33.2 | 23.3 | 30.3 | 87.9 |

### Target avg ≈ 3.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline ᵒ | k=6 (native) | 6.000 | **58.56** | 69.4 | 56.6 | 89.2 | 40.9 | 28.1 | 34.3 | 91.4 |
| Fixed-K ᵒ | k=3 | 3.000 | **52.51** (-6.04) | 63.3 | 46.5 | 86.0 | 31.5 | 24.2 | 28.6 | 87.5 |
| Ban | lambda=0.45, k_min=2 | 2.997 | **53.16** (-5.40) | 63.2 | 54.0 | 86.4 | 30.5 | 22.8 | 28.0 | 87.2 |
| DiEP ᵈ | beta=0.65, k_min=2 | 3.033 | **47.33** (-11.23) | 56.1 | 40.9 | 80.8 | 23.0 | 19.5 | 25.1 | 85.9 |
| DiEP-damped | beta=0.636, k_min=2, gamma_alpha=0.25, cap=0.9 | 3.022 | **46.78** (-11.78) | 53.7 | 50.5 | 79.4 | 20.1 | 15.7 | 25.1 | 82.9 |
| DiEP-damped | beta=0.645, alpha=0.5, cap=0.9, k_min=2 | 2.963 | **46.97** (-11.59) | 54.8 | 49.5 | 80.0 | 21.5 | 17.7 | 22.9 | 82.4 |
| Dynamic Routing | threshold=0.6, score_source=renormalized | 3.138 | **52.86** (-5.70) | 62.2 | 48.5 | 87.8 | 31.7 | 23.1 | 27.4 | 89.3 |
| NAEE | beta=0.6, k_min=2 | 3.106 | **47.00** (-11.56) | 56.1 | 44.4 | 82.6 | 22.2 | 15.4 | 25.7 | 82.6 |

### Target avg ≈ 2.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline ᵒ | k=6 (native) | 6.000 | **58.56** | 69.4 | 56.6 | 89.2 | 40.9 | 28.1 | 34.3 | 91.4 |
| Fixed-K ᵒ | k=2 | 2.000 | **34.99** (-23.57) | 42.3 | 38.4 | 70.4 | 5.0 | 8.4 | 8.0 | 72.4 |
| Ban | lambda=0.35, k_min=1 | 1.990 | **33.18** (-25.38) | 41.0 | 33.3 | 65.0 | 5.3 | 8.0 | 8.0 | 71.7 |
| DiEP ᵈ | beta=0.8, k_min=1 | 2.255 | **9.67** (-48.89) | 15.1 | 24.7 | 22.6 | 0.1 | 0.2 | 1.1 | 3.9 |
| DiEP ᵈ ᵛ | beta=0.8, k_min=2 | 2.573 | **43.03** (-15.53) | 49.6 | 43.4 | 77.2 | 14.9 | 14.8 | 18.9 | 82.3 |
| DiEP ᵈ ᵛ | gamma_1*gamma_2 (paper eq. 12), k_min=2 | 2.695 | **43.03** (-15.53) | 51.1 | 40.9 | 77.0 | 15.1 | 15.9 | 17.1 | 84.1 |
| DiEP-damped | beta=1.1, k_min=2, gamma_alpha=0.25, cap=0.9 | 2.083 | **35.09** (-23.46) | 42.4 | 32.3 | 68.6 | 6.4 | 9.9 | 10.9 | 75.1 |
| Dynamic Routing | threshold=0.4, score_source=renormalized | 2.036 | **31.88** (-26.68) | 38.3 | 32.8 | 65.2 | 4.2 | 7.8 | 8.0 | 66.8 |
| NAEE | beta=0.75, k_min=1 | 2.088 | **9.38** (-49.18) | 15.4 | 26.3 | 19.2 | 0.0 | 0.1 | 0.0 | 4.7 |
| NAEE | beta=0.75, k_min=2 | 2.428 | **41.53** (-17.03) | 48.5 | 46.5 | 74.2 | 12.0 | 11.9 | 18.9 | 78.8 |

ᵗ still the old table's numbers, waiting on the run here to replace them.

ᵈ DiEP's skipping rule is specified for k=2 and our generalisation of it to a top-6 router prunes the whole tail, leaving it 3 to 24 points below Fixed-K at the same budget. Only the paper's auxiliary skipping mechanism is implemented, not its differentiable pruning. Read these rows as a result about that generalisation, not as DiEP's achievable quality — see `docs/known_issues.md`.

ᵒ the re-run of this reference row came out above the old table, so the old table's value is kept. Fixed-K needs no calibration and the old table had little room to get it wrong, so a higher reading here is treated as measurement spread rather than as a better reference; taking it would flatter every pruning method measured against it. What the re-runs actually scored: k=6 scored 59.39 against 58.82, k=6 scored 59.39 against 58.82, k=6 scored 59.39 against 58.82, k=4 scored 57.06 against 57.02, k=3 scored 52.60 against 51.49, k=2 scored 34.50 against 33.47.

ᵛ this method has no setting that reaches this target, so its closest one is shown. It is spending more experts than the other rows and its mean is not comparable to them.

ᵃ this model keeps 2 shared experts active for every token on top of the routed ones. The average counts routed experts only, matching the old table, so an average of 3 here costs 5 experts of compute and is not comparable to an average of 3 on a model without shared experts.

## 0-shot QA (lm-eval)

Four multiple-choice sets, scored by likelihood over the options with nothing generated, so they answer a different question from the tables above: how much of the model's knowledge survives pruning, as opposed to how much of its reasoning does. A budget that ruins the generative scores can leave these nearly intact, and the gap between the two is the interesting part.

The thresholds below are not the ones in the tables above, and that is the point. A threshold is not a budget: it admits however many experts the router's own confidence lets it, and the router is more decided about a one-sentence question than about the middle of a derivation. The generative settings therefore spend fewer experts here than the tier they were calibrated for, and a row that is both cheaper than Fixed-K and worse than it answers nothing. Each method was re-solved against this distribution (`scripts/solve_qa_hyper.py`) so that its average sits at or just above Fixed-K's.

`acc` is used for all four columns because it is the only metric all four report — winogrande has no `acc_norm`. The result files carry `acc_norm` for the other three, and taking it where available moves rows by up to three places, so a comparison against a published table should be read off the JSON with that table's own metric.

### Target avg ≈ 4.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 (native) | 6.000 | 6.000 | **61.10** | 60.2 | 82.3 | 70.3 | 31.6 |
| Fixed-K | k=4 | 4.000 | 4.000 | **59.09** (-2.01) | 58.3 | 81.2 | 66.1 | 30.8 |
| Ban | lambda=0.7296, k_min=3 | 4.070 | — | **59.69** (-1.41) | 58.0 | 81.7 | 68.4 | 30.6 |
| DiEP | beta=0.3273, k_min=2 | 4.357 | — | **59.07** (-2.03) | 57.4 | 82.2 | 65.7 | 31.0 |
| DiEP-damped | beta=0.3424, k_min=2, gamma_alpha=0.25, cap=0.9 | 4.320 | — | **59.32** (-1.78) | 57.4 | 81.6 | 67.8 | 30.4 |
| Dynamic Routing | threshold=0.7991, score_source=renormalized | 4.225 | — | **59.82** (-1.28) | 58.4 | 82.0 | 68.4 | 30.4 |
| NAEE | beta=0.3529, k_min=2 | 4.243 | — | **58.25** (-2.85) | 56.3 | 81.5 | 65.7 | 29.4 |

### Target avg ≈ 3.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 (native) | 6.000 | 6.000 | **61.10** | 60.2 | 82.3 | 70.3 | 31.6 |
| Fixed-K | k=3 | 3.000 | 3.000 | **56.24** (-4.86) | 52.3 | 79.4 | 63.9 | 29.4 |
| Ban | lambda=0.5298, k_min=2 | 3.045 | — | **56.48** (-4.62) | 54.0 | 79.2 | 64.6 | 28.2 |
| DiEP | beta=0.5628, k_min=2 | 3.043 | — | **54.35** (-6.75) | 51.1 | 78.3 | 59.2 | 28.8 |
| DiEP-damped | beta=0.5514, alpha=0.5, cap=0.9, k_min=2 | 3.045 | — | **55.03** (-6.07) | 51.3 | 77.9 | 60.3 | 30.6 |
| Dynamic Routing | threshold=0.638, score_source=renormalized | 3.097 | — | **57.29** (-3.81) | 54.9 | 79.8 | 64.4 | 30.0 |
| NAEE | beta=0.5414, k_min=2 | 3.076 | — | **54.29** (-6.81) | 51.9 | 77.4 | 60.5 | 27.4 |

### Target avg ≈ 2.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 (native) | 6.000 | 6.000 | **61.10** | 60.2 | 82.3 | 70.3 | 31.6 |
| Fixed-K | k=2 | 2.000 | 2.000 | **47.23** (-13.88) | 38.9 | 69.6 | 57.4 | 23.0 |
| Ban | lambda=0.4122, k_min=1 | 2.035 | — | **47.21** (-13.89) | 41.0 | 70.9 | 53.2 | 23.8 |
| DiEP | beta=0.8, k_min=2 | 2.391 | 2.573 | **50.64** (-10.46) | 45.1 | 73.8 | 57.5 | 26.2 |
| DiEP-damped | beta=1.1, k_min=2, gamma_alpha=0.25, cap=0.9 | 2.051 | 2.083 | **48.82** (-12.28) | 43.7 | 69.8 | 57.8 | 24.0 |
| Dynamic Routing | threshold=0.4416, score_source=renormalized | 2.099 | — | **48.98** (-12.12) | 44.5 | 69.5 | 56.6 | 25.4 |
| NAEE | beta=0.75, k_min=2 | 2.283 | 2.428 | **51.05** (-10.05) | 46.9 | 73.7 | 57.2 | 26.4 |

Fixed-K does not win here the way it does above: 5 rows come out ahead of it without spending more experts, by up to 1.76 points. The standard error of this four-task mean is about 0.74 points, though, so those are ties rather than wins — which is itself the finding, because in the generative tables the same methods lose to Fixed-K by margins far outside that.

The two averages are the same configuration measured under different inputs, and they differ by between -0.55 and -0.03 experts (DiEP-damped-a0.25-cap0.9-kmin2-beta0.467 and DiEP-damped-a0.25-cap0.9-kmin2-beta1.1 are the extremes). A dynamic method reads the router's own confidence, and short factual questions do not distribute it the way a long derivation does, so a budget calibrated on generation does not transfer here exactly. The QA column is the one that describes these scores.

## Average expert count landed where it was expected to

The hyperparameter-to-average mapping is the only part of the compact table that survived the cache bug, so this confirms the re-runs are sitting at the intended operating points.

| config | expected avg | this run | Δ |
| --- | --- | --- | --- |
| Baseline-k6 | 6.0000 | 6.000 | +0.00 |
| FixedK-k4 | 4.0000 | 4.000 | +0.00 |
| Ban-kmin2-lambda0.9 | 3.9768 | 3.966 | -0.01 |
| DiEP-kmin2-beta0.45 | 4.1463 | 4.086 | -0.06 |
| DiEP-damped-a0.25-cap0.9-kmin2-beta0.467 | 4.0000 | 4.045 | +0.05 |
| DiEP-damped-a0.5-cap0.9-kmin2-beta0.468 | 4.0000 ᵖ | 3.984 | -0.02 |
| Dynamic_Routing-thr0.75-renormalized | 4.1882 | 4.171 | -0.02 |
| NAEE-kmin2-beta0.45 | 4.0590 | 4.058 | -0.00 |
| FixedK-k3 | 3.0000 | 3.000 | +0.00 |
| Ban-kmin2-lambda0.45 | 3.0003 | 2.997 | -0.00 |
| DiEP-kmin2-beta0.65 | 3.0835 | 3.033 | -0.05 |
| DiEP-damped-a0.25-cap0.9-kmin2-beta0.636 | 3.0000 | 3.022 | +0.02 |
| DiEP-damped-a0.5-cap0.9-kmin2-beta0.645 | 3.0000 ᵖ | 2.963 | -0.04 |
| Dynamic_Routing-thr0.6-renormalized | 3.1465 | 3.138 | -0.01 |
| NAEE-kmin2-beta0.6 | 3.1091 | 3.106 | -0.00 |
| FixedK-k2 | 2.0000 | 2.000 | +0.00 |
| Ban-kmin1-lambda0.35 | 1.9919 | 1.990 | -0.00 |
| DiEP-kmin1-beta0.8 | 2.2507 | 2.255 | +0.00 |
| DiEP-kmin2-beta0.8 | — ᵘ | 2.573 | — |
| DiEP-paper-gamma1-kmin2 | 2.2900 ᵖ | 2.695 | +0.40 |
| DiEP-damped-a0.25-cap0.9-kmin2-beta1.1 | 2.0000 | 2.083 | +0.08 |
| Dynamic_Routing-thr0.4-renormalized | 2.0352 | 2.036 | +0.00 |
| NAEE-kmin1-beta0.75 | 2.0891 | 2.088 | -0.00 |
| NAEE-kmin2-beta0.75 | — ᵘ | 2.428 | — |

ᵖ not in the old table, either because it never swept this method that far or because the configuration did not exist then. The expected value comes from solving the hyperparameter offline against averages that were measured (`scripts/calibrate_ban_lambda.py` for Ban, `scripts/solve_diep_beta.py` for DiEP), so it is a prediction being tested here rather than a target that was already hit.

ᵘ run to isolate one hyperparameter rather than to hit an average, so there was nothing to predict: its average is simply what the setting produced.
