# Qwen3-Next-80B-A3B-Instruct — corrected expert-pruning results

25 of 29 configurations finished.

The compact table's scores cannot be used as a reference: 74% of its score cells are exact duplicates of another configuration's, because it predates keying the lighteval sample cache on the expert pruning configuration (see `docs/compact_cache_contamination.md`). What it still provides, and what these runs take from it, is which hyperparameters land a method near a given average expert count.

Every table opens with two reference rows: the unpruned model (its native top-k of 10) and the Fixed-K point for that target. Fixed-K is what a method has to beat, since it reaches the same average expert count by simply keeping fewer experts per token. The bracketed number after each mean is the gap to the unpruned model.

The reference rows are held to the lower of the two readings: where the re-run came out above the old table it keeps the old value and is marked ᵒ, with what it actually scored given in the footnote. This only ever moves the reference down, so no pruning method is credited with beating an inflated Fixed-K.

Settings: the seven generative datasets, temperature 0.7 / top_p 0.8 / top_k 20, max_model_len and max_new_tokens 32768 unless a row is marked ᵏ, TP=4. Every run has its own cache namespace keyed on the pruning configuration, so no two rows here can share samples.

## Results by target average expert count

### Target avg ≈ 6.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline ᵒ | k=10 (native) | 10.000 | **77.11** | 82.1 | 74.2 | 88.4 | 79.7 | 67.1 | 53.1 | 95.2 |
| Fixed-K ᵒ | k=6 | 6.000 | **78.37** (+1.26) | 82.1 | 76.3 | 90.8 | 82.2 | 66.6 | 54.9 | 95.7 |
| Ban | lambda=0.85, k_min=4 | 6.185 | **78.31** (+1.20) | 82.2 | 74.2 | 89.2 | 82.7 | 69.7 | 54.9 | 95.3 |
| DiEP ᵈ | beta=0.65, k_min=3 | 6.239 | **75.50** (-1.61) | 81.5 | 74.7 | 89.8 | 77.4 | 62.1 | 48.0 | 95.0 |
| Dynamic Routing | threshold=0.7, score_source=renormalized | 5.973 | **78.01** (+0.90) | 81.9 | 77.3 | 89.6 | 83.4 | 67.9 | 50.9 | 95.1 |
| NAEE | beta=0.4, k_min=2 | 6.156 | **75.32** (-1.79) | 81.2 | 69.2 | 89.6 | 80.1 | 63.0 | 49.1 | 95.1 |

### Target avg ≈ 5.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline ᵒ | k=10 (native) | 10.000 | **77.11** | 82.1 | 74.2 | 88.4 | 79.7 | 67.1 | 53.1 | 95.2 |
| Fixed-K | k=5 | 5.000 | **76.96** (-0.16) | 81.8 | 69.2 | 90.8 | 81.1 | 66.0 | 54.9 | 94.9 |
| Ban | lambda=0.65, k_min=3 | 5.083 | **77.51** (+0.40) | 81.9 | 72.2 | 89.6 | 82.9 | 65.3 | 55.4 | 95.2 |
| DiEP ᵈ ᵛ ᵘ | beta=0.9, k_min=3 | 4.352 | **74.16** (-2.96) | 81.1 | 69.2 | 90.2 | 75.5 | 58.5 | 49.7 | 94.9 |
| Dynamic Routing | threshold=0.65, score_source=renormalized | 5.361 | **77.97** (+0.85) | 82.3 | 72.2 | 91.2 | 81.1 | 67.6 | 56.0 | 95.5 |
| NAEE | beta=0.45, k_min=2 | 5.435 | **74.73** (-2.38) | 80.6 | 73.7 | 90.4 | 78.8 | 60.4 | 44.6 | 94.7 |

### Target avg ≈ 4.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline ᵒ | k=10 (native) | 10.000 | **77.11** | 82.1 | 74.2 | 88.4 | 79.7 | 67.1 | 53.1 | 95.2 |
| Fixed-K ᵒ | k=4 | 4.000 | **73.89** (-3.23) | 80.7 | 72.7 | 88.0 | 74.8 | 58.8 | 49.1 | 93.1 |
| Ban | lambda=0.45, k_min=3 | 4.269 | **76.94** (-0.17) | 81.3 | 73.2 | 88.6 | 79.4 | 64.6 | 57.1 | 94.4 |
| DiEP-damped | beta=0.57, k_min=2, gamma_alpha=0.25, cap=0.9 | 4.203 | **71.61** (-5.50) | 80.1 | 67.2 | 92.4 | 71.7 | 53.0 | 42.9 | 94.1 |
| Dynamic Routing | threshold=0.55, score_source=renormalized | 4.298 | **75.71** (-1.40) | 81.1 | 69.7 | 89.2 | 78.5 | 62.2 | 54.3 | 95.0 |
| NAEE | beta=0.55, k_min=2 | 4.165 | **71.37** (-5.74) | 78.7 | 70.7 | 90.4 | 70.9 | 53.4 | 42.3 | 93.2 |

### Target avg ≈ 3.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline ᵒ | k=10 (native) | 10.000 | **77.11** | 82.1 | 74.2 | 88.4 | 79.7 | 67.1 | 53.1 | 95.2 |
| Fixed-K ᵏ | k=3 | 3.000 | **64.54** (-12.57) | 78.0 | 66.7 | 87.6 | 57.8 | 40.8 | 34.9 | 86.1 |
| Ban | lambda=0.3, k_min=2 | 3.066 | **68.91** (-8.20) | 79.1 | 73.7 | 88.4 | 64.6 | 44.8 | 42.9 | 88.9 |
| Ban ᵏ | lambda=0.3, k_min=2, 16k cap | 3.066 | **67.57** (-9.54) | 78.9 | 69.7 | 87.2 | 61.6 | 44.4 | 41.7 | 89.5 |
| DiEP-damped ᵏ | beta=0.72, k_min=2, gamma_alpha=0.25, cap=0.9 | 3.062 | **61.05** (-16.06) | 77.5 | 64.6 | 85.8 | 48.8 | 34.7 | 24.0 | 91.9 |
| Dynamic Routing ᵏ | threshold=0.4, score_source=renormalized | 2.973 | **64.06** (-13.05) | 76.8 | 66.7 | 85.0 | 56.5 | 41.7 | 34.9 | 87.0 |
| NAEE ᵏ | beta=0.65, k_min=2 | 3.177 | **61.47** (-15.65) | 76.2 | 62.6 | 85.8 | 51.2 | 38.8 | 26.9 | 88.8 |

### Target avg ≈ 2.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline ᵒ | k=10 (native) | 10.000 | **77.11** | 82.1 | 74.2 | 88.4 | 79.7 | 67.1 | 53.1 | 95.2 |
| Fixed-K ᵏ | k=2 | 2.000 | **39.70** (-37.42) | 62.3 | 44.4 | 69.0 | 12.0 | 7.7 | 7.4 | 75.0 |
| Ban ᵛ | lambda=0.3, k_min=2 | 3.066 | **68.91** (-8.20) | 79.1 | 73.7 | 88.4 | 64.6 | 44.8 | 42.9 | 88.9 |
| Ban ᵛ ᵏ | lambda=0.3, k_min=2, 16k cap | 3.066 | **67.57** (-9.54) | 78.9 | 69.7 | 87.2 | 61.6 | 44.4 | 41.7 | 89.5 |
| Dynamic Routing ᵏ | threshold=0.3, score_source=renormalized | 2.215 | **44.23** (-32.88) | 62.3 | 47.5 | 77.6 | 18.4 | 11.7 | 10.9 | 81.3 |
| Dynamic Routing ᵛ ᵏ | threshold=0.4, score_source=renormalized | 2.973 | **64.06** (-13.05) | 76.8 | 66.7 | 85.0 | 56.5 | 41.7 | 34.9 | 87.0 |
| NAEE ᵛ ᵏ | beta=0.75, k_min=2 | 2.569 | **53.02** (-24.09) | 71.9 | 57.1 | 84.8 | 33.8 | 25.0 | 16.0 | 82.6 |

ᵘ spends fewer experts than Fixed-K does at that tier, by up to 0.65 (DiEP at beta=0.9, k_min=3). 1 row is in that position, and a mean below Fixed-K's there can be explained by the smaller budget rather than by the method. These are not being re-run: a generative run of this model costs a day, and the threshold that would fix it is a different operating point rather than a correction to this one. The 0-shot QA tables below were re-solved instead, since a run there is minutes.

ᵗ still the old table's numbers, waiting on the run here to replace them.

ᵈ DiEP's skipping rule is specified for k=2 and our generalisation of it to a top-10 router prunes the whole tail, leaving it 1 points below Fixed-K. Only the paper's auxiliary skipping mechanism is implemented, not its differentiable pruning. Read these rows as a result about that generalisation, not as DiEP's achievable quality — see `docs/known_issues.md`.

ᵒ the re-run of this reference row came out above the old table, so the old table's value is kept. Fixed-K needs no calibration and the old table had little room to get it wrong, so a higher reading here is treated as measurement spread rather than as a better reference; taking it would flatter every pruning method measured against it. What the re-runs actually scored: k=10 scored 79.31 against 79.05, k=10 scored 79.31 against 79.05, k=10 scored 79.31 against 79.05, k=10 scored 79.31 against 79.05, k=10 scored 79.31 against 79.05, k=6 scored 76.52 against 76.12, k=4 scored 72.04 against 70.98.

ᵛ this method has no setting that reaches this target, so its closest one is shown. It is spending more experts than the other rows and its mean is not comparable to them.

ᵏ generated under a shorter limit than the 32k used elsewhere (16k on 10 rows). At two or three experts the model rambles and runs almost every answer into the limit, and an answer that is not right by 16k is very rarely right at 32k, so the shorter window buys a large amount of time at little cost in accuracy. It is not free, though: a row marked here could in principle lose a point to truncation that the unpruned reference did not pay. `Ban-kmin2-lambda0.3-cap16k` measures exactly that, being the same configuration as `Ban-kmin2-lambda0.3` under the two limits.

ᵃ this model keeps 1 shared experts active for every token on top of the routed ones. The average counts routed experts only, matching the old table, so an average of 3 here costs 4 experts of compute and is not comparable to an average of 3 on a model without shared experts.

## 0-shot QA (lm-eval)

Four multiple-choice sets, scored by likelihood over the options with nothing generated, so they answer a different question from the tables above: how much of the model's knowledge survives pruning, as opposed to how much of its reasoning does. A budget that ruins the generative scores can leave these nearly intact, and the gap between the two is the interesting part.

The thresholds below are not the ones in the tables above, and that is the point. A threshold is not a budget: it admits however many experts the router's own confidence lets it, and the router is more decided about a one-sentence question than about the middle of a derivation. The generative settings therefore spend fewer experts here than the tier they were calibrated for, and a row that is both cheaper than Fixed-K and worse than it answers nothing. Each method was re-solved against this distribution (`scripts/solve_qa_hyper.py`) so that its average sits at or just above Fixed-K's.

`acc` is used for all four columns because it is the only metric all four report — winogrande has no `acc_norm`. The result files carry `acc_norm` for the other three, and taking it where available moves rows by up to three places, so a comparison against a published table should be read off the JSON with that table's own metric.

### Target avg ≈ 6.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=10 (native) | 10.000 | 10.000 | **65.12** | 63.1 | 86.9 | 76.2 | 34.2 |
| Fixed-K | k=6 | 6.000 | 6.000 | **64.02** (-1.10) | 63.1 | 85.6 | 73.9 | 33.6 |
| Ban | lambda=0.85, k_min=4 | 6.085 | 6.185 | **63.64** (-1.48) | 62.4 | 85.7 | 74.1 | 32.4 |
| DiEP | beta=0.65, k_min=3 | 5.924 ᵘ | 6.239 | **62.79** (-2.33) | 60.3 | 85.1 | 73.8 | 32.0 |
| Dynamic Routing | threshold=0.7664, score_source=renormalized | 6.203 | — | **64.14** (-0.99) | 62.2 | 85.7 | 73.0 | 35.6 |
| NAEE | beta=0.2893, k_min=2 | 6.537 | — | **62.95** (-2.18) | 60.2 | 84.7 | 74.1 | 32.8 |

### Target avg ≈ 5.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=10 (native) | 10.000 | 10.000 | **65.12** | 63.1 | 86.9 | 76.2 | 34.2 |
| Fixed-K | k=5 | 5.000 | 5.000 | **62.05** (-3.07) | 59.6 | 84.9 | 71.0 | 32.6 |
| Ban | lambda=0.65, k_min=3 | 4.922 ᵘ | 5.083 | **62.41** (-2.72) | 60.8 | 84.8 | 71.7 | 32.2 |
| DiEP | beta=0.9, k_min=3 | 4.914 ᵘ | 4.352 | **61.29** (-3.84) | 57.6 | 84.0 | 71.5 | 32.0 |
| Dynamic Routing | threshold=0.6771, score_source=renormalized | 5.090 | — | **62.50** (-2.63) | 58.9 | 85.0 | 72.1 | 34.0 |
| NAEE | beta=0.45, k_min=2 | 4.287 ᵘ | 5.435 | **57.66** (-7.47) | 54.0 | 80.8 | 66.5 | 29.4 |

### Target avg ≈ 4.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=10 (native) | 10.000 | 10.000 | **65.12** | 63.1 | 86.9 | 76.2 | 34.2 |
| Fixed-K | k=4 | 4.000 | 4.000 | **58.88** (-6.24) | 55.9 | 82.2 | 67.2 | 30.2 |
| Ban | lambda=0.45, k_min=3 | 4.183 | 4.269 | **60.97** (-4.16) | 59.8 | 83.5 | 67.8 | 32.8 |
| DiEP-damped | beta=0.57, k_min=2, gamma_alpha=0.25, cap=0.9 | 3.938 ᵘ | 4.203 | **56.31** (-8.82) | 51.6 | 79.4 | 65.6 | 28.6 |
| Dynamic Routing | threshold=0.55, score_source=renormalized | 3.812 ᵘ | 4.298 | **59.24** (-5.89) | 56.1 | 81.7 | 68.1 | 31.0 |
| NAEE | beta=0.55, k_min=2 | 3.336 ᵘ | 4.165 | **54.65** (-10.48) | 51.7 | 75.9 | 62.2 | 28.8 |

### Target avg ≈ 3.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=10 (native) | 10.000 | 10.000 | **65.12** | 63.1 | 86.9 | 76.2 | 34.2 |
| Fixed-K | k=3 | 3.000 | 3.000 | **54.83** (-10.29) | 50.0 | 77.4 | 62.1 | 29.8 |
| Ban | lambda=0.3, k_min=2 | 3.042 | 3.066 | **55.43** (-9.69) | 50.2 | 77.7 | 63.1 | 30.8 |
| DiEP-damped | beta=0.72, k_min=2, gamma_alpha=0.25, cap=0.9 | 3.002 | 3.062 | **52.58** (-12.55) | 46.8 | 73.5 | 60.5 | 29.6 |
| Dynamic Routing | threshold=0.4, score_source=renormalized | 2.620 ᵘ | 2.973 | **52.03** (-13.09) | 45.5 | 71.8 | 64.0 | 26.8 |
| NAEE | beta=0.65, k_min=2 | 2.704 ᵘ | 3.177 | **50.77** (-14.36) | 42.8 | 70.8 | 58.8 | 30.6 |

### Target avg ≈ 2.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=10 (native) | 10.000 | 10.000 | **65.12** | 63.1 | 86.9 | 76.2 | 34.2 |
| Fixed-K | k=2 | 2.000 | 2.000 | **44.04** (-21.08) | 36.8 | 61.1 | 56.1 | 22.2 |
| Ban | lambda=0.3, k_min=2 | 3.042 | 3.066 | **55.43** (-9.69) | 50.2 | 77.7 | 63.1 | 30.8 |
| Dynamic Routing | threshold=0.4, score_source=renormalized | 2.620 | 2.973 | **52.03** (-13.09) | 45.5 | 71.8 | 64.0 | 26.8 |
| NAEE | beta=0.75, k_min=2 | 2.325 | 2.569 | **47.98** (-17.15) | 38.9 | 67.2 | 59.8 | 26.0 |

ᵘ spends fewer experts than Fixed-K does at that tier, by up to 0.71 (NAEE-kmin2-beta0.45). 9 rows are in that position, which makes them unreadable as comparisons rather than merely imprecise: a lower score can be explained by the smaller budget. Their re-solved settings are queued, and each replaces the row above when it lands.

Fixed-K does not win here the way it does above: 4 rows come out ahead of it without spending more experts, by up to 0.60 points. The standard error of this four-task mean is about 0.74 points, though, so those are ties rather than wins — which is itself the finding, because in the generative tables the same methods lose to Fixed-K by margins far outside that.

The two averages are the same configuration measured under different inputs, and they differ by between -1.30 and +0.56 experts (NAEE-kmin2-beta0.4 and DiEP-kmin3-beta0.9 are the extremes). A dynamic method reads the router's own confidence, and short factual questions do not distribute it the way a long derivation does, so a budget calibrated on generation does not transfer here exactly. The QA column is the one that describes these scores.

## Average expert count landed where it was expected to

The hyperparameter-to-average mapping is the only part of the compact table that survived the cache bug, so this confirms the re-runs are sitting at the intended operating points.

| config | expected avg | this run | Δ |
| --- | --- | --- | --- |
| Baseline-k10 | 10.0000 | 10.000 | +0.00 |
| FixedK-k6 | 6.0000 | 6.000 | +0.00 |
| Ban-kmin4-lambda0.85 | 6.1783 | 6.185 | +0.01 |
| DiEP-kmin3-beta0.65 | 5.9314 | 6.239 | +0.31 |
| Dynamic_Routing-thr0.7-renormalized | 5.7384 | 5.973 | +0.23 |
| NAEE-kmin2-beta0.4 | 5.6515 | 6.156 | +0.50 |
| FixedK-k5 | 5.0000 | 5.000 | +0.00 |
| Ban-kmin3-lambda0.65 | 5.0413 | 5.083 | +0.04 |
| DiEP-kmin3-beta0.9 | 4.9175 | 4.352 | -0.57 |
| Dynamic_Routing-thr0.65-renormalized | 5.1496 | 5.361 | +0.21 |
| NAEE-kmin2-beta0.45 | 5.0029 | 5.435 | +0.43 |
| FixedK-k4 | 4.0000 | 4.000 | +0.00 |
| Ban-kmin3-lambda0.45 | 4.2389 | 4.269 | +0.03 |
| DiEP-damped-a0.25-cap0.9-kmin2-beta0.57 | 4.0000 | 4.203 | +0.20 |
| Dynamic_Routing-thr0.55-renormalized | 4.1157 | 4.298 | +0.18 |
| NAEE-kmin2-beta0.55 | 3.8539 | 4.165 | +0.31 |
| FixedK-k3 | 3.0000 | 3.000 | +0.00 |
| Ban-kmin2-lambda0.3 | 3.0719 | 3.066 | -0.01 |
| Ban-kmin2-lambda0.3-cap16k | 3.0719 | 3.066 | -0.01 |
| DiEP-damped-a0.25-cap0.9-kmin2-beta0.72 | 3.0000 | 3.062 | +0.06 |
| Dynamic_Routing-thr0.4-renormalized | 2.8442 | 2.973 | +0.13 |
| NAEE-kmin2-beta0.65 | 3.0076 | 3.177 | +0.17 |
| FixedK-k2 | 2.0000 | 2.000 | +0.00 |
| Dynamic_Routing-thr0.3-renormalized | 2.0608 | 2.215 | +0.15 |
| NAEE-kmin2-beta0.75 | 2.4887 | 2.569 | +0.08 |
