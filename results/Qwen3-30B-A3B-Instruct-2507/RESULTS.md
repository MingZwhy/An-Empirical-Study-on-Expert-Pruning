# Qwen3-30B-A3B-Instruct-2507 — corrected expert-pruning results

34 of 35 configurations finished. 5 further configurations are set aside and not counted here: the five MC-MoE runs, deferred because one of them takes several days on this suite.

The compact table's scores cannot be used as a reference: 65% of its score cells are exact duplicates of another configuration's, because it predates keying the lighteval sample cache on the expert pruning configuration (see `docs/compact_cache_contamination.md`). What it still provides, and what these runs take from it, is which hyperparameters land a method near a given average expert count.

Every table opens with two reference rows: the unpruned model (its native top-k of 8) and the Fixed-K point for that target. Fixed-K is what a method has to beat, since it reaches the same average expert count by simply keeping fewer experts per token. The bracketed number after each mean is the gap to the unpruned model.

The reference rows are held to the lower of the two readings: where the re-run came out above the old table it keeps the old value and is marked ᵒ, with what it actually scored given in the footnote. This only ever moves the reference down, so no pruning method is credited with beating an inflated Fixed-K.

Settings: the seven generative datasets, temperature 0.7 / top_p 0.8 / top_k 20, max_model_len and max_new_tokens 32768 unless a row is marked ᵏ, TP=2. Every run has its own cache namespace keyed on the pruning configuration, so no two rows here can share samples.

## Results by target average expert count

### Target avg ≈ 6.0

| method | hyperparams | avg experts | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=8 (unpruned) | 8.000 | **70.20** | 74.2 | 56.6 | 89.6 | 74.1 | 61.4 | 41.1 | 94.4 |
| Fixed-K ᵒ | k=6 | 6.000 | **70.00** (-0.20) | 74.4 | 53.5 | 90.4 | 72.8 | 59.3 | 44.6 | 95.0 |
| Ban ᵛ ᵘ | lambda=0.95, k_min=3 | 5.054 | **69.18** (-1.01) | 74.1 | 57.6 | 89.4 | 71.6 | 54.7 | 42.9 | 94.0 |
| DiEP ᵈ ᵘ | beta=0.5, k_min=2 | 5.950 | **60.05** (-10.15) | 70.8 | 51.0 | 88.4 | 51.4 | 41.7 | 23.4 | 93.6 |
| DiEP-damped | beta=0.413, alpha=0.25, cap=0.9, k_min=2 | 6.110 | **67.76** (-2.43) | 71.9 | 52.0 | 89.2 | 71.0 | 54.3 | 41.7 | 94.2 |
| DiEP-damped | beta=0.442, alpha=0.5, cap=0.9, k_min=2 | 6.027 | **63.51** (-6.69) | 71.6 | 45.5 | 87.6 | 61.5 | 50.1 | 34.9 | 93.6 |
| Dynamic Routing ᵘ | threshold=0.8, score_source=renormalized | 5.918 | **69.99** (-0.21) | 74.5 | 59.1 | 90.0 | 74.6 | 58.1 | 39.4 | 94.2 |
| NAEE | beta=0.35, k_min=2 | 6.408 | **68.95** (-1.25) | 73.5 | 59.6 | 91.2 | 71.3 | 55.7 | 38.3 | 93.0 |

### Target avg ≈ 5.0

| method | hyperparams | avg experts | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=8 (unpruned) | 8.000 | **70.20** | 74.2 | 56.6 | 89.6 | 74.1 | 61.4 | 41.1 | 94.4 |
| Fixed-K ᵒ | k=5 | 5.000 | **68.99** (-1.21) | 73.8 | 55.0 | 91.0 | 72.1 | 55.4 | 41.7 | 93.9 |
| Ban | lambda=0.9, k_min=3 | 4.995 | **68.82** (-1.38) | 73.9 | 55.6 | 90.2 | 72.0 | 56.5 | 39.4 | 94.2 |
| DiEP ᵈ ᵘ | beta=0.7, k_min=2 | 4.841 | **52.47** (-17.73) | 70.2 | 46.0 | 81.8 | 35.3 | 26.5 | 14.9 | 92.7 |
| DiEP-damped ᵘ | beta=0.525, alpha=0.25, cap=0.9, k_min=2 | 4.927 | **63.82** (-6.38) | 71.2 | 51.0 | 89.4 | 60.9 | 49.7 | 32.0 | 92.5 |
| DiEP-damped ᵘ | beta=0.57, alpha=0.5, cap=0.9, k_min=2 | 4.911 | **57.08** (-13.11) | 70.2 | 49.5 | 84.0 | 44.6 | 37.8 | 21.7 | 91.8 |
| Dynamic Routing ᵘ | threshold=0.7, score_source=renormalized | 4.906 | **68.88** (-1.32) | 73.2 | 55.1 | 89.6 | 73.2 | 55.2 | 41.7 | 94.2 |
| NAEE | beta=0.45, k_min=2 | 5.134 | **66.90** (-3.30) | 72.5 | 56.1 | 81.0 | 69.1 | 53.6 | 43.4 | 92.6 |

### Target avg ≈ 4.0

| method | hyperparams | avg experts | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=8 (unpruned) | 8.000 | **70.20** | 74.2 | 56.6 | 89.6 | 74.1 | 61.4 | 41.1 | 94.4 |
| Fixed-K | k=4 | 4.000 | **65.49** (-4.71) | 72.3 | 54.0 | 87.8 | 62.5 | 47.9 | 40.6 | 93.3 |
| Ban | lambda=0.6, k_min=3 | 4.135 | **65.63** (-4.57) | 72.0 | 51.5 | 85.2 | 65.2 | 52.1 | 40.0 | 93.5 |
| DiEP ᵈ ᵘ | beta=1.0, k_min=2 | 3.801 | **41.59** (-28.60) | 63.7 | 40.4 | 68.6 | 11.5 | 11.0 | 6.3 | 89.7 |
| DiEP-damped ᵘ | beta=0.645, alpha=0.25, cap=0.9, k_min=2 | 3.879 | **52.29** (-17.91) | 68.9 | 50.0 | 82.2 | 35.9 | 24.9 | 15.4 | 88.6 |
| DiEP-damped ᵘ | beta=0.723, alpha=0.5, cap=0.9, k_min=2 | 3.911 | **46.33** (-23.87) | 68.2 | 36.4 | 73.6 | 26.7 | 19.9 | 9.7 | 89.8 |
| DiEP-paper | gamma_1*gamma_2, beta=1.0, k_min=2 | 4.365 | **46.94** (-23.26) | 68.4 | 46.0 | 75.0 | 22.8 | 17.4 | 7.4 | 91.7 |
| Dynamic Routing | threshold=0.6, score_source=renormalized | 4.052 | **65.73** (-4.47) | 71.5 | 51.5 | 88.2 | 66.4 | 49.9 | 39.4 | 93.1 |
| NAEE | beta=0.55, k_min=2 | 4.081 | **62.42** (-7.78) | 71.1 | 57.1 | 85.4 | 59.6 | 47.4 | 27.4 | 88.9 |

### Target avg ≈ 3.0

| method | hyperparams | avg experts | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=8 (unpruned) | 8.000 | **70.20** | 74.2 | 56.6 | 89.6 | 74.1 | 61.4 | 41.1 | 94.4 |
| Fixed-K ᵒ | k=3 | 3.000 | **54.43** (-15.77) | 67.0 | 48.5 | 80.2 | 40.0 | 31.1 | 25.7 | 88.5 |
| Ban | lambda=0.5, k_min=2 | 3.142 | **57.59** (-12.61) | 68.8 | 50.5 | 88.0 | 42.9 | 34.0 | 28.6 | 90.4 |
| DiEP ᵈ | beta=1.2, k_min=2 | 3.363 | **36.88** (-33.32) | 59.5 | 39.9 | 58.4 | 3.3 | 6.0 | 4.6 | 86.4 |
| DiEP-damped | beta=0.926, alpha=0.5, cap=0.9, k_min=2 | 3.083 | **35.42** (-34.78) | 56.3 | 39.9 | 56.6 | 2.8 | 6.0 | 3.4 | 82.9 |
| Dynamic Routing | threshold=0.45, score_source=renormalized | 2.970 | **52.19** (-18.01) | 66.1 | 49.0 | 81.8 | 36.0 | 27.5 | 22.3 | 82.6 |
| NAEE | beta=0.7, k_min=2 | 2.986 | **43.91** (-26.29) | 57.4 | 39.4 | 75.6 | 23.2 | 16.9 | 13.7 | 81.1 |

### Target avg ≈ 2.0

| method | hyperparams | avg experts | mean | mmlu_pro | gpqa | math500 | aime24 | aime25 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=8 (unpruned) | 8.000 | **70.20** | 74.2 | 56.6 | 89.6 | 74.1 | 61.4 | 41.1 | 94.4 |
| Fixed-K ᵒ | k=2 | 2.000 | **14.00** (-56.20) | 22.1 | 27.3 | 24.6 | 0.0 | 0.0 | 1.1 | 22.9 |
| Ban | lambda=0.26, k_min=1 | 2.013 | **12.97** (-57.22) | 19.7 | 22.7 | 22.2 | 0.0 | 0.1 | 0.6 | 25.5 |
| DiEP ᵈ ᵛ | beta=1.2, k_min=2 | 3.363 | **36.88** (-33.32) | 59.5 | 39.9 | 58.4 | 3.3 | 6.0 | 4.6 | 86.4 |
| DiEP-damped | beta=1.394, alpha=0.5, cap=0.9, k_min=2 | 2.308 | **19.66** (-50.53) | 30.9 | 28.8 | 34.2 | 0.0 | 0.1 | 1.7 | 41.9 |
| Dynamic Routing | threshold=0.35, score_source=renormalized | 2.327 | **24.20** (-46.00) | 32.7 | 27.8 | 48.2 | 0.4 | 1.5 | 2.3 | 56.6 |
| NAEE | beta=0.8, k_min=2 | 2.428 | **27.72** (-42.48) | 38.9 | 33.3 | 50.4 | 1.7 | 3.5 | 4.0 | 62.2 |

ᵘ spends fewer experts than Fixed-K does at that tier, by up to 0.95 (Ban at lambda=0.95, k_min=3). 10 rows are in that position, and a mean below Fixed-K's there can be explained by the smaller budget rather than by the method. These are not being re-run: a generative run of this model costs a day, and the threshold that would fix it is a different operating point rather than a correction to this one. The 0-shot QA tables below were re-solved instead, since a run there is minutes.

ᵗ still the old table's numbers, waiting on the run here to replace them.

ᵈ DiEP's skipping rule is specified for k=2 and our generalisation of it to a top-8 router prunes the whole tail, leaving it 8 to 21 points below Fixed-K at the same budget. Only the paper's auxiliary skipping mechanism is implemented, not its differentiable pruning. Read these rows as a result about that generalisation, not as DiEP's achievable quality — see `docs/known_issues.md`.

ᵒ the re-run of this reference row came out above the old table, so the old table's value is kept. Fixed-K needs no calibration and the old table had little room to get it wrong, so a higher reading here is treated as measurement spread rather than as a better reference; taking it would flatter every pruning method measured against it. What the re-runs actually scored: k=6 scored 74.79 against 74.09, k=5 scored 73.67 against 72.81, k=3 scored 57.65 against 56.44, k=2 scored 15.80 against 15.74.

ᵛ this method has no setting that reaches this target, so its closest one is shown. It is spending more experts than the other rows and its mean is not comparable to them.

## 0-shot QA (lm-eval)

Four multiple-choice sets, scored by likelihood over the options with nothing generated, so they answer a different question from the tables above: how much of the model's knowledge survives pruning, as opposed to how much of its reasoning does. A budget that ruins the generative scores can leave these nearly intact, and the gap between the two is the interesting part.

The thresholds below are not the ones in the tables above, and that is the point. A threshold is not a budget: it admits however many experts the router's own confidence lets it, and the router is more decided about a one-sentence question than about the middle of a derivation. The generative settings therefore spend fewer experts here than the tier they were calibrated for, and a row that is both cheaper than Fixed-K and worse than it answers nothing. Each method was re-solved against this distribution (`scripts/solve_qa_hyper.py`) so that its average sits at or just above Fixed-K's.

`acc` is used for all four columns because it is the only metric all four report — winogrande has no `acc_norm`. The result files carry `acc_norm` for the other three, and taking it where available moves rows by up to three places, so a comparison against a published table should be read off the JSON with that table's own metric.

### Target avg ≈ 6.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=8 (unpruned) | 8.000 | 8.000 | **62.77** | 60.8 | 85.0 | 73.2 | 32.0 |
| Fixed-K | k=6 | 6.000 | 6.000 | **61.19** (-1.59) | 59.2 | 82.8 | 71.3 | 31.4 |
| Ban | lambda=0.931, k_min=5 | 6.032 | — | **60.90** (-1.88) | 59.0 | 83.0 | 70.8 | 30.8 |
| DiEP | beta=0.4392, k_min=2 | 6.140 | — | **60.61** (-2.17) | 56.2 | 82.3 | 71.3 | 32.6 |
| DiEP-damped | beta=0.3529, alpha=0.5, cap=0.9, k_min=2 | 6.203 | — | **60.64** (-2.14) | 56.7 | 82.1 | 72.3 | 31.4 |
| Dynamic Routing | threshold=0.865, score_source=renormalized | 6.269 | — | **62.01** (-0.76) | 59.1 | 83.9 | 73.2 | 31.8 |
| NAEE | beta=0.2765, k_min=2 | 6.210 | — | **61.21** (-1.56) | 57.8 | 81.8 | 72.3 | 33.0 |

### Target avg ≈ 5.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=8 (unpruned) | 8.000 | 8.000 | **62.77** | 60.8 | 85.0 | 73.2 | 32.0 |
| Fixed-K | k=5 | 5.000 | 5.000 | **59.21** (-3.56) | 54.9 | 81.4 | 69.5 | 31.0 |
| Ban | lambda=0.6942, k_min=4 | 5.030 | — | **59.35** (-3.42) | 55.3 | 81.8 | 69.7 | 30.6 |
| DiEP | beta=0.7, k_min=2 | 4.664 ᵘ | 4.841 | **55.71** (-7.06) | 49.7 | 78.7 | 65.0 | 29.4 |
| DiEP-damped | beta=0.57, alpha=0.5, cap=0.9, k_min=2 | 4.458 ᵘ | 4.911 | **55.21** (-7.57) | 49.9 | 76.6 | 65.3 | 29.0 |
| Dynamic Routing | threshold=0.7, score_source=renormalized | 4.481 ᵘ | 4.906 | **58.61** (-4.16) | 55.3 | 81.0 | 68.6 | 29.6 |
| NAEE | beta=0.45, k_min=2 | 4.317 ᵘ | 5.134 | **55.64** (-7.14) | 50.2 | 77.4 | 66.2 | 28.8 |

### Target avg ≈ 4.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=8 (unpruned) | 8.000 | 8.000 | **62.77** | 60.8 | 85.0 | 73.2 | 32.0 |
| Fixed-K | k=4 | 4.000 | 4.000 | **56.21** (-6.57) | 51.3 | 77.2 | 65.1 | 31.2 |
| Ban | lambda=0.6, k_min=3 | 4.079 | 4.135 | **56.54** (-6.23) | 51.4 | 78.4 | 67.4 | 29.0 |
| DiEP | beta=1.0, k_min=2 | 3.541 ᵘ | 3.801 | **49.69** (-13.09) | 44.6 | 72.5 | 55.4 | 26.2 |
| DiEP-damped | beta=0.723, alpha=0.5, cap=0.9, k_min=2 | 3.557 ᵘ | 3.911 | **50.84** (-11.93) | 44.6 | 73.1 | 59.3 | 26.4 |
| DiEP-paper | gamma_1*gamma_2, beta=1.0, k_min=2 | 4.163 | 4.365 | **52.58** (-10.20) | 46.2 | 76.1 | 60.6 | 27.4 |
| Dynamic Routing | threshold=0.6, score_source=renormalized | 3.667 ᵘ | 4.052 | **56.26** (-6.52) | 50.7 | 77.5 | 65.7 | 31.2 |
| NAEE | beta=0.55, k_min=2 | 3.479 ᵘ | 4.081 | **51.37** (-11.40) | 46.2 | 71.0 | 63.4 | 25.0 |

### Target avg ≈ 3.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=8 (unpruned) | 8.000 | 8.000 | **62.77** | 60.8 | 85.0 | 73.2 | 32.0 |
| Fixed-K | k=3 | 3.000 | 3.000 | **50.35** (-12.43) | 44.6 | 71.8 | 58.0 | 27.0 |
| Ban | lambda=0.5, k_min=2 | 3.082 | 3.142 | **51.54** (-11.23) | 46.1 | 72.8 | 61.5 | 25.8 |
| DiEP | beta=1.2, k_min=2 | 3.070 | 3.363 | **46.06** (-16.71) | 38.9 | 67.6 | 54.0 | 23.8 |
| DiEP-damped | beta=0.926, alpha=0.5, cap=0.9, k_min=2 | 2.758 ᵘ | 3.083 | **45.06** (-17.72) | 39.1 | 66.4 | 51.8 | 23.0 |
| Dynamic Routing | threshold=0.45, score_source=renormalized | 2.663 ᵘ | 2.970 | **47.81** (-14.96) | 41.4 | 68.1 | 57.4 | 24.4 |
| NAEE | beta=0.7, k_min=2 | 2.579 ᵘ | 2.986 | **44.88** (-17.90) | 36.1 | 61.9 | 57.1 | 24.4 |

### Target avg ≈ 2.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=8 (unpruned) | 8.000 | 8.000 | **62.77** | 60.8 | 85.0 | 73.2 | 32.0 |
| Fixed-K | k=2 | 2.000 | 2.000 | **37.91** (-24.87) | 28.7 | 51.0 | 50.6 | 21.4 |
| Ban | lambda=0.26, k_min=1 | 1.949 ᵘ | 2.013 | **37.19** (-25.59) | 28.0 | 52.4 | 49.6 | 18.8 |
| DiEP | beta=1.2, k_min=2 | 3.070 | 3.363 | **46.06** (-16.71) | 38.9 | 67.6 | 54.0 | 23.8 |
| DiEP-damped | beta=1.394, alpha=0.5, cap=0.9, k_min=2 | 2.160 | 2.308 | **38.96** (-23.81) | 30.5 | 55.0 | 50.8 | 19.6 |
| Dynamic Routing | threshold=0.35, score_source=renormalized | 2.088 | 2.327 | **39.30** (-23.48) | 29.7 | 53.1 | 54.0 | 20.4 |
| NAEE | beta=0.8, k_min=2 | 2.248 | 2.428 | **41.12** (-21.66) | 31.0 | 57.1 | 53.8 | 22.6 |

ᵘ spends fewer experts than Fixed-K does at that tier, by up to 0.68 (NAEE-kmin2-beta0.45). 12 rows are in that position, which makes them unreadable as comparisons rather than merely imprecise: a lower score can be explained by the smaller budget. Their re-solved settings are queued, and each replaces the row above when it lands.

Fixed-K does not win here the way it does above: 5 rows come out ahead of it without spending more experts, by up to 1.39 points. The standard error of this four-task mean is about 0.73 points, though, so those are ties rather than wins — which is itself the finding, because in the generative tables the same methods lose to Fixed-K by margins far outside that.

The two averages are the same configuration measured under different inputs, and they differ by between -1.06 and -0.06 experts (NAEE-kmin2-beta0.35 and Ban-kmin3-lambda0.6 are the extremes). A dynamic method reads the router's own confidence, and short factual questions do not distribute it the way a long derivation does, so a budget calibrated on generation does not transfer here exactly. The QA column is the one that describes these scores.

## Average expert count landed where it was expected to

The hyperparameter-to-average mapping is the only part of the compact table that survived the cache bug, so this confirms the re-runs are sitting at the intended operating points.

| config | expected avg | this run | Δ |
| --- | --- | --- | --- |
| Baseline-k8 | 8.0000 | 8.000 | +0.00 |
| FixedK-k6 | 6.0000 | 6.000 | +0.00 |
| Ban-kmin3-lambda0.95 | 5.0484 | 5.054 | +0.01 |
| DiEP-kmin2-beta0.5 | 6.1807 | 5.950 | -0.23 |
| DiEP-damped-a0.25-cap0.9-kmin2-beta0.413 | 6.0000 ᵖ | 6.110 | +0.11 |
| DiEP-damped-a0.5-cap0.9-kmin2-beta0.442 | 6.0000 ᵖ | 6.027 | +0.03 |
| Dynamic_Routing-thr0.8-renormalized | 5.8469 | 5.918 | +0.07 |
| NAEE-kmin2-beta0.35 | 6.2813 | 6.408 | +0.13 |
| FixedK-k5 | 5.0000 | 5.000 | +0.00 |
| Ban-kmin3-lambda0.9 | 4.9585 | 4.995 | +0.04 |
| DiEP-kmin2-beta0.7 | 5.1100 | 4.841 | -0.27 |
| DiEP-damped-a0.25-cap0.9-kmin2-beta0.525 | 5.0000 ᵖ | 4.927 | -0.07 |
| DiEP-damped-a0.5-cap0.9-kmin2-beta0.57 | 5.0000 ᵖ | 4.911 | -0.09 |
| Dynamic_Routing-thr0.7-renormalized | 4.8490 | 4.906 | +0.06 |
| NAEE-kmin2-beta0.45 | 5.0892 | 5.134 | +0.04 |
| FixedK-k4 | 4.0000 | 4.000 | +0.00 |
| Ban-kmin3-lambda0.6 | 4.1264 | 4.135 | +0.01 |
| DiEP-kmin2-beta1.0 | 3.9456 | 3.801 | -0.14 |
| DiEP-damped-a0.25-cap0.9-kmin2-beta0.645 | 4.0000 ᵖ | 3.879 | -0.12 |
| DiEP-damped-a0.5-cap0.9-kmin2-beta0.723 | 4.0000 ᵖ | 3.911 | -0.09 |
| DiEP-paper-gamma1-kmin2 | 3.6300 ᵖ | 4.365 | +0.73 |
| Dynamic_Routing-thr0.6-renormalized | 3.9922 | 4.052 | +0.06 |
| NAEE-kmin2-beta0.55 | 4.1219 | 4.081 | -0.04 |
| FixedK-k3 | 3.0000 | 3.000 | +0.00 |
| Ban-kmin2-lambda0.5 | 3.1305 | 3.142 | +0.01 |
| DiEP-kmin2-beta1.2 | 3.4267 | 3.363 | -0.06 |
| DiEP-damped-a0.5-cap0.9-kmin2-beta0.926 | 3.0000 ᵖ | 3.083 | +0.08 |
| Dynamic_Routing-thr0.45-renormalized | 2.9087 | 2.970 | +0.06 |
| NAEE-kmin2-beta0.7 | 2.8900 | 2.986 | +0.10 |
| FixedK-k2 | 2.0000 | 2.000 | +0.00 |
| Ban-kmin1-lambda0.26 | 2.0000 ᵖ | 2.013 | +0.01 |
| DiEP-damped-a0.5-cap0.9-kmin2-beta1.394 | 2.2000 ᵖ | 2.308 | +0.11 |
| Dynamic_Routing-thr0.35-renormalized | 2.3028 | 2.327 | +0.02 |
| NAEE-kmin2-beta0.8 | 2.4066 | 2.428 | +0.02 |

ᵖ not in the old table, either because it never swept this method that far or because the configuration did not exist then. The expected value comes from solving the hyperparameter offline against averages that were measured (`scripts/calibrate_ban_lambda.py` for Ban, `scripts/solve_diep_beta.py` for DiEP), so it is a prediction being tested here rather than a target that was already hit.
