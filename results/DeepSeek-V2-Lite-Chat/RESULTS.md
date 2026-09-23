# DeepSeek-V2-Lite-Chat — expert-pruning results

6 of 6 configurations finished.

Every table opens with two reference rows: the unpruned model (its native top-k of 6) and the Fixed-K point for that target. Fixed-K is what a method has to beat, since it reaches the same average expert count by simply keeping fewer experts per token. The bracketed number after each mean is the gap to the unpruned model.

Settings: five generative datasets (AIME24/25 omitted — this model does not solve them), temperature 0.7 / top_p 0.8 / top_k 20, max_model_len and max_new_tokens 32768 unless a row is marked ᵏ, TP=2. Every run has its own cache namespace keyed on the pruning configuration, so no two rows here can share samples.

## Results by target average expert count

### Target avg ≈ 5.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 | 6.000 | **30.82** | 27.4 | 28.3 | 25.0 | 10.3 | 63.1 |
| Fixed-K | k=5 | 5.000 | **31.14** (+0.32) | 28.2 | 30.3 | 24.6 | 10.9 | 61.7 |

### Target avg ≈ 4.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 | 6.000 | **30.82** | 27.4 | 28.3 | 25.0 | 10.3 | 63.1 |
| Fixed-K | k=4 | 4.000 | **30.74** (-0.08) | 26.9 | 30.8 | 24.0 | 10.3 | 61.7 |

### Target avg ≈ 3.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 | 6.000 | **30.82** | 27.4 | 28.3 | 25.0 | 10.3 | 63.1 |
| Fixed-K | k=3 | 3.000 | **27.94** (-2.88) | 24.3 | 26.3 | 26.2 | 6.9 | 56.1 |

### Target avg ≈ 2.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 | 6.000 | **30.82** | 27.4 | 28.3 | 25.0 | 10.3 | 63.1 |
| Fixed-K | k=2 | 2.000 | **22.95** (-7.87) | 18.8 | 25.8 | 16.6 | 5.7 | 47.8 |

### Target avg ≈ 1.0

| method | hyperparams | avg experts ᵃ | mean | mmlu_pro | gpqa | math500 | lcb | gsm8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 | 6.000 | **30.82** | 27.4 | 28.3 | 25.0 | 10.3 | 63.1 |
| Fixed-K | k=1 | 1.000 | **11.11** (-19.71) | 11.1 | 25.3 | 3.2 | 1.1 | 14.9 |

ᵃ this model keeps 2 shared experts active for every token on top of the routed ones. The average counts routed experts only, matching the old table, so an average of 3 here costs 5 experts of compute and is not comparable to an average of 3 on a model without shared experts.

## 0-shot QA (lm-eval)

Four multiple-choice sets, scored by likelihood over the options with nothing generated, so they answer a different question from the tables above: how much of the model's knowledge survives pruning, as opposed to how much of its reasoning does. A budget that ruins the generative scores can leave these nearly intact, and the gap between the two is the interesting part.

The thresholds below are not the ones in the tables above, and that is the point. A threshold is not a budget: it admits however many experts the router's own confidence lets it, and the router is more decided about a one-sentence question than about the middle of a derivation. The generative settings therefore spend fewer experts here than the tier they were calibrated for, and a row that is both cheaper than Fixed-K and worse than it answers nothing. Each method was re-solved against this distribution (`scripts/solve_qa_hyper.py`) so that its average sits at or just above Fixed-K's.

`acc` is used for all four columns because it is the only metric all four report — winogrande has no `acc_norm`. The result files carry `acc_norm` for the other three, and taking it where available moves rows by up to three places, so a comparison against a published table should be read off the JSON with that table's own metric.

### Target avg ≈ 5.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 | 6.000 | 6.000 | **59.65** | 50.9 | 80.1 | 71.4 | 36.2 |
| Fixed-K | k=5 | 5.000 | 5.000 | **59.04** (-0.61) | 50.3 | 79.4 | 70.8 | 35.6 |

### Target avg ≈ 4.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 | 6.000 | 6.000 | **59.65** | 50.9 | 80.1 | 71.4 | 36.2 |
| Fixed-K | k=4 | 4.000 | 4.000 | **58.69** (-0.97) | 50.3 | 79.1 | 70.0 | 35.4 |

### Target avg ≈ 3.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 | 6.000 | 6.000 | **59.65** | 50.9 | 80.1 | 71.4 | 36.2 |
| Fixed-K | k=3 | 3.000 | 3.000 | **57.16** (-2.49) | 47.6 | 77.4 | 69.9 | 33.8 |

### Target avg ≈ 2.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 | 6.000 | 6.000 | **59.65** | 50.9 | 80.1 | 71.4 | 36.2 |
| Fixed-K | k=2 | 2.000 | 2.000 | **53.22** (-6.43) | 43.1 | 73.4 | 67.6 | 28.8 |

### Target avg ≈ 1.0

| method | hyperparams | avg experts (QA) | avg (gen) | mean | arc-c | arc-e | winogrande | obqa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | k=6 | 6.000 | 6.000 | **59.65** | 50.9 | 80.1 | 71.4 | 36.2 |
| Fixed-K | k=1 | 1.000 | 1.000 | **46.07** (-13.58) | 33.6 | 63.7 | 61.8 | 25.2 |

The QA average expert count is blank for this model for the same reason its generative rows take theirs from a probe: expert selection is inlined into the compiled region, and a run that keeps CUDA graphs cannot count it. The generative probe's value is not substituted here, because the number below would then describe a different input distribution from the scores next to it.
