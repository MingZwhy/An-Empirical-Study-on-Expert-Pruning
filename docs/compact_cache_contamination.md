# Cache-reuse damage in the compact tables

The compact tables predate keying the lighteval sample cache on the expert pruning configuration, so runs replayed each other's samples. Identical scores across configurations that should differ are the fingerprint of that. Only the mapping from hyperparameters to average expert count survives; the scores do not.

## Qwen3-30B-A3B-Instruct-2507

49 distinct configurations.

| dataset | duplicated cells | most repeated value | shared by |
| --- | --- | --- | --- |
| mmlu_pro | 2/49 | 73.95 | 2 configurations |
| gpqa | 32/49 | 50.00 | 4 configurations |
| math_500 | 22/49 | 91.00 | 4 configurations |
| aime24 | 41/49 | 66.72 | 13 configurations |
| aime25 | 41/49 | 50.31 | 13 configurations |
| lcb | 45/49 | 41.14 | 18 configurations |
| gsm8k | 17/49 | 94.47 | 4 configurations |
| hellaswag | 41/49 | 87.43 | 13 configurations |
| livebench | 44/49 | 92.00 | 15 configurations |

Overall **285/441 (65%)** of the score cells are exact duplicates of another configuration's.

Worst single case: `lcb` = 41.14 appears for 18 configurations whose average expert counts span 2.30–7.24:

- MC-MoE (beta=0.3, k_min=2, p=0.2) — avg 7.24
- MC-MoE (beta=0.4, k_min=2, p=0.2) — avg 6.17
- DiEP (beta=0.6, k_min=2) — avg 5.54
- DiEP (beta=0.7, k_min=2) — avg 5.11
- Ban (lambda=0.95, k_min=3) — avg 5.05
- MC-MoE (beta=0.5, k_min=2, p=0.2) — avg 4.97
- DiEP (beta=0.85, k_min=2) — avg 4.47
- MC-MoE (beta=0.55, k_min=2, p=0.2) — avg 4.45
- DiEP (beta=1.0, k_min=2) — avg 3.95
- MC-MoE (beta=0.6, k_min=2, p=0.2) — avg 3.94
- Ban (lambda=0.6, k_min=2) — avg 3.53
- DiEP (beta=1.2, k_min=2) — avg 3.43
- Ban (lambda=0.5, k_min=2) — avg 3.13
- MC-MoE (beta=0.7, k_min=2, p=0.2) — avg 3.11
- NAEE (beta=0.7, k_min=2) — avg 2.89
- MC-MoE (beta=0.8, k_min=2, p=0.2) — avg 2.63
- Dynamic Routing (threshold=0.4, score_source=renormalized) — avg 2.63
- Dynamic Routing (threshold=0.35, score_source=renormalized) — avg 2.30

## gpt-oss-20b

26 distinct configurations.

| dataset | duplicated cells | most repeated value | shared by |
| --- | --- | --- | --- |
| mmlu_pro | 2/26 | 74.46 | 2 configurations |
| gpqa | 13/26 | 65.66 | 4 configurations |
| math_500 | 13/26 | 91.20 | 3 configurations |
| aime24 | 24/26 | 76.30 | 9 configurations |
| aime25 | 24/26 | 71.93 | 9 configurations |
| lcb | 24/26 | 61.14 | 7 configurations |
| gsm8k | 9/26 | 85.97 | 3 configurations |
| hellaswag | 24/26 | 27.81 | 9 configurations |
| livebench | 25/26 | 72.00 | 9 configurations |

Overall **158/234 (68%)** of the score cells are exact duplicates of another configuration's.

Worst single case: `aime24` = 76.30 appears for 9 configurations whose average expert counts span 1.95–3.84:

- DiEP (beta=0.3, k_min=1) — avg 3.84
- NAEE (beta=0.35, k_min=1) — avg 3.71
- NAEE (beta=0.4, k_min=1) — avg 3.57
- DiEP (beta=0.6, k_min=1) — avg 3.50
- NAEE (beta=0.55, k_min=1) — avg 2.93
- Ban (lambda=0.9, k_min=1) — avg 2.49
- NAEE (beta=0.65, k_min=1) — avg 2.45
- Ban (lambda=0.55, k_min=1) — avg 2.00
- Dynamic Routing (threshold=0.5, score_source=renormalized) — avg 1.95

## Ling-lite-1.5-2507

42 distinct configurations.

| dataset | duplicated cells | most repeated value | shared by |
| --- | --- | --- | --- |
| mmlu_pro | 2/42 | 69.41 | 2 configurations |
| gpqa | 21/42 | 56.57 | 3 configurations |
| math_500 | 13/42 | 88.40 | 3 configurations |
| aime24 | 38/42 | 34.38 | 19 configurations |
| aime25 | 38/42 | 24.43 | 19 configurations |
| lcb | 40/42 | 30.29 | 19 configurations |
| gsm8k | 8/42 | 91.43 | 2 configurations |
| hellaswag | 38/42 | 61.94 | 19 configurations |
| livebench | 38/42 | 30.67 | 19 configurations |

Overall **236/378 (62%)** of the score cells are exact duplicates of another configuration's.

Worst single case: `aime24` = 34.38 appears for 19 configurations whose average expert counts span 2.57–5.14:

- NAEE (beta=0.3, k_min=2) — avg 5.14
- DiEP (beta=0.3, k_min=2) — avg 5.12
- Dynamic Routing (threshold=0.8, score_source=renormalized) — avg 4.61
- DiEP (beta=0.4, k_min=2) — avg 4.47
- NAEE (beta=0.4, k_min=2) — avg 4.41
- Dynamic Routing (threshold=0.75, score_source=renormalized) — avg 4.19
- DiEP (beta=0.45, k_min=2) — avg 4.15
- NAEE (beta=0.45, k_min=2) — avg 4.06
- Ban (lambda=0.9, k_min=2) — avg 3.98
- DiEP (beta=0.55, k_min=2) — avg 3.56
- Ban (lambda=0.7, k_min=2) — avg 3.52
- Dynamic Routing (threshold=0.65, score_source=renormalized) — avg 3.47
- NAEE (beta=0.55, k_min=2) — avg 3.40
- Dynamic Routing (threshold=0.6, score_source=renormalized) — avg 3.15
- NAEE (beta=0.6, k_min=2) — avg 3.11
- DiEP (beta=0.65, k_min=2) — avg 3.08
- Ban (lambda=0.45, k_min=2) — avg 3.00
- NAEE (beta=0.7, k_min=2) — avg 2.61
- Dynamic Routing (threshold=0.5, score_source=renormalized) — avg 2.57

## Qwen3-Next-80B-A3B-Instruct

56 distinct configurations.

| dataset | duplicated cells | most repeated value | shared by |
| --- | --- | --- | --- |
| mmlu_pro | 12/56 | 82.06 | 2 configurations |
| gpqa | 45/56 | 72.22 | 7 configurations |
| math_500 | 36/56 | 89.80 | 6 configurations |
| aime24 | 49/56 | 79.22 | 23 configurations |
| aime25 | 49/56 | 65.52 | 23 configurations |
| lcb | 51/56 | 48.57 | 24 configurations |
| gsm8k | 30/56 | 95.07 | 7 configurations |
| hellaswag | 47/56 | 35.29 | 23 configurations |
| livebench | 52/56 | 86.67 | 23 configurations |

Overall **371/504 (74%)** of the score cells are exact duplicates of another configuration's.

Worst single case: `lcb` = 48.57 appears for 24 configurations whose average expert counts span 3.01–8.19:

- DiEP (beta=0.3, k_min=3) — avg 8.19
- DiEP (beta=0.4, k_min=3) — avg 7.44
- DiEP (beta=0.45, k_min=3) — avg 7.10
- Dynamic Routing (threshold=0.8, score_source=renormalized) — avg 7.06
- DiEP (beta=0.55, k_min=3) — avg 6.47
- NAEE (beta=0.35, k_min=2) — avg 6.46
- Dynamic Routing (threshold=0.75, score_source=renormalized) — avg 6.36
- DiEP (beta=0.65, k_min=3) — avg 5.93
- Dynamic Routing (threshold=0.7, score_source=renormalized) — avg 5.74
- NAEE (beta=0.4, k_min=2) — avg 5.65
- Ban (lambda=0.8, k_min=3) — avg 5.41
- Dynamic Routing (threshold=0.65, score_source=renormalized) — avg 5.15
- Ban (lambda=0.65, k_min=3) — avg 5.04
- NAEE (beta=0.45, k_min=2) — avg 5.00
- Fixed-K (k=5) — avg 5.00
- Dynamic Routing (threshold=0.6, score_source=renormalized) — avg 4.61
- Ban (lambda=0.5, k_min=3) — avg 4.49
- NAEE (beta=0.5, k_min=2) — avg 4.37
- Ban (lambda=0.45, k_min=3) — avg 4.24
- Dynamic Routing (threshold=0.55, score_source=renormalized) — avg 4.12
- NAEE (beta=0.55, k_min=2) — avg 3.85
- Dynamic Routing (threshold=0.5, score_source=renormalized) — avg 3.66
- NAEE (beta=0.6, k_min=2) — avg 3.37
- NAEE (beta=0.65, k_min=2) — avg 3.01

