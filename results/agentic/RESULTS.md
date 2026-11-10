# Fixed-K on agentic benchmarks

Uniform truncation (Fixed-K) on seven agentic benchmarks with five fine-grained MoE models, run after the paper was
submitted. As in the main study, the checkpoints are unmodified and only the number of experts activated per token
changes. Agentic tasks chain many model calls, each conditioned on tool output, so a small loss per call can compound
over a trajectory.

## Setup

| | |
|---|---|
| Models | DeepSeek-V4-Flash (K = 6) and Gemma 4 26B-A4B (K = 8) from the main study; Qwen3.6-35B-A3B (K = 8), Qwen3.8-Flash-Next (K = 10) and GLM-5.3-Flash (K = 8) |
| SWE-bench Verified | 500 tasks with SWE-agent, one attempt per task (a 304-task sample for Qwen3.6-35B-A3B) |
| SWE-bench Pro V2 | 642 tasks with SWE-agent, one attempt per task (DeepSeek-V4-Flash's runs cover 633) |
| τ³-bench | airline, retail, telecom and banking-knowledge domains (three domains for DeepSeek-V4-Flash), with Qwen3.8-27B as user simulator and judge |
| Terminal-Bench 2.0 | 89 tasks with the Terminus-2 agent, three attempts each |
| DABstep | 450 data-analysis questions with the Terminus-2 agent |
| FeatureBench | 142 feature-implementation tasks scored by hidden tests, with mini-swe-agent |
| SkillsBench | 77 tasks, four attempts each |

Models are served with vLLM, with thinking enabled and each model's recommended sampling. Fixed-K is applied through
the same `num_experts_per_tok` overlay as in the main study; the Gemma 4 DABstep runs and the FeatureBench runs use an
equivalent top-k override in the server. Every pruned run is paired by task with an unpruned run from the same
session. Standard errors of the paired differences are mostly 1.5 to 2.6 points, and repeated unpruned runs differ by
about as much: DeepSeek-V4-Flash scored 86.6 and 89.4 on SWE-bench Verified in two unpruned runs.

## Results

Scores in percent at k experts per token; K is the native count. A dash means the budget was not run. Cells marked †
are the unpruned score plus the paired difference to the same-session unpruned run, which is how those benchmarks were
analysed.

### DeepSeek-V4-Flash (K = 6)

| Benchmark | Unpruned | k = 5 | k = 4 | k = 3 |
|---|---:|---:|---:|---:|
| SWE-bench Verified | 89.4 | 89.8 | 88.6 | 86.6 |
| SWE-bench Pro V2 | 91.3 | 91.5 | 93.4 | 90.7 |
| τ³-bench | 89.7 | 90.5 | 90.1 | 89.0 |
| Terminal-Bench 2.0 | 75.3 | 74.2† | 73.8† | 68.6† |

### Gemma 4 26B-A4B (K = 8)

| Benchmark | Unpruned | k = 7 | k = 6 | k = 5 | k = 4 | k = 3 |
|---|---:|---:|---:|---:|---:|---:|
| SWE-bench Verified | 54.4 | 54.6 | 52.4 | 51.8 | 41.0 | 24.4 |
| SWE-bench Pro V2 | 49.2 | 49.1 | 44.1 | 42.5 | 34.9 | 17.3 |
| τ³-bench | 58.5 | 56.9 | 57.5 | 56.0 | 54.6 | 43.4 |
| DABstep | 32.9 | – | 33.8† | 29.8† | 23.6† | – |

### Qwen3.6-35B-A3B (K = 8)

| Benchmark | Unpruned | k = 7 | k = 6 | k = 5 | k = 4 | k = 3 |
|---|---:|---:|---:|---:|---:|---:|
| SWE-bench Verified | 68.1 | 72.0† | 72.0† | 68.1† | – | – |
| τ³-bench | 67.8 | 68.1 | 66.5 | 66.4 | 63.6 | 32.3 |
| Terminal-Bench 2.0 | 53.9 | 51.3† | 50.0† | 52.0† | 48.3† | 20.4† |
| SkillsBench | 26.9 | 21.0† | 22.4† | 16.7† | – | – |
| DABstep | 37.6 | 38.0† | 37.6† | 36.3† | 34.0† | 28.6† |

### Qwen3.8-Flash-Next (K = 10)

| Benchmark | Unpruned | k = 8 | k = 6 | k = 5 | k = 4 | k = 3 |
|---|---:|---:|---:|---:|---:|---:|
| SWE-bench Verified | 79.2 | 79.8 | 76.4 | 72.4 | 66.2 | 0.6 |
| SWE-bench Pro V2 | 92.7 | 91.4 | 89.7 | 86.8 | 76.6 | 0.5 |
| τ³-bench | 77.9 | 76.2 | 74.2 | 69.2 | 48.8 | 0.4 |
| Terminal-Bench 2.0 | 88.4 | 88.8† | 82.8† | 73.8† | 36.0† | – |
| DABstep | 59.8 | 54.0† | 48.2† | 45.8† | 39.4† | – |
| FeatureBench | 79.8 | 80.3† | 78.5† | 79.9† | 67.6† | – |

### GLM-5.3-Flash (K = 8)

| Benchmark | Unpruned | k = 7 | k = 6 | k = 5 | k = 4 | k = 3 |
|---|---:|---:|---:|---:|---:|---:|
| Terminal-Bench 2.0 | 80.1 | 78.2† | 76.2† | 73.0† | 59.5† | 7.9† |
| DABstep | 54.0 | 54.4† | 49.8† | 48.2† | 38.9† | – |

## Summary

Near ⌈2K/3⌉ the 21 model-benchmark pairs keep 96.0 % of the unpruned score on average
(median 96.8 %), counting Qwen3.8-Flash-Next at k = 6, below its ⌈2K/3⌉ = 7; counting it at
k = 8 instead gives 97.3 %. This is about two points below the eleven benchmarks of the main
study. At K/2 the 19 pairs with that budget keep 87.0 % on average. Two pairs lose
more than 10 % near ⌈2K/3⌉: SkillsBench on Qwen3.6-35B-A3B and DABstep on Qwen3.8-Flash-Next. At k = 3,
Qwen3.8-Flash-Next fails nearly every trial on output format.
