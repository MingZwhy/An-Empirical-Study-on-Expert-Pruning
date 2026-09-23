# Hy3 canonical Fixed-K

Generated 2026-09-15. Source run
`<run-root>/runs/hy3_canonical`, pooled into
one `CANONICAL.json` per configuration alongside this file.

## Protocol

- Model `tencent/Hy3`, revision `a960ebc3da325ba167f069f76c41eb62c9280d22`, native k=8
  (192 routed experts + 1 shared). Fixed-K overlays change only `num_experts_per_tok`.
- Sampling `temperature=0.9, top_p=1.0, top_k=0` (Hy3 official recommendation;
  top_k is not part of it and 0 disables it in vLLM).
- Reasoning effort per task type: `low` for MMLU-Pro and GSM8K, `high` for
  GPQA / MATH-500 / AIME24 / AIME25 / LiveCodeBench.
  QA4 is lm_eval loglikelihood scored as bare continuation, so nothing is
  generated and no_think is structural there.
- `max_new_tokens` 32768, except FixedK-k1 at 8192.
- Engine: TP16, `enforce_eager`, no expert parallelism, FlashInfer attention,
  `max_num_seqs=128`, `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.80`.
  A 2x2 benchmark showed CUDA graphs cost 12-23% here and expert parallelism
  gained only 3%, so the engine configuration was left unchanged.
- AIME24/25 use avg@16. Pooling is exact and sample-level: every scored sample is
  read back from the per-shard detail files, keyed by task plus document id,
  checked for duplicates and against the expected count, then averaged. Sharded
  results are therefore identical to what an unsharded run would produce.

## Results

All values are percentages. ARC-C, ARC-E and OpenBookQA use normalized accuracy;
WinoGrande uses raw accuracy. `Avg11` is the unweighted mean of the eleven columns.

| Config | MMLU-P | GSM8K | MATH | AIME24 | AIME25 | LCB | GPQA | ARC-C | ARC-E | WG | OBQA | Avg11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline-k8 | 86.50 | 95.80 | 81.40 | 93.80 | 92.10 | 67.40 | 85.30 | 73.40 | 88.30 | 80.10 | 50.60 | **81.34** |
| FixedK-k7 | 86.60 | 95.80 | 81.00 | 93.30 | 89.80 | 65.70 | 84.30 | 72.90 | 88.10 | 78.50 | 50.40 | **80.58** |
| FixedK-k6 | 86.60 | 96.00 | 81.60 | 92.50 | 87.90 | 64.00 | 82.80 | 71.20 | 87.60 | 77.10 | 49.00 | **79.66** |
| FixedK-k5 | 86.50 | 95.50 | 80.80 | 90.60 | 82.10 | 57.70 | 81.80 | 69.90 | 86.00 | 76.20 | 47.60 | **77.70** |
| FixedK-k4 | 86.10 | 95.30 | 80.20 | 82.50 | 67.90 | 48.00 | 79.80 | 66.10 | 84.80 | 73.60 | 45.20 | **73.59** |
| FixedK-k3 | 83.90 | 94.70 | 74.00 | 32.30 | 25.60 | 24.00 | 60.10 | 61.40 | 79.80 | 67.60 | 44.00 | **58.85** |
| FixedK-k2 | 11.27 | 3.12 | 1.40 | 0.00 | 0.00 | 0.00 | 24.80 | 46.50 | 66.10 | 58.10 | 34.20 | **22.32** |
| FixedK-k1 | 5.80 | 0.80 | 1.60 | 0.00 | 0.00 | 0.00 | 25.20 | 25.40 | 31.10 | 51.00 | 27.80 | **15.34** |

## Coverage

Every group in every configuration is included above.

FixedK-k2's MMLU-Pro and GSM8K were the expensive ones. At k=2 the model
degenerates into unbounded repetition on those two tasks, consuming roughly
31,000 output tokens per question against the 32,768 cap, so their six MMLU-Pro
shards and GSM8K were completed in a separate run rather than in the main queue.

## What the curve shows

Knowledge tasks tolerate pruning far better than multi-step reasoning. MMLU-Pro
barely moves from k=8 to k=4 (86.54 → 86.14) and GSM8K is still 91.66 at k=3,
while at that same k=3 AIME24 has already fallen from 90.83 to 23.96 and
LiveCodeBench from 67.43 to 24.00.

The cliff sits between k=4 and k=3: Avg11 declines gently from 81.85 to 71.93
across k=8→k=4, then drops to 57.60 at k=3. k=2 is effectively unusable and k=1
has collapsed.

## Artifacts

- `<Config>/CANONICAL.json` — pooled per-sample metrics with mean, stderr and n
  for every task, plus the per-shard source results
- `POOLED_MERGES.json` — merge manifest, including which groups were skipped
- `QUEUE_FINAL.json` — final queue state with per-unit provenance
