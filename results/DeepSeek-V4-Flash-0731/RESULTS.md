# DeepSeek-V4-Flash-0731 canonical Fixed-K

## Protocol

- Checkpoint `deepseek-ai/DeepSeek-V4-Flash-0731`, revision
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, native FP8 e4m3 weights with block
  scaling [128,128]. Native k=6. Fixed-K overlays change only the per-token expert
  count; the checkpoint itself is never modified.
- Engine: TP4 / EP4, `max_num_seqs=128`, `max_num_batched_tokens=2048`,
  `max_model_len=32768`, `gpu_memory_utilization=0.85`, FP8 KV cache,
  `block_size=256`, enforce-eager, chunked prefill, seed 42.
- Sampling: `temperature=0.7, top_p=0.8, top_k=20`.
- Reasoning protocol: the checkpoint ships no chat template, so the official
  `encoding/encoding_dsv4.py` encoder is hooked through `apply_chat_template`.
  `no_think` maps to `thinking_mode=chat`; `low` and `high` map to
  `thinking_mode=thinking` with the matching `reasoning_effort`.
- Effort per task: `low` for MMLU-Pro and GSM8K, `high` for GPQA, MATH-500, LCB,
  AIME24 and AIME25, `no_think` for the four likelihood-scored QA sets.
- `max_new_tokens` 32768, except FixedK-k1 at 8192.

Generated: 2026-09-15T01:51:30.745435+00:00


## Results

All values are percentages. `Avg11` is the unweighted mean of the eleven
benchmark columns. ARC-C, ARC-E, and OpenBookQA use normalized accuracy; WinoGrande uses
raw accuracy.

| Config | K | ARC-C | ARC-E | WG | OBQA | MATH | AIME24 | AIME25 | GSM8K | LCB | MMLU-P | GPQA | Avg11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline-k6 | 6 | 67.30 | 87.80 | 78.80 | 47.80 | 88.80 | 92.10 | 91.70 | 75.00 | 29.70 | 86.30 | 87.40 | 75.70 |
| FixedK-k5 | 5 | 66.20 | 86.80 | 78.10 | 48.80 | 88.00 | 92.90 | 93.50 | 76.80 | 32.00 | 86.70 | 86.90 | **76.06** |
| FixedK-k4 | 4 | 62.20 | 85.20 | 75.00 | 46.00 | 85.80 | 92.10 | 92.90 | 75.70 | 28.60 | 86.50 | 89.90 | 74.54 |
| FixedK-k3 | 3 | 58.00 | 81.70 | 70.80 | 44.60 | 85.40 | 91.00 | 89.20 | 81.70 | 29.10 | 85.30 | 86.40 | 73.02 |
| FixedK-k2 | 2 | 48.80 | 70.50 | 63.10 | 37.80 | 77.80 | 74.40 | 68.50 | 88.70 | 44.00 | 75.30 | 72.70 | 65.60 |
| FixedK-k1 | 1 | 27.60 | 39.30 | 52.30 | 27.20 | 1.20 | 0.00 | 0.00 | 1.70 | 0.00 | 6.10 | 12.60 | 15.27 |

## Canonical artifacts

- Baseline-k6: `Baseline-k6/CANONICAL.json` — `15c8691305f7a2105dc2c897e407626911dee899cb52feecbabbee7d770b25ea`
- FixedK-k5: `FixedK-k5/CANONICAL.json` — `91f13671cded8b68e55d3d717c6d931918f9b6f78cc93ddfa22ed7e9c6cd1db9`
- FixedK-k4: `FixedK-k4/CANONICAL.json` — `ef11d42f1ef6f2b0dba1232a53d38331f2857ffb6e1ef23ad9ae9c95af9b5e1d`
- FixedK-k3: `FixedK-k3/CANONICAL.json` — `d519a8992c62adbd9f14a8424267ff51e2bf139113b8d7fb4a24a751079dc7ef`
- FixedK-k2: `FixedK-k2/CANONICAL.json` — `a1cb4db9a7cbfa16904a213da09a82f57bcca5b143aeb6241bc9faa80c572f74`
- FixedK-k1: `FixedK-k1/CANONICAL.json` — `7e08207ac5851b28d63df065de20794a5a45c3a89c4b28d968dcaa7e60e23ff6`
