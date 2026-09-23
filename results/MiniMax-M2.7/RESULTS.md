# MiniMax-M2.7 canonical Fixed-K

## Protocol

- Native k=8. Fixed-K overlays change only the per-token expert count; the
  checkpoint itself is never modified.
- Sampling `temperature=0.7, top_p=0.8, top_k=20`; `max_new_tokens` 32768.
- AIME24/25 use avg@16. The four likelihood-scored QA sets are scored as bare
  continuations, so nothing is generated for them.

## Results

All values are percentages. `Avg11` is the unweighted mean of the eleven
benchmark columns.

| Config | K | MMLU-P | GSM8K | MATH | AIME24 | AIME25 | LCB | GPQA | ARC-C | ARC-E | WG | OBQA | Avg11 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline-k8 | 8 | 82.00 | 91.60 | 90.20 | 81.50 | 81.70 | 47.30 | 85.40 | 70.10 | 86.80 | 76.20 | 38.80 | 75.60 |
| FixedK-k7 | 7 | 81.90 | 92.00 | 89.60 | 80.80 | 82.10 | 44.00 | 87.90 | 69.70 | 86.30 | 74.60 | 37.80 | **75.15** |
| FixedK-k6 | 6 | 81.50 | 91.50 | 88.80 | 78.30 | 80.60 | 42.30 | 84.30 | 69.20 | 85.60 | 74.40 | 37.40 | **73.99** |
| FixedK-k5 | 5 | 80.90 | 91.70 | 89.00 | 73.20 | 75.80 | 44.60 | 84.30 | 66.30 | 84.30 | 70.00 | 37.00 | **72.46** |
| FixedK-k4 | 4 | 78.70 | 90.10 | 85.40 | 54.20 | 42.90 | 33.10 | 78.80 | 63.10 | 82.90 | 68.00 | 35.80 | **64.82** |
| FixedK-k3 | 3 | 59.40 | 83.60 | 65.80 | 4.20 | 10.80 | 9.10 | 51.00 | 52.30 | 75.90 | 60.50 | 32.60 | **45.93** |
| FixedK-k2 | 2 | 9.40 | 2.30 | 3.00 | 0.00 | 0.00 | 0.00 | 24.20 | 34.90 | 53.80 | 53.20 | 22.20 | **18.45** |
| FixedK-k1 | 1 | 5.20 | 2.00 | 1.40 | 0.00 | 0.00 | 0.00 | 23.20 | 20.60 | 26.10 | 49.60 | 15.60 | **13.06** |

The per-configuration metric dumps this table was built from, with the raw
artifact paths and checksums, are kept alongside the run rather than here.
