# Native router distribution observer

Observe Qwen3 MoE **native** routing softmax distributions without changing
selected experts or model outputs.

## What it records

Per MoE layer and stage (`prefill`, `decode`, `unknown`):

- Online aggregates over **all** routed tokens in the sampled prompt subset
- Mean sorted 128-probability curve (+ variance)
- Cumulative mass at ranks 1/2/4/8/16/32/64
- Normalized entropy, effective experts `exp(H)`, HHI (`sum p²`), top1−top2 margin
- Histograms for top1 / entropy / margin
- Deterministic reservoir of full sorted vectors (`float16`), default 0.5% rate,
  cap 2048 vectors per layer/stage/worker

## Output layout

Each vLLM worker writes atomically under `--router_distribution_dir`:

```
router_distribution_<pid>.json   # compact online summary
router_distribution_<pid>.npz    # arrays + reservoir samples
```

JSON schema (`schema_version=1`):

```json
{
  "dataset": "arc_challenge",
  "task": "arc_challenge",
  "harness": "lm_eval",
  "model_path": "...",
  "num_experts": 128,
  "layers": {
    "0": {
      "prefill": {
        "token_count": 12345,
        "sorted_prob_mean": [128 floats],
        "cum_mass_mean": {"1": 0.31, "8": 0.92},
        "entropy_mean": 1.42,
        "effective_experts_mean": 4.14,
        "hhi_mean": 0.18,
        "top1_top2_margin_mean": 0.07
      }
    }
  }
}
```

## Storage estimate (defaults)

For 48 layers × 3 stages × 128 experts:

- Online JSON aggregates: ~200 KB per worker snapshot
- NPZ reservoir (worst case): `48 × 3 × 2048 × 128 × 2 bytes ≈ 72 MiB` per worker
- Typical run (0.5% sampling): ~360 KiB reservoir per worker

Never stores all token logits.

## Commands

### Text QA (Qwen3-30B-A3B-Instruct, 4 tasks × 256 samples)

```bash
conda activate expertpruning
python main.py \
  --harness lm_eval \
  --lm_eval_tasks arc_challenge \
  --model_path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tensor_parallel_size 2 \
  --max_samples 256 \
  --enforce_eager \
  --collect_router_distribution \
  --output_dir results/router_distribution/Qwen3-30B/arc_challenge
```

Repeat once per task (`arc_easy`, `winogrande`, `openbookqa`).

### VL multimodal (Qwen3-VL-30B-A3B-Instruct, 9 tasks × 128 samples)

```bash
conda activate qwen35
python main.py \
  --harness lmms_eval \
  --lmms_eval_tasks chartqa_lite \
  --model_path /path/to/Qwen3-VL-30B-A3B-Instruct \
  --tensor_parallel_size 2 \
  --max_samples 128 \
  --enforce_eager \
  --collect_router_distribution \
  --output_dir results/router_distribution/Qwen3-VL/chartqa_lite
```

### 8-GPU sweep (4 lanes × TP=2)

```bash
HARNESS=lm_eval MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  bash scripts/run_router_distribution_sweep.sh
```

### Merge task runs

```bash
python scripts/merge_router_distribution.py \
  results/router_distribution/Qwen3-30B/*/ \
  --output-dir results/router_distribution/Qwen3-30B/merged \
  --label Qwen3-30B
```

Produces `*_summary.json`, `*_summary.md`, `*_summary.csv`, and optional
`*_reservoirs.npz`. Raw per-task files are preserved.

## Constraints

- Incompatible with `--use_local_expert_router`, non-`none` pruning, Fixed-K
  (`--num_experts_per_tok`), or `--attention_sink_probe`
- Automatically enables `--enforce_eager`
- Records on TP rank 0 only
- Skips vLLM dummy/profile runs
- Multimodal token-type tags are **not** emitted unless reliable; dataset/stage
  labels are always recorded

## Smoke test (no model)

```bash
pytest tests/test_router_distribution.py -q
python -m compileall expert_pruning/router_distribution.py expert_pruning/vllm_patch.py main.py
```
