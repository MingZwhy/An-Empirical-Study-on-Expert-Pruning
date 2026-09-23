# Hyperparameters

`main.py` accepts about 160 flags. Roughly fifteen of them matter for ordinary
use; the rest are either engine plumbing you set once for your machine or knobs
belonging to research variants that did not make the paper.

This page covers the ones you would actually reach for. `python main.py --help`
prints all of them.

- [The budget knob](#the-budget-knob)
- [Choosing a rule](#choosing-a-rule)
- [Matching budgets, which is the whole trick](#matching-budgets-which-is-the-whole-trick)
- [What to evaluate](#what-to-evaluate)
- [Decoding](#decoding)
- [Engine and hardware](#engine-and-hardware)
- [Output and caching](#output-and-caching)
- [Checking that pruning actually happened](#checking-that-pruning-actually-happened)
- [Research variants](#research-variants)

## The budget knob

Everything in this study changes one quantity: how many experts a token is
allowed to use. Uniformly, that is a single integer.

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--num_experts_per_tok K` | int | model's own | Overrides `num_experts_per_tok` in the checkpoint's config. The model's own router still ranks the experts; only the cut-off moves. |

```bash
# keep the 6 highest-scoring experts of the 8 this model would normally use
python main.py --model_path $M --datasets gsm8k --num_experts_per_tok 6
```

This is Fixed-K, and it is worth being clear about how little it does. There is
no engine patch, no kernel change, no calibration, and no weight surgery: the
config is overlaid in a scratch directory, the model loads with a smaller `k`,
and its router does what it always did. That is why Fixed-K runs on any MoE the
engine can load, including architectures released long after the pinned engine
in the other environment.

It requires a **local checkpoint directory**, since there has to be a config to
overlay. A bare Hugging Face model id will not do.

## Choosing a rule

Everything other than Fixed-K decides `k` per token, which means intercepting
expert selection rather than editing a config. Two flags switch that on:

| Flag | What it does |
|---|---|
| `--use_local_expert_router` | Replaces vLLM's fused top-k routing with this repository's implementation. Required by every rule below. Without it the rule flags are silently inert. |
| `--expert_pruning_method NAME` | Which rule. `none` (default), `NAEE`, `Dynamic_Routing`, `DiEP`, `MC_MoE`, `EAC_MoE`, `Ban`, plus the research variants. |
| `--enforce_eager` *or* `--allow_compiled_router` | One of the two is mandatory whenever the router patch is on. See below. |

This only works on architectures the patch recognises: Qwen3-30B-A3B,
Qwen3-Next-80B, Ling-lite-1.5 and GPT-OSS-20B. On anything else it refuses to
start rather than leaving the model quietly unpruned.

### Why the router patch needs an execution mode

Expert selection is inlined into the region `torch.compile` handles, and the
counters that record how many experts each token kept are traced away with it.
Pruning still applies — the generated text differs from the unpruned model either
way — but the average expert count becomes unobservable, and that average is the
axis every comparison in this study is matched on. Rather than report a budget it
cannot stand behind, `main.py` stops:

```
RuntimeError: The expert router reported no statistics although the model generated
4 samples, so average_selected_experts is unavailable ... Re-run with --enforce_eager.
```

So pick one deliberately:

- **`--enforce_eager`** for unquantized models. On Qwen3-30B-A3B it costs
  essentially nothing — 69 s against 70 s on a four-sample run — so this is the
  default the sweep scripts use.
- **`--allow_compiled_router`** for quantized models, where eager is punitive:
  6.7× on gpt-oss, 166 against 1138 tok/s, the difference between a day and two
  weeks. The score then comes from the graph run and the expert count from a
  separate eager pass over a smaller sample —
  `bash scripts/models/gpt-oss-20b/probe.sh`. Under CUDA graphs vLLM also pads
  each batch to a captured size and those padding rows would be counted as
  tokens, so the probe is the more correct place to measure it regardless.

`scripts/run.sh` supplies the right one per model from `EP_ROUTER_EXEC_MODE` in
`model.env`, so this only bites when composing a `main.py` command by hand.

### Per-rule knobs

Each rule has one knob that sets the budget and, usually, a floor that keeps a
token from being starved entirely.

**Dynamic Routing** keeps the shortest prefix of the ranked experts whose
probability mass reaches a threshold.

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--dynamic_routing_threshold P` | float | 0.8 | Higher keeps more experts. Paper values run 0.3–0.87 depending on model and target. |
| `--dynamic_routing_score_source SRC` | str | `raw` | `raw` accumulates the post-softmax probabilities; `renormalized` first normalises over the top-k. The paper uses `renormalized` throughout, because `raw` makes the threshold mean different things in models with different native `k`. |

**NAEE** drops expert *i* once its gate weight falls below `beta` times the
top-1 weight.

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--naee_beta B` | float | 0.3 | **Higher prunes more**, opposite to Dynamic Routing. Paper values run 0.27–1.95. |
| `--naee_k_min N` | int | 2 | Never go below this many experts. Matters more than it looks: at aggressive budgets a floor of 1 versus 2 can be worth 30 accuracy points. |

**DiEP** is NAEE with the threshold scaled per expert by a calibrated similarity
term. It needs an artifact and reuses NAEE's flags for the threshold.

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--diep_artifact_path PATH` | str | — | Required. Produced by calibration; holds `sim_matrix`, `mean_sim`, `gamma_1`. |
| `--naee_beta`, `--naee_k_min` | | | As above. |
| `--diep_pruning_mode MODE` | str | `independent` | `independent` judges each expert on its own `gamma_2`; `prefix` prunes everything after the first cut, like NAEE. |
| `--diep_gamma_alpha A` | float | 1.0 | Exponent on `gamma_2`. 1.0 is the published rule, 0.0 removes the similarity signal entirely and degenerates to NAEE, and values in between compress its spread. The `DiEP-damped` rows in the result tables are `0.25` and `0.5`, which is what makes DiEP competitive at these budgets. |
| `--diep_threshold_cap C` | float | unset | Upper bound on the threshold relative to the top-1 weight. Must be below 1 to do anything: at exactly 1 the threshold equals the top-1 weight and every other weight is already below it. |
| `--diep_use_gamma1` | flag | off | Multiplies the calibrated `gamma_1` into the threshold, which is equation (12) of the DiEP paper exactly. Combine with `--naee_beta 1.0`. Off by default because `--naee_beta` stands in for `gamma_1` and is easier to sweep. |

**Ban** combines an offline per-layer sensitivity with an online per-token one,
so the budget moves across depth as well as across tokens.

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--ban_artifact_path PATH` | str | — | Required. Holds `layer_sensitivity`, `r_min`, `r_max`. |
| `--ban_lambda L` | float | 0.7 | Conservatism: **higher keeps more experts**. Paper values run 0.26–0.95. |
| `--ban_k_min N` | int | 3 | Floor per token per layer. |

**MC-MoE** and **EAC-MoE** are implemented and runnable but sit outside the
matched-budget comparison, because they prune on a different axis.

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--mc_moe_protection_ratio R` | float | 0.02 | Fraction of tokens protected by importance before thresholding. Uses `--naee_beta` for the threshold itself. |
| `--eac_alpha A` | float | 0.5 | An expert is dropped for a prefill segment when it received fewer than `l*k/n*alpha` tokens in it. |

### Calibration

Ban and DiEP read a tensor artifact computed once per model over a C4 sample.
The artifacts are small but not shippable, being derived from a particular
checkpoint: Ban's is a per-layer sensitivity vector, a few kilobytes. DiEP's holds
an expert-by-expert similarity matrix per layer, so it grows with the square of the
expert count — measured at 465 KB for Ling-lite's 64 experts, and proportionally
larger on wider routers.

```bash
bash scripts/models/<model>/calibrate.sh     # writes calib_utils/results/<model>/c4_{ban,diep}.pt
```

Ban is the slow half, costing samples times layers because it measures a KL
divergence per layer per sample. The default is 128 samples, roughly an hour and a
half on a 48-layer model across four GPUs. The published Qwen3-30B and Qwen3-Next
artifacts used 512, which took 6h11m; Ling-lite and GPT-OSS used 128. Override with
`BAN_NSAMPLES` and `DIEP_NSAMPLES`.

`EP_CALIB_BAN_K_PRUNED` in each `model.env` is the k that Ban prunes to while
measuring sensitivity, so it describes the model rather than the budget: 3 for the
two Qwen models, 2 for Ling-lite and GPT-OSS, matching the published artifacts.

The sweep scripts pass `--ban_artifact_path` and `--diep_artifact_path` for you
and fail with an explicit message if the artifact is missing.

## Matching budgets, which is the whole trick

A rule that keeps more experts will usually score better, so comparing rules at
their default knobs measures nothing. Every comparison in the paper is at a
**matched measured budget**: the knob is solved so that the average number of
experts actually retained per token lands on the Fixed-K value it is being
compared against.

Two consequences you will run into:

**The knob has to be solved per model.** `--ban_lambda 0.5` means something
different on a model with `k=4` than on one with `k=10`. The published values are
in `results/<model>/RESULTS.md` and in the sweep scripts.

**The knob has to be solved per regime.** The same threshold produces a
different average on generated text than on log-likelihood scoring of short
multiple-choice continuations, because the token mix differs. This is why
`qa.sh` uses different numbers from `dynamic_routing.sh` for the same tier, and
why the paper's tables carry two budgets per rule.

To solve a knob for a target average on your own model:

```bash
python scripts/solve_qa_hyper.py --help    # search a knob for a target average k
```

Then confirm where it landed, rather than trusting the search — see below.

## What to evaluate

| Flag | Default | Notes |
|---|---|---|
| `--harness {lighteval,lm_eval,lmms_eval}` | `lighteval` | `lighteval` for generative tasks, `lm_eval` for log-likelihood multiple choice, `lmms_eval` for image-text. The pruning path is shared by all three. |
| `--datasets A,B,C` | `gsm8k` | Generative tasks, for `--harness lighteval`. |
| `--lm_eval_tasks A,B` | the four QA sets | For `--harness lm_eval`. |
| `--lmms_eval_tasks A,B` | nine VL sets | For `--harness lmms_eval`. |
| `--max_samples N` | all | Cap per task. Indispensable for smoke tests, and a good way to burn a week of GPU time if you forget you set it. |
| `--lm_eval_apply_chat_template` | off | Wraps QA questions in the instruct template. **Leave it off** to stay comparable with published numbers for those four sets, which are scored as bare continuations. |
| `--custom_tasks PATH` | repo's `tasks/custom_tasks.py` | Task definitions added on top of the harness. |

The paper's generative suite is

```
mmlu_pro,gpqa:diamond,math_500,aime24_avg,aime25_avg,lcb:codegeneration_v6,gsm8k
```

and the QA suite is `arc_challenge,arc_easy,winogrande,openbookqa`, eleven benchmarks
between them. `tasks/` also implements `hellaswag_fixed` and `livebench_20241125`,
which earlier sweeps collected and the paper does not report; pass them to
`--datasets` if you want them.

`aime24_avg` and `aime25_avg` average over 16 samples per question, because AIME
is 30 questions and a single sample per question is too noisy to compare
budgets with.

Ten of the eleven are ungated. GPQA is not: accept the terms at
[Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) — approval is
automatic — and provide a token, or drop `gpqa:diamond` from `--datasets`. The
sweep scripts check for access before loading a model; a bare `main.py` call does
not, and fails on a 401 once the engine is already up.

## Decoding

| Flag | Default | Paper |
|---|---|---|
| `--temperature T` | 0.7 | 0.7, except Hunyuan3 at 0.9 per its model card |
| `--top_p P` | 0.8 | 0.8, except Hunyuan3 at 1.0 |
| `--top_k N` | 20 | 20, except Hunyuan3 which disables it with 0 |
| `--max_new_tokens N` | 32768 | 32768; 8192 for the `k=1` runs, which otherwise never stop |
| `--seed N` | 1234 | 1234 |

Aggressively pruned models lose the ability to stop cleanly long before they
lose the ability to answer, so a generation cap is not a formality. At `k=1` or
`k=2` some models degenerate into unbounded repetition, and the cap is what keeps
a sweep from stalling on a handful of samples.

## Engine and hardware

These describe your machine, not the experiment.

| Flag | Default | Notes |
|---|---|---|
| `--tensor_parallel_size N` | visible GPU count | Splits each layer across N GPUs. |
| `--pipeline_parallel_size N` | 1 | Splits by layer instead. Combine so that TP×PP equals your GPU count. |
| `--gpu_memory_utilization F` | 0.95 | The sweeps used 0.9, and 0.80–0.85 for the largest models where the KV cache needs room. |
| `--max_model_len N` | 32768 | Must be at least prompt plus `--max_new_tokens`. |
| `--batch_size N` | harness default | vLLM's `max_num_seqs`. Leaving this at 1 was worth a factor of 30 in one of our own runs, so it is worth checking. |
| `--dtype {auto,bfloat16,float16,float32}` | `auto` | |
| `--enforce_eager` | off | Disables CUDA graphs. Slower, but required to count retained experts on quantized models. |
| `--allow_compiled_router` | off | Keeps CUDA graphs on for quantized MoE models; see [above](#why-the-router-patch-needs-an-execution-mode). Pruning still applies but routing statistics cannot be counted. On gpt-oss the same 100 MMLU-Pro samples took 182 minutes eager and 5 minutes with graphs, so the usual approach is graphs for scores plus one short eager run for the expert count. |
| `--cpu_offload_gb F` | 0 | Offload this many GiB of weights per GPU to host memory. Makes an oversized model fit, at a large speed cost. |
| `--trust_remote_code` / `--no-trust_remote_code` | on | Several of these checkpoints ship their own modelling code. |
| `--language_model_only` | off | Skip the vision encoder in a unified multimodal checkpoint. Newer vLLM only; it is dropped automatically on the pinned engine. |

## Output and caching

| Flag | Default | Notes |
|---|---|---|
| `--output_dir PATH` | `./results` | Scores and per-sample details. |
| `--lighteval_cache_dir PATH` | auto | Generation cache. When the local router is active this defaults to `<output_dir>/.lighteval_cache`, with the method, its knobs, the artifact contents and the routing source code all fingerprinted into the subdirectory name. That fingerprint is the reason a sweep cannot silently reuse samples generated under a different pruning configuration; see [compact_cache_contamination.md](compact_cache_contamination.md). |
| `--sample_shard_id I` / `--sample_num_shards N` | unset | Deterministic sharding by document id, for spreading one task across hosts. Set both. |

## Checking that pruning actually happened

The failure worth worrying about is not a crash. It is a run that looks fine and
pruned nothing, which shows up as suspiciously good accuracy.

| Flag | Notes |
|---|---|
| `--expert_pruning_debug` | Prints the routing statistics for the first few calls, including the average number of experts actually retained. This is the number to check. |
| `--expert_pruning_debug_max_prints N` | How many (default 8). |

```bash
python main.py --model_path $M --datasets gsm8k --max_samples 4 \
    --use_local_expert_router --enforce_eager \
    --expert_pruning_method Dynamic_Routing \
    --dynamic_routing_threshold 0.8 --dynamic_routing_score_source renormalized \
    --expert_pruning_debug
```

On Qwen3-30B-A3B at `--dynamic_routing_threshold 0.8` this reports

```
平均专家选择数: 5.858166 (method=dynamic_routing, topk=8, routed_tokens=93504)
```

which is the shape of a healthy answer: strictly between the floor and the native
`k` of 8.

If the average comes back at the model's native `k`, the rule is not taking
effect: usually `--use_local_expert_router` is missing, or the architecture is
not one the patch covers. If it comes back at exactly `--naee_k_min`, the
threshold is so aggressive that the floor is doing all the work, and the knob is
no longer the thing being measured.

The averages recorded this way are what the `avg experts` column of every
results table reports, and what "matched budget" is matched on.

## Research variants

`TopK_Biased_Renorm`, `LayerWise_Dynamic_Routing`, `Budget_Dynamic_Routing`,
`LayerBudget_Dynamic_Routing`, `BandedLayerBudget_Dynamic_Routing`,
`BanSensitiveLayerBudget_Dynamic_Routing` and `RouterValue_Dynamic_Routing` are
implemented and selectable. They allocate a per-layer or per-token-difficulty
budget, and they are why the flag list is as long as it is: the
`--band_layer_budget_*` family alone is about forty flags.

They are not in the paper. Nothing was hidden — they simply did not beat the
rules that are. `python main.py --help` documents each one, and they are kept
because the negative result is worth being able to reproduce.
