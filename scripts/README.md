# scripts/

Three layers, from "just run it" to "read exactly what it does".

```
run.sh                  one entry point for every sweep
models/<model>/         one directory per model: every method, every budget
lib/common.sh           the shared plumbing the model scripts call
```

Everything else here is analysis: report generators, plotting, knob solvers, and
the speed benchmarks.

## Start here

```bash
bash scripts/run.sh list                              # models and their stages
bash scripts/run.sh qwen3-30b ban --dry-run           # see the commands, run nothing
bash scripts/run.sh qwen3-30b baseline --smoke        # check the plumbing on a few samples
bash scripts/run.sh qwen3-30b all                     # the paper's column for this model
bash scripts/run.sh all                               # the whole study
```

`--dry-run` needs neither a GPU nor a checkpoint, which makes it the fastest way
to understand a stage. Every flag is explained in
[`docs/hyperparameters.md`](../docs/hyperparameters.md).

## models/&lt;model&gt;/

One directory per model, one script per method, one line per configuration:

| File | What it runs |
|---|---|
| `model.env` | the model's canonical settings: native k, dataset list, sampling, tensor parallel |
| `baseline.sh` | the released checkpoint, nothing overridden |
| `fixedk.sh` | keep the k best experts per token, for every k from native−1 down to 1 |
| `naee.sh` | gate weight relative to the top-1 expert |
| `dynamic_routing.sh` | shortest expert prefix reaching a probability mass |
| `diep.sh` | NAEE scaled by calibrated expert similarity |
| `ban.sh` | calibrated layer sensitivity plus token sensitivity |
| `mc_moe.sh`, `eac_moe.sh` | implemented, outside the matched-budget comparison |
| `qa.sh` | the four log-likelihood multiple-choice sets |
| `calibrate.sh` | the one C4 pass that Ban and DiEP read |

Each script is independently runnable and takes the same options as `run.sh`:

```bash
bash scripts/models/qwen3-30b-a3b-instruct-2507/ban.sh --dry-run
bash scripts/models/qwen3-30b-a3b-instruct-2507/fixedk.sh --only k6 --gpus 0,1
```

The four models with the full method set need `install_expertpruning.sh`; the
newer architectures are Fixed-K only and need `install_fixedk.sh`. `run.sh`
checks which environment is active before it starts anything.

### Reading a configuration

```bash
# matched to Fixed-K k=3
ep_run Ban-kmin2-lambda0.5 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.5 --ban_k_min 2 \
    --ban_artifact_path "$(ep_artifact ban)"
```

`Ban-kmin2-lambda0.5` is both the output directory under `results/<model>/` and
the row label in `results/<model>/RESULTS.md`, so a line here and a row there
are the same run. The comment records which Fixed-K budget the knob was solved
to match — that matching is what makes the comparison mean anything, and
[docs/hyperparameters.md](../docs/hyperparameters.md#matching-budgets-which-is-the-whole-trick)
explains why.

`qa.sh` deliberately uses different knob values from the generative scripts: the
same threshold retains a different average number of experts when scoring short
multiple-choice continuations than when generating, so each regime has its own
solved value.

## Behaviour worth knowing

**Runs resume.** A configuration whose output directory already holds a scores
file is skipped, so re-invoking a stage after an interruption picks up where it
stopped. `--force` redoes them. The two regimes are tracked separately even when
they share a directory.

**Failures do not stop the sweep.** A failing configuration prints the tail of
its log and the stage moves on, then reports a non-zero count at the end. Set
`EP_KEEP_GOING=0` to stop at the first failure instead.

**Logs** go to `local_logs/<model>/<config>.log`.

**Nothing writes outside the repository** except what you point `MODELS_DIR` at.
Results go to `results/`, overridable with `EP_OUT_ROOT`.

## Checking the scripts

```bash
python scripts/check_sweep_scripts.py
```

Dry-runs every configuration and checks each emitted command against `main.py`'s
argument parser: unknown flags, flags missing values, invalid choices, colliding
output directories, and router-patch runs that name no execution mode. It needs no
GPU and no checkpoint, and it is how a typo in a knob name gets caught before a
run loads 60 GB of weights and dies at argument parsing. The count it reports does
not depend on what you have already run.

Two properties are checked directly, both on CPU in about a second:

```bash
python scripts/test_fixedk_overlay.py
```

That the Fixed-K overlay rewrites the config key a checkpoint actually uses, and
invents no other. Checkpoints disagree on the name — `num_experts_per_tok` at the
top level, the same under `text_config`, `top_k_experts` for Gemma 4 — and the name
in the config is the name the engine reads. Writing the wrong one prunes nothing
while the run still reports a pruned budget, which is the one failure here that is
both silent and plausible-looking.

```bash
python scripts/test_diep_gamma_alpha.py
```

The three properties of DiEP's threshold that the result tables rely on: that
`--diep_gamma_alpha 0` reproduces NAEE token for token, that the defaults leave
published behaviour untouched, and that a threshold cap of 1.0 is a no-op. The
reasoning is in [docs/known_issues.md](../docs/known_issues.md).

## Analysis and benchmarks

| Script | What it does |
|---|---|
| `report_results.py` | regenerates `results/<model>/RESULTS.md` from raw run directories |
| `report_qa_summary.py` | the cross-model QA table in `docs/qa_summary.md` |
| `report_vl.py` | the multimodal tables |
| `plot_expert_pruning_results.py` | the Fixed-K curves |
| `solve_qa_hyper.py`, `solve_diep_beta.py`, `calibrate_ban_lambda.py` | solve a knob for a target average expert count |
| `bench_fixed_shapes.py`, `run_fixed_shapes_sweep.sh` | prefill and decode throughput at fixed shapes, vLLM and Transformers |
| `run_long_decode_sweep.sh` | long-generation throughput, vLLM only |
| `analyze_router_distribution_text_vs_vl.py`, `merge_router_distribution.py`, `run_router_distribution_sweep.sh` | router weight concentration, text against multimodal |
| `check_compact_cache_contamination.py` | verifies no two configurations shared cached generations |
| `patch_gptoss_kernels_k3.py` | lets gpt-oss's Triton kernels take a non-power-of-two k |

The report generators rebuild the published tables from raw runs, which are tens
of GB per model and are not shipped. They are here so the path from a run to a
table is inspectable, not because they will work on a fresh clone.
