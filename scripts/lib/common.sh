#!/usr/bin/env bash
#
# Shared plumbing for the per-model sweep scripts in scripts/models/.
#
# A model script declares *what* to run; everything about *how* to run it lives
# here. That keeps the per-model files short enough to read as a list of
# configurations, which is the point: every line in one of those files
# corresponds to one row in the matching results/<model>/RESULTS.md table.
#
# A model script looks like this:
#
#     EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     source "$EP_MODEL_HOME/../../lib/common.sh"
#     ep_begin "Ban" "$@"
#     ep_run Ban-kmin3-lambda0.95 --expert_pruning_method Ban --ban_lambda 0.95 --ban_k_min 3
#     ep_end
#
# ep_run fills in the model path, the dataset list, the sampling parameters, the
# output directory and the router-patch flag from the model's model.env, so the
# call site carries only what actually distinguishes one configuration from
# another.

set -euo pipefail

EP_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---------------------------------------------------------------- diagnostics

ep_die() { printf '\n\033[31m[error]\033[0m %s\n' "$*" >&2; exit 1; }
ep_warn() { printf '\033[33m[warn]\033[0m %s\n' "$*" >&2; }
ep_info() { printf '\033[36m[%s]\033[0m %s\n' "${EP_STAGE:-scripts}" "$*"; }

# ------------------------------------------------------------- flag handling

ep_usage() {
    cat <<EOF
$(basename "${BASH_SOURCE[1]:-$0}") -- ${EP_STAGE:-sweep stage} on ${EP_MODEL_NAME:-a model}

  -n, --dry-run     print the commands and exit without running anything
      --smoke       shrink every run to a few samples and short generations,
                    to check the plumbing before spending GPU hours
  -f, --force       re-run configurations that already have results
      --only PAT    only run configurations whose name contains PAT
      --gpus LIST   set CUDA_VISIBLE_DEVICES (e.g. --gpus 0,1)
  -h, --help        this text

Environment:
  MODEL_PATH     explicit checkpoint directory, overrides MODELS_DIR
  MODELS_DIR     directory holding checkpoints, one subdirectory per model
  EP_OUT_ROOT    where results go (default: <repo>/results)
EOF
}

ep_parse_flags() {
    EP_DRY_RUN=0 EP_SMOKE=0 EP_FORCE=0 EP_ONLY=""
    while (($#)); do
        case "$1" in
            -n|--dry-run) EP_DRY_RUN=1 ;;
            --smoke)      EP_SMOKE=1 ;;
            -f|--force)   EP_FORCE=1 ;;
            --only)       shift; [[ $# -gt 0 ]] || ep_die "--only needs a pattern"; EP_ONLY="$1" ;;
            --gpus)       shift; [[ $# -gt 0 ]] || ep_die "--gpus needs a list"; export CUDA_VISIBLE_DEVICES="$1" ;;
            -h|--help)    ep_usage; exit 0 ;;
            *)            ep_die "unknown option: $1 (try --help)" ;;
        esac
        shift
    done
}

# ------------------------------------------------------------ model resolution

# Find the checkpoint. In dry-run we deliberately do not insist that it exists,
# so that the commands can be inspected before committing to a 60 GB download.
ep_resolve_model() {
    local candidate
    if [[ -n "${MODEL_PATH:-}" ]]; then
        EP_MODEL_PATH="$MODEL_PATH"
    elif [[ -n "${MODELS_DIR:-}" ]]; then
        for candidate in "$MODELS_DIR/$EP_MODEL_NAME" "$MODELS_DIR/$EP_HF_REPO" \
                         "$MODELS_DIR/${EP_HF_REPO##*/}"; do
            if [[ -d "$candidate" ]]; then EP_MODEL_PATH="$candidate"; break; fi
        done
        EP_MODEL_PATH="${EP_MODEL_PATH:-$MODELS_DIR/$EP_MODEL_NAME}"
    else
        # Nothing is set. Keep the variable unexpanded so that the dry-run output
        # stays copy-pasteable once MODELS_DIR is exported.
        EP_MODEL_PATH="\$MODELS_DIR/$EP_MODEL_NAME"
    fi

    if [[ ! -d "$EP_MODEL_PATH" ]]; then
        if [[ $EP_DRY_RUN -eq 1 ]]; then
            ep_warn "checkpoint not found at $EP_MODEL_PATH (fine for --dry-run)"
        else
            ep_die "$(cat <<EOF
no checkpoint at: $EP_MODEL_PATH

Point the scripts at a local copy of $EP_HF_REPO, either

  export MODELS_DIR=/path/to/models     # expects \$MODELS_DIR/$EP_MODEL_NAME
  export MODEL_PATH=/path/to/checkpoint # or name the directory outright

or fetch it first:

  hf download $EP_HF_REPO --local-dir \$MODELS_DIR/$EP_MODEL_NAME

Use --dry-run to inspect the commands without a checkpoint.
EOF
)"
        fi
    fi
}

# ------------------------------------------------------------------- lifecycle

# ep_begin "<stage name>" "$@"
ep_begin() {
    EP_STAGE="$1"; shift
    [[ -n "${EP_MODEL_HOME:-}" ]] || ep_die "model script must set EP_MODEL_HOME before sourcing"
    # shellcheck source=/dev/null
    source "$EP_MODEL_HOME/model.env"
    ep_parse_flags "$@"

    EP_OUT_ROOT="${EP_OUT_ROOT:-$EP_REPO_ROOT/results}"
    EP_LOG_DIR="${EP_LOG_DIR:-$EP_REPO_ROOT/local_logs/$EP_MODEL_NAME}"
    EP_RAN=0 EP_SKIPPED=0 EP_FAILED=0 EP_FILTERED=0

    ep_resolve_model

    # Tensor parallel defaults to however many GPUs are visible, which is what
    # main.py does too; we pass it explicitly so the log records it.
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        EP_TP="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
    else
        EP_TP="${EP_TP:-1}"
    fi

    # Check dataset access before anything loads a model. Set
    # EP_SKIP_DATASET_CHECK=1 to skip it; it costs a couple of seconds.
    if (( ! EP_DRY_RUN )) && [[ "${EP_SKIP_DATASET_CHECK:-0}" != "1" ]]; then
        python "$EP_REPO_ROOT/scripts/lib/preflight_datasets.py" "$EP_GEN_DATASETS" || exit 1
    fi

    printf '\n\033[1m== %s :: %s ==\033[0m\n' "$EP_MODEL_NAME" "$EP_STAGE"
    printf '   checkpoint  %s\n' "$EP_MODEL_PATH"
    printf '   results     %s/%s/\n' "$EP_OUT_ROOT" "$EP_MODEL_NAME"
    printf '   gpus        %s (tensor parallel %s)\n' "${CUDA_VISIBLE_DEVICES:-all}" "$EP_TP"
    (( EP_SMOKE ))   && printf '   \033[33msmoke mode: %s samples, %s new tokens\033[0m\n' \
                              "$EP_SMOKE_SAMPLES" "$EP_SMOKE_NEW_TOKENS"
    [[ -n "${EP_SAMPLE_CAP:-}" ]] && (( ! EP_SMOKE )) && \
        printf '   %s samples per task, %s new tokens\n' "$EP_SAMPLE_CAP" "$EP_MAX_NEW_TOKENS"
    (( EP_DRY_RUN )) && printf '   \033[33mdry run: nothing will be executed, and every configuration is listed\n              whether or not it already has results\033[0m\n'
    printf '\n'
}

ep_end() {
    # A failed configuration is still a match, so it has to count here; otherwise a
    # stage whose only selected run failed reports that the pattern matched nothing.
    if [[ -n "${EP_ONLY:-}" ]] && (( EP_RAN + EP_SKIPPED + EP_FAILED == 0 )); then
        ep_warn "--only '$EP_ONLY' matched none of the $EP_FILTERED configurations in this stage"
    fi
    printf '\n\033[1m-- %s :: %s --\033[0m  ran %d, skipped %d, failed %d\n' \
        "$EP_MODEL_NAME" "$EP_STAGE" "$EP_RAN" "$EP_SKIPPED" "$EP_FAILED"
    (( EP_FAILED == 0 )) || return 1
}

# ------------------------------------------------------------------ execution

# A configuration counts as done when main.py has written a scores file under
# its output directory. That is what makes re-invoking a stage safe: finished
# work is skipped, interrupted work is redone.
#
# The two regimes are tracked separately even though they share a directory. A
# tier's QA knob often works out to the same value as its generative knob, and
# then both regimes write into one directory: generative leaves results_*.json,
# QA leaves lm_eval_results_*.json. Looking for either would make a finished
# generative run mask a QA run that never happened.
ep_is_complete() {
    local out="$1" regime="$2"
    [[ -d "$out" ]] || return 1
    case "$regime" in
        generative) compgen -G "$out/*/results/results_*.json" >/dev/null 2>&1 ;;
        qa)         compgen -G "$out/*/results/lm_eval_results_*.json" >/dev/null 2>&1 ;;
    esac
}

# Where a calibration artifact lives. This only prints: it is called from inside
# "$(...)", and an exit there ends the subshell while the caller carries on with an
# empty path -- which turned one missing precondition into one failure per
# configuration. The check that can actually stop a stage is ep_require_artifact.
ep_artifact() {
    printf '%s' "$EP_REPO_ROOT/calib_utils/results/$EP_MODEL_NAME/c4_${1}.pt"
}

# Calibration artifacts are inputs, not outputs: Ban and DiEP cannot run without
# one. A stage states the requirement up front, so a missing artifact is a single
# refusal before anything starts rather than a pile of run failures.
ep_require_artifact() {
    local kind path msg
    for kind in "$@"; do
        path="$(ep_artifact "$kind")"
        [[ -f "$path" ]] && continue
        msg="$(cat <<EOF
missing $kind calibration artifact: $path

$kind needs one pass of calibration over C4 before it can prune. Run:

  bash scripts/models/$(basename "$EP_MODEL_HOME")/calibrate.sh

Artifacts are derived from this particular checkpoint, which is why they are not
shipped. Expect a few kilobytes for Ban and up to a few megabytes for DiEP.
EOF
)"
        if (( EP_DRY_RUN )); then ep_warn "$msg"; else ep_die "$msg"; fi
    done
}

_ep_shellquote() {
    local out=() a
    for a in "$@"; do
        if [[ "$a" =~ ^[A-Za-z0-9_./:=,+\$-]+$ ]]; then out+=("$a"); else out+=("$(printf '%q' "$a")"); fi
    done
    printf '%s' "${out[*]}"
}

# ep_run <config-name> [extra main.py flags...]
#
# Runs the generative suite. The model's own sampling parameters and dataset
# list come from model.env; anything passed here is what makes this
# configuration different from the others.
ep_run() { _ep_invoke generative "$@"; }

# ep_run_qa <config-name> [extra main.py flags...]
#
# Same, for the four log-likelihood multiple-choice sets. Nothing is generated,
# so pruning is measured without "did the model stop cleanly" as a confound.
ep_run_qa() { _ep_invoke qa "$@"; }

_ep_invoke() {
    local regime="$1" name="$2"; shift 2
    if [[ -n "$EP_ONLY" && "$name" != *"$EP_ONLY"* ]]; then
        (( ++EP_FILTERED )); return 0
    fi

    local out="$EP_OUT_ROOT/$EP_MODEL_NAME/$name"
    # A dry run describes the stage, not the work left in it, so it lists every
    # configuration regardless of what is already on disk. That also keeps it
    # reproducible: check_sweep_scripts.py counts these, and a count that shrank
    # as results accumulated would be reporting the machine, not the scripts.
    if (( ! EP_DRY_RUN && ! EP_FORCE )) && ep_is_complete "$out" "$regime"; then
        printf '  \033[90mskip\033[0m  %-46s already has results\n' "$name"
        (( ++EP_SKIPPED )); return 0
    fi

    local cmd=(python "main.py" --model_path "$EP_MODEL_PATH")
    case "$regime" in
        generative) cmd+=(--datasets "$EP_GEN_DATASETS") ;;
        qa)         cmd+=(--harness lm_eval --lm_eval_tasks "$EP_QA_TASKS") ;;
    esac
    # --smoke substitutes the lengths rather than appending a second copy of the
    # flags: argparse would take the last occurrence either way, but a command
    # printed with the same flag twice is not one you can safely copy.
    local model_len="$EP_MAX_MODEL_LEN" new_tokens="$EP_MAX_NEW_TOKENS"
    if (( EP_SMOKE )); then
        model_len="$EP_SMOKE_MODEL_LEN"; new_tokens="$EP_SMOKE_NEW_TOKENS"
    fi
    cmd+=(--max_model_len "$model_len"
          --max_new_tokens "$new_tokens"
          --gpu_memory_utilization "$EP_GPU_MEM_UTIL")
    # gpt-oss and Hunyuan3 do not take the same sampling knobs as the Qwen line,
    # so model.env decides which are meaningful rather than this file.
    [[ -n "${EP_TEMPERATURE:-}" ]] && cmd+=(--temperature "$EP_TEMPERATURE")
    [[ -n "${EP_TOP_P:-}" ]]       && cmd+=(--top_p "$EP_TOP_P")
    [[ -n "${EP_TOP_K:-}" ]]       && cmd+=(--top_k "$EP_TOP_K")
    [[ -n "${EP_EXTRA_ARGS:-}" ]]  && cmd+=(${EP_EXTRA_ARGS})
    cmd+=("$@")

    # A run that uses the router patch has to be told how to execute it, and
    # main.py refuses to guess. Expert selection gets inlined into the compiled
    # region, torch.compile traces the counters away, and the average expert
    # count is then unavailable -- which is fatal here, because that average is
    # the axis every comparison in the study is matched on. Pruning itself still
    # applies; only the measurement is lost, which is exactly the failure worth
    # refusing. So: eager by default, and quantized models opt into the graph
    # path plus a separate eager probe (see EP_ROUTER_EXEC_MODE in model.env).
    if [[ " $* " == *" --use_local_expert_router "* ]] \
       && [[ " $* " != *" --enforce_eager "* ]] \
       && [[ " $* " != *" --allow_compiled_router "* ]]; then
        cmd+=(${EP_ROUTER_EXEC_MODE:---enforce_eager})
    fi

    cmd+=(--output_dir "$out")

    # A stage may cap samples on its own account -- the expert-count probe is a
    # short measurement pass by design, not something you have to ask for. --smoke
    # is stricter and wins.
    if (( EP_SMOKE )); then
        cmd+=(--max_samples "$EP_SMOKE_SAMPLES")
    elif [[ -n "${EP_SAMPLE_CAP:-}" ]]; then
        cmd+=(--max_samples "$EP_SAMPLE_CAP")
    fi

    if (( EP_DRY_RUN )); then
        # One flag per line, each with whatever values belong to it. Splitting on
        # "starts with --" rather than on pairs keeps valueless flags such as
        # --use_local_expert_router on their own line instead of swallowing the
        # next flag's value.
        local -a lines=() cur=()
        local tok
        for tok in "${cmd[@]}"; do
            if [[ "$tok" == --* && ${#cur[@]} -gt 0 ]]; then
                lines+=("$(_ep_shellquote "${cur[@]}")"); cur=("$tok")
            else
                cur+=("$tok")
            fi
        done
        (( ${#cur[@]} )) && lines+=("$(_ep_shellquote "${cur[@]}")")

        printf '  \033[90m# %s\033[0m\n' "$name"
        local i
        for ((i = 0; i < ${#lines[@]}; i++)); do
            if (( i == 0 )); then printf '  %s' "${lines[i]}"
            else printf '      %s' "${lines[i]}"; fi
            (( i < ${#lines[@]} - 1 )) && printf ' \\'
            printf '\n'
        done
        printf '\n'
        (( ++EP_RAN )); return 0
    fi

    mkdir -p "$EP_LOG_DIR"
    # The regime belongs in the filename for the same reason it belongs in the
    # completeness check: a tier's two knobs often coincide, both regimes then
    # share an output directory, and a single log name means the second run
    # destroys the first one's evidence.
    local log="$EP_LOG_DIR/$name.$regime.log" start elapsed
    printf '  \033[36mrun \033[0m  %-46s -> %s\n' "$name" "${log/#$EP_REPO_ROOT\//}"
    start="$(date +%s)"
    if (cd "$EP_REPO_ROOT" && "${cmd[@]}") >"$log" 2>&1; then
        elapsed=$(( $(date +%s) - start ))
        printf '  \033[32mok  \033[0m  %-46s %dm%02ds\n' "$name" $((elapsed / 60)) $((elapsed % 60))
        (( ++EP_RAN ))
    else
        elapsed=$(( $(date +%s) - start ))
        printf '  \033[31mFAIL\033[0m  %-46s %dm%02ds, tail of %s:\n' \
            "$name" $((elapsed / 60)) $((elapsed % 60)) "$log"
        tail -n 12 "$log" | sed 's/^/          /'
        (( ++EP_FAILED ))
        [[ "${EP_KEEP_GOING:-1}" == "1" ]] || exit 1
    fi
}
