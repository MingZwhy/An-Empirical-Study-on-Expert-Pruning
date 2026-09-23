#!/usr/bin/env bash
#
# One entry point for every sweep in the study.
#
#   scripts/run.sh list                          what is available
#   scripts/run.sh qwen3-30b baseline fixedk     some stages of one model
#   scripts/run.sh qwen3-30b all                 everything the paper reports
#   scripts/run.sh all                           the whole study, model by model
#
# Add --dry-run to any of these to print the commands without running them,
# which is the fastest way to see exactly what a stage does.
#
# The per-model scripts under models/<slug>/ are the real content and can be run
# directly; this file only resolves names, checks the environment, and sequences
# stages. Anything it does, you can do by hand.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
MODELS_ROOT="$SCRIPTS_DIR/models"

# Stages in the order a full reproduction wants them: calibration first, because
# Ban and DiEP cannot run without it, then the cheap runs, then the rest.
ALL_STAGES=(calibrate baseline fixedk naee dynamic_routing diep ban probe qa)
# Implemented and runnable, but outside the paper's matched-budget comparison,
# so a reproduction does not need them.
EXTRA_STAGES=(mc_moe eac_moe)

red()  { printf '\033[31m%s\033[0m' "$*"; }
dim()  { printf '\033[90m%s\033[0m' "$*"; }
bold() { printf '\033[1m%s\033[0m' "$*"; }
die()  { printf '\n%s %s\n' "$(red '[error]')" "$*" >&2; exit 1; }

usage() {
    cat <<EOF
$(bold 'scripts/run.sh') -- reproduce the expert-pruning sweeps

$(bold USAGE)
  scripts/run.sh list                       list models and their stages
  scripts/run.sh <model> <stage>...         run those stages
  scripts/run.sh <model> all                every stage the paper reports
  scripts/run.sh <model> extras             MC-MoE and EAC-MoE
  scripts/run.sh all                        every model, in increasing GPU order

$(bold OPTIONS)   (passed through to the per-model scripts)
  -n, --dry-run    print commands, run nothing
      --smoke      a few samples and short generations, to test the plumbing
  -f, --force      redo configurations that already have results
      --only PAT   only configurations whose name contains PAT
      --gpus LIST  set CUDA_VISIBLE_DEVICES, e.g. --gpus 0,1
  -h, --help       this text

$(bold ENVIRONMENT)
  MODELS_DIR   directory of checkpoints, one subdirectory per model  (required)
  MODEL_PATH   a single checkpoint directory, overrides MODELS_DIR
  EP_OUT_ROOT  where results are written                (default: results/)

$(bold EXAMPLES)
  # see what the Ban stage would do, without a checkpoint or a GPU
  scripts/run.sh qwen3-30b ban --dry-run

  # check the plumbing end to end on two GPUs, a few samples per task
  export MODELS_DIR=\$HOME/models
  scripts/run.sh qwen3-30b baseline fixedk --smoke --gpus 0,1

  # the paper's Qwen3-30B column, about a day on 2 GPUs
  scripts/run.sh qwen3-30b all --gpus 0,1

$(bold 'MODEL NAMES') are matched loosely: qwen3-30b, qwen3-30b-a3b-instruct-2507
and Qwen3-30B-A3B-Instruct-2507 all select the same model.
EOF
}

# ------------------------------------------------------------------ discovery

model_slugs() { (cd "$MODELS_ROOT" && ls -d */ 2>/dev/null | tr -d /); }

# model_field <slug> <VAR>
#
# Sourced in a subshell rather than read with grep. model.env writes its values as
# ${VAR:-default} so that the environment can override them, and reading the file
# textually returns that expression instead of the value -- which silently fed the
# literal string to the environment guard and made it refuse every model.
model_field() {
    ( source "$MODELS_ROOT/$1/model.env" 2>/dev/null; printf '%s' "${!2-}" )
}

model_stages() {
    (cd "$MODELS_ROOT/$1" && ls *.sh 2>/dev/null | sed 's/\.sh$//')
}

# Accept a slug, a unique prefix of one, or the canonical model name.
resolve_model() {
    local want="${1,,}" slug hits=()
    for slug in $(model_slugs); do
        [[ "$slug" == "$want" ]] && { printf '%s' "$slug"; return; }
        [[ "${slug,,}" == "$want" ]] && { printf '%s' "$slug"; return; }
        if [[ "$(model_field "$slug" EP_MODEL_NAME | tr 'A-Z' 'a-z')" == "$want" ]]; then
            printf '%s' "$slug"; return
        fi
        [[ "$slug" == "$want"* ]] && hits+=("$slug")
    done
    case ${#hits[@]} in
        1) printf '%s' "${hits[0]}" ;;
        0) die "no such model: $1
$(printf '  known: %s\n' "$(model_slugs | tr '\n' ' ')")
Run 'scripts/run.sh list' for the full table." ;;
        *) die "'$1' is ambiguous: ${hits[*]}" ;;
    esac
}

cmd_list() {
    printf '\n%s\n\n' "$(bold 'Models')"
    printf '  %-30s %-14s %4s %5s  %s\n' slug environment k GPUs stages
    printf '  %-30s %-14s %4s %5s  %s\n' \
        '------------------------------' -------------- ---- ----- '------'
    local slug n
    for slug in $(model_slugs); do
        n="$(model_stages "$slug" | wc -l)"
        printf '  %-30s %-14s %4s %5s  %s\n' \
            "$slug" \
            "$(model_field "$slug" EP_ENVIRONMENT)" \
            "$(model_field "$slug" EP_NATIVE_K)" \
            "$(model_field "$slug" EP_TP)" \
            "$( (( n > 4 )) && echo "all $n" || model_stages "$slug" | tr '\n' ' ')"
    done
    cat <<EOF

$(dim '  environment  which install script the model needs: install_expertpruning.sh or install_fixedk.sh')
$(dim '  k      experts the router selects per token before pruning')
$(dim '  GPUs   tensor parallel size the published runs used')

$(bold 'Stages')

  baseline          the released checkpoint, nothing overridden
  fixedk            keep the k best experts per token, for every k
  naee              gate weight relative to the top-1 expert
  dynamic_routing   shortest expert prefix reaching a probability mass
  diep              NAEE scaled by calibrated expert similarity
  ban               calibrated layer sensitivity plus token sensitivity
  qa                the four log-likelihood multiple-choice sets
  calibrate         the C4 pass that Ban and DiEP read
  probe             quantized models only: an eager pass to count retained experts
  mc_moe, eac_moe   implemented, outside the matched-budget comparison

$(bold 'What each knob means:') docs/hyperparameters.md
EOF
}

# ---------------------------------------------------------------- environment

# Both environments can load a model the other cannot, and the failure when you
# get it backwards is a confusing engine error rather than a clear one.
#
# A model declares the environment it needs: "expertpruning" when a stage patches
# the router, "fixedk" when the architecture postdates the pinned engine, and
# "either" for the older Fixed-K-only models that both engines can load.
check_environment() {
    local slug="$1" want vllm
    want="$(model_field "$slug" EP_ENVIRONMENT)"
    vllm="$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)"
    [[ -n "$vllm" ]] || { printf '%s vLLM is not importable; is the environment activated?\n' \
        "$(red '[error]')" >&2; return 1; }
    case "$want:$vllm" in
        expertpruning:0.10.2*|fixedk:0.28*|either:0.10.2*|either:0.28*) return 0 ;;
    esac
    printf '\n%s %s needs the "%s" environment, but vLLM %s is active.\n' \
        "$(red '[error]')" "$(model_field "$slug" EP_MODEL_NAME)" "$want" "$vllm" >&2
    if [[ "$want" == expertpruning ]]; then
        printf '        The adaptive rules patch vLLM 0.10.2 in place:\n          bash install_expertpruning.sh\n' >&2
    else
        printf '        This architecture postdates the pinned engine:\n          bash install_fixedk.sh\n' >&2
    fi
    printf '        See the table at the top of the README. Override with EP_SKIP_ENV_CHECK=1.\n' >&2
    return 1
}

# --------------------------------------------------------------------- driver

run_stage() {
    local slug="$1" stage="$2"; shift 2
    local script="$MODELS_ROOT/$slug/$stage.sh"
    [[ -f "$script" ]] || {
        printf '  %s %s has no %s stage\n' "$(dim 'skip')" "$slug" "$stage"; return 0; }
    bash "$script" "$@"
}

main() {
    local passthru=() positional=()
    while (($#)); do
        case "$1" in
            -h|--help) usage; exit 0 ;;
            -n|--dry-run|--smoke|-f|--force) passthru+=("$1") ;;
            --only|--gpus) passthru+=("$1" "${2:?$1 needs a value}"); shift ;;
            -*) die "unknown option: $1 (try --help)" ;;
            *) positional+=("$1") ;;
        esac
        shift
    done

    (( ${#positional[@]} )) || { usage; exit 0; }

    if [[ "${positional[0]}" == "list" ]]; then cmd_list; exit 0; fi

    local dry=0 p
    for p in "${passthru[@]}"; do [[ "$p" == "-n" || "$p" == "--dry-run" ]] && dry=1; done

    # scripts/run.sh all -- every model, cheapest first
    if [[ "${positional[0]}" == "all" && ${#positional[@]} -eq 1 ]]; then
        MAIN_PASSTHRU=(${passthru[@]+"${passthru[@]}"})
        MAIN_DRY=$dry
        local slug rc=0
        for slug in $(for s in $(model_slugs); do
                          printf '%s %s\n' "$(model_field "$s" EP_TP)" "$s"; done |
                      sort -n | awk '{print $2}'); do
            printf '\n%s\n' "$(bold "########## $slug ##########")"
            main_one "$slug" all || rc=1
        done
        return $rc
    fi

    local slug
    slug="$(resolve_model "${positional[0]}")"
    local stages=("${positional[@]:1}")
    (( ${#stages[@]} )) || stages=(all)
    MAIN_PASSTHRU=(${passthru[@]+"${passthru[@]}"})
    MAIN_DRY=$dry
    main_one "$slug" "${stages[@]}"
}

main_one() {
    local slug="$1"; shift
    local stages=("$@") expanded=() s
    for s in "${stages[@]}"; do
        case "$s" in
            all)    expanded+=("${ALL_STAGES[@]}") ;;
            extras) expanded+=("${EXTRA_STAGES[@]}") ;;
            *)      [[ -f "$MODELS_ROOT/$slug/$s.sh" ]] ||
                        die "$slug has no stage '$s'. It has: $(model_stages "$slug" | tr '\n' ' ')"
                    expanded+=("$s") ;;
        esac
    done

    if (( ! ${MAIN_DRY:-0} )) && [[ "${EP_SKIP_ENV_CHECK:-0}" != "1" ]]; then
        check_environment "$slug" || exit 1
    fi

    local rc=0
    for s in "${expanded[@]}"; do
        run_stage "$slug" "$s" ${MAIN_PASSTHRU[@]+"${MAIN_PASSTHRU[@]}"} || rc=1
    done
    return $rc
}

main "$@"
