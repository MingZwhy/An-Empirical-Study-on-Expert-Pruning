#!/usr/bin/env bash
#
# EAC-MoE -- Qwen3-Next-80B-A3B-Instruct
#
# Decides from per-expert traffic and only prunes during prefill.
# Also outside the matched-budget comparison.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "EAC-MoE" "$@"

ep_run EAC_MoE-alpha0.5 \
    --use_local_expert_router --expert_pruning_method EAC_MoE --eac_alpha 0.5

ep_end
