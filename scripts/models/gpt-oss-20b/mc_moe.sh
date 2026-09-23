#!/usr/bin/env bash
#
# MC-MoE -- gpt-oss-20b
#
# Protects the most important tokens via attention sinks before
# thresholding. Outside the matched-budget comparison because it
# prunes on a different axis.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "MC-MoE" "$@"

ep_run MC_MoE-beta0.3-kmin2-p0.02 \
    --use_local_expert_router --expert_pruning_method MC_MoE --naee_beta 0.3 --naee_k_min 2 --mc_moe_protection_ratio 0.02 --attention_sink_probe_max_query_tokens 2048 --attention_sink_probe_max_key_tokens 2048

ep_end
