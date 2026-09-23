#!/usr/bin/env bash
#
# MC-MoE -- Qwen3-30B-A3B-Instruct-2507
#
# Protects the most important tokens via attention sinks before
# thresholding. Outside the matched-budget comparison because it
# prunes on a different axis.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "MC-MoE" "$@"

ep_run MC_MoE-beta0.55-kmin2-p0.20 \
    --use_local_expert_router --expert_pruning_method MC_MoE --naee_beta 0.55 --naee_k_min 2 --mc_moe_protection_ratio 0.20 --attention_sink_probe_max_query_tokens 2048 --attention_sink_probe_max_key_tokens 2048
ep_run MC_MoE-beta0.60-kmin2-p0.20 \
    --use_local_expert_router --expert_pruning_method MC_MoE --naee_beta 0.60 --naee_k_min 2 --mc_moe_protection_ratio 0.20 --attention_sink_probe_max_query_tokens 2048 --attention_sink_probe_max_key_tokens 2048
ep_run MC_MoE-beta0.65-kmin2-p0.20 \
    --use_local_expert_router --expert_pruning_method MC_MoE --naee_beta 0.65 --naee_k_min 2 --mc_moe_protection_ratio 0.20 --attention_sink_probe_max_query_tokens 2048 --attention_sink_probe_max_key_tokens 2048
ep_run MC_MoE-beta0.75-kmin2-p0.20 \
    --use_local_expert_router --expert_pruning_method MC_MoE --naee_beta 0.75 --naee_k_min 2 --mc_moe_protection_ratio 0.20 --attention_sink_probe_max_query_tokens 2048 --attention_sink_probe_max_key_tokens 2048

ep_end
