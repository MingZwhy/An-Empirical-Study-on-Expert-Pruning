#!/usr/bin/env bash
#
# Dynamic Routing -- Qwen3-30B-A3B-Instruct-2507
#
# Keep the shortest prefix of the ranked experts whose renormalised
# probability mass reaches the threshold. Cheapest rule that adapts
# per token.
#
# One block per budget tier. Within a tier the knob is set so the
# measured average expert count matches the Fixed-K row it is
# compared against, which is what makes the comparison fair.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "Dynamic Routing" "$@"

# matched to Fixed-K k=6
ep_run Dynamic_Routing-thr0.8-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.8 --dynamic_routing_score_source renormalized

# matched to Fixed-K k=5
ep_run Dynamic_Routing-thr0.7-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.7 --dynamic_routing_score_source renormalized

# matched to Fixed-K k=4
ep_run Dynamic_Routing-thr0.6-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.6 --dynamic_routing_score_source renormalized

# matched to Fixed-K k=3
ep_run Dynamic_Routing-thr0.45-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.45 --dynamic_routing_score_source renormalized

# matched to Fixed-K k=2
ep_run Dynamic_Routing-thr0.35-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.35 --dynamic_routing_score_source renormalized

ep_end
