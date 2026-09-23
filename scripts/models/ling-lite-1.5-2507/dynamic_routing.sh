#!/usr/bin/env bash
#
# Dynamic Routing -- Ling-lite-1.5-2507
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

# matched to Fixed-K k=4
ep_run Dynamic_Routing-thr0.75-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.75 --dynamic_routing_score_source renormalized

# matched to Fixed-K k=3
ep_run Dynamic_Routing-thr0.6-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.6 --dynamic_routing_score_source renormalized

# matched to Fixed-K k=2
ep_run Dynamic_Routing-thr0.4-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.4 --dynamic_routing_score_source renormalized

ep_end
