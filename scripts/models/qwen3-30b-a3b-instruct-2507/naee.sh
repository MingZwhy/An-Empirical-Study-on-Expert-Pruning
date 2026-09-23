#!/usr/bin/env bash
#
# NAEE -- Qwen3-30B-A3B-Instruct-2507
#
# Prune expert i when its gate weight falls below beta times the
# top-1 weight. One threshold, no calibration.
#
# One block per budget tier. Within a tier the knob is set so the
# measured average expert count matches the Fixed-K row it is
# compared against, which is what makes the comparison fair.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "NAEE" "$@"

# matched to Fixed-K k=6
ep_run NAEE-kmin2-beta0.35 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.35 --naee_k_min 2

# matched to Fixed-K k=5
ep_run NAEE-kmin2-beta0.45 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.45 --naee_k_min 2

# matched to Fixed-K k=4
ep_run NAEE-kmin2-beta0.55 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.55 --naee_k_min 2

# matched to Fixed-K k=3
ep_run NAEE-kmin2-beta0.7 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.7 --naee_k_min 2

# matched to Fixed-K k=2
ep_run NAEE-kmin2-beta0.8 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.8 --naee_k_min 2

ep_end
