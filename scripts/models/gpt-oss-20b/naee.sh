#!/usr/bin/env bash
#
# NAEE -- gpt-oss-20b
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

# matched to Fixed-K k=3
ep_run NAEE-kmin1-beta0.55 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.55 --naee_k_min 1

# matched to Fixed-K k=2
ep_run NAEE-kmin1-beta0.75 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.75 --naee_k_min 1
ep_run NAEE-kmin2-beta0.75 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.75 --naee_k_min 2

ep_end
