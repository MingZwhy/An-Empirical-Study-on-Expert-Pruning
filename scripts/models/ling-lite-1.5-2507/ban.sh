#!/usr/bin/env bash
#
# Ban -- Ling-lite-1.5-2507
#
# Offline layer sensitivity combined with online token sensitivity,
# so the budget moves across layers as well as tokens. Needs a
# calibration artifact.
#
# One block per budget tier. Within a tier the knob is set so the
# measured average expert count matches the Fixed-K row it is
# compared against, which is what makes the comparison fair.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "Ban" "$@"
ep_require_artifact ban

# matched to Fixed-K k=4
ep_run Ban-kmin2-lambda0.9 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.9 --ban_k_min 2 \
    --ban_artifact_path "$(ep_artifact ban)"

# matched to Fixed-K k=3
ep_run Ban-kmin2-lambda0.45 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.45 --ban_k_min 2 \
    --ban_artifact_path "$(ep_artifact ban)"

# matched to Fixed-K k=2
ep_run Ban-kmin1-lambda0.35 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.35 --ban_k_min 1 \
    --ban_artifact_path "$(ep_artifact ban)"

ep_end
