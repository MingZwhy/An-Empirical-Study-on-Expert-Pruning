#!/usr/bin/env bash
#
# Ban -- Qwen3-30B-A3B-Instruct-2507
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

# matched to Fixed-K k=6
ep_run Ban-kmin3-lambda0.95 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.95 --ban_k_min 3 \
    --ban_artifact_path "$(ep_artifact ban)"

# matched to Fixed-K k=5
ep_run Ban-kmin3-lambda0.9 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.9 --ban_k_min 3 \
    --ban_artifact_path "$(ep_artifact ban)"

# matched to Fixed-K k=4
ep_run Ban-kmin3-lambda0.6 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.6 --ban_k_min 3 \
    --ban_artifact_path "$(ep_artifact ban)"

# matched to Fixed-K k=3
ep_run Ban-kmin2-lambda0.5 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.5 --ban_k_min 2 \
    --ban_artifact_path "$(ep_artifact ban)"

# matched to Fixed-K k=2
ep_run Ban-kmin1-lambda0.26 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.26 --ban_k_min 1 \
    --ban_artifact_path "$(ep_artifact ban)"

ep_end
