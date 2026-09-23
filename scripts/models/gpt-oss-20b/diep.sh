#!/usr/bin/env bash
#
# DiEP -- gpt-oss-20b
#
# NAEE's threshold, scaled per expert by a calibrated similarity
# term. Needs a calibration artifact; the -damped variants temper
# that term, which is what makes it competitive here.
#
# One block per budget tier. Within a tier the knob is set so the
# measured average expert count matches the Fixed-K row it is
# compared against, which is what makes the comparison fair.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "DiEP" "$@"
ep_require_artifact diep

# matched to Fixed-K k=3
ep_run DiEP-kmin1-beta0.9 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.9 --naee_k_min 1 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run DiEP-damped-a0.25-cap0.9-kmin2-beta0.68 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.68 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.25 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"

# matched to Fixed-K k=2
ep_run DiEP-kmin1-beta1.95 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 1.95 --naee_k_min 1 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run DiEP-kmin2-beta1.95 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 1.95 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"

ep_end
