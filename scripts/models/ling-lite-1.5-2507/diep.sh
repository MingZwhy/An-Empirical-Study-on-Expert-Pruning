#!/usr/bin/env bash
#
# DiEP -- Ling-lite-1.5-2507
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

# matched to Fixed-K k=4
ep_run DiEP-kmin2-beta0.45 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.45 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run DiEP-damped-a0.25-cap0.9-kmin2-beta0.467 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.467 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.25 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run DiEP-damped-a0.5-cap0.9-kmin2-beta0.468 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.468 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.5 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"

# matched to Fixed-K k=3
ep_run DiEP-kmin2-beta0.65 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.65 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run DiEP-damped-a0.25-cap0.9-kmin2-beta0.636 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.636 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.25 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run DiEP-damped-a0.5-cap0.9-kmin2-beta0.645 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.645 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.5 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"

# matched to Fixed-K k=2
ep_run DiEP-kmin1-beta0.8 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.8 --naee_k_min 1 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run DiEP-kmin2-beta0.8 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.8 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run DiEP-paper-gamma1-kmin2 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.8 --naee_k_min 2 --diep_pruning_mode independent --diep_use_gamma1 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run DiEP-damped-a0.25-cap0.9-kmin2-beta1.1 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 1.1 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.25 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"

ep_end
