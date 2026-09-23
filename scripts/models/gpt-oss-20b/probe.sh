#!/usr/bin/env bash
#
# Eager expert-count probe -- gpt-oss-20b
#
# Measures the average number of experts each rule actually retains.
#
# The scoring runs for this model keep CUDA graphs, which makes the
# count unobservable but is 6.7x faster on generation -- 166 against
# 1138 tok/s in the published measurement. This stage re-runs each
# configuration eagerly over a much smaller sample: 40 prompts each
# of MMLU-Pro, MATH-500 and GSM8K, still a few million routing
# decisions, and spread over three input distributions because the
# router reacts to its input.
#
# The scores from this stage are not the reported ones; only the
# average expert count is. Being a measurement pass rather than a
# smoke test, it sets its own sample cap without being asked; passing
# --smoke narrows it further.

EP_SAMPLE_CAP=40
EP_MAX_NEW_TOKENS=2048
EP_MAX_MODEL_LEN=8192
EP_GEN_DATASETS=mmlu_pro,math_500,gsm8k

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "probe" "$@"

ep_run Ban-kmin2-lambda0.9-probe \
    --use_local_expert_router --enforce_eager --expert_pruning_method Ban --ban_lambda 0.9 --ban_k_min 2 \
    --ban_artifact_path "$(ep_artifact ban)"

ep_run DiEP-kmin1-beta0.9-probe \
    --use_local_expert_router --enforce_eager --expert_pruning_method DiEP --naee_beta 0.9 --naee_k_min 1 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"

ep_run DiEP-damped-a0.25-cap0.9-kmin2-beta0.68-probe \
    --use_local_expert_router --enforce_eager --expert_pruning_method DiEP --naee_beta 0.68 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.25 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"

ep_run Dynamic_Routing-thr0.75-renormalized-probe \
    --use_local_expert_router --enforce_eager --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.75 --dynamic_routing_score_source renormalized

ep_run NAEE-kmin1-beta0.55-probe \
    --use_local_expert_router --enforce_eager --expert_pruning_method NAEE --naee_beta 0.55 --naee_k_min 1

ep_run Ban-kmin1-lambda0.55-probe \
    --use_local_expert_router --enforce_eager --expert_pruning_method Ban --ban_lambda 0.55 --ban_k_min 1 \
    --ban_artifact_path "$(ep_artifact ban)"

ep_run DiEP-kmin1-beta1.95-probe \
    --use_local_expert_router --enforce_eager --expert_pruning_method DiEP --naee_beta 1.95 --naee_k_min 1 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"

ep_run DiEP-kmin2-beta1.95-probe \
    --use_local_expert_router --enforce_eager --expert_pruning_method DiEP --naee_beta 1.95 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"

ep_run Dynamic_Routing-thr0.5-renormalized-probe \
    --use_local_expert_router --enforce_eager --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.5 --dynamic_routing_score_source renormalized

ep_run NAEE-kmin1-beta0.75-probe \
    --use_local_expert_router --enforce_eager --expert_pruning_method NAEE --naee_beta 0.75 --naee_k_min 1

ep_run NAEE-kmin2-beta0.75-probe \
    --use_local_expert_router --enforce_eager --expert_pruning_method NAEE --naee_beta 0.75 --naee_k_min 2

ep_end
