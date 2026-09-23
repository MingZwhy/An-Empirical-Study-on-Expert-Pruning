#!/usr/bin/env bash
#
# Zero-shot multiple choice -- Ling-lite-1.5-2507
#
# ARC-Easy, ARC-Challenge, WinoGrande and OpenBookQA through
# lm-evaluation-harness. These are scored by log-likelihood over the
# candidate answers, so nothing is generated and pruning is measured
# without "did the model stop cleanly" as a confound.
#
# The knobs here differ from the generative scripts on purpose: the
# budget knob is solved separately per regime, so reaching the same
# average expert count on QA needs a different threshold.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "qa" "$@"

# unpruned, then the Fixed-K ladder
ep_run_qa Baseline-k6
ep_run_qa FixedK-k5 --num_experts_per_tok 5
ep_run_qa FixedK-k4 --num_experts_per_tok 4
ep_run_qa FixedK-k3 --num_experts_per_tok 3
ep_run_qa FixedK-k2 --num_experts_per_tok 2
ep_run_qa FixedK-k1 --num_experts_per_tok 1

# adaptive rules matched to Fixed-K k=4
ep_run_qa Ban-kmin3-lambda0.7296 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.7296 --ban_k_min 3 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-kmin2-beta0.3273 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.3273 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa DiEP-damped-a0.25-cap0.9-kmin2-beta0.3424 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.3424 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.25 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.7991-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.7991 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.3529 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.3529 --naee_k_min 2

# adaptive rules matched to Fixed-K k=3
ep_run_qa Ban-kmin2-lambda0.5298 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.5298 --ban_k_min 2 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-kmin2-beta0.5628 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.5628 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa DiEP-damped-a0.5-cap0.9-kmin2-beta0.5514 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.5514 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.5 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.638-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.638 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.5414 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.5414 --naee_k_min 2

# adaptive rules matched to Fixed-K k=2
ep_run_qa Ban-kmin1-lambda0.4122 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.4122 --ban_k_min 1 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-kmin2-beta0.8 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.8 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa DiEP-damped-a0.25-cap0.9-kmin2-beta1.1 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 1.1 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.25 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.4416-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.4416 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.75 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.75 --naee_k_min 2

ep_end
