#!/usr/bin/env bash
#
# Zero-shot multiple choice -- Qwen3-30B-A3B-Instruct-2507
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
ep_run_qa Baseline-k8
ep_run_qa FixedK-k7 --num_experts_per_tok 7
ep_run_qa FixedK-k6 --num_experts_per_tok 6
ep_run_qa FixedK-k5 --num_experts_per_tok 5
ep_run_qa FixedK-k4 --num_experts_per_tok 4
ep_run_qa FixedK-k3 --num_experts_per_tok 3
ep_run_qa FixedK-k2 --num_experts_per_tok 2
ep_run_qa FixedK-k1 --num_experts_per_tok 1

# adaptive rules matched to Fixed-K k=6
ep_run_qa Ban-kmin5-lambda0.931 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.931 --ban_k_min 5 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-kmin2-beta0.4392 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.4392 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa DiEP-damped-a0.5-cap0.9-kmin2-beta0.3529 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.3529 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.5 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.865-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.865 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.2765 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.2765 --naee_k_min 2

# adaptive rules matched to Fixed-K k=5
ep_run_qa Ban-kmin4-lambda0.6942 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.6942 --ban_k_min 4 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-kmin2-beta0.7 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.7 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa DiEP-damped-a0.5-cap0.9-kmin2-beta0.57 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.57 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.5 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.7-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.7 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.45 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.45 --naee_k_min 2

# adaptive rules matched to Fixed-K k=4
ep_run_qa Ban-kmin3-lambda0.6 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.6 --ban_k_min 3 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-kmin2-beta1.0 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 1.0 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa DiEP-damped-a0.5-cap0.9-kmin2-beta0.723 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.723 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.5 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa DiEP-paper-gamma1-kmin2 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 1.0 --naee_k_min 2 --diep_pruning_mode independent --diep_use_gamma1 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.6-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.6 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.55 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.55 --naee_k_min 2

# adaptive rules matched to Fixed-K k=3
ep_run_qa Ban-kmin2-lambda0.5 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.5 --ban_k_min 2 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-kmin2-beta1.2 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 1.2 --naee_k_min 2 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"   # also the best this rule reaches for k=2
ep_run_qa DiEP-damped-a0.5-cap0.9-kmin2-beta0.926 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.926 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.5 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.45-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.45 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.7 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.7 --naee_k_min 2

# adaptive rules matched to Fixed-K k=2
ep_run_qa Ban-kmin1-lambda0.26 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.26 --ban_k_min 1 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-damped-a0.5-cap0.9-kmin2-beta1.394 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 1.394 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.5 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.35-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.35 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.8 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.8 --naee_k_min 2

ep_end
