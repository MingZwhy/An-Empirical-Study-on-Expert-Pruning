#!/usr/bin/env bash
#
# Zero-shot multiple choice -- Qwen3-Next-80B-A3B-Instruct
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
ep_run_qa Baseline-k10
ep_run_qa FixedK-k9 --num_experts_per_tok 9
ep_run_qa FixedK-k8 --num_experts_per_tok 8
ep_run_qa FixedK-k7 --num_experts_per_tok 7
ep_run_qa FixedK-k6 --num_experts_per_tok 6
ep_run_qa FixedK-k5 --num_experts_per_tok 5
ep_run_qa FixedK-k4 --num_experts_per_tok 4
ep_run_qa FixedK-k3 --num_experts_per_tok 3
ep_run_qa FixedK-k2 --num_experts_per_tok 2
ep_run_qa FixedK-k1 --num_experts_per_tok 1

# adaptive rules matched to Fixed-K k=6
ep_run_qa Ban-kmin4-lambda0.85 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.85 --ban_k_min 4 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-kmin3-beta0.65 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.65 --naee_k_min 3 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.7-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.7 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.4 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.4 --naee_k_min 2

# adaptive rules matched to Fixed-K k=5
ep_run_qa Ban-kmin3-lambda0.65 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.65 --ban_k_min 3 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-kmin3-beta0.9 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.9 --naee_k_min 3 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.6771-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.6771 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.45 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.45 --naee_k_min 2

# adaptive rules matched to Fixed-K k=4
ep_run_qa Ban-kmin3-lambda0.45 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.45 --ban_k_min 3 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-damped-a0.25-cap0.9-kmin2-beta0.57 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.57 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.25 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.55-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.55 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.55 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.55 --naee_k_min 2

# adaptive rules matched to Fixed-K k=3
ep_run_qa Ban-kmin2-lambda0.3 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.3 --ban_k_min 2 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-damped-a0.25-cap0.9-kmin2-beta0.72 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.72 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.25 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.4-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.4 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.65 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.65 --naee_k_min 2

# adaptive rules matched to Fixed-K k=2
ep_run_qa Dynamic_Routing-thr0.3-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.3 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin2-beta0.75 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.75 --naee_k_min 2

ep_end
