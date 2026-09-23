#!/usr/bin/env bash
#
# Zero-shot multiple choice -- gpt-oss-20b
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
ep_run_qa Baseline-k4
ep_run_qa FixedK-k3 --num_experts_per_tok 3
ep_run_qa FixedK-k2 --num_experts_per_tok 2
ep_run_qa FixedK-k1 --num_experts_per_tok 1

# adaptive rules matched to Fixed-K k=3
ep_run_qa Ban-kmin2-lambda0.9 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.9 --ban_k_min 2 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-kmin1-beta0.9 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.9 --naee_k_min 1 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa DiEP-damped-a0.25-cap0.9-kmin2-beta0.68 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 0.68 --naee_k_min 2 --diep_pruning_mode independent --diep_gamma_alpha 0.25 --diep_threshold_cap 0.9 \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.75-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.75 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin1-beta0.55 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.55 --naee_k_min 1

# adaptive rules matched to Fixed-K k=2
ep_run_qa Ban-kmin1-lambda0.55 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.55 --ban_k_min 1 \
    --ban_artifact_path "$(ep_artifact ban)"
ep_run_qa DiEP-kmin1-beta1.95 \
    --use_local_expert_router --expert_pruning_method DiEP --naee_beta 1.95 --naee_k_min 1 --diep_pruning_mode independent \
    --diep_artifact_path "$(ep_artifact diep)"
ep_run_qa Dynamic_Routing-thr0.5-renormalized \
    --use_local_expert_router --expert_pruning_method Dynamic_Routing --dynamic_routing_threshold 0.5 --dynamic_routing_score_source renormalized
ep_run_qa NAEE-kmin1-beta0.75 \
    --use_local_expert_router --expert_pruning_method NAEE --naee_beta 0.75 --naee_k_min 1

ep_end
