#!/usr/bin/env bash
#
# Zero-shot multiple choice -- MiniMax-M2.7
#
# ARC-Easy, ARC-Challenge, WinoGrande and OpenBookQA through
# lm-evaluation-harness, scored by log-likelihood over the candidate
# answers. Nothing is generated.

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

ep_end
