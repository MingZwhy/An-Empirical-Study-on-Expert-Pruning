#!/usr/bin/env bash
#
# Fixed-K ladder -- DeepSeek-V2-Lite-Chat
#
# Keep the k highest-scoring experts per token and drop the rest, for
# every k from 5 down to 1. This is a config overlay: it rewrites
# num_experts_per_tok and lets the model's own router rank as usual, so
# it needs no engine patch and works in either environment.
#
# The whole ladder is 5 runs. Use --only to pick one, e.g.
#   bash deepseek-v2-lite-chat/fixedk.sh --only k4

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "fixedk" "$@"

ep_run FixedK-k5 --num_experts_per_tok 5
ep_run FixedK-k4 --num_experts_per_tok 4   # 2/3 budget
ep_run FixedK-k3 --num_experts_per_tok 3
ep_run FixedK-k2 --num_experts_per_tok 2
ep_run FixedK-k1 --num_experts_per_tok 1

ep_end
