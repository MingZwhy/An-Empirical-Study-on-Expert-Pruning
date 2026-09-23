#!/usr/bin/env bash
#
# Fixed-K ladder -- gpt-oss-20b
#
# Keep the k highest-scoring experts per token and drop the rest, for
# every k from 3 down to 1. This is a config overlay: it rewrites
# num_experts_per_tok and lets the model's own router rank as usual, so
# it needs no engine patch and works in either environment.
#
# The whole ladder is 3 runs. Use --only to pick one, e.g.
#   bash gpt-oss-20b/fixedk.sh --only k3

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "fixedk" "$@"

ep_run FixedK-k3 --num_experts_per_tok 3   # 2/3 budget
ep_run FixedK-k2 --num_experts_per_tok 2
ep_run FixedK-k1 --num_experts_per_tok 1

ep_end
