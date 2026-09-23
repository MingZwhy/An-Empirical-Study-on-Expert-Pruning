#!/usr/bin/env bash
#
# Unpruned baseline -- Qwen3-Next-80B-A3B-Instruct
#
# The reference every pruned configuration is measured against: the
# released checkpoint with its own router, selecting all 10 experts
# per token. Nothing is overridden here.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "baseline" "$@"

ep_run Baseline-k10

ep_end
