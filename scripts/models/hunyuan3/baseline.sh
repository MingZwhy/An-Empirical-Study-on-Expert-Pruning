#!/usr/bin/env bash
#
# Unpruned baseline -- Hy3
#
# The reference every pruned configuration is measured against: the
# released checkpoint with its own router, selecting all 8 experts
# per token. Nothing is overridden here.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "baseline" "$@"

ep_run Baseline-k8

ep_end
