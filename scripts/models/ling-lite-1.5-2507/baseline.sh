#!/usr/bin/env bash
#
# Unpruned baseline -- Ling-lite-1.5-2507
#
# The reference every pruned configuration is measured against: the
# released checkpoint with its own router, selecting all 6 experts
# per token. Nothing is overridden here.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "baseline" "$@"

ep_run Baseline-k6

ep_end
