#!/usr/bin/env bash
#
# Calibration for Ban and DiEP -- gpt-oss-20b
#
# Ban and DiEP each read a tensor artifact computed once per model
# from a C4 sample: layer sensitivities for Ban, expert similarity
# for DiEP. Derived from this checkpoint, so not shipped.
#
# This wraps calib_utils; it needs the checkpoint but no sweep.
#
# Ban is the slow half, costing samples x layers. The default 128
# samples is roughly an hour and a half on a 48-layer model across
# four GPUs; the published Qwen artifacts used 512, which took
# 6h11m. BAN_NSAMPLES and DIEP_NSAMPLES override it.

EP_MODEL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EP_MODEL_HOME/../../lib/common.sh"
ep_begin "calibrate" "$@"

ep_info "calibrating on C4 -- reads the checkpoint once, evaluates nothing"
if (( EP_DRY_RUN )); then
    printf '  MODEL_PATH=%s MODEL_NAME=%s BAN_K_PRUNED=%s bash scripts/calib.sh\n' \
        "$EP_MODEL_PATH" "$EP_MODEL_NAME" "$EP_CALIB_BAN_K_PRUNED"
else
    MODEL_PATH="$EP_MODEL_PATH" MODEL_NAME="$EP_MODEL_NAME" \
        BAN_K_PRUNED="$EP_CALIB_BAN_K_PRUNED" \
        bash "$EP_REPO_ROOT/scripts/calib.sh"
fi
ep_end
