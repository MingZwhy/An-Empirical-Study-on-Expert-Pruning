#!/bin/bash
# vLLM-only long-decode ladder: input=1024, bsz={1,4},
# output={2048,4096,8192}.
#
# Usage:
#   run_long_decode_sweep.sh <model_path> <result_name> <native_k> <cards>
set -u

MODEL=${1:?model path}
NAME=${2:?result name}
NATIVE=${3:?native k}
CARDS=${4:?CUDA card list}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=$REPO/results/_speed_long_decode/$NAME
LOG=$REPO/local_logs/long_decode_${NAME}.log
CELLS=$REPO/scripts/fixed_shapes_long_decode.json
TP=$(( $(tr -cd ',' <<<"$CARDS" | wc -c) + 1 ))
# Cooperative lock so a speed benchmark is not measured against a busy GPU.
LOCK="${SPEED_BENCH_LOCK:-${TMPDIR:-/tmp}/ep_speed_bench.lock}"
LOCK_OWNED=0

mkdir -p "$OUT" $REPO/local_logs
if [ ! -f "$LOCK" ]; then
    touch "$LOCK"
    LOCK_OWNED=1
fi
cleanup() {
    [ "$LOCK_OWNED" = 1 ] && rm -f "$LOCK"
}
trap cleanup EXIT

is_complete() {
    local path=$1
    [ -f "$path" ] || return 1
    python3 - "$path" <<'PY'
import json, sys
try:
    ok = json.load(open(sys.argv[1])).get("complete") is True
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

echo "[$(date '+%F %T')] long-decode sweep $NAME native_k=$NATIVE cards=$CARDS tp=$TP" | tee -a "$LOG"
for k in $(seq "$NATIVE" -1 1); do
    output=$OUT/vllm_k${k}.json
    if is_complete "$output"; then
        echo "[$(date '+%F %T')] skip complete $NAME k=$k" | tee -a "$LOG"
        continue
    fi
    echo "[$(date '+%F %T')] start $NAME k=$k" | tee -a "$LOG"
    # vLLM already profiles, compiles, and captures CUDA graphs at engine
    # startup. Avoid another full 2k+4k+8k generation as per-cell warmup.
    CUDA_VISIBLE_DEVICES=$CARDS python scripts/bench_fixed_shapes.py \
        --backend vllm \
        --model_path "$MODEL" \
        --k "$k" \
        --tp "$TP" \
        --out "$output" \
        --cells_json "$CELLS" \
        --warmups 0 \
        --trials 2 >>"$LOG" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] done $NAME k=$k rc=$rc" | tee -a "$LOG"
done
echo "[$(date '+%F %T')] long-decode sweep finished $NAME" | tee -a "$LOG"
