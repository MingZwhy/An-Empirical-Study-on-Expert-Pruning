#!/bin/bash
# Sequential expanded fixed-shape ladder for one model.
#
# Usage:
#   run_fixed_shapes_sweep.sh <model_path> <result_name> <native_k> <cards>
#
# Each k runs vLLM tensor parallelism and then Hugging Face Transformers on the
# same physical cards.  HF uses one process with Accelerate layer sharding when
# more than one card is visible.  Completed JSON files make the sweep
# restartable.
set -u

MODEL=${1:?model path}
NAME=${2:?result name}
NATIVE=${3:?native k}
CARDS=${4:?CUDA card list}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=$REPO/results/_speed_fixed_expanded/$NAME
LOG=$REPO/local_logs/fixed_shapes_${NAME}.log
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
    complete = json.load(open(sys.argv[1])).get("complete") is True
except Exception:
    complete = False
raise SystemExit(0 if complete else 1)
PY
}

run_one() {
    local backend=$1 k=$2
    local output=$OUT/${backend}_k${k}.json
    if is_complete "$output"; then
        echo "[$(date '+%F %T')] skip complete $NAME $backend k=$k" | tee -a "$LOG"
        return 0
    fi

    echo "[$(date '+%F %T')] start $NAME $backend k=$k cards=$CARDS" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$CARDS python scripts/bench_fixed_shapes.py \
        --backend "$backend" \
        --model_path "$MODEL" \
        --k "$k" \
        --tp "$TP" \
        --out "$output" \
        --warmups 1 \
        --trials 2 >>"$LOG" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] done $NAME $backend k=$k rc=$rc" | tee -a "$LOG"
    return "$rc"
}

echo "[$(date '+%F %T')] sweep $NAME native_k=$NATIVE cards=$CARDS tp=$TP" | tee -a "$LOG"
for k in $(seq "$NATIVE" -1 1); do
    run_one vllm "$k" || echo "WARN: $NAME vllm k=$k failed; continuing" | tee -a "$LOG"
    run_one hf "$k" || echo "WARN: $NAME hf k=$k failed; continuing" | tee -a "$LOG"
done
echo "[$(date '+%F %T')] sweep finished $NAME" | tee -a "$LOG"
