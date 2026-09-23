#!/usr/bin/env bash
# Queue native router-distribution observation jobs on an 8-GPU node (4× TP=2 lanes).
# One task per process so dataset labels stay exact. Skips completed outputs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"
HARNESS="${HARNESS:-lm_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/router_distribution}"
LANES=(0,1 2,3 4,5 6,7)
TP="${TENSOR_PARALLEL_SIZE:-2}"

declare -a JOBS=()
if [[ "$HARNESS" == "lm_eval" ]]; then
  TASKS=(arc_challenge arc_easy winogrande openbookqa)
  MAX_SAMPLES="${MAX_SAMPLES:-256}"
  for task in "${TASKS[@]}"; do
    JOBS+=("$task")
  done
elif [[ "$HARNESS" == "lmms_eval" ]]; then
  TASKS=(mmbench_en_dev_lite chartqa_lite gqa_lite textvqa_val_lite infovqa_val_lite \
         vizwiz_vqa_val_lite mmmu_val mmstar mmerealworld_lite)
  MAX_SAMPLES="${MAX_SAMPLES:-128}"
  for task in "${TASKS[@]}"; do
    JOBS+=("$task")
  done
else
  echo "Unsupported HARNESS=$HARNESS" >&2
  exit 1
fi

short_name="$(basename "${MODEL_PATH%/}")"
run_one() {
  local lane="$1"
  local task="$2"
  local out_dir="${OUTPUT_ROOT}/${short_name}/${task}"
  local dist_dir="${out_dir}/.router_distribution"
  local done_marker="${out_dir}/.completed"
  if [[ -f "$done_marker" ]]; then
    echo "[skip] $task already completed"
    return 0
  fi
  mkdir -p "$out_dir"
  IFS=',' read -r gpu_a gpu_b <<< "$lane"
  echo "[run] task=$task gpus=${gpu_a},${gpu_b} out=$out_dir"
  CUDA_VISIBLE_DEVICES="${gpu_a},${gpu_b}" python main.py \
    --harness "$HARNESS" \
    --model_path "$MODEL_PATH" \
    --tensor_parallel_size "$TP" \
    --max_samples "$MAX_SAMPLES" \
    --enforce_eager \
    --collect_router_distribution \
    --router_distribution_dir "$dist_dir" \
    --output_dir "$out_dir" \
    $( [[ "$HARNESS" == "lm_eval" ]] && echo --lm_eval_tasks "$task" ) \
    $( [[ "$HARNESS" == "lmms_eval" ]] && echo --lmms_eval_tasks "$task" )
  touch "$done_marker"
}

lane_idx=0
for task in "${JOBS[@]}"; do
  lane="${LANES[$((lane_idx % ${#LANES[@]}))]}"
  lane_idx=$((lane_idx + 1))
  run_one "$lane" "$task" &
  if (( lane_idx % ${#LANES[@]} == 0 )); then
    wait
  fi
done
wait

echo "All jobs finished. Merge with:"
echo "  python scripts/merge_router_distribution.py ${OUTPUT_ROOT}/${short_name}/* --output-dir ${OUTPUT_ROOT}/${short_name}/merged --label ${short_name}"
