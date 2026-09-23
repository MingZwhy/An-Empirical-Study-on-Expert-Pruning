#!/usr/bin/env bash
set -euo pipefail

# Router hyper-parameter calibration for expert pruning.
#
# Produces the two artifacts Ban and DiEP read. Both passes walk a C4 sample
# through the checkpoint once; nothing is evaluated and nothing is generated.
#
# Ban is the slow half: it measures a KL divergence per layer per sample, so its
# cost is samples x layers. 512 samples on Qwen3-30B-A3B took 6h11m across four
# GPUs, which is why the default here is 128 -- about an hour and a half on that
# model, and what the published Ling-lite and GPT-OSS artifacts used. The
# Qwen3-30B and Qwen3-Next artifacts behind the paper used 512; set
# BAN_NSAMPLES=512 to match them, at four times the wait. DiEP used 128
# throughout and is cheaper per sample.
#
# DiEP output:
#   ${OUTPUT_DIR}/${MODEL_NAME}/c4_diep.pt
#   ${OUTPUT_DIR}/${MODEL_NAME}/c4_diep.json
# Ban output:
#   ${OUTPUT_DIR}/${MODEL_NAME}/c4_ban.pt
#   ${OUTPUT_DIR}/${MODEL_NAME}/c4_ban.json
#   ${OUTPUT_DIR}/${MODEL_NAME}/c4_ban_layer_sensitivity.png  (needs matplotlib,
#     which neither install script pulls in; skipped with a notice if absent)

# Calibration reads a C4 sample. Offline is opt-in: on a first run the dataset
# still has to be fetched, and forcing offline here turned that into a confusing
# cache miss rather than a download.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -z "${MODEL_PATH:-}" ]]; then
    cat >&2 <<'EOF'
[calib.sh] MODEL_PATH is not set.

  MODEL_PATH=/path/to/checkpoint MODEL_NAME=My-Model bash scripts/calib.sh

Or let the per-model wrapper do it:

  bash scripts/models/<model>/calibrate.sh
EOF
    exit 1
fi
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
OUTPUT_DIR="${OUTPUT_DIR:-calib_utils/results}"
MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}"

NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
PROGRESS="${PROGRESS:-1}"
RUN_DIEP="${RUN_DIEP:-1}"
RUN_BAN="${RUN_BAN:-1}"
DIEP_NSAMPLES="${DIEP_NSAMPLES:-128}"
DIEP_MAX_OUTPUTS_PER_EXPERT="${DIEP_MAX_OUTPUTS_PER_EXPERT:-2048}"
DIEP_MIN_TOKENS_FOR_CKA="${DIEP_MIN_TOKENS_FOR_CKA:-16}"
DIEP_CKA_DEVICE="${DIEP_CKA_DEVICE:-}"
BAN_NSAMPLES="${BAN_NSAMPLES:-128}"
BAN_K_PRUNED="${BAN_K_PRUNED:-3}"
BAN_TOPN="${BAN_TOPN:-1000}"
BAN_KL_POSITIONS="${BAN_KL_POSITIONS:-last}"

if [[ "${PROGRESS}" == "0" ]]; then
  DIEP_PROGRESS_ARGS=(--no-progress)
else
  DIEP_PROGRESS_ARGS=(--progress)
fi
if [[ -n "${DIEP_CKA_DEVICE}" ]]; then
  DIEP_CKA_DEVICE_ARGS=(--cka_device "${DIEP_CKA_DEVICE}")
else
  DIEP_CKA_DEVICE_ARGS=()
fi

mkdir -p "${MODEL_OUTPUT_DIR}"

if [[ "${RUN_DIEP}" != "0" ]]; then
  echo "Running DiEP calibration on C4..."
  python -m calib_utils.diep_calibration \
    --model_path "${MODEL_PATH}" \
    --dataset c4 \
    --nsamples "${DIEP_NSAMPLES}" \
    --seqlen "${SEQLEN}" \
    --seed "${SEED}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --device_map "${DEVICE_MAP}" \
    --max_outputs_per_expert "${DIEP_MAX_OUTPUTS_PER_EXPERT}" \
    --min_tokens_for_cka "${DIEP_MIN_TOKENS_FOR_CKA}" \
    --local_files_only \
    --output_prefix "${MODEL_OUTPUT_DIR}/c4_diep" \
    "${DIEP_CKA_DEVICE_ARGS[@]}" \
    "${DIEP_PROGRESS_ARGS[@]}"
fi

if [[ "${RUN_BAN}" != "0" ]]; then
  echo "Running Ban calibration on C4..."
  python -m calib_utils.ban_calibration \
    --model_path "${MODEL_PATH}" \
    --dataset c4 \
    --nsamples "${BAN_NSAMPLES}" \
    --seqlen "${SEQLEN}" \
    --seed "${SEED}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --device_map "${DEVICE_MAP}" \
    --k_pruned "${BAN_K_PRUNED}" \
    --topn "${BAN_TOPN}" \
    --kl_positions "${BAN_KL_POSITIONS}" \
    --local_files_only \
    --output_prefix "${MODEL_OUTPUT_DIR}/c4_ban" \
    "${DIEP_PROGRESS_ARGS[@]}"
fi

echo "Calibration finished."
echo "Results written to: ${MODEL_OUTPUT_DIR}"
