#!/usr/bin/env bash

# This file is a command cookbook. Copy one command block into the terminal to
# run it; do not execute this file directly as a batch script.

# Checkpoint locations are site specific. Point MODELS_DIR at the directory that
# holds your local checkpoints, e.g.
#   export MODELS_DIR=$HOME/models
MODELS_DIR="${MODELS_DIR:?set MODELS_DIR to the directory containing your checkpoints}"
echo "This is a command cookbook. Copy one command block from this file and run it in the terminal."
exit 0

#
# Multimodal evaluation on lmms-eval, for the vision-language MoE checkpoints.
#
# Why a third harness: the other two only ever feed the model text. A VL MoE routes image tokens
# through the same experts as text tokens, so a budget calibrated on text says nothing about what the
# visual path can afford to lose — the question has to be asked again with images in the prompt.
#
# The nine candidate sets, each in its smallest published variant (which is what --lmms_eval_tasks
# defaults to). Sizes are per task, and all of them are already downloaded to the shared HF cache:
#
#   mmbench_en_dev_lite      500     chartqa_lite        500     gqa_lite            500
#   textvqa_val_lite         500     infovqa_val_lite    500     vizwiz_vqa_val_lite 500
#   mmmu_val                 900     mmstar             1500     mmerealworld_lite  1919
#
# The six *_lite sets are the published 500-question subsets from lmms-lab/LMMs-Eval-Lite. mmmu and
# mmstar have no lite release: mmmu's test split is unlabelled so val is the only scorable one, and
# mmstar ships as a single set. Rows stay comparable to each other but not to numbers quoted on the
# full sets, so the variant is recorded in every result file.
#
# Defaults worth knowing: --lmms_eval_num_fewshot is 0; the chat template is ON (the opposite of the
# text QA suites — these tasks are generative, the checkpoints are instruct versions, and the image
# placeholders live in the template), and --lmms_eval_no_chat_template turns it off for a control run;
# --lmms_eval_max_images is 8, because MMMU asks about up to seven images at once and vLLM rejects a
# prompt above the limit outright.
#
# The weights are read from the sweep's own filesystem rather than the slower shared one the standby
# models were downloaded to: a cold read there measures 40 MB/s, and on a busy node the first attempt at
# loading this checkpoint was managing about nine minutes per shard. That share is still the only path
# some nodes can see, so anything running there needs its own copy.
#
# Run these in the `qwen35` environment, not `expertpruning`: Qwen3-VL-MoE is newer than the vLLM
# commit the sweep environment pins, and the newer engine that knows it is exactly what `qwen35`
# already exists for. `conda activate qwen35` first.
#
# Two limits to know before planning a sweep (see docs/known_issues.md):
#
#   * Dynamic pruning does not work in this environment yet. vLLM 0.17.0 moved `fused_topk` to
#     `fused_moe/router/fused_topk_router.py` and gave it a `scoring_func` argument, so the router
#     patch needs a small port first. Unpruned baselines and Fixed-K (--num_experts_per_tok, which
#     rewrites the config and never touches the router) work today.
#   * Kimi-VL's language tower routes through grouped_topk (its config asks for noaux_tc) rather than
#     the fused_topk our patch replaces, so it needs a second piece of work on top of that port. A
#     pruned run stops on our own assertion instead of quietly producing unpruned numbers.
#


# 1. Unpruned reference for Qwen3-VL-30B-A3B (128 experts, top-8 — the same router shape as
#    Qwen3-30B-A3B). Two cards: 61 GB of bf16 weights does not leave room on one 71 GB card.
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --harness lmms_eval \
  --model_path $MODELS_DIR/Qwen3-VL-30B-A3B-Instruct \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.85 \
  --max_model_length 16384 \
  --output_dir results_lmms_eval/Qwen3-VL-30B-A3B-Instruct/Baseline-k8


# 2. Unpruned reference for Kimi-VL-A3B. One card is enough (16B total, ~32 GB in bf16).
CUDA_VISIBLE_DEVICES=0 python main.py \
  --harness lmms_eval \
  --model_path $MODELS_DIR/Kimi-VL-A3B-Instruct \
  --gpu_memory_utilization 0.85 \
  --max_model_length 16384 \
  --output_dir results_lmms_eval/Kimi-VL-A3B-Instruct/Baseline-k6


# 3. A single dataset, for a quick check or for splitting a sweep across cards.
CUDA_VISIBLE_DEVICES=0 python main.py \
  --harness lmms_eval \
  --lmms_eval_tasks chartqa_lite \
  --model_path $MODELS_DIR/Kimi-VL-A3B-Instruct \
  --gpu_memory_utilization 0.85 \
  --max_model_length 16384 \
  --output_dir results_lmms_eval/Kimi-VL-A3B-Instruct/Baseline-k6-chartqa


# 4. Fixed-K: the same average expert count reached by simply keeping fewer experts per token.
#    This is the bar a dynamic method has to clear, and it needs no router patch.
CUDA_VISIBLE_DEVICES=0 python main.py \
  --harness lmms_eval \
  --model_path $MODELS_DIR/Kimi-VL-A3B-Instruct \
  --num_experts_per_tok 4 \
  --gpu_memory_utilization 0.85 \
  --max_model_length 16384 \
  --output_dir results_lmms_eval/Kimi-VL-A3B-Instruct/FixedK-k4


# 5. Dynamic pruning, in the form it takes once the router patch covers grouped_topk. The pruning
#    options mean exactly what they mean in the text runs; only --harness changes.
CUDA_VISIBLE_DEVICES=0 python main.py \
  --harness lmms_eval \
  --model_path $MODELS_DIR/Kimi-VL-A3B-Instruct \
  --use_local_expert_router \
  --expert_pruning_method NAEE \
  --naee_beta 0.6 \
  --naee_k_min 2 \
  --gpu_memory_utilization 0.85 \
  --max_model_length 16384 \
  --output_dir results_lmms_eval/Kimi-VL-A3B-Instruct/NAEE-kmin2-beta0.6


# 6. Control run without the chat template, to check how much of a drop is the template rather than
#    the pruning.
CUDA_VISIBLE_DEVICES=0 python main.py \
  --harness lmms_eval \
  --lmms_eval_no_chat_template \
  --model_path $MODELS_DIR/Kimi-VL-A3B-Instruct \
  --gpu_memory_utilization 0.85 \
  --max_model_length 16384 \
  --output_dir results_lmms_eval/Kimi-VL-A3B-Instruct/Baseline-k6-nochat
