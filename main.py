#!/usr/bin/env python3
"""使用 vLLM 部署模型并在 lighteval 上做评估。

本脚本直接从 ``3rdparty/lighteval/src`` 导入 lighteval，仓库对 lighteval 的本地
改动（见 ``3rdparty/lighteval`` 的 ``expert-pruning-mods`` 分支）会随子模块一
起生效，协作者只需：

1. 拉取子模块：``git submodule update --init --recursive``
2. 安装依赖：``bash install_expertpruning.sh``（在 ``expertpruning`` 环境下，主线，
   会改 vllm 源码）；或 ``bash install_qwen35.sh``（在 ``qwen35`` 环境下，只跑
   baseline、不改 vllm 源码）。两个环境共用本脚本和 ``tasks/``。
3. 运行本脚本即可。

CLI 与 ``Half-Experts-Candoall/main_backup_3rdparty_lighteval.py`` 保持一致，额外
暴露了 ``--enforce_eager`` / ``--cpu_offload_gb``（依赖本仓库对 lighteval 的
改动，见 ``3rdparty/lighteval/src/lighteval/models/vllm/vllm_model.py``）。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

# 优先使用仓库内的 lighteval 源码（附带本仓库的修改）；使用 resolve() 是因为
# vLLM 等库用 multiprocessing spawn 会再次执行本文件，需稳定得到仓库根目录。
REPO_ROOT = str(Path(__file__).resolve().parent)
LIGHTEVAL_SRC = os.path.join(REPO_ROOT, "3rdparty", "lighteval", "src")
VLLM_SRC = os.path.join(REPO_ROOT, "3rdparty", "vllm")
if not os.path.isdir(LIGHTEVAL_SRC):
    raise RuntimeError(
        f"未找到 {LIGHTEVAL_SRC}：请先执行 `git submodule update --init --recursive`。"
    )
if LIGHTEVAL_SRC not in sys.path:
    sys.path.insert(0, LIGHTEVAL_SRC)


def _vllm_submodule_is_the_installed_vllm() -> bool:
    """两条安装线装 vLLM 的方式不同，此处决定要不要把子模块源码放到 sys.path 最前面。

    `expertpruning` 用 `3rdparty/vllm` 的 editable 安装（路由补丁要改它的源码），
    `qwen35` 直接从 PyPI 装更新的 wheel、根本用不到这个子模块。而两条线常共用同一份
    clone，子模块无论如何都是 checkout 过的——无条件前置就会用一份和已装 torch
    对不上的源码把 wheel 顶掉（`import vllm` 时报 undefined symbol）。
    所以只在"已装的 vLLM 本来就是这个子模块"时才前置。
    """
    if not os.path.isdir(os.path.join(VLLM_SRC, "vllm")):
        return False
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.origin:
        return True  # 没装过：子模块就是唯一的来源
    return os.path.realpath(spec.origin).startswith(os.path.realpath(VLLM_SRC))


if _vllm_submodule_is_the_installed_vllm():
    if VLLM_SRC not in sys.path:
        sys.path.insert(0, VLLM_SRC)
elif importlib.util.find_spec("vllm") is None:
    raise RuntimeError(
        f"既没有装好的 vllm，也没找到 {VLLM_SRC}/vllm："
        "请先执行 `git submodule update --init --recursive` 并按 README 安装。"
    )
# 让自定义任务能通过 `from tasks import ...` 被子模块内部的 spec loader 找到
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _install_local_expert_pruning_router_from_env() -> None:
    """Install the local vLLM MoE router in subprocesses that re-import main."""
    if os.environ.get("EXPERT_PRUNING_PATCH_VLLM_ROUTER") == "1":
        from expert_pruning import install_vllm_expert_pruning_router

        install_vllm_expert_pruning_router(
            method=os.environ.get("EXPERT_PRUNING_METHOD", "none"),
            naee_beta=float(os.environ.get("EXPERT_PRUNING_NAEE_BETA",
                                           "0.3")),
            naee_k_min=int(os.environ.get("EXPERT_PRUNING_NAEE_K_MIN", "2")),
            dynamic_routing_threshold=float(
                os.environ.get("EXPERT_PRUNING_DYNAMIC_ROUTING_THRESHOLD",
                               "0.8")),
            dynamic_routing_score_source=os.environ.get(
                "EXPERT_PRUNING_DYNAMIC_ROUTING_SCORE_SOURCE", "raw"),
            dynamic_routing_next_rank_penalty=float(
                os.environ.get(
                    "EXPERT_PRUNING_DYNAMIC_ROUTING_NEXT_RANK_PENALTY",
                    "1.0")),
            layerwise_dynamic_base_threshold=float(
                os.environ.get(
                    "EXPERT_PRUNING_LAYERWISE_DYNAMIC_BASE_THRESHOLD",
                    "0.55")),
            layerwise_dynamic_layer_alpha=float(
                os.environ.get(
                    "EXPERT_PRUNING_LAYERWISE_DYNAMIC_LAYER_ALPHA", "0.15")),
            layerwise_dynamic_num_layers=int(
                os.environ.get(
                    "EXPERT_PRUNING_LAYERWISE_DYNAMIC_NUM_LAYERS", "48")),
            layerwise_dynamic_k_min=int(
                os.environ.get("EXPERT_PRUNING_LAYERWISE_DYNAMIC_K_MIN",
                               "1")),
            budget_dynamic_easy_k=int(
                os.environ.get("EXPERT_PRUNING_BUDGET_DYNAMIC_EASY_K", "4")),
            budget_dynamic_base_k=int(
                os.environ.get("EXPERT_PRUNING_BUDGET_DYNAMIC_BASE_K", "5")),
            budget_dynamic_hard_k=int(
                os.environ.get("EXPERT_PRUNING_BUDGET_DYNAMIC_HARD_K", "6")),
            budget_dynamic_easy_threshold=float(
                os.environ.get(
                    "EXPERT_PRUNING_BUDGET_DYNAMIC_EASY_THRESHOLD", "0.90")),
            budget_dynamic_hard_threshold=float(
                os.environ.get(
                    "EXPERT_PRUNING_BUDGET_DYNAMIC_HARD_THRESHOLD", "0.75")),
            budget_dynamic_score_topn=int(
                os.environ.get("EXPERT_PRUNING_BUDGET_DYNAMIC_SCORE_TOPN",
                               "3")),
            layer_budget_dynamic_easy_layer_alpha=float(
                os.environ.get(
                    "EXPERT_PRUNING_LAYER_BUDGET_DYNAMIC_EASY_LAYER_ALPHA",
                    "0.08")),
            layer_budget_dynamic_hard_layer_alpha=float(
                os.environ.get(
                    "EXPERT_PRUNING_LAYER_BUDGET_DYNAMIC_HARD_LAYER_ALPHA",
                    "0.08")),
            layer_budget_dynamic_num_layers=int(
                os.environ.get(
                    "EXPERT_PRUNING_LAYER_BUDGET_DYNAMIC_NUM_LAYERS",
                    "48")),
            band_layer_budget_middle_start=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_MIDDLE_START",
                    "19")),
            band_layer_budget_middle_end=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_MIDDLE_END",
                    "28")),
            band_layer_budget_per_layer_k=os.environ.get(
                "EXPERT_PRUNING_BAND_LAYER_BUDGET_PER_LAYER_K", ""),
            band_layer_budget_easy_layers=os.environ.get(
                "EXPERT_PRUNING_BAND_LAYER_BUDGET_EASY_LAYERS", ""),
            band_layer_budget_hard_layers=os.environ.get(
                "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYERS", ""),
            band_layer_budget_token_gate_ratio=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_TOKEN_GATE_RATIO",
                    "-1")),
            band_layer_budget_extra_middle_start=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_EXTRA_MIDDLE_START",
                    "-1")),
            band_layer_budget_extra_middle_end=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_EXTRA_MIDDLE_END",
                    "-1")),
            band_layer_budget_middle_phase=os.environ.get(
                "EXPERT_PRUNING_BAND_LAYER_BUDGET_MIDDLE_PHASE", "all"),
            band_layer_budget_easy_start=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_EASY_START",
                    "0")),
            band_layer_budget_easy_end=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_EASY_END",
                    "47")),
            band_layer_budget_easy_phase=os.environ.get(
                "EXPERT_PRUNING_BAND_LAYER_BUDGET_EASY_PHASE", "all"),
            band_layer_budget_hard_phase=os.environ.get(
                "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PHASE", "all"),
            band_layer_budget_easy_tail_threshold=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_EASY_TAIL_THRESHOLD",
                    "1.0")),
            band_layer_budget_hard_min_rank_weight=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_MIN_RANK_WEIGHT",
                    "0.0")),
            band_layer_budget_hard_max_rank_weight=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_MAX_RANK_WEIGHT",
                    "1.0")),
            band_layer_budget_hard_decode_min_rank_weight=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_MIN_RANK_WEIGHT",
                    "-1")),
            band_layer_budget_mixed_rescue_min_rank_weight=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_MIXED_RESCUE_MIN_RANK_WEIGHT",
                    "-1")),
            band_layer_budget_mixed_rescue_max_concentration=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_MIXED_RESCUE_MAX_CONCENTRATION",
                    "-1")),
            band_layer_budget_hard_prefill_max_tokens=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_TOKENS",
                    "-1")),
            band_layer_budget_hard_prefill_max_segment_tokens=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_TOKENS",
                    "-1")),
            band_layer_budget_hard_prefill_max_density=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_DENSITY",
                    "-1")),
            band_layer_budget_hard_prefill_min_density=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MIN_DENSITY",
                    "-1")),
            band_layer_budget_hard_prefill_max_segment_hard_ratio=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_HARD_RATIO",
                    "-1")),
            band_layer_budget_hard_prefill_segment_cap_score=os.environ.get(
                "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_SEGMENT_CAP_SCORE",
                "concentration"),
            band_layer_budget_hard_prefill_conditional_segment_hard_ratio=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_SEGMENT_HARD_RATIO",
                    "-1")),
            band_layer_budget_hard_prefill_conditional_min_density=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_DENSITY",
                    "-1")),
            band_layer_budget_hard_prefill_conditional_min_seq_len=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_SEQ_LEN",
                    "-1")),
            band_layer_budget_hard_prefill_conditional_token_ids=(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_IDS",
                    "")),
            band_layer_budget_hard_prefill_conditional_token_ngrams=(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_NGRAMS",
                    "")),
            band_layer_budget_hard_prefill_marker_segment_hard_ratio=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_SEGMENT_HARD_RATIO",
                    "-1")),
            band_layer_budget_hard_prefill_marker_token_ids=(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_IDS",
                    "")),
            band_layer_budget_hard_prefill_marker_token_ngrams=(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_NGRAMS",
                    "")),
            band_layer_budget_hard_prefill_exclude_token_ids=(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_EXCLUDE_TOKEN_IDS",
                    "")),
            band_layer_budget_hard_prefill_min_relative_pos=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MIN_RELATIVE_POS",
                    "-1")),
            band_layer_budget_hard_prefill_max_relative_pos=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_RELATIVE_POS",
                    "-1")),
            band_layer_budget_hard_prefill_start=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_START",
                    "-1")),
            band_layer_budget_hard_prefill_end=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_END",
                    "-1")),
            band_layer_budget_hard_decode_min_seq_len=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_MIN_SEQ_LEN",
                    "-1")),
            band_layer_budget_hard_decode_max_seq_len=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_MAX_SEQ_LEN",
                    "-1")),
            band_layer_budget_hard_decode_seq_len_scope=os.environ.get(
                "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_SEQ_LEN_SCOPE",
                "all"),
            band_layer_budget_hard_decode_min_offset=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_MIN_OFFSET",
                    "-1")),
            band_layer_budget_hard_decode_max_offset=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_MAX_OFFSET",
                    "-1")),
            band_layer_budget_hard_decode_start=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_START",
                    "-1")),
            band_layer_budget_hard_decode_end=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_END",
                    "-1")),
            band_layer_budget_hard_decode_token_ids=(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_TOKEN_IDS",
                    "")),
            band_layer_budget_hard_layer_prior_start=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_START",
                    "-1")),
            band_layer_budget_hard_layer_prior_end=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_END",
                    "-1")),
            band_layer_budget_hard_layer_prior_extra_start=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_START",
                    "-1")),
            band_layer_budget_hard_layer_prior_extra_end=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_END",
                    "-1")),
            band_layer_budget_hard_layer_prior_alpha=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_ALPHA",
                    "0")),
            band_layer_budget_hard_layer_sensitivity_alpha=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_SENSITIVITY_ALPHA",
                    "0")),
            band_layer_budget_late_concentration_start=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_LATE_CONCENTRATION_START",
                    "-1")),
            band_layer_budget_late_concentration_threshold=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_LATE_CONCENTRATION_THRESHOLD",
                    "-1")),
            band_layer_budget_late_rescue_start=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_LATE_RESCUE_START",
                    "-1")),
            band_layer_budget_late_rescue_concentration_threshold=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_LATE_RESCUE_CONCENTRATION_THRESHOLD",
                    "-1")),
            band_layer_budget_late_rescue_mixed_max=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_LATE_RESCUE_MIXED_MAX",
                    "-1")),
            band_layer_budget_prompt_profile1_token_ngrams=os.environ.get(
                "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_TOKEN_NGRAMS",
                ""),
            band_layer_budget_prompt_profile1_hard_threshold=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_THRESHOLD",
                    "-1")),
            band_layer_budget_prompt_profile1_hard_min_rank_weight=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_MIN_RANK_WEIGHT",
                    "-1")),
            band_layer_budget_prompt_profile1_middle_start=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_START",
                    "-1")),
            band_layer_budget_prompt_profile1_middle_end=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_END",
                    "-1")),
            band_layer_budget_prompt_profile1_prefill_segment_hard_ratio=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_PREFILL_SEGMENT_HARD_RATIO",
                    "-1")),
            band_layer_budget_prompt_profile2_token_ngrams=os.environ.get(
                "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_TOKEN_NGRAMS",
                ""),
            band_layer_budget_prompt_profile2_hard_threshold=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_THRESHOLD",
                    "-1")),
            band_layer_budget_prompt_profile2_hard_min_rank_weight=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_MIN_RANK_WEIGHT",
                    "-1")),
            band_layer_budget_prompt_profile2_middle_start=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_START",
                    "-1")),
            band_layer_budget_prompt_profile2_middle_end=int(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_END",
                    "-1")),
            band_layer_budget_prompt_profile2_prefill_segment_hard_ratio=float(
                os.environ.get(
                    "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_PREFILL_SEGMENT_HARD_RATIO",
                    "-1")),
            diep_artifact_path=os.environ.get(
                "EXPERT_PRUNING_DIEP_ARTIFACT_PATH"),
            diep_pruning_mode=os.environ.get("EXPERT_PRUNING_DIEP_PRUNING_MODE",
                                             "independent"),
            diep_use_gamma1=os.environ.get(
                "EXPERT_PRUNING_DIEP_USE_GAMMA1", "0") == "1",
            diep_gamma_alpha=float(
                os.environ.get("EXPERT_PRUNING_DIEP_GAMMA_ALPHA", "1.0")),
            diep_threshold_cap=(
                float(os.environ["EXPERT_PRUNING_DIEP_THRESHOLD_CAP"])
                if os.environ.get("EXPERT_PRUNING_DIEP_THRESHOLD_CAP") else None),
            ban_artifact_path=os.environ.get(
                "EXPERT_PRUNING_BAN_ARTIFACT_PATH"),
            ban_lambda=float(os.environ.get("EXPERT_PRUNING_BAN_LAMBDA",
                                            "0.7")),
            ban_k_min=int(os.environ.get("EXPERT_PRUNING_BAN_K_MIN", "3")),
            eac_alpha=float(os.environ.get("EXPERT_PRUNING_EAC_ALPHA",
                                           "0.5")),
            biased_renorm_keep_topn=int(
                os.environ.get("EXPERT_PRUNING_BIASED_RENORM_KEEP_TOPN",
                               "3")),
            router_value_artifact_path=os.environ.get(
                "EXPERT_PRUNING_ROUTER_VALUE_ARTIFACT_PATH"),
            router_value_base_k=int(
                os.environ.get("EXPERT_PRUNING_ROUTER_VALUE_BASE_K", "3")),
            router_value_quota_ratio=float(
                os.environ.get("EXPERT_PRUNING_ROUTER_VALUE_QUOTA_RATIO",
                               "0.10")),
            router_value_selection_mode=os.environ.get(
                "EXPERT_PRUNING_ROUTER_VALUE_SELECTION_MODE", "threshold"),
            router_value_swap_direction=os.environ.get(
                "EXPERT_PRUNING_ROUTER_VALUE_SWAP_DIRECTION", "both"),
            mc_moe_protection_ratio=float(
                os.environ.get("EXPERT_PRUNING_MC_MOE_PROTECTION_RATIO",
                               "0.02")),
            debug=os.environ.get("EXPERT_PRUNING_DEBUG", "0") == "1",
            debug_max_prints=int(
                os.environ.get("EXPERT_PRUNING_DEBUG_MAX_PRINTS", "8")),
            attention_sink_probe=os.environ.get(
                "EXPERT_PRUNING_ATTENTION_SINK_PROBE", "0") == "1",
            attention_sink_probe_max_prints=int(
                os.environ.get("EXPERT_PRUNING_ATTENTION_SINK_PROBE_MAX_PRINTS",
                               "8")),
            attention_sink_probe_max_query_tokens=int(
                os.environ.get(
                    "EXPERT_PRUNING_ATTENTION_SINK_PROBE_MAX_QUERY_TOKENS",
                    "2048")),
            attention_sink_probe_max_key_tokens=int(
                os.environ.get(
                    "EXPERT_PRUNING_ATTENTION_SINK_PROBE_MAX_KEY_TOKENS",
                    "2048")),
            attention_sink_probe_topk=int(
                os.environ.get("EXPERT_PRUNING_ATTENTION_SINK_PROBE_TOPK",
                               "4")),
        )


def _install_router_distribution_from_env() -> None:
    """Install native router distribution observer in vLLM worker subprocesses."""
    if os.environ.get("ROUTER_DISTRIBUTION_ENABLED") != "1":
        return
    from expert_pruning.vllm_patch import install_router_distribution_observer

    install_router_distribution_observer(
        output_dir=os.environ["ROUTER_DISTRIBUTION_DIR"],
        sample_rate=float(
            os.environ.get("ROUTER_DISTRIBUTION_SAMPLE_RATE", "0.005")),
        sample_cap=int(os.environ.get("ROUTER_DISTRIBUTION_SAMPLE_CAP",
                                        "2048")),
        histogram_bins=int(
            os.environ.get("ROUTER_DISTRIBUTION_HISTOGRAM_BINS", "16")),
        flush_interval=int(
            os.environ.get("ROUTER_DISTRIBUTION_FLUSH_INTERVAL", "512")),
        dataset=os.environ.get("ROUTER_DISTRIBUTION_DATASET", ""),
        model_path=os.environ.get("ROUTER_DISTRIBUTION_MODEL_PATH", ""),
        harness=os.environ.get("ROUTER_DISTRIBUTION_HARNESS", ""),
        task=os.environ.get("ROUTER_DISTRIBUTION_TASK", ""),
        seed=int(os.environ.get("ROUTER_DISTRIBUTION_SEED", "0")),
    )


_install_local_expert_pruning_router_from_env()
_install_router_distribution_from_env()


def _configure_hf_dataset_cache_mode() -> None:
    """Use cached HF datasets directly when the configured mirror is unreliable."""
    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if endpoint != "https://hf-mirror.com":
        return

    false_values = {"0", "false", "no", "off"}
    if any(
        os.environ.get(name, "").strip().lower() in false_values
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    ):
        return

    changed = []
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        if not os.environ.get(name):
            os.environ[name] = "1"
            changed.append(name)

    if changed:
        print(
            "HF datasets: detected HF_ENDPOINT=https://hf-mirror.com; "
            f"set {', '.join(changed)}=1 to use local dataset cache directly."
        )


def get_short_model_name(model_path: str) -> str:
    """从模型路径或 HF id 得到简短名称，仅用最后一级（如 Qwen3-30B-A3B-Instruct-2507）。"""
    return model_path.rstrip("/").split("/")[-1]


# 每 token 专家数的字段定位与改写放在 expert_pruning.config_shims：那是纯 config
# 逻辑，不该被本文件顶部的 submodule 守卫挡住，否则裸 clone 里连它的单元测试都跑不了。
from expert_pruning.config_shims import (  # noqa: E402
    _MOE_TOPK_SITES,
    get_moe_num_experts_per_tok as _get_moe_num_experts_per_tok,
    set_moe_num_experts_per_tok as _set_moe_num_experts_per_tok,
)


def prepare_model_dir_with_num_experts_per_tok(model_path: str, num_experts_per_tok: int) -> str:
    """为 MoE 模型创建一份只改了「每 token 专家数」的「视图」目录（其余文件 symlink）。

    返回该目录路径，供 vLLM 加载。仅当 ``model_path`` 为本地目录且存在
    ``config.json`` 时有效。字段名与嵌套位置各家不同，由 ``_moe_topk_site``
    就地定位后写回原键，见其上方注释；原 checkpoint 不会被修改。
    """
    model_path = os.path.abspath(model_path)
    if not os.path.isdir(model_path):
        raise ValueError(f"num_experts_per_tok 仅支持本地模型目录，当前 model_path 不是目录: {model_path}")
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"未找到 config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if _get_moe_num_experts_per_tok(config) is None:
        raise ValueError(
            "config.json 中找不到每 token 专家数字段（顶层或 text_config 下的 "
            f"{' / '.join(sorted({k for _, k in _MOE_TOPK_SITES}))}），"
            f"当前可能不是 MoE 模型: {config_path}")
    _set_moe_num_experts_per_tok(config, num_experts_per_tok)
    override_root = os.path.join(REPO_ROOT, ".moe_override")
    short_name = get_short_model_name(model_path)
    # Two checkpoints can share a basename, so the source path has to be part of the
    # directory name; otherwise the second model silently reuses the first one's links.
    # The source listing is part of it too, so a view is never stale and never has to be
    # rewritten in place — rewriting raced with concurrent runs reading the same view and
    # made vLLM fail on a shard that vanished mid-load.
    source_digest = hashlib.sha256(
        "\n".join([model_path] + _source_listing_lines(model_path)).encode()
    ).hexdigest()[:12]
    dest_dir = os.path.join(
        override_root,
        f"{short_name}_k{num_experts_per_tok}_{source_digest}")
    if os.path.isdir(dest_dir):
        return dest_dir

    # Build somewhere private, then publish with a single atomic rename.
    os.makedirs(override_root, exist_ok=True)
    tmp_dir = f"{dest_dir}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    os.makedirs(tmp_dir)
    try:
        for name in os.listdir(model_path):
            src = os.path.join(model_path, name)
            dst = os.path.join(tmp_dir, name)
            if name == "config.json":
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                os.symlink(src, dst)
        try:
            os.rename(tmp_dir, dest_dir)
        except OSError:
            # Another process published the same view first; it is byte-for-byte the view
            # we just built, so use theirs.
            if not os.path.isdir(dest_dir):
                raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return dest_dir


def _source_listing_lines(model_path: str) -> list[str]:
    """Identify the source checkpoint's contents, so a changed one gets a fresh view."""
    lines = []
    for entry in sorted(os.scandir(model_path), key=lambda e: e.name):
        try:
            stat = entry.stat(follow_symlinks=True)
            lines.append(f"{entry.name}:{stat.st_size}:{int(stat.st_mtime)}")
        except OSError:
            lines.append(f"{entry.name}:missing")
    return lines


def _make_expert_pruning_stats_dir() -> str:
    stats_dir = Path(REPO_ROOT) / ".expert_pruning_stats" / uuid.uuid4().hex
    stats_dir.mkdir(parents=True, exist_ok=True)
    return str(stats_dir)


def _make_router_distribution_dir(base_dir: str | None = None) -> str:
    root = Path(base_dir) if base_dir else Path(REPO_ROOT) / ".router_distribution"
    dist_dir = root / uuid.uuid4().hex
    dist_dir.mkdir(parents=True, exist_ok=True)
    return str(dist_dir)


def _router_distribution_dataset_label(args) -> str:
    if args.harness == "lm_eval":
        return args.lm_eval_tasks.replace(",", "_")
    if args.harness == "lmms_eval":
        return args.lmms_eval_tasks.replace(",", "_")
    return args.tasks.replace(",", "_") if args.tasks else "lighteval"


def _clear_router_distribution_env() -> None:
    for key in (
            "ROUTER_DISTRIBUTION_ENABLED",
            "ROUTER_DISTRIBUTION_DIR",
            "ROUTER_DISTRIBUTION_SAMPLE_RATE",
            "ROUTER_DISTRIBUTION_SAMPLE_CAP",
            "ROUTER_DISTRIBUTION_HISTOGRAM_BINS",
            "ROUTER_DISTRIBUTION_FLUSH_INTERVAL",
            "ROUTER_DISTRIBUTION_DATASET",
            "ROUTER_DISTRIBUTION_MODEL_PATH",
            "ROUTER_DISTRIBUTION_HARNESS",
            "ROUTER_DISTRIBUTION_TASK",
            "ROUTER_DISTRIBUTION_SEED",
    ):
        os.environ.pop(key, None)


def _moe_routing_is_inlined_by_compile(model_path: str) -> bool:
    """Would this model's expert selection be traced away by torch.compile?

    Unquantized MoE layers reach the experts through a vLLM custom op, which Dynamo
    treats as opaque, so ``fused_topk`` keeps running as Python on every forward and the
    router can record statistics. Quantized MoE methods (mxfp4, for instance) instead
    call ``select_experts`` inline, so Dynamo traces the routing once and replays it as
    graph ops: the pruning still applies, but the Python-side counters never run again
    and ``average_selected_experts`` — the x-axis of every comparison — is lost.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return False
    return bool(config.get("quantization_config"))


def _update_digest_from_file(digest, path: Path) -> None:
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)


def _expert_pruning_cache_fingerprint() -> str:
    """Fingerprint router behavior omitted from Lighteval's model config."""
    operational_keys = {
        "EXPERT_PRUNING_STATS_DIR",
        "EXPERT_PRUNING_STATS_FLUSH_INTERVAL",
        "EXPERT_PRUNING_DEBUG",
        "EXPERT_PRUNING_DEBUG_MAX_PRINTS",
        "EXPERT_PRUNING_ATTENTION_SINK_PROBE_MAX_PRINTS",
    }
    settings = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("EXPERT_PRUNING_") and key not in operational_keys
    }

    digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    )
    for key, value in settings.items():
        if not key.endswith("_ARTIFACT_PATH") or not value:
            continue
        artifact = Path(value).expanduser()
        if artifact.is_file():
            digest.update(key.encode())
            _update_digest_from_file(digest, artifact)

    repo_root = Path(REPO_ROOT)
    source_paths = [Path(__file__).resolve()]
    for source_root in (
        repo_root / "expert_pruning",
        repo_root / "tasks",
        repo_root / "3rdparty" / "lighteval" / "src" / "lighteval",
        repo_root / "3rdparty" / "vllm" / "vllm" / "model_executor" / "models",
        repo_root
        / "3rdparty"
        / "vllm"
        / "vllm"
        / "model_executor"
        / "layers"
        / "fused_moe",
    ):
        source_paths.extend(source_root.rglob("*.py"))
    for source_path in sorted(set(source_paths)):
        digest.update(str(source_path.relative_to(repo_root)).encode())
        _update_digest_from_file(digest, source_path)
    return digest.hexdigest()[:16]


def _new_rank_weight_aggregate() -> dict:
    return {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
        "hist": [],
    }


def _new_segment_density_aggregate() -> dict:
    aggregate = _new_rank_weight_aggregate()
    aggregate["hard_tokens"] = 0
    aggregate["segment_tokens"] = 0
    return aggregate


def _merge_rank_weight_aggregate(aggregate: dict, item: dict) -> None:
    count = int(item.get("count", 0) or 0)
    aggregate["count"] += count
    aggregate["sum"] += float(item.get("sum", 0.0) or 0.0)
    item_min = item.get("min")
    item_max = item.get("max")
    if item_min is not None:
        aggregate["min"] = (
            float(item_min) if aggregate["min"] is None else
            min(float(item_min), aggregate["min"]))
    if item_max is not None:
        aggregate["max"] = (
            float(item_max) if aggregate["max"] is None else
            max(float(item_max), aggregate["max"]))
    hist = [int(v) for v in item.get("hist", [])]
    if len(aggregate["hist"]) < len(hist):
        aggregate["hist"].extend([0] * (len(hist) - len(aggregate["hist"])))
    for idx, value in enumerate(hist):
        aggregate["hist"][idx] += value


def _merge_segment_density_aggregate(aggregate: dict, item: dict) -> None:
    _merge_rank_weight_aggregate(aggregate, item)
    aggregate["hard_tokens"] += int(item.get("hard_tokens", 0) or 0)
    aggregate["segment_tokens"] += int(item.get("segment_tokens", 0) or 0)


def _finalize_rank_weight_aggregate(aggregate: dict) -> dict:
    count = int(aggregate["count"])
    return {
        "count": count,
        "sum": float(aggregate["sum"]),
        "mean": float(aggregate["sum"]) / count if count else 0.0,
        "min": aggregate["min"],
        "max": aggregate["max"],
        "hist": [int(v) for v in aggregate["hist"]],
    }


def _finalize_segment_density_aggregate(aggregate: dict) -> dict:
    result = _finalize_rank_weight_aggregate(aggregate)
    hard_tokens = int(aggregate.get("hard_tokens", 0))
    segment_tokens = int(aggregate.get("segment_tokens", 0))
    result["hard_tokens"] = hard_tokens
    result["segment_tokens"] = segment_tokens
    result["token_weighted_mean"] = (
        hard_tokens / segment_tokens if segment_tokens else 0.0)
    return result


def _collect_expert_pruning_stats(stats_dir: str | None) -> dict | None:
    if not stats_dir:
        return None
    stats_path = Path(stats_dir)
    if not stats_path.is_dir():
        return None

    total_selected = 0.0
    total_tokens = 0
    total_calls = 0
    worker_stats = []
    method = None
    topk = None
    naee_beta = None
    naee_k_min = None
    dynamic_routing_threshold = None
    dynamic_routing_score_source = None
    dynamic_routing_next_rank_penalty = None
    layerwise_dynamic_base_threshold = None
    layerwise_dynamic_layer_alpha = None
    layerwise_dynamic_num_layers = None
    layerwise_dynamic_k_min = None
    budget_dynamic_easy_k = None
    budget_dynamic_base_k = None
    budget_dynamic_hard_k = None
    budget_dynamic_easy_threshold = None
    budget_dynamic_hard_threshold = None
    budget_dynamic_score_topn = None
    layer_budget_dynamic_easy_layer_alpha = None
    layer_budget_dynamic_hard_layer_alpha = None
    layer_budget_dynamic_num_layers = None
    band_layer_budget_middle_start = None
    band_layer_budget_middle_end = None
    band_layer_budget_extra_middle_start = None
    band_layer_budget_extra_middle_end = None
    band_layer_budget_middle_phase = None
    band_layer_budget_easy_start = None
    band_layer_budget_easy_end = None
    band_layer_budget_easy_phase = None
    band_layer_budget_hard_phase = None
    band_layer_budget_easy_tail_threshold = None
    band_layer_budget_hard_min_rank_weight = None
    band_layer_budget_hard_max_rank_weight = None
    band_layer_budget_hard_decode_min_rank_weight = None
    band_layer_budget_mixed_rescue_min_rank_weight = None
    band_layer_budget_mixed_rescue_max_concentration = None
    band_layer_budget_hard_prefill_max_tokens = None
    band_layer_budget_hard_prefill_max_segment_tokens = None
    band_layer_budget_hard_prefill_max_density = None
    band_layer_budget_hard_prefill_min_density = None
    band_layer_budget_hard_prefill_max_segment_hard_ratio = None
    band_layer_budget_hard_prefill_segment_cap_score = None
    band_layer_budget_hard_prefill_conditional_segment_hard_ratio = None
    band_layer_budget_hard_prefill_conditional_min_density = None
    band_layer_budget_hard_prefill_conditional_min_seq_len = None
    band_layer_budget_hard_prefill_conditional_token_ids = None
    band_layer_budget_hard_prefill_conditional_token_ngrams = None
    band_layer_budget_hard_prefill_marker_segment_hard_ratio = None
    band_layer_budget_hard_prefill_marker_token_ids = None
    band_layer_budget_hard_prefill_marker_token_ngrams = None
    band_layer_budget_hard_prefill_exclude_token_ids = None
    band_layer_budget_hard_prefill_min_relative_pos = None
    band_layer_budget_hard_prefill_max_relative_pos = None
    band_layer_budget_hard_prefill_start = None
    band_layer_budget_hard_prefill_end = None
    band_layer_budget_hard_decode_min_seq_len = None
    band_layer_budget_hard_decode_max_seq_len = None
    band_layer_budget_hard_decode_seq_len_scope = None
    band_layer_budget_hard_decode_min_offset = None
    band_layer_budget_hard_decode_max_offset = None
    band_layer_budget_hard_decode_start = None
    band_layer_budget_hard_decode_end = None
    band_layer_budget_hard_decode_token_ids = None
    band_layer_budget_hard_layer_prior_start = None
    band_layer_budget_hard_layer_prior_end = None
    band_layer_budget_hard_layer_prior_extra_start = None
    band_layer_budget_hard_layer_prior_extra_end = None
    band_layer_budget_hard_layer_prior_alpha = None
    band_layer_budget_hard_layer_sensitivity_alpha = None
    band_layer_budget_late_concentration_start = None
    band_layer_budget_late_concentration_threshold = None
    band_layer_budget_late_rescue_start = None
    band_layer_budget_late_rescue_concentration_threshold = None
    band_layer_budget_late_rescue_mixed_max = None
    band_layer_budget_prompt_profile1_token_ngrams = None
    band_layer_budget_prompt_profile1_hard_threshold = None
    band_layer_budget_prompt_profile1_hard_min_rank_weight = None
    band_layer_budget_prompt_profile1_middle_start = None
    band_layer_budget_prompt_profile1_middle_end = None
    band_layer_budget_prompt_profile1_prefill_segment_hard_ratio = None
    band_layer_budget_prompt_profile2_token_ngrams = None
    band_layer_budget_prompt_profile2_hard_threshold = None
    band_layer_budget_prompt_profile2_hard_min_rank_weight = None
    band_layer_budget_prompt_profile2_middle_start = None
    band_layer_budget_prompt_profile2_middle_end = None
    band_layer_budget_prompt_profile2_prefill_segment_hard_ratio = None
    diep_artifact_path = None
    diep_pruning_mode = None
    eac_alpha = None
    mc_moe_protection_ratio = None
    biased_renorm_keep_topn = None
    ban_artifact_path = None
    ban_lambda = None
    ban_k_min = None
    total_protected_tokens = 0
    eac_prefill_tokens = 0
    eac_segments = 0
    eac_calls_with_segments = 0
    eac_pruned_selections = 0
    eac_disabled_experts = 0
    ban_total_tokens = 0
    ban_total_k = 0.0
    budget_easy_tokens = 0
    budget_base_tokens = 0
    budget_hard_tokens = 0
    budget_layer_counts: dict[int, list[int]] = {}
    budget_prompt_profile_counts: dict[int, list[int]] = {}
    budget_stage_counts: dict[str, list[int]] = {}
    budget_hard_rank_weight_bounds = None
    budget_hard_rank_weight_candidate = _new_rank_weight_aggregate()
    budget_hard_rank_weight_selected = _new_rank_weight_aggregate()
    budget_hard_seq_len_bounds = None
    budget_hard_seq_len_candidate = _new_rank_weight_aggregate()
    budget_hard_seq_len_selected = _new_rank_weight_aggregate()
    budget_hard_decode_offset_bounds = None
    budget_hard_decode_offset_candidate = _new_rank_weight_aggregate()
    budget_hard_decode_offset_selected = _new_rank_weight_aggregate()
    budget_prefill_segment_density_bounds = None
    budget_prefill_segment_density_candidate = (
        _new_segment_density_aggregate())
    budget_prefill_segment_density_selected = (
        _new_segment_density_aggregate())
    budget_prefill_segment_density_by_layer: dict[int, dict[str, dict]] = {}

    for path in sorted(stats_path.glob("routing_stats_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                item = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        selected = float(item.get("total_selected_experts", 0.0))
        tokens = int(item.get("total_routed_tokens", 0))
        calls = int(item.get("total_routing_calls", 0))
        if tokens <= 0:
            continue

        total_selected += selected
        total_tokens += tokens
        total_calls += calls
        method = method or item.get("method")
        topk = topk if topk is not None else item.get("topk")
        naee_beta = naee_beta if naee_beta is not None else item.get("naee_beta")
        naee_k_min = naee_k_min if naee_k_min is not None else item.get("naee_k_min")
        dynamic_routing_threshold = (
            dynamic_routing_threshold
            if dynamic_routing_threshold is not None else
            item.get("dynamic_routing_threshold"))
        dynamic_routing_score_source = (
            dynamic_routing_score_source
            if dynamic_routing_score_source is not None else
            item.get("dynamic_routing_score_source"))
        dynamic_routing_next_rank_penalty = (
            dynamic_routing_next_rank_penalty
            if dynamic_routing_next_rank_penalty is not None else
            item.get("dynamic_routing_next_rank_penalty"))
        layerwise_dynamic_base_threshold = (
            layerwise_dynamic_base_threshold
            if layerwise_dynamic_base_threshold is not None else
            item.get("layerwise_dynamic_base_threshold"))
        layerwise_dynamic_layer_alpha = (
            layerwise_dynamic_layer_alpha
            if layerwise_dynamic_layer_alpha is not None else
            item.get("layerwise_dynamic_layer_alpha"))
        layerwise_dynamic_num_layers = (
            layerwise_dynamic_num_layers
            if layerwise_dynamic_num_layers is not None else
            item.get("layerwise_dynamic_num_layers"))
        layerwise_dynamic_k_min = (
            layerwise_dynamic_k_min
            if layerwise_dynamic_k_min is not None else
            item.get("layerwise_dynamic_k_min"))
        budget_dynamic_easy_k = (
            budget_dynamic_easy_k
            if budget_dynamic_easy_k is not None else
            item.get("budget_dynamic_easy_k"))
        budget_dynamic_base_k = (
            budget_dynamic_base_k
            if budget_dynamic_base_k is not None else
            item.get("budget_dynamic_base_k"))
        budget_dynamic_hard_k = (
            budget_dynamic_hard_k
            if budget_dynamic_hard_k is not None else
            item.get("budget_dynamic_hard_k"))
        budget_dynamic_easy_threshold = (
            budget_dynamic_easy_threshold
            if budget_dynamic_easy_threshold is not None else
            item.get("budget_dynamic_easy_threshold"))
        budget_dynamic_hard_threshold = (
            budget_dynamic_hard_threshold
            if budget_dynamic_hard_threshold is not None else
            item.get("budget_dynamic_hard_threshold"))
        budget_dynamic_score_topn = (
            budget_dynamic_score_topn
            if budget_dynamic_score_topn is not None else
            item.get("budget_dynamic_score_topn"))
        layer_budget_dynamic_easy_layer_alpha = (
            layer_budget_dynamic_easy_layer_alpha
            if layer_budget_dynamic_easy_layer_alpha is not None else
            item.get("layer_budget_dynamic_easy_layer_alpha"))
        layer_budget_dynamic_hard_layer_alpha = (
            layer_budget_dynamic_hard_layer_alpha
            if layer_budget_dynamic_hard_layer_alpha is not None else
            item.get("layer_budget_dynamic_hard_layer_alpha"))
        layer_budget_dynamic_num_layers = (
            layer_budget_dynamic_num_layers
            if layer_budget_dynamic_num_layers is not None else
            item.get("layer_budget_dynamic_num_layers"))
        band_layer_budget_middle_start = (
            band_layer_budget_middle_start
            if band_layer_budget_middle_start is not None else
            item.get("band_layer_budget_middle_start"))
        band_layer_budget_middle_end = (
            band_layer_budget_middle_end
            if band_layer_budget_middle_end is not None else
            item.get("band_layer_budget_middle_end"))
        band_layer_budget_extra_middle_start = (
            band_layer_budget_extra_middle_start
            if band_layer_budget_extra_middle_start is not None else
            item.get("band_layer_budget_extra_middle_start"))
        band_layer_budget_extra_middle_end = (
            band_layer_budget_extra_middle_end
            if band_layer_budget_extra_middle_end is not None else
            item.get("band_layer_budget_extra_middle_end"))
        band_layer_budget_middle_phase = (
            band_layer_budget_middle_phase
            if band_layer_budget_middle_phase is not None else
            item.get("band_layer_budget_middle_phase"))
        band_layer_budget_easy_start = (
            band_layer_budget_easy_start
            if band_layer_budget_easy_start is not None else
            item.get("band_layer_budget_easy_start"))
        band_layer_budget_easy_end = (
            band_layer_budget_easy_end
            if band_layer_budget_easy_end is not None else
            item.get("band_layer_budget_easy_end"))
        band_layer_budget_easy_phase = (
            band_layer_budget_easy_phase
            if band_layer_budget_easy_phase is not None else
            item.get("band_layer_budget_easy_phase"))
        band_layer_budget_hard_phase = (
            band_layer_budget_hard_phase
            if band_layer_budget_hard_phase is not None else
            item.get("band_layer_budget_hard_phase"))
        band_layer_budget_easy_tail_threshold = (
            band_layer_budget_easy_tail_threshold
            if band_layer_budget_easy_tail_threshold is not None else
            item.get("band_layer_budget_easy_tail_threshold"))
        band_layer_budget_hard_min_rank_weight = (
            band_layer_budget_hard_min_rank_weight
            if band_layer_budget_hard_min_rank_weight is not None else
            item.get("band_layer_budget_hard_min_rank_weight"))
        band_layer_budget_hard_max_rank_weight = (
            band_layer_budget_hard_max_rank_weight
            if band_layer_budget_hard_max_rank_weight is not None else
            item.get("band_layer_budget_hard_max_rank_weight"))
        band_layer_budget_hard_decode_min_rank_weight = (
            band_layer_budget_hard_decode_min_rank_weight
            if band_layer_budget_hard_decode_min_rank_weight is not None else
            item.get("band_layer_budget_hard_decode_min_rank_weight"))
        band_layer_budget_mixed_rescue_min_rank_weight = (
            band_layer_budget_mixed_rescue_min_rank_weight
            if band_layer_budget_mixed_rescue_min_rank_weight is not None
            else item.get("band_layer_budget_mixed_rescue_min_rank_weight"))
        band_layer_budget_mixed_rescue_max_concentration = (
            band_layer_budget_mixed_rescue_max_concentration
            if band_layer_budget_mixed_rescue_max_concentration is not None
            else item.get("band_layer_budget_mixed_rescue_max_concentration"))
        band_layer_budget_hard_prefill_max_tokens = (
            band_layer_budget_hard_prefill_max_tokens
            if band_layer_budget_hard_prefill_max_tokens is not None else
            item.get("band_layer_budget_hard_prefill_max_tokens"))
        band_layer_budget_hard_prefill_max_segment_tokens = (
            band_layer_budget_hard_prefill_max_segment_tokens
            if band_layer_budget_hard_prefill_max_segment_tokens is not None
            else item.get("band_layer_budget_hard_prefill_max_segment_tokens"))
        band_layer_budget_hard_prefill_max_density = (
            band_layer_budget_hard_prefill_max_density
            if band_layer_budget_hard_prefill_max_density is not None else
            item.get("band_layer_budget_hard_prefill_max_density"))
        band_layer_budget_hard_prefill_min_density = (
            band_layer_budget_hard_prefill_min_density
            if band_layer_budget_hard_prefill_min_density is not None else
            item.get("band_layer_budget_hard_prefill_min_density"))
        band_layer_budget_hard_prefill_max_segment_hard_ratio = (
            band_layer_budget_hard_prefill_max_segment_hard_ratio
            if band_layer_budget_hard_prefill_max_segment_hard_ratio
            is not None else
            item.get(
                "band_layer_budget_hard_prefill_max_segment_hard_ratio"))
        band_layer_budget_hard_prefill_segment_cap_score = (
            band_layer_budget_hard_prefill_segment_cap_score
            if band_layer_budget_hard_prefill_segment_cap_score is not None
            else item.get(
                "band_layer_budget_hard_prefill_segment_cap_score"))
        band_layer_budget_hard_prefill_conditional_segment_hard_ratio = (
            band_layer_budget_hard_prefill_conditional_segment_hard_ratio
            if band_layer_budget_hard_prefill_conditional_segment_hard_ratio
            is not None else item.get(
                "band_layer_budget_hard_prefill_conditional_segment_hard_ratio"))
        band_layer_budget_hard_prefill_conditional_min_density = (
            band_layer_budget_hard_prefill_conditional_min_density
            if band_layer_budget_hard_prefill_conditional_min_density
            is not None else item.get(
                "band_layer_budget_hard_prefill_conditional_min_density"))
        band_layer_budget_hard_prefill_conditional_min_seq_len = (
            band_layer_budget_hard_prefill_conditional_min_seq_len
            if band_layer_budget_hard_prefill_conditional_min_seq_len
            is not None else item.get(
                "band_layer_budget_hard_prefill_conditional_min_seq_len"))
        band_layer_budget_hard_prefill_conditional_token_ids = (
            band_layer_budget_hard_prefill_conditional_token_ids
            if band_layer_budget_hard_prefill_conditional_token_ids
            is not None else item.get(
                "band_layer_budget_hard_prefill_conditional_token_ids"))
        band_layer_budget_hard_prefill_conditional_token_ngrams = (
            band_layer_budget_hard_prefill_conditional_token_ngrams
            if band_layer_budget_hard_prefill_conditional_token_ngrams
            is not None else item.get(
                "band_layer_budget_hard_prefill_conditional_token_ngrams"))
        band_layer_budget_hard_prefill_marker_segment_hard_ratio = (
            band_layer_budget_hard_prefill_marker_segment_hard_ratio
            if (band_layer_budget_hard_prefill_marker_segment_hard_ratio
                is not None)
            else item.get(
                "band_layer_budget_hard_prefill_marker_segment_hard_ratio"))
        band_layer_budget_hard_prefill_marker_token_ids = (
            band_layer_budget_hard_prefill_marker_token_ids
            if band_layer_budget_hard_prefill_marker_token_ids is not None
            else item.get("band_layer_budget_hard_prefill_marker_token_ids"))
        band_layer_budget_hard_prefill_marker_token_ngrams = (
            band_layer_budget_hard_prefill_marker_token_ngrams
            if band_layer_budget_hard_prefill_marker_token_ngrams is not None
            else item.get(
                "band_layer_budget_hard_prefill_marker_token_ngrams"))
        band_layer_budget_hard_prefill_exclude_token_ids = (
            band_layer_budget_hard_prefill_exclude_token_ids
            if band_layer_budget_hard_prefill_exclude_token_ids is not None
            else item.get("band_layer_budget_hard_prefill_exclude_token_ids"))
        band_layer_budget_hard_prefill_min_relative_pos = (
            band_layer_budget_hard_prefill_min_relative_pos
            if band_layer_budget_hard_prefill_min_relative_pos is not None
            else item.get(
                "band_layer_budget_hard_prefill_min_relative_pos"))
        band_layer_budget_hard_prefill_max_relative_pos = (
            band_layer_budget_hard_prefill_max_relative_pos
            if band_layer_budget_hard_prefill_max_relative_pos is not None
            else item.get(
                "band_layer_budget_hard_prefill_max_relative_pos"))
        band_layer_budget_hard_prefill_start = (
            band_layer_budget_hard_prefill_start
            if band_layer_budget_hard_prefill_start is not None else
            item.get("band_layer_budget_hard_prefill_start"))
        band_layer_budget_hard_prefill_end = (
            band_layer_budget_hard_prefill_end
            if band_layer_budget_hard_prefill_end is not None else
            item.get("band_layer_budget_hard_prefill_end"))
        band_layer_budget_hard_decode_min_seq_len = (
            band_layer_budget_hard_decode_min_seq_len
            if band_layer_budget_hard_decode_min_seq_len is not None else
            item.get("band_layer_budget_hard_decode_min_seq_len"))
        band_layer_budget_hard_decode_max_seq_len = (
            band_layer_budget_hard_decode_max_seq_len
            if band_layer_budget_hard_decode_max_seq_len is not None else
            item.get("band_layer_budget_hard_decode_max_seq_len"))
        band_layer_budget_hard_decode_seq_len_scope = (
            band_layer_budget_hard_decode_seq_len_scope
            if band_layer_budget_hard_decode_seq_len_scope is not None else
            item.get("band_layer_budget_hard_decode_seq_len_scope"))
        band_layer_budget_hard_decode_min_offset = (
            band_layer_budget_hard_decode_min_offset
            if band_layer_budget_hard_decode_min_offset is not None else
            item.get("band_layer_budget_hard_decode_min_offset"))
        band_layer_budget_hard_decode_max_offset = (
            band_layer_budget_hard_decode_max_offset
            if band_layer_budget_hard_decode_max_offset is not None else
            item.get("band_layer_budget_hard_decode_max_offset"))
        band_layer_budget_hard_decode_start = (
            band_layer_budget_hard_decode_start
            if band_layer_budget_hard_decode_start is not None else
            item.get("band_layer_budget_hard_decode_start"))
        band_layer_budget_hard_decode_end = (
            band_layer_budget_hard_decode_end
            if band_layer_budget_hard_decode_end is not None else
            item.get("band_layer_budget_hard_decode_end"))
        band_layer_budget_hard_decode_token_ids = (
            band_layer_budget_hard_decode_token_ids
            if band_layer_budget_hard_decode_token_ids is not None else
            item.get("band_layer_budget_hard_decode_token_ids"))
        band_layer_budget_hard_layer_prior_start = (
            band_layer_budget_hard_layer_prior_start
            if band_layer_budget_hard_layer_prior_start is not None else
            item.get("band_layer_budget_hard_layer_prior_start"))
        band_layer_budget_hard_layer_prior_end = (
            band_layer_budget_hard_layer_prior_end
            if band_layer_budget_hard_layer_prior_end is not None else
            item.get("band_layer_budget_hard_layer_prior_end"))
        band_layer_budget_hard_layer_prior_extra_start = (
            band_layer_budget_hard_layer_prior_extra_start
            if band_layer_budget_hard_layer_prior_extra_start is not None else
            item.get("band_layer_budget_hard_layer_prior_extra_start"))
        band_layer_budget_hard_layer_prior_extra_end = (
            band_layer_budget_hard_layer_prior_extra_end
            if band_layer_budget_hard_layer_prior_extra_end is not None else
            item.get("band_layer_budget_hard_layer_prior_extra_end"))
        band_layer_budget_hard_layer_prior_alpha = (
            band_layer_budget_hard_layer_prior_alpha
            if band_layer_budget_hard_layer_prior_alpha is not None else
            item.get("band_layer_budget_hard_layer_prior_alpha"))
        band_layer_budget_hard_layer_sensitivity_alpha = (
            band_layer_budget_hard_layer_sensitivity_alpha
            if band_layer_budget_hard_layer_sensitivity_alpha is not None else
            item.get("band_layer_budget_hard_layer_sensitivity_alpha"))
        band_layer_budget_late_concentration_start = (
            band_layer_budget_late_concentration_start
            if band_layer_budget_late_concentration_start is not None else
            item.get("band_layer_budget_late_concentration_start"))
        band_layer_budget_late_concentration_threshold = (
            band_layer_budget_late_concentration_threshold
            if band_layer_budget_late_concentration_threshold is not None else
            item.get("band_layer_budget_late_concentration_threshold"))
        band_layer_budget_late_rescue_start = (
            band_layer_budget_late_rescue_start
            if band_layer_budget_late_rescue_start is not None else
            item.get("band_layer_budget_late_rescue_start"))
        band_layer_budget_late_rescue_concentration_threshold = (
            band_layer_budget_late_rescue_concentration_threshold
            if band_layer_budget_late_rescue_concentration_threshold
            is not None else
            item.get("band_layer_budget_late_rescue_concentration_threshold"))
        band_layer_budget_late_rescue_mixed_max = (
            band_layer_budget_late_rescue_mixed_max
            if band_layer_budget_late_rescue_mixed_max is not None else
            item.get("band_layer_budget_late_rescue_mixed_max"))
        band_layer_budget_prompt_profile1_token_ngrams = (
            band_layer_budget_prompt_profile1_token_ngrams
            if band_layer_budget_prompt_profile1_token_ngrams is not None
            else item.get(
                "band_layer_budget_prompt_profile1_token_ngrams"))
        band_layer_budget_prompt_profile1_hard_threshold = (
            band_layer_budget_prompt_profile1_hard_threshold
            if band_layer_budget_prompt_profile1_hard_threshold is not None
            else item.get(
                "band_layer_budget_prompt_profile1_hard_threshold"))
        band_layer_budget_prompt_profile1_hard_min_rank_weight = (
            band_layer_budget_prompt_profile1_hard_min_rank_weight
            if (band_layer_budget_prompt_profile1_hard_min_rank_weight
                is not None)
            else item.get(
                "band_layer_budget_prompt_profile1_hard_min_rank_weight"))
        band_layer_budget_prompt_profile1_middle_start = (
            band_layer_budget_prompt_profile1_middle_start
            if band_layer_budget_prompt_profile1_middle_start is not None
            else item.get(
                "band_layer_budget_prompt_profile1_middle_start"))
        band_layer_budget_prompt_profile1_middle_end = (
            band_layer_budget_prompt_profile1_middle_end
            if band_layer_budget_prompt_profile1_middle_end is not None
            else item.get(
                "band_layer_budget_prompt_profile1_middle_end"))
        band_layer_budget_prompt_profile1_prefill_segment_hard_ratio = (
            band_layer_budget_prompt_profile1_prefill_segment_hard_ratio
            if (band_layer_budget_prompt_profile1_prefill_segment_hard_ratio
                is not None)
            else item.get(
                "band_layer_budget_prompt_profile1_prefill_segment_hard_ratio"))
        band_layer_budget_prompt_profile2_token_ngrams = (
            band_layer_budget_prompt_profile2_token_ngrams
            if band_layer_budget_prompt_profile2_token_ngrams is not None
            else item.get(
                "band_layer_budget_prompt_profile2_token_ngrams"))
        band_layer_budget_prompt_profile2_hard_threshold = (
            band_layer_budget_prompt_profile2_hard_threshold
            if band_layer_budget_prompt_profile2_hard_threshold is not None
            else item.get(
                "band_layer_budget_prompt_profile2_hard_threshold"))
        band_layer_budget_prompt_profile2_hard_min_rank_weight = (
            band_layer_budget_prompt_profile2_hard_min_rank_weight
            if (band_layer_budget_prompt_profile2_hard_min_rank_weight
                is not None)
            else item.get(
                "band_layer_budget_prompt_profile2_hard_min_rank_weight"))
        band_layer_budget_prompt_profile2_middle_start = (
            band_layer_budget_prompt_profile2_middle_start
            if band_layer_budget_prompt_profile2_middle_start is not None
            else item.get(
                "band_layer_budget_prompt_profile2_middle_start"))
        band_layer_budget_prompt_profile2_middle_end = (
            band_layer_budget_prompt_profile2_middle_end
            if band_layer_budget_prompt_profile2_middle_end is not None
            else item.get(
                "band_layer_budget_prompt_profile2_middle_end"))
        band_layer_budget_prompt_profile2_prefill_segment_hard_ratio = (
            band_layer_budget_prompt_profile2_prefill_segment_hard_ratio
            if (band_layer_budget_prompt_profile2_prefill_segment_hard_ratio
                is not None)
            else item.get(
                "band_layer_budget_prompt_profile2_prefill_segment_hard_ratio"))
        diep_artifact_path = (
            diep_artifact_path if diep_artifact_path is not None else
            item.get("diep_artifact_path"))
        diep_pruning_mode = (
            diep_pruning_mode if diep_pruning_mode is not None else
            item.get("diep_pruning_mode"))
        eac_alpha = eac_alpha if eac_alpha is not None else item.get(
            "eac_alpha")
        mc_moe_protection_ratio = (
            mc_moe_protection_ratio
            if mc_moe_protection_ratio is not None else
            item.get("mc_moe_protection_ratio"))
        biased_renorm_keep_topn = (
            biased_renorm_keep_topn
            if biased_renorm_keep_topn is not None else
            item.get("biased_renorm_keep_topn"))
        ban_artifact_path = (
            ban_artifact_path if ban_artifact_path is not None else
            item.get("ban_artifact_path"))
        ban_lambda = ban_lambda if ban_lambda is not None else item.get(
            "ban_lambda")
        ban_k_min = ban_k_min if ban_k_min is not None else item.get(
            "ban_k_min")
        total_protected_tokens += int(item.get("total_protected_tokens", 0))
        eac_prefill_tokens += int(item.get("eac_prefill_tokens", 0))
        eac_segments += int(item.get("eac_segments", 0))
        eac_calls_with_segments += int(
            item.get("eac_calls_with_segments", 0))
        eac_pruned_selections += int(
            item.get("eac_pruned_selections", 0))
        eac_disabled_experts += int(item.get("eac_disabled_experts", 0))
        ban_total_tokens += int(item.get("ban_total_tokens", 0))
        ban_total_k += float(item.get("ban_average_dynamic_k", 0.0)) * int(
            item.get("ban_total_tokens", 0))
        budget_easy_tokens += int(item.get("budget_easy_tokens", 0))
        budget_base_tokens += int(item.get("budget_base_tokens", 0))
        budget_hard_tokens += int(item.get("budget_hard_tokens", 0))
        for layer, counts in item.get("budget_layer_counts", {}).items():
            aggregate = budget_layer_counts.setdefault(int(layer), [0, 0, 0])
            for i in range(min(3, len(counts))):
                aggregate[i] += int(counts[i])
        for profile, counts in item.get(
                "budget_prompt_profile_counts", {}).items():
            aggregate = budget_prompt_profile_counts.setdefault(
                int(profile),
                [0, 0, 0],
            )
            for i in range(min(3, len(counts))):
                aggregate[i] += int(counts[i])
        for stage, counts in item.get("budget_stage_counts", {}).items():
            aggregate = budget_stage_counts.setdefault(stage, [0, 0, 0])
            for i in range(min(3, len(counts))):
                aggregate[i] += int(counts[i])
        rank_weight = item.get("budget_hard_rank_weight")
        if isinstance(rank_weight, dict):
            if budget_hard_rank_weight_bounds is None:
                budget_hard_rank_weight_bounds = rank_weight.get("bounds")
            candidate = rank_weight.get("candidate")
            selected = rank_weight.get("selected")
            if isinstance(candidate, dict):
                _merge_rank_weight_aggregate(
                    budget_hard_rank_weight_candidate, candidate)
            if isinstance(selected, dict):
                _merge_rank_weight_aggregate(
                    budget_hard_rank_weight_selected, selected)
        seq_len_stats = item.get("budget_hard_seq_len")
        if isinstance(seq_len_stats, dict):
            if budget_hard_seq_len_bounds is None:
                budget_hard_seq_len_bounds = seq_len_stats.get("bounds")
            candidate = seq_len_stats.get("candidate")
            selected = seq_len_stats.get("selected")
            if isinstance(candidate, dict):
                _merge_rank_weight_aggregate(
                    budget_hard_seq_len_candidate, candidate)
            if isinstance(selected, dict):
                _merge_rank_weight_aggregate(
                    budget_hard_seq_len_selected, selected)
        decode_offset_stats = item.get("budget_hard_decode_offset")
        if isinstance(decode_offset_stats, dict):
            if budget_hard_decode_offset_bounds is None:
                budget_hard_decode_offset_bounds = (
                    decode_offset_stats.get("bounds"))
            candidate = decode_offset_stats.get("candidate")
            selected = decode_offset_stats.get("selected")
            if isinstance(candidate, dict):
                _merge_rank_weight_aggregate(
                    budget_hard_decode_offset_candidate, candidate)
            if isinstance(selected, dict):
                _merge_rank_weight_aggregate(
                    budget_hard_decode_offset_selected, selected)
        segment_density = item.get("budget_prefill_segment_hard_density")
        if isinstance(segment_density, dict):
            if budget_prefill_segment_density_bounds is None:
                budget_prefill_segment_density_bounds = (
                    segment_density.get("bounds"))
            candidate = segment_density.get("candidate")
            selected = segment_density.get("selected")
            if isinstance(candidate, dict):
                _merge_segment_density_aggregate(
                    budget_prefill_segment_density_candidate, candidate)
            if isinstance(selected, dict):
                _merge_segment_density_aggregate(
                    budget_prefill_segment_density_selected, selected)
            for layer, layer_item in segment_density.get(
                    "by_layer", {}).items():
                layer_aggregate = (
                    budget_prefill_segment_density_by_layer.setdefault(
                        int(layer),
                        {
                            "candidate": _new_segment_density_aggregate(),
                            "selected": _new_segment_density_aggregate(),
                        },
                    ))
                if not isinstance(layer_item, dict):
                    continue
                candidate = layer_item.get("candidate")
                selected = layer_item.get("selected")
                if isinstance(candidate, dict):
                    _merge_segment_density_aggregate(
                        layer_aggregate["candidate"], candidate)
                if isinstance(selected, dict):
                    _merge_segment_density_aggregate(
                        layer_aggregate["selected"], selected)
        worker_stats.append(item)

    if total_tokens == 0:
        return None
    average_selected_experts = total_selected / total_tokens
    if method == "ban" and ban_total_tokens == 0:
        ban_total_tokens = total_tokens
        ban_total_k = total_selected

    return {
        "method": method,
        "topk": topk,
        "naee_beta": naee_beta,
        "naee_k_min": naee_k_min,
        "dynamic_routing_threshold": dynamic_routing_threshold,
        "dynamic_routing_score_source": dynamic_routing_score_source,
        "dynamic_routing_next_rank_penalty": (
            dynamic_routing_next_rank_penalty),
        "layerwise_dynamic_base_threshold": (
            layerwise_dynamic_base_threshold),
        "layerwise_dynamic_layer_alpha": layerwise_dynamic_layer_alpha,
        "layerwise_dynamic_num_layers": layerwise_dynamic_num_layers,
        "layerwise_dynamic_k_min": layerwise_dynamic_k_min,
        "budget_dynamic_easy_k": budget_dynamic_easy_k,
        "budget_dynamic_base_k": budget_dynamic_base_k,
        "budget_dynamic_hard_k": budget_dynamic_hard_k,
        "budget_dynamic_easy_threshold": budget_dynamic_easy_threshold,
        "budget_dynamic_hard_threshold": budget_dynamic_hard_threshold,
        "budget_dynamic_score_topn": budget_dynamic_score_topn,
        "layer_budget_dynamic_easy_layer_alpha": (
            layer_budget_dynamic_easy_layer_alpha),
        "layer_budget_dynamic_hard_layer_alpha": (
            layer_budget_dynamic_hard_layer_alpha),
        "layer_budget_dynamic_num_layers": layer_budget_dynamic_num_layers,
        "band_layer_budget_middle_start": band_layer_budget_middle_start,
        "band_layer_budget_middle_end": band_layer_budget_middle_end,
        "band_layer_budget_extra_middle_start": (
            band_layer_budget_extra_middle_start),
        "band_layer_budget_extra_middle_end": (
            band_layer_budget_extra_middle_end),
        "band_layer_budget_middle_phase": band_layer_budget_middle_phase,
        "band_layer_budget_easy_start": band_layer_budget_easy_start,
        "band_layer_budget_easy_end": band_layer_budget_easy_end,
        "band_layer_budget_easy_phase": band_layer_budget_easy_phase,
        "band_layer_budget_hard_phase": band_layer_budget_hard_phase,
        "band_layer_budget_easy_tail_threshold": (
            band_layer_budget_easy_tail_threshold),
        "band_layer_budget_hard_min_rank_weight": (
            band_layer_budget_hard_min_rank_weight),
        "band_layer_budget_hard_max_rank_weight": (
            band_layer_budget_hard_max_rank_weight),
        "band_layer_budget_hard_decode_min_rank_weight": (
            band_layer_budget_hard_decode_min_rank_weight),
        "band_layer_budget_mixed_rescue_min_rank_weight": (
            band_layer_budget_mixed_rescue_min_rank_weight),
        "band_layer_budget_mixed_rescue_max_concentration": (
            band_layer_budget_mixed_rescue_max_concentration),
        "band_layer_budget_hard_prefill_max_tokens": (
            band_layer_budget_hard_prefill_max_tokens),
        "band_layer_budget_hard_prefill_max_segment_tokens": (
            band_layer_budget_hard_prefill_max_segment_tokens),
        "band_layer_budget_hard_prefill_max_density": (
            band_layer_budget_hard_prefill_max_density),
        "band_layer_budget_hard_prefill_min_density": (
            band_layer_budget_hard_prefill_min_density),
        "band_layer_budget_hard_prefill_max_segment_hard_ratio": (
            band_layer_budget_hard_prefill_max_segment_hard_ratio),
        "band_layer_budget_hard_prefill_segment_cap_score": (
            band_layer_budget_hard_prefill_segment_cap_score),
        "band_layer_budget_hard_prefill_conditional_segment_hard_ratio": (
            band_layer_budget_hard_prefill_conditional_segment_hard_ratio),
        "band_layer_budget_hard_prefill_conditional_min_density": (
            band_layer_budget_hard_prefill_conditional_min_density),
        "band_layer_budget_hard_prefill_conditional_min_seq_len": (
            band_layer_budget_hard_prefill_conditional_min_seq_len),
        "band_layer_budget_hard_prefill_conditional_token_ids": (
            band_layer_budget_hard_prefill_conditional_token_ids),
        "band_layer_budget_hard_prefill_conditional_token_ngrams": (
            band_layer_budget_hard_prefill_conditional_token_ngrams),
        "band_layer_budget_hard_prefill_marker_segment_hard_ratio": (
            band_layer_budget_hard_prefill_marker_segment_hard_ratio),
        "band_layer_budget_hard_prefill_marker_token_ids": (
            band_layer_budget_hard_prefill_marker_token_ids),
        "band_layer_budget_hard_prefill_marker_token_ngrams": (
            band_layer_budget_hard_prefill_marker_token_ngrams),
        "band_layer_budget_hard_prefill_exclude_token_ids": (
            band_layer_budget_hard_prefill_exclude_token_ids),
        "band_layer_budget_hard_prefill_min_relative_pos": (
            band_layer_budget_hard_prefill_min_relative_pos),
        "band_layer_budget_hard_prefill_max_relative_pos": (
            band_layer_budget_hard_prefill_max_relative_pos),
        "band_layer_budget_hard_prefill_start": (
            band_layer_budget_hard_prefill_start),
        "band_layer_budget_hard_prefill_end": (
            band_layer_budget_hard_prefill_end),
        "band_layer_budget_hard_decode_min_seq_len": (
            band_layer_budget_hard_decode_min_seq_len),
        "band_layer_budget_hard_decode_max_seq_len": (
            band_layer_budget_hard_decode_max_seq_len),
        "band_layer_budget_hard_decode_seq_len_scope": (
            band_layer_budget_hard_decode_seq_len_scope),
        "band_layer_budget_hard_decode_min_offset": (
            band_layer_budget_hard_decode_min_offset),
        "band_layer_budget_hard_decode_max_offset": (
            band_layer_budget_hard_decode_max_offset),
        "band_layer_budget_hard_decode_start": (
            band_layer_budget_hard_decode_start),
        "band_layer_budget_hard_decode_end": (
            band_layer_budget_hard_decode_end),
        "band_layer_budget_hard_decode_token_ids": (
            band_layer_budget_hard_decode_token_ids),
        "band_layer_budget_hard_layer_prior_start": (
            band_layer_budget_hard_layer_prior_start),
        "band_layer_budget_hard_layer_prior_end": (
            band_layer_budget_hard_layer_prior_end),
        "band_layer_budget_hard_layer_prior_extra_start": (
            band_layer_budget_hard_layer_prior_extra_start),
        "band_layer_budget_hard_layer_prior_extra_end": (
            band_layer_budget_hard_layer_prior_extra_end),
        "band_layer_budget_hard_layer_prior_alpha": (
            band_layer_budget_hard_layer_prior_alpha),
        "band_layer_budget_hard_layer_sensitivity_alpha": (
            band_layer_budget_hard_layer_sensitivity_alpha),
        "band_layer_budget_late_concentration_start": (
            band_layer_budget_late_concentration_start),
        "band_layer_budget_late_concentration_threshold": (
            band_layer_budget_late_concentration_threshold),
        "band_layer_budget_late_rescue_start": (
            band_layer_budget_late_rescue_start),
        "band_layer_budget_late_rescue_concentration_threshold": (
            band_layer_budget_late_rescue_concentration_threshold),
        "band_layer_budget_late_rescue_mixed_max": (
            band_layer_budget_late_rescue_mixed_max),
        "band_layer_budget_prompt_profile1_token_ngrams": (
            band_layer_budget_prompt_profile1_token_ngrams),
        "band_layer_budget_prompt_profile1_hard_threshold": (
            band_layer_budget_prompt_profile1_hard_threshold),
        "band_layer_budget_prompt_profile1_hard_min_rank_weight": (
            band_layer_budget_prompt_profile1_hard_min_rank_weight),
        "band_layer_budget_prompt_profile1_middle_start": (
            band_layer_budget_prompt_profile1_middle_start),
        "band_layer_budget_prompt_profile1_middle_end": (
            band_layer_budget_prompt_profile1_middle_end),
        "band_layer_budget_prompt_profile1_prefill_segment_hard_ratio": (
            band_layer_budget_prompt_profile1_prefill_segment_hard_ratio),
        "band_layer_budget_prompt_profile2_token_ngrams": (
            band_layer_budget_prompt_profile2_token_ngrams),
        "band_layer_budget_prompt_profile2_hard_threshold": (
            band_layer_budget_prompt_profile2_hard_threshold),
        "band_layer_budget_prompt_profile2_hard_min_rank_weight": (
            band_layer_budget_prompt_profile2_hard_min_rank_weight),
        "band_layer_budget_prompt_profile2_middle_start": (
            band_layer_budget_prompt_profile2_middle_start),
        "band_layer_budget_prompt_profile2_middle_end": (
            band_layer_budget_prompt_profile2_middle_end),
        "band_layer_budget_prompt_profile2_prefill_segment_hard_ratio": (
            band_layer_budget_prompt_profile2_prefill_segment_hard_ratio),
        "diep_artifact_path": diep_artifact_path,
        "diep_pruning_mode": diep_pruning_mode,
        "eac_alpha": eac_alpha,
        "mc_moe_protection_ratio": mc_moe_protection_ratio,
        "biased_renorm_keep_topn": biased_renorm_keep_topn,
        "ban_artifact_path": ban_artifact_path,
        "ban_lambda": ban_lambda,
        "ban_k_min": ban_k_min,
        "total_protected_tokens": total_protected_tokens,
        "protected_token_ratio": (
            total_protected_tokens / total_tokens if total_tokens else 0.0),
        "eac_prefill_tokens": eac_prefill_tokens,
        "eac_segments": eac_segments,
        "eac_calls_with_segments": eac_calls_with_segments,
        "eac_pruned_selections": eac_pruned_selections,
        "eac_disabled_experts": eac_disabled_experts,
        "eac_pruned_selection_ratio": (
            eac_pruned_selections / (eac_prefill_tokens * topk)
            if eac_prefill_tokens and topk else 0.0),
        "ban_average_dynamic_k": (
            ban_total_k / ban_total_tokens if ban_total_tokens else 0.0),
        "ban_total_tokens": ban_total_tokens,
        "budget_easy_tokens": budget_easy_tokens,
        "budget_base_tokens": budget_base_tokens,
        "budget_hard_tokens": budget_hard_tokens,
        "budget_layer_counts": {
            str(layer): counts
            for layer, counts in sorted(budget_layer_counts.items())
        },
        "budget_layer_ratios": {
            str(layer): [
                count / sum(counts) if sum(counts) else 0.0
                for count in counts
            ]
            for layer, counts in sorted(budget_layer_counts.items())
        },
        "budget_prompt_profile_counts": {
            str(profile): counts
            for profile, counts in sorted(
                budget_prompt_profile_counts.items())
        },
        "budget_prompt_profile_ratios": {
            str(profile): [
                count / sum(counts) if sum(counts) else 0.0
                for count in counts
            ]
            for profile, counts in sorted(
                budget_prompt_profile_counts.items())
        },
        "budget_stage_counts": {
            stage: counts
            for stage, counts in sorted(budget_stage_counts.items())
        },
        "budget_stage_ratios": {
            stage: [
                count / sum(counts) if sum(counts) else 0.0
                for count in counts
            ]
            for stage, counts in sorted(budget_stage_counts.items())
        },
        "budget_hard_rank_weight": {
            "bounds": budget_hard_rank_weight_bounds or [],
            "candidate": _finalize_rank_weight_aggregate(
                budget_hard_rank_weight_candidate),
            "selected": _finalize_rank_weight_aggregate(
                budget_hard_rank_weight_selected),
        },
        "budget_hard_seq_len": {
            "bounds": budget_hard_seq_len_bounds or [],
            "candidate": _finalize_rank_weight_aggregate(
                budget_hard_seq_len_candidate),
            "selected": _finalize_rank_weight_aggregate(
                budget_hard_seq_len_selected),
        },
        "budget_hard_decode_offset": {
            "bounds": budget_hard_decode_offset_bounds or [],
            "candidate": _finalize_rank_weight_aggregate(
                budget_hard_decode_offset_candidate),
            "selected": _finalize_rank_weight_aggregate(
                budget_hard_decode_offset_selected),
        },
        "budget_prefill_segment_hard_density": {
            "bounds": budget_prefill_segment_density_bounds or [],
            "candidate": _finalize_segment_density_aggregate(
                budget_prefill_segment_density_candidate),
            "selected": _finalize_segment_density_aggregate(
                budget_prefill_segment_density_selected),
            "by_layer": {
                str(layer): {
                    "candidate": _finalize_segment_density_aggregate(
                        stats["candidate"]),
                    "selected": _finalize_segment_density_aggregate(
                        stats["selected"]),
                }
                for layer, stats in sorted(
                    budget_prefill_segment_density_by_layer.items())
            },
        },
        "budget_easy_ratio": (
            budget_easy_tokens / total_tokens if total_tokens else 0.0),
        "budget_base_ratio": (
            budget_base_tokens / total_tokens if total_tokens else 0.0),
        "budget_hard_ratio": (
            budget_hard_tokens / total_tokens if total_tokens else 0.0),
        "average_selected_experts": average_selected_experts,
        "total_selected_experts": total_selected,
        "total_routed_tokens": total_tokens,
        "total_routing_calls": total_calls,
        "num_worker_stat_files": len(worker_stats),
        "worker_stats": worker_stats,
    }


class CustomEvaluationTracker:
    """自定义 Tracker：结果与 details 保存在 ``output_dir/<short_model_name>/{results,details}`` 下，且 details 存为 JSON。"""

    def __init__(self, output_dir: str, short_model_name: str, save_details: bool = True, **kwargs):
        from dataclasses import asdict
        from datasets import Dataset
        from lighteval.logging.evaluation_tracker import EvaluationTracker, EnhancedJSONEncoder

        self._asdict = asdict
        self._Dataset = Dataset
        self._EnhancedJSONEncoder = EnhancedJSONEncoder
        self._short_model_name = short_model_name
        self._tracker = EvaluationTracker(output_dir=output_dir, save_details=save_details, **kwargs)
        self._expert_pruning_stats = None
        self._sample_shard_provenance = None

    def __getattr__(self, name):
        return getattr(self._tracker, name)

    def set_expert_pruning_stats(self, stats: dict | None) -> None:
        self._expert_pruning_stats = stats

    def set_sample_shard_provenance(self, provenance: dict | None) -> None:
        self._sample_shard_provenance = provenance

    def save(self):
        """覆盖 save：使用 ``output_dir/<short_model_name>/{results,details}``，details 写 JSON。"""
        from datetime import datetime

        date_id = datetime.now().isoformat().replace(":", "-")
        results_dict = self._tracker.results
        if self._expert_pruning_stats is not None and isinstance(results_dict, dict):
            results_dict["expert_pruning"] = self._expert_pruning_stats
        if self._sample_shard_provenance is not None and isinstance(
            results_dict, dict
        ):
            results_dict["_sample_shard"] = self._sample_shard_provenance
        details_datasets = {}
        for task_name, task_details in self._tracker.details_logger.details.items():
            dataset = self._Dataset.from_list([self._asdict(d) for d in task_details])
            col = [c for c in dataset.column_names if c != "id"] or dataset.column_names
            dataset = dataset.select_columns(sorted(col))
            details_datasets[task_name] = dataset
        self.save_results(date_id, results_dict)
        if self._tracker.should_save_details:
            self.save_details(date_id, details_datasets)
        if self._tracker.should_push_to_hub:
            self._tracker.push_to_hub(date_id=date_id, details=details_datasets, results_dict=results_dict)
        if getattr(self._tracker, "use_wandb", False):
            self._tracker.push_to_wandb(results_dict=results_dict, details_datasets=details_datasets)
        if getattr(self._tracker, "should_push_results_to_tensorboard", False):
            self._tracker.push_to_tensorboard(
                results=self._tracker.metrics_logger.metric_aggregated,
                details=self._tracker.details_logger.compiled_details,
            )

    def save_results(self, date_id: str, results_dict: dict):
        output_dir_results = Path(self._tracker.output_dir) / self._short_model_name / "results"
        self._tracker.fs.mkdirs(str(output_dir_results), exist_ok=True)
        output_results_file = output_dir_results / f"results_{date_id}.json"
        with self._tracker.fs.open(str(output_results_file), "w") as f:
            f.write(json.dumps(results_dict, cls=self._EnhancedJSONEncoder, indent=2, ensure_ascii=False))

    def _get_gold_from_doc(self, doc: dict):
        """从 doc 的 ``choices + gold_index`` 解析出参考答案，与 lighteval ``Doc.get_golds()`` 语义一致。"""
        choices = doc.get("choices")
        gold_index = doc.get("gold_index")
        if choices is None or gold_index is None:
            return None
        gold_indices = [gold_index] if isinstance(gold_index, int) else gold_index
        golds = []
        for ix in gold_indices:
            if ix < 0 or ix >= len(choices):
                continue
            c = choices[ix]
            if c is None:
                continue
            if isinstance(c, list):
                golds.extend(c)
            else:
                golds.append(c)
        return golds if golds else None

    def _filter_detail_record(self, record: dict) -> dict:
        """每条 detail 保留：``doc.id``、``doc.specific``、``doc.gold``、``metric``、``model_response``，保证各类任务都能看到 gold。"""
        doc = record.get("doc") or {}
        model_response = record.get("model_response") or {}
        gold = self._get_gold_from_doc(doc)
        choices = doc.get("choices") or []
        logprobs = model_response.get("logprobs") or []
        predicted_choice = None
        if choices and logprobs:
            n_choices = min(len(choices), len(logprobs))
            if n_choices:
                best_idx = max(range(n_choices), key=lambda i: logprobs[i])
                predicted_choice = choices[best_idx]
        out = {
            "doc": {"id": doc.get("id"), "specific": doc.get("specific")},
            "metric": record.get("metric"),
            "model_response": {
                "input": model_response.get("input"),
                "text": model_response.get("text"),
                "text_post_processed": model_response.get("text_post_processed"),
                "logprobs": logprobs,
                "output_tokens": model_response.get("output_tokens"),
                "predicted_choice": predicted_choice,
            },
        }
        if gold is not None:
            out["gold"] = gold
        return out

    def save_details(self, date_id: str, details_datasets: dict):
        output_dir_details_sub_folder = Path(self._tracker.output_dir) / self._short_model_name / "details" / date_id
        self._tracker.fs.mkdirs(str(output_dir_details_sub_folder), exist_ok=True)
        for task_name, dataset in details_datasets.items():
            output_file = output_dir_details_sub_folder / f"details_{task_name}_{date_id}.json"
            records = dataset.to_list() if hasattr(dataset, "to_list") else [dataset[i] for i in range(len(dataset))]
            filtered = [self._filter_detail_record(r) for r in records]
            with self._tracker.fs.open(str(output_file), "w") as f:
                f.write(json.dumps(filtered, indent=2, ensure_ascii=False, default=str))


def parse_args():
    # The task list lives with the runner that documents why each variant was chosen; the import is
    # local so that building the parser stays independent of whether lmms-eval is installed.
    from expert_pruning.lmms_eval_runner import DEFAULT_TASKS as LMMS_EVAL_DEFAULT_TASKS

    parser = argparse.ArgumentParser(description="vLLM + lighteval 评估脚本")
    parser.add_argument(
        "--datasets",
        type=str,
        default="gsm8k",
        help="评估数据集，逗号分隔，如 a,b,c。默认: gsm8k",
    )
    parser.add_argument(
        "--harness",
        choices=["lighteval", "lm_eval", "lmms_eval"],
        default="lighteval",
        help="用哪个评测框架。lighteval 是本仓库的生成式主线（--datasets 生效）；lm_eval 跑 "
             "log-likelihood 排序的选择题（--lm_eval_tasks 生效），不生成任何 token，因此剪枝的"
             "影响不掺入生成长度这个混杂因素；lmms_eval 跑图文多模态集（--lmms_eval_tasks 生效），"
             "用于 VL 模型。剪枝路径三者完全共用。默认: lighteval",
    )
    parser.add_argument(
        "--lm_eval_tasks",
        type=str,
        default="arc_challenge,arc_easy,winogrande,openbookqa",
        help="--harness lm_eval 时的任务名，逗号分隔。默认为四个 0-shot QA 集: "
             "arc_challenge,arc_easy,winogrande,openbookqa",
    )
    parser.add_argument(
        "--lm_eval_num_fewshot",
        type=int,
        default=0,
        help="--harness lm_eval 时的 few-shot 条数。默认: 0",
    )
    parser.add_argument(
        "--lm_eval_apply_chat_template",
        action="store_true",
        help="--harness lm_eval 时把题目套进 instruct 对话模板。这四个 QA 集公开的分数都是按裸"
             "续写打的，套模板后不可与之比较，故默认关闭。",
    )
    parser.add_argument(
        "--lmms_eval_tasks",
        type=str,
        default=",".join(LMMS_EVAL_DEFAULT_TASKS),
        help="--harness lmms_eval 时的任务名，逗号分隔。默认是九个候选集各自最小的已发布版本: "
             + ",".join(LMMS_EVAL_DEFAULT_TASKS),
    )
    parser.add_argument(
        "--lmms_eval_num_fewshot",
        type=int,
        default=0,
        help="--harness lmms_eval 时的 few-shot 条数。默认: 0",
    )
    parser.add_argument(
        "--lmms_eval_max_images",
        type=int,
        default=8,
        help="--harness lmms_eval 时单条 prompt 允许的图片数上限（vLLM 的 limit_mm_per_prompt）。"
             "MMMU 单题最多问 7 张图，超限会被直接拒绝，故默认 8。",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="vLLM 张量并行 (TP)：单层内参数切分到的 GPU 数。默认: 当前可见 GPU 数",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=None,
        help="vLLM 流水线并行 (PP)：按层切分到多卡，降低单卡显存；通常与 --tensor_parallel_size 联用，使 TP×PP 等于所用 GPU 总数。不设则为 1",
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=None,
        help="vLLM 数据并行副本数（多份完整模型，lighteval 会启用 Ray）。主要用于吞吐，一般不减单卡显存。不设则为 1",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="生成最大 token 数。默认: 32768",
    )
    parser.add_argument(
        "--max_model_length",
        "--max_model_len",
        type=int,
        default=32768,
        dest="max_model_length",
        help="模型最大序列长度（与 vLLM max_model_len 一致）。--max_model_len 为同义简写。默认: 32768",
    )
    parser.add_argument("--kv_cache_dtype", default="auto")
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--enable_expert_parallel", action="store_true")
    parser.add_argument(
        "--allow_compiled_router",
        action="store_true",
        help="量化 MoE 模型保留 CUDA graph 运行，不再自动切 eager。剪枝照常生效，但路由"
             "统计无法计数，平均专家数须由单独的 --enforce_eager 探针 run 测得。实测在 "
             "gpt-oss 上同样的 100 条 mmlu_pro，eager 用 182 分钟，graph 用 5 分钟。"
             "默认: 关闭",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个任务最多评估样本数，不设则全量。默认: 不设置",
    )
    parser.add_argument(
        "--sample_shard_id",
        type=int,
        default=None,
        help="按原始 Doc.id 对样本取模分片；必须与 --sample_num_shards 同时设置。",
    )
    parser.add_argument(
        "--sample_num_shards",
        type=int,
        default=None,
        help="确定性样本分片总数；不设置时保持原有全量行为。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度。默认: 0.7",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="top_p 采样。默认: 0.8",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="top_k 采样。默认: 20",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="结果与 details 保存目录。默认: ./results",
    )
    parser.add_argument(
        "--lighteval_cache_dir",
        type=str,
        default=None,
        help="lighteval 生成缓存根目录。不设时普通评测使用默认缓存；启用本地 expert router 时自动使用 output_dir/.lighteval_cache。router 方法、超参、artifact 内容和本地路由源码的指纹会自动加入子目录，避免跨配置复用旧生成结果。",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="模型路径或 HuggingFace 模型 id。默认: Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM GPU 显存利用率。默认: 0.95",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="传给 vLLM 的权重/计算 dtype。默认 auto；可显式固定 bfloat16。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="vLLM max_num_seqs：单轮迭代中最大并行序列数（与 EngineArgs.max_num_seqs 一致）。不设则使用 lighteval/vLLM 默认值",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Generation seed forwarded to vLLM and LightEval harness RNGs. Default: 1234.",
    )
    parser.add_argument(
        "--custom_tasks",
        type=str,
        default=None,
        help="自定义任务 Python 文件路径。默认: 仓库内 tasks/custom_tasks.py（聚合 aime/triviaqa/simpleqa/hellaswag/if_pass_at_n 全部任务）",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=None,
        help="MoE 模型每 token 激活的专家数，覆盖 config.json。仅支持本地模型目录。用于 pruning 实验（如 k,k-1,...,1）",
    )
    parser.add_argument(
        "--use_local_expert_router",
        action="store_true",
        help="启用本仓库 expert_pruning.routing.torch_fused_topk 替换 vLLM 默认专家选择；不加则走 vLLM 原路径。",
    )
    parser.add_argument(
        "--expert_pruning_method",
        type=str,
        choices=[
            "none", "NAEE", "Dynamic_Routing", "DiEP", "MC_MoE", "EAC_MoE",
            "Ban", "TopK_Biased_Renorm", "LayerWise_Dynamic_Routing",
            "Budget_Dynamic_Routing", "LayerBudget_Dynamic_Routing",
            "BandedLayerBudget_Dynamic_Routing",
            "BanSensitiveLayerBudget_Dynamic_Routing",
            "RouterValue_Dynamic_Routing",
        ],
        default="none",
        help="动态专家剪枝方法。none 表示不剪枝；NAEE 表示按 top-k softmax 权重阈值剪枝；Dynamic_Routing 表示保留累计概率达到阈值的最短专家前缀；DiEP 表示使用校准得到的专家相似度修正 NAEE 阈值；MC_MoE 表示用 attention sink 保护重要 token 后按 NAEE 剪枝；EAC_MoE 表示仅 prefill 阶段按专家分配量裁剪低活跃专家；Ban 表示结合离线层敏感度和在线 token 敏感度动态决定保留专家数；TopK_Biased_Renorm 表示保留固定 top-k 但只放大前若干专家权重；LayerWise_Dynamic_Routing 表示按层调整累计概率阈值；Budget_Dynamic_Routing 表示按 token 的 router 分布集中度分配不同专家数；LayerBudget_Dynamic_Routing 表示进一步按层位置调整 Budget 阈值；BandedLayerBudget_Dynamic_Routing 表示只在指定中间层 band 提升 hard token 专家数，并只在 band 外降低 easy token 专家数；BanSensitiveLayerBudget_Dynamic_Routing 表示只在 Ban 高敏感层提升专家数。默认: none",
    )
    parser.add_argument(
        "--naee_beta",
        type=float,
        default=0.3,
        help="NAEE 阈值系数 beta：若 wi < w1 * beta，则从 wi 开始剪枝。默认: 0.3",
    )
    parser.add_argument(
        "--naee_k_min",
        type=int,
        default=2,
        help="NAEE 每个 token 至少保留的专家数。默认: 2",
    )
    parser.add_argument(
        "--dynamic_routing_threshold",
        type=float,
        default=0.8,
        help="Dynamic_Routing 累计概率阈值 p：保留 softmax 后累计概率达到 p 的最少专家。默认: 0.8",
    )
    parser.add_argument(
        "--dynamic_routing_score_source",
        type=str,
        choices=[
            "raw",
            "renormalized",
            "next_rank_weight",
            "dropped_mass",
            "renorm_minus_next_rank",
            "renorm_minus_scaled_next_rank",
        ],
        default="raw",
        help="Dynamic_Routing/Budget 路由的分数来源。raw 表示使用 full-softmax 后的 top-k 原始概率累计；renormalized 表示先对 top-k 专家权重归一化后再累计；next_rank_weight 表示用 base_k 后第一个被丢弃专家的归一化权重，越大越 hard；dropped_mass 表示用 base_k 之后所有被丢弃专家的归一化权重和，越大越 hard；renorm_minus_next_rank 表示 topn 归一化累计权重减去第一个被丢弃专家权重，越小越 hard；renorm_minus_scaled_next_rank 表示 topn 归一化累计权重减去 alpha 倍第一个被丢弃专家权重，alpha 由 --dynamic_routing_next_rank_penalty 控制。默认: raw",
    )
    parser.add_argument(
        "--dynamic_routing_next_rank_penalty",
        type=float,
        default=1.0,
        help="renorm_minus_scaled_next_rank 的第一个被丢弃专家权重惩罚系数 alpha。默认: 1.0",
    )
    parser.add_argument(
        "--layerwise_dynamic_base_threshold",
        type=float,
        default=0.55,
        help="LayerWise_Dynamic_Routing 的基础累计概率阈值。默认: 0.55",
    )
    parser.add_argument(
        "--layerwise_dynamic_layer_alpha",
        type=float,
        default=0.15,
        help="LayerWise_Dynamic_Routing 的层位置阈值增量，中间层更保守。默认: 0.15",
    )
    parser.add_argument(
        "--layerwise_dynamic_num_layers",
        type=int,
        default=48,
        help="LayerWise_Dynamic_Routing 计算层位置时使用的总层数。Qwen3-30B 默认为 48。",
    )
    parser.add_argument(
        "--layerwise_dynamic_k_min",
        type=int,
        default=1,
        help="LayerWise_Dynamic_Routing 每个 token 至少保留的专家数。默认: 1",
    )
    parser.add_argument(
        "--budget_dynamic_easy_k",
        type=int,
        default=4,
        help="Budget_Dynamic_Routing 中 router 分布很集中时保留的专家数。默认: 4",
    )
    parser.add_argument(
        "--budget_dynamic_base_k",
        type=int,
        default=5,
        help="Budget_Dynamic_Routing 中普通 token 保留的专家数。默认: 5",
    )
    parser.add_argument(
        "--budget_dynamic_hard_k",
        type=int,
        default=6,
        help="Budget_Dynamic_Routing 中 router 分布较分散时保留的专家数。默认: 6",
    )
    parser.add_argument(
        "--budget_dynamic_easy_threshold",
        type=float,
        default=0.90,
        help="Budget_Dynamic_Routing 的 easy token 集中度阈值，高于该值使用 easy_k。默认: 0.90",
    )
    parser.add_argument(
        "--budget_dynamic_hard_threshold",
        type=float,
        default=0.75,
        help="Budget_Dynamic_Routing 的 hard token 集中度阈值，低于该值使用 hard_k。默认: 0.75",
    )
    parser.add_argument(
        "--budget_dynamic_score_topn",
        type=int,
        default=3,
        help="Budget_Dynamic_Routing 用前几个专家累计权重作为集中度分数。默认: 3",
    )
    parser.add_argument(
        "--layer_budget_dynamic_easy_layer_alpha",
        type=float,
        default=0.08,
        help="LayerBudget_Dynamic_Routing 中层位置对 easy 阈值的增量；越大越不容易在中间层降专家数。默认: 0.08",
    )
    parser.add_argument(
        "--layer_budget_dynamic_hard_layer_alpha",
        type=float,
        default=0.08,
        help="LayerBudget_Dynamic_Routing 中层位置对 hard 阈值的增量；越大越容易在中间层增加专家数。默认: 0.08",
    )
    parser.add_argument(
        "--layer_budget_dynamic_num_layers",
        type=int,
        default=48,
        help="LayerBudget_Dynamic_Routing 计算层位置时使用的总层数。Qwen3-30B 默认为 48。",
    )
    parser.add_argument(
        "--band_layer_budget_middle_start",
        type=int,
        default=19,
        help="BandedLayerBudget_Dynamic_Routing 中允许 hard token 增加专家数的起始层（包含）。默认: 19",
    )
    parser.add_argument(
        "--band_layer_budget_middle_end",
        type=int,
        default=28,
        help="BandedLayerBudget_Dynamic_Routing 中允许 hard token 增加专家数的结束层（包含）。默认: 28",
    )
    parser.add_argument(
        "--band_layer_budget_per_layer_k",
        type=str,
        default="",
        help=(
            "可选的逐层固定专家数列表，逗号或空格分隔。设置后覆盖 "
            "BandedLayerBudget_Dynamic_Routing 的动态规则；为空时保持原行为。"
        ),
    )
    parser.add_argument(
        "--band_layer_budget_easy_layers",
        type=str,
        default="",
        help=(
            "可选的允许 easy token 降低专家数的显式层列表；设置后覆盖 "
            "easy_start/easy_end，逗号或空格分隔。"
        ),
    )
    parser.add_argument(
        "--band_layer_budget_hard_layers",
        type=str,
        default="",
        help=(
            "可选的允许 hard token 增加专家数的显式层列表；设置后覆盖 "
            "middle/extra layer band，逗号或空格分隔。"
        ),
    )
    parser.add_argument(
        "--band_layer_budget_token_gate_ratio",
        type=float,
        default=-1.0,
        help=(
            "显式 easy/hard 层中的批内等额 token 配额；例如 0.05 分别将 "
            "最高置信 5%% 降为 easy_k、最低置信 5%% 升为 hard_k。-1 关闭。"
        ),
    )
    parser.add_argument(
        "--band_layer_budget_extra_middle_start",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中允许 hard token 增加专家数的额外起始层（包含）；-1 表示关闭额外层段。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_extra_middle_end",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中允许 hard token 增加专家数的额外结束层（包含）；-1 表示关闭额外层段。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_middle_phase",
        choices=("all", "prefill", "decode"),
        default="all",
        help="BandedLayerBudget_Dynamic_Routing 中 middle band 生效的阶段。默认: all",
    )
    parser.add_argument(
        "--band_layer_budget_easy_start",
        type=int,
        default=0,
        help="BandedLayerBudget_Dynamic_Routing 中允许 easy token 降低专家数的起始层（包含）。默认: 0",
    )
    parser.add_argument(
        "--band_layer_budget_easy_end",
        type=int,
        default=47,
        help="BandedLayerBudget_Dynamic_Routing 中允许 easy token 降低专家数的结束层（包含）。默认: 47",
    )
    parser.add_argument(
        "--band_layer_budget_easy_phase",
        type=str,
        choices=["all", "prefill", "decode"],
        default="all",
        help="BandedLayerBudget_Dynamic_Routing 中允许 easy token 降低专家数的 token 阶段。默认: all",
    )
    parser.add_argument(
        "--band_layer_budget_hard_phase",
        type=str,
        choices=["all", "prefill", "decode"],
        default="all",
        help="BandedLayerBudget_Dynamic_Routing 中允许 hard token 增加专家数的 token 阶段。默认: all",
    )
    parser.add_argument(
        "--band_layer_budget_easy_tail_threshold",
        type=float,
        default=1.0,
        help="BandedLayerBudget_Dynamic_Routing 中 easy 降专家数时允许丢弃的 top-rank 累计权重上限；1.0 表示关闭该限制。默认: 1.0",
    )
    parser.add_argument(
        "--band_layer_budget_hard_min_rank_weight",
        type=float,
        default=0.0,
        help="BandedLayerBudget_Dynamic_Routing 中 hard 升专家数时新增第一个 rank 的最小权重；0.0 表示关闭该限制。默认: 0.0",
    )
    parser.add_argument(
        "--band_layer_budget_hard_max_rank_weight",
        type=float,
        default=1.0,
        help="BandedLayerBudget_Dynamic_Routing 中 hard 升专家数时新增第一个 rank 的最大权重；1.0 表示关闭该限制。默认: 1.0",
    )
    parser.add_argument(
        "--band_layer_budget_hard_decode_min_rank_weight",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing 中仅对 decode hard token 生效的新增第一个 rank 最小权重；-1 表示关闭该限制。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_mixed_rescue_min_rank_weight",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing 中补回 mixed-only hard token 时新增第一个 rank 的最小权重；-1 表示关闭。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_mixed_rescue_max_concentration",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing 中补回 mixed-only hard token 时允许的最大 topn concentration；-1 表示关闭。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_max_tokens",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中允许 prefill hard 升专家数的最大 segment 长度；-1 表示关闭该限制。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_max_segment_tokens",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中允许 prefill hard 升专家数的最大当前 segment/chunk 长度；-1 表示关闭该限制。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_max_density",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing 中允许 prefill hard 升专家数的最大 hard-token segment 密度；-1 表示关闭该限制。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_min_density",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing 中允许 prefill hard 升专家数的最小 hard-token segment 密度；-1 表示关闭该限制。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_max_segment_hard_ratio",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing 中每个 prefill segment 内最多允许升专家数的 hard token 比例；-1 表示关闭该限制。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_segment_cap_score",
        type=str,
        choices=["concentration", "rank4", "rank4_over_topn"],
        default="concentration",
        help="BandedLayerBudget_Dynamic_Routing 中 prefill segment hard ratio 截断时的候选排序分数。默认: concentration",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_conditional_segment_hard_ratio",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing 中满足条件的 prefill segment 使用的 hard token 比例上限；-1 表示关闭条件上限。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_conditional_min_density",
        type=float,
        default=-1.0,
        help="触发 conditional prefill segment hard ratio 的最小候选 hard-token 密度；-1 表示不按密度触发。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_conditional_min_seq_len",
        type=int,
        default=-1,
        help="触发 conditional prefill segment hard ratio 的最小 segment 序列长度；-1 表示不按长度触发。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_conditional_token_ids",
        type=str,
        default="",
        help="触发 conditional prefill segment hard ratio 所需的 token id 列表，逗号或空格分隔；为空表示不按 token id 触发。默认: 空",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_conditional_token_ngrams",
        type=str,
        default="",
        help="触发 conditional prefill segment hard ratio 所需的 token id n-gram 列表；组内逗号或空格分隔，多组用分号分隔；为空表示不按 n-gram 触发。默认: 空",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_marker_segment_hard_ratio",
        type=float,
        default=-1.0,
        help="包含 marker token 的 prefill segment 使用的 hard token 比例上限，会覆盖普通/conditional segment ratio；-1 表示关闭。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_marker_token_ids",
        type=str,
        default="",
        help="触发 marker prefill segment hard ratio override 的 token id 列表，逗号或空格分隔；为空表示关闭 marker override。默认: 空",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_marker_token_ngrams",
        type=str,
        default="",
        help="触发 marker prefill segment hard ratio override 的 token id n-gram 列表；组内逗号或空格分隔，多组用分号分隔；为空表示关闭 marker override。默认: 空",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_exclude_token_ids",
        type=str,
        default="",
        help="BandedLayerBudget_Dynamic_Routing 中 prefill hard 升专家数候选需要排除的 token id 列表，逗号或空格分隔；为空表示不排除。默认: 空",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_min_relative_pos",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing 中允许 prefill hard 升专家数的最小 segment 相对位置；0 为开头、1 为结尾，-1 表示关闭。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_max_relative_pos",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing 中允许 prefill hard 升专家数的最大 segment 相对位置；0 为开头、1 为结尾，-1 表示关闭。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_start",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中额外允许 prefill hard 升专家数的起始层（包含）；-1 表示关闭额外 prefill 层带。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_prefill_end",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中额外允许 prefill hard 升专家数的结束层（包含）；-1 表示关闭额外 prefill 层带。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_decode_min_seq_len",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中允许 decode hard 升专家数的最小序列长度；-1 表示关闭该限制。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_decode_max_seq_len",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中允许 decode hard 升专家数的最大序列长度；-1 表示关闭该限制。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_decode_seq_len_scope",
        choices=("all", "extra"),
        default="all",
        help="BandedLayerBudget_Dynamic_Routing 中 decode 序列长度限制的作用范围；all 表示约束所有 decode hard，extra 表示只约束额外 decode 层带。默认: all",
    )
    parser.add_argument(
        "--band_layer_budget_hard_decode_min_offset",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中允许 decode hard 升专家数的最小生成偏移；0 表示第一个 decode 输入 token，-1 表示关闭。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_decode_max_offset",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中允许 decode hard 升专家数的最大生成偏移；-1 表示关闭。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_decode_start",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中额外允许 decode hard 升专家数的起始层（包含）；-1 表示沿用 middle 层带。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_decode_end",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中额外允许 decode hard 升专家数的结束层（包含）；-1 表示沿用 middle 层带。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_decode_token_ids",
        default="",
        help="BandedLayerBudget_Dynamic_Routing 中仅允许指定 decode 输入 token id 触发 hard 升专家数；逗号或空格分隔，空字符串表示关闭。默认: 空",
    )
    parser.add_argument(
        "--band_layer_budget_hard_layer_prior_start",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中 hard 阈值层先验 boost 的起始层（包含）；-1 表示关闭该层带。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_layer_prior_end",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中 hard 阈值层先验 boost 的结束层（包含）；-1 表示关闭该层带。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_layer_prior_extra_start",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中第二段 hard 阈值层先验 boost 的起始层（包含）；-1 表示关闭该层带。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_layer_prior_extra_end",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中第二段 hard 阈值层先验 boost 的结束层（包含）；-1 表示关闭该层带。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_hard_layer_prior_alpha",
        type=float,
        default=0.0,
        help="BandedLayerBudget_Dynamic_Routing 中对指定 hard layer-prior 层带额外增大 hard 阈值的系数。默认: 0.0",
    )
    parser.add_argument(
        "--band_layer_budget_hard_layer_sensitivity_alpha",
        type=float,
        default=0.0,
        help="BandedLayerBudget_Dynamic_Routing 中使用 Ban layer_sensitivity 增大 hard 阈值的系数；需要同时提供 --ban_artifact_path。默认: 0.0",
    )
    parser.add_argument(
        "--band_layer_budget_late_concentration_start",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中从该层开始改用普通 topn concentration 作为 hard 分数；-1 表示关闭。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_late_concentration_threshold",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing late concentration 分支的 hard 阈值；-1 表示沿用 --budget_dynamic_hard_threshold。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_late_rescue_start",
        type=int,
        default=-1,
        help="BandedLayerBudget_Dynamic_Routing 中从该层开始启用 late token rescue；-1 表示关闭。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_late_rescue_concentration_threshold",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing late token rescue 的 concentration hard 阈值；-1 表示关闭。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_late_rescue_mixed_max",
        type=float,
        default=-1.0,
        help="BandedLayerBudget_Dynamic_Routing late token rescue 允许的 mixed hard score 上界；-1 表示不限制。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile1_token_ngrams",
        type=str,
        default="",
        help="BandedLayerBudget_Dynamic_Routing prompt profile1 的精确 token n-gram 列表；每个 n-gram 用逗号/空格分隔，多个 n-gram 用分号分隔。默认: 空",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile1_hard_threshold",
        type=float,
        default=-1.0,
        help="prompt profile1 命中后覆盖 hard 阈值；-1 表示不覆盖。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile1_hard_min_rank_weight",
        type=float,
        default=-1.0,
        help="prompt profile1 命中后覆盖 hard 升专家数时新增第一个 rank 的最小权重；-1 表示不覆盖。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile1_middle_start",
        type=int,
        default=-1,
        help="prompt profile1 命中后覆盖 hard 层带起始层；-1 表示不覆盖。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile1_middle_end",
        type=int,
        default=-1,
        help="prompt profile1 命中后覆盖 hard 层带结束层；-1 表示不覆盖。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile1_prefill_segment_hard_ratio",
        type=float,
        default=-1.0,
        help="prompt profile1 命中后覆盖 prefill segment hard token 比例上限；-1 表示不覆盖。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile2_token_ngrams",
        type=str,
        default="",
        help="BandedLayerBudget_Dynamic_Routing prompt profile2 的精确 token n-gram 列表；每个 n-gram 用逗号/空格分隔，多个 n-gram 用分号分隔。默认: 空",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile2_hard_threshold",
        type=float,
        default=-1.0,
        help="prompt profile2 命中后覆盖 hard 阈值；-1 表示不覆盖。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile2_hard_min_rank_weight",
        type=float,
        default=-1.0,
        help="prompt profile2 命中后覆盖 hard 升专家数时新增第一个 rank 的最小权重；-1 表示不覆盖。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile2_middle_start",
        type=int,
        default=-1,
        help="prompt profile2 命中后覆盖 hard 层带起始层；-1 表示不覆盖。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile2_middle_end",
        type=int,
        default=-1,
        help="prompt profile2 命中后覆盖 hard 层带结束层；-1 表示不覆盖。默认: -1",
    )
    parser.add_argument(
        "--band_layer_budget_prompt_profile2_prefill_segment_hard_ratio",
        type=float,
        default=-1.0,
        help="prompt profile2 命中后覆盖 prefill segment hard token 比例上限；-1 表示不覆盖。默认: -1",
    )
    parser.add_argument(
        "--diep_artifact_path",
        type=str,
        default=None,
        help="DiEP 校准 artifact (.pt) 路径，包含 sim_matrix/mean_sim/gamma_1。使用 --expert_pruning_method DiEP 时必须提供。",
    )
    parser.add_argument(
        "--diep_pruning_mode",
        type=str,
        choices=["independent", "prefix"],
        default="independent",
        help="DiEP 剪枝模式。independent 表示每个 expert 按自己的 gamma2 独立判断；prefix 表示继承 NAEE 的首次剪枝后后续全剪。默认: independent",
    )
    parser.add_argument(
        "--diep_use_gamma1",
        action="store_true",
        help="把校准得到的 gamma_1 乘进 DiEP 阈值，即论文式 (12) 的 gamma = gamma_1 * gamma_2。"
             "配合 --naee_beta 1.0 使用可精确复现论文设定；默认关闭，由 --naee_beta 代替 gamma_1。",
    )
    parser.add_argument(
        "--diep_gamma_alpha",
        type=float,
        default=1.0,
        help="给 gamma_2 加的指数：阈值用 beta * gamma_2**alpha。1.0 是论文规则的直接推广；"
             "0.0 完全去掉相似度信号即退化为 NAEE；取中间值可压缩 gamma_2 的离散度。默认: 1.0",
    )
    parser.add_argument(
        "--diep_threshold_cap",
        type=float,
        default=None,
        help="阈值相对 top-1 门控权重的上界，即 beta * gamma_2**alpha 的上限。必须小于 1 才有作用："
             "等于 1 时阈值恰为 w_e0，降序权重仍全部低于它，决定与不设上界相同。默认: 不设",
    )
    parser.add_argument(
        "--ban_artifact_path",
        type=str,
        default=None,
        help="Ban 校准 artifact (.pt) 路径，包含 layer_sensitivity/r_min/r_max。使用 --expert_pruning_method Ban 时必须提供。",
    )
    parser.add_argument(
        "--ban_lambda",
        type=float,
        default=0.7,
        help="Ban 剪枝保守系数 lambda，越大平均保留专家越多。默认: 0.7",
    )
    parser.add_argument(
        "--ban_k_min",
        type=int,
        default=3,
        help="Ban 每个 token 每层至少保留的专家数。默认: 3",
    )
    parser.add_argument(
        "--expert_pruning_debug",
        action="store_true",
        help="打印前几次本地专家剪枝 routing 的统计信息，用于确认实际保留专家数。",
    )
    parser.add_argument(
        "--expert_pruning_debug_max_prints",
        type=int,
        default=8,
        help="--expert_pruning_debug 启用时最多打印多少次 routing 统计。默认: 8",
    )
    parser.add_argument(
        "--mc_moe_protection_ratio",
        type=float,
        default=0.02,
        help="MC_MoE 中按重要性保护的 token 比例 tau_h。默认: 0.02",
    )
    parser.add_argument(
        "--eac_alpha",
        type=float,
        default=0.5,
        help="EAC_MoE 专家活跃阈值系数 alpha。若某专家在 prefill 段内实际分配 token 数 < l*k/n*alpha，则该专家在该段内被剪枝。默认: 0.5",
    )
    parser.add_argument(
        "--biased_renorm_keep_topn",
        type=int,
        default=3,
        help="TopK_Biased_Renorm 中只放大前几个专家权重；剩余专家保留原始 softmax 权重。默认: 3",
    )
    parser.add_argument(
        "--router_value_artifact_path",
        type=str,
        default="",
        help="RouterValue_Dynamic_Routing 的 router-only 线性预测器 JSON 路径。",
    )
    parser.add_argument(
        "--router_value_base_k",
        type=int,
        default=3,
        help="RouterValue_Dynamic_Routing 的基础专家数。默认: 3",
    )
    parser.add_argument(
        "--router_value_quota_ratio",
        type=float,
        default=0.10,
        help="RouterValue_Dynamic_Routing 中从 K 提升到 K+1 的 token 比例。默认: 0.10",
    )
    parser.add_argument(
        "--router_value_selection_mode",
        choices=["threshold", "quota"],
        default="threshold",
        help="RouterValue token 选择方式：校准绝对阈值或当前 routing batch 内配额。默认: threshold",
    )
    parser.add_argument(
        "--router_value_swap_direction",
        choices=["both", "promote_only", "demote_only"],
        default="both",
        help="RouterValue 在基础 K 附近允许的预算方向。默认: both",
    )
    parser.add_argument(
        "--attention_sink_probe",
        action="store_true",
        help="实验性开关：在线观察 Qwen3-MoE attention 的 Q/K 信号，并确认该信息能传到后续 MoE；暂不改变专家剪枝结果。",
    )
    parser.add_argument(
        "--attention_sink_probe_max_prints",
        type=int,
        default=8,
        help="attention sink probe 最多打印多少条诊断信息。默认: 8",
    )
    parser.add_argument(
        "--attention_sink_probe_max_query_tokens",
        type=int,
        default=2048,
        help="attention sink probe 每次最多使用多少个 query token。默认: 2048",
    )
    parser.add_argument(
        "--attention_sink_probe_max_key_tokens",
        type=int,
        default=2048,
        help="attention sink probe 每次最多检查前多少个 key token。默认: 2048",
    )
    parser.add_argument(
        "--attention_sink_probe_topk",
        type=int,
        default=4,
        help="attention sink probe 每层临时标记的高 sink token 数，仅用于诊断，不参与剪枝。默认: 4",
    )
    parser.add_argument(
        "--load_multilingual_tasks",
        action="store_true",
        help="加载 lighteval 多语言任务注册表（含 ceval/cmmlu/agieval 等）。评 C-Eval 等时必须加此选项，与官方 CLI 的 --load-tasks-multilingual 一致",
    )
    parser.add_argument(
        "--collect_router_distribution",
        action="store_true",
        help="Observe native vLLM router softmax distributions without changing "
             "expert selection. Incompatible with pruning, Fixed-K overrides, "
             "and non-native model configs.",
    )
    parser.add_argument(
        "--router_distribution_dir",
        type=str,
        default=None,
        help="Directory for router-distribution worker snapshots. Default: "
             "<output_dir>/.router_distribution/<uuid>/",
    )
    parser.add_argument(
        "--router_distribution_sample_rate",
        type=float,
        default=0.005,
        help="Deterministic reservoir sampling rate for full sorted softmax "
             "vectors. Default: 0.005 (0.5%%).",
    )
    parser.add_argument(
        "--router_distribution_sample_cap",
        type=int,
        default=2048,
        help="Maximum stored raw sorted-probability vectors per layer/stage "
             "and worker. Default: 2048.",
    )
    parser.add_argument(
        "--router_distribution_histogram_bins",
        type=int,
        default=16,
        help="Histogram bin count for concentration metrics. Default: 16.",
    )
    parser.add_argument(
        "--router_distribution_flush_interval",
        type=int,
        default=512,
        help="Flush online aggregates every N native routing calls. "
             "Default: 512.",
    )
    parser.add_argument(
        "--router_distribution_seed",
        type=int,
        default=0,
        help="Deterministic seed for reservoir sampling. Default: 0.",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="传给 vLLM：enforce_eager=True（关闭 CUDAGraph 等，便于排错但更慢）。默认 False，与 vLLM 一致。"
             "依赖本仓库对 lighteval 的本地修改（VLLMModelConfig.enforce_eager 字段）。",
    )
    parser.add_argument(
        "--disable_custom_all_reduce",
        action="store_true",
        help="传给 vLLM：disable_custom_all_reduce=True。默认 False。"
             "当多卡 CUDA graph/custom all-reduce warmup 报错时可用于诊断或绕过。",
    )
    parser.add_argument(
        "--language_model_only",
        action="store_true",
        help="传给新版 vLLM：language_model_only=True，跳过多模态编码器并仅加载语言模型。"
             "用于 Qwen3.5/Qwen3.6 等统一多模态 checkpoint 的纯文本评测。默认 False。",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否允许 Hugging Face remote code。默认保持兼容行为 True；"
             "已进入 Transformers/vLLM 注册表的模型应显式传 --no-trust_remote_code。",
    )
    parser.add_argument(
        "--cpu_offload_gb",
        type=float,
        default=0,
        help="每张 GPU 将多少 GiB 的模型权重卸载到 CPU 内存。默认 0（不卸载）。"
             "当 GPU 显存不够装完整权重时使用，例如用 4 张 80G 卡跑 235B 模型可设 --cpu_offload_gb 40。"
             "注意：会显著降低推理速度。依赖本仓库对 lighteval 的本地修改（VLLMModelConfig.cpu_offload_gb 字段）。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.sample_shard_id is None) != (args.sample_num_shards is None):
        raise ValueError(
            "--sample_shard_id and --sample_num_shards must be set together"
        )
    if args.sample_num_shards is not None:
        if args.max_samples is not None:
            raise ValueError("sample sharding cannot be combined with --max_samples")
        if args.sample_num_shards < 1:
            raise ValueError("--sample_num_shards must be >= 1")
        if not 0 <= args.sample_shard_id < args.sample_num_shards:
            raise ValueError(
                "--sample_shard_id must satisfy 0 <= id < sample_num_shards"
            )
        os.environ["EXPERT_PRUNING_SAMPLE_SHARD_ID"] = str(
            args.sample_shard_id
        )
        os.environ["EXPERT_PRUNING_SAMPLE_NUM_SHARDS"] = str(
            args.sample_num_shards
        )

    from expert_pruning.config_shims import register_missing_configs
    added = register_missing_configs()
    if added:
        print(f"[config_shims] 为 transformers 不认识的 model_type 注册了配置类: {added}")

    use_local_expert_router = (
        args.use_local_expert_router
        or args.expert_pruning_method != "none"
        or args.attention_sink_probe
    )
    if args.naee_beta < 0:
        raise ValueError("--naee_beta must be non-negative")
    if args.naee_k_min < 1:
        raise ValueError("--naee_k_min must be >= 1")
    if not 0 < args.dynamic_routing_threshold <= 1:
        raise ValueError("--dynamic_routing_threshold must be in (0, 1]")
    if not 0 < args.layerwise_dynamic_base_threshold <= 1:
        raise ValueError(
            "--layerwise_dynamic_base_threshold must be in (0, 1]")
    if args.layerwise_dynamic_num_layers < 1:
        raise ValueError("--layerwise_dynamic_num_layers must be >= 1")
    if args.layerwise_dynamic_k_min < 1:
        raise ValueError("--layerwise_dynamic_k_min must be >= 1")
    if args.budget_dynamic_easy_k < 1:
        raise ValueError("--budget_dynamic_easy_k must be >= 1")
    if args.budget_dynamic_base_k < 1:
        raise ValueError("--budget_dynamic_base_k must be >= 1")
    if args.budget_dynamic_hard_k < 1:
        raise ValueError("--budget_dynamic_hard_k must be >= 1")
    if not 0 <= args.budget_dynamic_hard_threshold <= 1:
        raise ValueError("--budget_dynamic_hard_threshold must be in [0, 1]")
    if not 0 <= args.budget_dynamic_easy_threshold <= 1:
        raise ValueError("--budget_dynamic_easy_threshold must be in [0, 1]")
    if args.budget_dynamic_hard_threshold > args.budget_dynamic_easy_threshold:
        raise ValueError(
            "--budget_dynamic_hard_threshold must be <= "
            "--budget_dynamic_easy_threshold")
    if args.budget_dynamic_score_topn < 1:
        raise ValueError("--budget_dynamic_score_topn must be >= 1")
    if args.dynamic_routing_next_rank_penalty < 0:
        raise ValueError(
            "--dynamic_routing_next_rank_penalty must be >= 0")
    if args.layer_budget_dynamic_num_layers < 1:
        raise ValueError("--layer_budget_dynamic_num_layers must be >= 1")
    if args.band_layer_budget_middle_start < 0:
        raise ValueError("--band_layer_budget_middle_start must be >= 0")
    if args.band_layer_budget_middle_end < args.band_layer_budget_middle_start:
        raise ValueError(
            "--band_layer_budget_middle_end must be >= "
            "--band_layer_budget_middle_start")
    if args.band_layer_budget_extra_middle_start < -1:
        raise ValueError(
            "--band_layer_budget_extra_middle_start must be >= -1")
    if args.band_layer_budget_extra_middle_end < -1:
        raise ValueError(
            "--band_layer_budget_extra_middle_end must be >= -1")
    if ((args.band_layer_budget_extra_middle_start >= 0
         or args.band_layer_budget_extra_middle_end >= 0)
            and args.band_layer_budget_extra_middle_end
            < args.band_layer_budget_extra_middle_start):
        raise ValueError(
            "--band_layer_budget_extra_middle_end must be >= "
            "--band_layer_budget_extra_middle_start")
    if args.band_layer_budget_middle_phase not in {"all", "prefill", "decode"}:
        raise ValueError(
            "--band_layer_budget_middle_phase must be all, prefill, or "
            "decode")
    if args.band_layer_budget_easy_start < 0:
        raise ValueError("--band_layer_budget_easy_start must be >= 0")
    if args.band_layer_budget_easy_end < args.band_layer_budget_easy_start:
        raise ValueError(
            "--band_layer_budget_easy_end must be >= "
            "--band_layer_budget_easy_start")
    if args.band_layer_budget_easy_phase not in {"all", "prefill", "decode"}:
        raise ValueError(
            "--band_layer_budget_easy_phase must be all, prefill, or decode")
    if args.band_layer_budget_hard_phase not in {"all", "prefill", "decode"}:
        raise ValueError(
            "--band_layer_budget_hard_phase must be all, prefill, or decode")
    if not 0 <= args.band_layer_budget_easy_tail_threshold <= 1:
        raise ValueError(
            "--band_layer_budget_easy_tail_threshold must be in [0, 1]")
    if not 0 <= args.band_layer_budget_hard_min_rank_weight <= 1:
        raise ValueError(
            "--band_layer_budget_hard_min_rank_weight must be in [0, 1]")
    if not 0 <= args.band_layer_budget_hard_max_rank_weight <= 1:
        raise ValueError(
            "--band_layer_budget_hard_max_rank_weight must be in [0, 1]")
    if (args.band_layer_budget_hard_decode_min_rank_weight < 0
            and args.band_layer_budget_hard_decode_min_rank_weight != -1):
        raise ValueError(
            "--band_layer_budget_hard_decode_min_rank_weight must be in "
            "[0, 1] or -1")
    if args.band_layer_budget_hard_decode_min_rank_weight > 1:
        raise ValueError(
            "--band_layer_budget_hard_decode_min_rank_weight must be in "
            "[0, 1] or -1")
    if (args.band_layer_budget_mixed_rescue_min_rank_weight < 0
            and args.band_layer_budget_mixed_rescue_min_rank_weight != -1):
        raise ValueError(
            "--band_layer_budget_mixed_rescue_min_rank_weight must be in "
            "[0, 1] or -1")
    if args.band_layer_budget_mixed_rescue_min_rank_weight > 1:
        raise ValueError(
            "--band_layer_budget_mixed_rescue_min_rank_weight must be in "
            "[0, 1] or -1")
    if (args.band_layer_budget_mixed_rescue_max_concentration < 0
            and args.band_layer_budget_mixed_rescue_max_concentration != -1):
        raise ValueError(
            "--band_layer_budget_mixed_rescue_max_concentration must be in "
            "[0, 1] or -1")
    if args.band_layer_budget_mixed_rescue_max_concentration > 1:
        raise ValueError(
            "--band_layer_budget_mixed_rescue_max_concentration must be in "
            "[0, 1] or -1")
    if args.band_layer_budget_hard_prefill_max_tokens < -1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_tokens must be >= -1")
    if args.band_layer_budget_hard_prefill_max_segment_tokens < -1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_segment_tokens must be "
            ">= -1")
    if (args.band_layer_budget_hard_prefill_max_density < 0
            and args.band_layer_budget_hard_prefill_max_density != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_density must be in "
            "[0, 1] or -1")
    if args.band_layer_budget_hard_prefill_max_density > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_density must be in "
            "[0, 1] or -1")
    if (args.band_layer_budget_hard_prefill_min_density < 0
            and args.band_layer_budget_hard_prefill_min_density != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_min_density must be in "
            "[0, 1] or -1")
    if args.band_layer_budget_hard_prefill_min_density > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_min_density must be in "
            "[0, 1] or -1")
    if (args.band_layer_budget_hard_prefill_max_segment_hard_ratio < 0
            and args.band_layer_budget_hard_prefill_max_segment_hard_ratio
            != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_segment_hard_ratio must be "
            "in [0, 1] or -1")
    if args.band_layer_budget_hard_prefill_max_segment_hard_ratio > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_segment_hard_ratio must be "
            "in [0, 1] or -1")
    if (args.band_layer_budget_hard_prefill_conditional_segment_hard_ratio < 0
            and
            args.band_layer_budget_hard_prefill_conditional_segment_hard_ratio
            != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_conditional_segment_hard_ratio "
            "must be in [0, 1] or -1")
    if args.band_layer_budget_hard_prefill_conditional_segment_hard_ratio > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_conditional_segment_hard_ratio "
            "must be in [0, 1] or -1")
    if (args.band_layer_budget_hard_prefill_conditional_min_density < 0
            and args.band_layer_budget_hard_prefill_conditional_min_density
            != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_conditional_min_density must "
            "be in [0, 1] or -1")
    if args.band_layer_budget_hard_prefill_conditional_min_density > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_conditional_min_density must "
            "be in [0, 1] or -1")
    if args.band_layer_budget_hard_prefill_conditional_min_seq_len < -1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_conditional_min_seq_len must "
            "be >= -1")
    for token_id in str(
            args.band_layer_budget_hard_prefill_conditional_token_ids
    ).replace(",", " ").split():
        try:
            parsed_token_id = int(token_id)
        except ValueError as exc:
            raise ValueError(
                "--band_layer_budget_hard_prefill_conditional_token_ids "
                "must contain comma/space separated integers") from exc
        if parsed_token_id < 0:
            raise ValueError(
                "--band_layer_budget_hard_prefill_conditional_token_ids "
                "must contain non-negative token ids")
    for ngram in str(
            args.band_layer_budget_hard_prefill_conditional_token_ngrams
    ).split(";"):
        for token_id in ngram.replace(",", " ").split():
            try:
                parsed_token_id = int(token_id)
            except ValueError as exc:
                raise ValueError(
                    "--band_layer_budget_hard_prefill_conditional_token_ngrams "
                    "must contain semicolon separated integer n-grams"
                ) from exc
            if parsed_token_id < 0:
                raise ValueError(
                    "--band_layer_budget_hard_prefill_conditional_token_ngrams "
                    "must contain non-negative token ids")
    if (args.band_layer_budget_hard_prefill_marker_segment_hard_ratio < 0
            and args.band_layer_budget_hard_prefill_marker_segment_hard_ratio
            != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_marker_segment_hard_ratio "
            "must be in [0, 1] or -1")
    if args.band_layer_budget_hard_prefill_marker_segment_hard_ratio > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_marker_segment_hard_ratio "
            "must be in [0, 1] or -1")
    for token_id in str(
            args.band_layer_budget_hard_prefill_marker_token_ids
    ).replace(",", " ").split():
        try:
            parsed_token_id = int(token_id)
        except ValueError as exc:
            raise ValueError(
                "--band_layer_budget_hard_prefill_marker_token_ids "
                "must contain comma/space separated integers") from exc
        if parsed_token_id < 0:
            raise ValueError(
                "--band_layer_budget_hard_prefill_marker_token_ids "
                "must contain non-negative token ids")
    for ngram in str(
            args.band_layer_budget_hard_prefill_marker_token_ngrams
    ).split(";"):
        for token_id in ngram.replace(",", " ").split():
            try:
                parsed_token_id = int(token_id)
            except ValueError as exc:
                raise ValueError(
                    "--band_layer_budget_hard_prefill_marker_token_ngrams "
                    "must contain semicolon separated integer n-grams"
                ) from exc
            if parsed_token_id < 0:
                raise ValueError(
                    "--band_layer_budget_hard_prefill_marker_token_ngrams "
                    "must contain non-negative token ids")
    for token_id in str(
            args.band_layer_budget_hard_prefill_exclude_token_ids
    ).replace(",", " ").split():
        try:
            parsed_token_id = int(token_id)
        except ValueError as exc:
            raise ValueError(
                "--band_layer_budget_hard_prefill_exclude_token_ids "
                "must contain comma/space separated integers") from exc
        if parsed_token_id < 0:
            raise ValueError(
                "--band_layer_budget_hard_prefill_exclude_token_ids "
                "must contain non-negative token ids")
    if (args.band_layer_budget_hard_prefill_min_relative_pos < 0
            and args.band_layer_budget_hard_prefill_min_relative_pos != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_min_relative_pos must be "
            "in [0, 1] or -1")
    if args.band_layer_budget_hard_prefill_min_relative_pos > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_min_relative_pos must be "
            "in [0, 1] or -1")
    if (args.band_layer_budget_hard_prefill_max_relative_pos < 0
            and args.band_layer_budget_hard_prefill_max_relative_pos != -1):
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_relative_pos must be "
            "in [0, 1] or -1")
    if args.band_layer_budget_hard_prefill_max_relative_pos > 1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_relative_pos must be "
            "in [0, 1] or -1")
    if (args.band_layer_budget_hard_prefill_min_relative_pos >= 0
            and args.band_layer_budget_hard_prefill_max_relative_pos >= 0
            and args.band_layer_budget_hard_prefill_max_relative_pos
            < args.band_layer_budget_hard_prefill_min_relative_pos):
        raise ValueError(
            "--band_layer_budget_hard_prefill_max_relative_pos must be >= "
            "--band_layer_budget_hard_prefill_min_relative_pos")
    if args.band_layer_budget_hard_prefill_start < -1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_start must be >= -1")
    if args.band_layer_budget_hard_prefill_end < -1:
        raise ValueError(
            "--band_layer_budget_hard_prefill_end must be >= -1")
    if ((args.band_layer_budget_hard_prefill_start >= 0
         or args.band_layer_budget_hard_prefill_end >= 0)
            and args.band_layer_budget_hard_prefill_end
            < args.band_layer_budget_hard_prefill_start):
        raise ValueError(
            "--band_layer_budget_hard_prefill_end must be >= "
            "--band_layer_budget_hard_prefill_start")
    if args.band_layer_budget_hard_decode_min_seq_len < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_min_seq_len must be >= -1")
    if args.band_layer_budget_hard_decode_max_seq_len < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_max_seq_len must be >= -1")
    if (args.band_layer_budget_hard_decode_min_seq_len >= 0
            and args.band_layer_budget_hard_decode_max_seq_len >= 0
            and args.band_layer_budget_hard_decode_max_seq_len
            < args.band_layer_budget_hard_decode_min_seq_len):
        raise ValueError(
            "--band_layer_budget_hard_decode_max_seq_len must be >= "
            "--band_layer_budget_hard_decode_min_seq_len")
    if args.band_layer_budget_hard_decode_seq_len_scope not in {"all", "extra"}:
        raise ValueError(
            "--band_layer_budget_hard_decode_seq_len_scope must be all or "
            "extra")
    if args.band_layer_budget_hard_decode_min_offset < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_min_offset must be >= -1")
    if args.band_layer_budget_hard_decode_max_offset < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_max_offset must be >= -1")
    if (args.band_layer_budget_hard_decode_min_offset >= 0
            and args.band_layer_budget_hard_decode_max_offset >= 0
            and args.band_layer_budget_hard_decode_max_offset
            < args.band_layer_budget_hard_decode_min_offset):
        raise ValueError(
            "--band_layer_budget_hard_decode_max_offset must be >= "
            "--band_layer_budget_hard_decode_min_offset")
    if args.band_layer_budget_hard_decode_start < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_start must be >= -1")
    if args.band_layer_budget_hard_decode_end < -1:
        raise ValueError(
            "--band_layer_budget_hard_decode_end must be >= -1")
    if ((args.band_layer_budget_hard_decode_start >= 0
         or args.band_layer_budget_hard_decode_end >= 0)
            and args.band_layer_budget_hard_decode_end
            < args.band_layer_budget_hard_decode_start):
        raise ValueError(
            "--band_layer_budget_hard_decode_end must be >= "
            "--band_layer_budget_hard_decode_start")
    for item in str(args.band_layer_budget_hard_decode_token_ids).replace(
            ",", " ").split():
        if int(item) < 0:
            raise ValueError(
                "--band_layer_budget_hard_decode_token_ids must contain "
                "non-negative token ids")
    if args.band_layer_budget_hard_layer_prior_start < -1:
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_start must be >= -1")
    if args.band_layer_budget_hard_layer_prior_end < -1:
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_end must be >= -1")
    if ((args.band_layer_budget_hard_layer_prior_start >= 0
         or args.band_layer_budget_hard_layer_prior_end >= 0)
            and args.band_layer_budget_hard_layer_prior_end
            < args.band_layer_budget_hard_layer_prior_start):
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_end must be >= "
            "--band_layer_budget_hard_layer_prior_start")
    if args.band_layer_budget_hard_layer_prior_extra_start < -1:
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_extra_start must be >= -1")
    if args.band_layer_budget_hard_layer_prior_extra_end < -1:
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_extra_end must be >= -1")
    if ((args.band_layer_budget_hard_layer_prior_extra_start >= 0
         or args.band_layer_budget_hard_layer_prior_extra_end >= 0)
            and args.band_layer_budget_hard_layer_prior_extra_end
            < args.band_layer_budget_hard_layer_prior_extra_start):
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_extra_end must be >= "
            "--band_layer_budget_hard_layer_prior_extra_start")
    if args.band_layer_budget_hard_layer_prior_alpha < 0:
        raise ValueError(
            "--band_layer_budget_hard_layer_prior_alpha must be >= 0")
    if args.band_layer_budget_hard_layer_sensitivity_alpha < 0:
        raise ValueError(
            "--band_layer_budget_hard_layer_sensitivity_alpha must be >= 0")
    if args.band_layer_budget_late_concentration_start < -1:
        raise ValueError(
            "--band_layer_budget_late_concentration_start must be >= -1")
    if (args.band_layer_budget_late_concentration_threshold < 0
            and args.band_layer_budget_late_concentration_threshold != -1):
        raise ValueError(
            "--band_layer_budget_late_concentration_threshold must be in "
            "[0, 1] or -1")
    if args.band_layer_budget_late_concentration_threshold > 1:
        raise ValueError(
            "--band_layer_budget_late_concentration_threshold must be in "
            "[0, 1] or -1")
    if args.band_layer_budget_late_rescue_start < -1:
        raise ValueError(
            "--band_layer_budget_late_rescue_start must be >= -1")
    if (args.band_layer_budget_late_rescue_concentration_threshold < 0
            and args.band_layer_budget_late_rescue_concentration_threshold
            != -1):
        raise ValueError(
            "--band_layer_budget_late_rescue_concentration_threshold must be "
            "in [0, 1] or -1")
    if args.band_layer_budget_late_rescue_concentration_threshold > 1:
        raise ValueError(
            "--band_layer_budget_late_rescue_concentration_threshold must be "
            "in [0, 1] or -1")
    if (args.band_layer_budget_late_rescue_mixed_max < 0
            and args.band_layer_budget_late_rescue_mixed_max != -1):
        raise ValueError(
            "--band_layer_budget_late_rescue_mixed_max must be in [0, 1] "
            "or -1")
    if args.band_layer_budget_late_rescue_mixed_max > 1:
        raise ValueError(
            "--band_layer_budget_late_rescue_mixed_max must be in [0, 1] "
            "or -1")
    for profile_id in (1, 2):
        token_ngrams = getattr(
            args,
            f"band_layer_budget_prompt_profile{profile_id}_token_ngrams",
        )
        for ngram in str(token_ngrams).split(";"):
            for token_id in ngram.replace(",", " ").split():
                try:
                    parsed_token_id = int(token_id)
                except ValueError as exc:
                    raise ValueError(
                        "--band_layer_budget_prompt_profile"
                        f"{profile_id}_token_ngrams must contain "
                        "comma/space separated integers; separate multiple "
                        "n-grams with ';'") from exc
                if parsed_token_id < 0:
                    raise ValueError(
                        "--band_layer_budget_prompt_profile"
                        f"{profile_id}_token_ngrams must contain "
                        "non-negative token ids")
        hard_threshold = getattr(
            args,
            f"band_layer_budget_prompt_profile{profile_id}_hard_threshold",
        )
        if hard_threshold < 0 and hard_threshold != -1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_hard_threshold must be in [0, 1] or -1")
        if hard_threshold > 1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_hard_threshold must be in [0, 1] or -1")
        hard_min_rank_weight = getattr(
            args,
            "band_layer_budget_prompt_profile"
            f"{profile_id}_hard_min_rank_weight",
        )
        if hard_min_rank_weight < 0 and hard_min_rank_weight != -1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_hard_min_rank_weight must be in [0, 1] "
                "or -1")
        if hard_min_rank_weight > 1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_hard_min_rank_weight must be in [0, 1] "
                "or -1")
        middle_start = getattr(
            args,
            f"band_layer_budget_prompt_profile{profile_id}_middle_start",
        )
        middle_end = getattr(
            args,
            f"band_layer_budget_prompt_profile{profile_id}_middle_end",
        )
        if middle_start < -1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_middle_start must be >= -1")
        if middle_end < -1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_middle_end must be >= -1")
        if ((middle_start >= 0 or middle_end >= 0)
                and middle_end < middle_start):
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_middle_end must be >= corresponding "
                "middle_start")
        segment_ratio = getattr(
            args,
            "band_layer_budget_prompt_profile"
            f"{profile_id}_prefill_segment_hard_ratio",
        )
        if segment_ratio < 0 and segment_ratio != -1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_prefill_segment_hard_ratio must be in "
                "[0, 1] or -1")
        if segment_ratio > 1:
            raise ValueError(
                "--band_layer_budget_prompt_profile"
                f"{profile_id}_prefill_segment_hard_ratio must be in "
                "[0, 1] or -1")
    for method, flag, path in (("DiEP", "--diep_artifact_path", args.diep_artifact_path),
                               ("Ban", "--ban_artifact_path", args.ban_artifact_path)):
        if args.expert_pruning_method != method:
            continue
        if not path:
            raise ValueError(
                f"{flag} is required for {method}. It reads a calibration artifact "
                f"computed once per model; generate it with "
                f"`bash scripts/models/<model>/calibrate.sh`.")
        # Fail here rather than let torch.load raise a bare FileNotFoundError several
        # frames deeper, where nothing says what the file is or how to produce it.
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{method} calibration artifact not found: {path}\n"
                f"Generate it with `bash scripts/models/<model>/calibrate.sh`, or point "
                f"{flag} at an existing one. Artifacts are 3-48 MB per model and are not "
                f"shipped with the repository.")
    if not 0 <= args.ban_lambda <= 1:
        raise ValueError("--ban_lambda must be in [0, 1]")
    if args.ban_k_min < 1:
        raise ValueError("--ban_k_min must be >= 1")
    if args.expert_pruning_debug_max_prints < 0:
        raise ValueError("--expert_pruning_debug_max_prints must be >= 0")
    if not 0 < args.mc_moe_protection_ratio <= 1:
        raise ValueError("--mc_moe_protection_ratio must be in (0, 1]")
    if not 0 < args.eac_alpha < 1:
        raise ValueError("--eac_alpha must be in (0, 1)")
    if args.biased_renorm_keep_topn < 1:
        raise ValueError("--biased_renorm_keep_topn must be >= 1")
    if (args.expert_pruning_method == "RouterValue_Dynamic_Routing"
            and not args.router_value_artifact_path):
        raise ValueError(
            "--router_value_artifact_path is required for "
            "RouterValue_Dynamic_Routing")
    if args.router_value_base_k < 1:
        raise ValueError("--router_value_base_k must be >= 1")
    if not 0 <= args.router_value_quota_ratio <= 1:
        raise ValueError("--router_value_quota_ratio must be in [0, 1]")
    if args.attention_sink_probe_max_prints < 0:
        raise ValueError("--attention_sink_probe_max_prints must be >= 0")
    if args.attention_sink_probe_max_query_tokens < 1:
        raise ValueError(
            "--attention_sink_probe_max_query_tokens must be >= 1")
    if args.attention_sink_probe_max_key_tokens < 1:
        raise ValueError("--attention_sink_probe_max_key_tokens must be >= 1")
    if args.attention_sink_probe_topk < 1:
        raise ValueError("--attention_sink_probe_topk must be >= 1")
    if args.expert_pruning_method == "MC_MoE" and not args.enforce_eager:
        args.enforce_eager = True
        print("MC_MoE requires online attention information; automatically "
              "enabling --enforce_eager.")
    if (args.expert_pruning_method == "BanSensitiveLayerBudget_Dynamic_Routing"
            and not args.enforce_eager):
        # Its per-layer budget bookkeeping touches the host while a CUDA graph is being
        # captured, which aborts capture with "operation not permitted when stream is
        # capturing".
        args.enforce_eager = True
        print("BanSensitiveLayerBudget_Dynamic_Routing cannot be CUDA-graph "
              "captured; automatically enabling --enforce_eager.")
    if (use_local_expert_router and not args.enforce_eager
            and _moe_routing_is_inlined_by_compile(args.model_path)):
        if args.allow_compiled_router:
            print("Quantized MoE routing runs inside the compiled region. Keeping CUDA "
                  "graphs as asked: the pruning still applies, but this run will not be "
                  "able to count experts, so average_selected_experts has to come from a "
                  "separate --enforce_eager probe of the same configuration.")
        else:
            args.enforce_eager = True
            print("Quantized MoE routing runs inside the compiled region, which traces "
                  "away the router's Python-side statistics; automatically enabling "
                  "--enforce_eager so expert counts are still measured.")

    if args.collect_router_distribution:
        if use_local_expert_router or args.expert_pruning_method != "none":
            raise ValueError(
                "--collect_router_distribution requires native routing: do not "
                "combine with --use_local_expert_router or a non-none "
                "--expert_pruning_method")
        if args.num_experts_per_tok is not None:
            raise ValueError(
                "--collect_router_distribution requires the native model config; "
                "remove --num_experts_per_tok / Fixed-K overrides")
        if args.attention_sink_probe:
            raise ValueError(
                "--collect_router_distribution is incompatible with "
                "--attention_sink_probe")
        if not 0 < args.router_distribution_sample_rate <= 1:
            raise ValueError(
                "--router_distribution_sample_rate must be in (0, 1]")
        if args.router_distribution_sample_cap < 1:
            raise ValueError(
                "--router_distribution_sample_cap must be >= 1")
        if args.router_distribution_histogram_bins < 4:
            raise ValueError(
                "--router_distribution_histogram_bins must be >= 4")
        if args.router_distribution_flush_interval < 1:
            raise ValueError(
                "--router_distribution_flush_interval must be >= 1")
        args.enforce_eager = True
        print("Router distribution observer requires Python-side routing hooks; "
              "automatically enabling --enforce_eager.")

    router_distribution_dir = None
    if args.collect_router_distribution:
        base_dir = args.router_distribution_dir
        if base_dir is None:
            base_dir = str(Path(args.output_dir) / ".router_distribution")
        router_distribution_dir = _make_router_distribution_dir(base_dir)
        dataset_label = _router_distribution_dataset_label(args)
        os.environ["ROUTER_DISTRIBUTION_ENABLED"] = "1"
        os.environ["ROUTER_DISTRIBUTION_DIR"] = router_distribution_dir
        os.environ["ROUTER_DISTRIBUTION_SAMPLE_RATE"] = str(
            args.router_distribution_sample_rate)
        os.environ["ROUTER_DISTRIBUTION_SAMPLE_CAP"] = str(
            args.router_distribution_sample_cap)
        os.environ["ROUTER_DISTRIBUTION_HISTOGRAM_BINS"] = str(
            args.router_distribution_histogram_bins)
        os.environ["ROUTER_DISTRIBUTION_FLUSH_INTERVAL"] = str(
            args.router_distribution_flush_interval)
        os.environ["ROUTER_DISTRIBUTION_DATASET"] = dataset_label
        os.environ["ROUTER_DISTRIBUTION_MODEL_PATH"] = args.model_path
        os.environ["ROUTER_DISTRIBUTION_HARNESS"] = args.harness
        os.environ["ROUTER_DISTRIBUTION_TASK"] = dataset_label
        os.environ["ROUTER_DISTRIBUTION_SEED"] = str(
            args.router_distribution_seed)
        from expert_pruning.vllm_patch import install_router_distribution_observer

        install_router_distribution_observer(
            output_dir=router_distribution_dir,
            sample_rate=args.router_distribution_sample_rate,
            sample_cap=args.router_distribution_sample_cap,
            histogram_bins=args.router_distribution_histogram_bins,
            flush_interval=args.router_distribution_flush_interval,
            dataset=dataset_label,
            model_path=args.model_path,
            harness=args.harness,
            task=dataset_label,
            seed=args.router_distribution_seed,
        )
        print(f"Router distribution observer: writing snapshots to "
              f"{router_distribution_dir}")

    expert_pruning_stats_dir = None
    installed_router_patch = False
    if use_local_expert_router:
        # Keep vLLM source untouched: route only the MoE expert-selection step
        # through this repository's local implementation.
        expert_pruning_stats_dir = _make_expert_pruning_stats_dir()
        os.environ["EXPERT_PRUNING_PATCH_VLLM_ROUTER"] = "1"
        os.environ["EXPERT_PRUNING_METHOD"] = args.expert_pruning_method
        os.environ["EXPERT_PRUNING_NAEE_BETA"] = str(args.naee_beta)
        os.environ["EXPERT_PRUNING_NAEE_K_MIN"] = str(args.naee_k_min)
        os.environ["EXPERT_PRUNING_DYNAMIC_ROUTING_THRESHOLD"] = str(
            args.dynamic_routing_threshold)
        os.environ["EXPERT_PRUNING_DYNAMIC_ROUTING_SCORE_SOURCE"] = (
            args.dynamic_routing_score_source)
        os.environ["EXPERT_PRUNING_DYNAMIC_ROUTING_NEXT_RANK_PENALTY"] = str(
            args.dynamic_routing_next_rank_penalty)
        os.environ[
            "EXPERT_PRUNING_LAYERWISE_DYNAMIC_BASE_THRESHOLD"] = str(
                args.layerwise_dynamic_base_threshold)
        os.environ[
            "EXPERT_PRUNING_LAYERWISE_DYNAMIC_LAYER_ALPHA"] = str(
                args.layerwise_dynamic_layer_alpha)
        os.environ["EXPERT_PRUNING_LAYERWISE_DYNAMIC_NUM_LAYERS"] = str(
            args.layerwise_dynamic_num_layers)
        os.environ["EXPERT_PRUNING_LAYERWISE_DYNAMIC_K_MIN"] = str(
            args.layerwise_dynamic_k_min)
        os.environ["EXPERT_PRUNING_BUDGET_DYNAMIC_EASY_K"] = str(
            args.budget_dynamic_easy_k)
        os.environ["EXPERT_PRUNING_BUDGET_DYNAMIC_BASE_K"] = str(
            args.budget_dynamic_base_k)
        os.environ["EXPERT_PRUNING_BUDGET_DYNAMIC_HARD_K"] = str(
            args.budget_dynamic_hard_k)
        os.environ["EXPERT_PRUNING_BUDGET_DYNAMIC_EASY_THRESHOLD"] = str(
            args.budget_dynamic_easy_threshold)
        os.environ["EXPERT_PRUNING_BUDGET_DYNAMIC_HARD_THRESHOLD"] = str(
            args.budget_dynamic_hard_threshold)
        os.environ["EXPERT_PRUNING_BUDGET_DYNAMIC_SCORE_TOPN"] = str(
            args.budget_dynamic_score_topn)
        os.environ[
            "EXPERT_PRUNING_LAYER_BUDGET_DYNAMIC_EASY_LAYER_ALPHA"] = str(
                args.layer_budget_dynamic_easy_layer_alpha)
        os.environ[
            "EXPERT_PRUNING_LAYER_BUDGET_DYNAMIC_HARD_LAYER_ALPHA"] = str(
                args.layer_budget_dynamic_hard_layer_alpha)
        os.environ["EXPERT_PRUNING_LAYER_BUDGET_DYNAMIC_NUM_LAYERS"] = str(
            args.layer_budget_dynamic_num_layers)
        os.environ["EXPERT_PRUNING_BAND_LAYER_BUDGET_MIDDLE_START"] = str(
            args.band_layer_budget_middle_start)
        os.environ["EXPERT_PRUNING_BAND_LAYER_BUDGET_MIDDLE_END"] = str(
            args.band_layer_budget_middle_end)
        os.environ["EXPERT_PRUNING_BAND_LAYER_BUDGET_PER_LAYER_K"] = (
            args.band_layer_budget_per_layer_k)
        os.environ["EXPERT_PRUNING_BAND_LAYER_BUDGET_EASY_LAYERS"] = (
            args.band_layer_budget_easy_layers)
        os.environ["EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYERS"] = (
            args.band_layer_budget_hard_layers)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_TOKEN_GATE_RATIO"] = str(
                args.band_layer_budget_token_gate_ratio)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_EXTRA_MIDDLE_START"] = str(
                args.band_layer_budget_extra_middle_start)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_EXTRA_MIDDLE_END"] = str(
                args.band_layer_budget_extra_middle_end)
        os.environ["EXPERT_PRUNING_BAND_LAYER_BUDGET_MIDDLE_PHASE"] = (
            args.band_layer_budget_middle_phase)
        os.environ["EXPERT_PRUNING_BAND_LAYER_BUDGET_EASY_START"] = str(
            args.band_layer_budget_easy_start)
        os.environ["EXPERT_PRUNING_BAND_LAYER_BUDGET_EASY_END"] = str(
            args.band_layer_budget_easy_end)
        os.environ["EXPERT_PRUNING_BAND_LAYER_BUDGET_EASY_PHASE"] = (
            args.band_layer_budget_easy_phase)
        os.environ["EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PHASE"] = (
            args.band_layer_budget_hard_phase)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_EASY_TAIL_THRESHOLD"] = str(
                args.band_layer_budget_easy_tail_threshold)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_MIN_RANK_WEIGHT"] = str(
                args.band_layer_budget_hard_min_rank_weight)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_MAX_RANK_WEIGHT"] = str(
                args.band_layer_budget_hard_max_rank_weight)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_MIN_RANK_WEIGHT"] = str(
                args.band_layer_budget_hard_decode_min_rank_weight)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_MIXED_RESCUE_MIN_RANK_WEIGHT"] = str(
                args.band_layer_budget_mixed_rescue_min_rank_weight)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_MIXED_RESCUE_MAX_CONCENTRATION"] = str(
                args.band_layer_budget_mixed_rescue_max_concentration)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_TOKENS"] = str(
                args.band_layer_budget_hard_prefill_max_tokens)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_TOKENS"] = str(
                args.band_layer_budget_hard_prefill_max_segment_tokens)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_DENSITY"] = str(
                args.band_layer_budget_hard_prefill_max_density)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MIN_DENSITY"] = str(
                args.band_layer_budget_hard_prefill_min_density)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_SEGMENT_HARD_RATIO"] = str(
                args.band_layer_budget_hard_prefill_max_segment_hard_ratio)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_SEGMENT_CAP_SCORE"] = str(
                args.band_layer_budget_hard_prefill_segment_cap_score)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_SEGMENT_HARD_RATIO"] = str(
                args.band_layer_budget_hard_prefill_conditional_segment_hard_ratio)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_DENSITY"] = str(
                args.band_layer_budget_hard_prefill_conditional_min_density)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_MIN_SEQ_LEN"] = str(
                args.band_layer_budget_hard_prefill_conditional_min_seq_len)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_IDS"] = str(
                args.band_layer_budget_hard_prefill_conditional_token_ids)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_CONDITIONAL_TOKEN_NGRAMS"] = str(
                args.band_layer_budget_hard_prefill_conditional_token_ngrams)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_SEGMENT_HARD_RATIO"] = str(
                args.band_layer_budget_hard_prefill_marker_segment_hard_ratio)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_IDS"] = str(
                args.band_layer_budget_hard_prefill_marker_token_ids)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MARKER_TOKEN_NGRAMS"] = str(
                args.band_layer_budget_hard_prefill_marker_token_ngrams)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_EXCLUDE_TOKEN_IDS"] = str(
                args.band_layer_budget_hard_prefill_exclude_token_ids)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MIN_RELATIVE_POS"] = str(
                args.band_layer_budget_hard_prefill_min_relative_pos)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_MAX_RELATIVE_POS"] = str(
                args.band_layer_budget_hard_prefill_max_relative_pos)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_START"] = str(
                args.band_layer_budget_hard_prefill_start)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_PREFILL_END"] = str(
                args.band_layer_budget_hard_prefill_end)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_MIN_SEQ_LEN"] = str(
                args.band_layer_budget_hard_decode_min_seq_len)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_MAX_SEQ_LEN"] = str(
                args.band_layer_budget_hard_decode_max_seq_len)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_SEQ_LEN_SCOPE"] = (
                args.band_layer_budget_hard_decode_seq_len_scope)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_MIN_OFFSET"] = str(
                args.band_layer_budget_hard_decode_min_offset)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_MAX_OFFSET"] = str(
                args.band_layer_budget_hard_decode_max_offset)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_START"] = str(
                args.band_layer_budget_hard_decode_start)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_END"] = str(
                args.band_layer_budget_hard_decode_end)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_DECODE_TOKEN_IDS"] = (
                args.band_layer_budget_hard_decode_token_ids)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_START"] = str(
                args.band_layer_budget_hard_layer_prior_start)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_END"] = str(
                args.band_layer_budget_hard_layer_prior_end)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_START"] = str(
                args.band_layer_budget_hard_layer_prior_extra_start)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_EXTRA_END"] = str(
                args.band_layer_budget_hard_layer_prior_extra_end)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_PRIOR_ALPHA"] = str(
                args.band_layer_budget_hard_layer_prior_alpha)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_HARD_LAYER_SENSITIVITY_ALPHA"] = str(
                args.band_layer_budget_hard_layer_sensitivity_alpha)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_LATE_CONCENTRATION_START"] = str(
                args.band_layer_budget_late_concentration_start)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_LATE_CONCENTRATION_THRESHOLD"] = str(
                args.band_layer_budget_late_concentration_threshold)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_LATE_RESCUE_START"] = str(
                args.band_layer_budget_late_rescue_start)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_LATE_RESCUE_CONCENTRATION_THRESHOLD"] = str(
                args.band_layer_budget_late_rescue_concentration_threshold)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_LATE_RESCUE_MIXED_MAX"] = str(
                args.band_layer_budget_late_rescue_mixed_max)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_TOKEN_NGRAMS"] = (
                args.band_layer_budget_prompt_profile1_token_ngrams)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_THRESHOLD"] = str(
                args.band_layer_budget_prompt_profile1_hard_threshold)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_HARD_MIN_RANK_WEIGHT"] = str(
                args.band_layer_budget_prompt_profile1_hard_min_rank_weight)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_START"] = str(
                args.band_layer_budget_prompt_profile1_middle_start)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_MIDDLE_END"] = str(
                args.band_layer_budget_prompt_profile1_middle_end)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE1_PREFILL_SEGMENT_HARD_RATIO"] = str(
                args.band_layer_budget_prompt_profile1_prefill_segment_hard_ratio)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_TOKEN_NGRAMS"] = (
                args.band_layer_budget_prompt_profile2_token_ngrams)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_THRESHOLD"] = str(
                args.band_layer_budget_prompt_profile2_hard_threshold)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_HARD_MIN_RANK_WEIGHT"] = str(
                args.band_layer_budget_prompt_profile2_hard_min_rank_weight)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_START"] = str(
                args.band_layer_budget_prompt_profile2_middle_start)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_MIDDLE_END"] = str(
                args.band_layer_budget_prompt_profile2_middle_end)
        os.environ[
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_PROMPT_PROFILE2_PREFILL_SEGMENT_HARD_RATIO"] = str(
                args.band_layer_budget_prompt_profile2_prefill_segment_hard_ratio)
        if args.diep_artifact_path:
            os.environ["EXPERT_PRUNING_DIEP_ARTIFACT_PATH"] = (
                args.diep_artifact_path)
        os.environ["EXPERT_PRUNING_DIEP_PRUNING_MODE"] = (
            args.diep_pruning_mode)
        # Only exported when set to something other than the default. The cache fingerprint
        # hashes every EXPERT_PRUNING_* variable, so exporting these unconditionally would
        # change the namespace of runs that predate the flags and throw away their cached
        # samples for no reason. Absent means the default, which is what those runs did.
        if args.diep_use_gamma1:
            os.environ["EXPERT_PRUNING_DIEP_USE_GAMMA1"] = "1"
        if args.diep_gamma_alpha != 1.0:
            os.environ["EXPERT_PRUNING_DIEP_GAMMA_ALPHA"] = str(
                args.diep_gamma_alpha)
        if args.diep_threshold_cap is not None:
            os.environ["EXPERT_PRUNING_DIEP_THRESHOLD_CAP"] = str(
                args.diep_threshold_cap)
        if args.ban_artifact_path:
            os.environ["EXPERT_PRUNING_BAN_ARTIFACT_PATH"] = (
                args.ban_artifact_path)
        os.environ["EXPERT_PRUNING_BAN_LAMBDA"] = str(args.ban_lambda)
        os.environ["EXPERT_PRUNING_BAN_K_MIN"] = str(args.ban_k_min)
        os.environ["EXPERT_PRUNING_EAC_ALPHA"] = str(args.eac_alpha)
        os.environ["EXPERT_PRUNING_BIASED_RENORM_KEEP_TOPN"] = str(
            args.biased_renorm_keep_topn)
        if args.router_value_artifact_path:
            os.environ["EXPERT_PRUNING_ROUTER_VALUE_ARTIFACT_PATH"] = (
                args.router_value_artifact_path)
        os.environ["EXPERT_PRUNING_ROUTER_VALUE_BASE_K"] = str(
            args.router_value_base_k)
        os.environ["EXPERT_PRUNING_ROUTER_VALUE_QUOTA_RATIO"] = str(
            args.router_value_quota_ratio)
        os.environ["EXPERT_PRUNING_ROUTER_VALUE_SELECTION_MODE"] = (
            args.router_value_selection_mode)
        os.environ["EXPERT_PRUNING_ROUTER_VALUE_SWAP_DIRECTION"] = (
            args.router_value_swap_direction)
        os.environ["EXPERT_PRUNING_MC_MOE_PROTECTION_RATIO"] = str(
            args.mc_moe_protection_ratio)
        os.environ["EXPERT_PRUNING_DEBUG"] = (
            "1" if args.expert_pruning_debug else "0")
        os.environ["EXPERT_PRUNING_DEBUG_MAX_PRINTS"] = str(
            args.expert_pruning_debug_max_prints)
        attention_sink_probe_enabled = (
            args.attention_sink_probe
            or args.expert_pruning_method == "MC_MoE")
        os.environ["EXPERT_PRUNING_ATTENTION_SINK_PROBE"] = (
            "1" if attention_sink_probe_enabled else "0")
        os.environ["EXPERT_PRUNING_ATTENTION_SINK_PROBE_MAX_PRINTS"] = str(
            args.attention_sink_probe_max_prints)
        os.environ[
            "EXPERT_PRUNING_ATTENTION_SINK_PROBE_MAX_QUERY_TOKENS"] = str(
                args.attention_sink_probe_max_query_tokens)
        os.environ[
            "EXPERT_PRUNING_ATTENTION_SINK_PROBE_MAX_KEY_TOKENS"] = str(
                args.attention_sink_probe_max_key_tokens)
        os.environ["EXPERT_PRUNING_ATTENTION_SINK_PROBE_TOPK"] = str(
            args.attention_sink_probe_topk)
        os.environ["EXPERT_PRUNING_STATS_DIR"] = expert_pruning_stats_dir
        os.environ["EXPERT_PRUNING_STATS_FLUSH_INTERVAL"] = "1"
        from expert_pruning import install_vllm_expert_pruning_router

        installed_router_patch = install_vllm_expert_pruning_router(
            method=args.expert_pruning_method,
            naee_beta=args.naee_beta,
            naee_k_min=args.naee_k_min,
            dynamic_routing_threshold=args.dynamic_routing_threshold,
            dynamic_routing_score_source=args.dynamic_routing_score_source,
            dynamic_routing_next_rank_penalty=(
                args.dynamic_routing_next_rank_penalty),
            layerwise_dynamic_base_threshold=(
                args.layerwise_dynamic_base_threshold),
            layerwise_dynamic_layer_alpha=(
                args.layerwise_dynamic_layer_alpha),
            layerwise_dynamic_num_layers=(
                args.layerwise_dynamic_num_layers),
            layerwise_dynamic_k_min=args.layerwise_dynamic_k_min,
            budget_dynamic_easy_k=args.budget_dynamic_easy_k,
            budget_dynamic_base_k=args.budget_dynamic_base_k,
            budget_dynamic_hard_k=args.budget_dynamic_hard_k,
            budget_dynamic_easy_threshold=(
                args.budget_dynamic_easy_threshold),
            budget_dynamic_hard_threshold=(
                args.budget_dynamic_hard_threshold),
            budget_dynamic_score_topn=args.budget_dynamic_score_topn,
            layer_budget_dynamic_easy_layer_alpha=(
                args.layer_budget_dynamic_easy_layer_alpha),
            layer_budget_dynamic_hard_layer_alpha=(
                args.layer_budget_dynamic_hard_layer_alpha),
            layer_budget_dynamic_num_layers=(
                args.layer_budget_dynamic_num_layers),
            band_layer_budget_middle_start=(
                args.band_layer_budget_middle_start),
            band_layer_budget_middle_end=args.band_layer_budget_middle_end,
            band_layer_budget_per_layer_k=(
                args.band_layer_budget_per_layer_k),
            band_layer_budget_easy_layers=(
                args.band_layer_budget_easy_layers),
            band_layer_budget_hard_layers=(
                args.band_layer_budget_hard_layers),
            band_layer_budget_token_gate_ratio=(
                args.band_layer_budget_token_gate_ratio),
            band_layer_budget_extra_middle_start=(
                args.band_layer_budget_extra_middle_start),
            band_layer_budget_extra_middle_end=(
                args.band_layer_budget_extra_middle_end),
            band_layer_budget_middle_phase=(
                args.band_layer_budget_middle_phase),
            band_layer_budget_easy_start=args.band_layer_budget_easy_start,
            band_layer_budget_easy_end=args.band_layer_budget_easy_end,
            band_layer_budget_easy_phase=args.band_layer_budget_easy_phase,
            band_layer_budget_hard_phase=args.band_layer_budget_hard_phase,
            band_layer_budget_easy_tail_threshold=(
                args.band_layer_budget_easy_tail_threshold),
            band_layer_budget_hard_min_rank_weight=(
                args.band_layer_budget_hard_min_rank_weight),
            band_layer_budget_hard_max_rank_weight=(
                args.band_layer_budget_hard_max_rank_weight),
            band_layer_budget_hard_decode_min_rank_weight=(
                args.band_layer_budget_hard_decode_min_rank_weight),
            band_layer_budget_mixed_rescue_min_rank_weight=(
                args.band_layer_budget_mixed_rescue_min_rank_weight),
            band_layer_budget_mixed_rescue_max_concentration=(
                args.band_layer_budget_mixed_rescue_max_concentration),
            band_layer_budget_hard_prefill_max_tokens=(
                args.band_layer_budget_hard_prefill_max_tokens),
            band_layer_budget_hard_prefill_max_segment_tokens=(
                args.band_layer_budget_hard_prefill_max_segment_tokens),
            band_layer_budget_hard_prefill_max_density=(
                args.band_layer_budget_hard_prefill_max_density),
            band_layer_budget_hard_prefill_min_density=(
                args.band_layer_budget_hard_prefill_min_density),
            band_layer_budget_hard_prefill_max_segment_hard_ratio=(
                args.band_layer_budget_hard_prefill_max_segment_hard_ratio),
            band_layer_budget_hard_prefill_segment_cap_score=(
                args.band_layer_budget_hard_prefill_segment_cap_score),
            band_layer_budget_hard_prefill_conditional_segment_hard_ratio=(
                args.band_layer_budget_hard_prefill_conditional_segment_hard_ratio),
            band_layer_budget_hard_prefill_conditional_min_density=(
                args.band_layer_budget_hard_prefill_conditional_min_density),
            band_layer_budget_hard_prefill_conditional_min_seq_len=(
                args.band_layer_budget_hard_prefill_conditional_min_seq_len),
            band_layer_budget_hard_prefill_conditional_token_ids=(
                args.band_layer_budget_hard_prefill_conditional_token_ids),
            band_layer_budget_hard_prefill_conditional_token_ngrams=(
                args.band_layer_budget_hard_prefill_conditional_token_ngrams),
            band_layer_budget_hard_prefill_marker_segment_hard_ratio=(
                args.band_layer_budget_hard_prefill_marker_segment_hard_ratio),
            band_layer_budget_hard_prefill_marker_token_ids=(
                args.band_layer_budget_hard_prefill_marker_token_ids),
            band_layer_budget_hard_prefill_marker_token_ngrams=(
                args.band_layer_budget_hard_prefill_marker_token_ngrams),
            band_layer_budget_hard_prefill_exclude_token_ids=(
                args.band_layer_budget_hard_prefill_exclude_token_ids),
            band_layer_budget_hard_prefill_min_relative_pos=(
                args.band_layer_budget_hard_prefill_min_relative_pos),
            band_layer_budget_hard_prefill_max_relative_pos=(
                args.band_layer_budget_hard_prefill_max_relative_pos),
            band_layer_budget_hard_prefill_start=(
                args.band_layer_budget_hard_prefill_start),
            band_layer_budget_hard_prefill_end=(
                args.band_layer_budget_hard_prefill_end),
            band_layer_budget_hard_decode_min_seq_len=(
                args.band_layer_budget_hard_decode_min_seq_len),
            band_layer_budget_hard_decode_max_seq_len=(
                args.band_layer_budget_hard_decode_max_seq_len),
            band_layer_budget_hard_decode_seq_len_scope=(
                args.band_layer_budget_hard_decode_seq_len_scope),
            band_layer_budget_hard_decode_min_offset=(
                args.band_layer_budget_hard_decode_min_offset),
            band_layer_budget_hard_decode_max_offset=(
                args.band_layer_budget_hard_decode_max_offset),
            band_layer_budget_hard_decode_start=(
                args.band_layer_budget_hard_decode_start),
            band_layer_budget_hard_decode_end=(
                args.band_layer_budget_hard_decode_end),
            band_layer_budget_hard_decode_token_ids=(
                args.band_layer_budget_hard_decode_token_ids),
            band_layer_budget_hard_layer_prior_start=(
                args.band_layer_budget_hard_layer_prior_start),
            band_layer_budget_hard_layer_prior_end=(
                args.band_layer_budget_hard_layer_prior_end),
            band_layer_budget_hard_layer_prior_extra_start=(
                args.band_layer_budget_hard_layer_prior_extra_start),
            band_layer_budget_hard_layer_prior_extra_end=(
                args.band_layer_budget_hard_layer_prior_extra_end),
            band_layer_budget_hard_layer_prior_alpha=(
                args.band_layer_budget_hard_layer_prior_alpha),
            band_layer_budget_hard_layer_sensitivity_alpha=(
                args.band_layer_budget_hard_layer_sensitivity_alpha),
            band_layer_budget_late_concentration_start=(
                args.band_layer_budget_late_concentration_start),
            band_layer_budget_late_concentration_threshold=(
                args.band_layer_budget_late_concentration_threshold),
            band_layer_budget_late_rescue_start=(
                args.band_layer_budget_late_rescue_start),
            band_layer_budget_late_rescue_concentration_threshold=(
                args.band_layer_budget_late_rescue_concentration_threshold),
            band_layer_budget_late_rescue_mixed_max=(
                args.band_layer_budget_late_rescue_mixed_max),
            band_layer_budget_prompt_profile1_token_ngrams=(
                args.band_layer_budget_prompt_profile1_token_ngrams),
            band_layer_budget_prompt_profile1_hard_threshold=(
                args.band_layer_budget_prompt_profile1_hard_threshold),
            band_layer_budget_prompt_profile1_hard_min_rank_weight=(
                args.band_layer_budget_prompt_profile1_hard_min_rank_weight),
            band_layer_budget_prompt_profile1_middle_start=(
                args.band_layer_budget_prompt_profile1_middle_start),
            band_layer_budget_prompt_profile1_middle_end=(
                args.band_layer_budget_prompt_profile1_middle_end),
            band_layer_budget_prompt_profile1_prefill_segment_hard_ratio=(
                args.band_layer_budget_prompt_profile1_prefill_segment_hard_ratio),
            band_layer_budget_prompt_profile2_token_ngrams=(
                args.band_layer_budget_prompt_profile2_token_ngrams),
            band_layer_budget_prompt_profile2_hard_threshold=(
                args.band_layer_budget_prompt_profile2_hard_threshold),
            band_layer_budget_prompt_profile2_hard_min_rank_weight=(
                args.band_layer_budget_prompt_profile2_hard_min_rank_weight),
            band_layer_budget_prompt_profile2_middle_start=(
                args.band_layer_budget_prompt_profile2_middle_start),
            band_layer_budget_prompt_profile2_middle_end=(
                args.band_layer_budget_prompt_profile2_middle_end),
            band_layer_budget_prompt_profile2_prefill_segment_hard_ratio=(
                args.band_layer_budget_prompt_profile2_prefill_segment_hard_ratio),
            diep_artifact_path=args.diep_artifact_path,
            diep_pruning_mode=args.diep_pruning_mode,
            diep_use_gamma1=args.diep_use_gamma1,
            diep_gamma_alpha=args.diep_gamma_alpha,
            diep_threshold_cap=args.diep_threshold_cap,
            ban_artifact_path=args.ban_artifact_path,
            ban_lambda=args.ban_lambda,
            ban_k_min=args.ban_k_min,
            eac_alpha=args.eac_alpha,
            biased_renorm_keep_topn=args.biased_renorm_keep_topn,
            router_value_artifact_path=args.router_value_artifact_path,
            router_value_base_k=args.router_value_base_k,
            router_value_quota_ratio=args.router_value_quota_ratio,
            router_value_selection_mode=args.router_value_selection_mode,
            router_value_swap_direction=args.router_value_swap_direction,
            mc_moe_protection_ratio=args.mc_moe_protection_ratio,
            debug=args.expert_pruning_debug,
            debug_max_prints=args.expert_pruning_debug_max_prints,
            attention_sink_probe=attention_sink_probe_enabled,
            attention_sink_probe_max_prints=(
                args.attention_sink_probe_max_prints),
            attention_sink_probe_max_query_tokens=(
                args.attention_sink_probe_max_query_tokens),
            attention_sink_probe_max_key_tokens=(
                args.attention_sink_probe_max_key_tokens),
            attention_sink_probe_topk=args.attention_sink_probe_topk,
        )
        router_msg = (
            "MoE expert router: using local expert_pruning.routing.torch_fused_topk"
        )
        if args.expert_pruning_method == "NAEE":
            router_msg += (
                f" with NAEE(beta={args.naee_beta}, "
                f"k_min={args.naee_k_min})"
            )
        elif args.expert_pruning_method == "DiEP":
            router_msg += (
                f" with DiEP(beta={args.naee_beta}, "
                f"k_min={args.naee_k_min}, "
                f"mode={args.diep_pruning_mode}, "
                f"gamma_1={'on' if args.diep_use_gamma1 else 'off'}, "
                f"gamma_alpha={args.diep_gamma_alpha}, "
                f"cap={args.diep_threshold_cap if args.diep_threshold_cap is not None else 'none'}, "
                f"artifact={args.diep_artifact_path})")
        elif args.expert_pruning_method == "Dynamic_Routing":
            router_msg += (
                " with Dynamic_Routing"
                f"(threshold={args.dynamic_routing_threshold}, "
                f"score_source={args.dynamic_routing_score_source})"
            )
        elif args.expert_pruning_method == "LayerWise_Dynamic_Routing":
            router_msg += (
                " with LayerWise_Dynamic_Routing"
                f"(base={args.layerwise_dynamic_base_threshold}, "
                f"alpha={args.layerwise_dynamic_layer_alpha}, "
                f"num_layers={args.layerwise_dynamic_num_layers}, "
                f"k_min={args.layerwise_dynamic_k_min}, "
                f"score_source={args.dynamic_routing_score_source})")
        elif args.expert_pruning_method == "Budget_Dynamic_Routing":
            router_msg += (
                " with Budget_Dynamic_Routing"
                f"(easy_k={args.budget_dynamic_easy_k}, "
                f"base_k={args.budget_dynamic_base_k}, "
                f"hard_k={args.budget_dynamic_hard_k}, "
                f"easy_threshold={args.budget_dynamic_easy_threshold}, "
                f"hard_threshold={args.budget_dynamic_hard_threshold}, "
                f"score_topn={args.budget_dynamic_score_topn}, "
                f"score_source={args.dynamic_routing_score_source})")
        elif args.expert_pruning_method == "LayerBudget_Dynamic_Routing":
            router_msg += (
                " with LayerBudget_Dynamic_Routing"
                f"(easy_k={args.budget_dynamic_easy_k}, "
                f"base_k={args.budget_dynamic_base_k}, "
                f"hard_k={args.budget_dynamic_hard_k}, "
                f"easy_threshold={args.budget_dynamic_easy_threshold}, "
                f"hard_threshold={args.budget_dynamic_hard_threshold}, "
                f"score_topn={args.budget_dynamic_score_topn}, "
                f"easy_layer_alpha="
                f"{args.layer_budget_dynamic_easy_layer_alpha}, "
                f"hard_layer_alpha="
                f"{args.layer_budget_dynamic_hard_layer_alpha}, "
                f"num_layers={args.layer_budget_dynamic_num_layers}, "
                f"score_source={args.dynamic_routing_score_source})")
        elif args.expert_pruning_method == "BandedLayerBudget_Dynamic_Routing":
            router_msg += (
                " with BandedLayerBudget_Dynamic_Routing"
                f"(easy_k={args.budget_dynamic_easy_k}, "
                f"base_k={args.budget_dynamic_base_k}, "
                f"hard_k={args.budget_dynamic_hard_k}, "
                f"easy_threshold={args.budget_dynamic_easy_threshold}, "
                f"hard_threshold={args.budget_dynamic_hard_threshold}, "
                f"middle_start={args.band_layer_budget_middle_start}, "
                f"middle_end={args.band_layer_budget_middle_end}, "
                f"per_layer_k={args.band_layer_budget_per_layer_k}, "
                f"easy_layers={args.band_layer_budget_easy_layers}, "
                f"hard_layers={args.band_layer_budget_hard_layers}, "
                f"token_gate_ratio="
                f"{args.band_layer_budget_token_gate_ratio}, "
                f"extra_middle_start="
                f"{args.band_layer_budget_extra_middle_start}, "
                f"extra_middle_end="
                f"{args.band_layer_budget_extra_middle_end}, "
                f"middle_phase={args.band_layer_budget_middle_phase}, "
                f"easy_start={args.band_layer_budget_easy_start}, "
                f"easy_end={args.band_layer_budget_easy_end}, "
                f"easy_phase={args.band_layer_budget_easy_phase}, "
                f"hard_phase={args.band_layer_budget_hard_phase}, "
                f"easy_tail_threshold="
                f"{args.band_layer_budget_easy_tail_threshold}, "
                f"hard_min_rank_weight="
                f"{args.band_layer_budget_hard_min_rank_weight}, "
                f"hard_max_rank_weight="
                f"{args.band_layer_budget_hard_max_rank_weight}, "
                f"hard_decode_min_rank_weight="
                f"{args.band_layer_budget_hard_decode_min_rank_weight}, "
                f"mixed_rescue_min_rank_weight="
                f"{args.band_layer_budget_mixed_rescue_min_rank_weight}, "
                f"mixed_rescue_max_concentration="
                f"{args.band_layer_budget_mixed_rescue_max_concentration}, "
                f"hard_prefill_max_tokens="
                f"{args.band_layer_budget_hard_prefill_max_tokens}, "
                f"hard_prefill_max_segment_tokens="
                f"{args.band_layer_budget_hard_prefill_max_segment_tokens}, "
                f"hard_prefill_max_density="
                f"{args.band_layer_budget_hard_prefill_max_density}, "
                f"hard_prefill_min_density="
                f"{args.band_layer_budget_hard_prefill_min_density}, "
                f"hard_prefill_max_segment_hard_ratio="
                f"{args.band_layer_budget_hard_prefill_max_segment_hard_ratio}, "
                f"hard_prefill_segment_cap_score="
                f"{args.band_layer_budget_hard_prefill_segment_cap_score}, "
                f"hard_prefill_conditional_segment_hard_ratio="
                f"{args.band_layer_budget_hard_prefill_conditional_segment_hard_ratio}, "
                f"hard_prefill_conditional_min_density="
                f"{args.band_layer_budget_hard_prefill_conditional_min_density}, "
                f"hard_prefill_conditional_min_seq_len="
                f"{args.band_layer_budget_hard_prefill_conditional_min_seq_len}, "
                f"hard_prefill_conditional_token_ids="
                f"{args.band_layer_budget_hard_prefill_conditional_token_ids}, "
                f"hard_prefill_conditional_token_ngrams="
                f"{args.band_layer_budget_hard_prefill_conditional_token_ngrams}, "
                f"hard_prefill_marker_segment_hard_ratio="
                f"{args.band_layer_budget_hard_prefill_marker_segment_hard_ratio}, "
                f"hard_prefill_marker_token_ids="
                f"{args.band_layer_budget_hard_prefill_marker_token_ids}, "
                f"hard_prefill_marker_token_ngrams="
                f"{args.band_layer_budget_hard_prefill_marker_token_ngrams}, "
                f"hard_prefill_exclude_token_ids="
                f"{args.band_layer_budget_hard_prefill_exclude_token_ids}, "
                f"hard_prefill_min_relative_pos="
                f"{args.band_layer_budget_hard_prefill_min_relative_pos}, "
                f"hard_prefill_max_relative_pos="
                f"{args.band_layer_budget_hard_prefill_max_relative_pos}, "
                f"hard_prefill_start="
                f"{args.band_layer_budget_hard_prefill_start}, "
                f"hard_prefill_end="
                f"{args.band_layer_budget_hard_prefill_end}, "
                f"hard_decode_min_seq_len="
                f"{args.band_layer_budget_hard_decode_min_seq_len}, "
                f"hard_decode_max_seq_len="
                f"{args.band_layer_budget_hard_decode_max_seq_len}, "
                f"hard_decode_seq_len_scope="
                f"{args.band_layer_budget_hard_decode_seq_len_scope}, "
                f"hard_decode_min_offset="
                f"{args.band_layer_budget_hard_decode_min_offset}, "
                f"hard_decode_max_offset="
                f"{args.band_layer_budget_hard_decode_max_offset}, "
                f"hard_decode_start="
                f"{args.band_layer_budget_hard_decode_start}, "
                f"hard_decode_end="
                f"{args.band_layer_budget_hard_decode_end}, "
                f"hard_layer_prior_start="
                f"{args.band_layer_budget_hard_layer_prior_start}, "
                f"hard_layer_prior_end="
                f"{args.band_layer_budget_hard_layer_prior_end}, "
                f"hard_layer_prior_extra_start="
                f"{args.band_layer_budget_hard_layer_prior_extra_start}, "
                f"hard_layer_prior_extra_end="
                f"{args.band_layer_budget_hard_layer_prior_extra_end}, "
                f"hard_layer_prior_alpha="
                f"{args.band_layer_budget_hard_layer_prior_alpha}, "
                f"hard_layer_sensitivity_alpha="
                f"{args.band_layer_budget_hard_layer_sensitivity_alpha}, "
                f"late_concentration_start="
                f"{args.band_layer_budget_late_concentration_start}, "
                f"late_concentration_threshold="
                f"{args.band_layer_budget_late_concentration_threshold}, "
                f"late_rescue_start="
                f"{args.band_layer_budget_late_rescue_start}, "
                f"late_rescue_concentration_threshold="
                f"{args.band_layer_budget_late_rescue_concentration_threshold}, "
                f"late_rescue_mixed_max="
                f"{args.band_layer_budget_late_rescue_mixed_max}, "
                f"profile1_ngrams="
                f"{args.band_layer_budget_prompt_profile1_token_ngrams}, "
                f"profile1_hard_threshold="
                f"{args.band_layer_budget_prompt_profile1_hard_threshold}, "
                f"profile1_hard_min_rank_weight="
                f"{args.band_layer_budget_prompt_profile1_hard_min_rank_weight}, "
                f"profile1_middle="
                f"{args.band_layer_budget_prompt_profile1_middle_start}:"
                f"{args.band_layer_budget_prompt_profile1_middle_end}, "
                f"profile1_prefill_segment_hard_ratio="
                f"{args.band_layer_budget_prompt_profile1_prefill_segment_hard_ratio}, "
                f"profile2_ngrams="
                f"{args.band_layer_budget_prompt_profile2_token_ngrams}, "
                f"profile2_hard_threshold="
                f"{args.band_layer_budget_prompt_profile2_hard_threshold}, "
                f"profile2_hard_min_rank_weight="
                f"{args.band_layer_budget_prompt_profile2_hard_min_rank_weight}, "
                f"profile2_middle="
                f"{args.band_layer_budget_prompt_profile2_middle_start}:"
                f"{args.band_layer_budget_prompt_profile2_middle_end}, "
                f"profile2_prefill_segment_hard_ratio="
                f"{args.band_layer_budget_prompt_profile2_prefill_segment_hard_ratio}, "
                f"score_topn={args.budget_dynamic_score_topn}, "
                f"score_source={args.dynamic_routing_score_source})")
        elif (args.expert_pruning_method ==
              "BanSensitiveLayerBudget_Dynamic_Routing"):
            router_msg += (
                " with BanSensitiveLayerBudget_Dynamic_Routing"
                f"(base_k={args.budget_dynamic_base_k}, "
                f"hard_k={args.budget_dynamic_hard_k}, "
                f"layer_sensitivity_threshold="
                f"{args.budget_dynamic_hard_threshold}, "
                f"artifact={args.ban_artifact_path})")
        elif args.expert_pruning_method == "MC_MoE":
            router_msg += (
                f" with MC_MoE(beta={args.naee_beta}, "
                f"k_min={args.naee_k_min}, "
                f"protection_ratio={args.mc_moe_protection_ratio})")
        elif args.expert_pruning_method == "EAC_MoE":
            router_msg += f" with EAC_MoE(alpha={args.eac_alpha})"
        elif args.expert_pruning_method == "Ban":
            router_msg += (
                f" with Ban(lambda={args.ban_lambda}, "
                f"k_min={args.ban_k_min}, "
                f"artifact={args.ban_artifact_path})")
        elif args.expert_pruning_method == "TopK_Biased_Renorm":
            router_msg += (
                " with TopK_Biased_Renorm"
                f"(keep_topn={args.biased_renorm_keep_topn})")
        elif args.expert_pruning_method == "RouterValue_Dynamic_Routing":
            router_msg += (
                " with RouterValue_Dynamic_Routing"
                f"(base_k={args.router_value_base_k}, "
                f"quota_ratio={args.router_value_quota_ratio}, "
                f"selection_mode={args.router_value_selection_mode}, "
                f"swap_direction={args.router_value_swap_direction}, "
                f"artifact={args.router_value_artifact_path})")
        print(router_msg +
              (" (installed)" if installed_router_patch else
               " (already installed)"))
        if attention_sink_probe_enabled:
            attention_probe_desc = (
                "This only observes Q/K-derived sink signals; routing is unchanged."
                if args.expert_pruning_method != "MC_MoE" else
                "This supplies protected-token masks to MC_MoE routing.")
            print(
                "Attention sink probe: enabled "
                f"(max_query_tokens={args.attention_sink_probe_max_query_tokens}, "
                f"max_key_tokens={args.attention_sink_probe_max_key_tokens}, "
                f"topk={args.attention_sink_probe_topk}). "
                f"{attention_probe_desc}"
            )
    else:
        os.environ.pop("EXPERT_PRUNING_PATCH_VLLM_ROUTER", None)
        os.environ.pop("EXPERT_PRUNING_METHOD", None)
        os.environ.pop("EXPERT_PRUNING_NAEE_BETA", None)
        os.environ.pop("EXPERT_PRUNING_NAEE_K_MIN", None)
        os.environ.pop("EXPERT_PRUNING_DYNAMIC_ROUTING_THRESHOLD", None)
        os.environ.pop("EXPERT_PRUNING_DYNAMIC_ROUTING_SCORE_SOURCE", None)
        os.environ.pop(
            "EXPERT_PRUNING_DYNAMIC_ROUTING_NEXT_RANK_PENALTY", None)
        os.environ.pop(
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_MIXED_RESCUE_MIN_RANK_WEIGHT",
            None)
        os.environ.pop(
            "EXPERT_PRUNING_BAND_LAYER_BUDGET_MIXED_RESCUE_MAX_CONCENTRATION",
            None)
        os.environ.pop("EXPERT_PRUNING_LAYERWISE_DYNAMIC_BASE_THRESHOLD",
                       None)
        os.environ.pop("EXPERT_PRUNING_LAYERWISE_DYNAMIC_LAYER_ALPHA", None)
        os.environ.pop("EXPERT_PRUNING_LAYERWISE_DYNAMIC_NUM_LAYERS", None)
        os.environ.pop("EXPERT_PRUNING_LAYERWISE_DYNAMIC_K_MIN", None)
        os.environ.pop("EXPERT_PRUNING_BUDGET_DYNAMIC_EASY_K", None)
        os.environ.pop("EXPERT_PRUNING_BUDGET_DYNAMIC_BASE_K", None)
        os.environ.pop("EXPERT_PRUNING_BUDGET_DYNAMIC_HARD_K", None)
        os.environ.pop("EXPERT_PRUNING_BUDGET_DYNAMIC_EASY_THRESHOLD", None)
        os.environ.pop("EXPERT_PRUNING_BUDGET_DYNAMIC_HARD_THRESHOLD", None)
        os.environ.pop("EXPERT_PRUNING_BUDGET_DYNAMIC_SCORE_TOPN", None)
        os.environ.pop("EXPERT_PRUNING_LAYER_BUDGET_DYNAMIC_EASY_LAYER_ALPHA",
                       None)
        os.environ.pop("EXPERT_PRUNING_LAYER_BUDGET_DYNAMIC_HARD_LAYER_ALPHA",
                       None)
        os.environ.pop("EXPERT_PRUNING_LAYER_BUDGET_DYNAMIC_NUM_LAYERS", None)
        os.environ.pop("EXPERT_PRUNING_DIEP_ARTIFACT_PATH", None)
        os.environ.pop("EXPERT_PRUNING_DIEP_PRUNING_MODE", None)
        os.environ.pop("EXPERT_PRUNING_BAN_ARTIFACT_PATH", None)
        os.environ.pop("EXPERT_PRUNING_BAN_LAMBDA", None)
        os.environ.pop("EXPERT_PRUNING_BAN_K_MIN", None)
        os.environ.pop("EXPERT_PRUNING_EAC_ALPHA", None)
        os.environ.pop("EXPERT_PRUNING_BIASED_RENORM_KEEP_TOPN", None)
        os.environ.pop("EXPERT_PRUNING_ROUTER_VALUE_ARTIFACT_PATH", None)
        os.environ.pop("EXPERT_PRUNING_ROUTER_VALUE_BASE_K", None)
        os.environ.pop("EXPERT_PRUNING_ROUTER_VALUE_QUOTA_RATIO", None)
        os.environ.pop("EXPERT_PRUNING_ROUTER_VALUE_SELECTION_MODE", None)
        os.environ.pop("EXPERT_PRUNING_ROUTER_VALUE_SWAP_DIRECTION", None)
        os.environ.pop("EXPERT_PRUNING_MC_MOE_PROTECTION_RATIO", None)
        os.environ.pop("EXPERT_PRUNING_DEBUG", None)
        os.environ.pop("EXPERT_PRUNING_DEBUG_MAX_PRINTS", None)
        os.environ.pop("EXPERT_PRUNING_ATTENTION_SINK_PROBE", None)
        os.environ.pop("EXPERT_PRUNING_ATTENTION_SINK_PROBE_MAX_PRINTS", None)
        os.environ.pop("EXPERT_PRUNING_ATTENTION_SINK_PROBE_MAX_QUERY_TOKENS",
                       None)
        os.environ.pop("EXPERT_PRUNING_ATTENTION_SINK_PROBE_MAX_KEY_TOKENS",
                       None)
        os.environ.pop("EXPERT_PRUNING_ATTENTION_SINK_PROBE_TOPK", None)
        os.environ.pop("EXPERT_PRUNING_STATS_DIR", None)
        os.environ.pop("EXPERT_PRUNING_STATS_FLUSH_INTERVAL", None)
        print("MoE expert router: using vLLM default fused_topk")

    _configure_hf_dataset_cache_mode()

    # 若指定 num_experts_per_tok，为本地 MoE 模型准备覆盖目录并替换 model_path
    # 覆盖目录名里带源路径摘要以避免同名 checkpoint 互相复用，但结果目录不该被摘要污染，
    # 所以保存名单独由原始模型路径推导。
    result_name_source = args.model_path
    if args.num_experts_per_tok is not None:
        args.model_path = prepare_model_dir_with_num_experts_per_tok(
            args.model_path, args.num_experts_per_tok
        )
        print(f"MoE 已覆盖 num_experts_per_tok={args.num_experts_per_tok}，使用目录: {args.model_path}")

    # tensor_parallel_size 未设置时取当前可见 GPU 数
    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size is None:
        try:
            import torch

            tensor_parallel_size = torch.cuda.device_count()
        except Exception:
            tensor_parallel_size = 1
        if tensor_parallel_size <= 0:
            tensor_parallel_size = 1

    short_model_name = get_short_model_name(result_name_source)
    if args.num_experts_per_tok is not None:
        short_model_name = f"{short_model_name}_k{args.num_experts_per_tok}"

    if args.harness in ("lm_eval", "lmms_eval"):
        # Everything above — the pruning options, their export, the router patch, the
        # num_experts_per_tok override — is shared with the generative runs, so the router evaluated
        # here is the same one the lighteval numbers come from.
        if args.harness == "lm_eval":
            from expert_pruning.lm_eval_runner import run_lm_eval as run_harness
        else:
            from expert_pruning.lmms_eval_runner import run_lmms_eval as run_harness

        return run_harness(
            args,
            tensor_parallel_size=tensor_parallel_size,
            short_model_name=short_model_name,
            stats_dir=expert_pruning_stats_dir,
            collect_stats=_collect_expert_pruning_stats,
            use_local_expert_router=use_local_expert_router,
            installed_router_patch=installed_router_patch,
            cache_fingerprint=_expert_pruning_cache_fingerprint(),
            router_distribution_dir=router_distribution_dir,
        )

    from lighteval.models.model_input import GenerationParameters
    from lighteval.models.vllm.vllm_model import VLLMModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    evaluation_tracker = CustomEvaluationTracker(
        output_dir=args.output_dir,
        short_model_name=short_model_name,
        save_details=True,
    )
    if args.custom_tasks == "":
        custom_tasks_path = None
    elif args.custom_tasks is None:
        custom_tasks_path = os.path.join(REPO_ROOT, "tasks", "custom_tasks.py")
    else:
        custom_tasks_path = args.custom_tasks
    kwargs = dict(
        launcher_type=ParallelismManager.VLLM,
        max_samples=args.max_samples,
        sample_shard_id=args.sample_shard_id,
        sample_num_shards=args.sample_num_shards,
        load_tasks_multilingual=args.load_multilingual_tasks,
    )
    if custom_tasks_path and os.path.isfile(custom_tasks_path):
        kwargs["custom_tasks_directory"] = os.path.abspath(custom_tasks_path)

    # GPQA is gated on the Hub. If a local copy fetched from the authors' own
    # archive is present, read from it instead; see expert_pruning/gpqa_local.py.
    from expert_pruning.gpqa_local import use_local_gpqa
    use_local_gpqa()

    pipeline_params = PipelineParameters(**kwargs)

    vllm_model_kwargs = dict(
        model_name=args.model_path,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_length=args.max_model_length,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        language_model_only=args.language_model_only,
        kv_cache_dtype=args.kv_cache_dtype,
        block_size=args.block_size,
        enable_expert_parallel=args.enable_expert_parallel,
        cpu_offload_gb=args.cpu_offload_gb,
        generation_parameters=GenerationParameters(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        ),
    )
    lighteval_cache_dir = args.lighteval_cache_dir
    lighteval_cache_policy = "lighteval_default"
    expert_pruning_cache_fingerprint = None
    cache_root = lighteval_cache_dir or os.path.join(
        args.output_dir, ".lighteval_cache")
    expert_pruning_cache_fingerprint = _expert_pruning_cache_fingerprint()
    if use_local_expert_router:
        lighteval_cache_dir = os.path.join(
            cache_root,
            f"expert-router-{expert_pruning_cache_fingerprint}",
        )
        lighteval_cache_policy = "expert_router_fingerprint"
    else:
        # Baselines (including --num_experts_per_tok) must not share lighteval's global
        # cache: the task hash cannot see edits to tasks/ prompt code, so a shared cache
        # would replay generations produced by an older prompt template.
        lighteval_cache_dir = os.path.join(
            cache_root,
            f"baseline-{expert_pruning_cache_fingerprint}",
        )
        lighteval_cache_policy = "baseline_fingerprint"
    if lighteval_cache_dir:
        vllm_model_kwargs["cache_dir"] = lighteval_cache_dir
        print(
            f"lighteval 生成缓存目录: {lighteval_cache_dir} "
            f"(policy={lighteval_cache_policy})")
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch_size 须为正整数")
        vllm_model_kwargs["max_num_seqs"] = args.batch_size
    if args.seed < 0:
        raise ValueError("--seed 须为非负整数")
    vllm_model_kwargs["seed"] = args.seed
    os.environ["LIGHTEVAL_RANDOM_SEED"] = str(args.seed)
    if args.pipeline_parallel_size is not None:
        if args.pipeline_parallel_size < 1:
            raise ValueError("--pipeline_parallel_size 须为 >= 1 的整数")
        vllm_model_kwargs["pipeline_parallel_size"] = args.pipeline_parallel_size
    if args.data_parallel_size is not None:
        if args.data_parallel_size < 1:
            raise ValueError("--data_parallel_size 须为 >= 1 的整数")
        vllm_model_kwargs["data_parallel_size"] = args.data_parallel_size
    model_config = VLLMModelConfig(**vllm_model_kwargs)

    pipeline = Pipeline(
        tasks=args.datasets,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    if args.sample_num_shards is not None:
        task_indices = {
            task_name: [int(doc.id) for doc in docs]
            for task_name, docs in pipeline.documents_dict.items()
        }
        evaluation_tracker.set_sample_shard_provenance(
            {
                "method": "original_doc_index_modulo",
                "sample_shard_id": args.sample_shard_id,
                "sample_num_shards": args.sample_num_shards,
                "task_original_indices": task_indices,
                "task_sample_counts": {
                    task_name: len(indices)
                    for task_name, indices in task_indices.items()
                },
            }
        )

    par = f"TP={tensor_parallel_size}"
    if args.pipeline_parallel_size is not None:
        par += f", PP={args.pipeline_parallel_size}"
    if args.data_parallel_size is not None:
        par += f", DP={args.data_parallel_size}"
    print(
        f"使用 vLLM 后端评估: datasets={args.datasets}, model={args.model_path} "
        f"(保存名: {short_model_name}), 参数并行: {par}"
    )
    pipeline.evaluate()
    pipeline.show_results()
    results = pipeline.get_results()
    details = pipeline.get_details()
    expert_pruning_stats = _collect_expert_pruning_stats(
        expert_pruning_stats_dir)
    if args.allow_compiled_router and expert_pruning_stats:
        # Some counting does survive the compiled region, but not honestly: vLLM pads every batch
        # up to a captured graph size and the padding rows are counted as tokens, so the average
        # reads low — 1.954 against the 2.028 the same configuration measures in eager mode on
        # gpt-oss. The number is kept for reference under its own name, and the measurement this
        # run is placed by has to come from the eager probe instead of from here.
        padded = expert_pruning_stats.pop("average_selected_experts", None)
        if padded is not None:
            expert_pruning_stats["average_selected_experts_padded"] = padded
            expert_pruning_stats["average_selected_experts"] = None
            expert_pruning_stats["statistics_note"] = (
                "ran with CUDA graphs (--allow_compiled_router); the average here would count "
                "CUDA-graph padding rows as tokens, so it is reported separately and the "
                "operating point comes from a --enforce_eager probe of this configuration")
    if (expert_pruning_stats is None and use_local_expert_router
            and args.expert_pruning_method != "none"):
        from lighteval.utils.cache_management import cache_usage_counters

        generated = cache_usage_counters().get("processed", 0)
        if generated and args.allow_compiled_router:
            print("No router statistics, which is expected under CUDA graphs; the average "
                  "comes from the eager probe of this configuration.")
            expert_pruning_stats = {
                "method": args.expert_pruning_method,
                "router_invoked": True,
                "average_selected_experts": None,
                "total_routed_tokens": 0,
            }
        elif generated:
            raise RuntimeError(
                f"The expert router reported no statistics although the model generated "
                f"{generated} samples, so average_selected_experts is unavailable and "
                f"this result cannot be placed on the expert-count axis. The routing "
                f"itself still applied; what got lost is the measurement, which happens "
                f"when the model's expert selection is inlined into the compiled region "
                f"and torch.compile traces the counters away. Re-run with "
                f"--enforce_eager.")
    if expert_pruning_stats is None and use_local_expert_router:
        # A run whose generations all came from cache never invokes the router, so no
        # stats exist. Still record what configuration this output belongs to, otherwise
        # a stale-cache replay is indistinguishable from a real experiment afterwards.
        expert_pruning_stats = {
            "method": args.expert_pruning_method,
            "router_invoked": False,
            "total_routed_tokens": 0,
        }
    if expert_pruning_stats is not None:
        expert_pruning_stats.setdefault("router_invoked", True)
        expert_pruning_stats["router_patch_installed"] = installed_router_patch
        expert_pruning_stats["lighteval_cache_dir"] = lighteval_cache_dir
        expert_pruning_stats["lighteval_cache_policy"] = (
            lighteval_cache_policy)
        expert_pruning_stats["expert_pruning_cache_fingerprint"] = (
            expert_pruning_cache_fingerprint)
        evaluation_tracker.set_expert_pruning_stats(expert_pruning_stats)
        results["expert_pruning"] = expert_pruning_stats
        if isinstance(getattr(evaluation_tracker._tracker, "results", None),
                      dict):
            evaluation_tracker._tracker.results[
                "expert_pruning"] = expert_pruning_stats
    if (expert_pruning_stats is not None
            and expert_pruning_stats.get("router_invoked")
            and expert_pruning_stats.get("average_selected_experts") is None):
        print("平均专家选择数: 本 run 未测量（"
              f"{expert_pruning_stats.get('statistics_note', '无统计')}）"
              + (f"，仅供参考的含 padding 读数: "
                 f"{expert_pruning_stats['average_selected_experts_padded']:.6f}"
                 if expert_pruning_stats.get("average_selected_experts_padded") else ""))
    elif expert_pruning_stats is not None and expert_pruning_stats.get(
            "router_invoked"):
        print(
            "平均专家选择数: "
            f"{expert_pruning_stats['average_selected_experts']:.6f} "
            f"(method={expert_pruning_stats.get('method')}, "
            f"topk={expert_pruning_stats.get('topk')}, "
            f"routed_tokens={expert_pruning_stats['total_routed_tokens']})")
        if expert_pruning_stats.get("total_protected_tokens", 0):
            print(
                "保护 token 比例: "
                f"{expert_pruning_stats['protected_token_ratio']:.6f} "
                f"(protected_tokens="
                f"{expert_pruning_stats['total_protected_tokens']})")
        if expert_pruning_stats.get("method") == "eac_moe":
            print(
                "EAC prefill剪枝比例: "
                f"{expert_pruning_stats['eac_pruned_selection_ratio']:.6f} "
                f"(prefill_tokens="
                f"{expert_pruning_stats['eac_prefill_tokens']}, "
                f"segments={expert_pruning_stats['eac_segments']}, "
                f"pruned_selections="
                f"{expert_pruning_stats['eac_pruned_selections']})")
        if expert_pruning_stats.get("method") == "ban":
            print(
                "Ban 平均动态K: "
                f"{expert_pruning_stats['ban_average_dynamic_k']:.6f} "
                f"(lambda={expert_pruning_stats.get('ban_lambda')}, "
                f"k_min={expert_pruning_stats.get('ban_k_min')})")
    elif use_local_expert_router:
        print("平均专家选择数: 未收集到本地 expert router 统计 "
              f"(router_patch_installed={installed_router_patch})；"
              "本次所有生成均来自缓存，或 router 未被调用。"
              f"缓存目录: {lighteval_cache_dir}")

    # 保存前用简短模型名，使 results 里的 config 也一致
    evaluation_tracker.general_config_logger.model_name = short_model_name
    pipeline.save_and_push_results()

    print(f"评估完成。结果与 details 已保存至: {args.output_dir}/{short_model_name}/")
    return results, details


if __name__ == "__main__":
    main()
