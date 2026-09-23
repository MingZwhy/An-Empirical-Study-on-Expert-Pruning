# Legacy environment: `qwen35` (vLLM 0.17.0)

This environment has been superseded by `install_fixedk.sh` (vLLM 0.28.0), which
covers every model this one did plus the newer MoE architectures.

It is kept here for provenance: the published **multimodal** numbers
(`--harness lmms_eval`, Qwen3-VL-30B-A3B) were produced in this environment,
and they have not been re-run on vLLM 0.28. If you need to reproduce those
specific numbers rather than the text results, recreate this environment from
the script below.

```bash
#!/usr/bin/env bash
# 在仓库根目录执行: bash install_qwen35.sh
#
# ------------------------------------------------------------------------------
# 用途
# ------------------------------------------------------------------------------
# 本脚本面向**"需要新版 vLLM 才能跑的模型"**这一条副线，现在有两类用途：
#
#   (a) Qwen3.5 基线评测：Qwen3.5-35B-A3B 这类新 MoE 架构，v0.10.2 起不来。
#   (b) VL 多模态评测（--harness lmms_eval）：Qwen3-VL-30B-A3B 声明的
#       `Qwen3VLMoeForConditionalGeneration` 不在 expertpruning 那条线所钉的 vllm
#       （01efc7e，2025-09-15）的模型表里——模型比那个 pin 还新。0.17.0 里
#       `qwen3_vl_moe.py` 是现成的，所以 VL 归到这条线，而不是去动那个正跑着
#       整个 sweep 的环境。
#
# 两类用途共用一个 conda env（`qwen35`），因为它们要的恰好是同一件事：一份够新的
# vLLM。**不修改 vllm 源码**，只把它当推理后端。因此：
#
#   - vllm 直接 `pip install vllm==0.17.0`（不做 editable，不动 3rdparty/vllm
#     子模块）。好处是让 pip 自动把匹配的 `torch==2.10.0+cu128` 一并解出来，
#     完全规避了 editable + `VLLM_USE_PRECOMPILED=1` 那一套 "去 wheels.vllm.ai
#     拉 rolling dev wheel 导致 C++ ABI 跟 stable torch 对不上" 的问题
#     （见 docs/vllm_install_notes.md 方案 C）。
#   - 要跑 Qwen3.5 必须是 vLLM >=0.16 左右（新 MoE 架构）；v0.10.2 跑不了，
#     所以这边需要 0.17.0。
#   - lighteval 仍然用本仓库 fork 的 `3rdparty/lighteval`（editable），因为
#     我们加的 `enforce_eager / cpu_offload_gb` 字段、`TokensPrompt` 适配、
#     `is_package_available` 放宽版本校验这些补丁在 0.17.0 上同样需要（尤其
#     是 "放宽版本校验"：lighteval v0.13.0 的 pyproject 要求 `vllm<0.10.2`，
#     不打这个补丁它会拒绝把 0.17.0 当作可用的 vllm）。
#
# 与 `install_expertpruning.sh` 是**完全独立的两条路径**，对应两个互不干扰的
# conda env：`expertpruning`（可改 vllm 源码、跑文本 sweep）和 `qwen35`（新版 vLLM，
# 跑 baseline 与 VL 多模态）。两个脚本的区别详见 README.md 和 docs/vllm_install_notes.md。
#
# 注意这条线上**动态剪枝还没打通**：路由补丁换的是 vllm 的 `fused_topk`，而 0.17.0 把它从
# `fused_moe/fused_moe.py` 挪到了 `fused_moe/router/fused_topk_router.py`，签名也多了一个
# `scoring_func`。所以这个环境目前能跑的是未剪枝基线和 Fixed-K（`--num_experts_per_tok`，
# 靠改 config 实现、不需要补丁）；`--use_local_expert_router` 要等补丁适配这个版本。
# ------------------------------------------------------------------------------
#
# 要求：
#   - 当前已处于 conda env "qwen35" 下（Python 3.11 建议）
#   - 仓库已 clone，且 lighteval submodule 已拉下来：
#       git submodule update --init -- 3rdparty/lighteval
#     （vllm submodule 在这条路径下**不会用到**，可以不拉；如果你已经 init 过
#     也无所谓，脚本不会去碰它。）
#
# 组件版本（与本仓库验证过的组合一致）：
#   vllm         = 0.17.0   （从 PyPI 直接装，非 editable）
#   torch        = 2.10.0+cu128  （由 pip install vllm==0.17.0 自动解出）
#   transformers = 4.57.6   （同上，vllm 0.17.0 的 Requires-Dist 恰好会把它解到这个版本）
#   lighteval    = v0.13.0  （3rdparty/lighteval 子模块, editable, 含 expert-pruning-mods 分支的补丁）
#   lmms_eval    = v0.7.2   （3rdparty/lmms-eval 子模块 tag, editable --no-deps；
#                             跑 --harness lmms_eval 的图文多模态集）
#   Python       = 3.11     （conda env `qwen35`）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# -------- 0. 基本检查 --------
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "qwen35" ]]; then
    echo "[install_qwen35.sh] 警告：当前 conda env 不是 qwen35（实际: ${CONDA_DEFAULT_ENV:-none}）。"
    echo "[install_qwen35.sh] 请先执行： conda activate qwen35"
    echo "[install_qwen35.sh] 5 秒后继续..."
    sleep 5
fi

echo "[install_qwen35.sh] Python: $(which python) $(python --version)"
echo "[install_qwen35.sh] Pip:    $(which pip)"

# -------- 1. 同步 lighteval / lmms-eval submodule --------
# vllm 子模块这条路径下用不到（这里的 vllm 从 PyPI 装），所以不 init；避免占磁盘 / 误改。
for sub in 3rdparty/lighteval 3rdparty/lmms-eval; do
    if [[ ! -d "$REPO_ROOT/$sub/.git" && ! -f "$REPO_ROOT/$sub/.git" ]]; then
        echo "[install_qwen35.sh] 初始化 submodule: $sub"
        git submodule update --init -- "$sub"
    else
        echo "[install_qwen35.sh] $sub 已初始化，跳过 submodule update"
    fi
done

# -------- 2. vLLM (从 PyPI 直接装, NOT editable) --------
# 关键点：这里**不要**加 VLLM_USE_PRECOMPILED=1，也**不要**指定
# VLLM_PRECOMPILED_WHEEL_LOCATION。让 pip 用标准依赖解析，它会：
#   (1) 从 PyPI 下载 vllm-0.17.0 的 cp3*-abi3 wheel（二进制本身）
#   (2) 按 Requires-Dist 自动选一份 ABI 对齐的 torch（2.10.0+cu128）
# 这样 vllm 的 `_C.abi3.so` 和 torch 的 `libc10.so` 天然对得上符号，
# `import vllm` 不会报 `undefined symbol: _ZN3c1013MessageLogger...`。
echo "[install_qwen35.sh] ==> 安装 vLLM (PyPI wheel, non-editable)"
pip install "vllm==0.17.0"

# -------- 3. lighteval (editable, 不带 [vllm] extras) --------
# 和 expertpruning 脚本同理：
#   - lighteval[vllm] 会拉 "vllm>=0.11" 这条约束把刚装好的 0.17.0 顶掉重装
#     （实际会解到 PyPI 最新版，很可能又触发其它兼容性问题），所以不带 extras。
#   - editable 安装是因为我们要用 3rdparty/lighteval 里本仓库打过的补丁
#     （TokensPrompt wrap、enforce_eager 字段、is_package_available 放宽等）。
echo "[install_qwen35.sh] ==> 安装 lighteval (editable, 来自 3rdparty/lighteval)"
pushd "$REPO_ROOT/3rdparty/lighteval" >/dev/null
pip install --editable .
popd >/dev/null

# -------- 4. 额外依赖 --------
echo "[install_qwen35.sh] ==> 安装额外依赖"
# more_itertools : 绝大多数情况下会被 vllm / 其它包作为传递依赖拉进来，
#                  但显式装一次更稳，不然 lighteval 的 vllm 后端跑 batch
#                  iter 的时候会 ImportError。
# langdetect     : IFEval 评测里判语言要用，PyPI 不会自动带。
# 注意：这里**不再手动钉 transformers==4.57.6**，因为 vllm 0.17.0 的
# Requires-Dist 已经把 transformers 解到 4.57.6 了（实测过）。将来如果
# vllm 换版本顺便把 transformers 解到 >=5.x，并且运行时报
# `all_special_tokens_extended` 相关错误，再回头在这里钉版本也不迟。
pip install more_itertools langdetect

# -------- 5. lmms_eval（editable --no-deps + 手工补依赖） --------
# --harness lmms_eval 那条路：图文多模态集（mmmu / mmbench / mmstar / mme-realworld / chartqa /
# textvqa / infovqa / gqa / vizwiz，各取最小的已发布版本）。执行体见
# expert_pruning/lmms_eval_runner.py，它在进程内建 vllm.LLM，所以走的是本环境这份 vLLM。
#
# 为什么不直接 `pip install -e .` 让 pip 解依赖：lmms_eval 把 wandb 钉在 0.25.0，而 wandb 要求
# protobuf<7；vLLM 这边解出来的 protobuf 是 7.x，带 constraint 装会直接解不出来。wandb 只用于
# 上报日志、且在 lmms_eval 里全是函数内的懒导入，所以本体 --no-deps 装，再手工补运行真正要用的
# 包；开发工具（black / isort / pre-commit）和 wandb 一并跳过。
#
# constraint 的作用和上一条线一样：把此刻已装的承重包钉住，pip 只许补新的、不许升级已有的
# ——尤其是 torch / transformers，它们是 vLLM wheel 的 ABI 对手方，被换掉就 import 不起来。
# 清单里带上 math-verify / latex2sympy2* 是踩过的坑：lmms_eval 要 math-verify，而全新环境里
# pip 会解到最新版，它依赖 latex2sympy2_extended 1.11.0，把 lighteval 钉死的 ==1.0.6 顶掉
# ——受影响的正是 aime / math 这些题的判分。钉住后 pip 会沿用 lighteval 装好的那一份。
# decord 和 evaluate 是额外补的：decord 不在 lmms_eval 的依赖表里但 vLLM 后端在模块顶层就
# import 它；evaluate 则是 lmms_eval.api.registry 顶层就 import，而它自己没声明（在
# expertpruning 那条线上是 lm_eval 顺带带进来的，这条线没有 lm_eval，就得自己补）。
echo "[install_qwen35.sh] ==> 安装 lmms_eval (editable --no-deps, 手工补依赖)"
LMMS_CONSTRAINTS="$(mktemp)"
python - > "$LMMS_CONSTRAINTS" <<'PY'
import importlib.metadata as md

critical = """torch transformers datasets numpy accelerate huggingface-hub tokenizers
safetensors scipy scikit-learn pandas pyarrow ray pillow fsspec aiohttp requests protobuf
sentencepiece outlines xgrammar flashinfer-python triton torchvision torchaudio vllm
math-verify latex2sympy2-extended latex2sympy2""".split()
for name in critical:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
PY
# Levenshtein is imported at module scope by mathvista's scorer, so its absence is not a scoring
# fallback but a crashed run — and only after the engine has loaded and the earlier tasks have been
# answered, which is an expensive way to find out.
pip install --constraint "$LMMS_CONSTRAINTS" \
    loguru openpyxl decord "av<16.0.0" ftfy timm pycocoevalcap zss math-verify \
    latex2sympy2 "qwen-vl-utils>=0.0.14" sentence_transformers yt-dlp evaluate Levenshtein
pushd "$REPO_ROOT/3rdparty/lmms-eval" >/dev/null
pip install --editable . --no-deps
popd >/dev/null
rm -f "$LMMS_CONSTRAINTS"

echo "[install_qwen35.sh] 完成。"
echo "[install_qwen35.sh] 冒烟测试："
echo "[install_qwen35.sh]   基线:    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --datasets winogrande --max_samples 8 --model_path <Qwen3.5 path>"
echo "[install_qwen35.sh]   多模态:  CUDA_VISIBLE_DEVICES=0,1 python main.py --harness lmms_eval --lmms_eval_tasks chartqa_lite --max_samples 8 --tensor_parallel_size 2 --model_path <VL 模型>"
```
