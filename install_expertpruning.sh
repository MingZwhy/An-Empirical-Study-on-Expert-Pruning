#!/usr/bin/env bash
# 在仓库根目录执行: bash install_expertpruning.sh
#
# ------------------------------------------------------------------------------
# 用途
# ------------------------------------------------------------------------------
# 本脚本面向 **expert-pruning 主线实验**：我们**会修改 vllm 源码**来插入 expert
# 剪枝相关的逻辑，所以必须把 vllm 以 submodule + `pip install --editable .` 的
# 方式装上，而不能走纯 pip 的 prebuilt wheel。这条路也是本仓库 README 主推的
# 默认方案（conda env: `expertpruning`）。
#
# 如果你只需要在较新的 MoE 架构上跑 Fixed-K、不改 vllm 源码，请走另一条更轻的
# 路径：`install_fixedk.sh`（发布版 vLLM wheel，非可编辑）。两个脚本对应两个
# 独立环境，互不干扰。
# ------------------------------------------------------------------------------
#
# 要求：
#   - 当前已处于 conda env "expertpruning" 下（Python 3.11 建议）
#   - 已经 clone 本仓库并拉取 submodule:
#       git submodule update --init --recursive
#
# 组件版本（与本仓库验证过的组合一致）：
#   vllm         = v0.10.2   （3rdparty/vllm 子模块 tag, editable）
#   lighteval    = v0.13.0   （带本仓库 expert-pruning-mods 分支的改动, editable）
#   lm_eval      = v0.4.13   （3rdparty/lm-evaluation-harness 子模块 tag, editable；
#                              跑 --harness lm_eval 的 0-shot 选择题，见下）
#   （lmms_eval 不在本环境：多模态那条路要新版 vLLM，见 docs/legacy_qwen35_env.md）
#   transformers = 4.57.6    （>=5.0 移除了 Qwen2Tokenizer.all_special_tokens_extended，
#                              会让 vllm.transformers_utils.tokenizer.get_cached_tokenizer 崩）
#
# 为什么不是更新的 vLLM？
#   v0.17.0 / v0.19.0 的 **editable + `VLLM_USE_PRECOMPILED=1`** 流程会去
#   wheels.vllm.ai 拉滚动 dev wheel，那个二进制跟着 torch nightly 编，引用了
#   只在更新 torch 里才有的 c10 符号（`c10::MessageLogger(c10::SourceLocation,
#   int, bool)`），而 PyPI 上 stable `torch==2.10.0` 的 libc10.so 还没有这个
#   符号，import 时就会报 `undefined symbol: _ZN3c1013MessageLogger...`。
#   见 docs/vllm_install_notes.md 里更详细的三代 torch ABI 分析，以及源码编译
#   作为备选方案。本脚本目前采用 "回退到 v0.10.2 + 锚定 PyPI wheel" 这条最稳路径。
#
# 后续如果 lighteval / vllm 更新得跟不上，可能需要调整 3rdparty/*/ 对应子
# 模块指向的 commit；总体流程不变。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# -------- 0. 基本检查 --------
# conda env 或 venv 都可以；关键是不要把十几 GB 的 CUDA wheel 装进系统解释器。
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    ENV_KIND="conda:${CONDA_DEFAULT_ENV:-unknown}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    ENV_KIND="venv:$(basename "$VIRTUAL_ENV")"
else
    cat >&2 <<'MSG'
[install_expertpruning.sh] 错误：没有激活任何 conda env 或 virtualenv。
    本脚本会以 editable 方式安装 vLLM 并拉入 torch/CUDA wheel（十几 GB），
    拒绝直接写入系统解释器。请先创建隔离环境：

      conda create -y -n expertpruning python=3.12 && conda activate expertpruning
    或
      python3.12 -m venv ~/envs/expertpruning && source ~/envs/expertpruning/bin/activate

    然后重新运行。确实需要跳过检查时设置 EP_ALLOW_SYSTEM_PYTHON=1。
MSG
    [[ "${EP_ALLOW_SYSTEM_PYTHON:-0}" == "1" ]] || exit 1
    ENV_KIND="system (forced)"
fi

PY_VERSION="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PY_VERSION" in
    3.12|3.11|3.10) ;;
    *) echo "[install_expertpruning.sh] 错误：未验证的 Python $PY_VERSION，请用 3.12 / 3.11 / 3.10。" >&2; exit 1 ;;
esac
echo "[install_expertpruning.sh] Env:    $ENV_KIND"
echo "[install_expertpruning.sh] Python: $(which python) ($PY_VERSION)"
echo "[install_expertpruning.sh] Pip:    $(which pip)"

# 这条线的引擎钉在 2025-09 的 vLLM v0.10.2，为的是让路由补丁挂得住。代价是它
# 只认识那个时间点之前的模型架构和 GPU。装完后脚本的自检会打印 torch 的
# arch_list，可以对照自己的卡；实测 v0.10.2 所带的 torch 2.8.0+cu128 的
# arch_list 里是有 sm_120 的，但论文的 sweep 跑在更早的卡上。
# 如果你的模型或显卡比这个 pin 更新，请用 install_fixedk.sh（vLLM 0.28），
# 代价是只有 Fixed-K，没有自适应规则。

# 三个索引都要能连上；提前失败比装到一半再超时容易排查。
echo "[install_expertpruning.sh] ==> 检查索引可达性"
for url in https://pypi.org/simple/ https://files.pythonhosted.org/; do
    code="$(curl -sI --max-time 20 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo 000)"
    # 只有 000（DNS/代理/防火墙打不通）才算不可达；有些索引根路径对裸 HEAD 就是
    # 返回 404，把 4xx 当失败会误报。
    if [[ "$code" == 000 ]]; then
        echo "[install_expertpruning.sh] 错误：无法访问 $url（无响应）。" >&2
        echo "[install_expertpruning.sh]        如果在代理后面，请 export https_proxy/http_proxy 再重试。" >&2
        exit 1
    fi
done

# -------- 1. 同步 submodule --------
# 只在子模块尚未拉取（目录为空）时执行 init，避免在本地做过 checkout 之后被强
# 制重置。首次 clone 时建议用 `git clone --recurse-submodules`，如果没有加这
# 个参数再由本脚本兜底。
for sub in 3rdparty/vllm 3rdparty/lighteval 3rdparty/lm-evaluation-harness; do
    if [[ ! -d "$REPO_ROOT/$sub/.git" && ! -f "$REPO_ROOT/$sub/.git" ]]; then
        echo "[install_expertpruning.sh] 初始化 submodule: $sub"
        git submodule update --init -- "$sub"
    else
        echo "[install_expertpruning.sh] $sub 已初始化，跳过 submodule update"
    fi
done

# -------- 2. vLLM（editable, 使用预编译 wheel 里的二进制） --------
# 先装 vllm；lighteval 随后不带 [vllm] extras，避免它重新从 PyPI 拉一份把我们
# 这份 editable 覆盖掉。
#
# VLLM_PRECOMPILED_WHEEL_LOCATION 的用意：默认情况下 setup.py 会去 wheels.vllm.ai
# 拉"当前 main commit" 对应的 rolling dev wheel；但这个 wheel 是**每天都在重建**
# 的——即使我们钉死了 submodule 的 tag 为 v0.10.2，setup.py 抓到的二进制也可能
# 是今天刚重编的、和我们的 torch 运行时不兼容（实测会在 `import vllm._C` 时直接
# abort 成 `std::bad_alloc`）。
# 直接把 URL 指向 PyPI 上真正发布的 v0.10.2 wheel，就能复现当时发布时的二进制
# 状态，不受后续 rolling 重编影响。详情见 docs/vllm_install_notes.md。
VLLM_PYPI_WHEEL_URL="https://files.pythonhosted.org/packages/a2/1a/365479f413e7408b314c0237d6c929569874d5c002bc7c8b5a7fbf40c7d9/vllm-0.10.2-cp38-abi3-manylinux1_x86_64.whl"

echo "[install_expertpruning.sh] ==> 安装 vLLM (editable, VLLM_USE_PRECOMPILED=1, 指向 PyPI v0.10.2 wheel)"
pushd "$REPO_ROOT/3rdparty/vllm" >/dev/null
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_PYPI_WHEEL_URL" \
    pip install --editable .
popd >/dev/null

# -------- 3. lighteval（editable, 不带 vllm extras） --------
# lighteval[vllm] 等价于同时安装 ["vllm>=0.11.0", "ray", "more_itertools"]，
# 但我们已经手动装了 vllm==0.10.2（editable），不想被 >=0.11 的约束覆盖。
# ray 已由 vllm 0.10.2 依赖；这里只手动补 more_itertools（见下一步）。
echo "[install_expertpruning.sh] ==> 安装 lighteval (editable)"
pushd "$REPO_ROOT/3rdparty/lighteval" >/dev/null
pip install --editable .
popd >/dev/null

# -------- 4. 额外依赖 --------
echo "[install_expertpruning.sh] ==> 安装额外依赖"
# more_itertools : lighteval 跑 vllm 后端时 batch iter 要用，[vllm] extras 里原本会带
# langdetect     : IFEval 的语言识别要用
# transformers   : lighteval editable 安装会把 transformers 升到 >=5；而 vllm 0.10.2
#                  的 get_cached_tokenizer 还依赖 5.x 里已经删掉的
#                  `tokenizer.all_special_tokens_extended` 属性。钉死 4.57.6
#                  （老仓库验证过的最后一个可用版本）。
pip install more_itertools langdetect
pip install "transformers==4.57.6"

# -------- 5. lm_eval（editable, 只补缺的依赖，不许升级已装的） --------
# lm_eval 提供 --harness lm_eval 那条路：arc_challenge / arc_easy / winogrande / openbookqa 这类
# 按 log-likelihood 排序打分的选择题，不生成 token，所以剪枝的影响不掺入"生成是否收得住"这个
# 混杂因素。剪枝路径与 lighteval 那条完全共用（见 expert_pruning/lm_eval_runner.py）。
#
# 放在钉死 transformers 之后、并且带 constraint：lm_eval 声明的是 transformers>=4.1、
# datasets>=2.16 这类宽约束，直接装会顺手把 transformers 升回 5.x —— 那正是上一步专门降到
# 4.57.6 要避免的崩法。这里把承重的几个包按"此刻已装版本"钉住，pip 只会去补 lm_eval 真正缺的那
# 几个纯 Python 包（实测 7 个：evaluate / numexpr / peft / pybind11 / sqlitedict /
# tqdm-multiprocess / word2number）。如果哪天 lm_eval 的要求真的和这套组合冲突，pip 会直接报解
# 不出来，而不是悄悄把环境改坏。
echo "[install_expertpruning.sh] ==> 安装 lm_eval (editable, 锁定承重依赖)"
HARNESS_CONSTRAINTS="$(mktemp)"
python - > "$HARNESS_CONSTRAINTS" <<'PY'
import importlib.metadata as md

critical = """torch transformers datasets numpy accelerate huggingface-hub tokenizers
safetensors scipy scikit-learn pandas pyarrow ray pillow fsspec aiohttp requests protobuf
sentencepiece outlines xgrammar flashinfer-python triton torchvision torchaudio
math-verify latex2sympy2-extended latex2sympy2""".split()
for name in critical:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
PY
pushd "$REPO_ROOT/3rdparty/lm-evaluation-harness" >/dev/null
pip install --editable . --constraint "$HARNESS_CONSTRAINTS"
popd >/dev/null
rm -f "$HARNESS_CONSTRAINTS"

# 多模态那条路（--harness lmms_eval）不装在这个环境里：它要的引擎是新版 vLLM（Qwen3-VL-MoE 不在
# 本脚本所钉的 01efc7e 的模型表里），而这个环境的 vllm 是 editable 且路由补丁挂在它的 fused_topk
# 上，整个 sweep 都跑在这个组合上。所以 VL 评测归到另一条线，见 docs/legacy_qwen35_env.md。
# -------- 6. 自检 --------
echo "[install_expertpruning.sh] ==> 自检"
python - <<'PY'
import importlib.metadata as md
import torch
import vllm

def ver(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "MISSING"

print(f"  vllm         {vllm.__version__}")
print(f"  torch        {torch.__version__}")
print(f"  transformers {ver('transformers')}")
print(f"  lighteval    {ver('lighteval')}")
print(f"  lm_eval      {ver('lm_eval')}")
print(f"  torch archs  {' '.join(torch.cuda.get_arch_list())}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    mine = f"sm_{cap[0]}{cap[1]}"
    covered = mine in torch.cuda.get_arch_list()
    print(f"  this GPU     {mine} {'(covered)' if covered else '(NOT in arch_list -- expect kernel errors)'}")

assert vllm.__version__.startswith("0.10.2"), f"期望 vLLM 0.10.2，实际 {vllm.__version__}"
assert ver("transformers").startswith("4.57"), \
    f"transformers 必须留在 4.57.x（vLLM 0.10.2 的 get_cached_tokenizer 依赖 5.x 已删除的属性），实际 {ver('transformers')}"

# 这条线的意义就在于 vLLM 是 editable 的：四个剪枝方法要给它的路由打补丁。
direct = md.distribution("vllm").read_text("direct_url.json")
assert direct and '"editable": true' in direct.lower(), \
    "vLLM 不是 editable 安装；剪枝补丁将无法生效"
print("  vllm editable  yes")

# 四个方法的实现入口都要能导入
from expert_pruning import routing, vllm_patch  # noqa: F401
import lighteval, lm_eval  # noqa: F401
print("  expert_pruning / lighteval / lm_eval  import OK")
print("VERIFY_OK")
PY

cat <<'EOF'

[install_expertpruning.sh] 完成。

冒烟测试。$M 是本地 MoE checkpoint 目录，$NAME 是它的目录名（校准产物按名字存放）：

  M=/path/to/Qwen3-30B-A3B-Instruct-2507
  NAME="$(basename "$M")"

  # 张量并行默认等于可见 GPU 数，所以在多卡机器上要显式限定，否则会以整机
  # TP 启动引擎并失败。
  export CUDA_VISIBLE_DEVICES=0,1

  # 未剪枝基线
  python main.py --datasets gsm8k --max_samples 8 --model_path "$M"

  # 方法一 Fixed-K：只保留 router 打分最高的 K 个专家（改 config，不打补丁）
  python main.py --datasets gsm8k --max_samples 8 --num_experts_per_tok 4 --model_path "$M"

  # 自适应规则走本地路由补丁。--enforce_eager 是必需的：专家选择一旦被编进
  # compiled region，torch.compile 会把计数器优化掉，剪枝照常生效但平均专家数
  # 测不到，而那个平均数正是所有对比的横轴，所以 main.py 会直接报错而不是
  # 报告一个它不敢保证的数字。量化模型（gpt-oss）另有做法，见 README。
  python main.py --datasets gsm8k --max_samples 8 --model_path "$M" \
      --use_local_expert_router --enforce_eager \
      --expert_pruning_method Dynamic_Routing \
      --dynamic_routing_threshold 0.8 --dynamic_routing_score_source renormalized

  # Ban 还要先做一次校准，产物按模型名存放
  bash scripts/models/qwen3-30b-a3b-instruct-2507/calibrate.sh
  python main.py --datasets gsm8k --max_samples 8 --model_path "$M" \
      --use_local_expert_router --enforce_eager \
      --expert_pruning_method Ban --ban_lambda 0.7 --ban_k_min 3 \
      --ban_artifact_path "calib_utils/results/$NAME/c4_ban.pt"

  # 0-shot 选择题（log-likelihood 打分，不生成 token）
  python main.py --harness lm_eval --lm_eval_tasks arc_challenge --max_samples 8 --model_path "$M"

  # 整套 sweep 用统一入口，不必手拼命令，也不会漏掉上面这些细节：
  #   bash scripts/run.sh list
  #   bash scripts/run.sh qwen3-30b all --gpus 0,1

多模态（--harness lmms_eval）：见 docs/legacy_qwen35_env.md。
新架构 MoE（DeepSeek-V4-Flash / Hy3 等）：用 install_fixedk.sh。
EOF
