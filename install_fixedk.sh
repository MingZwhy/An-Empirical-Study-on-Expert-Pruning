#!/usr/bin/env bash
# Run from the repository root:  bash install_fixedk.sh [--with-lmms-eval]
#
# ------------------------------------------------------------------------------
# What this environment is for
# ------------------------------------------------------------------------------
# The second of the two environments in this repository. It runs **Fixed-K
# only** (`--num_experts_per_tok`, i.e. keep the top-K experts the router
# already ranked highest) on MoE checkpoints whose architectures are too new
# for the pinned vLLM used by the pruning environment:
#
#   DeepSeek-V4-Flash-0731 (DeepseekV4ForCausalLM)
#   Hy3                    (HYV3ForCausalLM)
#   MiniMax-M2.7, GLM-4.7-Flash, Qwen3.5-35B-A3B, ...
#
# Fixed-K needs no vLLM source changes: main.py rewrites `num_experts_per_tok`
# in a config-only overlay and vLLM's own router then selects K experts. That is
# why this environment installs vLLM as a **released, non-editable wheel** and
# stays easy to upgrade, whereas `install_expertpruning.sh` must keep vLLM
# editable to host the four learned pruning methods.
#
#   | | install_expertpruning.sh | install_fixedk.sh (this file) |
#   |---|---|---|
#   | conda env | expertpruning | fixedk |
#   | vLLM | v0.10.2, submodule, editable | 0.28.0, PyPI wheel, not editable |
#   | torch | 2.8.x (pinned by that wheel) | 2.13.0+cu130 |
#   | transformers | 4.57.6 (pinned) | 5.17.0 (resolved by vLLM) |
#   | Methods | Fixed-K + Ban + Dynamic Routing + Router-Guided | Fixed-K only |
#   | Models | Qwen3-30B-A3B, Qwen3-Next-80B, Ling-lite, gpt-oss-20b | the new MoE checkpoints above |
#
# The two environments are independent; installing one never touches the other.
#
# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
# The load-bearing versions are declared below and the resolver picks the rest,
# which is what lets this survive harmless upstream churn. Small version drift is
# expected and fine: the study's conclusions do not rest on a byte-identical
# environment, and the self-check at the end asserts the parts that do matter.
#
#   --with-lmms-eval  Also install lmms-eval for the image-text suite. Note that
#               the published multimodal numbers came from the legacy
#               install_qwen35.sh environment; this combination installs and
#               imports but we have not re-run the VL suite on vLLM 0.28.
#
# ------------------------------------------------------------------------------
# Why these exact versions
# ------------------------------------------------------------------------------
# vLLM 0.28.0 is the first release whose model registry carries both
# DeepseekV4ForCausalLM and HYV3ForCausalLM. Its wheel is built against
# torch 2.13.0+cu130, so torch/flashinfer must come from the cu130 indexes or
# `import vllm._C` dies with an undefined c10 symbol. flashinfer is split into
# three distributions and only `flashinfer-python` is on PyPI; the cubin and
# jit-cache halves live on flashinfer.ai and must match 0.6.17 exactly.
#
# See docs/vllm_install_notes.md for the ABI analysis behind all of this.
set -euo pipefail

WITH_LMMS=0
for arg in "$@"; do
    case "$arg" in
        --with-lmms-eval) WITH_LMMS=1 ;;
        -h|--help) sed -n '1,58p' "$0"; exit 0 ;;
        *) echo "[install_fixedk.sh] unknown argument: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VLLM_VERSION=0.28.0
TORCH_BACKEND=cu130
FLASHINFER_VERSION=0.6.17

# -------- 0. environment checks --------
# Either a conda env or a plain venv is fine; what matters is that we are not
# about to install ~20 GB of CUDA wheels into the system interpreter.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    ENV_KIND="conda:${CONDA_DEFAULT_ENV:-unknown}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    ENV_KIND="venv:$(basename "$VIRTUAL_ENV")"
else
    cat >&2 <<'MSG'
[install_fixedk.sh] ERROR: no conda env and no virtualenv is active.
    This installs vLLM, torch/cu130 and flashinfer (~20 GB). Refusing to touch
    the system interpreter. Create an isolated environment first, either:

      conda create -y -n fixedk python=3.12 && conda activate fixedk
    or
      python3.12 -m venv ~/envs/fixedk && source ~/envs/fixedk/bin/activate

    Then re-run this script. Override with FIXEDK_ALLOW_SYSTEM_PYTHON=1 if you
    really know what you are doing.
MSG
    [[ "${FIXEDK_ALLOW_SYSTEM_PYTHON:-0}" == "1" ]] || exit 1
    ENV_KIND="system (forced)"
fi

PY_VERSION="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PY_VERSION" in
    3.12|3.11) ;;
    *) echo "[install_fixedk.sh] ERROR: Python $PY_VERSION is untested; use 3.12 (or 3.11)." >&2; exit 1 ;;
esac
echo "[install_fixedk.sh] Environment: $ENV_KIND"
echo "[install_fixedk.sh] Python:      $(which python) ($PY_VERSION)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[install_fixedk.sh] WARNING: nvidia-smi not found. Installation will still work, but nothing will run."
fi

# Three separate indexes have to be reachable. Failing here with a clear message
# beats failing twenty minutes in with a wall of resolver timeouts -- on a
# proxied network you usually just need https_proxy exported.
echo "[install_fixedk.sh] ==> checking package index reachability"
UNREACHABLE=()
for url in https://pypi.org/simple/ \
           "https://download.pytorch.org/whl/$TORCH_BACKEND" \
           https://flashinfer.ai/whl; do
    code="$(curl -sI --max-time 20 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo 000)"
    # Any HTTP status proves the endpoint answered; 000 is the failure mode we
    # care about (DNS, proxy, firewall). Some index roots legitimately 404 on a
    # bare HEAD, so treating 4xx as unreachable produces false alarms.
    if [[ "$code" == 000 ]]; then
        UNREACHABLE+=("$url (no response)")
    fi
done
if (( ${#UNREACHABLE[@]} )); then
    echo "[install_fixedk.sh] ERROR: cannot reach:" >&2
    printf '[install_fixedk.sh]   %s\n' "${UNREACHABLE[@]}" >&2
    echo "[install_fixedk.sh] If you are behind a proxy, export https_proxy/http_proxy and retry." >&2
    exit 1
fi

# uv resolves the torch/vLLM/flashinfer matrix far faster than pip and knows the
# --torch-backend flag. Install it into this env if it is not already present.
if ! python -m uv --version >/dev/null 2>&1; then
    echo "[install_fixedk.sh] ==> bootstrapping uv"
    pip install -q -U uv
fi
UV="python -m uv"

# -------- 1. submodules --------
# lighteval carries this project's patches (see 3rdparty/lighteval history:
# every commit is tagged [ExpertPruning-mod]). lm-evaluation-harness is plain
# upstream v0.4.13 -- no fork, no patch.
for sub in 3rdparty/lighteval 3rdparty/lm-evaluation-harness; do
    if [[ ! -e "$REPO_ROOT/$sub/.git" ]]; then
        echo "[install_fixedk.sh] ==> initialising submodule $sub"
        git submodule update --init -- "$sub"
    else
        echo "[install_fixedk.sh] $sub already initialised"
    fi
done
# The vLLM submodule is not needed here (we install a released wheel) and
# lmms-eval only when explicitly requested.
if [[ "$WITH_LMMS" -eq 1 && ! -e "$REPO_ROOT/3rdparty/lmms-eval/.git" ]]; then
    git submodule update --init -- 3rdparty/lmms-eval
fi

# -------- 2. vLLM and the evaluation stack, in one resolution --------
# These must be solved together, not in sequence. vLLM 0.28 on its own resolves
# fsspec 2026.7.0, but datasets 5.0.1 requires fsspec <= 2026.6.0; installing
# vLLM first and then constraining fsspec to what it picked makes the eval
# dependencies unsatisfiable. Handing the solver both at once lets it choose the
# fsspec that satisfies everything.
#
# ray is not a vLLM 0.28 dependency any more, but lighteval's vLLM backend
# imports it at module scope, so it has to be here or importing the model class
# fails with ModuleNotFoundError.
echo "[install_fixedk.sh] ==> installing vLLM $VLLM_VERSION and the evaluation stack (single resolution)"
$UV pip install --torch-backend="$TORCH_BACKEND" \
    "vllm==$VLLM_VERSION" \
    ray \
    "datasets==5.0.1" more-itertools langdetect "math-verify==0.5.2" \
    "latex2sympy2-extended==1.0.6" latex2sympy2 "evaluate==0.4.6" \
    numexpr peft pybind11 sqlitedict tqdm-multiprocess word2number \
    GitPython hf-xet "termcolor==2.3.0" pytablewriter colorlog \
    "aenum==3.1.15" "nltk==3.9.1" scikit-learn scipy sacrebleu \
    "rouge-score==0.1.2" pycountry langcodes

# -------- 3. flashinfer: three distributions, three indexes --------
# Installed with --no-deps on purpose: they must not be allowed to move torch.
#
# Only flashinfer-python is on PyPI; the cubin and jit-cache halves live on
# flashinfer.ai and all three have to be the same version.
#
# Note that this deliberately overrides vLLM's own requirement. vLLM 0.28.0 pins
# flashinfer-python==0.6.16.post3 exactly, step 2 therefore installs that, and
# this step replaces it with 0.6.17 -- the version the published runs used, and
# the one the self-check below asserts. Both versions are complete on the
# flashinfer indexes, so this is a recorded choice rather than a forced one.
#
# Two consequences worth knowing rather than discovering: flashinfer-python is
# downloaded twice, which cannot be avoided by pinning it in step 2 without
# contradicting vLLM's exact requirement and failing the resolve; and `pip check`
# will report vllm 0.28.0 wanting 0.6.16.post3 against 0.6.17 installed. That
# report is expected and is not a broken environment.
echo "[install_fixedk.sh] ==> installing flashinfer $FLASHINFER_VERSION (python + cubin + jit-cache)"
$UV pip install --no-deps "flashinfer-python==$FLASHINFER_VERSION"
$UV pip install --no-deps --index-url https://flashinfer.ai/whl "flashinfer-cubin==$FLASHINFER_VERSION"
$UV pip install --no-deps --index-url "https://flashinfer.ai/whl/$TORCH_BACKEND" \
    "flashinfer-jit-cache==${FLASHINFER_VERSION}+${TORCH_BACKEND}"

# -------- shared: constraints for the two editable installs --------
# The editable installs below must be allowed to add missing packages but never
# to move the ones that matter. torch and transformers are the ABI counterparties
# of the vLLM wheel: replace either and `import vllm` stops working.
write_constraints() {
    python - <<'PY'
import importlib.metadata as md
# Only what must not move: the binary counterparties of the vLLM wheel, and the
# two packages that silently change how AIME/MATH answers are parsed. Everything
# else is left to the resolver -- over-constraining pure-Python packages is what
# makes the eval dependencies unsatisfiable.
critical = """torch torchvision torchaudio triton vllm transformers tokenizers
numpy flashinfer-python flashinfer-cubin flashinfer-jit-cache
math-verify latex2sympy2-extended latex2sympy2""".split()
for name in critical:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
PY
}

# -------- 4. lighteval (editable, no [vllm] extras) --------
# Without extras on purpose: lighteval[vllm] declares vllm>=0.11 and would
# reinstall vLLM from PyPI, undoing the pinned wheel above.
echo "[install_fixedk.sh] ==> installing lighteval (editable)"
CONSTRAINTS="$(mktemp)"
write_constraints > "$CONSTRAINTS"
$UV pip install --editable "$REPO_ROOT/3rdparty/lighteval" --constraint "$CONSTRAINTS"
rm -f "$CONSTRAINTS"

# -------- 5. lm-evaluation-harness (editable) --------
# Pinned to upstream v0.4.13. That release already carries everything this stack
# needs -- both vLLM tokenizer import paths are probed, swap_space is accepted
# and ignored, hf_vlms uses the Transformers 5 class name, and the winogrande /
# openbookqa dataset ids are namespaced (huggingface_hub >= 1.0 rejects bare
# ids with "Invalid HF URI ... must be 'namespace/name'"). Relative to the
# v0.4.9.2 the paper was produced on, the four QA task configs differ only by
# that namespacing, so the published QA numbers reproduce unchanged.
echo "[install_fixedk.sh] ==> installing lm-evaluation-harness (editable)"
CONSTRAINTS="$(mktemp)"
write_constraints > "$CONSTRAINTS"
$UV pip install --editable "$REPO_ROOT/3rdparty/lm-evaluation-harness" --constraint "$CONSTRAINTS"
rm -f "$CONSTRAINTS"

# -------- 6. optional: lmms-eval --------
if [[ "$WITH_LMMS" -eq 1 ]]; then
    echo "[install_fixedk.sh] ==> installing lmms-eval (editable --no-deps)"
    CONSTRAINTS="$(mktemp)"
    write_constraints > "$CONSTRAINTS"
    $UV pip install --constraint "$CONSTRAINTS" \
        loguru openpyxl "av<16.0.0" ftfy timm pycocoevalcap zss \
        "qwen-vl-utils>=0.0.14" sentence_transformers yt-dlp evaluate Levenshtein
    $UV pip install --editable "$REPO_ROOT/3rdparty/lmms-eval" --no-deps
    rm -f "$CONSTRAINTS"
fi

# -------- 7. verify --------
echo "[install_fixedk.sh] ==> verifying"
python - <<'PY'
import importlib.metadata as md
import torch
import vllm
from vllm.model_executor.models.registry import ModelRegistry

def show(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "MISSING"

print(f"  vllm                 {vllm.__version__}")
print(f"  torch                {torch.__version__}")
print(f"  transformers         {show('transformers')}")
print(f"  flashinfer-python    {show('flashinfer-python')}")
print(f"  flashinfer-cubin     {show('flashinfer-cubin')}")
print(f"  flashinfer-jit-cache {show('flashinfer-jit-cache')}")

assert vllm.__version__ == "0.28.0", f"expected vLLM 0.28.0, got {vllm.__version__}"
assert torch.__version__.startswith("2.13.0"), f"expected torch 2.13.0+cu130, got {torch.__version__}"

direct = md.distribution("vllm").read_text("direct_url.json")
assert not direct or '"editable": true' not in direct.lower(), \
    "vLLM must NOT be editable in this environment; use install_expertpruning.sh for that"

archs = set(ModelRegistry.get_supported_archs())
required = {"DeepseekV4ForCausalLM", "HYV3ForCausalLM"}
missing = required - archs
assert not missing, f"vLLM registry is missing {missing}"
print(f"  registry             {len(archs)} architectures, new MoE checkpoints present")

import lighteval, lm_eval  # noqa: F401
from lighteval.models.vllm.vllm_model import VLLMModelConfig  # exercises the patched import
print("  lighteval / lm_eval  import OK")
print("VERIFY_OK")
PY

cat <<'EOF'

[install_fixedk.sh] Done.

Smoke tests (replace <model> with a local checkpoint directory):

  # unpruned baseline, 8 samples
  python main.py --datasets gsm8k --max_samples 8 \
      --model_path <model> --tensor_parallel_size 8

  # Fixed-K: keep the 4 highest-scoring experts per token
  python main.py --datasets gsm8k --max_samples 8 \
      --num_experts_per_tok 4 \
      --model_path <model> --tensor_parallel_size 8

  # 0-shot multiple choice (log-likelihood scored, nothing generated)
  python main.py --harness lm_eval --lm_eval_tasks arc_challenge --max_samples 8 \
      --model_path <model> --tensor_parallel_size 8

EOF
