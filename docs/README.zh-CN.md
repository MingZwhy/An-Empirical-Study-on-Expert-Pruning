# 专家剪枝实证研究

[![arXiv](https://img.shields.io/badge/arXiv-2609.25809-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.25809)
[![License: MIT](https://img.shields.io/badge/License-MIT-37b24d?style=flat-square)](../LICENSE)

> **[You Only Need 2/3 of the Chosen Experts: An Empirical Study of Dynamic Expert
> Pruning in Fine-Grained MoE LLMs](https://arxiv.org/abs/2609.25809)**
>
> [English README](../README.md)

本仓库是一项关于**细粒度 MoE 大模型推理期专家剪枝**的实证研究的代码与结果：
每个 token 的专家选择到底有多冗余、现有剪枝方法能把这份冗余利用到什么程度、
以及是什么决定了一个模型对剪枝的敏感程度。

覆盖 **9 个架构族的 12 个 checkpoint**，**3 个任务族共 11 个 benchmark**。
全部在未改动的公开权重上测量——不微调、不蒸馏、不改权重，
唯一变化的是每个 token 激活的专家数量。

![总览图](figures/paper/teaser_v13_arxiv.png)

## 核心结论

| | |
|---|---|
| **三分之二的专家就够了。** | 保留 `⌈2K/3⌉` 个专家，平均保住未剪枝性能的 **98.8%**（9 个模型 × 3 个任务族）。 |
| **这只是改一个整数。** | 不微调、不校准、不动权重——而且在两种服务后端上实测快 **1.2–1.7 倍**。 |
| **自适应规则只有在激进预算下才值得。** | 保守预算下比统一截断平均只高 **+0.06**；激进剪枝下高 **+2.26**。 |
| **敏感度是模型的属性，不是预算的函数。** | 更大的模型和推理调优模型更耐剪，多模态模型明显更脆弱。 |

27 个「模型 × 任务族」组合里有 6 个在 2/3 预算下持平或超过自己的未剪枝分数。
checkpoint 本身一个字节都没动，改变的只是每个 token 能用多少个专家。

2/3 预算下各架构的准确率保留：

| | 模型 | 总参数 | 激活参数 | 专家数 | 每 token 激活 | 原生 → 剪后 | 保留 |
|---|---|---:|---:|---:|---:|:---:|:---:|
| <img src="figures/logos/openai.png" width="20"> | [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) | 21 B | 3.6 B | 32 | 4 | ![4 to 3](https://img.shields.io/badge/4%20%E2%86%92%203-4c6ef5?style=flat-square) | ![100.7 percent](https://img.shields.io/badge/100.7%25-2b8a3e?style=flat-square) |
| <img src="figures/logos/qwen.png" width="20"> | [Qwen3-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | 80 B | 3 B | 512 | 10 | ![10 to 7](https://img.shields.io/badge/10%20%E2%86%92%207-4c6ef5?style=flat-square) | ![100.5 percent](https://img.shields.io/badge/100.5%25-2b8a3e?style=flat-square) |
| <img src="figures/logos/deepseek.png" width="20"> | [DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) | 15.7 B | 2.4 B | 64 | 6 | ![6 to 4](https://img.shields.io/badge/6%20%E2%86%92%204-4c6ef5?style=flat-square) | ![99.9 percent](https://img.shields.io/badge/99.9%25-37b24d?style=flat-square) |
| <img src="figures/logos/deepseek.png" width="20"> | [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | 165 B | 11.5 B | 256 | 6 | ![6 to 4](https://img.shields.io/badge/6%20%E2%86%92%204-4c6ef5?style=flat-square) | ![98.8 percent](https://img.shields.io/badge/98.8%25-37b24d?style=flat-square) |
| <img src="figures/logos/qwen.png" width="20"> | [Qwen3-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | 30.5 B | 3.3 B | 128 | 8 | ![8 to 6](https://img.shields.io/badge/8%20%E2%86%92%206-4c6ef5?style=flat-square) | ![98.6 percent](https://img.shields.io/badge/98.6%25-37b24d?style=flat-square) |
| <img src="figures/logos/minimax.png" width="20"> | [MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) | 229 B | 11 B | 256 | 8 | ![8 to 6](https://img.shields.io/badge/8%20%E2%86%92%206-4c6ef5?style=flat-square) | ![98.1 percent](https://img.shields.io/badge/98.1%25-37b24d?style=flat-square) |
| <img src="figures/logos/hunyuan.png" width="20"> | [Hunyuan3](https://huggingface.co/tencent/Hy3) | 295 B | 21 B | 192 | 8 | ![8 to 6](https://img.shields.io/badge/8%20%E2%86%92%206-4c6ef5?style=flat-square) | ![98.0 percent](https://img.shields.io/badge/98.0%25-37b24d?style=flat-square) |
| <img src="figures/logos/google.png" width="20"> | [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) | 25.2 B | 3.8 B | 128 | 8 | ![8 to 6](https://img.shields.io/badge/8%20%E2%86%92%206-4c6ef5?style=flat-square) | ![97.5 percent](https://img.shields.io/badge/97.5%25-74b816?style=flat-square) |
| <img src="figures/logos/antgroup.png" width="20"> | [Ling-lite-1.5-2507](https://huggingface.co/inclusionAI/Ling-lite-1.5) | 16.8 B | 2.75 B | 64 | 6 | ![6 to 4](https://img.shields.io/badge/6%20%E2%86%92%204-4c6ef5?style=flat-square) | ![97.4 percent](https://img.shields.io/badge/97.4%25-74b816?style=flat-square) |

*总参数* 与 *激活参数* 是参数量，*专家数* 是每层的 routed 专家数，
*每 token 激活* 是 router 原生为每个 token 激活的数量。
共享专家（每个 token 无条件经过）不计入这两列：Qwen3-Next-80B、Hunyuan3、
DeepSeek-V4-Flash、Gemma 4 各有 1 个，Ling-lite-1.5 与 DeepSeek-V2-Lite 各 2 个，其余为 0。

11 个 benchmark，分三族：

| 任务族 | 打分方式 | Benchmark |
|---|---|---|
| 知识问答 | log-likelihood，zero-shot | ARC-Easy, ARC-Challenge, WinoGrande, OpenBookQA |
| 数学与代码推理 | 生成 | AIME-24, AIME-25, MATH-500, GSM8K, LiveCodeBench-v6 |
| 知识与通用推理 | 生成 | MMLU-Pro, GPQA-Diamond |

## 方法

每个规则都保留全部专家常驻，只改变每个 token 实际计算其中几个。所有规则都是在同一个
路由补丁上重新实现的，所以对比结果里没有任何一部分能用"harness 不同"来解释。

`--expert_pruning_method` 选一个：

| 方法 | 保留依据 | 分层预算 | 需校准 | k 下限 | 旋钮 | 论文 |
|---|---|:---:|:---:|:---:|---|---|
| **Fixed-K**（基线） | 所有 token 用同一个常数 k | 否 | 否 | — | `--num_experts_per_tok` | 本研究 |
| **NAEE** | 门控权重相对 top-1 专家的比例 | 否 | 否 | 2 | `--naee_beta` | [ACL 2024](https://aclanthology.org/2024.acl-long.334/) |
| **Dynamic Routing** | 累计保留的概率质量 | 否 | 否 | — | `--dynamic_routing_threshold` | [ACL 2024](https://aclanthology.org/2024.acl-long.696/) |
| **DiEP** | 用专家相似度缩放 top-1 比值 | 否 | **是** | 2 | `--naee_beta`, `--diep_artifact_path` | [NeurIPS 2025](https://doi.org/10.52202/085713-1878) |
| **Ban** | 层敏感度 + token 敏感度 | **是** | **是** | 3 | `--ban_lambda`, `--ban_artifact_path` | [EMNLP 2026](https://arxiv.org/abs/2509.06346) |

另外两个已发表的规则也实现了、可选，但不在同等预算对比之内，因为它们剪的是
另一个维度：
**MC-MoE** ([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/abc1943857a42935ceacff03c524bb44-Abstract-Conference.html))
先用 attention sink 保护重要 token 再做阈值剪枝；
**EAC-MoE** ([ACL 2025](https://aclanthology.org/2025.acl-long.633/)) 把剪枝限制在 prefill 阶段，依据每个专家收到的流量决定。若干 layer-budget
变体（`TopK_Biased_Renorm`、`LayerWise_Dynamic_Routing`、`Budget_Dynamic_Routing` 等）
是我们自己的；完整列表见 `python main.py --help`。

每个参数是什么、怎么选：[**docs/hyperparameters.md**](hyperparameters.md)（英文）。

**自适应规则支持哪些模型。** 除 Fixed-K 之外都需要 `--use_local_expert_router`，
它会给 vLLM 的 fused top-k 路由打补丁。这个补丁认识本研究用到它的那几个模型的
MoE 实现——Qwen3-30B-A3B、Qwen3-Next-80B、Ling-lite-1.5、GPT-OSS-20B。
遇到它不覆盖的架构时，它会**直接拒绝启动**，而不是静默地让模型不被剪枝：

```
RuntimeError: Expert pruning cannot intercept expert selection for MoE layer ...
```

DeepSeek-V2-Lite 就是这种情况，这也是论文里它只有 Fixed-K 结果的原因。
Fixed-K 本身没有这个限制：它只是一层 config overlay，引擎能加载的 MoE 它都能跑。

这个差别也正是为什么安装分成两条路径而不是一条，见下面的
[两个环境](#两个环境)。

## 结果

### 1. 细粒度 MoE 到底需要多少路由？

九个架构上的 Fixed-K。阴影带是到 2/3 预算的安全区，三条曲线是三个任务族。
**知识问答在每个模型上都是最后崩的，数学与代码推理都是最先崩的**：

![九个模型的 Fixed-K 曲线，按任务族拆分](figures/paper/fixedk_grid33.png)

每个预算、每个数据集的完整 sweep 见
[`docs/expert_pruning_results.md`](expert_pruning_results.md)
以及 [`results/`](../results) 下各模型的表。

### 2. 什么时候动态分配才胜过统一剪枝？

[论文表 3](https://arxiv.org/abs/2609.25809)。每个模型给两档预算，每档下把统一截断与四个动态规则中**在同等实测预算
下**表现最好的那个配对：

![逐数据集的同等预算对比：四个模型各在保守与激进两档预算下，把 Fixed-K 与最好的动态规则在 11 个数据集上配对](figures/paper/table3_isobudget.png)

顺着上标往下读：保守预算下差值在 `-0.53` 到 `+0.67` 之间，平均 **+0.06**——统一
截断已经追平了最好的规则，那套机制什么也没换来。把预算收紧，每个模型都翻转：
**+2.45**、**+2.71**、**+2.98**、**+0.90**，平均 **+2.26**。知道该丢**哪些**专家，
只有在剩下的专家少到"选择"开始起作用之后才有回报。

各规则在两档预算下的逐数据集明细见
[`docs/expert_pruning_results.md`](expert_pruning_results.md) 与
[`docs/qa_summary.md`](qa_summary.md)。

### 3. 是什么决定了对剪枝的敏感度？

在其他条件固定时，推理调优版比它的 instruct 兄弟更稳，同族里更大的模型比更小的
更稳——而且**差距只有在预算变激进之后才拉开**：

![两组配对对比：Qwen3-30B-A3B Thinking vs Instruct，以及 Qwen3-30B-A3B vs Qwen3-235B-A22B](figures/paper/axes_pairs.png)

多模态方向相反，而 router 给出了解释：VL 模型分配在它所选专家上的
softmax 质量要分散得多，可供移除的冗余因此少得多：

![左：Qwen3-VL-30B-A3B 比 Qwen3-30B-A3B 退化更快。右：VL 模型的 router 权重集中度低得多](figures/paper/modality.png)

右图背后的 router 统计见 [`docs/router_distribution.md`](router_distribution.md)。

### 各模型结果表

| 模型 | 覆盖的方法 |
|---|---|
| [Qwen3-30B-A3B-Instruct](../results/Qwen3-30B-A3B-Instruct-2507/RESULTS.md) | 全部规则，两档预算 |
| [Qwen3-Next-80B-A3B](../results/Qwen3-Next-80B-A3B-Instruct/RESULTS.md) | 全部规则，两档预算 |
| [Ling-lite-1.5-2507](../results/Ling-lite-1.5-2507/RESULTS.md) | 全部规则，两档预算 |
| [GPT-OSS-20B](../results/gpt-oss-20b/RESULTS.md) | 全部规则，两档预算 |
| [MiniMax-M2.7](../results/MiniMax-M2.7/RESULTS.md) | Fixed-K sweep |
| [DeepSeek-V2-Lite-Chat](../results/DeepSeek-V2-Lite-Chat/RESULTS.md) | Fixed-K sweep |
| [DeepSeek-V4-Flash-0731](../results/DeepSeek-V4-Flash-0731/RESULTS.md) | Fixed-K sweep |
| [Hunyuan3](../results/Hy3/RESULTS.md) | Fixed-K sweep |

Gemma 4 26B-A4B 在论文的 Fixed-K sweep 里、也在上面的表里，但它的单模型页面还没有
放进仓库，不过它的 k 阶梯和其他模型一样可以复现。配对对比与模态那两张
图背后的三个单因子 checkpoint——Qwen3-30B-A3B-Thinking、Qwen3-235B-A22B、
Qwen3-VL-30B-A3B——同样是论文有、这里没有单独页面。

## 两个环境

有两个安装脚本，选哪个只取决于你要跑的模型。

| | `install_expertpruning.sh` | `install_fixedk.sh` |
|---|---|---|
| **能跑什么** | 全部方法，含自适应规则 | 只有 Fixed-K |
| **vLLM** | v0.10.2，submodule，**可编辑** | 0.28.0，发布版 wheel，非可编辑 |
| **torch** | 2.8.0+cu128（由那个 wheel 锚定） | 2.13.0+cu130 |
| **transformers** | 4.57.6（钉死） | 5.17.0（由 vLLM 解出） |
| **模型** | Qwen3-30B-A3B, Qwen3-Next-80B, Ling-lite-1.5, GPT-OSS-20B, … | DeepSeek-V4-Flash, Hunyuan3, MiniMax-M2.7, Gemma 4, Qwen3.5, … |

DeepSeek-V2-Lite 老到两个引擎都能加载、而且只有 Fixed-K，所以两个环境都能跑；
`scripts/run.sh` 知道哪个模型属于哪一类，并在启动任何东西之前先检查。

两个环境互不影响，装其中一个不会碰到另一个。

<details>
<summary><b>为什么不能只用一个环境？</b></summary>

自适应规则要替换 vLLM 的 fused 路由 kernel，所以那个环境必须让 vLLM 保持可编辑、
因而也就被钉在某个版本上。而被钉住的引擎装不了比它更晚发布的模型架构。

Fixed-K 完全不需要改引擎：它只在一份 config-only overlay 里改写
`num_experts_per_tok`，让模型自己的 router 去选它本来就已经排在前面的 top-K，
所以那个环境可以跟着 vLLM 的发布走——这正是新 checkpoint 能跑起来的唯一原因。

也就是说，分环境不是因为哪种剪枝规则更有意思，而是因为一个打了补丁的引擎不可能
同时是一个最新的引擎。

</details>

## 安装

两个脚本都拒绝在隔离环境之外运行，也都会在下载那约 20 GB（耗时 35 到 40 分钟）之前
先检查所需的索引是否可达。

```bash
git clone --recurse-submodules https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning.git
cd An-Empirical-Study-on-Expert-Pruning
```

**全部方法**（可编辑 vLLM v0.10.2）：

```bash
conda create -y -n expertpruning python=3.12 && conda activate expertpruning
bash install_expertpruning.sh
```

**新架构上的 Fixed-K**（发布版 vLLM 0.28.0）：

```bash
conda create -y -n fixedk python=3.12 && conda activate fixedk
bash install_fixedk.sh
```

用 `venv` 和 conda 一样可以。

两个脚本都只声明必须成立的版本、其余交给解析器，所以和上表有一点偏差是正常的，
也不会让结论失效——本研究的结论不依赖于一个逐字节相同的环境。

两个脚本最后都会自检：断言所依赖的版本、确认 vLLM 只在该可编辑的地方可编辑、
以及模型注册表里确实有这个环境要跑的架构。

### 怎么自检你的安装

别看分数。几条样本上的准确率本来就是噪声，而一个没生效的剪枝规则照样能给出一个
体面的分数——这恰恰是它最麻烦的地方。

要看的是每个 token 实际保留了几个专家。这个数**没法被一个坏掉的环境伪造**：补丁没挂上
会报原生的 k，阈值过激会贴在 `k_min` 地板上，只有整条链路都对（补丁挂载、阈值语义被
正确解释、计数器还活着）才会落在发表值附近。两张卡两分钟：

```bash
export MODELS_DIR=$HOME/models
EP_GEN_DATASETS=gsm8k bash scripts/run.sh qwen3-30b dynamic_routing \
    --only thr0.8 --smoke --gpus 0,1

grep 平均专家选择数 local_logs/Qwen3-30B-A3B-Instruct-2507/*.generative.log
```

原生 k=8，对照[发表的 sweep](../results/Qwen3-30B-A3B-Instruct-2507/RESULTS.md)：

| 配置 | 发表值 | 我们的运行 |
|---|---:|---:|
| Dynamic Routing，阈值 0.8 | 5.918 | 5.936 |
| NAEE，β=0.35，k_min=2 | 6.408 | 6.456 |

差在 0.1 以内就算对上了。这个量能跨数据集比较，是因为它取决于 router 权重分布的形状
而不是题目本身——这也是四条 GSM8K 样本能和九数据集 sweep 相比的原因。如果读数正好是
8.000、或者根本没有这一行，说明规则没在跑，见
[诊断那一节](hyperparameters.md#checking-that-pruning-actually-happened)。

## 快速开始

`main.py` 是唯一入口，`--model_path` 指向本地 checkpoint 目录。

张量并行默认等于可见 GPU 数，所以在多卡机器上先设好 `CUDA_VISIBLE_DEVICES`——
否则下面这些命令会试图把一个 30B 模型切到全部卡上，并在引擎初始化时失败。

```bash
export CUDA_VISIBLE_DEVICES=0,1

# 未剪枝基线
python main.py --datasets gsm8k --max_samples 8 --model_path <model>

# Fixed-K：每个 token 只保留打分最高的 4 个专家
python main.py --datasets gsm8k --max_samples 8 --num_experts_per_tok 4 --model_path <model>

# log-likelihood 选择题（不生成 token，所以剪枝效果不掺入
# "模型能不能干净收尾" 这个混杂因素）
python main.py --harness lm_eval --lm_eval_tasks arc_challenge --max_samples 8 --model_path <model>
```

Fixed-K 就这么多：一个整数，不动引擎。自适应规则额外需要路由补丁，
所以只能在 `expertpruning` 环境里跑：

```bash
# Dynamic Routing：保留累计概率达到阈值的最短专家前缀
python main.py --datasets gsm8k --max_samples 8 --model_path <model> \
    --use_local_expert_router --enforce_eager \
    --expert_pruning_method Dynamic_Routing \
    --dynamic_routing_threshold 0.8 --dynamic_routing_score_source renormalized

# Ban：结合离线层敏感度与在线 token 敏感度
python main.py --datasets gsm8k --max_samples 8 --model_path <model> \
    --use_local_expert_router --enforce_eager \
    --expert_pruning_method Ban \
    --ban_artifact_path calib_utils/results/<model-dir-name>/c4_ban.pt \
    --ban_lambda 0.7 --ban_k_min 3
```

**这里的 `--enforce_eager` 不是可选项。** 专家选择会被编进 compiled region，
`torch.compile` 把计数器优化掉，于是"平均保留了几个专家"就观测不到了。剪枝照常发生，
丢的只是测量——但那个平均数正是本研究所有对比对齐的横轴，所以 `main.py` 会直接停下，
而不是报告一个它无法背书的数字。量化模型是例外：在 gpt-oss 上 eager 要付 6.7 倍吞吐代价，
所以那里分数来自 CUDA graph 运行（`--allow_compiled_router`），专家数由单独的 eager 探针测。

给上面任何一条加 `--expert_pruning_debug` 就能看到计数过程中的保留均值。
值得看一眼：一个没生效的规则表现出来不是报错，而是准确率好得反常。

每个参数是什么、怎么选：[**docs/hyperparameters.md**](hyperparameters.md)（英文）。

## 复现整个研究

所有 sweep 都由一个入口驱动，它只需要一个装着 checkpoint 的目录：

```bash
export MODELS_DIR=$HOME/models

bash scripts/run.sh list                    # 有哪些模型和阶段
bash scripts/run.sh qwen3-30b all           # 一个模型，论文报告的全部方法
bash scripts/run.sh all                     # 整个研究，从最省卡的模型开始
```

九个模型，用下面的名字或任何无歧义的前缀都可以。带全部自适应规则的那四个，正是路由
补丁覆盖的那四个；其余只有 Fixed-K——它不需要给引擎打补丁，所以引擎能加载的模型它都能跑。

| | 方法 | 卡数 | 环境 |
|---|---|:---:|---|
| `qwen3-30b` | 全部规则 | 2 | `expertpruning` |
| `qwen3-next` | 全部规则 | 4 | `expertpruning` |
| `ling-lite` | 全部规则 | 2 | `expertpruning` |
| `gpt-oss` | 全部规则 | 2 | `expertpruning` |
| `deepseek-v2` | Fixed-K | 2 | 两者皆可 |
| `gemma-4` | Fixed-K | 2 | `fixedk` |
| `deepseek-v4` | Fixed-K | 4 | `fixedk` |
| `minimax` | Fixed-K | 8 | `fixedk` |
| `hunyuan3` | Fixed-K | 16 | `fixedk` |

卡数是发表运行所用的数量，`--gpus` 可以覆盖它，张量并行随之变化。`run.sh` 会在启动
任何东西之前检查当前环境与模型是否匹配——这一对搞错的话，表现出来是一个难懂的引擎报错。

### 以单个模型为例

论文里关于一个模型的全部内容都在一个目录下：

```
scripts/models/qwen3-30b-a3b-instruct-2507/
    model.env             原生 k=8、数据集列表、采样参数、张量并行
    baseline.sh           发布的 checkpoint，什么都不覆盖
    fixedk.sh             k = 7, 6, 5, 4, 3, 2, 1
    naee.sh               按门控权重与 top-1 专家的比值剪
    dynamic_routing.sh    保留累计概率达到阈值的最短专家前缀
    diep.sh               NAEE 阈值乘以校准得到的专家相似度
    ban.sh                层敏感度 + token 敏感度
    mc_moe.sh eac_moe.sh  已实现，但不在同等预算对比之内
    qa.sh                 四个 log-likelihood 选择题集
    calibrate.sh          Ban 和 DiEP 要读的那一次 C4 校准
```

先看，再跑。`--dry-run` 既不需要 GPU 也不需要 checkpoint：

```console
$ bash scripts/run.sh qwen3-30b ban --dry-run

== Qwen3-30B-A3B-Instruct-2507 :: Ban ==
   checkpoint  $MODELS_DIR/Qwen3-30B-A3B-Instruct-2507
   results     results/Qwen3-30B-A3B-Instruct-2507/
   gpus        all (tensor parallel 2)
   dry run: nothing will be executed

  # Ban-kmin3-lambda0.95
  python main.py \
      --model_path $MODELS_DIR/Qwen3-30B-A3B-Instruct-2507 \
      --datasets mmlu_pro,gpqa:diamond,math_500,aime24_avg,... \
      --max_model_len 32768 --max_new_tokens 32768 \
      --temperature 0.7 --top_p 0.8 --top_k 20 \
      --use_local_expert_router \
      --expert_pruning_method Ban --ban_lambda 0.95 --ban_k_min 3 \
      --ban_artifact_path calib_utils/results/.../c4_ban.pt \
      --output_dir results/Qwen3-30B-A3B-Instruct-2507/Ban-kmin3-lambda0.95
  ...
```

然后是实际的执行顺序，在两张卡上大约一天：

```bash
bash scripts/run.sh qwen3-30b calibrate --gpus 0,1        # 只需一次，Ban 和 DiEP 读它
bash scripts/run.sh qwen3-30b baseline fixedk --gpus 0,1  # 基线与 k 阶梯
bash scripts/run.sh qwen3-30b ban dynamic_routing --gpus 0,1
bash scripts/run.sh qwen3-30b qa --gpus 0,1               # 四个选择题集
```

结果落在 `results/<model>/<配置名>/`，日志在 `local_logs/`。已经有结果的配置会被跳过，
所以中断后重跑同一条命令就能续上。

**GPQA 已经帮你处理好了。** 它在 Hub 上是 gated 的、又在每个模型的数据集列表里，
所以第一个用到它的生成阶段会自己取——2.3 MB，来自作者自己的压缩包，不需要账号，几秒钟——
并告诉你它做了什么：

```
[preflight] gpqa:diamond is gated on the Hub and no local copy is present; fetching it
            from the authors' archive instead (2.3 MB, no account needed).
[preflight] done; reading GPQA from data/gpqa/
```

不用做任何事，也不会重复做：副本落在 `data/gpqa/`，该目录已被 gitignore。如果你更愿意用
Hub，同意门禁并登录之后就会走 Hub（那边是自动批准，不是人工审批队列）。`EP_NO_AUTO_FETCH=1`
可以关掉自动获取并改为列出选项，`python scripts/fetch_gpqa.py` 则是显式执行一次——
跑 sweep 的机器没有外网时你会需要它。

**本仓库不携带这些题目，也不会携带。** 数据集附带的条款要求不要让它的样例以明文出现在
网上，以免进入训练语料——而这恰好是本研究自己的数字所依赖的 benchmark，没有理由去污染它。
作者那个带密码的压缩包正好兼顾了两头：它不是可爬取的文本，这就是它存在的意义。

开放镜像也有，但不能替代：行数对得上（198）的那几个把三个错误选项压成了一个 solution
字符串，还原不出四选一。

两个选项值得在烧卡之前知道：`--smoke` 把每次运行缩到几条样本和很短的生成，
`--only` 从一个阶段里挑出单个配置。

```bash
bash scripts/run.sh qwen3-30b baseline fixedk --smoke --gpus 0,1
bash scripts/run.sh qwen3-30b fixedk --only k6 --gpus 0,1
```

### 怎么读一个配置

这些脚本是写给人看的。`ban.sh` 就是一行一个配置：

```bash
# matched to Fixed-K k=6
ep_run Ban-kmin3-lambda0.95 \
    --use_local_expert_router --expert_pruning_method Ban --ban_lambda 0.95 --ban_k_min 3 \
    --ban_artifact_path "$(ep_artifact ban)"
```

`Ban-kmin3-lambda0.95` 既是 `results/<model>/` 下的输出目录名，也是
[`results/Qwen3-30B-A3B-Instruct-2507/RESULTS.md`](../results/Qwen3-30B-A3B-Instruct-2507/RESULTS.md)
里的行标签，所以脚本里的一行和表里的一行是同一次运行。注释记录的是这个 `lambda`
被解到与哪一档 Fixed-K 预算对齐——按各自默认参数比较规则是量不出东西的，
因为保留更多专家的规则本来就会分高。

不需要 GPU 和 checkpoint 就能检查这些脚本本身：

```bash
python scripts/check_sweep_scripts.py
```

它会 dry-run 全部 294 个配置，并把每条命令发出的每个 flag 对着 `main.py` 的
参数解析器核一遍。更多细节见 [`scripts/README.md`](../scripts/README.md)。

## 目录结构

```
main.py                       所有 harness 与方法的唯一入口
install_expertpruning.sh      环境一：全部方法，可编辑 vLLM v0.10.2
install_fixedk.sh             环境二：Fixed-K，发布版 vLLM 0.28.0

expert_pruning/               实现本体
  routing.py                    各剪枝规则
  vllm_patch.py                 替换 vLLM 的 fused top-k router
  config_shims.py               Fixed-K config overlay
calib_utils/                  Ban 与 DiEP 的校准
tasks/, tasks_lm_eval/        在 harness 之上补充的任务定义

scripts/
  run.sh                        所有 sweep 的唯一入口
  models/<model>/               按模型：每个方法、每档预算
  lib/common.sh                 这些脚本共用的底层逻辑
  check_sweep_scripts.py        不需要 GPU 就能校验它们
  report_*.py, plot_*.py        从原始产物生成表格与图
  bench_*.py, solve_*.py        速度 benchmark、旋钮求解

docs/
  hyperparameters.md            每个参数及其取法
  expert_pruning_results.md     完整 sweep
  qa_summary.md                 跨模型 QA 表
  known_issues.md               粗糙之处，写下来而不是藏起来
  vllm_install_notes.md         两个环境背后的 ABI 分析
results/<model>/RESULTS.md     各模型结果表
3rdparty/                     vLLM、lighteval、lm-evaluation-harness、lmms-eval
```

## 复现性说明

- **harness 以 submodule 钉版本。** `lighteval` 是本项目的 fork，
  所带补丁在其历史里全部标了 `[ExpertPruning-mod]`；
  `lm-evaluation-harness` 是上游 v0.4.13，未改动。
- **log-likelihood QA 按裸续写打分**，不套 chat template，因为那四个集合的
  公开分数就是这么打的。套模板会改变数值并使其不可比。
- **生成缓存按剪枝配置指纹化**，所以一次 sweep 不会静默复用在别的设置下
  生成的样本。见 [`docs/compact_cache_contamination.md`](compact_cache_contamination.md)。
- **已知的粗糙之处都写下来了**，没有藏：
  [`docs/known_issues.md`](known_issues.md)，
  以及两环境拆分背后的 torch/vLLM ABI 分析
  [`docs/vllm_install_notes.md`](vllm_install_notes.md)。
- 结果表会标注哪个数字是人工覆盖的以及原因。例如 Hunyuan3 的
  FixedK-k2 在 MMLU-Pro 和 GSM8K 上被标为未测量。

## 许可

[MIT](../LICENSE)。`3rdparty/` 下 vendored 的评测 harness
各自沿用自己的许可，清单见 [NOTICE](../NOTICE)。

## 引用

```bibtex
@misc{chen2026need23chosenexperts,
      title={You Only Need 2/3 of the Chosen Experts: An Empirical Study of Dynamic Expert Pruning in Fine-Grained MoE LLMs}, 
      author={Yuanteng Chen and Qiwei Lai and Chen Tianqi and Peisong Wang and Yuantian Shao and Nanxin Zeng and Zhilei Liu and Chuangyi Li and Jing Liu and Jian Cheng},
      year={2026},
      eprint={2609.25809},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.25809}, 
}
```
