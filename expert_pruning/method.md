# Expert Pruning Methods

本文档简要记录当前仓库已复现的三种动态专家剪枝方法。运行时需要先启用本地 router：

```bash
--use_local_expert_router
```

然后通过 `--expert_pruning_method` 选择方法：

```bash
--expert_pruning_method {none,NAEE,Dynamic_Routing,DiEP,MC_MoE,EAC_MoE,Ban}
```

## NAEE

对每个 token，先按 vLLM 原逻辑从所有专家中选出 top-k 个专家，并得到经过 softmax 的权重：

```text
w1 >= w2 >= ... >= wk
```

NAEE 使用手动设置的 `beta` 判断弱专家：若某个位置满足

```text
wi < w1 * beta
```

则从该位置开始剪枝后续专家。为了避免剪得过狠，可以设置最小保留专家数 `k_min`，即前 `k_min` 个专家始终保留。

相关参数：

- `--expert_pruning_method NAEE`
- `--naee_beta`：剪枝阈值系数，默认 `0.3`
- `--naee_k_min`：每个 token 至少保留的专家数，默认 `2`

## Dynamic_Routing

对每个 token，先得到 top-k 专家的 softmax 权重，并按权重从大到小排序。Dynamic_Routing 保留累计概率达到阈值 `p` 的最短专家前缀。

例如：

```text
w1 + w2 < p, 但 w1 + w2 + w3 >= p
```

则保留前三个专家，其余专家剪枝。

相关参数：

- `--expert_pruning_method Dynamic_Routing`
- `--dynamic_routing_threshold`：累计概率阈值 `p`，默认 `0.8`
- `--dynamic_routing_score_source`：累计概率使用的分数来源，默认 `raw`

`--dynamic_routing_score_source` 有两个选项：

- `raw`：使用 full-softmax 后的 top-k 原始概率做累计判断
- `renormalized`：先对 top-k 权重重新归一化，再做累计判断；剪枝后仍会基于原始 top-k scores 重新归一化

## DiEP

DiEP 可以看作加入专家相似度修正的 NAEE。它仍然使用手动设置的 `beta`，但阈值会乘上校准集得到的专家相似度因子 `gamma2`。

对当前 token，设 top-1 专家为 `e1`，待判断专家为 `ei`，则：

```text
gamma2 = sim_matrix[layer, e1, ei] / mean_sim[layer]
```

剪枝判断为：

```text
wi < w1 * beta * gamma2
```

因此，当两个专家相似度高于该层平均相似度时，`gamma2 > 1`，剪枝阈值会变高，该专家更容易被剪掉。

相关参数：

- `--expert_pruning_method DiEP`
- `--naee_beta`：DiEP 复用 NAEE 的手动 beta，默认 `0.3`
- `--naee_k_min`：每个 token 至少保留的专家数，默认 `2`
- `--diep_artifact_path`：DiEP 校准 artifact 路径，使用 DiEP 时必须提供
- `--diep_pruning_mode`：剪枝模式，默认 `independent`

`--diep_pruning_mode` 有两个选项：

- `independent`：每个非强制保留专家按自己的 `gamma2` 独立判断，默认选项
- `prefix`：一旦某个位置被剪枝，则该位置之后的所有专家都剪枝，更接近 NAEE 的前缀式剪枝

常用示例：

```bash
--use_local_expert_router \
--expert_pruning_method DiEP \
--naee_beta 0.3 \
--naee_k_min 2 \
--diep_pruning_mode independent \
--diep_artifact_path calib_utils/results/Qwen3-30B-A3B-Instruct-2507/c4_diep.pt
```

## MC_MoE

MC_MoE 基于 NAEE，但会先用当前层在线 attention 信息保护少量重要 token。对未保护 token，剪枝规则与 NAEE 相同；对保护 token，保留该 token 原本选中的全部 top-k 专家。

当前适配到细粒度 MoE 的重要性计算为：

```text
importance[j] = ||hidden_states[j]||_1 * attention_score[j]
```

其中 `attention_score[j]` 来自当前 attention 层中 token `j` 被其他 token 关注的平均分数。每个连续序列段内保护 top `ceil(protection_ratio * seq_len)` 个 token，默认比例为 `0.02`。

相关参数：

- `--expert_pruning_method MC_MoE`
- `--naee_beta`：复用 NAEE 的手动 beta，默认 `0.3`
- `--naee_k_min`：未保护 token 至少保留的专家数，默认 `2`
- `--mc_moe_protection_ratio`：保护 token 比例，默认 `0.02`
- `--attention_sink_probe_max_query_tokens` / `--attention_sink_probe_max_key_tokens`：在线 attention 统计窗口，默认均为 `2048`

使用 `MC_MoE` 时会自动启用 `--enforce_eager`，因为该方法需要在线 attention 信息。

## EAC_MoE

EAC_MoE 只在 prefill/chunked-prefill 阶段做专家剪枝。设当前连续序列段长度为 `l`，该层专家数为 `n`，每个 token 默认选择 `k` 个专家。若完全均匀分配，每个专家应接收：

```text
l * k / n
```

个 token 分配。引入超参 `alpha` 后，如果某专家实际被选中的 token 次数满足：

```text
count(expert) < l * k / n * alpha
```

则该专家在当前 prefill 段内被禁用。实现上会将所有选择该专家的 token 对应该专家权重置为 0，然后重新归一化。decode 阶段不应用该剪枝。

相关参数：

- `--expert_pruning_method EAC_MoE`
- `--eac_alpha`：专家活跃阈值系数，默认 `0.5`

`EAC_MoE` 不需要 `--enforce_eager`。实现会优先读取 vLLM forward context 中的 `attn_metadata.num_prefill_tokens` 来区分 prefill 与 decode；若能安全读取 `query_start_loc`，则按每个 prefill request 切段，否则将当前 forward 的全部 prefill tokens 作为一个段处理。

## Ban

Ban 先在校准集上计算每层敏感度，再在推理时结合当前 token 的 router 权重动态决定保留专家数。

校准阶段默认使用 C4，得到：

- `layer_sensitivity[l]`：每层 top8 与“仅该层 top3”输出 logits 的 top1000 KL 差异归一化结果
- `r_min` / `r_max`：校准集中所有 token、所有层的 `sum(top3) / sum(top8)` 最小值和最大值

推理时，对每个 token、每层：

```text
R_i = sum(top3_weights) / sum(topk_weights)
T_i = (R_max - R_i) / (R_max - R_min)
S_i_l = lambda * 0.5 * (layer_sensitivity[l] + T_i)
K_i_l = round(k_min + (k_base - k_min) * S_i_l)
```

然后只保留当前 top-k 中前 `K_i_l` 个专家。`k_base` 直接使用模型默认专家选择数，不作为超参。

相关参数：

- `--expert_pruning_method Ban`
- `--ban_artifact_path`：Ban 校准 artifact 路径，使用 Ban 时必须提供
- `--ban_lambda`：剪枝保守系数，默认 `0.7`，越大保留专家越多
- `--ban_k_min`：每个 token 至少保留专家数，默认 `3`

示例：

```bash
--expert_pruning_method Ban \
--ban_artifact_path calib_utils/results/Qwen3-30B-A3B-Instruct-2507/c4_ban.pt \
--ban_lambda 0.7 \
--ban_k_min 3
```
