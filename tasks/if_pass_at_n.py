"""IFEval / IFBench 上的 pass@n（每题 64 采样）。

与 lighteval 内置的 ``PassAtK`` 无偏估计不同，这里直接按「前 n 次采样中至少
一次整题通过」统计 pass@n。对每个 doc：

1. 用 vLLM 一次请求生成 ``N_TOTAL=64`` 条 completion（需 ``temperature>0``）。
2. 对每条 completion 算与官方任务一致的 prompt-level strict / loose
   （整题所有约束是否全部满足）。
3. 对每个 ``n ∈ KS``：``pass@n = 1`` 当且仅当前 n 条里至少有一条 strict 全对
   （loose 同理），最后在语料上取平均。

任务名（供 ``--datasets`` 使用）：

- ``ifeval_pass_at_n``
- ``ifbench_test_pass_at_n``

依赖与内置 ``ifeval`` / ``ifbench_test`` 相同（如 ``langdetect``、``spacy``、
``syllapy`` 等）。
"""
from __future__ import annotations

import numpy as np
from inspect_ai.solver import generate

from lighteval.metrics.metrics_sample import SamplingMetric, SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.ifeval.main import (
    IFEvalMetrics,
    ifeval_prompt,
    ifeval_scorer,
    record_to_sample as ifeval_record_to_sample,
)
from lighteval.tasks.tasks.ifbench.main import (
    IFBench,
    ifbench_prompt,
    ifbench_scorer,
    record_to_sample as ifbench_record_to_sample,
)

# 每题采样总数与各 pass@n 档位（n 为「前 n 条里至少一次全对」）
N_TOTAL = 64
KS = (2, 4, 8, 16, 32, 64)


def _grouping_metric_names(prefix: str) -> list[str]:
    names = []
    for k in KS:
        names.append(f"{prefix}_strict_pass@{k}")
        names.append(f"{prefix}_loose_pass@{k}")
    return names


class _IfPassAtNBase(SamplingMetric, SampleLevelComputation):
    """对单次 Doc 的 N 条生成计算 strict/loose 的 pass@k 档位。"""

    # 子类需要提供一个可调用的 evaluator 类（返回 {"prompt_level_strict_acc", "prompt_level_loose_acc"})
    evaluator_cls = None

    def __init__(self, n: int = N_TOTAL, ks: tuple[int, ...] = KS, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.ks = ks
        self._eval = self.evaluator_cls()

    def num_samples(self) -> int:
        return self.n

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> dict:
        preds = list(model_response.final_text)
        if len(preds) < self.n:
            preds = preds + [""] * (self.n - len(preds))
        preds = preds[: self.n]

        strict_bits: list[int] = []
        loose_bits: list[int] = []
        for response in preds:
            mr = ModelResponse(text=[response])
            out = self._eval.compute(doc, mr)
            strict_bits.append(int(out["prompt_level_strict_acc"]))
            loose_bits.append(int(out["prompt_level_loose_acc"]))

        row: dict[str, int] = {}
        for k in self.ks:
            row[f"prompt_strict_pass@{k}"] = 1 if any(strict_bits[:k]) else 0
            row[f"prompt_loose_pass@{k}"] = 1 if any(loose_bits[:k]) else 0
        return row


class IfevalPassAtNComputer(_IfPassAtNBase):
    evaluator_cls = IFEvalMetrics


class IfbenchPassAtNComputer(_IfPassAtNBase):
    evaluator_cls = IFBench


_pass_at_n_subnames = _grouping_metric_names("prompt")

ifeval_pass_at_n_metrics = SampleLevelMetricGrouping(
    metric_name=_pass_at_n_subnames,
    higher_is_better=dict.fromkeys(_pass_at_n_subnames, True),
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=IfevalPassAtNComputer(),
    corpus_level_fn={m: np.mean for m in _pass_at_n_subnames},
)

ifbench_pass_at_n_metrics = SampleLevelMetricGrouping(
    metric_name=_pass_at_n_subnames,
    higher_is_better=dict.fromkeys(_pass_at_n_subnames, True),
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=IfbenchPassAtNComputer(),
    corpus_level_fn={m: np.mean for m in _pass_at_n_subnames},
)


ifeval_pass_at_n = LightevalTaskConfig(
    name="ifeval_pass_at_n",
    prompt_function=ifeval_prompt,
    hf_repo="google/IFEval",
    hf_subset="default",
    metrics=[ifeval_pass_at_n_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],
    version="0.1",
    sample_fields=ifeval_record_to_sample,
    solver=[generate(cache=True)],
    scorer=ifeval_scorer(),
)

ifbench_test_pass_at_n = LightevalTaskConfig(
    name="ifbench_test_pass_at_n",
    prompt_function=ifbench_prompt,
    hf_repo="allenai/IFBench_test",
    hf_subset="default",
    metrics=[ifbench_pass_at_n_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],
    version="0.1",
    sample_fields=ifbench_record_to_sample,
    solver=[generate(cache=True)],
    scorer=ifbench_scorer(),
)


TASKS_TABLE = [ifeval_pass_at_n, ifbench_test_pass_at_n]
