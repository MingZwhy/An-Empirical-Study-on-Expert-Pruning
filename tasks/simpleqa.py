"""SimpleQA 上适配当前数据集 schema + 修正后的 EM。

HuggingFace 上的 ``lighteval/SimpleQA`` 已经把列名改成了 ``problem`` /
``answer``，而 lighteval 内置任务仍按旧列名 (``question`` / ``choices``) 取
字段，直接跑会 KeyError。

同时，模型常输出整句话，单纯的整句精确匹配 EM≈0，所以这里也走「规范化 +
后缀匹配」，与 :mod:`tasks.triviaqa` 保持一致。
"""
import numpy as np

from lighteval.metrics.metrics import SampleLevelMetric
from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.metrics.normalizations import harness_triviaqa_normalizer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


def simpleqa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['problem']}\nAnswer:",
        gold_index=0,
        choices=[[line["answer"]]],
    )


simpleqa_v2_em_metric = SampleLevelMetric(
    metric_name="em",
    sample_level_fn=ExactMatches(
        strip_strings=True,
        normalize_gold=harness_triviaqa_normalizer,
        normalize_pred=harness_triviaqa_normalizer,
        type_exact_match="suffix",
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


simpleqa_v2 = LightevalTaskConfig(
    name="simpleqa_v2",
    prompt_function=simpleqa_prompt,
    hf_repo="lighteval/SimpleQA",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="few_shot",
    few_shots_select=None,
    generation_size=2048,
    metrics=[simpleqa_v2_em_metric],
    stop_sequence=["\n"],
    version=0,
)


TASKS_TABLE = [simpleqa_v2]
