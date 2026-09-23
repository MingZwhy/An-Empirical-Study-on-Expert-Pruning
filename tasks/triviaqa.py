"""TriviaQA 上修正后的 EM。

lighteval 内置的 ``triviaqa`` 使用 ``ExactMatches(strip_strings=True)`` 做全句
精确匹配，而模型通常会输出一整句话（例如 ``"Prime Minister of Canada."``），
标准答案却是规范化后的短答案（例如 ``"canada"``）——这会导致 EM≈0。

本任务改用「规范化 + 后缀匹配」：只要答案出现在模型输出的末尾就算命中。
"""
import string

import numpy as np

from lighteval.metrics.metrics import SampleLevelMetric
from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.metrics.normalizations import harness_triviaqa_normalizer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


def _triviaqa_remove_prefixes(aliases):
    aliases = sorted(aliases)
    ret = [aliases[0]]
    for alias in aliases[1:]:
        if not alias.startswith(ret[-1]):
            ret.append(alias)
    return ret


def triviaqa_prompt(line, task_name: str = None):
    list_of_candidates = [
        alias.lower().translate(str.maketrans("", "", string.punctuation))
        for alias in _triviaqa_remove_prefixes(line["answer"]["aliases"])
    ]
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        gold_index=0,
        choices=[list_of_candidates],
    )


triviaqa_em_metric = SampleLevelMetric(
    metric_name="em",
    sample_level_fn=ExactMatches(
        strip_strings=True,
        normalize_pred=harness_triviaqa_normalizer,
        type_exact_match="suffix",
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


triviaqa_em = LightevalTaskConfig(
    name="triviaqa_em",
    prompt_function=triviaqa_prompt,
    hf_repo="mandarjoshi/trivia_qa",
    hf_subset="rc.nocontext",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=20,
    metrics=[triviaqa_em_metric],
    stop_sequence=["\n", ".", ","],
    version=0,
)


TASKS_TABLE = [triviaqa_em]
