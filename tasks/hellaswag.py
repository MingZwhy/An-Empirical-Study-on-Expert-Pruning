"""HellaSwag 修复版（见 lighteval#275）。

lighteval 内置的 ``hellaswag`` 在 Instruct / chat 模型上普遍 0% 准确率，原因有二：

1. ``choices=[" A", " B", ...]`` 前面带空格，而模型通常输出 ``"A"`` 无前导空格，
   直接字符串比对不匹配。
2. Instruct 模型常常输出整段解释（例如 ``"The correct answer is: **A**. ..."``），
   需要从文本中提取选项字母再与 gold 比较。

此处同时修正了以上两点，并把指标命名为 ``acc``（准确率）——与多项选择语义
一致。
"""
import re
from string import ascii_uppercase

import numpy as np

from lighteval.metrics.metrics import SampleLevelMetric
from lighteval.metrics.metrics_sample import ExactMatches
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


def _hellaswag_extract_choice(text: str) -> str:
    """从模型输出中提取选项字母 A/B/C/D，用于与 gold 比较。"""
    if not text:
        return ""
    text = text.strip()
    # 只看前一段，避免从后文解释里误匹配
    head = text[:800] if len(text) > 800 else text
    # 优先：**A** / **A.** 或 Answer: A / Correct Answer: A / correct answer is A
    m = re.search(r"\*\*([A-D])(?:\.|\*\*)?", head)
    if m:
        return m.group(1).upper()
    m = re.search(
        r"(?:^|\n)\s*(?:Answer|Correct answer|correct answer)\s*[:\s]+\*?\s*([A-D])\b",
        head,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:answer|choice)\s+is\s*[:\s]*\*?\s*([A-D])\b", head, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 首行单独一个字母如 "A." 或 "A)"
    m = re.search(r"^\s*([A-D])[\s.)]", head, re.MULTILINE)
    if m:
        return m.group(1).upper()
    # 兜底：第一个出现的独立字母 A-D
    m = re.search(r"\b([A-D])\b", head)
    if m:
        return m.group(1).upper()
    return ""


def hellaswag_fixed_prompt(line, task_name: str = None):
    query = "The following are multiple choice questions (with answers) about common sense.\n\n"
    query += f"Question: {line['activity_label']}: {line['ctx_a']} {line['ctx_b'].capitalize()}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(ascii_uppercase, line["endings"])])
    query += "Answer:"
    gold_ix = int(line["label"]) if line["label"] != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=[i for i in ascii_uppercase[: len(line["endings"])]],
        gold_index=gold_ix,
        instruction="The following are multiple choice questions (with answers) about common sense.\n\n",
    )


hellaswag_fixed_metric = SampleLevelMetric(
    metric_name="acc",
    sample_level_fn=ExactMatches(
        strip_strings=True,
        normalize_pred=_hellaswag_extract_choice,
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


hellaswag_fixed = LightevalTaskConfig(
    name="hellaswag_fixed",
    prompt_function=hellaswag_fixed_prompt,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[hellaswag_fixed_metric],
    stop_sequence=["\n"],
    version=0,
)


TASKS_TABLE = [hellaswag_fixed]
