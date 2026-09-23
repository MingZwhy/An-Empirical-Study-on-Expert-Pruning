"""AIME 2024 / 2025 的自定义任务（avg@N）。

- ``aime24_avg`` / ``aime25_avg``：avg@16
- ``aime24_avg8`` / ``aime25_avg8``：avg@8
- ``aime24_avg5`` / ``aime25_avg5``：avg@5（更省时间的小规模对比）

采样数从 64 降到 16（2026-08-12）。这两个任务合起来只有 60 道题，但按 64 次采样就是
3840 条请求，而在会先思考再作答的模型上，这一组比 mmlu_pro 的 12032 条还贵：实测
Qwen3-30B-A3B-Thinking-2507 的 FixedK-k5 在 mmlu_pro 上用了 5.0 小时，aime 这一组要
11.8 小时，占整套九个数据集的一半以上。16 次采样对 avg@n 这个指标已经够稳，换来整套
时长从 mmlu_pro 阶段的约 4 倍降到约 2.25 倍。

任务名故意不变。avg@16 与 avg@64 估计的是同一个量，报告里不区分这两批数字；结果 JSON
里的指标键会从 ``avg@n:n=64`` 变成 ``avg@n:n=16``，而报告脚本按任务名取该任务下所有
数值型指标，不认键名，所以已完成的四个模型的表不受影响。

Prompt 与 ``lighteval/tasks/tasks/aime.py`` 内置版本一致。

说明：上游 ``lighteval.tasks.tasks.aime`` 虽然写了 ``aime24_avg`` / ``aime25_avg``
对象，但**没有**把它们放进该模块的 ``TASKS_TABLE``，因此 lighteval 注册表里
实际上加载不到这两个名字。这里在自定义 ``TASKS_TABLE`` 里显式注册，才能在
``--datasets aime24_avg,aime25_avg`` 时使用。
"""
from textwrap import dedent

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# 与 lighteval/tasks/tasks/aime.py 内置 prompt 保持一致
MATH_PROMPT_TEMPLATE = dedent("""
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{prompt}
""").strip()


def record_to_sample(record):
    from inspect_ai.dataset import Sample

    return Sample(input=record["problem"], target=record["answer"])


def aime_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT_TEMPLATE.format(prompt=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def _make_aime(name: str, hf_repo: str, n: int) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=name,
        prompt_function=aime_prompt,
        sample_fields=record_to_sample,
        hf_repo=hf_repo,
        hf_subset="default",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=None,
        metrics=[Metrics.avg_at_n_math(sample_params={"n": n})],
        version=2,
    )


aime24_avg = _make_aime("aime24_avg", "HuggingFaceH4/aime_2024", n=16)
aime25_avg = _make_aime("aime25_avg", "yentinglin/aime_2025", n=16)
aime24_avg8 = _make_aime("aime24_avg8", "HuggingFaceH4/aime_2024", n=8)
aime25_avg8 = _make_aime("aime25_avg8", "yentinglin/aime_2025", n=8)
aime24_avg5 = _make_aime("aime24_avg5", "HuggingFaceH4/aime_2024", n=5)
aime25_avg5 = _make_aime("aime25_avg5", "yentinglin/aime_2025", n=5)


TASKS_TABLE = [
    aime24_avg,
    aime25_avg,
    aime24_avg8,
    aime25_avg8,
    aime24_avg5,
    aime25_avg5,
]
