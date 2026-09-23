"""本地自定义任务集合。

将原始仓库 (``Half-Experts-Candoall``) 中堆在 ``aime_custom.py`` 一个文件里的
任务，按数据集拆到独立模块，便于按需复用：

- :mod:`tasks.aime`          AIME 2024/2025 avg@N 评估
- :mod:`tasks.triviaqa`      TriviaQA 上修正后的 EM
- :mod:`tasks.simpleqa`      SimpleQA (HF 当前列名 ``problem``/``answer``)
- :mod:`tasks.hellaswag`     HellaSwag 修复版 (见 lighteval#275)
- :mod:`tasks.if_pass_at_n`  IFEval / IFBench 的 pass@n

每个模块都暴露自己的 ``TASKS_TABLE``。:mod:`tasks.custom_tasks` 汇总所有
模块的 ``TASKS_TABLE``，是 ``--custom_tasks`` 的默认入口。
"""
