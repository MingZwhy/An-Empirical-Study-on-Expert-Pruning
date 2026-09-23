"""自定义任务的总入口——默认给 ``main.py --custom_tasks`` 使用。

聚合所有 ``tasks/*.py`` 的 ``TASKS_TABLE``：

- :mod:`tasks.aime`          AIME avg@N
- :mod:`tasks.triviaqa`      TriviaQA EM 修正版
- :mod:`tasks.simpleqa`      SimpleQA 适配版
- :mod:`tasks.hellaswag`     HellaSwag 修复版
- :mod:`tasks.livebench_reasoning` LiveBench 2024-11-25 reasoning
- :mod:`tasks.if_pass_at_n`  IFEval / IFBench pass@n

如果某个任务需要的额外依赖没装（例如 IFEval 需要 ``langdetect``），这里会
跳过对应模块并在 stderr 给出一条警告，而不是让整个 ``--custom_tasks``
load 失败。
"""
from __future__ import annotations

import sys as _sys

from tasks import aime as _aime
from tasks import hellaswag as _hellaswag
from tasks import simpleqa as _simpleqa
from tasks import triviaqa as _triviaqa


TASKS_TABLE = []
TASKS_TABLE.extend(_aime.TASKS_TABLE)
TASKS_TABLE.extend(_triviaqa.TASKS_TABLE)
TASKS_TABLE.extend(_simpleqa.TASKS_TABLE)
TASKS_TABLE.extend(_hellaswag.TASKS_TABLE)

try:
    from tasks import livebench_reasoning as _livebench_reasoning

    TASKS_TABLE.extend(_livebench_reasoning.TASKS_TABLE)
except Exception as exc:  # pragma: no cover - 依赖/数据 schema 变化时的软失败
    print(
        f"[tasks.custom_tasks] 跳过 LiveBench reasoning 任务：{type(exc).__name__}: {exc}",
        file=_sys.stderr,
    )

try:
    from tasks import if_pass_at_n as _if_pass_at_n

    TASKS_TABLE.extend(_if_pass_at_n.TASKS_TABLE)
except Exception as exc:  # pragma: no cover - 依赖缺失时的软失败
    print(
        f"[tasks.custom_tasks] 跳过 IFEval / IFBench pass@n 任务：{type(exc).__name__}: {exc}",
        file=_sys.stderr,
    )
