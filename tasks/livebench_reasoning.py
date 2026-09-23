"""LiveBench 2024-11-25 reasoning tasks for lighteval.

Only the reasoning category is registered here.  By default this task loads
``livebench/reasoning`` from Hugging Face.  Set ``LIVEBENCH_REASONING_DATA_DIR``
to a local dataset directory when running in an environment without Hub access.

LiveBench prompts already include their task-specific answer-format
instructions in the first turn, so the prompt function intentionally passes
that turn through without adding an extra wrapper.
"""
from __future__ import annotations

import ast
import json
import os
import re
import string
from datetime import date, datetime
from functools import partial
from typing import Any

import numpy as np

from lighteval.metrics.metrics import SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


LIVEBENCH_RELEASE_DATE = "2024-11-25"
LIVEBENCH_REASONING_TASKS = ("web_of_lies_v2", "zebra_puzzle", "spatial")
LIVEBENCH_REASONING_DATA_DIR = os.environ.get("LIVEBENCH_REASONING_DATA_DIR")
LIVEBENCH_REASONING_HF_REPO = "livebench/reasoning"


def _date_key(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else text


def _available_on_release(line: dict[str, Any], release_date: str = LIVEBENCH_RELEASE_DATE) -> bool:
    """Select examples active in the requested LiveBench public release."""
    released = _date_key(line.get("livebench_release_date"))
    removed = _date_key(line.get("livebench_removal_date"))
    if released is not None and released > release_date:
        return False
    if removed is not None and removed <= release_date:
        return False
    return True


def _task_filter(task: str, line: dict[str, Any]) -> bool:
    return line.get("task") == task and _available_on_release(line)


def _first_turn(line: dict[str, Any]) -> str:
    turns = line.get("turns")
    if isinstance(turns, list) and turns:
        first = turns[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    for key in ("prompt", "question", "input"):
        if line.get(key) not in (None, ""):
            return str(line[key])
    raise KeyError("LiveBench reasoning row has no usable prompt field.")


def _ground_truth(line: dict[str, Any]) -> Any:
    for key in ("ground_truth", "answer", "target"):
        if key in line and line[key] not in (None, ""):
            return line[key]
    raise KeyError("LiveBench reasoning row has no ground-truth field.")


def _as_gold_list(value: Any) -> list[str]:
    if isinstance(value, dict):
        for key in ("answer", "answers", "solution", "target", "value"):
            if key in value:
                return _as_gold_list(value[key])
        return [json.dumps(value, sort_keys=True, ensure_ascii=False)]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def record_to_sample(record: dict[str, Any]):
    from inspect_ai.dataset import Sample

    return Sample(input=_first_turn(record), target="\n".join(_as_gold_list(_ground_truth(record))))


def livebench_reasoning_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    golds = _as_gold_list(_ground_truth(line))
    return Doc(
        task_name=task_name,
        query=_first_turn(line),
        choices=[golds],
        gold_index=0,
        specific={
            "question_id": line.get("question_id"),
            "category": line.get("category", "reasoning"),
            "subtask": line.get("task"),
            "livebench_release_date": _date_key(line.get("livebench_release_date")),
            "livebench_removal_date": _date_key(line.get("livebench_removal_date")),
            "ground_truth": golds,
        },
    )


def _strip_outer_punctuation(text: str) -> str:
    punctuation = string.punctuation.replace("-", "")
    return text.strip().strip(punctuation).strip()


def _extract_tagged(text: str, tag: str) -> list[str]:
    pattern = rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>"
    return [match.strip() for match in re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)]


def _candidate_predictions(text: str) -> list[str]:
    if not text:
        return [""]

    candidates: list[str] = []
    for tag in ("solution", "answer", "final_answer", "final"):
        candidates.extend(_extract_tagged(text, tag))

    candidates.extend(match.strip() for match in re.findall(r"\*\*(.*?)\*\*", text, flags=re.DOTALL))

    answer_markers = (
        r"(?:final\s+answer|answer|solution)\s*(?:is)?\s*[:：]\s*(.+)",
        r"(?:therefore|thus),?\s+(.+)",
    )
    for pattern in answer_markers:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            candidates.append(str(match).strip())

    # The last non-empty line is a useful fallback for prompts that ask for a
    # final answer line but the model does not use tags.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        candidates.append(lines[-1])
    candidates.append(text)
    return candidates


def _maybe_literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def _normalize_value(value: Any) -> str:
    if isinstance(value, str):
        text = value
        parsed = _maybe_literal(text)
        if parsed is not None and not isinstance(parsed, str):
            return _normalize_value(parsed)
    elif isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    elif isinstance(value, (list, tuple)):
        return json.dumps([_normalize_value(item) for item in value], ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value)

    text = re.sub(r"```(?:\w+)?|```", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\boxed", " ")
    text = re.sub(r"\s+", " ", text)
    text = _strip_outer_punctuation(text.lower())
    return text


def _boolean_aliases(text: str) -> set[str]:
    norm = _normalize_value(text)
    if norm in {"yes", "true", "correct", "1"}:
        return {"yes", "true", "correct", "1"}
    if norm in {"no", "false", "incorrect", "0"}:
        return {"no", "false", "incorrect", "0"}
    return {norm}


def _matches_any_gold(prediction: str, golds: list[str]) -> bool:
    pred_norm = _normalize_value(prediction)
    pred_aliases = _boolean_aliases(prediction)

    for gold in golds:
        gold_norm = _normalize_value(gold)
        if pred_norm == gold_norm:
            return True
        if pred_aliases & _boolean_aliases(gold):
            return True

    return False


class LiveBenchReasoningAccuracy(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> int:
        predictions = model_response.final_text or [""]
        golds = list(doc.specific.get("ground_truth", [])) if doc.specific else []
        if not golds:
            golds = doc.get_golds()

        for output in predictions:
            for candidate in _candidate_predictions(output):
                if _matches_any_gold(candidate, golds):
                    return 1
        return 0


livebench_reasoning_acc = SampleLevelMetric(
    metric_name="acc",
    sample_level_fn=LiveBenchReasoningAccuracy(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


def _make_livebench_reasoning_task(subtask: str) -> LightevalTaskConfig:
    dataset_path = LIVEBENCH_REASONING_DATA_DIR or LIVEBENCH_REASONING_HF_REPO
    return LightevalTaskConfig(
        name=f"livebench_20241125:{subtask}",
        prompt_function=livebench_reasoning_prompt,
        sample_fields=record_to_sample,
        hf_repo=dataset_path,
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=4096,
        metrics=[livebench_reasoning_acc],
        stop_sequence=[],
        hf_filter=partial(_task_filter, subtask),
        version=0,
    )


TASKS_TABLE = [_make_livebench_reasoning_task(subtask) for subtask in LIVEBENCH_REASONING_TASKS]
