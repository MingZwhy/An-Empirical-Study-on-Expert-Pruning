"""Chat-template settings that can differ between the tasks of one run.

Several checkpoints take their reasoning protocol from the chat template rather than from a sampling
parameter: Hy3 reads ``reasoning_effort``, Gemma 4 reads ``enable_thinking``, and DeepSeek-V4 ships no
template at all, only an encoder module next to its weights. lighteval renders every prompt with
``tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`` and passes
nothing else, so this module supplies the rest: keyword arguments for every task, overrides for named
tasks, and optionally an encoder that stands in for the template.

The paper's runs set one protocol per invocation and split the suite by task family (Hy3 and
DeepSeek-V4: low effort for MMLU-Pro and GSM8K, high for the other generative tasks). Keying the
overrides by task renders the same prompts from a single invocation.
"""
from __future__ import annotations

import contextvars
import importlib.util
import json
from pathlib import Path

_TASK: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "expert_pruning_chat_task", default=None)


def task_key(task_name: str) -> str:
    """lighteval names a task ``name|fewshot`` or ``suite|name|fewshot``; keep the name."""
    parts = task_name.split("|")
    return parts[1] if len(parts) >= 3 else parts[0]


def _parse_object(text: str | None, flag: str) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag} 不是合法的 JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{flag} 须为 JSON 对象，得到 {type(value).__name__}")
    return value


class ChatProtocol:
    def __init__(self, common: dict | None = None, by_task: dict | None = None,
                 encoder: str | None = None):
        self.common = dict(common or {})
        self.by_task = {name: dict(kwargs) for name, kwargs in (by_task or {}).items()}
        self.encoder_path = encoder
        self.encode = _load_encoder(encoder) if encoder else None
        self._reported: set[str | None] = set()

    @classmethod
    def from_args(cls, args) -> "ChatProtocol":
        by_task = _parse_object(args.chat_template_kwargs_by_task, "--chat_template_kwargs_by_task")
        for name, kwargs in by_task.items():
            if not isinstance(kwargs, dict):
                raise ValueError(f"--chat_template_kwargs_by_task 中 {name!r} 的值须为 JSON 对象")
        encoder = args.chat_encoder
        if encoder and not Path(encoder).is_absolute():
            encoder = str(Path(args.model_path) / encoder)
        if encoder and not Path(encoder).is_file():
            raise ValueError(f"--chat_encoder 指向的文件不存在: {encoder}")
        return cls(_parse_object(args.chat_template_kwargs, "--chat_template_kwargs"),
                   by_task, encoder)

    def __bool__(self) -> bool:
        return bool(self.common or self.by_task or self.encoder_path)

    def check_tasks(self, datasets: list[str]) -> None:
        """Name the overrides this run will not use. A model's settings cover its whole suite and
        a run often takes a subset, so this is not an error, but a misspelt task name shows up
        here instead of quietly leaving that task on the default protocol."""
        known = set(datasets) | {name.split(":")[0] for name in datasets}
        unused = sorted(set(self.by_task) - known)
        if unused:
            print(f"[chat_protocol] --chat_template_kwargs_by_task 中 {unused} 不在本次 --datasets "
                  f"({','.join(datasets)}) 里，不起作用", flush=True)

    def kwargs_for(self, task_name: str | None) -> dict:
        kwargs = dict(self.common)
        if task_name:
            key = task_key(task_name)
            override = self.by_task.get(key, self.by_task.get(key.split(":")[0]))
            if override:
                kwargs.update(override)
        return kwargs

    def describe(self) -> dict:
        return {"chat_template_kwargs": self.common,
                "chat_template_kwargs_by_task": self.by_task,
                "chat_encoder": self.encoder_path}

    def _report(self, task_name: str | None, kwargs: dict) -> None:
        if task_name in self._reported:
            return
        self._reported.add(task_name)
        via = f"encoder {Path(self.encoder_path).name}" if self.encode else "chat template"
        print(f"[chat_protocol] {task_name or '(no task)'}: {kwargs} via {via}", flush=True)


def _load_encoder(path: str):
    spec = importlib.util.spec_from_file_location("expert_pruning_chat_encoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encode = getattr(module, "encode_messages", None)
    if encode is None:
        raise ValueError(f"{path} 没有定义 encode_messages")
    return encode


def install(protocol: ChatProtocol) -> None:
    """Route every chat-template call in this process through ``protocol``.

    Prompts are rendered in the process that runs the harness, never in the engine's workers, so
    patching here is enough. Arguments the caller passes explicitly win over the protocol's.
    """
    from lighteval.tasks.prompt_manager import PromptManager
    from transformers import PreTrainedTokenizerBase

    prepare_prompt = PromptManager.prepare_prompt

    def prepare_prompt_for_task(self, doc):
        token = _TASK.set(getattr(doc, "task_name", None))
        try:
            return prepare_prompt(self, doc)
        finally:
            _TASK.reset(token)

    PromptManager.prepare_prompt = prepare_prompt_for_task

    apply_chat_template = PreTrainedTokenizerBase.apply_chat_template

    def apply_chat_template_with_protocol(self, conversation, *args, **kwargs):
        task_name = _TASK.get()
        extra = protocol.kwargs_for(task_name)
        protocol._report(task_name, extra)
        if protocol.encode is None:
            for key, value in extra.items():
                kwargs.setdefault(key, value)
            return apply_chat_template(self, conversation, *args, **kwargs)
        if conversation and isinstance(conversation[0], list):
            return [apply_chat_template_with_protocol(self, item, *args, **kwargs)
                    for item in conversation]
        prompt = protocol.encode([dict(message) for message in conversation], **extra)
        if kwargs.get("tokenize", False):
            return self(prompt, add_special_tokens=False)["input_ids"]
        return prompt

    PreTrainedTokenizerBase.apply_chat_template = apply_chat_template_with_protocol
