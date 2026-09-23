"""Calibration dataset helpers.

The public entry point is ``get_loaders``.  It returns the common GPTQ/AWQ
style calibration list ``[(input_ids, labels), ...]``, where each tensor has
shape ``[1, seqlen]`` and labels mask every token except the last one.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterable


def _require_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise ImportError(
            "calib_utils.data_utils requires the `datasets` package. "
            "Please run this in the expertpruning environment or install "
            "`datasets`.") from exc
    return datasets


def _require_transformers():
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "calib_utils.data_utils requires the `transformers` package."
        ) from exc
    return transformers


def _tokenizer_kwargs(hf_token: str | None = None,
                      use_fast: bool = False) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"use_fast": use_fast}
    if hf_token is not None:
        kwargs["token"] = hf_token
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        kwargs["local_files_only"] = True
    return kwargs


def _load_tokenizer(model: str, hf_token: str | None = None):
    transformers = _require_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model,
        **_tokenizer_kwargs(hf_token, use_fast=False),
    )
    if callable(tokenizer):
        return tokenizer

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model,
        **_tokenizer_kwargs(hf_token, use_fast=True),
    )
    if callable(tokenizer):
        return tokenizer
    raise TypeError(
        "AutoTokenizer.from_pretrained returned a non-callable object for "
        f"{model}: {type(tokenizer).__name__}.")


def _safe_window_start(num_tokens: int, seqlen: int) -> int:
    if num_tokens < seqlen:
        raise ValueError(
            f"Tokenized sample is shorter than seqlen: {num_tokens} < {seqlen}")
    return random.randint(0, num_tokens - seqlen)


def _read_text_file(path: str | Path) -> str:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return f.read()


def _resolve_split_file(local_data_path: str | None, split: str,
                        env_name: str) -> Path | None:
    env_path = os.environ.get(env_name)
    if env_path:
        return Path(env_path).expanduser()
    if not local_data_path:
        return None

    base = Path(local_data_path).expanduser()
    if base.is_file():
        return base
    candidates = [
        base / f"{split}.txt",
        base / f"ptb.{split}.txt",
        base / f"penn_treebank_{split}.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _make_lm_sample(input_ids, start: int, seqlen: int):
    inp = input_ids[:, start:start + seqlen]
    tar = inp.clone()
    tar[:, :-1] = -100
    return inp, tar


def _sample_from_tokenized_stream(input_ids, nsamples: int, seed: int,
                                  seqlen: int):
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        start = _safe_window_start(input_ids.shape[1], seqlen)
        trainloader.append(_make_lm_sample(input_ids, start, seqlen))
    return trainloader


def _build_token_stream_from_texts(tokenizer, texts, min_tokens: int):
    chunks = []
    token_count = 0
    for text in texts:
        if text is None or text == "":
            continue
        chunks.append(text)
        if len(chunks) % 256 == 0:
            enc = tokenizer(" ".join(chunks), return_tensors="pt")
            token_count = int(enc.input_ids.shape[1])
            if token_count >= min_tokens:
                return enc
    enc = tokenizer(" ".join(chunks), return_tensors="pt")
    token_count = int(enc.input_ids.shape[1])
    if token_count < min_tokens:
        raise ValueError(
            f"Only found {token_count} C4 tokens, need at least {min_tokens}.")
    return enc


def _load_dataset(*args, **kwargs):
    datasets = _require_datasets()
    return datasets.load_dataset(*args, **kwargs)


def _load_dataset_from_arrow_files(files: Iterable[Path]):
    datasets = _require_datasets()
    arrow_files = [str(path) for path in sorted(files) if path.is_file()]
    if not arrow_files:
        raise FileNotFoundError("No cached Arrow files were found.")
    pieces = [datasets.Dataset.from_file(path) for path in arrow_files]
    if len(pieces) == 1:
        return pieces[0]
    return datasets.concatenate_datasets(pieces)


def _cached_hf_dataset_root(dataset_dir: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" /
                                  "huggingface"))
    return hf_home / "datasets" / dataset_dir


def _load_cached_c4_split(split: str):
    """Load cached C4 Arrow shards when ``load_dataset`` misses the cache key."""
    cache_root = _cached_hf_dataset_root("allenai___c4")
    if split == "train":
        pattern = "c4-train-*.arrow"
    elif split == "validation":
        pattern = "c4-validation*.arrow"
    else:
        raise ValueError(f"Unsupported C4 split: {split}")
    return _load_dataset_from_arrow_files(cache_root.glob(f"**/{pattern}"))


def get_wikitext2(nsamples,
                  seed,
                  seqlen,
                  model,
                  hf_token=None,
                  eval_mode=False):
    tokenizer = _load_tokenizer(model, hf_token)

    if eval_mode:
        testdata = _load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        return tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    traindata = _load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    return _sample_from_tokenized_stream(trainenc.input_ids, nsamples, seed,
                                         seqlen)


def get_c4_new(nsamples,
               seed,
               seqlen,
               model,
               hf_token=None,
               eval_mode=False):
    tokenizer = _load_tokenizer(model, hf_token)

    if eval_mode:
        try:
            valdata = _load_dataset(
                "allenai/c4",
                data_files={
                    "validation": "en/c4-validation.00000-of-00008.json.gz"
                },
                split="validation",
            )
        except Exception:
            valdata = _load_cached_c4_split("validation")

        valenc = tokenizer(" ".join(valdata[:1100]["text"]),
                           return_tensors="pt")
        valenc = valenc.input_ids[:, :(256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        return TokenizerWrapper(valenc)

    try:
        traindata = _load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
    except Exception:
        traindata = _load_cached_c4_split("train")

    min_tokens = nsamples * seqlen + 1
    trainenc = _build_token_stream_from_texts(tokenizer, traindata["text"],
                                             min_tokens)
    return _sample_from_tokenized_stream(trainenc.input_ids, nsamples, seed,
                                         seqlen)


def get_ptb_new(nsamples,
                seed,
                seqlen,
                model,
                hf_token=None,
                eval_mode=False,
                local_data_path: str | None = None):
    tokenizer = _load_tokenizer(model, hf_token)

    local_file = _resolve_split_file(
        local_data_path,
        "test" if eval_mode else "train",
        "CALIB_PTB_TEST_FILE" if eval_mode else "CALIB_PTB_TRAIN_FILE",
    )
    if local_file is not None:
        enc = tokenizer(_read_text_file(local_file), return_tensors="pt")
        if eval_mode:
            return enc
        return _sample_from_tokenized_stream(enc.input_ids, nsamples, seed,
                                             seqlen)

    try:
        if eval_mode:
            testdata = _load_dataset(
                "ptb_text_only",
                "penn_treebank",
                split="test",
            )
            return tokenizer(" ".join(testdata["sentence"]),
                             return_tensors="pt")

        traindata = _load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="train",
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load PTB (`ptb_text_only`, `penn_treebank`). "
            "This dataset is not currently cached in the local Hugging Face "
            "datasets cache; run once with network access, set "
            "`CALIB_PTB_TRAIN_FILE` / `CALIB_PTB_TEST_FILE`, or pass "
            "`local_data_path` pointing to a PTB text file/directory.") from exc

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    return _sample_from_tokenized_stream(trainenc.input_ids, nsamples, seed,
                                         seqlen)


def get_loaders(name,
                nsamples=128,
                seed=0,
                seqlen=2048,
                model="",
                hf_token=None,
                eval_mode=False,
                local_data_path: str | None = None):
    dataset_name = name.lower()
    if "wikitext2" in dataset_name or "wikitext" in dataset_name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token,
                             eval_mode)
    if "ptb" in dataset_name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode,
                           local_data_path)
    if "c4" in dataset_name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    raise ValueError(
        f"Unknown calibration dataset `{name}`. Supported: wikitext2, ptb, c4."
    )
