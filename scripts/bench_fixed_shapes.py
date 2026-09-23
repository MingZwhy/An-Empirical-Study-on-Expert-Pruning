"""Expanded fixed-shape benchmark for vLLM and Hugging Face Transformers.

The two backends receive the same random token IDs and the same explicit
(batch size, input length, output length) cells.  ``out_len=1`` is the matching
prefill measurement for a (batch, input) pair.  Decode throughput for longer
cells subtracts that matching prefill wall time and counts the remaining
``out_len - 1`` tokens per sequence.

This benchmark intentionally measures the normal public generation API of each
backend.  Hugging Face multi-GPU runs use Accelerate layer sharding, whereas
vLLM runs use tensor parallelism; that distinction is recorded in the output.
Prefix caching is disabled in vLLM, and all runs are forced to emit exactly the
requested number of tokens.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from main import prepare_model_dir_with_num_experts_per_tok  # noqa: E402


# A broad prefill matrix rather than a Cartesian product full of redundant or
# guaranteed-OOM corners.  In particular, the requested (4, 2048) and
# (8, 1024) cases are first-class cells.
PREFILL_SHAPES = (
    (1, 256),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (2, 1024),
    (2, 2048),
    (4, 512),
    (4, 1024),
    (4, 2048),
    (8, 256),
    (8, 512),
    (8, 1024),
    (16, 256),
    (16, 512),
    (32, 256),
)

# out=32 is applied to every shape above.  These longer cells expose whether
# throughput changes as decode dominates, without making the full ladder take
# days under eager Transformers.
LONG_DECODE_CELLS = (
    (1, 256, 512),
    (1, 2048, 128),
    (4, 2048, 128),
    (8, 1024, 128),
)


def default_cells() -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for bsz, in_len in PREFILL_SHAPES:
        cells.append((bsz, in_len, 1))
        cells.append((bsz, in_len, 32))
    cells.extend(LONG_DECODE_CELLS)
    return cells


def gpu_meta() -> list[str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:  # noqa: BLE001
        return [f"{type(exc).__name__}: {exc}"]


def visible_gpu_peak_mib() -> list[float]:
    import torch

    return [
        torch.cuda.max_memory_allocated(i) / 1024**2
        for i in range(torch.cuda.device_count())
    ]


def reset_gpu_peak() -> None:
    import torch

    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def sync_gpus() -> None:
    import torch

    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def enable_gptoss_torch_routing(model: Any) -> int:
    """Use the kernels package's arbitrary-k routing fallback for gpt-oss.

    kernels-community/triton_kernels 0.12.x compiles its fast top-k router only
    when k is a power of two: ``tl.arange(0, 3)`` fails for Fixed-K=3.  The
    Transformers MXFP4 integration already contains a stable PyTorch routing
    path for distributed inference.  A one-rank process group and the same
    marker used by that integration select this path for *all* k values, so
    baseline and Fixed-K remain comparable while the expert matmuls stay MXFP4.
    """
    if getattr(model.config, "model_type", None) != "gpt_oss":
        return 0

    import torch.distributed as dist

    if not dist.is_initialized():
        init_file = f"/tmp/gptoss_routing_{os.getpid()}_{time.time_ns()}"
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
        )
    count = 0
    for module in model.modules():
        if module.__class__.__name__ == "GptOssMLP":
            module._is_hooked = True
            count += 1
    return count


def parse_cells(path: str | None) -> list[tuple[int, int, int]]:
    if path is None:
        return default_cells()
    raw = json.loads(Path(path).read_text())
    cells = []
    for item in raw:
        if isinstance(item, dict):
            cell = (item["bsz"], item["in_len"], item["out_len"])
        else:
            cell = tuple(item)
        if len(cell) != 3 or min(cell) < 1:
            raise ValueError(f"Invalid cell: {item}")
        cells.append(tuple(map(int, cell)))
    return cells


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


def time_trials(
    run_once: Callable[[], None],
    *,
    warmups: int,
    trials: int,
) -> list[float]:
    for _ in range(warmups):
        run_once()
    measured = []
    for _ in range(trials):
        sync_gpus()
        started = time.perf_counter()
        run_once()
        sync_gpus()
        measured.append(time.perf_counter() - started)
    return measured


def add_metrics(
    cell: dict[str, Any],
    prefill_s: dict[tuple[int, int], float],
) -> None:
    bsz = cell["bsz"]
    in_len = cell["in_len"]
    out_len = cell["out_len"]
    wall = cell["wall_s"]
    cell["prompt_tok"] = bsz * in_len
    cell["gen_tok"] = bsz * out_len
    if out_len == 1:
        prefill_s[(bsz, in_len)] = wall
        cell["kind"] = "prefill"
        cell["prefill_tok_s"] = bsz * in_len / wall
        return

    cell["kind"] = "prefill+decode"
    cell["total_tok_s"] = (bsz * (in_len + out_len)) / wall
    prefill = prefill_s.get((bsz, in_len))
    if prefill is not None and wall > prefill:
        cell["decode_s"] = wall - prefill
        cell["decode_tok_s"] = bsz * (out_len - 1) / (wall - prefill)
    else:
        cell["decode_s"] = None
        cell["decode_tok_s"] = None


def make_token_ids(vocab_size: int, in_len: int) -> list[int]:
    low = 100
    high = max(low + 1, vocab_size - 16)
    return [low + (i % (high - low)) for i in range(in_len)]


def run_vllm(
    args: argparse.Namespace,
    model_path: str,
    cells: list[tuple[int, int, int]],
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    from vllm import LLM, SamplingParams

    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        from vllm import TokensPrompt

    max_len = max(in_len + out_len for _, in_len, out_len in cells) + 16
    kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": args.tp,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max_len,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "seed": args.seed,
    }
    if args.tp >= 4:
        kwargs["disable_custom_all_reduce"] = True

    started = time.perf_counter()
    llm = LLM(**kwargs)
    payload["load_s"] = time.perf_counter() - started
    payload["engine_kwargs"] = kwargs
    payload["parallelism"] = "vllm_tensor_parallel"
    payload["vocab_size"] = llm.get_tokenizer().vocab_size
    write_payload(output_path, payload)

    prefill_s: dict[tuple[int, int], float] = {}
    prompt_cache: dict[tuple[int, int], list[Any]] = {}
    for bsz, in_len, out_len in cells:
        cell: dict[str, Any] = {
            "bsz": bsz,
            "in_len": in_len,
            "out_len": out_len,
        }
        try:
            key = (bsz, in_len)
            if key not in prompt_cache:
                prompt = TokensPrompt(
                    prompt_token_ids=make_token_ids(payload["vocab_size"], in_len)
                )
                prompt_cache[key] = [prompt] * bsz
            params = SamplingParams(
                max_tokens=out_len,
                min_tokens=out_len,
                ignore_eos=True,
                temperature=0.0,
            )

            def run_once() -> None:
                llm.generate(prompt_cache[key], params, use_tqdm=False)

            times = time_trials(
                run_once,
                warmups=args.warmups,
                trials=args.trials,
            )
            cell["trials_s"] = times
            cell["wall_s"] = statistics.median(times)
            # vLLM owns CUDA contexts in worker processes, so parent-process
            # torch.cuda memory counters are not meaningful here.
            add_metrics(cell, prefill_s)
        except Exception as exc:  # noqa: BLE001
            cell["error"] = f"{type(exc).__name__}: {exc}"
        payload["cells"].append(cell)
        write_payload(output_path, payload)
        print(json.dumps(cell, ensure_ascii=False), flush=True)


def run_hf(
    args: argparse.Namespace,
    model_path: str,
    cells: list[tuple[int, int, int]],
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import Cache

    # Ling-lite's checkpoint code targets Transformers 4.53, where Cache
    # exposed ``seen_tokens``.  In 4.57 the equivalent public information is
    # returned by get_seq_length().  Add the compatibility property rather
    # than pinning a second Transformers installation for one model.
    if not hasattr(Cache, "seen_tokens"):
        Cache.seen_tokens = property(lambda cache: cache.get_seq_length())
    if not hasattr(Cache, "get_usable_length"):
        Cache.get_usable_length = (
            lambda cache, new_seq_length, layer_idx=0: cache.get_seq_length(
                layer_idx
            )
        )

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.tp == 1:
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["device_map"] = "balanced"
        load_kwargs["max_memory"] = {
            i: f"{args.hf_max_memory_gib}GiB" for i in range(args.tp)
        }
    if args.hf_attn_implementation:
        load_kwargs["attn_implementation"] = args.hf_attn_implementation

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    gptoss_patched_mlps = enable_gptoss_torch_routing(model)
    payload["load_s"] = time.perf_counter() - started
    payload["load_kwargs"] = load_kwargs
    payload["parallelism"] = (
        "single_gpu" if args.tp == 1 else "transformers_accelerate_layer_sharding"
    )
    payload["transformers_version"] = transformers.__version__
    payload["torch_version"] = torch.__version__
    payload["hf_device_map"] = getattr(model, "hf_device_map", None)
    payload["attn_implementation"] = getattr(
        model.config, "_attn_implementation", None
    )
    if gptoss_patched_mlps:
        payload["gptoss_routing"] = "torch_stable_argsort"
        payload["gptoss_patched_mlps"] = gptoss_patched_mlps
    payload["vocab_size"] = model.config.vocab_size

    # CPU/disk offload is a different benchmark and usually hides an
    # insufficient-GPU-memory configuration behind PCIe traffic.
    device_map = payload["hf_device_map"] or {}
    if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(f"Refusing CPU/disk offload: {device_map}")

    input_device = model.get_input_embeddings().weight.device
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.eos_token_id = None
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = getattr(model.config, "pad_token_id", 0)
    write_payload(output_path, payload)

    prefill_s: dict[tuple[int, int], float] = {}
    tensor_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for bsz, in_len, out_len in cells:
        cell: dict[str, Any] = {
            "bsz": bsz,
            "in_len": in_len,
            "out_len": out_len,
        }
        try:
            key = (bsz, in_len)
            if key not in tensor_cache:
                one = torch.tensor(
                    make_token_ids(payload["vocab_size"], in_len),
                    dtype=torch.long,
                    device=input_device,
                )
                input_ids = one.unsqueeze(0).expand(bsz, -1).contiguous()
                attention_mask = torch.ones_like(input_ids)
                tensor_cache[key] = input_ids, attention_mask
            input_ids, attention_mask = tensor_cache[key]

            def run_once() -> None:
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                        max_new_tokens=out_len,
                        min_new_tokens=out_len,
                        use_cache=True,
                        synced_gpus=False,
                    )
                del generated

            reset_gpu_peak()
            times = time_trials(
                run_once,
                warmups=args.warmups,
                trials=args.trials,
            )
            cell["trials_s"] = times
            cell["wall_s"] = statistics.median(times)
            cell["peak_allocated_mib"] = visible_gpu_peak_mib()
            add_metrics(cell, prefill_s)
        except Exception as exc:  # noqa: BLE001
            cell["error"] = f"{type(exc).__name__}: {exc}"
            gc.collect()
            torch.cuda.empty_cache()
        payload["cells"].append(cell)
        write_payload(output_path, payload)
        print(json.dumps(cell, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("vllm", "hf"), required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cells_json")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--hf_max_memory_gib", type=int, default=46)
    parser.add_argument("--hf_attn_implementation")
    args = parser.parse_args()

    if args.trials < 1 or args.warmups < 0:
        raise ValueError("trials must be >=1 and warmups must be >=0")
    cells = parse_cells(args.cells_json)
    model_path = prepare_model_dir_with_num_experts_per_tok(
        args.model_path, args.k
    )
    output_path = Path(args.out)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "complete": False,
        "backend": args.backend,
        "model": args.model_path,
        "model_override": model_path,
        "k": args.k,
        "tp_or_gpu_count": args.tp,
        "cards": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host": os.uname().nodename,
        "gpu_meta": gpu_meta(),
        "trials": args.trials,
        "warmups": args.warmups,
        "seed": args.seed,
        "cells_spec": [
            {"bsz": bsz, "in_len": in_len, "out_len": out_len}
            for bsz, in_len, out_len in cells
        ],
        "cells": [],
    }
    write_payload(output_path, payload)

    if args.backend == "vllm":
        run_vllm(args, model_path, cells, payload, output_path)
    else:
        run_hf(args, model_path, cells, payload, output_path)

    payload["complete"] = True
    payload["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    write_payload(output_path, payload)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
