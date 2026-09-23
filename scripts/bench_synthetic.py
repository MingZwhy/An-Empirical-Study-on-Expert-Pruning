"""Fixed-shape prefill/decode throughput for Baseline vs Fixed-K.

Loads the model once, then walks a (batch, input length, output length) grid. Output length 1 is
treated as a prefill measurement. Longer outputs are timed as a whole pass; decode tok/s is estimated
by subtracting the matching prefill (same batch and input length, out=1) when that cell exists.

Random token ids are used so the measurement is not prompt-content dependent. Prefix caching is off.
OOM on a cell is recorded and the rest of the grid continues.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from main import prepare_model_dir_with_num_experts_per_tok  # noqa: E402


BSZ = (1, 8, 32)
IN_LEN = (256, 2048)
OUT_LEN = (1, 128, 1024)


def _gpu_meta():
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,power.limit",
             "--format=csv,noheader"], text=True)
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception as e:  # noqa: BLE001
        return [str(e)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--trials", type=int, default=2)
    args = p.parse_args()

    model = prepare_model_dir_with_num_experts_per_tok(args.model_path, args.k)
    max_len = max(IN_LEN) + max(OUT_LEN) + 64
    kwargs = dict(
        model=model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_len,
        trust_remote_code=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=1,
    )
    if args.tp >= 4:
        kwargs["disable_custom_all_reduce"] = True

    from vllm import LLM, SamplingParams
    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        from vllm import TokensPrompt  # older layout


    t_load = time.perf_counter()
    llm = LLM(**kwargs)
    load_s = time.perf_counter() - t_load
    vocab = llm.get_tokenizer().vocab_size
    # Stay away from the very first ids (often special) and the tail.
    base = 100

    prefill_s: dict[tuple[int, int], float] = {}
    cells = []
    for bsz in BSZ:
        for in_len in IN_LEN:
            prompt = TokensPrompt(prompt_token_ids=[base + (i % (vocab - base - 16))
                                                    for i in range(in_len)])
            prompts = [prompt] * bsz
            for out_len in OUT_LEN:
                params = SamplingParams(max_tokens=out_len, ignore_eos=True,
                                        temperature=0.0, top_p=1.0)
                cell = {"bsz": bsz, "in_len": in_len, "out_len": out_len}
                try:
                    llm.generate(prompts, params, use_tqdm=False)  # warmup
                    times = []
                    for _ in range(args.trials):
                        t0 = time.perf_counter()
                        llm.generate(prompts, params, use_tqdm=False)
                        times.append(time.perf_counter() - t0)
                    times.sort()
                    wall = times[len(times) // 2]
                    cell["wall_s"] = wall
                    cell["trials_s"] = times
                    prompt_tok = bsz * in_len
                    gen_tok = bsz * out_len
                    cell["prompt_tok"] = prompt_tok
                    cell["gen_tok"] = gen_tok
                    if out_len == 1:
                        prefill_s[(bsz, in_len)] = wall
                        cell["kind"] = "prefill"
                        cell["prefill_tok_s"] = prompt_tok / wall if wall else None
                    else:
                        cell["kind"] = "prefill+decode"
                        cell["total_tok_s"] = (prompt_tok + gen_tok) / wall if wall else None
                        pref = prefill_s.get((bsz, in_len))
                        if pref is not None and wall > pref:
                            cell["decode_s"] = wall - pref
                            cell["decode_tok_s"] = (bsz * (out_len - 1)) / (wall - pref)
                        else:
                            cell["decode_s"] = None
                            cell["decode_tok_s"] = None
                except Exception as e:  # noqa: BLE001
                    cell["error"] = f"{type(e).__name__}: {e}"
                cells.append(cell)
                print(json.dumps(cell, ensure_ascii=False), flush=True)

    payload = {
        "model": args.model_path,
        "k": args.k,
        "tp": args.tp,
        "load_s": load_s,
        "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_meta": _gpu_meta(),
        "host": os.uname().nodename,
        "grid": {"bsz": list(BSZ), "in_len": list(IN_LEN), "out_len": list(OUT_LEN)},
        "cells": cells,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
