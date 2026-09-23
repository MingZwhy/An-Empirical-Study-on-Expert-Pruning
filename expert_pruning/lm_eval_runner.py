"""Evaluate a pruned router on lm-evaluation-harness, for the zero-shot multiple-choice QA suites.

The lighteval suite this repository runs is almost entirely generative: the model writes an answer and
a parser grades it. That makes it sensitive to things a pruned router does to *fluency* — a collapsed
configuration rambles to the token limit and scores zero on tasks it could arguably still answer. The
suites here (arc_challenge, arc_easy, winogrande and openbookqa by default; the nine-task list adds
boolq, piqa, social_iqa, mmlu and hellaswag) are scored the opposite way: each
candidate answer is appended to the prompt and ranked by log-likelihood, nothing is generated, and the
grade depends only on which continuation the model finds most probable. Pruning is therefore measured
without the generation-length confound, and the two harnesses together separate "the router lost the
knowledge" from "the router lost the ability to stop".

Everything about the pruning path is shared with the generative runs: this is reached from ``main.py``
after it has parsed the pruning options, exported them, and installed the router patch, so the router
here is byte-identical to the one the lighteval numbers come from. Only the harness differs.

Two deliberate choices:

* No harness-side caching. lm_eval can cache requests and responses (``--use_cache``,
  ``cache_requests``), keyed by task and model name — which does not distinguish two expert-pruning
  configurations of the same checkpoint. That is precisely the reuse this repository already had to fix
  once in lighteval, where different methods silently replayed each other's generations. Since these
  tasks generate nothing and cost minutes, recomputation is cheaper than the risk.

* No chat template by default. These suites are scored as raw continuations, which is how their
  published numbers are produced; wrapping the prompt in an instruct template changes what the
  log-likelihoods mean and is not comparable to them. ``--lm_eval_apply_chat_template`` is available
  when the instruct-formatted variant is what you want, and the choice is recorded in the output.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from expert_pruning.harness_teardown import shutdown_vllm

REPO_ROOT = Path(__file__).resolve().parent.parent
LM_EVAL_ROOT = REPO_ROOT / "3rdparty" / "lm-evaluation-harness"
# Task definitions this repository adds to the harness's own, for suites whose shipped definition cannot
# load under the installed datasets version. This directory is the source of truth for them.
LOCAL_TASKS_DIR = REPO_ROOT / "tasks_lm_eval"
# ...but the harness cannot read them from here. get_task_dict logs every selected task as a path
# relative to lm_eval/tasks, via Path.relative_to, which raises for a task that came from an include path
# outside that tree — so `TaskManager(include_path=...)` crashes the run before the first request. The
# files are therefore mirrored into a directory inside the tree, where the harness's own recursive scan
# picks them up and its path arithmetic holds. Mirroring at run time rather than at install time keeps
# every host in step with the repository without a provisioning step.
MIRROR_DIR = LM_EVAL_ROOT / "lm_eval" / "tasks" / "_repo_include"


def _import_lm_eval():
    if not (LM_EVAL_ROOT / "lm_eval").is_dir():
        raise RuntimeError(
            f"未找到 {LM_EVAL_ROOT}/lm_eval：请先执行 "
            "`git submodule update --init 3rdparty/lm-evaluation-harness`，"
            "并用 install_expertpruning.sh 安装。")
    if str(LM_EVAL_ROOT) not in sys.path:
        sys.path.insert(0, str(LM_EVAL_ROOT))
    import lm_eval  # noqa: E402

    return lm_eval


def _mirror_local_tasks() -> None:
    """Copy this repository's task definitions where the harness expects to find task files."""
    if not LOCAL_TASKS_DIR.is_dir():
        return
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(LOCAL_TASKS_DIR.glob("*.yaml")):
        dst = MIRROR_DIR / src.name
        payload = src.read_bytes()
        if dst.is_file() and dst.read_bytes() == payload:
            continue
        # Written through a temporary name because several runs share this checkout and may do this at
        # the same moment: a reader must see either the old file or the new one, never a partial one.
        tmp = MIRROR_DIR / f".{src.name}.{os.getpid()}"
        tmp.write_bytes(payload)
        os.replace(tmp, dst)


def _harness_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(LM_EVAL_ROOT), "describe", "--tags", "--always"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _model_kwargs(args, tensor_parallel_size: int) -> dict:
    """vLLM options, mirroring what the lighteval path hands to the same engine."""
    kwargs = dict(
        pretrained=args.model_path,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_length,
        dtype=args.dtype,
        # "auto" hands every request to vLLM at once and lets its scheduler form the batches, which
        # is what we want; the integer default would feed them one at a time.
        batch_size="auto",
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        language_model_only=args.language_model_only,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_expert_parallel=args.enable_expert_parallel,
    )
    if args.block_size is not None:
        kwargs["block_size"] = args.block_size
    if args.cpu_offload_gb:
        kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    if args.batch_size is not None:
        # Same meaning as in the lighteval path: a ceiling on vLLM's concurrent sequences.
        kwargs["max_num_seqs"] = args.batch_size
    if args.pipeline_parallel_size is not None:
        kwargs["pipeline_parallel_size"] = args.pipeline_parallel_size
    if args.data_parallel_size is not None:
        kwargs["data_parallel_size"] = args.data_parallel_size
    return _drop_unsupported_engine_args(kwargs)


# lm-eval hands anything it does not recognise straight to vllm.LLM, which
# forwards it to EngineArgs. The two environments in this repository run engines
# nine months apart, and their argument surfaces differ -- language_model_only,
# for instance, exists only on the newer one. Passing it to the pinned v0.10.2
# aborts with "EngineArgs.__init__() got an unexpected keyword argument" after
# the harness has already loaded its datasets. Filtering against the installed
# signature keeps one call site working on both.
_LM_EVAL_OWN_ARGS = frozenset({"pretrained", "batch_size"})


def _drop_unsupported_engine_args(kwargs: dict) -> dict:
    import inspect

    accepted = None
    for path, attr in (("vllm.engine.arg_utils", "EngineArgs"), ("vllm.config", "EngineArgs")):
        try:
            module = __import__(path, fromlist=[attr])
            params = inspect.signature(getattr(module, attr).__init__).parameters
        except Exception:  # noqa: BLE001 - anything here means "do not filter"
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs  # accepts **kwargs, so we cannot tell what is supported
        accepted = set(params)
        break
    if accepted is None:
        return kwargs
    dropped = sorted(k for k in kwargs if k not in accepted and k not in _LM_EVAL_OWN_ARGS)
    if dropped:
        print(f"[expert-pruning] installed vLLM does not accept {', '.join(dropped)}; dropping")
    return {k: v for k, v in kwargs.items()
            if k in accepted or k in _LM_EVAL_OWN_ARGS}


def run_lm_eval(args, *, tensor_parallel_size: int, short_model_name: str,
                stats_dir: str | None, collect_stats, use_local_expert_router: bool,
                installed_router_patch: bool, cache_fingerprint: str | None,
                router_distribution_dir: str | None = None):
    """Run the requested lm_eval tasks and write one result file next to the lighteval ones."""
    lm_eval = _import_lm_eval()
    _mirror_local_tasks()
    import expert_pruning.vllm_patch  # noqa: F401 - env bootstrap for workers
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.utils import handle_non_serializable

    tasks = [t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("--lm_eval_tasks 为空")

    print(f"使用 lm_eval 后端评估: tasks={tasks}, num_fewshot={args.lm_eval_num_fewshot}, "
          f"model={args.model_path} (保存名: {short_model_name}), TP={tensor_parallel_size}, "
          f"chat_template={'on' if args.lm_eval_apply_chat_template else 'off'}")

    started = time.time()
    model_kwargs = _model_kwargs(args, tensor_parallel_size)
    lm = VLLM(**model_kwargs)
    try:
        return _evaluate_and_save(
            args, lm_eval, lm, tasks, started,
            handle_non_serializable=handle_non_serializable,
            model_kwargs=model_kwargs,
            tensor_parallel_size=tensor_parallel_size,
            short_model_name=short_model_name,
            stats_dir=stats_dir,
            collect_stats=collect_stats,
            use_local_expert_router=use_local_expert_router,
            installed_router_patch=installed_router_patch,
            cache_fingerprint=cache_fingerprint,
            router_distribution_dir=router_distribution_dir,
        )
    finally:
        if router_distribution_dir:
            from expert_pruning.router_distribution import flush_snapshot

            flush_snapshot()
        # After the statistics have been read: the counters are flushed by the engine's own
        # processes, and tearing the engine down first could drop the last of them.
        shutdown_vllm(lm)


def _evaluate_and_save(args, lm_eval, lm, tasks, started, *, handle_non_serializable, model_kwargs,
                       tensor_parallel_size, short_model_name, stats_dir, collect_stats,
                       use_local_expert_router, installed_router_patch, cache_fingerprint,
                       router_distribution_dir: str | None = None):
    raw = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.lm_eval_num_fewshot,
        limit=args.max_samples,
        log_samples=True,
        apply_chat_template=args.lm_eval_apply_chat_template,
        # Both caches stay off; see the module docstring.
        use_cache=None,
        cache_requests=False,
    )
    if raw is None:  # only the non-zero ranks of a distributed run
        return None, None
    elapsed = time.time() - started

    samples = raw.pop("samples", None)
    stats = collect_stats(stats_dir)
    if args.allow_compiled_router and stats:
        # Same treatment as the generative path: counting survives the compiled region but vLLM pads
        # batches to a captured graph size and the padding rows are counted as tokens, so the average
        # reads low. Keep it under its own name and take the operating point from an eager probe.
        padded = stats.pop("average_selected_experts", None)
        if padded is not None:
            stats["average_selected_experts_padded"] = padded
            stats["average_selected_experts"] = None
            stats["statistics_note"] = (
                "ran with CUDA graphs (--allow_compiled_router); the average here would count "
                "CUDA-graph padding rows as tokens, so it is reported separately and the "
                "operating point comes from a --enforce_eager probe of this configuration")
    if stats is None and use_local_expert_router:
        raise RuntimeError(
            "expert router 未回报任何统计，本次结果无法定位到平均专家数轴上。这些任务只做 "
            "prefill 打分、不生成，所以路由一定被调用过；统计丢失通常是专家选择被编译区 "
            "inline 掉了（量化 MoE 常见）。加 --enforce_eager 重跑。")
    if stats is not None:
        stats.setdefault("router_invoked", True)
        stats["router_patch_installed"] = installed_router_patch
        stats["expert_pruning_cache_fingerprint"] = cache_fingerprint
        stats["harness"] = "lm_eval"

    out_root = Path(args.output_dir) / short_model_name
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    payload = {
        "harness": {
            "name": "lm_eval",
            "version": getattr(lm_eval, "__version__", None),
            "commit": _harness_commit(),
        },
        "config_general": {
            "model_name": short_model_name,
            "model_path": args.model_path,
            "num_fewshot": args.lm_eval_num_fewshot,
            "apply_chat_template": bool(args.lm_eval_apply_chat_template),
            "limit": args.max_samples,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_length": args.max_model_length,
            "enforce_eager": args.enforce_eager,
            # lm_eval only records model_args when the model is named on its own command line; the
            # engine is built here, so the settings are written down here too.
            "vllm_model_args": model_kwargs,
            "total_evaluation_time_secondes": str(elapsed),
        },
        "results": raw.get("results"),
        "groups": raw.get("groups"),
        "n-shot": raw.get("n-shot"),
        "higher_is_better": raw.get("higher_is_better"),
        "versions": raw.get("versions"),
        "config_tasks": raw.get("configs"),
        "lm_eval_config": raw.get("config"),
        "git_hash": raw.get("git_hash"),
        "expert_pruning": stats,
        "router_distribution": ({
            "output_dir": router_distribution_dir,
            "dataset": args.lm_eval_tasks,
            "sample_rate": getattr(args, "router_distribution_sample_rate", None),
            "sample_cap": getattr(args, "router_distribution_sample_cap", None),
        } if router_distribution_dir else None),
    }
    # The filename keeps these apart from the lighteval outputs, which report_results.py globs as
    # results_*.json — a shared config directory therefore stays unambiguous.
    out_file = results_dir / f"lm_eval_results_{stamp}.json"
    out_file.write_text(json.dumps(payload, indent=2, default=handle_non_serializable,
                                   ensure_ascii=False))

    if samples:
        details_dir = out_root / "details"
        details_dir.mkdir(parents=True, exist_ok=True)
        for task, rows in samples.items():
            with (details_dir / f"lm_eval_samples_{task}_{stamp}.jsonl").open("w") as fh:
                for row in rows:
                    fh.write(json.dumps(row, default=handle_non_serializable,
                                        ensure_ascii=False) + "\n")

    print(lm_eval.utils.make_table(raw))
    if stats is not None and stats.get("average_selected_experts") is not None:
        print(f"平均专家选择数: {stats['average_selected_experts']:.6f} "
              f"(method={stats.get('method')}, topk={stats.get('topk')}, "
              f"routed_tokens={stats.get('total_routed_tokens')})")
    elif stats is not None:
        print("平均专家选择数: 本 run 未测量（"
              f"{stats.get('statistics_note', '无统计')}）")
    print(f"评估完成（{elapsed / 60:.1f} 分钟）。结果已保存至: {out_file}")
    return payload, samples
