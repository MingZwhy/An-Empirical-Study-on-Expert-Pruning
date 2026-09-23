"""Evaluate a pruned router on lmms-eval, for the multimodal suites the VL models are scored on.

The two other harnesses in this repository only ever feed the model text. A vision-language MoE
routes image tokens through the same experts as text tokens, and there is no reason to assume a
budget calibrated on text keeps the visual path intact — so the question "which experts can this
model afford to lose" has to be asked again with images in the prompt. That is what this harness is
for; the pruning path itself is unchanged.

Everything about the pruning is shared with the other two harnesses: this is reached from ``main.py``
after it has parsed the pruning options, exported them, and installed the router patch, so the router
measured here is byte-identical to the one the text numbers come from. Only the harness differs.

Four deliberate choices:

* The mini/lite variant of each dataset. Several of these sets exist in lmms-eval in more than one
  size, and the sweep multiplies every dataset by every method and budget; the lite variants (500
  sampled questions each, from ``lmms-lab/LMMs-Eval-Lite``) keep a full sweep affordable. They are
  the published subsets, so rows stay comparable to each other, but not to numbers quoted on the full
  sets — which is why the choice is recorded in the output rather than left to the reader.

* No harness-side caching. lmms-eval can cache both requests and responses, keyed by task and model
  name — which does not distinguish two expert-pruning configurations of the same checkpoint. That is
  exactly the reuse this repository already had to fix once in lighteval, where different methods
  silently replayed each other's generations.

* Samples are logged without their media. lmms-eval embeds the images in the per-sample log unless
  told otherwise, which turns a 500-question task into gigabytes of base64.

* The chat format comes from the model wrapper, not from the harness-level switch, which is off; see
  the note on ``apply_chat_template`` in ``docs/known_issues.md``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from expert_pruning.harness_teardown import shutdown_vllm

REPO_ROOT = Path(__file__).resolve().parent.parent
LMMS_EVAL_ROOT = REPO_ROOT / "3rdparty" / "lmms-eval"

# The nine candidate sets, each in its smallest published variant. mmmu_val and mmstar have no lite
# release: mmmu's test split is unlabelled, so val (900 questions) is the only scorable one, and
# mmstar ships as a single 1500-question set.
DEFAULT_TASKS = ("mmmu_val", "mmbench_en_dev_lite", "mmstar", "mmerealworld_lite", "chartqa_lite",
                 "textvqa_val_lite", "infovqa_val_lite", "gqa_lite", "vizwiz_vqa_val_lite")


class CliArgs(SimpleNamespace):
    """Stands in for lmms-eval's own argparse namespace, which its tasks read settings off.

    Anything not set here reads as None, i.e. as an option that was left alone — which is what a task
    would see from the upstream CLI in that case. The alternative is an AttributeError raised during
    aggregation, after every question has already been answered: the missing `output_path` below cost
    a 4.5 hour run of the nine-dataset suite, and only `ocrbench`, which writes a submission file,
    went looking for it.
    """

    def __getattr__(self, name: str):
        return None


def _import_lmms_eval():
    if not (LMMS_EVAL_ROOT / "lmms_eval").is_dir():
        raise RuntimeError(
            f"未找到 {LMMS_EVAL_ROOT}/lmms_eval：请先执行 "
            "`git submodule update --init 3rdparty/lmms-eval`，"
            "并用 install_expertpruning.sh 安装。")
    if str(LMMS_EVAL_ROOT) not in sys.path:
        sys.path.insert(0, str(LMMS_EVAL_ROOT))
    import lmms_eval  # noqa: E402

    return lmms_eval


def _harness_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(LMMS_EVAL_ROOT), "describe", "--tags", "--always"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _model_kwargs(args, tensor_parallel_size: int) -> dict:
    """vLLM options, mirroring what the other harnesses hand to the same engine.

    Anything not named by lmms-eval's wrapper is forwarded to ``vllm.LLM`` unchanged, which is how
    the multimodal-specific limits get set.
    """
    kwargs = dict(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_length,
        # A question with several images is common here (MMMU asks about up to seven at once) and
        # vLLM rejects anything above this limit outright, so it is raised rather than left at 1.
        limit_mm_per_prompt={"image": args.lmms_eval_max_images},
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        disable_log_stats=True,
    )
    if args.cpu_offload_gb:
        kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    # This one goes to the wrapper rather than to the engine, and it is not optional: the wrapper makes
    # one client.chat() call per batch, so its default of 1 serialises the entire run — the engine never
    # has a second sequence to overlap with, and 7000 questions crawl. vLLM's own scheduler still bounds
    # how many of a batch are resident at once, so this is a floor on concurrency, not a promise.
    kwargs["batch_size"] = args.batch_size or 16
    if args.pipeline_parallel_size is not None:
        kwargs["pipeline_parallel_size"] = args.pipeline_parallel_size
    return kwargs


def run_lmms_eval(args, *, tensor_parallel_size: int, short_model_name: str,
                  stats_dir: str | None, collect_stats, use_local_expert_router: bool,
                  installed_router_patch: bool, cache_fingerprint: str | None,
                  router_distribution_dir: str | None = None):
    """Run the requested lmms-eval tasks and write one result file next to the lighteval ones."""
    lmms_eval = _import_lmms_eval()
    import expert_pruning.vllm_patch  # noqa: F401 - env bootstrap for workers
    from lmms_eval.models import get_model
    from lmms_eval.utils import handle_non_serializable

    tasks = [t.strip() for t in args.lmms_eval_tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("--lmms_eval_tasks 为空")

    print(f"使用 lmms-eval 后端评估: tasks={tasks}, num_fewshot={args.lmms_eval_num_fewshot}, "
          f"model={args.model_path} (保存名: {short_model_name}), TP={tensor_parallel_size}, "
          f"每题最多 {args.lmms_eval_max_images} 张图")

    started = time.time()
    model_kwargs = _model_kwargs(args, tensor_parallel_size)
    # The chat wrapper, which is what get_model returns for "vllm" unless force_simple is asked for:
    # it builds the multimodal messages these instruct checkpoints expect.
    lm = get_model("vllm")(**model_kwargs)
    try:
        return _evaluate_and_save(
            args, lmms_eval, lm, tasks, started,
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
        shutdown_vllm(lm, attr="client")


def _evaluate_and_save(args, lmms_eval, lm, tasks, started, *, handle_non_serializable, model_kwargs,
                       tensor_parallel_size, short_model_name, stats_dir, collect_stats,
                       use_local_expert_router, installed_router_patch, cache_fingerprint,
                       router_distribution_dir: str | None = None):
    from lmms_eval import evaluator

    raw = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.lmms_eval_num_fewshot,
        limit=args.max_samples,
        log_samples=True,
        # Off, and not configurable: the wrapper get_model("vllm") returns builds its own OpenAI-style
        # messages out of each task's doc_to_messages, so this is not the switch that decides whether
        # these instruct checkpoints see a chat format. Turning it on only crashes — evaluate() then
        # does getattr(lm, "apply_chat_template"), and no model class in v0.7.2 defines that.
        apply_chat_template=False,
        # Both caches stay off; see the module docstring.
        use_cache=None,
        cache_requests=False,
        # lmms-eval reads a handful of settings off the CLI namespace. process_with_media keeps the
        # images out of the per-sample log, and output_path decides where a task that writes a
        # submission file puts it — under this run's own directory, so two configurations of the same
        # checkpoint cannot overwrite each other's.
        cli_args=CliArgs(process_with_media=False, reasoning_tags=None,
                         agentic_trace_mode="basic",
                         output_path=str(Path(args.output_dir) / short_model_name)),
    )
    if raw is None:  # only the non-zero ranks of a distributed run
        return None, None
    elapsed = time.time() - started

    samples = raw.pop("samples", None)
    stats = collect_stats(stats_dir)
    if args.allow_compiled_router and stats:
        # Same treatment as the other paths: counting survives the compiled region but vLLM pads
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
            "expert router 未回报任何统计，本次结果无法定位到平均专家数轴上。统计丢失通常是专家"
            "选择被编译区 inline 掉了（量化 MoE 常见）。加 --enforce_eager 重跑。")
    if stats is not None:
        stats.setdefault("router_invoked", True)
        stats["router_patch_installed"] = installed_router_patch
        stats["expert_pruning_cache_fingerprint"] = cache_fingerprint
        stats["harness"] = "lmms_eval"

    out_root = Path(args.output_dir) / short_model_name
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    payload = {
        "harness": {
            "name": "lmms_eval",
            "version": getattr(lmms_eval, "__version__", None),
            "commit": _harness_commit(),
        },
        "config_general": {
            "model_name": short_model_name,
            "model_path": args.model_path,
            "num_fewshot": args.lmms_eval_num_fewshot,
            "apply_chat_template": False,  # the chat wrapper formats messages itself
            "limit": args.max_samples,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_length": args.max_model_length,
            "enforce_eager": args.enforce_eager,
            # lmms-eval only records model_args when the model is named on its own command line; the
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
        "lmms_eval_config": raw.get("config"),
        "git_hash": raw.get("git_hash"),
        "expert_pruning": stats,
        "router_distribution": ({
            "output_dir": router_distribution_dir,
            "dataset": args.lmms_eval_tasks,
            "sample_rate": getattr(args, "router_distribution_sample_rate", None),
            "sample_cap": getattr(args, "router_distribution_sample_cap", None),
        } if router_distribution_dir else None),
    }
    # The filename keeps these apart from the lighteval outputs, which report_results.py globs as
    # results_*.json — a shared config directory therefore stays unambiguous.
    out_file = results_dir / f"lmms_eval_results_{stamp}.json"
    out_file.write_text(json.dumps(payload, indent=2, default=handle_non_serializable,
                                   ensure_ascii=False))

    if samples:
        details_dir = out_root / "details"
        details_dir.mkdir(parents=True, exist_ok=True)
        for task, rows in samples.items():
            with (details_dir / f"lmms_eval_samples_{task}_{stamp}.jsonl").open("w") as fh:
                for row in rows:
                    fh.write(json.dumps(row, default=handle_non_serializable,
                                        ensure_ascii=False) + "\n")

    print(lmms_eval.utils.make_table(raw))
    if stats is not None and stats.get("average_selected_experts") is not None:
        print(f"平均专家选择数: {stats['average_selected_experts']:.6f} "
              f"(method={stats.get('method')}, topk={stats.get('topk')}, "
              f"routed_tokens={stats.get('total_routed_tokens')})")
    elif stats is not None:
        print("平均专家选择数: 本 run 未测量（"
              f"{stats.get('statistics_note', '无统计')}）")
    print(f"评估完成（{elapsed / 60:.1f} 分钟）。结果已保存至: {out_file}")
    return payload, samples
