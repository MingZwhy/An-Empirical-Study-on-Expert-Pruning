"""Shut a vLLM engine down cleanly when a harness has no hook of its own.

Left to itself, a process that built ``vllm.LLM`` in-process and then falls off the end of ``main``
ends in ``terminate called without an active exception`` and SIGABRT, *after* the results are safely
written: torch warns that ``destroy_process_group()`` was never called, and the NCCL process group's
watchdog thread is still joinable when the libraries unload, which is a std::terminate. Only the exit
code is wrong, which sounds harmless until a queue worker reads it and runs the finished evaluation
again. A second failure hides behind the same cause: an engine that outlives the run keeps its share
of the card, so the next configuration scheduled onto it cannot allocate.

lighteval's model wrapper does this in its ``cleanup``. lm_eval's and lmms-eval's wrappers have no
equivalent hook, so both call in here instead of each keeping a copy.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

# Python's own helper process, which exists to catch leaked shared memory and is meant to outlive
# everything else until the parent goes. It holds no GPU, so waiting for it would be waiting forever.
_IGNORED_CHILDREN = ("multiprocessing.resource_tracker",)


def shutdown_vllm(engine_holder, attr: str = "model") -> None:
    """Tear the engine down, then wait for it to actually be gone.

    ``engine_holder`` is the harness's model wrapper and ``attr`` the attribute holding the
    ``vllm.LLM``; dropping the reference is what starts the shutdown.

    Each step is attempted on its own: an early failure (parallel state that was never initialised in
    this process, say) must not skip destroying the process group, which is the step that matters.
    Failures are printed unbuffered, because a later abort would swallow anything still in the buffer.
    """
    import gc

    import torch
    from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                                 destroy_model_parallel)

    def drop_engine():
        if getattr(engine_holder, attr, None) is not None:
            delattr(engine_holder, attr)
        gc.collect()

    def drop_process_group():
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    for name, step in (("destroy_model_parallel", destroy_model_parallel),
                       ("drop engine", drop_engine),
                       ("destroy_distributed_environment", destroy_distributed_environment),
                       ("destroy_process_group", drop_process_group),
                       ("empty_cache", torch.cuda.empty_cache)):
        try:
            step()
        except Exception as exc:  # noqa: BLE001 - results are on disk; teardown must not fail the run
            print(f"vLLM 关停步骤 {name} 报错（不影响已保存的结果）: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    await_engine_exit()


def _reap_children() -> None:
    """Collect children that have already exited, so they stop counting as processes.

    An exited child stays listed until its parent reaps it, and a zombie cannot be killed — without
    this, the engine process looked like something that refused both to exit and to die.
    """
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except (ChildProcessError, OSError):
            return
        if pid == 0:
            return


def _engine_children() -> list[int]:
    """Live child processes worth waiting for. vLLM runs the engine in one of them."""
    _reap_children()
    try:
        out = subprocess.run(["pgrep", "-P", str(os.getpid())],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    live = []
    for pid in (int(p) for p in out.stdout.split()):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
            cmdline = Path(f"/proc/{pid}/cmdline").read_text()
        except (OSError, IndexError):
            continue
        if state == "Z" or any(marker in cmdline for marker in _IGNORED_CHILDREN):
            continue
        live.append(pid)
    return live


def _describe(pid: int) -> str:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ").strip()
        return f"{pid}({cmd[:160] or Path(f'/proc/{pid}/comm').read_text().strip()})"
    except OSError:
        return str(pid)


def await_engine_exit(grace: float = 30.0) -> None:
    """Make sure nothing of the engine is left before this process exits.

    Dropping the engine only starts its shutdown, and returning straight away caused both problems
    described at the top of this module. Merely adding the child-process check below, worth tens of
    milliseconds, was enough to turn one run from exit code 134 into 0, so waiting on purpose is the
    honest version of that accident.

    The evaluation is written to disk before any of this, so once the grace period is up there is
    nothing left worth waiting for and whatever remains is killed.
    """
    deadline = time.monotonic() + grace
    left = _engine_children()
    while left and time.monotonic() < deadline:
        time.sleep(0.5)
        left = _engine_children()
    if not left:
        # The engine is gone but its threads unwind slightly after it; the abort above is exactly that
        # window, so leave a margin rather than exiting into it.
        time.sleep(2)
        return

    print(f"关停后仍在的子进程 {[_describe(p) for p in left]} 超过 {grace:.0f}s 没退出，强制结束，"
          "否则它会继续占着这张卡的显存", flush=True)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in left:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        for _ in range(20):
            left = _engine_children()
            if not left:
                return
            time.sleep(0.5)
    print(f"仍未结束: {[_describe(p) for p in left]}，需要人工清理", flush=True)
