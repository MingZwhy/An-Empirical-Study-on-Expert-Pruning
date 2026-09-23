"""Point lighteval's GPQA tasks at a local copy, so no Hub account is needed.

GPQA is gated, and the terms accepted to obtain it say not to reveal its examples
in plain text online -- so this repository does not carry them. The authors publish
the same data as a password-protected archive on their own GitHub, which needs no
account; `scripts/fetch_gpqa.py` fetches and unpacks it into `data/gpqa/`.

lighteval's task definitions name the Hub repository, `Idavidrein/gpqa`, and
`datasets.load_dataset` accepts a local directory in the same position. So when a
local copy is present the task's `hf_repo` is rewritten to point at it. Nothing
else about the task changes: the subset names, splits, prompt function, scorer and
metrics are untouched, so the measurement is the same one.

The rewrite happens only when a copy is actually there, which keeps the Hub the
default and means a stale local directory cannot silently take over.
"""
from __future__ import annotations

import os
from pathlib import Path

_HUB_REPO = "Idavidrein/gpqa"

# The configs the tasks ask for, and the files a local copy must therefore have.
_REQUIRED_FILES = ("README.md", "gpqa_diamond.csv", "gpqa_main.csv", "gpqa_extended.csv")


def local_gpqa_dir() -> Path | None:
    """The local GPQA copy to use, or None to leave the Hub in place.

    EP_GPQA_PATH wins if set; otherwise data/gpqa/ under the repository root.
    """
    explicit = os.environ.get("EP_GPQA_PATH")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        candidates.append(Path(__file__).resolve().parent.parent / "data" / "gpqa")

    for path in candidates:
        if all((path / f).is_file() for f in _REQUIRED_FILES):
            return path
        if explicit and path is candidates[0]:
            # Being told where it is and not finding it there is a mistake worth
            # reporting, rather than quietly falling back to a gated download.
            missing = [f for f in _REQUIRED_FILES if not (path / f).is_file()]
            raise FileNotFoundError(
                f"EP_GPQA_PATH={path} does not hold a usable GPQA copy "
                f"(missing {missing}). Run `python scripts/fetch_gpqa.py --dest {path}`."
            )
    return None


def use_local_gpqa() -> Path | None:
    """Rewrite the GPQA tasks to read from a local copy. Returns the path used."""
    path = local_gpqa_dir()
    if path is None:
        return None

    try:
        from lighteval.tasks.lighteval_task import LightevalTaskConfig
        from lighteval.tasks.tasks import gpqa as gpqa_tasks
    except Exception:
        # lighteval not importable, or its layout moved: leave the Hub default
        return None

    rewritten = []
    for name in dir(gpqa_tasks):
        task = getattr(gpqa_tasks, name)
        if isinstance(task, LightevalTaskConfig) and task.hf_repo == _HUB_REPO:
            task.hf_repo = str(path)
            rewritten.append(task.name)
    if not rewritten:
        return None
    print(f"GPQA: reading from {path} instead of {_HUB_REPO} "
          f"({', '.join(sorted(rewritten))})")
    return path
