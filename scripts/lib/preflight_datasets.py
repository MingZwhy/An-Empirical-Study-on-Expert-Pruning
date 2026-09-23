#!/usr/bin/env python3
"""Check dataset access before a run loads a model.

One of the eleven benchmarks is gated. Without access the run still starts, spends
several minutes bringing up the engine and placing 60-160 GB of weights, and only
then fails on a 401 while fetching the first task. This turns that into a two
second answer.

    python scripts/lib/preflight_datasets.py <comma-separated task list>

Exits 0 when everything the list needs is reachable, 1 with an actionable message
when it is not, and 0 with a warning if the check itself cannot run -- a broken
check must not block a sweep.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Only GPQA is gated. The others in the suite are public, and this map stays
# deliberately short rather than trying to mirror lighteval's task registry:
# a wrong entry here would block a run that would otherwise have worked.
# repo, the data file to probe, and the page to send someone to.
#
# The probe has to be a data file, not the repo metadata: a gated: auto repo serves
# its metadata and its README to anyone, and only refuses the files. dataset_info()
# therefore succeeds without access and proves nothing.
GATED = {
    "gpqa": (
        "Idavidrein/gpqa",
        "gpqa_diamond.csv",
        "https://huggingface.co/datasets/Idavidrein/gpqa",
    ),
}


def _auto_fetch(task: str) -> bool:
    """Fetch GPQA from the authors' archive rather than asking the user to.

    It is 2.3 MB from a public URL, and it is needed for the run that was just
    started, so doing it is less surprising than stopping to explain it. Set
    EP_NO_AUTO_FETCH=1 to be told instead.
    """
    if os.environ.get("EP_NO_AUTO_FETCH") == "1":
        return False
    script = Path(__file__).resolve().parents[1] / "fetch_gpqa.py"
    if not script.is_file():
        return False
    print(f"[preflight] {task} is gated on the Hub and no local copy is present; "
          f"fetching it\n"
          f"            from the authors' archive instead (2.3 MB, no account "
          f"needed).", file=sys.stderr)
    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True)
    if proc.returncode == 0:
        print("[preflight] done; reading GPQA from data/gpqa/", file=sys.stderr)
        return True
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    print("[preflight] automatic fetch failed:", file=sys.stderr)
    for line in tail:
        print(f"            {line}", file=sys.stderr)
    return False


def _denied(task: str, url: str) -> str:
    return f"""
[preflight] {task} needs access to a gated dataset, and this run would fail on it
            only after loading the model.

It is normally fetched for you; that did not happen, either because
EP_NO_AUTO_FETCH is set or because the download failed. Pick one. From the
authors' own archive, which needs no account:

  python scripts/fetch_gpqa.py

or accept the gate on the Hub and authenticate:

  1. open {url}          (approval is automatic, not a review queue)
  2. hf auth login       (or export HF_TOKEN=...)

or leave it out of this run, which costs you one of the eleven benchmarks:

  EP_GEN_DATASETS=mmlu_pro,math_500,aime24_avg,aime25_avg,lcb:codegeneration_v6,gsm8k

The gate exists to limit contamination, which is also why this repository does not
carry the questions. The open mirrors are not a substitute: the ones with the right
198 rows collapse the three wrong answers into a single solution string, so a
four-way multiple choice cannot be rebuilt from them.
""".strip()


def needed(task_list: str) -> list[tuple[str, str, str, str]]:
    tasks = [t.strip().lower() for t in task_list.split(",") if t.strip()]
    out = []
    for key, (repo, probe, url) in GATED.items():
        hit = next((t for t in tasks if t.split(":")[0] == key or t.startswith(key)), None)
        if hit:
            out.append((hit, repo, probe, url))
    return out


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        return 0
    wanted = needed(sys.argv[1])
    if not wanted:
        return 0

    # A local copy fetched from the authors' archive needs no Hub access at all.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from expert_pruning.gpqa_local import local_gpqa_dir
        if local_gpqa_dir() is not None:
            return 0
    except FileNotFoundError as exc:      # EP_GPQA_PATH set but wrong
        print(f"[preflight] {exc}", file=sys.stderr)
        return 1
    except Exception:
        pass

    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.utils import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except Exception as exc:  # the check is a convenience, never a gate of its own
        print(f"[preflight] cannot check dataset access ({exc}); continuing", file=sys.stderr)
        return 0

    for task, repo, probe, url in wanted:
        try:
            # a HEAD against the data file: no download, and it is the request that
            # actually gets refused without access
            get_hf_file_metadata(hf_hub_url(repo, probe, repo_type="dataset"))
        except GatedRepoError:
            if _auto_fetch(task):
                continue
            print(_denied(task, url), file=sys.stderr)
            return 1
        except RepositoryNotFoundError:
            print(f"[preflight] {repo} not found; is the name right?", file=sys.stderr)
            return 1
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                if _auto_fetch(task):
                    continue
                print(_denied(task, url), file=sys.stderr)
                return 1
            # offline, rate limited, behind a proxy: not ours to adjudicate
            print(f"[preflight] could not reach {repo} (HTTP {status}); continuing",
                  file=sys.stderr)
            return 0
        except OSError as exc:
            print(f"[preflight] could not reach {repo} ({exc.__class__.__name__}); "
                  "continuing", file=sys.stderr)
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
