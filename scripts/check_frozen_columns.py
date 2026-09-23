"""Why the old table never showed DiEP collapsing.

DiEP and NAEE are driven by the same two flags, --naee_beta and --naee_k_min. If the old
lighteval cache key did not separate them, runs would replay each other's samples, and a
replayed score cannot move when the pruning gets stronger. So the tell is a score column that
stays constant as beta rises, and is shared between the two methods.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COLS = ["mmlu_pro", "gpqa", "math500", "aime24", "aime25", "lcb", "gsm8k"]


def rows(model: str) -> dict:
    """(method, beta, k_min) -> (matched_avg, mean, [7 scores]); one entry per config."""
    out = {}
    for line in (REPO / "compact" / f"{model}.md").read_text().splitlines():
        if not line.startswith("|"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 5 + len(COLS) or c[0] in ("method", "---"):
            continue
        if c[0] not in ("NAEE", "DiEP"):
            continue
        beta = re.search(r"beta=([0-9.]+)", c[2])
        kmin = re.search(r"k_min=(\d+)", c[2])
        if not beta or not kmin:
            continue
        try:
            scores = [float(x) for x in c[5:5 + len(COLS)]]
            avg, mean = float(c[1]), float(c[3])
        except ValueError:
            continue
        out[(c[0], float(beta.group(1)), kmin.group(1))] = (avg, mean, scores)
    return out


for model in sys.argv[1:] or ["Qwen3-30B-A3B-Instruct-2507"]:
    table = rows(model)
    print(f"\n{'=' * 108}\n{model}: old-table scores as beta rises (k_min=2)\n{'=' * 108}")
    for method in ("NAEE", "DiEP"):
        keys = sorted(k for k in table if k[0] == method and k[2] == "2")
        if not keys:
            continue
        print(f"\n{method}")
        print(f"  {'beta':>5s} {'avg':>6s} {'mean':>6s} | "
              + " ".join(f"{c:>9s}" for c in COLS))
        for key in keys:
            avg, mean, scores = table[key]
            print(f"  {key[1]:5.2f} {avg:6.2f} {mean:6.2f} | "
                  + " ".join(f"{s:9.2f}" for s in scores))
        frozen = [c for i, c in enumerate(COLS)
                  if len({table[k][2][i] for k in keys}) == 1]
        print(f"  columns that never move across {len(keys)} beta settings: "
              + (", ".join(frozen) if frozen else "none"))
