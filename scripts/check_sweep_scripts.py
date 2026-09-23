#!/usr/bin/env python3
"""Check the sweep scripts without a GPU or a checkpoint.

Every configuration in scripts/models/ is dry-run, and each emitted command is
checked against main.py's actual argument parser: unknown flags, flags given no
value, duplicate configuration names, and output directories that collide. A
typo in a knob name would otherwise surface only once a real run had loaded 60 GB
of weights and failed at argument parsing.

    python scripts/check_sweep_scripts.py            # all models, all stages
    python scripts/check_sweep_scripts.py -v         # also print every command
"""
from __future__ import annotations

import argparse
import ast
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "scripts" / "models"
ANSI = re.compile(r"\033\[[0-9;]*m")


def parser_surface() -> tuple[dict[str, bool], set[str]]:
    """Read main.py's parse_args and return {flag: takes_a_value}, plus choices-bearing flags."""
    tree = ast.parse((REPO / "main.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
    flags: dict[str, bool] = {}
    choices: dict[str, list[str]] = {}
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        names = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        kw = {k.arg: k.value for k in node.keywords}
        action = kw.get("action")
        takes_value = not (isinstance(action, ast.Constant)
                           and str(action.value).startswith("store_"))
        for name in names:
            flags[name] = takes_value
            if "choices" in kw:
                try:
                    choices[name] = list(ast.literal_eval(kw["choices"]))
                except Exception:
                    pass
    return flags, choices


def dry_run(script: Path) -> str:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "TERM": "dumb"}
    proc = subprocess.run(["bash", str(script), "--dry-run"],
                          capture_output=True, text=True, cwd=REPO, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"[error] {script.relative_to(REPO)} failed to dry-run:\n"
                         f"{proc.stdout}\n{proc.stderr}")
    return ANSI.sub("", proc.stdout)


def commands(text: str):
    """Yield (config_name, argv) for each command block in a dry-run transcript."""
    name = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("  # "):
            name = line[4:].strip()
            continue
        stripped = line.strip()
        if stripped.startswith("python main.py") or (buf and stripped.startswith("--")):
            buf.append(stripped.rstrip("\\").strip())
            if not line.rstrip().endswith("\\"):
                yield name, shlex.split(" ".join(buf))
                buf = []
    if buf:
        yield name, shlex.split(" ".join(buf))


def check_model_fields(model_dirs) -> list[str]:
    """Every model.env value must survive the trip through run.sh.

    model.env writes values as ${VAR:-default} so the environment can override
    them, and run.sh has to evaluate that rather than read the file as text.
    Reading it textually is silent: the guard compares a literal ${...} against a
    vLLM version, refuses every model, and inverts its own advice, while dry runs
    keep passing because they skip the guard. So check the values, not the parsing.
    """
    problems = []

    # The per-model tables are the only thing under results/ that is tracked, and a
    # stray rm -rf while testing took one out without anything noticing until a
    # README link broke. They are cheap to assert.
    for link in ("Qwen3-30B-A3B-Instruct-2507", "Qwen3-Next-80B-A3B-Instruct",
                 "Ling-lite-1.5-2507", "gpt-oss-20b", "MiniMax-M2.7",
                 "DeepSeek-V2-Lite-Chat", "DeepSeek-V4-Flash-0731", "Hy3"):
        table = REPO / "results" / link / "RESULTS.md"
        if not table.is_file():
            problems.append(f"results/{link}/RESULTS.md is missing; the README links to it")

    listing = subprocess.run(["bash", str(REPO / "scripts" / "run.sh"), "list"],
                             capture_output=True, text=True, cwd=REPO)
    if "${" in ANSI.sub("", listing.stdout):
        problems.append("scripts/run.sh list prints an unexpanded ${...}; "
                        "model.env is being read as text rather than sourced")

    for mdir in model_dirs:
        for var, valid in (("EP_ENVIRONMENT", {"expertpruning", "fixedk", "either"}), ("EP_NATIVE_K", None),
                           ("EP_TP", None), ("EP_MODEL_NAME", None)):
            got = subprocess.run(
                ["bash", "-c", f'source "{mdir}/model.env" 2>/dev/null; printf "%s" "${{{var}-}}"'],
                capture_output=True, text=True).stdout
            if not got:
                problems.append(f"{mdir.name}: {var} is unset")
            elif "${" in got:
                problems.append(f"{mdir.name}: {var} resolves to {got!r}, not a value")
            elif valid and got not in valid:
                problems.append(f"{mdir.name}: {var}={got!r} is not one of {sorted(valid)}")
    return problems


def check_readme_model_table(model_dirs) -> list[str]:
    """The READMEs list each model's environment and GPU count; keep them true.

    Duplicating those two facts into prose is worth it -- a reader should not have to
    install anything to see which models exist -- but only if the copy cannot drift
    away from the scripts, which is what this checks.
    """
    problems = []
    declared = {}
    for mdir in model_dirs:
        got = {}
        for var in ("EP_ENVIRONMENT", "EP_TP"):
            got[var] = subprocess.run(
                ["bash", "-c", f'source "{mdir}/model.env" 2>/dev/null; printf "%s" "${{{var}-}}"'],
                capture_output=True, text=True).stdout
        declared[mdir.name] = got

    for readme in (REPO / "README.md", REPO / "docs" / "README.zh-CN.md"):
        text = readme.read_text()
        for name, got in declared.items():
            # rows look like: | `qwen3-30b` | every rule | 2 | `expertpruning` |
            short = next((m for m in re.findall(r"^\| `([a-z0-9.-]+)` \|.*$", text, re.M)
                          if name.startswith(m)), None)
            if short is None:
                problems.append(f"{readme.name}: no row for {name}")
                continue
            row = next(l for l in text.splitlines()
                       if l.startswith(f"| `{short}` |"))
            cells = [c.strip().strip("`") for c in row.strip("|").split("|")]
            gpus, env = cells[2], cells[3]
            if gpus != got["EP_TP"]:
                problems.append(f"{readme.name}: {short} lists {gpus} GPUs, "
                                f"model.env says {got['EP_TP']}")
            expected_env = got["EP_ENVIRONMENT"]
            if expected_env == "either":
                if env not in ("either", "两者皆可"):
                    problems.append(f"{readme.name}: {short} lists {env!r}, expected either")
            elif env != expected_env:
                problems.append(f"{readme.name}: {short} lists {env!r}, "
                                f"model.env says {expected_env!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--model", help="only this model directory")
    args = ap.parse_args()

    flags, choices = parser_surface()
    print(f"main.py exposes {len(flags)} flags, {len(choices)} of them with fixed choices\n")

    problems: list[str] = []
    total_cmds = 0
    seen_out: dict[str, str] = {}

    model_dirs = sorted(d for d in MODELS.iterdir() if d.is_dir())
    if args.model:
        model_dirs = [d for d in model_dirs if args.model in d.name]

    for mdir in model_dirs:
        stages = sorted(mdir.glob("*.sh"))
        names_here: Counter[str] = Counter()
        n_cmds = 0
        for script in stages:
            text = dry_run(script)
            for name, argv in commands(text):
                n_cmds += 1
                total_cmds += 1
                if name:
                    names_here[name] += 1
                where = f"{mdir.name}/{script.name}:{name}"

                if argv[:2] != ["python", "main.py"]:
                    problems.append(f"{where}: does not invoke main.py: {argv[:2]}")
                    continue

                i, rest = 2, argv[2:]
                while i - 2 < len(rest):
                    tok = rest[i - 2]
                    i += 1
                    if not tok.startswith("--"):
                        problems.append(f"{where}: stray token {tok!r}")
                        continue
                    if tok not in flags:
                        near = [f for f in flags if f.strip("-")[:6] == tok.strip("-")[:6]]
                        problems.append(f"{where}: unknown flag {tok}"
                                        + (f" (did you mean {', '.join(near)}?)" if near else ""))
                        continue
                    if flags[tok]:
                        nxt = rest[i - 2] if i - 2 < len(rest) else None
                        if nxt is None or nxt.startswith("--"):
                            problems.append(f"{where}: {tok} needs a value")
                        else:
                            if tok in choices and nxt not in choices[tok]:
                                problems.append(f"{where}: {tok}={nxt} not in {choices[tok]}")
                            i += 1

                # A flag-level check cannot see this one: every flag is valid, but
                # a router-patch run that names neither execution mode dies at the
                # end, after the whole evaluation, because main.py will not report
                # an average expert count it cannot trust.
                if "--use_local_expert_router" in rest and not (
                        {"--enforce_eager", "--allow_compiled_router"} & set(rest)):
                    problems.append(f"{where}: uses --use_local_expert_router with neither "
                                    "--enforce_eager nor --allow_compiled_router; the run would "
                                    "fail at the end with average_selected_experts unavailable")

                out = next((rest[j + 1] for j, t in enumerate(rest) if t == "--output_dir"), None)
                if out is None:
                    problems.append(f"{where}: no --output_dir")
                else:
                    # A generative stage and the QA stage legitimately share a
                    # directory when a tier's two knobs coincide; the two write
                    # differently named scores files. Different configurations
                    # sharing a directory is a real collision.
                    prev = seen_out.get(out)
                    if prev is not None and prev.split(":")[-1] != (name or ""):
                        problems.append(f"{where}: --output_dir collides with {prev}")
                    seen_out[out] = where

                if args.verbose:
                    print(f"  {where}\n      {' '.join(argv[2:])}")

        # A configuration may appear at most once per regime: once in a
        # generative stage and once in qa.sh. Twice within one regime is a bug.
        dupes = [n for n, c in names_here.items() if c > 2]
        if dupes:
            problems.append(f"{mdir.name}: configuration appears more than twice: "
                            f"{', '.join(dupes)}")
        print(f"  {mdir.name:32} {len(stages):2} stages, {n_cmds:3} configurations")

    problems += check_model_fields(model_dirs)
    if not args.model:
        problems += check_readme_model_table(model_dirs)

    print(f"\n{total_cmds} commands checked across {len(model_dirs)} models")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("no unknown flags, no missing values, no colliding output directories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
