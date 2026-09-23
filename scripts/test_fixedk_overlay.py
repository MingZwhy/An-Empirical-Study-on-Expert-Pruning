#!/usr/bin/env python3
"""Check that the Fixed-K overlay finds, and writes back to, the right config key.

Checkpoints disagree on what to call the per-token expert count, and the name a
config uses is the name vLLM reads. So the one property that matters is that the
overlay rewrites *that* key and introduces no other: adding `num_experts_per_tok`
to a config that spells it `top_k_experts` is silently ignored by the engine, and
the run then computes the unpruned budget while reporting a pruned one. Nothing
crashes, and the numbers look plausible, which is what makes it worth a test.

Runs on CPU in under a second; needs no model and no GPU.

    python scripts/test_fixedk_overlay.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expert_pruning.config_shims import (  # noqa: E402
    get_moe_num_experts_per_tok as get_k,
    set_moe_num_experts_per_tok as set_k,
)

# name, config, expected native k, the key that must change, keys that must not appear
CASES = [
    ("Qwen3 / DeepSeek / MiniMax / Ling (top level)",
     {"num_experts_per_tok": 8, "num_experts": 128},
     8, ("num_experts_per_tok",), ["top_k_experts"]),

    ("Qwen3.5 (under text_config)",
     {"text_config": {"num_experts_per_tok": 8, "num_experts": 128}},
     8, ("text_config", "num_experts_per_tok"), ["top_k_experts"]),

    ("Gemma 4 (text_config.top_k_experts)",
     {"text_config": {"top_k_experts": 8, "num_experts": 128}},
     8, ("text_config", "top_k_experts"), ["num_experts_per_tok"]),

    ("gpt-oss (duplicate experts_per_token)",
     {"num_experts_per_tok": 4, "experts_per_token": 4, "num_local_experts": 32},
     4, ("num_experts_per_tok",), ["top_k_experts"]),
]

TARGET_K = 6


def at(config: dict, path: tuple[str, ...]):
    node = config
    for part in path:
        node = node[part]
    return node


def flatten(config: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def main() -> int:
    failures = []

    for name, original, native_k, path, forbidden in CASES:
        config = copy.deepcopy(original)

        read = get_k(config)
        if read != native_k:
            failures.append(f"{name}: read {read}, expected {native_k}")
            continue

        set_k(config, TARGET_K)

        written = at(config, path[:-1])[path[-1]]
        if written != TARGET_K:
            failures.append(f"{name}: {'.'.join(path)} is {written}, expected {TARGET_K}")

        # No key may be invented. This is the property that keeps a silently
        # unpruned run from being reported as a pruned one.
        appeared = set(flatten(config)) - set(flatten(original))
        if appeared:
            failures.append(f"{name}: overlay invented {sorted(appeared)}; "
                            "the engine would ignore it and prune nothing")
        for key in forbidden:
            if key in flatten(config) or any(key == k.split(".")[-1] for k in flatten(config)):
                failures.append(f"{name}: {key} must not be present")

        # Every other field has to survive untouched.
        changed = {k for k, v in flatten(config).items()
                   if flatten(original).get(k) != v}
        expected_changed = {".".join(path)}
        if "experts_per_token" in original:
            expected_changed.add("experts_per_token")
        if changed != expected_changed:
            failures.append(f"{name}: changed {sorted(changed)}, "
                            f"expected only {sorted(expected_changed)}")

        print(f"  ok  {name}\n        {'.'.join(path)}: {native_k} -> {written}"
              f"{'  (experts_per_token kept in step)' if 'experts_per_token' in original else ''}")

    # A dense model must not look like a prunable one.
    dense = {"hidden_size": 2816, "num_hidden_layers": 30}
    if get_k(dense) is not None:
        failures.append("a config with no expert count should read as None")
    else:
        print("  ok  dense config reads as None, so main.py refuses the override")

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
