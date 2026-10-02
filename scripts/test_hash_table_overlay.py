#!/usr/bin/env python3
"""Check the hash-table slicing behind DeepSeek-V4's Fixed-K overlay, on a synthetic checkpoint.

The sliced shard has to keep the table's first k columns, keep every other tensor bit for bit, and
keep the shard metadata; a table of unexpected width, or one missing from the shard its index
names, has to be refused rather than written.

Runs on CPU in a second or two; needs torch and safetensors (either environment has both), no
model and no GPU.

    python scripts/test_hash_table_overlay.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:
    print(f"skipped: {exc}")
    sys.exit(0)

from expert_pruning.hash_routing import hash_table_shards, write_sliced_shard  # noqa: E402

TABLE = "layers.0.ffn.gate.tid2eid"


def main() -> int:
    failures = []

    def check(ok: bool, what: str) -> None:
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
        if not ok:
            failures.append(what)

    gen = torch.Generator().manual_seed(0)
    table = torch.randint(0, 256, (11, 6), generator=gen, dtype=torch.int64)
    gate = torch.randn(4, 3, generator=gen)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "model"
        src.mkdir()
        save_file({TABLE: table, "layers.0.ffn.gate.weight": gate}, src / "a.safetensors",
                  metadata={"format": "pt"})
        save_file({"layers.3.ffn.gate.weight": torch.randn(2, 2, generator=gen)},
                  src / "b.safetensors", metadata={"format": "pt"})
        weight_map = {TABLE: "a.safetensors", "layers.0.ffn.gate.weight": "a.safetensors",
                      "layers.3.ffn.gate.weight": "b.safetensors"}
        (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

        shards = hash_table_shards(str(src))
        check(shards == {"a.safetensors": [TABLE]}, "only the shard holding the table is selected")

        out = Path(tmp) / "a_k4.safetensors"
        write_sliced_shard(str(src / "a.safetensors"), str(out), shards["a.safetensors"], 4, 6)
        with safe_open(str(out), framework="pt") as handle:
            check(torch.equal(handle.get_tensor(TABLE), table[:, :4]), "table keeps its first 4 columns")
            check(torch.equal(handle.get_tensor("layers.0.ffn.gate.weight"), gate),
                  "the shard's other tensors are unchanged")
            check(handle.metadata() == {"format": "pt"}, "shard metadata is kept")

        for what, tables, native_k in (("a table of unexpected width is refused", [TABLE], 8),
                                       ("a table missing from its shard is refused",
                                        ["layers.1.ffn.gate.tid2eid"], 6)):
            try:
                write_sliced_shard(str(src / "a.safetensors"), str(Path(tmp) / "bad.safetensors"),
                                   tables, 4, native_k)
                check(False, what)
            except ValueError:
                check(not (Path(tmp) / "bad.safetensors").exists(), what)

    print("\nall checks passed" if not failures else f"\n{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
