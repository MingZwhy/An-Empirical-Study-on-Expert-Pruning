"""Fixed-K for layers that take their experts from a table instead of from router scores.

DeepSeek-V4 routes its first ``num_hash_layers`` layers through ``tid2eid``, a table of expert ids
per token id whose second axis is physically the native k wide, and vLLM sizes that parameter from
``num_experts_per_tok``. Rewriting the config alone therefore cannot load: the checkpoint's table is
native-k wide and the parameter is k wide. The Fixed-K overlay instead carries copies of the shards
holding those tables, each table cut to its first k columns, which is what the paper's runs did. No
model code ranks the columns, so the prefix is one deterministic convention among several; the
score-routed layers take ordinary Fixed-K.
"""
from __future__ import annotations

import json
from pathlib import Path

HASH_TABLE_SUFFIX = ".tid2eid"


def hash_table_shards(model_dir: str) -> dict[str, list[str]]:
    """Shard file name -> the hash-routing tables it holds, read from the safetensors index."""
    index_path = Path(model_dir) / "model.safetensors.index.json"
    if not index_path.is_file():
        return {}
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.endswith(HASH_TABLE_SUFFIX):
            shards.setdefault(shard, []).append(name)
    return shards


def write_sliced_shard(source: str, destination: str, tables: list[str], k: int,
                       native_k: int) -> None:
    """Copy a shard with each named table cut to its first k columns; every other tensor as is.

    The copy has to hold the shard's other tensors too: vLLM loads every tensor in every shard the
    index names, so a full-width table left in the original shard would be loaded as well.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if name in tables:
                if tensor.dim() != 2 or tensor.shape[1] != native_k:
                    raise ValueError(f"{name} in {source} has shape {tuple(tensor.shape)}, "
                                     f"expected (*, {native_k})")
                tensor = tensor[:, :k].contiguous()
            tensors[name] = tensor
    missing = set(tables) - set(tensors)
    if missing:
        raise ValueError(f"{source} lacks {sorted(missing)}, which its index places there")
    save_file(tensors, destination, metadata=metadata)
