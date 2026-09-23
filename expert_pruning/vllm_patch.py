"""Monkey patches that route vLLM MoE expert selection through local code."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Optional

import torch

from expert_pruning.attention_protection import (
    configure_attention_position_context,
    configure_attention_sink_probe,
    get_moe_protected_mask,
    get_moe_token_segments,
    observe_attention_qk,
    record_attention_positions,
    skip_attention_sink_probe,
)
from expert_pruning.routing import (
    configure_expert_router,
    current_routing_method,
    is_expert_routing_stats_skipped,
    prepare_ban_tensors_for_device,
    prepare_diep_tensors_for_device,
    prepare_router_value_tensors_for_device,
    set_current_eac_prefill_mask,
    set_current_input_token_ids,
    set_current_moe_layer,
    set_current_protected_token_mask,
    set_current_token_decode_offsets,
    set_current_token_segment_seq_lens,
    set_current_token_segments,
    set_current_token_seq_lens,
    skip_expert_routing_stats,
    torch_fused_topk,
)
from expert_pruning.router_distribution import (
    configure_router_distribution,
    is_router_distribution_enabled,
    is_router_distribution_skipped,
    skip_router_distribution,
    wrap_fused_topk_observer,
)

_original_fused_topk: Optional[Callable] = None
_original_package_fused_topk: Optional[Callable] = None
_patched_dummy_run_classes: set[type] = set()
_patched_fused_moe_layer_classes: set[type] = set()
_patched_attention_layer_classes: set[type] = set()
_patched_input_token_model_classes: set[type] = set()
_decode_offset_base_by_block_key: dict[tuple[int, ...], int] = {}


def _is_cuda_graph_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is None or not hasattr(compiler, "is_compiling"):
        return False
    return bool(compiler.is_compiling())


def _iter_attention_metadata(attn_metadata: Any):
    if attn_metadata is None:
        return
    if isinstance(attn_metadata, dict):
        yield from attn_metadata.values()
    else:
        yield attn_metadata


def _get_forward_context_attention_metadata() -> Optional[Any]:
    try:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
    except Exception:
        return None

    for metadata in _iter_attention_metadata(
            getattr(forward_context, "attn_metadata", None)):
        if (hasattr(metadata, "num_prefill_tokens")
                or (hasattr(metadata, "query_start_loc")
                    and hasattr(metadata, "max_query_len")
                    and hasattr(metadata, "num_actual_tokens"))):
            return metadata
    return None


def _query_start_loc_values(metadata: Any) -> Optional[list[int]]:
    query_start_loc = getattr(metadata, "query_start_loc_cpu", None)
    if query_start_loc is None:
        query_start_loc = getattr(metadata, "query_start_loc", None)
    if query_start_loc is None:
        return None
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None
    return [int(x) for x in query_start_loc.detach().to("cpu").tolist()]


def _seq_lens_values(metadata: Any) -> Optional[list[int]]:
    seq_lens = getattr(metadata, "seq_lens_cpu", None)
    if seq_lens is None:
        seq_lens = getattr(metadata, "seq_lens", None)
    if seq_lens is None:
        return None
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None
    if isinstance(seq_lens, torch.Tensor):
        return [int(x) for x in seq_lens.detach().to("cpu").tolist()]
    return [int(x) for x in seq_lens]


def _num_computed_tokens_values(metadata: Any) -> Optional[list[int]]:
    values = getattr(metadata, "num_computed_tokens_cpu", None)
    if values is None:
        return None
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None
    if isinstance(values, torch.Tensor):
        return [int(x) for x in values.detach().to("cpu").tolist()]
    return [int(x) for x in values]


def _block_table_key(metadata: Any, request_index: int) -> Optional[tuple[int, ...]]:
    block_table = getattr(metadata, "block_table_tensor", None)
    if block_table is None:
        return None
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None
    try:
        row = block_table[request_index].detach().to("cpu").reshape(-1)
    except Exception:
        return None
    values = [int(x) for x in row.tolist() if int(x) >= 0]
    if not values:
        return None
    return tuple(values[:8])


def _decode_offsets_from_forward_context(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None

    metadata = _get_forward_context_attention_metadata()
    if metadata is None:
        return None

    starts = _query_start_loc_values(metadata)
    computed_tokens = _num_computed_tokens_values(metadata)
    seq_lens = _seq_lens_values(metadata)
    using_seq_lens_fallback = computed_tokens is None
    length_values = computed_tokens if computed_tokens is not None else seq_lens
    if length_values is None:
        return None

    values = torch.full(
        (num_tokens, ),
        -1,
        device=device,
        dtype=torch.long,
    )
    wrote_offset = False
    if starts is not None and len(starts) >= 2:
        for idx, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
            if idx >= len(length_values):
                break
            start = max(0, min(int(start), num_tokens))
            end = max(0, min(int(end), num_tokens))
            query_len = end - start
            if query_len != 1:
                continue
            length_value = int(length_values[idx])
            key = (
                (idx, ) if using_seq_lens_fallback else
                _block_table_key(metadata, idx))
            if key is None:
                key = (idx, )
            base = _decode_offset_base_by_block_key.get(key)
            if base is None or length_value < base:
                base = length_value
                _decode_offset_base_by_block_key[key] = base
                if len(_decode_offset_base_by_block_key) > 4096:
                    _decode_offset_base_by_block_key.clear()
                    _decode_offset_base_by_block_key[key] = base
            values[start] = max(0, length_value - base)
            wrote_offset = True

    if not wrote_offset and len(length_values) == num_tokens:
        prefill_mask = _prefill_mask_from_forward_context(num_tokens, device)
        if prefill_mask is not None:
            prefill_mask_cpu = prefill_mask.detach().to("cpu").tolist()
        else:
            prefill_mask_cpu = [False] * num_tokens
        for idx, length_value in enumerate(length_values):
            if bool(prefill_mask_cpu[idx]):
                continue
            length_value = int(length_value)
            key = (
                (idx, ) if using_seq_lens_fallback else
                _block_table_key(metadata, idx))
            if key is None:
                key = (idx, )
            base = _decode_offset_base_by_block_key.get(key)
            if base is None or length_value < base:
                base = length_value
                _decode_offset_base_by_block_key[key] = base
                if len(_decode_offset_base_by_block_key) > 4096:
                    _decode_offset_base_by_block_key.clear()
                    _decode_offset_base_by_block_key[key] = base
            values[idx] = max(0, length_value - base)
            wrote_offset = True

    return values if wrote_offset else None


def _v1_prefill_segments_from_metadata(
        metadata: Any) -> Optional[list[tuple[int, int]]]:
    if not (hasattr(metadata, "query_start_loc")
            and hasattr(metadata, "max_query_len")
            and hasattr(metadata, "num_actual_tokens")):
        return None

    max_query_len = int(getattr(metadata, "max_query_len", 0) or 0)
    if max_query_len <= 1:
        return None

    starts = _query_start_loc_values(metadata)
    if starts is None or len(starts) < 2:
        return None

    query_lens = [end - start for start, end in zip(starts, starts[1:])]
    first_prefill = None
    for idx, query_len in enumerate(query_lens):
        if query_len > 1:
            first_prefill = idx
            break
    if first_prefill is None:
        return None

    num_actual_tokens = int(getattr(metadata, "num_actual_tokens", 0) or 0)
    segments: list[tuple[int, int]] = []
    for start, end in zip(starts[first_prefill:],
                          starts[first_prefill + 1:]):
        start = max(0, min(int(start), num_actual_tokens))
        end = max(0, min(int(end), num_actual_tokens))
        if end - start > 1:
            segments.append((start, end))
    return segments or None


def _v1_prefill_segment_seq_lens_from_metadata(
        metadata: Any) -> Optional[list[tuple[int, int, int]]]:
    if not (hasattr(metadata, "query_start_loc")
            and hasattr(metadata, "max_query_len")
            and hasattr(metadata, "num_actual_tokens")):
        return None

    max_query_len = int(getattr(metadata, "max_query_len", 0) or 0)
    if max_query_len <= 1:
        return None

    starts = _query_start_loc_values(metadata)
    seq_lens = _seq_lens_values(metadata)
    if starts is None or seq_lens is None or len(starts) < 2:
        return None

    query_lens = [end - start for start, end in zip(starts, starts[1:])]
    first_prefill = None
    for idx, query_len in enumerate(query_lens):
        if query_len > 1:
            first_prefill = idx
            break
    if first_prefill is None:
        return None

    num_actual_tokens = int(getattr(metadata, "num_actual_tokens", 0) or 0)
    segments: list[tuple[int, int, int]] = []
    for idx, (start, end) in enumerate(
            zip(starts[first_prefill:], starts[first_prefill + 1:]),
            start=first_prefill):
        start = max(0, min(int(start), num_actual_tokens))
        end = max(0, min(int(end), num_actual_tokens))
        if end - start > 1 and idx < len(seq_lens):
            segments.append((start, end, int(seq_lens[idx])))
    return segments or None


def _query_start_loc_segments(metadata: Any,
                              num_prefill_tokens: int
                              ) -> Optional[list[tuple[int, int]]]:
    query_start_loc = getattr(metadata, "query_start_loc", None)
    num_prefills = int(getattr(metadata, "num_prefills", 0) or 0)
    if query_start_loc is None or num_prefills <= 0:
        return None
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None

    starts = query_start_loc[:num_prefills + 1].detach().to("cpu").tolist()
    segments: list[tuple[int, int]] = []
    for start, end in zip(starts, starts[1:]):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        start = max(0, min(start, num_prefill_tokens))
        end = max(0, min(end, num_prefill_tokens))
        if end > start:
            segments.append((start, end))
    return segments or None


def _query_start_loc_segment_seq_lens(
        metadata: Any,
        num_prefill_tokens: int) -> Optional[list[tuple[int, int, int]]]:
    query_start_loc = getattr(metadata, "query_start_loc", None)
    num_prefills = int(getattr(metadata, "num_prefills", 0) or 0)
    seq_lens = _seq_lens_values(metadata)
    if query_start_loc is None or num_prefills <= 0 or seq_lens is None:
        return None
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None

    starts = query_start_loc[:num_prefills + 1].detach().to("cpu").tolist()
    segments: list[tuple[int, int, int]] = []
    for idx, (start, end) in enumerate(zip(starts, starts[1:])):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        start = max(0, min(start, num_prefill_tokens))
        end = max(0, min(end, num_prefill_tokens))
        if end > start and idx < len(seq_lens):
            segments.append((start, end, int(seq_lens[idx])))
    return segments or None


def _prefill_segments_from_forward_context(
        num_tokens: int) -> Optional[list[tuple[int, int]]]:
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None

    metadata = _get_forward_context_attention_metadata()
    if metadata is None:
        return None

    v1_segments = _v1_prefill_segments_from_metadata(metadata)
    if v1_segments is not None:
        segments = [
            (max(0, min(start, num_tokens)), max(0, min(end, num_tokens)))
            for start, end in v1_segments if end > start
        ]
        return segments or None

    num_prefill_tokens = int(
        getattr(metadata, "num_prefill_tokens", 0) or 0)
    num_prefill_tokens = max(0, min(num_prefill_tokens, num_tokens))
    if num_prefill_tokens <= 1:
        return None

    return (_query_start_loc_segments(metadata, num_prefill_tokens)
            or [(0, num_prefill_tokens)])


def _prefill_segment_seq_lens_from_forward_context(
        num_tokens: int) -> Optional[list[tuple[int, int, int]]]:
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None

    metadata = _get_forward_context_attention_metadata()
    if metadata is None:
        return None

    v1_segments = _v1_prefill_segment_seq_lens_from_metadata(metadata)
    if v1_segments is not None:
        segments = [
            (
                max(0, min(start, num_tokens)),
                max(0, min(end, num_tokens)),
                int(seq_len),
            )
            for start, end, seq_len in v1_segments if end > start
        ]
        return segments or None

    num_prefill_tokens = int(
        getattr(metadata, "num_prefill_tokens", 0) or 0)
    num_prefill_tokens = max(0, min(num_prefill_tokens, num_tokens))
    if num_prefill_tokens <= 1:
        return None

    return (_query_start_loc_segment_seq_lens(metadata, num_prefill_tokens)
            or [(0, num_prefill_tokens, num_prefill_tokens)])


def _token_seq_lens_from_forward_context(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None

    metadata = _get_forward_context_attention_metadata()
    if metadata is None:
        return None

    seq_lens = _seq_lens_values(metadata)
    if seq_lens is None:
        return None
    if len(seq_lens) == num_tokens:
        return torch.tensor(seq_lens, device=device, dtype=torch.long)

    starts = _query_start_loc_values(metadata)
    if starts is not None and len(starts) >= 2:
        values = torch.full(
            (num_tokens, ),
            -1,
            device=device,
            dtype=torch.long,
        )
        for idx, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
            if idx >= len(seq_lens):
                break
            start = max(0, min(int(start), num_tokens))
            end = max(0, min(int(end), num_tokens))
            if end > start:
                values[start:end] = int(seq_lens[idx])
        if (values >= 0).any().item():
            return values

    if 0 < len(seq_lens) <= num_tokens:
        values = torch.full(
            (num_tokens, ),
            -1,
            device=device,
            dtype=torch.long,
        )
        values[:len(seq_lens)] = torch.tensor(
            seq_lens,
            device=device,
            dtype=torch.long,
        )
        return values
    return None


def _prefill_mask_from_forward_context(
        num_tokens: int,
        device: torch.device) -> Optional[torch.Tensor]:
    if _is_cuda_graph_capturing() or _is_torch_compiling():
        return None

    metadata = _get_forward_context_attention_metadata()
    if metadata is None:
        return None

    query_start_loc = getattr(metadata, "query_start_loc", None)
    if query_start_loc is not None and hasattr(metadata, "max_query_len"):
        query_start_loc = query_start_loc.to(device=device)
        if query_start_loc.numel() < 2:
            return None
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        is_prefill = query_lens > 1
        has_prefill = is_prefill.any()
        first_prefill = is_prefill.to(dtype=torch.int64).argmax(dim=-1)
        no_prefill_start = torch.full(
            (),
            num_tokens,
            dtype=query_start_loc.dtype,
            device=device,
        )
        prefill_start = torch.where(
            has_prefill,
            query_start_loc[first_prefill],
            no_prefill_start,
        )
        prefill_end = min(
            int(getattr(metadata, "num_actual_tokens", num_tokens)
                or num_tokens),
            num_tokens,
        )
        token_positions = torch.arange(num_tokens, device=device)
        return ((token_positions >= prefill_start.to(device=device))
                & (token_positions < prefill_end))

    num_prefill_tokens = int(
        getattr(metadata, "num_prefill_tokens", 0) or 0)
    num_prefill_tokens = max(0, min(num_prefill_tokens, num_tokens))
    if num_prefill_tokens <= 0:
        return None
    token_positions = torch.arange(num_tokens, device=device)
    return token_positions < num_prefill_tokens


def _patch_dummy_run_stats_guard(module_name: str, class_name: str) -> None:
    """Disable local routing stats inside vLLM dummy/profile runs."""
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return

    cls = getattr(module, class_name, None)
    if cls is None or cls in _patched_dummy_run_classes:
        return

    original_dummy_run = getattr(cls, "_dummy_run", None)
    if original_dummy_run is None:
        return

    def wrapped_dummy_run(self, *args, **kwargs):
        with skip_expert_routing_stats(), skip_attention_sink_probe(
        ), skip_router_distribution():
            return original_dummy_run(self, *args, **kwargs)

    wrapped_dummy_run.__name__ = getattr(original_dummy_run, "__name__",
                                         "_dummy_run")
    wrapped_dummy_run.__doc__ = getattr(original_dummy_run, "__doc__", None)
    setattr(cls, "_dummy_run", wrapped_dummy_run)
    _patched_dummy_run_classes.add(cls)


def _install_dummy_run_stats_guards() -> None:
    _patch_dummy_run_stats_guard("vllm.v1.worker.gpu_model_runner",
                                 "GPUModelRunner")
    _patch_dummy_run_stats_guard("vllm.worker.model_runner",
                                 "GPUModelRunnerBase")
    _patch_dummy_run_stats_guard("vllm.worker.model_runner", "ModelRunner")


def _assert_layer_routes_through_patch(layer: Any) -> None:
    """Fail loudly if a MoE layer selects experts on a path this patch cannot reach.

    ``FusedMoE.select_experts`` only calls ``fused_topk`` — the function we replace —
    when the layer uses neither grouped top-k nor a custom routing function. Models that
    do (DeepSeek-style grouped routers, for instance) would run completely unpruned while
    still reporting as an expert-pruning experiment, making every method look identical
    to the baseline.
    """
    if current_routing_method() == "none":
        return
    reason = None
    if getattr(layer, "use_grouped_topk", False):
        reason = "use_grouped_topk=True (routes through grouped_topk)"
    elif getattr(layer, "custom_routing_function", None) is not None:
        reason = "custom_routing_function is set"
    if reason is None:
        return
    raise RuntimeError(
        "Expert pruning cannot intercept expert selection for MoE layer "
        f"{getattr(layer, 'layer_name', '<unknown>')}: {reason}. "
        "vLLM would bypass the patched fused_topk, so the run would silently produce "
        "unpruned baseline results. Extend expert_pruning to this routing path before "
        "evaluating this model.")


def _patch_fused_moe_layer_context() -> None:
    try:
        module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.layer")
    except Exception:
        return

    cls = getattr(module, "FusedMoE", None)
    if cls is None or cls in _patched_fused_moe_layer_classes:
        return

    # vLLM <=0.10 exposes forward_impl methods. vLLM 0.17's FusedMoE is a
    # CustomOp whose only class-level entrypoint is forward(), which dispatches
    # through an instance _forward_method. Wrap exactly one layer boundary:
    # prefer the implementation methods when present, otherwise forward().
    method_names = [
        name for name in ("forward_impl", "forward_impl_chunked")
        if getattr(cls, name, None) is not None
    ]
    if not method_names and getattr(cls, "forward", None) is not None:
        method_names = ["forward"]

    for method_name in method_names:
        original = getattr(cls, method_name, None)
        if original is None:
            continue

        def make_wrapped(original_method):
            def wrapped(self, *args, **kwargs):
                _assert_layer_routes_through_patch(self)
                hidden_states = args[0] if args else kwargs.get(
                    "hidden_states")
                protected_mask = None
                token_segments = None
                token_segment_seq_lens = None
                token_seq_lens = None
                token_decode_offsets = None
                eac_prefill_mask = None
                if isinstance(hidden_states, torch.Tensor):
                    method = current_routing_method()
                    prepare_diep_tensors_for_device(hidden_states.device)
                    prepare_ban_tensors_for_device(hidden_states.device)
                    prepare_router_value_tensors_for_device(
                        hidden_states.device)
                    if method == "mc_moe":
                        protected_mask = get_moe_protected_mask(
                            getattr(self, "layer_name", None), hidden_states)
                    if ((method in {
                            "eac_moe",
                            "bandedlayerbudget_dynamic_routing",
                    } or is_router_distribution_enabled())
                            and not is_expert_routing_stats_skipped()
                            and not is_router_distribution_skipped()
                            and not _is_cuda_graph_capturing()
                            and not _is_torch_compiling()):
                        eac_prefill_mask = _prefill_mask_from_forward_context(
                            hidden_states.size(0), hidden_states.device)
                        token_segments = get_moe_token_segments(
                            getattr(self, "layer_name", None),
                            hidden_states.size(0))
                        token_segment_seq_lens = (
                            _prefill_segment_seq_lens_from_forward_context(
                                hidden_states.size(0)))
                        token_seq_lens = _token_seq_lens_from_forward_context(
                            hidden_states.size(0), hidden_states.device)
                        token_decode_offsets = (
                            _decode_offsets_from_forward_context(
                                hidden_states.size(0), hidden_states.device))
                        if token_segments is None:
                            if token_segment_seq_lens is not None:
                                token_segments = [
                                    (start, end)
                                    for start, end, _ in
                                    token_segment_seq_lens
                                ]
                            else:
                                token_segments = (
                                    _prefill_segments_from_forward_context(
                                        hidden_states.size(0)))
                with set_current_moe_layer(getattr(self, "layer_name", None)):
                    with set_current_protected_token_mask(protected_mask):
                        with set_current_eac_prefill_mask(eac_prefill_mask):
                            with set_current_token_segments(token_segments):
                                with set_current_token_segment_seq_lens(
                                        token_segment_seq_lens):
                                    with set_current_token_seq_lens(
                                            token_seq_lens):
                                        with set_current_token_decode_offsets(
                                                token_decode_offsets):
                                            return original_method(
                                                self, *args, **kwargs)

            wrapped.__name__ = getattr(original_method, "__name__",
                                       method_name)
            wrapped.__doc__ = getattr(original_method, "__doc__", None)
            return wrapped

        setattr(cls, method_name, make_wrapped(original))
    _patched_fused_moe_layer_classes.add(cls)


def _patch_attention_probe() -> None:
    try:
        module = importlib.import_module("vllm.attention.layer")
    except Exception:
        return

    cls = getattr(module, "Attention", None)
    if cls is None or cls in _patched_attention_layer_classes:
        return

    original = getattr(cls, "forward", None)
    if original is None:
        return

    def wrapped(self, query, key, value, *args, **kwargs):
        if isinstance(query, torch.Tensor) and isinstance(key, torch.Tensor):
            positions = getattr(self, "_expert_pruning_positions", None)
            if positions is None:
                positions = torch.arange(query.size(0), device=query.device)
            observe_attention_qk(
                layer_name=getattr(self, "layer_name", None),
                positions=positions,
                q=query,
                k=key,
                num_heads=getattr(self, "num_heads"),
                num_kv_heads=getattr(self, "num_kv_heads"),
                head_dim=getattr(self, "head_size"),
                scaling=getattr(self.impl, "scale", 1.0),
            )
        return original(self, query, key, value, *args, **kwargs)

    wrapped.__name__ = getattr(original, "__name__", "forward")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    setattr(cls, "forward", wrapped)
    _patched_attention_layer_classes.add(cls)


def _position_arg_from_forward_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    position_arg_names: tuple[str, ...],
    position_arg_index: int,
) -> Optional[torch.Tensor]:
    for name in position_arg_names:
        value = kwargs.get(name)
        if isinstance(value, torch.Tensor):
            return value
    if len(args) > position_arg_index and isinstance(
            args[position_arg_index], torch.Tensor):
        return args[position_arg_index]
    return None


def _input_ids_arg_from_forward_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    input_arg_names: tuple[str, ...],
    input_arg_index: int,
) -> Optional[torch.Tensor]:
    for name in input_arg_names:
        value = kwargs.get(name)
        if isinstance(value, torch.Tensor):
            return value
    if len(args) > input_arg_index and isinstance(
            args[input_arg_index], torch.Tensor):
        return args[input_arg_index]
    return None


def _patch_input_token_ids(
    module_name: str,
    class_name: str,
    *,
    input_arg_names: tuple[str, ...] = ("input_ids", ),
    input_arg_index: int = 0,
    position_arg_names: tuple[str, ...] = ("positions", ),
    position_arg_index: int = 1,
) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return

    cls = getattr(module, class_name, None)
    if cls is None or cls in _patched_input_token_model_classes:
        return

    original = getattr(cls, "forward", None)
    if original is None:
        return

    def wrapped(self, *args, **kwargs):
        input_ids = _input_ids_arg_from_forward_call(
            args,
            kwargs,
            input_arg_names=input_arg_names,
            input_arg_index=input_arg_index,
        )
        positions = _position_arg_from_forward_call(
            args,
            kwargs,
            position_arg_names=position_arg_names,
            position_arg_index=position_arg_index,
        )
        with set_current_input_token_ids(input_ids, positions):
            return original(self, *args, **kwargs)

    wrapped.__name__ = getattr(original, "__name__", "forward")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    setattr(cls, "forward", wrapped)
    _patched_input_token_model_classes.add(cls)


def _patch_model_input_token_ids() -> None:
    _patch_input_token_ids(
        "vllm.model_executor.models.qwen3_moe",
        "Qwen3MoeForCausalLM",
        input_arg_names=("input_ids", ),
        input_arg_index=0,
    )


def _patch_observer_model_input_token_ids() -> None:
    """Patch text/VL model forwards so token ids are visible to the observer."""
    candidates = (
        ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeForCausalLM",
         ("input_ids", ), 0),
        ("vllm.model_executor.models.qwen3_vl_moe",
         "Qwen3VLMoeForConditionalGeneration", ("input_ids", ), 0),
        ("vllm.model_executor.models.qwen3_vl", "Qwen3VLForConditionalGeneration",
         ("input_ids", ), 0),
    )
    for module_name, class_name, input_names, input_index in candidates:
        _patch_input_token_ids(
            module_name,
            class_name,
            input_arg_names=input_names,
            input_arg_index=input_index,
        )


def _patch_attention_positions(
    module_name: str,
    class_name: str,
    *,
    attention_attr: str = "attn",
    position_arg_names: tuple[str, ...] = ("positions", ),
    position_arg_index: int = 0,
) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return

    cls = getattr(module, class_name, None)
    if cls is None or getattr(cls, "_expert_pruning_positions_patched",
                              False):
        return

    original = getattr(cls, "forward", None)
    if original is None:
        return

    def wrapped(self, *args, **kwargs):
        positions = _position_arg_from_forward_call(
            args,
            kwargs,
            position_arg_names=position_arg_names,
            position_arg_index=position_arg_index,
        )
        attention = getattr(self, attention_attr, None)
        if not isinstance(positions, torch.Tensor) or attention is None:
            return original(self, *args, **kwargs)

        attention._expert_pruning_positions = positions
        record_attention_positions(
            getattr(attention, "layer_name", None), positions)
        try:
            return original(self, *args, **kwargs)
        finally:
            attention._expert_pruning_positions = None

    wrapped.__name__ = getattr(original, "__name__", "forward")
    wrapped.__doc__ = getattr(original, "__doc__", None)
    setattr(cls, "forward", wrapped)
    setattr(cls, "_expert_pruning_positions_patched", True)


def _patch_model_attention_positions() -> None:
    _patch_attention_positions(
        "vllm.model_executor.models.qwen3_moe",
        "Qwen3MoeAttention",
        position_arg_names=("positions", ),
        position_arg_index=0,
    )
    _patch_attention_positions(
        "vllm.model_executor.models.gpt_oss",
        "OAIAttention",
        position_arg_names=("positions", ),
        position_arg_index=1,
    )
    _patch_attention_positions(
        "vllm.model_executor.models.qwen3_next",
        "Qwen3NextAttention",
        position_arg_names=("positions", ),
        position_arg_index=0,
    )
    _patch_attention_positions(
        "vllm.model_executor.models.bailing_moe",
        "BailingAttention",
        position_arg_names=("position_ids", "positions"),
        position_arg_index=1,
    )


def install_vllm_expert_pruning_router(
    method: str = "none",
    naee_beta: float = 0.3,
    naee_k_min: int = 2,
    dynamic_routing_threshold: float = 0.8,
    dynamic_routing_score_source: str = "raw",
    dynamic_routing_next_rank_penalty: float = 1.0,
    layerwise_dynamic_base_threshold: float = 0.55,
    layerwise_dynamic_layer_alpha: float = 0.15,
    layerwise_dynamic_num_layers: int = 48,
    layerwise_dynamic_k_min: int = 1,
    budget_dynamic_easy_k: int = 4,
    budget_dynamic_base_k: int = 5,
    budget_dynamic_hard_k: int = 6,
    budget_dynamic_easy_threshold: float = 0.90,
    budget_dynamic_hard_threshold: float = 0.75,
    budget_dynamic_score_topn: int = 3,
    layer_budget_dynamic_easy_layer_alpha: float = 0.08,
    layer_budget_dynamic_hard_layer_alpha: float = 0.08,
    layer_budget_dynamic_num_layers: int = 48,
    band_layer_budget_middle_start: int = 19,
    band_layer_budget_middle_end: int = 28,
    band_layer_budget_per_layer_k: str | None = None,
    band_layer_budget_easy_layers: str | None = None,
    band_layer_budget_hard_layers: str | None = None,
    band_layer_budget_token_gate_ratio: float = -1.0,
    band_layer_budget_extra_middle_start: int = -1,
    band_layer_budget_extra_middle_end: int = -1,
    band_layer_budget_middle_phase: str = "all",
    band_layer_budget_easy_start: int = 0,
    band_layer_budget_easy_end: int = 47,
    band_layer_budget_easy_phase: str = "all",
    band_layer_budget_hard_phase: str = "all",
    band_layer_budget_easy_tail_threshold: float = 1.0,
    band_layer_budget_hard_min_rank_weight: float = 0.0,
    band_layer_budget_hard_max_rank_weight: float = 1.0,
    band_layer_budget_hard_decode_min_rank_weight: float = -1.0,
    band_layer_budget_mixed_rescue_min_rank_weight: float = -1.0,
    band_layer_budget_mixed_rescue_max_concentration: float = -1.0,
    band_layer_budget_hard_prefill_max_tokens: int = -1,
    band_layer_budget_hard_prefill_max_segment_tokens: int = -1,
    band_layer_budget_hard_prefill_max_density: float = -1.0,
    band_layer_budget_hard_prefill_min_density: float = -1.0,
    band_layer_budget_hard_prefill_max_segment_hard_ratio: float = -1.0,
    band_layer_budget_hard_prefill_segment_cap_score: str = "concentration",
    band_layer_budget_hard_prefill_conditional_segment_hard_ratio:
    float = -1.0,
    band_layer_budget_hard_prefill_conditional_min_density: float = -1.0,
    band_layer_budget_hard_prefill_conditional_min_seq_len: int = -1,
    band_layer_budget_hard_prefill_conditional_token_ids: str | None = None,
    band_layer_budget_hard_prefill_conditional_token_ngrams:
    str | None = None,
    band_layer_budget_hard_prefill_marker_segment_hard_ratio:
    float = -1.0,
    band_layer_budget_hard_prefill_marker_token_ids: str | None = None,
    band_layer_budget_hard_prefill_marker_token_ngrams:
    str | None = None,
    band_layer_budget_hard_prefill_exclude_token_ids: str | None = None,
    band_layer_budget_hard_prefill_min_relative_pos: float = -1.0,
    band_layer_budget_hard_prefill_max_relative_pos: float = -1.0,
    band_layer_budget_hard_prefill_start: int = -1,
    band_layer_budget_hard_prefill_end: int = -1,
    band_layer_budget_hard_decode_min_seq_len: int = -1,
    band_layer_budget_hard_decode_max_seq_len: int = -1,
    band_layer_budget_hard_decode_seq_len_scope: str = "all",
    band_layer_budget_hard_decode_min_offset: int = -1,
    band_layer_budget_hard_decode_max_offset: int = -1,
    band_layer_budget_hard_decode_start: int = -1,
    band_layer_budget_hard_decode_end: int = -1,
    band_layer_budget_hard_decode_token_ids: str | None = None,
    band_layer_budget_hard_layer_prior_start: int = -1,
    band_layer_budget_hard_layer_prior_end: int = -1,
    band_layer_budget_hard_layer_prior_extra_start: int = -1,
    band_layer_budget_hard_layer_prior_extra_end: int = -1,
    band_layer_budget_hard_layer_prior_alpha: float = 0.0,
    band_layer_budget_hard_layer_sensitivity_alpha: float = 0.0,
    band_layer_budget_late_concentration_start: int = -1,
    band_layer_budget_late_concentration_threshold: float = -1.0,
    band_layer_budget_late_rescue_start: int = -1,
    band_layer_budget_late_rescue_concentration_threshold: float = -1.0,
    band_layer_budget_late_rescue_mixed_max: float = -1.0,
    band_layer_budget_prompt_profile1_token_ngrams: str | None = None,
    band_layer_budget_prompt_profile1_hard_threshold: float = -1.0,
    band_layer_budget_prompt_profile1_hard_min_rank_weight: float = -1.0,
    band_layer_budget_prompt_profile1_middle_start: int = -1,
    band_layer_budget_prompt_profile1_middle_end: int = -1,
    band_layer_budget_prompt_profile1_prefill_segment_hard_ratio:
    float = -1.0,
    band_layer_budget_prompt_profile2_token_ngrams: str | None = None,
    band_layer_budget_prompt_profile2_hard_threshold: float = -1.0,
    band_layer_budget_prompt_profile2_hard_min_rank_weight: float = -1.0,
    band_layer_budget_prompt_profile2_middle_start: int = -1,
    band_layer_budget_prompt_profile2_middle_end: int = -1,
    band_layer_budget_prompt_profile2_prefill_segment_hard_ratio:
    float = -1.0,
    diep_artifact_path: str | None = None,
    diep_pruning_mode: str = "independent",
    diep_use_gamma1: bool = False,
    diep_gamma_alpha: float = 1.0,
    diep_threshold_cap: float | None = None,
    ban_artifact_path: str | None = None,
    ban_lambda: float = 0.7,
    ban_k_min: int = 3,
    eac_alpha: float = 0.5,
    biased_renorm_keep_topn: int = 3,
    router_value_artifact_path: str | None = None,
    router_value_base_k: int = 3,
    router_value_quota_ratio: float = 0.10,
    router_value_selection_mode: str = "threshold",
    router_value_swap_direction: str = "both",
    debug: bool = False,
    debug_max_prints: int = 8,
    attention_sink_probe: bool = False,
    attention_sink_probe_max_prints: int = 8,
    attention_sink_probe_max_query_tokens: int = 2048,
    attention_sink_probe_max_key_tokens: int = 2048,
    attention_sink_probe_topk: int = 4,
    mc_moe_protection_ratio: float = 0.02,
) -> bool:
    """Patch vLLM's default non-grouped MoE top-k router.

    The patched function keeps vLLM's public ``fused_topk`` signature, so the
    rest of vLLM still uses its optimized expert execution kernels after local
    routing has produced ``topk_weights`` and ``topk_ids``.

    Returns:
        True if this call installed the patch, False if it was already active.
    """
    global _original_fused_topk, _original_package_fused_topk

    configure_expert_router(
        method=method,
        naee_beta=naee_beta,
        naee_k_min=naee_k_min,
        dynamic_routing_threshold=dynamic_routing_threshold,
        dynamic_routing_score_source=dynamic_routing_score_source,
        dynamic_routing_next_rank_penalty=(
            dynamic_routing_next_rank_penalty),
        layerwise_dynamic_base_threshold=layerwise_dynamic_base_threshold,
        layerwise_dynamic_layer_alpha=layerwise_dynamic_layer_alpha,
        layerwise_dynamic_num_layers=layerwise_dynamic_num_layers,
        layerwise_dynamic_k_min=layerwise_dynamic_k_min,
        budget_dynamic_easy_k=budget_dynamic_easy_k,
        budget_dynamic_base_k=budget_dynamic_base_k,
        budget_dynamic_hard_k=budget_dynamic_hard_k,
        budget_dynamic_easy_threshold=budget_dynamic_easy_threshold,
        budget_dynamic_hard_threshold=budget_dynamic_hard_threshold,
        budget_dynamic_score_topn=budget_dynamic_score_topn,
        layer_budget_dynamic_easy_layer_alpha=(
            layer_budget_dynamic_easy_layer_alpha),
        layer_budget_dynamic_hard_layer_alpha=(
            layer_budget_dynamic_hard_layer_alpha),
        layer_budget_dynamic_num_layers=layer_budget_dynamic_num_layers,
        band_layer_budget_middle_start=band_layer_budget_middle_start,
        band_layer_budget_middle_end=band_layer_budget_middle_end,
        band_layer_budget_per_layer_k=band_layer_budget_per_layer_k,
        band_layer_budget_easy_layers=band_layer_budget_easy_layers,
        band_layer_budget_hard_layers=band_layer_budget_hard_layers,
        band_layer_budget_token_gate_ratio=(
            band_layer_budget_token_gate_ratio),
        band_layer_budget_extra_middle_start=(
            band_layer_budget_extra_middle_start),
        band_layer_budget_extra_middle_end=(
            band_layer_budget_extra_middle_end),
        band_layer_budget_middle_phase=band_layer_budget_middle_phase,
        band_layer_budget_easy_start=band_layer_budget_easy_start,
        band_layer_budget_easy_end=band_layer_budget_easy_end,
        band_layer_budget_easy_phase=band_layer_budget_easy_phase,
        band_layer_budget_hard_phase=band_layer_budget_hard_phase,
        band_layer_budget_easy_tail_threshold=(
            band_layer_budget_easy_tail_threshold),
        band_layer_budget_hard_min_rank_weight=(
            band_layer_budget_hard_min_rank_weight),
        band_layer_budget_hard_max_rank_weight=(
            band_layer_budget_hard_max_rank_weight),
        band_layer_budget_hard_decode_min_rank_weight=(
            band_layer_budget_hard_decode_min_rank_weight),
        band_layer_budget_mixed_rescue_min_rank_weight=(
            band_layer_budget_mixed_rescue_min_rank_weight),
        band_layer_budget_mixed_rescue_max_concentration=(
            band_layer_budget_mixed_rescue_max_concentration),
        band_layer_budget_hard_prefill_max_tokens=(
            band_layer_budget_hard_prefill_max_tokens),
        band_layer_budget_hard_prefill_max_segment_tokens=(
            band_layer_budget_hard_prefill_max_segment_tokens),
        band_layer_budget_hard_prefill_max_density=(
            band_layer_budget_hard_prefill_max_density),
        band_layer_budget_hard_prefill_min_density=(
            band_layer_budget_hard_prefill_min_density),
        band_layer_budget_hard_prefill_max_segment_hard_ratio=(
            band_layer_budget_hard_prefill_max_segment_hard_ratio),
        band_layer_budget_hard_prefill_segment_cap_score=(
            band_layer_budget_hard_prefill_segment_cap_score),
        band_layer_budget_hard_prefill_conditional_segment_hard_ratio=(
            band_layer_budget_hard_prefill_conditional_segment_hard_ratio),
        band_layer_budget_hard_prefill_conditional_min_density=(
            band_layer_budget_hard_prefill_conditional_min_density),
        band_layer_budget_hard_prefill_conditional_min_seq_len=(
            band_layer_budget_hard_prefill_conditional_min_seq_len),
        band_layer_budget_hard_prefill_conditional_token_ids=(
            band_layer_budget_hard_prefill_conditional_token_ids),
        band_layer_budget_hard_prefill_conditional_token_ngrams=(
            band_layer_budget_hard_prefill_conditional_token_ngrams),
        band_layer_budget_hard_prefill_marker_segment_hard_ratio=(
            band_layer_budget_hard_prefill_marker_segment_hard_ratio),
        band_layer_budget_hard_prefill_marker_token_ids=(
            band_layer_budget_hard_prefill_marker_token_ids),
        band_layer_budget_hard_prefill_marker_token_ngrams=(
            band_layer_budget_hard_prefill_marker_token_ngrams),
        band_layer_budget_hard_prefill_exclude_token_ids=(
            band_layer_budget_hard_prefill_exclude_token_ids),
        band_layer_budget_hard_prefill_min_relative_pos=(
            band_layer_budget_hard_prefill_min_relative_pos),
        band_layer_budget_hard_prefill_max_relative_pos=(
            band_layer_budget_hard_prefill_max_relative_pos),
        band_layer_budget_hard_prefill_start=(
            band_layer_budget_hard_prefill_start),
        band_layer_budget_hard_prefill_end=(
            band_layer_budget_hard_prefill_end),
        band_layer_budget_hard_decode_min_seq_len=(
            band_layer_budget_hard_decode_min_seq_len),
        band_layer_budget_hard_decode_max_seq_len=(
            band_layer_budget_hard_decode_max_seq_len),
        band_layer_budget_hard_decode_seq_len_scope=(
            band_layer_budget_hard_decode_seq_len_scope),
        band_layer_budget_hard_decode_min_offset=(
            band_layer_budget_hard_decode_min_offset),
        band_layer_budget_hard_decode_max_offset=(
            band_layer_budget_hard_decode_max_offset),
        band_layer_budget_hard_decode_start=(
            band_layer_budget_hard_decode_start),
        band_layer_budget_hard_decode_end=(
            band_layer_budget_hard_decode_end),
        band_layer_budget_hard_decode_token_ids=(
            band_layer_budget_hard_decode_token_ids),
        band_layer_budget_hard_layer_prior_start=(
            band_layer_budget_hard_layer_prior_start),
        band_layer_budget_hard_layer_prior_end=(
            band_layer_budget_hard_layer_prior_end),
        band_layer_budget_hard_layer_prior_extra_start=(
            band_layer_budget_hard_layer_prior_extra_start),
        band_layer_budget_hard_layer_prior_extra_end=(
            band_layer_budget_hard_layer_prior_extra_end),
        band_layer_budget_hard_layer_prior_alpha=(
            band_layer_budget_hard_layer_prior_alpha),
        band_layer_budget_hard_layer_sensitivity_alpha=(
            band_layer_budget_hard_layer_sensitivity_alpha),
        band_layer_budget_late_concentration_start=(
            band_layer_budget_late_concentration_start),
        band_layer_budget_late_concentration_threshold=(
            band_layer_budget_late_concentration_threshold),
        band_layer_budget_late_rescue_start=(
            band_layer_budget_late_rescue_start),
        band_layer_budget_late_rescue_concentration_threshold=(
            band_layer_budget_late_rescue_concentration_threshold),
        band_layer_budget_late_rescue_mixed_max=(
            band_layer_budget_late_rescue_mixed_max),
        band_layer_budget_prompt_profile1_token_ngrams=(
            band_layer_budget_prompt_profile1_token_ngrams),
        band_layer_budget_prompt_profile1_hard_threshold=(
            band_layer_budget_prompt_profile1_hard_threshold),
        band_layer_budget_prompt_profile1_hard_min_rank_weight=(
            band_layer_budget_prompt_profile1_hard_min_rank_weight),
        band_layer_budget_prompt_profile1_middle_start=(
            band_layer_budget_prompt_profile1_middle_start),
        band_layer_budget_prompt_profile1_middle_end=(
            band_layer_budget_prompt_profile1_middle_end),
        band_layer_budget_prompt_profile1_prefill_segment_hard_ratio=(
            band_layer_budget_prompt_profile1_prefill_segment_hard_ratio),
        band_layer_budget_prompt_profile2_token_ngrams=(
            band_layer_budget_prompt_profile2_token_ngrams),
        band_layer_budget_prompt_profile2_hard_threshold=(
            band_layer_budget_prompt_profile2_hard_threshold),
        band_layer_budget_prompt_profile2_hard_min_rank_weight=(
            band_layer_budget_prompt_profile2_hard_min_rank_weight),
        band_layer_budget_prompt_profile2_middle_start=(
            band_layer_budget_prompt_profile2_middle_start),
        band_layer_budget_prompt_profile2_middle_end=(
            band_layer_budget_prompt_profile2_middle_end),
        band_layer_budget_prompt_profile2_prefill_segment_hard_ratio=(
            band_layer_budget_prompt_profile2_prefill_segment_hard_ratio),
        diep_artifact_path=diep_artifact_path,
        diep_pruning_mode=diep_pruning_mode,
        diep_use_gamma1=diep_use_gamma1,
        diep_gamma_alpha=diep_gamma_alpha,
        diep_threshold_cap=diep_threshold_cap,
        ban_artifact_path=ban_artifact_path,
        ban_lambda=ban_lambda,
        ban_k_min=ban_k_min,
        eac_alpha=eac_alpha,
        mc_moe_protection_ratio=mc_moe_protection_ratio,
        biased_renorm_keep_topn=biased_renorm_keep_topn,
        router_value_artifact_path=router_value_artifact_path,
        router_value_base_k=router_value_base_k,
        router_value_quota_ratio=router_value_quota_ratio,
        router_value_selection_mode=router_value_selection_mode,
        router_value_swap_direction=router_value_swap_direction,
        debug=debug,
        debug_max_prints=debug_max_prints,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        prepare_diep_tensors_for_device(device)
        prepare_ban_tensors_for_device(device)
        prepare_router_value_tensors_for_device(device)
    configure_attention_sink_probe(
        enabled=attention_sink_probe or method.lower() == "mc_moe",
        debug_max_prints=attention_sink_probe_max_prints,
        max_query_tokens=attention_sink_probe_max_query_tokens,
        max_key_tokens=attention_sink_probe_max_key_tokens,
        protect_topk=attention_sink_probe_topk,
        protection_ratio=mc_moe_protection_ratio,
    )
    configure_attention_position_context(
        enabled=method.lower() == "mc_moe" or attention_sink_probe)

    fused_moe_package = importlib.import_module(
        "vllm.model_executor.layers.fused_moe")
    binding_modules = [
        fused_moe_package,
        importlib.import_module(
            "vllm.model_executor.layers.fused_moe.fused_moe"),
    ]
    try:
        binding_modules.append(
            importlib.import_module(
                "vllm.model_executor.layers.fused_moe.router."
                "fused_topk_router"))
    except ImportError:
        pass
    bindings = [
        (module, getattr(module, "fused_topk"))
        for module in binding_modules
        if callable(getattr(module, "fused_topk", None))
    ]
    if not bindings:
        raise RuntimeError(
            "Could not locate a vLLM fused_topk binding; expert pruning "
            "cannot intercept this runtime")

    if all(fn is torch_fused_topk for _, fn in bindings):
        _install_dummy_run_stats_guards()
        _patch_fused_moe_layer_context()
        _patch_model_input_token_ids()
        _patch_model_attention_positions()
        _patch_attention_probe()
        return False

    if _original_fused_topk is None:
        _original_fused_topk = next(
            fn for _, fn in bindings if fn is not torch_fused_topk)
    if _original_package_fused_topk is None:
        _original_package_fused_topk = getattr(fused_moe_package, "fused_topk",
                                               None)

    for module, _ in bindings:
        module.fused_topk = torch_fused_topk
    _install_dummy_run_stats_guards()
    _patch_fused_moe_layer_context()
    _patch_model_input_token_ids()
    _patch_model_attention_positions()
    _patch_attention_probe()
    return True


def get_original_fused_topk() -> Optional[Callable]:
    """Return the original vLLM router if the patch has been installed."""
    return _original_fused_topk


def _discover_fused_topk_bindings() -> list[tuple[Any, str, Callable]]:
    """Locate every import alias of vLLM's native ``fused_topk``."""
    bindings: list[tuple[Any, str, Callable]] = []
    module_names = (
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe.router.fused_topk_router",
    )
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        fn = getattr(module, "fused_topk", None)
        if callable(fn):
            bindings.append((module, "fused_topk", fn))
    try:
        package = importlib.import_module(
            "vllm.model_executor.layers.fused_moe")
        fn = getattr(package, "fused_topk", None)
        if callable(fn):
            bindings.append((package, "fused_topk", fn))
    except Exception:
        pass
    if not bindings:
        raise RuntimeError(
            "Could not locate vLLM fused_topk; observer cannot be installed")
    unique: list[tuple[Any, str, Callable]] = []
    seen_ids: set[int] = set()
    for module, attr, fn in bindings:
        original = getattr(fn, "_router_distribution_original", fn)
        fn_id = id(original)
        if fn_id in seen_ids:
            continue
        seen_ids.add(fn_id)
        unique.append((module, attr, fn))
    return unique


def install_router_distribution_observer(
    *,
    output_dir: str,
    sample_rate: float = 0.005,
    sample_cap: int = 2048,
    histogram_bins: int = 16,
    flush_interval: int = 512,
    dataset: str = "",
    model_path: str = "",
    harness: str = "",
    task: str = "",
    seed: int = 0,
) -> bool:
    """Wrap native ``fused_topk`` to observe logits without changing outputs."""
    configure_router_distribution(
        enabled=True,
        output_dir=output_dir,
        sample_rate=sample_rate,
        sample_cap=sample_cap,
        histogram_bins=histogram_bins,
        flush_interval=flush_interval,
        dataset=dataset,
        model_path=model_path,
        harness=harness,
        task=task,
        seed=seed,
    )
    bindings = _discover_fused_topk_bindings()
    if all(getattr(fn, "_router_distribution_wrapper", False)
           for _, _, fn in bindings):
        _install_dummy_run_stats_guards()
        _patch_fused_moe_layer_context()
        _patch_observer_model_input_token_ids()
        return False
    patched_any = False
    for module, attr, fn in _discover_fused_topk_bindings():
        if getattr(fn, "_router_distribution_wrapper", False):
            patched_any = True
            continue
        original = getattr(fn, "_router_distribution_original", fn)
        wrapped = wrap_fused_topk_observer(original)
        setattr(module, attr, wrapped)
        patched_any = True
    if not patched_any:
        raise RuntimeError(
            "router distribution observer did not patch any fused_topk alias")
    _install_dummy_run_stats_guards()
    _patch_fused_moe_layer_context()
    _patch_observer_model_input_token_ids()
    return True


def _bootstrap_router_distribution_from_env() -> None:
    """Install observer in any process that imports this module with env set."""
    import os

    if os.environ.get("ROUTER_DISTRIBUTION_ENABLED") != "1":
        return
    try:
        install_router_distribution_observer(
            output_dir=os.environ["ROUTER_DISTRIBUTION_DIR"],
            sample_rate=float(
                os.environ.get("ROUTER_DISTRIBUTION_SAMPLE_RATE", "0.005")),
            sample_cap=int(
                os.environ.get("ROUTER_DISTRIBUTION_SAMPLE_CAP", "2048")),
            histogram_bins=int(
                os.environ.get("ROUTER_DISTRIBUTION_HISTOGRAM_BINS", "16")),
            flush_interval=int(
                os.environ.get("ROUTER_DISTRIBUTION_FLUSH_INTERVAL", "512")),
            dataset=os.environ.get("ROUTER_DISTRIBUTION_DATASET", ""),
            model_path=os.environ.get("ROUTER_DISTRIBUTION_MODEL_PATH", ""),
            harness=os.environ.get("ROUTER_DISTRIBUTION_HARNESS", ""),
            task=os.environ.get("ROUTER_DISTRIBUTION_TASK", ""),
            seed=int(os.environ.get("ROUTER_DISTRIBUTION_SEED", "0")),
        )
    except RuntimeError:
        return


_bootstrap_router_distribution_from_env()
