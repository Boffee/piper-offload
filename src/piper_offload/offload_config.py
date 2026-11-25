"""Configuration and automatic partitioning discovery for model offloading."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, cast

import torch
from torch import nn

from ._dtensor import local_shard
from .block_compile import BlockCompileConfig
from .tensor_adapter_registry import buffer_tensor_id, param_representation, param_tensor_id


class _UseConfigBudget(Enum):
    VALUE = 0


_USE_CONFIG_BUDGET: Final = _UseConfigBudget.VALUE
_NON_CALLABLE_CONTAINERS = (
    nn.ModuleList,
    nn.ModuleDict,
    nn.ParameterList,
    nn.ParameterDict,
)


def validate_vram_budget(vram_budget_bytes: object) -> int | None:
    """Validate a VRAM budget value (non-negative ``int`` or ``None``)."""
    if vram_budget_bytes is None:
        return None
    if isinstance(vram_budget_bytes, bool) or not isinstance(vram_budget_bytes, int):
        raise TypeError(
            f"vram_budget_bytes must be an int or None, got {type(vram_budget_bytes).__name__}."
        )
    if vram_budget_bytes < 0:
        raise ValueError(
            f"vram_budget_bytes must be >= 0 or None, got {vram_budget_bytes}."
        )
    return vram_budget_bytes


def _validate_path_sequence(paths: object, field_name: str) -> tuple[str, ...]:
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise TypeError(
            f"{field_name} must be a sequence of attribute paths, not {type(paths).__name__}."
        )
    seen: set[str] = set()
    validated: list[str] = []
    for path in paths:
        if not isinstance(path, str):
            raise TypeError(
                f"{field_name} entries must be str, got {type(path).__name__}."
            )
        if not path or any(not seg or seg.strip() != seg for seg in path.split(".")):
            raise ValueError(
                f"{field_name} entries must be non-empty dotted attribute paths "
                f"without blank or whitespace-padded segments; got {path!r}."
            )
        if path in seen:
            raise ValueError(f"Duplicate path {path!r} in {field_name}.")
        seen.add(path)
        validated.append(path)
    return tuple(validated)


@dataclass(frozen=True, kw_only=True, slots=True, init=False)
class OffloadConfig:
    """Keyword-only configuration for model offloading and residency.

    ``offload_paths`` accepts a sequence of dotted attribute paths naming
    callable submodules or ``nn.ModuleList`` containers managed by the
    residency engine (normalized to ``tuple[str, ...]``). Parameters and
    buffers outside those paths stay resident on the active device for the
    duration of an activation session.

    ``vram_budget_bytes`` bounds decoded CUDA target bytes held simultaneously
    by offloaded units (excluding resident weights, activations, KV cache, and
    transient scratch; ignored when ``offload_paths`` is empty):
    ``None`` keeps at most one active unit and one prefetched lookahead unit on
    GPU; ``0`` releases each active unit immediately on unit exit without
    async lookahead; a positive integer caches units up to that byte budget
    using Belady MIN eviction (preloading all units without pinning when total
    unit bytes fit within budget).

    ``block_compile`` optionally enables per-unit ``torch.compile`` on units in
    ``offload_paths`` when activating on CUDA or CPU.
    """

    offload_paths: tuple[str, ...]
    vram_budget_bytes: int | None
    block_compile: BlockCompileConfig | None

    def __init__(
        self,
        *,
        offload_paths: Sequence[str] = (),
        vram_budget_bytes: int | None = None,
        block_compile: BlockCompileConfig | None = None,
    ) -> None:
        normalized_paths = _validate_path_sequence(offload_paths, "offload_paths")
        validated_budget = validate_vram_budget(vram_budget_bytes)
        if block_compile is not None and not isinstance(
            block_compile, BlockCompileConfig
        ):
            raise TypeError(
                "block_compile must be a BlockCompileConfig or None, "
                f"got {type(block_compile).__name__}."
            )
        object.__setattr__(self, "offload_paths", normalized_paths)
        object.__setattr__(self, "vram_budget_bytes", validated_budget)
        object.__setattr__(self, "block_compile", block_compile)


def validate_offload_config(config: object) -> OffloadConfig:
    """Validate that ``config`` is an :class:`OffloadConfig` instance."""
    if not isinstance(config, OffloadConfig):
        raise TypeError(
            f"Expected OffloadConfig, got {type(config).__name__}."
        )
    return config


def resolve_offload_config(config: OffloadConfig | None) -> OffloadConfig:
    """Normalize ``config``, returning default :class:`OffloadConfig()` for ``None``."""
    if config is None:
        return OffloadConfig()
    return validate_offload_config(config)


def _packed_backing_parts(rep: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
    """Extract underlying CPU tensors from packed/projected tensor wrappers."""
    storage_tensors = getattr(rep, "storage_tensors", None)
    if callable(storage_tensors):
        return tuple(
            part
            for part in cast(Iterable[torch.Tensor], storage_tensors())
            if isinstance(part, torch.Tensor)
        )
    if (
        isinstance(rep, nn.Parameter)
        and hasattr(rep, "quant_type")
        and callable(getattr(rep, "as_tensor", None))
    ):
        raw = cast(Any, rep).as_tensor()
        if isinstance(raw, torch.Tensor):
            return (raw.as_subclass(torch.Tensor) if type(raw) is not torch.Tensor else raw,)
    flatten = getattr(rep, "__tensor_flatten__", None)
    if callable(flatten):
        attr_names = cast(tuple[Sequence[str], object], flatten())[0]
        return tuple(
            inner
            for name in attr_names
            if isinstance(inner := getattr(rep, name, None), torch.Tensor)
        )
    return None


def _estimate_tensor_bytes(tensor: torch.Tensor) -> int:
    seen: set[tuple[object, ...]] = set()
    kept_refs: list[torch.Tensor] = []
    total = 0

    def _record_leaf(rep: torch.Tensor) -> None:
        nonlocal total
        key: tuple[object, ...] | None = None
        if not rep.is_meta and rep.numel() > 0 and rep.layout is torch.strided:
            try:
                data_ptr = rep.untyped_storage().data_ptr()
            except RuntimeError:
                data_ptr = None
            if data_ptr is not None:
                key = (
                    "storage",
                    rep.device,
                    data_ptr,
                    rep.storage_offset(),
                    tuple(rep.shape),
                    rep.stride(),
                    rep.dtype,
                )
        if key is None:
            kept_refs.append(rep)
            key = ("id", id(rep))
        if key not in seen:
            seen.add(key)
            total += rep.numel() * rep.element_size()

    def _accumulate(t: torch.Tensor) -> None:
        rep = t if type(t) is torch.Tensor else param_representation(t)
        if type(rep) is torch.Tensor or type(rep) is nn.Parameter:
            _record_leaf(rep.data if type(rep) is nn.Parameter else rep)
            return
        local = local_shard(rep)
        if local is not rep:
            _accumulate(local)
            return
        parts = _packed_backing_parts(rep)
        if parts is not None:
            for part in parts:
                _accumulate(part)
            return
        data = rep.data
        if data is not rep and type(data) is not type(rep):
            _accumulate(data)
            return
        _record_leaf(rep)

    _accumulate(tensor)
    return total


def _safe_tensor_id(
    tensor: torch.Tensor,
    prefix: str,
    id_fn: Callable[[Any], tuple[Any, ...]],
) -> tuple[object, ...]:
    if tensor.layout is torch.strided:
        try:
            return (prefix, id_fn(tensor))
        except RuntimeError:
            pass
    return (f"{prefix}_id", id(tensor))


def _direct_module_tids(
    module: nn.Module,
) -> tuple[
    Counter[tuple[object, ...]],
    dict[tuple[object, ...], int],
    set[tuple[object, ...]],
]:
    tie_counts: Counter[tuple[object, ...]] = Counter()
    byte_sizes: dict[tuple[object, ...], int] = {}
    size_tids: set[tuple[object, ...]] = set()
    for param in module._parameters.values():
        if param is None:
            continue
        tid = _safe_tensor_id(param, "p", param_tensor_id)
        tie_counts[tid] += 1
        size_tids.add(tid)
        byte_sizes.setdefault(tid, _estimate_tensor_bytes(param))
    for leaf, buf in module._buffers.items():
        if buf is None:
            continue
        # Track buffer slot identity for cross-boundary ties so shared
        # read-only buffers (e.g. RoPE tables) across distinct blocks are
        # allowed while aliased module buffer slots still stay together.
        tie_counts[("bslot", id(module), leaf)] += 1
        size_tid = _safe_tensor_id(buf, "b", buffer_tensor_id)
        size_tids.add(size_tid)
        byte_sizes.setdefault(size_tid, _estimate_tensor_bytes(buf))
    return tie_counts, byte_sizes, size_tids


def _subtree_tids(
    mod: nn.Module,
) -> tuple[Counter[tuple[object, ...]], set[tuple[object, ...]]]:
    tie_counts, _, size_tids = _direct_module_tids(mod)
    for child in mod._modules.values():
        if child is not None:
            child_ties, child_sizes = _subtree_tids(child)
            tie_counts.update(child_ties)
            size_tids.update(child_sizes)
    return tie_counts, size_tids


def _qualifies_as_module_list(
    mod: nn.ModuleList,
    *,
    global_counts: Counter[tuple[object, ...]],
    byte_sizes: dict[tuple[object, ...], int],
    min_unit_bytes: int,
) -> bool:
    if len(mod) == 0:
        return False
    direct_ties, _, _ = _direct_module_tids(mod)
    list_ties, list_sizes = _subtree_tids(mod)
    if direct_ties or not list_ties:
        return False
    if any(count != global_counts[tid] for tid, count in list_ties.items()):
        return False
    seen_element_ids: set[int] = set()
    seen_in_elements: set[tuple[object, ...]] = set()
    for child in mod._modules.values():
        if child is None:
            continue
        if isinstance(child, _NON_CALLABLE_CONTAINERS):
            return False
        child_id = id(child)
        if child_id in seen_element_ids:
            continue
        seen_element_ids.add(child_id)
        child_ties, _ = _subtree_tids(child)
        if seen_in_elements.intersection(child_ties):
            return False
        seen_in_elements.update(child_ties)
    total = sum(byte_sizes[tid] for tid in list_sizes)
    return bool(seen_in_elements) and total > 0 and total >= min_unit_bytes


def auto_offload_paths(
    model: nn.Module,
    *,
    min_unit_bytes: int = 16 * 1024 * 1024,
) -> tuple[str, ...]:
    """Select default ``offload_paths`` for ``model``.

    Finds outermost ``nn.ModuleList`` containers and large callable standalone
    submodules whose unique parameter/buffer storage is at least
    ``min_unit_bytes``, skipping non-callable containers and candidates that
    share tied parameters or stateful buffer slots across their boundary or
    across distinct ``nn.ModuleList`` elements.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}.")
    if isinstance(min_unit_bytes, bool) or not isinstance(min_unit_bytes, int):
        raise TypeError(
            f"min_unit_bytes must be an int, got {type(min_unit_bytes).__name__}."
        )
    if min_unit_bytes < 0:
        raise ValueError(f"min_unit_bytes must be >= 0, got {min_unit_bytes}.")

    global_counts: Counter[tuple[object, ...]] = Counter()
    byte_sizes: dict[tuple[object, ...], int] = {}

    def _collect_global(mod: nn.Module) -> None:
        direct_ties, direct_bytes, _ = _direct_module_tids(mod)
        global_counts.update(direct_ties)
        for tid, nbytes in direct_bytes.items():
            byte_sizes.setdefault(tid, nbytes)
        for child in mod._modules.values():
            if child is not None:
                _collect_global(child)

    _collect_global(model)

    def _visit(mod: nn.Module, path: str) -> list[str]:
        if path and isinstance(mod, nn.ModuleList) and _qualifies_as_module_list(
            mod,
            global_counts=global_counts,
            byte_sizes=byte_sizes,
            min_unit_bytes=min_unit_bytes,
        ):
            return [path]

        child_paths: list[str] = []
        for child_name, child in mod.named_children():
            child_path = f"{path}.{child_name}" if path else child_name
            child_paths.extend(_visit(child, child_path))
        if child_paths or not path or isinstance(mod, _NON_CALLABLE_CONTAINERS):
            return child_paths

        tie_counts, size_tids = _subtree_tids(mod)
        if not tie_counts or any(
            count != global_counts[tid] for tid, count in tie_counts.items()
        ):
            return []
        total = sum(byte_sizes[tid] for tid in size_tids)
        if total > 0 and total >= min_unit_bytes:
            return [path]
        return []

    return tuple(_visit(model, ""))


__all__ = [
    "OffloadConfig",
    "auto_offload_paths",
    "resolve_offload_config",
    "validate_offload_config",
]
