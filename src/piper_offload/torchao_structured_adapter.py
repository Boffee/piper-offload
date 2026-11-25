"""Shared base for TorchAO structured-tensor adapters.

TorchAO represents each quantized weight as a tensor subclass wrapping a
handful of inner storage tensors (packed elements + scales) plus opaque
dispatch metadata. Moving such a weight across the CPU<->GPU boundary is
the same mechanical dance for every format: clone each inner tensor into
host memory, allocate stride-matching GPU storage, bulk-copy, and
rebuild the wrapper from storage + a metadata snapshot. Only a few things
vary per format: which inner tensors exist (and whether some are
optional), how the wrapper is reconstructed, the identity/layout metadata
that distinguishes incompatible variants, and the logical compute dtype.

:class:`TorchaoStructuredAdapter` captures the shared mechanics once and
exposes the per-format parts as small hooks. Concrete adapters
(:class:`~piper_offload.float8_adapter.Float8Adapter`,
:class:`~piper_offload.mx_adapter.MxAdapter`,
:class:`~piper_offload.nvfp4_adapter.Nvfp4Adapter`, ...) subclass it and
implement those hooks.

Capabilities beyond inference movement and metadata queries (dequantize/requantize
conversion, representation-preserving copy, and staged factorized or dense
merge) are added by the concrete subclass that supports them — and only that
subclass — so the ``@runtime_checkable`` capability protocols, which test for
method presence on the class, report each format's true capability. The base
deliberately implements no mutation capabilities: subclasses that add nothing
stay inference-only movement, while a subclass that defines the extra methods
advertises exactly those capabilities.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, NoReturn, cast

import torch
from torch import nn

from .tensor_adapters import (
    DenseMergeTargetValidationTensorAdapter,
    DenseMergeTensorAdapter,
    DenseMergeValidationTensorAdapter,
    DequantizeTensorAdapter,
    LoRAMergeTensorAdapter,
    LoRAMergeValidationTensorAdapter,
    adapter_name,
    capture_host_tensor,
    empty_like_strided,
    is_plain_meta_placeholder,
    matches_layout_signature,
    optional_tensor_id,
    tensor_layout,
    transfer_,
)

__all__ = [
    "TorchaoGpu",
    "TorchaoHost",
    "TorchaoStructuredAdapter",
    "_ProjectedTorchaoShard",
    "_ProjectedTorchaoShardAdapter",
    "copy_storage",
]


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    running = 1
    for idx in range(len(shape) - 1, -1, -1):
        strides[idx] = running
        running *= shape[idx]
    return tuple(strides)


@dataclass(slots=True, frozen=True)
class TorchaoHost[MetaT]:
    """CPU state for a TorchAO structured tensor.

    ``storage`` holds the inner storage tensors positionally, parallel to
    the adapter's ``_STORAGE_NAMES``; an entry is ``None`` for an absent
    optional tensor (e.g. NVFP4 global scales). ``meta`` is the immutable
    metadata snapshot used to rebuild the wrapper on each move, and
    ``logical_shape`` is the dense wrapper shape preserved across offload.
    When ``target_shapes`` is not ``None`` (e.g., non-canonical projected host
    views such as ``Shard(1)`` or swizzled tile-grid slices), CPU stays behind
    :class:`_ProjectedTorchaoShard` and device activation allocates contiguous
    buffers matching each slot's target shape.
    """

    storage: tuple[torch.Tensor | None, ...]
    meta: MetaT
    logical_shape: tuple[int, ...]
    target_shapes: tuple[tuple[int, ...] | None, ...] | None = None


def _resolve_target_shapes(
    storage: tuple[torch.Tensor | None, ...],
    target_shapes: tuple[tuple[int, ...] | None, ...] | None,
) -> tuple[tuple[int, ...] | None, ...]:
    """Validate and resolve per-slot target buffer shapes."""
    if target_shapes is None:
        return tuple(tuple(s.shape) if s is not None else None for s in storage)
    if len(target_shapes) != len(storage):
        raise ValueError(
            f"Projected target_shapes length {len(target_shapes)} does not match "
            f"storage slot count {len(storage)}."
        )
    resolved: list[tuple[int, ...] | None] = []
    for s, shape in zip(storage, target_shapes, strict=True):
        if s is None:
            if shape is not None:
                raise ValueError(
                    f"Absent projected storage slot cannot have target shape {shape}."
                )
            resolved.append(None)
            continue
        target = tuple(s.shape) if shape is None else tuple(shape)
        if any(dim < 0 for dim in target):
            raise ValueError(
                f"Projected storage slot target shape {target} cannot contain "
                "negative dimensions."
            )
        if math.prod(target) != s.numel():
            raise ValueError(
                f"Projected storage slot target shape {target} ({math.prod(target)} "
                f"elements) does not match view element count {s.numel()}."
            )
        resolved.append(target)
    return tuple(resolved)


class _ProjectedTorchaoShard(torch.Tensor):
    """Resting CPU placeholder for a non-canonical projected quantized shard.

    Keeps zero-copy host views inside :class:`TorchaoHost` while presenting a
    non-meta CPU local tensor of the exact shard shape and compute dtype to
    :class:`~piper_offload.dtensor_adapter.DTensorAdapter` and update preflight.
    """

    _adapter_cls: type[TorchaoStructuredAdapter[Any]]
    _state: TorchaoHost[Any]
    _meta_tensor: torch.Tensor
    _layout_signature: tuple[object, ...]

    @staticmethod
    def __new__(
        cls: type[_ProjectedTorchaoShard],
        adapter_cls: type[TorchaoStructuredAdapter[Any]],
        state: TorchaoHost[Any],
        meta_tensor: torch.Tensor,
        layout_signature: tuple[object, ...],
    ) -> _ProjectedTorchaoShard:
        tensor = cast(
            _ProjectedTorchaoShard,
            torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                cls,
                tuple(meta_tensor.shape),
                dtype=meta_tensor.dtype,
                device=torch.device("cpu"),
                requires_grad=False,
            ),
        )
        tensor._adapter_cls = adapter_cls
        tensor._state = state
        tensor._meta_tensor = meta_tensor
        tensor._layout_signature = layout_signature
        return tensor

    def __repr__(self, *, tensor_contents: object = None) -> str:
        del tensor_contents
        return (
            f"_ProjectedTorchaoShard(adapter={self._adapter_cls.__name__}, "
            f"shape={tuple(self.shape)}, dtype={self.dtype})"
        )

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls,
        func: object,
        _types: tuple[type, ...],
        args: tuple[object, ...] = (),
        _kwargs: dict[str, object] | None = None,
    ) -> _ProjectedTorchaoShard:
        if args and isinstance(args[0], cls):
            target = args[0]
            identity_view = (
                (
                    func in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default)
                    and tuple(cast(tuple[int, ...] | list[int], args[1])) == tuple(target.shape)
                )
                or (
                    func == torch.ops.aten.view_as.default
                    and tuple(cast(torch.Tensor, args[1]).shape) == tuple(target.shape)
                )
                or func in (torch.ops.aten.alias.default, torch.ops.aten.detach.default)
            )
            if identity_view:
                return cls(
                    target._adapter_cls,
                    target._state,
                    target._meta_tensor,
                    target._layout_signature,
                )
        raise RuntimeError(
            "Projected quantized shard requires CUDA activation before "
            f"executing {func}."
        )


def _require_projected_shard(t: torch.Tensor) -> _ProjectedTorchaoShard:
    if not isinstance(t, _ProjectedTorchaoShard):
        raise TypeError(
            f"expected _ProjectedTorchaoShard, got {type(t).__name__}"
        )
    return t


class _ProjectedTorchaoShardAdapter:
    """Internal adapter matched when inspecting a resting projected shard."""

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        return isinstance(t, _ProjectedTorchaoShard)

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple[object, ...]:
        shard = _require_projected_shard(t)
        return (
            "projected-torchao",
            shard._adapter_cls._TAG,
            *(optional_tensor_id(s) for s in shard._state.storage),
            shard._layout_signature,
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
        return _require_projected_shard(t)._layout_signature

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        return tuple(_require_projected_shard(t).shape)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        return _require_projected_shard(t).dtype

    @staticmethod
    def capture_host(t: torch.Tensor) -> NoReturn:
        _require_projected_shard(t)
        raise ValueError(
            "Resting projected quantized shards cannot be captured directly "
            "into host state; bind from the originating store or activate on CUDA."
        )

    @staticmethod
    def storage_tensors(state: object) -> NoReturn:
        del state
        raise RuntimeError(
            "Resting projected quantized shards do not expose direct host capture state."
        )

    @staticmethod
    def cpu_param(state: object) -> NoReturn:
        del state
        raise RuntimeError(
            "Resting projected quantized shards do not expose direct host capture state."
        )

    @staticmethod
    def alloc_gpu(state: object, device: torch.device) -> NoReturn:
        del state, device
        raise RuntimeError(
            "Resting projected quantized shards must be activated via their owning store."
        )

    @staticmethod
    def gpu_param(host: object, gpu_state: object) -> NoReturn:
        del host, gpu_state
        raise RuntimeError(
            "Resting projected quantized shards must be activated via their owning store."
        )

    @staticmethod
    def copy_to_gpu(
        src: object,
        dst: object,
        *,
        non_blocking: bool = False,
    ) -> NoReturn:
        del src, dst, non_blocking
        raise RuntimeError(
            "Resting projected quantized shards must be activated via their owning store."
        )

    @staticmethod
    def cache_bytes(state: object) -> int:
        del state
        raise RuntimeError(
            "Resting projected quantized shards do not expose direct host capture state."
        )

    @staticmethod
    def validate_permanent_update() -> NoReturn:
        raise ValueError(
            "Permanent updates to projected quantized shards are unsupported; "
            "use activation-time adapter merge instead."
        )

    @staticmethod
    def dequantize(t: torch.Tensor) -> NoReturn:
        shard = _require_projected_shard(t)
        adapter = shard._adapter_cls()
        if not isinstance(adapter, DequantizeTensorAdapter):
            raise ValueError(
                f"{adapter_name(adapter)} does not support dequantize."
            )
        raise RuntimeError(
            "Projected quantized shard must be activated on CUDA before "
            "dequantizing."
        )

    @staticmethod
    def validate_lora_merge(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        shard = _require_projected_shard(target)
        adapter = shard._adapter_cls()
        if not isinstance(adapter, LoRAMergeTensorAdapter):
            raise ValueError(
                f"{adapter_name(adapter)} does not support LoRA merge. "
                "Use routed LoRA for this tensor type."
            )
        if isinstance(adapter, LoRAMergeValidationTensorAdapter):
            adapter.validate_lora_merge(
                shard._meta_tensor,
                b,
                a,
                alpha,
                beta=beta,
                rounding_seed=rounding_seed,
            )

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> NoReturn:
        del b, a, alpha, beta, rounding_seed
        _require_projected_shard(target)
        raise RuntimeError(
            "Projected quantized shard must be activated on CUDA before merging LoRA."
        )

    @staticmethod
    def validate_dense_merge_target(
        target: torch.Tensor,
        *,
        rounding_seed: int | None = None,
    ) -> bool:
        shard = _require_projected_shard(target)
        adapter = shard._adapter_cls()
        if not isinstance(adapter, DenseMergeTensorAdapter):
            raise ValueError(
                f"{adapter_name(adapter)} does not support dense parameter merge."
            )
        if isinstance(adapter, DenseMergeTargetValidationTensorAdapter):
            return adapter.validate_dense_merge_target(
                shard._meta_tensor,
                rounding_seed=rounding_seed,
            )
        return isinstance(adapter, DenseMergeValidationTensorAdapter)

    @staticmethod
    def validate_dense_merge(
        target: torch.Tensor,
        update: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        shard = _require_projected_shard(target)
        adapter = shard._adapter_cls()
        if not isinstance(adapter, DenseMergeTensorAdapter):
            raise ValueError(
                f"{adapter_name(adapter)} does not support dense parameter merge."
            )
        if isinstance(adapter, DenseMergeValidationTensorAdapter):
            adapter.validate_dense_merge(
                shard._meta_tensor,
                update,
                alpha,
                beta=beta,
                rounding_seed=rounding_seed,
            )

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> NoReturn:
        del update, alpha, beta, rounding_seed
        _require_projected_shard(target)
        raise RuntimeError(
            "Projected quantized shard must be activated on CUDA before merging dense updates."
        )


@dataclass(slots=True, frozen=True)
class TorchaoGpu:
    """GPU state for a TorchAO structured tensor: device storage only;
    metadata lives in the originating :class:`TorchaoHost`."""

    storage: tuple[torch.Tensor | None, ...]


def copy_storage(
    src: tuple[torch.Tensor | None, ...],
    dst: tuple[torch.Tensor | None, ...],
    *,
    non_blocking: bool,
) -> None:
    """Per-tensor bulk copy of a parallel storage tuple, skipping absent
    (``None``) entries for H2D movement.

    Movement keeps the same representation on both sides, so a ``None`` on
    one side is ``None`` on the other; skipping on ``src`` therefore also
    leaves the matching ``dst`` slot untouched."""
    for s, d in zip(src, dst, strict=True):
        if s is None:
            continue
        assert d is not None
        transfer_(d, s, non_blocking=non_blocking)


def copy_storage_into(
    src: tuple[torch.Tensor | None, ...],
    dst: tuple[torch.Tensor | None, ...],
    *,
    non_blocking: bool,
) -> None:
    """Fill every present ``dst`` slot from its ``src`` counterpart,
    preserving ``dst``'s slot presence.

    Mirror of :func:`copy_storage` but driven by ``dst``: used by
    ``copy_into`` after a requantize, where the re-encoded ``src`` may
    populate an optional slot that ``dst``'s representation omits — e.g. a
    symmetric int8 weight stored with ``zero_point=None``, while
    ``Int8Tensor.from_hp`` always emits a zeros ``zero_point``. Such a slot
    carries no information ``dst`` lacks, so it is skipped rather than
    forced. A ``dst`` slot that *is* present must have a ``src`` to fill
    it (the requantized tensor mirrors ``dst``'s required slots)."""
    for s, d in zip(src, dst, strict=True):
        if d is None:
            continue
        assert s is not None
        transfer_(d, s, non_blocking=non_blocking)


class TorchaoStructuredAdapter[MetaT](ABC):
    """Base adapter for TorchAO structured (subclass-wrapped) weights.

    Implements the full
    :class:`~piper_offload.tensor_adapters.TensorAdapter` movement and
    identity contract in terms of per-format hooks. A concrete adapter
    sets :attr:`_TAG` / :attr:`_STORAGE_NAMES` and implements the ``_*``
    hooks; it opts into capabilities beyond inference movement by defining
    the relevant methods (``dequantize`` / ``requantize`` / ``copy_into``,
    ``merge_lora_``, or ``merge_dense_``) itself.

    The hooks operate on the validated wrapper, typed :data:`Any` because
    TorchAO tensor subclasses are untyped — the same boundary the
    per-format ``require_*_tensor`` helpers expose.
    """

    _TAG: ClassVar[str]
    _STORAGE_NAMES: ClassVar[tuple[str, ...]]

    # --- per-format hooks -------------------------------------------------

    @staticmethod
    @abstractmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        """True if ``t`` is this format's TorchAO subclass (and a supported
        variant). Mirrors the per-format ``is_*_tensor`` predicate."""

    @staticmethod
    @abstractmethod
    def _validate_layout(t: torch.Tensor) -> None:
        """Raise if ``t`` is missing the attributes this adapter preserves."""

    @staticmethod
    @abstractmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        """Return ``t`` as the validated wrapper, or raise."""

    @staticmethod
    @abstractmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        """Live inner storage tensors, positional and parallel to
        :attr:`_STORAGE_NAMES`; ``None`` for an absent optional tensor."""

    @staticmethod
    @abstractmethod
    def _meta_of(t: Any) -> MetaT:  # noqa: ANN401
        """Snapshot the wrapper's reconstruction metadata."""

    @staticmethod
    @abstractmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...], meta: MetaT
    ) -> torch.Tensor:
        """Rebuild the TorchAO wrapper from storage + metadata snapshot."""

    @staticmethod
    @abstractmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        """Format-specific identity metadata fragments for :meth:`tensor_id`."""

    @classmethod
    def _layout_metadata(cls, t: Any) -> tuple[object, ...]:  # noqa: ANN401
        """Layout metadata fragments for :meth:`layout_signature`.

        Defaults to the identity metadata — correct when block-pool
        compatibility needs exactly the fields that distinguish identity.
        Override only when layout needs fewer/different fields (e.g. Float8
        drops the logical dtype, which the standard dtype slot already
        carries)."""
        return cls._id_metadata(t)

    @staticmethod
    @abstractmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        """Logical matmul/output dtype read off the validated wrapper."""

    # --- shared mechanics -------------------------------------------------

    @classmethod
    def matches(cls, t: torch.Tensor) -> bool:
        if not cls._is_tensor(t):
            return False
        cls._validate_layout(t)
        return True

    @classmethod
    def tensor_id(cls, t: torch.Tensor) -> tuple[object, ...]:
        w = cls._require(t)
        return (
            cls._TAG,
            *(optional_tensor_id(s) for s in cls._storage_of(w)),
            tuple(w.shape),
            w.stride(),
            *cls._id_metadata(w),
        )

    @classmethod
    def layout_signature(cls, t: torch.Tensor) -> tuple[object, ...]:
        """Hashable layout metadata used by block-pool compatibility checks."""
        w = cls._require(t)
        return (
            tuple(w.shape),
            w.dtype,
            w.stride(),
            *cls._layout_metadata(w),
            *(
                (name, tensor_layout(s))
                for name, s in zip(
                    cls._STORAGE_NAMES, cls._storage_of(w), strict=True
                )
            ),
        )

    @classmethod
    def capture_host(
        cls,
        t: torch.Tensor,
    ) -> TorchaoHost[MetaT]:
        w = cls._require(t)
        # preserve_format (capture_host_tensor default): inner-tensor stride
        # ordering can encode a transposed quantized tensor.
        storage = tuple(
            capture_host_tensor(s) if s is not None else None
            for s in cls._storage_of(w)
        )
        return TorchaoHost(
            storage=storage,
            meta=cls._meta_of(w),
            logical_shape=tuple(w.shape),
        )

    @classmethod
    def _meta_representation(cls, state: TorchaoHost[MetaT]) -> torch.Tensor:
        """Build a 0-byte ``meta`` wrapper with canonical target buffer shapes."""
        slot_shapes = _resolve_target_shapes(state.storage, state.target_shapes)
        meta_storage = tuple(
            torch.empty(shape, dtype=s.dtype, device="meta")
            if s is not None and shape is not None
            else None
            for s, shape in zip(state.storage, slot_shapes, strict=True)
        )
        return cls._reconstruct(meta_storage, state.meta)

    @classmethod
    def _finalize_projected_shard(
        cls,
        storage: tuple[torch.Tensor | None, ...],
        meta: MetaT,
        logical_shape: tuple[int, ...],
        *,
        target_shapes: tuple[tuple[int, ...] | None, ...] | None = None,
    ) -> tuple[TorchaoHost[MetaT], tuple[object, ...]]:
        """Build zero-copy projected host state and its canonical layout signature.

        Contiguous views matching their target buffer shapes stay native host
        wrappers on CPU; non-contiguous or reshaped tile views keep zero-copy
        slices in ``storage`` while populating ``target_shapes`` for
        contiguous device allocation.
        """
        resolved_shapes = _resolve_target_shapes(storage, target_shapes)
        canonical_host = all(
            s is None
            or (
                shape is not None
                and tuple(s.shape) == shape
                and s.is_contiguous()
                and s.stride() == _contiguous_strides(tuple(s.shape))
            )
            for s, shape in zip(storage, resolved_shapes, strict=True)
        )
        host = TorchaoHost(
            storage=storage,
            meta=meta,
            logical_shape=logical_shape,
            target_shapes=None if canonical_host else resolved_shapes,
        )
        probe = (
            cls._reconstruct(storage, meta)
            if canonical_host
            else cls._meta_representation(host)
        )
        cls._validate_layout(probe)
        if tuple(probe.shape) != logical_shape:
            raise ValueError(
                f"{cls.__name__} projected logical shape {tuple(probe.shape)} "
                f"does not match target shard shape {logical_shape}."
            )
        return host, cls.layout_signature(probe)

    @classmethod
    def accepts_bind_target(
        cls,
        state: TorchaoHost[MetaT],
        target: torch.Tensor,
        *,
        layout_signature: object,
    ) -> bool:
        return (
            is_plain_meta_placeholder(target, state.logical_shape)
            or (
                isinstance(target, _ProjectedTorchaoShard)
                and target._adapter_cls is cls
                and target._layout_signature == layout_signature
            )
            or matches_layout_signature(cls, layout_signature, target)
        )

    @staticmethod
    def storage_tensors(state: TorchaoHost[MetaT]) -> tuple[torch.Tensor, ...]:
        return tuple(tensor for tensor in state.storage if tensor is not None)

    @classmethod
    def cpu_param(cls, state: TorchaoHost[MetaT]) -> nn.Parameter:
        if state.target_shapes is not None:
            meta_tensor = cls._meta_representation(state)
            return nn.Parameter(
                _ProjectedTorchaoShard(
                    cls,
                    state,
                    meta_tensor,
                    cls.layout_signature(meta_tensor),
                ),
                requires_grad=False,
            )
        return nn.Parameter(
            cls._reconstruct(state.storage, state.meta),
            requires_grad=False,
        )

    @classmethod
    def alloc_gpu(
        cls, state: TorchaoHost[MetaT], device: torch.device
    ) -> TorchaoGpu:
        if state.target_shapes is not None:
            slot_shapes = _resolve_target_shapes(state.storage, state.target_shapes)
            return TorchaoGpu(
                storage=tuple(
                    torch.empty(shape, dtype=s.dtype, device=device)
                    if s is not None and shape is not None
                    else None
                    for s, shape in zip(state.storage, slot_shapes, strict=True)
                )
            )
        return TorchaoGpu(
            storage=tuple(
                empty_like_strided(s, device) if s is not None else None
                for s in state.storage
            )
        )

    @classmethod
    def gpu_param(
        cls,
        host: TorchaoHost[MetaT],
        gpu_state: TorchaoGpu,
    ) -> nn.Parameter:
        return nn.Parameter(
            cls._reconstruct(gpu_state.storage, host.meta),
            requires_grad=False,
        )

    @classmethod
    def copy_to_gpu(
        cls,
        src: TorchaoHost[MetaT],
        dst: TorchaoGpu,
        *,
        non_blocking: bool = False,
    ) -> None:
        dst_storage = dst.storage
        if src.target_shapes is not None:
            # View the contiguous GPU buffer to the host slice's tile shape so
            # ``s`` stays unwrapped for ``PinManager._leased_views[id(s)]``.
            dst_storage = tuple(
                d
                if s is None or d is None or tuple(d.shape) == tuple(s.shape)
                else d.view(s.shape)
                for s, d in zip(src.storage, dst.storage, strict=True)
            )
        copy_storage(src.storage, dst_storage, non_blocking=non_blocking)

    @classmethod
    def cache_bytes(cls, state: TorchaoHost[MetaT]) -> int:
        return sum(s.nbytes for s in state.storage if s is not None)

    @classmethod
    def compute_dtype(cls, t: torch.Tensor) -> torch.dtype:
        # Validate once through the shared _require, then read the logical
        # dtype off the wrapper — same single-validation path as the other
        # hooks (subclasses implement _compute_dtype on the wrapper).
        return cls._compute_dtype(cls._require(t))

    @classmethod
    def logical_shape(cls, t: torch.Tensor) -> tuple[int, ...]:
        """Dense shape carried directly by every supported TorchAO wrapper."""
        return tuple(cls._require(t).shape)
