"""Tensor-type adapters: per-type capture/move/wrap mechanics.

Different tensor subclasses need different machinery to move bytes
across the CPU↔GPU boundary while preserving correctness:

- Plain ``torch.Tensor``: single contiguous host buffer; consumers replace
  ``module._parameters[leaf]`` with a frozen :class:`nn.Parameter` wrapping it.
- Quantized tensor subclasses: packed values, scales, and metadata.
  The adapter reconstructs the wrapper over the active storage.

Each adapter encapsulates the mechanics for one tensor type. The rest
of the package is type-agnostic and dispatches through
``tensor_adapter_registry.select_adapter``. The base adapter contract is
intentionally small: host capture and storage enumeration, move to GPU,
rebuild wrappers, report cache bytes, and report the logical compute dtype.
Adapters also provide a layout signature for block-pool compatibility checks. Extra
operations are expressed as small optional protocols so callers ask for
the exact capability they need instead of hard-coding tensor classes.

The base :class:`TensorAdapter` contract is public through
:mod:`piper_offload`; downstream implementations can register themselves with
:func:`piper_offload.register_adapter`. The remaining helpers, optional
capability protocols, and plain tensor implementation live here; built-in and
external adapter selection lives in :mod:`tensor_adapter_registry`.
"""

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal, Protocol, cast, runtime_checkable

import torch
from torch import nn

from .pin_manager import transfer_

__all__ = [
    "BindTargetTensorAdapter",
    "BroadcastMergeTensorAdapter",
    "DenseMergeTargetValidationTensorAdapter",
    "DenseMergeTensorAdapter",
    "DenseMergeValidationTensorAdapter",
    "DequantRequantTensorAdapter",
    "DequantizeTensorAdapter",
    "GpuTargetBytesTensorAdapter",
    "LoRAMergeTensorAdapter",
    "LoRAMergeValidationTensorAdapter",
    "LogicalShapeTensorAdapter",
    "MergeLocalityTensorAdapter",
    "PermanentUpdateValidationTensorAdapter",
    "ShardProjectionTensorAdapter",
    "TensorAdapter",
    "TensorCopyIntoAdapter",
    "accepts_bind_target",
    "adapter_name",
    "capture_host_tensor",
    "empty_like_strided",
    "gpu_target_bytes",
    "host_state_is_owned",
    "is_plain_meta_placeholder",
    "matches_layout_signature",
    "metadata_key",
    "optional_tensor_id",
    "tensor_layout",
    "transfer_",
    "validate_broadcast_merge",
]

_host_capture_mode: ContextVar[Literal["transfer", "view", "copy"]] = ContextVar(
    "host_capture_mode", default="transfer",
)


def host_state_is_owned[HostStateT](
    adapter: TensorAdapter[HostStateT, Any], state: HostStateT,
) -> bool:
    """Whether every physical tensor of ``state`` lives in storage this process allocated.

    A view into a file mapping, or into a buffer the process did not
    allocate, is not owned. Piper never writes into a file mapping: the pin
    manager may return its pages to the file, so merge targets modified in place
    must be owned first.
    """
    return all(tensor.untyped_storage().resizable() for tensor in adapter.storage_tensors(state))


def view_host_state[S](adapter: TensorAdapter[S, Any], tensor: torch.Tensor) -> S:
    """Request the live physical CPU views. An adapter with its own copy logic may still return a copy."""
    with _host_capture_mode.set("view"):
        return adapter.capture_host(tensor)


def clone_host_state[S](adapter: TensorAdapter[S, Any], tensor: torch.Tensor, source: S) -> S:
    """Capture writable CPU storage disjoint from the source's physical tensors."""
    with _host_capture_mode.set("copy"):
        clone = adapter.capture_host(tensor)
    source_pointers = {t.untyped_storage().data_ptr() for t in adapter.storage_tensors(source) if t.numel()}
    for part in adapter.storage_tensors(clone):
        storage = part.untyped_storage()
        if not storage.resizable() or (part.numel() and storage.data_ptr() in source_pointers):
            raise ValueError(f"{adapter_name(adapter)} did not produce independent, owned CPU storage.")
    return clone


@runtime_checkable
class TensorAdapter[HostStateT, GpuStateT](Protocol):
    """Adapter encoding the mechanics of host capture, movement, and wrapping
    one tensor type. Adapter instances are stateless; they hold no
    per-param data.

    Generic over two opaque state types: ``HostStateT`` (the CPU
    representation) and ``GpuStateT`` (the GPU storage). Each
    adapter specializes these with its own concrete dataclasses; consumers
    round-trip the opaque types without inspecting them.

    The Protocol is methods-only — capability is determined by what an
    adapter implements, not by declarative flags. If a workload needs an
    operation beyond inference movement, it should check one of the
    smaller capability protocols below.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        """True if this adapter handles tensor ``t``. Implementations
        should be conservative — :class:`RegularAdapter` matches only plain
        ``torch.Tensor``, not unrecognized subclasses."""
        ...

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple:
        """Composite identity key for tied-weight detection. Two tensors
        with the same key share backing data and quant metadata; different
        keys must not be deduped. Includes device and view layout
        (shape/stride/offset) so distinct devices or views into the
        same buffer don't collapse."""
        ...

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple:
        """Hashable tensor layout metadata for block-pool compatibility.

        Unlike :meth:`tensor_id`, this must not include tensor identity.
        It captures only fields that must match for one GPU
        pool target to safely receive bytes from multiple block instances.
        """
        ...

    @staticmethod
    def capture_host(t: torch.Tensor) -> HostStateT:
        """Capture a pageable CPU representation of ``t``.

        Implementations may retain compatible complete CPU allocations and
        views into non-resizable storage so file mappings survive capture.
        Pass physical tensors to :func:`capture_host_tensor` so a permanent merge
        can inspect their original views or request independent, writable storage.
        Preserve the encoded data and metadata required by this adapter's
        movement and reconstruction operations.
        """
        ...

    @staticmethod
    def storage_tensors(state: HostStateT) -> tuple[torch.Tensor, ...]:
        """Return the physical CPU tensors retained by captured host state.

        Return existing plain strided tensors, including packed data, scales,
        and tensor-valued metadata even when that metadata stays on the CPU.
        Do not copy, dequantize, reconstruct wrappers, or move data. Composing
        adapters delegate to their local representation.

        Preserve views and empty tensors. Entries may share allocations;
        callers must deduplicate underlying storage when accounting for or
        registering memory. Tuple order has no cross-adapter meaning.
        """
        ...

    @staticmethod
    def cpu_param(state: HostStateT) -> nn.Parameter:
        """Build a frozen :class:`nn.Parameter` wrapping the host state.

        Used as the deactivated-state registry value
        (``module._parameters[leaf] = cpu_param``).
        """
        ...

    @staticmethod
    def alloc_gpu(state: HostStateT, device: torch.device) -> GpuStateT:
        """Allocate empty GPU storage mirroring this state's layout.
        Returns opaque adapter-specific state."""
        ...

    @staticmethod
    def gpu_param(host: HostStateT, gpu_state: GpuStateT) -> nn.Parameter:
        """Build a frozen :class:`nn.Parameter` wrapping the GPU state.
        Reused across many :meth:`copy_to_gpu` calls.

        Structured adapters use captured metadata from ``host`` to
        reconstruct the GPU-side wrapper.
        """
        ...

    @staticmethod
    def copy_to_gpu(
        src: HostStateT, dst: GpuStateT, *, non_blocking: bool = False
    ) -> None:
        """Bulk DMA the host state's bytes into pre-allocated GPU storage.

        Copy every physical tensor through :func:`transfer_` rather than
        ``Tensor.copy_``: a checkpoint storage pinned through an owned copy
        is read from that copy only when the transfer asks for it, and an
        adapter that copies from ``src`` directly pays for the copy without
        using it.
        """
        ...

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        """Return the logical compute dtype for operations using ``t``.

        For plain tensors this is simply ``t.dtype``. Quantized wrappers
        should return their logical matmul/output dtype, not necessarily
        the dtype of packed inner storage.
        """
        ...

    @staticmethod
    def cache_bytes(state: HostStateT) -> int:
        """Logical representation bytes reserved in the :class:`ResourceCache`.

        Count the physical representation tensors, including auxiliary
        storage and metadata retained by the adapter.
        """
        ...

@runtime_checkable
class LogicalShapeTensorAdapter[HostStateT, GpuStateT](
    TensorAdapter[HostStateT, GpuStateT],
    Protocol,
):
    """Optional capability for reading logical shape without materialization.

    Packed tensor subclasses may report a physical storage shape from their
    outer object. Adapters that can recover the logical shape from wrapper or
    quantization metadata expose it here, allowing validation paths to avoid a
    full dequantization solely to inspect ``dense.shape``.
    """

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        """Return the dense logical shape represented by ``t``."""
        ...


@runtime_checkable
class GpuTargetBytesTensorAdapter[HostStateT, GpuStateT](
    TensorAdapter[HostStateT, GpuStateT],
    Protocol,
):
    """Optional capability for adapters whose GPU target size differs from host ``cache_bytes``."""

    @staticmethod
    def gpu_target_bytes(state: HostStateT) -> int:
        """Return unique GPU storage bytes allocated by :meth:`alloc_gpu` for ``state``."""
        ...


def gpu_target_bytes[HostStateT](
    adapter: TensorAdapter[HostStateT, Any],
    state: HostStateT,
) -> int:
    """Return GPU target storage bytes for ``state``, defaulting to ``cache_bytes``."""
    if isinstance(adapter, GpuTargetBytesTensorAdapter):
        return adapter.gpu_target_bytes(state)
    return adapter.cache_bytes(state)


@runtime_checkable
class PermanentUpdateValidationTensorAdapter(Protocol):
    """Optional preflight for permanent in-place parameter updates.

    Source adapters whose active device representation supports updates while
    their resting representation does not can reject permanent mutation here.
    Activation loading does not consult this capability.
    """

    @staticmethod
    def validate_permanent_update() -> None:
        """Raise when the resting representation cannot be updated in place."""
        ...


@runtime_checkable
class BroadcastMergeTensorAdapter(Protocol):
    """Opt into PyTorch broadcasting for the adapter's merge capabilities.

    Without this capability each source update must match the target's full
    logical shape. The caller validates that every source broadcasts to that
    shape before calling this preflight once for a broadcast merge. Staged
    LoRA products and dense updates may retain a smaller broadcastable shape;
    merge implementations must preserve the target's shape and storage.
    This capability does not itself provide LoRA or dense merge support.
    """

    @staticmethod
    def validate_broadcast_merge(target: torch.Tensor) -> None:
        """Reject target representations or layouts that cannot broadcast updates."""
        ...


def validate_broadcast_merge[HostStateT, GpuStateT](
    adapter: TensorAdapter[HostStateT, GpuStateT], target: torch.Tensor,
) -> None:
    """Require explicit representation support after validating update shapes."""
    if not isinstance(adapter, BroadcastMergeTensorAdapter):
        raise ValueError(
            f"{adapter_name(adapter)} does not support broadcast additive merges; "
            "updates must match the target's full logical shape."
        )
    adapter.validate_broadcast_merge(target)


@runtime_checkable
class LoRAMergeTensorAdapter[HostStateT, GpuStateT](
    LogicalShapeTensorAdapter[HostStateT, GpuStateT],
    Protocol,
):
    """Optional capability for an in-place staged LoRA merge.

    The caller validates and stages factors for one combined ``B @ A`` update
    on the target device. Finite coefficients follow ``addmm``: ``alpha`` scales
    the update and ``beta`` scales the current target. ``alpha`` is the effective
    merge coefficient, not the checkpoint's intrinsic LoRA alpha.
    The adapter combines both terms before encoding once, preserving the target
    tensor's object and storage identities. Implementations may select a
    format-specific kernel or a regular framework-operator fallback.

    Merge and staged validation must accept keyword-only ``beta=1.0`` and
    ``rounding_seed: int | None = None``, including implementations that only
    support deterministic rounding. ``None`` selects deterministic behavior;
    :func:`piper_offload.seeding.derive_seed` provides reproducible substreams.
    Apply source scaling before addition, preserving activation calibration and
    encoding only the final result. A cancelling base delta can overflow or
    lose low-precision updates; it is not an equivalent implementation.
    """

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        """Write ``beta * target + alpha * (b @ a)`` with one encoding."""
        ...


@runtime_checkable
class LoRAMergeValidationTensorAdapter(Protocol):
    """Optional non-mutating validation for an advertised LoRA merge.

    Some representations support merge only for a subset of otherwise valid
    tensor layouts or staged factor values. Implementations expose those
    constraints here so permanent merge can reject every unsupported operation
    before mutating any model parameter.
    """

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
        """Validate the staged update and optional rounding mode."""
        ...


@runtime_checkable
class DenseMergeTensorAdapter[HostStateT, GpuStateT](
    LogicalShapeTensorAdapter[HostStateT, GpuStateT],
    Protocol,
):
    """Optional capability for an in-place staged full-rank merge.

    ``update`` is a dense logical delta on the target device and in the
    adapter's compute dtype. Finite coefficients follow the LoRA merge contract:
    ``alpha`` scales the update and ``beta`` scales the current target before
    addition, without encoding an intermediate scaled base. Implementations
    preserve the target tensor's object and storage identities; structured
    formats may use a native kernel or dequantize/add/requantize internally.
    Merge and staged validation follow :class:`LoRAMergeTensorAdapter`'s
    coefficient, rounding, and calibration-preservation contract.
    """

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        """Write ``beta * target + alpha * update`` with one encoding."""
        ...


@runtime_checkable
class DenseMergeTargetValidationTensorAdapter(Protocol):
    """Optional target-only preflight for an advertised dense merge."""

    @staticmethod
    def validate_dense_merge_target(
        target: torch.Tensor,
        *,
        rounding_seed: int | None = None,
    ) -> bool:
        """Validate ``target`` and report whether update validation is needed.

        Returning ``True`` requires the adapter to also implement
        :class:`DenseMergeValidationTensorAdapter`. This supports composing
        wrappers whose inner adapter may or may not need to inspect the staged
        update without forcing target-only formats to stage it during preflight.
        """
        ...


@runtime_checkable
class DenseMergeValidationTensorAdapter(Protocol):
    """Optional staged-update validation for an advertised dense merge."""

    @staticmethod
    def validate_dense_merge(
        target: torch.Tensor,
        update: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        """Validate the affine full-rank update and rounding mode."""
        ...


@runtime_checkable
class MergeLocalityTensorAdapter(Protocol):
    """Optional target-local region used to stage logical merge inputs.

    Composing adapters expose the local shape and global offsets of the target
    region. This capability does not itself advertise a merge operation.
    """

    @staticmethod
    def merge_local_shape_and_offsets(
        target: torch.Tensor,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return this target's local shape and global offsets."""
        ...


@runtime_checkable
class DequantizeTensorAdapter[HostStateT, GpuStateT](
    TensorAdapter[HostStateT, GpuStateT],
    Protocol,
):
    """Optional capability for exposing a dense logical representation.

    The returned tensor must be accepted as a local or global update by the
    same adapter's :class:`DenseMergeTensorAdapter` implementation. Composing
    distributed adapters may therefore return the dense local shard.
    """

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        """Return the dense logical value represented by ``t``."""
        ...


@runtime_checkable
class DequantRequantTensorAdapter[HostStateT, GpuStateT](
    DequantizeTensorAdapter[HostStateT, GpuStateT],
    Protocol,
):
    """Optional capability for shape-preserving dequantize/requantize conversion.

    ``dequantize(t)`` returns a dense logical tensor for ``t`` in
    ``compute_dtype(t)``. ``requantize(t, like=...)`` converts a
    shape-compatible dense tensor back into the same representation/layout
    as ``like``. Device follows the dense input tensor; callers can move
    tensors explicitly before calling.

    Data-dependent weight quantization parameters are recomputed from the new
    dense values. Callers must not assume scale bytes survive a
    dequantize/requantize round trip. Built-in quantized adapters internally
    compose stochastic terminal-code recoding with this same conversion, but
    the public structural protocol remains deterministic.
    This conversion capability does not by itself advertise in-place copy or
    LoRA merge support.
    """

    @staticmethod
    def requantize(
        t: torch.Tensor,
        *,
        like: torch.Tensor,
        rounding_seed: int | None = None,
    ) -> torch.Tensor:
        """Return ``t`` encoded in the same representation as ``like``."""
        ...


@runtime_checkable
class TensorCopyIntoAdapter[HostStateT, GpuStateT](
    TensorAdapter[HostStateT, GpuStateT],
    Protocol,
):
    """Optional capability for representation-preserving copy into ``target``.

    ``copy_into(src, target=...)`` copies ``src``'s representation into
    ``target`` while preserving ``target``'s object identity and storage.
    Structured tensor wrappers use this when generic ``target.copy_(src)``
    does not update their internal storage correctly.
    """

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        """Copy ``src`` into ``target``'s existing representation."""
        ...


@runtime_checkable
class BindTargetTensorAdapter[HostStateT, GpuStateT](
    TensorAdapter[HostStateT, GpuStateT],
    Protocol,
):
    """Optional capability: own and relax bind target validation.

    By default, store-to-module bind validation requires the target parameter
    to match this adapter and compare equal on
    :meth:`TensorAdapter.layout_signature`. Adapters implementing this protocol
    own the bind compatibility decision — for example, accepting plain
    tensors whose dtype differs from the store, plain strided ``meta``
    placeholders matching a frozen quantized store's logical shape, or
    ``DTensor`` wrappers resting on another mesh device type — without
    requiring :meth:`TensorAdapter.layout_signature` to succeed on every
    incompatible target first.

    Consulted only for frozen physical host stores.
    """

    @staticmethod
    def accepts_bind_target(
        state: HostStateT,
        target: torch.Tensor,
        *,
        layout_signature: object,
    ) -> bool:
        """Return whether ``target`` is a compatible bind target for ``state``."""
        ...


@runtime_checkable
class ShardProjectionTensorAdapter[HostStateT, GpuStateT](
    TensorAdapter[HostStateT, GpuStateT],
    Protocol,
):
    """Optional capability: project an unsharded host state onto a contiguous shard.

    Adapters implementing this protocol slice or alias the full host backing
    into a rank-local host state without copying payload bytes, returning
    ``(projected_state, canonical_inner_layout_signature)``. Formats with
    quantization block/group structure validate sharding alignment here and
    derive shard-local metadata and layout signatures.
    """

    @staticmethod
    def project_shard(
        state: HostStateT,
        *,
        offsets: tuple[int, ...],
        local_shape: tuple[int, ...],
    ) -> tuple[HostStateT, tuple[object, ...]]:
        """Project a zero-copy rank-local shard from captured host ``state``."""
        ...


def is_plain_meta_placeholder(
    target: torch.Tensor,
    logical_shape: tuple[int, ...],
) -> bool:
    """Return whether ``target`` is a plain strided ``meta`` tensor with ``logical_shape``."""
    return (
        type(target) is torch.Tensor
        and target.is_meta
        and target.layout is torch.strided
        and tuple(target.shape) == logical_shape
    )


def matches_layout_signature(
    adapter: TensorAdapter[Any, Any] | type[TensorAdapter[Any, Any]],
    layout_signature: object,
    target: torch.Tensor,
) -> bool:
    """Return whether ``target`` matches ``adapter`` and its captured ``layout_signature``."""
    return adapter.matches(target) and adapter.layout_signature(target) == layout_signature


def accepts_bind_target(
    adapter: TensorAdapter[Any, Any],
    state: object,
    layout_signature: object,
    target: torch.Tensor,
) -> bool:
    """Return whether ``target`` satisfies strict or adapter-relaxed bind validation."""
    if isinstance(adapter, BindTargetTensorAdapter):
        return cast(BindTargetTensorAdapter[Any, Any], adapter).accepts_bind_target(
            state,
            target,
            layout_signature=layout_signature,
        )
    return matches_layout_signature(adapter, layout_signature, target)


# ---------------------------------------------------------------------------
# RegularAdapter — plain torch.Tensor (bf16/fp16/fp32, etc.)
# ---------------------------------------------------------------------------


def capture_host_tensor(
    t: torch.Tensor,
    *,
    memory_format: torch.memory_format = torch.preserve_format,
) -> torch.Tensor:
    """Transfer ownership of compatible CPU storage or copy into host memory.

    Complete CPU allocations and non-empty views into non-resizable CPU storage
    with the requested layout are retained directly. Safetensors and file-backed
    storages are non-resizable, so split checkpoint views keep their mappings.
    Device tensors and partial views into ordinary resizable allocations are
    normalized into a new CPU allocation.
    """
    source = t.detach()
    mode = _host_capture_mode.get()
    if mode == "view":
        return source
    source_storage = (
        source.untyped_storage() if source.layout is torch.strided else None
    )
    owns_complete_storage = False
    if source_storage is not None:
        storage_span = 0 if source.numel() == 0 else 1 + sum(
            (size - 1) * stride
            for size, stride in zip(source.shape, source.stride(), strict=True)
        )
        owns_complete_storage = (
            source.storage_offset() == 0
            and source_storage.nbytes() == storage_span * source.element_size()
        )
    requested_layout = (
        memory_format is torch.preserve_format
        or source.is_contiguous(memory_format=memory_format)
    )
    transferable_storage = owns_complete_storage or (
        source.device.type == "cpu"
        and source_storage is not None
        and source.numel() > 0
        and not source_storage.resizable()
    )
    if (
        mode == "transfer"
        and source.device.type == "cpu"
        and not source.is_pinned()
        and transferable_storage
        and requested_layout
    ):
        return source

    # Preserve exact strides for device sources and clone-style layout
    # normalization for CPU sources. Allocate the final host destination once
    # outside inference mode so captured buffers are not inference tensors.
    with torch.inference_mode(False):
        if source.device.type != "cpu" and memory_format == torch.preserve_format:
            host = torch.empty_strided(
                tuple(source.shape),
                source.stride(),
                dtype=source.dtype,
                device="cpu",
                pin_memory=False,
            )
        else:
            host = torch.empty_like(
                source,
                device="cpu",
                memory_format=memory_format,
                pin_memory=False,
            )
        host.copy_(source)
    return host


# ---------------------------------------------------------------------------
# Shared helpers for structured-tensor adapters (nvfp4, float8, ...)
# ---------------------------------------------------------------------------


def empty_like_strided(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Allocate an empty tensor on ``device`` mirroring ``t``'s shape,
    stride, and dtype. Structured-tensor adapters use this so inner
    storage tensors keep their stride ordering (e.g. transposed packed
    data) across the host/GPU boundary."""
    return torch.empty_strided(
        tuple(t.shape),
        t.stride(),
        dtype=t.dtype,
        device=device,
    )


def _storage_identity(t: torch.Tensor) -> object:
    ptr = t.data_ptr()
    if ptr != 0:
        return ptr
    if t.layout is torch.strided:
        return ("storage", t.untyped_storage()._cdata)
    return ("tensor", id(t))


def _physical_tensor_id(t: torch.Tensor) -> tuple[object, ...]:
    return (
        t.device,
        _storage_identity(t),
        t.dtype,
        tuple(t.shape),
        t.stride(),
        t.storage_offset(),
        t.is_conj(),
        t.is_neg(),
    )


def optional_tensor_id(t: torch.Tensor | None) -> tuple[object, ...] | None:
    """Identity key fragment for one (possibly absent) inner storage
    tensor, for use in adapter :meth:`TensorAdapter.tensor_id` keys."""
    if t is None:
        return None
    return _physical_tensor_id(t)


def tensor_layout(t: torch.Tensor | None) -> tuple[object, ...] | None:
    """Identity-free layout fragment for one (possibly absent) inner
    storage tensor, for use in adapter
    :meth:`TensorAdapter.layout_signature` values."""
    if t is None:
        return None
    return (tuple(t.shape), t.dtype, t.stride())


def metadata_key(value: object | None) -> object | None:
    """Hashable key for opaque quant metadata (dataclasses, configs).

    Dataclass instances become a structural (type, fields) key so two
    equal-valued instances compare equal; anything else falls back to
    ``repr`` for stability over unhashable or unknown types.
    """
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            _make_hashable(asdict(value)),
        )
    return repr(value)


def _make_hashable(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(
            (repr(k), _make_hashable(v))
            for k, v in sorted(value.items(), key=lambda item: repr(item[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_make_hashable(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_make_hashable(v) for v in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


@dataclass(slots=True)
class _RegularHost:
    """CPU state for a regular tensor, possibly a projected noncontiguous view."""

    data: torch.Tensor


@dataclass(slots=True)
class _RegularGpu:
    """GPU state for a regular tensor: one contiguous device buffer."""

    data: torch.Tensor


class RegularAdapter:
    """Adapter for plain ``torch.Tensor`` (no subclass machinery).

    Builds frozen :class:`nn.Parameter` objects wrapping the CPU and GPU
    storages. Model-bound callers replace the module registry via
    ``module._parameters[leaf] = ...`` with a CPU wrapper or active GPU wrapper.

    Conservative on dispatch: only matches exactly
    ``type(t) is torch.Tensor`` (or ``nn.Parameter``). Unrecognized
    tensor subclasses fall through to other adapters or raise via the
    factory.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        # A subclass can carry representation metadata beyond its storage.
        return type(t) is torch.Tensor or type(t) is nn.Parameter

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple:
        return ("regular", *_physical_tensor_id(t))

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple:
        return (tuple(t.shape), t.dtype, t.layout)

    @staticmethod
    def accepts_bind_target(
        state: _RegularHost,
        target: torch.Tensor,
        *,
        layout_signature: object,
    ) -> bool:
        del layout_signature
        return (
            RegularAdapter.matches(target)
            and target.layout is torch.strided
            and tuple(target.shape) == tuple(state.data.shape)
        )

    @staticmethod
    def capture_host(t: torch.Tensor) -> _RegularHost:
        return _RegularHost(
            data=capture_host_tensor(
                t.data,
                memory_format=torch.contiguous_format,
            )
        )

    @staticmethod
    def capture_host_view(view: torch.Tensor) -> _RegularHost:
        """Wrap an existing host view without normalizing or copying it."""
        return _RegularHost(data=view.detach())

    @staticmethod
    def project_shard(
        state: _RegularHost,
        *,
        offsets: tuple[int, ...],
        local_shape: tuple[int, ...],
    ) -> tuple[_RegularHost, tuple[object, ...]]:
        """Project a zero-copy host view for a contiguous ``DTensor`` shard."""
        source = state.data
        if len(offsets) != source.ndim or len(local_shape) != source.ndim:
            raise ValueError(
                "Shard projection dimensions must match the source tensor: "
                f"source={tuple(source.shape)}, offsets={offsets}, local_shape={local_shape}."
            )
        local = source
        for dim, (offset, size) in enumerate(zip(offsets, local_shape, strict=True)):
            if offset < 0 or size < 0 or offset + size > local.shape[dim]:
                raise ValueError(
                    "Shard projection range exceeds source dimensions: "
                    f"source={tuple(source.shape)}, dim={dim}, offset={offset}, size={size}."
                )
            if offset != 0 or size != local.shape[dim]:
                local = local.narrow(dim, offset, size)
        return (
            RegularAdapter.capture_host_view(local),
            RegularAdapter.layout_signature(local),
        )

    @staticmethod
    def storage_tensors(state: _RegularHost) -> tuple[torch.Tensor, ...]:
        return (state.data,)

    @staticmethod
    def cpu_param(state: _RegularHost) -> nn.Parameter:
        return nn.Parameter(state.data, requires_grad=False)

    @staticmethod
    def alloc_gpu(state: _RegularHost, device: torch.device) -> _RegularGpu:
        # Allocate contiguous device storage so projected non-contiguous views
        # (e.g., Shard(1)) materialize contiguously on the device.
        return _RegularGpu(
            data=torch.empty_like(
                state.data,
                device=device,
                memory_format=torch.contiguous_format,
            )
        )

    @staticmethod
    def gpu_param(host: _RegularHost, gpu_state: _RegularGpu) -> nn.Parameter:
        _ = host
        return nn.Parameter(gpu_state.data, requires_grad=False)

    @staticmethod
    def copy_to_gpu(
        src: _RegularHost, dst: _RegularGpu, *, non_blocking: bool = False
    ) -> None:
        transfer_(dst.data, src.data, non_blocking=non_blocking)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        return t.dtype

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        return tuple(t.shape)

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        """Return an independent dense logical value."""
        return t.clone()

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge in the staged compute dtype, preserving plain-tensor rounding.

        An exact 2D target uses fused ``addmm_``. A broadcast target first forms
        the compact ``b @ a`` product with autocast disabled, then adds it through
        ``merge_dense_``. That product rounds before addition, so the paths need
        not be bit-identical. Neither path promotes staged factors; PyTorch's
        kernels determine internal accumulation precision.
        """
        del rounding_seed
        if target.shape == (b.shape[0], a.shape[1]):
            target.addmm_(b, a, beta=beta, alpha=alpha)
        else:
            with torch.autocast(device_type=target.device.type, enabled=False):
                RegularAdapter.merge_dense_(target, b @ a, alpha, beta=beta)

    @staticmethod
    def validate_broadcast_merge(target: torch.Tensor) -> None:
        if target.dtype.is_floating_point and torch.finfo(target.dtype).bits == 8:
            raise ValueError("Broadcast merges do not support float8 targets with plain storage.")

    @staticmethod
    def validate_dense_merge_target(
        target: torch.Tensor,
        *,
        rounding_seed: int | None = None,
    ) -> bool:
        del rounding_seed
        if target.dtype.is_floating_point and torch.finfo(target.dtype).bits == 8:
            raise ValueError(
                "Dense merges do not support float8 targets with plain storage; "
                f"got {target.dtype}."
            )
        return False

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        del rounding_seed
        if beta == 1.0:
            target.add_(update, alpha=alpha)
        else:
            target.copy_(target.mul(beta).add_(update, alpha=alpha))

    @staticmethod
    def cache_bytes(state: _RegularHost) -> int:
        return state.data.numel() * state.data.element_size()


def adapter_name(adapter: TensorAdapter[Any, Any]) -> str:
    """Human-readable name for an adapter instance."""
    return type(adapter).__name__
