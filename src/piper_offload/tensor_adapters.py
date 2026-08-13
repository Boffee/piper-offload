"""Tensor-type adapters: per-type pin/move/wrap mechanics.

Different tensor subclasses need different machinery to move bytes
across the CPU↔GPU boundary while preserving correctness:

- Plain ``torch.Tensor``: single contiguous pinned
  buffer; consumers either replace ``module._parameters[leaf]`` with a
  fresh :class:`nn.Parameter` wrapping it, or ``.data``-swap to preserve
  identity for trainable params.
- Quanto ``WeightQBytesTensor``: two pinned tensors (``_data`` + ``_scale``)
  plus quant metadata; the wrapper must be reconstructed on each move.
  ``.data``-swap doesn't work for quanto — its quant state is part of
  the Parameter's wrapped object, not its bytes — so quanto stays
  frozen-only via registry replacement.

Each adapter encapsulates the mechanics for one tensor type. The rest
of the package is type-agnostic and dispatches through
``tensor_adapter_registry.select_adapter``. The base adapter contract is
intentionally small: clone/pin, move to GPU, rebuild wrappers, report
cache bytes, and report the logical compute dtype. Adapters also
provide a layout signature for block-pool compatibility checks. Extra
operations are expressed as small optional protocols so callers ask for
the exact capability they need instead of hard-coding tensor classes.

The base :class:`TensorAdapter` contract is public through
:mod:`piper_offload`; downstream implementations can register themselves with
:func:`piper_offload.register_adapter`. The remaining helpers, optional
capability protocols, and plain tensor implementation live here; built-in and
external adapter selection lives in :mod:`tensor_adapter_registry`.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

__all__ = [
    "AdoptableTensorAdapter",
    "BindLayoutTensorAdapter",
    "CpuRoundTripTensorAdapter",
    "DequantRequantTensorAdapter",
    "LoRAMergeTensorAdapter",
    "LoRAMergeValidationTensorAdapter",
    "LogicalShapeTensorAdapter",
    "ParameterDataSwapTensorAdapter",
    "PostLoadRearmTensorAdapter",
    "TensorAdapter",
    "TensorCopyIntoAdapter",
    "adapter_name",
    "adopt_cpu_storage",
    "clone_to_pinned_cpu",
    "empty_like_strided",
    "metadata_key",
    "optional_tensor_id",
    "tensor_layout",
]

@runtime_checkable
class TensorAdapter[PinnedStateT, GpuStateT](Protocol):
    """Adapter encoding the mechanics of pinning, moving, and wrapping
    one tensor type. Adapter instances are stateless; they hold no
    per-param data.

    Generic over two opaque state types: ``PinnedStateT`` (the pinned
    host representation) and ``GpuStateT`` (the GPU storage). Each
    adapter pins these to its own concrete dataclasses; consumers
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
    def clone_pin(t: torch.Tensor) -> PinnedStateT:
        """Clone ``t`` into pinned (or regular) host memory. Returns
        opaque adapter-specific state used by subsequent operations."""
        ...

    @staticmethod
    def cpu_param(
        state: PinnedStateT, *, requires_grad: bool = False
    ) -> nn.Parameter:
        """Build a stable :class:`nn.Parameter` wrapping the host state.
        Used as the deactivated-state registry value
        (``module._parameters[leaf] = cpu_param``).

        ``requires_grad`` defaults to ``False`` to match frozen
        registry-replacement callers; pass ``True`` when building a wrapper
        for trainable storage.
        """
        ...

    @staticmethod
    def alloc_gpu(state: PinnedStateT, device: torch.device) -> GpuStateT:
        """Allocate empty GPU storage mirroring this state's layout.
        Returns opaque adapter-specific state."""
        ...

    @staticmethod
    def gpu_param(
        pinned: PinnedStateT, gpu_state: GpuStateT, *, requires_grad: bool = False
    ) -> nn.Parameter:
        """Build a stable :class:`nn.Parameter` wrapping the GPU state.
        Reused across many :meth:`copy_to_gpu` calls.

        Takes both the pinned host state and the GPU state because
        adapters with structured tensors (e.g. quanto) need metadata
        captured at pin time to reconstruct the GPU-side wrapper. Plain
        adapters ignore ``pinned``.

        ``requires_grad`` defaults to ``False``; pass ``True`` for
        trainable use cases where the wrapper participates in autograd.
        """
        ...

    @staticmethod
    def copy_to_gpu(
        src: PinnedStateT, dst: GpuStateT, *, non_blocking: bool = False
    ) -> None:
        """Bulk DMA the pinned state's bytes into pre-allocated GPU storage."""
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
    def cache_bytes(state: PinnedStateT) -> int:
        """Logical representation bytes charged to :class:`ResourceCache`.

        Count the tensor bytes represented by ``state``. This is deliberately
        independent of an adopted tensor's underlying allocation capacity,
        checkpoint file size, and current mmap residency.
        """
        ...


@runtime_checkable
class AdoptableTensorAdapter[PinnedStateT](Protocol):
    """Optional capability for adopting existing CPU host state.

    This stays separate from :class:`TensorAdapter` so existing third-party
    adapters remain compatible with the default pinned path.
    """

    @staticmethod
    def adopt_host(t: torch.Tensor) -> PinnedStateT:
        """Return adapter state that aliases existing CPU storage."""
        ...


@runtime_checkable
class CpuRoundTripTensorAdapter[PinnedStateT, GpuStateT](
    TensorAdapter[PinnedStateT, GpuStateT],
    Protocol,
):
    """Optional D2H counterpart to the base H2D movement contract."""

    @staticmethod
    def copy_to_cpu(
        src: GpuStateT, dst: PinnedStateT, *, non_blocking: bool = False
    ) -> None:
        """Bulk D2H the GPU state's bytes into pinned host storage.

        Symmetric counterpart to :meth:`copy_to_gpu`. Used to sync the
        pinned host clone with post-update GPU contents — e.g., after
        an optimizer step has written into the GPU param, scatter the
        update back to the pinned state so the next H2D reads it.

        Adapters whose GPU representation is not round-trippable should
        not implement this capability.
        """
        ...


@runtime_checkable
class LogicalShapeTensorAdapter[PinnedStateT, GpuStateT](
    TensorAdapter[PinnedStateT, GpuStateT],
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
class LoRAMergeTensorAdapter[PinnedStateT, GpuStateT](
    LogicalShapeTensorAdapter[PinnedStateT, GpuStateT],
    Protocol,
):
    """Optional capability for an in-place staged LoRA merge.

    The caller validates and stages one combined ``B @ A`` update on the
    target device. The adapter applies it while preserving the target tensor's
    object and storage identities. Implementations may select a format-specific
    kernel or a regular framework-operator fallback.
    """

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge the staged update, optionally using stochastic rounding."""
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
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Validate the staged update and optional rounding mode."""
        ...


@runtime_checkable
class DequantRequantTensorAdapter[PinnedStateT, GpuStateT](
    TensorAdapter[PinnedStateT, GpuStateT],
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
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        """Return a dense logical tensor in ``compute_dtype(t)``."""
        ...

    @staticmethod
    def requantize(t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        """Return ``t`` encoded in the same representation as ``like``."""
        ...


@runtime_checkable
class ParameterDataSwapTensorAdapter[PinnedStateT, GpuStateT](
    TensorAdapter[PinnedStateT, GpuStateT],
    Protocol,
):
    """Optional capability for trainable streaming via ``Parameter.data`` swap."""

    @staticmethod
    def validate_parameter_data_swap_target(t: torch.Tensor) -> None:
        """Raise if ``t`` cannot safely round-trip through ``param.data =``.

        Streamed trainables preserve user Parameter identity by swapping
        only ``.data``. Tensor subclasses with wrapper metadata generally
        must not opt into this capability.
        """
        ...


@runtime_checkable
class TensorCopyIntoAdapter[PinnedStateT, GpuStateT](
    TensorAdapter[PinnedStateT, GpuStateT],
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
class BindLayoutTensorAdapter[PinnedStateT, GpuStateT](
    TensorAdapter[PinnedStateT, GpuStateT],
    Protocol,
):
    """Optional capability: relaxed layout for store↔module bind validation.

    Binding replaces every managed tensor in the target module with
    store-backed storage, so a placeholder's dtype carries no information
    past validation — a meta skeleton built from config alone (e.g. fp32
    defaults) is structurally compatible with a store pinned from natively
    loaded bf16 or mixed-precision weights. Adapters that opt in supply a
    bind signature without the fields binding discards; adapters that
    don't (quantized wrappers, whose placeholder representation is
    structural) keep the strict :meth:`TensorAdapter.layout_signature`
    comparison.
    """

    @staticmethod
    def bind_layout_signature(t: torch.Tensor) -> tuple:
        """Hashable structural layout for bind validation; excludes
        fields binding overwrites (dtype for plain tensors)."""
        ...


@runtime_checkable
class PostLoadRearmTensorAdapter[PinnedStateT, GpuStateT](
    TensorAdapter[PinnedStateT, GpuStateT],
    Protocol,
):
    """Optional capability: re-arm the active GPU wrapper after each load.

    The offloader builds one GPU wrapper per pool target and reuses it
    across loads, refilling its buffers each time. That assumes the wrapper
    keeps describing itself across refills — true for plain tensors, 4-bit,
    and quanto, whose reconstructed wrapper aliases the refilled buffers.

    bitsandbytes int8 breaks the assumption: the first forward migrates
    ``CB``/``SCB`` onto the owning module (``MatmulLtState``) and nulls them
    on the wrapper, so the reused wrapper never re-initializes — and in the
    pooled streaming path, where blocks share buffers, the stale module
    state reads another block's bytes. Adapters that opt in re-point the
    wrapper's migrated state at the freshly-loaded GPU storage after each
    copy, so the next forward re-initializes from the current data.
    """

    @staticmethod
    def rearm_after_load(param: nn.Parameter, gpu_state: GpuStateT) -> None:
        """Re-point ``param``'s migrated quant state at ``gpu_state``,
        called once per load after :meth:`TensorAdapter.copy_to_gpu` and
        before the wrapper is installed into its module."""
        ...


# ---------------------------------------------------------------------------
# RegularAdapter — plain torch.Tensor (bf16/fp16/fp32, etc.)
# ---------------------------------------------------------------------------


def clone_to_pinned_cpu(
    t: torch.Tensor,
    *,
    memory_format: torch.memory_format = torch.preserve_format,
) -> torch.Tensor:
    """Clone ``t`` into pinned CPU memory from any source device."""
    source = t.detach()
    # Allocate the final destination in pinned memory. ``clone().pin_memory()``
    # first materializes a complete pageable clone and then copies it again
    # into a pinned allocation, adding one source-tensor-sized construction
    # temporary. Preserve the existing exact-stride behavior for CUDA sources;
    # the CPU path follows ``clone(memory_format=...)`` normalization via
    # ``empty_like``.
    if source.device.type != "cpu" and memory_format == torch.preserve_format:
        pinned = torch.empty_strided(
            tuple(source.shape),
            source.stride(),
            dtype=source.dtype,
            device="cpu",
            pin_memory=True,
        )
    else:
        pinned = torch.empty_like(
            source,
            device="cpu",
            memory_format=memory_format,
            pin_memory=True,
        )
    pinned.copy_(source)
    return pinned


def adopt_cpu_storage(
    t: torch.Tensor,
    *,
    memory_format: torch.memory_format = torch.preserve_format,
) -> torch.Tensor:
    """Return a detached alias of compatible existing CPU storage.

    Adoption is deliberately strict: callers are asking to retain the source
    allocation (including mmap/file backing), so this helper never moves,
    clones, or normalizes a tensor implicitly.
    """
    source = t.detach()
    if source.device.type != "cpu":
        raise ValueError(
            "adopted host backing requires an existing CPU tensor; "
            f"got device {source.device}. Move the model to CPU first or use "
            "host_backing='pinned'."
        )
    if (
        memory_format == torch.contiguous_format
        and not source.is_contiguous()
    ):
        raise ValueError(
            "adopted host backing cannot retain a non-contiguous tensor "
            "where this adapter requires contiguous storage; materialize a "
            "contiguous CPU tensor before constructing the offloader."
        )
    if memory_format not in (
        torch.preserve_format,
        torch.contiguous_format,
    ):
        raise ValueError(
            "adopted host backing only supports preserve_format or "
            "contiguous_format adoption."
        )
    return source


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


def optional_tensor_id(t: torch.Tensor | None) -> tuple[object, ...] | None:
    """Identity key fragment for one (possibly absent) inner storage
    tensor, for use in adapter :meth:`TensorAdapter.tensor_id` keys."""
    if t is None:
        return None
    return (
        t.device,
        t.data_ptr(),
        t.dtype,
        tuple(t.shape),
        t.stride(),
        t.storage_offset(),
    )


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
class _RegularPinned:
    """Pinned-CPU state for a regular tensor: one contiguous host buffer."""

    data: torch.Tensor


@dataclass(slots=True)
class _RegularGpu:
    """GPU state for a regular tensor: one contiguous device buffer."""

    data: torch.Tensor


class RegularAdapter:
    """Adapter for plain ``torch.Tensor`` (no subclass machinery).

    Builds fresh :class:`nn.Parameter` objects wrapping the pinned-CPU
    and GPU storages. Frozen model-bound callers replace the module registry via
    ``module._parameters[leaf] = ...`` with a pinned CPU wrapper or
    active GPU wrapper; trainable callers preserve Parameter identity
    by skipping registry replacement and ``.data``-swapping into their own
    persistent Parameter. Both paths are supported by the shape of this
    adapter (plain tensors round-trip through ``.data =`` cleanly).

    Conservative on dispatch: only matches exactly
    ``type(t) is torch.Tensor`` (or ``nn.Parameter``). Unrecognized
    tensor subclasses fall through to other adapters or raise via the
    factory.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        # Strict identity match on the base class. PEFT, FSDP, quanto,
        # DTensor, etc. are subclasses with extra state; a silent fallback
        # to RegularAdapter would clone-and-dequantize quanto or break
        # distributed placement. Each subclass needs its own adapter.
        return type(t) is torch.Tensor or type(t) is nn.Parameter

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple:
        return (
            "regular",
            t.device,
            t.data_ptr(),
            t.dtype,
            tuple(t.shape),
            t.stride(),
            t.storage_offset(),
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple:
        return (tuple(t.shape), t.dtype)

    @staticmethod
    def bind_layout_signature(t: torch.Tensor) -> tuple:
        # dtype excluded: bind replaces plain-tensor placeholders with
        # store-backed storage, so only the shape is structural.
        return (tuple(t.shape),)

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _RegularPinned:
        return _RegularPinned(
            data=clone_to_pinned_cpu(
                t.data,
                memory_format=torch.contiguous_format,
            )
        )

    @staticmethod
    def adopt_host(t: torch.Tensor) -> _RegularPinned:
        return _RegularPinned(
            data=adopt_cpu_storage(
                t.data,
                memory_format=torch.contiguous_format,
            )
        )

    @staticmethod
    def cpu_param(
        state: _RegularPinned, *, requires_grad: bool = False
    ) -> nn.Parameter:
        return nn.Parameter(state.data, requires_grad=requires_grad)

    @staticmethod
    def alloc_gpu(state: _RegularPinned, device: torch.device) -> _RegularGpu:
        return _RegularGpu(data=torch.empty_like(state.data, device=device))

    @staticmethod
    def gpu_param(
        pinned: _RegularPinned,
        gpu_state: _RegularGpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        _ = pinned
        # pinned unused: regular tensors carry no metadata beyond storage.
        # Argument kept for Protocol parity with TensorAdapter — quanto
        # and other structured tensors need it to reconstruct wrappers.
        return nn.Parameter(gpu_state.data, requires_grad=requires_grad)

    @staticmethod
    def copy_to_gpu(
        src: _RegularPinned, dst: _RegularGpu, *, non_blocking: bool = False
    ) -> None:
        dst.data.copy_(src.data, non_blocking=non_blocking)

    @staticmethod
    def copy_to_cpu(
        src: _RegularGpu, dst: _RegularPinned, *, non_blocking: bool = False
    ) -> None:
        dst.data.copy_(src.data, non_blocking=non_blocking)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        return t.dtype

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        return tuple(t.shape)

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        del rounding_seed
        target.addmm_(b, a, alpha=strength)

    @staticmethod
    def validate_parameter_data_swap_target(t: torch.Tensor) -> None:
        if type(t) is not torch.Tensor:
            raise NotImplementedError(
                f"Parameter data-swap target is {type(t).__name__}; "
                "Parameter.data swap requires a plain torch.Tensor."
            )

    @staticmethod
    def cache_bytes(state: _RegularPinned) -> int:
        return state.data.numel() * state.data.element_size()


def adapter_name(adapter: TensorAdapter[Any, Any]) -> str:
    """Human-readable name for an adapter instance."""
    return type(adapter).__name__
