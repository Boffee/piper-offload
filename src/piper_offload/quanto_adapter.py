"""Quanto :class:`WeightQBytesTensor` adapter.

Quanto-quantized weights are subclassed tensors that wrap multiple
internal tensors (``_data``, ``_scale``) plus quant metadata
(``qtype``, ``axis``, ``activation_qtype``). The wrapper does not
support ``p.data = ...`` storage swap — its quant state is part of the
Parameter's wrapped object, not its bytes. So this adapter:

- Decomposes ``WeightQBytesTensor`` into ``_data`` and ``_scale``,
  pins each separately.
- Reconstructs a fresh ``WeightQBytesTensor`` (and thus a fresh
  :class:`nn.Parameter`) on each activate via registry replacement.
  PyTorch optimizers keyed by the user's pre-wrap Parameter id are
  orphaned across cycles — quanto-quantized weights are inference-only.

Device-optimized subclasses need one additional boundary rule.
optimum-quanto 0.2.7's ``MarlinF8QBytesTensor`` stores packed INT32 data
and permuted scales, and its wrapper-level ``copy_`` unpacks instead of
updating the packed leaf. The adapter therefore canonicalizes Marlin
inputs to the kernel-agnostic ``WeightQBytesTensor`` representation when
pinning and keeps that raw representation throughout streaming. Direct
updates to an existing Marlin wrapper are repacked into its physical
``_data._data`` buffer so wrapper, workspace, and storage identity remain
stable.

Reaches into quanto's private attributes (``_data``, ``_scale``,
``qtype``, ``axis``, ``activation_qtype``). Pinned to the
``WeightQBytesTensor`` layout in optimum-quanto as of the version this
repo depends on. If quanto refactors the wrapper class, the
:class:`QuantoAdapter` will fail with a clear validation error at
:meth:`matches` (validates the expected attributes exist on first
match).

CUDA qint8 and qfloat8 merges use one fixed-scale Triton pipeline when
available. Unsupported layouts and installations without Triton keep the
same dequantize/GEMM/requantize fallback.

Selected by :mod:`tensor_adapter_registry`. Importing fails silently if
optimum-quanto is not installed — quanto support is optional.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ._quanto import (
    canonical_qbytes_storage_layout,
    canonicalize_qbytes_tensor,
    copy_qbytes_tensor_,
    create_qbytes_tensor,
    dequantize_qbytes_tensor,
    is_marlin_f8_qbytes_tensor,
    is_weight_qbytes_tensor,
    qbytes_activation_qtype,
    qbytes_data_storage,
    requantize_qbytes_tensor,
    require_qbytes_tensor,
    validate_layout,
)
from .tensor_adapters import clone_to_pinned_cpu

try:
    from ._triton_quanto_lora import (
        merge_quanto_qfloat8_lora as _triton_merge_quanto_qfloat8_lora,
    )
    from ._triton_quanto_lora import (
        merge_quanto_qint8_lora as _triton_merge_quanto_qint8_lora,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_merge_quanto_qfloat8_lora = None
    _triton_merge_quanto_qint8_lora = None


@dataclass(slots=True)
class _QuantoPinned:
    """Pinned-CPU state for a quanto tensor: two pinned tensors
    plus the quant metadata needed to reconstruct the wrapper."""

    data: torch.Tensor   # pinned int8/fp8
    scale: torch.Tensor  # pinned fp16/fp32
    qtype: object
    axis: int | None
    size: torch.Size
    stride: tuple[int, ...]
    act_qt: object | None


@dataclass(slots=True)
class _QuantoGpu:
    """GPU state for a quanto tensor: the two GPU tensors. Quant
    metadata lives in the originating :class:`_QuantoPinned`; only
    storage moves to GPU."""

    data: torch.Tensor
    scale: torch.Tensor


def _build_qbytes(
    state: _QuantoPinned, data: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Reconstruct a :class:`WeightQBytesTensor` from raw pieces +
    cached quant metadata."""
    return create_qbytes_tensor(
        state.qtype,
        state.axis,
        state.size,
        state.stride,
        data,
        scale,
        state.act_qt,
    )


def _torch_merge_quanto_lora(
    target: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Merge through the ordinary dequantize/addmm/requantize path."""
    dense = dequantize_qbytes_tensor(target)
    dense.addmm_(b, a, alpha=strength)
    return requantize_qbytes_tensor(dense, like=target)


def _is_qint8_layout(qt: Any) -> bool:  # noqa: ANN401
    qtype = qt.qtype
    return (
        qt._data.dtype is torch.int8
        and getattr(qtype, "bits", None) == 8
        and not getattr(qtype, "is_floating_point", True)
    )


def _is_qfloat8_layout(qt: Any) -> bool:  # noqa: ANN401
    qtype = qt.qtype
    return (
        qt._data.dtype
        in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        and getattr(qtype, "bits", None) == 8
        and getattr(qtype, "is_floating_point", False)
    )


def _has_supported_scale_layout(qt: Any) -> bool:  # noqa: ANN401
    rows, cols = qt.size()
    axis = qt.axis
    scale_shape = tuple(qt._scale.shape)
    return (
        (axis is None and qt._scale.numel() == 1)
        or (axis == 0 and scale_shape == (rows, 1))
        or (axis in (-1, 1) and scale_shape == (1, cols))
    )


def _has_triton_compatible_layout(
    qt: Any,  # noqa: ANN401
    b: torch.Tensor,
    a: torch.Tensor,
) -> bool:
    data = qt._data
    scale = qt._scale
    return (
        data.device.type == "cuda"
        and data.ndim == 2
        and tuple(data.shape) == tuple(qt.size())
        and _has_supported_scale_layout(qt)
        and scale.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and scale.dtype is b.dtype
        and b.dtype is a.dtype
        and data.device == scale.device == b.device == a.device
    )


class QuantoAdapter:
    """Adapter for ``optimum.quanto.WeightQBytesTensor``.

    Decompose-on-pin, reconstruct-on-move. Each activate creates a
    fresh ``WeightQBytesTensor`` and a fresh :class:`nn.Parameter`,
    installed via registry replacement. This breaks PyTorch optimizer
    references — quanto-quantized weights are inference-only.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        if not is_weight_qbytes_tensor(t):
            return False
        # Validate the private layout we read in clone_pin/_build_qbytes.
        # Cheap (four hasattr calls) and runs on every dispatch.
        validate_layout(t)
        return True

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple[object, ...]:
        # Composite identity: the two underlying buffers plus the quant
        # metadata. Two distinct WeightQBytesTensors with the same
        # underlying _data/_scale storage AND matching quant params are
        # the same logical tensor and dedup safely.
        qt = require_qbytes_tensor(t)
        data = qbytes_data_storage(qt)
        return (
            "quanto",
            data.device,
            data.data_ptr(),
            data.dtype,
            tuple(data.shape),
            data.stride(),
            data.storage_offset(),
            qt._scale.device,
            qt._scale.data_ptr(),
            qt._scale.dtype,
            tuple(qt._scale.shape),
            qt._scale.stride(),
            qt._scale.storage_offset(),
            qt.qtype,
            qt.axis,
            tuple(qt.size()),
            qt.stride(),
            qbytes_activation_qtype(qt),
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
        qt = require_qbytes_tensor(t)
        data_shape, data_dtype, scale_shape, scale_dtype = (
            canonical_qbytes_storage_layout(qt)
        )
        return (
            tuple(qt.shape),
            qt.dtype,
            qt.stride(),
            qt.qtype,
            qt.axis,
            qbytes_activation_qtype(qt),
            ("_data", data_shape, data_dtype),
            ("_scale", scale_shape, scale_dtype),
        )

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _QuantoPinned:
        qt = canonicalize_qbytes_tensor(t)
        # contiguous_format clone: fp8-quanto leaves some _data buffers
        # strided via internal transposes; the default preserve_format
        # would carry that through pin_memory(), breaking downstream
        # assumptions of a contiguous pinned buffer. The original quant
        # stride is captured separately and reapplied when rebuilding the
        # canonical WeightQBytesTensor wrapper.
        return _QuantoPinned(
            data=clone_to_pinned_cpu(
                qt._data,
                memory_format=torch.contiguous_format,
            ),
            scale=clone_to_pinned_cpu(
                qt._scale,
                memory_format=torch.contiguous_format,
            ),
            qtype=qt.qtype,
            axis=qt.axis,
            size=qt.size(),
            stride=qt.stride(),
            act_qt=qbytes_activation_qtype(qt),
        )

    @staticmethod
    def cpu_param(
        state: _QuantoPinned, *, requires_grad: bool = False
    ) -> nn.Parameter:
        qt = _build_qbytes(state, state.data, state.scale)
        return nn.Parameter(qt, requires_grad=requires_grad)

    @staticmethod
    def alloc_gpu(state: _QuantoPinned, device: torch.device) -> _QuantoGpu:
        return _QuantoGpu(
            data=torch.empty_like(state.data, device=device),
            scale=torch.empty_like(state.scale, device=device),
        )

    @staticmethod
    def gpu_param(
        pinned: _QuantoPinned,
        gpu_state: _QuantoGpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        # Quant metadata comes from the pinned state; only the storage
        # tensors come from the GPU side.
        qt = _build_qbytes(pinned, gpu_state.data, gpu_state.scale)
        return nn.Parameter(qt, requires_grad=requires_grad)

    @staticmethod
    def copy_to_gpu(
        src: _QuantoPinned, dst: _QuantoGpu, *, non_blocking: bool = False
    ) -> None:
        dst.data.copy_(src.data, non_blocking=non_blocking)
        dst.scale.copy_(src.scale, non_blocking=non_blocking)

    @staticmethod
    def copy_to_cpu(
        src: _QuantoGpu, dst: _QuantoPinned, *, non_blocking: bool = False
    ) -> None:
        # Symmetric D2H of both packed tensors. Quant metadata lives on
        # the pinned state already and is unaffected by GPU operations,
        # so only the int8/fp8 _data and the fp16/fp32 _scale need to
        # round-trip back to host.
        dst.data.copy_(src.data, non_blocking=non_blocking)
        dst.scale.copy_(src.scale, non_blocking=non_blocking)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        qt = require_qbytes_tensor(t)
        return qt.dtype

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        return tuple(require_qbytes_tensor(t).size())

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_qbytes_tensor(t)

    @staticmethod
    def requantize(t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        return requantize_qbytes_tensor(t, like=like)

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
    ) -> None:
        """Merge with Triton when possible, except for packed Marlin targets."""
        target_qt = require_qbytes_tensor(target)
        qt = canonicalize_qbytes_tensor(target_qt)
        triton_merge = None
        if not is_marlin_f8_qbytes_tensor(target_qt):
            if _is_qint8_layout(qt):
                triton_merge = _triton_merge_quanto_qint8_lora
            elif _is_qfloat8_layout(qt):
                triton_merge = _triton_merge_quanto_qfloat8_lora

        if triton_merge is not None and _has_triton_compatible_layout(qt, b, a):
            data = triton_merge(
                qt._data,
                qt._scale,
                qt.axis,
                b,
                a,
                strength,
            )
            merged = create_qbytes_tensor(
                qt.qtype,
                qt.axis,
                qt.size(),
                qt.stride(),
                data,
                qt._scale,
                qbytes_activation_qtype(qt),
            )
            copy_qbytes_tensor_(merged, target_qt)
            return

        merged = require_qbytes_tensor(
            _torch_merge_quanto_lora(qt, b, a, strength)
        )
        copy_qbytes_tensor_(merged, target_qt)

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        copy_qbytes_tensor_(src, target)

    @staticmethod
    def cache_bytes(state: _QuantoPinned) -> int:
        return (
            state.data.numel() * state.data.element_size()
            + state.scale.numel() * state.scale.element_size()
        )
