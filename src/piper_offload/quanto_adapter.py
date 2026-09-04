"""Quanto :class:`WeightQBytesTensor` adapter.

Quanto-quantized weights are subclassed tensors that wrap multiple
internal tensors (``_data``, ``_scale``) plus quant metadata
(``qtype``, ``axis``, ``activation_qtype``). The wrapper does not
support ``p.data = ...`` storage swap — its quant state is part of the
Parameter's wrapped object, not its bytes. So this adapter:

- Decomposes ``WeightQBytesTensor`` into ``_data`` and ``_scale``,
  captures each separately.
- Reconstructs a fresh ``WeightQBytesTensor`` (and thus a fresh
  :class:`nn.Parameter`) on each activate via registry replacement.
  PyTorch optimizers keyed by the user's pre-wrap Parameter id are
  orphaned across cycles — quanto-quantized weights are inference-only.

Device-optimized subclasses need one additional boundary rule.
optimum-quanto 0.2.7's ``MarlinF8QBytesTensor`` stores packed INT32 data
and permuted scales, and its wrapper-level ``copy_`` unpacks instead of
updating the packed leaf. The adapter therefore canonicalizes Marlin
inputs to the kernel-agnostic ``WeightQBytesTensor`` representation when
capture and keeps that raw representation throughout streaming. Direct
updates to an existing Marlin wrapper are repacked into its physical
``_data._data`` buffer so wrapper, workspace, and storage identity remain
stable.

Reaches into quanto's private attributes (``_data``, ``_scale``,
``qtype``, ``axis``, ``activation_qtype``). Host to the
``WeightQBytesTensor`` layout in optimum-quanto as of the version this
repo depends on. If quanto refactors the wrapper class, the
:class:`QuantoAdapter` will fail with a clear validation error at
:meth:`matches` (validates the expected attributes exist on first
match).

CUDA qint8 and qfloat8 merges use a Triton merge/reduction/requantize pipeline
when available. Both it and the generic dequantize/GEMM/requantize fallback
recompute Quanto's data-dependent absmax weight scale from the merged values.

Selected by :mod:`tensor_adapter_registry`. Importing fails silently if
optimum-quanto is not installed — quanto support is optional.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from ._dense_merge import merge_dense_requantize_
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
from .tensor_adapters import clone_to_host_cpu

try:
    from ._triton_quanto_lora import (
        merge_quanto_qfloat8_dense as _triton_merge_quanto_qfloat8_dense,
    )
    from ._triton_quanto_lora import (
        merge_quanto_qfloat8_lora as _triton_merge_quanto_qfloat8_lora,
    )
    from ._triton_quanto_lora import (
        merge_quanto_qint8_dense as _triton_merge_quanto_qint8_dense,
    )
    from ._triton_quanto_lora import (
        merge_quanto_qint8_lora as _triton_merge_quanto_qint8_lora,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_merge_quanto_qfloat8_dense = None
    _triton_merge_quanto_qfloat8_lora = None
    _triton_merge_quanto_qint8_dense = None
    _triton_merge_quanto_qint8_lora = None


@dataclass(slots=True)
class _QuantoHost:
    """CPU state for a quanto tensor: two host tensors
    plus the quant metadata needed to reconstruct the wrapper."""

    data: torch.Tensor   # host int8/fp8
    scale: torch.Tensor  # host fp16/fp32
    qtype: object
    axis: int | None
    size: torch.Size
    stride: tuple[int, ...]
    act_qt: object | None


@dataclass(slots=True)
class _QuantoGpu:
    """GPU state for a quanto tensor: the two GPU tensors. Quant
    metadata lives in the originating :class:`_QuantoHost`; only
    storage moves to GPU."""

    data: torch.Tensor
    scale: torch.Tensor


def _build_qbytes(
    state: _QuantoHost, data: torch.Tensor, scale: torch.Tensor
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
    *,
    rounding_seed: int | None = None,
) -> torch.Tensor:
    """Merge through the ordinary dequantize/addmm/requantize path."""
    dense = dequantize_qbytes_tensor(target)
    dense.addmm_(b, a, alpha=strength)
    return requantize_qbytes_tensor(
        dense,
        like=target,
        rounding_seed=rounding_seed,
    )


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
        and data.numel() != 0
        and tuple(data.shape) == tuple(qt.size())
        and _has_supported_scale_layout(qt)
        and scale.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and scale.dtype is b.dtype
        and b.dtype is a.dtype
        and data.device == scale.device == b.device == a.device
    )


def _has_triton_compatible_dense_layout(
    qt: Any,  # noqa: ANN401
    update: torch.Tensor,
) -> bool:
    data = qt._data
    scale = qt._scale
    return (
        data.device.type == "cuda"
        and data.ndim == 2
        and data.numel() != 0
        and tuple(data.shape) == tuple(qt.size()) == tuple(update.shape)
        and _has_supported_scale_layout(qt)
        and scale.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and scale.dtype is update.dtype
        and data.device == scale.device == update.device
    )


class QuantoAdapter:
    """Adapter for ``optimum.quanto.WeightQBytesTensor``.

    Decompose-on-capture, reconstruct-on-move. Each activate creates a
    fresh ``WeightQBytesTensor`` and a fresh :class:`nn.Parameter`,
    installed via registry replacement. This breaks PyTorch optimizer
    references — quanto-quantized weights are inference-only.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        if not is_weight_qbytes_tensor(t):
            return False
        # Validate the private layout we read in capture_host/_build_qbytes.
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
    def capture_host(
        t: torch.Tensor,
    ) -> _QuantoHost:
        qt = canonicalize_qbytes_tensor(t)
        # fp8-quanto leaves some _data buffers strided via internal transposes;
        # capture contiguous bytes for the shared transfer path. The original quant
        # stride is captured separately and reapplied when rebuilding the
        # canonical WeightQBytesTensor wrapper.
        return _QuantoHost(
            data=clone_to_host_cpu(
                qt._data,
                memory_format=torch.contiguous_format,
            ),
            scale=clone_to_host_cpu(
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
        state: _QuantoHost, *, requires_grad: bool = False
    ) -> nn.Parameter:
        qt = _build_qbytes(state, state.data, state.scale)
        return nn.Parameter(qt, requires_grad=requires_grad)

    @staticmethod
    def alloc_gpu(state: _QuantoHost, device: torch.device) -> _QuantoGpu:
        return _QuantoGpu(
            data=torch.empty_like(state.data, device=device),
            scale=torch.empty_like(state.scale, device=device),
        )

    @staticmethod
    def gpu_param(
        host: _QuantoHost,
        gpu_state: _QuantoGpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        # Quant metadata comes from the host state; only the storage
        # tensors come from the GPU side.
        qt = _build_qbytes(host, gpu_state.data, gpu_state.scale)
        return nn.Parameter(qt, requires_grad=requires_grad)

    @staticmethod
    def copy_to_gpu(
        src: _QuantoHost, dst: _QuantoGpu, *, non_blocking: bool = False
    ) -> None:
        dst.data.copy_(src.data, non_blocking=non_blocking)
        dst.scale.copy_(src.scale, non_blocking=non_blocking)

    @staticmethod
    def copy_to_cpu(
        src: _QuantoGpu, dst: _QuantoHost, *, non_blocking: bool = False
    ) -> None:
        # Symmetric D2H of both packed tensors. Quant metadata lives on
        # the host state already and is unaffected by GPU operations,
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
    def requantize(
        t: torch.Tensor,
        *,
        like: torch.Tensor,
        rounding_seed: int | None = None,
    ) -> torch.Tensor:
        return requantize_qbytes_tensor(t, like=like, rounding_seed=rounding_seed)

    @staticmethod
    def validate_lora_merge(
        target: torch.Tensor,
        _b: torch.Tensor,
        a: torch.Tensor,
        _strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Validate canonicalization and absmax-requantization layout."""
        del rounding_seed
        QuantoAdapter._validate_merge_target(target, kind="lora")
        if a.shape[0] == 0:
            raise ValueError("Quanto LoRA merge requires a positive LoRA rank.")

    @staticmethod
    def validate_dense_merge_target(
        target: torch.Tensor,
        *,
        rounding_seed: int | None = None,
    ) -> bool:
        del rounding_seed
        QuantoAdapter._validate_merge_target(target, kind="dense")
        return False

    @staticmethod
    def _validate_merge_target(
        target: torch.Tensor,
        *,
        kind: Literal["lora", "dense"],
    ) -> Any:  # noqa: ANN401
        """Validate canonicalization and absmax-requantization layout."""
        label = "LoRA" if kind == "lora" else "dense"
        qt = canonicalize_qbytes_tensor(target)
        if qt._data.ndim != 2 or tuple(qt._data.shape) != tuple(qt.size()):
            raise ValueError(
                f"Quanto {label} merge requires a rank-two weight whose qbytes "
                "storage matches its logical shape."
            )
        if getattr(qt.qtype, "bits", None) != 8:
            raise ValueError(f"Quanto qbytes {label} merge requires an 8-bit qtype.")
        qmax = getattr(qt.qtype, "qmax", None)
        if not isinstance(qmax, (int, float)) or qmax <= 0:
            raise ValueError(
                f"Quanto qbytes {label} merge requires a qtype with a positive qmax."
            )
        if getattr(qt.qtype, "dtype", None) is not qt._data.dtype:
            raise ValueError("Quanto qbytes storage dtype does not match its qtype metadata.")
        if not qt._scale.dtype.is_floating_point:
            raise ValueError(f"Quanto qbytes {label} merge requires floating-point scales.")
        if qt._data.device != qt._scale.device:
            raise ValueError("Quanto qbytes data and scales must be on the same device.")
        if not _has_supported_scale_layout(qt):
            raise ValueError(
                f"Quanto {label} merge expects a scalar scale, shape (rows, 1) "
                "for axis 0, or shape (1, columns) for the last axis."
            )
        return qt

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a validated update, except via Triton for Marlin targets."""
        target_qt = require_qbytes_tensor(target)
        qt = canonicalize_qbytes_tensor(target_qt)
        triton_merge = None
        if not is_marlin_f8_qbytes_tensor(target_qt):
            if _is_qint8_layout(qt):
                triton_merge = _triton_merge_quanto_qint8_lora
            elif _is_qfloat8_layout(qt):
                triton_merge = _triton_merge_quanto_qfloat8_lora

        if triton_merge is not None and _has_triton_compatible_layout(qt, b, a):
            data, scale = triton_merge(
                qt._data,
                qt._scale,
                qt.axis,
                b,
                a,
                strength,
                rounding_seed=rounding_seed,
            )
            merged = create_qbytes_tensor(
                qt.qtype,
                qt.axis,
                qt.size(),
                qt.stride(),
                data,
                scale,
                qbytes_activation_qtype(qt),
            )
            copy_qbytes_tensor_(merged, target_qt)
            return

        merged = _torch_merge_quanto_lora(
            qt,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )
        merged = require_qbytes_tensor(merged)
        copy_qbytes_tensor_(merged, target_qt)

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a full-rank update, preferring Triton for canonical storage."""
        target_qt = require_qbytes_tensor(target)
        qt = canonicalize_qbytes_tensor(target_qt)
        triton_merge = None
        if not is_marlin_f8_qbytes_tensor(target_qt):
            if _is_qint8_layout(qt):
                triton_merge = _triton_merge_quanto_qint8_dense
            elif _is_qfloat8_layout(qt):
                triton_merge = _triton_merge_quanto_qfloat8_dense

        if triton_merge is not None and _has_triton_compatible_dense_layout(qt, update):
            data, scale = triton_merge(
                qt._data,
                qt._scale,
                qt.axis,
                update,
                strength,
                rounding_seed=rounding_seed,
            )
            merged = create_qbytes_tensor(
                qt.qtype,
                qt.axis,
                qt.size(),
                qt.stride(),
                data,
                scale,
                qbytes_activation_qtype(qt),
            )
            copy_qbytes_tensor_(merged, target_qt)
            return
        merge_dense_requantize_(
            QuantoAdapter,
            target,
            update,
            strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        copy_qbytes_tensor_(src, target)

    @staticmethod
    def cache_bytes(state: _QuantoHost) -> int:
        return (
            state.data.numel() * state.data.element_size()
            + state.scale.numel() * state.scale.element_size()
        )
