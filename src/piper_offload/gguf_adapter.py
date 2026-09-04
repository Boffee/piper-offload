"""Direct GGUF-to-ConvRot INT8 offload.

GGUF bytes remain compact in host backing. Each active target owns a reusable
CUDA staging buffer and a BF16 ConvRot INT8 representation. Refills DMA the
packed bytes and ask Piper Kernels to decode, rotate, and requantize directly
into that target; no dense weight is materialized.

Diffusers ``GGUFParameter`` objects are consumed through their existing
``quant_type`` and ``as_tensor()`` interface without taking a Diffusers
dependency or introducing another parameter wrapper.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from ._piper_convrot_int8 import create_convrot_int8_tensor
from .piper_convrot_int8_adapter import PiperConvRotInt8Adapter
from .tensor_adapters import adopt_cpu_storage, clone_to_pinned_cpu

__all__ = ["GgufAdapter"]

_CONVROT_GROUP_SIZES = (256, 64, 16)
_LOGICAL_DTYPE = torch.bfloat16


def _is_gguf_parameter(tensor: torch.Tensor) -> bool:
    """Recognize Diffusers' parameter contract without importing Diffusers."""
    return (
        isinstance(tensor, nn.Parameter)
        and type(tensor) is not nn.Parameter
        and hasattr(tensor, "quant_type")
        and hasattr(tensor, "quant_shape")
        and callable(getattr(tensor, "as_tensor", None))
    )


def _source_data(tensor: torch.Tensor) -> torch.Tensor:
    if not _is_gguf_parameter(tensor):
        raise TypeError(f"expected a GGUF parameter, got {type(tensor).__name__}")
    source = tensor.as_tensor()  # type: ignore[attr-defined]
    if not isinstance(source, torch.Tensor):
        raise TypeError("GGUFParameter.as_tensor() must return a torch.Tensor")
    if type(source) is not torch.Tensor:
        source = source.as_subclass(torch.Tensor)

    if source.is_meta:
        raise ValueError("GGUF parameters must own physical storage")
    if source.layout is not torch.strided:
        raise ValueError(f"GGUF parameters require strided storage, got {source.layout}")
    if not source.is_contiguous():
        raise ValueError("GGUF packed storage must be contiguous")
    return source.detach().view(torch.uint8)


def _quant_type(tensor: torch.Tensor) -> int:
    value = getattr(tensor, "quant_type", None)
    if value is None:
        raise TypeError("GGUF parameters must expose quant_type")
    if isinstance(value, bool):
        raise TypeError("GGUF quant_type must be an integer, not bool")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"GGUF quant_type must be integer-like, got {value!r}") from error


def _logical_shape(tensor: torch.Tensor) -> tuple[int, int]:
    shape = tuple(getattr(tensor, "quant_shape", ()))
    if len(shape) != 2:
        raise ValueError(
            f"GGUF ConvRot sources must represent a matrix, got logical shape {shape}"
        )
    return cast(tuple[int, int], shape)


def _convrot_group_size(features: int) -> int:
    for group_size in _CONVROT_GROUP_SIZES:
        if features % group_size == 0:
            return group_size
    raise ValueError(
        "GGUF ConvRot INT8 requires in_features divisible by 16; "
        f"got {features}"
    )


@dataclass(slots=True, frozen=True)
class _GgufPinned:
    data: torch.Tensor
    quant_type: int
    logical_shape: tuple[int, int]
    group_size: int
    parameter_type: type[nn.Parameter]


@dataclass(slots=True, frozen=True)
class _GgufGpu:
    staging: torch.Tensor
    target: torch.Tensor


class GgufAdapter:
    """Keep GGUF bytes on the host and refill fixed ConvRot INT8 targets."""

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        return _is_gguf_parameter(t)

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple[object, ...]:
        data = _source_data(t)
        return (
            "gguf",
            data.device,
            data.untyped_storage().data_ptr(),
            data.storage_offset(),
            tuple(data.shape),
            data.stride(),
            _quant_type(t),
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
        data = _source_data(t)
        quant_type = _quant_type(t)
        logical_shape = _logical_shape(t)
        return (
            logical_shape,
            _LOGICAL_DTYPE,
            tuple(data.shape),
            data.stride(),
            quant_type,
            _convrot_group_size(logical_shape[1]),
        )

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _GgufPinned:
        return GgufAdapter._host_state(t, clone_to_pinned_cpu)

    @staticmethod
    def adopt_host(t: torch.Tensor) -> _GgufPinned:
        return GgufAdapter._host_state(t, adopt_cpu_storage)

    @staticmethod
    def _host_state(
        tensor: torch.Tensor,
        capture: Callable[..., torch.Tensor],
    ) -> _GgufPinned:
        source = _source_data(tensor)
        logical_shape = _logical_shape(tensor)
        data = capture(source, memory_format=torch.contiguous_format)
        quant_type = _quant_type(tensor)
        return _GgufPinned(
            data=data,
            quant_type=quant_type,
            logical_shape=logical_shape,
            group_size=_convrot_group_size(logical_shape[1]),
            parameter_type=cast(type[nn.Parameter], type(tensor)),
        )

    @staticmethod
    def cpu_param(
        state: _GgufPinned,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        if requires_grad:
            raise ValueError("GGUF parameters are inference-only")
        parameter = cast(Any, state.parameter_type)(
            state.data,
            requires_grad=False,
            quant_type=state.quant_type,
        )
        if not isinstance(parameter, nn.Parameter):
            raise TypeError("GGUF parameter constructor must return nn.Parameter")
        return parameter

    @staticmethod
    def alloc_gpu(state: _GgufPinned, device: torch.device) -> _GgufGpu:
        rows, features = state.logical_shape
        return _GgufGpu(
            staging=torch.empty_like(state.data, device=device),
            target=create_convrot_int8_tensor(
                torch.empty((rows, features), dtype=torch.int8, device=device),
                torch.empty((rows, 1), dtype=torch.float32, device=device),
                state.group_size,
                _LOGICAL_DTYPE,
            ),
        )

    @staticmethod
    def gpu_param(
        pinned: _GgufPinned,
        gpu_state: _GgufGpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        del pinned
        if requires_grad:
            raise ValueError("GGUF parameters are inference-only")
        return nn.Parameter(gpu_state.target, requires_grad=False)

    @staticmethod
    def copy_to_gpu(
        src: _GgufPinned,
        dst: _GgufGpu,
        *,
        non_blocking: bool = False,
    ) -> None:
        dst.staging.copy_(src.data, non_blocking=non_blocking)
        cast(Any, dst.target).copy_from_gguf_(
            dst.staging,
            quant_type=src.quant_type,
        )

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        _source_data(t)
        return _LOGICAL_DTYPE

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        _source_data(t)
        return _logical_shape(t)

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        """Delegate active-value dequantization to ConvRot INT8."""
        return PiperConvRotInt8Adapter.dequantize(t)

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Delegate an activation-time LoRA update to ConvRot INT8."""
        PiperConvRotInt8Adapter.merge_lora_(
            target,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Delegate an activation-time dense update to ConvRot INT8."""
        PiperConvRotInt8Adapter.merge_dense_(
            target,
            update,
            strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def cache_bytes(state: _GgufPinned) -> int:
        return state.data.nbytes
