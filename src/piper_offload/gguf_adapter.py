"""Direct GGUF-to-ConvRot INT8 offload.

GGUF bytes remain compact in host backing. Each active target owns a reusable
BF16 ConvRot INT8 representation without retaining permanent GPU decode staging
buffers. Refills stage the packed bytes into transient stream-ordered scratch
memory and ask Piper Kernels to decode, rotate, and requantize directly into
that target; no dense weight is materialized.

Offload's ``GgufParameter`` and externally supplied Diffusers ``GGUFParameter``
objects share the ``quant_type``, ``quant_shape``, and ``as_tensor()`` interface.
No Diffusers dependency is needed for reading or activating GGUF checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from ._convrot_int8_source import (
    ConvRotInt8GpuTarget,
    ConvRotInt8SourceAdapter,
    convrot_int8_target_bytes,
    stage_packed_tensors,
)
from ._piper_convrot_int8 import create_convrot_int8_tensor
from .gguf_parameter import GGUF_COMPUTE_DTYPE
from .tensor_adapters import (
    capture_host_tensor,
    is_plain_meta_placeholder,
    matches_layout_signature,
)

__all__ = ["GgufAdapter"]

_CONVROT_GROUP_SIZES = (256, 64, 16)


def _is_gguf_parameter(tensor: torch.Tensor) -> bool:
    """Recognize native and external GGUF parameters without importing Diffusers."""
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
    """Choose the largest supported rotation group dividing ``features``."""
    for group_size in _CONVROT_GROUP_SIZES:
        if features % group_size == 0:
            return group_size
    raise ValueError(
        "GGUF ConvRot INT8 requires in_features divisible by 16; "
        f"got {features}"
    )


@dataclass(slots=True, frozen=True)
class _GgufHost:
    data: torch.Tensor
    quant_type: int
    logical_shape: tuple[int, int]
    group_size: int
    parameter_type: type[nn.Parameter]


class GgufAdapter(ConvRotInt8SourceAdapter):
    """Keep GGUF bytes on the host and refill fixed ConvRot INT8 targets."""

    _SOURCE_NAME = "packed GGUF"

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
            _logical_shape(t),
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
        _source_data(t)
        _quant_type(t)
        logical_shape = _logical_shape(t)
        return (
            logical_shape,
            GGUF_COMPUTE_DTYPE,
            _convrot_group_size(logical_shape[1]),
        )

    @classmethod
    def accepts_bind_target(
        cls,
        state: _GgufHost,
        target: torch.Tensor,
        *,
        layout_signature: object,
    ) -> bool:
        return is_plain_meta_placeholder(target, state.logical_shape) or (
            not target.is_meta and matches_layout_signature(cls, layout_signature, target)
        )

    @staticmethod
    def capture_host(
        t: torch.Tensor,
    ) -> _GgufHost:
        source = _source_data(t)
        logical_shape = _logical_shape(t)
        data = capture_host_tensor(source, memory_format=torch.contiguous_format)
        quant_type = _quant_type(t)
        return _GgufHost(
            data=data,
            quant_type=quant_type,
            logical_shape=logical_shape,
            group_size=_convrot_group_size(logical_shape[1]),
            parameter_type=cast(type[nn.Parameter], type(t)),
        )

    @staticmethod
    def storage_tensors(state: _GgufHost) -> tuple[torch.Tensor, ...]:
        return (state.data,)

    @staticmethod
    def cpu_param(state: _GgufHost) -> nn.Parameter:
        parameter = cast(Any, state.parameter_type)(
            state.data,
            requires_grad=False,
            quant_type=state.quant_type,
        )
        if not isinstance(parameter, nn.Parameter):
            raise TypeError("GGUF parameter constructor must return nn.Parameter")
        return parameter

    @staticmethod
    def alloc_gpu(state: _GgufHost, device: torch.device) -> ConvRotInt8GpuTarget:
        rows, features = state.logical_shape
        return ConvRotInt8GpuTarget(
            target=create_convrot_int8_tensor(
                torch.empty((rows, features), dtype=torch.int8, device=device),
                torch.empty((rows, 1), dtype=torch.float32, device=device),
                state.group_size,
                GGUF_COMPUTE_DTYPE,
            ),
        )

    @staticmethod
    def copy_to_gpu(
        src: _GgufHost,
        dst: ConvRotInt8GpuTarget,
        *,
        non_blocking: bool = False,
    ) -> None:
        (staged_data,) = stage_packed_tensors(
            (src.data,),
            dst.target.device,
            non_blocking=non_blocking,
        )
        cast(Any, dst.target).copy_from_gguf_(
            staged_data,
            quant_type=src.quant_type,
        )

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        _source_data(t)
        return GGUF_COMPUTE_DTYPE

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        _source_data(t)
        return _logical_shape(t)

    @staticmethod
    def cache_bytes(state: _GgufHost) -> int:
        return state.data.nbytes

    @staticmethod
    def gpu_target_bytes(state: _GgufHost) -> int:
        rows, features = state.logical_shape
        return convrot_int8_target_bytes(rows, features)
