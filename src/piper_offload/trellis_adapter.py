"""Trellis-coded ConvRot INT8 offload.

Trellis streams stay packed in host backing, at 3 to 6 bits per weight. Each
active target owns reusable CUDA staging for the packed tensors and a ConvRot
INT8 representation. Refills DMA the packed tensors and ask Piper Kernels to
decode them into that target, so no dense weight is materialized. Decoding needs
a Triton accelerator supported by Piper Kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from ._piper_convrot_int8 import create_convrot_int8_tensor
from .piper_convrot_int8_adapter import PiperConvRotInt8Adapter
from .tensor_adapters import capture_host_tensor, transfer_
from .trellis_parameter import TrellisParameter

__all__ = ["TrellisAdapter"]


def _require_trellis_parameter(tensor: torch.Tensor) -> TrellisParameter:
    if not isinstance(tensor, TrellisParameter):
        raise TypeError(f"expected a TrellisParameter, got {type(tensor).__name__}")
    if tensor.is_meta:
        raise ValueError("trellis parameters must own physical storage")
    return tensor


@dataclass(slots=True, frozen=True)
class _TrellisHost:
    parameter: TrellisParameter
    """The captured parameter, whose tensors are the host backing."""


@dataclass(slots=True, frozen=True)
class _TrellisGpu:
    staging: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """Bitstreams, block scales and row scale, as on the host."""
    target: torch.Tensor


class TrellisAdapter:
    """Keep trellis streams on the host and refill fixed ConvRot INT8 targets."""

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        return isinstance(t, TrellisParameter)

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple[object, ...]:
        parameter = _require_trellis_parameter(t)
        return (
            "trellis",
            parameter.device,
            *(
                (
                    tensor.untyped_storage().data_ptr(),
                    tensor.storage_offset(),
                    tuple(tensor.shape),
                    tensor.stride(),
                )
                for tensor in parameter.storage_tensors()
            ),
            parameter.block_size,
            parameter.group_size,
            parameter.dtype,
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
        parameter = _require_trellis_parameter(t)
        return (
            tuple(parameter.shape),
            parameter.dtype,
            tuple(parameter.bitstreams.shape),
            parameter.block_size,
            parameter.group_size,
            parameter.act_per_tensor_scale is not None,
        )

    @staticmethod
    def capture_host(t: torch.Tensor) -> _TrellisHost:
        parameter = _require_trellis_parameter(t)
        captured = [
            capture_host_tensor(tensor, memory_format=torch.contiguous_format)
            for tensor in parameter.storage_tensors()
        ]
        return _TrellisHost(parameter.rebuild(captured))

    @staticmethod
    def storage_tensors(state: _TrellisHost) -> tuple[torch.Tensor, ...]:
        return state.parameter.storage_tensors()

    @staticmethod
    def cpu_param(state: _TrellisHost, *, requires_grad: bool = False) -> nn.Parameter:
        if requires_grad:
            raise ValueError("trellis parameters are inference-only")
        # Each binding gets its own wrapper over the shared host tensors.
        parameter = state.parameter
        return parameter.rebuild(parameter.storage_tensors())

    @staticmethod
    def alloc_gpu(state: _TrellisHost, device: torch.device) -> _TrellisGpu:
        parameter = state.parameter
        rows, features = parameter.shape
        act_scale = parameter.act_per_tensor_scale
        return _TrellisGpu(
            staging=(
                torch.empty_like(parameter.bitstreams, device=device),
                torch.empty_like(parameter.block_scales, device=device),
                torch.empty_like(parameter.scale, device=device),
            ),
            target=create_convrot_int8_tensor(
                torch.empty((rows, features), dtype=torch.int8, device=device),
                torch.empty((rows, 1), dtype=torch.float32, device=device),
                parameter.group_size,
                parameter.dtype,
                None if act_scale is None else torch.empty_like(act_scale, device=device),
            ),
        )

    @staticmethod
    def gpu_param(
        host: _TrellisHost,
        gpu_state: _TrellisGpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        del host
        if requires_grad:
            raise ValueError("trellis parameters are inference-only")
        return nn.Parameter(gpu_state.target, requires_grad=False)

    @staticmethod
    def copy_to_gpu(src: _TrellisHost, dst: _TrellisGpu, *, non_blocking: bool = False) -> None:
        parameter = src.parameter
        host_tensors = (parameter.bitstreams, parameter.block_scales, parameter.scale)
        for staging, host in zip(dst.staging, host_tensors, strict=True):
            transfer_(staging, host, non_blocking=non_blocking)
        # ConvRotInt8Tensor is an optional import, typed Any at its boundary module.
        target = cast(Any, dst.target)
        if parameter.act_per_tensor_scale is not None:
            transfer_(
                target.act_per_tensor_scale,
                parameter.act_per_tensor_scale,
                non_blocking=non_blocking,
            )
        target.copy_from_trellis_(*dst.staging)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        return _require_trellis_parameter(t).dtype

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        return tuple(_require_trellis_parameter(t).shape)

    @staticmethod
    def validate_permanent_update() -> None:
        raise ValueError(
            "Permanent updates to trellis-coded parameters are unsupported; "
            "use activation-time adapter merge instead."
        )

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        """Delegate active-value dequantization to ConvRot INT8."""
        return PiperConvRotInt8Adapter.dequantize(t)

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
        """Delegate an activation-time LoRA update to ConvRot INT8."""
        PiperConvRotInt8Adapter.merge_lora_(
            target, b, a, alpha, beta=beta, rounding_seed=rounding_seed
        )

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        """Delegate an activation-time dense update to ConvRot INT8."""
        PiperConvRotInt8Adapter.merge_dense_(
            target, update, alpha, beta=beta, rounding_seed=rounding_seed
        )

    @staticmethod
    def cache_bytes(state: _TrellisHost) -> int:
        return sum(tensor.nbytes for tensor in state.parameter.storage_tensors())
