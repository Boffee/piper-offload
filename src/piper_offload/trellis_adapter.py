"""Trellis-coded ConvRot INT8 offload.

Trellis streams stay packed in host backing, at 2 to 6 bits per weight. Each
active target owns a reusable ConvRot INT8 representation without retaining
permanent decode staging buffers. Refills stage the packed tensors into
transient stream-ordered scratch memory and ask Piper Kernels to decode them
directly into that target, so no dense weight is materialized. Decoding needs
a Triton accelerator supported by Piper Kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from torch import nn

from ._convrot_int8_source import (
    ConvRotInt8GpuTarget,
    ConvRotInt8SourceAdapter,
    convrot_int8_target_bytes,
    stage_packed_tensors,
)
from .tensor_adapters import (
    capture_host_tensor,
    is_plain_meta_placeholder,
    matches_layout_signature,
    optional_tensor_id,
    transfer_,
)
from .trellis_parameter import TrellisParameter, validate_row_widths_cpu

__all__ = ["TrellisAdapter"]


def _require_trellis_parameter(tensor: torch.Tensor) -> TrellisParameter:
    if not isinstance(tensor, TrellisParameter):
        raise TypeError(f"expected a TrellisParameter, got {type(tensor).__name__}")
    if tensor.is_meta or any(part.is_meta for part in tensor.storage_tensors()):
        raise ValueError("trellis parameters must own physical storage")
    return tensor


@dataclass(slots=True, frozen=True)
class _TrellisHost:
    parameter: TrellisParameter
    """The captured parameter, whose tensors are the host backing."""


class TrellisAdapter(ConvRotInt8SourceAdapter):
    """Keep trellis streams on the host and refill fixed ConvRot INT8 targets."""

    _SOURCE_NAME = "trellis-coded"

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        return isinstance(t, TrellisParameter)

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple[object, ...]:
        parameter = _require_trellis_parameter(t)
        return (
            "trellis",
            parameter.device,
            *(optional_tensor_id(tensor) for tensor in parameter.storage_tensors()),
            parameter.block_size,
            parameter.group_size,
            parameter.dtype,
            parameter.row_widths is not None,
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
        parameter = _require_trellis_parameter(t)
        return (
            tuple(parameter.shape),
            parameter.dtype,
            parameter.group_size,
            parameter.act_per_tensor_scale is not None,
        )

    @classmethod
    def accepts_bind_target(
        cls,
        state: _TrellisHost,
        target: torch.Tensor,
        *,
        layout_signature: object,
    ) -> bool:
        return is_plain_meta_placeholder(target, tuple(state.parameter.shape)) or (
            not target.is_meta and matches_layout_signature(cls, layout_signature, target)
        )

    @staticmethod
    def capture_host(t: torch.Tensor) -> _TrellisHost:
        parameter = _require_trellis_parameter(t)
        captured_widths: torch.Tensor | None = None
        if parameter.row_widths is not None:
            captured_widths = capture_host_tensor(
                parameter.row_widths, memory_format=torch.contiguous_format
            )
            if captured_widths.device.type == "cpu":
                validate_row_widths_cpu(
                    captured_widths,
                    rows=parameter.shape[0],
                    blocks=parameter.block_scales.shape[1],
                    block_size=parameter.block_size,
                    bitstream_bytes=parameter.bitstreams.numel(),
                )
        bitstreams = capture_host_tensor(
            parameter.bitstreams, memory_format=torch.contiguous_format
        )
        block_scales = capture_host_tensor(
            parameter.block_scales, memory_format=torch.contiguous_format
        )
        scale = capture_host_tensor(parameter.scale, memory_format=torch.contiguous_format)
        act_scale = (
            None
            if parameter.act_per_tensor_scale is None
            else capture_host_tensor(
                parameter.act_per_tensor_scale, memory_format=torch.contiguous_format
            )
        )
        return _TrellisHost(
            TrellisParameter._wrap_validated(
                bitstreams,
                block_scales,
                scale,
                block_size=parameter.block_size,
                group_size=parameter.group_size,
                dtype=parameter.dtype,
                act_per_tensor_scale=act_scale,
                row_widths=captured_widths,
            )
        )

    @staticmethod
    def storage_tensors(state: _TrellisHost) -> tuple[torch.Tensor, ...]:
        return state.parameter.storage_tensors()

    @staticmethod
    def cpu_param(state: _TrellisHost) -> nn.Parameter:
        # Each binding gets its own wrapper over the shared host tensors.
        parameter = state.parameter
        return parameter._rewrap_validated(parameter.storage_tensors())

    @staticmethod
    def alloc_gpu(state: _TrellisHost, device: torch.device) -> ConvRotInt8GpuTarget:
        parameter = state.parameter
        rows, features = parameter.shape
        act_scale = parameter.act_per_tensor_scale
        return ConvRotInt8GpuTarget(
            target=ConvRotInt8Tensor(
                torch.empty((rows, features), dtype=torch.int8, device=device),
                torch.empty((rows, 1), dtype=torch.float32, device=device),
                parameter.group_size,
                parameter.dtype,
                None if act_scale is None else torch.empty_like(act_scale, device=device),
            ),
        )

    @staticmethod
    def copy_to_gpu(
        src: _TrellisHost,
        dst: ConvRotInt8GpuTarget,
        *,
        non_blocking: bool = False,
    ) -> None:
        parameter = src.parameter
        host_tensors: tuple[torch.Tensor, ...] = (
            parameter.bitstreams,
            parameter.block_scales,
            parameter.scale,
        )
        if parameter.row_widths is not None:
            host_tensors = (*host_tensors, parameter.row_widths)
        target = cast(Any, dst.target)
        staged = stage_packed_tensors(
            host_tensors,
            dst.target.device,
            non_blocking=non_blocking,
        )
        if parameter.act_per_tensor_scale is not None:
            transfer_(
                target.act_per_tensor_scale,
                parameter.act_per_tensor_scale,
                non_blocking=non_blocking,
            )
        if parameter.row_widths is None:
            target.copy_from_trellis_(*staged)
        else:
            target.copy_from_trellis_(*staged[:3], row_widths=staged[3])

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        return _require_trellis_parameter(t).dtype

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        return tuple(_require_trellis_parameter(t).shape)

    @staticmethod
    def cache_bytes(state: _TrellisHost) -> int:
        return sum(tensor.nbytes for tensor in state.parameter.storage_tensors())

    @staticmethod
    def gpu_target_bytes(state: _TrellisHost) -> int:
        rows, features = state.parameter.shape
        return convrot_int8_target_bytes(
            rows,
            features,
            state.parameter.act_per_tensor_scale,
        )
