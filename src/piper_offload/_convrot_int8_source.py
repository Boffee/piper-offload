"""Shared behavior of packed sources that activate as ConvRot INT8 targets.

GGUF and trellis-coded sources keep their packed bytes in host backing, and each
refill stages them into a transient stream-ordered scratch buffer and decodes
directly into a reusable ConvRot INT8 target without retaining permanent GPU
staging allocations. Once active, the parameter is an ordinary
``ConvRotInt8Tensor``, so LoRA and dense updates delegate to
:class:`PiperConvRotInt8Adapter`. The packed source is inference-only and cannot
be updated permanently. Each subclass owns its capture, identity, and refill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from .piper_convrot_int8_adapter import PiperConvRotInt8Adapter
from .tensor_adapters import transfer_

_SCRATCH_ALIGNMENT = 256
_SCRATCH_BUCKET_MIN_BYTES = 2 * 1024 * 1024


@dataclass(slots=True, frozen=True)
class ConvRotInt8GpuTarget:
    """Reusable active ConvRot INT8 GPU target without permanent decode staging."""

    target: torch.Tensor


def _aligned_size(nbytes: int) -> int:
    return ((nbytes + _SCRATCH_ALIGNMENT - 1) // _SCRATCH_ALIGNMENT) * _SCRATCH_ALIGNMENT


def _bucketed_scratch_bytes(total_bytes: int) -> int:
    """Round CUDA scratch sizes >= 2 MiB to < 12.5% geometric steps to curb allocator fragmentation."""
    if total_bytes < _SCRATCH_BUCKET_MIN_BYTES:
        return total_bytes
    bit = (total_bytes - 1).bit_length()
    step = 1 << max(20, bit - 4)
    return ((total_bytes + step - 1) // step) * step


def stage_packed_tensors(
    host_tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Copy packed host tensors into a transient stream-ordered device scratch buffer.

    The backing ``scratch`` buffer is allocated on the active stream of ``device``
    (bucketed on CUDA via :func:`_bucketed_scratch_bytes`), all returned views
    alias it at 256-byte-aligned offsets, and every copy routes through
    :func:`transfer_` on that same stream. Callers must enqueue their decode
    kernel on that same stream before dropping the returned views so
    ``CUDACachingAllocator`` orders buffer reuse after both H2D DMA and decode
    finish.
    """
    offsets: list[int] = []
    pos = 0
    for host in host_tensors:
        if not host.is_contiguous():
            raise ValueError("packed decode staging requires contiguous host tensors")
        offsets.append(pos)
        pos += _aligned_size(host.nbytes)

    alloc_bytes = _bucketed_scratch_bytes(pos) if device.type == "cuda" else pos
    scratch = torch.empty((alloc_bytes,), dtype=torch.uint8, device=device)
    staged: list[torch.Tensor] = []
    for host, offset in zip(host_tensors, offsets, strict=True):
        view = scratch[offset : offset + host.nbytes].view(host.dtype).view(host.shape)
        transfer_(view, host, non_blocking=non_blocking)
        staged.append(view)
    return tuple(staged)


class ConvRotInt8SourceAdapter:
    """Base adapter for packed sources whose active representation is ConvRot INT8."""

    _SOURCE_NAME: ClassVar[str]
    """The source format's name in error messages, such as ``"packed GGUF"``."""

    @classmethod
    def gpu_param(
        cls,
        host: object,
        gpu_state: ConvRotInt8GpuTarget,
    ) -> nn.Parameter:
        del cls, host
        return nn.Parameter(gpu_state.target, requires_grad=False)

    @classmethod
    def validate_permanent_update(cls) -> None:
        raise ValueError(
            f"Permanent updates to {cls._SOURCE_NAME} parameters are unsupported; "
            "use activation-time adapter merge instead."
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


def convrot_int8_target_bytes(
    rows: int,
    features: int,
    act_scale: torch.Tensor | None = None,
) -> int:
    """Unique GPU target bytes allocated for a ConvRot INT8 target."""
    return rows * features + rows * 4 + (0 if act_scale is None else act_scale.nbytes)


__all__ = [
    "ConvRotInt8GpuTarget",
    "ConvRotInt8SourceAdapter",
    "convrot_int8_target_bytes",
    "stage_packed_tensors",
]
