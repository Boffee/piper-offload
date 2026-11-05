"""Trellis-coded ConvRot INT8 source parameters with logical shape and compute dtype.

A trellis-coded weight stores a ConvRot INT8 matrix in 3 to 6 bits per weight
(Piper Kernels' ``piper_kernels.trellis`` format). ``TrellisParameter`` keeps
those packed tensors on the host and presents the logical ``[out, in]`` weight;
Offload activation decodes it into ConvRot INT8 (``TrellisAdapter``). Loaders
split or regroup checkpoint storage before constructing it: the parameter
supports only detach, alias, and clone, which share or copy all of its tensors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn


class TrellisParameter(nn.Parameter):
    """A logical parameter backed by trellis-coded storage, without decoding it.

    For a logical ``[out, in]`` weight in blocks of ``block_size`` weights,
    ``bitstreams`` is uint8 ``[out, in / block_size, bytes]``, ``block_scales`` is
    float8_e4m3fn ``[out, in / block_size]``, and ``scale`` is the float32 row scale
    ``[out, 1]``. The streams hold the weight rotated for ConvRot ``group_size``.
    ``dtype`` is the logical dtype of the decoded ConvRot INT8 weight, and
    ``act_per_tensor_scale`` its optional static activation scale. Use
    ``ModelOffloader`` to activate the parameter before executing its module.
    """

    bitstreams: torch.Tensor
    block_scales: torch.Tensor
    scale: torch.Tensor
    act_per_tensor_scale: torch.Tensor | None
    block_size: int
    group_size: int
    # PyTorch's C sentinel disables this hook; its stub is not a classmethod.
    __torch_function__ = torch._C._disabled_torch_function_impl  # pyright: ignore[reportAssignmentType]

    @staticmethod
    def __new__(
        cls: type[TrellisParameter],
        bitstreams: torch.Tensor,
        block_scales: torch.Tensor,
        scale: torch.Tensor,
        *,
        block_size: int,
        group_size: int,
        dtype: torch.dtype = torch.bfloat16,
        act_per_tensor_scale: torch.Tensor | None = None,
    ) -> TrellisParameter:
        if bitstreams.dtype is not torch.uint8 or bitstreams.ndim != 3:
            raise ValueError(
                "trellis bitstreams must be a uint8 [out, blocks, bytes] tensor, "
                f"got {bitstreams.dtype} {tuple(bitstreams.shape)}"
            )
        rows, blocks, _ = bitstreams.shape
        if block_scales.dtype is not torch.float8_e4m3fn or tuple(block_scales.shape) != (
            rows,
            blocks,
        ):
            raise ValueError(
                f"trellis block_scales must be float8_e4m3fn {(rows, blocks)}, "
                f"got {block_scales.dtype} {tuple(block_scales.shape)}"
            )
        if scale.dtype is not torch.float32 or tuple(scale.shape) != (rows, 1):
            raise ValueError(
                f"trellis scale must be float32 {(rows, 1)}, got {scale.dtype} {tuple(scale.shape)}"
            )
        parameter = torch.Tensor._make_wrapper_subclass(
            cls, (rows, blocks * block_size), dtype=dtype, device=bitstreams.device,
            requires_grad=False,
        )  # fmt: skip
        parameter.bitstreams = bitstreams.detach()
        parameter.block_scales = block_scales.detach()
        parameter.scale = scale.detach()
        parameter.act_per_tensor_scale = (
            None if act_per_tensor_scale is None else act_per_tensor_scale.detach()
        )
        parameter.block_size = int(block_size)
        parameter.group_size = int(group_size)
        return parameter

    def __init__(
        self,
        bitstreams: torch.Tensor,
        block_scales: torch.Tensor,
        scale: torch.Tensor,
        *,
        block_size: int,
        group_size: int,
        dtype: torch.dtype = torch.bfloat16,
        act_per_tensor_scale: torch.Tensor | None = None,
    ) -> None:
        # Tensor construction happens in __new__; spell out the constructor
        # so type checkers do not inherit Tensor's unrelated overloads.
        del bitstreams, block_scales, scale, block_size, group_size, dtype, act_per_tensor_scale

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """The packed tensors, then the activation scale when present."""
        tensors = (self.bitstreams, self.block_scales, self.scale)
        scale = self.act_per_tensor_scale
        return tensors if scale is None else (*tensors, scale)

    def rebuild(self, tensors: Sequence[torch.Tensor]) -> TrellisParameter:
        """This parameter's metadata around replacements for ``storage_tensors()``."""
        return TrellisParameter(
            tensors[0],
            tensors[1],
            tensors[2],
            block_size=self.block_size,
            group_size=self.group_size,
            dtype=self.dtype,
            act_per_tensor_scale=tensors[3] if len(tensors) == 4 else None,
        )

    def __repr__(self, *, tensor_contents: object = None) -> str:
        del tensor_contents
        return (
            f"TrellisParameter(shape={tuple(self.shape)}, block_size={self.block_size}, "
            f"bytes_per_block={self.bitstreams.shape[-1]}, group_size={self.group_size}, "
            f"device={self.device})"
        )

    @classmethod
    def __torch_dispatch__(
        cls, func: Callable[..., Any], types: Sequence[type[torch.Tensor]],
        args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        # PyTorch's dispatcher is untyped: contain its heterogeneous arguments
        # here, and only implement the operations a loaded source weight needs.
        del types
        aten = torch.ops.aten
        if func not in (aten.detach.default, aten.alias.default, aten.clone.default):
            raise NotImplementedError(
                f"{func} is unavailable on trellis-coded sources; split checkpoint storage "
                "before constructing the parameter, and activate with ModelOffloader"
            )
        source: TrellisParameter = args[0]
        kwargs = kwargs or {}
        return source.rebuild([func(tensor, **kwargs) for tensor in source.storage_tensors()])


__all__ = ["TrellisParameter"]
