"""Trellis-coded ConvRot INT8 source parameters with logical shape and compute dtype.

A trellis-coded weight stores a ConvRot INT8 matrix in 2 to 6 bits per weight
(Piper Kernels' ``piper_kernels.trellis`` format), either at a uniform block
width or with per-row byte widths. ``TrellisParameter`` keeps those packed
tensors on the host and presents the logical ``[out, in]`` weight; Offload
activation decodes it into ConvRot INT8 (``TrellisAdapter``). Loaders split or
regroup checkpoint storage before constructing it: the parameter supports only
detach, alias, and clone, which share or copy all of its tensors.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

_MIN_BLOCK_SIZE = 16
_MAX_BLOCK_SIZE = 256
_MIN_BITS_PER_WEIGHT = 2
_MAX_BITS_PER_WEIGHT = 6


def _validate_block_size(block_size: int) -> int:
    if isinstance(block_size, bool):
        raise TypeError(f"trellis block_size must be an integer, got {block_size!r}")
    try:
        resolved = operator.index(block_size)
    except TypeError as error:
        raise TypeError(f"trellis block_size must be an integer, got {block_size!r}") from error
    if not _MIN_BLOCK_SIZE <= resolved <= _MAX_BLOCK_SIZE:
        raise ValueError(
            f"trellis block_size must be between {_MIN_BLOCK_SIZE} and {_MAX_BLOCK_SIZE}, "
            f"got {resolved}"
        )
    return resolved


def _require_strided(tensor: torch.Tensor, name: str) -> None:
    if tensor.layout is not torch.strided:
        raise ValueError(f"trellis {name} must have strided layout, got {tensor.layout}")


def validate_row_widths_cpu(
    row_widths: torch.Tensor,
    *,
    rows: int,
    blocks: int,
    block_size: int,
    bitstream_bytes: int,
) -> None:
    """Validate CPU ``row_widths`` bounds and minimum bitstream capacity."""
    min_bytes = (_MIN_BITS_PER_WEIGHT * block_size + 7) // 8
    max_bytes = (_MAX_BITS_PER_WEIGHT * block_size) // 8
    if rows > 0:
        min_width = int(row_widths.min().item())
        max_width = int(row_widths.max().item())
        if min_width < min_bytes or max_width > max_bytes:
            raise ValueError(
                f"trellis row_widths for block_size={block_size} must be in [{min_bytes}, "
                f"{max_bytes}], got [{min_width}, {max_width}]"
            )
    expected_bytes = int(row_widths.sum(dtype=torch.int64).item()) * blocks
    if bitstream_bytes < expected_bytes:
        raise ValueError(
            f"trellis bitstreams length ({bitstream_bytes}) is smaller than "
            f"row_widths sum * blocks ({expected_bytes})"
        )


def _validate_uniform_inputs(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
) -> tuple[int, int]:
    _require_strided(bitstreams, "bitstreams")
    if bitstreams.dtype is not torch.uint8 or bitstreams.ndim != 3:
        raise ValueError(
            "trellis bitstreams must be a uint8 [out, blocks, bytes] tensor, "
            f"got {bitstreams.dtype} {tuple(bitstreams.shape)}"
        )
    rows, blocks, _ = bitstreams.shape
    if blocks <= 0:
        raise ValueError("trellis weights must contain at least one block per row")
    _require_strided(block_scales, "block_scales")
    if block_scales.dtype is not torch.float8_e4m3fn or tuple(block_scales.shape) != (
        rows,
        blocks,
    ):
        raise ValueError(
            f"trellis block_scales must be float8_e4m3fn {(rows, blocks)}, "
            f"got {block_scales.dtype} {tuple(block_scales.shape)}"
        )
    return rows, blocks


def _validate_variable_row_inputs(
    bitstreams: torch.Tensor,
    block_scales: torch.Tensor,
    row_widths: torch.Tensor,
    *,
    block_size: int,
) -> tuple[int, int]:
    _require_strided(bitstreams, "bitstreams")
    if bitstreams.dtype is not torch.uint8 or bitstreams.ndim != 1:
        raise ValueError(
            "variable-row trellis bitstreams must be a 1-D uint8 [total_bytes] tensor, "
            f"got {bitstreams.dtype} {tuple(bitstreams.shape)}"
        )
    _require_strided(block_scales, "block_scales")
    if block_scales.dtype is not torch.float8_e4m3fn or block_scales.ndim != 2:
        raise ValueError(
            "trellis block_scales must be a 2-D float8_e4m3fn [out, blocks] tensor, "
            f"got {block_scales.dtype} {tuple(block_scales.shape)}"
        )
    rows, blocks = block_scales.shape
    if blocks <= 0:
        raise ValueError("trellis weights must contain at least one block per row")
    _require_strided(row_widths, "row_widths")
    if row_widths.dtype is not torch.uint8 or tuple(row_widths.shape) != (rows,):
        raise ValueError(
            f"trellis row_widths must be uint8 {(rows,)}, "
            f"got {row_widths.dtype} {tuple(row_widths.shape)}"
        )
    # Skip content checks on meta placeholders and device tensors (no device sync on construction;
    # GPU parameters copy and validate row_widths before copying bitstreams in capture_host).
    if row_widths.device.type == "cpu":
        validate_row_widths_cpu(
            row_widths,
            rows=rows,
            blocks=blocks,
            block_size=block_size,
            bitstream_bytes=bitstreams.numel(),
        )
    return rows, blocks


class TrellisParameter(nn.Parameter):
    """A logical parameter backed by trellis-coded storage, without decoding it.

    For a logical ``[out, in]`` weight in blocks of ``block_size`` weights:

    - With uniform rows (``row_widths=None``), ``bitstreams`` is uint8
      ``[out, in / block_size, bytes]``.
    - With variable rows, ``row_widths`` is uint8 ``[out]`` (bytes per block for
      each row) and ``bitstreams`` is 1-D uint8 holding row-concatenated streams
      with at least ``(in / block_size) * sum(row_widths)`` bytes.

    In both layouts, ``block_scales`` is float8_e4m3fn ``[out, in / block_size]``,
    and ``scale`` is the float32 row scale ``[out, 1]``. The streams hold the weight
    rotated for ConvRot ``group_size``. ``dtype`` is the logical dtype of the
    decoded ConvRot INT8 weight, and ``act_per_tensor_scale`` its optional static
    strided float32 scalar activation scale. CPU construction validates
    ``row_widths`` bounds (2 to 6 code bits per weight) and minimum
    ``bitstreams`` capacity. Use ``ModelOffloader`` to activate the parameter before
    executing its module.
    """

    bitstreams: torch.Tensor
    block_scales: torch.Tensor
    scale: torch.Tensor
    row_widths: torch.Tensor | None
    act_per_tensor_scale: torch.Tensor | None
    block_size: int
    group_size: int
    # PyTorch's C sentinel disables this hook; its stub is not a classmethod.
    __torch_function__ = torch._C._disabled_torch_function_impl  # pyright: ignore[reportAssignmentType]

    @classmethod
    def _wrap_validated(
        cls,
        bitstreams: torch.Tensor,
        block_scales: torch.Tensor,
        scale: torch.Tensor,
        *,
        block_size: int,
        group_size: int,
        dtype: torch.dtype,
        act_per_tensor_scale: torch.Tensor | None = None,
        row_widths: torch.Tensor | None = None,
    ) -> TrellisParameter:
        rows, blocks = block_scales.shape
        parameter = torch.Tensor._make_wrapper_subclass(
            cls, (rows, blocks * block_size), dtype=dtype, device=bitstreams.device,
            requires_grad=False,
        )  # fmt: skip
        parameter.bitstreams = bitstreams.detach()
        parameter.block_scales = block_scales.detach()
        parameter.scale = scale.detach()
        parameter.row_widths = None if row_widths is None else row_widths.detach()
        parameter.act_per_tensor_scale = (
            None if act_per_tensor_scale is None else act_per_tensor_scale.detach()
        )
        parameter.block_size = block_size
        parameter.group_size = int(group_size)
        return parameter

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
        row_widths: torch.Tensor | None = None,
    ) -> TrellisParameter:
        resolved_block_size = _validate_block_size(block_size)
        if row_widths is None:
            rows, _ = _validate_uniform_inputs(bitstreams, block_scales)
        else:
            rows, _ = _validate_variable_row_inputs(
                bitstreams,
                block_scales,
                row_widths,
                block_size=resolved_block_size,
            )
        _require_strided(scale, "scale")
        if scale.dtype is not torch.float32 or tuple(scale.shape) != (rows, 1):
            raise ValueError(
                f"trellis scale must be float32 {(rows, 1)}, got {scale.dtype} {tuple(scale.shape)}"
            )
        if act_per_tensor_scale is not None:
            _require_strided(act_per_tensor_scale, "act_per_tensor_scale")
            if act_per_tensor_scale.ndim != 0 or act_per_tensor_scale.dtype is not torch.float32:
                raise ValueError(
                    "trellis act_per_tensor_scale must be a strided float32 scalar, "
                    f"got {act_per_tensor_scale.dtype} {tuple(act_per_tensor_scale.shape)}"
                )
        is_meta = bitstreams.is_meta
        if any(
            tensor is not None and tensor.is_meta != is_meta
            for tensor in (block_scales, scale, row_widths, act_per_tensor_scale)
        ):
            raise ValueError("trellis storage tensors cannot mix meta and physical devices")
        return cls._wrap_validated(
            bitstreams,
            block_scales,
            scale,
            block_size=resolved_block_size,
            group_size=group_size,
            dtype=dtype,
            act_per_tensor_scale=act_per_tensor_scale,
            row_widths=row_widths,
        )

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
        row_widths: torch.Tensor | None = None,
    ) -> None:
        # Tensor construction happens in __new__; spell out the constructor
        # so type checkers do not inherit Tensor's unrelated overloads.
        del (
            bitstreams,
            block_scales,
            scale,
            block_size,
            group_size,
            dtype,
            act_per_tensor_scale,
            row_widths,
        )

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """The packed tensors, optional row widths, then the activation scale."""
        tensors: tuple[torch.Tensor, ...] = (self.bitstreams, self.block_scales, self.scale)
        if self.row_widths is not None:
            tensors = (*tensors, self.row_widths)
        if self.act_per_tensor_scale is not None:
            tensors = (*tensors, self.act_per_tensor_scale)
        return tensors

    def _unpack_storage_tensors(
        self, tensors: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        has_row_widths = self.row_widths is not None
        has_act_scale = self.act_per_tensor_scale is not None
        expected = 3 + has_row_widths + has_act_scale
        if len(tensors) != expected:
            raise ValueError(f"expected {expected} storage tensors, got {len(tensors)}")
        index = 3
        row_widths = tensors[index] if has_row_widths else None
        if has_row_widths:
            index += 1
        act_scale = tensors[index] if has_act_scale else None
        return row_widths, act_scale

    def _rewrap_validated(self, tensors: Sequence[torch.Tensor]) -> TrellisParameter:
        row_widths, act_scale = self._unpack_storage_tensors(tensors)
        return type(self)._wrap_validated(
            tensors[0],
            tensors[1],
            tensors[2],
            block_size=self.block_size,
            group_size=self.group_size,
            dtype=self.dtype,
            act_per_tensor_scale=act_scale,
            row_widths=row_widths,
        )

    def rebuild(self, tensors: Sequence[torch.Tensor]) -> TrellisParameter:
        """This parameter's metadata around replacements for ``storage_tensors()``."""
        row_widths, act_scale = self._unpack_storage_tensors(tensors)
        return type(self)(
            tensors[0],
            tensors[1],
            tensors[2],
            block_size=self.block_size,
            group_size=self.group_size,
            dtype=self.dtype,
            act_per_tensor_scale=act_scale,
            row_widths=row_widths,
        )

    def __repr__(self, *, tensor_contents: object = None) -> str:
        del tensor_contents
        rate = (
            f"bytes_per_block={self.bitstreams.shape[-1]}"
            if self.row_widths is None
            else f"total_bytes={self.bitstreams.numel()}, variable_rows=True"
        )
        return (
            f"TrellisParameter(shape={tuple(self.shape)}, block_size={self.block_size}, "
            f"{rate}, group_size={self.group_size}, device={self.device})"
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
        return source._rewrap_validated([func(tensor, **kwargs) for tensor in source.storage_tensors()])


__all__ = ["TrellisParameter"]
