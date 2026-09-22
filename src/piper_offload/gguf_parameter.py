"""Packed GGUF source parameters with logical shape and compute dtype.

These parameters support the structural operations used while loading a model:
row splits, row indexing, concatenating and regrouping rows, detach, and clone.
Arithmetic is provided by the ConvRot target during Offload activation. The
packed bytes are immutable; operations that copy have no checkpoint provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from piper_kernels.gguf import GGUF_QUANT_SIZES, GGUFQuantizationType, logical_shape
from torch import nn

GGUF_COMPUTE_DTYPE = torch.bfloat16


class GgufParameter(nn.Parameter):
    """A logical BF16 parameter backed by packed GGUF bytes, without decoding them.

    ``as_tensor()`` exposes the byte tensor for the tensor adapter. Structural
    operations must preserve complete encoded rows. Use ``ModelOffloader`` to
    activate the parameter before executing its module.
    """

    _packed: torch.Tensor
    quant_type: int
    # PyTorch's C sentinel disables this hook; its stub is not a classmethod.
    __torch_function__ = torch._C._disabled_torch_function_impl  # pyright: ignore[reportAssignmentType]

    @staticmethod
    def __new__(
        cls: type[GgufParameter], data: torch.Tensor, requires_grad: bool = False, *, quant_type: int,
    ) -> GgufParameter:
        if requires_grad:
            raise ValueError("GGUF parameters are inference-only")
        packed = data.detach().view(torch.uint8)
        shape = logical_shape(tuple(packed.shape), quant_type)
        block_size, type_size = GGUF_QUANT_SIZES[GGUFQuantizationType(quant_type)]
        strides = (*[stride // type_size * block_size for stride in packed.stride()[:-1]], 1)
        parameter = torch.Tensor._make_wrapper_subclass(
            cls, shape, strides=strides, dtype=GGUF_COMPUTE_DTYPE,
            device=packed.device, requires_grad=False,
        )
        parameter._packed = packed
        parameter.quant_type = int(quant_type)
        return parameter

    def __init__(self, data: torch.Tensor, requires_grad: bool = False, *, quant_type: int) -> None:
        # Tensor construction happens in __new__; spell out the constructor
        # so type checkers do not inherit Tensor's unrelated overloads.
        del data, requires_grad, quant_type

    @property
    def quant_shape(self) -> torch.Size:
        """Logical shape under the shared GGUF tensor adapter contract."""
        return self.shape

    def as_tensor(self) -> torch.Tensor:
        """The packed bytes, sharing their original storage and file provenance."""
        return self._packed

    def __repr__(self, *, tensor_contents: object = None) -> str:
        del tensor_contents
        return f"GgufParameter(shape={tuple(self.shape)}, quant_type={self.quant_type}, device={self.device})"

    @classmethod
    def __torch_dispatch__(
        cls, func: Callable[..., Any], types: Sequence[type[torch.Tensor]],
        args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        # PyTorch's dispatcher is untyped: contain its heterogeneous arguments
        # here, and only implement the structural operations of a source weight.
        del types
        kwargs = kwargs or {}
        aten = torch.ops.aten
        source: GgufParameter = args[0][0] if func is aten.cat.default else args[0]
        packed = source._packed
        if func is aten.cat.default:
            tensors: Sequence[GgufParameter] = args[0]
            dim = args[1] if len(args) > 1 else 0
            if dim % source.ndim == source.ndim - 1 or any(t.quant_type != source.quant_type for t in tensors):
                raise ValueError("GGUF concatenation requires complete rows with the same encoding")
            result = torch.cat([tensor._packed for tensor in tensors], dim=dim)
        elif func in (aten.detach.default, aten.alias.default, aten.clone.default):
            result = func(packed, **kwargs)
        elif func in (aten.view.default, aten._unsafe_view.default):
            shape = list(args[1])
            # Reshaping the encoded axis would reinterpret scales as weights.
            if shape[-1] != source.shape[-1]:
                raise ValueError("GGUF reshapes must preserve complete encoded rows")
            shape[-1] = packed.shape[-1]
            result = func(packed, shape)
        elif func in (aten.split.Tensor, aten.split_with_sizes.default):
            sizes = args[1]
            dim = args[2] if len(args) > 2 else 0
            if dim % source.ndim == source.ndim - 1:
                raise ValueError("GGUF splits must preserve complete encoded rows")
            return tuple(cls(piece, quant_type=source.quant_type) for piece in func(packed, sizes, dim))
        elif func in (aten.slice.Tensor, aten.select.int):
            dim = args[1] if len(args) > 1 else 0
            if dim % source.ndim == source.ndim - 1:
                # A full slice is emitted by ordinary multidimensional indexing.
                if func is aten.slice.Tensor and args[2:] == (0, 9223372036854775807):
                    return cls(packed, quant_type=source.quant_type)
                raise ValueError("GGUF indexing must preserve complete encoded rows")
            result = func(packed, *args[1:], **kwargs)
        else:
            raise NotImplementedError(f"{func} is unavailable on packed GGUF sources; activate with ModelOffloader")
        return cls(result, quant_type=source.quant_type)


__all__ = ["GgufParameter"]
