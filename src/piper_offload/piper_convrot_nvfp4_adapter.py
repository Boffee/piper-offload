"""Piper ``ConvRotNVFP4Tensor`` offload adapter.

Piper Offload owns packed-storage movement and merge integration. Piper
Kernels owns rotation, scale recomputation, terminal-code selection, and the
in-place updates through ``ConvRotNVFP4Tensor.addmm_`` and
``ConvRotNVFP4Tensor.add_``.
"""

from dataclasses import dataclass
from typing import Any

import torch

from ._piper_convrot_nvfp4 import (
    create_convrot_nvfp4_tensor,
    is_convrot_nvfp4_tensor,
    require_convrot_nvfp4_add,
    require_convrot_nvfp4_addmm,
    require_convrot_nvfp4_tensor,
    validate_layout,
)
from .tensor_adapters import metadata_key
from .torchao_structured_adapter import TorchaoStructuredAdapter


@dataclass(slots=True, frozen=True)
class _PiperConvRotNVFP4Meta:
    """NVFP4 and rotation metadata needed to reconstruct a wrapper."""

    wrapper_type: type[torch.Tensor]
    block_size: int
    orig_dtype: torch.dtype
    group_size: int
    is_swizzled_scales: bool
    use_triton_kernel: bool
    act_quant_kwargs: object | None


class PiperConvRotNVFP4Adapter(TorchaoStructuredAdapter[_PiperConvRotNVFP4Meta]):
    """Adapter for Piper Kernels ``ConvRotNVFP4Tensor`` weights."""

    _TAG = "piper-kernels-convrot-nvfp4"
    _STORAGE_NAMES = (
        "qdata",
        "scale",
        "per_tensor_scale",
        "act_per_tensor_scale",
    )

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_convrot_nvfp4_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_convrot_nvfp4_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale, t.per_tensor_scale, t.act_per_tensor_scale)

    @staticmethod
    def _meta_of(t: Any) -> _PiperConvRotNVFP4Meta:  # noqa: ANN401
        return _PiperConvRotNVFP4Meta(
            wrapper_type=type(t),
            block_size=t.block_size,
            orig_dtype=t.orig_dtype,
            group_size=t.group_size,
            is_swizzled_scales=t.is_swizzled_scales,
            use_triton_kernel=t.use_triton_kernel,
            act_quant_kwargs=t.act_quant_kwargs,
        )

    @staticmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...],
        meta: _PiperConvRotNVFP4Meta,
    ) -> torch.Tensor:
        qdata, scale, per_tensor_scale, act_per_tensor_scale = storage
        assert qdata is not None
        assert scale is not None
        return create_convrot_nvfp4_tensor(
            qdata,
            scale,
            meta.block_size,
            meta.orig_dtype,
            meta.group_size,
            per_tensor_scale,
            act_per_tensor_scale,
            meta.is_swizzled_scales,
            meta.use_triton_kernel,
            meta.act_quant_kwargs,
            wrapper_type=meta.wrapper_type,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (
            type(t),
            t.block_size,
            t.orig_dtype,
            t.group_size,
            t.is_swizzled_scales,
            t.use_triton_kernel,
            metadata_key(t.act_quant_kwargs),
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.orig_dtype

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Delegate a validated staged update to Piper Kernels."""
        require_convrot_nvfp4_addmm(target).addmm_(
            b,
            a,
            alpha=strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def validate_lora_merge(
        target: torch.Tensor,
        _b: torch.Tensor,
        _a: torch.Tensor,
        _strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        del rounding_seed
        require_convrot_nvfp4_addmm(target)

    @staticmethod
    def validate_dense_merge_target(
        target: torch.Tensor,
        *,
        rounding_seed: int | None = None,
    ) -> bool:
        """Validate kernel support without staging the dense update."""
        del rounding_seed
        require_convrot_nvfp4_add(target)
        return False

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Delegate a validated dense update to Piper Kernels."""
        require_convrot_nvfp4_add(target).add_(
            update,
            alpha=strength,
            rounding_seed=rounding_seed,
        )
