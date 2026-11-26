"""Piper ``PiperNVFP4Tensor`` offload adapter.

Piper Offload owns packed-storage movement and merge integration. Piper
Kernels owns NVFP4 dequantization, linear dispatch, and the in-place
``PiperNVFP4Tensor.addmm_`` and ``PiperNVFP4Tensor.add_`` updates, including
both update and base scaling.
"""

from dataclasses import dataclass
from typing import Any

import torch

from ._nvfp4_projection import project_nvfp4_storage
from ._piper_nvfp4 import (
    create_convrot_nvfp4_tensor,
    create_piper_nvfp4_tensor,
    is_piper_nvfp4_tensor,
    require_piper_nvfp4_matrix,
    require_piper_nvfp4_tensor,
    validate_layout,
)
from .structured_adapter import StructuredHost, _KernelMergeStructuredAdapter
from .tensor_adapters import metadata_key


@dataclass(slots=True, frozen=True)
class _PiperNvfp4Meta:
    """NVFP4 metadata needed to reconstruct ``PiperNVFP4Tensor`` or ``ConvRotNVFP4Tensor``."""

    wrapper_type: type[torch.Tensor]
    block_size: int
    orig_dtype: torch.dtype
    is_swizzled_scales: bool
    use_triton_kernel: bool
    act_quant_kwargs: object | None
    high_first: bool
    group_size: int | None = None


class _PiperNvfp4BaseAdapter(_KernelMergeStructuredAdapter[_PiperNvfp4Meta]):
    """Shared structured adapter mechanics for Piper NVFP4 and ConvRot NVFP4."""

    _STORAGE_NAMES = (
        "qdata",
        "scale",
        "per_tensor_scale",
        "act_per_tensor_scale",
    )

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale, t.per_tensor_scale, t.act_per_tensor_scale)

    @staticmethod
    def _meta_from_nvfp4(
        t: Any,  # noqa: ANN401
        *,
        group_size: int | None = None,
    ) -> _PiperNvfp4Meta:
        return _PiperNvfp4Meta(
            wrapper_type=type(t),
            block_size=t.block_size,
            orig_dtype=t.orig_dtype,
            is_swizzled_scales=t.is_swizzled_scales,
            use_triton_kernel=t.use_triton_kernel,
            act_quant_kwargs=t.act_quant_kwargs,
            high_first=t.high_first,
            group_size=group_size,
        )

    @staticmethod
    def _id_metadata_nvfp4(t: Any, *extra: object) -> tuple[object, ...]:  # noqa: ANN401
        return (
            type(t),
            t.block_size,
            t.orig_dtype,
            t.is_swizzled_scales,
            t.use_triton_kernel,
            metadata_key(t.act_quant_kwargs),
            t.high_first,
            *extra,
        )

    @staticmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...],
        meta: _PiperNvfp4Meta,
    ) -> torch.Tensor:
        qdata, scale, per_tensor_scale, act_per_tensor_scale = storage
        assert qdata is not None
        assert scale is not None
        if meta.group_size is not None:
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
                meta.high_first,
                wrapper_type=meta.wrapper_type,
            )
        return create_piper_nvfp4_tensor(
            qdata,
            scale,
            meta.block_size,
            meta.orig_dtype,
            per_tensor_scale,
            act_per_tensor_scale,
            meta.is_swizzled_scales,
            meta.use_triton_kernel,
            meta.act_quant_kwargs,
            meta.high_first,
            wrapper_type=meta.wrapper_type,
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.orig_dtype

    @classmethod
    def project_shard(
        cls,
        state: StructuredHost[_PiperNvfp4Meta],
        *,
        offsets: tuple[int, ...],
        local_shape: tuple[int, ...],
    ) -> tuple[StructuredHost[_PiperNvfp4Meta], tuple[object, ...]]:
        """Project a zero-copy NVFP4 shard from ``state``."""
        storage, target_shapes = project_nvfp4_storage(
            adapter_name_label=cls.__name__,
            storage=state.storage,
            logical_shape=state.logical_shape,
            offsets=offsets,
            local_shape=local_shape,
            block_size=state.meta.block_size,
            is_swizzled_scales=state.meta.is_swizzled_scales,
        )
        return cls._finalize_projected_shard(
            storage,
            state.meta,
            local_shape,
            target_shapes=target_shapes,
        )


class PiperNvfp4Adapter(_PiperNvfp4BaseAdapter):
    """Adapter for Piper Kernels ``PiperNVFP4Tensor`` weights (2-D and 3-D MoE)."""

    _TAG = "piper-kernels-nvfp4"

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_piper_nvfp4_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_piper_nvfp4_tensor(t)

    @staticmethod
    def _require_matrix(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_piper_nvfp4_matrix(t)

    @staticmethod
    def _meta_of(t: Any) -> _PiperNvfp4Meta:  # noqa: ANN401
        return _PiperNvfp4BaseAdapter._meta_from_nvfp4(t)

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return _PiperNvfp4BaseAdapter._id_metadata_nvfp4(t)
