"""Piper ``ConvRotNVFP4Tensor`` offload adapter.

Piper Offload owns packed-storage movement and merge integration. Piper
Kernels owns rotation, scale recomputation, terminal-code selection, and the
in-place updates through ``ConvRotNVFP4Tensor.addmm_`` and
``ConvRotNVFP4Tensor.add_``, including both update and base scaling.
"""

from typing import Any

import torch

from ._nvfp4_projection import project_nvfp4_storage
from ._piper_nvfp4 import (
    is_convrot_nvfp4_tensor,
    require_convrot_nvfp4_matrix,
    require_convrot_nvfp4_tensor,
    validate_convrot_nvfp4_layout,
)
from .piper_nvfp4_adapter import _PiperNvfp4BaseAdapter, _PiperNvfp4Meta
from .structured_adapter import StructuredHost


class PiperConvRotNVFP4Adapter(_PiperNvfp4BaseAdapter):
    """Adapter for Piper Kernels ``ConvRotNVFP4Tensor`` weights."""

    _TAG = "piper-kernels-convrot-nvfp4"

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_convrot_nvfp4_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_convrot_nvfp4_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_convrot_nvfp4_tensor(t)

    @staticmethod
    def _require_matrix(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_convrot_nvfp4_matrix(t)

    @staticmethod
    def _meta_of(t: Any) -> _PiperNvfp4Meta:  # noqa: ANN401
        return _PiperNvfp4BaseAdapter._meta_from_nvfp4(t, group_size=t.group_size)

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return _PiperNvfp4BaseAdapter._id_metadata_nvfp4(t, t.group_size)

    @classmethod
    def project_shard(
        cls,
        state: StructuredHost[_PiperNvfp4Meta],
        *,
        offsets: tuple[int, ...],
        local_shape: tuple[int, ...],
    ) -> tuple[StructuredHost[_PiperNvfp4Meta], tuple[object, ...]]:
        """Project a zero-copy 2-D ``ConvRotNVFP4Tensor`` shard from ``state``."""
        if len(state.logical_shape) != 2:
            raise ValueError(
                f"{cls.__name__} shard projection only supports 2-D tensors, "
                f"got shape {state.logical_shape}."
            )
        group_size = state.meta.group_size
        if group_size is None or group_size <= 0:
            raise ValueError(
                f"{cls.__name__} rotation group_size must be positive, got {group_size}."
            )
        storage, target_shapes = project_nvfp4_storage(
            adapter_name_label=cls.__name__,
            storage=state.storage,
            logical_shape=state.logical_shape,
            offsets=offsets,
            local_shape=local_shape,
            block_size=state.meta.block_size,
            is_swizzled_scales=state.meta.is_swizzled_scales,
        )
        col_off, local_cols = offsets[1], local_shape[1]
        if col_off % group_size != 0 or local_cols % group_size != 0:
            raise ValueError(
                f"{cls.__name__} Shard(1) offset/width ({col_off}, {local_cols}) "
                f"must align to rotation group_size {group_size}."
            )
        return cls._finalize_projected_shard(
            storage,
            state.meta,
            local_shape,
            target_shapes=target_shapes,
        )
