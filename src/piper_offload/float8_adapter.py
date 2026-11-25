"""TorchAO ``Float8Tensor`` (scaled-fp8) adapter.

TorchAO's scaled-fp8 workflow stores weights as a tensor subclass with
fp8 bytes (``qdata``), per-row or per-tensor fp32 scales (``scale``),
and metadata controlling the matmul dispatch (``block_size``,
``mm_config``, ``kernel_preference``, ``act_quant_kwargs``). The shared
:class:`~piper_offload.torchao_structured_adapter.TorchaoStructuredAdapter`
base preserves that representation across CPU and GPU storage;
this module supplies the Float8-specific hooks.

Beyond inference movement, this adapter opts into:

- Format-specific LoRA merge: CUDA per-row, per-tensor, and standard
  per-group weights use raw Triton kernels when available. Other layouts
  and installations without Triton use the existing
  dequantize/GEMM/requantize path.
- Dequantize/requantize and ``copy_into`` remain separate conversion and copy
  capabilities and supply the adapter's reference merge fallback.
  Requantization recomputes scales via the public ``Float8Tensor.from_hp``,
  which is lossy but standard practice for permanent merges into quantized
  weights.
"""

from dataclasses import dataclass
from typing import Any

import torch

from ._dense_merge import merge_dense_requantize_
from ._torchao_float8 import (
    create_float8_tensor,
    dequantize_float8_tensor,
    is_float8_tensor,
    requantize_float8_tensor,
    require_float8_tensor,
    validate_float8_requantize_layout,
    validate_layout,
)
from .tensor_adapters import metadata_key
from .torchao_structured_adapter import TorchaoHost, TorchaoStructuredAdapter

try:
    from . import _triton_float8_lora
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _TRITON_MAX_GROUP_SIZE = 0
    _triton_merge_float8_dense = None
    _triton_merge_float8_lora = None
else:
    _TRITON_MAX_GROUP_SIZE = _triton_float8_lora.MAX_GROUP_SIZE
    _triton_merge_float8_dense = _triton_float8_lora.merge_float8_dense
    _triton_merge_float8_lora = _triton_float8_lora.merge_float8_lora


def _is_triton_float8_layout(
    t: Any,  # noqa: ANN401
    b: torch.Tensor,
    a: torch.Tensor,
) -> bool:
    """Return whether ``t`` uses a raw layout supported by the Triton path."""
    if (
        _triton_merge_float8_lora is None
        or t.qdata.device.type != "cuda"
        or t.qdata.ndim != 2
        or t.qdata.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
        or t.scale.dtype is not torch.float32
        or b.ndim != 2
        or a.ndim != 2
        or b.dtype is not a.dtype
        or b.dtype is not t.dtype
        or b.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or t.qdata.device != t.scale.device
        or t.qdata.device != b.device
        or t.qdata.device != a.device
    ):
        return False
    rows, cols = t.qdata.shape
    rank = a.shape[0]
    if rows == 0 or cols == 0 or rank == 0 or b.shape != (rows, rank) or a.shape[1] != cols:
        return False

    block_size = tuple(t.block_size)
    if block_size == (rows, cols):
        return t.scale.numel() == 1
    if len(block_size) != 2 or block_size[0] != 1 or not 0 < block_size[1] <= cols or cols % block_size[1] != 0:
        return False
    group_size = block_size[1]
    if group_size < cols and group_size > _TRITON_MAX_GROUP_SIZE:
        return False
    return tuple(t.scale.shape) == (rows, cols // group_size)


def _is_triton_float8_dense_layout(
    t: Any,  # noqa: ANN401
    update: torch.Tensor,
) -> bool:
    """Return whether ``t`` and a dense update fit the Triton path."""
    if (
        _triton_merge_float8_dense is None
        or t.qdata.device.type != "cuda"
        or t.qdata.ndim != 2
        or t.qdata.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
        or t.scale.dtype is not torch.float32
        or update.ndim != 2
        or update.dtype is not t.dtype
        or update.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or t.qdata.device != t.scale.device
        or t.qdata.device != update.device
    ):
        return False
    rows, cols = t.qdata.shape
    if rows == 0 or cols == 0 or tuple(update.shape) != (rows, cols):
        return False

    block_size = tuple(t.block_size)
    if block_size == (rows, cols):
        return t.scale.numel() == 1
    if len(block_size) != 2 or block_size[0] != 1 or not 0 < block_size[1] <= cols or cols % block_size[1] != 0:
        return False
    group_size = block_size[1]
    if group_size < cols and group_size > _TRITON_MAX_GROUP_SIZE:
        return False
    return tuple(t.scale.shape) == (rows, cols // group_size)


def _torch_merge_float8_lora_(
    target: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    alpha: float,
    *,
    beta: float = 1.0,
    rounding_seed: int | None = None,
) -> None:
    """Apply the established wrapper-based Float8 merge in place."""
    dense = dequantize_float8_tensor(target)
    dense.addmm_(b, a, beta=beta, alpha=alpha)
    requantized = requantize_float8_tensor(
        dense,
        like=target,
        rounding_seed=rounding_seed,
    )
    target_f8 = require_float8_tensor(target)
    source_f8 = require_float8_tensor(requantized)
    target_f8.qdata.copy_(source_f8.qdata)
    target_f8.scale.copy_(source_f8.scale)


def _validate_float8_shard_geometry(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    *,
    global_shape: tuple[int, ...],
    block_size: tuple[int, ...],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> None:
    ndim = len(global_shape)
    if (
        len(offsets) != ndim
        or len(local_shape) != ndim
        or len(block_size) != ndim
    ):
        raise ValueError(
            "Float8 shard projection dimensions must match weight rank: "
            f"global_shape={global_shape}, block_size={block_size}, "
            f"offsets={offsets}, local_shape={local_shape}."
        )
    if any(size <= 0 for size in local_shape):
        raise ValueError(
            "Float8 DTensor shard projection requires non-empty local "
            f"shards; got local_shape={local_shape}."
        )
    if any(
        offset < 0 or offset + size > global_size
        for offset, size, global_size in zip(
            offsets, local_shape, global_shape, strict=True
        )
    ):
        raise ValueError(
            "Float8 shard projection range exceeds global weight shape: "
            f"global_shape={global_shape}, offsets={offsets}, local_shape={local_shape}."
        )
    if (
        not qdata.is_contiguous()
        or not scale.is_contiguous()
        or tuple(qdata.shape) != global_shape
    ):
        raise ValueError(
            "Float8 DTensor shard projection requires contiguous, "
            "untransposed qdata and scale storage matching global_shape; "
            f"got qdata={tuple(qdata.shape)} (contiguous={qdata.is_contiguous()}), "
            f"scale={tuple(scale.shape)} (contiguous={scale.is_contiguous()})."
        )
    if any(
        block <= 0 or global_size % block != 0
        for global_size, block in zip(global_shape, block_size, strict=True)
    ):
        raise ValueError(
            f"Float8 block_size {block_size} does not divide global_shape {global_shape}."
        )


def _project_float8_scale(
    scale: torch.Tensor,
    *,
    global_shape: tuple[int, ...],
    block_size: tuple[int, ...],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if block_size == global_shape and scale.numel() == 1:
        return scale, local_shape

    expected_scale_shape = tuple(
        global_size // block
        for global_size, block in zip(global_shape, block_size, strict=True)
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "Float8 scale shape does not match block grid: "
            f"expected {expected_scale_shape}, got {tuple(scale.shape)}."
        )
    scale_local = scale
    block_dims = list(block_size)
    for dim, (offset, size, global_size, block) in enumerate(
        zip(offsets, local_shape, global_shape, block_size, strict=True)
    ):
        if offset == 0 and size == global_size:
            continue
        if block == global_size:
            block_dims[dim] = size
            continue
        if offset % block != 0 or size % block != 0:
            raise ValueError(
                f"Float8 shard along dim {dim} (offset={offset}, size={size}) "
                f"must align to quantization block_size {block}."
            )
        scale_local = scale_local.narrow(dim, offset // block, size // block)
    return scale_local, tuple(block_dims)


@dataclass(slots=True, frozen=True)
class _Float8Meta:
    """Reconstruction metadata snapshot for a TorchAO scaled-fp8 tensor."""

    block_size: tuple[int, ...]
    mm_config: object | None
    act_quant_kwargs: object | None
    kernel_preference: object
    dtype: torch.dtype  # logical (pre-quantization) dtype


class Float8Adapter(TorchaoStructuredAdapter[_Float8Meta]):
    """Adapter for TorchAO ``Float8Tensor`` (scaled-fp8) weights."""

    _TAG = "torchao-float8"
    _STORAGE_NAMES = ("qdata", "scale")

    # --- per-format hooks -------------------------------------------------

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_float8_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_float8_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale)

    @staticmethod
    def _meta_of(t: Any) -> _Float8Meta:  # noqa: ANN401
        return _Float8Meta(
            block_size=tuple(t.block_size),
            mm_config=t.mm_config,
            act_quant_kwargs=t.act_quant_kwargs,
            kernel_preference=t.kernel_preference,
            dtype=t.dtype,
        )

    @staticmethod
    def _reconstruct(storage: tuple[torch.Tensor | None, ...], meta: _Float8Meta) -> torch.Tensor:
        qdata, scale = storage
        assert qdata is not None
        assert scale is not None
        return create_float8_tensor(
            qdata,
            scale,
            list(meta.block_size),
            meta.mm_config,
            meta.act_quant_kwargs,
            meta.kernel_preference,
            meta.dtype,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (
            tuple(t.block_size),
            t.dtype,
            metadata_key(t.mm_config),
            metadata_key(t.kernel_preference),
            metadata_key(t.act_quant_kwargs),
        )

    @classmethod
    def _layout_metadata(cls, t: Any) -> tuple[object, ...]:  # noqa: ANN401
        # Float8 diverges from identity metadata: drop the logical dtype,
        # which layout_signature's standard dtype slot already carries.
        return (
            tuple(t.block_size),
            metadata_key(t.mm_config),
            metadata_key(t.kernel_preference),
            metadata_key(t.act_quant_kwargs),
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.dtype

    @classmethod
    def project_shard(
        cls,
        state: TorchaoHost[_Float8Meta],
        *,
        offsets: tuple[int, ...],
        local_shape: tuple[int, ...],
    ) -> tuple[TorchaoHost[_Float8Meta], tuple[object, ...]]:
        """Project a zero-copy host shard for ``Float8Tensor`` weights."""
        qdata, scale = state.storage
        assert qdata is not None
        assert scale is not None
        global_shape = state.logical_shape
        block_size = state.meta.block_size
        _validate_float8_shard_geometry(
            qdata,
            scale,
            global_shape=global_shape,
            block_size=block_size,
            offsets=offsets,
            local_shape=local_shape,
        )

        qdata_local = qdata
        for dim, (offset, size) in enumerate(zip(offsets, local_shape, strict=True)):
            if offset != 0 or size != qdata_local.shape[dim]:
                qdata_local = qdata_local.narrow(dim, offset, size)

        scale_local, new_block_size = _project_float8_scale(
            scale,
            global_shape=global_shape,
            block_size=block_size,
            offsets=offsets,
            local_shape=local_shape,
        )

        meta = _Float8Meta(
            block_size=new_block_size,
            mm_config=state.meta.mm_config,
            act_quant_kwargs=state.meta.act_quant_kwargs,
            kernel_preference=state.meta.kernel_preference,
            dtype=state.meta.dtype,
        )
        return cls._finalize_projected_shard(
            (qdata_local, scale_local),
            meta,
            local_shape,
        )

    # --- capabilities beyond inference movement ---------------------------

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_float8_tensor(t)

    @staticmethod
    def requantize(
        t: torch.Tensor,
        *,
        like: torch.Tensor,
        rounding_seed: int | None = None,
    ) -> torch.Tensor:
        return requantize_float8_tensor(t, like=like, rounding_seed=rounding_seed)

    @staticmethod
    def validate_lora_merge(
        target: torch.Tensor,
        _b: torch.Tensor,
        _a: torch.Tensor,
        _alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        del beta, rounding_seed
        validate_float8_requantize_layout(target)

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
        """Merge a validated staged update while preserving the wrapper."""
        f8 = require_float8_tensor(target)
        if _is_triton_float8_layout(f8, b, a):
            assert _triton_merge_float8_lora is not None
            qdata, scale = _triton_merge_float8_lora(
                f8.qdata,
                f8.scale,
                tuple(f8.block_size),
                b,
                a,
                alpha,
                beta=beta,
                rounding_seed=rounding_seed,
            )
            f8.qdata.copy_(qdata)
            f8.scale.copy_(scale)
            return
        _torch_merge_float8_lora_(
            target,
            b,
            a,
            alpha,
            beta=beta,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def validate_dense_merge_target(
        target: torch.Tensor,
        *,
        rounding_seed: int | None = None,
    ) -> bool:
        del rounding_seed
        validate_float8_requantize_layout(target)
        return False

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a full-rank update while preserving the wrapper."""
        f8 = require_float8_tensor(target)
        if _is_triton_float8_dense_layout(f8, update):
            assert _triton_merge_float8_dense is not None
            qdata, scale = _triton_merge_float8_dense(
                f8.qdata,
                f8.scale,
                tuple(f8.block_size),
                update,
                alpha,
                beta=beta,
                rounding_seed=rounding_seed,
            )
            f8.qdata.copy_(qdata)
            f8.scale.copy_(scale)
            return
        merge_dense_requantize_(
            Float8Adapter,
            target,
            update,
            alpha,
            beta=beta,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        target_f8 = require_float8_tensor(target)
        src_f8 = require_float8_tensor(src)
        target_f8.qdata.copy_(src_f8.qdata)
        target_f8.scale.copy_(src_f8.scale)
