"""TorchAO ``Float8Tensor`` (scaled-fp8) adapter.

TorchAO's scaled-fp8 workflow stores weights as a tensor subclass with
fp8 bytes (``qdata``), per-row or per-tensor fp32 scales (``scale``),
and metadata controlling the matmul dispatch (``block_size``,
``mm_config``, ``kernel_preference``, ``act_quant_kwargs``). The shared
:class:`~piper_offload.torchao_structured_adapter.TorchaoStructuredAdapter`
base preserves that representation across pinned CPU and GPU storage;
this module supplies the Float8-specific hooks.

Beyond inference movement, this adapter opts into:

- CPU round-trip: GPU storage is the identical fp8 bytes, so D2H back
  into the pinned host state is lossless.
- Format-specific LoRA merge: CUDA per-row, per-tensor, and standard
  per-group weights use raw Triton kernels when available. Other layouts
  and installations without Triton use the existing
  dequantize/GEMM/requantize path.
- Dequantize/requantize and ``copy_into`` remain separate conversion and copy
  capabilities and supply the adapter's reference merge fallback.
  Requantization recomputes scales via the public ``Float8Tensor.from_hp``,
  which is lossy but standard practice for permanent merges into quantized
  weights.

No trainable ``Parameter.data`` swap — the quant state lives in the
wrapper object, not its bytes, so scaled-fp8 weights stay frozen.
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
from .torchao_structured_adapter import (
    TorchaoGpu,
    TorchaoPinned,
    TorchaoStructuredAdapter,
    copy_storage,
)

try:
    from . import _triton_float8_lora
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _TRITON_MAX_GROUP_SIZE = 0
    _triton_merge_float8_lora = None
else:
    _TRITON_MAX_GROUP_SIZE = _triton_float8_lora.MAX_GROUP_SIZE
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


def _torch_merge_float8_lora_(
    target: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> None:
    """Apply the established wrapper-based Float8 merge in place."""
    dense = dequantize_float8_tensor(target)
    dense.addmm_(b, a, alpha=strength)
    requantized = requantize_float8_tensor(
        dense,
        like=target,
        rounding_seed=rounding_seed,
    )
    target_f8 = require_float8_tensor(target)
    source_f8 = require_float8_tensor(requantized)
    target_f8.qdata.copy_(source_f8.qdata)
    target_f8.scale.copy_(source_f8.scale)


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

    # --- capabilities beyond inference movement ---------------------------

    @staticmethod
    def copy_to_cpu(src: TorchaoGpu, dst: TorchaoPinned[_Float8Meta], *, non_blocking: bool = False) -> None:
        # GPU storage is the identical fp8 bytes + scales, so D2H is a
        # lossless byte copy. Quant metadata lives on the pinned state.
        copy_storage(src.storage, dst.storage, non_blocking=non_blocking)

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
        _strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        del rounding_seed
        validate_float8_requantize_layout(target)

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
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
                strength,
                rounding_seed=rounding_seed,
            )
            f8.qdata.copy_(qdata)
            f8.scale.copy_(scale)
            return
        _torch_merge_float8_lora_(
            target,
            b,
            a,
            strength,
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
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a full-rank update through the reference requantization path."""
        merge_dense_requantize_(
            Float8Adapter,
            target,
            update,
            strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        target_f8 = require_float8_tensor(target)
        src_f8 = require_float8_tensor(src)
        target_f8.qdata.copy_(src_f8.qdata)
        target_f8.scale.copy_(src_f8.scale)
