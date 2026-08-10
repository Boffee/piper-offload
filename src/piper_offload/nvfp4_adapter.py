"""TorchAO ``NVFP4Tensor`` adapter.

TorchAO's NVFP4 workflow stores weights as a tensor subclass with packed
FP4 bytes (``qdata``), FP8 block scales (``scale``), optional global
per-tensor scales (``per_tensor_scale`` / ``act_per_tensor_scale``), and
metadata controlling the matmul dispatch. The shared
:class:`~piper_offload.torchao_structured_adapter.TorchaoStructuredAdapter`
base preserves that representation across pinned CPU and GPU storage;
this module supplies the NVFP4-specific hooks. The two global scales are
optional, represented as ``None`` entries in the storage tuple so the
base's clone/alloc/copy/accounting skip them.

Beyond inference movement, this adapter opts into a format-specific CUDA
LoRA merge plus dequantize/requantize and ``copy_into``. Contiguous
rank-two NVFP4 weights use raw Triton kernels when available, covering
regular or swizzled block scales and single- or two-level scaling without
materializing a dense weight. Unsupported representations use the
existing generic path. Both re-derive the FP8 (E4M3) block scales — and,
for two-level scaling, the global ``per_tensor_scale``. Like any merge
into a quantized base it is lossy, and NVFP4's 4-bit grid makes it coarse,
so choosing merge vs routed (non-destructive) LoRA is the caller's
tradeoff.

It does not opt into CPU round-trip or trainable ``Parameter.data`` swap:
the quant state lives in the wrapper object, not its bytes, so NVFP4
weights stay frozen for streaming/training. Routed LoRA remains the
non-destructive alternative when the owning module is a logical
``nn.Linear`` with compatible shape/dtype.
"""

from dataclasses import dataclass
from typing import Any

import torch

from ._torchao_nvfp4 import (
    create_nvfp4_tensor,
    dequantize_nvfp4_tensor,
    is_nvfp4_tensor,
    requantize_nvfp4_tensor,
    require_nvfp4_tensor,
    validate_layout,
)
from .tensor_adapters import metadata_key
from .torchao_structured_adapter import TorchaoStructuredAdapter, copy_storage_into

try:
    from ._triton_nvfp4_lora import (
        merge_nvfp4_lora as _triton_merge_nvfp4_lora,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_merge_nvfp4_lora = None


def _is_triton_nvfp4_layout(
    nv: Any,  # noqa: ANN401
    b: torch.Tensor,
    a: torch.Tensor,
) -> bool:
    """Return whether the raw NVFP4 representation has a supported layout."""
    if (
        _triton_merge_nvfp4_lora is None
        or nv.qdata.device.type != "cuda"
        or nv.qdata.dtype is not torch.uint8
        or nv.scale.dtype is not torch.float8_e4m3fn
        or nv.qdata.ndim != 2
        or nv.scale.ndim != 2
        or not nv.qdata.is_contiguous()
        or not nv.scale.is_contiguous()
        or nv.block_size != 16
        or nv.orig_dtype not in (torch.bfloat16, torch.float32)
        or b.ndim != 2
        or a.ndim != 2
        or b.dtype is not a.dtype
        or b.dtype is not nv.orig_dtype
        or nv.qdata.device != nv.scale.device
        or nv.qdata.device != b.device
        or nv.qdata.device != a.device
    ):
        return False

    if len(nv.shape) != 2:
        return False
    rows, cols = nv.shape
    rank = a.shape[0]
    if (
        rows == 0
        or cols == 0
        or rank == 0
        or cols % 16 != 0
        or tuple(nv.qdata.shape) != (rows, cols // 2)
        or b.shape != (rows, rank)
        or a.shape[1] != cols
    ):
        return False

    scale_cols = cols // nv.block_size
    expected_scale_shape = (
        (
            (rows + 127) // 128 * 32,
            (cols + 63) // 64 * 16,
        )
        if nv.is_swizzled_scales
        else (rows, scale_cols)
    )
    if tuple(nv.scale.shape) != expected_scale_shape:
        return False
    if nv.per_tensor_scale is None:
        return True
    return (
        nv.per_tensor_scale.dtype is torch.float32
        and nv.per_tensor_scale.device == nv.qdata.device
        and nv.per_tensor_scale.numel() == 1
    )


def _torch_merge_nvfp4_lora_(
    target: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> None:
    """Apply the established wrapper-based NVFP4 merge in place."""
    dense = dequantize_nvfp4_tensor(target)
    dense.addmm_(b, a, alpha=strength)
    requantized = requantize_nvfp4_tensor(
        dense,
        like=target,
        rounding_seed=rounding_seed,
    )
    Nvfp4Adapter.copy_into(requantized, target=target)


@dataclass(slots=True, frozen=True)
class _Nvfp4Meta:
    """Reconstruction metadata snapshot for a TorchAO NVFP4 tensor."""

    block_size: int
    orig_dtype: torch.dtype
    is_swizzled_scales: bool
    use_triton_kernel: bool
    act_quant_kwargs: object | None


class Nvfp4Adapter(TorchaoStructuredAdapter[_Nvfp4Meta]):
    """Adapter for TorchAO ``NVFP4Tensor`` weights."""

    _TAG = "torchao-nvfp4"
    _STORAGE_NAMES = ("qdata", "scale", "per_tensor_scale", "act_per_tensor_scale")

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_nvfp4_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_nvfp4_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale, t.per_tensor_scale, t.act_per_tensor_scale)

    @staticmethod
    def _meta_of(t: Any) -> _Nvfp4Meta:  # noqa: ANN401
        return _Nvfp4Meta(
            block_size=t.block_size,
            orig_dtype=t.orig_dtype,
            is_swizzled_scales=t.is_swizzled_scales,
            use_triton_kernel=t.use_triton_kernel,
            act_quant_kwargs=t.act_quant_kwargs,
        )

    @staticmethod
    def _reconstruct(storage: tuple[torch.Tensor | None, ...], meta: _Nvfp4Meta) -> torch.Tensor:
        qdata, scale, per_tensor_scale, act_per_tensor_scale = storage
        assert qdata is not None
        assert scale is not None
        return create_nvfp4_tensor(
            qdata,
            scale,
            meta.block_size,
            meta.orig_dtype,
            per_tensor_scale,
            act_per_tensor_scale,
            meta.is_swizzled_scales,
            meta.use_triton_kernel,
            meta.act_quant_kwargs,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (
            t.block_size,
            t.orig_dtype,
            t.is_swizzled_scales,
            t.use_triton_kernel,
            metadata_key(t.act_quant_kwargs),
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.orig_dtype

    # --- capabilities beyond inference movement ---------------------------

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_nvfp4_tensor(t)

    @staticmethod
    def requantize(
        t: torch.Tensor,
        *,
        like: torch.Tensor,
        rounding_seed: int | None = None,
    ) -> torch.Tensor:
        return requantize_nvfp4_tensor(t, like=like, rounding_seed=rounding_seed)

    @staticmethod
    def validate_lora_merge(
        target: torch.Tensor,
        _b: torch.Tensor,
        _a: torch.Tensor,
        _strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Reject layouts whose weight quantizers cannot be preserved."""
        del rounding_seed
        nv = require_nvfp4_tensor(target)
        if not nv.qdata.is_contiguous():
            raise ValueError(
                "Cannot merge LoRA into a non-contiguous (e.g. transposed) "
                "NVFP4 weight: requantization produces the standard packed "
                "layout, which cannot fill a transposed target. Use routed "
                "LoRA for this weight."
            )
        if nv.per_tensor_scale is not None and nv.per_tensor_scale.numel() != 1:
            raise ValueError(
                "Cannot merge LoRA into an NVFP4 weight with a non-scalar "
                "per_tensor_scale (e.g. per-expert grouped/MoE scales); the "
                "merge recomputes a single global scale and would drop "
                "per-group precision. Use routed LoRA for this weight."
            )

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a staged LoRA update while preserving target storage."""
        Nvfp4Adapter.validate_lora_merge(
            target,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )
        nv = require_nvfp4_tensor(target)
        if _is_triton_nvfp4_layout(nv, b, a):
            assert _triton_merge_nvfp4_lora is not None
            qdata, scale, per_tensor_scale = _triton_merge_nvfp4_lora(
                nv.qdata,
                nv.scale,
                nv.per_tensor_scale,
                nv.block_size,
                nv.is_swizzled_scales,
                b,
                a,
                strength,
                rounding_seed=rounding_seed,
            )
            nv.qdata.copy_(qdata)
            nv.scale.copy_(scale)
            if nv.per_tensor_scale is not None:
                assert per_tensor_scale is not None
                nv.per_tensor_scale.copy_(per_tensor_scale)
            return
        _torch_merge_nvfp4_lora_(
            target,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        # Fill target's present storage slots (packed FP4 + E4M3 block
        # scales, plus the optional global per-tensor/activation scales)
        # from the requantized src, preserving target's wrapper identity.
        copy_storage_into(
            Nvfp4Adapter._storage_of(require_nvfp4_tensor(src)),
            Nvfp4Adapter._storage_of(require_nvfp4_tensor(target)),
            non_blocking=False,
        )
