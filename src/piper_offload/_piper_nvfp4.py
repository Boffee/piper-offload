"""Internal optional-import boundary for Piper NVFP4 and ConvRot NVFP4 support.

``piper-kernels`` owns :class:`PiperNVFP4Tensor` and :class:`ConvRotNVFP4Tensor`,
including dequantization, linear dispatch, grouped rotation, and in-place
``addmm_`` / ``add_`` updates. Piper Offload preserves their public packed
storage and metadata during movement and delegates updates, including base
scaling, to the public wrapper methods.

``piper-kernels`` is a required dependency; TorchAO is not. Importing these
wrappers reaches TorchAO's NVFP4 encoder, so imports fail without the
``torchao`` extra and the adapters report the formats unavailable.
"""

from typing import Any, cast

import torch

NVFP4_LAYOUT_ATTRS = (
    "qdata",
    "scale",
    "block_size",
    "orig_dtype",
    "per_tensor_scale",
    "act_per_tensor_scale",
    "is_swizzled_scales",
    "use_triton_kernel",
    "act_quant_kwargs",
    "high_first",
)
"""Public ``PiperNVFP4Tensor`` fields preserved by Piper Offload."""

CONVROT_NVFP4_LAYOUT_ATTRS = (*NVFP4_LAYOUT_ATTRS, "group_size")
"""Public ``ConvRotNVFP4Tensor`` fields preserved by Piper Offload."""


try:
    from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor

    PIPER_NVFP4_AVAILABLE = True
except ImportError:
    PIPER_NVFP4_AVAILABLE = False
    PiperNVFP4Tensor: Any = None

try:
    from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor

    PIPER_CONVROT_NVFP4_AVAILABLE = True
except ImportError:
    PIPER_CONVROT_NVFP4_AVAILABLE = False
    ConvRotNVFP4Tensor: Any = None


def is_piper_nvfp4_tensor(t: object) -> bool:
    """Return whether ``t`` is a plain Piper ``PiperNVFP4Tensor`` (not ConvRot)."""
    if not PIPER_NVFP4_AVAILABLE or not isinstance(t, PiperNVFP4Tensor):
        return False
    return not is_convrot_nvfp4_tensor(t)


def is_convrot_nvfp4_tensor(t: object) -> bool:
    """Return whether ``t`` is a Piper ``ConvRotNVFP4Tensor``."""
    return PIPER_CONVROT_NVFP4_AVAILABLE and isinstance(t, ConvRotNVFP4Tensor)


def _create_nvfp4_wrapper(  # noqa: PLR0913
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
    orig_dtype: torch.dtype,
    per_tensor_scale: torch.Tensor | None,
    act_per_tensor_scale: torch.Tensor | None,
    is_swizzled_scales: bool,
    use_triton_kernel: bool,
    act_quant_kwargs: object | None,
    high_first: bool,
    *,
    is_convrot: bool,
    group_size: int | None = None,
    wrapper_type: type[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Rebuild a ``PiperNVFP4Tensor`` or ``ConvRotNVFP4Tensor`` wrapper."""
    if is_convrot:
        if not PIPER_CONVROT_NVFP4_AVAILABLE:
            raise RuntimeError(
                "piper-offload[torchao] is required to create a ConvRotNVFP4Tensor"
            )
        constructor = cast(
            Any,
            ConvRotNVFP4Tensor if wrapper_type is None else wrapper_type,
        )
        if not issubclass(constructor, ConvRotNVFP4Tensor):
            raise TypeError(
                "ConvRot NVFP4 wrapper type must subclass ConvRotNVFP4Tensor, "
                f"got {constructor.__name__}"
            )
        return constructor(
            qdata=qdata,
            scale=scale,
            block_size=block_size,
            orig_dtype=orig_dtype,
            group_size=group_size,
            per_tensor_scale=per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale,
            is_swizzled_scales=is_swizzled_scales,
            use_triton_kernel=use_triton_kernel,
            act_quant_kwargs=act_quant_kwargs,
            high_first=high_first,
        )

    if not PIPER_NVFP4_AVAILABLE:
        raise RuntimeError(
            "piper-offload[torchao] is required to create a PiperNVFP4Tensor"
        )
    constructor = cast(Any, PiperNVFP4Tensor if wrapper_type is None else wrapper_type)
    if not issubclass(constructor, PiperNVFP4Tensor) or (
        isinstance(ConvRotNVFP4Tensor, type)
        and issubclass(constructor, ConvRotNVFP4Tensor)
    ):
        raise TypeError(
            "NVFP4 wrapper type must subclass PiperNVFP4Tensor "
            f"(not ConvRotNVFP4Tensor), got {constructor.__name__}"
        )
    return constructor(
        qdata=qdata,
        scale=scale,
        block_size=block_size,
        orig_dtype=orig_dtype,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=act_per_tensor_scale,
        is_swizzled_scales=is_swizzled_scales,
        use_triton_kernel=use_triton_kernel,
        act_quant_kwargs=act_quant_kwargs,
        high_first=high_first,
    )


def create_piper_nvfp4_tensor(  # noqa: PLR0913
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
    orig_dtype: torch.dtype,
    per_tensor_scale: torch.Tensor | None,
    act_per_tensor_scale: torch.Tensor | None,
    is_swizzled_scales: bool,
    use_triton_kernel: bool,
    act_quant_kwargs: object | None,
    high_first: bool,
    *,
    wrapper_type: type[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Rebuild a ``PiperNVFP4Tensor`` wrapper from storage and metadata."""
    return _create_nvfp4_wrapper(
        qdata,
        scale,
        block_size,
        orig_dtype,
        per_tensor_scale,
        act_per_tensor_scale,
        is_swizzled_scales,
        use_triton_kernel,
        act_quant_kwargs,
        high_first,
        is_convrot=False,
        wrapper_type=wrapper_type,
    )


def create_convrot_nvfp4_tensor(  # noqa: PLR0913, PLR0917
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
    orig_dtype: torch.dtype,
    group_size: int,
    per_tensor_scale: torch.Tensor | None,
    act_per_tensor_scale: torch.Tensor | None,
    is_swizzled_scales: bool,
    use_triton_kernel: bool,
    act_quant_kwargs: object | None,
    high_first: bool,
    *,
    wrapper_type: type[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Rebuild a ``ConvRotNVFP4Tensor`` wrapper from storage and metadata."""
    return _create_nvfp4_wrapper(
        qdata,
        scale,
        block_size,
        orig_dtype,
        per_tensor_scale,
        act_per_tensor_scale,
        is_swizzled_scales,
        use_triton_kernel,
        act_quant_kwargs,
        high_first,
        is_convrot=True,
        group_size=group_size,
        wrapper_type=wrapper_type,
    )


def _validate_nvfp4_layout(
    t: torch.Tensor,
    *,
    attrs: tuple[str, ...],
    class_name: str,
    with_group_size: bool,
) -> None:
    missing = [attr for attr in attrs if not hasattr(t, attr)]
    if missing:
        raise RuntimeError(
            f"{class_name} is missing expected attributes {missing!r}; "
            f"piper-offload expects the public layout {attrs}. "
            "Upgrade piper-offload to match piper-kernels."
        )
    wrapped: Any = t
    _create_nvfp4_wrapper(
        wrapped.qdata,
        wrapped.scale,
        wrapped.block_size,
        wrapped.orig_dtype,
        wrapped.per_tensor_scale,
        wrapped.act_per_tensor_scale,
        wrapped.is_swizzled_scales,
        wrapped.use_triton_kernel,
        wrapped.act_quant_kwargs,
        wrapped.high_first,
        is_convrot=with_group_size,
        group_size=wrapped.group_size if with_group_size else None,
        wrapper_type=type(wrapped),
    )


def validate_layout(t: torch.Tensor) -> None:
    """Validate the public ``PiperNVFP4Tensor`` storage contract."""
    _validate_nvfp4_layout(
        t,
        attrs=NVFP4_LAYOUT_ATTRS,
        class_name="PiperNVFP4Tensor",
        with_group_size=False,
    )


def validate_convrot_nvfp4_layout(t: torch.Tensor) -> None:
    """Validate the public ``ConvRotNVFP4Tensor`` storage contract."""
    _validate_nvfp4_layout(
        t,
        attrs=CONVROT_NVFP4_LAYOUT_ATTRS,
        class_name="ConvRotNVFP4Tensor",
        with_group_size=True,
    )


def require_piper_nvfp4_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated ``PiperNVFP4Tensor``, or raise."""
    if not is_piper_nvfp4_tensor(t):
        raise TypeError(
            f"expected piper_kernels.weights.nvfp4.PiperNVFP4Tensor, got {type(t).__name__}"
        )
    validate_layout(t)
    return t


def require_convrot_nvfp4_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated ``ConvRotNVFP4Tensor``, or raise."""
    if not is_convrot_nvfp4_tensor(t):
        raise TypeError(
            "expected piper_kernels.weights.convrot.nvfp4.ConvRotNVFP4Tensor, "
            f"got {type(t).__name__}"
        )
    validate_convrot_nvfp4_layout(t)
    return t


def validate_nvfp4_matrix(tensor: Any, *, label: str) -> Any:  # noqa: ANN401
    """Enforce 2-D untransposed contiguous NVFP4 layout invariants before updates."""
    if tensor.ndim != 2:
        raise ValueError(f"{label} weight updates require a 2-D weight, got {tensor.ndim}-D")
    if tensor.orig_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"{label} weight updates require float16, bfloat16, or float32 orig_dtype, "
            f"got {tensor.orig_dtype}"
        )
    if tensor.transposed or not tensor.qdata.is_contiguous() or not tensor.scale.is_contiguous():
        raise ValueError(f"{label} weight updates require contiguous untransposed packed storage")
    if tensor.block_size != 16:
        raise ValueError(f"{label} weight updates require block_size=16, got {tensor.block_size}")
    rows, features = tensor.shape
    if rows <= 0 or features <= 0 or features % 16 != 0:
        raise ValueError(
            f"{label} weight updates require nonempty block-16 storage, got shape {tuple(tensor.shape)}"
        )
    expected_qdata = (rows, features // 2)
    if tensor.qdata.dtype is not torch.uint8 or tuple(tensor.qdata.shape) != expected_qdata:
        raise ValueError(
            f"{label} weight updates require uint8 qdata with shape {expected_qdata}, "
            f"got {tensor.qdata.dtype} {tuple(tensor.qdata.shape)}"
        )
    expected_scale = (
        (((rows + 127) // 128) * 32, ((features + 63) // 64) * 16)
        if tensor.is_swizzled_scales
        else (rows, features // 16)
    )
    if tensor.scale.dtype is not torch.float8_e4m3fn or tuple(tensor.scale.shape) != expected_scale:
        raise ValueError(
            f"{label} weight updates require float8_e4m3fn scales with shape {expected_scale}, "
            f"got {tensor.scale.dtype} {tuple(tensor.scale.shape)}"
        )
    if tensor.per_tensor_scale is not None and tensor.per_tensor_scale.numel() != 1:
        raise ValueError(f"{label} weight updates require a scalar per_tensor_scale")
    storage = (tensor.qdata, tensor.scale, tensor.per_tensor_scale, tensor.act_per_tensor_scale)
    if any(value is not None and value.device != tensor.device for value in storage):
        raise ValueError(f"{label} weight updates require storage tensors on the same device")
    return tensor


def require_piper_nvfp4_matrix(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Require a 2-D contiguous untransposed ``PiperNVFP4Tensor`` before updates."""
    return validate_nvfp4_matrix(require_piper_nvfp4_tensor(t), label="Piper NVFP4")


def require_convrot_nvfp4_matrix(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Require a 2-D contiguous untransposed ``ConvRotNVFP4Tensor`` before updates."""
    return validate_nvfp4_matrix(require_convrot_nvfp4_tensor(t), label="ConvRot NVFP4")
