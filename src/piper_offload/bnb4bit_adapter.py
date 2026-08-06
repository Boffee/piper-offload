"""bitsandbytes 4-bit (``Params4bit``) adapter — NF4 and FP4.

bitsandbytes 4-bit weights are ``Params4bit`` tensors (an ``nn.Parameter``
subclass) carrying a ``QuantState``: a packed ``uint8`` weight (two 4-bit
codes per byte) plus per-block ``absmax`` scales, a 16-entry codebook
(``quant_map``), and — when double-quant is on — a nested second-level
``absmax``/codebook/offset. The wrapper does not support ``p.data = ...``
storage swap: its quant state is part of the wrapped object, not its
bytes. So this adapter, like :class:`~piper_offload.quanto_adapter.QuantoAdapter`:

- Decomposes ``Params4bit`` into the packed weight + the quant-state
  tensors (via ``QuantState.as_dict(packed=True)``), pinning each.
- Reconstructs a fresh ``Params4bit`` (and thus a fresh
  :class:`nn.Parameter`) on each activate via registry replacement.
  PyTorch optimizers keyed by the user's pre-wrap Parameter id are
  orphaned across cycles — 4-bit weights are inference-only.

The mechanics are codebook- and nesting-agnostic: movement carries
whatever tensors the quant state serializes, so one adapter covers NF4
and FP4, double-quant on or off. The fixed scalar metadata (blocksize,
shape, dtype, quant type) rides in a small packed blob that bitsandbytes
keeps host-resident; only the packed weight and the per-block scales DMA
to the GPU.

CUDA LoRA updates use a raw-storage Triton merge for standard 64-value
blocks whose physical storage contains exactly two logical values per byte.
This covers NF4 and FP4, ordinary and nested scales, and byte-aligned
uint8/fp16/bf16/fp32 quant storage. Other layouts keep the precise generic
dequantize/addmm/requantize path.

Reaches into bitsandbytes' 4-bit layout through :mod:`._bnb`; if
bitsandbytes refactors, the pin/read paths fail with a clear validation
error (``require_params_4bit`` checks the expected attributes before reading
the quant state). :meth:`Bnb4bitAdapter.matches` stays pure type recognition
so an unquantized placeholder can still serve as a bind target.

Selected by :mod:`tensor_adapter_registry`. Importing fails silently if
bitsandbytes is not installed — 4-bit support is optional.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ._bnb import (
    build_params_4bit,
    dequantize_params_4bit,
    is_params_4bit,
    metadata_blob_key,
    quant_stats,
    requantize_params_4bit,
    require_params_4bit,
)
from .tensor_adapters import (
    adopt_cpu_storage,
    clone_to_pinned_cpu,
    empty_like_strided,
    optional_tensor_id,
    tensor_layout,
)

try:
    from ._triton_bnb4_lora import (
        merge_bnb4_lora as _triton_merge_bnb4_lora,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_merge_bnb4_lora = None

_TRITON_STORAGE_DTYPES = (
    torch.uint8,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


def _torch_merge_bnb4_lora(
    target: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Merge through the ordinary dequantize/addmm/requantize path."""
    dense = dequantize_params_4bit(target)
    dense.addmm_(b, a, alpha=strength)
    return requantize_params_4bit(dense, like=target)


def _can_use_triton_merge(
    qt: Any,  # noqa: ANN401
    b: torch.Tensor,
    a: torch.Tensor,
) -> bool:
    state = qt.quant_state
    logical_shape = tuple(state.shape)
    if len(logical_shape) != 2:
        return False
    rows, cols = logical_shape
    numel = rows * cols
    num_blocks = (numel + state.blocksize - 1) // state.blocksize
    common = (
        _triton_merge_bnb4_lora is not None
        and qt.data.device.type == "cuda"
        and qt.data.dtype in _TRITON_STORAGE_DTYPES
        and qt.data.is_contiguous()
        and qt.data.storage_offset() == 0
        and cols % 64 == 0
        and numel % 2 == 0
        and qt.data.numel() * qt.data.element_size() == numel // 2
        and state.quant_type in ("nf4", "fp4")
        and state.blocksize == 64
        and state.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and state.code.dtype is torch.float32
        and state.code.numel() == 16
        and state.absmax.numel() == num_blocks
        and b.ndim == 2
        and a.ndim == 2
        and b.dtype is state.dtype
        and a.dtype is state.dtype
        and b.shape == (rows, a.shape[0])
        and a.shape[1] == cols
        and qt.data.device == state.absmax.device == state.code.device
        and qt.data.device == b.device == a.device
    )
    if not common:
        return False
    if not state.nested:
        return state.absmax.dtype is torch.float32

    state2 = state.state2
    offset = state.offset
    expected_nested_blocks = (num_blocks + 255) // 256
    return (
        state.absmax.dtype is torch.uint8
        and state2 is not None
        and state2.blocksize == 256
        and state2.dtype is torch.float32
        and state2.absmax.dtype is torch.float32
        and state2.absmax.numel() == expected_nested_blocks
        and state2.code.dtype is torch.float32
        and state2.code.numel() == 256
        and isinstance(offset, torch.Tensor)
        and offset.dtype is torch.float32
        and offset.numel() == 1
        and qt.data.device
        == state2.absmax.device
        == state2.code.device
        == offset.device
    )


@dataclass(slots=True)
class _Bnb4bitPinned:
    """Pinned-CPU state for a 4-bit tensor: the packed weight, the
    per-block scale tensors (keyed as bitsandbytes serializes them), and
    the small host-resident metadata blob needed to rebuild the wrapper."""

    data: torch.Tensor                    # pinned uint8 packed weight
    buffers: dict[str, torch.Tensor]      # pinned absmax / quant_map / nested_*
    blob_key: str                         # key of the packed metadata blob
    blob: torch.Tensor                    # uint8 scalar-metadata (host-resident)
    offset: torch.Tensor | None           # nested double-quant offset (data-dependent)


@dataclass(slots=True)
class _Bnb4bitGpu:
    """GPU state for a 4-bit tensor: the packed weight + per-block scale
    tensors. Scalar metadata stays in the originating :class:`_Bnb4bitPinned`;
    only storage moves to GPU."""

    data: torch.Tensor
    buffers: dict[str, torch.Tensor]
    offset: torch.Tensor | None


def _gpu_stats(
    pinned: _Bnb4bitPinned, buffers: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Assemble the packed stats dict ``from_prequantized`` expects:
    the (GPU or pinned) scale buffers plus the host-resident blob."""
    return {**buffers, pinned.blob_key: pinned.blob}


class Bnb4bitAdapter:
    """Adapter for ``bitsandbytes.nn.Params4bit`` (NF4 / FP4).

    Decompose-on-pin, reconstruct-on-move. Each activate creates a fresh
    ``Params4bit`` and a fresh :class:`nn.Parameter`, installed via
    registry replacement. This breaks PyTorch optimizer references —
    4-bit weights are inference-only.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        # Pure type recognition — never raises. An unquantized Params4bit
        # (quant_state is None) is a legitimate *bind target*: a config-built
        # meta skeleton carries one, and binding replaces it wholesale with a
        # store-reconstructed Params4bit. The layout validation that *reading*
        # the quant state needs lives in require_params_4bit, so it still fires
        # on every pin/read path (clone_pin, layout_signature, tensor_id, …) —
        # where a real tensor must be fully quantized — without rejecting a
        # placeholder that is only ever bound, never pinned.
        return is_params_4bit(t)

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple:
        # Composite identity: the packed weight plus the quant-state
        # tensors plus the scalar quant params. Two Params4bit sharing the
        # same packed storage AND matching quant params are the same
        # logical tensor and dedup safely. Read the live quant_state
        # tensors directly (not as_dict, which mints a fresh metadata blob
        # on each call and would perturb identity).
        qt = require_params_4bit(t)
        quant_state = qt.quant_state
        nested = quant_state.nested
        state2 = quant_state.state2 if nested else None
        return (
            "bnb4bit",
            optional_tensor_id(qt.data),
            optional_tensor_id(quant_state.absmax),
            optional_tensor_id(quant_state.code),
            optional_tensor_id(getattr(state2, "absmax", None)),
            optional_tensor_id(getattr(state2, "code", None)),
            optional_tensor_id(quant_state.offset if nested else None),
            quant_state.quant_type,
            quant_state.blocksize,
            tuple(quant_state.shape),
            quant_state.dtype,
            nested,
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple:
        qt = require_params_4bit(t)
        quant_state = qt.quant_state
        state2 = quant_state.state2 if quant_state.nested else None
        return (
            tuple(quant_state.shape),
            quant_state.dtype,
            quant_state.quant_type,
            quant_state.blocksize,
            quant_state.nested,
            ("data", tensor_layout(qt.data)),
            ("absmax", tensor_layout(quant_state.absmax)),
            ("code", tensor_layout(quant_state.code)),
            ("nested_absmax", tensor_layout(getattr(state2, "absmax", None))),
            ("nested_code", tensor_layout(getattr(state2, "code", None))),
        )

    @staticmethod
    def bind_layout_signature(t: torch.Tensor) -> tuple:
        # Relaxed counterpart of layout_signature for store↔module bind
        # validation. Bind replaces the placeholder with a store-reconstructed
        # Params4bit — its packed data *and* quant_state — so every
        # store-supplied field (dtype, quant_type, blocksize, nesting, and the
        # packed/scale tensor layouts) is overwritten and carries no
        # information past validation, exactly as RegularAdapter drops dtype.
        # Only the logical [out, in] shape is structural: read it from
        # quant_state for a real param (whose ``.shape`` is the *packed*
        # (numel/2, 1) storage) and from ``.shape`` for an unquantized
        # placeholder (quant_state is None — its data is still logical).
        quant_state = getattr(t, "quant_state", None)
        shape = tuple(quant_state.shape) if quant_state is not None else tuple(t.shape)
        return (shape,)

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _Bnb4bitPinned:
        return Bnb4bitAdapter._host_state(t, clone_to_pinned_cpu)

    @staticmethod
    def adopt_host(t: torch.Tensor) -> _Bnb4bitPinned:
        return Bnb4bitAdapter._host_state(t, adopt_cpu_storage)

    @staticmethod
    def _host_state(
        t: torch.Tensor,
        clone: Callable[..., torch.Tensor],
    ) -> _Bnb4bitPinned:
        qt = require_params_4bit(t)
        quant_state = qt.quant_state
        stats = quant_stats(qt)
        blob_key = metadata_blob_key(stats)
        return _Bnb4bitPinned(
            data=clone(
                qt.data, memory_format=torch.contiguous_format
            ),
            buffers={
                key: clone(
                    value, memory_format=torch.contiguous_format
                )
                for key, value in stats.items()
                if key != blob_key
            },
            blob_key=blob_key,
            # Metadata blob is tiny, constant, and host-resident — pin it
            # once and inject it at reconstruction; it never DMAs.
            blob=clone(
                stats[blob_key], memory_format=torch.contiguous_format
            ),
            # The nested double-quant offset is data-dependent but lives in
            # the (host-resident, template-block) blob, so a wrapper reused
            # across pooled streamed blocks would keep the wrong offset. Pin
            # it separately and alias it per load like the other scales.
            offset=(
                clone(
                    quant_state.offset, memory_format=torch.contiguous_format
                )
                if quant_state.nested
                else None
            ),
        )

    @staticmethod
    def cpu_param(
        state: _Bnb4bitPinned, *, requires_grad: bool = False
    ) -> nn.Parameter:
        # Params4bit is itself an nn.Parameter subclass.
        return build_params_4bit(
            state.data,
            _gpu_stats(state, state.buffers),
            device=state.data.device,
            requires_grad=requires_grad,
        )

    @staticmethod
    def alloc_gpu(state: _Bnb4bitPinned, device: torch.device) -> _Bnb4bitGpu:
        return _Bnb4bitGpu(
            data=empty_like_strided(state.data, device),
            buffers={
                key: empty_like_strided(value, device)
                for key, value in state.buffers.items()
            },
            offset=(
                empty_like_strided(state.offset, device)
                if state.offset is not None
                else None
            ),
        )

    @staticmethod
    def gpu_param(
        pinned: _Bnb4bitPinned,
        gpu_state: _Bnb4bitGpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        # Scalar metadata comes from the pinned blob; the packed weight and
        # scale tensors come from the (pre-allocated) GPU side. Reconstruction
        # aliases those buffers, so the later copy_to_gpu DMA is visible here.
        param = build_params_4bit(
            gpu_state.data,
            _gpu_stats(pinned, gpu_state.buffers),
            device=gpu_state.data.device,
            requires_grad=requires_grad,
        )
        if gpu_state.offset is not None:
            # from_prequantized baked the offset from the (template) blob;
            # re-point it at the per-load DMA'd buffer so a wrapper reused
            # across pooled streamed blocks sees each block's own offset
            # (the absmax/nested_absmax buffers already alias this way).
            param.quant_state.offset = gpu_state.offset
        return param

    @staticmethod
    def copy_to_gpu(
        src: _Bnb4bitPinned, dst: _Bnb4bitGpu, *, non_blocking: bool = False
    ) -> None:
        dst.data.copy_(src.data, non_blocking=non_blocking)
        for key, value in src.buffers.items():
            dst.buffers[key].copy_(value, non_blocking=non_blocking)
        if src.offset is not None and dst.offset is not None:
            dst.offset.copy_(src.offset, non_blocking=non_blocking)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        return require_params_4bit(t).quant_state.dtype

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        return tuple(require_params_4bit(t).quant_state.shape)

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_params_4bit(t)

    @staticmethod
    def requantize(t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        return requantize_params_4bit(t, like=like)

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
    ) -> None:
        """Merge into BNB4 storage, preferring the raw Triton path."""
        qt = require_params_4bit(target)
        if _can_use_triton_merge(qt, b, a):
            assert _triton_merge_bnb4_lora is not None
            state = qt.quant_state
            state2 = state.state2 if state.nested else None
            packed, absmax, nested_absmax, offset = _triton_merge_bnb4_lora(
                qt.data.view(torch.uint8).reshape(-1),
                state.absmax,
                state.code,
                state2.absmax if state2 is not None else None,
                state2.code if state2 is not None else None,
                state.offset if state.nested else None,
                tuple(state.shape),
                state.blocksize,
                state.quant_type,
                b,
                a,
                strength,
            )
            qt.data.view(torch.uint8).reshape(-1).copy_(packed)
            state.absmax.copy_(absmax)
            if state.nested:
                assert state2 is not None
                assert nested_absmax is not None
                assert offset is not None
                state2.absmax.copy_(nested_absmax)
                state.offset.copy_(offset)
            return

        merged = _torch_merge_bnb4_lora(target, b, a, strength)
        Bnb4bitAdapter.copy_into(merged, target=target)

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        # Preserve target identity/storage: overwrite the packed weight and
        # the data-dependent block scales in place. The codebook
        # (quant_map / code) is fixed by quant_type and shared, so it needs
        # no copy.
        target_qt = require_params_4bit(target)
        src_qt = require_params_4bit(src)
        target_qt.data.copy_(src_qt.data)
        target_qt.quant_state.absmax.copy_(src_qt.quant_state.absmax)
        if target_qt.quant_state.nested:
            target_qt.quant_state.state2.absmax.copy_(
                src_qt.quant_state.state2.absmax
            )
            target_qt.quant_state.offset.copy_(src_qt.quant_state.offset)

    @staticmethod
    def cache_bytes(state: _Bnb4bitPinned) -> int:
        total = state.data.numel() * state.data.element_size()
        total += state.blob.numel() * state.blob.element_size()
        for value in state.buffers.values():
            total += value.numel() * value.element_size()
        if state.offset is not None:
            total += state.offset.numel() * state.offset.element_size()
        return total
