"""bitsandbytes 8-bit (``Int8Params`` / LLM.int8) adapter.

bitsandbytes 8-bit weights are ``Int8Params`` tensors (an ``nn.Parameter``
subclass) carrying two pieces: ``CB`` (the row-major int8 weight, also
``weight.data``) and ``SCB`` (the per-output-row float32 scale). Like
:class:`~piper_offload.bnb4bit_adapter.Bnb4bitAdapter` this is a
decompose-on-capture / reconstruct-on-move, inference-only adapter: each
activate builds a fresh ``Int8Params`` installed via registry replacement,
so optimizers keyed on the pre-wrap Parameter are orphaned.

Two things differ from 4-bit, both because int8 keeps no self-describing
``QuantState``:

- The representation is just two tensors (``CB`` + ``SCB``) — no packed
  metadata blob, no codebook, no nested double-quant. Movement is simpler.
- ``Int8Params`` carries ``CB``/``SCB`` only **before** the owning
  ``Linear8bitLt`` runs its first forward. The first forward migrates them
  onto the module's ``MatmulLtState`` and nulls them on the weight. A
  tensor-scoped adapter cannot see the module, so it must capture a pre-forward
  weight; the capture/read path (``require_int8_params``) raises on a post-forward
  weight rather than silently capturing ``None``, while
  :meth:`Bnb8bitAdapter.matches` stays pure type recognition so an unquantized
  placeholder can still serve as a bind target.

This is safe across activations because the reconstructed ``Int8Params``
carries ``CB``, so each forward re-runs ``init_8bit_state`` and repopulates
the module state from the freshly-installed weight — the offloader never
has to touch the owning module.

Reaches into bitsandbytes' int8 layout through :mod:`._bnb`; if
bitsandbytes refactors, the capture/read path fails with a clear validation error
(``require_int8_params`` checks the layout before reading CB/SCB). Selected by
:mod:`tensor_adapter_registry`. Importing fails silently if bitsandbytes is
not installed — 8-bit support is optional.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ._bnb import (
    build_int8_params,
    dequantize_int8_params,
    is_int8_params,
    requantize_int8_params,
    require_int8_params,
)
from ._dense_merge import merge_dense_requantize_
from .tensor_adapters import (
    clone_to_host_cpu,
    empty_like_strided,
    optional_tensor_id,
    tensor_layout,
)

try:
    from ._triton_bnb8_lora import (
        merge_bnb8_dense as _triton_merge_bnb8_dense,
    )
    from ._triton_bnb8_lora import (
        merge_bnb8_lora as _triton_merge_bnb8_lora,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_merge_bnb8_dense = None
    _triton_merge_bnb8_lora = None

_TRITON_MAX_COLUMNS = 8_388_608


def _torch_merge_bnb8_lora(
    target: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge through the ordinary dequantize/addmm/requantize path."""
    dense = dequantize_int8_params(target)
    dense.addmm_(b, a, alpha=strength)
    merged = require_int8_params(
        requantize_int8_params(
            dense,
            like=target,
            rounding_seed=rounding_seed,
        )
    )
    return merged.CB, merged.SCB


def _can_use_triton_merge(
    cb: torch.Tensor,
    scb: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
) -> bool:
    if not (
        _triton_merge_bnb8_lora is not None
        and cb.device.type == "cuda"
        and cb.dtype is torch.int8
        and scb.dtype is torch.float32
        and b.dtype is torch.float16
        and a.dtype is torch.float16
        and cb.device == scb.device == b.device == a.device
        and cb.ndim == b.ndim == a.ndim == 2
    ):
        return False

    rows, cols = cb.shape
    rank = a.shape[0]
    return (
        rows > 0
        and 0 < cols <= _TRITON_MAX_COLUMNS
        and rank > 0
        and scb.numel() == rows
        and b.shape == (rows, rank)
        and a.shape[1] == cols
    )


def _can_use_triton_dense_merge(
    cb: torch.Tensor,
    scb: torch.Tensor,
    update: torch.Tensor,
) -> bool:
    if not (
        _triton_merge_bnb8_dense is not None
        and cb.device.type == "cuda"
        and cb.dtype is torch.int8
        and scb.dtype is torch.float32
        and update.dtype is torch.float16
        and cb.device == scb.device == update.device
        and cb.ndim == update.ndim == 2
    ):
        return False
    rows, cols = cb.shape
    return rows > 0 and 0 < cols <= _TRITON_MAX_COLUMNS and scb.numel() == rows and tuple(update.shape) == (rows, cols)


@dataclass(slots=True)
class _Bnb8bitHost:
    """CPU state for an 8-bit tensor: the int8 weight (``CB``) and
    the per-row float32 scale (``SCB``)."""

    data: torch.Tensor  # host int8 CB (== weight.data)
    scb: torch.Tensor   # host float32 per-row scale


@dataclass(slots=True)
class _Bnb8bitGpu:
    """GPU state for an 8-bit tensor: the int8 weight + the per-row scale."""

    data: torch.Tensor
    scb: torch.Tensor


class Bnb8bitAdapter:
    """Adapter for ``bitsandbytes.nn.Int8Params`` (LLM.int8).

    Decompose-on-capture (``CB`` + ``SCB``), reconstruct a fresh ``Int8Params``
    on each activate. Inference-only. Must capture a pre-forward weight — int8
    quant state migrates onto the owning module after the first forward,
    out of a tensor-scoped adapter's reach.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        # Pure type recognition — never raises. An Int8Params with CB/SCB None
        # (an unquantized meta-skeleton placeholder) is a legitimate *bind
        # target*: binding replaces it wholesale with a store-reconstructed
        # Int8Params. The pre-forward / quantized validation that *reading*
        # CB/SCB needs lives in require_int8_params, so it still fires on every
        # capture/read path (capture_host, layout_signature, tensor_id, …) — rejecting
        # a post-forward or unquantized weight there — without rejecting a
        # placeholder that is only ever bound, never host.
        return is_int8_params(t)

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple:
        # Composite identity: the int8 weight + the per-row scale. Two
        # Int8Params sharing the same CB/SCB storage are the same logical
        # tensor and dedup safely.
        qt = require_int8_params(t)
        # CB is the logical weight tensor; optional_tensor_id already embeds
        # its shape, so no separate shape field is needed (cf. 4-bit, where
        # the packed qt.data shape differs from the logical quant_state.shape).
        return (
            "bnb8bit",
            optional_tensor_id(qt.CB),
            optional_tensor_id(qt.SCB),
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple:
        qt = require_int8_params(t)
        return (
            tuple(qt.CB.shape),
            ("CB", tensor_layout(qt.CB)),
            ("SCB", tensor_layout(qt.SCB)),
        )

    @staticmethod
    def bind_layout_signature(t: torch.Tensor) -> tuple:
        # Relaxed counterpart of layout_signature for store↔module bind
        # validation. Bind replaces the placeholder with a store-reconstructed
        # Int8Params (CB + SCB), so those tensors' layouts are store-supplied
        # and carry no information past validation, as RegularAdapter drops
        # dtype. Only the logical [out, in] shape is structural — and unlike
        # 4-bit it is just ``t.shape`` in every state: int8 is full-byte (no
        # sub-byte packing), so ``.shape`` is logical whether CB is present
        # (real, ``.shape == CB.shape``) or None (unquantized placeholder). The
        # only CB-None tensor that reaches bind is the skeleton placeholder; a
        # post-forward real weight is rejected earlier on the capture/target path.
        return (tuple(t.shape),)

    @staticmethod
    def capture_host(
        t: torch.Tensor,
    ) -> _Bnb8bitHost:
        qt = require_int8_params(t)
        return _Bnb8bitHost(
            data=clone_to_host_cpu(
                qt.CB, memory_format=torch.contiguous_format
            ),
            scb=clone_to_host_cpu(
                qt.SCB, memory_format=torch.contiguous_format
            ),
        )

    @staticmethod
    def storage_tensors(state: _Bnb8bitHost) -> tuple[torch.Tensor, ...]:
        return (state.data, state.scb)

    @staticmethod
    def cpu_param(
        state: _Bnb8bitHost, *, requires_grad: bool = False
    ) -> nn.Parameter:
        # Int8Params is itself an nn.Parameter subclass.
        return build_int8_params(
            state.data, state.scb, requires_grad=requires_grad
        )

    @staticmethod
    def alloc_gpu(state: _Bnb8bitHost, device: torch.device) -> _Bnb8bitGpu:
        return _Bnb8bitGpu(
            data=empty_like_strided(state.data, device),
            scb=empty_like_strided(state.scb, device),
        )

    @staticmethod
    def gpu_param(
        host: _Bnb8bitHost,
        gpu_state: _Bnb8bitGpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        _ = host
        # int8 carries no host-resident metadata (unlike the 4-bit blob), so
        # the wrapper is built purely from the GPU buffers. Reconstruction
        # aliases them, so the later copy_to_gpu DMA is visible here, and the
        # rebuilt wrapper carries CB so the next forward re-inits module state.
        return build_int8_params(
            gpu_state.data, gpu_state.scb, requires_grad=requires_grad
        )

    @staticmethod
    def copy_to_gpu(
        src: _Bnb8bitHost, dst: _Bnb8bitGpu, *, non_blocking: bool = False
    ) -> None:
        dst.data.copy_(src.data, non_blocking=non_blocking)
        dst.scb.copy_(src.scb, non_blocking=non_blocking)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        require_int8_params(t)
        # LLM.int8 dequantizes/accumulates in fp16; there is no module-level
        # compute_dtype the routed-LoRA path can read off Linear8bitLt.
        return torch.float16

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        return tuple(require_int8_params(t).CB.shape)

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_int8_params(t)

    @staticmethod
    def requantize(
        t: torch.Tensor,
        *,
        like: torch.Tensor,
        rounding_seed: int | None = None,
    ) -> torch.Tensor:
        return requantize_int8_params(t, like=like, rounding_seed=rounding_seed)

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge into BNB8 storage, preferring the raw Triton path."""
        qt = require_int8_params(target)
        if _can_use_triton_merge(qt.CB, qt.SCB, b, a):
            assert _triton_merge_bnb8_lora is not None
            merged = _triton_merge_bnb8_lora(
                qt.CB,
                qt.SCB,
                b,
                a,
                strength,
                rounding_seed=rounding_seed,
            )
        else:
            merged = _torch_merge_bnb8_lora(
                target,
                b,
                a,
                strength,
                rounding_seed=rounding_seed,
            )
        cb, scb = merged
        qt.CB.copy_(cb)
        qt.SCB.copy_(scb)

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a full-rank update, preferring the raw Triton path."""
        qt = require_int8_params(target)
        if _can_use_triton_dense_merge(qt.CB, qt.SCB, update):
            assert _triton_merge_bnb8_dense is not None
            cb, scb = _triton_merge_bnb8_dense(
                qt.CB,
                qt.SCB,
                update,
                strength,
                rounding_seed=rounding_seed,
            )
            qt.CB.copy_(cb)
            qt.SCB.copy_(scb)
            return
        merge_dense_requantize_(
            Bnb8bitAdapter,
            target,
            update,
            strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def rearm_after_load(param: nn.Parameter, gpu_state: _Bnb8bitGpu) -> None:
        # The previous forward migrated CB/SCB onto the owning module and
        # nulled them here; re-point them at the freshly-loaded buffers so the
        # next forward re-runs init_8bit_state from the current data. Pool
        # buffers are shared across streamed blocks, so without this the
        # migrated module state would read another block's bytes. CB is None
        # at this point (post-forward), so this cannot go through the
        # CB-present validation in require_int8_params.
        wrapper: Any = param
        wrapper.CB = gpu_state.data
        wrapper.SCB = gpu_state.scb

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        # Preserve target identity/storage: overwrite CB and SCB in place.
        # CB and weight.data alias the same storage, so copying CB updates
        # both.
        target_qt = require_int8_params(target)
        src_qt = require_int8_params(src)
        target_qt.CB.copy_(src_qt.CB)
        target_qt.SCB.copy_(src_qt.SCB)

    @staticmethod
    def cache_bytes(state: _Bnb8bitHost) -> int:
        return (
            state.data.numel() * state.data.element_size()
            + state.scb.numel() * state.scb.element_size()
        )
