"""Internal helper: invert TorchAO's ``get_block_size`` to a granularity.

TorchAO's affine-quantized tensors (``Float8Tensor``, ``Int8Tensor``)
store the per-block partition as a ``block_size`` list rather than the
``Granularity`` object it was derived from, but the re-encoding
constructors (``from_hp``) take a ``Granularity``. Recovering it is the
same arithmetic for every such format — it depends only on ``block_size``
vs ``shape`` and the shared ``PerRow`` / ``PerTensor`` classes, not on any
per-format layout — so it lives here once instead of in each
``_torchao_<format>`` module.
"""

from typing import Any

import torch

try:
    from torchao.quantization.granularity import PerGroup, PerRow, PerTensor

    TORCHAO_GRANULARITY_AVAILABLE = True
except ImportError:
    TORCHAO_GRANULARITY_AVAILABLE = False
    PerGroup: Any = None
    PerRow: Any = None
    PerTensor: Any = None


def granularity_from_block_size(
    block_size: tuple[int, ...], shape: tuple[int, ...], *, label: str,
) -> object:
    """Invert TorchAO's ``get_block_size`` for the affine granularities.

    Supports ``PerTensor`` (block covers the whole tensor), ``PerRow``
    (block covers one axis fully, 1 elsewhere — the per-output-channel
    weight recipe gives ``[1, in]``), and ``PerGroup`` (block groups the
    last axis into chunks of ``group_size``, giving ``[1, group_size]``
    with ``group_size`` a proper divisor of the last dim). Shapes where two
    readings coincide (e.g. a dim of size 1, or ``group_size == in``)
    produce identical block partitions either way, so any matching reading
    is correct; ``PerRow`` is checked first. ``label`` names the wrapper
    type for the error raised on an unrecognized granularity.
    """
    if block_size == shape:
        return PerTensor()
    for dim, size in enumerate(shape):
        per_row = tuple(size if i == dim else 1 for i in range(len(shape)))
        if block_size == per_row:
            return PerRow(dim=dim)
    # PerGroup chunks the last axis: 1 on every leading axis, and a last
    # entry that proper-divides the last dim (the == case is PerRow, caught
    # above).
    *lead, last = block_size
    if (
        PerGroup is not None
        and all(b == 1 for b in lead)
        and 0 < last < shape[-1]
        and shape[-1] % last == 0
    ):
        return PerGroup(last)
    raise ValueError(
        f"{label} block_size {block_size!r} for shape {shape!r} matches "
        "no PerTensor / PerRow / PerGroup granularity; TorchAO likely added "
        "a granularity this adapter does not support yet."
    )


def expand_block_parameter(
    parameter: torch.Tensor,
    *,
    block_size: tuple[int, ...],
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Expand one quantization parameter per block to logical elements."""
    if len(block_size) != len(shape) or any(
        block <= 0 or size % block != 0
        for size, block in zip(shape, block_size, strict=True)
    ):
        raise ValueError(
            f"Invalid quantization block layout: shape={shape}, "
            f"block_size={block_size}."
        )
    grid = tuple(
        size // block for size, block in zip(shape, block_size, strict=True)
    )
    expected = 1
    for size in grid:
        expected *= size
    if parameter.numel() != expected:
        raise ValueError(
            "Quantization parameter storage does not match its block grid: "
            f"parameter={tuple(parameter.shape)}, expected={grid}."
        )
    expanded = parameter.reshape(grid)
    for dim, repeat in enumerate(block_size):
        if repeat != 1:
            expanded = expanded.repeat_interleave(repeat, dim=dim)
    return expanded
