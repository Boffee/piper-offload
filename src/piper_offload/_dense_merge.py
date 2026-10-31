"""Shared reference implementation for structured dense merges."""

from typing import Protocol

import torch


class _DenseRequantizeAdapter(Protocol):
    """Static operations required by the reference dense-merge path."""

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor: ...

    @staticmethod
    def requantize(
        t: torch.Tensor,
        *,
        like: torch.Tensor,
        rounding_seed: int | None = None,
    ) -> torch.Tensor: ...

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None: ...


def merge_dense_requantize_(
    adapter: type[_DenseRequantizeAdapter],
    target: torch.Tensor,
    update: torch.Tensor,
    alpha: float,
    *,
    beta: float = 1.0,
    rounding_seed: int | None = None,
) -> None:
    """Scale the dense base, add the update, and encode into target once."""
    dense = adapter.dequantize(target)
    if beta != 1.0:
        dense.mul_(beta)
    dense.add_(update, alpha=alpha)
    requantized = adapter.requantize(
        dense,
        like=target,
        rounding_seed=rounding_seed,
    )
    adapter.copy_into(requantized, target=target)
