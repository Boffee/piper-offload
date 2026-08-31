"""Shared fixtures for block-compilation tests."""

import torch
from torch import nn

from piper_offload import (
    BlockCompileConfig,
    ModelOffloader,
)


class _Block(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(x))


class _BlockModel(nn.Module):
    def __init__(
        self,
        *,
        num_blocks: int = 2,
        width: int = 8,
        blocks: list[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if blocks is None:
            blocks = [_Block(width) for _ in range(num_blocks)]
        self.blocks = nn.ModuleList(blocks)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _make_offloader(
    model: nn.Module,
    *,
    block_paths: list[str] | None = None,
    transient_block_paths: tuple[str, ...] = (),
    block_compile: BlockCompileConfig | None = None,
) -> ModelOffloader:
    if block_paths is None:
        block_paths = [] if transient_block_paths else ["blocks"]
    return ModelOffloader.from_module(
        model,
        block_paths=block_paths,
        transient_block_paths=transient_block_paths,
        block_compile=block_compile,
    )


__all__ = ["_Block", "_BlockModel", "_make_offloader"]
