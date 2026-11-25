"""Diffusers-compatible GGUF test parameters without that dependency."""

import torch
from torch import nn


class GGUFParameter(nn.Parameter):
    """Minimal test double for Diffusers ``GGUFParameter``."""

    quant_type: int
    quant_shape: torch.Size

    @staticmethod
    def __new__(
        cls: type[GGUFParameter],
        data: torch.Tensor,
        requires_grad: bool = False,
        quant_type: int | None = None,
    ) -> GGUFParameter:
        assert quant_type is not None
        parameter = torch.Tensor._make_subclass(cls, data, requires_grad)
        parameter.quant_type = int(quant_type)
        from piper_kernels.gguf import logical_shape

        parameter.quant_shape = torch.Size(
            logical_shape(tuple(data.detach().view(torch.uint8).shape), quant_type)
        )
        return parameter

    def as_tensor(self) -> torch.Tensor:
        return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)


def packed_q4_0(rows: int, features: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Make complete Q4_0 rows with arbitrary codes and finite, nonzero scales."""
    blocks_per_row = features // 32
    blocks = torch.randint(256, (rows, blocks_per_row, 18), dtype=torch.uint8, generator=generator)
    scales = torch.rand((rows, blocks_per_row, 1), generator=generator).mul_(0.2).add_(0.02).half()
    blocks[..., :2] = scales.view(torch.uint8)
    return blocks.flatten(1)


def packed_q8_0(rows: int, features: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Make complete Q8_0 rows with arbitrary codes and finite, nonzero scales."""
    blocks_per_row = features // 32
    blocks = torch.randint(256, (rows, blocks_per_row, 34), dtype=torch.uint8, generator=generator)
    scales = torch.rand((rows, blocks_per_row, 1), generator=generator).mul_(0.2).add_(0.02).half()
    blocks[..., :2] = scales.view(torch.uint8)
    return blocks.flatten(1)
