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
