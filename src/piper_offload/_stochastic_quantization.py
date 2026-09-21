"""Stochastic terminal-code selection for quantized LoRA merges.

`piper-kernels` owns the rounding itself. This module adds the 8-bit float
codebook that TorchAO's and optimum-quanto's FP8 formats round against, which
the kernel package has no use for because its own formats are INT8 and NVFP4.
"""

import torch
from piper_kernels.stochastic_quantization import (
    stochastic_codebook_indices,
    stochastic_round_to_int,
)

__all__ = ["stochastic_codebook_indices", "stochastic_round_to_int"]


def _float_codebook(dtype: torch.dtype, *, device: torch.device) -> torch.Tensor:
    """Enumerate the finite values represented by an 8-bit float dtype."""
    if not dtype.is_floating_point or torch.finfo(dtype).bits != 8:
        raise ValueError(f"Unsupported stochastic float codebook dtype {dtype}.")
    return torch.arange(256, device=device, dtype=torch.uint8).view(dtype).to(torch.float32)


def _stochastic_cast_float8(
    values: torch.Tensor,
    dtype: torch.dtype,
    *,
    seed: int,
    deterministic: torch.Tensor,
) -> torch.Tensor:
    deterministic_bits = deterministic.contiguous().view(torch.uint8).to(torch.int64)
    indices = stochastic_codebook_indices(
        values,
        _float_codebook(dtype, device=values.device),
        seed=seed,
        deterministic=deterministic_bits,
    )
    return indices.to(torch.uint8).view(dtype)
