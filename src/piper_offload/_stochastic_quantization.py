"""Stochastic terminal-code selection for quantized LoRA merges."""

import torch

__all__: list[str] = []
def _uniform(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Draw reproducibly without consuming the process-global RNG."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand(
        shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )


def _stochastic_round_to_int(
    values: torch.Tensor,
    *,
    seed: int,
    quant_min: int,
    quant_max: int,
    deterministic: torch.Tensor,
) -> torch.Tensor:
    """Round scaled values to adjacent integers with unbiased probability."""
    if deterministic.shape != values.shape:
        raise ValueError("Deterministic integer qdata does not match the values.")
    finite = torch.nan_to_num(
        values.to(torch.float32),
        nan=0.0,
        posinf=float(quant_max),
        neginf=float(quant_min),
    ).clamp_(quant_min, quant_max)
    lower = finite.floor()
    probability = finite - lower
    uniform = _uniform(
        tuple(values.shape),
        device=values.device,
        seed=seed,
    )
    rounded = lower.add_(uniform < probability).to(torch.int64)
    interior = (
        torch.isfinite(values)
        & (finite > quant_min)
        & (finite < quant_max)
        & (probability > 0)
    )
    return torch.where(
        interior,
        rounded,
        deterministic.to(device=values.device, dtype=torch.int64),
    )


def _stochastic_codebook_indices(
    values: torch.Tensor,
    codebook: torch.Tensor,
    *,
    seed: int,
    deterministic: torch.Tensor,
) -> torch.Tensor:
    """Select adjacent finite codebook entries with unbiased probability."""
    if deterministic.shape != values.shape:
        raise ValueError("Deterministic codebook qdata does not match the values.")
    if codebook.ndim != 1 or codebook.numel() < 2:
        raise ValueError(
            "Stochastic rounding requires a one-dimensional codebook with "
            "at least two entries."
        )
    finite_mask = torch.isfinite(codebook)
    if not bool(finite_mask.any()):
        raise ValueError("Stochastic-rounding codebook has no finite entries.")

    storage_indices = torch.arange(
        codebook.numel(),
        device=codebook.device,
        dtype=torch.int64,
    )[finite_mask]
    levels, order = torch.sort(codebook[finite_mask].to(torch.float32))
    storage_indices = storage_indices[order]
    finite_values = torch.nan_to_num(
        values.to(device=levels.device, dtype=torch.float32),
        nan=0.0,
        posinf=float(levels[-1]),
        neginf=float(levels[0]),
    ).clamp_(float(levels[0]), float(levels[-1]))
    upper_index = torch.searchsorted(levels, finite_values).clamp_(
        max=levels.numel() - 1
    )
    upper = levels[upper_index]
    exact = upper == finite_values
    lower_index = torch.where(
        exact,
        upper_index,
        (upper_index - 1).clamp_(min=0),
    )
    lower = levels[lower_index]
    width = upper - lower
    probability = torch.where(
        width > 0,
        (finite_values - lower) / width,
        torch.zeros_like(width),
    )
    uniform = _uniform(
        tuple(values.shape),
        device=values.device,
        seed=seed,
    )
    chosen = storage_indices[
        torch.where(uniform < probability, upper_index, lower_index)
    ]
    interior = (
        torch.isfinite(values)
        & (finite_values > levels[0])
        & (finite_values < levels[-1])
        & ~exact
    )
    return torch.where(
        interior,
        chosen,
        deterministic.to(device=values.device, dtype=torch.int64),
    )


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
    indices = _stochastic_codebook_indices(
        values,
        _float_codebook(dtype, device=values.device),
        seed=seed,
        deterministic=deterministic_bits,
    )
    return indices.to(torch.uint8).view(dtype)
