"""Shared stochastic terminal-code selection tests."""

import pytest
import torch

from piper_offload import derive_seed
from piper_offload._stochastic_quantization import (
    _float_codebook,
    stochastic_codebook_indices,
    _stochastic_cast_float8,
    stochastic_round_to_int,
)


def _seed(
    *,
    key: str = "layers.0.weight",
    merge_index: int = 0,
) -> int:
    return derive_seed(key, merge_index)


def test_derive_seed_has_stable_typed_encoding() -> None:
    assert derive_seed("layer.0.weight", 0) == 0x9C3E_A68D_BAC9_1AAD
    assert derive_seed("1") != derive_seed(1)
    assert derive_seed("ab", "c") != derive_seed("a", "bc")


def test_derive_seed_validates_integer_range_and_part_types() -> None:
    assert 0 <= derive_seed(0, (1 << 64) - 1) < 1 << 64
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        derive_seed(-1)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        derive_seed(1 << 64)
    with pytest.raises(TypeError, match="strings or unsigned 64-bit"):
        derive_seed(True)
    with pytest.raises(TypeError, match="strings or unsigned 64-bit"):
        derive_seed(1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_rounding_replays_without_advancing_global_rng(device: str) -> None:
    # The draw is seeded, so a merge replays; it must not perturb the caller's
    # own sampling, which piper-kernels achieves with a private generator.
    values = torch.full((64, 128), 0.3, device=device)
    deterministic = torch.zeros_like(values, dtype=torch.int64)
    seed = _seed()
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = (
        torch.cuda.get_rng_state().clone() if device == "cuda" else None
    )

    first = stochastic_round_to_int(
        values,
        seed=seed,
        quant_min=-2,
        quant_max=2,
        deterministic=deterministic,
    )
    second = stochastic_round_to_int(
        values,
        seed=seed,
        quant_min=-2,
        quant_max=2,
        deterministic=deterministic,
    )

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert set(first.unique().tolist()) == {0, 1}
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    if cuda_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_before)


def test_target_path_and_merge_index_decorrelate_samples() -> None:
    values = torch.full((64, 128), 0.3)
    deterministic = torch.zeros_like(values, dtype=torch.int64)

    def rounded(seed: int) -> torch.Tensor:
        return stochastic_round_to_int(
            values,
            seed=seed,
            quant_min=-2,
            quant_max=2,
            deterministic=deterministic,
        )

    base = rounded(_seed())
    other_path = rounded(_seed(key="layers.1.weight"))
    other_merge = rounded(_seed(merge_index=1))

    assert not torch.equal(base, other_path)
    assert not torch.equal(base, other_merge)


def test_uniform_grid_selection_is_statistically_unbiased() -> None:
    values = torch.full((256, 512), 0.3)
    rounded = stochastic_round_to_int(
        values,
        seed=_seed(),
        quant_min=-2,
        quant_max=2,
        deterministic=torch.zeros_like(values, dtype=torch.int64),
    )
    assert set(rounded.unique().tolist()) == {0, 1}
    assert rounded.to(torch.float32).mean().item() == pytest.approx(
        0.3, abs=0.005
    )


def test_nonuniform_codebook_selection_is_statistically_unbiased() -> None:
    # 1.5 lies 25% of the way from 1 to 3.
    values = torch.full((256, 512), 1.5)
    codebook = torch.tensor([-2.0, 0.0, 1.0, 3.0])
    codes = stochastic_codebook_indices(
        values,
        codebook,
        seed=_seed(),
        deterministic=torch.zeros_like(values, dtype=torch.int64),
    )
    decoded = codebook[codes]
    assert set(decoded.unique().tolist()) == {1.0, 3.0}
    assert decoded.mean().item() == pytest.approx(1.5, abs=0.01)


def test_exact_endpoints_and_saturation_keep_deterministic_codes() -> None:
    values = torch.tensor([[-10.0, -1.0, 0.0, 1.0, 10.0]])
    deterministic = torch.tensor([[9, 0, 1, 2, 8]])
    codes = stochastic_codebook_indices(
        values,
        torch.tensor([-1.0, 0.0, 1.0]),
        seed=_seed(),
        deterministic=deterministic,
    )
    assert torch.equal(codes, deterministic)


def test_integer_endpoints_nonfinite_and_saturation_keep_deterministic_codes() -> None:
    values = torch.tensor(
        [[-torch.inf, -2.0, 0.0, 2.0, torch.inf, torch.nan]]
    )
    deterministic = torch.tensor([[41, 42, 43, 44, 45, 46]])

    rounded = stochastic_round_to_int(
        values,
        seed=_seed(),
        quant_min=-2,
        quant_max=2,
        deterministic=deterministic,
    )

    assert torch.equal(rounded, deterministic)


@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_float8_codebook_rounds_across_smallest_subnormal(dtype: torch.dtype) -> None:
    shape = (128, 512)
    finite = _float_codebook(dtype, device=torch.device("cpu"))
    smallest_positive = finite[(finite > 0) & torch.isfinite(finite)].min()
    values = torch.full(
        shape,
        smallest_positive.item() / 2,
        dtype=torch.float32,
    )
    deterministic = torch.zeros(shape, dtype=dtype)

    rounded = _stochastic_cast_float8(
        values,
        dtype,
        seed=_seed(),
        deterministic=deterministic,
    ).to(torch.float32)

    assert set(rounded.unique().tolist()) == {0.0, smallest_positive.item()}
    upper_rate = (rounded > 0).to(torch.float32).mean().item()
    assert upper_rate == pytest.approx(0.5, abs=0.01)


def test_float8_exact_negative_zero_keeps_upstream_sign_bit() -> None:
    values = torch.tensor([[-0.0]])
    deterministic = values.to(torch.float8_e4m3fn)
    assert deterministic.view(torch.uint8).item() == 0x80

    rounded = _stochastic_cast_float8(
        values,
        torch.float8_e4m3fn,
        seed=_seed(),
        deterministic=deterministic,
    )

    assert rounded.view(torch.uint8).item() == 0x80
