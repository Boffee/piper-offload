"""Shared Triton primitives for stochastic terminal-code selection."""

# Triton JIT helper signatures intentionally use untyped tensor parameters
# and upper-case constexpr names.
# ruff: noqa: ANN001, ANN202, N803

import triton
import triton.language as tl


def _seed_argument(seed: int | None) -> int:
    """Return a launch-safe signed scalar with the seed's uint64 bit pattern."""
    if seed is None:
        return 0
    return seed if seed < (1 << 63) else seed - (1 << 64)


@triton.jit
def _random(seed, offsets):
    """Draw by logical element offset so launch geometry cannot affect samples."""
    return tl.rand(seed, offsets.to(tl.uint64))


@triton.jit
def _stochastic_round_to_int(
    values,
    deterministic,
    seed,
    offsets,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
):
    """Round finite interior values to adjacent integers."""
    interior = (values > QMIN) & (values < QMAX)
    safe = tl.where(interior, values, 0.0)
    lower = tl.floor(safe)
    probability = safe - lower
    rounded = lower + (_random(seed, offsets) < probability)
    return tl.where(interior & (probability > 0.0), rounded, deterministic)


@triton.jit
def _stochastic_sorted_code(
    values,
    deterministic,
    code_ptr,
    seed,
    offsets,
    MAX_INDEX: tl.constexpr,
    STEPS: tl.constexpr,
):
    """Select adjacent entries from a monotonically increasing codebook."""
    low = tl.zeros(values.shape, dtype=tl.int32)
    high = low + MAX_INDEX
    for _ in range(STEPS):
        middle = (low + high) // 2
        middle_value = tl.load(code_ptr + middle)
        move_right = middle_value < values
        low = tl.where(move_right, middle + 1, low)
        high = tl.where(move_right, high, middle)

    upper_index = low
    upper = tl.load(code_ptr + upper_index)
    exact = upper == values
    lower_index = tl.where(exact, upper_index, tl.maximum(upper_index - 1, 0))
    lower = tl.load(code_ptr + lower_index)
    width = upper - lower
    probability = tl.where(width > 0.0, (values - lower) / width, 0.0)
    chosen = tl.where(
        _random(seed, offsets) < probability,
        upper_index,
        lower_index,
    )
    interior = (
        (values > tl.load(code_ptr))
        & (values < tl.load(code_ptr + MAX_INDEX))
        & ~exact
    )
    return tl.where(interior, chosen, deterministic)


@triton.jit
def _e2m1_value(code):
    """Decode the magnitude bits of an E2M1 storage code."""
    code &= 7
    return tl.where(
        code == 0,
        0.0,
        tl.where(
            code == 1,
            0.5,
            tl.where(
                code == 2,
                1.0,
                tl.where(
                    code == 3,
                    1.5,
                    tl.where(
                        code == 4,
                        2.0,
                        tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )


@triton.jit
def _stochastic_e2m1_code(values, deterministic, seed, offsets):
    """Select adjacent E2M1 values and return their four-bit storage codes."""
    magnitude = tl.abs(values.to(tl.float32))
    nearest_code = deterministic.to(tl.int32) & 7
    nearest = _e2m1_value(nearest_code)
    rounded_down = nearest > magnitude
    lower_code = tl.where(rounded_down, nearest_code - 1, nearest_code)
    upper_code = tl.where(rounded_down, nearest_code, nearest_code + 1)
    lower = _e2m1_value(lower_code)
    upper = _e2m1_value(upper_code)
    probability = tl.where(upper > lower, (magnitude - lower) / (upper - lower), 0.0)
    magnitude_code = tl.where(
        _random(seed, offsets) < probability,
        upper_code,
        lower_code,
    )
    sign = (values.to(tl.int32, bitcast=True) >> 28) & 8
    chosen = magnitude_code | sign
    interior = (magnitude > 0.0) & (magnitude < 6.0) & (probability > 0.0)
    return tl.where(interior, chosen, deterministic)


@triton.jit
def _stochastic_float8(
    values,
    seed,
    offsets,
    E4M3: tl.constexpr,
    LIMIT: tl.constexpr,
):
    """Round to adjacent finite E4M3FN or E5M2 values."""
    values_f32 = values.to(tl.float32)
    magnitude = tl.abs(values_f32)
    if E4M3:  # noqa: SIM108 - constexpr branches return distinct Triton types.
        deterministic_magnitude = magnitude.to(tl.float8e4nv)
    else:
        deterministic_magnitude = magnitude.to(tl.float8e5)
    deterministic_bits = deterministic_magnitude.to(tl.uint8, bitcast=True)
    deterministic_value = deterministic_magnitude.to(tl.float32)

    rounded_down = deterministic_value > magnitude
    lower_bits = tl.where(rounded_down, deterministic_bits - 1, deterministic_bits)
    upper_bits = tl.where(rounded_down, deterministic_bits, deterministic_bits + 1)
    if E4M3:
        lower = lower_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        upper = upper_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        lower = lower_bits.to(tl.float8e5, bitcast=True).to(tl.float32)
        upper = upper_bits.to(tl.float8e5, bitcast=True).to(tl.float32)

    probability = tl.where(upper > lower, (magnitude - lower) / (upper - lower), 0.0)
    magnitude_bits = tl.where(
        _random(seed, offsets) < probability,
        upper_bits,
        lower_bits,
    )
    sign = (values_f32.to(tl.int32, bitcast=True) >> 24) & 0x80
    chosen_bits = magnitude_bits | sign
    deterministic_bits |= sign
    interior = (magnitude < LIMIT) & (probability > 0.0)
    output_bits = tl.where(interior, chosen_bits, deterministic_bits).to(tl.uint8)
    if E4M3:
        output = output_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        output = output_bits.to(tl.float8e5, bitcast=True).to(tl.float32)
    return output


__all__: list[str] = []
