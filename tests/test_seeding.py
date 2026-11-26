"""Stable deterministic seed derivation tests."""

import pytest

from piper_offload import derive_seed


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
