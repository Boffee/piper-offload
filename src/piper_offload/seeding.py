"""Stable seed derivation shared by Piper Offload and external adapters."""

from hashlib import blake2s

__all__ = ["derive_seed"]

_UINT64_LIMIT = 1 << 64


def derive_seed(*parts: str | int) -> int:
    """Return a stable unsigned 64-bit seed derived from typed identity parts.

    Strings are UTF-8 encoded with their length; integers must fit in an
    unsigned 64-bit value. The encoding is versioned and does not depend on
    Python's process-randomized :func:`hash`.
    """
    digest = blake2s(digest_size=8)
    digest.update(b"piper-offload-seed-v1\0")
    for part in parts:
        if isinstance(part, str):
            encoded = part.encode("utf-8")
            digest.update(b"s")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        elif isinstance(part, int) and not isinstance(part, bool):
            if not 0 <= part < _UINT64_LIMIT:
                raise ValueError(
                    "derive_seed() integer parts must be unsigned 64-bit values"
                )
            digest.update(b"i")
            digest.update(part.to_bytes(8, "little"))
        else:
            raise TypeError(
                "derive_seed() parts must be strings or unsigned 64-bit integers"
            )
    return int.from_bytes(digest.digest(), "little")
