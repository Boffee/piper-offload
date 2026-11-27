"""Select the platform host memory helpers once at import; native calls remain lazy.

Linux allocations use anonymous mappings and Windows allocations use
``VirtualAlloc`` regions. Budget fractions reflect safe defaults (70% pinned /
80% total host on Linux bounded by cgroups; 45% pinned / 75% total host on
Windows WDDM).
"""

import logging
import mmap
import sys
from collections.abc import Buffer
from fractions import Fraction
from typing import Protocol

if sys.platform == "win32":
    from ._host_memory_windows import (
        DEFAULT_HOST_FRACTION,
        DEFAULT_PIN_FRACTION,
        available_memory,
    )
    from ._host_memory_windows import (
        allocate_aligned_region as _platform_allocate_aligned_region,
    )
else:
    from ._host_memory_linux import (
        DEFAULT_HOST_FRACTION,
        DEFAULT_PIN_FRACTION,
        available_memory,
    )
    from ._host_memory_linux import (
        allocate_aligned_region as _platform_allocate_aligned_region,
    )

logger = logging.getLogger(__name__)


class AlignedRegion(Buffer, Protocol):
    """Page-aligned OS host allocation exposing the buffer protocol and ``close()``."""

    def close(self) -> None: ...


def allocate_aligned_region(size: int) -> AlignedRegion:
    """Allocate ``size`` bytes of page-aligned private host memory."""
    return _platform_allocate_aligned_region(size)


def _default_budget(fraction: float, label: str) -> int:
    try:
        total = max(0, available_memory())
        ratio = Fraction(str(fraction))
        scaled = (total * ratio.numerator) // ratio.denominator
        return (scaled // mmap.PAGESIZE) * mmap.PAGESIZE
    except (AttributeError, OSError, ValueError) as error:
        logger.warning(
            "Cannot determine physical RAM; the default %s is zero: %s",
            label,
            str(error),
        )
        return 0


def default_pin_budget() -> int:
    """Default Tier 2a pinned host budget, rounded down to OS pages without touching CUDA."""
    return _default_budget(DEFAULT_PIN_FRACTION, "host pin budget")


def default_host_budget() -> int:
    """Default total Tier 2a + Tier 2b host memory ceiling, rounded down to OS pages."""
    return _default_budget(DEFAULT_HOST_FRACTION, "host memory budget")


__all__ = [
    "DEFAULT_HOST_FRACTION",
    "DEFAULT_PIN_FRACTION",
    "AlignedRegion",
    "allocate_aligned_region",
    "available_memory",
    "default_host_budget",
    "default_pin_budget",
]
