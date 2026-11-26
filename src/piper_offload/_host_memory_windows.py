"""Windows host memory: physical RAM query and ``VirtualAlloc`` aligned regions.

A Windows host slab or region lives in its own ``VirtualAlloc`` allocation and
is freed with ``VirtualFree``. The region exports its bytes through the buffer
protocol and counts live exports, refusing ``close()`` while an export is
alive; an unclosed collected region is freed via a non-atexit finalizer so no
pointer outlives its pages. Native calls are bound on first use, never at import.
"""

import ctypes
import functools
import logging
import mmap
import sys
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

logger = logging.getLogger(__name__)

DEFAULT_PIN_FRACTION = 0.45
DEFAULT_HOST_FRACTION = 0.75

_MEM_COMMIT = 0x1000
_MEM_RESERVE = 0x2000
_MEM_RELEASE = 0x8000
_PAGE_READWRITE = 0x04
_PYBUF_WRITE = 0x200


@functools.cache
def _from_memory() -> Callable[[int, int, int], memoryview]:
    """CPython's ``PyMemoryView_FromMemory``, which views raw pages without owning them."""
    view = ctypes.pythonapi.PyMemoryView_FromMemory
    view.argtypes = (ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int)
    view.restype = ctypes.py_object
    return cast(Callable[[int, int, int], memoryview], view)


@dataclass(frozen=True, slots=True)
class _Kernel32:
    allocate: Callable[[int | None, int, int, int], int | None]
    free: Callable[[int, int, int], int]


@functools.cache
def _kernel32() -> _Kernel32:
    if sys.platform != "win32":
        raise OSError("VirtualAlloc regions exist only on Windows")
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    allocate = library.VirtualAlloc
    allocate.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32)
    allocate.restype = ctypes.c_void_p
    free = library.VirtualFree
    free.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
    free.restype = ctypes.c_int
    return _Kernel32(
        cast(Callable[[int | None, int, int, int], int | None], allocate),
        cast(Callable[[int, int, int], int], free),
    )


def _last_error() -> int:
    """The calling thread's last Windows error; zero elsewhere, where ``ctypes`` has no such call."""
    return ctypes.get_last_error() if sys.platform == "win32" else 0


def _release(kernel: _Kernel32, address: int) -> None:
    if not kernel.free(address, 0, _MEM_RELEASE):
        logger.warning("VirtualFree failed with Windows error %d; the region stays committed", _last_error())


class VirtualRegion:
    """``size`` bytes of zeroed, page-aligned private memory, committed by ``VirtualAlloc``.

    ``memoryview(region)`` reads and writes it while it is open; a closed
    region refuses new exports with ``ValueError``.
    """

    __slots__ = ("__weakref__", "_exports", "_kernel", "_released", "address", "size")

    def __init__(self, size: int) -> None:
        kernel = _kernel32()
        address = kernel.allocate(None, size, _MEM_COMMIT | _MEM_RESERVE, _PAGE_READWRITE)
        if not address:
            raise OSError(f"VirtualAlloc of {size} bytes failed with Windows error {_last_error()}")
        self.address = address
        self.size = size
        self._kernel = kernel
        self._exports = 0
        # Registered memory must not be freed at interpreter exit, before the runtime unregisters it.
        self._released = weakref.finalize(self, _release, kernel, address)
        self._released.atexit = False

    def __buffer__(self, flags: int, /) -> memoryview:
        if self.closed:
            raise ValueError("region is closed")
        view = _from_memory()(self.address, self.size, _PYBUF_WRITE)
        self._exports += 1
        return view

    def __release_buffer__(self, view: memoryview, /) -> None:
        if hasattr(self, "_exports"):
            self._exports -= 1

    @property
    def closed(self) -> bool:
        return not self._released.alive

    def close(self) -> None:
        """Release the pages; idempotent. Raises ``BufferError`` while an export is alive."""
        if self._exports:
            raise BufferError("cannot close a region while its buffer is exported")
        self._released()


class _MemoryStatus(ctypes.Structure):
    # Windows MEMORYSTATUSEX; fixed-width types also allow testing on POSIX.
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("load", ctypes.c_uint32),
        ("total_physical", ctypes.c_uint64),
        ("available_physical", ctypes.c_uint64),
        ("total_page_file", ctypes.c_uint64),
        ("available_page_file", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64),
        ("available_virtual", ctypes.c_uint64),
        ("available_extended_virtual", ctypes.c_uint64),
    ]


def available_memory() -> int:
    """Physical RAM reported by Windows, without initializing CUDA."""
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    # WinDLL exists only on Windows; resolve it at this native-call boundary
    # so the module can also be tested with an injected library on Linux.
    query = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True).GlobalMemoryStatusEx  # noqa: B009
    query.argtypes = (ctypes.POINTER(_MemoryStatus),)
    query.restype = ctypes.c_int
    if not query(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    if status.total_physical <= 0:
        raise ValueError("physical RAM is unavailable")
    return status.total_physical


def allocate_aligned_region(size: int) -> VirtualRegion:
    """Allocate ``size`` bytes of page-aligned private memory via ``VirtualAlloc``."""
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % mmap.PAGESIZE != 0:
        raise ValueError(
            f"Aligned host region size must be a positive multiple of {mmap.PAGESIZE}, got {size!r}."
        )
    return VirtualRegion(size)


__all__ = [
    "DEFAULT_HOST_FRACTION",
    "DEFAULT_PIN_FRACTION",
    "VirtualRegion",
    "allocate_aligned_region",
    "available_memory",
]
