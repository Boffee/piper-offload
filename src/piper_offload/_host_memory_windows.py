"""Windows memory for pinned copies, which can be offered and reclaimed, and reads that rank below it.

A Windows copy lives in its own ``VirtualAlloc`` region, freed with
``VirtualFree``. Once unregistered, its pages can be *offered*
(``OfferVirtualMemory``): they leave the working set and become inaccessible,
Windows may discard them under memory pressure, and they stay committed.
``ReclaimVirtualMemory`` makes them accessible again and reports whether they
survived.

The region exports its bytes through the buffer protocol and counts the
exports, as ``mmap`` does, so anything that reads it must hold a
``memoryview``; a tensor over it is built from one. It refuses to be offered
or closed while an export is alive, and a region collected without being
closed is released then, so no pointer into it outlives its pages.

Filling a copy through the file cache leaves a second copy of every byte in
the cache, ranked above offered memory at the normal memory priority, so
while copies can be offered a fill reads at a lower one
(:func:`reading_below_offers`). The native calls are bound on first use,
never at import.
"""

import contextlib
import ctypes
import functools
import io
import logging
import sys
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import cast

logger = logging.getLogger(__name__)

_MEM_COMMIT = 0x1000
_MEM_RESERVE = 0x2000
_MEM_RELEASE = 0x8000
_PAGE_READWRITE = 0x04
_ERROR_BUSY = 170
# Windows drops offered memory at VeryLow priority all at once as soon as free
# memory runs short, and at Normal only as much as it needs: under the same
# 7.6 GiB of demand, 17 of 20 offered 128 MiB copies survived at Normal and
# none at VeryLow.
_VM_OFFER_PRIORITY_NORMAL = 4
_PYBUF_WRITE = 0x200
# SetThreadInformation's ThreadMemoryPriority class and its lowest value. The
# file-cache pages a thread's reads bring in take its memory priority, and at
# the lowest they rank below offered memory where the normal priority ranks
# above it. Windows does not document where offered memory ranks; measured,
# it sits between memory priorities 3 and 4, so the lowest leaves two levels
# of margin: with 21 GiB of RAM available, reloading an 18 GiB checkpoint
# after loading a 6.9 GiB one kept 10.8 to 14.4 GiB of its offered copies
# with the fill reading at 1, 2 or 3, and 3.6 to 6.2 GiB at 4 or 5.
_THREAD_MEMORY_PRIORITY = 0
_MEMORY_PRIORITY_VERY_LOW = 1
# The kernel event the memory manager sets when commitment nears the most the
# system can ever commit and the paging files cannot grow, which is when
# allocations start to fail. The current commit limit is no guide: with
# system-managed paging files Windows grows it on demand, and a tier that kept
# a tenth of it free threw away all 18 GiB of an offered checkpoint, and its
# 13 s reload became a 39 s read from disk, with commitment never short.
_MAXIMUM_COMMIT_CONDITION = "\\KernelObjects\\MaximumCommitCondition"
_SYNCHRONIZE = 0x00100000
_OBJ_CASE_INSENSITIVE = 0x40
_WAIT_OBJECT_0 = 0
# Offered memory is discarded in whole spans, so a copy is offered in spans
# rather than as one range: reacquiring an 18 GiB checkpoint after a 7 GiB one
# had to refill 6.28 GiB of it whole-region, 6.06 in 64 MiB spans, 5.93 in
# 16 MiB and 5.86 in 1 MiB, while offering and reclaiming 6.9 GiB cost 0.40 s
# whole-region, 0.38 in 16 MiB spans, 0.48 in 4 MiB and 0.86 in 1 MiB.
_OFFER_GRAIN = 16 * 2**20


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
    offer: Callable[[int, int, int], int]
    reclaim: Callable[[int, int], int]


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
    offer = library.OfferVirtualMemory
    offer.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    offer.restype = ctypes.c_uint32
    reclaim = library.ReclaimVirtualMemory
    reclaim.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    reclaim.restype = ctypes.c_uint32
    return _Kernel32(
        cast(Callable[[int | None, int, int, int], int | None], allocate),
        cast(Callable[[int, int, int], int], free),
        cast(Callable[[int, int, int], int], offer),
        cast(Callable[[int, int], int], reclaim),
    )


def _last_error() -> int:
    """The calling thread's last Windows error; zero elsewhere, where ``ctypes`` has no such call."""
    return ctypes.get_last_error() if sys.platform == "win32" else 0


def _release(kernel: _Kernel32, address: int) -> None:
    if not kernel.free(address, 0, _MEM_RELEASE):
        logger.warning("VirtualFree failed with Windows error %d; the region stays committed", _last_error())


class VirtualRegion:
    """``size`` bytes of zeroed, page-aligned private memory, committed by ``VirtualAlloc``.

    ``memoryview(region)`` reads and writes it while it is accessible; an
    offered or closed region refuses new exports with ``ValueError``.
    """

    __slots__ = ("__weakref__", "_exports", "_kernel", "_offered", "_released", "address", "size")

    def __init__(self, size: int) -> None:
        kernel = _kernel32()
        address = kernel.allocate(None, size, _MEM_COMMIT | _MEM_RESERVE, _PAGE_READWRITE)
        if not address:
            raise OSError(f"VirtualAlloc of {size} bytes failed with Windows error {_last_error()}")
        self.address = address
        self.size = size
        self._kernel = kernel
        self._exports = 0
        self._offered = False
        # Registered memory must not be freed at interpreter exit, before the runtime unregisters it.
        self._released = weakref.finalize(self, _release, kernel, address)
        self._released.atexit = False

    def __buffer__(self, flags: int, /) -> memoryview:
        if self._offered or self.closed:
            raise ValueError("region is offered or closed")
        self._exports += 1
        return _from_memory()(self.address, self.size, _PYBUF_WRITE)

    def __release_buffer__(self, view: memoryview, /) -> None:
        self._exports -= 1

    @property
    def closed(self) -> bool:
        return not self._released.alive

    @property
    def offered(self) -> bool:
        return self._offered

    def close(self) -> None:
        """Release the pages, offered or not; idempotent. Raises ``BufferError`` while an export is alive."""
        if self._exports:
            raise BufferError("cannot close a region while its buffer is exported")
        self._released()

    def _spans(self) -> Iterator[tuple[int, int]]:
        """The region as offsets and lengths of at most ``_OFFER_GRAIN`` bytes each."""
        for start in range(0, self.size, _OFFER_GRAIN):
            yield start, min(_OFFER_GRAIN, self.size - start)

    def offer(self) -> None:
        """Offer the pages to Windows, which may discard them; they are inaccessible until reclaimed.

        Offering in spans of ``_OFFER_GRAIN`` lets a reclaim find what survived in
        the same spans, so a discard costs one of them rather than the whole
        region. Raises ``BufferError``, changing nothing, while an export is
        alive. An ``OSError`` leaves the pages in an unknown state; close the
        region.
        """
        if self._exports:
            raise BufferError("cannot offer a region while its buffer is exported")
        self._offered = True
        for start, length in self._spans():
            code = self._kernel.offer(self.address + start, length, _VM_OFFER_PRIORITY_NORMAL)
            if code:
                raise OSError(f"OfferVirtualMemory failed with Windows error {code}")

    def reclaim(self) -> list[tuple[int, int]]:
        """Make offered pages accessible again, returning the spans Windows discarded; empty if all survived.

        A discarded span's contents (``ERROR_BUSY``) are undefined and must be
        rewritten. Any other failure raises ``OSError`` and leaves the region
        offered; close it.
        """
        discarded: list[tuple[int, int]] = []
        for start, length in self._spans():
            code = self._kernel.reclaim(self.address + start, length)
            if code not in (0, _ERROR_BUSY):
                raise OSError(f"ReclaimVirtualMemory failed with Windows error {code}")
            if code:
                if discarded and discarded[-1][0] + discarded[-1][1] == start:
                    previous, covered = discarded[-1]
                    discarded[-1] = (previous, covered + length)
                else:
                    discarded.append((start, length))
        self._offered = False
        return discarded


@dataclass(frozen=True, slots=True)
class _MemoryPriority:
    get: Callable[[], int]
    set: Callable[[int], None]


@functools.cache
def _thread_memory_priority() -> _MemoryPriority:
    """The calling thread's memory priority, read and set through ``Get/SetThreadInformation``."""
    if sys.platform != "win32":
        raise OSError("thread memory priorities exist only on Windows")
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    current = library.GetCurrentThread
    current.restype = ctypes.c_void_p
    calls = []
    for name in ("GetThreadInformation", "SetThreadInformation"):
        call = getattr(library, name)
        call.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
        call.restype = ctypes.c_int
        calls.append(call)
    get_information, set_information = calls

    def get() -> int:
        value = ctypes.c_uint32()
        if not get_information(current(), _THREAD_MEMORY_PRIORITY, ctypes.byref(value), ctypes.sizeof(value)):
            raise OSError(f"GetThreadInformation failed with Windows error {_last_error()}")
        return value.value

    def set_priority(priority: int) -> None:
        value = ctypes.c_uint32(priority)
        if not set_information(current(), _THREAD_MEMORY_PRIORITY, ctypes.byref(value), ctypes.sizeof(value)):
            raise OSError(f"SetThreadInformation failed with Windows error {_last_error()}")

    return _MemoryPriority(get, set_priority)


@contextlib.contextmanager
def reading_below_offers() -> Iterator[None]:
    """Run the calling thread at the lowest memory priority, then restore the one it had.

    File-cache pages the thread's reads bring in meanwhile rank below offered
    memory, so the cache gives them up first. So do private pages the thread
    faults in, which is why a fill faults its copy in before it reads.
    """
    priority = _thread_memory_priority()
    previous = priority.get()
    priority.set(_MEMORY_PRIORITY_VERY_LOW)
    try:
        yield
    finally:
        priority.set(previous)


class _UnicodeString(ctypes.Structure):
    _fields_ = (("length", ctypes.c_ushort), ("maximum_length", ctypes.c_ushort), ("buffer", ctypes.c_wchar_p))


class _ObjectAttributes(ctypes.Structure):
    _fields_ = (
        ("length", ctypes.c_ulong),
        ("root_directory", ctypes.c_void_p),
        ("object_name", ctypes.POINTER(_UnicodeString)),
        ("attributes", ctypes.c_ulong),
        ("security_descriptor", ctypes.c_void_p),
        ("security_quality_of_service", ctypes.c_void_p),
    )


def _open_maximum_commit_condition() -> Callable[[int], bool]:
    """A wait of up to so many milliseconds on ``MaximumCommitCondition``, true if it is set; Windows only."""
    if sys.platform != "win32":
        raise OSError("commit conditions exist only on Windows")
    open_event = ctypes.WinDLL("ntdll").NtOpenEvent
    open_event.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong, ctypes.POINTER(_ObjectAttributes))
    open_event.restype = ctypes.c_long
    size = len(_MAXIMUM_COMMIT_CONDITION) * ctypes.sizeof(ctypes.c_wchar)
    name = _UnicodeString(size, size + ctypes.sizeof(ctypes.c_wchar), _MAXIMUM_COMMIT_CONDITION)
    attributes = _ObjectAttributes(
        ctypes.sizeof(_ObjectAttributes), None, ctypes.pointer(name), _OBJ_CASE_INSENSITIVE, None, None,
    )
    handle = ctypes.c_void_p()
    status = open_event(ctypes.byref(handle), _SYNCHRONIZE, ctypes.byref(attributes))
    if status:
        raise OSError(f"NtOpenEvent failed with NTSTATUS {status & 0xFFFFFFFF:#010x}")
    wait = ctypes.WinDLL("kernel32", use_last_error=True).WaitForSingleObject
    wait.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    wait.restype = ctypes.c_uint32
    event = handle.value
    return lambda milliseconds: wait(event, milliseconds) == _WAIT_OBJECT_0


@functools.cache
def _maximum_commit_condition() -> Callable[[int], bool] | None:
    """The condition's wait, opened once for the process; None, with a warning, where it cannot be opened."""
    try:
        return _open_maximum_commit_condition()
    except OSError as error:
        logger.warning("Cannot watch Windows commitment; offered copies are freed only by their budgets: %s", error)
        return None


def commit_exhausted(timeout: float = 0.0) -> bool:
    """Whether Windows reports its commitment exhausted, waiting up to ``timeout`` seconds for it to be.

    That is commitment near the most the system can ever commit, with the
    paging files unable to grow, which is when allocations anywhere begin to
    fail. Returns at once when it is, and false where it cannot be observed.
    """
    wait = _maximum_commit_condition()
    if wait is None:
        time.sleep(timeout)
        return False
    return wait(int(timeout * 1000))


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


def new_region(size: int) -> VirtualRegion:
    """An owned page-aligned allocation whose pages can be offered after unregistration."""
    return VirtualRegion(size)


class Readers(contextlib.AbstractContextManager["Readers"]):
    """Checkpoint reads through one handle per file per worker, closed after the fill.

    A Windows handle has one position and the kernel serializes concurrent
    reads on it. Each worker opens its own handle from the provenance path;
    ``MappedCheckpoint`` keeps that file open on disk throughout the fill.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self._opened: list[io.FileIO] = []
        self._lock = threading.Lock()

    def __exit__(self, *_exc: object) -> None:
        for handle in self._opened:
            handle.close()

    def read_at(self, file: io.BufferedReader) -> Callable[[memoryview, int], int]:
        """A positional read of ``file`` that any worker may call."""
        return functools.partial(self._read_alone, file)

    def _read_alone(self, file: io.BufferedReader, buffer: memoryview, offset: int) -> int:
        handles: dict[int, io.FileIO] = getattr(self._local, "handles", None) or {}
        self._local.handles = handles
        handle = handles.get(id(file))
        if handle is None:
            handle = open(file.name, "rb", buffering=0)  # noqa: SIM115 - closed with the fill's readers
            handles[id(file)] = handle
            with self._lock:
                self._opened.append(handle)
        handle.seek(offset)
        return handle.readinto(buffer)


__all__ = ["Readers", "VirtualRegion", "available_memory", "commit_exhausted", "new_region", "reading_below_offers"]
