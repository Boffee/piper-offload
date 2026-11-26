"""Process memory, read traffic, and page-cache control on Linux and Windows.

The checkpoint benchmarks report what Tier 2a pinned slabs and Tier 2b
overflow copies cost and where their bytes come from. Linux answers from
``/proc``: the resident set, the shared part anonymous mappings land in, and the
bytes the process read from disk. Windows reports the process's *working set*
and *commitment* (``PrivateUsage``). Its per-process I/O counter also counts
reads served from the cache, so it bounds disk traffic rather than measuring it.
"""

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_READ_CHUNK = 64 * 2**20


@dataclass(frozen=True, slots=True)
class Memory:
    """What the process holds, in bytes.

    ``resident`` is the working set on Windows and ``VmRSS`` on Linux.
    ``committed`` is the commitment Windows has promised this process
    (``PrivateUsage``), and on Linux the closest equivalent, its private
    anonymous pages and what of them is swapped out. ``shared`` is the part of
    the resident set backed by shared memory, which Linux counts separately
    and Windows does not report per process.
    """

    resident: int
    committed: int
    shared: int


class _MemoryCounters(ctypes.Structure):
    # Windows PROCESS_MEMORY_COUNTERS_EX.
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("page_faults", ctypes.c_uint32),
        ("peak_working_set", ctypes.c_size_t),
        ("working_set", ctypes.c_size_t),
        ("peak_paged_pool", ctypes.c_size_t),
        ("paged_pool", ctypes.c_size_t),
        ("peak_non_paged_pool", ctypes.c_size_t),
        ("non_paged_pool", ctypes.c_size_t),
        ("page_file", ctypes.c_size_t),
        ("peak_page_file", ctypes.c_size_t),
        ("private_usage", ctypes.c_size_t),
    ]


class _IoCounters(ctypes.Structure):
    # Windows IO_COUNTERS.
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "reads", "writes", "other", "read_bytes", "written_bytes", "other_bytes",
    )]


class _MemoryStatus(ctypes.Structure):
    # Windows MEMORYSTATUSEX.
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("load", ctypes.c_uint32),
        ("total_physical", ctypes.c_uint64),
        ("available_physical", ctypes.c_uint64),
        ("total_commit", ctypes.c_uint64),
        ("available_commit", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64),
        ("available_virtual", ctypes.c_uint64),
        ("available_extended", ctypes.c_uint64),
    ]


def _proc_status() -> dict[str, int]:
    values: dict[str, int] = {}
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            name, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[name] = int(parts[0]) * 1024
    return values


def memory() -> Memory:
    """What the process holds right now."""
    if sys.platform == "win32":
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        query = ctypes.WinDLL("kernel32", use_last_error=True).K32GetProcessMemoryInfo
        if not query(ctypes.c_void_p(-1), ctypes.byref(counters), ctypes.sizeof(counters)):
            raise OSError("K32GetProcessMemoryInfo failed")
        return Memory(counters.working_set, counters.private_usage, 0)
    status = _proc_status()
    return Memory(
        status.get("VmRSS", 0),
        status.get("RssAnon", 0) + status.get("VmSwap", 0),
        status.get("RssShmem", 0),
    )


def read_bytes() -> int:
    """Bytes this process has read: from disk on Linux, from disk or the cache on Windows."""
    if sys.platform == "win32":
        counters = _IoCounters()
        query = ctypes.WinDLL("kernel32", use_last_error=True).GetProcessIoCounters
        if not query(ctypes.c_void_p(-1), ctypes.byref(counters)):
            raise OSError("GetProcessIoCounters failed")
        return counters.read_bytes
    with open("/proc/self/io", encoding="utf-8") as counters_file:
        for line in counters_file:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return 0


def system_commit() -> tuple[int, int]:
    """The system's commit limit and what this process could still commit; zeros off Windows."""
    if sys.platform != "win32":
        return (0, 0)
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    query = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
    if not query(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return (status.total_commit, status.available_commit)


def fill_cache(path: Path) -> None:
    """Read the whole file, so a refill of it touches no disk."""
    with open(path, "rb", buffering=0) as source:
        while source.read(_READ_CHUNK):
            pass


def drop_from_cache(path: Path) -> None:
    """Evict the file's pages from the page cache, as far as the OS allows.

    Windows purges a file's cached pages when it is opened unbuffered, but
    only while nothing maps it; a checkpoint already mapped by the reader
    keeps the pages its mapping covers, so time a cold read before mapping it.
    """
    if sys.platform == "win32":
        # CreateFileW(GENERIC_READ, share all, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING).
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateFileW
        create.argtypes = (ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                           ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
        create.restype = ctypes.c_void_p
        handle = create(str(path), 0x80000000, 0x00000007, None, 3, 0x20000000, None)
        if handle is None or handle == 2**64 - 1:
            raise OSError(f"CreateFileW failed with Windows error {ctypes.get_last_error()}")
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


__all__ = ["Memory", "drop_from_cache", "fill_cache", "memory", "read_bytes", "system_commit"]
