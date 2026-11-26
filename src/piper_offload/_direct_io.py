"""Direct I/O ``FileSlice`` reader with sector alignment, coalescing, and fallback.

Reads raw checkpoint byte spans directly into aligned host destinations using
``O_DIRECT`` on Linux/POSIX and ``FILE_FLAG_NO_BUFFERING`` with pooled Win32
handles on Windows, avoiding OS page-cache double caching and per-page faults.
Filesystems or open-file topologies that refuse Direct I/O (such as ``tmpfs``,
FUSE mounts, or unlinked/replaced files) fall back cleanly to buffered reads.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import functools
import io
import mmap
import os
import sys
import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Protocol, Self, cast

from . import _host_memory as host_memory
from .checkpoint import FileSlice

if sys.platform == "win32":
    import msvcrt

DEFAULT_SECTOR_SIZE = 4096
DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024
DEFAULT_COALESCE_GAP = 64 * 1024
DEFAULT_COALESCE_MAX_BYTES = 4 * 1024 * 1024
DEFAULT_COALESCE_THRESHOLD = 512 * 1024

_WIN32_MAX_READ_BYTES = 1 << 30  # 1 GiB (sector-aligned, well below uint32 limit)

_O_DIRECT = cast(int, getattr(os, "O_DIRECT", 0))
_POSIX_FADV_DONTNEED = cast(int, getattr(os, "POSIX_FADV_DONTNEED", 4))

_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_OPEN_EXISTING = 3
_FILE_FLAG_NO_BUFFERING = 0x20000000
_FILE_FLAG_RANDOM_ACCESS = 0x10000000
_ERROR_INVALID_FUNCTION = 1
_ERROR_HANDLE_EOF = 38
_ERROR_NOT_SUPPORTED = 50
_ERROR_INVALID_PARAMETER = 87
_WIN_DOWNGRADE_ERRORS = frozenset(
    {_ERROR_INVALID_PARAMETER, _ERROR_INVALID_FUNCTION, _ERROR_NOT_SUPPORTED}
)


@dataclass(frozen=True, slots=True)
class AlignedSpan:
    """Sector-aligned byte range covering a checkpoint byte span."""

    aligned_offset: int
    aligned_length: int
    head_pad: int
    data_length: int

    @property
    def aligned_end(self) -> int:
        return self.aligned_offset + self.aligned_length


@dataclass(frozen=True, slots=True)
class DirectReadRequest:
    """One checkpoint byte-range read target."""

    source: FileSlice
    destination: memoryview
    head_pad: int = 0
    aligned_target: bool = True


def _validate_int(value: int, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        bound = "non-negative" if minimum == 0 else ("positive" if minimum == 1 else f">= {minimum}")
        raise ValueError(f"{name} must be a {bound} integer, got {value!r}.")


def _validate_power_of_two(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"{name} must be a positive power of two, got {value!r}.")


def align_span(
    offset: int,
    length: int,
    *,
    sector_size: int = DEFAULT_SECTOR_SIZE,
) -> AlignedSpan:
    """Round ``[offset, offset + length)`` outward to ``sector_size`` boundaries."""
    _validate_power_of_two(sector_size, "sector_size")
    _validate_int(offset, "File offset", minimum=0)
    _validate_int(length, "Read length", minimum=0)
    if length == 0:
        return AlignedSpan(0, 0, 0, 0)
    aligned_offset = (offset // sector_size) * sector_size
    head_pad = offset - aligned_offset
    end = offset + length
    aligned_end = ((end + sector_size - 1) // sector_size) * sector_size
    return AlignedSpan(
        aligned_offset=aligned_offset,
        aligned_length=aligned_end - aligned_offset,
        head_pad=head_pad,
        data_length=length,
    )


def _buffer_pointer(view: memoryview) -> int:
    if view.nbytes == 0:
        return 0
    return ctypes.addressof(ctypes.c_char.from_buffer(view))


def _same_open_file(
    file: io.BufferedReader,
    other_fd_or_path: int | str | bytes | os.PathLike[str] | os.PathLike[bytes],
) -> bool:
    orig_stat = os.fstat(file.fileno())
    target_stat = (
        os.fstat(other_fd_or_path)
        if isinstance(other_fd_or_path, int)
        else Path(os.fsdecode(other_fd_or_path)).stat()
    )
    if orig_stat.st_ino == 0 and target_stat.st_ino == 0:
        # Filesystems/platforms without inode identifiers rely on the caller's open reader path.
        return True
    return (target_stat.st_dev, target_stat.st_ino) == (orig_stat.st_dev, orig_stat.st_ino)


class _CloseGate:
    """One-way closing gate tracking active operations and draining before teardown."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._closing = False
        self._closed = False
        self._inflight = 0

    @contextmanager
    def enter(self) -> Iterator[None]:
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("DirectFileReader is closed.")
            self._inflight += 1
        try:
            yield
        finally:
            with self._lock:
                self._inflight -= 1
                if self._inflight == 0:
                    self._cv.notify_all()

    def close_and_drain(self) -> tuple[bool, BaseException | None]:
        interrupt: BaseException | None = None
        with self._lock:
            if self._closed or self._closing:
                return False, None
            self._closing = True
            while self._inflight > 0:
                try:
                    self._cv.wait()
                except KeyboardInterrupt as exc:
                    if interrupt is None:
                        interrupt = exc
            self._closed = True
        return True, interrupt


class _PoolClosedError(RuntimeError):
    """Raised when checking out from a drained/closed resource pool."""


class _ResourcePool[T]:
    """Bounded checkout/return pool closing idle items on drain and active items on checkin."""

    def __init__(
        self,
        factory: Callable[[], T],
        destroy: Callable[[T], None],
        *,
        max_idle: int = 8,
    ) -> None:
        self._factory = factory
        self._destroy = destroy
        self._max_idle = max_idle
        self._lock = threading.Lock()
        self._idle: list[T] = []
        self._closed = False

    def _safe_destroy(self, item: T) -> None:
        with contextlib.suppress(OSError, BufferError):
            self._destroy(item)

    def seed(self, item: T) -> None:
        destroy_now = False
        with self._lock:
            if self._closed or len(self._idle) >= self._max_idle:
                destroy_now = True
            else:
                self._idle.append(item)
        if destroy_now:
            self._safe_destroy(item)

    def checkout(
        self,
        accept: Callable[[T], bool] | None = None,
        create: Callable[[], T] | None = None,
    ) -> T:
        evict: T | None = None
        with self._lock:
            if self._closed:
                raise _PoolClosedError("Resource pool is closed.")
            for idx in range(len(self._idle) - 1, -1, -1):
                candidate = self._idle[idx]
                if accept is None or accept(candidate):
                    return self._idle.pop(idx)
            if self._idle and (accept is not None or len(self._idle) >= self._max_idle):
                evict = self._idle.pop(0)
        if evict is not None:
            self._safe_destroy(evict)

        item = (self._factory if create is None else create)()
        with self._lock:
            closed = self._closed
        if closed:
            self._safe_destroy(item)
            raise _PoolClosedError("Resource pool is closed.")
        return item

    def checkin(self, item: T, *, retire: bool = False) -> None:
        destroy_now = False
        with self._lock:
            if retire or self._closed or len(self._idle) >= self._max_idle:
                destroy_now = True
            else:
                self._idle.append(item)
        if destroy_now:
            self._safe_destroy(item)

    def drain_and_close(self) -> None:
        with self._lock:
            self._closed = True
            items, self._idle = self._idle, []
        for item in items:
            self._safe_destroy(item)


class _DowngradeDirectError(Exception):
    """Internal control signal requesting permanent downgrade to buffered reads."""


class _IOBackend(Protocol):
    @property
    def is_direct(self) -> bool: ...

    @property
    def sector_size(self) -> int: ...

    def disable_direct(self) -> None: ...

    def file_size(self) -> int | None: ...

    def read_direct_chunk(self, target: memoryview, offset: int) -> int: ...

    def read_buffered_chunk(self, target: memoryview, offset: int) -> int: ...

    def after_buffered_read(self, aligned_offset: int, aligned_length: int, /) -> None: ...

    def close(self) -> None: ...


class _ManagedFileHandle:
    """Single owner of aligned validation, short-read/EOF classification, and fallback retry."""

    def __init__(self, backend: _IOBackend) -> None:
        self._backend = backend

    @property
    def is_direct(self) -> bool:
        return self._backend.is_direct

    @property
    def sector_size(self) -> int:
        return self._backend.sector_size

    def _validate_read_args(
        self,
        aligned_offset: int,
        aligned_length: int,
        min_required_bytes: int,
        target: memoryview,
    ) -> None:
        sector = self._backend.sector_size
        _validate_int(aligned_offset, "aligned_offset", minimum=0)
        _validate_int(aligned_length, "aligned_length", minimum=0)
        _validate_int(min_required_bytes, "min_required_bytes", minimum=0)
        if aligned_offset % sector != 0:
            raise ValueError(f"aligned_offset {aligned_offset} is not aligned to {sector} bytes.")
        if aligned_length % sector != 0:
            raise ValueError(f"aligned_length {aligned_length} is not aligned to {sector} bytes.")
        if min_required_bytes > aligned_length:
            raise ValueError(
                f"min_required_bytes {min_required_bytes} exceeds aligned_length {aligned_length}."
            )
        if target.readonly:
            raise ValueError("Aligned read target must be writable.")
        if not target.c_contiguous:
            raise ValueError("Aligned read target must be C-contiguous.")
        if len(target) < aligned_length:
            raise ValueError(
                f"Aligned read target length {len(target)} is smaller than {aligned_length}."
            )
        if aligned_length > 0 and _buffer_pointer(target) % sector != 0:
            raise ValueError(f"Direct read buffer must be aligned to {sector} bytes.")

    def _attempt_read(
        self,
        direct_mode: bool,
        aligned_offset: int,
        aligned_length: int,
        min_required_bytes: int,
        target: memoryview,
    ) -> int:
        sector = self._backend.sector_size
        done = 0
        while done < min_required_bytes:
            if direct_mode and (done % sector != 0 or (aligned_length - done) % sector != 0):
                known_size = self._backend.file_size()
                if known_size is not None and aligned_offset + done >= known_size:
                    raise OSError(f"checkpoint ended at byte {aligned_offset + done}")
                raise _DowngradeDirectError
            with target[done:aligned_length] as slice_mv:
                count = (
                    self._backend.read_direct_chunk(slice_mv, aligned_offset + done)
                    if direct_mode
                    else self._backend.read_buffered_chunk(slice_mv, aligned_offset + done)
                )
            if count <= 0:
                raise OSError(f"checkpoint ended at byte {aligned_offset + done}")
            done += count
        return done

    def read_aligned(
        self,
        aligned_offset: int,
        aligned_length: int,
        min_required_bytes: int,
        target: memoryview,
    ) -> int:
        self._validate_read_args(aligned_offset, aligned_length, min_required_bytes, target)
        if min_required_bytes == 0:
            return 0
        while True:
            direct_mode = self._backend.is_direct
            try:
                done = self._attempt_read(
                    direct_mode,
                    aligned_offset,
                    aligned_length,
                    min_required_bytes,
                    target,
                )
            except _DowngradeDirectError:
                self._backend.disable_direct()
                continue
            if not direct_mode:
                self._backend.after_buffered_read(aligned_offset, aligned_length)
            return done

    def close(self) -> None:
        self._backend.close()


class _PosixBackend:
    """POSIX positional ``preadv`` backend preferring ``O_DIRECT`` with buffered fallback."""

    def __init__(self, file: io.BufferedReader, *, direct_io: bool) -> None:
        self._file_ref = weakref.ref(file)
        self._sector_size = DEFAULT_SECTOR_SIZE
        self._lock = threading.Lock()
        self._caller_file_lock = threading.Lock()
        self._direct_fd: int | None = None
        self._retired_fds: list[int] = []
        self._closed = False

        raw_name = getattr(file, "name", None)
        if direct_io and _O_DIRECT and hasattr(os, "preadv") and isinstance(raw_name, (str, bytes, os.PathLike)):
            try:
                opened = os.open(raw_name, os.O_RDONLY | _O_DIRECT)
            except OSError:
                opened = None
            if opened is not None:
                try:
                    if _same_open_file(file, opened):
                        self._direct_fd = opened
                    else:
                        os.close(opened)
                except BaseException:
                    with contextlib.suppress(OSError):
                        os.close(opened)
                    raise

    def _require_file(self) -> io.BufferedReader:
        file = self._file_ref()
        if file is None:
            raise RuntimeError("Backing file has been closed or garbage-collected.")
        return file

    @property
    def is_direct(self) -> bool:
        with self._lock:
            return self._direct_fd is not None

    @property
    def sector_size(self) -> int:
        return self._sector_size

    def disable_direct(self) -> None:
        with self._lock:
            if self._direct_fd is not None:
                self._retired_fds.append(self._direct_fd)
                self._direct_fd = None

    def file_size(self) -> int | None:
        file = self._file_ref()
        if file is None:
            return None
        try:
            return os.fstat(file.fileno()).st_size
        except (OSError, ValueError):
            return None

    def _preadv_or_fail(self, fd: int, target: memoryview, offset: int) -> int:
        preadv = cast(Callable[[int, list[memoryview], int], int] | None, getattr(os, "preadv", None))
        if preadv is None:
            raise OSError("os.preadv is unavailable on this platform.")
        return preadv(fd, [target], offset)

    def read_direct_chunk(self, target: memoryview, offset: int) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("DirectFileReader handle is closed.")
            fd = self._direct_fd
        if fd is None:
            raise _DowngradeDirectError
        try:
            return self._preadv_or_fail(fd, target, offset)
        except OSError as exc:
            if exc.errno == errno.EINVAL:
                raise _DowngradeDirectError from exc
            raise

    def read_buffered_chunk(self, target: memoryview, offset: int) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("DirectFileReader handle is closed.")
        file = self._require_file()
        if hasattr(os, "preadv"):
            return self._preadv_or_fail(file.fileno(), target, offset)
        with self._caller_file_lock:
            file.seek(offset)
            return file.readinto(target) or 0

    def after_buffered_read(self, aligned_offset: int, aligned_length: int) -> None:
        if aligned_length <= 0:
            return
        file = self._file_ref()
        if file is None:
            return
        posix_fadvise = cast(Callable[[int, int, int, int], None] | None, getattr(os, "posix_fadvise", None))
        if posix_fadvise is not None:
            with contextlib.suppress(OSError, ValueError):
                posix_fadvise(file.fileno(), aligned_offset, aligned_length, _POSIX_FADV_DONTNEED)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._direct_fd is not None:
                self._retired_fds.append(self._direct_fd)
                self._direct_fd = None
            to_close, self._retired_fds = self._retired_fds, []
        for fd in to_close:
            with contextlib.suppress(OSError):
                os.close(fd)


class _Overlapped(ctypes.Structure):
    _fields_ = (
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", ctypes.c_uint32),
        ("OffsetHigh", ctypes.c_uint32),
        ("hEvent", ctypes.c_void_p),
    )


class _FileTime(ctypes.Structure):
    _fields_ = (
        ("dwLowDateTime", ctypes.c_uint32),
        ("dwHighDateTime", ctypes.c_uint32),
    )


class _ByHandleFileInfo(ctypes.Structure):
    _fields_ = (
        ("dwFileAttributes", ctypes.c_uint32),
        ("ftCreationTime", _FileTime),
        ("ftLastAccessTime", _FileTime),
        ("ftLastWriteTime", _FileTime),
        ("dwVolumeSerialNumber", ctypes.c_uint32),
        ("nFileSizeHigh", ctypes.c_uint32),
        ("nFileSizeLow", ctypes.c_uint32),
        ("nNumberOfLinks", ctypes.c_uint32),
        ("nFileIndexHigh", ctypes.c_uint32),
        ("nFileIndexLow", ctypes.c_uint32),
    )


@dataclass(frozen=True, slots=True)
class _Win32IO:
    open_direct: Callable[[str], int]
    read_at: Callable[[int, memoryview, int], int]
    close_handle: Callable[[int], None]
    sector_size: Callable[[str], int]
    same_handle_file: Callable[[int, io.BufferedReader], bool] | None = None


@cache
def _configure_kernel32() -> object:
    windll = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
    windll.CreateFileW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    windll.CreateFileW.restype = ctypes.c_void_p
    windll.ReadFile.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(_Overlapped),
    )
    windll.ReadFile.restype = ctypes.c_int
    windll.CloseHandle.argtypes = (ctypes.c_void_p,)
    windll.CloseHandle.restype = ctypes.c_int
    windll.GetDiskFreeSpaceW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    )
    windll.GetDiskFreeSpaceW.restype = ctypes.c_int
    windll.GetVolumePathNameW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    windll.GetVolumePathNameW.restype = ctypes.c_int
    windll.GetFileInformationByHandle.argtypes = (ctypes.c_void_p, ctypes.POINTER(_ByHandleFileInfo))
    windll.GetFileInformationByHandle.restype = ctypes.c_int
    windll.GetFinalPathNameByHandleW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32)
    windll.GetFinalPathNameByHandleW.restype = ctypes.c_uint32
    return windll


@cache
def _default_win32_io() -> _Win32IO:
    if sys.platform != "win32":
        raise OSError("Windows Direct I/O is only available on win32.")

    windll = _configure_kernel32()
    invalid_handle = ctypes.c_void_p(-1).value

    def _raise_win_error(action: str, err: int) -> OSError:
        return OSError(errno.EIO, f"{action} failed with Windows error {err}", None, err)

    def open_direct(path: str) -> int:
        handle = windll.CreateFileW(  # type: ignore[attr-defined]
            path,
            _GENERIC_READ,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_RANDOM_ACCESS,
            None,
        )
        if handle is None or handle == invalid_handle:
            raise _raise_win_error("CreateFileW", ctypes.get_last_error())
        return int(handle)

    def read_at(handle: int, target: memoryview, offset: int) -> int:
        _validate_int(offset, "offset", minimum=0)
        if offset >= (1 << 64):
            raise ValueError(f"offset exceeds 64-bit Win32 range: {offset}.")
        if len(target) == 0:
            return 0
        chunk_len = min(len(target), _WIN32_MAX_READ_BYTES)
        overlapped = _Overlapped(
            Internal=0,
            InternalHigh=0,
            Offset=offset & 0xFFFFFFFF,
            OffsetHigh=(offset >> 32) & 0xFFFFFFFF,
            hEvent=None,
        )
        bytes_read = ctypes.c_uint32(0)
        ok = windll.ReadFile(  # type: ignore[attr-defined]
            ctypes.c_void_p(handle),
            ctypes.c_void_p(_buffer_pointer(target)),
            chunk_len,
            ctypes.byref(bytes_read),
            ctypes.byref(overlapped),
        )
        if not ok:
            err = ctypes.get_last_error()
            if err == _ERROR_HANDLE_EOF:
                return 0
            raise _raise_win_error("ReadFile", err)
        return int(bytes_read.value)

    def close_handle(handle: int) -> None:
        windll.CloseHandle(ctypes.c_void_p(handle))  # type: ignore[attr-defined]

    def sector_size(path: str) -> int:
        try:
            abs_path = str(Path(path).resolve())
        except OSError:
            abs_path = path
        vol_buf = ctypes.create_unicode_buffer(max(260, len(abs_path) + 8))
        if windll.GetVolumePathNameW(abs_path, vol_buf, len(vol_buf)):  # type: ignore[attr-defined]
            root: str | None = vol_buf.value
        else:
            drive, _ = os.path.splitdrive(abs_path)
            root = f"{drive}\\" if drive else None
        spc, bps, free_c, total_c = (ctypes.c_uint32(0) for _ in range(4))
        ok = windll.GetDiskFreeSpaceW(  # type: ignore[attr-defined]
            root, ctypes.byref(spc), ctypes.byref(bps), ctypes.byref(free_c), ctypes.byref(total_c)
        )
        if not ok:
            return DEFAULT_SECTOR_SIZE
        value = int(bps.value)
        if value <= 0 or (value & (value - 1)) != 0:
            return DEFAULT_SECTOR_SIZE
        return max(DEFAULT_SECTOR_SIZE, value)

    def same_handle_file(direct_handle: int, file: io.BufferedReader) -> bool:
        try:
            orig_handle = msvcrt.get_osfhandle(file.fileno())
        except (OSError, ValueError):
            return False
        info_a = _ByHandleFileInfo()
        info_b = _ByHandleFileInfo()
        fn = windll.GetFileInformationByHandle  # type: ignore[attr-defined]
        if not fn(ctypes.c_void_p(direct_handle), ctypes.byref(info_a)):
            return False
        if not fn(ctypes.c_void_p(orig_handle), ctypes.byref(info_b)):
            return False
        return (info_a.dwVolumeSerialNumber, info_a.nFileIndexHigh, info_a.nFileIndexLow) == (
            info_b.dwVolumeSerialNumber,
            info_b.nFileIndexHigh,
            info_b.nFileIndexLow,
        )

    return _Win32IO(
        open_direct=open_direct,
        read_at=read_at,
        close_handle=close_handle,
        sector_size=sector_size,
        same_handle_file=same_handle_file,
    )


def _resolve_open_path(file: io.BufferedReader) -> str:
    if sys.platform == "win32":
        windll = _configure_kernel32()
        try:
            os_handle = msvcrt.get_osfhandle(file.fileno())
            size = windll.GetFinalPathNameByHandleW(ctypes.c_void_p(os_handle), None, 0, 0)  # type: ignore[attr-defined]
            if size > 0:
                buf = ctypes.create_unicode_buffer(size + 1)
                written = windll.GetFinalPathNameByHandleW(  # type: ignore[attr-defined]
                    ctypes.c_void_p(os_handle), buf, len(buf), 0
                )
                if 0 < written < len(buf):
                    return buf.value
        except (OSError, ValueError):
            pass
    raw_name = getattr(file, "name", None)
    if isinstance(raw_name, (str, bytes, os.PathLike)):
        return str(Path(os.fsdecode(raw_name)).absolute())
    return ""


def _is_win_downgrade_error(exc: OSError) -> bool:
    winerror = getattr(exc, "winerror", None)
    return winerror in _WIN_DOWNGRADE_ERRORS


class _FallbackToCallerFileError(Exception):
    """Signals that the path could not be verified and buffered reads must use caller's file."""


class _WindowsBackend:
    """Windows ``FILE_FLAG_NO_BUFFERING`` backend with pooled handles and identity verification."""

    def __init__(
        self,
        file: io.BufferedReader,
        *,
        direct_io: bool,
        max_pool_handles: int,
        win32: _Win32IO | None = None,
    ) -> None:
        self._file_ref = weakref.ref(file)
        self._path = _resolve_open_path(file)
        self._win32 = _default_win32_io() if win32 is None else win32
        raw_sector = self._win32.sector_size(self._path) if self._path else DEFAULT_SECTOR_SIZE
        self._sector_size = max(DEFAULT_SECTOR_SIZE, raw_sector)
        _validate_power_of_two(self._sector_size, "sector_size")

        self._lock = threading.Lock()
        self._caller_file_lock = threading.Lock()
        self._direct_active = False
        self._use_caller_file = not self._path
        self._closed = False

        self._direct_pool = _ResourcePool(
            self._open_verified_direct,
            self._win32.close_handle,
            max_idle=max_pool_handles,
        )
        self._buffered_pool = _ResourcePool(
            self._open_verified_buffered,
            lambda f: f.close(),
            max_idle=max_pool_handles,
        )

        if direct_io and self._path:
            try:
                handle = self._open_verified_direct()
            except OSError:
                self._direct_pool.drain_and_close()
            else:
                self._direct_active = True
                self._direct_pool.seed(handle)
        else:
            self._direct_pool.drain_and_close()

    def _require_file(self) -> io.BufferedReader:
        file = self._file_ref()
        if file is None:
            raise RuntimeError("Backing file has been closed or garbage-collected.")
        return file

    @property
    def is_direct(self) -> bool:
        with self._lock:
            return self._direct_active

    @property
    def sector_size(self) -> int:
        return self._sector_size

    def _verify_direct_handle(self, handle: int) -> bool:
        file = self._file_ref()
        if file is None:
            return False
        if self._win32.same_handle_file is not None:
            return self._win32.same_handle_file(handle, file)
        try:
            return _same_open_file(file, self._path)
        except OSError:
            return False

    def _open_verified_direct(self) -> int:
        if not self._path:
            raise OSError("Cannot open direct handle without filesystem path.")
        opened = self._win32.open_direct(self._path)
        try:
            if not self._verify_direct_handle(opened):
                raise OSError("Opened direct path no longer matches original file.")
        except BaseException:
            with contextlib.suppress(OSError):
                self._win32.close_handle(opened)
            raise
        return opened

    def _open_verified_buffered(self) -> io.FileIO:
        if not self._path:
            raise _FallbackToCallerFileError
        file = self._require_file()
        try:
            opened = open(self._path, "rb", buffering=0)  # noqa: SIM115 - owned by _buffered_pool
        except OSError as exc:
            raise _FallbackToCallerFileError from exc
        try:
            if not _same_open_file(file, opened.fileno()):
                raise _FallbackToCallerFileError
        except BaseException:
            with contextlib.suppress(OSError):
                opened.close()
            raise
        return opened

    def disable_direct(self) -> None:
        with self._lock:
            self._direct_active = False
        self._direct_pool.drain_and_close()

    def file_size(self) -> int | None:
        file = self._file_ref()
        if file is None:
            return None
        try:
            return os.fstat(file.fileno()).st_size
        except (OSError, ValueError):
            return None

    def read_direct_chunk(self, target: memoryview, offset: int) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("DirectFileReader handle is closed.")
            if not self._direct_active:
                raise _DowngradeDirectError
        try:
            handle = self._direct_pool.checkout()
        except (_PoolClosedError, OSError) as exc:
            raise _DowngradeDirectError from exc

        retire = False
        try:
            return self._win32.read_at(handle, target, offset)
        except OSError as exc:
            if _is_win_downgrade_error(exc):
                retire = True
                raise _DowngradeDirectError from exc
            raise
        finally:
            self._direct_pool.checkin(handle, retire=retire)

    def read_buffered_chunk(self, target: memoryview, offset: int) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("DirectFileReader handle is closed.")
            use_caller = self._use_caller_file
        if not use_caller:
            try:
                buffered = self._buffered_pool.checkout()
            except (_PoolClosedError, _FallbackToCallerFileError):
                with self._lock:
                    self._use_caller_file = True
                self._buffered_pool.drain_and_close()
            else:
                try:
                    buffered.seek(offset)
                    return buffered.readinto(target) or 0
                finally:
                    self._buffered_pool.checkin(buffered)

        file = self._require_file()
        with self._caller_file_lock:
            file.seek(offset)
            return file.readinto(target) or 0

    def after_buffered_read(self, _aligned_offset: int, _aligned_length: int) -> None:
        return None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._direct_active = False
        self._direct_pool.drain_and_close()
        self._buffered_pool.drain_and_close()


@dataclass(slots=True)
class _Scratch:
    region: host_memory.AlignedRegion
    size: int
    base_ptr: int

    def close(self) -> None:
        with contextlib.suppress(BufferError, OSError):
            self.region.close()


def _normalize_byte_view(view: memoryview) -> tuple[memoryview, memoryview | None]:
    if view.format == "B" and view.ndim == 1:
        return view, None
    cast_view = view.cast("B")
    return cast_view, cast_view


@dataclass(frozen=True, slots=True)
class _PreparedItem:
    request: DirectReadRequest
    span: AlignedSpan
    dest_bytes: memoryview


@dataclass(slots=True)
class _HandleEntry:
    weak_file: weakref.ref[io.BufferedReader]
    handle: _ManagedFileHandle
    finalizer: weakref.finalize[[], Any]


class DirectFileReader:
    """Reusable parallel Direct I/O reader for ``FileSlice`` checkpoint spans."""

    def __init__(
        self,
        *,
        direct_io: bool = True,
        max_workers: int | None = None,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        coalesce_gap: int = DEFAULT_COALESCE_GAP,
        coalesce_max_bytes: int = DEFAULT_COALESCE_MAX_BYTES,
        coalesce_threshold: int = DEFAULT_COALESCE_THRESHOLD,
        win32_io: _Win32IO | None = None,
    ) -> None:
        _validate_int(chunk_bytes, "chunk_bytes", minimum=1)
        if chunk_bytes % DEFAULT_SECTOR_SIZE != 0:
            raise ValueError(f"chunk_bytes must be a multiple of {DEFAULT_SECTOR_SIZE}, got {chunk_bytes}.")
        _validate_int(coalesce_gap, "coalesce_gap", minimum=0)
        _validate_int(coalesce_max_bytes, "coalesce_max_bytes", minimum=1)
        if coalesce_max_bytes % DEFAULT_SECTOR_SIZE != 0:
            raise ValueError(
                f"coalesce_max_bytes must be a multiple of {DEFAULT_SECTOR_SIZE}, got {coalesce_max_bytes}."
            )
        _validate_int(coalesce_threshold, "coalesce_threshold", minimum=0)
        if max_workers is not None:
            _validate_int(max_workers, "max_workers", minimum=1)

        default_workers = min(8, max(4, os.process_cpu_count() or 4))
        self._direct_io = direct_io
        self._max_workers = default_workers if max_workers is None else max_workers
        self._chunk_bytes = chunk_bytes
        self._coalesce_gap = coalesce_gap
        self._coalesce_max_bytes = coalesce_max_bytes
        self._coalesce_threshold = coalesce_threshold
        self._win32_io = win32_io

        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._gate = _CloseGate()
        self._handles: dict[int, _HandleEntry] = {}
        self._pool: ThreadPoolExecutor | None = None
        self._scratch_pool = _ResourcePool(
            lambda: self._allocate_scratch(DEFAULT_SECTOR_SIZE, DEFAULT_SECTOR_SIZE),
            lambda scratch: scratch.close(),
            max_idle=max(self._max_workers, 4),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _allocate_scratch(size: int, alignment: int) -> _Scratch:
        page_unit = max(DEFAULT_SECTOR_SIZE, mmap.PAGESIZE)
        extra = max(0, alignment - mmap.PAGESIZE)
        new_size = -(-(size + extra) // page_unit) * page_unit
        region = host_memory.allocate_aligned_region(new_size)
        mv = memoryview(region)
        try:
            base_ptr = _buffer_pointer(mv)
        finally:
            mv.release()
        pad = (-base_ptr) % alignment
        if pad + size > new_size:
            region.close()
            raise ValueError(f"Allocated scratch buffer cannot satisfy {alignment}-byte sector alignment.")
        return _Scratch(region=region, size=new_size, base_ptr=base_ptr)

    def _prune_dead_handles_locked(self) -> list[_ManagedFileHandle]:
        dead_keys = [k for k, entry in self._handles.items() if entry.weak_file() is None]
        retired: list[_ManagedFileHandle] = []
        for k in dead_keys:
            entry = self._handles.pop(k)
            entry.finalizer.detach()
            retired.append(entry.handle)
        return retired

    def _create_handle(self, file: io.BufferedReader) -> _ManagedFileHandle:
        if self._win32_io is not None or sys.platform == "win32":
            backend: _IOBackend = _WindowsBackend(
                file,
                direct_io=self._direct_io,
                max_pool_handles=max(self._max_workers, 4),
                win32=self._win32_io,
            )
        else:
            backend = _PosixBackend(file, direct_io=self._direct_io)
        return _ManagedFileHandle(backend)

    def _handle_for(self, file: io.BufferedReader) -> _ManagedFileHandle:
        key = id(file)
        stale: list[_ManagedFileHandle] = []
        with self._lock:
            entry = self._handles.get(key)
            if entry is not None:
                if entry.weak_file() is file:
                    return entry.handle
                self._handles.pop(key)
                entry.finalizer.detach()
                stale.append(entry.handle)
        for handle in stale:
            handle.close()

        new_handle = self._create_handle(file)
        discard: list[_ManagedFileHandle] = []
        chosen = new_handle
        with self._lock:
            discard.extend(self._prune_dead_handles_locked())
            existing = self._handles.get(key)
            if existing is not None and existing.weak_file() is file:
                discard.append(new_handle)
                chosen = existing.handle
            else:
                if existing is not None:
                    self._handles.pop(key)
                    existing.finalizer.detach()
                    discard.append(existing.handle)
                fin: weakref.finalize[[], Any] = weakref.finalize(file, new_handle.close)
                fin.atexit = False
                self._handles[key] = _HandleEntry(weak_file=weakref.ref(file), handle=new_handle, finalizer=fin)
        for handle in discard:
            handle.close()
        return chosen

    def is_direct(self, file: io.BufferedReader) -> bool:
        """Return whether Direct I/O is currently active for ``file``."""
        with self._gate.enter():
            return self._handle_for(file).is_direct

    def sector_size(self, file: io.BufferedReader) -> int:
        """Return the required alignment sector size in bytes for ``file``."""
        with self._gate.enter():
            return self._handle_for(file).sector_size

    def read_aligned(
        self,
        file: io.BufferedReader,
        aligned_offset: int,
        aligned_length: int,
        min_required_bytes: int,
        destination: memoryview,
    ) -> int:
        """Read one already-aligned span from ``file`` into ``destination``."""
        with self._gate.enter():
            if destination.readonly:
                raise ValueError("Aligned read target must be writable.")
            if not destination.c_contiguous:
                raise ValueError("Aligned read target must be C-contiguous.")
            byte_view, temp_view = _normalize_byte_view(destination)
            try:
                return self._handle_for(file).read_aligned(
                    aligned_offset,
                    aligned_length,
                    min_required_bytes,
                    byte_view,
                )
            finally:
                if temp_view is not None:
                    temp_view.release()

    def _submit_or_inline(self, tasks: Sequence[Callable[[], object]]) -> None:
        if not tasks:
            return
        if len(tasks) == 1 or self._max_workers == 1:
            for task in tasks:
                task()
            return
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="piper-offload-dio",
                )
            pool = self._pool

        futures: list[Future[object]] = []
        try:
            for task in tasks:
                futures.append(pool.submit(task))
            for future in futures:
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            for future in futures:
                while not future.done():
                    with contextlib.suppress(BaseException):
                        future.result()
            raise

    def _prepare_requests(
        self,
        requests: Sequence[DirectReadRequest],
        owned_cast_views: list[memoryview],
    ) -> dict[int, tuple[_ManagedFileHandle, list[_PreparedItem]]]:
        grouped: dict[int, tuple[_ManagedFileHandle, list[_PreparedItem]]] = {}
        for request in requests:
            source = request.source
            _validate_int(source.offset, "FileSlice.offset", minimum=0)
            _validate_int(source.length, "FileSlice.length", minimum=0)
            _validate_int(request.head_pad, "head_pad", minimum=0)
            if source.length == 0:
                continue
            dest = request.destination
            if dest.readonly:
                raise ValueError("DirectReadRequest destination must be writable.")
            if not dest.c_contiguous:
                raise ValueError("DirectReadRequest destination must be C-contiguous.")
            byte_view, temp_cast = _normalize_byte_view(dest)
            if temp_cast is not None:
                owned_cast_views.append(temp_cast)

            handle = self._handle_for(source.file)
            sector = handle.sector_size
            span = align_span(source.offset, source.length, sector_size=sector)
            self._validate_target_view(request, span, byte_view, sector)
            key = id(source.file)
            if key not in grouped:
                grouped[key] = (handle, [])
            grouped[key][1].append(_PreparedItem(request=request, span=span, dest_bytes=byte_view))
        return grouped

    @staticmethod
    def _validate_target_view(
        request: DirectReadRequest,
        span: AlignedSpan,
        byte_view: memoryview,
        sector: int,
    ) -> None:
        if request.aligned_target:
            if request.head_pad != span.head_pad:
                raise ValueError(
                    f"Aligned DirectReadRequest head_pad {request.head_pad} does not match "
                    f"aligned span head_pad {span.head_pad}."
                )
            if len(byte_view) < span.aligned_length:
                raise ValueError(
                    f"Aligned DirectReadRequest destination length {len(byte_view)} < {span.aligned_length}."
                )
            if _buffer_pointer(byte_view) % sector != 0:
                raise ValueError(
                    f"Aligned DirectReadRequest destination must be {sector}-byte aligned."
                )
            return
        if len(byte_view) < request.head_pad + request.source.length:
            raise ValueError(
                f"DirectReadRequest destination length {len(byte_view)} < "
                f"{request.head_pad + request.source.length}."
            )

    def read_into(self, requests: Sequence[DirectReadRequest]) -> None:
        """Execute all ``requests`` using coalesced scratch or zero-copy direct reads."""
        with self._gate.enter():
            if not requests:
                return
            owned_views: list[memoryview] = []
            try:
                grouped = self._prepare_requests(requests, owned_views)
                if not grouped:
                    return
                tasks: list[Callable[[], object]] = []
                for handle, items in grouped.values():
                    items.sort(key=lambda item: (item.span.aligned_offset, item.span.aligned_length))
                    self._plan_file_tasks(handle, items, tasks, owned_views)
                self._submit_or_inline(tasks)
            finally:
                for view in owned_views:
                    view.release()

    def _enqueue_batch(
        self,
        handle: _ManagedFileHandle,
        batch: list[_PreparedItem],
        batch_start: int,
        batch_end: int,
        chunk_step: int,
        tasks: list[Callable[[], object]],
        owned_views: list[memoryview],
    ) -> None:
        if len(batch) == 1 and batch[0].request.aligned_target:
            self._append_direct_tasks(handle, batch[0], chunk_step, tasks, owned_views)
            return
        slices = [
            (item.request.source.offset, item.span.data_length, item.request.head_pad, item.dest_bytes)
            for item in batch
        ]
        tasks.append(functools.partial(self._run_scratch_batch, handle, batch_start, batch_end, slices))

    def _plan_file_tasks(
        self,
        handle: _ManagedFileHandle,
        sorted_items: list[_PreparedItem],
        tasks: list[Callable[[], object]],
        owned_views: list[memoryview],
    ) -> None:
        sector = handle.sector_size
        chunk_step = max(sector, (self._chunk_bytes // sector) * sector)
        scratch_step = max(sector, (self._coalesce_max_bytes // sector) * sector)
        coalesce_limit = min(self._coalesce_max_bytes, scratch_step)

        batch: list[_PreparedItem] = []
        batch_start = 0
        batch_end = 0
        for item in sorted_items:
            span = item.span
            eligible = span.aligned_length <= self._coalesce_threshold and span.aligned_length <= coalesce_limit
            if eligible and not batch:
                batch, batch_start, batch_end = [item], span.aligned_offset, span.aligned_end
                continue
            if eligible and max(0, span.aligned_offset - batch_end) <= self._coalesce_gap:
                new_end = max(batch_end, span.aligned_end)
                if new_end - batch_start <= coalesce_limit:
                    batch.append(item)
                    batch_end = new_end
                    continue
            if batch:
                self._enqueue_batch(handle, batch, batch_start, batch_end, chunk_step, tasks, owned_views)
                batch = []
            if eligible:
                batch, batch_start, batch_end = [item], span.aligned_offset, span.aligned_end
            elif item.request.aligned_target:
                self._append_direct_tasks(handle, item, chunk_step, tasks, owned_views)
            else:
                self._append_unaligned_scratch_tasks(handle, item, scratch_step, tasks, owned_views)

        if batch:
            self._enqueue_batch(handle, batch, batch_start, batch_end, chunk_step, tasks, owned_views)

    @staticmethod
    def _append_direct_tasks(
        handle: _ManagedFileHandle,
        item: _PreparedItem,
        chunk_step: int,
        tasks: list[Callable[[], object]],
        owned_views: list[memoryview],
    ) -> None:
        span = item.span
        dest = item.dest_bytes
        if span.aligned_length <= chunk_step:
            min_req = span.head_pad + span.data_length
            tasks.append(
                functools.partial(handle.read_aligned, span.aligned_offset, span.aligned_length, min_req, dest)
            )
            return

        payload_end = span.aligned_offset + span.head_pad + span.data_length
        for rel in range(0, span.aligned_length, chunk_step):
            chunk_len = min(chunk_step, span.aligned_length - rel)
            chunk_off = span.aligned_offset + rel
            if chunk_off >= payload_end:
                break
            chunk_req = min(chunk_len, payload_end - chunk_off)
            sub_view = dest[rel : rel + chunk_len]
            owned_views.append(sub_view)
            tasks.append(functools.partial(handle.read_aligned, chunk_off, chunk_len, chunk_req, sub_view))

    def _append_unaligned_scratch_tasks(
        self,
        handle: _ManagedFileHandle,
        item: _PreparedItem,
        scratch_step: int,
        tasks: list[Callable[[], object]],
        owned_views: list[memoryview],
    ) -> None:
        span = item.span
        payload_start = span.aligned_offset + span.head_pad
        payload_end = payload_start + span.data_length
        dest = item.dest_bytes
        dest_base = item.request.head_pad

        for sub_off in range(span.aligned_offset, span.aligned_end, scratch_step):
            sub_end = min(span.aligned_end, sub_off + scratch_step)
            copy_start = max(sub_off, payload_start)
            copy_end = min(sub_end, payload_end)
            if copy_start >= copy_end:
                continue
            copy_len = copy_end - copy_start
            dst_off = dest_base + (copy_start - payload_start)
            sub_dest = dest[dst_off : dst_off + copy_len]
            owned_views.append(sub_dest)
            slices = [(copy_start, copy_len, 0, sub_dest)]
            tasks.append(functools.partial(self._run_scratch_batch, handle, sub_off, sub_end, slices))

    def _run_scratch_batch(
        self,
        handle: _ManagedFileHandle,
        batch_start: int,
        batch_end: int,
        slices: Sequence[tuple[int, int, int, memoryview]],
    ) -> None:
        total_len = batch_end - batch_start
        sector = handle.sector_size
        max_payload_end = max(file_off + length for file_off, length, _, _ in slices)
        min_required = max_payload_end - batch_start
        scratch = self._scratch_pool.checkout(
            accept=lambda item: ((-item.base_ptr) % sector) + total_len <= item.size,
            create=functools.partial(self._allocate_scratch, total_len, sector),
        )
        try:
            pad = (-scratch.base_ptr) % sector
            with memoryview(scratch.region)[pad : pad + total_len] as scratch_mv:
                handle.read_aligned(batch_start, total_len, min_required, scratch_mv)
                for file_off, length, dst_off, dest_bytes in slices:
                    rel = file_off - batch_start
                    dest_bytes[dst_off : dst_off + length] = scratch_mv[rel : rel + length]
        finally:
            self._scratch_pool.checkin(scratch)

    @staticmethod
    def _shutdown_pool(pool: ThreadPoolExecutor | None, interrupt: BaseException | None) -> BaseException | None:
        if pool is None:
            return interrupt
        while True:
            try:
                pool.shutdown(wait=True)
                return interrupt
            except KeyboardInterrupt as exc:
                if interrupt is None:
                    interrupt = exc
            except BaseException as exc:
                return exc if interrupt is None else interrupt

    def close(self) -> None:
        """Drain active I/O calls, join workers, and release handles and scratch buffers."""
        with self._close_lock:
            opened, interrupt = self._gate.close_and_drain()
            if not opened:
                return
            with self._lock:
                pool, self._pool = self._pool, None
                handles, self._handles = self._handles, {}

            interrupt = self._shutdown_pool(pool, interrupt)

            for entry in handles.values():
                try:
                    entry.finalizer.detach()
                    entry.handle.close()
                except BaseException as exc:
                    if interrupt is None:
                        interrupt = exc
            try:
                self._scratch_pool.drain_and_close()
            except BaseException as exc:
                if interrupt is None:
                    interrupt = exc
            if interrupt is not None:
                raise interrupt


def read_file_slice(
    source: FileSlice,
    destination: memoryview,
    *,
    head_pad: int = 0,
    aligned_target: bool = True,
    direct_io: bool = True,
) -> None:
    """Convenience one-shot helper reading a single :class:`FileSlice` into ``destination``."""
    with DirectFileReader(direct_io=direct_io) as reader:
        reader.read_into(
            [
                DirectReadRequest(
                    source=source,
                    destination=destination,
                    head_pad=head_pad,
                    aligned_target=aligned_target,
                )
            ]
        )


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_COALESCE_GAP",
    "DEFAULT_COALESCE_MAX_BYTES",
    "DEFAULT_COALESCE_THRESHOLD",
    "DEFAULT_SECTOR_SIZE",
    "AlignedSpan",
    "DirectFileReader",
    "DirectReadRequest",
    "align_span",
    "read_file_slice",
]
