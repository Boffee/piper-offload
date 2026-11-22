"""Linux CUDA/HIP IPC peer staging for same-machine multi-GPU collectives.

Exports one device buffer per rank via ``cudaIpcGetMemHandle`` /
``hipIpcGetMemHandle`` and opens peer buffers directly into a bounded staging
ring. Any failure during probe or handle exchange rolls back cleanly (closing
imported peer handles across a barrier before freeing local device memory) and
falls back to host-memory staging.
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, cast

import torch

from ._host_registration import load_device_runtime_library

logger = logging.getLogger(__name__)

_IPC_HANDLE_SIZE = 64
_BUS_ID_SIZE = 32
_IPC_LAZY_ENABLE_PEER_ACCESS = 0x01
_MEMCPY_DEFAULT = 4
_CLOSE_RENDEZVOUS_MAX_SECONDS = 5.0
_STATUS_FALLBACK = 0
_STATUS_OK = 1
_STATUS_ERROR = 2
_HANDLE_STRUCT = struct.Struct(f"<BQ{_BUS_ID_SIZE}s{_IPC_HANDLE_SIZE}s")

type IpcExchange = Callable[[bytes, timedelta], list[bytes]]
type IpcCloseBarrier = Callable[[timedelta], None]


class _IpcMemHandle(ctypes.Structure):
    # unsigned char preserves embedded NUL bytes inside opaque 64-byte handles.
    _fields_ = [("reserved", ctypes.c_ubyte * _IPC_HANDLE_SIZE)]


class IpcRuntime(Protocol):
    """Protocol for CUDA/HIP device allocation, IPC handle, and peer copy calls."""

    def check_prior_error(self) -> None: ...

    def device_get_pci_bus_id(self, device_index: int) -> str | None: ...

    def device_get_by_pci_bus_id(self, bus_id: str) -> int | None: ...

    def device_can_access_peer(self, device_index: int, peer_device_index: int) -> bool: ...

    def malloc(self, nbytes: int) -> int | None: ...

    def free(self, ptr: int) -> None: ...

    def ipc_get_mem_handle(self, ptr: int) -> bytes | None: ...

    def ipc_open_mem_handle(self, handle: bytes) -> int | None: ...

    def ipc_close_mem_handle(self, ptr: int) -> None: ...

    def memcpy_async(self, dst_ptr: int, src_ptr: int, nbytes: int, stream_ptr: int) -> None: ...


class DeviceIpcRuntime:
    """Bind CUDA or HIP IPC and peer-copy runtime entry points via ctypes."""

    def __init__(self, library: ctypes.CDLL, prefix: str) -> None:
        self._library = library
        get_last_error = getattr(library, f"{prefix}GetLastError")
        get_last_error.argtypes = []
        get_last_error.restype = ctypes.c_int
        self._get_last_error = cast(Callable[[], int], get_last_error)

        dev_bus_id = getattr(library, f"{prefix}DeviceGetPCIBusId")
        dev_bus_id.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        dev_bus_id.restype = ctypes.c_int
        self._device_get_pci_bus_id = cast(Callable[[int, int, int], int], dev_bus_id)

        dev_by_bus = getattr(library, f"{prefix}DeviceGetByPCIBusId")
        dev_by_bus.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_char_p]
        dev_by_bus.restype = ctypes.c_int
        self._device_get_by_pci_bus_id = cast(Callable[[object, bytes], int], dev_by_bus)

        can_peer = getattr(library, f"{prefix}DeviceCanAccessPeer")
        can_peer.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        can_peer.restype = ctypes.c_int
        self._device_can_access_peer = cast(Callable[[object, int, int], int], can_peer)

        malloc = getattr(library, f"{prefix}Malloc")
        malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        malloc.restype = ctypes.c_int
        self._malloc = cast(Callable[[object, int], int], malloc)

        free = getattr(library, f"{prefix}Free")
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_int
        self._free = cast(Callable[[int], int], free)

        get_handle = getattr(library, f"{prefix}IpcGetMemHandle")
        get_handle.argtypes = [ctypes.POINTER(_IpcMemHandle), ctypes.c_void_p]
        get_handle.restype = ctypes.c_int
        self._ipc_get_mem_handle = cast(Callable[[object, int], int], get_handle)

        open_handle = getattr(library, f"{prefix}IpcOpenMemHandle")
        open_handle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcMemHandle, ctypes.c_uint]
        open_handle.restype = ctypes.c_int
        self._ipc_open_mem_handle = cast(Callable[[object, _IpcMemHandle, int], int], open_handle)

        close_handle = getattr(library, f"{prefix}IpcCloseMemHandle")
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        self._ipc_close_mem_handle = cast(Callable[[int], int], close_handle)

        memcpy_async = getattr(library, f"{prefix}MemcpyAsync")
        memcpy_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        memcpy_async.restype = ctypes.c_int
        self._memcpy_async = cast(Callable[[int, int, int, int, int], int], memcpy_async)

    def check_prior_error(self) -> None:
        code = self._get_last_error()
        if code:
            raise RuntimeError(f"Prior CUDA/HIP runtime work failed with error {code}")

    def _clear_failed_call(self, operation: str, code: int) -> None:
        pending = self._get_last_error()
        if pending not in (0, code):
            raise RuntimeError(f"CUDA/HIP runtime error {pending} while handling {operation} failure {code}")

    def _cleanup_ptr(self, operation: str, release: Callable[[int], int], ptr: int) -> None:
        if not ptr:
            return
        prior = self._get_last_error()
        if prior:
            logger.warning("Cleared CUDA/HIP error %d from earlier work before %s", prior, operation)
        code = release(ptr)
        if code:
            pending = self._get_last_error()
            if pending not in (0, code):
                logger.warning("CUDA/HIP error %d while handling %s failure %d", pending, operation, code)
            else:
                logger.warning("CUDA/HIP %s failed with error %d", operation, code)

    def device_get_pci_bus_id(self, device_index: int) -> str | None:
        self.check_prior_error()
        buf = ctypes.create_string_buffer(64)
        code = self._device_get_pci_bus_id(ctypes.addressof(buf), len(buf), device_index)
        if code:
            self._clear_failed_call("device_get_pci_bus_id", code)
            return None
        bus_id = buf.value.decode("ascii").strip().lower()
        return bus_id or None

    def device_get_by_pci_bus_id(self, bus_id: str) -> int | None:
        self.check_prior_error()
        out_dev = ctypes.c_int()
        code = self._device_get_by_pci_bus_id(ctypes.byref(out_dev), bus_id.encode("ascii"))
        if code:
            self._clear_failed_call("device_get_by_pci_bus_id", code)
            return None
        return int(out_dev.value)

    def device_can_access_peer(self, device_index: int, peer_device_index: int) -> bool:
        self.check_prior_error()
        can_access = ctypes.c_int()
        code = self._device_can_access_peer(ctypes.byref(can_access), device_index, peer_device_index)
        if code:
            self._clear_failed_call("device_can_access_peer", code)
            return False
        return bool(can_access.value)

    def malloc(self, nbytes: int) -> int | None:
        self.check_prior_error()
        if nbytes <= 0:
            raise ValueError("IPC device allocation size must be positive")
        out_ptr = ctypes.c_void_p()
        code = self._malloc(ctypes.byref(out_ptr), nbytes)
        if code:
            self._clear_failed_call("malloc", code)
            return None
        return int(out_ptr.value or 0) or None

    def free(self, ptr: int) -> None:
        self._cleanup_ptr("free", self._free, ptr)

    def ipc_get_mem_handle(self, ptr: int) -> bytes | None:
        self.check_prior_error()
        handle = _IpcMemHandle()
        code = self._ipc_get_mem_handle(ctypes.byref(handle), ptr)
        if code:
            self._clear_failed_call("ipc_get_mem_handle", code)
            return None
        return bytes(handle.reserved)

    def ipc_open_mem_handle(self, handle: bytes) -> int | None:
        self.check_prior_error()
        if len(handle) != _IPC_HANDLE_SIZE:
            return None
        raw = _IpcMemHandle((ctypes.c_ubyte * _IPC_HANDLE_SIZE).from_buffer_copy(handle))
        out_ptr = ctypes.c_void_p()
        code = self._ipc_open_mem_handle(ctypes.byref(out_ptr), raw, _IPC_LAZY_ENABLE_PEER_ACCESS)
        if code:
            self._clear_failed_call("ipc_open_mem_handle", code)
            return None
        return int(out_ptr.value or 0) or None

    def ipc_close_mem_handle(self, ptr: int) -> None:
        self._cleanup_ptr("ipc_close_mem_handle", self._ipc_close_mem_handle, ptr)

    def memcpy_async(self, dst_ptr: int, src_ptr: int, nbytes: int, stream_ptr: int) -> None:
        if nbytes == 0:
            return
        code = self._memcpy_async(dst_ptr, src_ptr, nbytes, _MEMCPY_DEFAULT, stream_ptr)
        if code:
            self._clear_failed_call("memcpy_async", code)
            raise RuntimeError(f"CUDA/HIP memcpyAsync failed with error {code}")


def load_ipc_runtime() -> IpcRuntime | None:
    """Return the Linux CUDA/HIP IPC runtime wrapper, or None when unsupported."""
    if sys.platform != "linux":
        return None
    loaded = load_device_runtime_library()
    if loaded is None:
        return None
    library, prefix = loaded
    try:
        return DeviceIpcRuntime(library, prefix)
    except AttributeError as error:
        logger.info("CUDA/HIP IPC runtime symbols unavailable (%s); falling back to shared host memory", error)
        return None


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _device_context(device: torch.device) -> AbstractContextManager[object]:
    if torch.cuda.is_available():
        return torch.cuda.device(device)
    return nullcontext()


class IpcArena:
    """Per-rank device staging buffer and imported peer IPC handles for one process group."""

    def __init__(
        self,
        device: torch.device,
        rank: int,
        world_size: int,
        slot_bytes: int,
        buffer_count: int,
        local_ptr: int,
        peer_ptrs: dict[int, int],
        close_barrier: IpcCloseBarrier,
        runtime: IpcRuntime,
    ) -> None:
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.slot_bytes = slot_bytes
        self.buffer_count = buffer_count
        self._local_ptr = local_ptr
        self._peer_ptrs = peer_ptrs
        self._close_barrier = close_barrier
        self._runtime = runtime
        self._closed = False

    def _validate_slice(self, lane: int, offset: int, tensor: torch.Tensor) -> int:
        if self._closed:
            raise RuntimeError("CUDA/HIP IPC arena is closed")
        if not 0 <= lane < self.buffer_count:
            raise IndexError(f"IPC slot lane {lane} is out of bounds")
        nbytes = tensor.nbytes
        if offset < 0 or offset + nbytes > self.slot_bytes:
            raise IndexError(f"IPC slot slice [{offset}, {offset + nbytes}) exceeds {self.slot_bytes}")
        if tensor.device != self.device:
            raise ValueError(f"Expected tensor on {self.device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError("IPC slot transfers require a contiguous tensor")
        return nbytes

    def copy_to_slot(self, lane: int, offset: int, src: torch.Tensor, stream: torch.cuda.Stream) -> None:
        nbytes = self._validate_slice(lane, offset, src)
        if nbytes == 0:
            return
        dst_ptr = self._local_ptr + lane * self.slot_bytes + offset
        self._runtime.memcpy_async(dst_ptr, src.data_ptr(), nbytes, int(stream.cuda_stream))

    def copy_from_slot(self, peer: int, lane: int, offset: int, dst: torch.Tensor, stream: torch.cuda.Stream) -> None:
        nbytes = self._validate_slice(lane, offset, dst)
        if nbytes == 0:
            return
        src_ptr = self._peer_ptrs[peer] + lane * self.slot_bytes + offset
        self._runtime.memcpy_async(dst.data_ptr(), src_ptr, nbytes, int(stream.cuda_stream))

    def close(self, *, broken: bool = False, timeout: timedelta = timedelta(seconds=5)) -> None:
        if self._closed:
            return
        self._closed = True
        with _device_context(self.device):
            for ptr in self._peer_ptrs.values():
                self._runtime.ipc_close_mem_handle(ptr)
            self._peer_ptrs.clear()
            if not self._local_ptr:
                return
        if broken:
            logger.warning("Skipping CUDA/HIP IPC buffer free on rank %d after broken collective", self.rank)
            return
        wait_timeout = min(timeout, timedelta(seconds=_CLOSE_RENDEZVOUS_MAX_SECONDS))
        try:
            self._close_barrier(wait_timeout)
        except Exception as error:
            logger.warning(
                "Skipping CUDA/HIP IPC buffer free on rank %d after close rendezvous failure: %s",
                self.rank,
                str(error),
            )
            return
        with _device_context(self.device):
            self._runtime.free(self._local_ptr)
            self._local_ptr = 0


@dataclass(frozen=True, slots=True)
class IpcSlotView:
    """Sliceable byte view over a local outgoing or peer incoming IPC staging slot."""

    arena: IpcArena
    rank: int
    lane: int
    offset: int
    width: int
    stream: torch.cuda.Stream

    def numel(self) -> int:
        return self.width

    def __getitem__(self, key: slice) -> IpcSlotView:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("IPC slot views only support unit-stride slice indexing")
        start, stop, _ = key.indices(self.width)
        return IpcSlotView(self.arena, self.rank, self.lane, self.offset + start, max(0, stop - start), self.stream)

    def copy_(self, src: torch.Tensor, non_blocking: bool = False) -> None:
        del non_blocking
        if self.rank != self.arena.rank:
            raise ValueError("Cannot write to a peer IPC slot")
        if src.nbytes != self.width:
            raise ValueError(f"Expected {self.width} bytes for IPC slot write, got {src.nbytes}")
        self.arena.copy_to_slot(self.lane, self.offset, src, self.stream)

    def copy_to(self, dst: torch.Tensor) -> None:
        if self.rank == self.arena.rank:
            raise ValueError("Cannot read local IPC slot as peer slot")
        if dst.nbytes != self.width:
            raise ValueError(f"Expected {self.width} bytes for IPC slot read, got {dst.nbytes}")
        self.arena.copy_from_slot(self.rank, self.lane, self.offset, dst, self.stream)


def _close_peer_ptrs(rt: IpcRuntime | None, device: torch.device, peer_ptrs: dict[int, int]) -> None:
    if rt is not None and peer_ptrs:
        with _device_context(device):
            for ptr in peer_ptrs.values():
                rt.ipc_close_mem_handle(ptr)
        peer_ptrs.clear()


def _prepare_local_handle(
    runtime: IpcRuntime | None,
    device: torch.device,
    device_index: int,
    nbytes: int,
) -> tuple[IpcRuntime | None, int, bytes, Exception | None]:
    rt: IpcRuntime | None = None
    local_ptr = 0
    try:
        rt = load_ipc_runtime() if runtime is None else runtime
        if rt is not None:
            with _device_context(device):
                rt.check_prior_error()
                bus_id = rt.device_get_pci_bus_id(device_index)
                raw_bus = b"" if bus_id is None else bus_id.encode("ascii", errors="strict")
                if 0 < len(raw_bus) <= _BUS_ID_SIZE:
                    ptr = rt.malloc(nbytes)
                    if ptr is not None:
                        handle = rt.ipc_get_mem_handle(ptr)
                        if handle is not None and len(handle) == _IPC_HANDLE_SIZE:
                            local_ptr = ptr
                            payload = _HANDLE_STRUCT.pack(_STATUS_OK, os.getpid(), raw_bus, handle)
                            return rt, local_ptr, payload, None
                        rt.free(ptr)
        return rt, 0, _HANDLE_STRUCT.pack(_STATUS_FALLBACK, os.getpid(), b"", b""), None
    except Exception as error:
        if rt is not None and local_ptr:
            with _device_context(device):
                rt.free(local_ptr)
        return rt, 0, _HANDLE_STRUCT.pack(_STATUS_ERROR, 0, b"", b""), error
    except BaseException:
        if rt is not None and local_ptr:
            with _device_context(device):
                rt.free(local_ptr)
        raise


def _open_one_peer_handle(
    rt: IpcRuntime,
    device_index: int,
    my_bus_id: str,
    raw_bus: bytes,
    peer_handle: bytes,
) -> int | None:
    if len(peer_handle) != _IPC_HANDLE_SIZE:
        return None
    try:
        peer_bus_id = raw_bus.rstrip(b"\x00").decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return None
    if not peer_bus_id:
        return None
    if peer_bus_id != my_bus_id:
        peer_dev = rt.device_get_by_pci_bus_id(peer_bus_id)
        if peer_dev is not None and not rt.device_can_access_peer(device_index, peer_dev):
            return None
    return rt.ipc_open_mem_handle(peer_handle)


def _open_peer_handles(
    rt: IpcRuntime,
    rank: int,
    device: torch.device,
    device_index: int,
    records: Sequence[tuple[int, int, bytes, bytes]],
    peer_ptrs: dict[int, int],
) -> tuple[int, Exception | None]:
    try:
        my_bus_id = records[rank][2].rstrip(b"\x00").decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return _STATUS_FALLBACK, None
    if not my_bus_id:
        return _STATUS_FALLBACK, None
    try:
        with _device_context(device):
            rt.check_prior_error()
            for peer, rec in enumerate(records):
                if peer == rank:
                    continue
                opened = _open_one_peer_handle(rt, device_index, my_bus_id, rec[2], rec[3])
                if opened is None:
                    return _STATUS_FALLBACK, None
                peer_ptrs[peer] = opened
        return _STATUS_OK, None
    except Exception as error:
        return _STATUS_ERROR, error
    except BaseException:
        _close_peer_ptrs(rt, device, peer_ptrs)
        raise


def open_ipc_arena(
    exchange: IpcExchange,
    close_barrier: IpcCloseBarrier,
    rank: int,
    world_size: int,
    device: torch.device,
    slot_bytes: int,
    buffer_count: int,
    timeout: timedelta,
    *,
    runtime: IpcRuntime | None = None,
) -> IpcArena | None:
    """Probe and open peer GPU IPC staging handles across all ranks, or return None to fall back."""
    device_index = _device_index(device)
    rt, local_ptr, local_payload, phase1_error = _prepare_local_handle(
        runtime, device, device_index, buffer_count * slot_bytes,
    )
    records = [_HANDLE_STRUCT.unpack(raw) for raw in exchange(local_payload, timeout)]
    if any(rec[0] == _STATUS_ERROR for rec in records):
        if rt is not None and local_ptr:
            with _device_context(device):
                rt.free(local_ptr)
        if phase1_error is not None:
            raise phase1_error
        raise RuntimeError("CUDA/HIP IPC initialization failed on a peer rank")
    if any(rec[0] != _STATUS_OK for rec in records) or len({rec[1] for rec in records}) != world_size:
        if rt is not None and local_ptr:
            with _device_context(device):
                rt.free(local_ptr)
        logger.info("CUDA/HIP IPC unavailable for rank %d on %s; falling back to shared host memory", rank, device)
        return None
    assert rt is not None
    peer_ptrs: dict[int, int] = {}
    local_ready, phase2_error = _open_peer_handles(rt, rank, device, device_index, records, peer_ptrs)
    try:
        ready_raw = exchange(bytes([local_ready]), timeout)
    except BaseException:
        _close_peer_ptrs(rt, device, peer_ptrs)
        raise
    ready_statuses = [raw[0] for raw in ready_raw]
    if all(status == _STATUS_OK for status in ready_statuses):
        return IpcArena(device, rank, world_size, slot_bytes, buffer_count, local_ptr, peer_ptrs, close_barrier, rt)
    _close_peer_ptrs(rt, device, peer_ptrs)
    exchange(b"\x00", timeout)
    with _device_context(device):
        rt.free(local_ptr)
    if phase2_error is not None:
        raise phase2_error
    if any(status == _STATUS_ERROR for status in ready_statuses):
        raise RuntimeError("CUDA/HIP IPC initialization failed on a peer rank")
    logger.info("CUDA/HIP IPC unavailable for rank %d on %s; falling back to shared host memory", rank, device)
    return None
