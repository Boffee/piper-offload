"""CPU source storage and optional anonymous storage of the same weight bytes.

CPU tensors retain their source storage: file-backed for mmap weights, or
anonymous for parameters constructed in RAM. Copies to GPU prefer the optional
anonymous storage when supplied. Allocation, pinning, and eviction are managed
by the caller; this module protects storage while it is being read.

Parameters, buffers, and views sharing a storage share one handle. Independent
storages wrapping overlapping addresses are not deduplicated (the same
restriction applies to HostMemoryManager).
"""

import threading
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Self

import torch

if TYPE_CHECKING:
    from .host_memory import HostMemoryManager


class CopyCompletion(Protocol):
    """Completion marker supplied by the copy layer, usually a CUDA event."""

    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


@dataclass
class _BackingState:
    """Keep allocations alive through final cleanup without retaining the handle."""

    source_storage: torch.UntypedStorage
    anonymous_storage: torch.UntypedStorage | None = None
    active_leases: int = 0
    pending_copies: list[CopyCompletion] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def discard_completed_copies(self) -> None:
        """Drop completed markers while the caller holds the state lock."""
        self.pending_copies[:] = [copy for copy in self.pending_copies if not copy.query()]

    def wait_for_copies(self) -> None:
        # Final cleanup runs after all leases release the handle.
        for completion in self.pending_copies:
            completion.synchronize()
        self.pending_copies.clear()


@dataclass
class _LeaseState:
    """Let the finalizer receive a completion marker without retaining the lease."""

    completion: CopyCompletion | None = None


class HostBackingLease:
    """Protect one resolved view until close, then until its completion marker.

    The tensor is borrowed: callers must not retain/use it after closing the
    lease. Async callers must supply a marker covering every read, including
    reads submitted before an exception. A live backing owner can defer reuse
    without waiting; dropping the last owner waits for pending work.
    """

    def __init__(self, backing: HostBacking, tensor: torch.Tensor) -> None:
        self._tensor: torch.Tensor | None = tensor
        self._state = _LeaseState()
        self._finalizer = weakref.finalize(self, backing._release, self._state)
        self._finalizer.atexit = False

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            raise RuntimeError("Host backing lease is closed")
        return self._tensor

    def close(self, completion: CopyCompletion | None = None) -> None:
        """Release protection immediately or after the supplied marker."""
        if not self._finalizer.alive:
            return
        self._state.completion = completion
        self._finalizer()
        self._tensor = None

    def __enter__(self) -> Self:
        if not self._finalizer.alive:
            raise RuntimeError("Host backing lease is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class HostBacking:
    """Track source storage and optional anonymous storage.

    Selection changes only when all acquired views and pending copies have
    finished. Source tensor shape, stride, dtype and storage offset determine
    each resolved view. Anonymous storage contains the entire source allocation's
    bytes, including portions outside an individual tensor view.
    """

    def __init__(self, storage: torch.UntypedStorage, memory_manager: HostMemoryManager) -> None:
        self._memory_manager = memory_manager
        self._state = _BackingState(storage)
        self._finalizer = weakref.finalize(self, self._state.wait_for_copies)
        self._finalizer.atexit = False

    @property
    def memory_manager(self) -> HostMemoryManager:
        """The manager that owns this allocation handle."""
        return self._memory_manager

    @property
    def nbytes(self) -> int:
        """Physical bytes of this allocation, independent of view sizes."""
        return self._state.source_storage.nbytes()

    @property
    def has_anonymous_storage(self) -> bool:
        """Whether an anonymous allocation is currently available for copies."""
        with self._state.lock:
            return self._state.anonymous_storage is not None

    def acquire(self, source: torch.Tensor) -> HostBackingLease:
        """Acquire the selected backing with the source's view."""
        state = self._state
        if source.untyped_storage()._cdata != state.source_storage._cdata:
            raise ValueError("Source does not belong to this host backing")
        with state.lock:
            state.discard_completed_copies()
            if state.anonymous_storage is None:
                tensor = source
            else:
                tensor = torch.empty(0, dtype=source.dtype, device="cpu").set_(
                    state.anonymous_storage,
                    source.storage_offset(),
                    source.shape,
                    source.stride(),
                )
            lease = HostBackingLease(self, tensor)
            state.active_leases += 1
            return lease

    def _release(self, lease: _LeaseState) -> None:
        state = self._state
        with state.lock:
            if lease.completion is not None:
                state.pending_copies.append(lease.completion)
            state.active_leases -= 1

    def try_set_anonymous_storage(self, tensor: torch.Tensor | None) -> bool:
        """Set anonymous storage, or clear it with None. Keep source storage intact.

        Return False while a lease or unfinished copy prevents the change.
        The caller must supply anonymous memory with equivalent immutable bytes
        and retain any pin until the anonymous storage is safely removed. This
        method does not copy bytes, pin, unpin, or verify equality or memory
        provenance. A future cache must check whether weights are immutable and
        recoverable before creating or discarding anonymous storage.
        """
        state = self._state
        anonymous_storage = None
        if tensor is not None:
            if (
                type(tensor) is not torch.Tensor
                or tensor.device.type != "cpu"
                or tensor.layout is not torch.strided
                or not tensor.is_contiguous()
                or tensor.storage_offset() != 0
                or tensor.nbytes != tensor.untyped_storage().nbytes()
                or tensor.nbytes != self.nbytes
            ):
                raise ValueError("Anonymous storage must be a complete CPU allocation of equal byte size")
            anonymous_storage = tensor.untyped_storage()
            if anonymous_storage._cdata == state.source_storage._cdata:
                raise ValueError("Use None to select source storage")
        with state.lock:
            if state.active_leases:
                return False
            state.discard_completed_copies()
            if state.pending_copies:
                return False
            state.anonymous_storage = anonymous_storage
            return True
