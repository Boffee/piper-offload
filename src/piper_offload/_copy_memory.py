"""Owned checkpoint copies and their preparation, independent of the host OS.

The pin manager owns all reservations and registration order. Memory owns a
copy until ``reserve`` hands it to an acquisition, and takes it back only
after successful unregistration. Metadata uses the manager's lock; preparation
and batch completion run outside it, since worker finalizers may need it.
"""

import contextlib
import io
import logging
import mmap
import operator
import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Buffer, Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import torch

from .checkpoint import FileSlice

logger = logging.getLogger(__name__)


class Region(Buffer, Protocol):
    def close(self) -> None: ...


class Readers(Protocol):
    def __enter__(self) -> Readers: ...
    def __exit__(self, *_exc: object) -> None: ...
    def read_at(self, file: io.BufferedReader) -> Callable[[memoryview, int], int]: ...


class Copy:
    """An owned page-aligned region holding one checkpoint storage's bytes.

    Page alignment keeps two copies from sharing an OS page, which the
    runtime would refuse to register twice. ``size`` is the registered and
    reserved extent, whole pages.
    """

    __slots__ = ("region", "size", "storage")

    def __init__(self, size: int, region: Region) -> None:
        assert size % mmap.PAGESIZE == 0
        self.size = size
        self.region = region
        # None once freed or offered.
        self.storage: torch.UntypedStorage | None = None
        self._expose()

    def _expose(self) -> None:
        # Built over a memoryview, so the storage and every view of it hold an
        # export of the region, which cannot close or be offered under them.
        self.storage = torch.frombuffer(memoryview(self.region), dtype=torch.uint8).untyped_storage()

    @property
    def pointer(self) -> int:
        assert self.storage is not None
        return self.storage.data_ptr()

    def touch(self) -> None:
        """Write one byte of every page back over itself, which brings the pages into the working set.

        Pages a copy holds can be outside the working set whether they were
        just reclaimed, which leaves them present but not in it, or filled
        long enough ago that memory pressure has trimmed them again.
        ``cudaHostRegister`` faults such pages in one at a time, at 8 GiB/s,
        where this write brings them back at 42 GiB/s and leaves registration
        running at 55. It leaves the bytes as they were.
        """
        with memoryview(self.region) as view:
            _fault_in(view, 0, self.size)

    def view(self, tensor: torch.Tensor) -> torch.Tensor:
        """``tensor``'s geometry over the copy, including its lazy conjugation and negation."""
        assert self.storage is not None
        view = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
            self.storage, tensor.storage_offset(), tensor.shape, tensor.stride(),
        )
        if tensor.is_conj():
            view = view.conj()
        if tensor.is_neg():
            view = torch._neg_view(view)
        return view

    def free(self) -> None:
        self.storage = None
        try:
            self.region.close()
        except BufferError:
            # A transfer view outlived its lease; the region returns with it.
            logger.warning("An evicted pinned copy is still referenced; its memory returns when the reference dies")


_FILL_SLICE = 64 * 2**20


def _read_range(
    read_at: Callable[[memoryview, int], int], view: memoryview, offset: int, start: int, stop: int,
) -> None:
    """Fill ``view[start:stop]`` from ``offset + start`` in the file, looping over short reads."""
    done = start
    while done < stop:
        count = read_at(view[done:stop], offset + done)
        if count <= 0:
            raise OSError(f"checkpoint ended at byte {offset + done}")
        done += count


def _fault_in(view: memoryview, start: int, stop: int) -> None:
    """Write one byte of every page of ``view[start:stop]`` back over itself, faulting the pages in."""
    with view[start:stop] as window:
        torch.frombuffer(window, dtype=torch.uint8)[:: mmap.PAGESIZE].bitwise_or_(0)


@dataclass(eq=False, slots=True, kw_only=True)
class CopyLoad:
    """A copy reserved for one request, filled and registered with the rest of its acquisition.

    ``offered`` marks a copy whose reclaim worker has not yet been joined;
    every reclaim finishes before anything is filled. ``missing`` is what the
    fill has to read of the copy, as offsets and lengths: all of a new or rebuilt one, the spans
    Windows discarded of a reclaimed one, and nothing of one it kept intact
    or whose fill completed successfully.
    """

    source: FileSlice
    copy: Copy
    missing: list[tuple[int, int]]
    offered: bool = False
    error: OSError | None = None

    def spans(self) -> list[tuple[int, int]]:
        """``missing`` within the storage's bytes, as start and stop offsets."""
        length = self.source.length
        return [(start, min(start + covered, length)) for start, covered in self.missing if start < length]

    @property
    def ready(self) -> bool:
        """Published by preparation after joining the worker that last touched this copy."""
        return not self.offered and (not self.missing or self.error is not None)


type ReadRange = Callable[[Callable[[memoryview, int], int], memoryview, int, int, int], None]


def _fill_copies(pending: list[CopyLoad], *, readers: Readers, read: ReadRange) -> None:
    """Fill every pending copy from its file; a read failure marks that copy's ``error``.

    Positional reads over fixed-size slices of every copy run together on
    one thread per logical core, hyperthreads included: the fill is bound
    by page faults on the fresh regions, which overlap across threads, and
    a checkpoint's tensors are mostly smaller than one slice. Each worker
    reads through the platform's reader. Runs without the manager's lock:
    workers run Python code, so a garbage collection on one of them may run a finalizer that
    needs that lock.
    """
    workers = os.process_cpu_count() or 1
    # The pool closes before the views, so a failure waits for the queued
    # slices to finish, and the readers close after all of them.
    with (
        readers as opened,
        contextlib.ExitStack() as views,
        ThreadPoolExecutor(workers, thread_name_prefix="piper-offload-fill") as pool,
    ):
        futures: list[tuple[CopyLoad, Future[None]]] = []
        for item in pending:
            view = views.enter_context(memoryview(item.copy.region)[: item.source.length])
            read_at = opened.read_at(item.source.file)
            for first, last in item.spans():
                for start in range(first, last, _FILL_SLICE):
                    stop = min(start + _FILL_SLICE, last)
                    futures.append((item, pool.submit(read, read_at, view, item.source.offset, start, stop)))
        for item, future in futures:
            try:
                future.result()
            except OSError as error:
                item.error = _detached(error)
    for item in pending:
        if item.error is None:
            item.missing.clear()


def _on_workers(action: Callable[[Copy], None], copies: list[Copy], workers: int, name: str) -> None:
    """Run ``action`` on every copy on up to ``workers`` threads named ``name``, or inline for one copy.

    Runs without the manager's lock: the workers run Python code, so a
    garbage collection on one of them may run a finalizer that needs it.
    """
    if len(copies) < 2:
        for copy in copies:
            action(copy)
        return
    with ThreadPoolExecutor(min(workers, len(copies)), thread_name_prefix=name) as pool:
        for future in [pool.submit(action, copy) for copy in copies]:
            future.result()


def _touch_copies(pending: list[CopyLoad]) -> None:
    """Bring the copies a fill has just written into the working set, on one thread per core.

    A long fill leaves the copies it wrote first outside the working set by
    the time the last one is read, and pages outside it are where
    ``cudaHostRegister`` faults them in one at a time at 8 GiB/s: writing a
    byte to each brings them back at 42 GiB/s and leaves registration running
    at 55, so the pair costs 60 ms/GiB against the 130 that registering them
    cold does.
    """
    ready = [item.copy for item in pending if item.error is None]
    _on_workers(operator.methodcaller("touch"), ready, os.process_cpu_count() or 1, "piper-offload-touch")


def _detached(error: OSError) -> OSError:
    """The error without its traceback, whose frames would hold a slice of the copy and keep it mapped."""
    return error.with_traceback(None)


# Unmapping is page-table teardown: freeing 32 GiB measured 1.79 s on one
# thread, 0.64 s on four, and no improvement beyond eight.
_FREE_WORKERS = 4


def _free_copies(copies: list[Copy]) -> None:
    """Free owned regions together, returning their memory to the OS."""
    _on_workers(operator.methodcaller("free"), copies, _FREE_WORKERS, "piper-offload-free")


class CopyOwner(Protocol):
    """The identity and lifetime of an unregistered copy; retained while its memory is cached."""

    pointer: int
    copy: Copy | None
    retired: bool


class ReleaseBatch(contextlib.AbstractContextManager[None]):
    """Enter under the manager lock; finish outside it, including on acquisition failure.

    The Linux implementation has nothing to defer. Windows starts offering
    each released copy immediately and joins those workers in ``finish``.
    """

    def __enter__(self) -> None:
        pass

    def __exit__(self, *_exc: object) -> None:
        pass

    def finish(self) -> None:
        pass


class Memory(ABC):
    """Copy lifetime shared by both platforms; Linux frees every released copy.

    ``occupied`` reports whether a storage is registered or pending. It is
    read only under ``lock`` and must not retain the manager, so a Windows
    commitment watcher cannot extend the manager's lifetime.
    """

    readers: type[Readers]
    offered_bytes = 0

    def __init__(self, max_offered_bytes: int, lock: threading.RLock, occupied: Callable[[int], bool]) -> None:
        self.max_offered_bytes = max_offered_bytes
        self._lock = lock
        self._occupied = occupied

    @abstractmethod
    def new_copy(self, size: int) -> Copy: ...

    def allocate(self, size: int) -> Copy | None:
        try:
            return self.new_copy(size)
        except (OSError, MemoryError) as error:
            logger.warning("Could not allocate a pinned copy; the checkpoint storage stays pageable: %s", error)
            return None

    def reserve(self, _pointer: int, size: int, source: FileSlice, fit: Callable[[int], bool]) -> CopyLoad | None:
        if not fit(size):
            return None
        copy = self.allocate(size)
        return None if copy is None else CopyLoad(source=source, copy=copy, missing=[(0, size)])

    def prepare(self, copies: list[CopyLoad]) -> Generator[None]:
        """Snapshot policy under the lock; advance and close the generator outside it.

        The first result lets the manager detect exhausted runtime capacity
        before reading the remaining copies. Close this generator before
        freeing any acquisition copy.
        """
        return self._filled(copies, read=_read_range)

    def _filled(self, copies: list[CopyLoad], *, read: ReadRange) -> Generator[None]:
        """Fill the first copy, yield for registration, then fill the rest together."""
        if copies:
            self._fill(copies[:1], read=read)
            yield
            if len(copies) > 1:
                self._fill(copies[1:], read=read)

    def _fill(self, copies: list[CopyLoad], *, read: ReadRange) -> None:
        _fill_copies(copies, readers=self.readers(), read=read)
        _touch_copies(copies)

    def release_batch(self) -> ReleaseBatch:
        return ReleaseBatch()

    def release(self, owner: CopyOwner, *, demote: bool) -> list[Copy]:  # noqa: ARG002 - platform contract
        """Take an unregistered copy; return detached copies for the caller to free."""
        assert owner.copy is not None
        return [owner.copy]

    def retire(self, _owner: CopyOwner) -> None:
        """Free any cached copy still belonging to this retired owner."""
        return

    def resize(self, value: int) -> list[Copy]:
        self.max_offered_bytes = value
        return []

    def clear(self) -> list[Copy]:
        """Detach all cached copies for the caller to free outside the lock."""
        return []
