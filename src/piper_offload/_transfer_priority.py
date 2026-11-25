"""Background host-to-device copies yield to foreground transfers on the same device.

Copies between pinned host memory and a GPU share that device's host link: two
run together take about as long as one after the other, so the only choice is
which waits. A transfer that queued GPU work waits for while background copies
could run, such as a collective whose output a queued kernel reads, runs inside
:func:`foreground_transfer` scoped to its device. Background work,
such as prefetching the next block while this one runs, runs inside
:func:`background_transfers`.

When a foreground producer is registered for a target device (via
:func:`register_foreground_producer`) or a :func:`foreground_transfer` is active
on that device, :func:`copy_host_to_device` waits for active foreground transfers
and limits queued background work: large contiguous copies pipeline in 2 MiB
chunks with at most two events in flight (4 MiB max pending), while smaller or
non-contiguous tensors synchronize at tensor boundaries. When no foreground
producer or transfer targets that device, background copies enqueue directly
without per-chunk or per-tensor synchronizations. Foreground activity on one
device never stalls background transfers targeting another device.
"""

import threading
import weakref
from collections import deque
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Self, cast

import torch

from ._devices import canonical_device

_BACKGROUND_CHUNK_BYTES = 2 * 1024 * 1024
_BACKGROUND_PIPELINE_DEPTH = 2


class _Foreground:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.counts: dict[torch.device, int] = {}
        self.wildcard_producers = 0
        self.device_producers: dict[torch.device, int] = {}

    def count(self, device: torch.device | str) -> int:
        key = canonical_device(device)
        with self.condition:
            return self.counts.get(key, 0)


class _Local(threading.local):
    background = False


_foreground = _Foreground()
_local = _Local()


def _decrement_devices_locked(devices: Iterable[torch.device]) -> None:
    for device in devices:
        remaining = _foreground.device_producers.get(device, 0) - 1
        if remaining > 0:
            _foreground.device_producers[device] = remaining
        else:
            _foreground.device_producers.pop(device, None)


def _release_producer(devices: tuple[torch.device, ...] | None) -> None:
    with _foreground.condition:
        if devices is None:
            if _foreground.wildcard_producers > 0:
                _foreground.wildcard_producers -= 1
            return
        _decrement_devices_locked(devices)


class ForegroundProducer:
    """Handle keeping chunked background yielding enabled while a producer lives."""

    def __init__(
        self,
        devices: Iterable[torch.device | str] | None = None,
    ) -> None:
        scoped_devices: tuple[torch.device, ...] | None = (
            None
            if devices is None
            else tuple(dict.fromkeys(canonical_device(device) for device in devices))
        )
        finalizer = weakref.finalize(self, _release_producer, scoped_devices)
        try:
            finalizer.atexit = False
            self._finalizer = finalizer
            added_wildcard = False
            added_devices: list[torch.device] = []
            with _foreground.condition:
                try:
                    if scoped_devices is None:
                        _foreground.wildcard_producers += 1
                        added_wildcard = True
                    else:
                        for device in scoped_devices:
                            _foreground.device_producers[device] = (
                                _foreground.device_producers.get(device, 0) + 1
                            )
                            added_devices.append(device)
                except BaseException:
                    if added_wildcard and _foreground.wildcard_producers > 0:
                        _foreground.wildcard_producers -= 1
                    elif added_devices:
                        _decrement_devices_locked(added_devices)
                    raise
        except BaseException:
            finalizer.detach()
            raise

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        """Unregister this foreground producer; idempotent."""
        self._finalizer()

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("Foreground producer registration is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def register_foreground_producer(
    devices: Iterable[torch.device | str] | None = None,
) -> ForegroundProducer:
    """Register an active foreground-transfer producer (e.g., ``piper_relay``).

    Pass ``devices`` to limit yielding to those canonical devices, or ``None``
    when the producer's devices are not known at construction time.
    """
    return ForegroundProducer(devices)


def should_yield(device: torch.device | str) -> bool:
    """Return whether background copies to ``device`` should yield between chunks."""
    key = canonical_device(device)
    with _foreground.condition:
        return bool(
            _foreground.wildcard_producers
            or _foreground.device_producers.get(key, 0)
            or _foreground.counts.get(key, 0)
        )


def _wait_for_foreground(device: torch.device) -> None:
    with _foreground.condition:
        _foreground.condition.wait_for(lambda: not _foreground.counts.get(device, 0))


def _can_chunk(destination: torch.Tensor, source: torch.Tensor) -> bool:
    return (
        type(destination) is torch.Tensor
        and type(source) is torch.Tensor
        and not destination.requires_grad
        and destination.shape == source.shape
        and destination.is_contiguous()
        and source.is_contiguous()
        and source.nbytes > _BACKGROUND_CHUNK_BYTES
    )


@contextmanager
def foreground_transfer(device: torch.device | str) -> Generator[None]:
    """Hold background copies targeting ``device`` back until the ``with`` exits.

    Background copies on the same canonical device wait for as long as this is
    held, including while the transfer itself waits, such as for a peer, so
    enter once its inputs are ready. Transfers on other devices are unaffected.

    Subsystems whose concurrent worker or forward calls enter this context
    alongside streaming prefetch should register a
    :func:`register_foreground_producer` handle for their lifetime so
    prefetches already begin yielding before the first collective enters.
    """
    key = canonical_device(device)
    with _foreground.condition:
        _foreground.counts[key] = _foreground.counts.get(key, 0) + 1
    try:
        yield
    finally:
        with _foreground.condition:
            remaining = _foreground.counts[key] - 1
            if remaining:
                _foreground.counts[key] = remaining
            else:
                del _foreground.counts[key]
                _foreground.condition.notify_all()


@contextmanager
def background_transfers() -> Generator[None]:
    """Make this thread's copies through :func:`copy_host_to_device` wait for foreground transfers.

    Not reentrant: exiting a nested ``with`` statement ends background mode for
    the thread.
    """
    _local.background = True
    try:
        yield
    finally:
        _local.background = False


def copy_host_to_device(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Asynchronously copy host ``source`` into GPU ``destination`` on the current stream.

    Inside :func:`background_transfers`, copies targeting a device with no
    registered foreground producer and no active foreground transfer enqueue
    directly. Otherwise the copy waits for active foreground transfers and
    synchronizes so queued work cannot overrun a subsequent collective: large
    contiguous copies (> 2 MiB) pipeline 2 MiB chunks with depth 2 (at most
    4 MiB in flight), and smaller or non-contiguous copies synchronize per
    tensor.
    """
    if not _local.background:
        destination.copy_(source, non_blocking=True)
        return
    device = canonical_device(destination.device)
    if not should_yield(device):
        destination.copy_(source, non_blocking=True)
        return
    stream = torch.cuda.current_stream(device)
    if not _can_chunk(destination, source):
        _wait_for_foreground(device)
        destination.copy_(source, non_blocking=True)
        stream.synchronize()
        return
    dst_flat = destination.view(-1)
    src_flat = source.view(-1)
    chunk_elements = max(1, _BACKGROUND_CHUNK_BYTES // source.element_size())
    pending: deque[torch.cuda.Event] = deque()
    try:
        for start in range(0, src_flat.numel(), chunk_elements):
            if len(pending) >= _BACKGROUND_PIPELINE_DEPTH:
                pending.popleft().synchronize()
            end = start + chunk_elements
            _wait_for_foreground(device)
            dst_flat[start:end].copy_(src_flat[start:end], non_blocking=True)
            try:
                pending.append(cast(torch.cuda.Event, stream.record_event()))
            except BaseException:
                pending.clear()
                stream.synchronize()
                raise
    finally:
        if pending:
            pending[-1].synchronize()
