"""Background host-to-device copies yield to foreground transfers on the same device.

Copies between pinned host memory and a GPU share that device's host link: two
run together take about as long as one after the other, so the only choice is
which waits. A transfer that queued GPU work waits for while background copies
could run, such as a collective whose output a queued kernel reads, runs inside
:func:`foreground_transfer` scoped to its device. Background work,
such as prefetching the next block while this one runs, runs inside
:func:`background_transfers`. There each copy through
:func:`copy_host_to_device` waits only for foreground transfers on the same
canonical device and synchronizes before moving on; large contiguous copies
split into 4 MiB chunks so a collective waits for at most one chunk rather than
an entire tensor or block. Foreground activity on one device never stalls
background transfers targeting another device.
"""

import threading
from collections.abc import Generator
from contextlib import contextmanager

import torch

from ._devices import canonical_device

_BACKGROUND_CHUNK_BYTES = 4 * 1024 * 1024


class _Foreground:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.counts: dict[torch.device, int] = {}

    def count(self, device: torch.device | str) -> int:
        key = canonical_device(device)
        with self.condition:
            return self.counts.get(key, 0)


class _Local(threading.local):
    background = False


_foreground = _Foreground()
_local = _Local()


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

    Inside :func:`background_transfers` the copy waits for foreground transfers
    targeting the same device to end before copying and has completed when this
    returns: a copy left queued could start on the GPU after a foreground
    transfer begins. Contiguous copies larger than 4 MiB chunk in 4 MiB slices
    so foreground transfers can interleave between chunks.
    """
    if not _local.background:
        destination.copy_(source, non_blocking=True)
        return
    device = canonical_device(destination.device)
    stream = torch.cuda.current_stream(device)
    if not _can_chunk(destination, source):
        _wait_for_foreground(device)
        destination.copy_(source, non_blocking=True)
        stream.synchronize()
        return
    dst_flat = destination.view(-1)
    src_flat = source.view(-1)
    chunk_elements = max(1, _BACKGROUND_CHUNK_BYTES // source.element_size())
    for start in range(0, src_flat.numel(), chunk_elements):
        end = start + chunk_elements
        _wait_for_foreground(device)
        dst_flat[start:end].copy_(src_flat[start:end], non_blocking=True)
        stream.synchronize()
