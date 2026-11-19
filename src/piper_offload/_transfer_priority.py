"""Background host-to-device copies yield to foreground transfers.

Copies between pinned host memory and a GPU share its host link: two run
together take about as long as one after the other, so the only choice is which
waits. A transfer that queued GPU work waits for while background copies could
run, such as a collective whose output a queued kernel reads, runs inside
:func:`foreground_transfer`. Background work, such as prefetching the next block
while this one runs, runs inside :func:`background_transfers`. There each copy
through :func:`copy_host_to_device` checks for active foreground transfers and
synchronizes before moving on; large contiguous copies split into 4 MiB chunks
so a collective waits for at most one chunk rather than an entire tensor or
block.

The signal is process-wide. With one rank per process, the measured setup, both
kinds share one GPU's host link; a process driving several GPUs would also hold
one GPU's background copies for another's foreground transfers. Neither side
learns what the other is.
"""

import threading
from collections.abc import Generator
from contextlib import contextmanager

import torch

_BACKGROUND_CHUNK_BYTES = 4 * 1024 * 1024


class _Foreground:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.count = 0


class _Local(threading.local):
    background = False


_foreground = _Foreground()
_local = _Local()


def _wait_for_foreground() -> None:
    with _foreground.condition:
        _foreground.condition.wait_for(lambda: not _foreground.count)


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
def foreground_transfer() -> Generator[None]:
    """Hold background copies back until the ``with`` statement exits.

    Background copies wait for as long as this is held, including while the
    transfer itself waits, such as for a peer, so enter once its inputs are ready.
    """
    with _foreground.condition:
        _foreground.count += 1
    try:
        yield
    finally:
        with _foreground.condition:
            _foreground.count -= 1
            if not _foreground.count:
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
    to end before copying and has completed when this returns: a copy left
    queued could start on the GPU after a foreground transfer begins. Contiguous
    copies larger than 4 MiB chunk in 4 MiB slices so foreground transfers can
    interleave between chunks.
    """
    if not _local.background:
        destination.copy_(source, non_blocking=True)
        return
    stream = torch.cuda.current_stream(destination.device)
    if not _can_chunk(destination, source):
        _wait_for_foreground()
        destination.copy_(source, non_blocking=True)
        stream.synchronize()
        return
    dst_flat = destination.view(-1)
    src_flat = source.view(-1)
    chunk_elements = max(1, _BACKGROUND_CHUNK_BYTES // source.element_size())
    for start in range(0, src_flat.numel(), chunk_elements):
        end = start + chunk_elements
        _wait_for_foreground()
        dst_flat[start:end].copy_(src_flat[start:end], non_blocking=True)
        stream.synchronize()
