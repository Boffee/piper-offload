"""Background host-to-device copies yield to foreground transfers.

Copies between pinned host memory and a GPU share its host link: two run
together take about as long as one after the other, so the only choice is which
waits. A transfer that queued GPU work waits for while background copies could
run, such as a collective whose output a queued kernel reads, runs inside
:func:`foreground_transfer`. Background work, such as prefetching the next block
while this one runs, runs inside :func:`background_transfers`. There each copy
through :func:`copy_host_to_device` starts only while no foreground transfer is
in progress and completes before the call returns, so a foreground transfer
shares the host link with at most one background tensor.

The signal is process-wide. With one rank per process, the measured setup, both
kinds share one GPU's host link; a process driving several GPUs would also hold
one GPU's background copies for another's foreground transfers. Neither side
learns what the other is.
"""

import threading
from collections.abc import Generator
from contextlib import contextmanager

import torch


class _Foreground:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.count = 0


class _Local(threading.local):
    background = False


_foreground = _Foreground()
_local = _Local()


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

    Inside :func:`background_transfers` the copy first waits for foreground
    transfers to end and has completed when this returns: a copy left queued
    could start on the GPU after a foreground transfer begins.
    """
    if not _local.background:
        destination.copy_(source, non_blocking=True)
        return
    with _foreground.condition:
        _foreground.condition.wait_for(lambda: not _foreground.count)
    destination.copy_(source, non_blocking=True)
    torch.cuda.current_stream(destination.device).synchronize()
