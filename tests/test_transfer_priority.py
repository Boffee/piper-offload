"""Background host-to-device copies wait for foreground transfers; other copies never do."""

import threading

import pytest
import torch

from piper_offload._transfer_priority import background_transfers, copy_host_to_device, foreground_transfer

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")


@CUDA
def test_background_copy_waits_for_a_foreground_transfer():
    # The next block's weights must not take the link while a collective moves
    # data that queued GPU work is waiting for.
    source = torch.arange(1024, dtype=torch.float32).pin_memory()
    destination = torch.zeros_like(source, device="cuda")
    copied = threading.Event()

    def stage() -> None:
        with background_transfers():
            copy_host_to_device(destination, source)
        copied.set()

    worker = threading.Thread(target=stage)
    with foreground_transfer():
        worker.start()
        assert not copied.wait(0.2)
    assert copied.wait(30)
    worker.join()
    assert torch.equal(destination.cpu(), source)


@CUDA
def test_other_copies_never_wait():
    # A copy the caller is about to use, here inside the foreground transfer
    # itself, would deadlock if it waited.
    source = torch.arange(1024, dtype=torch.float32).pin_memory()
    destination = torch.zeros_like(source, device="cuda")
    with foreground_transfer():
        copy_host_to_device(destination, source)
    torch.cuda.synchronize()
    assert torch.equal(destination.cpu(), source)
