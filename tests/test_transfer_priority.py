"""Background host-to-device copies wait for foreground transfers; other copies never do."""

import threading

import pytest
import torch

import piper_offload._transfer_priority as transfer_priority
from piper_offload._transfer_priority import background_transfers, copy_host_to_device, foreground_transfer

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")


def test_background_large_contiguous_copy_chunks_and_yields_mid_tensor(monkeypatch):
    pause_on_first = True

    class RecordingStream:
        def __init__(self) -> None:
            self.syncs = 0

        def synchronize(self) -> None:
            self.syncs += 1
            if pause_on_first and self.syncs == 1:
                in_first_chunk.set()
                resume_allowed.wait(5)

    stream = RecordingStream()
    in_first_chunk = threading.Event()
    resume_allowed = threading.Event()
    monkeypatch.setattr(transfer_priority, "_BACKGROUND_CHUNK_BYTES", 16)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)

    # Multi-dim contiguous tensor and non-zero storage_offset contiguous view (48 B -> 3 x 16 B chunks).
    src_2d = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    dst_2d = torch.zeros_like(src_2d)

    def run_bg(dst: torch.Tensor, src: torch.Tensor) -> None:
        with background_transfers():
            copy_host_to_device(dst, src)

    worker = threading.Thread(target=run_bg, args=(dst_2d, src_2d))
    with foreground_transfer():
        worker.start()
        assert not in_first_chunk.wait(0.15)
    assert in_first_chunk.wait(5)
    with foreground_transfer():
        # Release after chunk 1 while foreground is active; chunks 2-3 must wait.
        resume_allowed.set()
        assert stream.syncs == 1
        assert torch.equal(dst_2d.view(-1)[:4], src_2d.view(-1)[:4])
        assert torch.equal(dst_2d.view(-1)[4:], torch.zeros(8))
        worker.join(0.15)
        assert worker.is_alive()
    worker.join(5)
    assert not worker.is_alive()
    assert stream.syncs == 3
    assert torch.equal(dst_2d, src_2d)

    # Non-zero storage_offset contiguous slice also chunks cleanly.
    pause_on_first = False
    stream.syncs = 0
    base_src = torch.arange(20, dtype=torch.float32)
    src_slice = base_src[4:16].reshape(2, 6)
    dst_slice = torch.zeros(2, 6, dtype=torch.float32)
    with background_transfers():
        copy_host_to_device(dst_slice, src_slice)
    assert stream.syncs == 3
    assert torch.equal(dst_slice, src_slice)


def test_background_non_chunkable_copies_use_single_transfer(monkeypatch):
    class CountingStream:
        def __init__(self) -> None:
            self.calls = 0
            self.syncs = 0

        def synchronize(self) -> None:
            self.syncs += 1

    stream = CountingStream()

    def current_stream(_device: object) -> CountingStream:
        stream.calls += 1
        return stream

    monkeypatch.setattr(transfer_priority, "_BACKGROUND_CHUNK_BYTES", 16)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)

    class SubTensor(torch.Tensor):
        pass

    grad_dst = torch.zeros(8, dtype=torch.float32, requires_grad=True)
    cases = [
        (torch.zeros(4, dtype=torch.float32), torch.arange(4, dtype=torch.float32)),  # <= threshold (16 B)
        (torch.zeros(4, 4, dtype=torch.float32), torch.arange(16, dtype=torch.float32).reshape(4, 4).T),
        (torch.zeros(4, 4, dtype=torch.float32).T, torch.arange(16, dtype=torch.float32).reshape(4, 4)),
        (torch.zeros(2, 6, dtype=torch.float32), torch.arange(6, dtype=torch.float32).reshape(1, 6)),
        (grad_dst, torch.arange(8, dtype=torch.float32)),
        (torch.zeros(8, dtype=torch.float32), SubTensor(torch.arange(8, dtype=torch.float32))),
    ]
    with torch.no_grad(), background_transfers():
        for dst, src in cases:
            stream.syncs = 0
            copy_host_to_device(dst, src)
            assert stream.syncs == 1
            torch.testing.assert_close(
                dst.detach().as_subclass(torch.Tensor),
                src.as_subclass(torch.Tensor).expand_as(dst),
            )

    # Foreground/unscoped copies never check _foreground or call current_stream.
    stream.calls = stream.syncs = 0
    fg_dst = torch.zeros(12, dtype=torch.float32)
    fg_src = torch.arange(12, dtype=torch.float32)
    with foreground_transfer():
        copy_host_to_device(fg_dst, fg_src)
    assert stream.calls == 0 and stream.syncs == 0
    assert torch.equal(fg_dst, fg_src)


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
def test_cuda_background_copy_yields_mid_tensor(monkeypatch):
    source = torch.arange(2048, dtype=torch.float32).pin_memory()
    destination = torch.zeros_like(source, device="cuda")
    chunk_boundary = threading.Event()
    proceed = threading.Event()
    waits = 0
    original_wait = transfer_priority._wait_for_foreground

    def instrumented_wait() -> None:
        nonlocal waits
        if waits == 1:
            chunk_boundary.set()
            proceed.wait(30)
        original_wait()
        waits += 1

    monkeypatch.setattr(transfer_priority, "_BACKGROUND_CHUNK_BYTES", 1024)
    monkeypatch.setattr(transfer_priority, "_wait_for_foreground", instrumented_wait)

    def _run_bg(dst: torch.Tensor, src: torch.Tensor) -> None:
        with background_transfers():
            copy_host_to_device(dst, src)

    worker = threading.Thread(target=_run_bg, args=(destination, source))
    worker.start()
    assert chunk_boundary.wait(30)
    with foreground_transfer():
        proceed.set()
        assert torch.equal(destination[:256].cpu(), source[:256])
        assert torch.equal(destination[256:].cpu(), torch.zeros(2048 - 256))
        worker.join(0.15)
        assert worker.is_alive()
    worker.join(30)
    assert waits == 8
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
