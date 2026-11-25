"""Background host-to-device copies wait for foreground transfers; other copies never do."""

import gc
import threading

import pytest
import torch

import piper_offload._transfer_priority as transfer_priority
from piper_offload._transfer_priority import (
    background_transfers,
    copy_host_to_device,
    foreground_transfer,
    register_foreground_producer,
    should_yield,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")


def test_background_large_contiguous_copy_chunks_and_yields_mid_tensor(monkeypatch):
    pause_on_first = True

    class RecordingEvent:
        def __init__(self, stream: RecordingStream) -> None:
            self._stream = stream

        def synchronize(self) -> None:
            self._stream.event_syncs += 1
            if pause_on_first and self._stream.event_syncs == 1:
                in_first_sync.set()
                resume_allowed.wait(5)

    class RecordingStream:
        def __init__(self) -> None:
            self.calls = 0
            self.syncs = 0
            self.event_syncs = 0

        def record_event(self) -> RecordingEvent:
            return RecordingEvent(self)

        def synchronize(self) -> None:
            self.syncs += 1

    stream = RecordingStream()

    def current_stream(_device: object) -> RecordingStream:
        stream.calls += 1
        return stream

    in_first_sync = threading.Event()
    resume_allowed = threading.Event()
    monkeypatch.setattr(transfer_priority, "_BACKGROUND_CHUNK_BYTES", 16)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)

    # Multi-dim contiguous tensor (64 B -> 4 x 16 B chunks with depth-2 pipelining).
    src_2d = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    dst_2d = torch.zeros_like(src_2d)

    def run_bg(dst: torch.Tensor, src: torch.Tensor) -> None:
        with background_transfers():
            copy_host_to_device(dst, src)

    with register_foreground_producer(["cpu"]):
        worker = threading.Thread(target=run_bg, args=(dst_2d, src_2d))
        with foreground_transfer("cpu"):
            worker.start()
            assert not in_first_sync.wait(0.15)
        assert in_first_sync.wait(5)
        with foreground_transfer("cpu"):
            # Paused before chunk 3 while ev0 synchronized: chunks 1-2 (8 floats)
            # are copied, and chunks 3-4 must wait for foreground exit.
            resume_allowed.set()
            assert stream.event_syncs == 1
            assert stream.syncs == 0
            assert torch.equal(dst_2d.view(-1)[:8], src_2d.view(-1)[:8])
            assert torch.equal(dst_2d.view(-1)[8:], torch.zeros(8))
            worker.join(0.15)
            assert worker.is_alive()
        worker.join(5)
        assert not worker.is_alive()
        assert stream.event_syncs == 3
        assert stream.syncs == 0
        assert torch.equal(dst_2d, src_2d)

        # Non-zero storage_offset contiguous slice also chunks cleanly (48 B -> 3 x 16 B chunks).
        pause_on_first = False
        stream.event_syncs = 0
        base_src = torch.arange(20, dtype=torch.float32)
        src_slice = base_src[4:16].reshape(2, 6)
        dst_slice = torch.zeros(2, 6, dtype=torch.float32)
        with background_transfers():
            copy_host_to_device(dst_slice, src_slice)
        assert stream.event_syncs == 2
        assert stream.syncs == 0
        assert torch.equal(dst_slice, src_slice)

    # Without an active producer or transfer, background copies enqueue without syncs.
    stream.calls = stream.syncs = stream.event_syncs = 0
    fast_dst = torch.zeros_like(src_2d)
    run_bg(fast_dst, src_2d)
    assert stream.calls == 0 and stream.syncs == 0 and stream.event_syncs == 0
    assert torch.equal(fast_dst, src_2d)


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
    with register_foreground_producer(["cpu"]), torch.no_grad(), background_transfers():
        for dst, src in cases:
            stream.syncs = 0
            copy_host_to_device(dst, src)
            assert stream.syncs == 1
            torch.testing.assert_close(
                dst.detach().as_subclass(torch.Tensor),
                src.as_subclass(torch.Tensor).expand_as(dst),
            )

    # Background copies without an active producer or transfer avoid current_stream/syncs.
    stream.calls = stream.syncs = 0
    no_producer_dst = torch.zeros(4, dtype=torch.float32)
    with background_transfers():
        copy_host_to_device(no_producer_dst, cases[0][1])
    assert stream.calls == 0 and stream.syncs == 0
    assert torch.equal(no_producer_dst, cases[0][1])

    # Foreground/unscoped copies never check _foreground or call current_stream.
    stream.calls = stream.syncs = 0
    fg_dst = torch.zeros(12, dtype=torch.float32)
    fg_src = torch.arange(12, dtype=torch.float32)
    with foreground_transfer("cpu"):
        copy_host_to_device(fg_dst, fg_src)
    assert stream.calls == 0 and stream.syncs == 0
    assert torch.equal(fg_dst, fg_src)


def test_foreground_transfer_isolates_distinct_devices(monkeypatch):
    class NoopStream:
        def synchronize(self) -> None:
            pass

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: NoopStream())

    src = torch.arange(16, dtype=torch.float32)
    dst = torch.zeros_like(src)
    cpu_done = threading.Event()
    gpu1_done = threading.Event()
    gpu0_done = threading.Event()
    errors: list[BaseException] = []

    def run_checked(fn) -> None:
        try:
            fn()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def copy_cpu() -> None:
        with background_transfers():
            copy_host_to_device(dst, src)
        cpu_done.set()

    def wait_gpu(index: int, flag: threading.Event) -> None:
        transfer_priority._wait_for_foreground(torch.device("cuda", index))
        flag.set()

    w_cpu = threading.Thread(target=run_checked, args=(copy_cpu,))
    w_gpu1 = threading.Thread(target=run_checked, args=(lambda: wait_gpu(1, gpu1_done),))
    w_gpu0 = threading.Thread(target=run_checked, args=(lambda: wait_gpu(0, gpu0_done),))
    try:
        with foreground_transfer("cuda:0"):
            assert transfer_priority._foreground.count("cuda:0") == 1
            assert transfer_priority._foreground.count("cuda:1") == 0
            w_gpu0.start()
            with foreground_transfer("cuda:0"):
                assert transfer_priority._foreground.count("cuda:0") == 2
                w_cpu.start()
                w_gpu1.start()
                assert cpu_done.wait(5.0)
                assert gpu1_done.wait(5.0)
                assert not gpu0_done.wait(0.05)
            # Exiting nested block must keep cuda:0 waiter blocked until outer exits.
            assert transfer_priority._foreground.count("cuda:0") == 1
            assert not gpu0_done.wait(0.05)
        assert transfer_priority._foreground.counts == {}
        assert gpu0_done.wait(5.0)
    finally:
        with transfer_priority._foreground.condition:
            transfer_priority._foreground.counts.clear()
            transfer_priority._foreground.condition.notify_all()
        for worker in (w_cpu, w_gpu1, w_gpu0):
            if worker.ident is not None:
                worker.join(timeout=5.0)

    assert not errors
    assert torch.equal(dst, src)


def test_foreground_transfer_cleans_up_on_exception_and_canonicalizes_devices(monkeypatch):
    current = 1
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)

    # Bare "cuda" binds to entry-time current_device() and still cleans up that
    # exact key even if current_device() changes before exit.
    with foreground_transfer("cuda"):
        assert transfer_priority._foreground.count("cuda:1") == 1
        assert transfer_priority._foreground.count("cuda:0") == 0
        current = 0
        assert transfer_priority._foreground.count("cuda") == 0
    assert transfer_priority._foreground.counts == {}

    # "cpu:0" normalizes to canonical "cpu".
    with foreground_transfer("cpu:0"):
        assert transfer_priority._foreground.count("cpu") == 1
    assert transfer_priority._foreground.counts == {}

    # Exceptions in both nested and outermost scopes decrement counts and wake waiters.
    woken = threading.Event()
    errors: list[BaseException] = []

    def wait_gpu() -> None:
        try:
            transfer_priority._wait_for_foreground(torch.device("cuda", 0))
            woken.set()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    waiter = threading.Thread(target=wait_gpu)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with foreground_transfer("cuda:0"):
                waiter.start()
                with pytest.raises(ValueError, match="inner"):
                    with foreground_transfer("cuda:0"):
                        raise ValueError("inner")
                assert transfer_priority._foreground.count("cuda:0") == 1
                assert not woken.wait(0.05)
                raise RuntimeError("boom")
        assert transfer_priority._foreground.counts == {}
        assert woken.wait(5.0)
    finally:
        with transfer_priority._foreground.condition:
            transfer_priority._foreground.counts.clear()
            transfer_priority._foreground.condition.notify_all()
        if waiter.ident is not None:
            waiter.join(timeout=5.0)

    assert not errors


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
    with foreground_transfer(destination.device):
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

    def instrumented_wait(device: torch.device) -> None:
        nonlocal waits
        if waits == 2:
            chunk_boundary.set()
            proceed.wait(30)
        original_wait(device)
        waits += 1

    monkeypatch.setattr(transfer_priority, "_BACKGROUND_CHUNK_BYTES", 1024)
    monkeypatch.setattr(transfer_priority, "_wait_for_foreground", instrumented_wait)

    def _run_bg(dst: torch.Tensor, src: torch.Tensor) -> None:
        with background_transfers():
            copy_host_to_device(dst, src)

    with register_foreground_producer([destination.device]):
        worker = threading.Thread(target=_run_bg, args=(destination, source))
        worker.start()
        assert chunk_boundary.wait(30)
        with foreground_transfer(destination.device):
            proceed.set()
            assert torch.equal(destination[:256].cpu(), source[:256])
            assert torch.equal(destination[512:].cpu(), torch.zeros(2048 - 512))
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
    with foreground_transfer(destination.device):
        copy_host_to_device(destination, source)
    torch.cuda.synchronize()
    assert torch.equal(destination.cpu(), source)


@CUDA
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 CUDA/HIP GPUs")
def test_foreground_transfer_does_not_stall_background_copy_on_other_gpu():
    src = torch.arange(2048, dtype=torch.float32).pin_memory()
    dst0 = torch.zeros_like(src, device="cuda:0")
    dst1 = torch.zeros_like(src, device="cuda:1")
    copied0 = threading.Event()
    copied1 = threading.Event()
    errors: list[BaseException] = []

    def copy_to(dst: torch.Tensor, flag: threading.Event) -> None:
        try:
            with background_transfers():
                copy_host_to_device(dst, src)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            flag.set()

    with register_foreground_producer(["cuda:0", "cuda:1"]):
        w0 = threading.Thread(target=copy_to, args=(dst0, copied0))
        w1 = threading.Thread(target=copy_to, args=(dst1, copied1))
        try:
            with foreground_transfer("cuda:0"):
                w0.start()
                w1.start()
                assert copied1.wait(10.0)
                assert not errors
                assert not copied0.wait(0.15)
            assert copied0.wait(30.0)
            assert not errors
        finally:
            for worker in (w0, w1):
                if worker.ident is not None:
                    worker.join(timeout=30.0)

    assert torch.equal(dst0.cpu(), src)
    assert torch.equal(dst1.cpu(), src)


def test_register_foreground_producer_controls_should_yield() -> None:
    assert not should_yield("cpu")
    assert not should_yield("cuda:0")

    with foreground_transfer("cuda:0"):
        assert should_yield("cuda:0")
        assert not should_yield("cuda:1")
        assert not should_yield("cpu")

    # Wildcard producer (devices=None) enables yielding across all devices.
    first = register_foreground_producer()
    try:
        assert should_yield("cpu")
        assert should_yield("cuda:0")
        second = register_foreground_producer()
        try:
            assert should_yield("cuda:1")
            second.close()
            second.close()
            assert second.closed
            assert should_yield("cuda:1")
        finally:
            second.close()
        first.close()
        first.close()
        assert first.closed
        assert not should_yield("cuda:0")
    finally:
        first.close()

    with pytest.raises(RuntimeError, match="closed"):
        with first:
            pass


def test_scoped_foreground_producer_limits_yielding_to_canonical_devices() -> None:
    # Scoped producers only enable yielding for their canonical deduplicated devices.
    scoped = register_foreground_producer(["cuda:0", torch.device("cuda:0")])
    try:
        assert should_yield("cuda:0")
        assert not should_yield("cuda:1")
        assert not should_yield("cpu")
    finally:
        scoped.close()
    assert not should_yield("cuda:0")

    # Empty device list is a no-op producer handle.
    empty = register_foreground_producer(())
    try:
        assert not empty.closed
        assert not should_yield("cpu")
        assert not should_yield("cuda:0")
    finally:
        empty.close()
    assert empty.closed

    # Invalid device rolls back any earlier devices in the same registration call.
    with pytest.raises(RuntimeError):
        register_foreground_producer(["cpu", "invalid_device_xyz"])
    assert not should_yield("cpu")

    dropped = register_foreground_producer(["cuda:0"])
    assert should_yield("cuda:0")
    del dropped
    gc.collect()
    assert not should_yield("cuda:0")


def test_foreground_producer_init_rollback_avoids_double_decrement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weakref

    dev0 = transfer_priority.canonical_device("cuda:0")
    dev1 = transfer_priority.canonical_device("cuda:1")
    keeper = register_foreground_producer([dev0])
    try:
        detached_calls = 0

        class FailingFinalizer:
            def __init__(self, real: weakref.finalize) -> None:
                self._real = real

            @property
            def alive(self) -> bool:
                return self._real.alive

            @property
            def atexit(self) -> bool:
                return self._real.atexit

            @atexit.setter
            def atexit(self, _value: bool) -> None:
                raise BaseException("interrupted while configuring finalizer")

            def detach(self) -> object:
                nonlocal detached_calls
                detached_calls += 1
                return self._real.detach()

            def __call__(self) -> object:
                return self._real()

        real_finalize = weakref.finalize

        def failing_finalize(obj: object, func: object, *args: object) -> FailingFinalizer:
            return FailingFinalizer(real_finalize(obj, func, *args))  # type: ignore[arg-type]

        with monkeypatch.context() as patch:
            patch.setattr(weakref, "finalize", failing_finalize)
            with pytest.raises(BaseException, match="interrupted"):
                register_foreground_producer([dev0])
        gc.collect()
        assert detached_calls == 1
        assert should_yield(dev0)
        assert transfer_priority._foreground.device_producers == {dev0: 1}

        class PartialFailureMap(dict[torch.device, int]):
            def __setitem__(self, key: torch.device, value: int) -> None:
                if key == dev1:
                    raise BaseException("interrupted mid-registration")
                super().__setitem__(key, value)

        with transfer_priority._foreground.condition:
            original_map = transfer_priority._foreground.device_producers
            failing_map = PartialFailureMap(original_map)
            transfer_priority._foreground.device_producers = failing_map
        try:
            with pytest.raises(BaseException, match="mid-registration"):
                register_foreground_producer([dev0, dev1])
        finally:
            with transfer_priority._foreground.condition:
                retained = dict(failing_map)
                transfer_priority._foreground.device_producers = retained
        gc.collect()
        assert should_yield(dev0)
        assert not should_yield(dev1)
        assert transfer_priority._foreground.device_producers == {dev0: 1}
    finally:
        keeper.close()
    assert not should_yield(dev0)
    assert transfer_priority._foreground.device_producers == {}


@CUDA
def test_single_gpu_streaming_lookahead_skips_chunk_syncs_without_foreground_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from piper_offload import ModelOffloader, OffloadConfig
    from tests.conftest import activated_model

    class _BlockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                torch.nn.Linear(64, 64, bias=False) for _ in range(3)
            )
            self.requires_grad_(False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)
            return x

    caller_thread = threading.get_ident()
    stream_syncs = 0
    event_syncs = 0
    original_stream_sync = torch.cuda.Stream.synchronize
    original_event_sync = torch.cuda.Event.synchronize

    def counting_stream_sync(stream: torch.cuda.Stream) -> None:
        nonlocal stream_syncs
        if threading.get_ident() != caller_thread:
            stream_syncs += 1
        original_stream_sync(stream)

    def counting_event_sync(event: torch.cuda.Event) -> None:
        nonlocal event_syncs
        if threading.get_ident() != caller_thread:
            event_syncs += 1
        original_event_sync(event)

    monkeypatch.setattr(transfer_priority, "_BACKGROUND_CHUNK_BYTES", 1024)
    monkeypatch.setattr(torch.cuda.Stream, "synchronize", counting_stream_sync)
    monkeypatch.setattr(torch.cuda.Event, "synchronize", counting_event_sync)

    model = _BlockModel()
    offloader = ModelOffloader.from_module(
        model,
        OffloadConfig(offload_paths=("blocks",)),
    )
    x = torch.randn(2, 64, device="cuda")

    def run_pass() -> tuple[int, int]:
        nonlocal stream_syncs, event_syncs
        stream_syncs = 0
        event_syncs = 0
        with activated_model(offloader, "cuda"), torch.inference_mode():
            model(x)
            torch.cuda.synchronize()
        return stream_syncs, event_syncs

    assert run_pass() == (0, 0)
    with register_foreground_producer(["cpu"]):
        assert run_pass() == (0, 0)
    with register_foreground_producer(["cuda"]):
        s_syncs, e_syncs = run_pass()
        assert s_syncs == 0
        assert e_syncs > 0
    assert run_pass() == (0, 0)


@CUDA
def test_mid_block_foreground_transfer_yields_lookahead_with_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from piper_offload import ModelOffloader, OffloadConfig
    from tests.conftest import activated_model

    prefetch_started = threading.Event()
    fg_entered = threading.Event()
    after_yield = threading.Event()
    saw_yield = False

    class _CooperatingBlock(torch.nn.Module):
        def __init__(self, *, trigger_fg: bool) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(256, 256, bias=False)
            self._trigger_fg = trigger_fg

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self._trigger_fg:
                assert prefetch_started.wait(10.0)
                with foreground_transfer(x.device):
                    fg_entered.set()
                    assert not after_yield.wait(0.1)
            return self.linear(x)

    class _TwoBlockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [_CooperatingBlock(trigger_fg=True), _CooperatingBlock(trigger_fg=False)]
            )
            self.requires_grad_(False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)
            return x

    original_wait = transfer_priority._wait_for_foreground
    wait_calls = 0

    def instrumented_wait(device: torch.device) -> None:
        nonlocal wait_calls, saw_yield
        wait_calls += 1
        is_second = wait_calls == 2
        if is_second:
            prefetch_started.set()
            fg_entered.wait(10.0)
            saw_yield = True
        original_wait(device)
        if is_second:
            after_yield.set()

    monkeypatch.setattr(transfer_priority, "_BACKGROUND_CHUNK_BYTES", 16 * 1024)
    monkeypatch.setattr(transfer_priority, "_wait_for_foreground", instrumented_wait)

    model = _TwoBlockModel()
    offloader = ModelOffloader.from_module(
        model,
        OffloadConfig(offload_paths=("blocks",)),
    )
    x = torch.randn(2, 256, device="cuda")

    with register_foreground_producer([x.device]):
        with activated_model(offloader, "cuda"), torch.inference_mode():
            try:
                out = model(x)
                torch.cuda.synchronize()
            finally:
                fg_entered.set()
    assert out.shape == (2, 256)
    assert saw_yield
    assert after_yield.is_set()
