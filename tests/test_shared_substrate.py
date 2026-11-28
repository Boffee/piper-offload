"""Layer 1 shared substrate contracts: lease ordering, transfers, budget, dual arena, and specs."""

import contextlib
import mmap
import threading
import time
from dataclasses import dataclass
from typing import cast

import pytest
import torch
from torch import nn

import piper_offload
from piper_offload import (
    ArenaCapacityError,
    DeviceTransferContext,
    ExpertPoolStats,
    ExpertSlabPool,
    ExpertWave,
    KvBatch,
    KvChunkLocation,
    KvChunkPool,
    KvPoolStats,
    ModelCache,
    ModelOffloader,
    ModelSpec,
    OffloadConfig,
    PinManager,
    SlotSchema,
    Uint8DualArena,
    VramBudget,
    VramBudgetExhaustedError,
    VramReservation,
)
from piper_offload.dual_arena import (
    ArenaFieldSpec,
    HighSlotLayout,
    KvMetadataPool,
    LowSlotLayout,
    SideFieldSpec,
    StructuredSidePool,
    TypedSidePool,
)
from piper_offload.host_arena import PinnedBounceRing
from piper_offload.stream_lease import StreamLease
from piper_offload.tensor_adapters import (
    ArenaBindableTensorAdapter,
    BufferSpec,
    GpuAllocator,
    RegularAdapter,
    SliceableTensorAdapter,
)
from tests.conftest import Interrupt, activated_model, residency_engine

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Block(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(x))


class _BlockModel(nn.Module):
    def __init__(self, num_blocks: int = 4, width: int = 32) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(width) for _ in range(num_blocks))
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _ResidentAndBlocksModel(nn.Module):
    def __init__(self, num_blocks: int = 3, width: int = 32) -> None:
        super().__init__()
        self.embed = nn.Linear(width, width, bias=False)
        self.blocks = nn.ModuleList(_Block(width) for _ in range(num_blocks))
        self.head = nn.Linear(width, width, bias=False)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _make_fake_stream_types():
    class FakeEvent:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    class FakeStream:
        def __init__(self, *, record_failures: int = 0, wait_failures: int = 0) -> None:
            self.record_failures = record_failures
            self.wait_failures = wait_failures
            self.waited_events: list[FakeEvent] = []
            self.waited_streams: list[FakeStream] = []
            self.events: list[FakeEvent] = []
            self.synchronize_calls = 0

        def wait_event(self, event: FakeEvent) -> None:
            if self.wait_failures:
                self.wait_failures -= 1
                raise Interrupt
            self.waited_events.append(event)

        def wait_stream(self, stream: FakeStream) -> None:
            self.waited_streams.append(stream)

        def record_event(self) -> FakeEvent:
            if self.record_failures:
                self.record_failures -= 1
                raise Interrupt
            event = FakeEvent()
            self.events.append(event)
            return event

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    return FakeEvent, FakeStream


# ---------------------------------------------------------------------------
# Public Export Surface
# ---------------------------------------------------------------------------


def test_top_level_exports_expose_user_coordination_types_only() -> None:
    _ = (ExpertPoolStats, ExpertSlabPool, ExpertWave, KvBatch, KvChunkLocation, KvChunkPool, KvPoolStats)
    top_all = set(piper_offload.__all__)
    assert {
        "ArenaCapacityError",
        "DeviceTransferContext",
        "ExpertPoolStats",
        "ExpertSlabPool",
        "ExpertWave",
        "KvBatch",
        "KvChunkLocation",
        "KvChunkPool",
        "KvPoolStats",
        "SlotSchema",
        "Uint8DualArena",
        "VramBudget",
        "VramBudgetExhaustedError",
        "VramReservation",
    } <= top_all
    submodule_only_names = {
        "ArenaBindableTensorAdapter",
        "ArenaFieldSpec",
        "ArenaSlice",
        "BudgetTenantHandle",
        "BufferSpec",
        "GpuAllocator",
        "HighSlotLayout",
        "KvMetadataPool",
        "LowSlotLayout",
        "SideFieldSpec",
        "SliceableTensorAdapter",
        "StreamLease",
        "StructuredSidePool",
        "TypedSidePool",
    }
    assert top_all.isdisjoint(submodule_only_names)


# ---------------------------------------------------------------------------
# StreamLease
# ---------------------------------------------------------------------------


def test_stream_lease_multi_reader_and_selective_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, fake_stream_cls = _make_fake_stream_types()
    alloc_stream = fake_stream_cls()
    copy_stream = fake_stream_cls()
    reader_a = fake_stream_cls()
    reader_b = fake_stream_cls()
    side_stream = fake_stream_cls()
    lease = StreamLease(cast(torch.cuda.Stream, alloc_stream))
    monkeypatch.setattr(torch.cuda, "stream", lambda _s: contextlib.nullcontext())

    lease.stage_with(lambda: None, cast(torch.cuda.Stream, copy_stream))
    ready = copy_stream.events[-1]
    assert lease.is_staged
    assert not lease.is_acquired
    assert lease.reader_count == 0

    lease.acquire(cast(torch.cuda.Stream, reader_a), exclusive=False)
    lease.acquire(cast(torch.cuda.Stream, reader_b), exclusive=False)
    assert lease.is_acquired
    assert lease.reader_count == 2
    assert reader_a.waited_events == [ready]
    assert reader_b.waited_events == [ready]

    with pytest.raises(RuntimeError, match="already acquired"):
        lease.acquire(cast(torch.cuda.Stream, side_stream), exclusive=True)

    lease.record_stream(cast(torch.cuda.Stream, side_stream))
    assert side_stream.waited_events == [ready]
    with pytest.raises(ValueError, match="has not acquired"):
        lease.release(cast(torch.cuda.Stream, side_stream))

    lease.release(cast(torch.cuda.Stream, reader_a))
    assert lease.is_acquired
    assert lease.reader_count == 1
    assert len(reader_a.events) == 1
    assert side_stream.events == []
    with pytest.raises(ValueError, match="has not acquired"):
        lease.release(cast(torch.cuda.Stream, reader_a))

    with pytest.raises(RuntimeError, match="while it is in use"):
        lease.stage_with(lambda: None, cast(torch.cuda.Stream, copy_stream))

    # Final release snapshots reader_b and registered side_stream together.
    lease.release(cast(torch.cuda.Stream, reader_b))
    assert not lease.is_acquired
    assert lease.reader_count == 0
    assert len(reader_b.events) == 1
    assert len(side_stream.events) == 1

    # Restaging clears dependencies and orders copy_stream after prior readers.
    lease.stage_with(lambda: None, cast(torch.cuda.Stream, copy_stream))
    assert ready in copy_stream.waited_events
    assert reader_a.events[0] in copy_stream.waited_events
    assert reader_b.events[0] in copy_stream.waited_events
    assert side_stream.events[0] in copy_stream.waited_events

    lease.synchronize_and_reset()
    assert not lease.is_staged


def test_stream_lease_acquire_rolls_back_on_wait_event_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, fake_stream_cls = _make_fake_stream_types()
    alloc_stream = fake_stream_cls()
    copy_stream = fake_stream_cls()
    flaky_reader = fake_stream_cls(wait_failures=1)
    healthy_reader = fake_stream_cls()
    lease = StreamLease(cast(torch.cuda.Stream, alloc_stream))
    monkeypatch.setattr(torch.cuda, "stream", lambda _s: contextlib.nullcontext())

    lease.stage_with(lambda: None, cast(torch.cuda.Stream, copy_stream))
    with pytest.raises(Interrupt):
        lease.acquire(cast(torch.cuda.Stream, flaky_reader), exclusive=True)

    assert not lease.is_acquired
    assert lease.reader_count == 0
    assert not lease._exclusive
    assert lease._stream_readers == {}
    # Subsequent exclusive acquire succeeds cleanly once wait_event recovers.
    lease.acquire(cast(torch.cuda.Stream, healthy_reader), exclusive=True)
    assert lease.is_acquired
    assert lease.reader_count == 1
    lease.release()
    lease.synchronize_and_reset()


def test_stream_lease_gpu_born_produce_with_write_once_d2h_and_failure_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, fake_stream_cls = _make_fake_stream_types()
    alloc_stream = fake_stream_cls()
    compute_stream = fake_stream_cls()
    d2h_stream = fake_stream_cls()
    lease = StreamLease(cast(torch.cuda.Stream, alloc_stream))
    monkeypatch.setattr(torch.cuda, "stream", lambda _s: contextlib.nullcontext())

    # Partial write failure still captures ready event for sync without setting dirty/staged.
    with pytest.raises(Interrupt):
        lease.produce_with(
            lambda: (_ for _ in ()).throw(Interrupt),
            cast(torch.cuda.Stream, compute_stream),
        )
    assert compute_stream.waited_streams == [alloc_stream]
    assert len(compute_stream.events) == 1
    assert not lease.is_dirty
    assert not lease.is_staged

    lease.produce_with(lambda: None, cast(torch.cuda.Stream, compute_stream))
    assert lease.is_dirty
    assert lease.is_staged

    with pytest.raises(RuntimeError, match="Dirty"):
        lease.produce_with(lambda: None, cast(torch.cuda.Stream, compute_stream))
    with pytest.raises(RuntimeError, match="Dirty"):
        lease.stage_with(lambda: None, cast(torch.cuda.Stream, compute_stream))

    def fail_d2h() -> None:
        raise Interrupt

    with pytest.raises(Interrupt):
        lease.spill_to_host(fail_d2h, cast(torch.cuda.Stream, d2h_stream))
    assert lease.d2h_in_flight
    # Failed spill still syncs partial D2H event on wait_d2h(), but keeps is_dirty=True.
    lease.wait_d2h()
    assert not lease.d2h_in_flight
    assert lease.is_dirty
    assert d2h_stream.events[-1].synchronize_calls == 1

    lease.spill_to_host(lambda: None, cast(torch.cuda.Stream, d2h_stream))
    assert lease.d2h_in_flight
    with pytest.raises(RuntimeError, match="already been started"):
        lease.spill_to_host(lambda: None, cast(torch.cuda.Stream, d2h_stream))

    lease.wait_d2h()
    assert not lease.d2h_in_flight
    assert not lease.is_dirty

    lease.produce_with(lambda: None, cast(torch.cuda.Stream, compute_stream))
    lease.discard_dirty()
    assert not lease.is_dirty
    assert not lease.is_staged
    lease.synchronize_and_reset()
    assert compute_stream.events[0].synchronize_calls == 0
    assert compute_stream.events[-1].synchronize_calls == 1


def test_stream_lease_d2h_fallback_stream_exclusive_guard_and_sync_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, fake_stream_cls = _make_fake_stream_types()

    class FlakySyncStream(fake_stream_cls):
        def __init__(self, *, sync_failures: int = 0, **kwargs: object) -> None:
            super().__init__(**kwargs)  # type: ignore[arg-type]
            self.sync_failures = sync_failures

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            if self.sync_failures:
                self.sync_failures -= 1
                raise Interrupt("sync failed")

    alloc_stream = FlakySyncStream(sync_failures=1)
    compute_stream = FlakySyncStream()
    reader_stream = FlakySyncStream()
    d2h_stream = FlakySyncStream(record_failures=2)
    lease = StreamLease(cast(torch.cuda.Stream, alloc_stream))
    monkeypatch.setattr(torch.cuda, "stream", lambda _s: contextlib.nullcontext())

    # If both copy_fn or _record_event fail, d2h_fallback_stream keeps d2h_in_flight=True
    # so wait_d2h() and discard_dirty() still synchronize d2h_stream on the host.
    lease.produce_with(lambda: None, cast(torch.cuda.Stream, compute_stream))
    with pytest.raises(Interrupt):
        lease.spill_to_host(lambda: None, cast(torch.cuda.Stream, d2h_stream))
    assert lease.d2h_in_flight
    with pytest.raises(RuntimeError, match="exclusively acquired while a D2H"):
        lease.acquire(cast(torch.cuda.Stream, reader_stream), exclusive=True)
    with pytest.raises(RuntimeError, match="D2H spill is pending"):
        lease._d2h_succeeded = True
        lease._dirty = False
        lease.stage_with(lambda: None, cast(torch.cuda.Stream, compute_stream))

    lease._dirty = True
    lease._d2h_succeeded = False
    lease.discard_dirty()
    assert d2h_stream.synchronize_calls == 1
    assert not lease.d2h_in_flight
    assert not lease.is_dirty

    # Nested same-stream non-exclusive acquire decrements reader_count step by step.
    lease.stage_with(lambda: None, cast(torch.cuda.Stream, compute_stream))
    lease.acquire(cast(torch.cuda.Stream, reader_stream), exclusive=False)
    lease.acquire(cast(torch.cuda.Stream, reader_stream), exclusive=False)
    assert lease.reader_count == 2
    lease.release(cast(torch.cuda.Stream, reader_stream))
    assert lease.is_acquired
    assert lease.reader_count == 1
    lease.release(cast(torch.cuda.Stream, reader_stream))
    assert not lease.is_acquired

    # Interruption during _allocation_stream.synchronize() keeps acquired state intact for retry.
    lease.acquire(cast(torch.cuda.Stream, reader_stream), exclusive=True)
    with pytest.raises(Interrupt, match="sync failed"):
        lease.synchronize_and_reset()
    assert lease.is_acquired
    assert lease.is_staged
    lease.synchronize_and_reset()
    assert not lease.is_acquired
    assert not lease.is_staged
    assert alloc_stream.synchronize_calls == 2


@CUDA
def test_stream_lease_cuda_gpu_born_spill_and_concurrent_readers() -> None:
    device = torch.device("cuda")
    alloc_stream = torch.cuda.current_stream(device)
    compute_stream = torch.cuda.Stream(device=device)
    reader_stream_a = torch.cuda.Stream(device=device)
    reader_stream_b = torch.cuda.Stream(device=device)
    d2h_stream = torch.cuda.Stream(device=device)

    gpu_buf = torch.empty((256,), dtype=torch.float32, device=device)
    host_buf = torch.empty((256,), dtype=torch.float32, pin_memory=True)
    expected = torch.arange(256, dtype=torch.float32) * 1.5
    lease = StreamLease(alloc_stream)
    try:
        def write_gpu() -> None:
            gpu_buf.copy_(torch.arange(256, dtype=torch.float32, device=device) * 1.5)

        lease.produce_with(write_gpu, compute_stream)
        assert lease.is_dirty

        lease.acquire(reader_stream_a, exclusive=False)
        lease.acquire(reader_stream_b, exclusive=False)
        assert lease.reader_count == 2
        with torch.cuda.stream(reader_stream_a):
            read_a = gpu_buf.clone()
        with torch.cuda.stream(reader_stream_b):
            read_b = gpu_buf * 2.0

        lease.spill_to_host(
            lambda: host_buf.copy_(gpu_buf, non_blocking=True),
            d2h_stream,
        )
        lease.release(reader_stream_a)
        lease.release(reader_stream_b)
        lease.wait_d2h()
        assert not lease.is_dirty

        reader_stream_a.synchronize()
        reader_stream_b.synchronize()
        torch.testing.assert_close(host_buf, expected, rtol=0, atol=0)
        torch.testing.assert_close(read_a.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(read_b.cpu(), expected * 2.0, rtol=0, atol=0)
    finally:
        lease.synchronize_and_reset()


# ---------------------------------------------------------------------------
# DeviceTransferContext
# ---------------------------------------------------------------------------


def test_device_transfer_context_rejects_non_cuda() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        DeviceTransferContext("cpu")
    with pytest.raises(ValueError, match="int"):
        DeviceTransferContext("cuda", stream_priority=True)


@CUDA
def test_device_transfer_context_close_does_not_deadlock_worker_and_retries_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    proceed = threading.Event()
    seen: list[object] = []

    with DeviceTransferContext("cuda") as ctx:
        fut = ctx.submit(
            lambda: (
                entered.set(),
                proceed.wait(timeout=5.0),
                seen.append((ctx.copy_stream, ctx.borrower_count, ctx.is_closed)),
            )
        )
        assert entered.wait(timeout=5.0)
        proceed.set()
        ctx.close()
        fut.result(timeout=5.0)
    assert len(seen) == 1
    assert seen[0][0] is not None
    assert seen[0][1] == 0
    assert seen[0][2] is False
    assert ctx.is_closed

    # Sync failure leaves ctx in CLOSING state (rejecting re-open/borrow) until retry succeeds.
    flaky_ctx = DeviceTransferContext("cuda")
    stream = flaky_ctx.ensure_stream()
    original_sync = stream.synchronize
    failed_once = False

    def fail_sync_once() -> None:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise Interrupt("stream sync interrupted")
        original_sync()

    with monkeypatch.context() as patch:
        patch.setattr(stream, "synchronize", fail_sync_once)
        with pytest.raises(Interrupt, match="stream sync interrupted"):
            flaky_ctx.close()

    assert flaky_ctx.is_open
    assert not flaky_ctx.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        flaky_ctx.ensure_stream()
    with pytest.raises(RuntimeError, match="closed"):
        flaky_ctx._acquire_borrow()
    with pytest.raises(RuntimeError, match="closed"):
        flaky_ctx.submit(lambda: None)
    flaky_ctx.close()
    assert flaky_ctx.is_closed
    assert not flaky_ctx.is_open


@CUDA
def test_device_transfer_context_lazy_stream_without_thread_and_closed_guard() -> None:
    ctx = DeviceTransferContext("cuda")
    assert not ctx.is_open
    assert not ctx.is_closed
    assert ctx.borrower_count == 0

    stream = ctx.ensure_stream()
    assert stream is ctx.copy_stream
    assert ctx.executor is None
    assert ctx.is_open

    ctx.close()
    assert not ctx.is_open
    assert ctx.is_closed
    ctx.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        ctx.ensure_stream()
    with pytest.raises(RuntimeError, match="closed"):
        ctx.ensure()
    with pytest.raises(RuntimeError, match="closed"):
        ctx.submit(lambda: None)

    model = _BlockModel(num_blocks=2, width=16)
    offloader = ModelOffloader.from_module(model, OffloadConfig(offload_paths=("blocks",)))
    engine = residency_engine(offloader)
    with pytest.raises(RuntimeError, match="closed"):
        offloader.activate("cuda", transfer_context=ctx)
    assert engine.active_device is None
    assert not offloader.failed
    assert not engine.retains_resources


@CUDA
def test_device_transfer_context_rejects_mismatched_device() -> None:
    torch.manual_seed(0)
    model = _BlockModel(num_blocks=2, width=16)
    offloader = ModelOffloader.from_module(
        model,
        OffloadConfig(offload_paths=("blocks",)),
    )
    ctx = DeviceTransferContext(torch.device("cuda", 0))
    try:
        with pytest.raises(ValueError, match="does not match activation device"):
            offloader.activate(torch.device("cuda", 1), transfer_context=ctx)
        assert not offloader.failed
        assert not ctx.is_open
        assert ctx.borrower_count == 0
    finally:
        ctx.close()


@CUDA
def test_device_transfer_context_shared_and_refuses_close_while_borrowed() -> None:
    torch.manual_seed(0)
    model_a = _BlockModel(num_blocks=3, width=32)
    model_b = _BlockModel(num_blocks=3, width=32)
    x = torch.randn(2, 32, device="cuda")
    with torch.inference_mode():
        expected_a = model_a.to("cuda")(x).cpu()
        model_a.cpu()
        expected_b = model_b.to("cuda")(x).cpu()
        model_b.cpu()

    cfg = OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=0)
    offloader_a = ModelOffloader.from_module(model_a, cfg)
    cache = ModelCache()
    spec_b = ModelSpec(key="model_b", factory=lambda: model_b, config=cfg)

    ctx = DeviceTransferContext("cuda")
    assert not ctx.is_open
    try:
        offloader_a.activate("cuda", vram_budget_bytes=0, transfer_context=ctx)
        assert ctx.borrower_count == 1
        with cache.use(spec_b, device="cuda", vram_budget_bytes=0, transfer_context=ctx) as active_b:
            assert ctx.borrower_count == 2
            with pytest.raises(RuntimeError, match="active borrowers"):
                ctx.close()
            with torch.inference_mode():
                out_a = model_a(x).cpu()
                out_b = active_b(x).cpu()
            engine_a = residency_engine(offloader_a)
            assert engine_a._copy_stream is ctx.copy_stream
            assert engine_a._executor is ctx.executor
            assert ctx.is_open

        assert ctx.borrower_count == 1
        assert ctx.is_open
        with torch.inference_mode():
            out_a2 = model_a(x).cpu()
        offloader_a.deactivate()
        assert ctx.borrower_count == 0
        assert ctx.is_open

        torch.testing.assert_close(out_a, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(out_a2, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(out_b, expected_b, rtol=0, atol=0)
    finally:
        if offloader_a.active_device is not None:
            offloader_a.deactivate()
        ctx.close()
    assert ctx.is_closed
    assert not ctx.is_open


# ---------------------------------------------------------------------------
# VramBudget, ResidencyEngine, Option B, Concurrency & no_record_trace
# ---------------------------------------------------------------------------


def test_vram_budget_priority_reentrancy_and_resize() -> None:
    budget = VramBudget(1100, headroom_bytes=100)
    assert budget.limit_bytes == 1100
    assert budget.headroom_bytes == 100
    assert budget.effective_limit_bytes == 1000
    assert budget.available_bytes == 1000

    evict_log: list[tuple[str, int, int]] = []
    held_low_priority = [600]
    held_high_priority = [300]

    def evict_first(ask: int) -> int:
        used_seen = budget.used_bytes
        freed = min(ask, held_low_priority[0] - 100)
        held_low_priority[0] -= freed
        evict_log.append(("low_retention", ask, used_seen))
        return freed

    def evict_second(ask: int) -> int:
        used_seen = budget.used_bytes
        freed = min(ask, held_high_priority[0])
        held_high_priority[0] -= freed
        evict_log.append(("high_retention", ask, used_seen))
        return freed

    h_keep = budget.register_tenant(
        "kv",
        priority=0,
        evict_cb=evict_second,
        usage_fn=lambda: held_high_priority[0],
        min_bytes=200,
    )
    h_evict = budget.register_tenant(
        "weights",
        priority=10,
        evict_cb=evict_first,
        usage_fn=lambda: held_low_priority[0],
        min_bytes=100,
    )
    assert budget.used_bytes == 900
    assert budget.available_bytes == 100
    assert h_keep.reclaimable_headroom_bytes() == 100 + (600 - 100)

    with budget.reserve(700, name="kv_arena") as res:
        assert isinstance(res, VramReservation)
        assert [entry[0] for entry in evict_log] == ["low_retention", "high_retention"]
        assert held_low_priority[0] == 100
        assert held_high_priority[0] == 200
        assert res.nbytes == 700
        assert budget.used_bytes == 1000

        with pytest.raises(VramBudgetExhaustedError):
            res.resize(950)
        assert res.nbytes == 700

        res.resize(500)
        assert res.nbytes == 500
        assert budget.available_bytes == 200
    assert budget.reserved_bytes == 0

    held_low_priority[0] = 200
    held_high_priority[0] = 250
    budget.set_limit(400)  # effective = 300; deficit = 150 -> evicts weights (100) + kv (50)
    assert held_low_priority[0] == 100
    assert held_high_priority[0] == 200
    h_keep.unregister()
    h_evict.unregister()


def test_vram_budget_yield_to_peers_ownership_concurrent_retry_and_no_progress() -> None:
    # yield_to_peers controls equal-priority eviction; foreign/bool handles are rejected.
    budget = VramBudget(500)
    other_budget = VramBudget(500)
    peer_strict = [200]
    peer_yielding = [200]
    requester_used = [100]

    def evict_yielding(ask: int) -> int:
        freed = min(ask, peer_yielding[0])
        peer_yielding[0] -= freed
        return freed

    h_strict = budget.register_tenant(
        "kv_strict",
        priority=10,
        evict_cb=lambda ask: ask,
        usage_fn=lambda: peer_strict[0],
        yield_to_peers=False,
    )
    h_yield = budget.register_tenant(
        "weights_yielding",
        priority=10,
        evict_cb=evict_yielding,
        usage_fn=lambda: peer_yielding[0],
        yield_to_peers=True,
    )
    h_req = budget.register_tenant(
        "requester",
        priority=10,
        usage_fn=lambda: requester_used[0],
        yield_to_peers=True,
    )
    h_other = other_budget.register_tenant("other", usage_fn=lambda: 0)
    assert h_req.reclaimable_headroom_bytes() == 200  # only yielding peer, not strict peer
    new_ceiling = h_req.request_bytes(200)
    assert new_ceiling == 300
    assert peer_yielding[0] == 0
    assert peer_strict[0] == 200

    with pytest.raises(ValueError, match="does not belong"):
        budget.unregister_tenant(h_other)
    assert budget.unregister_tenant(h_other.tenant_id) is None  # own tenant id still works cleanly
    with pytest.raises(ValueError, match="does not belong"):
        budget.ceiling_for(h_other)
    with pytest.raises(TypeError, match="BudgetTenantHandle or int"):
        budget.unregister_tenant(True)

    h_strict.unregister()
    h_yield.unregister()
    h_req.unregister()

    # Concurrent reservation release during _evict_step avoids spurious VramBudgetExhaustedError.
    race_budget = VramBudget(100)
    blocker = race_budget.reserve(80, name="blocker")

    def evict_and_release(_ask: int) -> int:
        blocker.release()
        return 0  # callback released reservation directly without lowering tenant usage

    probe = [10]
    race_budget.register_tenant(
        "helper",
        priority=5,
        evict_cb=evict_and_release,
        usage_fn=lambda: probe[0],
    )
    with race_budget.reserve(50, name="after_release") as r2:
        assert r2.nbytes == 50

    blocker2 = race_budget.reserve(40, name="blocker2")
    with race_budget.reserve(10, name="growable") as grow_res:
        def evict_for_resize(_ask: int) -> int:
            blocker2.release()
            return 0

        race_budget.register_tenant(
            "helper2",
            priority=5,
            evict_cb=evict_for_resize,
            usage_fn=lambda: 10,
        )
        grow_res.resize(50)
        assert grow_res.nbytes == 50

    # No-progress evict_cb returning >0 without lowering usage_fn() must not hang.
    stuck_budget = VramBudget(100)
    stuck_budget.register_tenant(
        "stubborn",
        priority=10,
        evict_cb=lambda ask: ask,
        usage_fn=lambda: 100,
    )
    with pytest.raises(VramBudgetExhaustedError):
        stuck_budget.reserve(50, name="blocked")


def test_vram_budget_try_claim_atomic_eviction_and_validation() -> None:
    budget = VramBudget(100)
    peer_bytes = [80]
    strict_bytes = [20]

    def evict_peer(ask: int) -> int:
        freed = min(ask, peer_bytes[0])
        peer_bytes[0] -= freed
        return freed

    h_strict = budget.register_tenant(
        "strict_kv",
        priority=10,
        evict_cb=lambda ask: ask,
        usage_fn=lambda: strict_bytes[0],
        yield_to_peers=False,
    )
    h_peer = budget.register_tenant(
        "yielding_weights",
        priority=10,
        evict_cb=evict_peer,
        usage_fn=lambda: peer_bytes[0],
        yield_to_peers=True,
    )
    h_caller = budget.register_tenant(
        "caller",
        priority=10,
        usage_fn=lambda: 0,
        yield_to_peers=True,
    )

    # Atomic claim evicts yielding peer but leaves non-yielding strict peer alone.
    claim = h_caller.try_claim(80, name="slot_claim")
    assert claim is not None
    assert peer_bytes[0] == 0
    assert strict_bytes[0] == 20
    assert budget.used_bytes == 100
    assert budget.available_bytes == 0

    # Exhausted budget returns None rather than raising or overshooting.
    assert h_caller.try_claim(1) is None

    claim.release()
    assert budget.available_bytes == 80

    # Concurrent reservation release during eviction retry lets try_claim succeed.
    other = VramBudget(50)
    h_other = other.register_tenant("other", usage_fn=lambda: 0)
    with pytest.raises(ValueError, match="does not belong"):
        budget.try_claim(h_other, 10)
    with pytest.raises(TypeError, match="BudgetTenantHandle or int"):
        budget.try_claim(True, 10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        h_caller.try_claim(-1)

    blocker = other.reserve(80 if False else 40, name="blocker")
    other_usage = [10]
    other.register_tenant(
        "releaser",
        priority=10,
        evict_cb=lambda _ask: (blocker.release(), 0)[1],
        usage_fn=lambda: other_usage[0],
        yield_to_peers=True,
    )
    retry_claim = h_other.try_claim(40)
    assert retry_claim is not None
    assert retry_claim.nbytes == 40
    retry_claim.release()
    h_strict.unregister()
    h_peer.unregister()
    h_caller.unregister()


def _setup_fake_elastic_pools(
    monkeypatch: pytest.MonkeyPatch,
    *,
    limit_bytes: int,
    slot_bytes: int = 80,
):
    import time
    from piper_offload import vram_pool as vp

    class _FakeLease:
        def __init__(self, target_bytes: int) -> None:
            self.target = target_bytes
            self.is_staged = True

        def close(self) -> None:
            pass

    @dataclass(frozen=True)
    class _FakePlan:
        sig: str
        nbytes: int = slot_bytes

    def _fake_allocate(
        plan: object,
        _device: object,
        *,
        allocation_stream: object = None,
        buffer_names: object = None,
    ) -> _FakeLease:
        del allocation_stream, buffer_names
        time.sleep(0.01)
        return _FakeLease(cast(_FakePlan, plan).nbytes)

    monkeypatch.setattr(vp.CudaTargetLease, "allocate", _fake_allocate)
    monkeypatch.setattr(vp, "estimate_plan_gpu_bytes", lambda plan: cast(_FakePlan, plan).nbytes)
    monkeypatch.setattr(vp, "target_gpu_bytes", lambda target: cast(int, target))
    monkeypatch.setattr(
        vp,
        "plan_target_signature",
        lambda plan: cast(_FakePlan, plan).sig,
    )

    budget = VramBudget(limit_bytes)

    def _make_pool(name: str, static_budget: int | None = None):
        pool = vp.ElasticVramPool(torch.device("cpu"), vram_budget_bytes=static_budget)
        handle = budget.register_tenant(
            name,
            priority=10,
            evict_cb=lambda needed: pool.evict_idle_bytes(
                needed,
                distance_fn=lambda _idx: 1.0,
            ),
            usage_fn=lambda: pool.used_bytes,
            yield_to_peers=True,
        )
        pool.set_budget_callbacks(
            budget_fn=handle.ceiling_bytes,
            claim_bytes_fn=handle.try_claim,
        )
        return pool, handle

    return budget, _make_pool, _FakePlan


def test_concurrent_two_pool_lookahead_and_reserve_never_overshoot_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget, make_pool, fake_plan_cls = _setup_fake_elastic_pools(
        monkeypatch,
        limit_bytes=100,
        slot_bytes=80,
    )
    pool_a, h_a = make_pool("model_a")
    pool_b, h_b = make_pool("model_b")
    stream = cast(torch.cuda.Stream, object())
    try:
        # Codex reproducer: simultaneous 80B lookahead checkouts on 100B budget.
        for round_idx in range(4):
            barrier = threading.Barrier(2)
            results: list[object] = []
            errors: list[BaseException] = []

            def _worker(
                pool=pool_a,
                idx: int = round_idx,
                b: threading.Barrier = barrier,
                res: list[object] = results,
                err: list[BaseException] = errors,
            ) -> None:
                try:
                    b.wait(timeout=5.0)
                    slot, _ = pool.checkout(
                        idx,
                        cast(object, fake_plan_cls(f"sig_{idx}")),  # type: ignore[arg-type]
                        stream,
                        distance_fn=lambda _i: 1.0,
                        is_lookahead=True,
                    )
                    res.append(slot)
                except BaseException as exc:  # pragma: no cover
                    err.append(exc)

            t1 = threading.Thread(target=_worker, args=(pool_a, round_idx * 2))
            t2 = threading.Thread(target=_worker, args=(pool_b, round_idx * 2 + 1))
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)
            assert not errors
            admitted = [s for s in results if s is not None]
            assert len(admitted) <= 1
            assert budget.used_bytes <= 100
            pool_a.close_all()
            pool_b.close_all()

        # Concurrent lookahead checkout vs VramBudget.reserve(80) cannot both succeed.
        barrier2 = threading.Barrier(2)
        lookahead_slot: list[object] = []
        reserved: list[VramReservation] = []

        def _reserve_worker() -> None:
            barrier2.wait(timeout=5.0)
            with contextlib.suppress(VramBudgetExhaustedError):
                reserved.append(budget.reserve(80, name="kv"))

        t_res = threading.Thread(target=_reserve_worker)
        t_res.start()
        barrier2.wait(timeout=5.0)
        slot, _ = pool_a.checkout(
            0,
            cast(object, fake_plan_cls("sig_r")),  # type: ignore[arg-type]
            stream,
            distance_fn=lambda _i: 1.0,
            is_lookahead=True,
        )
        lookahead_slot.append(slot)
        t_res.join(timeout=5.0)
        assert not (lookahead_slot[0] is not None and len(reserved) == 1)
        assert budget.used_bytes <= 100
        for r in reserved:
            r.release()
    finally:
        pool_a.close_all()
        pool_b.close_all()
        h_a.unregister()
        h_b.unregister()


def test_heterogeneous_own_idle_reclaim_races_safely_against_peer_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget, make_pool, fake_plan_cls = _setup_fake_elastic_pools(
        monkeypatch,
        limit_bytes=100,
        slot_bytes=80,
    )
    pool_a, h_a = make_pool("model_a")
    pool_b, h_b = make_pool("model_b")
    stream = cast(torch.cuda.Stream, object())
    try:
        slot_0, _ = pool_a.checkout(
            0,
            cast(object, fake_plan_cls("sig_A")),  # type: ignore[arg-type]
            stream,
            distance_fn=lambda _i: 1.0,
            is_lookahead=False,
        )
        assert slot_0 is not None
        pool_a.return_slot(slot_0, staged_unit_idx=0, distance_fn=lambda _i: 1.0)
        assert budget.used_bytes == 80

        # pool_a checks out incompatible sig_B (must evict own sig_A and reclaim) while pool_b races.
        barrier = threading.Barrier(2)
        admitted: list[object] = []

        def _race(pool, unit_idx: int, sig: str) -> None:
            barrier.wait(timeout=5.0)
            s, _ = pool.checkout(
                unit_idx,
                cast(object, fake_plan_cls(sig)),  # type: ignore[arg-type]
                stream,
                distance_fn=lambda _i: 1.0,
                is_lookahead=True,
            )
            if s is not None:
                admitted.append(s)

        t1 = threading.Thread(target=_race, args=(pool_a, 1, "sig_B"))
        t2 = threading.Thread(target=_race, args=(pool_b, 2, "sig_C"))
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        assert len(admitted) <= 1
        assert budget.used_bytes <= 100
    finally:
        pool_a.close_all()
        pool_b.close_all()
        h_a.unregister()
        h_b.unregister()


@CUDA
def test_vram_budget_option_b_resident_reservation_and_failure_recovery() -> None:
    torch.manual_seed(0)
    model = _ResidentAndBlocksModel(num_blocks=3, width=32)
    x = torch.randn(2, 32, device="cuda")
    with torch.inference_mode():
        expected = model.to("cuda")(x).cpu()
        model.cpu()

    param_bytes = 32 * 32 * 4
    resident_bytes = 2 * param_bytes  # embed + head
    offloader = ModelOffloader.from_module(
        model,
        OffloadConfig(offload_paths=("blocks",)),
    )

    # Budget smaller than resident weights fails loudly without poisoning offloader/borrowers.
    tiny_budget = VramBudget(resident_bytes - 16)
    ctx = DeviceTransferContext("cuda")
    try:
        with pytest.raises(VramBudgetExhaustedError):
            offloader.activate("cuda", vram_budget=tiny_budget, transfer_context=ctx)
        assert not offloader.failed
        assert tiny_budget.used_bytes == 0
        assert ctx.borrower_count == 0
        assert not residency_engine(offloader).retains_resources

        # Exact resident + 2 offloaded blocks succeeds and tracks resident reservation separately.
        good_budget = VramBudget(resident_bytes + 2 * param_bytes)
        offloader.activate("cuda", vram_budget=good_budget, transfer_context=ctx)
        assert good_budget.reserved_bytes == resident_bytes
        with torch.inference_mode():
            out = model(x).cpu()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert good_budget.used_bytes <= resident_bytes + 2 * param_bytes

        # External consumer trying to take room needed by resident weights cannot steal it.
        with pytest.raises(VramBudgetExhaustedError):
            good_budget.reserve(2 * param_bytes + 1, name="oversized_kv")
    finally:
        if offloader.active_device is not None:
            offloader.deactivate()
        ctx.close()
    assert good_budget.used_bytes == 0
    assert not residency_engine(offloader).retains_resources


@CUDA
def test_vram_budget_concurrent_forward_and_reserve_eviction() -> None:
    torch.manual_seed(0)
    model = _BlockModel(num_blocks=4, width=64)
    x = torch.randn(2, 64, device="cuda")
    with torch.inference_mode():
        expected = model.to("cuda")(x).cpu()
        model.cpu()

    unit_bytes = 64 * 64 * 4
    budget = VramBudget(4 * unit_bytes)
    offloader = ModelOffloader.from_module(
        model,
        OffloadConfig(offload_paths=("blocks",)),
    )
    errors: list[BaseException] = []
    stop_event = threading.Event()

    def kv_worker() -> None:
        try:
            while not stop_event.is_set():
                try:
                    with budget.reserve(2 * unit_bytes, name="concurrent_kv"):
                        pass
                except VramBudgetExhaustedError:
                    pass
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    try:
        offloader.activate("cuda", vram_budget=budget)
        reserver = threading.Thread(target=kv_worker, daemon=True)
        reserver.start()
        try:
            with torch.inference_mode():
                for _ in range(20):
                    out = model(x).cpu()
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
        finally:
            stop_event.set()
            reserver.join(timeout=5.0)
        assert not errors
    finally:
        if offloader.active_device is not None:
            offloader.deactivate()
    assert budget.used_bytes == 0


@CUDA
def test_vram_budget_shared_two_models_and_model_cache() -> None:
    torch.manual_seed(0)
    model_a = _ResidentAndBlocksModel(num_blocks=4, width=32)
    model_b = _ResidentAndBlocksModel(num_blocks=4, width=32)
    x = torch.randn(2, 32, device="cuda")
    with torch.inference_mode():
        expected_a = model_a.to("cuda")(x).cpu()
        model_a.cpu()
        expected_b = model_b.to("cuda")(x).cpu()
        model_b.cpu()

    unit_bytes = 32 * 32 * 4
    resident_bytes = 2 * unit_bytes
    # Room for both resident weights + 3 unit slots: model A caches an idle slot, and model B reclaims it.
    total_limit = 2 * resident_bytes + 3 * unit_bytes
    budget = VramBudget(total_limit)

    offloader_a = ModelOffloader.from_module(model_a, OffloadConfig(offload_paths=("blocks",)))
    cache = ModelCache()
    spec_b = ModelSpec(
        key="model_b_shared",
        factory=lambda: model_b,
        config=OffloadConfig(offload_paths=("blocks",)),
    )

    try:
        offloader_a.activate("cuda", vram_budget=budget)
        with cache.use(spec_b, device="cuda", vram_budget=budget) as active_b:
            with torch.inference_mode():
                out_a1 = model_a(x).cpu()
                assert budget.used_bytes <= total_limit
                engine_a = residency_engine(offloader_a)
                assert engine_a.pool is not None and engine_a.pool.used_bytes == 3 * unit_bytes

                out_b1 = active_b(x).cpu()
                assert budget.used_bytes <= total_limit
                assert engine_a.pool.used_bytes < 3 * unit_bytes

                out_a2 = model_a(x).cpu()
                assert budget.used_bytes <= total_limit
        torch.testing.assert_close(out_a1, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(out_a2, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(out_b1, expected_b, rtol=0, atol=0)
    finally:
        if offloader_a.active_device is not None:
            offloader_a.deactivate()
    assert budget.used_bytes == 0


@CUDA
def test_vram_budget_activation_validation_and_static_cap_no_over_evict() -> None:
    torch.manual_seed(0)
    model_a = _ResidentAndBlocksModel(num_blocks=3, width=32)
    x = torch.randn(2, 32, device="cuda")
    with torch.inference_mode():
        expected_a = model_a.to("cuda")(x).cpu()
        model_a.cpu()

    unit_bytes = 32 * 32 * 4
    offloader_a = ModelOffloader.from_module(model_a, OffloadConfig(offload_paths=("blocks",)))
    with pytest.raises(TypeError, match="VramBudget"):
        offloader_a.activate("cpu", vram_budget=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DeviceTransferContext"):
        offloader_a.activate("cpu", transfer_context=object())  # type: ignore[arg-type]
    cuda_ctx = DeviceTransferContext("cuda")
    try:
        with pytest.raises(ValueError, match="does not match activation device"):
            offloader_a.activate("cpu", transfer_context=cuda_ctx)
        assert cuda_ctx.borrower_count == 0
    finally:
        cuda_ctx.close()

    peer_bytes = [3 * unit_bytes]
    peer_evicts = 0

    def peer_evict(ask: int) -> int:
        nonlocal peer_evicts
        peer_evicts += 1
        freed = min(ask, peer_bytes[0])
        peer_bytes[0] -= freed
        return freed

    # Tight static cap with zero excess shared headroom must still reuse own idle slots without evicting peers.
    wide_budget = VramBudget(2 * unit_bytes + 3 * unit_bytes + 2 * unit_bytes)
    peer_handle = wide_budget.register_tenant(
        "lower_retention_peer",
        priority=20,
        evict_cb=peer_evict,
        usage_fn=lambda: peer_bytes[0],
    )
    with activated_model(
        offloader_a,
        "cuda",
        vram_budget=wide_budget,
        vram_budget_bytes=2 * unit_bytes,
    ), torch.inference_mode():
        torch.testing.assert_close(model_a(x).cpu(), expected_a, rtol=0, atol=0)
        torch.testing.assert_close(model_a(x).cpu(), expected_a, rtol=0, atol=0)
    assert peer_evicts == 0
    assert peer_bytes[0] == 3 * unit_bytes
    peer_handle.unregister()


@CUDA
def test_vram_budget_concurrent_deactivate_and_set_limit_none_safety() -> None:
    torch.manual_seed(0)
    model = _BlockModel(num_blocks=3, width=32)
    offloader = ModelOffloader.from_module(model, OffloadConfig(offload_paths=("blocks",)))
    x = torch.randn(2, 32, device="cuda")
    budget = VramBudget(3 * 32 * 32 * 4)
    errors: list[BaseException] = []
    stop = threading.Event()

    def hammer() -> None:
        try:
            while not stop.is_set():
                _ = budget.used_bytes
                budget.set_limit(None, evict=False)
                budget.set_limit(3 * 32 * 32 * 4, evict=False)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    hammer_thread = threading.Thread(target=hammer, daemon=True)
    hammer_thread.start()
    try:
        for _ in range(15):
            offloader.activate("cuda", vram_budget=budget)
            with torch.inference_mode():
                _ = model(x)
            offloader.deactivate()
    finally:
        stop.set()
        hammer_thread.join(timeout=5.0)
        if offloader.active_device is not None:
            offloader.deactivate()
    assert not errors
    assert budget.used_bytes == 0


@CUDA
def test_no_record_trace_isolates_mtp_draft_history() -> None:
    torch.manual_seed(0)
    model = _BlockModel(num_blocks=4, width=32)
    offloader = ModelOffloader.from_module(
        model,
        OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=0),
    )
    x = torch.randn(2, 32, device="cuda")
    try:
        with activated_model(offloader, "cuda"), torch.inference_mode():
            engine = residency_engine(offloader)
            assert engine.is_recording_trace
            for idx in (0, 1, 2, 0, 1, 2):
                x = model.blocks[idx](x)
            assert engine._predictor is not None
            hist_before = list(engine._predictor._history)
            assert hist_before == [0, 1, 2, 0, 1, 2]
            assert engine._predictor.next_after(0) == 1

            with offloader.no_record_trace():
                assert not engine.is_recording_trace
                model.blocks[0](x)
                with offloader.no_record_trace():
                    assert not engine.is_recording_trace
                    model.blocks[3](x)
                assert not engine.is_recording_trace
                model.blocks[2](x)

            assert engine.is_recording_trace
            assert list(engine._predictor._history) == hist_before
            assert engine._predictor.next_after(0) == 1
    finally:
        if offloader.active_device is not None:
            offloader.deactivate()


# ---------------------------------------------------------------------------
# Uint8DualArena, TypedSidePool, StructuredSidePool, KvMetadataPool
# ---------------------------------------------------------------------------


def test_dual_arena_low_coalescing_blocking_slices_and_high_descending() -> None:
    budget = VramBudget(4096 + 256)
    evict_calls: list[int] = []
    budget.register_tenant(
        "peer",
        priority=10,
        evict_cb=lambda ask: (evict_calls.append(ask), 0)[1],
        usage_fn=lambda: 0,
    )
    with pytest.raises(ArenaCapacityError):
        Uint8DualArena(1024, device="cpu", slot_bytes=2048, alignment=256, budget=budget)
    assert budget.reserved_bytes == 0
    assert evict_calls == []
    with pytest.raises(ValueError, match="power of two"):
        Uint8DualArena(4096, device="cpu", alignment=48)

    arena = Uint8DualArena(4096, device="cpu", slot_bytes=1024, alignment=256, budget=budget)
    assert budget.reserved_bytes == arena.allocated_bytes
    assert arena.slot_layout is not None
    assert arena.slot_layout.max_slots == 4
    assert arena.available_high_slots == 4

    s3 = arena.alloc_high_slot()
    s2 = arena.alloc_high_slot()
    assert (s3, s2) == (3, 2)
    assert arena.high_frontier == 2048
    assert arena.high_slot_tensor(s3).data_ptr() % 256 == 0

    a = arena.alloc_low(100, tag="w0")  # [0, 256)
    b = arena.alloc_low(300, tag="w1")  # [256, 768)
    assert a.tensor.data_ptr() % 256 == 0
    assert b.tensor.data_ptr() % 256 == 0
    assert (a.offset, a.aligned_size) == (0, 256)
    assert (b.offset, b.aligned_size) == (256, 512)
    assert arena.low_frontier == 768

    with pytest.raises(ArenaCapacityError):
        arena.alloc_low(1281, tag="too_big")

    arena.free_low(a)
    assert arena.low_frontier == 768
    arena.free_low(b)
    assert arena.low_frontier == 0

    arena.free_high_slot(s3)
    assert arena.alloc_high_slot() == 3

    low0 = arena.alloc_low(900, tag="moe0")
    assert arena.overlapping_low_slices(0) == (low0,)
    assert arena.blocking_low_slices(0) == (low0,)
    assert arena.alloc_high_slot() == 1
    with pytest.raises(ArenaCapacityError):
        arena.alloc_high_slot()

    arena.close()
    assert budget.reserved_bytes == 0
    assert arena.available_high_slots == 0
    assert arena.free_gap_bytes == 0
    assert not arena.is_open


def test_dual_arena_blocking_ranges_and_slice_validation() -> None:
    small_arena = Uint8DualArena(1024, device="cpu", slot_bytes=256, alignment=256)
    s_low_a = small_arena.alloc_low(100, tag="h")  # [0, 256)
    s_low_b = small_arena.alloc_low(100, tag="p")  # [256, 512)
    assert small_arena.overlapping_low_slices_in_range(100, 100) == ()
    with pytest.raises(ValueError, match=">="):
        small_arena.overlapping_low_slices_in_range(200, 100)
    with pytest.raises(ValueError, match="non-negative integer"):
        small_arena.overlapping_low_slices_in_range(False, 10)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-negative"):
        s_low_a.as_uint8((-1, 4))
    assert tuple(s_low_a.as_uint8(d for d in (2, 50)).shape) == (2, 50)
    assert tuple(s_low_a.as_uint8(()).shape) == ()

    small_arena.free_low(s_low_a)
    with pytest.raises(RuntimeError, match="released"):
        _ = s_low_a.tensor
    assert not s_low_a.is_active
    assert small_arena.overlapping_low_slices(0) == ()
    assert small_arena.blocking_low_slices(0) == (s_low_b,)
    s_low_b.release()
    assert small_arena.blocking_low_slices(0) == ()
    small_arena.close()


def test_slot_schema_validation_and_alignment() -> None:
    schema = SlotSchema(
        {"up": 300, "down": (4, 64)},
        {"scale": (torch.float32, (3,))},
        field_alignment=256,
        side_alignment=16,
    )
    assert schema.arena_field_names == ("up", "down")
    assert schema.side_field_names == ("scale",)
    assert isinstance(schema.arena_field("up"), ArenaFieldSpec)
    assert isinstance(schema.side_field("scale"), SideFieldSpec)
    assert schema.side_field("scale").nbytes == 12
    assert schema.arena_field("up").offset == 0
    assert schema.arena_field("up").aligned_bytes == 512
    assert schema.arena_field("down").offset == 512
    assert schema.arena_field("down").aligned_bytes == 256
    assert schema.drawer_bytes == 768
    assert schema.side_bytes_per_slot == 16  # 3 * fp32 (12B) aligned up to 16B

    no_side = SlotSchema({"up": 64}, side_fields=None)
    assert no_side.side_fields == ()
    assert no_side.side_bytes_per_slot == 0
    with pytest.raises(KeyError):
        schema.arena_field("missing")
    with pytest.raises(KeyError):
        schema.side_field("missing")
    with pytest.raises(TypeError, match="mapping or None"):
        SlotSchema({"up": 64}, side_fields=[("scale", (torch.float32, (1,)))])  # type: ignore[arg-type]

    for bad_fn, match in (
        (lambda: SlotSchema({}), "at least one"),
        (lambda: SlotSchema({"bad": True}), "bool"),
        (lambda: SlotSchema({"bad": ()}), "non-empty"),
        (lambda: SlotSchema({"a": 64}, {"a": (torch.float32, (2,))}), "Duplicate"),
        (lambda: SlotSchema({"a": 64}, {"s": ("f32", (2,))}), "torch.dtype"),  # type: ignore[arg-type]
        (lambda: SlotSchema({"a": 64}, field_alignment=8), "power of two >= 16"),
        (lambda: SlotSchema({"a": 64}, field_alignment=48), "power of two >= 16"),
        (lambda: SlotSchema({"a": 64}, side_alignment=True), "power of two >= 16"),
    ):
        with pytest.raises(ValueError, match=match):
            bad_fn()


def test_symmetric_low_slots_and_views() -> None:
    with pytest.raises(ValueError, match="requires low_slot_bytes"):
        Uint8DualArena(2048, device="cpu", low_max_slots=2)
    with pytest.raises(ArenaCapacityError):
        Uint8DualArena(2048, device="cpu", low_slot_bytes=768, low_max_slots=3, alignment=256)

    # Reject fixed low slot configuration while variable alloc_low slices are active; allow once freed.
    with Uint8DualArena(2048, device="cpu", alignment=256) as mixed_arena:
        for fn in (
            mixed_arena.alloc_low_slot,
            lambda: mixed_arena.free_low_slot(0),
            lambda: mixed_arena.low_slot_tensor(0),
            lambda: mixed_arena.low_pool_view((64,)),
        ):
            with pytest.raises(RuntimeError, match="not configured"):
                fn()
        var_slc = mixed_arena.alloc_low(128)
        with pytest.raises(RuntimeError, match="while low allocations are active"):
            mixed_arena.configure_low_slots(256)
        var_slc.release()
        mixed_arena.configure_low_slots(256, max_slots=4)
        assert mixed_arena.low_slot_layout is not None
        assert mixed_arena.low_slot_layout.max_slots == 4

    with Uint8DualArena(
        2048,
        device="cpu",
        slot_bytes=512,
        low_slot_bytes=768,
        alignment=256,
    ) as arena:
        assert isinstance(arena.low_slot_layout, LowSlotLayout)
        assert arena.low_slot_layout.max_slots == 2
        assert arena.available_low_slots == 2
        with pytest.raises(RuntimeError, match="fixed low slots"):
            arena.alloc_low(64)

        low0 = arena.alloc_low_slot()
        low1 = arena.alloc_low_slot(max_slot_id=1)
        assert (low0, low1) == (0, 1)
        assert arena.active_low_slots == (0, 1)
        assert arena.low_frontier == 1536
        assert arena.active_low_bytes == 1536
        assert arena.raw_uint8_view().numel() == 2048

        # Freeing top slot shrinks frontier, double free is idempotent, invalid IDs raise.
        arena.free_low_slot(low1)
        arena.free_low_slot(low1)
        assert arena.low_frontier == 768
        assert arena.alloc_low_slot() == 1
        arena.free_low_slot(low0)
        assert arena.alloc_low_slot(max_slot_id=0, allow_reclaim=False) == 0
        with pytest.raises(ArenaCapacityError):
            arena.alloc_low_slot(max_slot_id=0, allow_reclaim=False)
        with pytest.raises(TypeError, match="int"):
            arena.free_low_slot(True)
        with pytest.raises(IndexError):
            arena.free_low_slot(99)

        down_view = arena.low_pool_view((4, 64), field_offset=512)
        assert tuple(down_view.shape) == (2, 4, 64)
        assert down_view.stride(0) == 768
        down_view[low1].fill_(42)
        assert int(arena.low_slot_tensor(low1)[512:].min().item()) == 42
        with pytest.raises(ValueError, match="multiple of 16"):
            arena.low_pool_view((4,), field_offset=8)


def test_peer_frontier_reclaimer_callbacks() -> None:
    with Uint8DualArena(
        2048,
        device="cpu",
        slot_bytes=512,
        low_slot_bytes=768,
        alignment=256,
    ) as arena:
        assert arena.alloc_low_slot() == 0
        low1 = arena.alloc_low_slot()
        assert low1 == 1
        assert arena.alloc_high_slot() == 3
        with pytest.raises(ArenaCapacityError):
            arena.alloc_high_slot()

        low_calls: list[int] = []

        def low_reclaim(target_low: int) -> None:
            low_calls.append(target_low)
            assert arena.high_frontier == 1536  # arena._lock is NOT held during callback
            arena.free_low_slot(low1)

        with pytest.raises(TypeError, match="callable"):
            arena.set_low_reclaimer(123)  # type: ignore[arg-type]
        arena.set_low_reclaimer(low_reclaim)
        with pytest.raises(RuntimeError, match="already registered"):
            arena.set_low_reclaimer(lambda _t: None)

        # Wrong callback identity does not clear the active reclaimer.
        arena.clear_low_reclaimer(lambda _t: None)
        hi2 = arena.alloc_high_slot()
        assert hi2 == 2
        assert low_calls == [1024]
        assert arena.low_frontier == 768

        high_calls: list[int] = []

        def high_reclaim(needed_high: int) -> None:
            high_calls.append(needed_high)
            arena.free_high_slot(hi2)

        arena.set_high_reclaimer(high_reclaim)
        assert arena.alloc_low_slot() == 1
        assert high_calls == [1536]
        assert arena.high_frontier == 1536

        stalled_calls: list[int] = []
        arena.clear_high_reclaimer(high_reclaim)
        arena.free_low_slot(1)
        assert arena.alloc_high_slot() == 2
        arena.set_high_reclaimer(stalled_calls.append)
        with pytest.raises(ArenaCapacityError):
            arena.alloc_low_slot()
        assert stalled_calls == [1536]

        # Stalled low_reclaimer stops cleanly; set_low_reclaimer(None) clears unconditionally.
        arena.set_low_reclaimer(None)
        stalled_low: list[int] = []
        arena.set_low_reclaimer(stalled_low.append)
        with pytest.raises(ArenaCapacityError):
            arena.alloc_high_slot()
        assert stalled_low == [512]
        arena.set_low_reclaimer(None)


def test_peer_frontier_reclaimer_closed_and_variable_slice_reclaim() -> None:
    arena = Uint8DualArena(1024, device="cpu", slot_bytes=512, low_slot_bytes=512, alignment=256)
    arena.close()
    with pytest.raises(RuntimeError, match="closed"):
        arena.set_low_reclaimer(lambda _t: None)
    with pytest.raises(RuntimeError, match="closed"):
        arena.set_high_reclaimer(lambda _t: None)

    # Variable alloc_low slices also work with low_reclaimer when alloc_high_slot collides.
    with Uint8DualArena(1024, device="cpu", slot_bytes=512, alignment=256) as var_arena:
        s0 = var_arena.alloc_low(512)
        var_arena.set_low_reclaimer(lambda _target: s0.release())
        assert var_arena.alloc_high_slot() == 1
        assert var_arena.alloc_high_slot() == 0


def test_interleaved_kv_views_and_structured_side_pools() -> None:
    h_kv, d, c_tok = 4, 64, 1024
    slot_bytes = 2 * h_kv * c_tok * d
    for num_slots in (1, 3):
        arena = Uint8DualArena(
            num_slots * slot_bytes, device="cpu", slot_bytes=slot_bytes, alignment=256
        )
        try:
            k, v = arena.interleaved_kv_views(num_kv_heads=h_kv, head_dim=d, chunk_tokens=c_tok)
            assert k.dtype is torch.uint8
            assert v.dtype is torch.uint8
            assert tuple(k.shape) == (num_slots, h_kv, c_tok, d)
            assert tuple(v.shape) == (num_slots, h_kv, 16, d, 64)
            assert arena.slot_layout is not None
            assert k.stride(0) == arena.slot_layout.slot_stride > 0
            assert v.stride(0) == arena.slot_layout.slot_stride > 0
            assert k.stride(0) % 16 == 0
            assert k.data_ptr() % 16 == 0
            assert v.data_ptr() % 16 == 0
            assert k.storage_offset() % 16 == 0
            assert v.storage_offset() % 16 == 0
            slot = arena.alloc_high_slot()
            k[slot, 0, 0, :4] = torch.tensor([11, 22, 33, 44], dtype=torch.uint8)
            assert arena.high_slot_tensor(slot)[:4].tolist() == [11, 22, 33, 44]
        finally:
            arena.close()

    with pytest.raises(ValueError, match="multiple of 16"):
        bad = Uint8DualArena(8192, device="cpu", slot_bytes=2 * 1 * 64 * 1, alignment=256)
        try:
            bad.interleaved_kv_views(num_kv_heads=1, head_dim=1, chunk_tokens=8, tail_subblock=4)
        finally:
            bad.close()


@CUDA
def test_dual_arena_and_side_pools_cuda_custom_alignment_and_inference_mode() -> None:
    device = torch.device("cuda")
    # Allocate an odd 512B CUDA block first to stress caching-allocator base offsets.
    spacer = torch.empty((512,), dtype=torch.uint8, device=device)
    try:
        with torch.inference_mode():
            arena = Uint8DualArena(
                4096, device=device, slot_bytes=1024, alignment=1024
            )
            pool = TypedSidePool(
                2, (4,), dtype=torch.float32, device=device, alignment_bytes=1024
            )
        try:
            slc = arena.alloc_low(64)
            slot_id = arena.alloc_high_slot()
            slot_t = arena.high_slot_tensor(slot_id)
            assert slc.tensor.data_ptr() % 1024 == 0
            assert slot_t.data_ptr() % 1024 == 0
            assert pool.tensor.data_ptr() % 1024 == 0
            assert not slc.tensor.is_inference()
            assert not pool.tensor.is_inference()
            # Writable from normal-mode background thread despite construction inside inference_mode().
            write_err: list[BaseException] = []

            def worker_write() -> None:
                try:
                    slc.tensor.fill_(9)
                    pool.tensor.fill_(3.5)
                except BaseException as exc:  # pragma: no cover
                    write_err.append(exc)

            t = threading.Thread(target=worker_write)
            t.start()
            t.join(timeout=5.0)
            torch.cuda.synchronize(device)
            assert not write_err
            assert int(slc.tensor[0].item()) == 9
            assert float(pool.slot(0)[0].item()) == 3.5
        finally:
            pool.close()
            arena.close()
    finally:
        del spacer

def test_structured_and_kv_metadata_side_pools() -> None:
    # Generic StructuredSidePool with mixed heterogeneous field shapes.
    side_budget = VramBudget(1024 * 1024)
    structured = StructuredSidePool(
        4,
        {"alpha": (2, 3), "beta": (5,)},
        dtype=torch.float32,
        device="cpu",
        budget=side_budget,
    )
    assert structured.field_names == ("alpha", "beta")
    assert side_budget.reserved_bytes == structured.nbytes
    for fname in structured.field_names:
        t = structured[fname]
        assert t.stride(0) == structured.stride0 > 0
        assert (t.stride(0) * 4) % 16 == 0
        assert (t.storage_offset() * 4) % 16 == 0
        assert t[0].is_contiguous()
    structured["alpha"].fill_(5.0)
    structured["beta"].fill_(9.0)
    assert torch.all(structured.field("alpha") == 5.0)
    assert torch.all(structured.field("beta") == 9.0)
    structured.close()
    assert side_budget.reserved_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        _ = structured["alpha"]

    # KvMetadataPool wrapper over StructuredSidePool.
    meta = TypedSidePool.for_kv_metadata(
        4,
        key_shape=(3, 1),
        value_shape=(3, 5),
        dtype=torch.float32,
        device="cpu",
        budget=side_budget,
    )
    assert isinstance(meta, KvMetadataPool)
    assert side_budget.reserved_bytes == meta.nbytes
    for tensor in (meta.key_scale, meta.key_center, meta.value_scale, meta.value_mean):
        assert tensor.dtype is torch.float32
        assert tensor.stride(0) == meta.stride0 > 0
        assert (tensor.stride(0) * 4) % 16 == 0
        assert (tensor.storage_offset() * 4) % 16 == 0
        assert tensor[0].is_contiguous()
    meta.close()
    assert side_budget.reserved_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        _ = meta.key_scale


@CUDA
def test_dual_arena_single_dtype_torch_compile_fullgraph(
    request: pytest.FixtureRequest,
) -> None:
    torch.compiler.reset()
    request.addfinalizer(torch.compiler.reset)
    device = torch.device("cuda")
    h_kv, d, c_tok = 2, 64, 1024
    slot_bytes = 2 * h_kv * c_tok * d
    total_bytes = 4 * slot_bytes

    arena = Uint8DualArena(total_bytes, device=device, slot_bytes=slot_bytes, alignment=256)
    meta = KvMetadataPool(
        arena.slot_layout.max_slots if arena.slot_layout is not None else 4,
        key_shape=(h_kv, d),
        value_shape=(h_kv, c_tok),
        dtype=torch.float32,
        device=device,
    )
    try:
        low_slice = arena.alloc_low(64, tag="moe_weight")
        moe_w = low_slice.as_uint8((8, 8))
        moe_w.fill_(3)

        slot_id = arena.alloc_high_slot()
        k_pool, v_pool = arena.interleaved_kv_views(
            num_kv_heads=h_kv,
            head_dim=d,
            chunk_tokens=c_tok,
        )
        k_pool[slot_id].zero_()
        v_pool[slot_id].zero_()
        meta.key_scale[slot_id].zero_()

        def compiled_step(
            w: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            k_scale: torch.Tensor,
        ) -> torch.Tensor:
            k[slot_id, :, 0, :].fill_(5)
            v[slot_id, :, 0, :, 0].fill_(7)
            k_scale[slot_id].fill_(2.5)
            return w.to(torch.float32).sum() + k_scale[slot_id].sum()

        fn = torch.compile(compiled_step, fullgraph=True)
        out1 = fn(moe_w, k_pool, v_pool, meta.key_scale)
        out2 = fn(moe_w, k_pool, v_pool, meta.key_scale)
        torch.cuda.synchronize(device)

        expected = torch.tensor(64 * 3.0 + h_kv * d * 2.5, device=device)
        torch.testing.assert_close(out1, expected)
        torch.testing.assert_close(out2, expected)
        assert int(k_pool[slot_id, :, 0, :].min().item()) == 5
        assert int(v_pool[slot_id, :, 0, :, 0].min().item()) == 7
        assert int(arena.high_slot_tensor(slot_id)[0].item()) == 5
    finally:
        meta.close()
        arena.close()


# ---------------------------------------------------------------------------
# BufferSpec, GpuAllocator, ArenaBindableTensorAdapter, SliceableTensorAdapter
# ---------------------------------------------------------------------------


def test_buffer_spec_validation_and_slice_host() -> None:
    default_spec = BufferSpec(name="f", shape=(4, 8), dtype=torch.float32)
    assert default_spec.arena_eligible is False
    spec = BufferSpec(
        name="w", shape=(4, 8), dtype=torch.uint8, alignment=256, arena_eligible=True
    )
    assert spec.nbytes == 32
    assert spec.arena_eligible is True
    for bad_kwargs, match in (
        ({"name": "", "shape": (4,), "dtype": torch.uint8}, "non-empty"),
        ({"name": "bad_dim", "shape": (4, -1), "dtype": torch.uint8}, "non-negative"),
        ({"name": "bad_align", "shape": (4,), "dtype": torch.uint8, "alignment": 0}, "power-of-two"),
        ({"name": "bad_align_np2", "shape": (4,), "dtype": torch.uint8, "alignment": 3}, "power-of-two"),
        ({"name": "bad_list", "shape": [4], "dtype": torch.uint8}, "tuple"),
        ({"name": "bad_dtype", "shape": (4,), "dtype": "uint8"}, "torch.dtype"),
        ({"name": "f32_arena", "shape": (4,), "dtype": torch.float32, "arena_eligible": True}, "Only torch.uint8"),
        ({"name": "empty_arena", "shape": (0,), "dtype": torch.uint8, "arena_eligible": True}, "Zero-byte"),
    ):
        with pytest.raises(ValueError, match=match):
            BufferSpec(**bad_kwargs)  # type: ignore[arg-type]

    stacked = torch.arange(3 * 4 * 8, dtype=torch.uint8).view(3, 4, 8)
    adapter = RegularAdapter()
    assert isinstance(adapter, ArenaBindableTensorAdapter)
    assert isinstance(adapter, SliceableTensorAdapter)

    host_stacked = adapter.capture_host(stacked)
    expert_1 = adapter.slice_host(host_stacked, 1, dim=0)
    expert_last = adapter.slice_host(host_stacked, -1, dim=0)
    dim1_slice = adapter.slice_host(host_stacked, 2, dim=1)
    assert expert_1.data.data_ptr() == host_stacked.data[1].data_ptr()
    torch.testing.assert_close(expert_last.data, stacked[-1], rtol=0, atol=0)
    torch.testing.assert_close(dim1_slice.data, stacked[:, 2], rtol=0, atol=0)
    with pytest.raises(IndexError):
        adapter.slice_host(host_stacked, 3, dim=0)
    with pytest.raises(IndexError):
        adapter.slice_host(host_stacked, 0, dim=3)
    with pytest.raises(TypeError):
        adapter.slice_host(host_stacked, True, dim=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scalar"):
        adapter.slice_host(adapter.capture_host(torch.tensor(1.0)), 0)

    specs = adapter.gpu_specs(expert_1)
    assert specs == (
        BufferSpec(name="data", shape=(4, 8), dtype=torch.uint8, arena_eligible=True),
    )
    f32_host = adapter.capture_host(torch.randn(4, 8, dtype=torch.float32))
    assert adapter.gpu_specs(f32_host)[0].arena_eligible is False
    empty_u8_host = adapter.capture_host(torch.empty((0, 8), dtype=torch.uint8))
    assert adapter.gpu_specs(empty_u8_host)[0].arena_eligible is False


def test_regular_adapter_bind_gpu_arena_and_invariant_checks() -> None:
    stacked = torch.arange(3 * 4 * 8, dtype=torch.uint8).view(3, 4, 8)
    adapter = RegularAdapter()
    expert_1 = adapter.slice_host(adapter.capture_host(stacked), 1, dim=0)
    f32_host = adapter.capture_host(torch.randn(4, 8, dtype=torch.float32))
    arena = Uint8DualArena(4096, device="cpu", alignment=256)

    @dataclass
    class _ArenaAllocator:
        arena_ref: Uint8DualArena

        def allocate(self, s: BufferSpec, *, device: torch.device) -> torch.Tensor:
            assert s.arena_eligible and device.type == "cpu"
            return self.arena_ref.alloc_low(s.nbytes, tag=s.name).as_uint8(s.shape)

    allocator: GpuAllocator = _ArenaAllocator(arena)
    gpu_state = adapter.bind_gpu(expert_1, allocator, torch.device("cpu"))
    adapter.copy_to_gpu(expert_1, gpu_state, non_blocking=False)
    torch.testing.assert_close(gpu_state.data, stacked[1], rtol=0, atol=0)
    assert arena.low_frontier == 256

    class _FnAllocator:
        def __init__(self, fn) -> None:
            self._fn = fn

        def allocate(self, s: BufferSpec, *, device: torch.device) -> torch.Tensor:
            return self._fn(s, device)

    with pytest.raises(ValueError, match="incompatible buffer"):
        adapter.bind_gpu(expert_1, _FnAllocator(lambda _s, d: torch.empty((1,), device=d)), torch.device("cpu"))
    with pytest.raises(ValueError, match="non-contiguous"):
        adapter.bind_gpu(
            expert_1,
            _FnAllocator(lambda s, d: torch.empty(tuple(reversed(s.shape)), dtype=s.dtype, device=d).t()),
            torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="non-contiguous"):
        adapter.bind_gpu(
            expert_1,
            _FnAllocator(lambda s, d: torch.empty((1, s.shape[1]), dtype=s.dtype, device=d).expand(s.shape)),
            torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="requires_grad"):
        adapter.bind_gpu(
            f32_host,
            _FnAllocator(lambda s, d: torch.empty(s.shape, dtype=s.dtype, device=d, requires_grad=True)),
            torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="plain strided"):
        sparse = torch.sparse_coo_tensor(
            [[0], [0]],
            torch.ones(1, dtype=torch.uint8),
            size=(4, 8),
            dtype=torch.uint8,
        )
        adapter.bind_gpu(expert_1, _FnAllocator(lambda _s, _d: sparse), torch.device("cpu"))
    with pytest.raises(ValueError, match="aliasing captured host"):
        adapter.bind_gpu(expert_1, _FnAllocator(lambda _s, _d: expert_1.data), torch.device("cpu"))
    with torch.inference_mode():
        inf_t = torch.empty((4, 8), dtype=torch.uint8)
    with pytest.raises(ValueError, match="inference-mode"):
        adapter.bind_gpu(expert_1, _FnAllocator(lambda _s, _d: inf_t), torch.device("cpu"))
    arena.close()


@CUDA
def test_regular_adapter_bind_gpu_rejects_misaligned_and_wrong_index_cuda_buffer() -> None:
    adapter = RegularAdapter()
    host = adapter.capture_host(torch.zeros((4, 8), dtype=torch.uint8))

    class _MisalignedAllocator:
        def allocate(self, s: BufferSpec, *, device: torch.device) -> torch.Tensor:
            raw = torch.empty((s.nbytes + 16,), dtype=torch.uint8, device=device)
            return raw[1 : 1 + s.nbytes].view(s.shape)

    with pytest.raises(ValueError, match="unaligned CUDA buffer"):
        adapter.bind_gpu(host, _MisalignedAllocator(), torch.device("cuda"))

    class _WrongIndexAllocator:
        def allocate(self, s: BufferSpec, *, device: torch.device) -> torch.Tensor:
            assert device == torch.device("cuda", 1)
            return torch.empty(s.shape, dtype=s.dtype, device="cuda:0")

    with pytest.raises(ValueError, match="incompatible buffer"):
        adapter.bind_gpu(host, _WrongIndexAllocator(), torch.device("cuda", 1))


def test_stream_lease_produce_and_poll_d2h(monkeypatch: pytest.MonkeyPatch) -> None:
    class _PollEvent:
        def __init__(self) -> None:
            self.done = True

        def query(self) -> bool:
            return self.done

        def synchronize(self) -> None:
            self.done = True

    class _PollStream:
        def __init__(self) -> None:
            self.events: list[_PollEvent] = []
            self.waited_events: list[_PollEvent] = []
            self.waited_streams: list[object] = []

        def wait_event(self, event: _PollEvent) -> None:
            self.waited_events.append(event)

        def wait_stream(self, stream: object) -> None:
            self.waited_streams.append(stream)

        def record_event(self) -> _PollEvent:
            ev = _PollEvent()
            self.events.append(ev)
            return ev

        def synchronize(self) -> None:
            for ev in self.events:
                ev.done = True

    alloc_stream = _PollStream()
    compute_stream = _PollStream()
    d2h_stream = _PollStream()
    lease = StreamLease(cast(torch.cuda.Stream, alloc_stream))
    monkeypatch.setattr(torch.cuda, "stream", lambda _s: contextlib.nullcontext())

    with pytest.raises(RuntimeError, match="boom"):
        with lease.produce(cast(torch.cuda.Stream, compute_stream)):
            raise RuntimeError("boom")
    assert not lease.is_dirty
    assert not lease.is_staged

    with lease.produce(cast(torch.cuda.Stream, compute_stream)):
        assert not lease.is_staged
    assert lease.is_dirty
    assert lease.is_staged
    assert compute_stream.waited_streams == [alloc_stream]

    lease.spill_to_host(lambda: None, cast(torch.cuda.Stream, d2h_stream))
    d2h_ev = d2h_stream.events[-1]
    d2h_ev.done = False
    assert lease.d2h_in_flight
    assert not lease.poll_d2h()
    assert lease.is_dirty

    d2h_ev.done = True
    assert lease.poll_d2h()
    assert not lease.d2h_in_flight
    assert not lease.is_dirty
    assert lease.poll_d2h()


@CUDA
def test_stream_lease_produce_and_poll_d2h_cuda() -> None:
    device = torch.device("cuda")
    compute_stream = torch.cuda.Stream(device=device)
    d2h_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(compute_stream):
        gpu_buf = torch.empty((1024,), dtype=torch.float32, device=device)
    host_buf = torch.empty((1024,), dtype=torch.float32, pin_memory=True)
    lease = StreamLease(compute_stream)

    with lease.produce(compute_stream):
        gpu_buf.fill_(4.25)
    assert lease.is_dirty
    lease.spill_to_host(lambda: host_buf.copy_(gpu_buf, non_blocking=True), d2h_stream)
    deadline = time.monotonic() + 10.0
    while not lease.poll_d2h():
        assert time.monotonic() < deadline
        d2h_stream.query()
        time.sleep(0.001)
    assert not lease.is_dirty
    torch.testing.assert_close(host_buf, torch.full((1024,), 4.25))


def test_symmetric_high_slots_limits_views_and_release_owner() -> None:
    with pytest.raises(ValueError, match="requires slot_bytes"):
        Uint8DualArena(2048, device="cpu", high_max_slots=2)
    with pytest.raises(ArenaCapacityError):
        Uint8DualArena(2048, device="cpu", slot_bytes=768, high_max_slots=3, alignment=256)

    arena = Uint8DualArena(2048, device="cpu", alignment=256)
    try:
        arena.configure_high_slots(512, max_slots=3)
        assert isinstance(arena.slot_layout, HighSlotLayout)
        assert arena.slot_layout.max_slots == 3
        s2 = arena.alloc_high_slot(min_slot_id=2)
        assert s2 == 2
        with pytest.raises(ArenaCapacityError):
            arena.alloc_high_slot(min_slot_id=2, allow_reclaim=False)
        arena.free_high_slot(s2)

        cb = lambda _need: None  # noqa: E731
        arena.set_high_reclaimer(cb)
        arena._release_high_owner(cb, reset_expected=arena.slot_layout)
        assert arena.slot_layout is None
    finally:
        arena.close()

    with Uint8DualArena(2048, device="cpu", slot_bytes=512, high_max_slots=3, alignment=256) as a2:
        view = a2.high_pool_view((128,), field_offset=256)
        assert tuple(view.shape) == (3, 128)
        assert view.stride(0) == 512
        sid = a2.alloc_high_slot()
        view[sid].fill_(7)
        assert int(a2.high_slot_tensor(sid)[256:384].min().item()) == 7
        with pytest.raises(ValueError, match="multiple of 16"):
            a2.high_pool_view((16,), field_offset=8)


class _SimpleSlabBackend:
    def __init__(self) -> None:
        self.registered: dict[int, int] = {}

    def register(self, pointer: int, size: int) -> bool:
        self.registered[pointer] = size
        return True

    def unregister(self, pointer: int) -> None:
        self.registered.pop(pointer, None)


def test_pin_manager_allocate_host_slab_validation_and_tier2a() -> None:
    page = mmap.PAGESIZE
    pm = PinManager(2 * page, max_host_bytes=4 * page, backend=_SimpleSlabBackend())
    try:
        for kwargs, match in (
            ({"nbytes": 0}, "positive integer"),
            ({"nbytes": True}, "positive integer"),
            ({"nbytes": 64, "alignment": 8}, "power of two >= 16"),
            ({"nbytes": 64, "batch_hint": 0}, "positive integer"),
        ):
            with pytest.raises(ValueError, match=match):
                pm.allocate_host_slab(**kwargs)  # type: ignore[arg-type]

        a1 = pm.allocate_host_slab(page, pinned=True)
        a2 = pm.allocate_host_slab(page, pinned=True)
        assert a1 is not None and a2 is not None
        assert a1.is_pinned and a2.is_pinned
        assert pm.allocate_host_slab(page, pinned=True) is None

        a1.release()
        a3 = pm.allocate_host_slab(page, pinned=True)
        assert a3 is not None
        a2.release()
        a3.release()
    finally:
        pm.clear()


def test_pin_manager_allocate_host_slab_tier2b_ceiling_holes_and_clear() -> None:
    page = mmap.PAGESIZE
    backend = _SimpleSlabBackend()
    pm = PinManager(2 * page, max_host_bytes=4 * page, backend=backend)
    try:
        # Hold two Tier 2a slabs, then fill remaining Tier 2b capacity with two pageable slabs.
        p1 = pm.allocate_host_slab(page, pinned=True)
        p2 = pm.allocate_host_slab(page, pinned=True)
        b1 = pm.allocate_host_slab(page, pinned=False)
        b2 = pm.allocate_host_slab(page, pinned=False)
        assert p1 is not None and p2 is not None and b1 is not None and b2 is not None
        assert not b1.is_pinned and not b2.is_pinned

        # Full-live Tier 2b returns None cleanly without looping; freeing one reuses its hole.
        assert pm.allocate_host_slab(page, pinned=False) is None
        b1.release()
        b3 = pm.allocate_host_slab(page, pinned=False)
        assert b3 is not None

        # Clear while allocations remain live keeps returned views writable until release().
        pm.clear()
        b3.as_uint8_tensor().fill_(5)
        assert int(b3.as_uint8_tensor()[0].item()) == 5
        p1.release()
        p2.release()
        b2.release()
        b3.release()
    finally:
        pm.clear()


def test_pin_manager_allocate_host_slab_evicts_idle_weights_for_budget() -> None:
    page = mmap.PAGESIZE
    buf = mmap.mmap(-1, page)
    idle: torch.Tensor | None = torch.frombuffer(buf, dtype=torch.uint8)
    lease = None
    pm = PinManager(2 * page, max_host_bytes=2 * page, backend=_SimpleSlabBackend())
    try:
        assert idle is not None
        lease = pm.acquire([idle], pin=True)
        lease.close()
        lease = None
        assert pm.stats.pinned_bytes == page
        assert pm.stats.idle_registrations == 1

        slab = pm.allocate_host_slab(2 * page, pinned=True)
        assert slab is not None
        assert pm.stats.idle_registrations == 0
        slab.release()
    finally:
        pm.clear()
        idle = None
        buf.close()


def test_stream_lease_write_guard_rejects_overlapping_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    class _GuardStream:
        def wait_event(self, _event: object) -> None:
            pass

        def wait_stream(self, _stream: object) -> None:
            pass

        def record_event(self) -> object:
            return type("_Ev", (), {"query": lambda _s: True, "synchronize": lambda _s: None})()

    monkeypatch.setattr(torch.cuda, "stream", lambda _s: contextlib.nullcontext())
    s = cast(torch.cuda.Stream, _GuardStream())
    lease = StreamLease(s)

    with lease.produce(s):
        with pytest.raises(RuntimeError, match="while a write is active"):
            with lease.produce(s):
                pass
        with pytest.raises(RuntimeError, match="while a write is active"):
            lease.acquire(s)
        with pytest.raises(RuntimeError, match="while a write is active"):
            lease.spill_to_host(lambda: None, s)
        with pytest.raises(RuntimeError, match="while a write is active"):
            lease.discard_dirty()
        with pytest.raises(RuntimeError, match="while a write is active"):
            lease.synchronize_and_reset()

    assert lease.is_dirty
    lease.acquire(s)
    lease.release(s)
    lease.discard_dirty()
    assert not lease.is_dirty


def test_dual_arena_rejects_external_reconfigure_while_pool_attached() -> None:
    schema = SlotSchema({"k": 256})
    with Uint8DualArena(4096, device="cpu", alignment=256) as arena:
        kv = KvChunkPool(arena, schema, max_slots=2)
        exp = ExpertSlabPool(arena, SlotSchema({"up": 256}), max_slots=2)
        assert arena.active_high_slots == ()
        assert arena.active_low_slots == ()
        arena.set_high_reclaimer(None)
        arena.set_low_reclaimer(None)
        with pytest.raises(RuntimeError):
            arena.configure_high_slots(schema.drawer_bytes)
        with pytest.raises(RuntimeError):
            arena.configure_low_slots(256)
        kv.close()
        exp.close()
        arena.configure_high_slots(schema.drawer_bytes, max_slots=1)
        arena.configure_low_slots(256, max_slots=1)


def test_pin_manager_allocate_host_slab_ceilings_and_alignment_eviction() -> None:
    page = mmap.PAGESIZE
    # Clamping max_pinned_bytes to max_host_bytes when max_host_bytes > 0.
    pm_clamp = PinManager(4 * page, max_host_bytes=page, backend=_SimpleSlabBackend())
    try:
        assert pm_clamp.allocate_host_slab(2 * page, pinned=True) is None
        s1 = pm_clamp.allocate_host_slab(page, pinned=True)
        assert s1 is not None
        s1.release()
    finally:
        pm_clamp.clear()

    # max_host_bytes=0 disables Tier 2b/host fallback while keeping Tier 2a pinned slabs active.
    pm_zero = PinManager(2 * page, max_host_bytes=0, backend=_SimpleSlabBackend())
    try:
        s_pin = pm_zero.allocate_host_slab(2 * page, pinned=True)
        assert s_pin is not None
        assert pm_zero.allocate_host_slab(page, pinned=False) is None
        s_pin.release()
    finally:
        pm_zero.clear()

    # Alignment pushes min_slab_size past free host_budget; impossible Tier 2b requests do not evict idle weights.
    buf = mmap.mmap(-1, 2 * page)
    idle: torch.Tensor | None = torch.frombuffer(buf, dtype=torch.uint8)
    pm_align = PinManager(2 * page, max_host_bytes=3 * page, backend=_SimpleSlabBackend())
    try:
        assert idle is not None
        lease = pm_align.acquire([idle], pin=True)
        lease.close()
        assert pm_align.allocate_host_slab(4 * page, pinned=False) is None
        assert pm_align.stats.idle_registrations == 1
        aligned = pm_align.allocate_host_slab(page, pinned=False, alignment=2 * page)
        assert aligned is not None
        assert pm_align.stats.idle_registrations == 0
        aligned.release()
    finally:
        pm_align.clear()
        idle = None
        buf.close()


def test_pin_manager_inflight_reservation_blocks_overcommit_without_spinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import piper_offload._host_memory as host_memory_mod

    page = mmap.PAGESIZE
    pm = PinManager(4 * page, max_host_bytes=2 * page, backend=_SimpleSlabBackend())
    buf = mmap.mmap(-1, page)
    w_page = torch.frombuffer(buf, dtype=torch.uint8)
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    orig_alloc_region = host_memory_mod.allocate_aligned_region
    result: list[object] = []
    result_t2: list[object] = []
    worker = threading.Thread(target=lambda: result.append(pm.allocate_host_slab(page, pinned=True)))
    t2 = threading.Thread(target=lambda: result_t2.append(pm.allocate_host_slab(page, pinned=True)))

    def gated_alloc_region(size: int) -> object:
        if threading.current_thread() is worker:
            entered.wait(timeout=5.0)
            proceed.wait(timeout=5.0)
        return orig_alloc_region(size)

    monkeypatch.setattr(host_memory_mod, "allocate_aligned_region", gated_alloc_region)
    b1 = None
    try:
        worker.start()
        entered.wait(timeout=5.0)
        assert pm._ledger.snapshot_atomic()[0] == page
        b1 = pm.allocate_host_slab(page, pinned=False)
        assert b1 is not None and pm.allocate_host_slab(page, pinned=False) is None
        with pm.acquire([w_page], pin=True) as w_lease:
            assert (
                w_lease.registered_bytes == 0
                and w_lease.pageable_bytes == page
                and pm.stats.registrations == 0
            )
        t2.start()
        proceed.wait(timeout=5.0)
        worker.join(timeout=5.0)
        t2.join(timeout=5.0)
        assert not worker.is_alive() and not t2.is_alive()
        assert result and result[0] is not None and getattr(result[0], "is_pinned", False)
        assert result_t2 == [None]
        t2a, t2b, bounce = pm._ledger.snapshot_atomic()
        assert pm._ledger.in_place_bytes + t2a + t2b + bounce <= 2 * page
    finally:
        entered.abort()
        proceed.abort()
        monkeypatch.undo()
        for thr in (worker, t2):
            if thr.is_alive():
                thr.join(timeout=2.0)
        for alloc in (b1, *result[:1], *result_t2[:1]):
            if alloc is not None:
                alloc.release()  # type: ignore[attr-defined]
        pm.clear()
        del w_page
        buf.close()


def test_pin_manager_concurrent_tier2a_host_epoch_rollback_under_host_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = mmap.PAGESIZE
    pm = PinManager(4 * page, max_host_bytes=3 * page, backend=_SimpleSlabBackend())
    buf = mmap.mmap(-1, page)
    w_page: torch.Tensor | None = torch.frombuffer(buf, dtype=torch.uint8)
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    orig_alloc = pm._tier2a.allocate
    result: list[object] = []
    worker = threading.Thread(target=lambda: result.append(pm.allocate_host_slab(page, pinned=True)))

    def gated_alloc(*args: object, **kwargs: object) -> object:
        if threading.current_thread() is worker:
            entered.wait(timeout=5.0)
            proceed.wait(timeout=5.0)
        return orig_alloc(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pm._tier2a, "allocate", gated_alloc)
    b1 = None
    lease = None
    try:
        worker.start()
        entered.wait(timeout=5.0)
        b1 = pm.allocate_host_slab(page, pinned=False)
        assert b1 is not None
        assert w_page is not None
        lease = pm.acquire([w_page], pin=True)
        assert pm.stats.pinned_bytes == page
        pm.max_host_bytes = 2 * page
        proceed.wait(timeout=5.0)
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert result == [None]
        assert pm._tier2a.stats.reserved_bytes == 0
    finally:
        entered.abort()
        proceed.abort()
        if worker.is_alive():
            worker.join(timeout=2.0)
        if lease is not None:
            lease.close()
        if b1 is not None:
            b1.release()
        pm.clear()
        w_page = None
        buf.close()


def test_pin_manager_budget_reduction_below_live_slab_does_not_spin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = mmap.PAGESIZE
    for pinned in (True, False):
        pm = PinManager(4 * page, max_host_bytes=4 * page, backend=_SimpleSlabBackend())
        slab = pm.allocate_host_slab(page, pinned=pinned, batch_hint=4)
        assert slab is not None
        arena = pm._tier2a if pinned else pm._tier2b
        calls = 0
        orig_alloc = arena.allocate

        def counted_alloc(
            *args: object, _orig: object = orig_alloc, **kwargs: object
        ) -> object:
            nonlocal calls
            calls += 1
            return _orig(*args, **kwargs)  # type: ignore[operator]

        monkeypatch.setattr(arena, "allocate", counted_alloc)
        try:
            pm.max_host_bytes = 2 * page
            assert pm.allocate_host_slab(page, pinned=pinned) is None
            assert calls <= 3
        finally:
            slab.release()
            pm.clear()


def test_pin_manager_tier2b_impossible_evict_epoch_snapshot_and_max_bytes_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = mmap.PAGESIZE
    # 1) Non-evictable Tier 2b leaves raw_target_pinned < 0 -> must not evict idle pinned weight.
    buf = mmap.mmap(-1, page)
    idle: torch.Tensor | None = torch.frombuffer(buf, dtype=torch.uint8)
    pm1 = PinManager(2 * page, max_host_bytes=3 * page, backend=_SimpleSlabBackend())
    b_live = None
    try:
        assert idle is not None
        lease = pm1.acquire([idle], pin=True)
        lease.close()
        b_live = pm1.allocate_host_slab(2 * page, pinned=False)
        assert b_live is not None
        assert pm1.allocate_host_slab(2 * page, pinned=False) is None
        assert pm1.stats.idle_registrations == 1
    finally:
        if b_live is not None:
            b_live.release()
        pm1.clear()
        idle = None
        buf.close()

    # 2) Concurrent 2-page and 1-page Tier 2a slabs under 3-page cap succeed without max_bytes clobbering.
    pm2 = PinManager(3 * page, max_host_bytes=3 * page, backend=_SimpleSlabBackend())
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    res_t1: list[object] = []
    orig_2a_alloc = pm2._tier2a.allocate
    t1 = threading.Thread(
        target=lambda: res_t1.append(pm2.allocate_host_slab(2 * page, pinned=True))
    )

    def gated_2a(*args: object, **kwargs: object) -> object:
        if threading.current_thread() is t1:
            entered.wait(timeout=5.0)
            proceed.wait(timeout=5.0)
        return orig_2a_alloc(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pm2._tier2a, "allocate", gated_2a)
    s2 = None
    try:
        t1.start()
        entered.wait(timeout=5.0)
        s2 = pm2.allocate_host_slab(page, pinned=True)
        assert s2 is not None
        proceed.wait(timeout=5.0)
        t1.join(timeout=5.0)
        assert len(res_t1) == 1 and res_t1[0] is not None
        res_t1[0].release()  # type: ignore[attr-defined]
    finally:
        entered.abort()
        proceed.abort()
        if s2 is not None:
            s2.release()
        pm2.clear()


def test_pin_manager_tier2b_epoch_snapshot_prevents_peak_host_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = mmap.PAGESIZE
    pm = PinManager(page, max_host_bytes=page, backend=_SimpleSlabBackend())
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    orig_bounce = pm._active_bounce_bytes_off_lock
    orig_2b_alloc = pm._tier2b.allocate
    max_bytes_seen: list[object] = []
    peak_total = 0
    res_2b: list[object] = []
    t_2b = threading.Thread(target=lambda: res_2b.append(pm.allocate_host_slab(page, pinned=False)))

    gate_once = True

    def gated_bounce() -> tuple[int, int, int]:
        nonlocal gate_once
        out = orig_bounce()
        if gate_once and threading.current_thread() is t_2b:
            gate_once = False
            entered.wait(timeout=5.0)
            proceed.wait(timeout=5.0)
        return out

    def spy_2b_alloc(*args: object, **kwargs: object) -> object:
        nonlocal peak_total
        max_bytes_seen.append(kwargs.get("max_bytes"))
        out = orig_2b_alloc(*args, **kwargs)  # type: ignore[arg-type]
        peak_total = max(peak_total, pm._tier2a.stats.reserved_bytes + pm._tier2b.stats.reserved_bytes)
        return out

    monkeypatch.setattr(pm, "_active_bounce_bytes_off_lock", gated_bounce)
    monkeypatch.setattr(pm._tier2b, "allocate", spy_2b_alloc)
    p1 = None
    try:
        t_2b.start()
        entered.wait(timeout=5.0)
        p1 = pm.allocate_host_slab(page, pinned=True)
        assert p1 is not None
        proceed.wait(timeout=5.0)
        t_2b.join(timeout=5.0)
        assert res_2b == [None]
        assert max_bytes_seen == [0]
        assert peak_total <= page
    finally:
        entered.abort()
        proceed.abort()
        if p1 is not None:
            p1.release()
        pm.clear()


def test_pin_manager_bounce_publish_respects_max_host_bytes_without_max_pinned() -> None:
    pm = PinManager(
        None,
        max_host_bytes=128 * 1024,
        bounce_bytes=128 * 1024,
        bounce_slot_bytes=64 * 1024,
        min_bounce_bytes=64 * 1024,
        backend=_SimpleSlabBackend(),
    )
    slab_small = pm.allocate_host_slab(64 * 1024, pinned=False)
    assert slab_small is not None
    try:
        with pm._use_bounce_ring() as ring:
            assert ring is not None
            assert pm.stats.bounce_bytes == 64 * 1024
            assert pm._tier2b.stats.reserved_bytes + pm.stats.bounce_bytes <= 128 * 1024
    finally:
        slab_small.release()
        pm.clear()

    slab_full = pm.allocate_host_slab(128 * 1024, pinned=False)
    assert slab_full is not None
    try:
        with pm._use_bounce_ring() as ring2:
            assert ring2 is None
        assert pm.stats.bounce_bytes == 0
    finally:
        slab_full.release()
        pm.clear()


class _SubstrateTestInterrupt(BaseException):
    pass


@CUDA
def test_stream_lease_produce_interrupt_after_yield_retains_writer_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    alloc_stream = torch.cuda.Stream(device=device)
    produce_stream = torch.cuda.Stream(device=device)
    reader_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(alloc_stream):
        gpu_buf = torch.zeros((256,), dtype=torch.float32, device=device)
    reader_buf = torch.empty_like(gpu_buf)
    lease = StreamLease(alloc_stream)

    # Part A: interrupt after yield while _record_ready_event runs normally.
    with pytest.raises(_SubstrateTestInterrupt, match="abort produce"), lease.produce(produce_stream):
        torch.cuda._sleep(25_000_000)
        gpu_buf.fill_(9.5)
        raise _SubstrateTestInterrupt("abort produce")
    assert not lease._writing and not lease.is_dirty and not lease.is_staged
    assert lease._ready_event is not None
    lease.stage_with(lambda: reader_buf.copy_(gpu_buf), reader_stream)
    reader_stream.synchronize()
    torch.testing.assert_close(reader_buf.cpu(), torch.full((256,), 9.5))

    # Part B: interrupt inside _record_ready_event retains produce_stream in _recorded_streams.
    orig_record_ready = lease._record_ready_event
    monkeypatch.setattr(
        lease,
        "_record_ready_event",
        lambda *_a, **_kw: (_ for _ in ()).throw(_SubstrateTestInterrupt("abort ready")),
    )
    with pytest.raises(_SubstrateTestInterrupt, match="abort ready"), lease.produce(produce_stream):
        torch.cuda._sleep(25_000_000)
        gpu_buf.fill_(11.25)
    assert not lease._writing and produce_stream in lease._recorded_streams
    with pytest.raises(_SubstrateTestInterrupt, match="abort ready"):
        lease.stage_with(lambda: None, produce_stream)
    assert not lease._writing and produce_stream in lease._recorded_streams
    monkeypatch.setattr(lease, "_record_ready_event", orig_record_ready)
    lease.stage_with(lambda: reader_buf.copy_(gpu_buf), reader_stream)
    reader_stream.synchronize()
    torch.testing.assert_close(reader_buf.cpu(), torch.full((256,), 11.25))


def _exercise_bounce_pause_window(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_host_pages: int,
    pause_after: bool,
) -> None:
    import piper_offload._host_memory as host_memory_mod

    page = mmap.PAGESIZE
    pm = PinManager(
        max_host_pages * page,
        max_host_bytes=max_host_pages * page,
        bounce_bytes=2 * page,
        bounce_slot_bytes=page,
        min_bounce_bytes=page,
        backend=_SimpleSlabBackend(),
    )
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    orig_init = PinnedBounceRing._ensure_initialized
    orig_alloc_region = host_memory_mod.allocate_aligned_region
    t_bounce_ref: list[threading.Thread] = []

    def gated_alloc_region(size: int) -> object:
        if not pause_after and t_bounce_ref and threading.current_thread() is t_bounce_ref[0]:
            entered.wait(timeout=5.0)
            proceed.wait(timeout=5.0)
        return orig_alloc_region(size)

    def gated_init(self_ring: PinnedBounceRing) -> bool:
        if pause_after:
            ok = orig_init(self_ring)
            entered.wait(timeout=5.0)
            proceed.wait(timeout=5.0)
            return ok
        return orig_init(self_ring)

    def _use_bounce() -> None:
        with pm._use_bounce_ring():
            pass

    monkeypatch.setattr(host_memory_mod, "allocate_aligned_region", gated_alloc_region)
    monkeypatch.setattr(PinnedBounceRing, "_ensure_initialized", gated_init)
    t_bounce = threading.Thread(target=_use_bounce)
    t_bounce_ref.append(t_bounce)
    slab = None
    try:
        t_bounce.start()
        entered.wait(timeout=5.0)
        assert pm._ledger.snapshot_atomic()[2] == 2 * page
        slab = pm.allocate_host_slab(page, pinned=False)
        assert (slab is not None) if pause_after else (slab is None)
        proceed.wait(timeout=5.0)
        t_bounce.join(timeout=5.0)
        _, ledger_2b, ledger_bounce = pm._ledger.snapshot_atomic()
        assert ledger_2b + ledger_bounce <= max_host_pages * page
        assert pm._tier2b.stats.reserved_bytes + pm.stats.bounce_bytes <= max_host_pages * page
    finally:
        entered.abort()
        proceed.abort()
        monkeypatch.undo()
        if slab is not None:
            slab.release()
        pm.clear()


def test_pin_manager_bounce_and_tier2b_peak_host_ceiling_and_no_double_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_bounce_pause_window(monkeypatch, max_host_pages=2, pause_after=False)
    _exercise_bounce_pause_window(monkeypatch, max_host_pages=3, pause_after=True)


def test_pin_manager_impossible_cold_aligned_request_does_not_evict_idle() -> None:
    page = mmap.PAGESIZE
    pm = PinManager(page, max_host_bytes=page, bounce_bytes=0, backend=_SimpleSlabBackend())
    buf = mmap.mmap(-1, page)
    idle_t: torch.Tensor | None = torch.frombuffer(buf, dtype=torch.uint8)
    try:
        assert idle_t is not None
        with pm.acquire([idle_t]):
            pass
        assert pm.stats.idle_registrations == 1
        assert pm._tier2a.stats.reserved_bytes == 0
        assert pm.allocate_host_slab(page, alignment=2 * page, pinned=True) is None
        assert pm.stats.idle_registrations == 1
        assert pm.stats.pinned_bytes == page
        assert idle_t.data_ptr() in pm._registrations
    finally:
        pm.clear()
        idle_t = None
        buf.close()


def test_pin_manager_bounce_ledger_blocks_concurrent_inplace_pinning_under_max_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import piper_offload._host_memory as host_memory_mod

    page = mmap.PAGESIZE
    pm = PinManager(
        2 * page,
        max_host_bytes=4 * page,
        bounce_bytes=2 * page,
        bounce_slot_bytes=page,
        min_bounce_bytes=page,
        backend=_SimpleSlabBackend(),
    )
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    orig_alloc = host_memory_mod.allocate_aligned_region
    t_holder: list[threading.Thread] = []

    def gated_alloc(size: int) -> object:
        if t_holder and threading.current_thread() is t_holder[0]:
            entered.wait(timeout=5.0)
            proceed.wait(timeout=5.0)
        return orig_alloc(size)

    def _use_bounce() -> None:
        with pm._use_bounce_ring():
            pass

    monkeypatch.setattr(host_memory_mod, "allocate_aligned_region", gated_alloc)
    t_bounce = threading.Thread(target=_use_bounce)
    t_holder.append(t_bounce)
    buf = mmap.mmap(-1, page)
    fg_t: torch.Tensor | None = torch.frombuffer(buf, dtype=torch.uint8)
    try:
        assert fg_t is not None
        t_bounce.start()
        entered.wait(timeout=5.0)
        assert pm._ledger.snapshot_atomic()[2] == 2 * page
        with pm.acquire([fg_t], pin=True) as lease:
            assert (lease.registered_bytes, lease.pageable_bytes) == (0, page)
        assert fg_t.data_ptr() not in pm._registrations
        assert pm._ledger.in_place_bytes + pm._ledger.bounce_bytes <= 2 * page
        proceed.wait(timeout=5.0)
        t_bounce.join(timeout=5.0)
    finally:
        entered.abort()
        proceed.abort()
        monkeypatch.undo()
        pm.clear()
        fg_t = None
        buf.close()


def _write_test_ckpt_tensor(tmp_path, nbytes: int, name: str) -> torch.Tensor:
    from piper_offload import MappedCheckpoint

    path = tmp_path / f"{name}.safetensors"
    header = b'{"w": {"dtype": "U8", "shape": [%d], "data_offsets": [0, %d]}}' % (nbytes, nbytes)
    header += b" " * (-(8 + len(header)) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + (bytes([1]) * nbytes))
    return MappedCheckpoint(path).get_tensor("w")


def test_pin_manager_demotion_respects_concurrent_foreground_pinning_under_max_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    page = mmap.PAGESIZE
    idle_ckpt = _write_test_ckpt_tensor(tmp_path, 2 * page, "idle_demote")
    pm = PinManager(
        4 * page,
        max_host_bytes=4 * page,
        slab_bytes=2 * page,
        bounce_bytes=0,
        backend=_SimpleSlabBackend(),
    )
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    orig_demote_slot = PinManager._allocate_demotion_slot
    t_holder: list[threading.Thread] = []

    def gated_demote_slot(self_pm: PinManager, *args: object) -> object:
        if t_holder and threading.current_thread() is t_holder[0]:
            entered.wait(timeout=5.0)
            proceed.wait(timeout=5.0)
        return orig_demote_slot(self_pm, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(PinManager, "_allocate_demotion_slot", gated_demote_slot)
    t_demote = threading.Thread(target=lambda: setattr(pm, "max_pinned_bytes", 0))
    t_holder.append(t_demote)
    buf = mmap.mmap(-1, 2 * page)
    fg_t: torch.Tensor | None = torch.frombuffer(buf, dtype=torch.uint8)
    fg_lease = None
    try:
        assert fg_t is not None
        with pm.acquire([idle_ckpt]):
            pass
        assert pm._ledger.snapshot_atomic()[0] == 2 * page
        t_demote.start()
        entered.wait(timeout=5.0)
        pm.max_pinned_bytes = 4 * page
        fg_lease = pm.acquire([fg_t], pin=True)
        assert fg_lease.registered_bytes == 2 * page
        proceed.wait(timeout=5.0)
        t_demote.join(timeout=5.0)
        t2a, t2b, bounce = pm._ledger.snapshot_atomic()
        assert pm._ledger.in_place_bytes + t2a + t2b + bounce <= 4 * page
        assert pm.stats.private_copy_bytes == 0
    finally:
        entered.abort()
        proceed.abort()
        monkeypatch.undo()
        if fg_lease is not None:
            fg_lease.close()
        pm.clear()
        fg_t = None
        buf.close()


def _setup_warm_tier2a_with_idle_reg(page: int) -> tuple[PinManager, mmap.mmap, torch.Tensor]:
    pm = PinManager(2 * page, max_host_bytes=2 * page, bounce_bytes=0, backend=_SimpleSlabBackend())
    buf = mmap.mmap(-1, page)
    idle_t = torch.frombuffer(buf, dtype=torch.uint8)
    warm = pm.allocate_host_slab(page, pinned=True)
    assert warm is not None
    warm.release()
    with pm.acquire([idle_t]):
        pass
    assert pm.allocate_host_slab(page, alignment=4 * page, pinned=True) is None
    assert pm.stats.idle_registrations == 1
    assert pm._tier2a.stats.reserved_bytes == page
    pm.max_pinned_bytes = page
    pm.max_pinned_bytes = 2 * page
    return pm, buf, idle_t


def test_pin_manager_warm_tier2a_trim_race_and_overaligned_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import piper_offload._host_memory as host_memory_mod

    page = mmap.PAGESIZE
    pm, buf, idle_t = _setup_warm_tier2a_with_idle_reg(page)
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    orig_alloc = host_memory_mod.allocate_aligned_region
    res_2a: list[object] = []
    t_holder: list[threading.Thread] = []

    def gated_alloc(size: int) -> object:
        if size == 2 * page and t_holder and threading.current_thread() is t_holder[0]:
            entered.wait(timeout=5.0)
            proceed.wait(timeout=5.0)
        return orig_alloc(size)

    s_2a = None
    s_2b = None
    try:
        _ = idle_t
        monkeypatch.setattr(host_memory_mod, "allocate_aligned_region", gated_alloc)
        t_2a = threading.Thread(
            target=lambda: res_2a.append(pm.allocate_host_slab(2 * page, pinned=True))
        )
        t_holder.append(t_2a)
        t_2a.start()
        entered.wait(timeout=5.0)
        s_2b = pm.allocate_host_slab(page, pinned=False)
        assert s_2b is None
        proceed.wait(timeout=5.0)
        t_2a.join(timeout=5.0)
        s_2a = res_2a[0]
        assert s_2a is not None
        t2a, t2b, bounce = pm._ledger.snapshot_atomic()
        assert pm._ledger.in_place_bytes + t2a + t2b + bounce <= 2 * page
    finally:
        entered.abort()
        proceed.abort()
        monkeypatch.undo()
        if s_2b is not None:
            s_2b.release()
        if s_2a is not None:
            s_2a.release()  # type: ignore[attr-defined]
        pm.clear()
        del idle_t
        buf.close()


@CUDA
def test_stream_lease_spill_interrupt_after_copy_keeps_fallback_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import piper_offload.stream_lease as stream_lease_mod

    device = torch.device("cuda")
    compute_stream = torch.cuda.Stream(device=device)
    d2h_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(compute_stream):
        gpu_buf = torch.empty((1024,), dtype=torch.float32, device=device)
    host_buf = torch.empty((1024,), dtype=torch.float32, pin_memory=True)
    lease = StreamLease(compute_stream)
    with lease.produce(compute_stream):
        gpu_buf.fill_(7.5)

    def fail_record(_stream: torch.cuda.Stream) -> object:
        raise _SubstrateTestInterrupt("post-copy-record-fail")

    seen_pre_copy: list[object] = []

    def copy_with_probe() -> None:
        seen_pre_copy.append(lease._d2h_fallback_stream)
        host_buf.copy_(gpu_buf, non_blocking=True)
        raise _SubstrateTestInterrupt("mid-copy-fail")

    monkeypatch.setattr(stream_lease_mod, "_record_event", fail_record)
    with pytest.raises(_SubstrateTestInterrupt, match="mid-copy-fail"):
        lease.spill_to_host(copy_with_probe, d2h_stream)
    monkeypatch.undo()
    assert seen_pre_copy == [d2h_stream]
    assert lease.d2h_in_flight and lease._d2h_fallback_stream is d2h_stream
    lease.wait_d2h()
    assert not lease.d2h_in_flight and lease._d2h_fallback_stream is None and lease.is_dirty
    torch.testing.assert_close(host_buf, torch.full((1024,), 7.5))


def test_weak_mmap_tracked_region_credits_ledger_on_buffer_error_gc() -> None:
    import gc

    page = mmap.PAGESIZE
    pm = PinManager(2 * page, max_host_bytes=2 * page, bounce_bytes=0, backend=_SimpleSlabBackend())
    mv: memoryview | None = None
    alloc = pm._tier2b.allocate(page)
    assert alloc is not None
    try:
        assert alloc._lease.slab is not None
        slab_bytes = alloc._lease.slab.size
        tracked = alloc._lease.slab.backing._state.region
        assert tracked is not None
        mv = memoryview(tracked)
        alloc.release()
        assert pm._tier2b.trim_empty(0) == 0
        assert pm._ledger.snapshot_atomic()[1] == slab_bytes
        del tracked
        gc.collect()
        assert pm._ledger.snapshot_atomic()[1] == slab_bytes
        del mv
        mv = None
        gc.collect()
        assert pm._ledger.snapshot_atomic()[1] == 0
    finally:
        if mv is not None:
            mv.release()
        pm.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wave_tracker_overlapping_handoff_during_reap_preserves_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import piper_offload._slab_pool_common as common_mod

    schema = SlotSchema({"k": 256, "v": 256}, {"scale": (torch.float32, (8,))}, field_alignment=256)
    pm = PinManager(4 * 1024 * 1024, max_host_bytes=4 * 1024 * 1024)
    entered = threading.Event()
    proceed = threading.Event()
    reap_err: list[BaseException] = []
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device="cuda", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None) as pool,
        ):
            with pool.produce(["a"]):
                pass
            slot_a = pool._chunks["a"].slot
            assert slot_a is not None
            lease1 = pool._wave_releases.new_lease()
            lease2 = pool._wave_releases.new_lease()
            with pool._lock:
                lease1.add_unlocked(slot_a)
                lease2.add_unlocked(slot_a)
            lease1.acquire()
            lease2.acquire()
            lease1.handoff_to_pool()

            orig_rel = common_mod._release_slot_stream_lease

            def paused_rel(*args: object, **kwargs: object) -> tuple[bool, BaseException | None]:
                entered.set()
                proceed.wait(timeout=5.0)
                return orig_rel(*args, **kwargs)  # type: ignore[arg-type]

            monkeypatch.setattr(common_mod, "_release_slot_stream_lease", paused_rel)

            def run_reap() -> None:
                try:
                    pool._wave_releases.reap(lambda: pool._closed or pool._closing)
                except BaseException as exc:
                    reap_err.append(exc)

            t_reap = threading.Thread(target=run_reap)
            t_reap.start()
            assert entered.wait(timeout=5.0)
            monkeypatch.undo()
            lease2.handoff_to_pool()
            proceed.set()
            t_reap.join(timeout=5.0)
            assert not reap_err and len(pool._wave_releases.pending) == 1
            pool.close()
            assert slot_a.active_readers == 0
    finally:
        entered.set()
        proceed.set()
        monkeypatch.undo()
        pm.clear()


@CUDA
def test_wave_lease_release_and_reap_propagate_interrupt_after_successful_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import piper_offload._slab_pool_common as common_mod

    device = torch.device("cuda")
    schema = SlotSchema({"k": 256, "v": 256}, {"scale": (torch.float32, (8,))}, field_alignment=256)
    pm = PinManager(4 * 1024 * 1024, max_host_bytes=4 * 1024 * 1024)
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device=device, alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None) as pool,
        ):
            with pool.produce(["a", "b"]) as batch:
                batch.arena_bytes.zero_()
            slot_a = pool._chunks["a"].slot
            slot_b = pool._chunks["b"].slot
            assert slot_a is not None and slot_b is not None

            orig_rel = common_mod._release_slot_stream_lease
            calls = 0

            def release_with_interrupt(
                slot: object,
                release_stream: torch.cuda.Stream,
                end_streams: tuple[torch.cuda.Stream, ...],
            ) -> tuple[bool, BaseException | None]:
                nonlocal calls
                calls += 1
                ok, _err = orig_rel(slot, release_stream, end_streams)  # type: ignore[arg-type]
                if calls == 1:
                    return ok, _SubstrateTestInterrupt("retry-succeeded-interrupt")
                return ok, None

            lease1 = pool._wave_releases.new_lease()
            with pool._lock:
                lease1.add_unlocked(slot_a)
            lease1.acquire()
            monkeypatch.setattr(common_mod, "_release_slot_stream_lease", release_with_interrupt)
            with pytest.raises(_SubstrateTestInterrupt, match="retry-succeeded-interrupt"):
                lease1.release()
            monkeypatch.undo()
            assert slot_a.active_readers == 0

            calls = 0
            lease2 = pool._wave_releases.new_lease()
            with pool._lock:
                lease2.add_unlocked(slot_a)
                lease2.add_unlocked(slot_b)
            lease2.acquire()
            lease2.handoff_to_pool()
            assert len(pool._wave_releases.pending) == 2
            monkeypatch.setattr(common_mod, "_release_slot_stream_lease", release_with_interrupt)
            with pytest.raises(_SubstrateTestInterrupt, match="retry-succeeded-interrupt"):
                pool._wave_releases.reap(lambda: pool._closed or pool._closing)
            monkeypatch.undo()
            assert slot_a.active_readers == 0
            assert slot_b.active_readers == 1
            assert len(pool._wave_releases.pending) == 1
            pool._wave_releases.reap(lambda: pool._closed or pool._closing)
            assert slot_b.active_readers == 0
            assert len(pool._wave_releases.pending) == 0
    finally:
        monkeypatch.undo()
        pm.clear()


def test_tier2b_eviction_uses_fresh_ledger_and_bounce_cap_without_pin_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = mmap.PAGESIZE
    buf = mmap.mmap(-1, 2 * page)
    idle: torch.Tensor | None = torch.frombuffer(buf, dtype=torch.uint8)
    pm = PinManager(4 * page, max_host_bytes=4 * page, bounce_bytes=0, backend=_SimpleSlabBackend())
    b1 = None
    b2 = None
    try:
        assert idle is not None
        with pm.acquire([idle], pin=True):
            pass
        assert pm.stats.idle_registrations == 1

        b1 = pm.allocate_host_slab(2 * page, pinned=False)
        assert b1 is not None
        orig_alloc = pm._tier2b.allocate
        released_once = False

        def concurrent_release_on_miss(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            nonlocal released_once
            res = orig_alloc(*args, **kwargs)  # type: ignore[arg-type]
            if res is None and not released_once:
                released_once = True
                assert b1 is not None
                b1.release()
                pm._tier2b.trim_empty(0)
            return res

        monkeypatch.setattr(pm._tier2b, "allocate", concurrent_release_on_miss)
        b2 = pm.allocate_host_slab(3 * page, pinned=False)
        monkeypatch.undo()
        assert released_once
        assert b2 is not None
        assert pm.stats.idle_registrations == 0

        pm_unbounded_pin = PinManager(
            None,
            max_host_bytes=None,
            bounce_bytes=2 * page,
            bounce_slot_bytes=page,
            min_bounce_bytes=page,
            backend=_SimpleSlabBackend(),
        )
        try:
            with pm_unbounded_pin._lock:
                assert pm_unbounded_pin._ledger.can_reserve_bounce_locked(2 * page)
                assert not pm_unbounded_pin._ledger.can_reserve_bounce_locked(3 * page)
        finally:
            pm_unbounded_pin.clear()
    finally:
        monkeypatch.undo()
        if b2 is not None and not b2.released:
            b2.release()
        if b1 is not None and not b1.released:
            b1.release()
        pm.clear()
        idle = None
        buf.close()
