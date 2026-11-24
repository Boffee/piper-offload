"""CUDA target lease ordering and stream lifetime."""

import contextlib
from typing import cast

import pytest
import torch
from torch import nn

from piper_offload import PinManager
from piper_offload.host_module import (
    HostModuleLoadPlan,
    HostModuleStore,
    HostModuleTarget,
)
from piper_offload.target_lease import CudaTargetLease
from tests.conftest import Interrupt

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@CUDA
def test_first_pinned_upload_waits_for_prior_work_on_reused_allocation() -> None:
    module = nn.Linear(64, 64, bias=False).requires_grad_(False)
    module.weight.fill_(7)
    plan = HostModuleStore.from_module(module).bind(module).resolve_load_plan()
    manager = PinManager(1024**2)
    device = torch.device("cuda")
    copy_stream = torch.cuda.Stream(device=device)
    allocation_stream = torch.cuda.current_stream(device)
    lease = None
    try:
        with manager.acquire(plan.sources["weight"].storage_tensors()) as pins:
            assert pins.registered_bytes > 0
            # Warm allocation and fill/copy kernels before enqueueing delayed
            # work, so lazy CUDA module loading cannot serialize the probe.
            warm = CudaTargetLease.allocate(plan, device, allocation_stream=allocation_stream)
            warm.stage(plan, copy_stream)
            warm.close()
            stale = torch.empty((64, 64), device=device).fill_(0)
            pointer = stale.data_ptr()
            torch.cuda._sleep(1)
            torch.cuda.synchronize(device)
            torch.cuda._sleep(500_000_000)
            stale.fill_(123)
            del stale

            lease = CudaTargetLease.allocate(plan, device, allocation_stream=allocation_stream)
            assert lease.target.param_targets["weight"].param.data_ptr() == pointer
            assert not allocation_stream.query()
            lease.stage(plan, copy_stream, non_blocking=True)
            actual = lease.acquire(allocation_stream).param_targets["weight"].param.cpu()
            copy_stream.synchronize()
            torch.testing.assert_close(actual, torch.full((64, 64), 7.0))
    finally:
        if lease is not None:
            lease.close()
        manager.clear()


@CUDA
def test_stage_is_registry_pure_and_reuse_waits_for_consumer() -> None:
    torch.manual_seed(0)
    first = nn.Linear(32, 32, bias=False)
    second = nn.Linear(32, 32, bias=False)
    for module in (first, second):
        module.weight.requires_grad_(False)
    value = torch.randn(4, 32)
    expected_first = first(value).cuda()
    expected_second = second(value).cuda()
    first_instance = HostModuleStore.from_module(first).bind(first)
    second_instance = HostModuleStore.from_module(second).bind(second)
    device = torch.device("cuda")
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    first_plan = first_instance.resolve_load_plan()
    second_plan = second_instance.resolve_load_plan()
    lease = CudaTargetLease.allocate(first_plan, device)

    lease.stage(first_plan, copy_stream)
    assert first.weight.device.type == "cpu"
    with torch.cuda.stream(compute_stream):
        first_instance.install_target(lease.acquire(compute_stream))
        actual_first = first(value.cuda())
    first_instance.install_host()
    lease.release()

    lease.stage(second_plan, copy_stream)
    assert second.weight.device.type == "cpu"
    with torch.cuda.stream(compute_stream):
        second_instance.install_target(lease.acquire(compute_stream))
        actual_second = second(value.cuda())
    second_instance.install_host()
    lease.release()
    lease.close()

    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual_first, expected_first)
    torch.testing.assert_close(actual_second, expected_second)


def test_close_synchronizes_target_work_before_drop() -> None:
    class Event:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    class Stream:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    allocation_stream = Stream()
    side_stream = Stream()
    consumer_stream = Stream()
    ready = Event()
    lease = CudaTargetLease(
        cast(HostModuleTarget, object()),
        cast(torch.cuda.Stream, allocation_stream),
    )
    lease._ready_event = cast(torch.cuda.Event, ready)
    lease._acquired = True
    lease.record_stream(cast(torch.cuda.Stream, consumer_stream))
    lease.record_stream(cast(torch.cuda.Stream, side_stream))

    lease.close()

    assert ready.synchronize_calls == 1
    assert allocation_stream.synchronize_calls == 1
    assert consumer_stream.synchronize_calls == 1
    assert side_stream.synchronize_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        _ = lease.target


def test_failed_close_keeps_the_target_for_a_retry() -> None:
    class Stream:
        def __init__(self, failures: int = 0) -> None:
            self.failures = failures
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            if self.failures:
                self.failures -= 1
                raise Interrupt

    allocation_stream = Stream()
    consumer_stream = Stream(failures=1)
    target = cast(HostModuleTarget, object())
    lease = CudaTargetLease(target, cast(torch.cuda.Stream, allocation_stream))
    lease._acquired = True
    lease.record_stream(cast(torch.cuda.Stream, consumer_stream))

    with pytest.raises(Interrupt):
        lease.close()
    # Its work was not confirmed finished, so the lease still owns the target.
    assert lease.target is target

    lease.close()

    assert consumer_stream.synchronize_calls == 2
    with pytest.raises(RuntimeError, match="closed"):
        _ = lease.target


def test_restage_waits_for_every_actual_use_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stream:
        def __init__(self) -> None:
            self.waited_events: list[object] = []
            self.waited_streams: list[object] = []
            self.recorded = object()

        def wait_event(self, event: object) -> None:
            self.waited_events.append(event)

        def wait_stream(self, stream: object) -> None:
            self.waited_streams.append(stream)

        def record_event(self) -> object:
            return self.recorded

    class Plan:
        def __init__(self) -> None:
            self.copied = False

        def copy_to_target(self, *_args: object, **_kwargs: object) -> None:
            self.copied = True

    allocation_stream = Stream()
    first_consumer = Stream()
    second_consumer = Stream()
    copy_stream = Stream()
    ready = object()
    plan = Plan()
    lease = CudaTargetLease(
        cast(HostModuleTarget, object()),
        cast(torch.cuda.Stream, allocation_stream),
    )
    lease._ready_event = cast(torch.cuda.Event, ready)
    lease._staged = True
    lease.acquire(cast(torch.cuda.Stream, first_consumer))
    lease.record_stream(cast(torch.cuda.Stream, second_consumer))
    lease.release()
    monkeypatch.setattr(
        torch.cuda,
        "stream",
        lambda _stream: contextlib.nullcontext(),
    )

    lease.stage(
        cast(HostModuleLoadPlan, plan),
        cast(torch.cuda.Stream, copy_stream),
    )

    assert copy_stream.waited_events[0] is ready
    assert set(copy_stream.waited_events) == {
        ready,
        first_consumer.recorded,
        second_consumer.recorded,
    }
    assert copy_stream.waited_streams == []
    assert plan.copied
    assert lease._ready_event is copy_stream.recorded


def test_close_synchronizes_release_events_and_retries_on_failure() -> None:
    class Stream:
        def __init__(self, failures: int = 0) -> None:
            self.synchronize_calls = 0
            self.event = Event(failures=failures)

        def record_event(self) -> object:
            return self.event

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    class Event:
        def __init__(self, failures: int = 0) -> None:
            self.failures = failures
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            if self.failures:
                self.failures -= 1
                raise Interrupt

    allocation_stream = Stream()
    consumer_stream = Stream(failures=1)
    target = cast(HostModuleTarget, object())
    lease = CudaTargetLease(target, cast(torch.cuda.Stream, allocation_stream))
    lease._acquired = True
    lease.record_stream(cast(torch.cuda.Stream, consumer_stream))
    lease.release()

    with pytest.raises(Interrupt):
        lease.close()
    assert lease.target is target
    assert consumer_stream.event.synchronize_calls == 1
    assert allocation_stream.synchronize_calls == 0

    lease.close()

    assert consumer_stream.event.synchronize_calls == 2
    assert allocation_stream.synchronize_calls == 1
    assert consumer_stream.synchronize_calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        _ = lease.target


@CUDA
def test_restage_after_release_overlaps_later_compute() -> None:
    first = nn.Linear(64, 64, bias=False).requires_grad_(False)
    second = nn.Linear(64, 64, bias=False).requires_grad_(False)
    first_plan = HostModuleStore.from_module(first).bind(first).resolve_load_plan()
    second_plan = HostModuleStore.from_module(second).bind(second).resolve_load_plan()
    manager = PinManager(1024**2)
    device = torch.device("cuda")
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    lease = None
    try:
        sources = [
            *first_plan.sources["weight"].storage_tensors(),
            *second_plan.sources["weight"].storage_tensors(),
        ]
        with manager.acquire(sources) as pins:
            assert pins.registered_bytes > 0
            warm = CudaTargetLease.allocate(
                first_plan, device, allocation_stream=compute_stream
            )
            warm.stage(first_plan, copy_stream, non_blocking=True)
            warm.acquire(compute_stream)
            warm.release()
            warm.close()
            with torch.cuda.stream(compute_stream):
                torch.cuda._sleep(1)
            compute_stream.synchronize()

            lease = CudaTargetLease.allocate(
                first_plan, device, allocation_stream=compute_stream
            )
            lease.stage(first_plan, copy_stream, non_blocking=True)
            lease.acquire(compute_stream)
            lease.release()

            with torch.cuda.stream(compute_stream):
                torch.cuda._sleep(500_000_000)

            lease.stage(second_plan, copy_stream, non_blocking=True)
            copy_stream.synchronize()
            assert not compute_stream.query()
            torch.testing.assert_close(
                lease.target.param_targets["weight"].param.cpu(),
                second.weight,
            )
    finally:
        compute_stream.synchronize()
        if lease is not None:
            lease.close()
        manager.clear()


@CUDA
def test_restage_after_release_still_waits_for_prior_compute() -> None:
    first = nn.Linear(64, 64, bias=False).requires_grad_(False)
    second = nn.Linear(64, 64, bias=False).requires_grad_(False)
    first.weight.fill_(3)
    second.weight.fill_(9)
    first_plan = HostModuleStore.from_module(first).bind(first).resolve_load_plan()
    second_plan = HostModuleStore.from_module(second).bind(second).resolve_load_plan()
    manager = PinManager(1024**2)
    device = torch.device("cuda")
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    lease = None
    try:
        sources = [
            *first_plan.sources["weight"].storage_tensors(),
            *second_plan.sources["weight"].storage_tensors(),
        ]
        with manager.acquire(sources) as pins:
            assert pins.registered_bytes > 0
            warm = CudaTargetLease.allocate(
                first_plan, device, allocation_stream=compute_stream
            )
            warm.stage(first_plan, copy_stream, non_blocking=True)
            warm.acquire(compute_stream)
            warm.release()
            warm.close()
            with torch.cuda.stream(compute_stream):
                torch.cuda._sleep(1)
            compute_stream.synchronize()

            lease = CudaTargetLease.allocate(
                first_plan, device, allocation_stream=compute_stream
            )
            lease.stage(first_plan, copy_stream, non_blocking=True)
            target = lease.acquire(compute_stream)
            with torch.cuda.stream(compute_stream):
                torch.cuda._sleep(500_000_000)
                observed = target.param_targets["weight"].param.clone()
            lease.release()

            lease.stage(second_plan, copy_stream, non_blocking=True)
            assert not copy_stream.query()
            compute_stream.synchronize()
            copy_stream.synchronize()
            torch.testing.assert_close(observed.cpu(), first.weight)
            torch.testing.assert_close(
                lease.target.param_targets["weight"].param.cpu(),
                second.weight,
            )
    finally:
        compute_stream.synchronize()
        if lease is not None:
            lease.close()
        manager.clear()


def test_stage_failure_preserves_dependencies_and_staged_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEvent:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    class FakeStream:
        def __init__(self, *, record_failures: int = 0) -> None:
            self.record_failures = record_failures
            self.events: list[FakeEvent] = []
            self.synchronize_calls = 0

        def wait_event(self, _event: FakeEvent) -> None:
            pass

        def wait_stream(self, _stream: FakeStream) -> None:
            pass

        def record_event(self) -> FakeEvent:
            if self.record_failures:
                self.record_failures -= 1
                raise Interrupt
            event = FakeEvent()
            self.events.append(event)
            return event

        def synchronize(self) -> None:
            self.synchronize_calls += 1

        def _as_parameter_(self) -> None:
            return None

        @property
        def device_index(self) -> int:
            return 0

        @property
        def cuda_stream(self) -> int:
            return 0

    class FakePlan:
        def __init__(self, *, fail_copy: bool = False) -> None:
            self.fail_copy = fail_copy

        def copy_to_target(self, _target: HostModuleTarget, *, non_blocking: bool) -> None:
            assert non_blocking is True
            if self.fail_copy:
                raise Interrupt

    allocation_stream = FakeStream()
    prior_stream = FakeStream()
    unrecorded_stream = FakeStream()
    copy_stream = FakeStream(record_failures=1)
    target = cast(HostModuleTarget, object())
    lease = CudaTargetLease(target, cast(torch.cuda.Stream, allocation_stream))
    lease._acquired = True
    lease.record_stream(cast(torch.cuda.Stream, prior_stream))
    lease.release()
    release_event = prior_stream.events[0]
    lease._acquired = True
    lease.record_stream(cast(torch.cuda.Stream, unrecorded_stream))
    lease._acquired = False

    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: contextlib.nullcontext())

    # Record failures during finally leave prior dependencies and staged=False.
    with pytest.raises(Interrupt):
        lease.stage(cast(HostModuleLoadPlan, FakePlan(fail_copy=False)), cast(torch.cuda.Stream, copy_stream))
    assert not lease._staged
    assert lease._ready_event is None
    assert release_event in lease._release_events.values()
    assert cast(torch.cuda.Stream, unrecorded_stream) in lease._recorded_streams
    assert cast(torch.cuda.Stream, copy_stream) in lease._recorded_streams
    # release() is a no-op when unacquired so cleanup dependencies stay intact.
    lease.release()
    assert cast(torch.cuda.Stream, unrecorded_stream) in lease._recorded_streams
    assert cast(torch.cuda.Stream, copy_stream) in lease._recorded_streams
    with pytest.raises(RuntimeError, match="must be staged"):
        lease.acquire(cast(torch.cuda.Stream, prior_stream))

    # A partial copy failure whose record_event succeeds captures prior dependencies
    # into a new ready_event and still leaves staged=False.
    with pytest.raises(Interrupt):
        lease.stage(cast(HostModuleLoadPlan, FakePlan(fail_copy=True)), cast(torch.cuda.Stream, copy_stream))
    assert not lease._staged
    assert lease._ready_event is copy_stream.events[0]
    assert lease._release_events == {}
    assert lease._recorded_streams == set()

    lease.close()
    assert copy_stream.events[0].synchronize_calls == 1
    assert allocation_stream.synchronize_calls == 1


def test_release_interruption_preserves_remaining_streams() -> None:
    record_calls = 0

    class Event:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    class Stream:
        def __init__(self) -> None:
            self.event: Event | None = None
            self.synchronize_calls = 0

        def record_event(self) -> Event:
            nonlocal record_calls
            record_calls += 1
            if record_calls == 2:
                raise Interrupt
            self.event = Event()
            return self.event

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    allocation_stream = Stream()
    stream_a = Stream()
    stream_b = Stream()
    target = cast(HostModuleTarget, object())
    lease = CudaTargetLease(target, cast(torch.cuda.Stream, allocation_stream))
    lease._acquired = True
    lease.record_stream(cast(torch.cuda.Stream, stream_a))
    lease.record_stream(cast(torch.cuda.Stream, stream_b))

    with pytest.raises(Interrupt):
        lease.release()

    assert lease._acquired
    assert len(lease._release_events) == 1
    assert len(lease._recorded_streams) == 1
    with pytest.raises(RuntimeError, match="while it is in use"):
        lease.stage(cast(HostModuleLoadPlan, object()), cast(torch.cuda.Stream, allocation_stream))

    lease.close()

    assert not lease._acquired
    recorded = [s for s in (stream_a, stream_b) if s.event is not None]
    unrecorded = [s for s in (stream_a, stream_b) if s.event is None]
    assert len(recorded) == 1
    assert len(unrecorded) == 1
    assert recorded[0].event is not None and recorded[0].event.synchronize_calls == 1
    assert unrecorded[0].synchronize_calls == 1
    assert allocation_stream.synchronize_calls == 1

    # Retrying release() finishes snapshotting the remaining stream and clears _acquired.
    record_calls = 0
    retry_stream_a = Stream()
    retry_stream_b = Stream()
    retry_lease = CudaTargetLease(target, cast(torch.cuda.Stream, allocation_stream))
    retry_lease._acquired = True
    retry_lease.record_stream(cast(torch.cuda.Stream, retry_stream_a))
    retry_lease.record_stream(cast(torch.cuda.Stream, retry_stream_b))
    with pytest.raises(Interrupt):
        retry_lease.release()
    assert retry_lease._acquired
    retry_lease.release()
    assert not retry_lease._acquired
    assert retry_lease._recorded_streams == set()
    assert len(retry_lease._release_events) == 2
    retry_lease.close()
