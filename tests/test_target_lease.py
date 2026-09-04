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

    assert copy_stream.waited_events == [ready]
    assert set(copy_stream.waited_streams) == {first_consumer, second_consumer}
    assert plan.copied
    assert lease._ready_event is copy_stream.recorded
