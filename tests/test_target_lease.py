"""CUDA target lease ordering and stream lifetime."""

import contextlib
from typing import cast

import pytest
import torch
from torch import nn

from piper_offload.pinned_module import (
    PinnedModuleInstance,
    PinnedModuleStore,
    PinnedModuleTarget,
)
from piper_offload.target_lease import _CudaTargetLease

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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
    first_instance = PinnedModuleStore.from_module(first).bind(first)
    second_instance = PinnedModuleStore.from_module(second).bind(second)
    device = torch.device("cuda")
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    lease = _CudaTargetLease.allocate(first_instance, device)

    lease.stage(first_instance, copy_stream)
    assert first.weight.device.type == "cpu"
    with torch.cuda.stream(compute_stream):
        first_instance.install_target(lease.acquire(compute_stream))
        actual_first = first(value.cuda())
    first_instance.install_pinned()
    lease.release()

    lease.stage(second_instance, copy_stream)
    assert second.weight.device.type == "cpu"
    with torch.cuda.stream(compute_stream):
        second_instance.install_target(lease.acquire(compute_stream))
        actual_second = second(value.cuda())
    second_instance.install_pinned()
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
    tracked_stream = Stream()
    consumer_stream = Stream()
    ready = Event()
    lease = _CudaTargetLease(
        cast(PinnedModuleTarget, object()),
        cast(torch.cuda.Stream, allocation_stream),
    )
    lease._ready_event = cast(torch.cuda.Event, ready)
    lease._acquired = True
    lease.mark_used(cast(torch.cuda.Stream, consumer_stream))
    lease.track_lifetime_stream(cast(torch.cuda.Stream, tracked_stream))

    lease.close()

    assert ready.synchronize_calls == 1
    assert allocation_stream.synchronize_calls == 1
    assert consumer_stream.synchronize_calls == 1
    assert tracked_stream.synchronize_calls == 1
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

    class Instance:
        def __init__(self) -> None:
            self.copied = False

        def copy_to_target(self, *_args: object, **_kwargs: object) -> None:
            self.copied = True

    allocation_stream = Stream()
    first_consumer = Stream()
    second_consumer = Stream()
    copy_stream = Stream()
    ready = object()
    instance = Instance()
    lease = _CudaTargetLease(
        cast(PinnedModuleTarget, object()),
        cast(torch.cuda.Stream, allocation_stream),
    )
    lease._ready_event = cast(torch.cuda.Event, ready)
    lease._staged = True
    lease.acquire(cast(torch.cuda.Stream, first_consumer))
    lease.mark_used(cast(torch.cuda.Stream, second_consumer))
    lease.release()
    monkeypatch.setattr(
        torch.cuda,
        "stream",
        lambda _stream: contextlib.nullcontext(),
    )

    lease.stage(
        cast(PinnedModuleInstance, instance),
        cast(torch.cuda.Stream, copy_stream),
    )

    assert copy_stream.waited_events == [ready]
    assert set(copy_stream.waited_streams) == {first_consumer, second_consumer}
    assert instance.copied
    assert lease._ready_event is copy_stream.recorded
