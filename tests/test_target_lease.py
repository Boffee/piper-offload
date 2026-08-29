"""CUDA target lease ordering and stream lifetime."""

from typing import cast

import pytest
import torch
from torch import nn

from piper_offload.pinned_module import PinnedModuleStore, PinnedModuleTarget
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
    lease.release(compute_stream)

    lease.stage(second_instance, copy_stream)
    assert second.weight.device.type == "cpu"
    with torch.cuda.stream(compute_stream):
        second_instance.install_target(lease.acquire(compute_stream))
        actual_second = second(value.cuda())
    second_instance.install_pinned()
    lease.release(compute_stream)
    lease.close()

    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual_first, expected_first)
    torch.testing.assert_close(actual_second, expected_second)


def test_close_hands_target_work_back_to_allocation_stream() -> None:
    class Stream:
        def __init__(self) -> None:
            self.waited_events: list[object] = []
            self.waited_streams: list[object] = []

        def wait_event(self, event: object) -> None:
            self.waited_events.append(event)

        def wait_stream(self, stream: object) -> None:
            self.waited_streams.append(stream)

    allocation_stream = Stream()
    tracked_stream = Stream()
    ready = object()
    lease = _CudaTargetLease(
        cast(PinnedModuleTarget, object()),
        cast(torch.cuda.Stream, allocation_stream),
    )
    lease._event = cast(torch.cuda.Event, ready)
    lease.track_stream(cast(torch.cuda.Stream, tracked_stream))

    lease.close()

    assert allocation_stream.waited_events == [ready]
    assert allocation_stream.waited_streams == [tracked_stream]
