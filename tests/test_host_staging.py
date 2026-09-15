"""Bounded host-to-device staging."""

import pytest
import torch

from piper_offload._host_staging import (
    _CHUNK_BYTES,
    _SLOT_COUNT,
    _HostStaging,
)

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA/HIP device required",
)


@CUDA
def test_staging_reuses_a_bounded_ping_pong_window() -> None:
    source = torch.arange(
        2 * _CHUNK_BYTES + 17,
        dtype=torch.uint8,
    )
    destination = torch.empty_like(source, device="cuda")
    staging = _HostStaging()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        assert staging.copy(destination, source, non_blocking=True)
    stream.synchronize()

    assert staging._slots is not None
    assert sum(slot.nbytes for slot in staging._slots) == (
        _SLOT_COUNT * _CHUNK_BYTES
    )
    torch.testing.assert_close(destination.cpu(), source)


@CUDA
def test_staging_preserves_blocking_copy_semantics() -> None:
    source = torch.arange(_CHUNK_BYTES + 1, dtype=torch.uint8)
    destination = torch.empty_like(source, device="cuda")
    staging = _HostStaging()

    assert staging.copy(destination, source, non_blocking=False)
    torch.testing.assert_close(destination.cpu(), source)


@CUDA
@pytest.mark.parametrize("pinned", [False, True])
def test_staging_packs_row_strided_source(pinned: bool) -> None:
    rows = 1025
    backing = torch.arange(rows * 16385, dtype=torch.uint8).view(rows, 16385)
    if pinned:
        backing = backing.pin_memory()
    source = backing[:, 1:8193]
    destination = torch.empty(source.shape, dtype=source.dtype, device="cuda")
    staging = _HostStaging()
    stream = torch.cuda.Stream()

    assert not source.is_contiguous()
    assert source.nbytes > _CHUNK_BYTES
    with torch.cuda.stream(stream):
        assert staging.copy(destination, source, non_blocking=True)
    stream.synchronize()
    torch.testing.assert_close(destination.cpu(), source)


@CUDA
def test_staging_reservation_is_idempotent() -> None:
    staging = _HostStaging()

    assert staging.reserve()
    slots = staging._slots
    assert slots is not None
    assert staging.reserve()
    assert staging._slots is slots


@CUDA
def test_staging_leaves_pinned_sources_on_the_direct_path() -> None:
    source = torch.ones(16, pin_memory=True)
    destination = torch.empty_like(source, device="cuda")

    assert not _HostStaging().copy(
        destination,
        source,
        non_blocking=True,
    )
