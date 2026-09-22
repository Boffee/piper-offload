"""Real HIP registration, owned-copy, and pageable-fallback checks."""

import mmap
import os
import warnings

import pytest
import torch

from piper_offload import PinManager

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available() or not hasattr(os, "preadv"),
    reason="Linux ROCm device and preadv required",
)


def _anonymous_bytes(pointer: int) -> int:
    """Return private page bytes for the mapping containing ``pointer``."""
    inside = False
    with open("/proc/self/smaps", encoding="utf-8") as smaps:
        for line in smaps:
            fields = line.split()
            if not fields[0].endswith(":"):
                start, end = (int(part, 16) for part in fields[0].split("-"))
                inside = start <= pointer < end
            elif inside and fields[0] == "Anonymous:":
                return int(fields[1]) * 1024
    raise AssertionError("mapping not found")


@pytest.mark.parametrize("loader", ["torch", "mmap", "safetensors"])
def test_private_mapping_registration_preserves_views_and_overlapping_leases(tmp_path, loader):
    expected = torch.arange(4096, dtype=torch.float32).reshape(128, 32)
    path = tmp_path / "weights.bin"
    if loader == "safetensors":
        safetensors = pytest.importorskip("safetensors.torch")
        safetensors.save_file({"weight": expected}, path)
        with safetensors.safe_open(path, framework="pt") as reader:
            tensor = reader.get_tensor("weight")
    else:
        path.write_bytes(b"#" * 128 + expected.numpy().tobytes())
        if loader == "torch":
            tensor = torch.from_file(str(path), shared=False, size=4128, dtype=torch.float32)[32:].reshape(128, 32)
        else:
            with path.open("rb") as file:
                mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)
            tensor = torch.frombuffer(mapping, dtype=torch.float32, count=4096, offset=128).reshape(128, 32)
    # Reader/file scope has ended; tensors must still own their mapping.
    original = path.read_bytes()
    views = [tensor[1::2, ::2], tensor.T[::3]]
    references = [expected[1::2, ::2], expected.T[::3]]
    size = tensor.untyped_storage().nbytes()
    manager = PinManager(size + 2 * mmap.PAGESIZE)
    stream = torch.cuda.Stream()
    try:
        for _ in range(2):
            first = manager.acquire(views)
            second = manager.acquire([tensor])
            try:
                assert first.registered_bytes == size
                assert second.registered_bytes == size
                assert manager.stats.registrations == 1
                assert manager.stats.active_leases == 2
                with torch.cuda.stream(stream):
                    targets = [view.to("cuda", non_blocking=True) for view in views]
                stream.synchronize()
                first.close()
                manager.clear()
                assert manager.stats.registrations == 1
                assert tensor.is_pinned()
                for actual, reference in zip(targets, references, strict=True):
                    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
            finally:
                stream.synchronize()
                first.close()
                second.close()
            manager.clear()
            assert manager.stats.pinned_bytes == 0
            assert not tensor.is_pinned()
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
            assert path.read_bytes() == original
    finally:
        manager.clear()


@pytest.mark.parametrize("read_limit", [4095, 65536])
def test_owned_anonymous_copy_fills_from_retained_file_and_transfers_views(tmp_path, read_limit):
    original = bytes(range(256)) * 256
    path = tmp_path / "weights.bin"
    path.write_bytes(original)
    manager = PinManager(len(original))
    stream = torch.cuda.Stream()
    with path.open("rb") as file:
        # Replacing the pathname must not change the source of a future fill.
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(b"\xff" * len(original))
        replacement.replace(path)
        for _ in range(2):
            allocation = mmap.mmap(-1, len(original), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            tensor = torch.frombuffer(allocation, dtype=torch.uint8).reshape(256, 256)
            assert tensor.data_ptr() % mmap.PAGESIZE == 0
            try:
                with manager.acquire([tensor]) as lease:
                    assert lease.registered_bytes == len(original)
                    with memoryview(allocation) as destination:
                        done = 0
                        while done < len(original):
                            with destination[done : done + read_limit] as remaining:
                                count = os.preadv(file.fileno(), [remaining], done)
                            assert count > 0
                            done += count
                    with torch.cuda.stream(stream):
                        target = tensor.T[::2].to("cuda", non_blocking=True)
                    stream.synchronize()
                expected = torch.arange(256, dtype=torch.uint8).repeat(256, 1).T[::2]
                torch.testing.assert_close(target.cpu(), expected, rtol=0, atol=0)
            finally:
                stream.synchronize()
                manager.clear()
            assert not tensor.is_pinned()
            del tensor
            allocation.close()
            assert manager.stats.pinned_bytes == 0
    assert path.read_bytes() == b"\xff" * len(original)


@pytest.mark.parametrize("non_blocking", [False, True])
def test_read_only_pageable_fallback_preserves_file_backing(tmp_path, non_blocking):
    original = bytes(range(256)) * 256
    path = tmp_path / "weights.bin"
    path.write_bytes(original)
    with path.open("rb") as file:
        mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        tensor = torch.frombuffer(mapping, dtype=torch.uint8)
    manager = PinManager(0)
    with manager.acquire([tensor]) as lease:
        assert lease.registered_bytes == 0
        assert lease.pageable_bytes == len(original)
        target = tensor.to("cuda", non_blocking=non_blocking)
        torch.cuda.synchronize()
    assert _anonymous_bytes(tensor.data_ptr()) == 0
    assert manager.stats.pinned_bytes == 0
    expected = torch.arange(256, dtype=torch.uint8).repeat(256)
    torch.testing.assert_close(target.cpu(), expected, rtol=0, atol=0)
    assert path.read_bytes() == original
