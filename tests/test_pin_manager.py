"""Registration ownership, page accounting, Tier 2a/2b arenas, and explicit pin leases."""

import ctypes
import gc
import logging
import mmap
import os
import random
import threading
import time
import weakref
from bisect import bisect_left
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import piper_offload._host_memory as host_memory
import piper_offload._host_memory_linux as linux_memory
import piper_offload._host_memory_windows as windows_memory
import piper_offload._host_registration as registration_module
import piper_offload.pin_manager as pin_module
from piper_offload import MappedCheckpoint, PinManager, file_slice, host_pin_manager
from piper_offload._direct_io import DEFAULT_COALESCE_THRESHOLD, DirectFileReader
from piper_offload._host_memory_windows import VirtualRegion
from piper_offload._host_registration import HostRegistrationError, RuntimeHostRegistration
from piper_offload.host_arena import ArenaAllocation, HostSlabArena, PinnedBounceRing
from piper_offload.pin_manager import PinLease, TransferLease

PAGE = mmap.PAGESIZE
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")


def _tensors(*ranges: tuple[int, int]) -> list[torch.Tensor]:
    size = max(start + length for start, length in ranges)
    buffer = mmap.mmap(-1, size)
    return [torch.frombuffer(buffer, dtype=torch.uint8, offset=start, count=length) for start, length in ranges]


class FakeBackend:
    def __init__(self) -> None:
        self.registered: dict[int, int] = {}
        self.register_calls: list[tuple[int, int]] = []
        self.unregister_calls: list[int] = []
        self.refuse: set[int] = set()
        self.register_errors: set[int] = set()
        self.unregister_errors: set[int] = set()
        self.capacity: int | None = None
        # Registration and memory calls in the order they happened.
        self.events: list[tuple[str, int]] = []

    def register(self, pointer: int, size: int) -> bool:
        self.register_calls.append((pointer, size))
        self.events.append(("register", pointer))
        if pointer in self.register_errors:
            raise HostRegistrationError("registration", 700)
        if pointer in self.refuse:
            return False
        if self.capacity is not None and sum(self.registered.values()) + size > self.capacity:
            return False
        assert pointer not in self.registered
        self.registered[pointer] = size
        return True

    def unregister(self, pointer: int) -> None:
        self.unregister_calls.append(pointer)
        self.events.append(("unregister", pointer))
        if pointer in self.unregister_errors:
            raise HostRegistrationError("unregistration", 700)
        del self.registered[pointer]


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def manager(backend: FakeBackend):
    result = PinManager(4 * PAGE, max_host_bytes=8 * PAGE, backend=backend)
    yield result
    backend.unregister_errors.clear()
    result.clear()


@pytest.fixture
def direct_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Every (offset, length) read through ``DirectFileReader.read_into``."""
    recorded: list[tuple[int, int]] = []
    original = DirectFileReader.read_into

    def recording(self: DirectFileReader, requests) -> None:
        for req in requests:
            recorded.append((req.source.offset, req.source.length))
        original(self, requests)

    monkeypatch.setattr(DirectFileReader, "read_into", recording)
    return recorded


def test_default_budget_is_finite_without_initializing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_runtime():
        raise AssertionError("construction and configuration must not initialize CUDA/HIP")

    monkeypatch.setattr(registration_module, "_load_runtime", unexpected_runtime)
    manager = PinManager()
    assert manager.max_pinned_bytes == pin_module._DEFAULT_PIN_BUDGET == host_memory.default_pin_budget()
    assert manager.max_host_bytes == pin_module._DEFAULT_HOST_BUDGET == host_memory.default_host_budget()
    assert manager.max_pinned_bytes is not None and manager.max_pinned_bytes > 0
    assert manager.max_host_bytes is not None and manager.max_host_bytes >= manager.max_pinned_bytes
    assert manager.max_pinned_bytes % PAGE == 0
    assert manager.max_host_bytes % PAGE == 0
    assert host_pin_manager.max_pinned_bytes == manager.max_pinned_bytes
    assert host_pin_manager.max_host_bytes == manager.max_host_bytes
    unbounded = PinManager(None, max_host_bytes=None)
    assert unbounded.max_pinned_bytes is None
    assert unbounded.max_host_bytes is None
    manager.max_pinned_bytes = PAGE
    manager.max_pinned_bytes = None
    manager.max_host_bytes = 2 * PAGE
    manager.max_host_bytes = None
    assert manager.stats.max_pinned_bytes is None
    assert manager.stats.max_host_bytes is None
    assert manager.stats.pinned_bytes == 0
    assert manager.stats.private_copy_bytes == 0
    assert manager.stats.bounce_bytes == 0


def test_default_budget_is_70_and_80_percent_on_linux(monkeypatch) -> None:
    monkeypatch.setattr(host_memory, "available_memory", linux_memory.available_memory)
    monkeypatch.setattr(host_memory, "DEFAULT_PIN_FRACTION", linux_memory.DEFAULT_PIN_FRACTION)
    monkeypatch.setattr(host_memory, "DEFAULT_HOST_FRACTION", linux_memory.DEFAULT_HOST_FRACTION)
    values = {"SC_PHYS_PAGES": 100, "SC_PAGE_SIZE": PAGE}
    monkeypatch.setattr(linux_memory.os, "sysconf", values.__getitem__, raising=False)
    monkeypatch.setattr(linux_memory, "_PROC_CGROUP", "/nonexistent/cgroup")
    assert host_memory.default_pin_budget() == 70 * PAGE
    assert host_memory.default_host_budget() == 80 * PAGE


def _cgroups(monkeypatch, tmp_path, membership: str, files: dict[str, str]) -> None:
    """Fake the process's cgroup membership and a cgroup mount holding ``files``."""
    monkeypatch.setattr(host_memory, "available_memory", linux_memory.available_memory)
    monkeypatch.setattr(host_memory, "DEFAULT_PIN_FRACTION", linux_memory.DEFAULT_PIN_FRACTION)
    monkeypatch.setattr(host_memory, "DEFAULT_HOST_FRACTION", linux_memory.DEFAULT_HOST_FRACTION)
    values = {"SC_PHYS_PAGES": 100, "SC_PAGE_SIZE": PAGE}
    monkeypatch.setattr(linux_memory.os, "sysconf", values.__getitem__, raising=False)
    (tmp_path / "cgroup").write_text(membership)
    for relative, text in files.items():
        path = tmp_path / "mount" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    monkeypatch.setattr(linux_memory, "_PROC_CGROUP", str(tmp_path / "cgroup"))
    monkeypatch.setattr(linux_memory, "_CGROUP_MOUNT", str(tmp_path / "mount"))


@pytest.mark.parametrize(
    ("membership", "files", "expected_pin_pages", "expected_host_pages"),
    [
        ("0::/a/b\n", {"a/b/memory.max": "20", "a/memory.max": "max", "memory.max": "max"}, 14, 16),
        ("0::/a/b\n", {"a/b/memory.max": "max", "a/memory.max": "10"}, 7, 8),
        ("0::/\n", {"memory.max": "30"}, 21, 24),
        ("0::/a\n", {"a/memory.max": "max", "memory.max": "max"}, 70, 80),
        ("4:memory:/a\n", {"memory/a/memory.limit_in_bytes": "20"}, 14, 16),
        ("4:memory:/a\n", {"memory/a/memory.limit_in_bytes": "9223372036854771712"}, 70, 80),
        ("3:cpu:/a\n", {"cpu/a/memory.max": "2"}, 70, 80),
    ],
    ids=[
        "v2-own", "v2-ancestor", "v2-namespaced-root", "v2-unlimited", "v1", "v1-unlimited", "no-memory-controller",
    ],
)
def test_default_budget_is_bounded_by_the_process_cgroup(
    monkeypatch, tmp_path, membership, files, expected_pin_pages, expected_host_pages,
) -> None:
    # Small numbers are page counts; the v1 unlimited sentinel is passed through.
    scaled = {
        name: str(int(text) * PAGE) if text.isdigit() and len(text) < 10 else text for name, text in files.items()
    }
    _cgroups(monkeypatch, tmp_path, membership, scaled)
    assert host_memory.default_pin_budget() == expected_pin_pages * PAGE
    assert host_memory.default_host_budget() == expected_host_pages * PAGE


@pytest.mark.parametrize("success", [False, True])
def test_windows_default_budget_uses_physical_ram_without_cuda(monkeypatch, success) -> None:
    def query(pointer):
        status = pointer._obj
        assert status.length == 64
        status.total_physical = 100 * PAGE
        return success

    monkeypatch.setattr(host_memory, "available_memory", windows_memory.available_memory)
    monkeypatch.setattr(host_memory, "DEFAULT_PIN_FRACTION", windows_memory.DEFAULT_PIN_FRACTION)
    monkeypatch.setattr(host_memory, "DEFAULT_HOST_FRACTION", windows_memory.DEFAULT_HOST_FRACTION)
    monkeypatch.setattr(
        windows_memory.ctypes, "WinDLL",
        lambda *_args, **_kwargs: SimpleNamespace(GlobalMemoryStatusEx=Mock(side_effect=query)), raising=False,
    )
    assert host_memory.default_pin_budget() == (45 * PAGE if success else 0)
    assert host_memory.default_host_budget() == (75 * PAGE if success else 0)


def test_unknown_physical_ram_disables_default_admission(monkeypatch) -> None:
    monkeypatch.setattr(host_memory, "available_memory", linux_memory.available_memory)
    monkeypatch.setattr(linux_memory.os, "sysconf", Mock(side_effect=OSError("unavailable")), raising=False)
    assert host_memory.default_pin_budget() == 0
    assert host_memory.default_host_budget() == 0


def test_zero_budget_disables_registration_without_initializing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_runtime():
        raise AssertionError("zero budget must not initialize CUDA/HIP")

    monkeypatch.setattr(registration_module, "_load_runtime", unexpected_runtime)
    manager = PinManager(0)
    tensor = torch.ones(8)
    with manager.acquire([tensor]) as lease:
        assert lease.registered_bytes == 0
        assert lease.pageable_bytes == tensor.nbytes
    assert manager.stats.pinned_bytes == 0


def test_aliases_share_whole_allocation_and_reference_counts(manager: PinManager, backend: FakeBackend) -> None:
    (tensor,) = _tensors((0, 2 * PAGE))
    view = tensor[64:128:2]
    first = manager.acquire([tensor, view, tensor])
    second = manager.acquire([view])
    assert backend.register_calls == [(tensor.data_ptr(), tensor.nbytes)]
    assert first.registered_bytes == second.registered_bytes == tensor.nbytes
    assert first.pageable_bytes == 0
    assert manager.stats.active_leases == 2

    first.close()
    first.close()
    manager.clear()
    assert backend.unregister_calls == []
    second.close()
    assert manager.stats.idle_registrations == 1
    with manager.acquire([tensor]):
        assert len(backend.register_calls) == 1
    manager.clear()
    assert backend.unregister_calls == [tensor.data_ptr()]


def test_lru_evicts_only_idle_registrations(backend: FakeBackend) -> None:
    manager = PinManager(2 * PAGE, backend=backend)
    a, b, c = _tensors((0, PAGE), (2 * PAGE, PAGE), (4 * PAGE, PAGE))
    for tensor in (a, b, a):
        with manager.acquire([tensor]):
            pass
    with manager.acquire([c]):
        assert backend.unregister_calls == [b.data_ptr()]
        assert manager.stats.pinned_bytes == 2 * PAGE
    manager.clear()


def test_an_acquisition_holds_existing_registrations_before_reserving(backend: FakeBackend) -> None:
    manager = PinManager(PAGE, backend=backend)
    cached, new = _tensors((0, PAGE), (2 * PAGE, PAGE))
    with manager.acquire([cached]):
        pass
    with manager.acquire([new, cached]) as lease:
        assert lease.registered_bytes == PAGE
        assert lease.pageable_bytes == PAGE
        assert backend.register_calls == [(cached.data_ptr(), PAGE)]
        assert backend.unregister_calls == []
    manager.clear()


def test_oversized_request_preserves_idle_cache(backend: FakeBackend) -> None:
    manager = PinManager(PAGE, backend=backend)
    cached, oversized = _tensors((0, PAGE), (2 * PAGE, 2 * PAGE))
    with manager.acquire([cached]):
        pass
    with manager.acquire([oversized]) as lease:
        assert lease.registered_bytes == 0
        assert lease.pageable_bytes == 2 * PAGE
        assert backend.unregister_calls == []
    manager.clear()


def test_shared_boundary_pages_are_reserved_once(backend: FakeBackend) -> None:
    manager = PinManager(PAGE, backend=backend)
    a, b = _tensors((64, 128), (512, 256))
    first, second = manager.acquire([a]), manager.acquire([b])
    assert manager.stats.registrations == 2
    assert manager.stats.pinned_bytes == PAGE
    assert backend.register_calls == [(a.data_ptr(), a.nbytes), (b.data_ptr(), b.nbytes)]
    first.close()
    manager.clear()
    assert manager.stats.pinned_bytes == PAGE
    assert set(backend.registered) == {b.data_ptr()}
    second.close()
    manager.clear()
    assert manager.stats.pinned_bytes == 0


def test_page_accounting_matches_union_through_release_and_eviction(backend: FakeBackend) -> None:
    manager = PinManager(32 * PAGE, backend=backend)
    tensors = _tensors((64, 128), (512, 256), (PAGE - 100, 400), (2 * PAGE + 100, 3 * PAGE), (8 * PAGE, PAGE))
    leases = [manager.acquire([tensor]) for tensor in tensors]

    def expected_bytes() -> int:
        pages = set()
        for pointer, size in backend.registered.items():
            pages.update(range(pointer // PAGE, (pointer + size - 1) // PAGE + 1))
        return len(pages) * PAGE

    assert manager.stats.pinned_bytes == expected_bytes()
    for index in (2, 0, 3, 1, 4):
        leases[index].close()
        manager.clear()
        assert manager.stats.pinned_bytes == expected_bytes()


def test_budget_reduction_waits_for_active_leases(manager: PinManager, backend: FakeBackend) -> None:
    (tensor,) = _tensors((0, 2 * PAGE))
    lease = manager.acquire([tensor])
    manager.max_pinned_bytes = 0
    assert manager.stats.pinned_bytes == 2 * PAGE
    assert backend.unregister_calls == []
    lease.close()
    assert manager.stats.pinned_bytes == 0


def test_capacity_failure_stops_later_registration_attempts(manager: PinManager, backend: FakeBackend) -> None:
    a, b, c = _tensors((0, PAGE), (2 * PAGE, PAGE), (4 * PAGE, PAGE))
    backend.refuse.add(a.data_ptr())
    with manager.acquire([a, b, c]) as lease:
        assert lease.registered_bytes == 0
        assert lease.pageable_bytes == 3 * PAGE
        assert manager.stats.registration_failures == 1
        assert backend.register_calls == [(a.data_ptr(), PAGE)]


def test_unbounded_budget_evicts_idle_lru_and_retries(backend: FakeBackend) -> None:
    manager = PinManager(None, backend=backend)
    a, b, c = _tensors((0, PAGE), (2 * PAGE, PAGE), (4 * PAGE, PAGE))
    backend.capacity = 2 * PAGE
    for tensor in (a, b):
        with manager.acquire([tensor]):
            pass

    with manager.acquire([c]) as lease:
        assert lease.registered_bytes == PAGE
        assert lease.pageable_bytes == 0
        assert backend.unregister_calls == [a.data_ptr()]
        assert set(backend.registered) == {b.data_ptr(), c.data_ptr()}
        assert manager.stats.max_pinned_bytes is None
        assert manager.stats.registration_failures == 1
    manager.clear()


def test_eviction_for_a_runtime_retry_keeps_the_requested_idle_registration(backend: FakeBackend) -> None:
    manager = PinManager(None, backend=backend)
    requested, unrelated, new = _tensors(
        (0, PAGE),
        (2 * PAGE, PAGE),
        (4 * PAGE, PAGE),
    )
    backend.capacity = 2 * PAGE
    for tensor in (requested, unrelated):
        with manager.acquire([tensor]):
            pass

    with manager.acquire([requested, new]) as lease:
        assert lease.registered_bytes == 2 * PAGE
        assert backend.unregister_calls == [unrelated.data_ptr()]
        assert set(backend.registered) == {requested.data_ptr(), new.data_ptr()}
    manager.clear()


def test_budget_can_switch_between_finite_and_opportunistic(backend: FakeBackend) -> None:
    manager = PinManager(PAGE, backend=backend)
    a, b = _tensors((0, PAGE), (2 * PAGE, PAGE))
    with manager.acquire([a]):
        pass

    manager.max_pinned_bytes = None
    with manager.acquire([b]):
        assert manager.stats.pinned_bytes == 2 * PAGE
    assert manager.max_pinned_bytes is None

    manager.max_pinned_bytes = PAGE
    assert manager.stats.pinned_bytes == PAGE
    manager.clear()


def test_pageable_storage_waits_for_all_active_leases_before_registration(backend: FakeBackend) -> None:
    manager = PinManager(0, backend=backend)
    (tensor,) = _tensors((0, PAGE))
    first = manager.acquire([tensor])
    manager.max_pinned_bytes = PAGE
    second = manager.acquire([tensor[64:128]])
    first.close()
    with manager.acquire([tensor]) as third:
        assert second.pageable_bytes == third.pageable_bytes == PAGE
        assert backend.register_calls == []
    second.close()
    with manager.acquire([tensor]) as fourth:
        assert fourth.registered_bytes == PAGE
        assert len(backend.register_calls) == 1
    manager.clear()


def test_active_pageable_ranges_also_reject_partial_overlaps(backend: FakeBackend) -> None:
    manager = PinManager(0, backend=backend)
    a, b = _tensors((0, 2 * PAGE), (PAGE, 2 * PAGE))
    with manager.acquire([a]):
        manager.max_pinned_bytes = 4 * PAGE
        with pytest.raises(ValueError, match="Overlapping"):
            manager.acquire([b])
        assert backend.register_calls == []
    with manager.acquire([b]) as lease:
        assert lease.registered_bytes == b.nbytes
    manager.clear()


def test_unexpected_registration_error_rolls_back_new_registrations(manager: PinManager, backend: FakeBackend) -> None:
    a, b = _tensors((0, PAGE), (2 * PAGE, PAGE))
    backend.register_errors.add(b.data_ptr())
    with pytest.raises(HostRegistrationError):
        manager.acquire([a, b])
    assert not backend.registered
    assert manager.stats.pinned_bytes == 0
    assert manager.stats.active_leases == 0


def test_validation_finishes_before_registration(manager: PinManager, backend: FakeBackend) -> None:
    (tensor,) = _tensors((0, PAGE))
    with pytest.raises(ValueError, match="CPU"):
        manager.acquire([tensor, torch.empty(8, device="meta")])
    assert backend.register_calls == []
    with pytest.raises(ValueError, match="strided"):
        manager.acquire([torch.empty(8).to_sparse()])
    with pytest.raises(TypeError, match="plain"):
        manager.acquire([object()])


def test_distinct_overlapping_storage_ranges_are_rejected_before_mutation(
    manager: PinManager,
    backend: FakeBackend,
) -> None:
    a, b, same_start = _tensors((0, 2 * PAGE), (PAGE, 2 * PAGE), (0, PAGE))
    for other in (b, same_start):
        with pytest.raises(ValueError, match="Overlapping"):
            manager.acquire([a, other])
        assert backend.register_calls == []
    with manager.acquire([a]):
        with pytest.raises(ValueError, match="Overlapping"):
            manager.acquire([b])
        assert len(backend.register_calls) == 1
        assert not backend.unregister_calls


def test_empty_views_do_not_register_their_backing(manager: PinManager, backend: FakeBackend) -> None:
    (tensor,) = _tensors((0, PAGE))
    with manager.acquire([tensor[:0], torch.empty(0)]) as lease:
        assert lease.registered_bytes == lease.pageable_bytes == 0
    assert backend.register_calls == []


def test_owner_disposal_unregisters_before_storage_dies(manager: PinManager, backend: FakeBackend) -> None:
    (tensor,) = _tensors((0, PAGE))
    pointer = tensor.data_ptr()
    tensor_ref = weakref.ref(tensor)
    storage_ref = weakref.ref(tensor.untyped_storage())
    with manager.acquire([tensor]):
        pass
    assert storage_ref() is not None
    del tensor
    gc.collect()
    assert tensor_ref() is None
    assert backend.unregister_calls == [pointer]
    assert storage_ref() is None
    assert manager.stats.pinned_bytes == 0


def test_disposed_owner_waits_for_an_alias_lease(manager: PinManager, backend: FakeBackend) -> None:
    (tensor,) = _tensors((0, PAGE))
    alias = tensor[16:32].detach()
    with manager.acquire([tensor]):
        pass
    lease = manager.acquire([alias])
    del tensor
    gc.collect()
    assert backend.unregister_calls == []
    lease.close()
    assert len(backend.unregister_calls) == 1
    assert manager.stats.pinned_bytes == 0


def test_failed_unregistration_retains_storage_and_reservation_for_retry(
    manager: PinManager, backend: FakeBackend,
) -> None:
    (tensor,) = _tensors((0, PAGE))
    storage_ref = weakref.ref(tensor.untyped_storage())
    pointer = tensor.data_ptr()
    with manager.acquire([tensor]):
        pass
    backend.unregister_errors.add(pointer)
    del tensor
    gc.collect()
    assert storage_ref() is not None
    assert manager.stats.pinned_bytes == PAGE
    with pytest.raises(RuntimeError, match="remains retained"):
        manager.clear()
    backend.unregister_errors.clear()
    manager.clear()
    assert storage_ref() is None
    assert manager.stats.pinned_bytes == 0


def test_failed_eviction_does_not_oversubscribe_budget(backend: FakeBackend) -> None:
    manager = PinManager(PAGE, backend=backend)
    a, b = _tensors((0, PAGE), (2 * PAGE, PAGE))
    with manager.acquire([a]):
        pass
    backend.unregister_errors.add(a.data_ptr())
    with manager.acquire([b]) as lease:
        assert lease.pageable_bytes == PAGE
        assert manager.stats.pinned_bytes == PAGE
        assert set(backend.registered) == {a.data_ptr()}
    backend.unregister_errors.clear()
    manager.clear()


def test_lease_holds_pageable_storage_until_close(backend: FakeBackend) -> None:
    manager = PinManager(0, backend=backend)
    tensor = torch.ones(8)
    tensor_ref = weakref.ref(tensor)
    lease = manager.acquire([tensor])
    del tensor
    gc.collect()
    assert tensor_ref() is not None
    assert manager.stats.active_leases == 1
    lease.close()
    assert tensor_ref() is None
    assert manager.stats.active_leases == 0


def test_a_dropped_lease_releases_its_registrations(manager: PinManager) -> None:
    (tensor,) = _tensors((0, PAGE))
    lease = manager.acquire([tensor])
    del lease
    gc.collect()
    assert manager.stats.active_leases == 0
    assert manager.stats.idle_registrations == 1


def test_registration_keeps_manager_alive_until_owner_disposal(backend: FakeBackend) -> None:
    manager = PinManager(PAGE, backend=backend)
    (tensor,) = _tensors((0, PAGE))
    manager_ref = weakref.ref(manager)
    with manager.acquire([tensor]):
        pass
    del manager
    gc.collect()
    assert manager_ref() is not None
    assert backend.registered
    del tensor
    gc.collect()
    assert manager_ref() is None
    assert not backend.registered


def test_negative_budget_is_rejected(manager: PinManager) -> None:
    with pytest.raises(ValueError, match=">= 0"):
        PinManager(-1)
    with pytest.raises(ValueError, match=">= 0"):
        manager.max_pinned_bytes = -1

def test_concurrent_leases_share_one_registration(manager: PinManager, backend: FakeBackend) -> None:
    (tensor,) = _tensors((0, PAGE))
    barrier = threading.Barrier(5)

    def use_storage() -> None:
        with manager.acquire([tensor]):
            barrier.wait(timeout=10)
            barrier.wait(timeout=10)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(use_storage) for _ in range(4)]
        barrier.wait(timeout=10)
        assert manager.stats.active_leases == 4
        assert len(backend.register_calls) == 1
        barrier.wait(timeout=10)
        for future in futures:
            future.result()
    assert manager.stats.idle_registrations == 1


class FakeRuntime:
    def __init__(self, code: int) -> None:
        self.code = code
        self.flags: int | None = None
        self.last_error = 0
        self.unregistered: list[int] = []

    def register(self, pointer: int, size: int, flags: int) -> int:
        self.flags = flags
        self.last_error = self.code
        return self.code

    def unregister(self, pointer: int) -> int:
        self.unregistered.append(pointer)
        self.last_error = self.code
        return self.code

    def get_last_error(self) -> int:
        result, self.last_error = self.last_error, 0
        return result


@pytest.mark.parametrize("checkpoint", [False, True])
def test_rejected_range_stays_pageable_without_evicting_or_skipping_other_storage(
    checkpoint: bool, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(0)
    register = runtime.register
    attempts = []

    def refuse_second(pointer: int, size: int, flags: int) -> int:
        runtime.code = int(len(attempts) == 1)
        attempts.append((pointer, size))
        return register(pointer, size, flags)

    monkeypatch.setattr(runtime, "register", refuse_second)
    monkeypatch.setattr(registration_module, "_load_runtime", lambda: runtime)
    manager = PinManager(4 * PAGE)
    cached, rejected, fresh = _tensors((0, PAGE), (2 * PAGE, PAGE), (4 * PAGE, PAGE))
    if checkpoint:
        rejected = _checkpoint(tmp_path, PAGE)
    try:
        with manager.acquire([cached]):
            pass
        with manager.acquire([rejected, fresh]) as lease:
            assert (lease.registered_bytes, lease.pageable_bytes) == (PAGE, PAGE)
            assert len(attempts) == 3
            assert attempts[0] == (cached.data_ptr(), PAGE)
            assert attempts[2] == (fresh.data_ptr(), PAGE)
            assert runtime.unregistered == []
            assert runtime.last_error == 0
            assert manager.stats.pinned_bytes == 2 * PAGE
            assert manager.stats.copy_bytes == 0
            assert manager.stats.registration_failures == 1
            torch.testing.assert_close(_transferred(manager, rejected), rejected)
        with manager.acquire([cached, fresh]) as lease:
            assert lease.registered_bytes == 2 * PAGE
            assert len(attempts) == 3
    finally:
        runtime.code = 0
        manager.clear()
    assert set(runtime.unregistered) == {cached.data_ptr(), fresh.data_ptr()}
    assert manager.stats.pinned_bytes == 0
    assert manager.stats.active_leases == 0


@pytest.mark.parametrize("code", [0, 2, 801, 1, 700, 712])
def test_runtime_backend_classifies_registration_errors(
    code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(code)
    monkeypatch.setattr(registration_module, "_load_runtime", lambda: runtime)
    backend = RuntimeHostRegistration()
    if code in (0, 2, 801):
        assert backend.register(PAGE, PAGE) == (code == 0)
    else:
        with pytest.raises(HostRegistrationError) as error:
            backend.register(PAGE, PAGE)
        assert error.value.code == code
    assert runtime.flags == 1
    assert runtime.last_error == 0
    if code == 0:
        backend.unregister(PAGE)
    else:
        with pytest.raises(HostRegistrationError):
            backend.unregister(PAGE)
    assert runtime.last_error == 0


def test_runtime_backend_without_device_does_not_initialize_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_runtime():
        raise AssertionError("CPU fallback must not initialize CUDA/HIP")

    monkeypatch.setattr(torch.cuda, "cudart", unexpected_runtime)
    assert not RuntimeHostRegistration().register(PAGE, PAGE)

@pytest.mark.parametrize("code", [1, 700])
def test_prior_runtime_error_is_reported_before_registering(code: int, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(0)
    runtime.last_error = code
    monkeypatch.setattr(registration_module, "_load_runtime", lambda: runtime)
    manager = PinManager(PAGE)
    (tensor,) = _tensors((0, PAGE))
    with pytest.raises(HostRegistrationError, match="prior runtime work") as error:
        manager.acquire([tensor])
    assert error.value.code == code
    assert runtime.flags is None
    assert manager.stats.active_leases == manager.stats.pinned_bytes == 0


def test_stale_runtime_error_does_not_block_unregistration(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    runtime = FakeRuntime(0)
    runtime.last_error = 700
    monkeypatch.setattr(registration_module, "_load_runtime", lambda: runtime)
    with caplog.at_level(logging.WARNING, logger="piper_offload._host_registration"):
        RuntimeHostRegistration().unregister(PAGE)
    assert runtime.unregistered == [PAGE]
    assert runtime.last_error == 0
    assert "700" in caplog.text


@pytest.mark.parametrize(
    ("hip", "filename", "prefix"),
    [
        (None, "libcudart.so.13", "cuda"),
        (None, "cudart64_13.dll", "cuda"),
        ("7.2.0", "libamdhip64.so.7", "hip"),
        ("7.2.0", "amdhip64_7.dll", "hip"),
    ],
)
def test_loader_binds_the_loaded_runtime_with_native_pointer_widths(
    hip: str | None,
    filename: str,
    prefix: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register, unregister, get_error = (Mock(return_value=0) for _ in range(3))
    library = SimpleNamespace(**{
        f"{prefix}HostRegister": register,
        f"{prefix}HostUnregister": unregister,
        f"{prefix}GetLastError": get_error,
    })
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(registration_module, "dllist", lambda: ["/loaded/libtorch.so", filename])
    load = Mock(return_value=library)
    monkeypatch.setattr(registration_module.ctypes, "CDLL", load)
    backend = RuntimeHostRegistration()
    assert backend.register(2**40, PAGE)
    backend.unregister(2**40)
    load.assert_called_once_with(filename)
    register.assert_called_once_with(2**40, PAGE, 1)
    unregister.assert_called_once_with(2**40)
    assert register.argtypes == [
        registration_module.ctypes.c_void_p,
        registration_module.ctypes.c_size_t,
        registration_module.ctypes.c_uint,
    ]


@CUDA
def test_real_failed_registration_does_not_poison_later_torch_work() -> None:
    backend = RuntimeHostRegistration()
    with pytest.raises(HostRegistrationError):
        backend.register(0, 0)
    # This kernel used to report the registration's leftover invalid-argument
    # error when using PyTorch's partial cudart bindings directly.
    result = torch.ones(8, device="cuda")
    torch.cuda.synchronize()
    assert result.sum().item() == 8


@CUDA
def test_real_registration_copy_and_unregistration() -> None:
    manager = PinManager(4 * PAGE)
    (tensor,) = _tensors((64, 2 * PAGE))
    tensor.fill_(17)
    pointer = tensor.data_ptr()
    stream = torch.cuda.Stream()
    lease = manager.acquire([tensor])
    try:
        assert lease.registered_bytes == tensor.nbytes
        assert tensor.is_pinned()
        with torch.cuda.stream(stream):
            target = tensor.to("cuda", non_blocking=True)
        stream.synchronize()
        lease.close()
        assert tensor.data_ptr() == pointer
        torch.testing.assert_close(target.cpu(), tensor)
    finally:
        lease.close()
        manager.clear()
    assert not tensor.is_pinned()


@CUDA
def test_real_fallback_can_copy_allocation_sharing_a_registered_page() -> None:
    manager = PinManager(PAGE)
    pinned, pageable = _tensors((64, 128), (512, 2 * PAGE))
    pageable.fill_(23)
    with manager.acquire([pinned, pageable]) as lease:
        assert lease.registered_bytes == pinned.nbytes
        assert lease.pageable_bytes == pageable.nbytes
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            target = pageable.to("cuda", non_blocking=True)
        stream.synchronize()
    manager.clear()
    torch.testing.assert_close(target.cpu(), pageable)


@CUDA
def test_real_foreign_registration_is_never_unregistered() -> None:
    backend = RuntimeHostRegistration()
    (tensor,) = _tensors((0, PAGE))
    pointer = tensor.data_ptr()
    assert backend.register(pointer, tensor.nbytes)
    manager = PinManager(PAGE)
    try:
        with pytest.raises(HostRegistrationError) as error:
            manager.acquire([tensor])
        assert error.value.code == 712
        manager.clear()
        assert tensor.is_pinned()
    finally:
        backend.unregister(pointer)


def test_pageable_lease_registers_nothing_and_holds_its_storage(backend: FakeBackend) -> None:
    manager = PinManager(4 * PAGE, backend=backend)
    first, second = _tensors((0, PAGE), (PAGE, PAGE))
    with manager.acquire([first, second], pin=False) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert backend.register_calls == []
        assert manager.stats.active_leases == 1
        assert manager.stats.pinned_bytes == 0
    assert manager.stats.active_leases == 0
    manager.clear()


@pytest.mark.parametrize("pin", [False, True], ids=["pageable", "over-budget"])
def test_pageable_acquisition_claims_storage_before_unlocking(backend, monkeypatch, request, pin) -> None:
    manager = PinManager(0 if pin else 2 * PAGE, backend=backend)
    request.addfinalizer(manager.clear)
    (tensor,) = _tensors((0, PAGE))
    unlocked = threading.Event()
    competing_acquire_done = threading.Event()
    lock = manager._lock

    class PauseAtFirstUnlock:
        def __enter__(self):
            lock.acquire()

        def __exit__(self, *_exc):
            lock.release()
            if not unlocked.is_set():
                unlocked.set()
                assert competing_acquire_done.wait(5)

    monkeypatch.setattr(manager, "_lock", PauseAtFirstUnlock())
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(manager.acquire, [tensor], pin=pin)
        try:
            assert unlocked.wait(5)
            manager.max_pinned_bytes = 2 * PAGE
            with manager.acquire([tensor]) as competing:
                assert competing.pageable_bytes == PAGE
                assert backend.register_calls == []
        finally:
            competing_acquire_done.set()
            first.result(timeout=5).close()
    with manager.acquire([tensor]) as later:
        assert later.registered_bytes == PAGE


def test_pageable_lease_holds_existing_registrations_out_of_eviction(backend: FakeBackend) -> None:
    manager = PinManager(2 * PAGE, backend=backend)
    held, fresh, extra = _tensors((0, PAGE), (PAGE, PAGE), (2 * PAGE, PAGE))
    with manager.acquire([held]):
        pass
    assert manager.stats.registrations == 1
    with manager.acquire([held], pin=False) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (PAGE, 0)
        with manager.acquire([fresh, extra]) as pinning:
            # The held entry is not idle, so only one more page fits the budget.
            assert (pinning.registered_bytes, pinning.pageable_bytes) == (PAGE, PAGE)
        assert backend.unregister_calls == []
    with manager.acquire([extra]) as later:
        # Once the pageable lease closes, its entry is idle and evictable again.
        assert later.registered_bytes == PAGE
        assert len(backend.unregister_calls) == 1
    manager.clear()


def test_transfer_lease_holds_its_storage_until_synchronization_succeeds(backend: FakeBackend, monkeypatch) -> None:
    manager = PinManager(backend=backend)
    (source,) = _tensors((0, PAGE))
    device = torch.device("cuda", 3)
    synchronized: list[torch.device] = []
    failing = True

    def synchronize(sync_device: torch.device) -> None:
        synchronized.append(sync_device)
        if failing:
            raise RuntimeError("injected synchronization failure")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    transfer = TransferLease()
    transfer.start(manager, [source], device)
    assert transfer.open and manager.stats.active_leases == 1
    assert backend.register_calls == []
    with pytest.raises(RuntimeError, match="still unfinished"):
        transfer.start(manager, [source], device)
    with pytest.raises(RuntimeError, match="injected"):
        transfer.finish()
    assert transfer.open and manager.stats.active_leases == 1
    failing = False
    transfer.finish()
    assert not transfer.open and manager.stats.active_leases == 0
    # Retries synchronize the transfer's own device, and a closed lease is inert.
    assert synchronized == [device, device]
    transfer.finish()
    assert synchronized == [device, device]


def _checkpoint(tmp_path, nbytes: int, name: str = "model") -> torch.Tensor:
    """A tensor over a read-only ``MappedCheckpoint`` mapping whose bytes follow a checkable pattern."""
    path = tmp_path / f"{name}.safetensors"
    header = b'{"w": {"dtype": "U8", "shape": [%d], "data_offsets": [0, %d]}}' % (nbytes, nbytes)
    header += b" " * (-(8 + len(header)) % 8)
    payload = (bytes(range(256)) * (nbytes // 256 + 1))[:nbytes]
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
    return MappedCheckpoint(path).get_tensor("w")


def _transferred(manager: PinManager, tensor: torch.Tensor) -> torch.Tensor:
    """What a synchronous transfer of ``tensor`` delivers."""
    destination = torch.zeros_like(tensor)
    manager.transfer(destination, tensor, non_blocking=False)
    return destination


def test_checkpoint_storage_pins_through_an_owned_copy(backend: FakeBackend, tmp_path) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE - 100)
    assert file_slice(tensor) is not None
    manager = PinManager(4 * PAGE, max_host_bytes=8 * PAGE, backend=backend)
    (anonymous,) = _tensors((0, PAGE))
    with manager.acquire([tensor, anonymous]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (tensor.nbytes + PAGE, 0)
        calls = dict(backend.register_calls)
        assert calls.pop(anonymous.data_ptr()) == PAGE
        # The mapping itself is never registered; its copy lives inside a page-aligned Tier 2a slab.
        ((copy_pointer, copy_size),) = calls.items()
        assert copy_pointer != tensor.data_ptr() and copy_pointer % PAGE == 0 and copy_size == 2 * PAGE
        assert manager.stats.pinned_bytes == 3 * PAGE and manager.stats.copy_bytes == 2 * PAGE
        alloc = manager._registrations[tensor.untyped_storage().data_ptr()].copy
        assert isinstance(alloc, ArenaAllocation)
        assert not alloc.released
        torch.testing.assert_close(_transferred(manager, tensor), tensor)
        # Transfers read the Tier 2a copy, not the mapping: a byte changed in the copy
        # shows up in what a transfer delivers, through a view's geometry too.
        with alloc.payload_view() as bytes_:
            bytes_[300] = (tensor[300].item() + 1) % 256
            expected = torch.frombuffer(bytearray(bytes_[300:1000]), dtype=torch.uint8).view(torch.int16)
        assert _transferred(manager, tensor)[300] != tensor[300]
        part = tensor[300:1000].view(torch.int16)
        torch.testing.assert_close(_transferred(manager, part), expected)
        torch.testing.assert_close(_transferred(manager, anonymous), anonymous)
    manager.clear()
    assert set(backend.unregister_calls) == {copy_pointer, anonymous.data_ptr()}
    assert alloc.released
    assert manager.stats.pinned_bytes == 0 and manager.stats.copy_bytes == 0


@pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm device required")
@pytest.mark.parametrize("budget", [0, 2 * PAGE])
def test_real_rocm_checkpoint_transfer_uses_owned_copy_or_pageable_fallback(tmp_path, budget: int) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE - 100)
    source = tensor[100:1100]
    original = tensor.clone()
    manager = PinManager(budget)
    destination = torch.empty_like(source, device="cuda")
    stream = torch.cuda.Stream()
    try:
        with manager.acquire([tensor]) as lease:
            assert lease.registered_bytes == (tensor.nbytes if budget else 0)
            assert manager.stats.copy_bytes == budget
            with torch.cuda.stream(stream):
                manager.transfer(destination, source, non_blocking=True)
            stream.synchronize()
        torch.testing.assert_close(destination.cpu(), original[100:1100], rtol=0, atol=0)
    finally:
        stream.synchronize()
        manager.clear()
    assert manager.stats.pinned_bytes == manager.stats.copy_bytes == 0
    torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def test_copy_eviction_reuses_pinned_slab_without_unregistration(backend: FakeBackend, tmp_path) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(2 * PAGE, max_host_bytes=0, backend=backend)
    with manager.acquire([first, second]) as lease:
        # Only one copy fits, and it is reserved before the second is considered.
        assert (lease.registered_bytes, lease.pageable_bytes) == (2 * PAGE, 2 * PAGE)
        assert len(backend.register_calls) == 1
        alloc = manager._registrations[first.untyped_storage().data_ptr()].copy
        assert isinstance(alloc, ArenaAllocation)
    with manager.acquire([second]) as lease2:
        # Evicting the idle span releases it back to the pre-pinned slab for reuse (0 unregistrations).
        assert alloc.released
        assert lease2.registered_bytes == 2 * PAGE
        assert backend.unregister_calls == []
        assert len(backend.register_calls) == 1
        assert manager.stats.copy_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, second), second)
    manager.clear()
    assert backend.unregister_calls == [backend.register_calls[0][0]]


@pytest.mark.skipif(not hasattr(os, "preadv"), reason="positional reads")
def test_sliced_fill_covers_the_copy_through_short_reads(backend: FakeBackend, tmp_path, monkeypatch) -> None:
    tensor = _checkpoint(tmp_path, 8 * PAGE)
    fs = file_slice(tensor)
    assert fs is not None
    original = os.preadv
    reads: list[tuple[int, int]] = []
    lock = threading.Lock()

    def recording_preadv(fd, buffers, offset):
        (buffer,) = buffers
        count = original(fd, [buffer[:100]], offset)  # short reads inside every slice too
        with lock:
            reads.append((offset, count))
        return count

    monkeypatch.setattr(os, "preadv", recording_preadv)
    manager = PinManager(16 * PAGE, backend=backend)
    with manager.acquire([tensor]) as lease:
        assert lease.registered_bytes == 8 * PAGE
        assert manager.stats.copy_bytes == 8 * PAGE
        torch.testing.assert_close(_transferred(manager, tensor), tensor)
    assert len(reads) > 1
    covered = 0
    for start, end in sorted((off, off + cnt) for off, cnt in reads if cnt > 0):
        target = fs.offset + covered
        if start <= target < end:
            covered = max(covered, end - fs.offset)
    assert covered >= fs.length
    manager.clear()


def test_copies_of_one_acquisition_pack_into_slab_and_batch_read(
    backend: FakeBackend, tmp_path, direct_reads,
) -> None:
    first = _checkpoint(tmp_path, 8 * PAGE, "first")
    second = _checkpoint(tmp_path, 4 * PAGE, "second")
    third = _checkpoint(tmp_path, 2 * PAGE, "third")
    manager = PinManager(16 * PAGE, backend=backend)
    with manager.acquire([first, second, third]) as lease:
        assert lease.registered_bytes == 14 * PAGE
        assert len(backend.register_calls) == 1
        assert backend.register_calls[0][1] == 14 * PAGE
        assert len(direct_reads) == 3
        for tensor in (first, second, third):
            torch.testing.assert_close(_transferred(manager, tensor), tensor)
    manager.clear()


def test_failed_fill_leaves_the_storage_pageable(backend: FakeBackend, tmp_path, monkeypatch, caplog) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)

    def broken(_self, _requests) -> None:
        raise OSError("injected read failure")

    monkeypatch.setattr(DirectFileReader, "read_into", broken)
    manager = PinManager(4 * PAGE, backend=backend)
    with caplog.at_level(logging.WARNING, logger="piper_offload"), manager.acquire([tensor]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert backend.registered == {} and manager.stats.pinned_bytes == 0 and manager.stats.copy_bytes == 0
    assert "injected read failure" in caplog.text
    manager.clear()


def test_copy_registration_error_rolls_back_the_acquisition(backend: FakeBackend, tmp_path, monkeypatch) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(8 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    register = backend.register

    def broken_after_one(pointer: int, size: int) -> bool:
        if backend.register_calls:
            raise HostRegistrationError("registration", 700)
        return register(pointer, size)

    with monkeypatch.context() as patch:
        patch.setattr(backend, "register", broken_after_one)
        with pytest.raises(HostRegistrationError):
            manager.acquire([first, second])
    # The first slab was registered and is retired; the second was never registered. Both are freed.
    assert backend.registered == {} and len(backend.unregister_calls) == 1
    assert manager.stats.pinned_bytes == 0 and manager.stats.copy_bytes == 0
    assert manager.stats.active_leases == 0
    with manager.acquire([first, second]) as lease:
        assert lease.registered_bytes == 4 * PAGE
    manager.clear()


def test_copy_allocation_failure_leaves_the_storage_pageable(
    backend: FakeBackend, tmp_path, monkeypatch, caplog,
) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)

    def exhausted(_size: int):
        raise MemoryError("injected allocation failure")

    manager = PinManager(4 * PAGE, backend=backend)
    monkeypatch.setattr(manager._tier2a, "_allocator", exhausted)
    with caplog.at_level(logging.WARNING, logger="piper_offload"), manager.acquire([tensor]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert backend.register_calls == [] and manager.stats.pinned_bytes == 0
    assert "injected allocation failure" in caplog.text
    manager.clear()


def test_a_buffered_allocation_warning_keeps_no_storage_alive(
    backend: FakeBackend, tmp_path, monkeypatch, caplog,
) -> None:
    """caplog keeps every record, as an application that buffers warnings does."""

    def exhausted(_size: int):
        raise MemoryError("injected allocation failure")

    tensor = _checkpoint(tmp_path, 2 * PAGE)
    alive = weakref.ref(tensor)
    manager = PinManager(4 * PAGE, backend=backend)
    monkeypatch.setattr(manager._tier2a, "_allocator", exhausted)
    with caplog.at_level(logging.WARNING, logger="piper_offload"), manager.acquire([tensor]):
        pass
    del tensor
    gc.collect()
    assert "injected allocation failure" in caplog.text
    # The record keeps the error's text, not the frames of the acquisition that held the tensor.
    assert alive() is None
    manager.clear()


def test_a_pinning_acquisition_waits_for_a_pending_fill_while_a_pageable_one_proceeds(
    backend: FakeBackend, tmp_path, monkeypatch,
) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    pinning, pageable = _tensors((0, PAGE), (2 * PAGE, PAGE))
    manager = PinManager(8 * PAGE, backend=backend)
    original = manager._reader.read_into
    leases: dict[str, PinLease] = {}
    pinner = threading.Thread(target=lambda: leases.__setitem__("pinning", manager.acquire([pinning])))
    pager = threading.Thread(target=lambda: leases.__setitem__("pageable", manager.acquire([pageable], pin=False)))
    during_fill: list[bool] = []

    def fill_while_others_acquire(requests):
        pinner.start()
        pager.start()
        pager.join(2.0)
        pinner.join(0.2)
        during_fill.extend((pager.is_alive(), pinner.is_alive()))
        original(requests)

    monkeypatch.setattr(manager._reader, "read_into", fill_while_others_acquire)
    with manager.acquire([tensor]) as lease:
        pinner.join()
        # The pageable acquisition of other storage completed during the fill;
        # the pinning one waited, so the copy took runtime capacity first.
        assert during_fill == [False, True]
        assert lease.registered_bytes == 2 * PAGE and leases["pinning"].registered_bytes == PAGE
        assert backend.register_calls[1] == (pinning.data_ptr(), PAGE)
    for other in leases.values():
        other.close()
    manager.clear()


def test_an_alias_of_pending_storage_is_rejected_during_the_fill(backend: FakeBackend, tmp_path, monkeypatch) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    anonymous, alias = _tensors((0, 2 * PAGE), (PAGE, 2 * PAGE))
    manager = PinManager(8 * PAGE, backend=backend)
    original = manager._reader.read_into
    rejected: list[bool] = []

    def fill_while_an_alias_is_acquired(requests):
        # The in-place storage is reserved/registered; a distinct storage
        # overlapping it is rejected during the unlocked fill window.
        with pytest.raises(ValueError, match="Overlapping"):
            manager.acquire([alias], pin=False)
        rejected.append(True)
        original(requests)

    monkeypatch.setattr(manager._reader, "read_into", fill_while_an_alias_is_acquired)
    with manager.acquire([tensor, anonymous]) as lease:
        assert rejected == [True] and lease.registered_bytes == 4 * PAGE
    manager.clear()


def test_a_pinning_acquisition_of_registered_storage_does_not_wait_for_pending_fill(
    backend: FakeBackend, tmp_path, monkeypatch,
) -> None:
    pending_tensor = _checkpoint(tmp_path, 2 * PAGE)
    (cached,) = _tensors((0, PAGE))
    manager = PinManager(8 * PAGE, backend=backend)
    with manager.acquire([cached]):
        pass

    original = manager._reader.read_into
    cached_leases: list[PinLease] = []
    blocker = threading.Thread(target=lambda: cached_leases.append(manager.acquire([cached])))
    alive_during_fill: list[bool] = []

    def fill_while_cached_acquires(requests):
        blocker.start()
        blocker.join(2.0)
        alive_during_fill.append(blocker.is_alive())
        original(requests)

    monkeypatch.setattr(manager._reader, "read_into", fill_while_cached_acquires)
    with manager.acquire([pending_tensor]):
        blocker.join()
        assert alive_during_fill == [False]
        assert len(cached_leases) == 1
        assert cached_leases[0].registered_bytes == PAGE
        assert cached_leases[0].pageable_bytes == 0
    for lease in cached_leases:
        lease.close()
    manager.clear()
    assert manager.stats.active_leases == 0 and manager.stats.pinned_bytes == 0


def test_a_mixed_pinning_acquisition_waits_without_holding_cached_storage_early(
    backend: FakeBackend, tmp_path, monkeypatch,
) -> None:
    pending_tensor = _checkpoint(tmp_path, 2 * PAGE)
    cached, uncached = _tensors((0, PAGE), (PAGE, PAGE))
    manager = PinManager(8 * PAGE, backend=backend)
    with manager.acquire([cached]):
        pass

    original = manager._reader.read_into
    mixed_leases: list[PinLease] = []
    mixed_thread = threading.Thread(target=lambda: mixed_leases.append(manager.acquire([cached, uncached])))
    observed: list[tuple[bool, int]] = []

    def fill_while_mixed_waits(requests):
        mixed_thread.start()
        mixed_thread.join(0.2)
        cached_reg = manager._registrations[cached.data_ptr()]
        observed.append((mixed_thread.is_alive(), cached_reg.leases))
        original(requests)

    monkeypatch.setattr(manager._reader, "read_into", fill_while_mixed_waits)
    with manager.acquire([pending_tensor]):
        mixed_thread.join(2.0)
        assert not mixed_thread.is_alive()
        assert observed == [(True, 0)]
        assert mixed_leases[0].registered_bytes == 2 * PAGE
    for lease in mixed_leases:
        lease.close()
    manager.clear()
    assert manager.stats.active_leases == 0 and manager.stats.pinned_bytes == 0


def test_registered_fast_path_rejects_different_size_alias(
    manager: PinManager, backend: FakeBackend,
) -> None:
    cached, big, same_start_smaller = _tensors((0, PAGE), (PAGE, 2 * PAGE), (PAGE, PAGE))
    with manager.acquire([cached, big]):
        pass
    assert len(backend.register_calls) == 2

    with pytest.raises(ValueError, match="Overlapping"):
        manager.acquire([cached, same_start_smaller])
    assert manager._registrations[cached.data_ptr()].leases == 0
    assert cached.data_ptr() in manager._idle
    assert manager._registrations[big.data_ptr()].leases == 0
    assert manager.stats.active_leases == 0


def test_copy_registration_failure_frees_the_copy(backend: FakeBackend, tmp_path) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    backend.capacity = PAGE
    manager = PinManager(4 * PAGE, backend=backend)
    with manager.acquire([tensor]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert manager.stats.pinned_bytes == 0 and manager.stats.registration_failures == 1
    manager.clear()


def test_capacity_refusal_stops_registering_the_rest_of_the_acquisition(backend: FakeBackend, tmp_path) -> None:
    copies = [_checkpoint(tmp_path, 2 * PAGE, name) for name in ("first", "second", "third")]
    backend.capacity = 2 * PAGE
    manager = PinManager(8 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    with manager.acquire(copies) as lease:
        # First registers a 2*PAGE slab; second is refused by capacity, stopping before third.
        assert (lease.registered_bytes, lease.pageable_bytes) == (2 * PAGE, 4 * PAGE)
        assert len(backend.register_calls) == 2 and manager.stats.registration_failures == 1
        assert manager.stats.pinned_bytes == 2 * PAGE and manager.stats.copy_bytes == 2 * PAGE
    manager.clear()
    assert manager.stats.pinned_bytes == 0


def test_runtime_capacity_goes_to_the_storage_requested_first(backend: FakeBackend, tmp_path) -> None:
    copy = _checkpoint(tmp_path, 2 * PAGE)
    (anonymous,) = _tensors((0, 2 * PAGE))
    backend.capacity = 2 * PAGE
    for first, second in ((copy, anonymous), (anonymous, copy)):
        manager = PinManager(8 * PAGE, backend=backend)
        with manager.acquire([first, second]) as lease:
            assert (lease.registered_bytes, lease.pageable_bytes) == (2 * PAGE, 2 * PAGE)
            assert len(backend.register_calls) == 2 and manager.stats.registration_failures == 1
            assert manager.stats.copy_bytes == (2 * PAGE if first is copy else 0)
        manager.clear()
        assert manager.stats.pinned_bytes == 0
        backend.register_calls.clear()


def test_an_exhausted_runtime_is_found_before_the_rest_is_read(backend: FakeBackend, tmp_path, direct_reads) -> None:
    copies = [_checkpoint(tmp_path, 2 * PAGE, name) for name in ("first", "second", "third")]
    backend.capacity = PAGE
    manager = PinManager(8 * PAGE, backend=backend)
    with manager.acquire(copies) as lease:
        # Tier 2a registration failed before disk I/O ran, so zero bytes were read from disk.
        assert direct_reads == []
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 6 * PAGE)
        assert manager.stats.pinned_bytes == 0
    manager.clear()


def test_the_budget_goes_to_the_storage_requested_first(backend: FakeBackend, tmp_path) -> None:
    copy = _checkpoint(tmp_path, 2 * PAGE)
    (anonymous,) = _tensors((0, 2 * PAGE))
    for first, second in ((copy, anonymous), (anonymous, copy)):
        manager = PinManager(2 * PAGE, backend=backend)
        with manager.acquire([first, second]) as lease:
            assert (lease.registered_bytes, lease.pageable_bytes) == (2 * PAGE, 2 * PAGE)
            assert len(backend.register_calls) == 1 and manager.stats.registration_failures == 0
            assert manager.stats.copy_bytes == (2 * PAGE if first is copy else 0)
        manager.clear()
        backend.register_calls.clear()


def test_one_failed_fill_leaves_the_other_copies_registered(
    backend: FakeBackend, tmp_path, monkeypatch, caplog,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    broken_file = file_slice(second).file
    original = DirectFileReader.read_into

    def failing_read_into(self: DirectFileReader, requests) -> None:
        if any(req.source.file is broken_file for req in requests):
            raise OSError("injected read failure")
        original(self, requests)

    monkeypatch.setattr(DirectFileReader, "read_into", failing_read_into)
    manager = PinManager(8 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    warnings = caplog.at_level(logging.WARNING, logger="piper_offload")
    with warnings, manager.acquire([first, second]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (2 * PAGE, 2 * PAGE)
        assert manager.stats.copy_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, first), first)
        torch.testing.assert_close(_transferred(manager, second), second)
    assert "injected read failure" in caplog.text
    manager.clear()


def test_an_acquisition_of_pending_storage_waits_until_it_registers(
    backend: FakeBackend, tmp_path, monkeypatch,
) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    manager = PinManager(8 * PAGE, backend=backend)
    original = manager._reader.read_into
    other: list[PinLease] = []
    thread = threading.Thread(target=lambda: other.append(manager.acquire([tensor], pin=False)))
    during_fill: list[tuple[int, bool]] = []

    def fill_while_another_acquisition_waits(requests):
        thread.start()
        thread.join(0.2)
        during_fill.append((manager.stats.copy_bytes, thread.is_alive()))
        original(requests)

    monkeypatch.setattr(manager._reader, "read_into", fill_while_another_acquisition_waits)
    with manager.acquire([tensor]) as lease:
        thread.join()
        assert lease.registered_bytes == 2 * PAGE
        # The fill ran without the lock, yet the pageable acquisition of the
        # same storage waited for the copy and holds its registration.
        assert during_fill == [(2 * PAGE, True)]
        assert (other[0].registered_bytes, other[0].pageable_bytes) == (2 * PAGE, 0)
        assert len(backend.register_calls) == 1 and manager.stats.active_leases == 2
    other[0].close()
    manager.clear()
    assert manager.stats.pinned_bytes == 0


def test_failed_copy_unregistration_keeps_the_copy_and_its_reservation(backend: FakeBackend, tmp_path) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    manager = PinManager(4 * PAGE, backend=backend)
    with manager.acquire([tensor]):
        ((copy_pointer, _size),) = backend.register_calls
    backend.unregister_errors.add(copy_pointer)
    with pytest.raises(RuntimeError, match="remains retained"):
        manager.clear()
    assert manager.stats.pinned_bytes == 2 * PAGE and manager.stats.copy_bytes == 2 * PAGE
    backend.unregister_errors.clear()
    manager.clear()
    assert manager.stats.pinned_bytes == 0


def test_synchronous_transfer_of_pinned_storage_needs_no_lease(backend: FakeBackend, tmp_path) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    manager = PinManager(4 * PAGE, backend=backend)
    with manager.acquire([tensor]):
        pass
    destination = torch.zeros(2 * PAGE, dtype=torch.uint8)
    # The copy is idle, but a synchronous transfer completes under the lock.
    manager.transfer(destination, tensor, non_blocking=False)
    torch.testing.assert_close(destination, tensor)
    manager.clear()
    destination.zero_()
    manager.transfer(destination, tensor, non_blocking=False)
    torch.testing.assert_close(destination, tensor)


@CUDA
def test_asynchronous_transfer_of_pinned_storage_requires_a_lease(backend: FakeBackend, tmp_path) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    (anonymous,) = _tensors((0, PAGE))
    manager = PinManager(4 * PAGE, backend=backend)
    with manager.acquire([tensor, anonymous]):
        pass
    destination = torch.zeros(2 * PAGE, dtype=torch.uint8, device="cuda")
    # Idle registrations may be evicted at any moment, so an asynchronous read without a lease raises.
    for source in (tensor, anonymous):
        with pytest.raises(RuntimeError, match="outside a lease"):
            manager.transfer(destination[: source.numel()], source, non_blocking=True)
    with manager.acquire([tensor], pin=False):
        manager.transfer(destination, tensor, non_blocking=True)
        torch.cuda.synchronize()
    torch.testing.assert_close(destination.cpu(), tensor)
    manager.clear()


def test_clear_frees_every_copy_off_the_lock(backend: FakeBackend, tmp_path) -> None:
    """Release and unregistration during ``clear()`` run with ``manager._lock`` released."""
    tensors = [_checkpoint(tmp_path, 2 * PAGE, f"copy{index}") for index in range(3)]
    manager = PinManager(slab_bytes=2 * PAGE, backend=backend)
    with manager.acquire(tensors):
        pass
    allocs = [manager._registrations[tensor.untyped_storage().data_ptr()].copy for tensor in tensors]
    freed_locks: list[bool] = []
    unreg_locks: list[bool] = []
    orig_release = ArenaAllocation.release
    orig_unregister = backend.unregister

    def watch_release(self_alloc: ArenaAllocation) -> None:
        acquired = manager._lock.acquire(blocking=False)
        if acquired:
            manager._lock.release()
        freed_locks.append(acquired)
        orig_release(self_alloc)

    def watch_unregister(ptr: int) -> None:
        acquired = manager._lock.acquire(blocking=False)
        if acquired:
            manager._lock.release()
        unreg_locks.append(acquired)
        orig_unregister(ptr)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ArenaAllocation, "release", watch_release)
        patch.setattr(backend, "unregister", watch_unregister)
        manager.clear()

    assert all(isinstance(a, ArenaAllocation) and a.released for a in allocs)
    assert len(freed_locks) == 3 and all(freed_locks)
    assert len(unreg_locks) == 3 and all(unreg_locks)
    assert (manager.stats.pinned_bytes, manager.stats.copy_bytes, manager.stats.private_copy_bytes) == (0, 0, 0)


def test_pin_manager_rejects_invalid_and_boolean_config_parameters(backend: FakeBackend) -> None:
    for invalid in (-1, True, False):
        with pytest.raises(ValueError, match=">= 0"):
            PinManager(invalid, backend=backend)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=">= 0"):
            PinManager(PAGE, max_host_bytes=invalid, backend=backend)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=">= 0"):
            PinManager(PAGE, bounce_bytes=invalid, backend=backend)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=">= 0"):
            PinManager(PAGE, min_bounce_bytes=invalid, backend=backend)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="multiple"):
        PinManager(PAGE, slab_bytes=PAGE + 1, backend=backend)
    with pytest.raises(ValueError, match="multiple"):
        PinManager(PAGE, bounce_slot_bytes=PAGE + 1, backend=backend)
    manager = PinManager(PAGE, backend=backend)
    with pytest.raises(ValueError, match=">= 0"):
        manager.max_host_bytes = -1
    with pytest.raises(ValueError, match=">= 0"):
        manager.max_host_bytes = True  # type: ignore[assignment]
    manager.clear()


def test_tier2a_demotion_to_tier2b_and_in_ram_promotion(
    backend: FakeBackend, direct_reads, tmp_path,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    # Simultaneous Tier 2b promotion and Tier 2a demotion transiently holds both copies plus Tier 2a.
    manager = PinManager(2 * PAGE, max_host_bytes=6 * PAGE, backend=backend)
    with manager.acquire([first]) as lease:
        assert lease.registered_bytes == 2 * PAGE
        assert manager.stats.copy_bytes == 2 * PAGE
        assert manager.stats.private_copy_bytes == 0
    assert len(direct_reads) == 1

    # Acquiring second evicts first from Tier 2a and demotes it into Tier 2b.
    with manager.acquire([second]) as lease2:
        assert lease2.registered_bytes == 2 * PAGE
        assert manager.stats.copy_bytes == 2 * PAGE
        assert manager.stats.private_copy_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, second), second)
    assert len(direct_reads) == 2

    # Re-acquiring first promotes from Tier 2b in RAM with zero disk reads and no unregistration.
    direct_reads.clear()
    with manager.acquire([first]) as lease3:
        assert lease3.registered_bytes == 2 * PAGE
        assert direct_reads == []
        assert manager.stats.copy_bytes == 2 * PAGE
        assert manager.stats.private_copy_bytes == 2 * PAGE
        assert backend.unregister_calls == []
        torch.testing.assert_close(_transferred(manager, first), first)
    manager.clear()
    assert (manager.stats.pinned_bytes, manager.stats.copy_bytes, manager.stats.private_copy_bytes) == (0, 0, 0)

    # Tight 4*PAGE ceiling still promotes first from RAM without transiently overshooting max_host_bytes.
    tight = PinManager(2 * PAGE, max_host_bytes=4 * PAGE, backend=backend)
    direct_reads.clear()
    with tight.acquire([first]):
        pass
    with tight.acquire([second]):
        assert tight.stats.private_copy_bytes == 2 * PAGE
    direct_reads.clear()
    with tight.acquire([first]) as lease4:
        assert lease4.registered_bytes == 2 * PAGE
        assert direct_reads == []
        assert tight.stats.pinned_bytes + tight.stats.private_copy_bytes <= 4 * PAGE
        assert tight.stats.private_copy_bytes == 0
    tight.clear()


def test_max_pinned_bytes_reduction_demotes_tier2a_and_preserves_active_leases(
    backend: FakeBackend, direct_reads, tmp_path,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(4 * PAGE, max_host_bytes=8 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    active = manager.acquire([first])
    with manager.acquire([second]):
        pass
    assert (manager.stats.copy_bytes, manager.stats.private_copy_bytes) == (4 * PAGE, 0)

    # Lowering to 2*PAGE demotes idle second to Tier 2b while active first stays pinned.
    manager.max_pinned_bytes = 2 * PAGE
    assert manager.stats.copy_bytes == 2 * PAGE
    assert manager.stats.private_copy_bytes == 2 * PAGE

    # Dropping budget to 0 waits for active lease, then demotes first to Tier 2b on close.
    manager.max_pinned_bytes = 0
    assert manager.stats.copy_bytes == 2 * PAGE
    active.close()
    assert manager.stats.copy_bytes == 0
    assert manager.stats.private_copy_bytes == 4 * PAGE

    # Restoring budget promotes both back without reading disk.
    direct_reads.clear()
    manager.max_pinned_bytes = 4 * PAGE
    with manager.acquire([first, second]) as lease:
        assert lease.registered_bytes == 4 * PAGE
        assert direct_reads == []
        for tensor in (first, second):
            torch.testing.assert_close(_transferred(manager, tensor), tensor)
    manager.clear()


def test_max_host_bytes_eviction_and_zero_disable(backend: FakeBackend, tmp_path) -> None:
    tensors = [_checkpoint(tmp_path, 2 * PAGE, f"ckpt{i}") for i in range(3)]
    manager = PinManager(2 * PAGE, max_host_bytes=6 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    for tensor in tensors:
        with manager.acquire([tensor]):
            pass
    # third is in Tier 2a (2*PAGE), first and second are warm in Tier 2b (4*PAGE).
    assert (manager.stats.copy_bytes, manager.stats.private_copy_bytes) == (2 * PAGE, 4 * PAGE)

    manager.max_host_bytes = 4 * PAGE
    assert (manager.stats.copy_bytes, manager.stats.private_copy_bytes) == (2 * PAGE, 2 * PAGE)
    first_ptr = tensors[0].untyped_storage().data_ptr()
    assert first_ptr not in manager._private_copies
    assert tensors[1].untyped_storage().data_ptr() in manager._private_copies

    manager.max_host_bytes = 0
    assert manager.stats.private_copy_bytes == 0
    assert manager.stats.copy_bytes == 2 * PAGE

    manager.max_host_bytes = None
    assert manager.stats.max_host_bytes is None
    manager.clear()


def test_tier2b_owner_disposal_and_lease_retention(backend: FakeBackend, tmp_path) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(2 * PAGE, max_host_bytes=4 * PAGE, backend=backend)
    with manager.acquire([first]):
        pass
    with manager.acquire([second]):
        pass
    assert manager.stats.private_copy_bytes == 2 * PAGE

    # Held pageable lease protects Tier 2b copy until lease closes, even after owner disappears.
    alias = first[:PAGE].detach()
    lease = manager.acquire([alias], pin=False)
    del first
    gc.collect()
    assert manager.stats.private_copy_bytes == 2 * PAGE
    assert torch.equal(_transferred(manager, alias), alias)
    del alias
    lease.close()
    assert manager.stats.private_copy_bytes == 0
    manager.clear()


def test_direct_io_aligns_large_unaligned_and_coalesces_small(
    backend: FakeBackend, tmp_path, monkeypatch,
) -> None:
    large = _checkpoint(tmp_path, DEFAULT_COALESCE_THRESHOLD + 8192, "large")
    small = _checkpoint(tmp_path, 2 * PAGE, "small")
    seen_reqs: list[tuple[int, int, bool]] = []
    orig_read_into = DirectFileReader.read_into

    def capture(self: DirectFileReader, requests) -> None:
        for req in requests:
            seen_reqs.append((req.source.length, req.head_pad, req.aligned_target))
        orig_read_into(self, requests)

    monkeypatch.setattr(DirectFileReader, "read_into", capture)
    manager = PinManager(4 * 1024 * 1024, backend=backend)
    with manager.acquire([large, small]) as lease:
        assert lease.registered_bytes == large.nbytes + small.nbytes
        torch.testing.assert_close(_transferred(manager, large), large)
        torch.testing.assert_close(_transferred(manager, small), small)
    by_len = {length: (pad, aligned) for length, pad, aligned in seen_reqs}
    assert by_len[large.nbytes][0] > 0 and by_len[large.nbytes][1] is True
    assert by_len[small.nbytes] == (0, False)
    manager.clear()


def test_tier2b_cpu_transfer_reads_private_copy_without_registration(
    backend: FakeBackend, tmp_path,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(2 * PAGE, max_host_bytes=4 * PAGE, backend=backend)
    with manager.acquire([first]):
        pass
    with manager.acquire([second]):
        pass
    priv = manager._private_copies[first.untyped_storage().data_ptr()]
    assert manager.stats.private_copy_bytes == 2 * PAGE
    with priv.alloc.payload_view() as view:
        view[17] = (first[17].item() + 1) % 256
    delivered = _transferred(manager, first)
    assert delivered[17] != first[17]
    manager.clear()


@CUDA
def test_tier2b_cuda_transfer_streams_via_bounce_ring(backend: FakeBackend, tmp_path) -> None:
    nbytes = 512 * 1024
    first = _checkpoint(tmp_path, nbytes, "first")
    second = _checkpoint(tmp_path, nbytes, "second")
    manager = PinManager(
        nbytes + 256 * 1024,
        max_host_bytes=3 * nbytes,
        bounce_bytes=256 * 1024,
        bounce_slot_bytes=64 * 1024,
        backend=backend,
    )
    try:
        with manager.acquire([first]):
            pass
        with manager.acquire([second]):
            pass
        assert manager.stats.private_copy_bytes >= nbytes
        assert manager.stats.bounce_bytes == 0

        dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        # Synchronous CUDA transfer from Tier 2b uses the lazy bounce ring cleanly.
        manager.transfer(dst, first, non_blocking=False)
        torch.testing.assert_close(dst.cpu(), first)
        assert manager.stats.bounce_bytes > 0
        assert manager.stats.pinned_bytes <= manager.max_pinned_bytes  # type: ignore[operator]

        # Asynchronous leased transfer from Tier 2b also routes cleanly through the bounce ring.
        dst.zero_()
        with manager.acquire([first], pin=False):
            manager.transfer(dst, first, non_blocking=True)
            torch.cuda.synchronize()
        torch.testing.assert_close(dst.cpu(), first)
    finally:
        manager.clear()


@CUDA
def test_direct_disk_to_bounce_ring_to_cuda_uses_zero_tier2_ram(backend: FakeBackend, tmp_path) -> None:
    nbytes = 512 * 1024
    ckpt = _checkpoint(tmp_path, nbytes, "direct_stream")
    manager = PinManager(
        2 * 1024 * 1024,
        bounce_bytes=512 * 1024,
        bounce_slot_bytes=64 * 1024,
        backend=backend,
    )
    try:
        dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        with manager.acquire([ckpt], pin=False) as lease:
            assert (lease.registered_bytes, lease.pageable_bytes) == (0, ckpt.nbytes)
            manager.transfer(dst, ckpt, non_blocking=True)
            torch.cuda.synchronize()
        torch.testing.assert_close(dst.cpu(), ckpt)
        assert manager.stats.copy_bytes == 0
        assert manager.stats.private_copy_bytes == 0
        assert manager.stats.bounce_bytes > 0

        # Synchronous unleased FileSlice transfer also works and synchronizes before return.
        sub = ckpt[8192 : 8192 + 300 * 1024]
        sub_dst = torch.empty_like(sub, device="cuda")
        manager.transfer(sub_dst, sub, non_blocking=False)
        torch.testing.assert_close(sub_dst.cpu(), sub)
    finally:
        manager.clear()


@CUDA
def test_bounce_ring_respects_budget_and_overflow_headroom(backend: FakeBackend, tmp_path) -> None:
    nbytes = 512 * 1024
    first = _checkpoint(tmp_path, nbytes, "first")
    second = _checkpoint(tmp_path, nbytes, "second")
    manager = PinManager(
        nbytes + 256 * 1024,
        max_host_bytes=nbytes + 256 * 1024,
        bounce_bytes=256 * 1024,
        bounce_slot_bytes=64 * 1024,
        backend=backend,
    )
    try:
        with manager.acquire([first, second]) as lease:
            assert lease.registered_bytes == nbytes
            assert lease.pageable_bytes == nbytes
            dst_first = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            dst_second = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            manager.transfer(dst_first, first, non_blocking=True)
            manager.transfer(dst_second, second, non_blocking=True)
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_first.cpu(), first)
            torch.testing.assert_close(dst_second.cpu(), second)
            assert manager.stats.bounce_bytes > 0
            assert manager.stats.pinned_bytes == (
                manager.stats.copy_bytes + manager.stats.bounce_bytes
            ) <= manager.max_pinned_bytes  # type: ignore[operator]
    finally:
        manager.clear()
    assert manager.stats.pinned_bytes == 0


@CUDA
def test_pageable_cuda_transfer_bounce_and_fallback(backend: FakeBackend, tmp_path) -> None:
    nbytes = 512 * 1024
    src = torch.arange(nbytes // 4, dtype=torch.int32)
    disabled = PinManager(0, backend=backend)
    try:
        dst = torch.empty_like(src, device="cuda")
        disabled.transfer(dst, src, non_blocking=False)
        torch.testing.assert_close(dst.cpu(), src)
        assert backend.register_calls == [] and disabled.stats.bounce_bytes == 0
    finally:
        disabled.clear()

    active = PinManager(1024 * 1024, bounce_bytes=256 * 1024, bounce_slot_bytes=64 * 1024, backend=backend)
    try:
        dst = torch.empty_like(src, device="cuda")
        active.transfer(dst, src, non_blocking=False)
        torch.testing.assert_close(dst.cpu(), src)
        assert active.stats.bounce_bytes > 0
        # Non-contiguous view falls back cleanly without breaking.
        non_contig = src[::2]
        dst_half = torch.empty_like(non_contig, device="cuda")
        active.transfer(dst_half, non_contig, non_blocking=False)
        torch.testing.assert_close(dst_half.cpu(), non_contig)
    finally:
        active.clear()


def test_bounce_ring_clear_retention_and_concurrent_wait(
    backend: FakeBackend, tmp_path, monkeypatch,
) -> None:
    manager = PinManager(
        1024 * 1024,
        bounce_bytes=256 * 1024,
        bounce_slot_bytes=64 * 1024,
        min_bounce_bytes=64 * 1024,
        backend=backend,
    )
    ring = manager._create_bounce_ring()
    assert ring is not None
    manager._resources.bounce_ring = ring
    ring_ptr = backend.register_calls[-1][0]

    # Wait semantics: clear() blocks while _bounce_users > 0.
    entered = threading.Event()
    release_user = threading.Event()
    clear_done = threading.Event()

    def hold_bounce() -> None:
        with manager._use_bounce_ring():
            entered.set()
            assert release_user.wait(5)

    def run_clear() -> None:
        manager.clear()
        clear_done.set()

    user_t = threading.Thread(target=hold_bounce)
    user_t.start()
    assert entered.wait(5)
    backend.unregister_errors.add(ring_ptr)
    clear_err: list[Exception] = []

    def run_clear_catch() -> None:
        try:
            run_clear()
        except Exception as exc:
            clear_err.append(exc)
            clear_done.set()

    clear_t = threading.Thread(target=run_clear_catch)
    clear_t.start()
    assert not clear_done.wait(0.1)
    release_user.set()
    user_t.join(5)
    clear_t.join(5)
    assert clear_done.is_set()
    assert clear_err and "remains retained" in str(clear_err[0])
    assert manager.stats.bounce_bytes > 0
    assert manager.stats.unregistration_failures >= 1

    # Clearing the error allows the next clear() to unregister cleanly while preserving counters.
    prev_unreg_fails = manager.stats.unregistration_failures
    backend.unregister_errors.clear()
    manager.clear()
    assert manager.stats.bounce_bytes == 0
    assert manager.stats.unregistration_failures == prev_unreg_fails
    assert ring_ptr in backend.unregister_calls


def test_pin_manager_gc_closes_direct_file_reader(backend: FakeBackend) -> None:
    mgr = PinManager(2 * PAGE, backend=backend)
    reader = mgr._reader
    del mgr
    gc.collect()
    assert reader._gate._closed


def test_simplified_windows_virtual_region_exports_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    regions: dict[int, mmap.mmap] = {}

    class _MockKernel:
        def allocate(self, _addr: object, size: int, _t: int, _p: int) -> int:
            if not size:
                return 0
            buf = mmap.mmap(-1, size)
            c = ctypes.c_char.from_buffer(buf)
            ptr = ctypes.addressof(c)
            del c
            regions[ptr] = buf
            return ptr

        def free(self, ptr: int, _sz: int, _ft: int) -> int:
            regions.pop(ptr).close()
            return 1

    monkeypatch.setattr(windows_memory, "_kernel32", _MockKernel)
    region = VirtualRegion(2 * PAGE)
    tensor = torch.frombuffer(memoryview(region), dtype=torch.uint8)
    with pytest.raises(BufferError, match="exported"):
        region.close()
    assert not region.closed
    del tensor
    region.close()
    assert region.closed
    region.close()
    with pytest.raises(ValueError, match="closed"):
        memoryview(region)
    assert not regions


def test_concurrent_acquire_demote_promote_transfer_and_clear_stress(
    backend: FakeBackend, tmp_path,
) -> None:
    ckpts = [_checkpoint(tmp_path, 2 * PAGE, f"stress{i}") for i in range(5)]
    anon = _tensors((0, PAGE), (2 * PAGE, PAGE))
    manager = PinManager(4 * PAGE, max_host_bytes=8 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    errors: list[BaseException] = []
    stop = threading.Event()

    def worker(idx: int) -> None:
        rng = random.Random(100 + idx)
        try:
            for _ in range(25):
                if stop.is_set():
                    break
                choice = rng.choice(ckpts)
                pin_flag = rng.random() < 0.7
                with manager.acquire([choice, anon[idx % 2]], pin=pin_flag):
                    dst = torch.empty_like(choice)
                    manager.transfer(dst, choice, non_blocking=False)
                    assert torch.equal(dst, choice)
        except BaseException as exc:
            errors.append(exc)
            stop.set()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for step in range(6):
        time.sleep(0.01)
        if step % 2 == 0:
            manager.max_pinned_bytes = 2 * PAGE
        else:
            manager.max_pinned_bytes = 6 * PAGE
            manager.clear()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()
    assert not errors, errors
    manager.clear()
    assert manager.stats.pinned_bytes == 0
    assert manager.stats.private_copy_bytes == 0
    assert manager.stats.active_leases == 0


def _assert_close_entry_interrupt_retry(
    local_manager: PinManager,
    tensor: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease2 = local_manager.acquire([tensor])
    orig_close_lease = local_manager._close_lease
    fail_enter = True

    def interrupt_close_entry(key: int) -> None:
        nonlocal fail_enter
        if fail_enter:
            fail_enter = False
            raise KeyboardInterrupt("injected entry interrupt")
        orig_close_lease(key)

    monkeypatch.setattr(local_manager, "_close_lease", interrupt_close_entry)
    try:
        with pytest.raises(KeyboardInterrupt, match="injected entry interrupt"):
            lease2.close()
        assert not lease2.closed
        lease2.close()
        assert lease2.closed
        assert local_manager.stats.active_leases == 0
    finally:
        fail_enter = False
        monkeypatch.setattr(local_manager, "_close_lease", orig_close_lease)


def test_interrupted_lease_close_retry_preserves_accounting(
    backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, = _tensors((0, PAGE))
    b, = _tensors((0, PAGE))
    c, = _tensors((0, PAGE))
    a_view = torch.empty(0, dtype=torch.uint8).set_(a.untyped_storage(), 0, (PAGE,), (1,))
    local_manager = PinManager(2 * PAGE, backend=backend)
    init_lease = local_manager.acquire([a])
    init_lease.close()
    lease = local_manager.acquire([a_view, b, c])
    del a
    gc.collect()

    orig_unregister = backend.unregister
    fail_once = True

    def interrupt_unregister(pointer: int) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise KeyboardInterrupt("injected unregister interrupt")
        orig_unregister(pointer)

    monkeypatch.setattr(backend, "unregister", interrupt_unregister)
    try:
        with pytest.raises(KeyboardInterrupt, match="injected unregister interrupt"):
            lease.close()
        assert not lease.closed
        lease.close()
        assert lease.closed
        assert local_manager.stats.active_leases == 0
        assert local_manager.stats.registrations == 1
        assert set(backend.registered) == {b.untyped_storage().data_ptr()}
        local_manager.clear()
        assert local_manager.stats.registrations == 0
        assert backend.registered == {}

        _assert_close_entry_interrupt_retry(local_manager, b, monkeypatch)
    finally:
        fail_once = False
        monkeypatch.undo()
        del a_view, b, c
        gc.collect()
        local_manager.clear()


def test_empty_view_into_idle_registered_storage_transfers_without_lease(
    backend: FakeBackend,
) -> None:
    (base,) = _tensors((0, PAGE))
    empty = base.narrow(0, PAGE, 0)
    assert empty.numel() == 0
    assert empty.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()

    manager = PinManager(2 * PAGE, backend=backend)
    try:
        # Leave the underlying storage registered with zero active leases.
        with manager.acquire([base]):
            pass
        assert manager.stats.registrations == 1
        assert manager.stats.active_leases == 0

        copies: list[tuple[int, bool]] = []
        fake_cuda_dst = SimpleNamespace(
            device=torch.device("cuda", 0),
            copy_=lambda src, *, non_blocking=False: copies.append((src.numel(), non_blocking)),
        )
        with manager.acquire([empty]) as lease:
            assert (lease.registered_bytes, lease.pageable_bytes) == (0, 0)
            manager.transfer(fake_cuda_dst, empty, non_blocking=True)  # type: ignore[arg-type]
        assert copies == [(0, True)]
    finally:
        manager.clear()


def test_sorted_pointers_matches_reference_oracle() -> None:
    rng = random.Random(42)
    index = pin_module._SortedPointers()
    oracle: list[int] = []

    assert index.neighbors(0) == ()
    with pytest.raises(KeyError):
        index.remove(123)

    next_ptr = 1024
    for step in range(2500):
        # Bias toward growth early so multiple bucket splits occur (> 512 items), then mix churn.
        if not oracle or step < 1400 or rng.random() < 0.55:
            ptr = next_ptr + rng.randrange(1, 2048)
            next_ptr = ptr if rng.random() < 0.7 else next_ptr
            if ptr not in oracle:
                index.add(ptr)
                index.add(ptr)  # duplicate add must stay idempotent
                oracle.insert(bisect_left(oracle, ptr), ptr)
        else:
            ptr = oracle[rng.randrange(len(oracle))]
            index.remove(ptr)
            oracle.remove(ptr)

        if step % 25 == 0:
            for probe in (0, next_ptr + 10_000, *(rng.sample(oracle, min(3, len(oracle))))):
                for query in (probe - 1, probe, probe + 1):
                    pos = bisect_left(oracle, query)
                    assert index.neighbors(query) == tuple(oracle[max(0, pos - 1):pos + 1])

    with pytest.raises(KeyError):
        index.remove(oracle[0] + 1 if oracle[0] + 1 not in oracle else -1)
    with pytest.raises(KeyError):
        index.remove(oracle[-1] + 9999)

    rng.shuffle(oracle)
    for idx, ptr in enumerate(oracle):
        index.remove(ptr)
        if idx % 20 == 0 and oracle[idx + 1 :]:
            sample_q = oracle[idx + 1]
            rem = sorted(oracle[idx + 1 :])
            for query in (sample_q - 1, sample_q, sample_q + 1):
                pos = bisect_left(rem, query)
                assert index.neighbors(query) == tuple(rem[max(0, pos - 1):pos + 1])
    assert index.neighbors(0) == ()
    with pytest.raises(KeyError):
        index.remove(oracle[0])


def test_leased_async_transfer_reuses_preresolved_view_without_manager_lock(
    backend: FakeBackend,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = _checkpoint(tmp_path, 2 * PAGE, "preresolved")
    sub = ckpt[16:128]
    manager = PinManager(4 * PAGE, backend=backend)
    view_calls = 0
    original_view = ArenaAllocation.tensor_view

    def counting_view(self: ArenaAllocation, tensor: torch.Tensor) -> torch.Tensor:
        nonlocal view_calls
        view_calls += 1
        return original_view(self, tensor)

    monkeypatch.setattr(ArenaAllocation, "tensor_view", counting_view)

    transferred: list[tuple[int, torch.Tensor]] = []
    errors: list[BaseException] = []
    fake_cuda_dst = SimpleNamespace(
        device=torch.device("cuda", 0),
        copy_=lambda src, *, non_blocking=False: transferred.append(
            (src.untyped_storage().data_ptr(), src.clone())
        ),
    )

    def run_transfer() -> None:
        try:
            manager.transfer(fake_cuda_dst, ckpt, non_blocking=True)  # type: ignore[arg-type]
        except BaseException as exc:  # pragma: no cover - re-raised below
            errors.append(exc)

    try:
        with manager.acquire([ckpt]) as lease:
            assert lease.registered_bytes == 2 * PAGE
            assert view_calls == 1
            reg_ptr = manager._registrations[ckpt.untyped_storage().data_ptr()].registered
            slab_ptr, slab_size = backend.register_calls[0]
            assert slab_ptr <= reg_ptr < slab_ptr + slab_size
            # Hold _lock on main thread while worker runs repeated async transfers.
            with manager._lock:
                for _ in range(5):
                    t = threading.Thread(target=run_transfer)
                    t.start()
                    t.join(timeout=2.0)
                    assert not t.is_alive()
            assert not errors
            assert view_calls == 1
            assert len(transferred) == 5
            assert all(ptr == reg_ptr and torch.equal(out, ckpt) for ptr, out in transferred)

            # Synchronous leased copy also reuses the pre-resolved view without re-calling tensor_view.
            cpu_dst = torch.empty_like(ckpt)
            manager.transfer(cpu_dst, ckpt, non_blocking=False)
            assert view_calls == 1
            assert torch.equal(cpu_dst, ckpt)

            # Derived sub-view falls back cleanly under the lock via ArenaAllocation.tensor_view.
            manager.transfer(fake_cuda_dst, sub, non_blocking=True)  # type: ignore[arg-type]
            assert view_calls == 2
            assert transferred[-1][0] == reg_ptr
            assert torch.equal(transferred[-1][1], sub)
            assert slab_ptr in backend.registered

        with pytest.raises(RuntimeError, match="outside a lease"):
            manager.transfer(fake_cuda_dst, ckpt, non_blocking=True)  # type: ignore[arg-type]
    finally:
        manager.clear()


def test_concurrent_leases_share_and_release_preresolved_views_for_tier2b_demotion(
    backend: FakeBackend,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(2 * PAGE, max_host_bytes=4 * PAGE, backend=backend)
    try:
        lease1 = manager.acquire([first])
        lease2 = manager.acquire([first])
        assert manager._leased_views[id(first)][1] == 2
        lease1.close()
        assert manager._leased_views[id(first)][1] == 1
        lease2.close()
        assert manager._leased_views == {}

        with caplog.at_level(logging.WARNING, logger="piper_offload"), manager.acquire([second]):
            assert (manager.stats.private_copy_bytes, manager.stats.pinned_bytes) == (2 * PAGE, 2 * PAGE)
        assert "still referenced" not in caplog.text
    finally:
        manager.clear()


def test_open_lease_view_failure_rolls_back_cleanly(
    backend: FakeBackend,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(8 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    calls = 0
    original_view = ArenaAllocation.tensor_view

    def fail_second_view(self: ArenaAllocation, tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected view failure")
        return original_view(self, tensor)

    monkeypatch.setattr(ArenaAllocation, "tensor_view", fail_second_view)
    try:
        with caplog.at_level(logging.WARNING, logger="piper_offload"), pytest.raises(
            RuntimeError, match="injected view failure",
        ):
            manager.acquire([first, second])
        assert manager._leased_views == {}
        assert manager.stats.active_leases == 0
        assert manager.stats.registrations == 0
        assert manager.stats.pinned_bytes == 0
        assert backend.registered == {}
        assert "still referenced" not in caplog.text
    finally:
        manager.clear()


def test_open_lease_pin_lease_init_failure_preserves_shared_lease(
    backend: FakeBackend,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = _checkpoint(tmp_path, 2 * PAGE, "shared")
    ptr = ckpt.untyped_storage().data_ptr()
    manager = PinManager(4 * PAGE, backend=backend)
    lease1 = manager.acquire([ckpt])
    try:
        reg_ptr = manager._registrations[ptr].registered
        real_finalize = weakref.finalize

        class FailingAtExitFinalizer:
            def __init__(self, obj: object, func: object, *args: object) -> None:
                self._real = real_finalize(obj, func, *args)  # type: ignore[arg-type]

            @property
            def alive(self) -> bool:
                return self._real.alive

            @property
            def atexit(self) -> bool:
                return self._real.atexit

            @atexit.setter
            def atexit(self, _value: bool) -> None:
                raise RuntimeError("injected PinLease init failure")

            def detach(self) -> object:
                return self._real.detach()

            def __call__(self) -> object:
                return self._real()

        with monkeypatch.context() as patch:
            patch.setattr(weakref, "finalize", FailingAtExitFinalizer)
            with pytest.raises(RuntimeError, match="injected PinLease init failure"):
                manager.acquire([ckpt])
        gc.collect()
        assert manager.stats.active_leases == 1
        assert manager._leased_views[id(ckpt)][1] == 1
        assert manager._registrations[ptr].leases == 1
        assert reg_ptr in backend.registered
        lease1.close()
        assert manager._leased_views == {}
        assert manager.stats.active_leases == 0
    finally:
        manager.clear()
        assert backend.registered == {}


def test_interrupted_release_views_retry_preserves_other_lease(
    backend: FakeBackend,
    tmp_path,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(8 * PAGE, backend=backend)
    lease1 = manager.acquire([first, second])
    lease2 = manager.acquire([first, second])
    try:
        class OnceInterruptingDeque(deque[int]):
            def __init__(self, iterable: deque[int]) -> None:
                super().__init__(iterable)
                self._calls = 0

            def popleft(self) -> int:
                self._calls += 1
                if self._calls == 2:
                    raise KeyboardInterrupt("injected release_views interrupt")
                return super().popleft()

        state = manager._leases[lease1._key]
        state.view_ids = OnceInterruptingDeque(state.view_ids)

        with pytest.raises(KeyboardInterrupt, match="injected release_views interrupt"):
            lease1.close()
        assert not lease1.closed
        assert len(state.view_ids) == 1

        lease1.close()
        assert lease1.closed
        assert manager._leased_views[id(first)][1] == 1
        assert manager._leased_views[id(second)][1] == 1

        lease2.close()
        assert manager._leased_views == {}
        assert manager.stats.active_leases == 0
    finally:
        manager.clear()


def test_bounce_ring_init_error_does_not_deadlock_future_transfers(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = PinManager(
        512 * 1024,
        bounce_bytes=128 * 1024,
        bounce_slot_bytes=64 * 1024,
        min_bounce_bytes=64 * 1024,
        backend=backend,
    )
    original_init = PinnedBounceRing._ensure_initialized
    calls = 0

    def fail_once(self: PinnedBounceRing) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HostRegistrationError("registration", 700)
        return original_init(self)

    monkeypatch.setattr(PinnedBounceRing, "_ensure_initialized", fail_once)
    with pytest.raises(HostRegistrationError), manager._use_bounce_ring():
        pass
    assert not manager._bounce_creating
    assert manager._resources.creating_ring is None
    with manager._use_bounce_ring() as ring:
        assert ring is not None
    manager.clear()
    assert backend.registered == {}


def test_open_lease_failure_rolls_back_created_and_promoted_cleanly(
    backend: FakeBackend, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = _checkpoint(tmp_path, 2 * PAGE, "promoted")
    other = _checkpoint(tmp_path, 2 * PAGE, "other")
    manager = PinManager(2 * PAGE, max_host_bytes=6 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    for item in (ckpt, other):
        with manager.acquire([item]):
            pass
    assert manager.stats.private_copy_bytes == 2 * PAGE

    original_view = ArenaAllocation.tensor_view
    entered, proceed, worker_done = threading.Event(), threading.Event(), threading.Event()
    calls = 0

    def fail_on_first_promotion_view(self: ArenaAllocation, tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert proceed.wait(5)
            raise RuntimeError("injected promotion view failure")
        return original_view(self, tensor)

    monkeypatch.setattr(ArenaAllocation, "tensor_view", fail_on_first_promotion_view)
    waiter_result: list[int] = []
    worker_blocked: list[bool] = []

    def concurrent_acquire() -> None:
        with manager.acquire([ckpt]) as lease:
            waiter_result.append(lease.registered_bytes)
            torch.testing.assert_close(_transferred(manager, ckpt), ckpt)
        worker_done.set()

    def drive_waiter() -> None:
        assert entered.wait(5)
        worker.start()
        time.sleep(0.02)
        worker_blocked.append(not worker_done.is_set())
        proceed.set()

    worker = threading.Thread(target=concurrent_acquire)
    waiter_driver = threading.Thread(target=drive_waiter)
    waiter_driver.start()
    try:
        with pytest.raises(RuntimeError, match="injected promotion view failure"):
            manager.acquire([ckpt])
    finally:
        waiter_driver.join(5)
        worker.join(5)
    assert worker_blocked == [True] and waiter_result == [2 * PAGE]
    manager.clear()
    assert manager._registrations == {} and manager._pending == {}
    assert manager.stats.copy_bytes == 0 and manager.stats.private_copy_bytes == 0
    assert backend.registered == {}


def test_exhausted_capacity_releases_unvisited_private_source_leases(
    backend: FakeBackend, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    third: torch.Tensor | None = _checkpoint(tmp_path, 2 * PAGE, "third")
    manager = PinManager(2 * PAGE, max_host_bytes=8 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    for item in (first, second, third):
        assert item is not None
        with manager.acquire([item]):
            pass
    # Drop the sole owner of the Tier 2a copy so Tier 2a shrinks to 0 while first/second stay in Tier 2b.
    del item, third
    gc.collect()
    first_ptr = first.untyped_storage().data_ptr()
    second_ptr = second.untyped_storage().data_ptr()
    assert manager.stats.copy_bytes == 0
    assert first_ptr in manager._private_copies and second_ptr in manager._private_copies

    backend.capacity = 0
    with manager.acquire([first, second]) as lease:
        assert lease.registered_bytes == 0
        assert lease.pageable_bytes == 4 * PAGE
        assert manager._private_copies[first_ptr].leases == 1
        assert manager._private_copies[second_ptr].leases == 1
    assert manager._private_copies[first_ptr].leases == 0
    assert manager._private_copies[second_ptr].leases == 0
    manager.max_host_bytes = 0
    assert manager.stats.private_copy_bytes == 0
    manager.clear()

    # Promoting a Tier 2b copy whose tracker flipped retired mid-promotion creates a live registration.
    backend.capacity = None
    manager.max_host_bytes = 6 * PAGE
    target = _checkpoint(tmp_path, 2 * PAGE, "retired_tracker")
    fourth = _checkpoint(tmp_path, 2 * PAGE, "fourth")
    target_ptr = target.untyped_storage().data_ptr()
    with manager.acquire([target]):
        pass
    with manager.acquire([fourth]):
        pass
    priv = manager._private_copies[target_ptr]
    original_u8 = ArenaAllocation.as_uint8_tensor
    flipped = False

    def mark_retired_during_copy(self: ArenaAllocation) -> torch.Tensor:
        nonlocal flipped
        if not flipped and self is priv.alloc:
            flipped = True
            priv.tracker.retired = True
        return original_u8(self)

    monkeypatch.setattr(ArenaAllocation, "as_uint8_tensor", mark_retired_during_copy)
    with manager.acquire([target]) as promoted_lease:
        assert promoted_lease.registered_bytes == 2 * PAGE
        assert target_ptr in manager._registrations
        assert not manager._registrations[target_ptr].retired
    assert target_ptr in manager._registrations
    manager.clear()


@pytest.mark.parametrize("scenario", ["zero", "full"])
def test_bounce_ring_creation_aborts_when_budget_vanishes(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch, scenario: str,
) -> None:
    manager = PinManager(
        512 * 1024,
        bounce_bytes=128 * 1024,
        bounce_slot_bytes=64 * 1024,
        min_bounce_bytes=64 * 1024,
        backend=backend,
    )
    entered = threading.Event()
    release = threading.Event()
    original_init = PinnedBounceRing._ensure_initialized

    def paused_init(self: PinnedBounceRing) -> bool:
        entered.set()
        assert release.wait(5)
        return original_init(self)

    monkeypatch.setattr(PinnedBounceRing, "_ensure_initialized", paused_init)
    outcomes: list[bool] = []

    def create_ring() -> None:
        with manager._use_bounce_ring() as ring:
            outcomes.append(ring is not None)

    creator = threading.Thread(target=create_ring)
    creator.start()
    assert entered.wait(5)
    hold_lease: PinLease | None = None
    if scenario == "zero":
        manager.max_pinned_bytes = 0
    else:
        (buf,) = _tensors((0, 512 * 1024))
        hold_lease = manager.acquire([buf])
    release.set()
    creator.join(5)
    monkeypatch.undo()
    assert outcomes == [False]
    assert manager._resources.creating_ring is None
    assert manager._resources.bounce_ring is None
    assert manager.stats.bounce_bytes == 0
    if hold_lease is not None:
        assert manager.stats.pinned_bytes <= 512 * 1024
        hold_lease.close()
    manager.clear()
    assert backend.registered == {}


def test_bounce_ring_closing_tracks_bounce_bytes_until_closed(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    closing_mgr = PinManager(
        512 * 1024,
        bounce_bytes=128 * 1024,
        bounce_slot_bytes=64 * 1024,
        min_bounce_bytes=64 * 1024,
        backend=backend,
    )
    with closing_mgr._use_bounce_ring() as ring:
        assert ring is not None
    close_entered = threading.Event()
    close_release = threading.Event()
    orig_close = PinnedBounceRing.close

    def paused_close(self: PinnedBounceRing) -> None:
        close_entered.set()
        assert close_release.wait(5)
        orig_close(self)

    monkeypatch.setattr(PinnedBounceRing, "close", paused_close)
    clear_t = threading.Thread(target=closing_mgr.clear)
    clear_t.start()
    assert close_entered.wait(5)
    assert closing_mgr.stats.bounce_bytes == 128 * 1024
    close_release.set()
    clear_t.join(5)
    assert closing_mgr.stats.bounce_bytes == 0
    assert backend.registered == {}


def test_demotion_respects_max_host_bytes_with_shared_slab_and_active_leases(
    backend: FakeBackend, tmp_path,
) -> None:
    active_ckpt = _checkpoint(tmp_path, 4 * PAGE, "active")
    idle_ckpt = _checkpoint(tmp_path, 4 * PAGE, "idle")

    # Tight host budget equals the shared 8*PAGE slab: idle sibling skips demotion rather than exceeding budget.
    tight = PinManager(8 * PAGE, max_host_bytes=8 * PAGE, slab_bytes=8 * PAGE, backend=backend)
    with tight.acquire([active_ckpt, idle_ckpt]) as both:
        assert both.registered_bytes == 8 * PAGE
    active_only = tight.acquire([active_ckpt])
    assert tight.stats.copy_bytes == 8 * PAGE
    tight.max_pinned_bytes = 0
    assert tight.stats.pinned_bytes + tight.stats.private_copy_bytes <= 8 * PAGE
    assert tight.stats.private_copy_bytes == 0
    active_only.close()
    tight.clear()

    # With 12*PAGE host ceiling (8*PAGE slab + 4*PAGE copy), idle_ckpt demotes within budget.
    roomy = PinManager(8 * PAGE, max_host_bytes=12 * PAGE, slab_bytes=8 * PAGE, backend=backend)
    with roomy.acquire([active_ckpt, idle_ckpt]):
        pass
    active_roomy = roomy.acquire([active_ckpt])
    roomy.max_pinned_bytes = 0
    assert roomy.stats.copy_bytes == 8 * PAGE
    assert roomy.stats.private_copy_bytes == 4 * PAGE
    assert roomy.stats.pinned_bytes + roomy.stats.private_copy_bytes <= 12 * PAGE
    active_roomy.close()
    roomy.clear()


def test_budget_setter_epoch_prevents_stale_arena_limit_overwrite(
    backend: FakeBackend, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # (a) Raising max_pinned_bytes = None concurrently with _close_lease() leaves _tier2a.max_bytes None.
    ckpt = _checkpoint(tmp_path, 4 * PAGE, "epoch_2a")
    manager = PinManager(4 * PAGE, max_host_bytes=8 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    lease = manager.acquire([ckpt])
    # Lower budget while lease stays active so _close_lease() enters _evict_idle_for_budget -> trim_empty.
    manager.max_pinned_bytes = 2 * PAGE
    entered = threading.Event()
    proceed = threading.Event()
    orig_trim = HostSlabArena.trim_empty

    def pause_first_trim(self: HostSlabArena, target_reserved_bytes: int) -> int:
        if self is manager._tier2a and not entered.is_set():
            entered.set()
            assert proceed.wait(5)
        return orig_trim(self, target_reserved_bytes)

    monkeypatch.setattr(HostSlabArena, "trim_empty", pause_first_trim)
    closer = threading.Thread(target=lease.close)
    closer.start()
    assert entered.wait(5)
    manager.max_pinned_bytes = None
    proceed.set()
    closer.join(5)
    monkeypatch.undo()
    assert manager.max_pinned_bytes is None
    assert manager._tier2a.max_bytes is None
    manager.clear()

    # (b) Setting max_host_bytes = 0 mid-demotion prevents stale _PrivateCopy commit or limit overwrite.
    ckpt2 = _checkpoint(tmp_path, 2 * PAGE, "epoch_2b")
    manager2 = PinManager(2 * PAGE, max_host_bytes=6 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    with manager2.acquire([ckpt2]):
        pass
    alloc_entered = threading.Event()
    alloc_proceed = threading.Event()
    orig_alloc = HostSlabArena.allocate

    def pause_tier2b_alloc(self: HostSlabArena, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self is manager2._tier2b and not alloc_entered.is_set():
            alloc_entered.set()
            assert alloc_proceed.wait(5)
        return orig_alloc(self, *args, **kwargs)

    monkeypatch.setattr(HostSlabArena, "allocate", pause_tier2b_alloc)
    demoter = threading.Thread(target=setattr, args=(manager2, "max_pinned_bytes", 0))
    demoter.start()
    assert alloc_entered.wait(5)
    manager2.max_host_bytes = 0
    alloc_proceed.set()
    demoter.join(5)
    monkeypatch.undo()
    assert manager2.max_host_bytes == 0
    assert manager2._tier2b.max_bytes == 0
    assert manager2.stats.private_copy_bytes == 0
    manager2.clear()


@pytest.mark.parametrize("source_kind", ["in_place", "tier2a"])
def test_sync_transfer_releases_manager_lock_during_copy(
    backend: FakeBackend, tmp_path, source_kind: str,
) -> None:
    manager = PinManager(8 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    if source_kind == "in_place":
        (source,) = _tensors((0, 2 * PAGE))
        source.copy_(torch.arange(2 * PAGE, dtype=torch.uint8) % 251)
    else:
        source = _checkpoint(tmp_path, 2 * PAGE, "ckpt")

    with manager.acquire([source]):
        pass
    assert len(backend.registered) == 1
    real_dst = torch.zeros_like(source)
    lock_free_results: list[bool] = []
    unregister_during_copy: list[int] = []

    class _CheckingDestination:
        device = torch.device("cpu")

        def copy_(self, view: torch.Tensor, *, non_blocking: bool) -> None:
            assert not non_blocking

            def probe() -> None:
                acquired = manager._lock.acquire(blocking=False)
                lock_free_results.append(acquired)
                if acquired:
                    manager._lock.release()
                manager.clear()
                unregister_during_copy.append(len(backend.unregister_calls))

            thread = threading.Thread(target=probe)
            thread.start()
            thread.join(5)
            real_dst.copy_(view, non_blocking=False)

    manager.transfer(_CheckingDestination(), source, non_blocking=False)  # type: ignore[arg-type]
    assert lock_free_results == [True]
    assert unregister_during_copy == [0]
    torch.testing.assert_close(real_dst, source)
    manager.clear()
    backend.unregister_calls.clear()


def test_bounce_ring_publish_eviction_exception_cleans_creating_ring(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = PinManager(
        256 * 1024,
        bounce_bytes=128 * 1024,
        bounce_slot_bytes=64 * 1024,
        min_bounce_bytes=64 * 1024,
        backend=backend,
    )
    (idle_buf,) = _tensors((0, 192 * 1024))
    with manager.acquire([idle_buf]):
        pass

    orig_evict = manager._evict_idle_for_budget
    evict_calls = 0

    def fail_on_publish_evict(
        target_pinned: int, *, protected: set[int], close_bounce_last: bool,
    ) -> bool:
        nonlocal evict_calls
        evict_calls += 1
        # First call frees space in _compute_bounce_target_bytes; inject error if called during publish,
        # or on the very first eviction when total > budget.
        raise RuntimeError("injected publish eviction error")

    # Force total_bounce + in_place > budget while bypassing _compute_bounce_target_bytes eviction.
    orig_compute = manager._compute_bounce_target_bytes
    monkeypatch.setattr(manager, "_compute_bounce_target_bytes", lambda: (True, 128 * 1024))
    monkeypatch.setattr(manager, "_evict_idle_for_budget", fail_on_publish_evict)
    with pytest.raises(RuntimeError, match="injected publish eviction error"), manager._use_bounce_ring():
        pass
    assert evict_calls == 1
    assert manager._resources.creating_ring is None
    assert manager._resources.bounce_ring is None
    assert not manager._bounce_creating
    assert manager.stats.bounce_bytes == 0
    monkeypatch.setattr(manager, "_compute_bounce_target_bytes", orig_compute)
    monkeypatch.setattr(manager, "_evict_idle_for_budget", orig_evict)
    manager.clear()
    assert backend.registered == {}


def test_demotion_respects_concurrent_pinned_growth_under_max_host_bytes(
    backend: FakeBackend, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_ckpt = _checkpoint(tmp_path, 2 * PAGE, "active")
    idle_ckpt = _checkpoint(tmp_path, 2 * PAGE, "idle")
    (anon_buf,) = _tensors((0, 4 * PAGE))
    manager = PinManager(8 * PAGE, max_host_bytes=8 * PAGE, slab_bytes=4 * PAGE, backend=backend)
    with manager.acquire([active_ckpt, idle_ckpt]):
        pass
    active_lease = manager.acquire([active_ckpt])
    assert manager.stats.copy_bytes == 4 * PAGE

    entered = threading.Event()
    proceed = threading.Event()
    orig_alloc = HostSlabArena.allocate

    def pause_tier2b_alloc(self: HostSlabArena, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self is manager._tier2b and not entered.is_set():
            entered.set()
            assert proceed.wait(5)
        return orig_alloc(self, *args, **kwargs)

    monkeypatch.setattr(HostSlabArena, "allocate", pause_tier2b_alloc)
    demoter = threading.Thread(target=setattr, args=(manager, "max_pinned_bytes", 2 * PAGE))
    demoter.start()
    assert entered.wait(5)

    # Grow pinned_bytes concurrently to consume the full 8*PAGE host budget.
    manager.max_pinned_bytes = 8 * PAGE
    anon_lease = manager.acquire([anon_buf])
    proceed.set()
    demoter.join(5)
    monkeypatch.undo()

    assert manager.stats.pinned_bytes == 8 * PAGE
    assert manager.stats.private_copy_bytes == 0
    assert manager.stats.pinned_bytes + manager.stats.private_copy_bytes <= 8 * PAGE
    assert manager._tier2b.max_bytes == 0
    anon_lease.close()
    active_lease.close()
    manager.clear()


def test_rollback_acquire_preserves_registration_leased_by_another_thread(
    backend: FakeBackend, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (first_anon,) = _tensors((0, 2 * PAGE))
    second_ckpt = _checkpoint(tmp_path, 2 * PAGE, "second_ckpt")
    manager = PinManager(8 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    first_ptr = first_anon.untyped_storage().data_ptr()

    other_leases: list[PinLease] = []
    orig_read_into = manager._reader.read_into

    def hijack_first_anon_during_fill(requests) -> None:
        other_leases.append(manager.acquire([first_anon]))
        orig_read_into(requests)

    def fail_view(self: ArenaAllocation, tensor: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("injected view failure after hijack")

    monkeypatch.setattr(manager._reader, "read_into", hijack_first_anon_during_fill)
    monkeypatch.setattr(ArenaAllocation, "tensor_view", fail_view)
    with pytest.raises(RuntimeError, match="injected view failure after hijack"):
        manager.acquire([first_anon, second_ckpt])

    assert len(other_leases) == 1
    assert first_ptr in manager._registrations
    assert not manager._registrations[first_ptr].retired
    assert manager._registrations[first_ptr].leases == 1
    assert first_ptr in backend.registered

    other_leases[0].close()
    assert first_ptr in manager._registrations
    assert not manager._registrations[first_ptr].retired
    assert manager.stats.registrations == 1
    manager.clear()
    assert backend.registered == {}


def test_transient_transfer_enforces_lowered_budget_when_external_lease_closes(
    backend: FakeBackend,
) -> None:
    (src,) = _tensors((0, 2 * PAGE))
    src.copy_(torch.arange(2 * PAGE, dtype=torch.uint8) % 251)
    manager = PinManager(4 * PAGE, backend=backend)
    ext_lease = manager.acquire([src])
    manager.max_pinned_bytes = 0
    assert manager.stats.pinned_bytes == 2 * PAGE

    real_dst = torch.zeros_like(src)
    registered_during_copy: list[bool] = []

    class _ClosingDst:
        device = torch.device("cpu")

        def copy_(self, view: torch.Tensor, *, non_blocking: bool) -> None:
            ext_lease.close()
            # Transient transfer hold still keeps the registration pinned until copy_ finishes.
            registered_during_copy.append(manager.stats.registrations == 1)
            real_dst.copy_(view, non_blocking=non_blocking)

    manager.transfer(_ClosingDst(), src, non_blocking=False)  # type: ignore[arg-type]
    assert registered_during_copy == [True]
    assert manager.stats.pinned_bytes == 0
    assert manager.stats.registrations == 0
    assert backend.registered == {}
    torch.testing.assert_close(real_dst, src)


def test_acquire_post_commit_error_closes_lease_without_double_rollback(
    backend: FakeBackend, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = _checkpoint(tmp_path, 2 * PAGE, "post_commit_interrupt")
    (pageable_anon,) = _tensors((0, 2 * PAGE))
    manager = PinManager(2 * PAGE, max_host_bytes=4 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    orig_enforce_host = manager._enforce_host_budget
    interrupted = False

    def interrupt_once_post_commit() -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("injected post-commit interrupt")
        orig_enforce_host()

    monkeypatch.setattr(manager, "_enforce_host_budget", interrupt_once_post_commit)
    with pytest.raises(KeyboardInterrupt, match="injected post-commit interrupt"):
        manager.acquire([ckpt, pageable_anon])

    monkeypatch.undo()
    gc.collect()
    assert manager.stats.active_leases == 0
    assert manager._leases == {}
    assert manager._leased_views == {}
    assert manager._pageable == {}
    assert all(reg.leases == 0 for reg in manager._registrations.values())
    assert all(priv.leases == 0 for priv in manager._private_copies.values())
    manager.clear()
    assert backend.registered == {}
    assert manager.stats.pinned_bytes == 0


@pytest.mark.parametrize("new_budget", [0, 4 * PAGE])
def test_cancelled_in_place_unregistration_failure_retains_retired_registration(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch, new_budget: int,
) -> None:
    manager = PinManager(8 * PAGE, backend=backend)
    (tensor,) = _tensors((0, 2 * PAGE))
    ptr = tensor.untyped_storage().data_ptr()
    orig_register = backend.register
    triggered = False

    def register_then_cancel_and_fail_unregister(pointer: int, size: int) -> bool:
        nonlocal triggered
        ok = orig_register(pointer, size)
        if not triggered:
            triggered = True
            backend.unregister_errors.add(pointer)
            manager.max_pinned_bytes = new_budget
        return ok

    monkeypatch.setattr(backend, "register", register_then_cancel_and_fail_unregister)
    lease = manager.acquire([tensor])
    monkeypatch.undo()

    assert len(backend.register_calls) == 1
    assert manager.stats.unregistration_failures == 1
    assert manager.stats.registrations == 1
    assert manager.stats.pinned_bytes == 2 * PAGE
    assert ptr in backend.registered
    assert manager._registrations[ptr].retired

    # Closing while unregister still fails retains the entry for retry via clear().
    lease.close()
    assert manager.stats.registrations == 1
    assert manager.stats.pinned_bytes == 2 * PAGE
    assert ptr in backend.registered

    backend.unregister_errors.clear()
    manager.clear()
    assert manager.stats.registrations == 0
    assert manager.stats.pinned_bytes == 0
    assert backend.registered == {}


def test_try_alloc_tier2a_span_releases_allocation_on_base_exception(
    backend: FakeBackend, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = _checkpoint(tmp_path, 2 * PAGE, "alloc_interrupt")
    manager = PinManager(4 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    orig_allocate = HostSlabArena.allocate
    orig_stats_prop = HostSlabArena.stats
    orig_stats_get = orig_stats_prop.fget
    assert orig_stats_get is not None
    allocated_after = False

    def recording_allocate(self: HostSlabArena, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal allocated_after
        alloc = orig_allocate(self, *args, **kwargs)
        if self is manager._tier2a and alloc is not None:
            allocated_after = True
        return alloc

    def interrupting_stats(self: HostSlabArena):  # type: ignore[no-untyped-def]
        if self is manager._tier2a and allocated_after:
            raise KeyboardInterrupt("injected post-allocate interrupt")
        return orig_stats_get(self)

    monkeypatch.setattr(HostSlabArena, "allocate", recording_allocate)
    monkeypatch.setattr(HostSlabArena, "stats", property(interrupting_stats))
    with pytest.raises(KeyboardInterrupt, match="injected post-allocate interrupt"):
        manager.acquire([ckpt])

    monkeypatch.undo()
    assert manager._pin_reserving == 0
    assert manager.stats.copy_bytes == 0
    assert backend.registered == {}
    with manager.acquire([ckpt]) as lease:
        assert lease.registered_bytes == 2 * PAGE
    manager.clear()
    assert backend.registered == {}


def test_clear_interrupt_releases_already_popped_tier2a_allocations(
    backend: FakeBackend, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = _checkpoint(tmp_path, 2 * PAGE, "clear_interrupt")
    (anon,) = _tensors((0, 2 * PAGE))
    manager = PinManager(8 * PAGE, slab_bytes=2 * PAGE, backend=backend)
    with manager.acquire([ckpt]):
        pass
    with manager.acquire([anon]):
        pass
    alloc = manager._registrations[ckpt.untyped_storage().data_ptr()].copy
    assert alloc is not None and not alloc.released

    orig_unregister = backend.unregister
    interrupted = False

    def interrupting_unregister_once(pointer: int) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("injected clear interrupt")
        orig_unregister(pointer)

    monkeypatch.setattr(backend, "unregister", interrupting_unregister_once)
    with pytest.raises(KeyboardInterrupt, match="injected clear interrupt"):
        manager.clear()

    monkeypatch.undo()
    assert alloc.released
    assert manager.stats.copy_bytes == 0
    manager.clear()
    assert manager.stats.pinned_bytes == 0
    assert backend.registered == {}


def test_bounce_ring_retained_headroom_and_oversubscribed_release(backend: FakeBackend) -> None:
    # Part A: retained rings count against bounce headroom and retry on budget pressure.
    manager = PinManager(
        512 * 1024,
        bounce_bytes=128 * 1024,
        bounce_slot_bytes=64 * 1024,
        min_bounce_bytes=64 * 1024,
        backend=backend,
    )
    with manager._use_bounce_ring() as ring:
        assert ring is not None
        (ring_ptr,) = tuple(backend.registered)
    backend.unregister_errors.add(ring_ptr)
    with pytest.raises(RuntimeError, match="Could not release"):
        manager.clear()
    assert len(manager._resources.retaining_rings) == 1
    assert manager.stats.bounce_bytes == 128 * 1024

    register_calls_before = len(backend.register_calls)
    with manager._use_bounce_ring() as second_ring:
        assert second_ring is None
    assert len(backend.register_calls) == register_calls_before

    backend.unregister_errors.clear()
    manager.max_pinned_bytes = 64 * 1024
    assert manager.stats.bounce_bytes == 0
    assert manager._resources.retaining_rings == []
    assert backend.registered == {}

    # Part B: an idle bounce ring whose total pinned bytes exceed a lowered budget closes on user exit.
    mgr2 = PinManager(
        512 * 1024,
        bounce_bytes=128 * 1024,
        bounce_slot_bytes=64 * 1024,
        min_bounce_bytes=64 * 1024,
        backend=backend,
    )
    (buf,) = _tensors((0, 256 * 1024))
    hold = mgr2.acquire([buf])
    with mgr2._use_bounce_ring() as ring:
        assert ring is not None
        assert mgr2.stats.pinned_bytes == 384 * 1024
        mgr2.max_pinned_bytes = 300 * 1024
        assert mgr2._resources.bounce_ring is ring
    assert mgr2._resources.bounce_ring is None
    assert mgr2.stats.bounce_bytes == 0
    assert mgr2.stats.pinned_bytes == 256 * 1024 <= 300 * 1024
    hold.close()
    mgr2.clear()
    assert backend.registered == {}
