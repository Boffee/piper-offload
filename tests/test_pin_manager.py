"""Registration ownership, page accounting, and explicit pin leases."""

import ctypes
import gc
import logging
import mmap
import os
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Self
from unittest.mock import Mock

import pytest
import torch

import piper_offload._copy_memory as copy_memory
import piper_offload._copy_memory_windows as windows_memory
import piper_offload._host_memory_linux as linux_memory
import piper_offload._host_memory_windows as memory_module
import piper_offload._host_registration as registration_module
import piper_offload.pin_manager as pin_module
from piper_offload._host_memory_windows import VirtualRegion
from piper_offload._host_registration import HostRegistrationError, RuntimeHostRegistration
from piper_offload import MappedCheckpoint, PinManager, file_slice, host_pin_manager
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


class FakeMemoryPriority:
    """Thread memory priorities as ``SetThreadInformation`` keeps them: per thread, starting at normal."""

    NORMAL = 5

    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self.changes: list[tuple[str, int]] = []

    def get(self) -> int:
        return getattr(self._local, "priority", self.NORMAL)

    def set(self, priority: int) -> None:
        self._local.priority = priority
        with self._lock:
            self.changes.append((threading.current_thread().name, priority))


class FakeKernel:
    """kernel32's memory calls over anonymous mappings, so the offered tier runs off Windows too.

    Offering poisons the pages and keeps their bytes, which a reclaim restores
    only when it reports them intact, so reading an offered or discarded copy
    shows up in whatever a transfer delivers.
    """

    POISON = 0xA5

    def __init__(self, events: list[tuple[str, int]]) -> None:
        self.events = events
        self.regions: dict[int, mmap.mmap] = {}
        self.saved: dict[tuple[int, int], bytes] = {}
        self.reclaim_code = 0  # 0 keeps the pages, 170 (ERROR_BUSY) discards them, anything else fails
        self.offer_code = 0
        self.allocation_failures = 0
        # Spans, as (region, offset), that Windows drops while the rest survives.
        self.discarded_spans: set[tuple[int, int]] = set()
        self.priority = FakeMemoryPriority()
        # The kernel's MaximumCommitCondition: set when Windows can commit no more.
        self.commit_exhausted = threading.Event()

    def _owner(self, address: int) -> tuple[int, int]:
        """The region an offered span belongs to, and where in it the span starts."""
        base = max(start for start in self.regions if start <= address)
        return base, address - base

    def allocate(self, _address: None, size: int, _type: int, _protect: int) -> int:
        if self.allocation_failures:
            self.allocation_failures -= 1
            self.events.append(("allocation failure", size))
            return 0
        region = mmap.mmap(-1, size)
        borrowed = ctypes.c_char.from_buffer(region)
        pointer = ctypes.addressof(borrowed)
        del borrowed  # the mapping must be closable again
        self.regions[pointer] = region
        self.events.append(("allocate", pointer))
        return pointer

    def free(self, pointer: int, _size: int, _type: int) -> int:
        self.events.append(("free", pointer))
        for span in [span for span in self.saved if span[0] == pointer]:
            del self.saved[span]
        self.regions.pop(pointer).close()
        return 1

    def offer(self, address: int, size: int, priority: int) -> int:
        base, offset = self._owner(address)
        self.events.append(("offer", base))
        assert priority == 4  # VMOfferPriorityNormal: VeryLow is dropped wholesale under any pressure
        if self.offer_code:
            return self.offer_code
        with memoryview(self.regions[base]) as pages:
            self.saved[(base, offset)] = bytes(pages[offset : offset + size])
            pages[offset : offset + size] = bytes([self.POISON]) * size
        return 0

    def reclaim(self, address: int, size: int) -> int:
        base, offset = self._owner(address)
        self.events.append(("reclaim", base))
        code = self.reclaim_code or (170 if (base, offset) in self.discarded_spans else 0)
        saved = self.saved.pop((base, offset), None)
        if code == 0 and saved is not None:
            with memoryview(self.regions[base]) as pages:
                pages[offset : offset + size] = saved
        return code


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def manager(backend: FakeBackend):
    result = PinManager(4 * PAGE, backend=backend)
    yield result
    backend.unregister_errors.clear()
    result.clear()


def install_fake_kernel(monkeypatch: pytest.MonkeyPatch, events: list[tuple[str, int]]) -> FakeKernel:
    """Allocate copies in offerable regions over a fake kernel32, on any platform."""
    kernel = FakeKernel(events)
    monkeypatch.setattr(memory_module, "_kernel32", lambda: kernel)
    monkeypatch.setattr(memory_module, "_thread_memory_priority", lambda: kernel.priority)
    monkeypatch.setattr(windows_memory, "commit_exhausted", lambda timeout=0.0: kernel.commit_exhausted.wait(timeout))
    monkeypatch.setattr(pin_module.host_memory, "Memory", windows_memory.Memory)
    return kernel


@pytest.fixture
def offering(backend: FakeBackend, monkeypatch: pytest.MonkeyPatch) -> FakeKernel:
    return install_fake_kernel(monkeypatch, backend.events)


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Every (offset, length) a fill reads from a checkpoint."""
    recorded: list[tuple[int, int]] = []
    original = copy_memory._read_range

    def recording(read_at, view, offset, start, stop) -> None:
        recorded.append((offset + start, stop - start))
        original(read_at, view, offset, start, stop)

    monkeypatch.setattr(copy_memory, "_read_range", recording)
    return recorded


def test_default_budget_is_finite_without_initializing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_runtime():
        raise AssertionError("construction and configuration must not initialize CUDA/HIP")

    monkeypatch.setattr(registration_module, "_load_runtime", unexpected_runtime)
    manager = PinManager()
    assert manager.max_pinned_bytes == pin_module._default_pin_budget()
    assert manager.max_pinned_bytes > 0
    assert manager.max_pinned_bytes % PAGE == 0
    assert host_pin_manager.max_pinned_bytes == manager.max_pinned_bytes
    assert PinManager(None).max_pinned_bytes is None
    manager.max_pinned_bytes = PAGE
    manager.max_pinned_bytes = None
    assert manager.stats.max_pinned_bytes is None
    assert manager.stats.pinned_bytes == 0


def test_default_budget_is_half_of_physical_ram(monkeypatch) -> None:
    monkeypatch.setattr(pin_module.host_memory, "available_memory", linux_memory.available_memory)
    values = {"SC_PHYS_PAGES": 101, "SC_PAGE_SIZE": PAGE}
    monkeypatch.setattr(linux_memory.os, "sysconf", values.__getitem__, raising=False)
    monkeypatch.setattr(linux_memory, "_PROC_CGROUP", "/nonexistent/cgroup")
    assert pin_module._default_pin_budget() == 50 * PAGE


def _cgroups(monkeypatch, tmp_path, membership: str, files: dict[str, str]) -> None:
    """Fake the process's cgroup membership and a cgroup mount holding ``files``."""
    monkeypatch.setattr(pin_module.host_memory, "available_memory", linux_memory.available_memory)
    values = {"SC_PHYS_PAGES": 101, "SC_PAGE_SIZE": PAGE}
    monkeypatch.setattr(linux_memory.os, "sysconf", values.__getitem__, raising=False)
    (tmp_path / "cgroup").write_text(membership)
    for relative, text in files.items():
        path = tmp_path / "mount" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    monkeypatch.setattr(linux_memory, "_PROC_CGROUP", str(tmp_path / "cgroup"))
    monkeypatch.setattr(linux_memory, "_CGROUP_MOUNT", str(tmp_path / "mount"))


@pytest.mark.parametrize(
    ("membership", "files", "expected_pages"),
    [
        ("0::/a/b\n", {"a/b/memory.max": "20", "a/memory.max": "max", "memory.max": "max"}, 10),
        ("0::/a/b\n", {"a/b/memory.max": "max", "a/memory.max": "12"}, 6),
        ("0::/\n", {"memory.max": "30"}, 15),
        ("0::/a\n", {"a/memory.max": "max", "memory.max": "max"}, 50),
        ("4:memory:/a\n", {"memory/a/memory.limit_in_bytes": "16"}, 8),
        ("4:memory:/a\n", {"memory/a/memory.limit_in_bytes": "9223372036854771712"}, 50),
        ("3:cpu:/a\n", {"cpu/a/memory.max": "2"}, 50),
    ],
    ids=[
        "v2-own", "v2-ancestor", "v2-namespaced-root", "v2-unlimited", "v1", "v1-unlimited", "no-memory-controller",
    ],
)
def test_default_budget_is_bounded_by_the_process_cgroup(
    monkeypatch, tmp_path, membership, files, expected_pages,
) -> None:
    # Small numbers are page counts; the v1 unlimited sentinel is passed through.
    scaled = {
        name: str(int(text) * PAGE) if text.isdigit() and len(text) < 10 else text for name, text in files.items()
    }
    _cgroups(monkeypatch, tmp_path, membership, scaled)
    assert pin_module._default_pin_budget() == expected_pages * PAGE


@pytest.mark.parametrize("success", [False, True])
def test_windows_default_budget_uses_physical_ram_without_cuda(monkeypatch, success) -> None:
    def query(pointer):
        status = pointer._obj
        assert status.length == 64
        status.total_physical = 101 * PAGE
        return success

    monkeypatch.setattr(pin_module.host_memory, "available_memory", memory_module.available_memory)
    monkeypatch.setattr(
        memory_module.ctypes, "WinDLL",
        lambda *_args, **_kwargs: SimpleNamespace(GlobalMemoryStatusEx=Mock(side_effect=query)), raising=False,
    )
    assert pin_module._default_pin_budget() == (50 * PAGE if success else 0)


def test_unknown_physical_ram_disables_default_admission(monkeypatch) -> None:
    monkeypatch.setattr(pin_module.host_memory, "available_memory", linux_memory.available_memory)
    monkeypatch.setattr(linux_memory.os, "sysconf", Mock(side_effect=OSError("unavailable")), raising=False)
    assert pin_module._default_pin_budget() == 0


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
    manager = PinManager(4 * PAGE, backend=backend)
    (anonymous,) = _tensors((0, PAGE))
    with manager.acquire([tensor, anonymous]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (tensor.nbytes + PAGE, 0)
        calls = dict(backend.register_calls)
        assert calls.pop(anonymous.data_ptr()) == PAGE
        # The mapping itself is never registered; its copy is page-aligned and page-rounded.
        ((copy_pointer, copy_size),) = calls.items()
        assert copy_pointer != tensor.data_ptr() and copy_pointer % PAGE == 0 and copy_size == 2 * PAGE
        assert manager.stats.pinned_bytes == 3 * PAGE and manager.stats.copy_bytes == 2 * PAGE
        region = manager._registrations[tensor.untyped_storage().data_ptr()].copy.region
        torch.testing.assert_close(_transferred(manager, tensor), tensor)
        # Transfers read the copy, not the mapping: a byte changed in the copy
        # shows up in what a transfer delivers, through a view's geometry too.
        with memoryview(region) as bytes_:
            bytes_[300] = (tensor[300].item() + 1) % 256
            expected = torch.frombuffer(bytearray(bytes_[300:1000]), dtype=torch.uint8).view(torch.int16)
        assert _transferred(manager, tensor)[300] != tensor[300]
        part = tensor[300:1000].view(torch.int16)
        torch.testing.assert_close(_transferred(manager, part), expected)
        torch.testing.assert_close(_transferred(manager, anonymous), anonymous)
    manager.clear()
    assert set(backend.unregister_calls) == {copy_pointer, anonymous.data_ptr()}
    assert region.closed
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


def test_copy_is_reserved_at_allocation_and_evicted_by_freeing(backend: FakeBackend, tmp_path) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(2 * PAGE, backend=backend)
    with manager.acquire([first, second]) as lease:
        # Only one copy fits, and it is reserved before the second is considered.
        assert (lease.registered_bytes, lease.pageable_bytes) == (2 * PAGE, 2 * PAGE)
        assert len(backend.register_calls) == 1
        region = manager._registrations[first.untyped_storage().data_ptr()].copy.region
    with manager.acquire([second]):
        # The idle copy is evicted to make room: unregistered and freed.
        assert backend.unregister_calls == [backend.register_calls[0][0]]
        assert region.closed
        assert manager.stats.copy_bytes == 2 * PAGE
    manager.clear()


@pytest.mark.skipif(not hasattr(os, "preadv"), reason="positional reads")
def test_sliced_fill_covers_the_copy_through_short_reads(backend: FakeBackend, tmp_path, monkeypatch) -> None:
    tensor = _checkpoint(tmp_path, 8 * PAGE)
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
    monkeypatch.setattr(copy_memory, "_FILL_SLICE", PAGE)
    manager = PinManager(16 * PAGE, backend=backend)
    with manager.acquire([tensor]):
        torch.testing.assert_close(_transferred(manager, tensor), tensor)
    base = file_slice(tensor).offset
    # Every byte was read exactly once, in slices of one page each.
    covered = sorted((offset - base, count) for offset, count in reads)
    assert sum(count for _offset, count in covered) == 8 * PAGE
    assert [start for start, _count in covered if start % PAGE == 0] == [page * PAGE for page in range(8)]
    manager.clear()


def test_copies_of_one_acquisition_fill_together(backend: FakeBackend, tmp_path, monkeypatch) -> None:
    first = _checkpoint(tmp_path, 8 * PAGE, "first")
    second = _checkpoint(tmp_path, 4 * PAGE, "second")

    class _Pool:
        """Stands in for the executor: a slice reads when awaited, noting how many slices were submitted by then."""

        in_flight_at_wait: list[int] = []

        def __init__(self, max_workers: int, *, thread_name_prefix: str) -> None:
            self.submitted = 0
            # Only the fill's slices are counted; the touch that follows it has a pool of its own.
            self.counted = "fill" in thread_name_prefix

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            pass

        def submit(self, function, *args) -> SimpleNamespace:
            self.submitted += 1

            def result() -> None:
                if self.counted:
                    self.in_flight_at_wait.append(self.submitted)
                function(*args)

            return SimpleNamespace(result=result)

    third = _checkpoint(tmp_path, 2 * PAGE, "third")
    monkeypatch.setattr(copy_memory, "ThreadPoolExecutor", _Pool)
    monkeypatch.setattr(copy_memory, "_FILL_SLICE", PAGE)
    manager = PinManager(16 * PAGE, backend=backend)
    with manager.acquire([first, second, third]) as lease:
        assert lease.registered_bytes == 14 * PAGE
        for tensor in (first, second, third):
            torch.testing.assert_close(_transferred(manager, tensor), tensor)
    # The first copy filled alone; every slice of the other two was submitted before any was waited on.
    assert _Pool.in_flight_at_wait == [8] * 8 + [6] * 6
    manager.clear()


def test_a_fill_without_positional_reads_uses_a_handle_per_worker(
    backend: FakeBackend, tmp_path, monkeypatch,
) -> None:
    """A Windows handle has one position, so each worker reads through its own; they close with the fill."""
    monkeypatch.setattr(pin_module.host_memory.Memory, "readers", memory_module.Readers)
    monkeypatch.setattr(copy_memory, "_FILL_SLICE", PAGE)
    tensors = [_checkpoint(tmp_path, 8 * PAGE, f"copy{index}") for index in range(3)]
    provenance = {file_slice(tensor).file for tensor in tensors}
    opened: list = []
    real_open = open

    def recording_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(memory_module, "open", recording_open, raising=False)
    manager = PinManager(64 * PAGE, backend=backend)
    with manager.acquire(tensors) as lease:
        assert lease.registered_bytes == 24 * PAGE
        for tensor in tensors:
            torch.testing.assert_close(_transferred(manager, tensor), tensor)
    # One handle each at least, none of them the reader's own, and none left open.
    assert len(opened) >= len(tensors)
    assert provenance.isdisjoint(opened)
    assert all(handle.closed for handle in opened)
    manager.clear()


def test_failed_fill_leaves_the_storage_pageable(backend: FakeBackend, tmp_path, monkeypatch, caplog) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)

    def broken(*_args):
        raise OSError("injected read failure")

    monkeypatch.setattr(copy_memory, "_read_range", broken)
    manager = PinManager(4 * PAGE, backend=backend)
    with caplog.at_level(logging.WARNING, logger="piper_offload"), manager.acquire([tensor]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert backend.register_calls == [] and manager.stats.pinned_bytes == 0
    assert "injected read failure" in caplog.text
    manager.clear()


def test_copy_registration_error_rolls_back_the_acquisition(backend: FakeBackend, tmp_path, monkeypatch) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(8 * PAGE, backend=backend)
    register = backend.register

    def broken_after_one(pointer: int, size: int) -> bool:
        if backend.register_calls:
            raise HostRegistrationError("registration", 700)
        return register(pointer, size)

    with monkeypatch.context() as patch:
        patch.setattr(backend, "register", broken_after_one)
        with pytest.raises(HostRegistrationError):
            manager.acquire([first, second])
    # The first copy was registered and is retired; the second was never registered. Both are freed.
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

    def exhausted(_size: int) -> None:
        raise MemoryError("injected allocation failure")

    monkeypatch.setattr(pin_module.host_memory.Memory, "new_copy", lambda _self, size: exhausted(size))
    manager = PinManager(4 * PAGE, backend=backend)
    with caplog.at_level(logging.WARNING, logger="piper_offload"), manager.acquire([tensor]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert backend.register_calls == [] and manager.stats.pinned_bytes == 0
    assert "injected allocation failure" in caplog.text
    manager.clear()


def test_a_pinning_acquisition_waits_for_a_pending_fill_while_a_pageable_one_proceeds(
    backend: FakeBackend, tmp_path, monkeypatch,
) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    pinning, pageable = _tensors((0, PAGE), (2 * PAGE, PAGE))
    manager = PinManager(8 * PAGE, backend=backend)
    original = copy_memory._fill_copies
    leases: dict[str, PinLease] = {}
    pinner = threading.Thread(target=lambda: leases.__setitem__("pinning", manager.acquire([pinning])))
    pager = threading.Thread(target=lambda: leases.__setitem__("pageable", manager.acquire([pageable], pin=False)))
    during_fill: list[bool] = []

    def fill_while_others_acquire(pending, **options):
        pinner.start()
        pager.start()
        pager.join(2.0)
        pinner.join(0.2)
        during_fill.extend((pager.is_alive(), pinner.is_alive()))
        original(pending, **options)

    monkeypatch.setattr(copy_memory, "_fill_copies", fill_while_others_acquire)
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
    original = copy_memory._fill_copies
    rejected: list[bool] = []

    def fill_while_an_alias_is_acquired(pending, **options):
        # The in-place storage is reserved but not yet registered; a distinct
        # storage overlapping it is still rejected, as it is once registered.
        with pytest.raises(ValueError, match="Overlapping"):
            manager.acquire([alias], pin=False)
        rejected.append(True)
        original(pending, **options)

    monkeypatch.setattr(copy_memory, "_fill_copies", fill_while_an_alias_is_acquired)
    with manager.acquire([tensor, anonymous]) as lease:
        assert rejected == [True] and lease.registered_bytes == 4 * PAGE
    manager.clear()


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
    manager = PinManager(8 * PAGE, backend=backend)
    with manager.acquire(copies) as lease:
        # The first copy registers, the second is refused, and the third is
        # not attempted; both of those are freed and only the first is reserved.
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
            # Registration follows request order whether the storage registers
            # in place or through a copy, as it did before copies filled together.
            assert (lease.registered_bytes, lease.pageable_bytes) == (2 * PAGE, 2 * PAGE)
            assert len(backend.register_calls) == 2 and manager.stats.registration_failures == 1
            assert manager.stats.copy_bytes == (2 * PAGE if first is copy else 0)
        manager.clear()
        assert manager.stats.pinned_bytes == 0
        backend.register_calls.clear()


def test_an_exhausted_runtime_is_found_before_the_rest_is_read(backend: FakeBackend, tmp_path, monkeypatch) -> None:
    copies = [_checkpoint(tmp_path, 2 * PAGE, name) for name in ("first", "second", "third")]
    backend.capacity = PAGE
    manager = PinManager(8 * PAGE, backend=backend)
    original = copy_memory._fill_copies
    filled: list[int] = []

    def counting_fill(pending, **options):
        filled.append(len(pending))
        original(pending, **options)

    monkeypatch.setattr(copy_memory, "_fill_copies", counting_fill)
    with manager.acquire(copies) as lease:
        # Only the first copy was read: its refusal evicted the other two unfilled.
        assert filled == [1] and len(backend.register_calls) == 1
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


@pytest.mark.skipif(not hasattr(os, "preadv"), reason="positional reads")
def test_one_failed_fill_leaves_the_other_copies_registered(
    backend: FakeBackend, tmp_path, monkeypatch, caplog,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    broken_fd = file_slice(second).file.fileno()
    original = os.preadv

    def failing_preadv(fd, buffers, offset):
        if fd == broken_fd:
            raise OSError("injected read failure")
        return original(fd, buffers, offset)

    monkeypatch.setattr(os, "preadv", failing_preadv)
    manager = PinManager(8 * PAGE, backend=backend)
    warnings = caplog.at_level(logging.WARNING, logger="piper_offload")
    with warnings, manager.acquire([first, second]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (2 * PAGE, 2 * PAGE)
        assert len(backend.register_calls) == 1 and manager.stats.copy_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, first), first)
        torch.testing.assert_close(_transferred(manager, second), second)
    assert "injected read failure" in caplog.text
    manager.clear()


def test_an_acquisition_of_pending_storage_waits_until_it_registers(
    backend: FakeBackend, tmp_path, monkeypatch,
) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    manager = PinManager(8 * PAGE, backend=backend)
    original = copy_memory._fill_copies
    other: list[PinLease] = []
    thread = threading.Thread(target=lambda: other.append(manager.acquire([tensor], pin=False)))
    during_fill: list[tuple[int, bool]] = []

    def fill_while_another_acquisition_waits(pending, **options):
        thread.start()
        thread.join(0.2)
        during_fill.append((manager.stats.copy_bytes, thread.is_alive()))
        original(pending, **options)

    monkeypatch.setattr(copy_memory, "_fill_copies", fill_while_another_acquisition_waits)
    with manager.acquire([tensor]) as lease:
        thread.join()
        assert lease.registered_bytes == 2 * PAGE
        # The fill ran without the lock, yet the pageable acquisition of the
        # same storage waited for the copy and holds its registration, rather
        # than tracking the storage as pageable; the copy counted meanwhile.
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
    # Idle registrations, the copy and the in-place one, may be evicted at any
    # moment, so an asynchronous read without a lease is refused, not raced.
    for source in (tensor, anonymous):
        with pytest.raises(RuntimeError, match="outside a lease"):
            manager.transfer(destination[: source.numel()], source, non_blocking=True)
    with manager.acquire([tensor], pin=False):
        manager.transfer(destination, tensor, non_blocking=True)
        torch.cuda.synchronize()
    torch.testing.assert_close(destination.cpu(), tensor)
    manager.clear()


def test_clear_frees_every_copy_off_the_lock(backend: FakeBackend, tmp_path) -> None:
    """Freeing runs on workers with the lock released, so a finalizer there cannot deadlock.

    The lock is reentrant, so the caller could always reacquire it; what matters
    is that the copies are freed from other threads and that those threads can
    take the lock, which they could not if ``clear`` still held it.
    """
    tensors = [_checkpoint(tmp_path, 2 * PAGE, f"copy{index}") for index in range(3)]
    manager = PinManager(backend=backend)
    with manager.acquire(tensors):
        pass
    regions = [
        manager._registrations[tensor.untyped_storage().data_ptr()].copy.region for tensor in tensors
    ]
    freed_by: list[tuple[str, bool]] = []
    original = copy_memory.Copy.free

    def watch_free(copy: copy_memory.Copy) -> None:
        acquired = manager._lock.acquire(blocking=False)
        if acquired:
            manager._lock.release()
        freed_by.append((threading.current_thread().name, acquired))
        original(copy)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(copy_memory.Copy, "free", watch_free)
        manager.clear()

    assert all(region.closed for region in regions)
    assert len(freed_by) == 3
    assert all(name.startswith("piper-offload-free") for name, _ in freed_by), freed_by
    assert all(acquired for _, acquired in freed_by), freed_by
    assert (manager.stats.pinned_bytes, manager.stats.copy_bytes) == (0, 0)


WINDOWS = pytest.mark.skipif(sys.platform != "win32", reason="Windows offered memory")


def _copy_pointer(backend: FakeBackend, index: int = 0) -> int:
    return backend.register_calls[index][0]


def _offered_manager(backend: FakeBackend, budget: int = 2 * PAGE, offered: int = 4 * PAGE) -> PinManager:
    return PinManager(budget, max_offered_bytes=offered, backend=backend)


def _offer_first(backend: FakeBackend, tmp_path) -> tuple[PinManager, torch.Tensor, torch.Tensor]:
    """A manager that pinned two checkpoints in turn, so the first one's copy is offered."""
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = _offered_manager(backend)
    for tensor in (first, second):
        with manager.acquire([tensor]):
            pass
    return manager, first, second


def _offer_three(backend: FakeBackend, tmp_path) -> tuple[PinManager, list[torch.Tensor]]:
    """A manager whose tier holds three copies, offered to make room for a fourth checkpoint."""
    tensors = [_checkpoint(tmp_path, 2 * PAGE, f"copy{index}") for index in range(3)]
    other = _checkpoint(tmp_path, 8 * PAGE, "other")
    manager = PinManager(8 * PAGE, max_offered_bytes=16 * PAGE, backend=backend)
    with manager.acquire(tensors):
        pass
    with manager.acquire([other]):
        pass
    assert manager.stats.offered_bytes == 6 * PAGE
    return manager, tensors


def _offered_after_unregistering(backend: FakeBackend, pointer: int) -> bool:
    """Whether ``pointer`` was offered, and only while the runtime had released its registration."""
    offer = backend.events.index(("offer", pointer))
    runtime = [event for event in backend.events[:offer] if event in (("register", pointer), ("unregister", pointer))]
    return bool(runtime) and runtime[-1] == ("unregister", pointer)


def test_an_eviction_offers_a_copy_only_after_its_unregistration(backend, offering, tmp_path) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE - 100, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = _offered_manager(backend)
    with manager.acquire([first]):
        pass
    offered_pointer = _copy_pointer(backend)
    with manager.acquire([second]) as lease:
        # Room for the second copy comes from the first, offered only once its
        # own unregistration succeeded and before the copy that replaces it
        # registers, which is before anything is read into that one.
        assert _offered_after_unregistering(backend, offered_pointer)
        replacement = _copy_pointer(backend, 1)
        assert backend.events.index(("offer", offered_pointer)) < backend.events.index(("register", replacement))
        assert sorted(backend.events) == sorted([
            ("allocate", offered_pointer), ("register", offered_pointer), ("unregister", offered_pointer),
            ("offer", offered_pointer), ("allocate", replacement), ("register", replacement),
        ])
        assert lease.registered_bytes == 2 * PAGE
        # The offered pages are whole and charged to their own budget, not the pin budget.
        stats = manager.stats
        assert (stats.offered_bytes, stats.max_offered_bytes) == (2 * PAGE, 4 * PAGE)
        assert (stats.pinned_bytes, stats.copy_bytes) == (2 * PAGE, 2 * PAGE)
        # Nothing reads the offered copy: a transfer of its storage takes the mapping.
        torch.testing.assert_close(_transferred(manager, first), first)
    manager.clear()
    assert (manager.stats.offered_bytes, manager.stats.pinned_bytes) == (0, 0)


def test_reclaiming_intact_pages_registers_the_copy_without_reading_the_file(
    backend, offering, reads, tmp_path,
) -> None:
    manager, first, second = _offer_first(backend, tmp_path)
    reclaimed, evicted = _copy_pointer(backend), _copy_pointer(backend, 1)
    reads.clear()
    backend.events.clear()
    with manager.acquire([first]) as lease:
        # The copy comes back intact, so it is registered again with nothing
        # read and nothing allocated, and delivers the checkpoint's bytes.
        assert backend.events == [
            ("unregister", evicted),
            ("offer", evicted),
            ("reclaim", reclaimed),
            ("register", reclaimed),
        ]
        assert reads == []
        assert lease.registered_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, first), first)
    assert (manager.stats.offered_bytes, manager.stats.pinned_bytes) == (2 * PAGE, 2 * PAGE)
    manager.clear()


def test_discarded_pages_are_refilled_in_full_before_the_copy_is_registered(
    backend, offering, reads, tmp_path, monkeypatch,
) -> None:
    manager, first, second = _offer_first(backend, tmp_path)
    offering.reclaim_code = 170  # ERROR_BUSY: Windows discarded the pages
    reads.clear()
    registered_during_fill: list[bool] = []
    original = copy_memory._fill_copies

    def fill_while_unregistered(pending, **options) -> None:
        registered_during_fill.append(_copy_pointer(backend) in backend.registered)
        original(pending, **options)

    monkeypatch.setattr(copy_memory, "_fill_copies", fill_while_unregistered)
    with manager.acquire([first]) as lease:
        # Every byte is read back, while the copy is still unregistered, so no
        # transfer can reach the undefined contents.
        assert sum(length for _offset, length in reads) == 2 * PAGE
        assert min(offset for offset, _length in reads) == file_slice(first).offset
        assert registered_during_fill == [False]
        assert lease.registered_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, first), first)
    manager.clear()


@pytest.mark.parametrize(("code", "pages_read"), [(0, 4), (170, 12)], ids=["intact", "discarded"])
def test_rotation_reuses_offered_copies_and_preserves_every_byte(
    code, pages_read, backend, offering, reads, tmp_path,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    offering.reclaim_code = code
    manager = _offered_manager(backend, offered=2 * PAGE)
    for step in range(6):
        tensor = (first, second)[step % 2]
        with manager.acquire([tensor]) as lease:
            assert lease.registered_bytes == 2 * PAGE
            torch.testing.assert_close(_transferred(manager, tensor), tensor)
        # From the first switch on, the idle copy is always the offered one.
        assert manager.stats.offered_bytes == (0 if step == 0 else 2 * PAGE)
    # Both copies were built once; every later switch reclaimed one, reading
    # only what Windows discarded. One copy is offered throughout.
    assert sum(length for _offset, length in reads) == pages_read * PAGE
    assert len([event for event in backend.events if event[0] == "allocate"]) == 2
    assert (manager.stats.pinned_bytes, manager.stats.offered_bytes) == (2 * PAGE, 2 * PAGE)
    manager.clear()
    assert (manager.stats.pinned_bytes, manager.stats.offered_bytes) == (0, 0)


def test_views_of_one_storage_share_a_single_offered_copy(backend, offering, reads, tmp_path) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE, "first")
    other = _checkpoint(tmp_path, 2 * PAGE, "second")
    views = [tensor[:PAGE].view(torch.int16), tensor[PAGE + 64:], tensor]
    manager = _offered_manager(backend)
    with manager.acquire(views):
        pass
    with manager.acquire([other]):
        pass
    reads.clear()
    with manager.acquire(views) as lease:
        # One storage, one copy: the views share the reclaim and the registration.
        assert lease.registered_bytes == 2 * PAGE
        assert [event for event in backend.events if event[0] == "reclaim"] == [("reclaim", _copy_pointer(backend))]
        assert manager.stats.registrations == 1
        assert reads == []
        for view in views:
            torch.testing.assert_close(_transferred(manager, view), view)
    manager.clear()


def test_a_copy_is_offered_only_once_every_lease_over_it_has_ended(backend, offering, tmp_path) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = _offered_manager(backend)
    pinning = manager.acquire([first])
    transfer = manager.acquire([first], pin=False)
    offered_pointer = _copy_pointer(backend)
    pinning.close()
    with manager.acquire([second]) as lease:
        # The transfer lease still holds the copy, so it is neither unregistered nor offered.
        assert backend.unregister_calls == [] and ("offer", offered_pointer) not in backend.events
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert manager.stats.offered_bytes == 0
    transfer.close()
    with manager.acquire([second]) as lease:
        assert backend.unregister_calls == [offered_pointer]
        assert _offered_after_unregistering(backend, offered_pointer)
        assert lease.registered_bytes == 2 * PAGE
    manager.clear()


def test_a_failed_unregistration_keeps_the_copy_registered_and_charged(backend, offering, tmp_path) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = _offered_manager(backend)
    with manager.acquire([first]):
        pass
    offered_pointer = _copy_pointer(backend)
    backend.unregister_errors.add(offered_pointer)
    with manager.acquire([second]) as lease:
        # Nothing may be offered while the runtime still holds the registration.
        assert ("offer", offered_pointer) not in backend.events
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        stats = manager.stats
        assert (stats.pinned_bytes, stats.copy_bytes, stats.offered_bytes) == (2 * PAGE, 2 * PAGE, 0)
    backend.unregister_errors.clear()
    with manager.acquire([second]) as lease:
        # The retry unregisters and then offers it, and the copy survives.
        assert _offered_after_unregistering(backend, offered_pointer)
        assert lease.registered_bytes == 2 * PAGE
    assert manager.stats.offered_bytes == 2 * PAGE
    manager.clear()


def test_a_failed_offer_frees_the_unregistered_copy(backend, offering, tmp_path, caplog) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = _offered_manager(backend)
    with manager.acquire([first]):
        pass
    offered_pointer = _copy_pointer(backend)
    offering.offer_code = 1450  # ERROR_NO_SYSTEM_RESOURCES
    warnings = caplog.at_level(logging.WARNING, logger="piper_offload")
    with warnings, manager.acquire([second]) as lease:
        replacement = _copy_pointer(backend, 1)
        assert _offered_after_unregistering(backend, offered_pointer)
        assert backend.events.index(("offer", offered_pointer)) < backend.events.index(("free", offered_pointer))
        assert backend.events.index(("free", offered_pointer)) < backend.events.index(("register", replacement))
        assert lease.registered_bytes == 2 * PAGE
        assert (manager.stats.offered_bytes, manager.stats.pinned_bytes) == (0, 2 * PAGE)
    assert "Could not offer an evicted pinned copy" in caplog.text
    manager.clear()


def test_a_view_that_outlives_its_lease_frees_the_copy_instead_of_offering_it(
    backend, offering, tmp_path, caplog,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = _offered_manager(backend)
    with manager.acquire([first]):
        stray = manager._registrations[first.untyped_storage().data_ptr()].copy.view(first)
    offered_pointer = _copy_pointer(backend)
    warnings = caplog.at_level(logging.WARNING, logger="piper_offload")
    with warnings, manager.acquire([second]):
        # Offered pages must not be readable, so a copy something still views
        # is freed instead, and its memory returns when that view dies.
        assert ("offer", offered_pointer) not in backend.events
        assert ("free", offered_pointer) not in backend.events
        assert manager.stats.offered_bytes == 0
    assert "still referenced" in caplog.text
    del stray
    gc.collect()
    assert ("free", offered_pointer) in backend.events
    manager.clear()


def test_an_unreclaimable_copy_is_freed_and_rebuilt(backend, offering, reads, tmp_path, caplog) -> None:
    manager, first, second = _offer_first(backend, tmp_path)
    offered_pointer = _copy_pointer(backend)
    offering.reclaim_code = 1453  # not ERROR_BUSY: the pages' state is unknown
    reads.clear()
    backend.events.clear()
    warnings = caplog.at_level(logging.WARNING, logger="piper_offload")
    with warnings, manager.acquire([first]) as lease:
        rebuilt = backend.register_calls[-1][0]
        assert backend.events[2:] == [
            ("reclaim", offered_pointer), ("free", offered_pointer), ("allocate", rebuilt), ("register", rebuilt),
        ]
        assert sum(length for _offset, length in reads) == 2 * PAGE
        assert lease.registered_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, first), first)
        assert (manager.stats.pinned_bytes, manager.stats.copy_bytes) == (2 * PAGE, 2 * PAGE)
    assert "Could not reclaim an offered pinned copy" in caplog.text
    manager.clear()


def test_a_rebuild_that_cannot_allocate_leaves_the_storage_pageable(backend, offering, tmp_path, caplog) -> None:
    manager, first, second = _offer_first(backend, tmp_path)
    offered_pointer = _copy_pointer(backend)
    offering.reclaim_code = 1453
    offering.allocation_failures = 1
    warnings = caplog.at_level(logging.WARNING, logger="piper_offload")
    with warnings, manager.acquire([first]) as lease:
        assert ("free", offered_pointer) in backend.events
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        # Nothing of this storage is charged; the other copy, evicted to make
        # room for the reclaim, is what the tier now holds.
        assert (manager.stats.pinned_bytes, manager.stats.offered_bytes) == (0, 2 * PAGE)
    assert "could not allocate its replacement" in caplog.text
    # The storage pins again from the file once allocation works.
    with manager.acquire([first]) as lease:
        assert lease.registered_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, first), first)
    manager.clear()


def test_a_refused_reclaimed_copy_is_freed_and_leaves_the_storage_pageable(backend, offering, tmp_path) -> None:
    manager, first, second = _offer_first(backend, tmp_path)
    offered_pointer = _copy_pointer(backend)
    backend.refuse.add(offered_pointer)
    with manager.acquire([first]) as lease:
        # Reclaimed, refused by the runtime, and freed: its reservation goes
        # back, and only the copy evicted for it stays offered.
        assert backend.events[-2:] == [("register", offered_pointer), ("free", offered_pointer)]
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert manager.stats.registration_failures == 1
        assert (manager.stats.pinned_bytes, manager.stats.offered_bytes) == (0, 2 * PAGE)
        torch.testing.assert_close(_transferred(manager, first), first)
    manager.clear()


def test_a_registration_error_rolls_back_a_reclaimed_copy(backend, offering, tmp_path) -> None:
    manager, first, second = _offer_first(backend, tmp_path)
    offered_pointer = _copy_pointer(backend)
    backend.register_errors.add(offered_pointer)
    with pytest.raises(HostRegistrationError):
        manager.acquire([first])
    assert ("free", offered_pointer) in backend.events
    stats = manager.stats
    # The rolled-back copy is gone; the one evicted for it is still offered.
    assert (stats.pinned_bytes, stats.offered_bytes, stats.active_leases) == (0, 2 * PAGE, 0)
    backend.register_errors.clear()
    with manager.acquire([first]) as lease:
        assert lease.registered_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, first), first)
    manager.clear()


def test_the_offered_budget_frees_the_least_recently_offered_copy(backend, offering, tmp_path) -> None:
    tensors = [_checkpoint(tmp_path, 2 * PAGE, name) for name in ("first", "second", "third")]
    manager = _offered_manager(backend, offered=2 * PAGE)
    for tensor in tensors[:2]:
        with manager.acquire([tensor]):
            pass
    first, second = _copy_pointer(backend), _copy_pointer(backend, 1)
    assert manager.stats.offered_bytes == 2 * PAGE
    backend.events.clear()
    with manager.acquire([tensors[2]]) as lease:
        # The tier holds one copy, so admitting the second's frees the first's.
        assert [event for event in backend.events if event[0] != "allocate"][:3] == [
            ("unregister", second), ("offer", second), ("free", first),
        ]
        assert (manager.stats.offered_bytes, lease.registered_bytes) == (2 * PAGE, 2 * PAGE)
    backend.events.clear()
    with manager.acquire([tensors[1]]) as lease:
        # The copy still in the tier is reclaimed; nothing is allocated for it.
        assert ("reclaim", second) in backend.events
        assert [name for name, _pointer in backend.events].count("allocate") == 0
        assert lease.registered_bytes == 2 * PAGE
    backend.events.clear()
    with manager.acquire([tensors[0]]) as lease:
        # The copy the tier gave up is built from the file again.
        assert [name for name, _pointer in backend.events].count("allocate") == 1
        assert [name for name, _pointer in backend.events].count("reclaim") == 0
        assert lease.registered_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, tensors[0]), tensors[0])
    manager.clear()


@pytest.mark.parametrize("offered", [0, PAGE], ids=["disabled", "too-small"])
def test_a_copy_the_offered_budget_cannot_hold_is_freed(offered, backend, offering, tmp_path) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = _offered_manager(backend, offered=offered)
    with manager.acquire([first]):
        pass
    offered_pointer = _copy_pointer(backend)
    with manager.acquire([second]):
        assert backend.events[2:4] == [("unregister", offered_pointer), ("free", offered_pointer)]
        assert manager.stats.offered_bytes == 0
    manager.clear()


def test_lowering_the_offered_budget_frees_offered_copies_now(backend, offering, tmp_path) -> None:
    tensors = [_checkpoint(tmp_path, 2 * PAGE, name) for name in ("first", "second", "third")]
    manager = _offered_manager(backend, budget=2 * PAGE, offered=4 * PAGE)
    for tensor in tensors:
        with manager.acquire([tensor]):
            pass
    assert manager.stats.offered_bytes == 4 * PAGE
    manager.max_offered_bytes = 2 * PAGE
    assert manager.max_offered_bytes == manager.stats.offered_bytes == 2 * PAGE
    assert ("free", _copy_pointer(backend)) in backend.events
    manager.max_offered_bytes = 0
    assert manager.stats.offered_bytes == 0
    assert ("free", _copy_pointer(backend, 1)) in backend.events
    with pytest.raises(ValueError, match=">= 0"):
        manager.max_offered_bytes = -1
    with pytest.raises(ValueError, match=">= 0"):
        PinManager(PAGE, max_offered_bytes=-1)
    manager.clear()


def test_clear_frees_every_offered_copy(backend, offering, tmp_path) -> None:
    manager, first, second = _offer_first(backend, tmp_path)
    assert manager.stats.offered_bytes == 2 * PAGE
    manager.clear()
    assert ("free", _copy_pointer(backend)) in backend.events
    assert (manager.stats.offered_bytes, manager.stats.pinned_bytes) == (0, 0)
    assert not offering.regions


def test_dropping_an_owner_frees_its_offered_copy(backend, offering, tmp_path) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    storage_ref = weakref.ref(first.untyped_storage())
    manager = _offered_manager(backend)
    for tensor in (first, second):
        with manager.acquire([tensor]):
            pass
    offered_pointer = _copy_pointer(backend)
    assert manager.stats.offered_bytes == 2 * PAGE
    del first
    gc.collect()
    # The offered copy held the storage it came from; losing the owner frees both.
    assert ("free", offered_pointer) in backend.events
    assert manager.stats.offered_bytes == 0
    assert storage_ref() is None
    manager.clear()


def test_a_failed_allocation_frees_offered_copies_and_retries(backend, offering, tmp_path) -> None:
    tensors = [_checkpoint(tmp_path, 2 * PAGE, name) for name in ("first", "second", "third")]
    manager = _offered_manager(backend, offered=4 * PAGE)
    for tensor in tensors[:2]:
        with manager.acquire([tensor]):
            pass
    offered_pointer = _copy_pointer(backend)
    offering.allocation_failures = 1
    with manager.acquire([tensors[2]]) as lease:
        # Offered pages stay committed, so the allocation that failed frees
        # them, least recently offered first, and succeeds on the retry.
        failure = backend.events.index(("allocation failure", 2 * PAGE))
        assert backend.events[failure + 1] == ("free", offered_pointer)
        assert backend.events[failure + 2] == ("allocate", _copy_pointer(backend, 2))
        assert lease.registered_bytes == 2 * PAGE
        assert manager.stats.offered_bytes == 2 * PAGE
    manager.clear()


def test_a_pageable_lease_leaves_an_offered_copy_offered(backend, offering, tmp_path) -> None:
    manager, first, second = _offer_first(backend, tmp_path)
    with manager.acquire([first], pin=False) as lease:
        # Leasing never decides pinning, so it neither reclaims nor reads the copy.
        assert ("reclaim", _copy_pointer(backend)) not in backend.events
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        torch.testing.assert_close(_transferred(manager, first), first)
    assert manager.stats.offered_bytes == 2 * PAGE
    manager.clear()


def test_an_offered_copy_stays_offered_when_the_pin_budget_is_full(backend, offering, tmp_path) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = _offered_manager(backend)
    with manager.acquire([first]):
        pass
    offered_pointer = _copy_pointer(backend)
    holder = manager.acquire([second])
    with manager.acquire([first]) as lease:
        # An active lease holds the whole pin budget, so the copy cannot be
        # taken back yet; it waits in the tier instead of being freed.
        assert ("reclaim", offered_pointer) not in backend.events
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert manager.stats.offered_bytes == 2 * PAGE
    holder.close()
    with manager.acquire([first]) as lease:
        assert ("reclaim", offered_pointer) in backend.events
        assert lease.registered_bytes == 2 * PAGE
    manager.clear()


def test_copies_are_freed_when_the_platform_cannot_offer_them(backend, tmp_path, monkeypatch) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    # Anonymous mappings, as on Linux: the budget is set, but nothing is offered.
    monkeypatch.setattr(linux_memory.Memory, "readers", pin_module.host_memory.Memory.readers)
    monkeypatch.setattr(pin_module.host_memory, "Memory", linux_memory.Memory)
    manager = _offered_manager(backend)
    with manager.acquire([first]):
        pass
    region = manager._registrations[first.untyped_storage().data_ptr()].copy.region
    with manager.acquire([second]) as lease:
        assert lease.registered_bytes == 2 * PAGE
        assert region.closed
        assert manager.stats.offered_bytes == 0
    manager.clear()


def test_a_region_counts_its_buffers_and_refuses_to_close_or_offer_under_one(offering) -> None:
    region = VirtualRegion(2 * PAGE)
    tensor = torch.frombuffer(memoryview(region), dtype=torch.uint8)
    for refuse in (region.close, region.offer):
        with pytest.raises(BufferError, match="exported"):
            refuse()
    assert not region.closed and not region.offered
    del tensor
    region.offer()
    assert region.offered
    with pytest.raises(ValueError, match="offered or closed"):
        memoryview(region)
    assert region.reclaim() == []  # nothing was discarded
    with memoryview(region) as pages:
        assert len(pages) == 2 * PAGE
    region.close()
    assert region.closed
    region.close()  # idempotent
    with pytest.raises(ValueError, match="offered or closed"):
        memoryview(region)


def test_a_collected_region_releases_its_pages(offering) -> None:
    region = VirtualRegion(PAGE)
    pointer = region.address
    del region
    gc.collect()
    assert ("free", pointer) in offering.events


def test_an_unreclaimable_region_stays_unreadable_until_it_is_closed(offering) -> None:
    region = VirtualRegion(PAGE)
    region.offer()
    offering.reclaim_code = 1453
    with pytest.raises(OSError, match="ReclaimVirtualMemory failed with Windows error 1453"):
        region.reclaim()
    assert region.offered
    with pytest.raises(ValueError, match="offered or closed"):
        memoryview(region)
    region.close()
    assert region.closed


def test_a_region_that_cannot_be_offered_is_left_unreadable(offering) -> None:
    region = VirtualRegion(PAGE)
    offering.offer_code = 1450
    with pytest.raises(OSError, match="OfferVirtualMemory failed with Windows error 1450"):
        region.offer()
    with pytest.raises(ValueError, match="offered or closed"):
        memoryview(region)
    region.close()


def test_a_region_that_cannot_be_allocated_raises(offering) -> None:
    offering.allocation_failures = 1
    with pytest.raises(OSError, match="VirtualAlloc of 4096 bytes failed"):
        VirtualRegion(4096)


@WINDOWS
def test_real_windows_region_offers_and_reclaims_its_pages() -> None:
    region = VirtualRegion(8 * PAGE)
    payload = (bytes(range(256)) * (8 * PAGE // 256))
    with memoryview(region) as pages:
        pages[:] = payload
    region.offer()
    with pytest.raises(ValueError, match="offered or closed"):
        memoryview(region)
    # Nothing needed the memory in between, so Windows should have kept it; a discard is still legal.
    intact = region.reclaim()
    with memoryview(region) as pages:
        assert bytes(pages) == payload if intact else len(pages) == 8 * PAGE
    region.close()
    assert region.closed


@CUDA
@WINDOWS
def test_real_offered_copy_is_reclaimed_registered_and_transferred(tmp_path) -> None:
    first = _checkpoint(tmp_path, 16 * PAGE, "first")
    second = _checkpoint(tmp_path, 16 * PAGE, "second")
    manager = PinManager(16 * PAGE, max_offered_bytes=32 * PAGE)
    try:
        with manager.acquire([first]):
            pass
        with manager.acquire([second]):
            # Real OfferVirtualMemory, after the real runtime unregistered the copy.
            assert manager.stats.offered_bytes == 16 * PAGE
        with manager.acquire([first]) as lease:
            assert lease.registered_bytes == first.nbytes
            assert manager.stats.offered_bytes == 16 * PAGE  # the second copy took its place
            target = torch.empty(first.numel(), dtype=torch.uint8, device="cuda")
            manager.transfer(target, first, non_blocking=True)
            torch.cuda.synchronize()
            torch.testing.assert_close(target.cpu(), first)
    finally:
        manager.clear()
    assert (manager.stats.offered_bytes, manager.stats.pinned_bytes) == (0, 0)


def test_a_mapped_payload_of_several_dtypes_survives_an_offer_and_reclaim(
    backend, offering, reads, tmp_path,
) -> None:
    """A packed payload, its scales, and a weight share one file, as a quantized parameter's storages do."""
    safetensors = pytest.importorskip("safetensors.torch")
    payload = {
        "packed": torch.randint(0, 255, (4096,), dtype=torch.uint8),
        "scales": torch.rand(256, dtype=torch.float32),
        "weight": torch.randn(64, 64, dtype=torch.bfloat16),
    }
    path = tmp_path / "quantized.safetensors"
    safetensors.save_file(payload, str(path))
    reader = MappedCheckpoint(path)
    names = reader.keys()
    tensors = [reader.get_tensor(name) for name in names]
    other = _checkpoint(tmp_path, 8 * PAGE, "other")
    # The offered budget holds these copies and the one whose admission evicts them.
    manager = PinManager(8 * PAGE, max_offered_bytes=16 * PAGE, backend=backend)
    with manager.acquire(tensors):
        pass
    with manager.acquire([other]):
        pass
    # Every copy was offered whole pages, whatever its tensor's dtype and size.
    assert manager.stats.offered_bytes == 4 * PAGE
    reads.clear()
    with manager.acquire(tensors) as lease:
        assert (lease.registered_bytes, reads) == (sum(t.nbytes for t in tensors), [])
        for tensor in tensors:
            torch.testing.assert_close(_transferred(manager, tensor), tensor)
        # A second reader call over the same bytes resolves to the same copy,
        # and a reinterpreting view of the payload delivers those bytes too.
        packed = reader.get_tensor("packed").view(torch.int32)
        torch.testing.assert_close(_transferred(manager, packed), packed)
    manager.clear()


def test_a_failed_refill_after_a_discard_leaves_the_storage_pageable(
    backend, offering, tmp_path, monkeypatch, caplog,
) -> None:
    manager, first, second = _offer_first(backend, tmp_path)
    offered_pointer = _copy_pointer(backend)
    offering.reclaim_code = 170

    def broken(*_args) -> None:
        raise OSError("injected read failure")

    monkeypatch.setattr(copy_memory, "_read_range", broken)
    warnings = caplog.at_level(logging.WARNING, logger="piper_offload")
    with warnings, manager.acquire([first]) as lease:
        # The discarded copy could not be rewritten, so it is freed rather than
        # registered, and the transfer falls back to the mapping.
        assert ("free", offered_pointer) in backend.events
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert (manager.stats.pinned_bytes, manager.stats.offered_bytes) == (0, 2 * PAGE)
        torch.testing.assert_close(_transferred(manager, first), first)
    assert "injected read failure" in caplog.text
    manager.clear()


def test_offered_copies_are_reclaimed_together_with_the_lock_released(backend, offering, tmp_path) -> None:
    """Every offered copy comes back together, before any fill, on a pool whose workers can take the lock.

    The acquisition takes the lock only briefly, to register each copy that
    came back, so a worker may have to wait for it but never waits forever.
    """
    manager, tensors = _offer_three(backend, tmp_path)
    reclaimed_by: list[tuple[str, bool]] = []
    original = windows_memory.WindowsCopy.reclaim

    def watch(copy: copy_memory.Copy) -> bool:
        acquired = manager._lock.acquire(timeout=5)
        if acquired:
            manager._lock.release()
        reclaimed_by.append((threading.current_thread().name, acquired))
        return original(copy)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(windows_memory.WindowsCopy, "reclaim", watch)
        with manager.acquire(tensors) as lease:
            assert lease.registered_bytes == 6 * PAGE
            for tensor in tensors:
                torch.testing.assert_close(_transferred(manager, tensor), tensor)
    assert len(reclaimed_by) == 3
    assert all(acquired for _name, acquired in reclaimed_by), reclaimed_by
    assert all(name.startswith("piper-offload-reclaim") for name, _ in reclaimed_by), reclaimed_by
    manager.clear()


def test_an_eviction_offers_each_copy_while_it_unregisters_the_rest(
    backend, offering, tmp_path, monkeypatch,
) -> None:
    """The runtime serializes unregistration, so the offers have to run beside it rather than after it."""
    tensors = [_checkpoint(tmp_path, 2 * PAGE, f"copy{index}") for index in range(3)]
    other = _checkpoint(tmp_path, 8 * PAGE, "other")
    manager = PinManager(8 * PAGE, max_offered_bytes=16 * PAGE, backend=backend)
    with manager.acquire(tensors):
        pass
    offering_started = threading.Event()
    during: list[bool] = []
    offer_copy, unregister = windows_memory._offer_copy, backend.unregister

    def offering_one(copy: copy_memory.Copy) -> str | None:
        failure = offer_copy(copy)
        offering_started.set()
        return failure

    def unregistering_one(pointer: int) -> None:
        if backend.unregister_calls:  # an earlier copy of this eviction is already being offered
            during.append(offering_started.wait(5))
        unregister(pointer)

    monkeypatch.setattr(windows_memory, "_offer_copy", offering_one)
    monkeypatch.setattr(backend, "unregister", unregistering_one)
    with manager.acquire([other]) as lease:
        assert lease.registered_bytes == 8 * PAGE
    assert during and all(during), during
    assert manager.stats.offered_bytes == 6 * PAGE
    manager.clear()


def test_an_acquisitions_evictions_are_offered_together_with_the_lock_released(backend, offering, tmp_path) -> None:
    tensors = [_checkpoint(tmp_path, 2 * PAGE, f"copy{index}") for index in range(3)]
    other = _checkpoint(tmp_path, 8 * PAGE, "other")
    manager = PinManager(8 * PAGE, max_offered_bytes=16 * PAGE, backend=backend)
    with manager.acquire(tensors):
        pass
    offered_by: list[tuple[str, bool]] = []
    original = windows_memory.WindowsCopy.offer

    def watch(copy: copy_memory.Copy) -> None:
        # A worker's garbage collection can need the lock, so waiting for one
        # while holding it would deadlock; the manager waits only after it
        # has let go, and this stands in for that finalizer.
        acquired = manager._lock.acquire(timeout=5)
        if acquired:
            manager._lock.release()
        offered_by.append((threading.current_thread().name, acquired))
        original(copy)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(windows_memory.WindowsCopy, "offer", watch)
        with manager.acquire([other]) as lease:
            assert lease.registered_bytes == 8 * PAGE
    # All three evictions made room for one acquisition, and were offered on a
    # pool whose workers could take the lock while it ran.
    assert len(offered_by) == 3
    assert all(acquired for _name, acquired in offered_by), offered_by
    assert all(name.startswith("piper-offload-offer") for name, _ in offered_by), offered_by
    assert manager.stats.offered_bytes == 6 * PAGE
    manager.clear()


def test_a_lease_that_closes_over_budget_offers_its_copy_as_it_closes(backend, offering, tmp_path) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    manager = _offered_manager(backend)
    lease = manager.acquire([first])
    manager.max_pinned_bytes = 0
    assert (manager.stats.pinned_bytes, manager.stats.offered_bytes) == (2 * PAGE, 0)
    offered_on: list[str] = []
    original = windows_memory.WindowsCopy.offer

    def watch(copy: copy_memory.Copy) -> None:
        offered_on.append(threading.current_thread().name)
        original(copy)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(windows_memory.WindowsCopy, "offer", watch)
        lease.close()
    # A lease can close from a finalizer, which is no place for a pool, so the
    # copy is offered right there.
    assert offered_on == [threading.current_thread().name]
    assert (manager.stats.pinned_bytes, manager.stats.offered_bytes) == (0, 2 * PAGE)
    manager.clear()


def test_a_copy_whose_storage_is_pinned_again_while_it_is_offered_is_freed(
    backend, offering, tmp_path, monkeypatch,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(4 * PAGE, max_offered_bytes=8 * PAGE, backend=backend)
    for tensor in (first, second):
        with manager.acquire([tensor]):
            pass
    old = _copy_pointer(backend)
    original = windows_memory._OfferBatch.take
    rebuilt: list[PinLease] = []
    racing: list[bool] = []

    def take_while_first_is_pinned_again(batch):
        if not racing:
            racing.append(True)  # before acquiring: that acquisition offers what it evicts too
            # The evicted copy is neither registered nor in the tier yet, so
            # this acquisition of its storage builds another one.
            rebuilt.append(manager.acquire([first]))
        return original(batch)

    monkeypatch.setattr(windows_memory._OfferBatch, "take", take_while_first_is_pinned_again)
    manager.max_pinned_bytes = 2 * PAGE  # evicts the least recent copy, the first
    new = manager._registrations[first.untyped_storage().data_ptr()].copy.pointer
    # The old copy was offered, then freed rather than kept beside its
    # replacement; the tier holds the second copy, evicted for the new one.
    assert new != old
    assert backend.events.index(("offer", old)) < backend.events.index(("free", old))
    assert manager.stats.offered_bytes == 2 * PAGE
    torch.testing.assert_close(_transferred(manager, first), first)
    rebuilt[0].close()
    manager.clear()


def test_a_copy_whose_owner_is_dropped_while_it_is_offered_is_freed(backend, offering, tmp_path, monkeypatch) -> None:
    owners = [_checkpoint(tmp_path, 2 * PAGE, "first")]
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = _offered_manager(backend)
    with manager.acquire(owners):
        pass
    offered_pointer = _copy_pointer(backend)
    original = windows_memory._OfferBatch.take

    def take_after_the_owner_is_dropped(batch):
        owners.clear()
        gc.collect()
        return original(batch)

    monkeypatch.setattr(windows_memory._OfferBatch, "take", take_after_the_owner_is_dropped)
    with manager.acquire([second]) as lease:
        assert lease.registered_bytes == 2 * PAGE
        assert backend.events.index(("offer", offered_pointer)) < backend.events.index(("free", offered_pointer))
        assert manager.stats.offered_bytes == 0
    manager.clear()


def test_only_the_spans_windows_discarded_are_read_again(
    backend, offering, reads, tmp_path, monkeypatch,
) -> None:
    """A copy is offered in spans, so one discarded span costs that span rather than the whole copy."""
    tensor = _checkpoint(tmp_path, 4 * PAGE)
    monkeypatch.setattr(memory_module, "_OFFER_GRAIN", PAGE)
    manager = PinManager(4 * PAGE, max_offered_bytes=4 * PAGE, backend=backend)
    with manager.acquire([tensor]):
        pass
    copy = _copy_pointer(backend)
    manager.max_pinned_bytes = 0  # offered as four spans of one page
    offering.discarded_spans = {(copy, PAGE)}  # Windows drops the second of them
    manager.max_pinned_bytes = 4 * PAGE
    reads.clear()
    backend.events.clear()
    with manager.acquire([tensor]) as lease:
        assert lease.registered_bytes == 4 * PAGE
        # The copy is whole again: the surviving spans kept their bytes and the lost one was read back.
        torch.testing.assert_close(_transferred(manager, tensor), tensor)
    base = file_slice(tensor).offset
    assert reads == [(base + PAGE, PAGE)]
    assert ("allocate", copy) not in backend.events  # the same region, not a replacement
    manager.clear()


@pytest.mark.parametrize("first_kind", ["fresh", "discarded"])
@pytest.mark.parametrize("capacity", [None, 2 * PAGE])
def test_an_intact_copy_registers_before_an_earlier_copy_is_read(
    first_kind, capacity, backend, offering, tmp_path, monkeypatch,
) -> None:
    """Waiting for the earlier fill would leave its pages for that fill to push out of the working set.

    So when the runtime has capacity for only one of them, the intact copy is the one pinned.
    """
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    kept = _checkpoint(tmp_path, 2 * PAGE, "kept")
    manager = PinManager(4 * PAGE, max_offered_bytes=4 * PAGE, backend=backend)
    with manager.acquire([first, kept] if first_kind == "discarded" else [kept]):
        pass
    kept_copy = _copy_pointer(backend, 1 if first_kind == "discarded" else 0)
    if first_kind == "discarded":
        offering.discarded_spans.add((_copy_pointer(backend), 0))
    manager.max_pinned_bytes = 0
    manager.max_pinned_bytes = 4 * PAGE
    backend.capacity = capacity
    original = copy_memory._fill_copies

    def noting(pending, **options):
        backend.events.append(("fill", len(pending)))
        return original(pending, **options)

    monkeypatch.setattr(copy_memory, "_fill_copies", noting)
    backend.events.clear()
    try:
        with manager.acquire([first, kept]) as lease:
            pinned = [kept] if capacity else [first, kept]
            assert set(manager._registrations) == {tensor.untyped_storage().data_ptr() for tensor in pinned}
            assert lease.registered_bytes == 2 * PAGE * len(pinned)
            for tensor in (first, kept):
                torch.testing.assert_close(_transferred(manager, tensor), tensor)
        assert backend.events.index(("register", kept_copy)) < backend.events.index(("fill", 1))
    finally:
        manager.clear()


def test_a_failed_earlier_fill_does_not_block_an_intact_copy(
    backend, offering, tmp_path, monkeypatch,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    kept = _checkpoint(tmp_path, 2 * PAGE, "kept")
    manager = PinManager(4 * PAGE, max_offered_bytes=4 * PAGE, backend=backend)
    with manager.acquire([kept]):
        pass
    kept_pointer = _copy_pointer(backend)
    manager.max_pinned_bytes = 0
    manager.max_pinned_bytes = 4 * PAGE
    backend.capacity = 2 * PAGE
    backend.register_calls.clear()

    def failed_read(*_args):
        raise OSError("injected read failure")

    monkeypatch.setattr(copy_memory, "_read_range", failed_read)
    try:
        with manager.acquire([first, kept]) as lease:
            assert (lease.registered_bytes, lease.pageable_bytes) == (2 * PAGE, 2 * PAGE)
            assert backend.register_calls == [(kept_pointer, 2 * PAGE)]
            assert not manager._pending
            assert manager.stats.pinned_bytes == 2 * PAGE
            for tensor in (first, kept):
                torch.testing.assert_close(_transferred(manager, tensor), tensor)
    finally:
        manager.clear()


def test_copies_that_come_back_intact_register_before_the_rest_is_read(
    backend, offering, tmp_path, monkeypatch,
) -> None:
    """Registering them first locks their pages, which a long fill would otherwise have the OS reclaim."""
    kept = _checkpoint(tmp_path, 2 * PAGE, "kept")
    fresh = _checkpoint(tmp_path, 2 * PAGE, "fresh")
    manager = PinManager(4 * PAGE, max_offered_bytes=4 * PAGE, backend=backend)
    with manager.acquire([kept]):
        pass
    offered = _copy_pointer(backend)
    manager.max_pinned_bytes = 0  # the idle copy is evicted, and so offered
    original = copy_memory._fill_copies

    def noting(pending, **options):
        backend.events.append(("fill", len(pending)))
        return original(pending, **options)

    monkeypatch.setattr(copy_memory, "_fill_copies", noting)
    manager.max_pinned_bytes = 8 * PAGE
    backend.events.clear()  # only what the reacquisition does is of interest
    with manager.acquire([kept, fresh]) as lease:
        assert lease.registered_bytes == 4 * PAGE
        for tensor in (kept, fresh):
            torch.testing.assert_close(_transferred(manager, tensor), tensor)
    fills = [event for event in backend.events if event[0] == "fill"]
    assert fills, backend.events
    # The copy Windows kept was reclaimed and registered before the other one was read at all.
    assert backend.events.index(("reclaim", offered)) < backend.events.index(("register", offered))
    assert backend.events.index(("register", offered)) < backend.events.index(fills[0])
    manager.clear()


def test_an_intact_copy_registers_while_the_rest_are_still_being_reclaimed(
    backend, offering, tmp_path, monkeypatch,
) -> None:
    """The runtime registers serially, so registration runs beside the reclaims rather than after them."""
    manager, tensors = _offer_three(backend, tmp_path)
    last = _copy_pointer(backend, 2)
    first_registered = threading.Event()
    waited: list[bool] = []
    reclaim, register = windows_memory.WindowsCopy.reclaim, backend.register

    def reclaiming(copy: copy_memory.Copy) -> list[tuple[int, int]]:
        if copy.region.address == last:
            waited.append(first_registered.wait(5))
        return reclaim(copy)

    def registering(pointer: int, size: int) -> bool:
        first_registered.set()
        return register(pointer, size)

    monkeypatch.setattr(windows_memory.WindowsCopy, "reclaim", reclaiming)
    monkeypatch.setattr(backend, "register", registering)
    with manager.acquire(tensors) as lease:
        assert lease.registered_bytes == 6 * PAGE
        for tensor in tensors:
            torch.testing.assert_close(_transferred(manager, tensor), tensor)
    assert waited == [True]
    manager.clear()


def test_capacity_running_out_on_a_reclaimed_copy_frees_the_rest_once_reclaimed(
    backend, offering, reads, tmp_path, monkeypatch,
) -> None:
    """No copy is freed while a worker is still reclaiming it, and nothing is read once capacity is gone."""
    manager, tensors = _offer_three(backend, tmp_path)
    fresh = _checkpoint(tmp_path, 2 * PAGE, "fresh")
    first = _copy_pointer(backend)
    backend.refuse.add(first)
    refused = threading.Event()
    reclaim, register = windows_memory.WindowsCopy.reclaim, backend.register

    def reclaiming(copy: copy_memory.Copy) -> list[tuple[int, int]]:
        address = copy.region.address
        if address != first:
            refused.wait(5)  # still reclaiming once the first copy has been refused
        discarded = reclaim(copy)
        backend.events.append(("reclaimed", address))
        return discarded

    def registering(pointer: int, size: int) -> bool:
        try:
            return register(pointer, size)
        finally:
            refused.set()

    monkeypatch.setattr(windows_memory.WindowsCopy, "reclaim", reclaiming)
    monkeypatch.setattr(backend, "register", registering)
    reads.clear()
    backend.register_calls.clear()
    with manager.acquire([*tensors, fresh]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 8 * PAGE)
        # One refused call, and the rest of the acquisition gives up without another or a read.
        assert backend.register_calls == [(first, 2 * PAGE)]
        assert reads == []
        reclaimed = [address for name, address in backend.events if name == "reclaimed"]
        assert len(reclaimed) == 3
        for pointer in reclaimed:
            assert backend.events.index(("reclaimed", pointer)) < backend.events.index(("free", pointer))
        assert manager.stats.pinned_bytes == 0
    manager.clear()


def test_every_copy_is_touched_after_it_comes_back_and_before_it_registers(
    backend, offering, tmp_path, monkeypatch,
) -> None:
    """Registration faults pages that left the working set one at a time, so a copy is touched first."""
    tensors = [_checkpoint(tmp_path, 2 * PAGE, f"model{index}") for index in range(2)]
    original = copy_memory.Copy.touch

    def recording(self) -> None:
        backend.events.append(("touch", self.pointer))
        original(self)

    monkeypatch.setattr(copy_memory.Copy, "touch", recording)
    manager = PinManager(8 * PAGE, max_offered_bytes=8 * PAGE, backend=backend)
    with manager.acquire(tensors) as lease:
        assert lease.registered_bytes == 4 * PAGE
    copies = [pointer for name, pointer in backend.events if name == "register"]
    assert len(copies) == 2
    for pointer in copies:
        assert backend.events.index(("touch", pointer)) < backend.events.index(("register", pointer))
    # The same holds for a copy that comes back from the tier, which is touched after its reclaim.
    manager.max_pinned_bytes = 0
    manager.max_pinned_bytes = 8 * PAGE
    backend.events.clear()
    with manager.acquire(tensors) as lease:
        for tensor in tensors:
            torch.testing.assert_close(_transferred(manager, tensor), tensor)
    for pointer in copies:
        reclaimed = backend.events.index(("reclaim", pointer))
        touched = backend.events.index(("touch", pointer))
        assert reclaimed < touched < backend.events.index(("register", pointer))
    manager.clear()


@pytest.mark.parametrize("offered", [0, 8 * PAGE], ids=["tier-off", "tier-on"])
def test_tier_fills_fault_each_slice_in_then_read_it_below_offers(
    offered, backend, offering, tmp_path, monkeypatch,
) -> None:
    """The file enters the cache below offered copies, and the copy's own pages fault in before the priority drops."""
    tensor = _checkpoint(tmp_path, 3 * PAGE)
    monkeypatch.setattr(copy_memory, "_FILL_SLICE", PAGE)
    steps: list[tuple[str, int, int]] = []
    fault_in, read_range = copy_memory._fault_in, copy_memory._read_range

    def faulting(view, start, stop) -> None:
        steps.append(("fault", start, offering.priority.get()))
        fault_in(view, start, stop)

    def reading(read_at, view, offset, start, stop) -> None:
        steps.append(("read", start, offering.priority.get()))
        read_range(read_at, view, offset, start, stop)

    monkeypatch.setattr(copy_memory, "_fault_in", faulting)
    monkeypatch.setattr(copy_memory, "_read_range", reading)
    manager = PinManager(8 * PAGE, max_offered_bytes=offered, backend=backend)
    with manager.acquire([tensor]) as lease:
        assert lease.registered_bytes == 3 * PAGE
        torch.testing.assert_close(_transferred(manager, tensor), tensor)
    if offered:
        # Before anything is offered too: otherwise this file would sit above the copies offered later.
        for start in (0, PAGE, 2 * PAGE):
            fault, read = steps.index(("fault", start, 5)), steps.index(("read", start, 1))
            assert fault < read
        # Every worker went back to the priority it had once its read was done.
        last: dict[str, int] = {}
        for thread, priority in offering.priority.changes:
            last[thread] = priority
        assert last and set(last.values()) == {FakeMemoryPriority.NORMAL}
    else:
        # With the tier off, fills read through the cache as any read does, which is what makes a refill warm.
        reads = [step for step in steps if step[0] == "read"]
        assert sorted(reads) == [("read", 0, 5), ("read", PAGE, 5), ("read", 2 * PAGE, 5)]
        # Nothing faults the copy in ahead of its reads; the one fault is the touch before registration.
        assert all(steps.index(read) < steps.index(("fault", 0, 5)) for read in reads)
        assert offering.priority.changes == []
    manager.clear()


def test_copies_the_platform_cannot_offer_fill_at_the_normal_priority(backend, tmp_path, monkeypatch) -> None:
    """Off Windows the budget has no effect, so the tier being on changes nothing about how copies fill."""
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    monkeypatch.setattr(linux_memory.Memory, "readers", pin_module.host_memory.Memory.readers)
    monkeypatch.setattr(pin_module.host_memory, "Memory", linux_memory.Memory)

    def no_priorities():
        raise AssertionError("a copy that cannot be offered changed its fill's memory priority")

    monkeypatch.setattr(memory_module, "_thread_memory_priority", no_priorities)
    manager = PinManager(8 * PAGE, max_offered_bytes=8 * PAGE, backend=backend)
    with manager.acquire([tensor]) as lease:
        assert lease.registered_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, tensor), tensor)
    manager.clear()


@WINDOWS
def test_real_tier_fill_reads_at_the_lowest_memory_priority(backend, tmp_path, monkeypatch) -> None:
    tensor = _checkpoint(tmp_path, 3 * PAGE + 100)
    priorities: list[int] = []
    read_range = copy_memory._read_range

    def reading(read_at, view, offset, start, stop) -> None:
        priorities.append(memory_module._thread_memory_priority().get())
        read_range(read_at, view, offset, start, stop)

    monkeypatch.setattr(copy_memory, "_read_range", reading)
    before = memory_module._thread_memory_priority().get()
    manager = PinManager(8 * PAGE, max_offered_bytes=8 * PAGE, backend=backend)
    with manager.acquire([tensor]) as lease:
        assert lease.registered_bytes == tensor.nbytes
        torch.testing.assert_close(_transferred(manager, tensor), tensor)
    assert priorities and set(priorities) == {memory_module._MEMORY_PRIORITY_VERY_LOW}
    assert memory_module._thread_memory_priority().get() == before
    manager.clear()



def test_exhausted_commitment_frees_every_offered_copy(backend, offering, tmp_path, caplog) -> None:
    """Offered copies keep their commitment after Windows takes their RAM, which is what others run short of."""
    manager, _tensors = _offer_three(backend, tmp_path)
    offered = [_copy_pointer(backend, index) for index in range(3)]
    assert manager._memory._keep_commit()
    assert manager.stats.offered_bytes == 6 * PAGE
    offering.commit_exhausted.set()
    with caplog.at_level(logging.WARNING, logger="piper_offload"):
        # Windows can commit no more: the tier gives all of it back, and there is nothing left to watch.
        assert not manager._memory._keep_commit()
    assert all(("free", pointer) in backend.events for pointer in offered)
    assert manager.stats.offered_bytes == 0
    assert not manager._memory._watching_commit
    assert "cannot commit more memory" in caplog.text
    manager.clear()


def test_an_evicted_copy_is_freed_instead_of_offered_while_commitment_is_exhausted(
    backend, offering, tmp_path,
) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = _offered_manager(backend)
    with manager.acquire([first]):
        pass
    offered_pointer = _copy_pointer(backend)
    offering.commit_exhausted.set()
    with manager.acquire([second]) as lease:
        assert lease.registered_bytes == 2 * PAGE
        # Offered as its eviction ran, then freed rather than kept.
        assert backend.events.index(("offer", offered_pointer)) < backend.events.index(("free", offered_pointer))
        assert manager.stats.offered_bytes == 0
        torch.testing.assert_close(_transferred(manager, second), second)
    manager.clear()


def test_a_new_copy_frees_the_tier_first_while_commitment_is_exhausted(
    backend, offering, tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(windows_memory, "_watch_commit", lambda _manager: None)  # only the allocation checks here
    manager, _tensors = _offer_three(backend, tmp_path)
    fresh = _checkpoint(tmp_path, 2 * PAGE, "fresh")
    offered = [_copy_pointer(backend, index) for index in range(3)]
    offering.commit_exhausted.set()
    with manager.acquire([fresh]) as lease:
        assert lease.registered_bytes == 2 * PAGE
        torch.testing.assert_close(_transferred(manager, fresh), fresh)
    # The fresh copy may reuse an address a freed one gave back, so compare with its own, last allocation.
    allocated = backend.register_calls[-1][0]
    last_allocation = max(index for index, event in enumerate(backend.events) if event == ("allocate", allocated))
    assert all(backend.events.index(("free", pointer)) < last_allocation for pointer in offered)
    manager.clear()


def test_a_watcher_frees_the_tier_as_soon_as_commitment_is_exhausted(backend, offering, tmp_path) -> None:
    manager, _tensors = _offer_three(backend, tmp_path)
    assert manager._memory._watching_commit
    # Another program takes the last of the commitment while the manager does nothing.
    offering.commit_exhausted.set()
    deadline = time.monotonic() + 5
    while (manager.stats.offered_bytes or manager._memory._watching_commit) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.stats.offered_bytes == 0
    assert not manager._memory._watching_commit
    manager.clear()


def test_offered_memory_does_not_retain_its_manager(backend, offering, tmp_path, monkeypatch) -> None:
    # Leave the real watcher waiting while the manager and its component die.
    entered, stopped = threading.Event(), threading.Event()
    watch = windows_memory._watch_commit

    def watching(memory_ref):
        entered.set()
        watch(memory_ref)
        stopped.set()

    monkeypatch.setattr(windows_memory, "_watch_commit", watching)
    manager, tensors = _offer_three(backend, tmp_path)
    assert tensors and manager.stats.offered_bytes == 6 * PAGE
    assert manager.stats.pinned_bytes == 0
    assert entered.wait(5)
    manager_ref, memory_ref = weakref.ref(manager), weakref.ref(manager._memory)
    del manager
    gc.collect()
    assert manager_ref() is None
    assert stopped.wait(5)
    assert memory_ref() is None
    assert offering.regions == {}


def test_a_watcher_stops_once_the_tier_is_empty(backend, offering, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(windows_memory, "_COMMIT_WATCH_SECONDS", 0.01)
    manager, _tensors = _offer_three(backend, tmp_path)
    assert manager._memory._watching_commit
    manager.clear()
    deadline = time.monotonic() + 5
    while manager._memory._watching_commit and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not manager._memory._watching_commit


def test_an_unobservable_commit_condition_is_never_exhausted(monkeypatch, caplog) -> None:
    def refused():
        raise OSError("NtOpenEvent failed")

    monkeypatch.setattr(memory_module, "_open_maximum_commit_condition", refused)
    memory_module._maximum_commit_condition.cache_clear()
    try:
        with caplog.at_level(logging.WARNING, logger="piper_offload._host_memory_windows"):
            assert not memory_module.commit_exhausted()
            assert not memory_module.commit_exhausted()
        assert caplog.text.count("Cannot watch Windows commitment") == 1
    finally:
        memory_module._maximum_commit_condition.cache_clear()


@WINDOWS
def test_real_commit_condition_can_be_watched() -> None:
    assert memory_module._maximum_commit_condition() is not None
    started = time.monotonic()
    assert not memory_module.commit_exhausted(0.05)
    assert time.monotonic() - started >= 0.04
