"""Registration ownership, page accounting, and explicit pin leases."""

import gc
import logging
import mmap
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import piper_offload._host_registration as registration_module
import piper_offload.pin_manager as pin_module
from piper_offload._host_registration import HostRegistrationError, RuntimeHostRegistration
from piper_offload import MappedCheckpoint, PinManager, file_slice, host_pin_manager
from piper_offload.pin_manager import TransferLease

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

    def register(self, pointer: int, size: int) -> bool:
        self.register_calls.append((pointer, size))
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
        if pointer in self.unregister_errors:
            raise HostRegistrationError("unregistration", 700)
        del self.registered[pointer]


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def manager(backend: FakeBackend):
    result = PinManager(4 * PAGE, backend=backend)
    yield result
    backend.unregister_errors.clear()
    result.clear()


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
    monkeypatch.setattr(pin_module.sys, "platform", "linux")
    values = {"SC_PHYS_PAGES": 101, "SC_PAGE_SIZE": PAGE}
    monkeypatch.setattr(pin_module.os, "sysconf", values.__getitem__, raising=False)
    monkeypatch.setattr(pin_module, "_PROC_CGROUP", "/nonexistent/cgroup")
    assert pin_module._default_pin_budget() == 50 * PAGE


def _cgroups(monkeypatch, tmp_path, membership: str, files: dict[str, str]) -> None:
    """Fake the process's cgroup membership and a cgroup mount holding ``files``."""
    monkeypatch.setattr(pin_module.sys, "platform", "linux")
    values = {"SC_PHYS_PAGES": 101, "SC_PAGE_SIZE": PAGE}
    monkeypatch.setattr(pin_module.os, "sysconf", values.__getitem__, raising=False)
    (tmp_path / "cgroup").write_text(membership)
    for relative, text in files.items():
        path = tmp_path / "mount" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    monkeypatch.setattr(pin_module, "_PROC_CGROUP", str(tmp_path / "cgroup"))
    monkeypatch.setattr(pin_module, "_CGROUP_MOUNT", str(tmp_path / "mount"))


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

    monkeypatch.setattr(pin_module.sys, "platform", "win32")
    monkeypatch.setattr(
        pin_module.ctypes, "WinDLL",
        lambda *_args, **_kwargs: SimpleNamespace(GlobalMemoryStatusEx=Mock(side_effect=query)), raising=False,
    )
    assert pin_module._default_pin_budget() == (50 * PAGE if success else 0)


def test_unknown_physical_ram_disables_default_admission(monkeypatch) -> None:
    monkeypatch.setattr(pin_module.sys, "platform", "linux")
    monkeypatch.setattr(pin_module.os, "sysconf", Mock(side_effect=OSError("unavailable")), raising=False)
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


def test_batch_protects_cached_inputs_before_new_admissions(backend: FakeBackend) -> None:
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


def test_shared_boundary_pages_are_charged_once(backend: FakeBackend) -> None:
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


def test_unbounded_budget_reclaims_idle_lru_and_retries(backend: FakeBackend) -> None:
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


def test_opportunistic_reclaim_protects_requested_idle_registration(backend: FakeBackend) -> None:
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


def test_source_disposal_unregisters_before_storage_dies(manager: PinManager, backend: FakeBackend) -> None:
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


def test_failed_unregistration_retains_storage_and_charge_for_retry(manager: PinManager, backend: FakeBackend) -> None:
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


def test_lease_retains_pageable_sources_until_close(backend: FakeBackend) -> None:
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


def test_abandoned_lease_releases_registration_protection(manager: PinManager) -> None:
    (tensor,) = _tensors((0, PAGE))
    lease = manager.acquire([tensor])
    del lease
    gc.collect()
    assert manager.stats.active_leases == 0
    assert manager.stats.idle_registrations == 1


def test_registration_keeps_manager_alive_until_source_disposal(backend: FakeBackend) -> None:
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


@pytest.mark.parametrize("code", [0, 2, 801, 1, 700, 712])
def test_runtime_backend_only_falls_back_for_capacity_or_unsupported_errors(
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

def test_prior_runtime_error_is_reported_before_registering(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(0)
    runtime.last_error = 700
    monkeypatch.setattr(registration_module, "_load_runtime", lambda: runtime)
    with pytest.raises(HostRegistrationError, match="prior runtime work") as error:
        RuntimeHostRegistration().register(PAGE, PAGE)
    assert error.value.code == 700
    assert runtime.flags is None


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


def test_pageable_lease_registers_nothing_and_tracks_its_sources(backend: FakeBackend) -> None:
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


def test_transfer_lease_keeps_protection_until_synchronization_succeeds(backend: FakeBackend, monkeypatch) -> None:
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
        (copy_pointer, copy_size), anonymous_call = backend.register_calls
        # The mapping itself is never registered; its copy is page-aligned and page-rounded.
        assert copy_pointer != tensor.data_ptr() and copy_pointer % PAGE == 0 and copy_size == 2 * PAGE
        assert anonymous_call == (anonymous.data_ptr(), PAGE)
        assert manager.stats.pinned_bytes == 3 * PAGE and manager.stats.copy_bytes == 2 * PAGE
        region = manager._entries[tensor.untyped_storage().data_ptr()].copy.region
        torch.testing.assert_close(_transferred(manager, tensor), tensor)
        # Transfers read the copy, not the mapping: a byte changed in the copy
        # shows up in what a transfer delivers, through a view's geometry too.
        region[300] = (tensor[300].item() + 1) % 256
        assert _transferred(manager, tensor)[300] != tensor[300]
        part = tensor[300:1000].view(torch.int16)
        expected = torch.frombuffer(region, dtype=torch.uint8)[300:1000].view(torch.int16)
        torch.testing.assert_close(_transferred(manager, part), expected)
        torch.testing.assert_close(_transferred(manager, anonymous), anonymous)
    manager.clear()
    assert backend.unregister_calls == [copy_pointer, anonymous.data_ptr()]
    assert region.closed
    assert manager.stats.pinned_bytes == 0 and manager.stats.copy_bytes == 0


def test_copy_is_charged_at_allocation_and_evicted_by_freeing(backend: FakeBackend, tmp_path) -> None:
    first = _checkpoint(tmp_path, 2 * PAGE, "first")
    second = _checkpoint(tmp_path, 2 * PAGE, "second")
    manager = PinManager(2 * PAGE, backend=backend)
    with manager.acquire([first, second]) as lease:
        # Only one copy fits, and it is charged before the second is considered.
        assert (lease.registered_bytes, lease.pageable_bytes) == (2 * PAGE, 2 * PAGE)
        assert len(backend.register_calls) == 1
        region = manager._entries[first.untyped_storage().data_ptr()].copy.region
    with manager.acquire([second]):
        # The idle copy is evicted to make room: unregistered and freed.
        assert backend.unregister_calls == [backend.register_calls[0][0]]
        assert region.closed
        assert manager.stats.copy_bytes == 2 * PAGE
    manager.clear()


@pytest.mark.skipif(not hasattr(os, "preadv"), reason="positional reads")
def test_copy_fill_loops_over_short_reads(backend: FakeBackend, tmp_path, monkeypatch) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    original = os.preadv
    calls = 0

    def short_preadv(fd, buffers, offset):
        nonlocal calls
        calls += 1
        (buffer,) = buffers
        return original(fd, [buffer[:100]], offset)

    monkeypatch.setattr(os, "preadv", short_preadv)
    manager = PinManager(4 * PAGE, backend=backend)
    with manager.acquire([tensor]):
        torch.testing.assert_close(_transferred(manager, tensor), tensor)
    assert calls == -(-2 * PAGE // 100)
    manager.clear()


def test_seek_and_read_fallback_fills_the_copy(backend: FakeBackend, tmp_path, monkeypatch) -> None:
    monkeypatch.delattr(os, "preadv", raising=False)
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    manager = PinManager(4 * PAGE, backend=backend)
    with manager.acquire([tensor]):
        torch.testing.assert_close(_transferred(manager, tensor), tensor)
    manager.clear()


def test_failed_fill_leaves_the_storage_pageable(backend: FakeBackend, tmp_path, monkeypatch, caplog) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)

    def broken(_copy, _source):
        raise OSError("injected read failure")

    monkeypatch.setattr(pin_module, "_fill_copy", broken)
    manager = PinManager(4 * PAGE, backend=backend)
    with caplog.at_level(logging.WARNING, logger="piper_offload.pin_manager"), manager.acquire([tensor]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert backend.register_calls == [] and manager.stats.pinned_bytes == 0
    assert "injected read failure" in caplog.text
    manager.clear()


def test_copy_registration_error_rolls_back_the_charge(backend: FakeBackend, tmp_path, monkeypatch) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    manager = PinManager(4 * PAGE, backend=backend)

    def broken(_pointer: int, _size: int) -> bool:
        raise HostRegistrationError("registration", 700)

    with monkeypatch.context() as patch:
        patch.setattr(backend, "register", broken)
        with pytest.raises(HostRegistrationError):
            manager.acquire([tensor])
    assert manager.stats.pinned_bytes == 0 and manager.stats.copy_bytes == 0
    assert manager.stats.active_leases == 0
    with manager.acquire([tensor]) as lease:
        assert lease.registered_bytes == 2 * PAGE
    manager.clear()


def test_copy_allocation_failure_leaves_the_storage_pageable(
    backend: FakeBackend, tmp_path, monkeypatch, caplog,
) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)

    def exhausted(_size: int) -> None:
        raise MemoryError("injected allocation failure")

    monkeypatch.setattr(pin_module, "_Copy", exhausted)
    manager = PinManager(4 * PAGE, backend=backend)
    with caplog.at_level(logging.WARNING, logger="piper_offload.pin_manager"), manager.acquire([tensor]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert backend.register_calls == [] and manager.stats.pinned_bytes == 0
    assert "injected allocation failure" in caplog.text
    manager.clear()


def test_copy_registration_failure_frees_the_copy(backend: FakeBackend, tmp_path) -> None:
    tensor = _checkpoint(tmp_path, 2 * PAGE)
    backend.capacity = PAGE
    manager = PinManager(4 * PAGE, backend=backend)
    with manager.acquire([tensor]) as lease:
        assert (lease.registered_bytes, lease.pageable_bytes) == (0, 2 * PAGE)
        assert manager.stats.pinned_bytes == 0 and manager.stats.registration_failures == 1
    manager.clear()


def test_failed_copy_unregistration_keeps_the_copy_and_its_charge(backend: FakeBackend, tmp_path) -> None:
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
