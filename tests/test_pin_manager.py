"""Registration ownership, page accounting, and explicit pin leases."""

import gc
import logging
import mmap
import sys
import tempfile
from pathlib import Path
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
from piper_offload import PinManager, host_pin_manager

PAGE = mmap.PAGESIZE
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
LINUX = pytest.mark.skipif(sys.platform != "linux", reason="private file mappings are recognized through /proc")
NVIDIA = pytest.mark.skipif(
    torch.version.hip is not None,
    reason="HIP repopulates a discarded private mapping on its own; in-place mapping registration retires in #119",
)


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


def test_default_budget_reserves_half_of_physical_ram(monkeypatch) -> None:
    monkeypatch.setattr(pin_module.sys, "platform", "linux")
    values = {"SC_PHYS_PAGES": 101, "SC_PAGE_SIZE": PAGE}
    monkeypatch.setattr(pin_module.os, "sysconf", values.__getitem__, raising=False)
    assert pin_module._default_pin_budget() == 50 * PAGE


@pytest.mark.parametrize("success", [False, True])
def test_windows_budget_uses_physical_ram_without_cuda(monkeypatch, success) -> None:
    def query(pointer):
        status = pointer._obj
        assert status.length == 64
        status.total_physical = 101 * PAGE
        return success

    query = Mock(side_effect=query)
    monkeypatch.setattr(pin_module.sys, "platform", "win32")
    monkeypatch.setattr(
        pin_module.ctypes, "WinDLL",
        lambda *_args, **_kwargs: SimpleNamespace(GlobalMemoryStatusEx=query), raising=False,
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


@pytest.mark.parametrize("budget", [None, 64 * PAGE])
def test_capacity_failure_reclaims_idle_lru_and_retries(backend: FakeBackend, budget) -> None:
    manager = PinManager(budget, backend=backend)
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
        assert manager.stats.max_pinned_bytes == budget
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


def _file_tensor(tmp_path, size: int) -> torch.Tensor:
    """A private mapping of a file whose bytes follow a checkable pattern."""
    path = tmp_path / "weights.bin"
    path.write_bytes(bytes(range(256)) * (size // 256))
    return torch.from_file(str(path), shared=False, size=size, dtype=torch.uint8)


@pytest.fixture
def madvise(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int, int]]:
    calls: list[tuple[int, int, int]] = []

    def record(start: int, length: int, advice: int) -> int:
        calls.append((start, length, advice))
        return 0

    monkeypatch.setattr(pin_module, "_madvise", lambda: record)
    return calls


def test_interior_pages_leave_boundary_pages_to_their_neighbours() -> None:
    assert pin_module._interior_pages(PAGE, 2 * PAGE) == (PAGE, 2 * PAGE)
    assert pin_module._interior_pages(PAGE + 100, 3 * PAGE) == (2 * PAGE, 2 * PAGE)
    assert pin_module._interior_pages(PAGE, PAGE + 1) == (PAGE, PAGE)
    assert pin_module._interior_pages(PAGE + 1, PAGE - 1) == (2 * PAGE, 0)
    assert pin_module._interior_pages(PAGE + 1, 2 * PAGE) == (2 * PAGE, PAGE)


@LINUX
def test_unregistering_a_file_mapping_discards_and_warms_its_pages(backend, madvise, tmp_path) -> None:
    manager = PinManager(backend=backend)
    tensor = _file_tensor(tmp_path, 4 * PAGE)
    with manager.acquire([tensor]) as lease:
        assert lease.registered_bytes == tensor.nbytes
        assert madvise == []
    # An idle registration is retained, pages and all, for a cheap reactivation.
    assert madvise == []
    assert manager.stats.idle_registrations == 1
    assert manager.stats.pinned_bytes == 4 * PAGE
    manager.clear()
    span = (tensor.data_ptr(), 4 * PAGE)
    assert madvise == [(*span, pin_module._MADV_DONTNEED), (*span, pin_module._MADV_WILLNEED)]
    assert backend.registered == {}
    assert manager.stats.pinned_bytes == 0


@LINUX
def test_anonymous_storage_is_never_discarded(backend, madvise) -> None:
    manager = PinManager(backend=backend)
    (buffer,) = _tensors((0, 4 * PAGE))
    heap = torch.full((4 * PAGE,), 9, dtype=torch.uint8)
    with manager.acquire([buffer, heap]) as lease:
        assert lease.registered_bytes == buffer.nbytes + heap.nbytes
    assert manager.stats.idle_registrations == 2
    assert manager.stats.pinned_bytes >= buffer.nbytes + heap.nbytes
    manager.clear()
    assert madvise == []
    assert heap.eq(9).all()


@LINUX
def test_dropping_the_last_owner_discards_without_reading_the_file_back(backend, madvise, tmp_path) -> None:
    manager = PinManager(backend=backend)
    tensor = _file_tensor(tmp_path, 4 * PAGE)
    span = (tensor.data_ptr(), 4 * PAGE)
    with manager.acquire([tensor]):
        pass
    assert not madvise
    del tensor
    gc.collect()
    assert madvise == [(*span, pin_module._MADV_DONTNEED)]
    assert backend.registered == {}


@LINUX
def test_budget_eviction_discards_the_evicted_mapping(backend, madvise, tmp_path) -> None:
    manager = PinManager(4 * PAGE, backend=backend)
    first = _file_tensor(tmp_path, 4 * PAGE)
    (second,) = _tensors((0, 4 * PAGE))
    with manager.acquire([first]):
        pass
    assert manager.stats.pinned_bytes == 4 * PAGE
    with manager.acquire([second]) as lease:
        assert lease.registered_bytes == second.nbytes
    assert [advice for _start, _length, advice in madvise] == [pin_module._MADV_DONTNEED, pin_module._MADV_WILLNEED]
    manager.clear()


@LINUX
def test_shared_mapping_is_retained_across_leases_and_can_be_registered_again(backend, madvise, tmp_path) -> None:
    manager = PinManager(backend=backend)
    tensor = _file_tensor(tmp_path, 4 * PAGE)
    view = tensor[128:256]  # a live view shares the storage; a dropped owner would retire it
    first = manager.acquire([tensor])
    second = manager.acquire([view])
    first.close()
    manager.clear()
    assert not backend.unregister_calls
    assert not madvise
    assert manager.stats.pinned_bytes == 4 * PAGE
    second.close()
    assert manager.stats.pinned_bytes == 4 * PAGE
    assert manager.stats.idle_registrations == 1
    manager.clear()
    assert manager.stats.pinned_bytes == 0
    assert len(madvise) == 2
    with manager.acquire([tensor]):
        assert len(backend.register_calls) == 2
    manager.clear()
    assert len(backend.unregister_calls) == 2


@LINUX
def test_file_unregister_failure_keeps_storage_and_charge_until_retry(backend, madvise, tmp_path) -> None:
    manager = PinManager(backend=backend)
    tensor = _file_tensor(tmp_path, 4 * PAGE)
    backend.unregister_errors.add(tensor.data_ptr())
    with manager.acquire([tensor]):
        pass
    assert not madvise
    assert manager.stats.active_leases == 0
    assert manager.stats.idle_registrations == 1
    assert manager.stats.pinned_bytes == 4 * PAGE
    assert manager._entries[tensor.data_ptr()].storage.data_ptr() == tensor.data_ptr()
    backend.unregister_errors.clear()
    manager.clear()
    assert manager.stats.pinned_bytes == 0
    assert len(madvise) == 2


@LINUX
def test_adjacent_file_registrations_discard_only_interior_pages(backend, madvise, tmp_path) -> None:
    path = tmp_path / "adjacent.bin"
    path.write_bytes(bytes(4 * PAGE))
    with path.open("r+b") as file:
        mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)
    first = torch.frombuffer(mapping, dtype=torch.uint8, count=PAGE + 100)
    second = torch.frombuffer(mapping, dtype=torch.uint8, offset=PAGE + 100)
    manager = PinManager(backend=backend)
    lease = manager.acquire([second])
    with manager.acquire([first]):
        assert manager.stats.pinned_bytes == 4 * PAGE
    # Trimming while the neighbour is leased evicts only the idle registration
    # and leaves the page they share intact.
    manager.clear()
    assert madvise == [(first.data_ptr(), PAGE, pin_module._MADV_DONTNEED),
                       (first.data_ptr(), PAGE, pin_module._MADV_WILLNEED)]
    assert manager.stats.pinned_bytes == 3 * PAGE
    assert second.data_ptr() in backend.registered
    lease.close()
    assert manager.stats.pinned_bytes == 3 * PAGE
    manager.clear()
    assert madvise[2][0:2] == (first.data_ptr() + 2 * PAGE, 2 * PAGE)
    assert manager.stats.pinned_bytes == 0


@LINUX
@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="tmpfs at /dev/shm required")
def test_file_mapping_on_tmpfs_is_discarded(backend, madvise) -> None:
    manager = PinManager(backend=backend)
    with tempfile.NamedTemporaryFile(dir="/dev/shm", prefix="piper-test-") as file:
        file.write(bytes(range(256)) * (4 * PAGE // 256))
        file.flush()
        tensor = torch.from_file(file.name, shared=False, size=4 * PAGE, dtype=torch.uint8)
        with manager.acquire([tensor]):
            pass
        manager.clear()
        span = (tensor.data_ptr(), 4 * PAGE)
        assert madvise == [(*span, pin_module._MADV_DONTNEED), (*span, pin_module._MADV_WILLNEED)]


def test_trim_evicts_idle_entries_least_recently_released_first(backend: FakeBackend) -> None:
    manager = PinManager(backend=backend)
    a, b, c = _tensors((0, PAGE), (2 * PAGE, PAGE), (4 * PAGE, PAGE))
    for tensor in (a, b, c):
        with manager.acquire([tensor]):
            pass
    lease = manager.acquire([c])
    result = manager.trim(PAGE)
    assert result == pin_module.TrimResult(released_bytes=PAGE, freed_bytes=0, failures=0)
    assert backend.unregister_calls == [a.data_ptr()]
    # The rest: b is idle and goes; c is leased and stays.
    result = manager.trim()
    assert result.released_bytes == PAGE and result.failures == 0
    assert set(backend.registered) == {c.data_ptr()}
    assert manager.stats.pinned_bytes == PAGE
    lease.close()
    manager.clear()


def test_trim_reports_failures_without_raising_and_clear_raises(backend: FakeBackend) -> None:
    manager = PinManager(backend=backend)
    (a,) = _tensors((0, PAGE))
    with manager.acquire([a]):
        pass
    backend.unregister_errors.add(a.data_ptr())
    assert manager.trim() == pin_module.TrimResult(released_bytes=0, freed_bytes=0, failures=1)
    assert manager.stats.pinned_bytes == PAGE
    with pytest.raises(RuntimeError, match="Could not release 1"):
        manager.clear()
    backend.unregister_errors.clear()
    manager.clear()
    assert manager.stats.pinned_bytes == 0


@LINUX
def test_trim_counts_only_returned_memory_as_freed(backend, madvise, tmp_path) -> None:
    manager = PinManager(backend=backend)
    mapped = _file_tensor(tmp_path, 4 * PAGE)
    (anonymous,) = _tensors((0, 4 * PAGE))
    with manager.acquire([mapped, anonymous]):
        pass
    result = manager.trim()
    assert result.released_bytes == 8 * PAGE
    assert result.freed_bytes == 4 * PAGE  # the mapping's pages; the anonymous storage is still the model's
    assert result.failures == 0


def test_reservation_holds_budget_until_consumed_or_released(backend: FakeBackend) -> None:
    manager = PinManager(4 * PAGE, backend=backend)
    a, b = _tensors((0, 4 * PAGE), (8 * PAGE, 4 * PAGE))
    with manager.acquire([a]):
        pass
    reservation = manager.reserve(4 * PAGE)
    assert reservation is not None and reservation.active
    # Admission evicted the idle entry to make room, and the hold shows in the stats.
    assert backend.unregister_calls == [a.data_ptr()]
    assert manager.stats.reserved_bytes == 4 * PAGE
    with manager.acquire([b]) as lease:
        assert lease.pageable_bytes == b.nbytes  # no budget left while the reservation stands
    with manager.acquire([b], reservation=reservation) as lease:
        assert lease.registered_bytes == b.nbytes
        assert not reservation.active
        assert manager.stats.reserved_bytes == 0
    assert manager.reserve(8 * PAGE) is None  # larger than the whole budget
    with manager.reserve(4 * PAGE) as held:
        assert held is not None and manager.stats.reserved_bytes == 4 * PAGE
    assert manager.stats.reserved_bytes == 0
    manager.clear()


def test_reservation_is_consumed_even_when_registration_is_refused(backend: FakeBackend) -> None:
    manager = PinManager(4 * PAGE, backend=backend)
    (a,) = _tensors((0, 4 * PAGE))
    reservation = manager.reserve(4 * PAGE)
    assert reservation is not None
    backend.refuse.add(a.data_ptr())
    with manager.acquire([a], reservation=reservation) as lease:
        assert lease.pageable_bytes == a.nbytes
    assert manager.stats.reserved_bytes == 0
    assert not reservation.active


def test_admissions_that_consume_memory_keep_the_headroom(backend: FakeBackend, madvise, tmp_path) -> None:
    available = {"bytes": 6 * PAGE}
    manager = PinManager(headroom_bytes=3 * PAGE, backend=backend, available_memory=lambda: available["bytes"])
    (anonymous,) = _tensors((0, 4 * PAGE))
    # Registering resident anonymous storage consumes nothing and is always admitted.
    with manager.acquire([anonymous]) as lease:
        assert lease.registered_bytes == anonymous.nbytes
    # A reservation is a copy about to be filled: 4 pages would leave 2, below the headroom.
    assert manager.reserve(4 * PAGE) is None
    available["bytes"] = 7 * PAGE
    reservation = manager.reserve(4 * PAGE)
    assert reservation is not None
    reservation.release()
    if sys.platform == "linux":
        # Registering a private file mapping copies its pages, so it is gated too.
        mapped = _file_tensor(tmp_path, 4 * PAGE)
        available["bytes"] = 6 * PAGE
        with manager.acquire([mapped]) as lease:
            assert lease.pageable_bytes == mapped.nbytes
        available["bytes"] = 7 * PAGE
        with manager.acquire([mapped]) as lease:
            assert lease.registered_bytes == mapped.nbytes
    manager.clear()


@LINUX
def test_ensure_available_trims_idle_entries_that_return_memory(backend, madvise, tmp_path) -> None:
    available = {"bytes": 2 * PAGE}
    manager = PinManager(headroom_bytes=PAGE, backend=backend, available_memory=lambda: available["bytes"])
    mapped = _file_tensor(tmp_path, 4 * PAGE)
    available["bytes"] = 9 * PAGE
    with manager.acquire([mapped]):
        pass
    available["bytes"] = 2 * PAGE

    def madvise_frees(start: int, length: int, advice: int) -> int:
        if advice == pin_module._MADV_DONTNEED:
            available["bytes"] += length
        return 0

    manager._available_memory = lambda: available["bytes"]
    pin_module_madvise = pin_module._madvise
    pin_module._madvise = lambda: madvise_frees
    try:
        assert manager.ensure_available(4 * PAGE) == 0
        assert manager.stats.pinned_bytes == 0
        assert available["bytes"] == 6 * PAGE
        # Nothing idle is left, so a bigger request reports what is missing.
        assert manager.ensure_available(8 * PAGE) == 3 * PAGE
    finally:
        pin_module._madvise = pin_module_madvise


def test_ensure_available_does_nothing_when_availability_is_unknown(backend: FakeBackend) -> None:
    manager = PinManager(backend=backend, available_memory=lambda: None)
    (a,) = _tensors((0, PAGE))
    with manager.acquire([a]):
        pass
    assert manager.ensure_available(2**40) == 0
    assert manager.stats.idle_registrations == 1
    manager.clear()


def test_headroom_configuration_and_stats(backend: FakeBackend) -> None:
    manager = PinManager(headroom_bytes=PAGE, backend=backend, available_memory=lambda: None)
    assert manager.headroom_bytes == PAGE
    assert manager.stats.headroom_bytes == PAGE
    manager.headroom_bytes = 2 * PAGE
    assert manager.stats.headroom_bytes == 2 * PAGE
    with pytest.raises(ValueError, match="headroom_bytes must be >= 0"):
        manager.headroom_bytes = -1
    with pytest.raises(ValueError, match="headroom_bytes must be >= 0"):
        PinManager(headroom_bytes=-1, backend=backend)
    default = PinManager(backend=backend)
    assert default.headroom_bytes >= 2**30 and default.headroom_bytes % PAGE == 0
    assert pin_module.available_memory_bytes() is None or pin_module.available_memory_bytes() > 0


def _anonymous_bytes(pointer: int) -> int:
    """Private page bytes of the mapping containing ``pointer``, from ``/proc/self/smaps``."""
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


@CUDA
@LINUX
@NVIDIA
def test_real_unregistration_returns_a_file_mapping_to_its_file(tmp_path) -> None:
    size = 16 * 2**20
    tensor = _file_tensor(tmp_path, size)
    pointer = tensor.data_ptr()
    expected = tensor.clone()
    assert _anonymous_bytes(pointer) == 0
    manager = PinManager(2 * size)
    try:
        for _ in range(2):
            with manager.acquire([tensor]) as lease:
                assert lease.registered_bytes == size
                # The writable lock made the kernel copy every page.
                assert _anonymous_bytes(pointer) == size
                target = tensor.to("cuda", non_blocking=True)
                torch.cuda.synchronize()
            # Idle, the registration and its pages are retained until evicted.
            assert _anonymous_bytes(pointer) == size
            assert tensor.is_pinned()
            manager.clear()
            assert _anonymous_bytes(pointer) == 0
            assert not tensor.is_pinned()
            torch.testing.assert_close(target.cpu(), expected)
            torch.testing.assert_close(tensor, expected)
    finally:
        manager.clear()
