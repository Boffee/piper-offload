"""Registration ownership, page accounting, and explicit host leases."""

import gc
import logging
import mmap
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import piper_offload._host_registration as registration_module
from piper_offload._host_registration import HostRegistrationError, RuntimeHostRegistration
from piper_offload import HostMemoryManager

PAGE = mmap.PAGESIZE
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")


def _tensors(*ranges: tuple[int, int]) -> list[torch.Tensor]:
    size = max(start + length for start, length in ranges)
    buffer = mmap.mmap(-1, size)
    return [torch.frombuffer(buffer, dtype=torch.uint8, offset=start, count=length) for start, length in ranges]


def _backings(manager: HostMemoryManager, *tensors: torch.Tensor):
    # Test tensors wrap anonymous mmaps for address control; opt them in.
    return [next(iter(manager.capture([tensor], pin_in_place=True).values())) for tensor in tensors]


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
    result = HostMemoryManager(4 * PAGE, backend=backend)
    yield result
    backend.unregister_errors.clear()
    result.clear()


def test_default_budget_is_unlimited_without_initializing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_runtime():
        raise AssertionError("construction and configuration must not initialize CUDA/HIP")

    monkeypatch.setattr(registration_module, "_load_runtime", unexpected_runtime)
    manager = HostMemoryManager()
    assert manager.max_pinned_bytes is None
    manager.max_pinned_bytes = PAGE
    manager.max_pinned_bytes = None
    assert manager.stats.max_pinned_bytes is None
    assert manager.stats.pinned_bytes == 0


def test_zero_budget_disables_registration_without_initializing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_runtime():
        raise AssertionError("zero budget must not initialize CUDA/HIP")

    monkeypatch.setattr(registration_module, "_load_runtime", unexpected_runtime)
    manager = HostMemoryManager(0)
    (backing,) = _backings(manager, torch.ones(8))
    with manager.acquire([backing]) as lease:
        assert not lease.pinned
        assert lease.backings == (backing,)
    assert manager.stats.pinned_bytes == 0


def _file_mapping(tmp_path) -> torch.Tensor:
    path = tmp_path / "weights.bin"
    path.write_bytes(bytes(2 * PAGE))
    return torch.from_file(str(path), shared=False, size=2 * PAGE, dtype=torch.uint8)


def test_file_mapping_is_pinned_through_a_copy_under_a_finite_budget(tmp_path, backend: FakeBackend) -> None:
    mapped = _file_mapping(tmp_path)
    manager = HostMemoryManager(2 * PAGE, backend=backend)
    (backing,) = manager.capture([mapped]).values()
    assert not backing.pin_in_place
    with manager.acquire([backing]) as lease:
        assert lease.pinned
        copy_pointer, size = backing.span
        assert copy_pointer != mapped.data_ptr() and size == 2 * PAGE
        assert backend.register_calls == [(copy_pointer, 2 * PAGE)]
        assert manager.stats.pinned_bytes == manager.stats.copy_bytes == 2 * PAGE
    manager.clear()
    assert backend.unregister_calls == [copy_pointer]
    assert manager.stats.copy_bytes == 0
    assert backing.span == (mapped.data_ptr(), 2 * PAGE)


def test_file_mapping_stays_pageable_under_an_unbounded_budget(tmp_path, backend: FakeBackend) -> None:
    mapped = _file_mapping(tmp_path)
    (idle,) = _backings(manager := HostMemoryManager(backend=backend), torch.empty(PAGE, dtype=torch.uint8))
    with manager.acquire([idle]):
        pass
    owned = torch.empty(PAGE, dtype=torch.uint8)
    (backing,) = manager.capture([mapped]).values()
    (owned_backing,) = manager.capture([owned]).values()
    # Skipping the mapping is not a refusal: nothing is evicted and the
    # anonymous weight after it in the batch is still registered.
    with manager.acquire([backing, owned_backing]) as lease:
        assert not lease.pinned
        assert not backing.pinned and backing.copy_bytes == 0
        assert owned_backing.pinned and idle.pinned
        assert backend.unregister_calls == []
    manager.clear()


def test_wrapped_memory_defaults_to_copying_and_owned_allocations_register(backend: FakeBackend) -> None:
    manager = HostMemoryManager(4 * PAGE, backend=backend)
    (wrapped,) = _tensors((0, PAGE))
    owned = torch.empty(PAGE, dtype=torch.uint8)
    wrapped_backing, owned_backing = (next(iter(manager.capture([t]).values())) for t in (wrapped, owned))
    assert not wrapped_backing.pin_in_place
    assert owned_backing.pin_in_place
    with manager.acquire([wrapped_backing, owned_backing]) as lease:
        assert lease.pinned
        assert wrapped_backing.copy_bytes == PAGE and owned_backing.copy_bytes == 0
        assert backend.register_calls == [(wrapped_backing.span[0], PAGE), (owned.data_ptr(), PAGE)]
    manager.clear()


def test_pin_in_place_override_is_fixed_at_first_capture(backend: FakeBackend) -> None:
    manager = HostMemoryManager(PAGE, backend=backend)
    (tensor,) = _tensors((0, PAGE))
    (backing,) = manager.capture([tensor], pin_in_place=True).values()
    assert manager.capture([tensor]) == {tensor.untyped_storage()._cdata: backing}
    with pytest.raises(ValueError, match="different pin_in_place"):
        manager.capture([tensor], pin_in_place=False)
    with manager.acquire([backing]) as lease:
        assert lease.pinned
    manager.clear()


def test_aliases_share_one_backing_and_count_per_backing(manager: HostMemoryManager, backend: FakeBackend) -> None:
    (tensor,) = _tensors((0, 2 * PAGE))
    view = tensor[64:128:2]
    handles = manager.capture([tensor, view, tensor], pin_in_place=True)
    (backing,) = handles.values()
    assert manager.capture([view]) == handles
    first = manager.acquire([backing, backing])
    second = manager.acquire([backing])
    assert backend.register_calls == [(tensor.data_ptr(), tensor.nbytes)]
    assert first.pinned and second.pinned
    assert first.backings == second.backings == (backing,)
    assert manager.stats.active_backings == 1
    assert manager.stats.registrations == 1

    first.close()
    first.close()
    manager.clear()
    assert backend.unregister_calls == []
    second.close()
    assert manager.stats.idle_registrations == 1
    with manager.acquire([backing]):
        assert len(backend.register_calls) == 1
    manager.clear()
    assert backend.unregister_calls == [tensor.data_ptr()]


def test_lru_evicts_only_idle_registrations(backend: FakeBackend) -> None:
    manager = HostMemoryManager(2 * PAGE, backend=backend)
    tensors = _tensors((0, PAGE), (2 * PAGE, PAGE), (4 * PAGE, PAGE))
    a, b, c = _backings(manager, *tensors)
    for backing in (a, b, a):
        with manager.acquire([backing]):
            pass
    with manager.acquire([c]):
        assert backend.unregister_calls == [tensors[1].data_ptr()]
        assert manager.stats.pinned_bytes == 2 * PAGE
    manager.clear()


def test_batch_protects_cached_inputs_before_new_admissions(backend: FakeBackend) -> None:
    manager = HostMemoryManager(PAGE, backend=backend)
    tensors = _tensors((0, PAGE), (2 * PAGE, PAGE))
    cached, new = _backings(manager, *tensors)
    with manager.acquire([cached]):
        pass
    with manager.acquire([new, cached]) as lease:
        assert not lease.pinned
        assert cached.pinned and not new.pinned
        assert backend.register_calls == [(tensors[0].data_ptr(), PAGE)]
        assert backend.unregister_calls == []
    manager.clear()


def test_oversized_request_preserves_idle_cache(backend: FakeBackend) -> None:
    manager = HostMemoryManager(PAGE, backend=backend)
    cached, oversized = _backings(manager, *_tensors((0, PAGE), (2 * PAGE, 2 * PAGE)))
    with manager.acquire([cached]):
        pass
    with manager.acquire([oversized]) as lease:
        assert not lease.pinned
        assert backend.unregister_calls == []
    manager.clear()


def test_pinned_bytes_are_page_rounded_per_allocation(backend: FakeBackend) -> None:
    manager = HostMemoryManager(4 * PAGE, backend=backend)
    tensors = _tensors((64, 128), (512, 256), (PAGE - 100, 400))
    handles = _backings(manager, *tensors)
    leases = [manager.acquire([backing]) for backing in handles]
    assert manager.stats.registrations == 3
    assert manager.stats.pinned_bytes == 4 * PAGE
    leases[2].close()
    manager.clear()
    assert manager.stats.pinned_bytes == 2 * PAGE
    for lease in leases:
        lease.close()
    manager.clear()
    assert manager.stats.pinned_bytes == 0


def test_budget_reduction_waits_for_active_leases(manager: HostMemoryManager, backend: FakeBackend) -> None:
    (backing,) = _backings(manager, *_tensors((0, 2 * PAGE)))
    lease = manager.acquire([backing])
    manager.max_pinned_bytes = 0
    assert manager.stats.pinned_bytes == 2 * PAGE
    assert backend.unregister_calls == []
    lease.close()
    assert manager.stats.pinned_bytes == 0


def test_capacity_failure_stops_later_registration_attempts(manager: HostMemoryManager, backend: FakeBackend) -> None:
    tensors = _tensors((0, PAGE), (2 * PAGE, PAGE), (4 * PAGE, PAGE))
    handles = _backings(manager, *tensors)
    backend.refuse.add(tensors[0].data_ptr())
    with manager.acquire(handles) as lease:
        assert not lease.pinned
        assert not any(backing.pinned for backing in handles)
        assert backend.register_calls == [(tensors[0].data_ptr(), PAGE)]


def test_default_budget_reclaims_idle_lru_and_retries(backend: FakeBackend) -> None:
    manager = HostMemoryManager(backend=backend)
    tensors = _tensors((0, PAGE), (2 * PAGE, PAGE), (4 * PAGE, PAGE))
    a, b, c = _backings(manager, *tensors)
    backend.capacity = 2 * PAGE
    for backing in (a, b):
        with manager.acquire([backing]):
            pass

    with manager.acquire([c]) as lease:
        assert lease.pinned
        assert backend.unregister_calls == [tensors[0].data_ptr()]
        assert set(backend.registered) == {tensors[1].data_ptr(), tensors[2].data_ptr()}
        assert manager.stats.max_pinned_bytes is None
    manager.clear()


def test_opportunistic_reclaim_protects_requested_idle_registration(backend: FakeBackend) -> None:
    manager = HostMemoryManager(None, backend=backend)
    tensors = _tensors((0, PAGE), (2 * PAGE, PAGE), (4 * PAGE, PAGE))
    requested, unrelated, new = _backings(manager, *tensors)
    backend.capacity = 2 * PAGE
    for backing in (requested, unrelated):
        with manager.acquire([backing]):
            pass

    with manager.acquire([requested, new]) as lease:
        assert lease.pinned
        assert backend.unregister_calls == [tensors[1].data_ptr()]
        assert set(backend.registered) == {tensors[0].data_ptr(), tensors[2].data_ptr()}
    manager.clear()


def test_budget_can_switch_between_finite_and_opportunistic(backend: FakeBackend) -> None:
    manager = HostMemoryManager(PAGE, backend=backend)
    a, b = _backings(manager, *_tensors((0, PAGE), (2 * PAGE, PAGE)))
    with manager.acquire([a]):
        pass

    manager.max_pinned_bytes = None
    with manager.acquire([b]):
        assert manager.stats.pinned_bytes == 2 * PAGE
    assert manager.max_pinned_bytes is None

    manager.max_pinned_bytes = PAGE
    assert manager.stats.pinned_bytes == PAGE
    manager.clear()


def test_pageable_backing_waits_for_all_active_leases_before_registration(backend: FakeBackend) -> None:
    manager = HostMemoryManager(0, backend=backend)
    (backing,) = _backings(manager, *_tensors((0, PAGE)))
    first = manager.acquire([backing])
    manager.max_pinned_bytes = PAGE
    second = manager.acquire([backing])
    first.close()
    with manager.acquire([backing]) as third:
        assert not second.pinned and not third.pinned
        assert backend.register_calls == []
    second.close()
    with manager.acquire([backing]) as fourth:
        assert fourth.pinned
        assert len(backend.register_calls) == 1
    manager.clear()


def test_unexpected_registration_error_rolls_back_new_registrations(
    manager: HostMemoryManager, backend: FakeBackend,
) -> None:
    tensors = _tensors((0, PAGE), (2 * PAGE, PAGE))
    a, b = _backings(manager, *tensors)
    backend.register_errors.add(tensors[1].data_ptr())
    with pytest.raises(HostRegistrationError):
        manager.acquire([a, b])
    assert not backend.registered
    assert manager.stats.pinned_bytes == 0
    assert manager.stats.active_backings == 0


def test_validation_finishes_before_registration(manager: HostMemoryManager, backend: FakeBackend) -> None:
    (backing,) = _backings(manager, *_tensors((0, PAGE)))
    with pytest.raises(ValueError, match="CPU"):
        manager.capture([torch.empty(8, device="meta")])
    with pytest.raises(ValueError, match="strided"):
        manager.capture([torch.empty(8).to_sparse()])
    with pytest.raises(TypeError, match="HostBacking"):
        manager.acquire([backing, object()])  # type: ignore[list-item]
    (foreign,) = _backings(HostMemoryManager(backend=backend), torch.ones(8))
    with pytest.raises(ValueError, match="not captured"):
        manager.acquire([backing, foreign])
    assert backend.register_calls == []
    assert manager.stats.active_backings == 0


def test_empty_allocations_are_never_registered(manager: HostMemoryManager, backend: FakeBackend) -> None:
    (backing,) = _backings(manager, torch.empty(0))
    with manager.acquire([backing]) as lease:
        assert not lease.pinned
    assert backend.register_calls == []


def test_disposing_the_last_owner_unregisters_before_storage_dies(
    manager: HostMemoryManager, backend: FakeBackend,
) -> None:
    (tensor,) = _tensors((0, PAGE))
    (backing,) = _backings(manager, tensor)
    pointer = tensor.data_ptr()
    storage_ref = weakref.ref(tensor.untyped_storage())
    with manager.acquire([backing]):
        pass
    del tensor
    gc.collect()
    assert storage_ref() is not None
    assert backend.unregister_calls == []
    del backing
    gc.collect()
    assert backend.unregister_calls == [pointer]
    assert storage_ref() is None
    assert manager.stats.pinned_bytes == 0
    assert manager.stats.backings == 0


def test_pending_copy_keeps_backing_pageable(backend: FakeBackend) -> None:
    manager = HostMemoryManager(PAGE, backend=backend)
    (tensor,) = _tensors((0, PAGE))
    (backing,) = _backings(manager, tensor)
    completion = SimpleNamespace(done=False, query=lambda: completion.done, synchronize=lambda: None)
    stream = SimpleNamespace(record_event=lambda: completion, synchronize=lambda: None)
    with backing._read(tensor, stream):
        pass
    with manager.acquire([backing]) as lease:
        assert not lease.pinned
        assert backend.register_calls == []
    completion.done = True
    with manager.acquire([backing]) as lease:
        assert lease.pinned
    manager.clear()


def test_lease_retains_backing_until_close(manager: HostMemoryManager, backend: FakeBackend) -> None:
    (tensor,) = _tensors((0, PAGE))
    (backing,) = _backings(manager, tensor)
    backing_ref = weakref.ref(backing)
    with manager.acquire([backing]):
        pass
    lease = manager.acquire([backing])
    del backing, tensor
    gc.collect()
    assert backing_ref() is not None
    assert backend.unregister_calls == []
    assert manager.stats.active_backings == 1
    lease.close()
    gc.collect()
    assert backing_ref() is None
    assert len(backend.unregister_calls) == 1
    assert manager.stats.pinned_bytes == 0


def test_failed_unregistration_of_idle_backing_is_retried_by_clear(
    manager: HostMemoryManager, backend: FakeBackend,
) -> None:
    (tensor,) = _tensors((0, PAGE))
    (backing,) = _backings(manager, tensor)
    pointer = tensor.data_ptr()
    with manager.acquire([backing]):
        pass
    backend.unregister_errors.add(pointer)
    with pytest.raises(RuntimeError, match="remains retained"):
        manager.clear()
    assert backing.pinned
    assert manager.stats.pinned_bytes == PAGE
    backend.unregister_errors.clear()
    manager.clear()
    assert not backing.pinned
    assert manager.stats.pinned_bytes == 0


def test_failed_unregistration_at_disposal_is_logged_and_storage_freed(
    manager: HostMemoryManager, backend: FakeBackend, caplog: pytest.LogCaptureFixture,
) -> None:
    (tensor,) = _tensors((0, PAGE))
    (backing,) = _backings(manager, tensor)
    pointer = tensor.data_ptr()
    storage_ref = weakref.ref(tensor.untyped_storage())
    with manager.acquire([backing]):
        pass
    backend.unregister_errors.add(pointer)
    with caplog.at_level(logging.WARNING, logger="piper_offload._host_backing"):
        del tensor, backing
        gc.collect()
    assert backend.unregister_calls == [pointer]
    assert storage_ref() is None
    assert manager.stats.backings == 0
    assert "unregistration failed during cleanup" in caplog.text
    backend.unregister_errors.clear()
    backend.unregister(pointer)


def test_failed_eviction_does_not_oversubscribe_budget(backend: FakeBackend) -> None:
    manager = HostMemoryManager(PAGE, backend=backend)
    tensors = _tensors((0, PAGE), (2 * PAGE, PAGE))
    a, b = _backings(manager, *tensors)
    with manager.acquire([a]):
        pass
    backend.unregister_errors.add(tensors[0].data_ptr())
    with manager.acquire([b]) as lease:
        assert not lease.pinned
        assert manager.stats.pinned_bytes == PAGE
        assert set(backend.registered) == {tensors[0].data_ptr()}
    backend.unregister_errors.clear()
    manager.clear()


def test_abandoned_lease_releases_registration_protection(manager: HostMemoryManager) -> None:
    (backing,) = _backings(manager, *_tensors((0, PAGE)))
    lease = manager.acquire([backing])
    del lease
    gc.collect()
    assert manager.stats.active_backings == 0
    assert manager.stats.idle_registrations == 1


def test_registration_outlives_its_manager(backend: FakeBackend) -> None:
    manager = HostMemoryManager(PAGE, backend=backend)
    (tensor,) = _tensors((0, PAGE))
    (backing,) = _backings(manager, tensor)
    manager_ref = weakref.ref(manager)
    with manager.acquire([backing]):
        pass
    del manager
    gc.collect()
    assert manager_ref() is None
    assert backing.pinned
    assert backend.registered
    del backing
    gc.collect()
    assert not backend.registered


def test_negative_budget_is_rejected(manager: HostMemoryManager) -> None:
    with pytest.raises(ValueError, match=">= 0"):
        HostMemoryManager(-1)
    with pytest.raises(ValueError, match=">= 0"):
        manager.max_pinned_bytes = -1


def test_concurrent_leases_share_one_registration(manager: HostMemoryManager, backend: FakeBackend) -> None:
    (backing,) = _backings(manager, *_tensors((0, PAGE)))
    barrier = threading.Barrier(5)

    def use_storage() -> None:
        with manager.acquire([backing]):
            barrier.wait(timeout=10)
            barrier.wait(timeout=10)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(use_storage) for _ in range(4)]
        barrier.wait(timeout=10)
        assert manager.stats.active_backings == 1
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

    def register(self, pointer: int, size: int, flags: int) -> int:
        self.flags = flags
        self.last_error = self.code
        return self.code

    def unregister(self, pointer: int) -> int:
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


def test_prior_runtime_error_does_not_skip_unregistration(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = FakeRuntime(0)
    unregistered = []
    runtime.unregister = lambda pointer: unregistered.append(pointer) or 0  # type: ignore[method-assign]
    runtime.last_error = 700
    monkeypatch.setattr(registration_module, "_load_runtime", lambda: runtime)
    with caplog.at_level(logging.WARNING, logger="piper_offload._host_registration"):
        RuntimeHostRegistration().unregister(PAGE)
    assert unregistered == [PAGE]
    assert runtime.last_error == 0
    assert "700" in caplog.text


def test_source_needing_a_copy_is_charged_for_the_aligned_copy(tmp_path, backend: FakeBackend) -> None:
    # A one-page payload starting mid-page spans two source pages, but its
    # page-aligned copy fits one, so a one-page budget admits it.
    (unaligned,) = _tensors((PAGE - 100, PAGE))
    manager = HostMemoryManager(PAGE, backend=backend)
    (backing,) = manager.capture([unaligned]).values()
    assert backing.needs_copy and backing.page_bytes == PAGE
    with manager.acquire([backing]) as lease:
        assert lease.pinned
        assert backing.span[0] % PAGE == 0 and backing.page_bytes == PAGE
        assert manager.stats.pinned_bytes == PAGE
    manager.clear()


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
    manager = HostMemoryManager(4 * PAGE)
    (tensor,) = _tensors((64, 2 * PAGE))
    tensor.fill_(17)
    pointer = tensor.data_ptr()
    (backing,) = _backings(manager, tensor)
    stream = torch.cuda.Stream()
    lease = manager.acquire([backing])
    try:
        assert lease.pinned
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
    manager = HostMemoryManager(PAGE)
    pinned, pageable = _tensors((64, 128), (512, 2 * PAGE))
    pageable.fill_(23)
    handles = _backings(manager, pinned, pageable)
    with manager.acquire(handles) as lease:
        assert not lease.pinned
        assert handles[0].pinned and not handles[1].pinned
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
    manager = HostMemoryManager(PAGE)
    (backing,) = _backings(manager, tensor)
    try:
        with pytest.raises(HostRegistrationError) as error:
            manager.acquire([backing])
        assert error.value.code == 712
        manager.clear()
        assert tensor.is_pinned()
    finally:
        backend.unregister(pointer)
