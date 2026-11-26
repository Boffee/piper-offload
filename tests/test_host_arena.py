"""Tests for Layer 1 HostSlabArena (Tier 2a/2b), PinnedBounceRing, and budget defaults."""

from __future__ import annotations

import gc
import logging
import math
import mmap
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

import piper_offload._host_memory as host_memory
import piper_offload._host_memory_linux as linux_memory
import piper_offload._host_memory_windows as windows_memory
import piper_offload._transfer_priority as transfer_priority
import piper_offload.host_arena as host_arena_mod
from piper_offload import MappedCheckpoint
from piper_offload._direct_io import DirectFileReader, DirectReadRequest, align_span
from piper_offload._host_registration import HostRegistrationError, HostRegistrationRefusedError
from piper_offload.checkpoint import file_slice
from piper_offload.host_arena import HostSlabArena, PinnedBounceRing

PAGE = mmap.PAGESIZE
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")


class _FakeBackend:
    def __init__(self) -> None:
        self.registered: dict[int, int] = {}
        self.register_calls: list[tuple[int, int]] = []
        self.unregister_calls: list[int] = []
        self.capacity: int | None = None
        self.refuse_pointers: set[int] = set()
        self.raise_refused_error = False
        self.register_errors: set[int] = set()
        self.unregister_errors: set[int] = set()
        self._lock = threading.Lock()

    def register(self, pointer: int, size: int) -> bool:
        with self._lock:
            self.register_calls.append((pointer, size))
            if self.raise_refused_error:
                raise HostRegistrationRefusedError("registration", 1)
            if pointer in self.register_errors:
                raise HostRegistrationError("registration", 700)
            if pointer in self.refuse_pointers:
                return False
            if self.capacity is not None and sum(self.registered.values()) + size > self.capacity:
                return False
            assert pointer not in self.registered
            self.registered[pointer] = size
            return True

    def unregister(self, pointer: int) -> None:
        with self._lock:
            self.unregister_calls.append(pointer)
            if pointer in self.unregister_errors:
                raise HostRegistrationError("unregistration", 700)
            del self.registered[pointer]


class TestHostMemoryBudgetsAndRegions:
    def test_platform_default_budgets_respect_fractions_without_cuda_init(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        init_cuda = Mock(side_effect=AssertionError("CUDA must not initialize"))
        monkeypatch.setattr(torch.cuda, "init", init_cuda)
        monkeypatch.setattr(host_memory, "available_memory", lambda: 100 * 1024 * PAGE)
        monkeypatch.setattr(host_memory, "DEFAULT_PIN_FRACTION", linux_memory.DEFAULT_PIN_FRACTION)
        monkeypatch.setattr(host_memory, "DEFAULT_HOST_FRACTION", linux_memory.DEFAULT_HOST_FRACTION)
        assert math.isclose(linux_memory.DEFAULT_PIN_FRACTION, 0.70, abs_tol=1e-12)
        assert math.isclose(linux_memory.DEFAULT_HOST_FRACTION, 0.80, abs_tol=1e-12)
        assert host_memory.default_pin_budget() == int(100 * 1024 * PAGE * 0.70) // PAGE * PAGE
        assert host_memory.default_host_budget() == int(100 * 1024 * PAGE * 0.80) // PAGE * PAGE

        monkeypatch.setattr(host_memory, "DEFAULT_PIN_FRACTION", windows_memory.DEFAULT_PIN_FRACTION)
        monkeypatch.setattr(host_memory, "DEFAULT_HOST_FRACTION", windows_memory.DEFAULT_HOST_FRACTION)
        assert math.isclose(windows_memory.DEFAULT_PIN_FRACTION, 0.45, abs_tol=1e-12)
        assert math.isclose(windows_memory.DEFAULT_HOST_FRACTION, 0.75, abs_tol=1e-12)
        assert host_memory.default_pin_budget() == int(100 * 1024 * PAGE * 0.45) // PAGE * PAGE
        assert host_memory.default_host_budget() == int(100 * 1024 * PAGE * 0.75) // PAGE * PAGE

        def boom() -> int:
            raise OSError("unavailable")

        monkeypatch.setattr(host_memory, "available_memory", boom)
        assert host_memory.default_pin_budget() == 0
        assert host_memory.default_host_budget() == 0

        # Exact rational rounding avoids losing a page to binary float truncation (90 * 0.70 == 63).
        monkeypatch.setattr(host_memory, "available_memory", lambda: 90 * PAGE)
        monkeypatch.setattr(host_memory, "DEFAULT_PIN_FRACTION", 0.70)
        assert host_memory.default_pin_budget() == 63 * PAGE

    def test_allocate_aligned_region_validates_size(self) -> None:
        with pytest.raises(ValueError, match="positive multiple"):
            host_memory.allocate_aligned_region(0)
        with pytest.raises(ValueError, match="positive multiple"):
            host_memory.allocate_aligned_region(PAGE + 1)
        with pytest.raises(ValueError, match="positive multiple"):
            host_memory.allocate_aligned_region(float(PAGE))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive multiple"):
            host_memory.allocate_aligned_region(True)  # type: ignore[arg-type]
        reg = host_memory.allocate_aligned_region(2 * PAGE)
        try:
            with memoryview(reg) as mv:
                assert len(mv) == 2 * PAGE
        finally:
            reg.close()


class TestHostSlabArena:
    def test_tier2a_registers_once_per_slab_and_tier2b_never_registers(self) -> None:
        backend = _FakeBackend()
        alloc_calls: list[int] = []

        def tracked_alloc(size: int):
            alloc_calls.append(size)
            return host_memory.allocate_aligned_region(size)

        with HostSlabArena(
            pinned=True,
            slab_bytes=16 * PAGE,
            backend=backend,
            allocator=tracked_alloc,
        ) as arena:
            allocs = [arena.allocate(2 * PAGE) for _ in range(6)]
            assert all(a is not None and a.is_pinned for a in allocs)
            assert len(backend.register_calls) == 1
            assert len(alloc_calls) == 1
            for alloc in allocs:
                assert alloc is not None
                alloc.release()
            # Re-allocating after release hits the same registered slab cleanly.
            again = arena.allocate(12 * PAGE)
            assert again is not None
            assert len(backend.register_calls) == 1
            assert len(alloc_calls) == 1
            again.release()
        assert len(backend.registered) == 0

        with HostSlabArena(
            pinned=False,
            slab_bytes=8 * PAGE,
            backend=backend,
        ) as unpinned:
            a = unpinned.allocate(4 * PAGE)
            assert a is not None and not a.is_pinned
            assert len(backend.register_calls) == 1  # unchanged from earlier Tier 2a test
            a.release()

    def test_best_fit_and_three_way_free_list_coalescing(self) -> None:
        with HostSlabArena(pinned=False, slab_bytes=8 * PAGE) as arena:
            a = arena.allocate(2 * PAGE)
            b = arena.allocate(2 * PAGE)
            c = arena.allocate(2 * PAGE)
            assert a is not None and b is not None and c is not None
            a.release()
            c.release()
            b.release()
            # Full 8-page span must coalesce back into one contiguous chunk.
            full = arena.allocate(8 * PAGE)
            assert full is not None
            assert arena.stats.slab_count == 1
            full.release()

            # Small gap vs big gap: best-fit picks smaller matching hole.
            s_big = arena.allocate(4 * PAGE)
            guard1 = arena.allocate(PAGE)
            s_small = arena.allocate(2 * PAGE)
            guard2 = arena.allocate(PAGE)
            assert s_big and guard1 and s_small and guard2
            s_big.release()
            s_small.release()
            picked = arena.allocate(2 * PAGE)
            assert picked is not None
            assert picked.offset == s_small.offset
            picked.release()
            guard1.release()
            guard2.release()

    def test_oversized_hint_and_power_of_two_alignments(self) -> None:
        backend = _FakeBackend()
        with HostSlabArena(pinned=True, slab_bytes=4 * PAGE, backend=backend) as arena:
            big = arena.allocate(10 * PAGE)
            assert big is not None and big.size == 10 * PAGE
            assert arena.stats.reserved_bytes == 10 * PAGE
            big.release()
            arena.clear()

            hinted = arena.allocate(PAGE, slab_hint_bytes=2 * PAGE)
            assert hinted is not None
            assert arena.stats.reserved_bytes == 2 * PAGE
            hinted.release()
            arena.clear()

            high_align = arena.allocate(128, alignment=8 * PAGE)
            assert high_align is not None
            assert high_align.aligned_pointer % (8 * PAGE) == 0
            high_align.release()

    def test_payload_views_tensor_types_and_direct_io_roundtrip(self, tmp_path: Path) -> None:
        st_path = tmp_path / "arena_roundtrip.safetensors"
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        cplx = torch.complex(torch.randn(3, 4), torch.randn(3, 4))
        bf16 = torch.randn(7, 5, dtype=torch.bfloat16)
        save_file({"base": base, "bf16": bf16}, st_path)

        with MappedCheckpoint(st_path) as ckpt:
            fs_base = file_slice(ckpt.get_tensor("base"))
            fs_bf16 = file_slice(ckpt.get_tensor("bf16"))
        assert fs_base is not None and fs_bf16 is not None

        with HostSlabArena(pinned=False, slab_bytes=8 * PAGE) as arena, DirectFileReader() as reader:
            for fs, tmpl in ((fs_base, base), (fs_bf16, bf16)):
                span = align_span(fs.offset, fs.length, sector_size=reader.sector_size(fs.file))
                alloc = arena.allocate(span.data_length, head_pad=span.head_pad)
                assert alloc is not None
                with alloc.aligned_view() as mv:
                    reader.read_into([DirectReadRequest(fs, mv, head_pad=alloc.head_pad, aligned_target=True)])
                rebuilt = alloc.tensor_view(tmpl)
                torch.testing.assert_close(rebuilt, tmpl, rtol=0, atol=0)
                del rebuilt
                alloc.release()

            # Check non-zero storage_offset, non-contiguous stride, complex conj, neg views, and as_uint8_tensor.
            src_cplx = cplx.t().conj()
            alloc2 = arena.allocate(src_cplx.untyped_storage().nbytes(), head_pad=19)
            assert alloc2 is not None
            u8 = alloc2.as_uint8_tensor()
            src_u8 = torch.empty(0, dtype=torch.uint8).set_(
                src_cplx.untyped_storage(), 0, (src_cplx.untyped_storage().nbytes(),), (1,)
            )
            u8.copy_(src_u8)
            rebuilt_cplx = alloc2.tensor_view(src_cplx)
            torch.testing.assert_close(rebuilt_cplx, src_cplx, rtol=0, atol=0)

            sliced_f = base.reshape(-1)[3:19].neg()
            alloc3 = arena.allocate(sliced_f.untyped_storage().nbytes(), head_pad=13)
            assert alloc3 is not None
            alloc3.as_uint8_tensor().copy_(
                torch.empty(0, dtype=torch.uint8).set_(
                    sliced_f.untyped_storage(), 0, (sliced_f.untyped_storage().nbytes(),), (1,)
                )
            )
            torch.testing.assert_close(alloc3.tensor_view(sliced_f), sliced_f, rtol=0, atol=0)
            del u8, rebuilt_cplx
            alloc2.release()
            alloc3.release()

    def test_max_bytes_trim_and_capacity_fallback(self) -> None:
        backend = _FakeBackend()
        with HostSlabArena(pinned=True, max_bytes=8 * PAGE, slab_bytes=4 * PAGE, backend=backend) as arena:
            a1 = arena.allocate(3 * PAGE)
            a2 = arena.allocate(3 * PAGE)
            assert a1 is not None and a2 is not None
            assert arena.stats.slab_count == 2
            assert arena.allocate(3 * PAGE) is None

            a2.release()
            arena.max_bytes = 2 * PAGE
            assert arena.stats.slab_count == 1
            a1.release()
            assert arena.stats.slab_count == 0
            assert arena.stats.reserved_bytes == 0

        backend2 = _FakeBackend()
        backend2.capacity = 6 * PAGE
        with HostSlabArena(pinned=True, slab_bytes=4 * PAGE, backend=backend2) as arena2:
            s1 = arena2.allocate(3 * PAGE)
            assert s1 is not None
            s2 = arena2.allocate(2 * PAGE)
            assert s2 is not None and s2.size == 2 * PAGE
            assert arena2.stats.registration_failures == 1
            assert arena2.allocate(4 * PAGE) is None
            s1.release()
            s2.release()

    def test_registration_refusal_unregistration_failure_and_dangling_view_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend_refused = _FakeBackend()
        backend_refused.raise_refused_error = True
        closed_regions: list[int] = []

        class _TrackedRegion:
            def __init__(self, inner: host_memory.AlignedRegion, sz: int) -> None:
                self._inner = inner
                self._sz = sz

            def __buffer__(self, flags: int, /) -> memoryview:
                return memoryview(self._inner)

            def __release_buffer__(self, view: memoryview, /) -> None:
                view.release()

            def close(self) -> None:
                closed_regions.append(self._sz)
                self._inner.close()

        def tracked_close_alloc(sz: int) -> host_memory.AlignedRegion:
            return _TrackedRegion(host_memory.allocate_aligned_region(sz), sz)

        with HostSlabArena(
            pinned=True, slab_bytes=2 * PAGE, backend=backend_refused, allocator=tracked_close_alloc
        ) as arena_ref:
            assert arena_ref.allocate(PAGE) is None
            assert closed_regions == [2 * PAGE]

        backend3 = _FakeBackend()
        arena3 = HostSlabArena(pinned=True, slab_bytes=4 * PAGE, backend=backend3)
        x = arena3.allocate(PAGE)
        assert x is not None
        ptr = next(iter(backend3.registered))
        backend3.unregister_errors.add(ptr)
        x.release()
        with pytest.raises(RuntimeError, match="Could not release 1"):
            arena3.clear()
        assert arena3.stats.slab_count == 1
        assert arena3.stats.unregistration_failures == 1
        backend3.unregister_errors.clear()
        arena3.close()
        assert arena3.stats.slab_count == 0

        arena4 = HostSlabArena(pinned=False, slab_bytes=2 * PAGE)
        y = arena4.allocate(PAGE)
        assert y is not None
        dangling_mv = y.payload_view()
        y.release()
        with caplog.at_level(logging.WARNING):
            arena4.close()
        assert arena4.stats.slab_count == 1
        assert "surviving buffer references" not in caplog.text
        dangling_mv.release()
        assert arena4.stats.slab_count == 0

    def test_finalizer_returns_slab_after_close_and_thread_stress(self) -> None:
        backend = _FakeBackend()
        arena = HostSlabArena(pinned=True, slab_bytes=8 * PAGE, backend=backend)
        alloc = arena.allocate(PAGE)
        assert alloc is not None
        arena.close()
        assert arena.stats.slab_count == 1
        del alloc
        gc.collect()
        assert arena.stats.slab_count == 0
        assert not backend.registered

        # Multi-threaded alloc/release stress test.
        with HostSlabArena(pinned=True, slab_bytes=16 * PAGE, backend=backend) as mt_arena:
            errors: list[BaseException] = []

            def worker(tid: int) -> None:
                try:
                    for idx in range(30):
                        a = mt_arena.allocate(1024 + (idx % 5) * 1024, head_pad=(tid * 17) % 128)
                        assert a is not None
                        a.as_uint8_tensor().fill_((tid + idx) % 251)
                        assert int(a.as_uint8_tensor()[0].item()) == (tid + idx) % 251
                        a.release()
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors
            assert mt_arena.stats.used_bytes == 0
            assert mt_arena.stats.live_allocations == 0
        assert not backend.registered

    def test_refusal_impossible_alignment_preserve_slabs_and_os_oom_recovers(self) -> None:
        backend = _FakeBackend()
        with HostSlabArena(pinned=True, max_bytes=12 * PAGE, slab_bytes=4 * PAGE, backend=backend) as arena:
            seed = arena.allocate(PAGE)
            assert seed is not None
            seed.release()
            assert arena.stats.empty_slab_count == 1

            # Refusal on a second slab must NOT evict existing empty slabs or retry tight size.
            backend.raise_refused_error = True
            assert arena.allocate(6 * PAGE) is None
            assert arena.stats.empty_slab_count == 1
            assert arena.stats.slab_count == 1
            backend.raise_refused_error = False

            # Impossible alignment (min_slab_size > max_bytes) must NOT evict empty slabs either.
            arena.max_bytes = 8 * PAGE
            assert arena.allocate(PAGE, alignment=8 * PAGE) is None
            assert arena.stats.empty_slab_count == 1

        # OS MemoryError on oversized candidate slab trims empty slabs and falls back to min_slab_size.
        fail_max = [2 * PAGE]
        oom_calls: list[int] = []

        def flaky_alloc(sz: int) -> host_memory.AlignedRegion:
            oom_calls.append(sz)
            if sz > fail_max[0]:
                if len(oom_calls) >= 3:
                    fail_max[0] = 3 * PAGE
                raise MemoryError("simulated OS fragmentation")
            return host_memory.allocate_aligned_region(sz)

        with HostSlabArena(pinned=False, slab_bytes=8 * PAGE, allocator=flaky_alloc) as oom_arena:
            cached = oom_arena.allocate(2 * PAGE)
            assert cached is not None
            cached.release()
            assert oom_arena.stats.slab_count == 1
            big = oom_arena.allocate(3 * PAGE)
            assert big is not None and big.size == 3 * PAGE
            assert oom_arena.stats.slab_count == 1
            big.release()

    def test_tensor_and_view_lifetimes_prevent_reuse_and_validate_templates(self) -> None:
        backend = _FakeBackend()
        arena = HostSlabArena(pinned=True, slab_bytes=4 * PAGE, backend=backend)

        a1 = arena.allocate(PAGE)
        assert a1 is not None
        off1 = a1.offset
        t1 = a1.as_uint8_tensor()
        t1.fill_(42)
        del a1
        gc.collect()
        # Dropping ArenaAllocation handle while t1 is alive must keep span leased.
        assert arena.stats.live_allocations == 1

        a2 = arena.allocate(PAGE)
        assert a2 is not None and a2.offset != off1
        a2.as_uint8_tensor().fill_(99)
        assert int(t1[0].item()) == 42

        # Validation rejects non-strided, negative-stride, out-of-bounds, and 0-elem huge offset.
        sparse = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (2, 2))
        with pytest.raises(ValueError, match="strided"):
            a2.tensor_view(sparse)
        too_big = torch.empty(PAGE + 1, dtype=torch.uint8)
        with pytest.raises(ValueError, match="exceeding"):
            a2.tensor_view(too_big)
        zero_oob = torch.empty(PAGE + 10, dtype=torch.uint8)[PAGE + 10 :]
        with pytest.raises(ValueError, match="exceeding"):
            a2.tensor_view(zero_oob)

        # Explicit release while tensor_view is alive protects span even past arena.close().
        tmpl = torch.arange(64, dtype=torch.float32)
        held_view = a2.tensor_view(tmpl)
        held_view.copy_(tmpl)
        a2.release()
        arena.close()
        assert arena.stats.slab_count == 1
        torch.testing.assert_close(held_view, tmpl, rtol=0, atol=0)

        del t1, held_view
        gc.collect()
        assert arena.stats.slab_count == 0
        assert not backend.registered

    def test_close_waits_for_inflight_creation_and_gc_unregisters(self) -> None:
        backend = _FakeBackend()
        alloc_entered = threading.Event()
        allow_alloc = threading.Event()

        def blocking_alloc(sz: int) -> host_memory.AlignedRegion:
            alloc_entered.set()
            assert allow_alloc.wait(timeout=3.0)
            return host_memory.allocate_aligned_region(sz)

        arena = HostSlabArena(pinned=True, slab_bytes=4 * PAGE, backend=backend, allocator=blocking_alloc)
        err_holder: list[BaseException] = []

        def worker() -> None:
            try:
                arena.allocate(PAGE)
            except BaseException as exc:
                err_holder.append(exc)

        t_alloc = threading.Thread(target=worker)
        t_alloc.start()
        assert alloc_entered.wait(timeout=3.0)
        close_done = threading.Event()
        t_close = threading.Thread(target=lambda: (arena.close(), close_done.set()))
        t_close.start()
        assert not close_done.wait(timeout=0.05)
        allow_alloc.set()
        t_alloc.join(timeout=3.0)
        t_close.join(timeout=3.0)
        assert close_done.is_set()
        assert len(err_holder) == 1 and isinstance(err_holder[0], RuntimeError)
        assert arena.stats.slab_count == 0 and arena.stats.reserved_bytes == 0
        assert not backend.registered

        # GC without explicit close unregisters pinned slabs before unmapping.
        arena_gc = HostSlabArena(pinned=True, slab_bytes=4 * PAGE, backend=backend)
        item = arena_gc.allocate(PAGE)
        assert item is not None
        item.release()
        assert len(backend.registered) == 1
        del arena_gc
        gc.collect()
        assert not backend.registered


class TestPinnedBounceRing:
    def test_lazy_init_budget_clamping_and_registration_refusal(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        init_cuda = Mock(side_effect=AssertionError("CUDA must not initialize"))
        monkeypatch.setattr(torch.cuda, "init", init_cuda)
        backend = _FakeBackend()

        ring = PinnedBounceRing(slot_bytes=8 * PAGE, num_slots=4, min_bounce_bytes=PAGE, backend=backend)
        assert not ring.is_initialized
        assert not backend.register_calls
        # CPU destination is ineligible -> does not initialize.
        assert not ring.copy_pageable_to_device(torch.empty(2 * PAGE), torch.empty(2 * PAGE))
        assert not ring.is_initialized
        ring.close()

        # Disabled when budget is smaller than 2 sectors.
        tiny = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, max_bytes=PAGE, backend=backend)
        assert not tiny._ensure_initialized()
        tiny.close()

        # Finite budget clamps geometry cleanly and registers once.
        capped = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=4, max_bytes=5 * PAGE, backend=backend)
        assert capped._ensure_initialized()
        assert capped.reserved_bytes == 4 * PAGE
        assert capped.num_slots == 2 and capped.slot_bytes == 2 * PAGE
        with caplog.at_level(logging.WARNING):
            capped.close()
        assert "surviving references" not in caplog.text
        assert not backend.registered

        # Registration failure rolls back cleanly.
        refused_backend = _FakeBackend()
        refused_backend.capacity = PAGE
        refused_ring = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, backend=refused_backend)
        assert not refused_ring._ensure_initialized()
        assert refused_ring.registration_failures == 1
        assert not refused_ring.is_initialized
        refused_ring.close()

    @CUDA
    def test_copy_pageable_to_device_dtypes_views_concurrency_and_priority(self) -> None:
        with PinnedBounceRing(
            slot_bytes=4 * PAGE,
            num_slots=2,
            min_bounce_bytes=128,
        ) as ring:
            # Multi-chunk wraparound across bf16, float32, complex64, and sliced storage_offset.
            for dtype, shape in (
                (torch.bfloat16, (64, 512)),
                (torch.float32, (64, 512)),
                (torch.complex64, (32, 256)),
            ):
                base = torch.randn(*shape, dtype=dtype)
                dst = torch.empty_like(base, device="cuda")
                assert ring.copy_pageable_to_device(dst, base)
                torch.cuda.synchronize()
                torch.testing.assert_close(dst.cpu(), base, rtol=0, atol=0)

            full = torch.arange(32 * 512, dtype=torch.float32)
            sliced = full[256 : 256 + 16 * 512]
            dst_slice = torch.empty_like(sliced, device="cuda")
            assert ring.copy_pageable_to_device(dst_slice, sliced)
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_slice.cpu(), sliced, rtol=0, atol=0)

            # Ineligible cases return False cleanly without corrupting ring.
            small = torch.randn(8, dtype=torch.float32)
            assert not ring.copy_pageable_to_device(torch.empty_like(small, device="cuda"), small)
            non_contig = torch.randn(512, 64, dtype=torch.float32).t()
            assert not ring.copy_pageable_to_device(torch.empty(64, 512, device="cuda"), non_contig)

            # Contended multi-thread copy through 2 slots.
            streams = [torch.cuda.Stream() for _ in range(4)]

            def run_worker(idx: int) -> None:
                s = streams[idx]
                src = torch.full((40 * 1024 // 4,), float(idx + 1), dtype=torch.float32)
                d = torch.empty_like(src, device="cuda")
                assert ring.copy_pageable_to_device(d, src, stream=s)
                s.synchronize()
                torch.testing.assert_close(d.cpu(), src, rtol=0, atol=0)

            with ThreadPoolExecutor(max_workers=4) as pool:
                for f in [pool.submit(run_worker, i) for i in range(4)]:
                    f.result()

            # Background copy yields while foreground_transfer is active on target device.
            fg_entered = threading.Event()
            bg_done = threading.Event()
            dev = torch.device("cuda:0")
            src_bg = torch.ones(1024, dtype=torch.float32)
            dst_bg = torch.empty_like(src_bg, device=dev)

            def bg_worker() -> None:
                fg_entered.wait(timeout=2.0)
                with transfer_priority.background_transfers():
                    assert ring.copy_pageable_to_device(dst_bg, src_bg)
                bg_done.set()

            t = threading.Thread(target=bg_worker)
            t.start()
            try:
                with transfer_priority.foreground_transfer(dev):
                    fg_entered.set()
                    assert not bg_done.wait(timeout=0.05)
                assert bg_done.wait(timeout=2.0)
            finally:
                t.join(timeout=2.0)
            torch.cuda.synchronize()
            torch.testing.assert_close(dst_bg.cpu(), src_bg, rtol=0, atol=0)

    @CUDA
    def test_stream_file_slice_to_device_single_pipelined_serial_and_partial(self, tmp_path: Path) -> None:
        st_path = tmp_path / "stream.safetensors"
        t_small = torch.randn(31, 47, dtype=torch.bfloat16)
        # > 100 KiB tensor so multi-chunk streaming spans multiple 32 KiB bounce slots.
        t_large = torch.randn(192, 192, dtype=torch.float32)
        save_file({"small": t_small, "large": t_large}, st_path)

        with MappedCheckpoint(st_path) as ckpt:
            fs_small = file_slice(ckpt.get_tensor("small"))
            fs_large = file_slice(ckpt.get_tensor("large"))
        assert fs_small is not None and fs_large is not None

        with DirectFileReader() as reader:
            # 1. Pipelined 4-slot ring streaming both single-chunk and multi-chunk files + shard view.
            with PinnedBounceRing(slot_bytes=8 * PAGE, num_slots=4, min_bounce_bytes=0) as ring:
                dst_small = torch.empty_like(t_small, device="cuda")
                assert ring.stream_file_slice_to_device(dst_small, fs_small, reader=reader)
                torch.cuda.synchronize()
                torch.testing.assert_close(dst_small.cpu(), t_small, rtol=0, atol=0)

                dst_large = torch.empty_like(t_large, device="cuda")
                assert ring.stream_file_slice_to_device(dst_large, fs_large, reader=reader)
                torch.cuda.synchronize()
                torch.testing.assert_close(dst_large.cpu(), t_large, rtol=0, atol=0)

                # Partial shard view at unaligned middle byte offset.
                row_bytes = t_large.shape[1] * 4
                sub_dst = torch.empty((41, 192), dtype=torch.float32, device="cuda")
                assert ring.stream_file_slice_to_device(
                    sub_dst,
                    fs_large,
                    reader=reader,
                    view_offset_bytes=17 * row_bytes,
                    view_nbytes=41 * row_bytes,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(sub_dst.cpu(), t_large[17:58], rtol=0, atol=0)

            # 2. Single-slot serial fallback also works across multi-chunk spans.
            with PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=1, min_bounce_bytes=0) as ring1:
                dst_serial = torch.empty_like(t_large, device="cuda")
                assert ring1.stream_file_slice_to_device(dst_serial, fs_large, reader=reader)
                torch.cuda.synchronize()
                torch.testing.assert_close(dst_serial.cpu(), t_large, rtol=0, atol=0)

    @CUDA
    def test_stream_file_slice_errors_do_not_leak_slots_and_close_retries(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "trunc.bin"
        bad_file.write_bytes(b"z" * 1000)

        from piper_offload.checkpoint import FileSlice

        backend = _FakeBackend()
        with bad_file.open("rb") as f, DirectFileReader(direct_io=False) as reader:
            ring = PinnedBounceRing(slot_bytes=2 * PAGE, num_slots=3, backend=backend)
            dst = torch.empty(40 * 1024 // 4, dtype=torch.float32, device="cuda")
            with pytest.raises(OSError, match="checkpoint ended"):
                ring.stream_file_slice_to_device(dst, FileSlice(f, 0, 40 * 1024), reader=reader)
            # Failed unregister on close retains ring and succeeds on retry.
            assert ring.is_initialized
            ptr = next(iter(backend.registered))
            backend.unregister_errors.add(ptr)
            with pytest.raises(HostRegistrationError):
                ring.close()
            assert ring.unregistration_failures == 1
            assert not ring.is_initialized
            assert not ring.copy_pageable_to_device(dst, torch.empty(dst.shape))
            backend.unregister_errors.clear()
            ring.close()
            assert not backend.registered

    def test_live_view_keeps_ring_export_and_gc_teardown_and_validation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = _FakeBackend()
        ring = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, backend=backend)
        assert ring._ensure_initialized()
        assert ring._res.backing is not None
        dangling_mv = memoryview(ring._res.backing.region)
        with caplog.at_level(logging.WARNING):
            ring.close()
        assert "surviving buffer references" in caplog.text
        dangling_mv.release()
        assert not backend.registered

        # GC without close unregisters cleanly before freeing region without leaking buffer exports.
        ring_gc = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, backend=backend)
        assert ring_gc._ensure_initialized()
        assert len(backend.registered) == 1
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            del ring_gc
            gc.collect()
        assert not backend.registered
        assert "surviving buffer references" not in caplog.text

        # Precondition checks without requiring CUDA hardware.
        ring_val = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, min_bounce_bytes=0, backend=backend)

        class _SubTensor(torch.Tensor):
            pass

        sub = _SubTensor(torch.empty(PAGE))
        assert not ring_val.copy_pageable_to_device(torch.empty(PAGE), sub)
        grad_cpu = torch.empty(PAGE, requires_grad=True)
        assert not ring_val.copy_pageable_to_device(grad_cpu, torch.empty(PAGE))
        ring_val.close()

    @CUDA
    def test_cuda_preconditions_wrong_stream_and_poisoned_slot(self) -> None:
        backend = _FakeBackend()
        with PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, min_bounce_bytes=0, backend=backend) as ring:
            src = torch.ones(4 * PAGE // 4, dtype=torch.float32)
            dst_grad = torch.empty_like(src, device="cuda", requires_grad=True)
            assert not ring.copy_pageable_to_device(dst_grad, src)

            bad_stream = Mock(device=torch.device("cuda:99"))
            with pytest.raises(ValueError, match="does not match destination device"):
                ring.copy_pageable_to_device(torch.empty_like(src, device="cuda"), src, stream=bad_stream)

        # Poisoned slot when record_event + fallback synchronize both fail retains region safely.
        p_backend = _FakeBackend()
        p_ring = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, min_bounce_bytes=0, backend=p_backend)

        class _BrokenStream(torch.cuda.Stream):
            def record_event(self, event: object = None) -> torch.cuda.Event:
                raise RuntimeError("event error")

            def synchronize(self) -> None:
                raise RuntimeError("sync error")

        with pytest.raises(RuntimeError, match="sync error"):
            p_ring.copy_pageable_to_device(
                torch.empty(PAGE, device="cuda"),
                torch.ones(PAGE),
                stream=_BrokenStream(device=0),
            )
        assert not p_ring.is_initialized
        with host_arena_mod._RETAINED_LOCK:
            before_keys = set(host_arena_mod._RETAINED_REGIONS)
        try:
            p_ring.close()
            with host_arena_mod._RETAINED_LOCK:
                assert len(set(host_arena_mod._RETAINED_REGIONS) - before_keys) >= 1
        finally:
            with host_arena_mod._RETAINED_LOCK:
                for key in set(host_arena_mod._RETAINED_REGIONS) - before_keys:
                    host_arena_mod._RETAINED_REGIONS.pop(key, None)

    @CUDA
    def test_concurrent_pipelined_streaming_no_deadlock_and_1slot_yield(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        st_path = tmp_path / "concurrent_stream.safetensors"
        tensor = torch.randn(192, 192, dtype=torch.float32)
        save_file({"w": tensor}, st_path)
        with MappedCheckpoint(st_path) as ckpt:
            fs = file_slice(ckpt.get_tensor("w"))
        assert fs is not None

        with DirectFileReader() as reader, PinnedBounceRing(slot_bytes=8 * PAGE, num_slots=3) as ring3:
            def stream_one() -> None:
                d = torch.empty_like(tensor, device="cuda")
                assert ring3.stream_file_slice_to_device(d, fs, reader=reader)
                torch.cuda.synchronize()
                torch.testing.assert_close(d.cpu(), tensor, rtol=0, atol=0)

            with ThreadPoolExecutor(max_workers=4) as pool:
                for fut in [pool.submit(stream_one) for _ in range(4)]:
                    fut.result()

        with DirectFileReader() as reader, PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=1) as ring1:
            dev = torch.device("cuda:0")
            dst1 = torch.empty_like(tensor, device=dev)
            monkeypatch.setattr(transfer_priority, "_BACKGROUND_CHUNK_BYTES", 4 * PAGE)
            fg_ready = threading.Event()
            bg_done = threading.Event()

            def bg_stream() -> None:
                fg_ready.wait(timeout=3.0)
                with transfer_priority.background_transfers():
                    assert ring1.stream_file_slice_to_device(dst1, fs, reader=reader)
                bg_done.set()

            th = threading.Thread(target=bg_stream)
            th.start()
            with transfer_priority.foreground_transfer(dev):
                fg_ready.set()
                assert not bg_done.wait(timeout=0.05)
            th.join(timeout=5.0)
            assert bg_done.is_set()
            torch.cuda.synchronize()
            torch.testing.assert_close(dst1.cpu(), tensor, rtol=0, atol=0)


class TestHostArenaAndRingRegressions:
    def test_cyclic_gc_allocation_and_view_during_allocate_no_deadlock(self) -> None:
        old_threshold = gc.get_threshold()
        with HostSlabArena(pinned=False, slab_bytes=8 * PAGE) as arena:
            try:
                gc.set_threshold(5, 2, 2)
                for _ in range(30):
                    alloc = arena.allocate(PAGE)
                    assert alloc is not None
                    mv = alloc.aligned_view()
                    cycle: list[object] = [alloc, mv]
                    cycle.append(cycle)
                    del alloc, mv, cycle
                for _ in range(10):
                    alloc = arena.allocate(PAGE)
                    assert alloc is not None
                    sv = host_arena_mod._SpanView(alloc._lease, alloc.offset, alloc.offset + alloc.size)
                    mv = memoryview(sv)
                    cycle = [alloc, sv, mv]
                    cycle.append(cycle)
                    del alloc, sv, mv, cycle
                gc.collect()
                assert arena.stats.used_bytes == 0
                assert arena.stats.live_allocations == 0
                assert not host_arena_mod._ACTIVE_SPAN_EXPORTS
            finally:
                gc.set_threshold(*old_threshold)

    def test_concurrent_payload_storage_vs_release_race(self) -> None:
        def take_storage(
            alloc: host_arena_mod.ArenaAllocation, barrier: threading.Barrier
        ) -> torch.UntypedStorage | str:
            barrier.wait()
            try:
                return alloc.payload_storage()
            except RuntimeError as exc:
                return str(exc)

        def release_alloc(alloc: host_arena_mod.ArenaAllocation, barrier: threading.Barrier) -> None:
            barrier.wait()
            alloc.release()

        with HostSlabArena(pinned=False, slab_bytes=8 * PAGE) as arena, ThreadPoolExecutor(max_workers=2) as pool:
            for _ in range(30):
                alloc = arena.allocate(PAGE)
                assert alloc is not None
                barrier = threading.Barrier(2)
                f_store = pool.submit(take_storage, alloc, barrier)
                f_rel = pool.submit(release_alloc, alloc, barrier)
                f_rel.result(timeout=5.0)
                storage = f_store.result(timeout=5.0)
                if isinstance(storage, str):
                    assert "released" in storage.lower()
                else:
                    assert storage.nbytes() == PAGE
                    del storage
                del f_store, f_rel, alloc
            gc.collect()
            assert arena.stats.live_allocations == 0
            assert arena.stats.used_bytes == 0

    def test_close_waits_for_inflight_trim_unregister(self) -> None:
        backend = _FakeBackend()
        unreg_started, allow_unreg = threading.Event(), threading.Event()
        orig_unreg = backend.unregister

        def blocking_unreg(pointer: int) -> None:
            unreg_started.set()
            assert allow_unreg.wait(timeout=5.0)
            orig_unreg(pointer)

        backend.unregister = blocking_unreg  # type: ignore[method-assign]
        arena = HostSlabArena(pinned=True, slab_bytes=4 * PAGE, backend=backend)
        try:
            alloc = arena.allocate(PAGE)
            assert alloc is not None and alloc.is_pinned
            alloc.release()
            t_trim = threading.Thread(target=arena.trim_empty)
            t_trim.start()
            assert unreg_started.wait(timeout=5.0)
            t_close = threading.Thread(target=arena.close)
            t_close.start()
            t_close.join(timeout=0.05)
            assert t_close.is_alive()
            allow_unreg.set()
            t_trim.join(timeout=5.0)
            t_close.join(timeout=5.0)
            assert not t_trim.is_alive()
            assert not t_close.is_alive()
            assert not backend.registered
        finally:
            allow_unreg.set()
            arena.close()

    def test_invalid_allocator_region_rejected_and_closed(self) -> None:
        class _EmptyRegion:
            def __init__(self) -> None:
                self.closed = False

            def __buffer__(self, flags: int, /) -> memoryview:
                return memoryview(bytearray(0))

            def close(self) -> None:
                self.closed = True

        class _UndersizedRegion:
            def __init__(self) -> None:
                self._inner = host_memory.allocate_aligned_region(PAGE)
                self.closed = False

            def __buffer__(self, flags: int, /) -> memoryview:
                return memoryview(self._inner)

            def close(self) -> None:
                self.closed = True
                self._inner.close()

        empty_reg = _EmptyRegion()
        with (
            HostSlabArena(pinned=False, slab_bytes=4 * PAGE, allocator=lambda _sz: empty_reg) as a0,
            pytest.raises(ValueError, match="Allocator returned invalid region"),
        ):
            a0.allocate(PAGE)
        assert empty_reg.closed

        small_reg = _UndersizedRegion()
        with (
            HostSlabArena(pinned=False, slab_bytes=4 * PAGE, allocator=lambda _sz: small_reg) as a1,
            pytest.raises(ValueError, match="Allocator returned invalid region"),
        ):
            a1.allocate(PAGE)
        assert small_reg.closed

    def test_bounce_ring_gc_sync_error_retains_without_unregister(self) -> None:
        class _ExplodingEvent:
            def synchronize(self) -> None:
                raise RuntimeError("event sync failed")

        backend = _FakeBackend()
        with host_arena_mod._RETAINED_LOCK:
            before_keys = set(host_arena_mod._RETAINED_REGIONS)
        try:
            ring = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, backend=backend)
            assert ring._ensure_initialized()
            assert len(backend.registered) == 1
            assert ring._res.slots is not None
            ring._res.slots[0].event = _ExplodingEvent()  # type: ignore[assignment]
            del ring
            gc.collect()
            assert len(backend.registered) == 1
            with host_arena_mod._RETAINED_LOCK:
                assert len(set(host_arena_mod._RETAINED_REGIONS) - before_keys) == 1
        finally:
            with host_arena_mod._RETAINED_LOCK:
                for key in set(host_arena_mod._RETAINED_REGIONS) - before_keys:
                    host_arena_mod._RETAINED_REGIONS.pop(key, None)

    def test_bounce_ring_close_recovers_after_keyboard_interrupt_in_checked_out_wait(self) -> None:
        backend = _FakeBackend()
        ring = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, backend=backend)
        try:
            assert ring._ensure_initialized()
            slot = ring._checkout_slot(blocking=False)
            assert slot is not None
            orig_wait = ring._cv.wait
            calls = 0

            def interrupting_wait(timeout: float | None = None) -> bool:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise KeyboardInterrupt
                return orig_wait(timeout)

            ring._cv.wait = interrupting_wait  # type: ignore[method-assign]
            with pytest.raises(KeyboardInterrupt):
                ring.close()
            assert not ring._closing_active
            ring._cv.wait = orig_wait  # type: ignore[method-assign]
            ring._checkin_slot(slot)
            ring.close()
            assert not backend.registered
        finally:
            ring.close()

    def test_copy_pageable_to_device_rejects_requires_grad(self) -> None:
        backend = _FakeBackend()
        with PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, min_bounce_bytes=0, backend=backend) as ring:
            src_grad = torch.ones(16, dtype=torch.float32, requires_grad=True)
            assert not ring.copy_pageable_to_device(torch.empty(16, dtype=torch.float32), src_grad)
            assert not ring.is_initialized
            if torch.cuda.is_available():
                dst_cuda = torch.empty(16, dtype=torch.float32, device="cuda")
                assert not ring.copy_pageable_to_device(dst_cuda, src_grad)
                assert not ring.is_initialized
                assert ring.copy_pageable_to_device(dst_cuda, torch.ones(16, dtype=torch.float32))
                assert ring.is_initialized
                torch.cuda.synchronize()

    def test_windows_virtual_region_failed_from_memory_preserves_exports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        released: list[int] = []

        class _FakeKernel:
            def allocate(self, _addr: object, _sz: int, _t: int, _p: int) -> int:
                return 0x10000

            def free(self, addr: int, _sz: int, _free_type: int) -> int:
                released.append(addr)
                return 1

        def exploding_from_memory(_addr: int, _sz: int, _flags: int) -> memoryview:
            raise RuntimeError("boom")

        monkeypatch.setattr(windows_memory, "_kernel32", _FakeKernel)
        monkeypatch.setattr(windows_memory, "_from_memory", lambda: exploding_from_memory)
        region = windows_memory.VirtualRegion(PAGE)
        with pytest.raises(RuntimeError, match="boom"):
            memoryview(region)
        assert region._exports == 0
        region.close()
        assert region.closed and released == [0x10000]


class TestPinnedBounceRingCudaRegressions:
    @CUDA
    def test_cuda_close_skips_sync_when_ring_poisoned(self) -> None:
        class _ExplodingEvent:
            def synchronize(self) -> None:
                raise RuntimeError("sync should be skipped")

        backend = _FakeBackend()
        with host_arena_mod._RETAINED_LOCK:
            before_keys = set(host_arena_mod._RETAINED_REGIONS)
        try:
            ring = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, backend=backend)
            assert ring._ensure_initialized()
            assert ring._res.slots is not None and len(ring._res.slots) == 2
            ring._res.slots[1].event = _ExplodingEvent()  # type: ignore[assignment]
            ring._res.poisoned = True
            ring.close()
            assert len(backend.registered) == 1
            with host_arena_mod._RETAINED_LOCK:
                assert len(set(host_arena_mod._RETAINED_REGIONS) - before_keys) == 1
        finally:
            with host_arena_mod._RETAINED_LOCK:
                for key in set(host_arena_mod._RETAINED_REGIONS) - before_keys:
                    host_arena_mod._RETAINED_REGIONS.pop(key, None)

    @staticmethod
    def _install_pipelined_interrupt_hooks(
        ring: PinnedBounceRing,
    ) -> tuple[threading.Event, threading.Event, threading.Event, list[tuple[bool, bool]], list[threading.Timer]]:
        read_lock = threading.Lock()
        blocked_assigned = False
        blocked_slot_ref: list[object | None] = [None]
        chunk1_started, allow_chunk1, chunk1_finished = threading.Event(), threading.Event(), threading.Event()
        checkin_records: list[tuple[bool, bool]] = []
        timers: list[threading.Timer] = []
        orig_read, orig_checkin = ring._read_into_slot, ring._checkin_slot

        def hooked_read(slot, reader_arg, file_obj, span):  # type: ignore[no-untyped-def]
            nonlocal blocked_assigned
            with read_lock:
                is_blocked = not blocked_assigned and span.aligned_offset > 0
                if is_blocked:
                    blocked_assigned = True
                    blocked_slot_ref[0] = slot
            if is_blocked:
                chunk1_started.set()
                assert allow_chunk1.wait(timeout=5.0)
            result = orig_read(slot, reader_arg, file_obj, span)
            if is_blocked:
                chunk1_finished.set()
            return result

        def hooked_copy(*_args: object, **_kwargs: object) -> None:
            assert chunk1_started.wait(timeout=5.0)
            timer = threading.Timer(0.03, allow_chunk1.set)
            timers.append(timer)
            timer.start()
            raise KeyboardInterrupt

        def hooked_checkin(slot) -> None:  # type: ignore[no-untyped-def]
            checkin_records.append((slot is blocked_slot_ref[0], chunk1_finished.is_set()))
            orig_checkin(slot)

        ring._read_into_slot = hooked_read  # type: ignore[method-assign]
        ring._copy_slot_to_cuda = hooked_copy  # type: ignore[method-assign]
        ring._checkin_slot = hooked_checkin  # type: ignore[method-assign]
        return chunk1_started, allow_chunk1, chunk1_finished, checkin_records, timers

    @CUDA
    def test_cuda_run_pipelined_chunks_drains_worker_before_checkin_on_interrupt(self, tmp_path: Path) -> None:
        st_path = tmp_path / "pipe_ki.safetensors"
        tensor = torch.arange(16 * PAGE, dtype=torch.uint8)
        save_file({"w": tensor}, st_path)
        backend = _FakeBackend()
        ring = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=3, backend=backend)
        orig_copy = ring._copy_slot_to_cuda
        started, allow, finished, checkin_records, timers = self._install_pipelined_interrupt_hooks(ring)
        try:
            with MappedCheckpoint(st_path) as ckpt, DirectFileReader() as reader:
                fs = file_slice(ckpt.get_tensor("w"))
                assert fs is not None
                dst = torch.empty_like(tensor, device="cuda")
                with pytest.raises(KeyboardInterrupt):
                    ring.stream_file_slice_to_device(dst, fs, reader=reader)
            assert started.is_set()
            assert finished.is_set()
            assert (True, True) in checkin_records
        finally:
            allow.set()
            for timer in timers:
                timer.join(timeout=5.0)
            ring._copy_slot_to_cuda = orig_copy  # type: ignore[method-assign]
            ring.close()
            assert not backend.registered

    @CUDA
    def test_cuda_bounce_ring_initialized_in_inference_mode_works_outside_inference_mode(self) -> None:
        with PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, min_bounce_bytes=0) as ring:
            src0 = torch.arange(PAGE, dtype=torch.float32)
            dst0 = torch.empty_like(src0, device="cuda")
            with torch.inference_mode():
                assert ring.copy_pageable_to_device(dst0, src0)
            torch.cuda.synchronize()
            torch.testing.assert_close(dst0.cpu(), src0, rtol=0, atol=0)

            src1 = torch.arange(PAGE, 2 * PAGE, dtype=torch.float32)
            dst1 = torch.empty_like(src1, device="cuda")
            assert ring.copy_pageable_to_device(dst1, src1)
            torch.cuda.synchronize()
            torch.testing.assert_close(dst1.cpu(), src1, rtol=0, atol=0)


class TestHostArenaFinalReviewRegressions:
    def test_ensure_geometry_rollback_interrupt_unwedges_initializing_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeBackend()
        ring = PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, backend=backend)
        try:
            orig_init_backing = ring._init_backing_off_lock
            tripped = False

            def interrupting_init_backing():  # type: ignore[no-untyped-def]
                result = orig_init_backing()
                assert result is not None
                orig_release = result[1].release_or_retain

                def interrupting_release(*, keep_region: bool = False) -> bool:
                    orig_release(keep_region=keep_region)
                    raise KeyboardInterrupt

                result[1].release_or_retain = interrupting_release  # type: ignore[method-assign]
                return result

            def interrupting_deque(items):  # type: ignore[no-untyped-def]
                nonlocal tripped
                if not tripped:
                    tripped = True
                    raise RuntimeError("boom after init_backing")
                return host_arena_mod.deque(items)

            monkeypatch.setattr(ring, "_init_backing_off_lock", interrupting_init_backing)
            monkeypatch.setattr(host_arena_mod, "deque", interrupting_deque)
            with pytest.raises(KeyboardInterrupt):
                ring._ensure_initialized()
            assert ring._state is host_arena_mod._RingState.UNINITIALIZED

            monkeypatch.undo()
            assert ring._ensure_initialized()
            ring.close()
            assert not backend.registered
        finally:
            ring.close()

    def test_arena_close_waits_for_uncommitted_slab_rollback_unregister(self) -> None:
        backend = _FakeBackend()
        create_started, close_waiting = threading.Event(), threading.Event()
        unreg_started, allow_unreg = threading.Event(), threading.Event()
        orig_unreg = backend.unregister

        def blocking_unreg(pointer: int) -> None:
            unreg_started.set()
            assert allow_unreg.wait(timeout=5.0)
            orig_unreg(pointer)

        backend.unregister = blocking_unreg  # type: ignore[method-assign]
        arena = HostSlabArena(pinned=True, slab_bytes=4 * PAGE, backend=backend)
        orig_alloc_backing = arena._allocate_slab_backing
        orig_wait = arena._wait_cv_locked
        alloc_errors: list[BaseException] = []

        def paused_alloc_backing(candidate_size: int, min_slab_size: int):  # type: ignore[no-untyped-def]
            res = orig_alloc_backing(candidate_size, min_slab_size)
            create_started.set()
            assert close_waiting.wait(timeout=5.0)
            return res

        def signaling_wait() -> None:
            close_waiting.set()
            orig_wait()

        arena._allocate_slab_backing = paused_alloc_backing  # type: ignore[method-assign]
        arena._wait_cv_locked = signaling_wait  # type: ignore[method-assign]

        def run_alloc() -> None:
            try:
                arena.allocate(PAGE)
            except BaseException as exc:
                alloc_errors.append(exc)

        try:
            t_alloc = threading.Thread(target=run_alloc)
            t_alloc.start()
            assert create_started.wait(timeout=5.0)
            t_close = threading.Thread(target=arena.close)
            t_close.start()
            assert unreg_started.wait(timeout=5.0)
            t_close.join(timeout=0.05)
            assert t_close.is_alive()
            allow_unreg.set()
            t_alloc.join(timeout=5.0)
            t_close.join(timeout=5.0)
            assert not t_close.is_alive()
            assert len(alloc_errors) == 1 and isinstance(alloc_errors[0], RuntimeError)
            assert not backend.registered
        finally:
            close_waiting.set()
            allow_unreg.set()
            arena.close()

    def test_arena_clear_reports_unregistration_failure_on_pending_return_slab(self) -> None:
        backend = _FakeBackend()
        arena = HostSlabArena(pinned=True, slab_bytes=4 * PAGE, backend=backend)
        try:
            alloc = arena.allocate(PAGE)
            assert alloc is not None and alloc.is_pinned
            arena.max_bytes = 0
            orig_unreg = backend.unregister

            def failing_unreg(_pointer: int) -> None:
                raise RuntimeError("unregister blocked")

            backend.unregister = failing_unreg  # type: ignore[method-assign]
            with arena._owned_lock():
                alloc.release()
            with pytest.raises(RuntimeError, match="Could not release"):
                arena.clear()
            backend.unregister = orig_unreg  # type: ignore[method-assign]
            arena.clear()
            assert not backend.registered
        finally:
            arena.close()

    def test_allocate_and_finalize_retain_on_base_exception_after_register(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeBackend()
        with host_arena_mod._RETAINED_LOCK:
            before_keys = set(host_arena_mod._RETAINED_REGIONS)

        def fail_unreg(_pointer: int) -> None:
            raise KeyboardInterrupt

        try:
            with monkeypatch.context() as ctx:
                ctx.setattr(
                    host_arena_mod._RegisteredRegion,
                    "__init__",
                    lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
                )
                ctx.setattr(backend, "unregister", fail_unreg)
                with pytest.raises(KeyboardInterrupt):
                    host_arena_mod._RegisteredRegion.allocate(
                        4 * PAGE,
                        allocator=host_memory.allocate_aligned_region,
                        pinned=True,
                        backend=backend,
                        context_label="test",
                    )
                with host_arena_mod._RETAINED_LOCK:
                    assert len(set(host_arena_mod._RETAINED_REGIONS) - before_keys) == 1
                assert len(backend.registered) == 1

            reg, status = host_arena_mod._RegisteredRegion.allocate(
                4 * PAGE,
                allocator=host_memory.allocate_aligned_region,
                pinned=True,
                backend=backend,
                context_label="test",
            )
            assert reg is not None and status is host_arena_mod._RegStatus.REGISTERED
            raw_region = reg.region
            monkeypatch.setattr(backend, "unregister", fail_unreg)
            with pytest.raises(KeyboardInterrupt):
                reg._finalizer()
            assert not raw_region.closed
            with host_arena_mod._RETAINED_LOCK:
                assert len(set(host_arena_mod._RETAINED_REGIONS) - before_keys) == 2
        finally:
            monkeypatch.undo()
            with host_arena_mod._RETAINED_LOCK:
                for key in set(host_arena_mod._RETAINED_REGIONS) - before_keys:
                    retained = host_arena_mod._RETAINED_REGIONS.pop(key, None)
                    if retained is not None:
                        retained.close()

    def test_unhashable_custom_region_retained_and_non_byte_region_rejected(self) -> None:
        class _UnhashableRegion:
            __hash__ = None  # type: ignore[assignment]

            def __init__(self) -> None:
                self._inner = host_memory.allocate_aligned_region(4 * PAGE)
                self.closed = False

            def __buffer__(self, flags: int, /) -> memoryview:
                return memoryview(self._inner)

            def close(self) -> None:
                self.closed = True
                self._inner.close()

        class _Float32Region:
            def __init__(self) -> None:
                self._tensor = torch.zeros(4 * PAGE, dtype=torch.float32)
                self.closed = False

            def __buffer__(self, flags: int, /) -> memoryview:
                return memoryview(self._tensor.numpy())

            def close(self) -> None:
                self.closed = True

        backend = _FakeBackend()
        with host_arena_mod._RETAINED_LOCK:
            before_keys = set(host_arena_mod._RETAINED_REGIONS)
        try:
            reg, _ = host_arena_mod._RegisteredRegion.allocate(
                4 * PAGE,
                allocator=lambda _sz: _UnhashableRegion(),
                pinned=True,
                backend=backend,
                context_label="unhashable",
            )
            assert reg is not None
            reg.release_or_retain(keep_region=True)
            with host_arena_mod._RETAINED_LOCK:
                assert len(set(host_arena_mod._RETAINED_REGIONS) - before_keys) == 1

            bad_reg = _Float32Region()
            with pytest.raises(ValueError, match="Allocator returned invalid region"):
                host_arena_mod._RegisteredRegion.allocate(
                    4 * PAGE,
                    allocator=lambda _sz: bad_reg,
                    pinned=False,
                    backend=None,
                    context_label="non-byte",
                )
            assert bad_reg.closed
        finally:
            with host_arena_mod._RETAINED_LOCK:
                for key in set(host_arena_mod._RETAINED_REGIONS) - before_keys:
                    retained = host_arena_mod._RETAINED_REGIONS.pop(key, None)
                    if retained is not None:
                        retained.close()

    def test_bounce_ring_initializes_under_non_cpu_default_device(self) -> None:
        backend = _FakeBackend()
        with (
            torch.device("meta"),
            PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, backend=backend) as ring,
        ):
            assert ring._ensure_initialized()
            assert ring._res.slots is not None
            assert all(slot.uint8_view.device.type == "cpu" for slot in ring._res.slots)
        assert not backend.registered

        if torch.cuda.is_available():
            src = torch.arange(PAGE, dtype=torch.float32, device="cpu")
            dst = torch.empty_like(src, device="cuda")
            with (
                torch.device("cuda"),
                PinnedBounceRing(slot_bytes=4 * PAGE, num_slots=2, min_bounce_bytes=0) as cuda_ring,
            ):
                assert cuda_ring.copy_pageable_to_device(dst, src)
                torch.cuda.synchronize()
            torch.testing.assert_close(dst.cpu(), src, rtol=0, atol=0)



