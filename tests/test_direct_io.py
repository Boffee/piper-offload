"""Tests for Layer 1 Direct I/O FileSlice reader (`_direct_io.py`)."""

from __future__ import annotations

import errno
import gc
import mmap
import os
import random
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

import piper_offload._direct_io as direct_io_mod
from piper_offload import MappedCheckpoint
from piper_offload._direct_io import (
    DEFAULT_SECTOR_SIZE,
    AlignedSpan,
    DirectFileReader,
    DirectReadRequest,
    _Overlapped,
    _Win32IO,
    align_span,
    read_file_slice,
)
from piper_offload._host_memory import allocate_aligned_region
from piper_offload.checkpoint import FileSlice, file_slice

_WORKSPACE_TMP_ROOT = Path(__file__).resolve().parent.parent / ".pytest_cache"


def _page_align(size: int) -> int:
    return ((max(size, mmap.PAGESIZE) + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE


def _can_probe_o_direct(directory: Path) -> bool:
    o_direct = getattr(os, "O_DIRECT", 0)
    if not o_direct or not hasattr(os, "preadv"):
        return False
    probe = directory / ".o_direct_probe.bin"
    try:
        probe.write_bytes(b"x" * 4096)
        fd = os.open(probe, os.O_RDONLY | o_direct)
    except OSError:
        return False
    try:
        buf = allocate_aligned_region(_page_align(4096))
        try:
            with memoryview(buf) as mv:
                os.preadv(fd, [mv], 0)
        finally:
            buf.close()
        return True
    except OSError:
        return False
    finally:
        os.close(fd)
        probe.unlink(missing_ok=True)


@contextmanager
def _direct_capable_dir() -> Iterator[tuple[Path, bool]]:
    _WORKSPACE_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_WORKSPACE_TMP_ROOT, prefix="dio-") as tmp:
        path = Path(tmp)
        yield path, _can_probe_o_direct(path)


class TestAlignSpan:
    def test_zero_length_returns_zero_span(self) -> None:
        assert align_span(12345, 0) == AlignedSpan(0, 0, 0, 0)
        assert align_span(0, 0, sector_size=8192).aligned_end == 0

    def test_aligned_and_unaligned_bounds(self) -> None:
        assert align_span(4096, 8192) == AlignedSpan(4096, 8192, 0, 8192)
        assert align_span(4100, 100) == AlignedSpan(4096, 4096, 4, 100)
        span = align_span(4095, 200)
        assert span == AlignedSpan(0, 8192, 4095, 200)
        assert span.aligned_end == 8192

    def test_custom_power_of_two_sector_sizes(self) -> None:
        assert align_span(520, 100, sector_size=512) == AlignedSpan(512, 512, 8, 100)
        assert align_span(100, 8192, sector_size=8192) == AlignedSpan(0, 16384, 100, 8192)

    @pytest.mark.parametrize(
        ("offset", "length", "sector_size"),
        [
            (-1, 16, 4096),
            (0, -1, 4096),
            (0, 16, 0),
            (0, 16, -4096),
            (0, 16, 6000),
            (0, 16, True),  # type: ignore[arg-type]
        ],
    )
    def test_rejects_invalid_arguments(self, offset: int, length: int, sector_size: int) -> None:
        with pytest.raises(ValueError, match="non-negative|positive|power of two"):
            align_span(offset, length, sector_size=sector_size)


class TestDirectFileReaderPosix:
    def test_reads_unaligned_and_eof_tail_and_safetensors_slices(self) -> None:
        with _direct_capable_dir() as (work_dir, o_direct_supported):
            raw_path = work_dir / "payload.bin"
            rng = random.Random(42)
            raw_bytes = rng.randbytes(4096 * 5 + 313)
            raw_path.write_bytes(raw_bytes)

            with raw_path.open("rb") as f, DirectFileReader() as reader:
                if o_direct_supported:
                    assert reader.is_direct(f)
                assert reader.sector_size(f) == DEFAULT_SECTOR_SIZE

                # Unaligned slice straddling sector boundaries.
                fs_mid = FileSlice(f, offset=4090, length=500)
                span_mid = align_span(fs_mid.offset, fs_mid.length)
                reg_mid = allocate_aligned_region(_page_align(span_mid.aligned_length))
                try:
                    with memoryview(reg_mid) as mv_mid:
                        reader.read_into([DirectReadRequest(fs_mid, mv_mid, head_pad=span_mid.head_pad)])
                        assert bytes(mv_mid[span_mid.head_pad : span_mid.head_pad + fs_mid.length]) == (
                            raw_bytes[4090 : 4090 + 500]
                        )
                finally:
                    reg_mid.close()

                # Tail slice ending at an unaligned EOF.
                tail_off = 4096 * 4 + 17
                tail_len = len(raw_bytes) - tail_off
                fs_tail = FileSlice(f, offset=tail_off, length=tail_len)
                span_tail = align_span(fs_tail.offset, fs_tail.length)
                reg_tail = allocate_aligned_region(_page_align(span_tail.aligned_length))
                try:
                    with memoryview(reg_tail) as mv_tail:
                        reader.read_into([DirectReadRequest(fs_tail, mv_tail, head_pad=span_tail.head_pad)])
                        assert bytes(mv_tail[span_tail.head_pad : span_tail.head_pad + tail_len]) == (
                            raw_bytes[tail_off:]
                        )
                finally:
                    reg_tail.close()

            # Verify end-to-end with MappedCheckpoint safetensors slices.
            st_path = work_dir / "weights.safetensors"
            t0 = torch.randn(31, 47, dtype=torch.bfloat16)
            t1 = torch.randint(0, 255, (19, 13), dtype=torch.uint8)
            save_file({"a": t0, "b": t1}, st_path)
            with MappedCheckpoint(st_path) as ckpt:
                a_mapped = ckpt.get_tensor("a")
                b_mapped = ckpt.get_tensor("b")
            fs_a = file_slice(a_mapped)
            fs_b = file_slice(b_mapped)
            assert fs_a is not None and fs_b is not None

            out_a = bytearray(fs_a.length)
            out_b = bytearray(fs_b.length)
            with DirectFileReader() as reader:
                with memoryview(out_a) as mv_a, memoryview(out_b) as mv_b:
                    reader.read_into(
                        [
                            DirectReadRequest(fs_a, mv_a, aligned_target=False),
                            DirectReadRequest(fs_b, mv_b, aligned_target=False),
                        ]
                    )
            assert torch.equal(torch.frombuffer(out_a, dtype=torch.bfloat16).reshape(t0.shape), t0)
            assert torch.equal(torch.frombuffer(out_b, dtype=torch.uint8).reshape(t1.shape), t1)
            del a_mapped, b_mapped, fs_a, fs_b
            gc.collect()

    def test_coalesces_nearby_small_requests_and_splits_large_reads(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "coalesce.bin"
        payload = random.Random(7).randbytes(256 * 1024 + 500)
        path.write_bytes(payload)

        orig_preadv = getattr(os, "preadv", None)
        if orig_preadv is None:
            pytest.skip("os.preadv unavailable on this platform")
        calls: list[tuple[int, int]] = []

        def tracked_preadv(fd: int, buffers: list[memoryview], offset: int) -> int:
            calls.append((offset, sum(len(b) for b in buffers)))
            return orig_preadv(fd, buffers, offset)

        monkeypatch.setattr(os, "preadv", tracked_preadv)

        with path.open("rb") as f, DirectFileReader(
            direct_io=False,
            chunk_bytes=32 * 1024,
            coalesce_gap=16 * 1024,
            coalesce_max_bytes=128 * 1024,
            coalesce_threshold=64 * 1024,
            max_workers=2,
        ) as reader:
            # 4 tiny nearby spans should coalesce into a single preadv call.
            out0 = bytearray(300)
            out1 = bytearray(400)
            out2 = bytearray(500)
            out3 = bytearray(200)
            reqs = [
                DirectReadRequest(FileSlice(f, 100, 300), memoryview(out0), aligned_target=False),
                DirectReadRequest(FileSlice(f, 12000, 400), memoryview(out1), aligned_target=False),
                DirectReadRequest(FileSlice(f, 25000, 500), memoryview(out2), aligned_target=False),
                DirectReadRequest(FileSlice(f, 35000, 200), memoryview(out3), aligned_target=False),
            ]
            reader.read_into(reqs)
            for req in reqs:
                req.destination.release()

            assert len(calls) == 1
            assert bytes(out0) == payload[100:400]
            assert bytes(out1) == payload[12000:12400]
            assert bytes(out2) == payload[25000:25500]
            assert bytes(out3) == payload[35000:35200]

            # Large aligned request (> coalesce_threshold) splits along chunk_bytes across workers.
            calls.clear()
            large_off = 4096 * 2 + 123
            large_len = 96 * 1024
            span = align_span(large_off, large_len)
            large_reg = allocate_aligned_region(_page_align(span.aligned_length))
            try:
                with memoryview(large_reg) as large_mv:
                    req = DirectReadRequest(FileSlice(f, large_off, large_len), large_mv, head_pad=span.head_pad)
                    reader.read_into([req])
                    assert bytes(large_mv[span.head_pad : span.head_pad + large_len]) == (
                        payload[large_off : large_off + large_len]
                    )
            finally:
                large_reg.close()
            assert len(calls) >= 3

            # Large unaligned request (> coalesce_max_bytes) chunks through thread-local scratch.
            unaligned_buf = bytearray(large_len)
            with memoryview(unaligned_buf) as umv:
                reader.read_into([DirectReadRequest(FileSlice(f, large_off, large_len), umv, aligned_target=False)])
            assert bytes(unaligned_buf) == payload[large_off : large_off + large_len]

    def test_einval_downgrades_to_buffered_with_fadvise(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(os, "preadv"):
            pytest.skip("POSIX os.preadv required")
        path = tmp_path / "fallback.bin"
        data = b"piper-direct-io-fallback-" * 600
        path.write_bytes(data)

        fadvise_calls: list[tuple[int, int, int, int]] = []
        orig_preadv = os.preadv
        orig_open = os.open
        orig_fstat = os.fstat

        with path.open("rb") as f:
            orig_fd = f.fileno()
            orig_stat = os.fstat(orig_fd)
            fake_direct_fd = os.dup(orig_fd)

            def fake_open(p: object, flags: int, *args: object) -> int:
                if flags & getattr(os, "O_DIRECT", 0):
                    return fake_direct_fd
                return orig_open(p, flags, *args)  # type: ignore[arg-type]

            def fake_fstat(fd: int) -> os.stat_result:
                return orig_stat if fd == fake_direct_fd else orig_fstat(fd)

            def fake_fadvise(fd: int, offset: int, length: int, advice: int) -> None:
                fadvise_calls.append((fd, offset, length, advice))

            def failing_preadv(fd: int, buffers: list[memoryview], offset: int) -> int:
                if fd == fake_direct_fd:
                    raise OSError(errno.EINVAL, "Invalid argument")
                return orig_preadv(fd, buffers, offset)

            monkeypatch.setattr(os, "open", fake_open)
            monkeypatch.setattr(os, "fstat", fake_fstat)
            monkeypatch.setattr(os, "O_DIRECT", 0x4000, raising=False)
            monkeypatch.setattr(os, "posix_fadvise", fake_fadvise, raising=False)
            monkeypatch.setattr(direct_io_mod, "_O_DIRECT", 0x4000)
            monkeypatch.setattr(os, "preadv", failing_preadv)

            out = bytearray(len(data))
            with DirectFileReader() as reader:
                assert reader.is_direct(f)
                with memoryview(out) as mv:
                    reader.read_into([DirectReadRequest(FileSlice(f, 0, len(data)), mv, aligned_target=False)])
                assert not reader.is_direct(f)
                assert bytes(out) == data
            assert any(advice == direct_io_mod._POSIX_FADV_DONTNEED for *_, advice in fadvise_calls)

    def test_unaligned_short_read_downgrades_to_buffered(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(os, "preadv"):
            pytest.skip("POSIX os.preadv required")
        path = tmp_path / "short_direct.bin"
        data = b"piper-short-read-" * 800
        path.write_bytes(data)

        orig_preadv = os.preadv
        orig_open = os.open
        orig_fstat = os.fstat

        with path.open("rb") as f:
            orig_fd = f.fileno()
            orig_stat = os.fstat(orig_fd)
            fake_direct_fd = os.dup(orig_fd)

            def fake_open(p: object, flags: int, *args: object) -> int:
                if flags & 0x4000:
                    return fake_direct_fd
                return orig_open(p, flags, *args)  # type: ignore[arg-type]

            def fake_fstat(fd: int) -> os.stat_result:
                return orig_stat if fd == fake_direct_fd else orig_fstat(fd)

            def short_preadv(fd: int, buffers: list[memoryview], _offset: int) -> int:
                if fd == fake_direct_fd:
                    view = buffers[0]
                    chunk = min(137, len(view))
                    view[:chunk] = b"\x00" * chunk
                    return chunk
                return orig_preadv(fd, buffers, _offset)

            monkeypatch.setattr(os, "open", fake_open)
            monkeypatch.setattr(os, "fstat", fake_fstat)
            monkeypatch.setattr(os, "O_DIRECT", 0x4000, raising=False)
            monkeypatch.setattr(direct_io_mod, "_O_DIRECT", 0x4000)
            monkeypatch.setattr(os, "preadv", short_preadv)

            out = bytearray(8192)
            with DirectFileReader() as reader:
                assert reader.is_direct(f)
                with memoryview(out) as mv:
                    reader.read_into([DirectReadRequest(FileSlice(f, 0, 8192), mv, aligned_target=False)])
                assert not reader.is_direct(f)
                assert bytes(out) == data[:8192]

    @pytest.mark.skipif(sys.platform == "win32", reason="open-unlink requires POSIX")
    def test_replaced_file_reads_original_inode_not_new_path(self) -> None:
        with _direct_capable_dir() as (work_dir, _o_direct):
            path = work_dir / "swap.bin"
            original = b"ORIGINAL" * 1024
            replacement = b"REPLACED" * 1024
            path.write_bytes(original)

            with path.open("rb") as f:
                fs = FileSlice(f, 0, len(original))
                path.unlink()
                path.write_bytes(replacement)

                out = bytearray(len(original))
                with DirectFileReader() as reader:
                    assert not reader.is_direct(f)
                    with memoryview(out) as mv:
                        reader.read_into([DirectReadRequest(fs, mv, aligned_target=False)])
                assert bytes(out) == original

    def test_empty_and_truncated_file_and_validation_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "short.bin"
        path.write_bytes(b"abcd" * 100)
        with path.open("rb") as f:
            empty_buf = bytearray(0)
            with memoryview(empty_buf) as empty_mv:
                read_file_slice(FileSlice(f, 0, 0), empty_mv, aligned_target=False, direct_io=False)

            too_short = bytearray(600)
            with memoryview(too_short) as short_mv, pytest.raises(OSError, match="checkpoint ended"):
                read_file_slice(FileSlice(f, 0, 600), short_mv, aligned_target=False, direct_io=False)

            readonly_mv = memoryview(b"0" * 16)
            with pytest.raises(ValueError, match="writable"):
                read_file_slice(FileSlice(f, 0, 16), readonly_mv, aligned_target=False, direct_io=False)

            strided = memoryview(bytearray(32))[::2]
            with pytest.raises(ValueError, match="C-contiguous"):
                read_file_slice(FileSlice(f, 0, 16), strided, aligned_target=False, direct_io=False)

            unaligned = bytearray(8192)
            with memoryview(unaligned) as un_mv:
                with pytest.raises(ValueError, match="head_pad"):
                    read_file_slice(FileSlice(f, 5, 16), un_mv, head_pad=0, aligned_target=True, direct_io=False)
                with pytest.raises(ValueError, match="length"):
                    read_file_slice(
                        FileSlice(f, 0, 8000), un_mv[:4096], head_pad=0, aligned_target=True, direct_io=False
                    )
                unaligned_slice = mmap.mmap(-1, 8192)
                try:
                    with memoryview(unaligned_slice)[1:4097] as off1_mv, pytest.raises(ValueError, match="aligned"):
                        read_file_slice(FileSlice(f, 0, 100), off1_mv, head_pad=0, aligned_target=True, direct_io=False)
                finally:
                    unaligned_slice.close()

    def test_read_aligned_validation_and_closed_reader_rejections(self, tmp_path: Path) -> None:
        path = tmp_path / "val.bin"
        path.write_bytes(b"v" * 8192)
        with path.open("rb") as f, DirectFileReader(direct_io=False) as reader:
            reg = allocate_aligned_region(_page_align(8192))
            try:
                with memoryview(reg) as mv:
                    with pytest.raises(ValueError, match="aligned_offset"):
                        reader.read_aligned(f, -4096, 4096, 100, mv)
                    with pytest.raises(ValueError, match="aligned"):
                        reader.read_aligned(f, 1, 4096, 100, mv)
                    with pytest.raises(ValueError, match="aligned"):
                        reader.read_aligned(f, 0, 100, 100, mv)
                    with pytest.raises(ValueError, match="exceeds"):
                        reader.read_aligned(f, 0, 4096, 5000, mv)
                    with pytest.raises(ValueError, match="C-contiguous"):
                        reader.read_aligned(f, 0, 4096, 100, mv[::2])
                    with pytest.raises(ValueError, match="aligned"):
                        reader.read_aligned(f, 0, 4096, 100, mv[1:4097])
                with pytest.raises(ValueError, match="writable"):
                    reader.read_aligned(f, 0, 4096, 100, memoryview(b"x" * 4096))
            finally:
                reg.close()

        with path.open("rb") as f:
            with pytest.raises(RuntimeError, match="closed"):
                reader.read_into([])
            with pytest.raises(RuntimeError, match="closed"):
                reader.read_aligned(f, 0, 0, 0, memoryview(bytearray(0)))
            with pytest.raises(RuntimeError, match="closed"):
                reader.is_direct(f)
            with pytest.raises(RuntimeError, match="closed"):
                reader.sector_size(f)

    def test_unaligned_truncated_eof_preserves_direct_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not hasattr(os, "preadv"):
            pytest.skip("POSIX os.preadv required")
        path = tmp_path / "trunc5000.bin"
        data = b"T" * 5000
        path.write_bytes(data)

        orig_open, orig_fstat, orig_preadv = os.open, os.fstat, os.preadv
        with path.open("rb") as f:
            orig_stat = os.fstat(f.fileno())
            fake_fd = os.dup(f.fileno())

            def fake_open(p: object, flags: int, *args: object) -> int:
                return fake_fd if (flags & 0x4000) else orig_open(p, flags, *args)  # type: ignore[arg-type]

            def fake_fstat(fd: int) -> os.stat_result:
                return orig_stat if fd == fake_fd else orig_fstat(fd)

            def direct_preadv(fd: int, buffers: list[memoryview], offset: int) -> int:
                return orig_preadv(fd, buffers, offset)

            monkeypatch.setattr(os, "open", fake_open)
            monkeypatch.setattr(os, "fstat", fake_fstat)
            monkeypatch.setattr(os, "O_DIRECT", 0x4000, raising=False)
            monkeypatch.setattr(direct_io_mod, "_O_DIRECT", 0x4000)
            monkeypatch.setattr(os, "preadv", direct_preadv)

            reg = allocate_aligned_region(_page_align(8192))
            try:
                with DirectFileReader() as reader, memoryview(reg) as mv:
                    assert reader.is_direct(f)
                    with pytest.raises(OSError, match="checkpoint ended at byte 5000"):
                        reader.read_aligned(f, 0, 8192, 6000, mv)
                    assert reader.is_direct(f)
            finally:
                reg.close()

    def test_overlapping_duplicate_2d_float32_and_weakref_collection(self, tmp_path: Path) -> None:
        path = tmp_path / "overlap.bin"
        payload = random.Random(55).randbytes(16384)
        path.write_bytes(payload)
        out_f32 = bytearray(64)
        dup_a, dup_b = bytearray(400), bytearray(400)
        with DirectFileReader(direct_io=False, coalesce_threshold=8192, coalesce_max_bytes=16384) as reader:
            f_temp = path.open("rb")
            with memoryview(out_f32).cast("f", shape=(4, 4)) as mv_2d, memoryview(dup_a) as ma, memoryview(dup_b) as mb:
                reader.read_into(
                    [
                        DirectReadRequest(FileSlice(f_temp, 100, 64), mv_2d, aligned_target=False),
                        DirectReadRequest(FileSlice(f_temp, 120, 400), ma, aligned_target=False),
                        DirectReadRequest(FileSlice(f_temp, 120, 400), mb, aligned_target=False),
                    ]
                )
            assert bytes(out_f32) == payload[100:164]
            assert bytes(dup_a) == bytes(dup_b) == payload[120:520]
            assert len(reader._handles) == 1
            entry = next(iter(reader._handles.values()))
            f_temp.close()
            del f_temp
            gc.collect()
            assert entry.weak_file() is None
            assert entry.handle._backend._closed
            prune_path = tmp_path / "prune.bin"
            prune_data = b"prune-test" * 8
            prune_path.write_bytes(prune_data)
            with prune_path.open("rb") as f_prune:
                out_prune = bytearray(32)
                with memoryview(out_prune) as mv_prune:
                    req_prune = DirectReadRequest(FileSlice(f_prune, 0, 32), mv_prune, aligned_target=False)
                    reader.read_into([req_prune])
                assert bytes(out_prune) == prune_data[:32]
                assert len(reader._handles) == 1
                assert reader._handles[id(f_prune)] is not entry

    def test_concurrent_close_drains_inflight_and_keyboard_interrupt_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not hasattr(os, "preadv"):
            pytest.skip("POSIX os.preadv required")
        path = tmp_path / "drain.bin"
        data = b"D" * 16384
        path.write_bytes(data)

        entered = threading.Event()
        release_read = threading.Event()
        orig_preadv = os.preadv

        def slow_preadv(fd: int, buffers: list[memoryview], offset: int) -> int:
            entered.set()
            assert release_read.wait(timeout=3.0)
            return orig_preadv(fd, buffers, offset)

        monkeypatch.setattr(os, "preadv", slow_preadv)
        out = bytearray(4096)
        reader = DirectFileReader(direct_io=False)
        close_done = threading.Event()
        with path.open("rb") as f:
            with memoryview(out) as mv:
                req = DirectReadRequest(FileSlice(f, 0, 4096), mv, aligned_target=False)
                t_read = threading.Thread(target=lambda: reader.read_into([req]))
                t_read.start()
                assert entered.wait(timeout=3.0)
                t_close = threading.Thread(target=lambda: (reader.close(), close_done.set()))
                t_close.start()
                assert not close_done.wait(timeout=0.05)
                release_read.set()
                t_read.join(timeout=3.0)
                t_close.join(timeout=3.0)
                assert close_done.is_set()
                assert bytes(out) == data[:4096]

        monkeypatch.setattr(os, "preadv", orig_preadv)
        call_num = 0
        orig_submit = ThreadPoolExecutor.submit

        def failing_submit(self_pool, fn, /, *args, **kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 2:
                raise KeyboardInterrupt("simulated submit cancel")
            return orig_submit(self_pool, fn, *args, **kwargs)

        monkeypatch.setattr(ThreadPoolExecutor, "submit", failing_submit)
        with path.open("rb") as f, DirectFileReader(
            direct_io=False, coalesce_threshold=0, coalesce_max_bytes=4096, max_workers=2
        ) as k_reader:
            b0, b1 = bytearray(4096), bytearray(4096)
            with memoryview(b0) as mv0, memoryview(b1) as mv1, pytest.raises(KeyboardInterrupt):
                k_reader.read_into(
                    [
                        DirectReadRequest(FileSlice(f, 0, 4096), mv0, aligned_target=False),
                        DirectReadRequest(FileSlice(f, 8192, 4096), mv1, aligned_target=False),
                    ]
                )


class TestDirectFileReaderWindowsEmulation:
    @staticmethod
    def _make_fake_win32(
        file_bytes: bytes,
    ) -> tuple[_Win32IO, dict[int, int], list[int], list[tuple[int, int, int]], list[int | None], list[bool]]:
        lock = threading.Lock()
        next_handle = 100
        opened_threads: dict[int, int] = {}
        closed_handles: list[int] = []
        recorded_reads: list[tuple[int, int, int]] = []
        fail_read_error: list[int | None] = [None]
        fail_open = [False]

        def open_direct(_path: str) -> int:
            if fail_open[0]:
                raise OSError(errno.EIO, "denied", None, 5)
            nonlocal next_handle
            with lock:
                h = next_handle
                next_handle += 1
                opened_threads[h] = threading.get_ident()
                return h

        def read_at(handle: int, target: memoryview, offset: int) -> int:
            assert (offset % 8192) == 0
            assert (len(target) % 8192) == 0
            if fail_read_error[0] is not None:
                err = OSError(errno.EINVAL, "bad param")
                err.winerror = fail_read_error[0]  # type: ignore[attr-defined]
                raise err
            with lock:
                recorded_reads.append((handle, offset, len(target)))
            real_off = offset % len(file_bytes)
            avail = min(len(target), len(file_bytes) - real_off)
            if avail <= 0:
                return 0
            target[:avail] = file_bytes[real_off : real_off + avail]
            return avail

        def close_handle(handle: int) -> None:
            with lock:
                closed_handles.append(handle)

        win32 = _Win32IO(
            open_direct=open_direct,
            read_at=read_at,
            close_handle=close_handle,
            sector_size=lambda _p: 8192,
        )
        return win32, opened_threads, closed_handles, recorded_reads, fail_read_error, fail_open

    def test_overlapped_64bit_split_high_and_low_offsets(self) -> None:
        offset = (5 << 32) + 0x12345000
        ov = _Overlapped(
            Internal=0,
            InternalHigh=0,
            Offset=offset & 0xFFFFFFFF,
            OffsetHigh=(offset >> 32) & 0xFFFFFFFF,
            hEvent=None,
        )
        assert ov.Offset == 0x12345000
        assert ov.OffsetHigh == 5

    def test_windows_no_buffering_large_offsets_and_custom_sector_size(self, tmp_path: Path) -> None:
        backing_path = tmp_path / "win_test.bin"
        file_bytes = random.Random(99).randbytes(65536)
        backing_path.write_bytes(file_bytes)
        win32, opened_threads, closed_handles, recorded_reads, _, _ = self._make_fake_win32(file_bytes)

        with backing_path.open("rb") as f:
            high_offset = (1 << 33) + 123
            region = allocate_aligned_region(_page_align(32768))
            try:
                with DirectFileReader(win32_io=win32, chunk_bytes=16384, max_workers=2) as reader:
                    assert reader.is_direct(f)
                    assert reader.sector_size(f) == 8192
                    span = align_span(high_offset, 10000, sector_size=8192)
                    assert span.aligned_length == 16384
                    with memoryview(region) as raw_mv:
                        pad = (-direct_io_mod._buffer_pointer(raw_mv)) % 8192
                        with raw_mv[pad : pad + 16384] as mv:
                            n = reader.read_aligned(
                                f, span.aligned_offset, span.aligned_length, span.head_pad + 10000, mv
                            )
                            assert n >= span.head_pad + 10000
                            start_in_file = high_offset % len(file_bytes)
                            assert bytes(mv[span.head_pad : span.head_pad + 10000]) == (
                                file_bytes[start_in_file : start_in_file + 10000]
                            )

                        b0 = bytearray(400)
                        b1 = bytearray(400)
                        with memoryview(b0) as mv_b0, memoryview(b1) as mv_b1:
                            reader.read_into(
                                [
                                    DirectReadRequest(FileSlice(f, 10, 400), mv_b0, aligned_target=False),
                                    DirectReadRequest(FileSlice(f, 50000, 400), mv_b1, aligned_target=False),
                                ]
                            )
                    assert bytes(b0) == file_bytes[10:410]
                    assert bytes(b1) == file_bytes[50000:50400]
                    assert any(off >= (1 << 33) for _, off, _ in recorded_reads)
                assert sorted(closed_handles) == sorted(opened_threads.keys())
            finally:
                region.close()

    def test_windows_no_buffering_read_and_open_fallbacks(self, tmp_path: Path) -> None:
        backing_path = tmp_path / "win_fallback.bin"
        file_bytes = random.Random(101).randbytes(65536)
        backing_path.write_bytes(file_bytes)
        win32, _, closed_handles, _, fail_read_error, fail_open = self._make_fake_win32(file_bytes)

        with backing_path.open("rb") as f:
            fail_read_error[0] = 87
            out_fb = bytearray(1200)
            with DirectFileReader(win32_io=win32) as reader_fb:
                assert reader_fb.is_direct(f)
                with memoryview(out_fb) as mv_fb:
                    reader_fb.read_into([DirectReadRequest(FileSlice(f, 50, 1200), mv_fb, aligned_target=False)])
                assert not reader_fb.is_direct(f)
                assert bytes(out_fb) == file_bytes[50:1250]
            assert closed_handles

            fail_read_error[0] = None
            fail_open[0] = True
            out_open_fb = bytearray(1200)
            with DirectFileReader(win32_io=win32) as reader_open_fb:
                assert not reader_open_fb.is_direct(f)
                with memoryview(out_open_fb) as mv_open_fb:
                    reader_open_fb.read_into(
                        [DirectReadRequest(FileSlice(f, 50, 1200), mv_open_fb, aligned_target=False)]
                    )
                assert bytes(out_open_fb) == file_bytes[50:1250]

    def test_windows_replaced_file_fdopen_and_ephemeral_thread_pooling(self, tmp_path: Path) -> None:
        backing_path = tmp_path / "win_swap.bin"
        original = random.Random(202).randbytes(24576)
        replacement = random.Random(303).randbytes(24576)
        backing_path.write_bytes(original)
        win32, opened_threads, closed_handles, _, fail_read_error, _ = self._make_fake_win32(replacement)

        # Replaced file must still read the original open file bytes (not replacement path).
        if sys.platform != "win32":
            with backing_path.open("rb") as f:
                backing_path.unlink()
                backing_path.write_bytes(replacement)
                out = bytearray(10000)
                with DirectFileReader(win32_io=win32) as reader:
                    assert not reader.is_direct(f)
                    with memoryview(out) as mv:
                        reader.read_into([DirectReadRequest(FileSlice(f, 123, 10000), mv, aligned_target=False)])
                    assert bytes(out) == original[123:10123]

        # Pathless reader falls back cleanly to caller file across platforms.
        backing_path.write_bytes(original)
        with (
            patch.object(direct_io_mod, "_resolve_open_path", return_value=""),
            backing_path.open("rb") as pathless_file,
            DirectFileReader(win32_io=win32) as reader_fd,
        ):
            assert not reader_fd.is_direct(pathless_file)
            out_fd = bytearray(500)
            with memoryview(out_fd) as mv_fd:
                req_fd = DirectReadRequest(FileSlice(pathless_file, 10, 500), mv_fd, aligned_target=False)
                reader_fd.read_into([req_fd])
            assert bytes(out_fd) == original[10:510]

        # Sequential ephemeral threads reuse one pooled handle and never double-close on downgrade (err=1).
        closed_handles.clear()
        opened_threads.clear()
        with backing_path.open("rb") as f2, DirectFileReader(win32_io=win32, max_workers=2) as reader_pool:
            for _ in range(4):
                out_ephem = bytearray(1024)

                def step(target_buf: bytearray = out_ephem) -> None:
                    with memoryview(target_buf) as mv_e:
                        reader_pool.read_into([DirectReadRequest(FileSlice(f2, 100, 1024), mv_e, aligned_target=False)])

                t = threading.Thread(target=step)
                t.start()
                t.join(timeout=3.0)
                assert bytes(out_ephem) == replacement[100:1124]
            assert len(opened_threads) == 1
            fail_read_error[0] = 1
            out_downgrade = bytearray(1024)
            with memoryview(out_downgrade) as mv_dg:
                reader_pool.read_into([DirectReadRequest(FileSlice(f2, 50, 1024), mv_dg, aligned_target=False)])
            assert not reader_pool.is_direct(f2)
        assert len(closed_handles) == len(set(closed_handles)) == len(opened_threads)

    def test_windows_non_wrapping_eof_and_short_read_downgrade(self, tmp_path: Path) -> None:
        file_bytes = random.Random(404).randbytes(10000)
        backing_path = tmp_path / "win_eof.bin"
        backing_path.write_bytes(file_bytes)
        short_mode = [False]

        def open_direct(_path: str) -> int:
            return 1

        def read_at(_handle: int, target: memoryview, offset: int) -> int:
            if short_mode[0]:
                target[:128] = file_bytes[offset : offset + 128]
                return 128
            avail = min(len(target), len(file_bytes) - offset)
            if avail <= 0:
                return 0
            target[:avail] = file_bytes[offset : offset + avail]
            return avail

        win32 = _Win32IO(
            open_direct=open_direct,
            read_at=read_at,
            close_handle=lambda _h: None,
            sector_size=lambda _p: 8192,
        )
        reg = allocate_aligned_region(_page_align(32768))
        try:
            with backing_path.open("rb") as f, DirectFileReader(win32_io=win32) as reader:
                with memoryview(reg) as raw_mv:
                    pad = (-direct_io_mod._buffer_pointer(raw_mv)) % 8192
                    with raw_mv[pad : pad + 16384] as mv:
                        with pytest.raises(OSError, match="checkpoint ended at byte 10000"):
                            reader.read_aligned(f, 0, 16384, 12000, mv)
                        assert reader.is_direct(f)
                        short_mode[0] = True
                        assert reader.read_aligned(f, 0, 8192, 4096, mv) >= 4096
                        assert not reader.is_direct(f)
                        assert bytes(mv[:4096]) == file_bytes[:4096]
        finally:
            reg.close()


class TestDirectFileReaderRegressions:
    def test_concurrent_close_drains_inflight_multi_request_read(self, tmp_path: Path) -> None:
        data1 = random.Random(11).randbytes(32768)
        data2 = random.Random(22).randbytes(32768)
        p1, p2 = tmp_path / "f1.bin", tmp_path / "f2.bin"
        p1.write_bytes(data1)
        p2.write_bytes(data2)
        first_open, proceed = threading.Event(), threading.Event()
        errors: list[BaseException] = []

        def open_direct(path: str) -> int:
            if Path(path).name == "f1.bin":
                first_open.set()
                assert proceed.wait(timeout=5.0)
                return 1
            return 2

        def read_at(handle: int, target: memoryview, offset: int) -> int:
            buf = data1 if handle == 1 else data2
            avail = min(len(target), len(buf) - offset)
            if avail > 0:
                target[:avail] = buf[offset : offset + avail]
            return max(avail, 0)

        win32 = _Win32IO(
            open_direct=open_direct,
            read_at=read_at,
            close_handle=lambda _h: None,
            sector_size=lambda _p: 8192,
        )
        reader = DirectFileReader(win32_io=win32, max_workers=2)
        try:
            with p1.open("rb") as f1, p2.open("rb") as f2:
                out1, out2 = bytearray(6000), bytearray(7000)

                def run_read() -> None:
                    try:
                        with memoryview(out1) as mv1, memoryview(out2) as mv2:
                            reader.read_into(
                                [
                                    DirectReadRequest(FileSlice(f1, 123, 6000), mv1, aligned_target=False),
                                    DirectReadRequest(FileSlice(f2, 321, 7000), mv2, aligned_target=False),
                                ]
                            )
                    except BaseException as exc:
                        errors.append(exc)

                t_read = threading.Thread(target=run_read)
                t_read.start()
                assert first_open.wait(timeout=5.0)
                t_close = threading.Thread(target=reader.close)
                t_close.start()
                deadline = time.monotonic() + 2.0
                while not reader._gate._closing and time.monotonic() < deadline:
                    time.sleep(0.005)
                assert reader._gate._closing
                proceed.set()
                t_read.join(timeout=5.0)
                t_close.join(timeout=5.0)
                assert not errors
                assert bytes(out1) == data1[123:6123]
                assert bytes(out2) == data2[321:7321]
                with memoryview(out1) as mv1, pytest.raises(RuntimeError, match="closed"):
                    reader.read_into([DirectReadRequest(FileSlice(f1, 0, 16), mv1, aligned_target=False)])
        finally:
            reader.close()

    def test_scratch_pool_reuses_wave_buffers_small_then_large(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        large_len = _page_align(8192) * 4
        data = random.Random(33).randbytes(large_len * 3)
        path = tmp_path / "wave.bin"
        path.write_bytes(data)
        win32, *_ = TestDirectFileReaderWindowsEmulation._make_fake_win32(data)
        alloc_sizes: list[int] = []
        orig_alloc = direct_io_mod.host_memory.allocate_aligned_region

        def tracked_alloc(size: int) -> object:
            alloc_sizes.append(size)
            return orig_alloc(size)

        monkeypatch.setattr(direct_io_mod.host_memory, "allocate_aligned_region", tracked_alloc)
        with path.open("rb") as f, DirectFileReader(win32_io=win32, chunk_bytes=large_len, max_workers=2) as reader:
            s0, s1 = bytearray(100), bytearray(100)
            with memoryview(s0) as mv0, memoryview(s1) as mv1:
                reader.read_into(
                    [
                        DirectReadRequest(FileSlice(f, 0, 100), mv0, aligned_target=False),
                        DirectReadRequest(FileSlice(f, 8192, 100), mv1, aligned_target=False),
                    ]
                )
            small_calls = len(alloc_sizes)
            b0, b1 = bytearray(large_len), bytearray(large_len)
            with memoryview(b0) as mb0, memoryview(b1) as mb1:
                reader.read_into(
                    [
                        DirectReadRequest(FileSlice(f, 0, large_len), mb0, aligned_target=False),
                        DirectReadRequest(FileSlice(f, large_len, large_len), mb1, aligned_target=False),
                    ]
                )
                after_large1 = len(alloc_sizes)
                assert after_large1 > small_calls
                reader.read_into(
                    [
                        DirectReadRequest(FileSlice(f, 0, large_len), mb0, aligned_target=False),
                        DirectReadRequest(FileSlice(f, large_len, large_len), mb1, aligned_target=False),
                    ]
                )
                assert len(alloc_sizes) == after_large1
            assert bytes(b0) == data[:large_len]
            assert bytes(b1) == data[large_len : 2 * large_len]

    def test_closed_file_probe_no_fd_leak_and_preadv_absent_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = random.Random(44).randbytes(16384)
        path = tmp_path / "probe.bin"
        path.write_bytes(data)
        opened_fds: set[int] = set()
        closed_fds: set[int] = set()
        orig_open, orig_close = os.open, os.close

        def spy_open(p: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
            try:
                fd = orig_open(p, flags, mode)
            except OSError:
                fd = orig_open(p, flags & ~getattr(os, "O_DIRECT", 0), mode)
            opened_fds.add(fd)
            return fd

        def spy_close(fd: int) -> None:
            if fd in opened_fds:
                closed_fds.add(fd)
            orig_close(fd)

        f_closed = path.open("rb")
        f_closed.close()
        with DirectFileReader() as reader:
            with monkeypatch.context() as ctx:
                ctx.setattr(direct_io_mod.os, "open", spy_open)
                ctx.setattr(direct_io_mod.os, "close", spy_close)
                with pytest.raises(ValueError, match="closed file"):
                    reader.is_direct(f_closed)
            assert not opened_fds
            assert len(reader._handles) == 0

        if hasattr(os, "O_DIRECT") and hasattr(os, "preadv"):
            with path.open("rb") as f_probe, monkeypatch.context() as ctx:
                ctx.setattr(direct_io_mod.os, "open", spy_open)
                ctx.setattr(direct_io_mod.os, "close", spy_close)
                ctx.setattr(
                    direct_io_mod,
                    "_same_open_file",
                    lambda *_args: (_ for _ in ()).throw(ValueError("probe boom")),
                )
                with pytest.raises(ValueError, match="probe boom"):
                    direct_io_mod._PosixBackend(f_probe, direct_io=True)
            assert len(opened_fds) == 1
            assert opened_fds == closed_fds

        monkeypatch.delattr(os, "preadv", raising=False)
        slices = [(0, 1500), (1500, 2300), (3800, 3100), (7000, 4000)]
        with path.open("rb") as f, DirectFileReader(direct_io=False, max_workers=4) as reader_buf:

            def read_one(item: tuple[int, int]) -> tuple[int, int, bytes]:
                off, length = item
                out = bytearray(length)
                with memoryview(out) as mv:
                    reader_buf.read_into([DirectReadRequest(FileSlice(f, off, length), mv, aligned_target=False)])
                return off, length, bytes(out)

            with ThreadPoolExecutor(max_workers=4) as pool:
                for off, length, payload in pool.map(read_one, slices):
                    assert payload == data[off : off + length]

    def test_close_keyboard_interrupt_retries_shutdown_and_closes_resources(self, tmp_path: Path) -> None:
        data = random.Random(55).randbytes(32768)
        p0, p1 = tmp_path / "ki0.bin", tmp_path / "ki1.bin"
        p0.write_bytes(data)
        p1.write_bytes(data)
        win32, _, closed_handles, *_ = TestDirectFileReaderWindowsEmulation._make_fake_win32(data)
        reader = DirectFileReader(win32_io=win32, max_workers=2)
        with p0.open("rb") as f0, p1.open("rb") as f1:
            b0, b1 = bytearray(500), bytearray(500)
            with memoryview(b0) as mv0, memoryview(b1) as mv1:
                reader.read_into(
                    [
                        DirectReadRequest(FileSlice(f0, 0, 500), mv0, aligned_target=False),
                        DirectReadRequest(FileSlice(f1, 0, 500), mv1, aligned_target=False),
                    ]
                )
            assert reader._pool is not None
            assert len(reader._handles) == 2
            assert len(reader._scratch_pool._idle) >= 1
            orig_shutdown = reader._pool.shutdown
            calls = 0

            def interrupting_shutdown(*args: object, **kwargs: object) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise KeyboardInterrupt
                orig_shutdown(*args, **kwargs)  # type: ignore[arg-type]

            reader._pool.shutdown = interrupting_shutdown  # type: ignore[method-assign]
            with pytest.raises(KeyboardInterrupt):
                reader.close()
            assert calls == 2
            assert len(reader._handles) == 0
            assert len(closed_handles) == 2
            assert reader._scratch_pool._closed
            assert not reader._scratch_pool._idle

    def test_win_downgrade_requires_winerror_and_read_aligned_rejects_bad_views(self, tmp_path: Path) -> None:
        assert not direct_io_mod._is_win_downgrade_error(OSError(errno.EPERM, "perm"))
        assert not direct_io_mod._is_win_downgrade_error(OSError(errno.EINVAL, "no winerror"))
        err_87 = OSError(errno.EINVAL, "bad param")
        err_87.winerror = 87  # type: ignore[attr-defined]
        assert direct_io_mod._is_win_downgrade_error(err_87)

        backing_path = tmp_path / "views.bin"
        backing_path.write_bytes(b"\x00" * _page_align(8192))
        strided = memoryview(torch.zeros(8192, dtype=torch.float32).numpy())[::2]
        ro = memoryview(b"\x00" * 4096)
        with backing_path.open("rb") as f, DirectFileReader(direct_io=False) as reader:
            with pytest.raises(ValueError, match="C-contiguous"):
                reader.read_aligned(f, 0, 4096, 4096, strided)
            with pytest.raises(ValueError, match="writable"):
                reader.read_aligned(f, 0, 4096, 4096, ro)

