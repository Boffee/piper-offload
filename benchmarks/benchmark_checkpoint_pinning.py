"""Separate checkpoint page reads, host registration, transfers, and release.

Linux diagnostic using the existing safetensors loader. The retain-private
comparison temporarily suppresses discard in this benchmark process only; it
shows the cost of registering pages whose private copies already exist.
"""

# ruff: noqa: T201 - benchmark reports progress and measurements.

import argparse
import ctypes
import json
import mmap
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open

import piper_offload.pin_manager as pin_module
from piper_offload import PinManager
from piper_offload._host_registration import RuntimeHostRegistration


class _TimedBackend(RuntimeHostRegistration):
    def __init__(self) -> None:
        self.register_s = 0.0
        self.unregister_s = 0.0
        self.register_calls = 0

    def register(self, pointer: int, size: int) -> bool:
        start = time.perf_counter()
        try:
            return super().register(pointer, size)
        finally:
            self.register_s += time.perf_counter() - start
            self.register_calls += 1

    def unregister(self, pointer: int) -> None:
        start = time.perf_counter()
        try:
            super().unregister(pointer)
        finally:
            self.unregister_s += time.perf_counter() - start


def _read_pages(tensor: torch.Tensor) -> None:
    data = tensor.reshape(-1).view(torch.uint8).numpy()
    data[::mmap.PAGESIZE].sum(dtype=np.uint64)
    data[-1].item()


def _memory(path: Path) -> dict[str, int]:
    result = {"Rss": 0, "Anonymous": 0, "Locked": 0}
    inside = False
    with open("/proc/self/smaps", encoding="utf-8") as smaps:
        for line in smaps:
            fields = line.split()
            if "-" in fields[0]:
                inside = fields[-1] == str(path)
            elif inside and fields[0].rstrip(":") in result:
                result[fields[0].rstrip(":")] += int(fields[1]) * 1024
    return result


def _disk_bytes() -> int:
    with open("/proc/self/io", encoding="utf-8") as io:
        return next(int(line.split()[1]) for line in io if line.startswith("read_bytes:"))


def _timed(operation: Callable[[], object]) -> float:
    start = time.perf_counter()
    operation()
    return time.perf_counter() - start


def _populate_private_pages(tensors: list[torch.Tensor], workers: int, *, reset_mapping: bool) -> None:
    """Experimental parallel COW preparation; never changes checkpoint bytes.

    MADV_POPULATE_WRITE (Linux 5.14+) prepares writable page tables for these
    private file views. Run only before registering the benchmark's selected
    checkpoint ranges. Production admission/failure handling is not implemented
    here; the benchmark has enough pin budget for every requested allocation.
    """
    ranges = []
    for tensor in tensors:
        storage = tensor.untyped_storage()
        pointer, size = storage.data_ptr(), storage.nbytes()
        start = pointer // mmap.PAGESIZE * mmap.PAGESIZE
        end = -(-(pointer + size) // mmap.PAGESIZE) * mmap.PAGESIZE
        ranges.append((start, end))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    chunk_size = 64 * 1024**2
    chunks = [(start, min(chunk_size, end - start))
              for begin, end in merged for start in range(begin, end, chunk_size)]
    madvise = pin_module._madvise()
    if reset_mapping:
        # These benchmark ranges have no outstanding leases. Drop existing
        # read-prefaulted PTEs before preparing writable ones; file-cache pages
        # stay cached. Do not apply this to arbitrary active model mappings.
        for start, end in merged:
            if madvise(start, end - start, pin_module._MADV_DONTNEED):
                raise OSError(ctypes.get_errno(), "MADV_DONTNEED")

    def populate(span: tuple[int, int]) -> None:
        if madvise(*span, 23):  # MADV_POPULATE_WRITE in Linux mman.h
            raise OSError(ctypes.get_errno(), "MADV_POPULATE_WRITE")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(populate, chunks))


def _transfer(tensors: list[torch.Tensor], target: torch.Tensor) -> dict[str, float | bool]:
    # Reuse a bounded GPU buffer and retain byte samples from every chunk.
    # Validation is outside the transfer timer; memory usage stays below 1 GiB.
    chunks = [chunk for tensor in tensors for chunk in tensor.reshape(-1).view(torch.uint8).split(target.numel())]
    checks = torch.empty((len(chunks), 2), device="cuda", dtype=torch.uint8)
    expected = torch.tensor([[chunk[0].item(), chunk[-1].item()] for chunk in chunks], dtype=torch.uint8)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for index, chunk in enumerate(chunks):
        target[:chunk.numel()].copy_(chunk, non_blocking=True)
        checks[index, :1].copy_(target[:1])
        checks[index, 1:].copy_(target[chunk.numel() - 1:chunk.numel()])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    torch.testing.assert_close(checks.cpu(), expected, rtol=0, atol=0)
    return {"transfer_s": elapsed, "byte_samples_match": True}


def _run_cycles(
    tensors: list[torch.Tensor],
    target: torch.Tensor,
    path: Path,
    manager: PinManager,
    backend: _TimedBackend,
    repeats: int,
    policies: list[str],
    populate_workers: int,
    reset_mapping: bool,
) -> list[dict[str, object]]:
    total = sum(tensor.nbytes for tensor in tensors)
    samples = []
    native_advice = pin_module._madvise()
    advice_times = {pin_module._MADV_DONTNEED: 0.0, pin_module._MADV_WILLNEED: 0.0, 23: 0.0}

    def timed_advice(pointer: int, size: int, advice: int) -> int:
        started = time.perf_counter()
        try:
            return native_advice(pointer, size, advice)
        finally:
            advice_times[advice] += time.perf_counter() - started

    try:
        with patch.object(pin_module, "_madvise", lambda: timed_advice):
            for policy in policies:
                for repeat in range(repeats):
                    before = _memory(path)
                    before_io = _disk_bytes()
                    backend.register_s = backend.unregister_s = 0.0
                    backend.register_calls = 0
                    advice_times.update(dict.fromkeys(advice_times, 0.0))
                    populate_s = 0.0
                    if populate_workers:
                        populate_s = _timed(lambda: _populate_private_pages(
                            tensors, populate_workers, reset_mapping=reset_mapping,
                        ))
                    populate_io = _disk_bytes() - before_io
                    before_io = _disk_bytes()
                    # Release advice timings exclude any mapping reset that
                    # was already included in populate_s.
                    advice_times.update(dict.fromkeys(advice_times, 0.0))
                    start = time.perf_counter()
                    lease = manager.acquire(tensors)
                    acquire_s = time.perf_counter() - start
                    register_io = _disk_bytes() - before_io
                    try:
                        assert lease.registered_bytes == total, (lease.registered_bytes, total)
                        registered = _memory(path)
                        transfers = _transfer(tensors, target)
                    finally:
                        if policy == "retain_private":
                            with patch.object(pin_module, "_discard_and_warm", lambda *_args, **_kwargs: None):
                                release_s = _timed(lease.close)
                        else:
                            release_s = _timed(lease.close)
                    sample = {
                        "policy": policy, "repeat": repeat, "acquire_s": acquire_s,
                        "populate_s": populate_s, "populate_disk_bytes": populate_io,
                        "total_setup_s": populate_s + acquire_s,
                        "native_register_s": backend.register_s, "register_calls": backend.register_calls,
                        "register_disk_bytes": register_io, "release_s": release_s,
                        "native_unregister_s": backend.unregister_s,
                        "discard_s": advice_times[pin_module._MADV_DONTNEED],
                        "warm_s": advice_times[pin_module._MADV_WILLNEED],
                        "before": before, "registered": registered, "released": _memory(path),
                        "stats": asdict(manager.stats), **transfers,
                    }
                    samples.append(sample)
                    print(json.dumps(sample), flush=True)
    finally:
        manager.clear()
        for tensor in tensors:
            storage = tensor.untyped_storage()
            pin_module._discard_and_warm(storage.data_ptr(), storage.nbytes(), warm=False)

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--exclude-prefix", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--populate-workers", type=int, default=0, help="experimental parallel private-page preparation",
    )
    parser.add_argument("--reset-mapping", action="store_true", help="reset read-populated mappings before preparation")
    parser.add_argument("--policies", nargs="+", choices=["discard", "retain_private"],
                        default=["discard", "retain_private"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repeats, args.workers) <= 0 or args.populate_workers < 0:
        parser.error("repeats and workers must be positive; populate-workers must be non-negative")
    if args.reset_mapping and not args.populate_workers:
        parser.error("--reset-mapping requires --populate-workers")
    path = args.path.expanduser().resolve()
    backend = _TimedBackend()
    manager = PinManager(None, backend=backend)
    target = torch.empty(64 * 1024**2, device="cuda", dtype=torch.uint8)
    with manager.acquire([torch.zeros(4096, dtype=torch.uint8)]):
        pass
    manager.clear()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with safe_open(path, framework="pt") as reader:
        names = reader.keys()
        tensors = [reader.get_tensor(name) for name in names if not name.startswith(tuple(args.exclude_prefix))]
    mapping_s = time.perf_counter() - started
    tensors = [tensor for tensor in tensors if tensor.numel()]
    total = sum(tensor.nbytes for tensor in tensors)
    report = {
        "path": str(path), "tensor_bytes": total, "tensors": len(tensors),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "mapping_s": mapping_s, "exclude_prefix": args.exclude_prefix, "samples": [],
        "populate_workers": args.populate_workers,
        "reset_mapping": args.reset_mapping,
    }
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}), flush=True)
    started = time.perf_counter()
    before_io = _disk_bytes()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(_read_pages, tensors))
    report["prefetch_s"] = time.perf_counter() - started
    report["prefetch_disk_bytes"] = _disk_bytes() - before_io
    print(json.dumps({"prefetch_s": report["prefetch_s"], "disk_bytes": report["prefetch_disk_bytes"]}), flush=True)

    report["pageable_transfer"] = _transfer(tensors, target)
    print(json.dumps({"pageable_transfer": report["pageable_transfer"]}), flush=True)
    report["samples"] = _run_cycles(
        tensors, target, path, manager, backend, args.repeats, args.policies, args.populate_workers, args.reset_mapping,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
