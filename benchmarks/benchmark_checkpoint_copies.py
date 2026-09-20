"""Time the owned-copy pin path on a real checkpoint, phase by phase.

Mirrors the archived H3 diagnosis: the selected tensors of the H3 transformer
checkpoint go through the real pin manager and the real CUDA backend. Phases:
a cold first pin (file evicted from the page cache before any tensor data
is touched),
a pinned transfer of every selected byte into a reused 64 MiB GPU buffer, a
cached reactivation, an eviction, a warm refill, and a warm pageable transfer
from the mapping. Fill and native registration time are split by
instrumenting the manager; the kernel's disk-read counter proves which phases
read from disk; and process RSS shows the copies' memory arriving and leaving.
"""

# ruff: noqa: T201 - benchmark CLI prints its report.

import argparse
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import torch

import piper_offload.pin_manager as pin_module
from piper_offload import MappedCheckpoint, PinManager
from piper_offload._host_registration import RuntimeHostRegistration

CHUNK = 64 * 2**20
# What Piper Engine's H3 transformer loader leaves out.
EXCLUDED_PREFIXES = ("token_refiner.", "condition_proj.", "context_embedder.")


def _rss() -> tuple[int, int]:
    """Process RSS and its shared part, which is where anonymous ``mmap`` regions are counted."""
    rss = shared = 0
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
            elif line.startswith("RssShmem:"):
                shared = int(line.split()[1]) * 1024
    return rss, shared


def _disk_read_bytes() -> int:
    with open("/proc/self/io", encoding="utf-8") as counters:
        for line in counters:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return 0


def _drop_from_cache(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _chunks(tensors: list[torch.Tensor]) -> Iterator[torch.Tensor]:
    for tensor in tensors:
        data = tensor.reshape(-1).view(torch.uint8)
        for start in range(0, data.numel(), CHUNK):
            yield data[start : start + CHUNK]


class _Timed(RuntimeHostRegistration):
    """The real backend, timing its native calls."""

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


class _Phases:
    """Times each phase and records the fill, registration, and RSS behind it."""

    def __init__(self, manager: PinManager, backend: _Timed) -> None:
        self.manager = manager
        self.backend = backend
        self.fill_s = 0.0
        self.result: dict[str, object] = {}
        self._original_fill = pin_module._fill_copies
        pin_module._fill_copies = self._timed_fill

    def _timed_fill(self, pending: list[pin_module._PendingCopy]) -> None:
        start = time.perf_counter()
        self._original_fill(pending)
        self.fill_s += time.perf_counter() - start

    def restore(self) -> None:
        pin_module._fill_copies = self._original_fill

    def start(self) -> float:
        self.backend.register_s = self.backend.unregister_s = 0.0
        self.backend.register_calls = 0
        self.fill_s = 0.0
        self._disk_before = _disk_read_bytes()
        return time.perf_counter()

    def stop(self, name: str, started: float) -> None:
        seconds = time.perf_counter() - started
        rss, shared = _rss()
        disk = _disk_read_bytes() - self._disk_before
        backend = self.backend
        self.result[name] = {
            "seconds": round(seconds, 3),
            "fill_s": round(self.fill_s, 3),
            "register_s": round(backend.register_s, 3),
            "unregister_s": round(backend.unregister_s, 3),
            "register_calls": backend.register_calls,
            "disk_read_bytes": disk,
            "rss_bytes": rss,
            "shared_rss_bytes": shared,
            "pinned_bytes": self.manager.stats.pinned_bytes,
        }
        print(
            f"{name:<20} {seconds:8.3f} s  fill {self.fill_s:7.3f}  register {backend.register_s:7.3f}  "
            f"unregister {backend.unregister_s:7.3f}  disk {disk / 1e9:6.2f} GB  rss {rss / 2**30:6.1f} GiB  "
            f"pinned {self.manager.stats.pinned_bytes / 2**30:6.1f} GiB",
            flush=True,
        )


def _transfer_all(manager: PinManager, tensors: list[torch.Tensor], device: torch.device) -> bool:
    """Copy every selected byte into a reused GPU buffer; check the first and last byte of each chunk."""
    buffer = torch.empty(CHUNK, dtype=torch.uint8, device=device)
    ok = True
    for chunk in _chunks(tensors):
        target = buffer[: chunk.numel()]
        manager.transfer(target, chunk, non_blocking=True)
        if chunk.numel() and (chunk[0].item() != target[0].item() or chunk[-1].item() != target[-1].item()):
            ok = False
    torch.cuda.synchronize(device)
    return ok


def _run(path: Path, device: torch.device) -> dict[str, object]:
    reader = MappedCheckpoint(path)
    keys = reader.keys()
    tensors = [reader.get_tensor(name) for name in keys if not name.startswith(EXCLUDED_PREFIXES)]
    backend = _Timed()
    manager = PinManager(backend=backend)
    phases = _Phases(manager, backend)
    phases.result.update(
        checkpoint=path.name,
        tensors=len(tensors),
        selected_bytes=sum(t.numel() * t.element_size() for t in tensors),
        budget_bytes=manager.max_pinned_bytes,
    )
    try:
        # No tensor data has been touched yet, so the drop actually evicts
        # it; a page already mapped into the process ignores the advice.
        _drop_from_cache(path)
        started = phases.start()
        lease = manager.acquire(tensors)
        phases.stop("first_pin_cold", started)
        phases.result["first_pin_pageable_bytes"] = lease.pageable_bytes

        started = phases.start()
        phases.result["pinned_transfer_ok"] = _transfer_all(manager, tensors, device)
        phases.stop("pinned_transfer", started)

        lease.close()
        started = phases.start()
        lease = manager.acquire(tensors)
        phases.stop("cached_reacquire", started)
        lease.close()

        started = phases.start()
        manager.clear()
        phases.stop("evict", started)

        # The cold fill left the file in the page cache.
        started = phases.start()
        lease = manager.acquire(tensors)
        phases.stop("refill_warm", started)
        lease.close()
        manager.clear()

        started = phases.start()
        with manager.acquire(tensors, pin=False):
            phases.result["pageable_transfer_ok"] = _transfer_all(manager, tensors, device)
        phases.stop("pageable_warm", started)
    finally:
        phases.restore()
    return phases.result


def _main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.empty(1, device=device)  # context creation stays out of the timings
    results = []
    for path in args.checkpoints:
        print(f"=== {path.name}", flush=True)
        results.append(_run(path, device))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    _main()
