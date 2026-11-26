"""Time the Tier 2a/2b owned-copy pin path on a real checkpoint, phase by phase.

Mirrors the archived H3 diagnosis: the selected tensors of a transformer
checkpoint go through the real pin manager and the real CUDA backend. Phases:
a cold first pin (file evicted from the page cache before any tensor data
is touched),
a pinned transfer of every selected byte into a reused 64 MiB GPU buffer, a
reacquisition of registered storage, an eviction, a warm refill, and a warm
pageable transfer from the mapping. Direct file fill and native registration
time are split by instrumenting the manager and backend; the kernel's read
counter shows which phases read, and process memory shows Tier 2a/2b slab
memory arriving and leaving.

``--host-gib`` adds Tier 2a <-> Tier 2b demotion/promotion phases right after
the free-and-refill pair so both run against the same page cache: lowering
``max_pinned_bytes`` to zero demotes idle copies into Tier 2b private slabs,
and reacquiring them promotes the copies back into Tier 2a in RAM without
reading disk.
"""

# ruff: noqa: T201 - benchmark CLI prints its report.

import argparse
import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import _process_memory as process_memory
import torch

from piper_offload import MappedCheckpoint, PinManager
from piper_offload._direct_io import DirectReadRequest
from piper_offload._host_registration import RuntimeHostRegistration

CHUNK = 64 * 2**20
# What Piper Engine's H3 transformer loader leaves out.
EXCLUDED_PREFIXES = ("token_refiner.", "condition_proj.", "context_embedder.")


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
    """Times each phase and records the fill, registration, and memory behind it."""

    def __init__(self, manager: PinManager, backend: _Timed) -> None:
        self.manager = manager
        self.backend = backend
        self.fill_s = 0.0
        self.result: dict[str, object] = {}
        self._original_read_into = manager._reader.read_into
        manager._reader.read_into = self._timed_read_into

    def _timed_read_into(self, requests: Sequence[DirectReadRequest]) -> None:
        start = time.perf_counter()
        try:
            self._original_read_into(requests)
        finally:
            self.fill_s += time.perf_counter() - start

    def restore(self) -> None:
        self.manager._reader.read_into = self._original_read_into

    def start(self) -> float:
        self.backend.register_s = self.backend.unregister_s = 0.0
        self.backend.register_calls = 0
        self.fill_s = 0.0
        self._read_before = process_memory.read_bytes()
        return time.perf_counter()

    def stop(self, name: str, started: float) -> None:
        seconds = time.perf_counter() - started
        held = process_memory.memory()
        read = process_memory.read_bytes() - self._read_before
        backend, stats = self.backend, self.manager.stats
        self.result[name] = {
            "seconds": round(seconds, 3),
            "fill_s": round(self.fill_s, 3),
            "register_s": round(backend.register_s, 3),
            "unregister_s": round(backend.unregister_s, 3),
            "register_calls": backend.register_calls,
            "read_bytes": read,
            "resident_bytes": held.resident,
            "committed_bytes": held.committed,
            "shared_resident_bytes": held.shared,
            "pinned_bytes": stats.pinned_bytes,
            "copy_bytes": stats.copy_bytes,
            "private_copy_bytes": stats.private_copy_bytes,
            "bounce_bytes": stats.bounce_bytes,
        }
        print(
            f"{name:<22} {seconds:8.3f} s  fill {self.fill_s:7.3f}  register {backend.register_s:7.3f}  "
            f"unregister {backend.unregister_s:7.3f}  read {read / 1e9:6.2f} GB  "
            f"resident {held.resident / 2**30:6.1f} GiB  commit {held.committed / 2**30:6.1f} GiB  "
            f"pinned {stats.pinned_bytes / 2**30:5.1f}  copies {stats.copy_bytes / 2**30:5.1f}  "
            f"private {stats.private_copy_bytes / 2**30:5.1f}  bounce {stats.bounce_bytes / 2**30:5.2f} GiB",
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


def _host_overflow_phases(
    manager: PinManager,
    phases: _Phases,
    tensors: list[torch.Tensor],
    host_bytes: int,
) -> None:
    """Demote Tier 2a copies into Tier 2b on pin pressure and promote them back in RAM."""
    budget = manager.max_pinned_bytes
    orig_host = manager.max_host_bytes
    manager.max_host_bytes = host_bytes
    try:
        started = phases.start()
        manager.max_pinned_bytes = 0
        phases.stop("evict_demote", started)

        manager.max_pinned_bytes = budget
        started = phases.start()
        with manager.acquire(tensors):
            phases.stop("promote_ram", started)
    finally:
        manager.max_host_bytes = 0
        manager.clear()
        manager.max_pinned_bytes = budget
        manager.max_host_bytes = orig_host


def _run(path: Path, device: torch.device, host_gib: float) -> dict[str, object]:
    # Before the file is mapped: Windows keeps the pages a mapping covers.
    process_memory.drop_from_cache(path)
    reader = MappedCheckpoint(path)
    keys = reader.keys()
    tensors = [reader.get_tensor(name) for name in keys if not name.startswith(EXCLUDED_PREFIXES)]
    backend = _Timed()
    manager = PinManager(backend=backend)
    phases = _Phases(manager, backend)
    limit, available = process_memory.system_commit()
    phases.result.update(
        checkpoint=path.name,
        tensors=len(tensors),
        selected_bytes=sum(t.numel() * t.element_size() for t in tensors),
        budget_bytes=manager.max_pinned_bytes,
        host_budget_bytes=manager.max_host_bytes,
        commit_limit_bytes=limit,
        available_commit_bytes=available,
    )
    try:
        started = phases.start()
        with manager.acquire(tensors) as lease:
            phases.stop("first_pin_cold", started)
            phases.result["first_pin_pageable_bytes"] = lease.pageable_bytes

            started = phases.start()
            phases.result["pinned_transfer_ok"] = _transfer_all(manager, tensors, device)
            phases.stop("pinned_transfer", started)

        started = phases.start()
        with manager.acquire(tensors):
            phases.stop("cached_reacquire", started)

        started = phases.start()
        manager.clear()
        phases.stop("evict", started)

        # Direct I/O bypasses the page cache on direct-capable filesystems, while buffered fallback warms it.
        started = phases.start()
        with manager.acquire(tensors):
            phases.stop("refill_warm", started)

        if host_gib > 0:
            _host_overflow_phases(manager, phases, tensors, int(host_gib * 2**30))
        else:
            manager.clear()

        started = phases.start()
        with manager.acquire(tensors, pin=False):
            phases.result["pageable_transfer_ok"] = _transfer_all(manager, tensors, device)
        phases.stop("pageable_warm", started)
    finally:
        phases.restore()
        manager.clear()
    return phases.result


def _main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--host-gib",
        type=float,
        default=0.0,
        help="combined Tier 2a + Tier 2b host ceiling in GiB to benchmark Tier 2b demotion/promotion",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.empty(1, device=device)  # context creation stays out of the timings
    results = []
    for path in args.checkpoints:
        print(f"=== {path.name}", flush=True)
        results.append(_run(path, device, args.host_gib))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    _main()
