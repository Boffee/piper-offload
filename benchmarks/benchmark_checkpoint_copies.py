"""Time the owned-copy pin path on a real checkpoint, phase by phase.

Mirrors the archived H3 diagnosis: the selected tensors of a transformer
checkpoint go through the real pin manager and the real CUDA backend. Phases:
a cold first pin (file evicted from the page cache before any tensor data
is touched),
a pinned transfer of every selected byte into a reused 64 MiB GPU buffer, a
reacquisition of registered storage, an eviction, a warm refill, and a warm pageable transfer
from the mapping. Fill and native registration time are split by
instrumenting the manager; the kernel's read counter shows which phases
read, and process memory shows the copies' memory arriving and leaving.

``--offered-gib`` adds the Windows offered tier to the same checkpoint, right
after the free-and-refill pair it is meant to replace, so the two are measured
against the same page cache: an eviction that offers the copies instead of
freeing them, and the reacquisition that takes them back. ``--pressure-gib``
then has a child process commit and touch that much memory, so Windows
discards the offered pages and the reacquisition after it measures the refill
that a discard costs.
"""

# ruff: noqa: T201 - benchmark CLI prints its report.

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Generator, Iterator
from pathlib import Path

import _process_memory as process_memory
import torch

import piper_offload.pin_manager as pin_module
from piper_offload import MappedCheckpoint, PinManager
from piper_offload._host_registration import RuntimeHostRegistration

CHUNK = 64 * 2**20
# What Piper Engine's H3 transformer loader leaves out.
EXCLUDED_PREFIXES = ("token_refiner.", "condition_proj.", "context_embedder.")

_PRESSURE = """
import ctypes, sys
target, held = int(sys.argv[1]), []
while sum(len(block) for block in held) < target:
    size = min(1 << 30, target - sum(len(block) for block in held))
    try:
        block = ctypes.create_string_buffer(size)
    except MemoryError:
        break
    ctypes.memset(block, 1, size)
    held.append(block)
print(sum(len(block) for block in held))
"""


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
        # Offers and reclaims run on pools, so their own elapsed times overlap; each stage is timed whole.
        self.reclaims = process_memory.Reclaims()
        self.fill_s = self.reclaim_s = self.offer_s = 0.0
        self.result: dict[str, object] = {}
        self._original_fill = pin_module._fill_copies
        self._original_reclaim = pin_module._reclaimed
        self._original_offer = pin_module._OfferBatch.take
        pin_module._fill_copies = self._timed_fill
        pin_module._reclaimed = self._timed_reclaim

        def timed_take(batch: pin_module._OfferBatch) -> tuple[list[object], list[str | None]]:
            """What an eviction still waits for once it has unregistered everything; the rest overlapped that."""
            start = time.perf_counter()
            try:
                return self._original_offer(batch)
            finally:
                self.offer_s += time.perf_counter() - start

        pin_module._OfferBatch.take = timed_take

    def _timed_fill(self, pending: list[pin_module._PendingCopy], **options: bool) -> None:
        start = time.perf_counter()
        self._original_fill(pending, **options)
        self.fill_s += time.perf_counter() - start

    def _timed_reclaim(self, pending: list[pin_module._PendingCopy]) -> Generator[pin_module._PendingCopy]:
        """The reclaim phase, including the registration of intact copies that runs beside it."""
        start = time.perf_counter()
        try:
            yield from self._original_reclaim(pending)
        finally:
            self.reclaim_s += time.perf_counter() - start

    def restore(self) -> None:
        pin_module._fill_copies = self._original_fill
        pin_module._reclaimed = self._original_reclaim
        pin_module._OfferBatch.take = self._original_offer
        self.reclaims.restore()

    def start(self) -> float:
        self.backend.register_s = self.backend.unregister_s = 0.0
        self.backend.register_calls = 0
        self.fill_s = self.reclaim_s = self.offer_s = 0.0
        self.reclaims.take()
        self._read_before = process_memory.read_bytes()
        return time.perf_counter()

    def stop(self, name: str, started: float) -> None:
        seconds = time.perf_counter() - started
        held = process_memory.memory()
        read = process_memory.read_bytes() - self._read_before
        backend, stats = self.backend, self.manager.stats
        intact, discarded = self.reclaims.take()
        self.result[name] = {
            "seconds": round(seconds, 3),
            "fill_s": round(self.fill_s, 3),
            "register_s": round(backend.register_s, 3),
            "unregister_s": round(backend.unregister_s, 3),
            "offer_s": round(self.offer_s, 3),
            "reclaim_s": round(self.reclaim_s, 3),
            "register_calls": backend.register_calls,
            "read_bytes": read,
            "resident_bytes": held.resident,
            "committed_bytes": held.committed,
            "shared_resident_bytes": held.shared,
            "pinned_bytes": stats.pinned_bytes,
            "offered_bytes": stats.offered_bytes,
            "intact_bytes": intact,
            "discarded_bytes": discarded,
        }
        print(
            f"{name:<22} {seconds:8.3f} s  fill {self.fill_s:7.3f}  register {backend.register_s:7.3f}  "
            f"unregister {backend.unregister_s:7.3f}  offer {self.offer_s:6.3f}  "
            f"reclaim {self.reclaim_s:6.3f}  read {read / 1e9:6.2f} GB  "
            f"resident {held.resident / 2**30:6.1f} GiB  commit {held.committed / 2**30:6.1f} GiB  "
            f"pinned {stats.pinned_bytes / 2**30:5.1f}  offered {stats.offered_bytes / 2**30:5.1f} GiB",
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


def _apply_pressure(gib: float) -> int:
    """Commit and touch ``gib`` GiB in a child process, so Windows discards what it can."""
    print(f"--- committing {gib:.1f} GiB in a child process", flush=True)
    finished = subprocess.run(
        [sys.executable, "-c", _PRESSURE, str(int(gib * 2**30))],
        check=True, capture_output=True, text=True,
    )
    touched = int(finished.stdout.strip() or 0)
    print(f"    it touched {touched / 2**30:.1f} GiB", flush=True)
    return touched


def _offered_phases(
    manager: PinManager,
    phases: _Phases,
    tensors: list[torch.Tensor],
    offered_bytes: int,
    pressure_gib: float,
) -> None:
    """Offer the copies instead of freeing them, and take them back, before and after memory pressure."""
    budget = manager.max_pinned_bytes
    manager.max_offered_bytes = offered_bytes

    def evict_into_the_tier(name: str) -> None:
        started = phases.start()
        manager.max_pinned_bytes = 0  # every idle copy is evicted, and so offered
        manager.max_pinned_bytes = budget
        phases.stop(name, started)

    def reacquire(name: str) -> None:
        started = phases.start()
        lease = manager.acquire(tensors)
        phases.stop(name, started)
        lease.close()

    evict_into_the_tier("evict_offer")
    reacquire("reacquire_intact")
    if pressure_gib > 0:
        evict_into_the_tier("evict_offer_again")
        phases.result["pressure_bytes"] = _apply_pressure(pressure_gib)
        reacquire("reacquire_after_pressure")
    manager.max_offered_bytes = 0
    manager.clear()


def _run(path: Path, device: torch.device, offered_gib: float, pressure_gib: float) -> dict[str, object]:
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
        commit_limit_bytes=limit,
        available_commit_bytes=available,
    )
    try:
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

        if offered_gib > 0:
            _offered_phases(manager, phases, tensors, int(offered_gib * 2**30), pressure_gib)
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
    parser.add_argument("--offered-gib", type=float, default=0.0, help="offered-copy budget; Windows only")
    parser.add_argument("--pressure-gib", type=float, default=0.0, help="memory a child commits to force a discard")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.empty(1, device=device)  # context creation stays out of the timings
    results = []
    for path in args.checkpoints:
        print(f"=== {path.name}", flush=True)
        results.append(_run(path, device, args.offered_gib, args.pressure_gib))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    _main()
