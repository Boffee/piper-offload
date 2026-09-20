"""Refill under a known page-cache state, and rotation that exceeds the pin budget.

Completes what the phase benchmark leaves out. Its warm refill runs on whatever
the previous phase happened to leave cached, so it reads some disk; here the
file is pulled fully into the page cache first, then fully evicted, to bound the
refill from both sides. Rotation then gives two checkpoints one manager whose
budget holds only one, so admitting either must evict the other in LRU order.

Times are host pinning only: no model is built and nothing is denoised.
"""

# ruff: noqa: T201 - benchmark CLI prints its report.

import argparse
import json
import os
import time
from pathlib import Path

import torch

from piper_offload import MappedCheckpoint, PinManager

# What Piper Engine's H3 transformer loader leaves out.
EXCLUDED_PREFIXES = ("token_refiner.", "condition_proj.", "context_embedder.")
READ_CHUNK = 64 * 2**20


def _rss() -> int:
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def _disk_read_bytes() -> int:
    with open("/proc/self/io", encoding="utf-8") as counters:
        for line in counters:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    return 0


def _evict_from_cache(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _fill_cache(path: Path) -> None:
    """Read the whole file so a refill of it touches no disk."""
    with open(path, "rb", buffering=0) as source:
        while source.read(READ_CHUNK):
            pass


def _selected(path: Path) -> list[torch.Tensor]:
    reader = MappedCheckpoint(path)
    keys = reader.keys()
    return [reader.get_tensor(name) for name in keys if not name.startswith(EXCLUDED_PREFIXES)]


def _acquire(manager: PinManager, tensors: list[torch.Tensor], label: str, out: dict) -> None:
    disk_before = _disk_read_bytes()
    started = time.perf_counter()
    lease = manager.acquire(tensors)
    seconds = time.perf_counter() - started
    disk = _disk_read_bytes() - disk_before
    stats = manager.stats
    lease.close()
    out[label] = {
        "seconds": round(seconds, 3),
        "disk_read_bytes": disk,
        "pinned_bytes": stats.pinned_bytes,
        "copy_bytes": stats.copy_bytes,
        "rss_bytes": _rss(),
    }
    print(
        f"{label:<26} {seconds:8.3f} s  disk {disk / 1e9:6.2f} GB  "
        f"pinned {stats.pinned_bytes / 2**30:6.1f} GiB  copies {stats.copy_bytes / 2**30:6.1f} GiB  "
        f"rss {_rss() / 2**30:6.1f} GiB",
        flush=True,
    )


def _refill(path: Path, out: dict) -> None:
    tensors = _selected(path)
    manager = PinManager()
    _fill_cache(path)
    _acquire(manager, tensors, "refill_fully_warm", out)
    manager.clear()
    _evict_from_cache(path)
    _acquire(manager, tensors, "refill_cold", out)
    manager.clear()


def _rotate(first: Path, second: Path, budget: int, rounds: int, out: dict) -> None:
    """One budget that holds a single checkpoint, so each switch evicts the other."""
    selected = {"A": _selected(first), "B": _selected(second)}
    manager = PinManager(budget)
    out["rotation_budget_bytes"] = budget
    _fill_cache(first)
    _fill_cache(second)
    for index in range(rounds):
        for name, tensors in selected.items():
            _acquire(manager, tensors, f"rotate_{name}_round{index}", out)
    manager.clear()


def _main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("checkpoints", nargs=2, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget-gib", type=float, default=40.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.empty(1, device=torch.device(args.device))  # context creation stays out of the timings
    out: dict[str, object] = {}
    print("=== refill under a known page-cache state", flush=True)
    _refill(args.checkpoints[0], out)
    print("=== rotation over a budget that holds one checkpoint", flush=True)
    _rotate(args.checkpoints[0], args.checkpoints[1], int(args.budget_gib * 2**30), args.rounds, out)
    if args.output:
        args.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    _main()
