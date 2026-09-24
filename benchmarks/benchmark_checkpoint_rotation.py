"""Refill under a known page-cache state, and rotation that exceeds the pin budget.

Completes what the phase benchmark leaves out. Its warm refill runs on whatever
the previous phase happened to leave cached, so it reads some disk; here the
file is pulled fully into the page cache first, then fully evicted, to bound the
refill from both sides. Rotation then gives two checkpoints one manager whose
budget holds only one, so admitting either must evict the other in LRU order.

``--offered-gib`` repeats the rotation with the Windows offered tier, which
turns each switch's eviction into an offer and each admission into a reclaim.
Both rotations run in the same process against the same page-cache state, so
their times, working sets, and commitment are comparable; ``--cold`` evicts
both files from the page cache first, which is when a reclaim has the most to
save. The tier's rotation also reports how much Windows kept intact.

Times are host pinning only: no model is built and nothing is denoised.
"""

# ruff: noqa: T201 - benchmark CLI prints its report.

import argparse
import json
import time
from pathlib import Path

import _process_memory as process_memory
import torch

from piper_offload import MappedCheckpoint, PinManager

# What Piper Engine's H3 transformer loader leaves out.
EXCLUDED_PREFIXES = ("token_refiner.", "condition_proj.", "context_embedder.")


def _selected(path: Path) -> list[torch.Tensor]:
    reader = MappedCheckpoint(path)
    keys = reader.keys()
    return [reader.get_tensor(name) for name in keys if not name.startswith(EXCLUDED_PREFIXES)]


def _acquire(
    manager: PinManager, tensors: list[torch.Tensor], label: str, out: dict, reclaims: process_memory.Reclaims,
) -> None:
    read_before = process_memory.read_bytes()
    started = time.perf_counter()
    lease = manager.acquire(tensors)
    seconds = time.perf_counter() - started
    read = process_memory.read_bytes() - read_before
    stats = manager.stats
    held = process_memory.memory()
    lease.close()
    intact, discarded = reclaims.take()
    out[label] = {
        "seconds": round(seconds, 3),
        "read_bytes": read,
        "pinned_bytes": stats.pinned_bytes,
        "copy_bytes": stats.copy_bytes,
        "offered_bytes": stats.offered_bytes,
        "resident_bytes": held.resident,
        "committed_bytes": held.committed,
        "intact_bytes": intact,
        "discarded_bytes": discarded,
    }
    print(
        f"{label:<26} {seconds:8.3f} s  read {read / 1e9:6.2f} GB  "
        f"pinned {stats.pinned_bytes / 2**30:5.1f}  copies {stats.copy_bytes / 2**30:5.1f}  "
        f"offered {stats.offered_bytes / 2**30:5.1f} GiB  resident {held.resident / 2**30:6.1f}  "
        f"commit {held.committed / 2**30:6.1f} GiB  intact {intact / 2**30:5.1f} GiB",
        flush=True,
    )


def _refill(path: Path, out: dict, reclaims: process_memory.Reclaims) -> None:
    tensors = _selected(path)
    manager = PinManager()
    process_memory.fill_cache(path)
    _acquire(manager, tensors, "refill_fully_warm", out, reclaims)
    manager.clear()
    process_memory.drop_from_cache(path)
    _acquire(manager, tensors, "refill_cold", out, reclaims)
    manager.clear()


def _rotate(
    first: Path,
    second: Path,
    budget: int,
    offered: int,
    rounds: int,
    cold: bool,
    out: dict,
    reclaims: process_memory.Reclaims,
) -> None:
    """One budget that holds a single checkpoint, so each switch evicts or offers the other."""
    # The page-cache state is set before the files are mapped: Windows keeps
    # the pages a mapping covers.
    for path in (first, second):
        if cold:
            process_memory.drop_from_cache(path)
        else:
            process_memory.fill_cache(path)
    selected = {"A": _selected(first), "B": _selected(second)}
    manager = PinManager(budget, max_offered_bytes=offered)
    suffix = "offered" if offered else "freed"
    out[f"rotation_{suffix}_budget_bytes"] = budget
    out[f"rotation_{suffix}_offered_budget_bytes"] = offered
    for index in range(rounds):
        for name, tensors in selected.items():
            _acquire(manager, tensors, f"rotate_{suffix}_{name}_round{index}", out, reclaims)
    manager.max_offered_bytes = 0
    manager.clear()


def _main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("checkpoints", nargs=2, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget-gib", type=float, default=40.0)
    parser.add_argument("--offered-gib", type=float, default=0.0, help="offered-copy budget; Windows only")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--cold", action="store_true", help="evict both checkpoints from the page cache first")
    parser.add_argument("--skip-refill", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.empty(1, device=torch.device(args.device))  # context creation stays out of the timings
    out: dict[str, object] = {}
    limit, available = process_memory.system_commit()
    out.update(commit_limit_bytes=limit, available_commit_bytes=available)
    reclaims = process_memory.Reclaims()
    budget = int(args.budget_gib * 2**30)
    try:
        if not args.skip_refill:
            print("=== refill under a known page-cache state", flush=True)
            _refill(args.checkpoints[0], out, reclaims)
        print("=== rotation over a budget that holds one checkpoint, freeing on eviction", flush=True)
        _rotate(args.checkpoints[0], args.checkpoints[1], budget, 0, args.rounds, args.cold, out, reclaims)
        if args.offered_gib > 0:
            print("=== the same rotation, offering evicted copies instead", flush=True)
            offered = int(args.offered_gib * 2**30)
            _rotate(args.checkpoints[0], args.checkpoints[1], budget, offered, args.rounds, args.cold, out, reclaims)
    finally:
        reclaims.restore()
    if args.output:
        args.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    _main()
