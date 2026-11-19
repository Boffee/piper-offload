"""Shared-memory signaling and rank agreement for same-machine relays.

Replaces Gloo control traffic on TSO hosts (x86_64/AMD64) with one unpinned
control mmap accessed through a memory-safe ``int64`` memoryview: a global abort
word plus cache-line-separated monotonic counters and double-buffered
collective metadata per rank. Waiters spin briefly, cooperatively yield the CPU
for multi-chunk transfers, then fall back to short bounded sleeps.
"""

from __future__ import annotations

import mmap
import os
import platform
import time
from collections.abc import Callable, Sequence
from datetime import timedelta

_WORD_BYTES = 8
_HEADER_WORDS = 8
_RANK_WORDS = 40

_SPIN_SECONDS = 150e-6
_SPIN_BATCH = 64
_YIELD_SECONDS = 4e-3
_YIELD_BATCH = 16
_SLEEP_MIN = 50e-6
_SLEEP_MAX = 500e-6

_SCHED_YIELD: Callable[[], None] | None = getattr(os, "sched_yield", None)


def _yield_cpu() -> None:
    if _SCHED_YIELD is not None:
        _SCHED_YIELD()
    else:
        time.sleep(0)


def shared_control_supported() -> bool:
    """Return whether aligned int64 memoryview loads/stores provide hardware TSO ordering."""
    return platform.machine().lower() in ("x86_64", "amd64")


def control_bytes(world_size: int) -> int:
    """Return page-rounded shared control size with cache-line-isolated rank state."""
    size = (_HEADER_WORDS + world_size * _RANK_WORDS) * _WORD_BYTES
    return max(4096, (size + 4095) // 4096 * 4096)


class SharedControl:
    """Shared-memory metadata agreement and chunk ready/free signaling across ranks.

    The ``memoryview`` retains the underlying ``mmap`` for its lifetime; callers
    drop the ``SharedControl`` instance rather than explicitly closing ``mapping``.
    """

    def __init__(self, mapping: mmap.mmap, rank: int, world_size: int) -> None:
        required = control_bytes(world_size)
        if len(mapping) < required:
            raise ValueError("shared relay control mapping is smaller than required")
        self.rank = rank
        self.world_size = world_size
        self._words = memoryview(mapping).cast("q")
        self._other_ranks = tuple(r for r in range(world_size) if r != rank)
        self._meta_idx = tuple(_HEADER_WORDS + r * _RANK_WORDS for r in range(world_size))
        self._ready_idx = tuple(_HEADER_WORDS + r * _RANK_WORDS + 8 for r in range(world_size))
        self._free_idx = tuple(_HEADER_WORDS + r * _RANK_WORDS + 16 for r in range(world_size))
        self._slot_idx = tuple(
            (_HEADER_WORDS + r * _RANK_WORDS + 24, _HEADER_WORDS + r * _RANK_WORDS + 32)
            for r in range(world_size)
        )
        self._epoch = 0
        self._local_ready = 0
        self._local_free = 0

    def _poll(self, indices: Sequence[int], pending: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        words = self._words
        if words[0]:
            raise RuntimeError("shared relay aborted by peer")
        remaining = [(peer, target) for peer, target in pending if words[indices[peer]] < target]
        if not remaining and words[0]:
            raise RuntimeError("shared relay aborted by peer")
        return remaining

    def _poll_batch(
        self,
        indices: Sequence[int],
        pending: list[tuple[int, int]],
        rounds: int,
        *,
        yield_cpu: bool = False,
    ) -> list[tuple[int, int]]:
        for _ in range(rounds):
            if yield_cpu:
                _yield_cpu()
            pending = self._poll(indices, pending)
            if not pending:
                return pending
        return pending

    def _wait(self, indices: Sequence[int], targets: Sequence[tuple[int, int]], timeout: timedelta) -> None:
        pending = self._poll(indices, targets)
        if not pending:
            return
        now = time.monotonic()
        deadline = now + max(0.0, timeout.total_seconds())
        spin_until = min(deadline, now + _SPIN_SECONDS)
        yield_until = min(deadline, now + _YIELD_SECONDS)
        sleep_s = _SLEEP_MIN
        while pending:
            if now < spin_until:
                pending = self._poll_batch(indices, pending, _SPIN_BATCH)
            elif now < yield_until:
                pending = self._poll_batch(indices, pending, _YIELD_BATCH, yield_cpu=True)
            else:
                remaining = deadline - now
                pending = self._poll(indices, pending)
                if not pending:
                    return
                if remaining <= 0.0:
                    raise RuntimeError("shared relay timed out waiting for peer")
                time.sleep(min(sleep_s, remaining))
                sleep_s = min(_SLEEP_MAX, sleep_s * 2.0)
            now = time.monotonic()

    def agree(
        self, op: int, nbytes: int, kind: int, detail: int, timeout: timedelta,
    ) -> tuple[list[int], list[int]]:
        """Exchange collective metadata and snapshot every rank's ready/free counters."""
        words = self._words
        self._epoch += 1
        slot_base = self._slot_idx[self.rank][self._epoch & 1]
        words[slot_base] = op
        words[slot_base + 1] = nbytes
        words[slot_base + 2] = kind
        words[slot_base + 3] = detail
        words[slot_base + 4] = self._local_ready
        words[slot_base + 5] = self._local_free
        words[self._meta_idx[self.rank]] = self._epoch

        if self._other_ranks:
            self._wait(self._meta_idx, [(peer, self._epoch) for peer in self._other_ranks], timeout)

        ready_bases = [0] * self.world_size
        free_bases = [0] * self.world_size
        ready_bases[self.rank] = self._local_ready
        free_bases[self.rank] = self._local_free
        expected = (op, nbytes, kind, detail)
        mismatched = False
        for peer in self._other_ranks:
            peer_slot = self._slot_idx[peer][self._epoch & 1]
            if (words[peer_slot], words[peer_slot + 1], words[peer_slot + 2], words[peer_slot + 3]) != expected:
                mismatched = True
            ready_bases[peer] = int(words[peer_slot + 4])
            free_bases[peer] = int(words[peer_slot + 5])
        if mismatched:
            raise ValueError("shared relay collective operation, payload bytes and root/dtype must match across ranks")
        return ready_bases, free_bases

    def publish_ready(self) -> None:
        self._local_ready += 1
        self._words[self._ready_idx[self.rank]] = self._local_ready

    def publish_free(self) -> None:
        self._local_free += 1
        self._words[self._free_idx[self.rank]] = self._local_free

    def wait_ready(self, targets: Sequence[tuple[int, int]], timeout: timedelta) -> None:
        if targets:
            self._wait(self._ready_idx, targets, timeout)

    def wait_free(self, targets: Sequence[tuple[int, int]], timeout: timedelta) -> None:
        if targets:
            self._wait(self._free_idx, targets, timeout)

    def abort(self) -> None:
        """Signal failure across the shared control region so waiting peers exit quickly."""
        self._words[0] = 1
