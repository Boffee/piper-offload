"""Linux host memory: physical and cgroup limits, anonymous mappings, and positional reads."""

import contextlib
import functools
import io
import mmap
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

_PROC_CGROUP = "/proc/self/cgroup"
_CGROUP_MOUNT = "/sys/fs/cgroup"


def _cgroup_memory_limit() -> int | None:
    """The tightest memory limit on the process's cgroup or any ancestor, v2 or v1; None if unlimited.

    A container or a systemd service can be far below the host's RAM, and its
    limit lives at the process's own cgroup path, not at the hierarchy root.
    The hierarchy is assumed to be mounted at the standard ``/sys/fs/cgroup``;
    set ``max_pinned_bytes`` explicitly on a host that mounts it elsewhere.
    """
    try:
        with open(_PROC_CGROUP, encoding="ascii") as membership:
            lines = membership.read().splitlines()
    except OSError:
        return None
    limits: list[int] = []
    for line in lines:
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0":
            base, name = Path(_CGROUP_MOUNT), "memory.max"
        elif "memory" in controllers.split(","):
            base, name = Path(_CGROUP_MOUNT, "memory"), "memory.limit_in_bytes"
        else:
            continue
        directory = base.joinpath(*path.strip("/").split("/")) if path.strip("/") else base
        while True:
            try:
                text = (directory / name).read_text(encoding="ascii").strip()
            except OSError:
                text = ""
            if text.isdigit():
                limits.append(int(text))
            if directory == base or base not in directory.parents:
                break
            directory = directory.parent
    return min(limits, default=None)


def available_memory() -> int:
    """Physical RAM, bounded by the process's cgroup when it has a tighter limit."""
    # POSIX-only os calls are bound here so Windows tests can inject them;
    # their callable types are fixed at this platform boundary.
    sysconf = cast(Callable[[str], int], getattr(os, "sysconf"))  # noqa: B009 - platform-specific os attribute
    total = sysconf("SC_PHYS_PAGES") * sysconf("SC_PAGE_SIZE")
    if total <= 0:
        raise ValueError("physical RAM is unavailable")
    limit = _cgroup_memory_limit()
    return total if limit is None else min(total, limit)


def new_region(size: int) -> mmap.mmap:
    """An owned page-aligned anonymous mapping, freed on eviction."""
    return mmap.mmap(-1, size)


def _preadv(fd: int, buffer: memoryview, offset: int) -> int:
    # preadv is absent on Windows, where the selector uses per-worker handles.
    preadv = cast(Callable[[int, list[memoryview], int], int], getattr(os, "preadv"))  # noqa: B009
    return preadv(fd, [buffer], offset)


class Readers(contextlib.AbstractContextManager["Readers"]):
    """Positional reads share the checkpoint's descriptor without moving its file position."""

    def read_at(self, file: io.BufferedReader) -> Callable[[memoryview, int], int]:
        return functools.partial(_preadv, file.fileno())

    def __exit__(self, *_exc: object) -> None:
        pass


__all__ = ["Readers", "available_memory", "new_region"]
