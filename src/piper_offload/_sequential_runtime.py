"""Scoped PyTorch 2.15 distributed and compiler state for in-process ranks."""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from collections.abc import Sequence
from datetime import timedelta
from functools import wraps
from itertools import count
from typing import Any, cast

import torch.distributed as dist

# The C++ binding is available but absent from PyTorch's type stubs.
from torch._C._distributed_c10d import _set_thread_isolation_mode  # type: ignore[attr-defined]
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch.distributed.device_mesh import DeviceMesh

_MAX_STORE_WAIT_SECONDS = 3600.0

# Process-wide lock shared by SequentialExecutor and ParallelExecutor because
# c10d._world and _set_thread_isolation_mode are global state.
session_lock = threading.Lock()
_session_owner_lock = threading.Lock()
_session_owner: object | None = None
_session_ids = count()


def acquire_session(owner: object) -> bool:
    """Claim ``session_lock`` non-blocking for ``owner``."""
    global _session_owner  # noqa: PLW0603 -- process-wide single-session ownership
    with _session_owner_lock:
        if _session_owner is not None:
            return False
        if not session_lock.acquire(blocking=False):
            return False
        try:
            _session_owner = owner
        except BaseException:
            with contextlib.suppress(RuntimeError):
                session_lock.release()
            raise
        return True


def owns_session(owner: object) -> bool:
    """Return True while ``owner`` holds the process-wide session."""
    with _session_owner_lock:
        return _session_owner is owner


def release_session(owner: object) -> None:
    """Release ``session_lock`` if ``owner`` owns it; safe to retry after SIGINT."""
    global _session_owner  # noqa: PLW0603 -- process-wide single-session ownership
    with _session_owner_lock:
        if _session_owner is not owner:
            return
        if session_lock.locked():
            session_lock.release()
        _session_owner = None


class StartupGate:
    """Per-session worker admission and cancellation barrier before c10d access."""

    __slots__ = ("_admitted", "_cancelled", "_cv", "_exited")

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._cancelled = False
        self._admitted = 0
        self._exited = 0

    def try_admit(self) -> bool:
        with self._cv:
            if self._cancelled:
                return False
            self._admitted += 1
            return True

    def mark_exited(self) -> None:
        with self._cv:
            self._exited += 1
            self._cv.notify_all()

    def cancel(self) -> None:
        with self._cv:
            self._cancelled = True
            self._cv.notify_all()

    def wait_exited(self, deadline: float) -> bool:
        with self._cv:
            while self._exited < self._admitted:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._cv.wait(timeout=min(remaining, 3600.0))
            return True


class _ThreadWorld(threading.local):
    """The c10d World extension point with rank-local registration state."""

    def __init__(self) -> None:
        self.default_pg: dist.ProcessGroup | None = None
        self.group_count = 0
        self.pg_map: dict[Any, Any] = {}
        self.pg_names: dict[Any, Any] = {}
        self.pg_group_ranks: dict[Any, Any] = {}
        self.pg_backend_config: dict[Any, Any] = {}
        self.tags_to_pg: dict[Any, Any] = {}
        self.pg_to_tag: dict[Any, Any] = {}
        self.pg_coalesce_state: dict[Any, Any] = {}
        self.pg_flight_recorder_hooks: dict[Any, Any] = {}
        self.comms: list[Any] = []


def _wrap_mesh_init(mesh_init: Any, backend: str, session_id: int) -> Any:  # noqa: ANN401
    @wraps(mesh_init)
    def init_rank_mesh(self: DeviceMesh, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        mesh_init(self, *args, **kwargs)
        if dist.is_initialized() and dist.get_backend() == backend:
            # PyTorch populates this compiler/cache identity only for its
            # built-in "threaded" backend. Include our logical ranks too, so
            # implicit Replicate -> Shard slicing cannot reuse another rank's
            # compiled graph or DTensor redistribution planner.
            # Packing (session_id, global_rank) into negative IDs <= -2 avoids
            # collisions across arbitrary session histories and world sizes
            # while preserving rank parity for persistent disk caches.
            self._thread_id = -2 - ((session_id << 32) | dist.get_rank())
            self._hash = None

    return init_rank_mesh


def _wrap_autotune(autotune: Any, tuning_lock: threading.Lock) -> Any:  # noqa: ANN401
    @wraps(autotune)
    def tune_once(self: CachingAutotuner, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        # Cached generated modules can share an autotuner across rank threads.
        # Precompilation has its own lock; choosing/pruning launchers does not.
        # Inductor calls this only when multiple candidate launchers remain.
        with tuning_lock:
            if len(self.launchers) > 1:
                autotune(self, *args, **kwargs)

    return tune_once


class _RuntimeIsolation:
    """Retryable process-wide distributed and compiler isolation scope."""

    def __init__(self, backend: str = "piper_sequential") -> None:
        c10d = dist.distributed_c10d
        self._backend = backend
        self._previous = (
            c10d._world,
            c10d._backend,
            c10d._default_pg_init_method,
            sys.excepthook,
        )
        self._session_id = next(_session_ids)
        self._mesh_init = DeviceMesh.__init__
        self._autotune = CachingAutotuner.autotune_to_one_config
        self._tuning_lock = threading.Lock()
        self._entered = False
        self._restored = False

    def __enter__(self) -> None:
        self.enter()

    def enter(self) -> None:
        c10d = dist.distributed_c10d
        init_rank_mesh = _wrap_mesh_init(self._mesh_init, self._backend, self._session_id)
        tune_once = _wrap_autotune(self._autotune, self._tuning_lock)
        self._entered = True
        self._restored = False
        try:
            c10d._world = _ThreadWorld()  # type: ignore[assignment]
            _set_thread_isolation_mode(True)
            DeviceMesh.__init__ = init_rank_mesh
            CachingAutotuner.autotune_to_one_config = tune_once
        except BaseException:
            self.restore()
            raise

    def restore(self) -> None:
        """Restore captured globals idempotently; stays retryable if interrupted."""
        if not self._entered or self._restored:
            return
        c10d = dist.distributed_c10d
        CachingAutotuner.autotune_to_one_config = self._autotune
        DeviceMesh.__init__ = self._mesh_init
        _set_thread_isolation_mode(False)
        c10d._world, c10d._backend, c10d._default_pg_init_method, sys.excepthook = self._previous
        self._restored = True

    def __exit__(self, *_exc: object) -> None:
        self.restore()


def isolate_runtime(backend: str = "piper_sequential") -> _RuntimeIsolation:
    """Build a retryable isolation scope capturing current globals before mutation."""
    return _RuntimeIsolation(backend)


class _InProcessStore(dist.Store):
    """Cross-platform in-memory ``dist.Store`` for in-process rank threads.

    PyTorch omits ``HashStore`` from Windows builds; implementing ``dist.Store``
    directly keeps Linux and Windows on one shared coordination path.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cv = threading.Condition()
        self._data: dict[str, bytes] = {}

    @staticmethod
    def _to_bytes(value: str | bytes) -> bytes:
        return value.encode("utf-8") if isinstance(value, str) else bytes(value)

    def _timeout_seconds(self) -> float:
        return cast("timedelta", cast("Any", self).timeout).total_seconds()

    def _wait_all_locked(self, keys: Sequence[str], timeout_seconds: float, missing_msg: str) -> None:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while any(key not in self._data for key in keys):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise dist.DistStoreError(missing_msg)  # type: ignore[attr-defined]
            self._cv.wait(timeout=min(remaining, _MAX_STORE_WAIT_SECONDS))

    def set(self, key: str, value: str | bytes) -> None:  # type: ignore[override]
        raw = self._to_bytes(value)
        with self._cv:
            self._data[key] = raw
            self._cv.notify_all()

    def get(self, key: str) -> bytes:
        with self._cv:
            self._wait_all_locked(
                (key,),
                self._timeout_seconds(),
                f"Timeout waiting for key {key!r} in in-process store",
            )
            return self._data[key]

    def add(self, key: str, value: int) -> int:
        with self._cv:
            raw = self._data.get(key)
            current = 0 if raw is None else int(raw)
            updated = current + int(value)
            self._data[key] = str(updated).encode("ascii")
            self._cv.notify_all()
            return updated

    def wait(self, keys: Sequence[str], timeout: timedelta | None = None) -> None:  # type: ignore[override]
        dur = timeout.total_seconds() if isinstance(timeout, timedelta) else self._timeout_seconds()
        with self._cv:
            self._wait_all_locked(
                keys,
                dur,
                f"Timeout waiting for keys {list(keys)!r} in in-process store",
            )

    def check(self, keys: Sequence[str]) -> bool:  # type: ignore[override]
        with self._cv:
            return all(key in self._data for key in keys)

    def compare_set(self, key: str, expected: str | bytes, desired: str | bytes) -> bytes:  # type: ignore[override]
        exp = self._to_bytes(expected)
        des = self._to_bytes(desired)
        with self._cv:
            current = self._data.get(key)
            if current is None:
                if exp != b"":
                    return exp
                self._data[key] = des
                self._cv.notify_all()
                return des
            if current == exp:
                self._data[key] = des
                self._cv.notify_all()
                return des
            return current

    def deleteKey(self, key: str) -> bool:  # noqa: N802 -- C++ PyStore virtual trampoline name
        with self._cv:
            return self._data.pop(key, None) is not None

    delete_key = deleteKey

    def getNumKeys(self) -> int:  # noqa: N802 -- C++ PyStore virtual trampoline name
        with self._cv:
            return len(self._data)

    num_keys = getNumKeys

    def append(self, key: str, value: str | bytes) -> None:  # type: ignore[override]
        raw = self._to_bytes(value)
        with self._cv:
            self._data[key] = self._data.get(key, b"") + raw
            self._cv.notify_all()

    def multi_get(self, keys: Sequence[str]) -> list[bytes]:  # type: ignore[override]
        with self._cv:
            self._wait_all_locked(
                keys,
                self._timeout_seconds(),
                f"Timeout waiting for keys {list(keys)!r} in in-process store",
            )
            return [self._data[key] for key in keys]

    def multi_set(self, keys: Sequence[str], values: Sequence[str | bytes]) -> None:  # type: ignore[override]
        if len(keys) != len(values):
            raise ValueError("keys and values must have the same length")
        with self._cv:
            for key, value in zip(keys, values, strict=True):
                self._data[key] = self._to_bytes(value)
            self._cv.notify_all()

    def clone(self) -> _InProcessStore:
        return self


def make_in_process_store() -> dist.Store:
    """Create a cross-platform in-memory ``dist.Store`` for in-process rank workers."""
    return _InProcessStore()
