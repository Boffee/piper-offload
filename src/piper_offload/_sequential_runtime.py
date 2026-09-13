"""Scoped PyTorch 2.14 distributed and compiler state for sequential ranks."""

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from itertools import count
from typing import Any

import torch.distributed as dist

# The C++ binding is available but absent from PyTorch's type stubs.
from torch._C._distributed_c10d import _set_thread_isolation_mode  # type: ignore[attr-defined]
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch.distributed.device_mesh import DeviceMesh

_session_ids = count()


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
        self.comms: list[Any] = []


@contextmanager
def isolate_runtime() -> Iterator[None]:
    """Restore process-wide state only after the executor has joined its workers."""
    c10d = dist.distributed_c10d
    previous = c10d._world, c10d._backend, c10d._default_pg_init_method, sys.excepthook
    # Thread IDs can be recycled while cached graphs retain earlier meshes.
    session_id = next(_session_ids)
    mesh_init = DeviceMesh.__init__
    autotune = CachingAutotuner.autotune_to_one_config
    tuning_lock = threading.Lock()

    @wraps(mesh_init)
    def init_rank_mesh(self: DeviceMesh, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        mesh_init(self, *args, **kwargs)
        if dist.is_initialized() and dist.get_backend() == "piper_sequential":
            # PyTorch populates this compiler/cache identity only for its
            # built-in "threaded" backend. Include our logical ranks too, so
            # implicit Replicate -> Shard slicing cannot reuse another rank's
            # compiled graph or DTensor redistribution planner.
            # Rank parity also prevents persistent caches from confusing ranks
            # when worker startup order differs between Python processes.
            self._thread_id = 2 * session_id + dist.get_rank()
            self._hash = None

    @wraps(autotune)
    def tune_once(self: CachingAutotuner, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        # Cached generated modules can share an autotuner across rank threads.
        # Precompilation has its own lock; choosing/pruning launchers does not.
        # Inductor calls this only when multiple candidate launchers remain.
        with tuning_lock:
            if len(self.launchers) > 1:
                autotune(self, *args, **kwargs)

    try:
        c10d._world = _ThreadWorld()  # type: ignore[assignment]
        _set_thread_isolation_mode(True)
        DeviceMesh.__init__ = init_rank_mesh
        CachingAutotuner.autotune_to_one_config = tune_once
        yield
    finally:
        CachingAutotuner.autotune_to_one_config = autotune
        DeviceMesh.__init__ = mesh_init
        _set_thread_isolation_mode(False)
        c10d._world, c10d._backend, c10d._default_pg_init_method, sys.excepthook = previous
