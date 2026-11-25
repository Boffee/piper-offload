"""Reusable asynchronous CUDA host-device copy stream and worker context."""

import enum
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Self

import torch

from ._devices import canonical_device


class _State(enum.Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


def _shutdown_executor(executor: ThreadPoolExecutor) -> None:
    """Shut down ``executor`` and wait for its worker thread even if interrupted."""
    interrupt: BaseException | None = None
    while True:
        try:
            executor.shutdown(wait=True)
            break
        except BaseException as exc:
            if interrupt is None:
                interrupt = exc
    if interrupt is not None:
        raise interrupt


class DeviceTransferContext:
    """Shared per-device CUDA copy stream and optional background staging executor.

    A transfer context may be shared across multiple offloaded models or KV/MoE
    pools targeting the same CUDA device:

    - Callers needing only asynchronous CUDA copies can call :meth:`ensure_stream`
      without spawning any background worker thread.
    - Callers needing background host-staging threads (such as
      :class:`ResidencyEngine`) call :meth:`ensure` or :meth:`submit`, which
      lazily start a single worker thread on demand.
    - Borrowed contexts track active borrowers and refuse :meth:`close` while a
      borrower session remains active; once closed, a context cannot be
      reopened.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        stream_priority: int = -1,
    ) -> None:
        canonical = canonical_device(device)
        if canonical.type != "cuda":
            raise ValueError(
                f"DeviceTransferContext requires a CUDA device; got {canonical}."
            )
        if isinstance(stream_priority, bool) or not isinstance(stream_priority, int):
            raise ValueError(f"stream_priority must be an int; got {stream_priority!r}.")
        self._device = canonical
        self._stream_priority = stream_priority
        self._copy_stream: torch.cuda.Stream | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._borrowers = 0
        self._state = _State.OPEN
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()

    @property
    def device(self) -> torch.device:
        """Return the canonical CUDA device bound to this transfer context."""
        return self._device

    @property
    def copy_stream(self) -> torch.cuda.Stream | None:
        """Return the lazily created CUDA copy stream, or ``None`` if uninitialized/closed."""
        with self._lock:
            return self._copy_stream

    @property
    def executor(self) -> ThreadPoolExecutor | None:
        """Return the lazily created single-worker executor, or ``None`` if uninitialized/closed."""
        with self._lock:
            return self._executor

    @property
    def is_open(self) -> bool:
        """Return whether the context currently holds an open stream or executor."""
        with self._lock:
            return self._copy_stream is not None or self._executor is not None

    @property
    def is_closed(self) -> bool:
        """Return whether :meth:`close` has permanently closed this context."""
        with self._lock:
            return self._state is _State.CLOSED

    @property
    def borrower_count(self) -> int:
        """Return the number of active sessions borrowing this context."""
        with self._lock:
            return self._borrowers

    def _require_open_unlocked(self) -> None:
        if self._state is not _State.OPEN:
            raise RuntimeError("DeviceTransferContext is closed.")

    def _ensure_stream_unlocked(self) -> torch.cuda.Stream:
        self._require_open_unlocked()
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(
                device=self._device,
                priority=self._stream_priority,
            )
        return self._copy_stream

    def _ensure_unlocked(self) -> tuple[torch.cuda.Stream, ThreadPoolExecutor]:
        stream = self._ensure_stream_unlocked()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="piper-offload-transfer",
            )
        return stream, self._executor

    def ensure_stream(self) -> torch.cuda.Stream:
        """Lazily allocate and return the CUDA copy stream without spawning a thread."""
        with self._lock:
            return self._ensure_stream_unlocked()

    def ensure(self) -> tuple[torch.cuda.Stream, ThreadPoolExecutor]:
        """Lazily allocate both the copy stream and single-worker executor."""
        with self._lock:
            return self._ensure_unlocked()

    def submit[**P, T](
        self,
        fn: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[T]:
        """Submit ``fn(*args, **kwargs)`` on the single-worker transfer executor."""
        with self._lock:
            _, executor = self._ensure_unlocked()
            return executor.submit(fn, *args, **kwargs)

    def _acquire_borrow(self) -> None:
        with self._lock:
            if self._state is not _State.OPEN:
                raise RuntimeError("Cannot borrow a closed DeviceTransferContext.")
            self._borrowers += 1

    def _release_borrow(self) -> None:
        with self._lock:
            if self._borrowers > 0:
                self._borrowers -= 1

    def _shutdown_executor(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            _shutdown_executor(executor)

    def _synchronize_stream(self) -> None:
        with self._lock:
            stream = self._copy_stream
        if stream is not None:
            stream.synchronize()
            with self._lock:
                if self._copy_stream is stream:
                    self._copy_stream = None

    def close(self) -> None:
        """Join pending executor work, synchronize the copy stream, and close permanently.

        Idempotent once closed, never holds the internal state lock across
        blocking joins or CUDA synchronization (so worker tasks may inspect
        properties such as :attr:`copy_stream` safely), and refuses closing
        while active borrowers hold references to this context. Retry-safe on
        interruption: once teardown begins, new borrows, stream creation, and
        executor submissions are rejected immediately, while if stream
        synchronization fails, the copy stream remains open and
        :attr:`is_closed` remains ``False`` until a retry succeeds.
        """
        with self._close_lock:
            with self._lock:
                if self._state is _State.CLOSED:
                    return
                if self._borrowers > 0:
                    raise RuntimeError(
                        "Cannot close DeviceTransferContext while active borrowers hold it."
                    )
                self._state = _State.CLOSING
                executor, self._executor = self._executor, None
                stream = self._copy_stream
            if executor is not None:
                _shutdown_executor(executor)
            if stream is not None:
                stream.synchronize()
            with self._lock:
                if self._copy_stream is stream:
                    self._copy_stream = None
                self._state = _State.CLOSED

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["DeviceTransferContext"]
