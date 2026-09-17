"""The token a session holds to keep host backings from being unpinned or evicted."""

import weakref
from collections.abc import Callable
from typing import Self

from ._host_backing import HostBacking


def _close_lease(backings: tuple[HostBacking, ...], on_close: Callable[[], None] | None) -> None:
    try:
        for backing in backings:
            backing.release()
    finally:
        if on_close is not None:
            on_close()


class HostLease:
    """Protect a batch of backings until close.

    Created by ``HostMemoryManager.acquire``. While open, the leased backings
    are neither unpinned nor evicted. Dropping the lease closes it. Close is
    idempotent, and a closed lease retains nothing.
    """

    def __init__(self, backings: tuple[HostBacking, ...], on_close: Callable[[], None] | None = None) -> None:
        self._backings = backings
        self._finalizer = weakref.finalize(self, _close_lease, backings, on_close)
        self._finalizer.atexit = False

    @property
    def backings(self) -> tuple[HostBacking, ...]:
        """The protected backings; unavailable once closed."""
        if self.closed:
            raise RuntimeError("Host lease is closed")
        return self._backings

    @property
    def pinned(self) -> bool:
        """Whether every leased backing is registered by its manager."""
        return all(backing.pinned for backing in self.backings)

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        try:
            self._finalizer()
        finally:
            self._backings = ()

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("Host lease is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["HostLease"]
