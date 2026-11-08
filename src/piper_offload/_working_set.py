"""Where a component's device working set stands between acquire and release."""

import enum


class WorkingSet(enum.Enum):
    """State of one component's device working set.

    ``acquire()`` and ``release()`` mark ``CHANGING`` before their first step
    and commit ``ACQUIRED`` or ``RELEASED`` only after their last, so an
    exception anywhere, including an interrupt, leaves ``CHANGING``. Any call
    that finds ``CHANGING`` releases first: release steps are idempotent, so
    that finishes an interrupted release and rolls back an interrupted
    acquire. See the scheduling-hooks rules in ``docs/development.md``.
    """

    RELEASED = "released"
    ACQUIRED = "acquired"
    CHANGING = "changing"


__all__ = ["WorkingSet"]
