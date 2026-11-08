"""Where a component's device working set stands between acquire and release."""

import enum


class WorkingSet(enum.Enum):
    """State of one component's device working set.

    ``CHANGING`` means an ``acquire()`` or ``release()`` started and has not
    finished, so the next call releases first. The scheduling-hooks rules in
    ``docs/development.md`` own this pattern.
    """

    RELEASED = "released"
    ACQUIRED = "acquired"
    CHANGING = "changing"


__all__ = ["WorkingSet"]
