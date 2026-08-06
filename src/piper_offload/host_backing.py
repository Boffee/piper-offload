"""Construction-time host backing policy."""

from typing import Literal, cast

type HostBacking = Literal["pinned", "adopt"]

def validate_host_backing(value: str) -> HostBacking:
    """Return a validated host-backing policy.

    Validation happens before store construction because pinned construction
    may repoint model parameters as each host clone is created and is therefore
    intentionally not rollback-safe after it starts.
    """
    if value not in ("pinned", "adopt"):
        raise ValueError(
            "host_backing must be 'pinned' or 'adopt'; "
            f"got {value!r}."
        )
    return cast("HostBacking", value)


__all__ = ["HostBacking"]
