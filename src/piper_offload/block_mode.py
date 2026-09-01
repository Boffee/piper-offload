"""Block residency strategy."""

from typing import Literal

type BlockMode = Literal["auto", "resident", "streaming", "rolling"]

__all__ = ["BlockMode"]
