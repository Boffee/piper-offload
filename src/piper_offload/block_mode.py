"""Block residency strategy."""

from typing import Literal

type BlockMode = Literal["resident", "streaming", "rolling"]

__all__ = ["BlockMode"]
