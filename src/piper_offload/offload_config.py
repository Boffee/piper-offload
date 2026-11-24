"""Configuration for model weight offloading and block scheduling."""

from dataclasses import dataclass

from .block_compile import BlockCompileConfig
from .block_mode import BlockMode

_VALID_BLOCK_MODES: frozenset[str] = frozenset(
    {"auto", "resident", "streaming", "rolling"}
)


def _validate_path_tuple(field_name: str, paths: tuple[str, ...]) -> None:
    if not isinstance(paths, tuple):
        raise TypeError(
            f"{field_name} must be a tuple of module paths, "
            f"got {type(paths).__name__}."
        )
    seen: set[str] = set()
    for entry in paths:
        if not isinstance(entry, str):
            raise TypeError(
                f"{field_name} entries must be str, got {type(entry).__name__}."
            )
        if entry in seen:
            raise ValueError(
                f"{field_name} must not contain duplicate path {entry!r}."
            )
        seen.add(entry)


@dataclass(frozen=True, kw_only=True, slots=True)
class OffloadConfig:
    """Topology and execution settings for binding host storage to a model.

    Parameters
    ----------
    block_paths:
        Dotted paths resolving to persistent ``nn.ModuleList`` block groups.
        Each group retains its CUDA working set for the whole activation.
    transient_block_paths:
        Dotted paths resolving to transient ``nn.ModuleList`` block groups that
        release their CUDA working set after their final block and reacquire
        between root forwards. Must be disjoint from ``block_paths`` and contain
        distinct block module objects.
    transient_paths:
        Dotted paths to non-block modules that release their CUDA working sets
        after their forwards finish and reacquire between root forwards.
    block_mode:
        CUDA execution mode for every declared block group: ``"streaming"``
        (default), ``"resident"``, ``"rolling"``, or ``"auto"``.
    block_compile:
        Optional Inductor compile configuration applied to every declared block
        group on CUDA and CPU activations.
    """

    block_paths: tuple[str, ...] = ()
    transient_block_paths: tuple[str, ...] = ()
    transient_paths: tuple[str, ...] = ()
    block_mode: BlockMode = "streaming"
    block_compile: BlockCompileConfig | None = None

    def __post_init__(self) -> None:
        _validate_path_tuple("block_paths", self.block_paths)
        _validate_path_tuple("transient_block_paths", self.transient_block_paths)
        _validate_path_tuple("transient_paths", self.transient_paths)
        overlap = set(self.block_paths) & set(self.transient_block_paths)
        if overlap:
            raise ValueError(
                "block_paths and transient_block_paths must be disjoint; "
                f"both contain {sorted(overlap)!r}."
            )
        if self.block_mode not in _VALID_BLOCK_MODES:
            raise ValueError(f"unsupported block mode: {self.block_mode!r}")
        if self.block_compile is not None and not isinstance(
            self.block_compile, BlockCompileConfig
        ):
            raise TypeError(
                "block_compile must be a BlockCompileConfig or None, "
                f"got {type(self.block_compile).__name__}."
            )
        if self.block_mode == "rolling" and not (
            self.block_compile is not None and self.block_compile.fullgraph
        ):
            raise ValueError(
                "block_mode='rolling' requires BlockCompileConfig(fullgraph=True)."
            )


def validate_offload_config(config: object) -> OffloadConfig:
    """Validate that ``config`` is an :class:`OffloadConfig` instance."""
    if not isinstance(config, OffloadConfig):
        raise TypeError(
            f"config must be an OffloadConfig, got {type(config).__name__}."
        )
    return config


def resolve_offload_config(config: OffloadConfig | None) -> OffloadConfig:
    """Return ``config`` when valid or a default :class:`OffloadConfig`."""
    if config is None:
        return OffloadConfig()
    if not isinstance(config, OffloadConfig):
        raise TypeError(
            f"config must be an OffloadConfig or None, got {type(config).__name__}."
        )
    return config


__all__ = ["OffloadConfig", "resolve_offload_config", "validate_offload_config"]
