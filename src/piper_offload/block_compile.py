"""Opt-in ``torch.compile`` policy for streamed model blocks."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Self, cast

import torch
from torch import nn

type _BlockForward = Callable[..., object]
type CompileBackend = str | Callable[..., object]
type _TorchCompileOption = str | int | bool | Callable[..., object]
_NO_INSTANCE_FORWARD = object()


@dataclass(frozen=True, kw_only=True, slots=True)
class BlockCompileConfig:
    """Compilation options applied to every declared streamed block group.

    Compilation is disabled unless an instance of this configuration is
    supplied to :meth:`piper_offload.ModelOffloader.from_module` (directly or
    through :class:`piper_offload.ModelSpec`). The initial API intentionally
    fixes the backend to Inductor's default mode and exposes only the two
    graph-capture controls needed by supported inference workloads, plus an
    optional mapping for target-specific compiler extensions.

    Attributes
    ----------
    dynamic:
        Forwarded to :func:`torch.compile`. ``True`` asks Dynamo to attempt
        dynamic kernels up front, which is the default here because streamed
        diffusion and transformer blocks commonly see changing sequence or
        spatial shapes. ``None`` selects PyTorch's adaptive behavior.
    fullgraph:
        Forwarded to :func:`torch.compile`. The default permits graph breaks;
        ``True`` requires the complete block forward to form one graph.
    options:
        Optional Inductor options forwarded to :func:`torch.compile`. The
        mapping is copied for every compiled block so compiler setup cannot
        mutate shared configuration state.
    rolling:
        Experimental inference-only single-target streaming. Inductor inserts
        a target-refill operation after each supported parameter's final graph
        use, allowing the next block to reuse that storage while the current
        block continues computing. Supported adapters are regular dense,
        TorchAO-family, Quanto, GGUF, and Piper ConvRot INT8. Requires
        ``fullgraph=True``.
    """

    dynamic: bool | None = True
    fullgraph: bool = False
    options: Mapping[str, object] | None = None
    rolling: bool = False

    def __post_init__(self) -> None:
        if self.dynamic is not None and not isinstance(self.dynamic, bool):
            raise TypeError(f"BlockCompileConfig.dynamic must be bool or None; got {type(self.dynamic).__name__}.")
        if not isinstance(self.fullgraph, bool):
            raise TypeError(f"BlockCompileConfig.fullgraph must be bool; got {type(self.fullgraph).__name__}.")
        if self.options is not None and not isinstance(self.options, Mapping):
            raise TypeError(f"BlockCompileConfig.options must be a mapping or None; got {type(self.options).__name__}.")
        if not isinstance(self.rolling, bool):
            raise TypeError(f"BlockCompileConfig.rolling must be bool; got {type(self.rolling).__name__}.")
        if self.rolling and not self.fullgraph:
            raise ValueError("BlockCompileConfig(rolling=True) requires fullgraph=True.")


@dataclass(slots=True)
class _CompiledBlockForward:
    """One block's compiled forward and exact original attribute state."""

    module: nn.Module
    compiled: _BlockForward
    original_instance_forward: object

    def install(self) -> None:
        self.module.__dict__["forward"] = self.compiled

    def restore(self) -> None:
        if self.original_instance_forward is _NO_INSTANCE_FORWARD:
            self.module.__dict__.pop("forward", None)
        else:
            self.module.__dict__["forward"] = self.original_instance_forward


@dataclass(slots=True)
class _BlockCompileState:
    """Cached compiled block forwards and their activation-scoped state."""

    config: BlockCompileConfig | None
    _forwards: tuple[_CompiledBlockForward, ...]
    _installed: bool = False

    @classmethod
    def create(
        cls,
        blocks: Sequence[nn.Module],
        config: BlockCompileConfig | None,
        *,
        backend: CompileBackend,
    ) -> Self:
        if config is None:
            return cls(config=None, _forwards=())

        forwards: list[_CompiledBlockForward] = []
        seen: set[int] = set()
        for block in blocks:
            block_id = id(block)
            if block_id in seen:
                continue
            seen.add(block_id)
            original_instance_forward = block.__dict__.get(
                "forward",
                _NO_INSTANCE_FORWARD,
            )
            compile_options = None if config.options is None else dict(config.options)
            torch_compile_options = cast(
                dict[str, _TorchCompileOption] | None,
                compile_options,
            )
            compiled = cast(
                _BlockForward,
                torch.compile(
                    block.forward,
                    backend=backend,
                    dynamic=config.dynamic,
                    fullgraph=config.fullgraph,
                    options=torch_compile_options,
                ),
            )
            forwards.append(
                _CompiledBlockForward(
                    module=block,
                    compiled=compiled,
                    original_instance_forward=original_instance_forward,
                )
            )
        return cls(config=config, _forwards=tuple(forwards))

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self, enabled: bool) -> None:
        if not enabled or not self._forwards:
            return
        if self._installed:
            raise RuntimeError("compiled block forwards are already installed")

        installed: list[_CompiledBlockForward] = []
        try:
            for forward in self._forwards:
                forward.install()
                installed.append(forward)
        except BaseException:
            for forward in reversed(installed):
                forward.restore()
            raise
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        for forward in reversed(self._forwards):
            forward.restore()
        self._installed = False


__all__ = ["BlockCompileConfig"]
