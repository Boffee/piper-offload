"""Contract: a deactivation that cannot finish releasing is reported and retried.

Each case activates on CUDA, runs a forward, then fails one release step once
while deactivating: the k-th CUDA synchronization or the k-th return of a
module to host storage, with an ordinary error or an interrupt. Afterwards
either every managed tensor is back on the host, or ``ModelOffloader.failed``
is true; it never ends inactive and unreported with managed tensors still on
the device. The next activation retries what was kept and runs the model as
before. A release that keeps failing makes activation raise
:class:`ModelRuntimeFailedError`. See the scheduling-hooks rules in
``docs/development.md``. New block modes, components, and hook-driven adapter
paths join ``_CONFIGS``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch import nn

from piper_offload import (
    Adapter,
    BlockCompileConfig,
    BlockMode,
    ModelOffloader,
    ModelRuntimeFailedError,
)
from piper_offload.host_module import HostModuleInstance
from tests.conftest import activated_model

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_WIDTH = 8
_NUM_BLOCKS = 3

type _Step = Literal["synchronize", "restore"]


class _Interrupt(BaseException):
    """Like ``KeyboardInterrupt``: not an ``Exception``."""


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(_WIDTH, _WIDTH, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(x))


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(_WIDTH, _WIDTH, bias=False)
        self.blocks = nn.ModuleList(_Block() for _ in range(_NUM_BLOCKS))
        self.head = nn.Linear(_WIDTH, _WIDTH, bias=False)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


@dataclass(frozen=True)
class _Config:
    block_mode: BlockMode | None
    transient: bool = False
    routed: bool = False

    def offloader(self, model: nn.Module) -> ModelOffloader:
        if self.block_mode is None:
            return ModelOffloader.from_module(model)
        return ModelOffloader.from_module(
            model,
            block_paths=() if self.transient else ("blocks",),
            transient_block_paths=("blocks",) if self.transient else (),
            transient_paths=("embed",) if self.transient else (),
            block_mode=self.block_mode,
            block_compile=(
                BlockCompileConfig(dynamic=False, fullgraph=True)
                if self.block_mode == "rolling"
                else None
            ),
        )

    def adapters(self) -> list[Adapter]:
        if not self.routed:
            return []
        state: dict[str, torch.Tensor] = {}
        for idx in range(_NUM_BLOCKS):
            state[f"blocks.{idx}.proj.lora_A.weight"] = torch.randn(1, _WIDTH)
            state[f"blocks.{idx}.proj.lora_B.weight"] = torch.randn(_WIDTH, 1)
        return [Adapter.from_state_dict(state)]


_CONFIGS = (
    pytest.param(_Config(None), id="host-only"),
    pytest.param(_Config("streaming"), id="streaming"),
    pytest.param(_Config("resident"), id="resident"),
    pytest.param(_Config("rolling"), id="rolling"),
    pytest.param(_Config("streaming", transient=True), id="streaming-transient"),
    pytest.param(_Config("streaming", routed=True), id="streaming-routed"),
)


def _fail(
    patch: pytest.MonkeyPatch,
    step: _Step,
    error: type[BaseException],
    *,
    occurrence: int,
    persistent: bool = False,
) -> None:
    """Make the ``occurrence``-th synchronization or host restore raise ``error``.

    With ``persistent``, every later one raises too.
    """
    calls = 0

    def failing[**P, R](original: Callable[P, R]) -> Callable[P, R]:
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal calls
            calls += 1
            if calls == occurrence or (persistent and calls > occurrence):
                raise error("injected release failure")
            return original(*args, **kwargs)

        return call

    if step == "synchronize":
        patch.setattr(torch.cuda, "synchronize", failing(torch.cuda.synchronize))
        patch.setattr(torch.cuda.Stream, "synchronize", failing(torch.cuda.Stream.synchronize))
        patch.setattr(torch.cuda.Event, "synchronize", failing(torch.cuda.Event.synchronize))
    else:
        patch.setattr(HostModuleInstance, "install_host", failing(HostModuleInstance.install_host))


def _on_device(model: nn.Module) -> list[str]:
    tensors = [*model.named_parameters(remove_duplicate=False), *model.named_buffers(remove_duplicate=False)]
    return [name for name, tensor in tensors if tensor.device.type == "cuda"]


@CUDA
@pytest.mark.parametrize("error", [RuntimeError, _Interrupt], ids=["error", "interrupt"])
@pytest.mark.parametrize("occurrence", [1, 2])
@pytest.mark.parametrize("step", ["synchronize", "restore"])
@pytest.mark.parametrize("config", _CONFIGS)
def test_failed_release_step_is_reported(
    request: pytest.FixtureRequest,
    config: _Config,
    step: _Step,
    occurrence: int,
    error: type[BaseException],
) -> None:
    if config.block_mode == "rolling":
        # Keep compiled variants below Dynamo's frame limit.
        torch.compiler.reset()
        request.addfinalizer(torch.compiler.reset)
    torch.manual_seed(0)
    model = _Model()
    adapters = config.adapters()
    offloader = config.offloader(model)
    x = torch.randn(2, _WIDTH, device="cuda")

    offloader.activate("cuda", adapters=adapters, adapter_mode="routed")
    with torch.inference_mode():
        expected = model(x).cpu()
    raised: BaseException | None = None
    with pytest.MonkeyPatch.context() as patch:
        _fail(patch, step, error, occurrence=occurrence)
        try:
            offloader.deactivate()
        except (RuntimeError, _Interrupt) as exc:
            raised = exc

    assert offloader.active_device is None
    if offloader.failed:
        assert raised is not None
    else:
        # Nothing was kept: no device tensors, no routed lease.
        assert _on_device(model) == []
        assert offloader._routed_lease is None

    # The next activation retries whatever was kept, then runs as before.
    with activated_model(offloader, "cuda", adapters=adapters, adapter_mode="routed"), torch.inference_mode():
        actual = model(x).cpu()
    assert not offloader.failed
    assert _on_device(model) == []
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
def test_activation_reports_a_release_that_keeps_failing() -> None:
    model = _Model()
    offloader = _Config("streaming").offloader(model)
    offloader.activate("cuda")
    with torch.inference_mode():
        model(torch.randn(2, _WIDTH, device="cuda"))

    with pytest.MonkeyPatch.context() as patch:
        _fail(patch, "synchronize", RuntimeError, occurrence=1, persistent=True)
        with pytest.raises(RuntimeError, match="injected release failure"):
            offloader.deactivate()
        assert offloader.failed
        with pytest.raises(ModelRuntimeFailedError, match="Recreate the CUDA worker") as caught:
            offloader.activate("cuda")
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert offloader.active_device is None
        assert offloader.failed

    # Once synchronization works again, a retry releases everything.
    offloader.deactivate()
    assert not offloader.failed
    assert _on_device(model) == []
