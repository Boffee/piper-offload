"""Contract: a deactivation that cannot finish releasing is reported, never hidden.

Each case activates on CUDA, runs a forward, then fails one release step once
while deactivating: the k-th CUDA synchronization or the k-th return of a
module to host storage, with an ordinary error or an interrupt. Afterwards
either every managed tensor is back on the host and the model activates again,
or ``ModelOffloader.failed`` is true and activation raises
:class:`ModelRuntimeFailedError`. It never ends inactive with managed tensors
still on the device. See the scheduling-hooks rules in ``docs/development.md``.
New block modes, components, and hook-driven adapter paths join ``_CONFIGS``.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch import nn

from piper_offload import (
    Adapter,
    BlockCompileConfig,
    BlockComponent,
    BlockMode,
    HostComponent,
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


def _fail_once(patch: pytest.MonkeyPatch, step: _Step, occurrence: int, error: type[BaseException]) -> None:
    """Make the ``occurrence``-th synchronization or host restore raise ``error`` once."""
    calls = 0

    def failing[**P, R](original: Callable[P, R]) -> Callable[P, R]:
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal calls
            calls += 1
            if calls == occurrence:
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


def _components(offloader: ModelOffloader) -> Iterator[HostComponent | BlockComponent]:
    return offloader._composite._components()


def _release_leftovers(offloader: ModelOffloader) -> None:
    """Finish what a failed deactivation kept, as recreating the CUDA worker would."""
    for component in _components(offloader):
        if isinstance(component, HostComponent):
            component.release()
            continue
        for runtime in (component._runtime, component._eager_runtime):
            if runtime.acquired:
                runtime.release()
        component._close_pin_lease()
        component._transfer.finish()
    if offloader._routed_lease is not None:
        offloader._routed_lease.close()
        offloader._routed_lease = None


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
        _fail_once(patch, step, occurrence, error)
        try:
            offloader.deactivate()
        except (RuntimeError, _Interrupt) as exc:
            raised = exc

    try:
        assert offloader.active_device is None
        if offloader.failed:
            assert raised is not None
            with pytest.raises(ModelRuntimeFailedError, match="Recreate the CUDA worker"):
                offloader.activate("cuda")
            assert offloader.active_device is None
        else:
            # A reusable offloader keeps nothing: no device tensors, no lease.
            assert _on_device(model) == []
            assert offloader._routed_lease is None
    finally:
        _release_leftovers(offloader)

    # Whatever was kept is releasable, after which the model works again.
    assert not offloader.failed
    assert _on_device(model) == []
    with activated_model(offloader, "cuda", adapters=adapters, adapter_mode="routed"), torch.inference_mode():
        actual = model(x).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
