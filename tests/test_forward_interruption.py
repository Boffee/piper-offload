"""Contract: an interrupted forward never breaks the next one.

Scheduling hooks must act on installed state rather than assume the previous
hook ran; see the scheduling-hooks section of ``docs/development.md``. Each
configuration exercises one scheduler. A new offloaded residency configuration or
hook-driven adapter path joins ``_CONFIGS``; a new way for hooks to miss each
other joins ``_INTERRUPTIONS``.
"""

import gc
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch import nn

from piper_offload import (
    Adapter,
    BlockCompileConfig,
    ModelOffloader,
    OffloadConfig,
    host_pin_manager,
)
from tests.conftest import Interrupt, activated_model

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_WIDTH = 8
_NUM_BLOCKS = 3
_MIDDLE = 1

type _Interruption = Literal[
    "error_after_block",
    "error_in_layer",
    "interrupt_in_layer",
    "error_in_unit",
    "interrupt_in_unit",
    "skipped_block",
    "direct_call",
]
_INTERRUPTIONS: tuple[_Interruption, ...] = (
    # Leaves the residency engine mid-traversal.
    "error_after_block",
    # Raise inside the resident head after its routed pre-hook staged.
    # ``always_call`` post-hooks run for an ``Exception`` but not for an interrupt.
    "error_in_layer",
    "interrupt_in_layer",
    # Raise inside an offloaded unit's forward so unit hooks unwind via
    # ``always_call`` (Exception) or must recover cleanly on next use (Interrupt).
    "error_in_unit",
    "interrupt_in_unit",
    "skipped_block",
    # Run the final offloaded unit directly outside the root forward.
    "direct_call",
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(_WIDTH, _WIDTH, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(x))


class _Model(nn.Module):
    """Embedding, blocks, and head; the forward can skip a block or raise after one."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(_WIDTH, _WIDTH, bias=False)
        self.blocks = nn.ModuleList(_Block() for _ in range(_NUM_BLOCKS))
        self.head = nn.Linear(_WIDTH, _WIDTH, bias=False)
        self.requires_grad_(False)
        self.skip: int | None = None
        self.fail_after_middle = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for idx, block in enumerate(self.blocks):
            if idx == self.skip:
                continue
            x = block(x)
            if idx == _MIDDLE and self.fail_after_middle:
                raise RuntimeError("simulated interruption")
        return self.head(x)


@dataclass(frozen=True)
class _Config:
    device: str
    offload_paths: tuple[str, ...] = ("blocks",)
    vram_budget_bytes: int | None = None
    compiled: bool = False
    routed: bool = False

    def offloader(self, model: nn.Module) -> ModelOffloader:
        return ModelOffloader.from_module(
            model,
            OffloadConfig(
                offload_paths=self.offload_paths,
                vram_budget_bytes=self.vram_budget_bytes,
                block_compile=(
                    BlockCompileConfig(dynamic=False, fullgraph=True)
                    if self.compiled
                    else None
                ),
            ),
        )


_UNIT_BYTES = _WIDTH * _WIDTH * 4

_CONFIGS = (
    pytest.param(_Config("cuda", ("blocks",), None), marks=CUDA, id="streaming"),
    pytest.param(_Config("cuda", ("blocks",), 0), marks=CUDA, id="zero-budget"),
    pytest.param(_Config("cuda", ("embed", "blocks"), None), marks=CUDA, id="mixed-paths"),
    pytest.param(_Config("cuda", ("blocks",), 2 * _UNIT_BYTES), marks=CUDA, id="partial-cache"),
    pytest.param(_Config("cuda", ("embed", "blocks"), 1024 * 1024 * 1024), marks=CUDA, id="all-resident"),
    pytest.param(_Config("cuda", ("blocks",), None, compiled=True), marks=CUDA, id="compiled"),
    pytest.param(_Config("cuda", ("blocks",), None, routed=True), marks=CUDA, id="streaming-routed"),
    # CPU activation schedules no CUDA units, but routed hooks still stage factors.
    pytest.param(_Config("cpu", ("blocks",), None, routed=True), id="cpu-routed"),
)


def _lora() -> Adapter:
    state: dict[str, torch.Tensor] = {}
    for name in (*(f"blocks.{idx}.proj" for idx in range(_NUM_BLOCKS)), "head"):
        state[f"{name}.lora_A.weight"] = torch.randn(2, _WIDTH)
        state[f"{name}.lora_B.weight"] = torch.randn(_WIDTH, 2)
    return Adapter.from_state_dict(state)


def _interrupt(model: _Model, x: torch.Tensor, interruption: _Interruption) -> None:
    if interruption == "skipped_block":
        model.skip = _MIDDLE
        try:
            model(x)
        finally:
            model.skip = None
    elif interruption == "direct_call":
        model.blocks[-1](model.embed(x))
    elif interruption == "error_after_block":
        model.fail_after_middle = True
        try:
            with pytest.raises(RuntimeError, match="simulated interruption"):
                model(x)
        finally:
            model.fail_after_middle = False
    else:
        error = (
            Interrupt
            if interruption in ("interrupt_in_layer", "interrupt_in_unit")
            else RuntimeError
        )
        target: nn.Module
        if interruption in ("error_in_unit", "interrupt_in_unit"):
            block = model.blocks[_MIDDLE]
            target = block if "forward" in block.__dict__ else block.proj
        else:
            target = model.head

        def fail(_module: nn.Module, _args: tuple[object, ...]) -> None:
            raise error("simulated interruption")

        # Registered after activation, so it runs after any routed pre-hook.
        handle = target.register_forward_pre_hook(fail)
        try:
            with pytest.raises(error, match="simulated interruption"):
                model(x)
        finally:
            handle.remove()


def _device_bytes(device: str) -> int:
    gc.collect()
    if device != "cuda":
        return 0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


@pytest.mark.parametrize("interruption", _INTERRUPTIONS)
@pytest.mark.parametrize("config", _CONFIGS)
def test_next_forward_recovers_from_interruption(
    request: pytest.FixtureRequest,
    config: _Config,
    interruption: _Interruption,
) -> None:
    if config.compiled:
        # Keep compiled variants below Dynamo's frame limit.
        torch.compiler.reset()
        request.addfinalizer(torch.compiler.reset)
    torch.manual_seed(0)
    model = _Model()
    adapters = [_lora()] if config.routed else []
    offloader = config.offloader(model)
    leases = host_pin_manager.stats.active_leases
    x = torch.randn(2, _WIDTH, device=config.device)

    with (
        activated_model(offloader, config.device, adapters=adapters, adapter_mode="routed"),
        torch.inference_mode(),
    ):
        expected = model(x).cpu()
        settled = _device_bytes(config.device)

        _interrupt(model, x, interruption)
        actual = model(x).cpu()

        # The forward after an interruption leaves the same working sets,
        # and nothing an interrupted call staged, on the device.
        assert _device_bytes(config.device) == settled

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert host_pin_manager.stats.active_leases == leases
