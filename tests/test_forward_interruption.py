"""Contract: an interrupted forward never breaks the next one.

Scheduling hooks must act on installed state rather than assume the previous
hook ran; see the scheduling-hooks section of ``docs/development.md``. Each
configuration exercises one scheduler. A new offloaded residency configuration
joins ``_CONFIGS``; a new way for hooks to miss each other joins
``_INTERRUPTIONS``.
"""

import gc
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch import nn

from piper_offload import (
    BlockCompileConfig,
    DeviceTransferContext,
    ModelOffloader,
    OffloadConfig,
    VramBudget,
    host_pin_manager,
)
from tests.conftest import Interrupt, activated_model, residency_engine

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
    # Raise inside the resident head before its forward runs to exercise
    # resident failure before the next pass.
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
    offload_paths: tuple[str, ...] = ("blocks",)
    vram_budget_bytes: int | None = None
    compiled: bool = False
    use_vram_budget: bool = False
    use_transfer_context: bool = False

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
    pytest.param(_Config(("blocks",), None), marks=CUDA, id="streaming"),
    pytest.param(_Config(("blocks",), 0), marks=CUDA, id="zero-budget"),
    pytest.param(_Config(("embed", "blocks"), None), marks=CUDA, id="mixed-paths"),
    pytest.param(_Config(("blocks",), 2 * _UNIT_BYTES), marks=CUDA, id="partial-cache"),
    pytest.param(_Config(("embed", "blocks"), 1024 * 1024 * 1024), marks=CUDA, id="all-resident"),
    pytest.param(_Config(("blocks",), None, use_vram_budget=True), marks=CUDA, id="shared-budget"),
    pytest.param(
        _Config(("blocks",), None, use_transfer_context=True),
        marks=CUDA,
        id="shared-transfer-context",
    ),
    pytest.param(_Config(("blocks",), None, compiled=True), marks=CUDA, id="compiled"),
)


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

        # Registered after activation so it runs in the layer's pre-hook chain.
        handle = target.register_forward_pre_hook(fail)
        try:
            with pytest.raises(error, match="simulated interruption"):
                model(x)
        finally:
            handle.remove()


def _device_bytes() -> int:
    gc.collect()
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
    offloader = config.offloader(model)
    leases = host_pin_manager.stats.active_leases
    x = torch.randn(2, _WIDTH, device="cuda")
    budget = VramBudget(4 * _UNIT_BYTES) if config.use_vram_budget else None
    transfer_ctx = (
        DeviceTransferContext("cuda") if config.use_transfer_context else None
    )

    try:
        with (
            activated_model(
                offloader,
                "cuda",
                vram_budget=budget,
                transfer_context=transfer_ctx,
            ),
            torch.inference_mode(),
        ):
            expected = model(x).cpu()
            settled = _device_bytes()

            _interrupt(model, x, interruption)
            actual = model(x).cpu()

            # The forward after an interruption leaves the same working sets,
            # and nothing an interrupted call staged, on the device.
            assert _device_bytes() == settled

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert host_pin_manager.stats.active_leases == leases
        if budget is not None:
            assert budget.used_bytes == 0
        if transfer_ctx is not None:
            assert transfer_ctx.borrower_count == 0
    finally:
        if transfer_ctx is not None:
            transfer_ctx.close()


@CUDA
@pytest.mark.parametrize("error", [RuntimeError, Interrupt], ids=["error", "interrupt"])
def test_no_record_trace_recovers_from_interruption(error: type[BaseException]) -> None:
    torch.manual_seed(0)
    model = _Model()
    offloader = _Config(("blocks",), 0).offloader(model)
    x = torch.randn(2, _WIDTH, device="cuda")

    with activated_model(offloader, "cuda"), torch.inference_mode():
        engine = residency_engine(offloader)
        expected = model(x).cpu()
        assert engine._predictor is not None
        hist_before = list(engine._predictor._history)

        def fail_on_middle(_module: nn.Module, _args: tuple[object, ...]) -> None:
            raise error("draft interruption")

        handle = model.blocks[_MIDDLE].proj.register_forward_pre_hook(fail_on_middle)
        try:
            with pytest.raises(error, match="draft interruption"):
                with offloader.no_record_trace():
                    model(x)
        finally:
            handle.remove()

        assert engine.is_recording_trace
        assert list(engine._predictor._history) == hist_before
        actual = model(x).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
