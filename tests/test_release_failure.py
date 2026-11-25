"""Contract: a deactivation that cannot finish releasing is reported and retried.

Each case activates on CUDA, runs a forward, then fails one release step once
while deactivating: the k-th CUDA synchronization or the k-th return of a
module to host storage, with an ordinary error or an interrupt. Each is a
release step, so ``ModelOffloader.failed`` is true exactly when one failed,
and otherwise every managed tensor is back on the host. The next activation
retries what was kept and runs the model as before. A release that keeps failing makes activation raise
:class:`ModelRuntimeFailedError`. See the scheduling-hooks rules in
``docs/development.md``. New block modes, components, and hook-driven adapter
paths join ``_CONFIGS``.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch import nn

from piper_offload import (
    Adapter,
    BlockCompileConfig,
    DeviceTransferContext,
    ModelOffloader,
    ModelRuntimeFailedError,
    OffloadConfig,
    VramBudget,
)
from piper_offload.block_compile import _BlockCompileState
from piper_offload.host_module import HostModuleInstance, HostModuleLoadPlan
from tests.conftest import Interrupt, activated_model, residency_engine

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_WIDTH = 8
_NUM_BLOCKS = 3
_UNIT_BYTES = _WIDTH * _WIDTH * 4

type _Step = Literal["synchronize", "restore"]


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
    offload_paths: tuple[str, ...]
    vram_budget_bytes: int | None = None
    compiled: bool = False
    routed: bool = False
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

    def adapters(self) -> list[Adapter]:
        if not self.routed:
            return []
        state: dict[str, torch.Tensor] = {}
        for idx in range(_NUM_BLOCKS):
            state[f"blocks.{idx}.proj.lora_A.weight"] = torch.randn(1, _WIDTH)
            state[f"blocks.{idx}.proj.lora_B.weight"] = torch.randn(_WIDTH, 1)
        return [Adapter.from_state_dict(state)]


_CONFIGS = (
    pytest.param(_Config(()), id="host-only"),
    pytest.param(_Config(("blocks",), None), id="streaming"),
    pytest.param(_Config(("blocks",), 0), id="zero-budget"),
    pytest.param(_Config(("blocks",), 2 * _UNIT_BYTES), id="partial-cache"),
    pytest.param(_Config(("blocks",), 1024 * 1024 * 1024), id="all-resident"),
    pytest.param(_Config(("blocks",), None, use_vram_budget=True), id="shared-budget"),
    pytest.param(_Config(("blocks",), None, use_transfer_context=True), id="shared-transfer-context"),
    pytest.param(_Config(("embed", "blocks"), None, compiled=True), id="mixed-compiled"),
    pytest.param(_Config(("blocks",), None, routed=True), id="streaming-routed"),
)


def _fail(
    patch: pytest.MonkeyPatch,
    step: _Step,
    error: type[BaseException],
    *,
    occurrence: int,
    persistent: bool = False,
    after: bool = False,
) -> None:
    """Make the ``occurrence``-th synchronization, host restore, or refill raise ``error``.

    With ``persistent``, every later one raises too. With ``after``, the step
    completes first, as when an interrupt lands right behind it.
    """
    # A lookahead may still synchronize on its worker during deactivation.
    # Its failure can surface after cleanup succeeds. Inject into the caller's
    # release or forward transition, whose unfinished work this test observes.
    caller = threading.get_ident()
    calls = 0

    def failing[**P, R](original: Callable[P, R]) -> Callable[P, R]:
        def call(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal calls
            if threading.get_ident() != caller:
                return original(*args, **kwargs)
            calls += 1
            hit = calls == occurrence or (persistent and calls > occurrence)
            if hit and not after:
                raise error("injected release failure")
            result = original(*args, **kwargs)
            if hit:
                raise error("injected release failure")
            return result

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
@pytest.mark.parametrize("error", [RuntimeError, Interrupt], ids=["error", "interrupt"])
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
    if config.compiled:
        # Keep compiled variants below Dynamo's frame limit.
        torch.compiler.reset()
        request.addfinalizer(torch.compiler.reset)
    torch.manual_seed(0)
    model = _Model()
    adapters = config.adapters()
    offloader = config.offloader(model)
    x = torch.randn(2, _WIDTH, device="cuda")
    budget = VramBudget(4 * _UNIT_BYTES) if config.use_vram_budget else None
    transfer_ctx = (
        DeviceTransferContext("cuda") if config.use_transfer_context else None
    )

    try:
        offloader.activate(
            "cuda",
            adapters=adapters,
            adapter_mode="routed",
            vram_budget=budget,
            transfer_context=transfer_ctx,
        )
        with torch.inference_mode():
            expected = model(x).cpu()
        raised: BaseException | None = None
        with pytest.MonkeyPatch.context() as patch:
            _fail(patch, step, error, occurrence=occurrence)
            try:
                offloader.deactivate()
            except (RuntimeError, Interrupt) as exc:
                raised = exc

        assert offloader.active_device is None
        # Every synchronization and host restore during deactivation is a release
        # step, so a failure must keep what that step could not free and report it.
        assert offloader.failed == (raised is not None)
        if raised is None:
            # The occurrence did not exist: nothing failed, so nothing was kept.
            assert _on_device(model) == []
            assert offloader._routed_lease is None

        # The next activation retries whatever was kept, then runs as before.
        with (
            activated_model(
                offloader,
                "cuda",
                adapters=adapters,
                adapter_mode="routed",
                vram_budget=budget,
                transfer_context=transfer_ctx,
            ),
            torch.inference_mode(),
        ):
            actual = model(x).cpu()
        assert not offloader.failed
        assert _on_device(model) == []
        if budget is not None:
            assert budget.used_bytes == 0
        if transfer_ctx is not None:
            assert transfer_ctx.borrower_count == 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        if offloader.failed or offloader.active_device is not None:
            offloader.deactivate()
        if transfer_ctx is not None:
            transfer_ctx.close()


@CUDA
def test_activation_reports_a_release_that_keeps_failing() -> None:
    model = _Model()
    offloader = _Config(("blocks",), None).offloader(model)
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


# Transitions that run inside a forward: unit switches and mid-forward
# evictions on budget/ceiling pressure.
_FORWARD_CASES = (
    pytest.param(_Config(("blocks",), None), "restore", id="streaming-switch"),
    pytest.param(_Config(("blocks",), 0), "restore", id="zero-budget-switch"),
    pytest.param(_Config(("blocks",), 2 * _UNIT_BYTES), "restore", id="partial-switch"),
    pytest.param(_Config(("embed", "blocks"), None), "synchronize", id="mixed-evict-sync"),
    pytest.param(_Config(("blocks",), 1024 * 1024 * 1024), "restore", id="all-resident-switch"),
)


@CUDA
@pytest.mark.parametrize("after", [False, True], ids=["before-step", "after-step"])
@pytest.mark.parametrize("occurrence", [1, 2])
@pytest.mark.parametrize(("config", "step"), _FORWARD_CASES)
def test_next_forward_finishes_a_transition_interrupted_mid_forward(
    request: pytest.FixtureRequest,
    config: _Config,
    step: _Step,
    occurrence: int,
    after: bool,
) -> None:
    if config.compiled:
        torch.compiler.reset()
        request.addfinalizer(torch.compiler.reset)
    torch.manual_seed(0)
    model = _Model()
    offloader = config.offloader(model)
    x = torch.randn(2, _WIDTH, device="cuda")

    with activated_model(offloader, "cuda"), torch.inference_mode():
        expected = model(x).cpu()
        interrupted = False
        with pytest.MonkeyPatch.context() as patch:
            _fail(patch, step, Interrupt, occurrence=occurrence, after=after)
            try:
                model(x)
            except Interrupt:
                interrupted = True
        if step == "restore":
            assert interrupted
        actual = model(x).cpu()
        again = model(x).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(again, expected, rtol=0, atol=0)
    assert not offloader.failed
    assert _on_device(model) == []


def _inject_lookahead_record_failure(
    patch: pytest.MonkeyPatch,
    lookahead_stream_id: int,
) -> tuple[list[object], list[object]]:
    from piper_offload.target_lease import CudaTargetLease

    caller = threading.get_ident()
    worker_record_calls = 0
    failed_leases: list[object] = []
    closed_leases: list[object] = []
    fail_lookahead_sync = True
    original_stage = CudaTargetLease.stage
    original_close = CudaTargetLease.close
    original_record_event = torch.cuda.Stream.record_event
    original_stream_sync = torch.cuda.Stream.synchronize

    def tracking_stage(
        lease: CudaTargetLease,
        plan: HostModuleLoadPlan,
        stream: torch.cuda.Stream,
        *,
        non_blocking: bool = True,
    ) -> None:
        try:
            original_stage(lease, plan, stream, non_blocking=non_blocking)
        except RuntimeError:
            failed_leases.append(lease)
            raise

    def tracking_close(lease: CudaTargetLease) -> None:
        original_close(lease)
        closed_leases.append(lease)

    def failing_record_event(stream: torch.cuda.Stream, *args: object, **kwargs: object) -> object:
        nonlocal worker_record_calls
        if threading.get_ident() != caller and stream.cuda_stream == lookahead_stream_id:
            worker_record_calls += 1
            if worker_record_calls >= 3:
                raise RuntimeError("injected lookahead record_event failure")
        return original_record_event(stream, *args, **kwargs)  # type: ignore[arg-type]

    def failing_stream_sync(stream: torch.cuda.Stream) -> None:
        nonlocal fail_lookahead_sync
        if stream.cuda_stream == lookahead_stream_id and fail_lookahead_sync:
            fail_lookahead_sync = False
            raise RuntimeError("injected lookahead synchronize failure")
        original_stream_sync(stream)

    patch.setattr(CudaTargetLease, "stage", tracking_stage)
    patch.setattr(CudaTargetLease, "close", tracking_close)
    patch.setattr(torch.cuda.Stream, "record_event", failing_record_event)
    patch.setattr(torch.cuda.Stream, "synchronize", failing_stream_sync)
    return failed_leases, closed_leases


@CUDA
def test_failed_lookahead_record_event_returns_lease_and_keeps_pin_lease_until_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    model = _Model()
    offloader = _Config(("blocks",), None).offloader(model)
    engine = residency_engine(offloader)
    x = torch.randn(2, _WIDTH, device="cuda")

    offloader.activate("cuda")
    with torch.inference_mode():
        expected = model(x).cpu()
    offloader.deactivate()

    offloader.activate("cuda")
    copy_stream = engine._copy_stream
    assert copy_stream is not None
    with monkeypatch.context() as patch:
        failed_leases, closed_leases = _inject_lookahead_record_failure(patch, copy_stream.cuda_stream)
        with torch.inference_mode():
            model(x)
        with pytest.raises(RuntimeError, match="injected lookahead synchronize failure"):
            offloader.deactivate()
        assert len(failed_leases) == 1
        assert failed_leases[0] not in closed_leases
        assert engine.retains_resources
        assert engine._pin_lease is not None
        assert offloader.failed
        # Even while record_event still fails on the worker stream, retry cleanup succeeds.
        offloader.deactivate()

    assert failed_leases[0] in closed_leases
    assert not engine.retains_resources
    assert engine._pin_lease is None
    assert not offloader.failed
    with activated_model(offloader, "cuda"), torch.inference_mode():
        torch.testing.assert_close(model(x).cpu(), expected, rtol=0, atol=0)
    assert _on_device(model) == []


@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", id="cpu"), pytest.param("cuda", marks=CUDA, id="cuda")],
)
def test_interrupt_in_block_compile_restore_marks_failed_and_retries(
    request: pytest.FixtureRequest,
    device: str,
) -> None:
    torch.compiler.reset()
    request.addfinalizer(torch.compiler.reset)
    torch.manual_seed(0)
    model = _Model()
    offloader = _Config(("blocks",), None, compiled=True).offloader(model)
    x = torch.randn(2, _WIDTH, device=device)

    offloader.activate(device)
    with torch.inference_mode():
        expected = model(x).cpu()

    original_restore = _BlockCompileState.restore
    raised_once = False

    def failing_restore(state: _BlockCompileState) -> None:
        nonlocal raised_once
        if not raised_once:
            raised_once = True
            raise Interrupt("injected restore failure")
        original_restore(state)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_BlockCompileState, "restore", failing_restore)
        with pytest.raises(Interrupt, match="injected restore failure"):
            offloader.deactivate()

    assert offloader.active_device is None
    assert offloader.failed
    assert residency_engine(offloader).retains_resources

    offloader.deactivate()
    assert not offloader.failed
    assert not residency_engine(offloader).retains_resources
    assert _on_device(model) == []
    for blk in model.blocks:
        assert "forward" not in blk.__dict__

    with activated_model(offloader, device), torch.inference_mode():
        actual = model(x).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert _on_device(model) == []


@CUDA
def test_return_slot_eviction_failure_does_not_wedge_active_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from piper_offload.target_lease import CudaTargetLease
    from piper_offload.vram_pool import ElasticVramPool

    torch.manual_seed(0)
    model = _Model()
    offloader = _Config(("blocks",), _UNIT_BYTES).offloader(model)
    engine = residency_engine(offloader)
    x = torch.randn(2, _WIDTH, device="cuda")

    with activated_model(offloader, "cuda"), torch.inference_mode():
        expected = model(x).cpu()
        assert engine.pool is not None
        assert engine._active is not None
        # Add two extra excess idle slots so _trim_to_budget inside _release_active_unit
        # must evict 2 slots to reach _UNIT_BYTES: first succeeds (removing active.slot),
        # second raises Interrupt.
        for extra_idx in (0, 1):
            plan = engine._unit_plans[extra_idx]
            extra_lease = CudaTargetLease.allocate(plan, torch.device("cuda"))
            extra_lease.stage(plan, torch.cuda.current_stream(), non_blocking=False)
            engine.pool.admit_preloaded_slot(extra_idx, extra_lease, engine._active.slot.signature)

        original_close = CudaTargetLease.close
        close_calls = 0

        def fail_second_close(lease: CudaTargetLease) -> None:
            nonlocal close_calls
            close_calls += 1
            if close_calls == 2:
                raise Interrupt("injected second eviction close failure")
            original_close(lease)

        monkeypatch.setattr(ElasticVramPool, "_pick_victim", staticmethod(lambda c, _d: c[0]))
        with monkeypatch.context() as patch:
            patch.setattr(CudaTargetLease, "close", fail_second_close)
            with pytest.raises(Interrupt, match="injected second eviction close failure"):
                engine._release_active_unit()

        assert close_calls == 2
        assert engine._active is None
        actual = model(x).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not offloader.failed
    assert _on_device(model) == []


@CUDA
def test_unit_release_failure_still_releases_resident_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from piper_offload.vram_pool import ElasticVramPool

    torch.manual_seed(0)
    model = _Model()
    offloader = _Config(("blocks",), None).offloader(model)
    engine = residency_engine(offloader)
    x = torch.randn(2, _WIDTH, device="cuda")

    offloader.activate("cuda")
    with torch.inference_mode():
        expected = model(x).cpu()

    original_close_all = ElasticVramPool.close_all
    failed_once = False

    def fail_once(pool: ElasticVramPool) -> None:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("injected pool close failure")
        original_close_all(pool)

    with monkeypatch.context() as patch:
        patch.setattr(ElasticVramPool, "close_all", fail_once)
        with pytest.raises(RuntimeError, match="injected pool close failure"):
            offloader.deactivate()

    # Resident embed/head weights and lease were released despite unit pool error.
    assert model.embed.weight.device.type == "cpu"
    assert model.head.weight.device.type == "cpu"
    assert engine._resident_lease is None
    assert offloader.failed

    offloader.deactivate()
    assert not offloader.failed
    assert _on_device(model) == []
    with activated_model(offloader, "cuda"), torch.inference_mode():
        torch.testing.assert_close(model(x).cpu(), expected, rtol=0, atol=0)


@CUDA
def test_vram_budget_and_borrowed_context_retained_until_release_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from piper_offload.vram_pool import ElasticVramPool

    torch.manual_seed(0)
    model = _Model()
    offloader = _Config(("blocks",), None).offloader(model)
    engine = residency_engine(offloader)
    budget = VramBudget(4 * _UNIT_BYTES)
    ctx = DeviceTransferContext("cuda")
    x = torch.randn(2, _WIDTH, device="cuda")

    try:
        offloader.activate("cuda", vram_budget=budget, transfer_context=ctx)
        with torch.inference_mode():
            expected = model(x).cpu()

        original_close_all = ElasticVramPool.close_all
        pool_failed_once = False

        def fail_pool_once(pool: ElasticVramPool) -> None:
            nonlocal pool_failed_once
            if not pool_failed_once:
                pool_failed_once = True
                raise Interrupt("injected pool close failure")
            original_close_all(pool)

        with monkeypatch.context() as patch:
            patch.setattr(ElasticVramPool, "close_all", fail_pool_once)
            with pytest.raises(Interrupt, match="injected pool close failure"):
                offloader.deactivate()

        assert offloader.failed
        assert engine._budget_handle is not None
        assert ctx.borrower_count == 1
        with pytest.raises(RuntimeError, match="active borrowers"):
            ctx.close()

        offloader.deactivate()
        assert not offloader.failed
        assert budget.used_bytes == 0
        assert ctx.borrower_count == 0
        with activated_model(offloader, "cuda", vram_budget=budget, transfer_context=ctx):
            with torch.inference_mode():
                torch.testing.assert_close(model(x).cpu(), expected, rtol=0, atol=0)
    finally:
        if offloader.failed or offloader.active_device is not None:
            offloader.deactivate()
        ctx.close()


@CUDA
def test_resident_reservation_and_borrowed_context_retained_until_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from piper_offload.target_lease import CudaTargetLease

    torch.manual_seed(0)
    model = _Model()
    offloader = _Config(("blocks",), None).offloader(model)
    engine = residency_engine(offloader)
    budget = VramBudget(4 * _UNIT_BYTES)
    ctx = DeviceTransferContext("cuda")
    x = torch.randn(2, _WIDTH, device="cuda")

    try:
        offloader.activate("cuda", vram_budget=budget, transfer_context=ctx)
        with torch.inference_mode():
            expected = model(x).cpu()

        original_lease_close = CudaTargetLease.close
        res_failed_once = False

        def fail_resident_once(lease: CudaTargetLease) -> None:
            nonlocal res_failed_once
            if lease is engine._resident_lease and not res_failed_once:
                res_failed_once = True
                raise Interrupt("injected resident close failure")
            original_lease_close(lease)

        with monkeypatch.context() as patch:
            patch.setattr(CudaTargetLease, "close", fail_resident_once)
            with pytest.raises(Interrupt, match="injected resident close failure"):
                offloader.deactivate()

        assert offloader.failed
        assert engine._resident_reservation is not None
        assert budget.reserved_bytes == 2 * _UNIT_BYTES
        assert ctx.borrower_count == 0

        offloader.deactivate()
        assert not offloader.failed
        assert budget.used_bytes == 0
        assert ctx.borrower_count == 0
        assert _on_device(model) == []
        with activated_model(offloader, "cuda", vram_budget=budget, transfer_context=ctx):
            with torch.inference_mode():
                torch.testing.assert_close(model(x).cpu(), expected, rtol=0, atol=0)
    finally:
        if offloader.failed or offloader.active_device is not None:
            offloader.deactivate()
        ctx.close()


@CUDA
def test_return_slot_budget_query_interrupt_does_not_wedge_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    model = _Model()
    offloader = _Config(("blocks",), None).offloader(model)
    engine = residency_engine(offloader)
    budget = VramBudget(4 * _UNIT_BYTES)
    x = torch.randn(2, _WIDTH, device="cuda")

    with activated_model(offloader, "cuda", vram_budget=budget), torch.inference_mode():
        expected = model(x).cpu()
        assert engine._active is not None
        assert engine.pool is not None
        released_slot = engine._active.slot

        original_ceiling_for = VramBudget.ceiling_for
        raised_once = False

        def flaky_ceiling(self_budget: VramBudget, tenant: object) -> int | None:
            nonlocal raised_once
            if not raised_once:
                raised_once = True
                raise Interrupt("injected ceiling interrupt")
            return original_ceiling_for(self_budget, tenant)  # type: ignore[arg-type]

        with monkeypatch.context() as patch:
            patch.setattr(VramBudget, "ceiling_for", flaky_ceiling)
            with pytest.raises(Interrupt, match="injected ceiling interrupt"):
                engine._release_active_unit()

        assert engine._active is None
        assert not released_slot.checked_out
        actual = model(x).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert budget.used_bytes == 0
