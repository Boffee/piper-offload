"""Tests for ``piper_offload.model_offloader.ModelOffloader``."""

import copy
from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.utils.checkpoint
from torch import nn

from piper_offload import (
    Adapter,
    ModelOffloader,
    ModelRuntimeFailedError,
    ModelStore,
    OffloadConfig,
    ResourceBinding,
    ResourceStore,
)
from piper_offload.host_module import HostModuleInstance, ParameterOverride
from piper_offload._transfer_priority import foreground_transfer
from piper_offload._working_set import WorkingSetState
from piper_offload.pin_manager import host_pin_manager
from piper_offload.residency_engine import (
    ResidencyEngine,
    _ActiveUnit,
    _ExecutionPredictor,
)
from piper_offload.vram_pool import ElasticVramPool, VramSlot, plan_target_signature

from tests.conftest import (
    CallbackParameterTransform,
    Interrupt,
    activated_model,
    offloaded_units,
    residency_engine,
    resident_instance,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_units_offloader(
    blocks: Sequence[nn.Module],
    *,
    blocks_path: str = "blocks",
    vram_budget_bytes: int | None = None,
) -> ModelOffloader:
    model = _make_block_list_model(blocks, blocks_path)
    return ModelOffloader.from_module(
        model,
        OffloadConfig(offload_paths=(blocks_path,), vram_budget_bytes=vram_budget_bytes),
    )


def _make_block_list_model(
    blocks: Sequence[nn.Module],
    blocks_path: str,
) -> nn.Module:
    class BlockListModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            parent: nn.Module = self
            parts = blocks_path.split(".")
            for part in parts[:-1]:
                child = nn.Module()
                setattr(parent, part, child)
                parent = child
            setattr(parent, parts[-1], nn.ModuleList(blocks))

    return BlockListModel()


def _make_block_model(num_blocks: int = 4, width: int = 8) -> nn.Module:
    """Tiny transformer-shaped model: nn.ModuleList of Linear blocks
    plus a non-block embed/head for non-block-resident testing.
    All params frozen."""

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(width, width, bias=False)
            self.transformer_blocks = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(num_blocks)])
            self.head = nn.Linear(width, width, bias=False)

        def forward(self, x):
            x = self.embed(x)
            for block in self.transformer_blocks:
                x = block(x)
            return self.head(x)

    m = TinyModel()
    for p in m.parameters():
        p.requires_grad = False
    return m


# ---------------------------------------------------------------------------
# ResourceBinding conformance
# ---------------------------------------------------------------------------


class TestResourceBindingConformance:
    def test_is_cached_store_and_runtime_binding(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            assert isinstance(strategy, ResourceStore)
            assert isinstance(strategy, ResourceBinding)
            assert callable(strategy.activate)
            assert callable(strategy.deactivate)
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Constructor captures; cache_bytes is final immediately
# ---------------------------------------------------------------------------


class TestConstructorPins:
    def test_constructor_captures_blocks(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            # Block weights are host via registry replacement.
            for block in m.transformer_blocks:
                assert not block.weight.is_pinned()
            # Non-block (embed/head) also host via resident instance.
            assert not m.embed.weight.is_pinned()
            assert not m.head.weight.is_pinned()
            assert strategy.cache_bytes > 0
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Lifecycle: constructor → activate → deactivate
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_cuda_reactivation_preserves_model_and_restores_host_weights(self) -> None:
        m = _make_block_model()
        expected = torch.device("cuda", torch.cuda.current_device())
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            for _ in range(2):
                with activated_model(strategy, "cuda"):
                    assert strategy.model is m
                    assert strategy.active_device == expected
                    assert residency_engine(strategy).active_device == expected
                    assert m.embed.weight.is_cuda
                    assert m.head.weight.is_cuda
                assert strategy.active_device is None
                for parameter in (m.embed.weight, m.head.weight):
                    assert parameter.device == torch.device("cpu")
                    assert not parameter.is_pinned()
        finally:
            strategy.deactivate()

    def test_activate_cpu_uses_host_backed_weights_without_streaming(self) -> None:
        torch.manual_seed(42)
        m_eager = _make_block_model(num_blocks=4, width=8)
        m_off = _make_block_model(num_blocks=4, width=8)
        m_off.load_state_dict(m_eager.state_dict())

        x = torch.randn(2, 8)
        with torch.no_grad():
            expected = m_eager(x)

        strategy = ModelOffloader.from_module(
            m_off,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            assert strategy.active_device is None
            strategy.deactivate()
            strategy.deactivate()
            host_ptrs = [block.weight.untyped_storage().data_ptr() for block in m_off.transformer_blocks]
            with activated_model(strategy, "cpu") as cpu_model:
                assert strategy.active_device == torch.device("cpu")
                assert residency_engine(strategy).active_device == torch.device("cpu")
                assert residency_engine(strategy).pool is None
                assert all(
                    block.weight.untyped_storage().data_ptr() == ptr
                    for block, ptr in zip(
                        m_off.transformer_blocks,
                        host_ptrs,
                        strict=True,
                    )
                )
                with torch.no_grad():
                    got = cpu_model(x)

            assert strategy.active_device is None
            torch.testing.assert_close(got, expected)
            for p in m_off.parameters():
                assert p.device == torch.device("cpu")
                assert not p.is_pinned()
        finally:
            strategy.deactivate()

    def test_double_activate_raises_before_component_activation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            strategy.activate("cpu")

            def fail_activate(device: torch.device | str | None = None, **kwargs: object) -> None:
                del device, kwargs
                raise AssertionError("engine activation should not run")

            monkeypatch.setattr(
                residency_engine(strategy),
                "activate",
                fail_activate,
            )
            with pytest.raises(RuntimeError, match=r"already.*active"):
                strategy.activate("cpu")
        finally:
            strategy.deactivate()


class TestResidencyEngineActivation:
    def test_direct_cpu_activation_uses_host_backed_weights(self) -> None:
        torch.manual_seed(42)
        m_eager = _make_block_model(num_blocks=4, width=8)
        m = _make_block_model(num_blocks=4, width=8)
        m.load_state_dict(m_eager.state_dict())

        x = torch.randn(2, 8)
        with torch.no_grad():
            expected = m_eager(x)

        offloader = _make_units_offloader(list(m.transformer_blocks), blocks_path="transformer_blocks")
        engine = residency_engine(offloader)
        try:
            host_ptrs = [block.weight.untyped_storage().data_ptr() for block in m.transformer_blocks]
            with activated_model(offloader, "cpu"):
                assert engine.active_device == torch.device("cpu")
                assert engine.pool is None
                assert all(
                    block.weight.untyped_storage().data_ptr() == ptr
                    for block, ptr in zip(
                        m.transformer_blocks,
                        host_ptrs,
                        strict=True,
                    )
                )
                assert all(len(block._forward_pre_hooks) == 0 for block in m.transformer_blocks)
                with torch.no_grad():
                    got = m(x)

            torch.testing.assert_close(got, expected)
            for block in m.transformer_blocks:
                assert block.weight.device == torch.device("cpu")
                assert not block.weight.is_pinned()
        finally:
            offloader.deactivate()

    @CUDA
    @pytest.mark.parametrize("vram_budget_bytes", [None, 1024 * 1024])
    def test_cuda_working_set_can_release_and_reactivate(
        self,
        vram_budget_bytes: int | None,
    ) -> None:
        torch.manual_seed(42)
        eager_blocks = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(3)]).requires_grad_(False)
        blocks = copy.deepcopy(eager_blocks)
        value = torch.randn(2, 8)
        with torch.no_grad():
            expected = value
            for block in eager_blocks:
                expected = block(expected)

        offloader = _make_units_offloader(list(blocks), vram_budget_bytes=vram_budget_bytes)
        engine = residency_engine(offloader)
        try:
            engine.activate(torch.device("cuda"))
            assert engine._working_set_state is WorkingSetState.ACQUIRED

            engine.release()
            engine.release()
            assert engine._working_set_state is WorkingSetState.RELEASED
            assert engine.pool is None
            assert all(block.weight.device.type == "cpu" for block in blocks)

            engine.deactivate()
            engine.activate(torch.device("cuda"))
            assert engine._working_set_state is WorkingSetState.ACQUIRED
            with torch.no_grad():
                actual = value.cuda()
                for block in blocks:
                    actual = block(actual)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual.cpu(), expected)
        finally:
            offloader.deactivate()

    def test_already_active_unit_tracks_the_forward_stream(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class Lease:
            def __init__(self) -> None:
                self.used: list[object] = []

            def record_stream(self, stream: object) -> None:
                self.used.append(stream)

        offloader = _make_units_offloader([nn.Linear(4, 4, bias=False).requires_grad_(False)])
        engine = residency_engine(offloader)
        lease = Lease()
        slot = VramSlot(lease=cast(Any, lease), signature=cast(Any, ()), nbytes=64)
        forward_stream = object()
        engine._active_device = torch.device("cuda")
        engine._pool = ElasticVramPool(torch.device("cuda"), vram_budget_bytes=None)
        engine._predictor = _ExecutionPredictor(1)
        engine._active = _ActiveUnit(0, slot, installed=True)
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda _device: forward_stream,
        )
        try:
            engine._before_unit_forward(0)
            assert lease.used == [forward_stream]
        finally:
            engine._active = None
            engine._pool = None
            engine._predictor = None
            engine._active_device = None
            offloader.deactivate()

    def test_direct_cpu_double_activate_raises(self) -> None:
        m = _make_block_model(num_blocks=2)
        offloader = _make_units_offloader(list(m.transformer_blocks), blocks_path="transformer_blocks")
        engine = residency_engine(offloader)
        try:
            engine.activate(torch.device("cpu"))
            with pytest.raises(RuntimeError, match="already active"):
                engine.activate(torch.device("cpu"))
        finally:
            offloader.deactivate()


# ---------------------------------------------------------------------------
# Cleanup contract: deactivate restores CPU state, drop refs to free
# ---------------------------------------------------------------------------


class TestCleanup:
    @CUDA
    def test_deactivate_restores_cpu_state(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        strategy.activate("cuda")
        strategy.deactivate()
        # Model is back in CPU/host state — usable, just without the
        # strategy's GPU streaming.
        for p in m.parameters():
            assert not p.is_cuda

    def test_deactivate_is_idempotent(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        strategy.deactivate()
        strategy.deactivate()  # no error

    @CUDA
    def test_deactivate_ends_the_residency_session(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(strategy)
        strategy.activate("cuda")
        assert engine.active_device is not None
        assert engine._working_set_state is WorkingSetState.ACQUIRED
        strategy.deactivate()
        assert engine.active_device is None
        assert not engine.retains_resources

    @CUDA
    def test_drop_strategy_without_deactivate_does_not_cycle(self) -> None:
        # Regression: pre/post hooks use weakref so dropping the strategy frees it
        # without needing the cyclic GC even if the user skipped deactivate().
        import gc
        import weakref

        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        strategy.activate("cuda")  # installs hooks; no deactivate
        engine = residency_engine(strategy)
        engine_ref = weakref.ref(engine)

        gc.disable()
        try:
            del strategy, engine
            assert engine_ref() is None
        finally:
            gc.enable()

    @CUDA
    def test_orphaned_hooks_noop_after_strategy_dropped(self) -> None:
        torch.manual_seed(0)
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        strategy.activate("cuda")
        del strategy

        x = torch.randn(2, 8, device="cpu")
        with torch.no_grad():
            try:
                _ = m(x)
            except Exception as e:
                if "AttributeError" in repr(e) or "NoneType" in repr(e):
                    raise AssertionError(f"orphan hook crashed: {e!r}") from e


# ---------------------------------------------------------------------------
# Hook lifecycle (CUDA-only)
# ---------------------------------------------------------------------------


class TestHookLifecycle:
    def test_named_forward_hook_is_caller_owned(self) -> None:
        model = _make_block_model()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        calls: list[nn.Module] = []
        remove_hook = offloader.register_forward_hook(
            "transformer_blocks.3",
            lambda module, _args, _output: calls.append(module),
        )
        try:
            with torch.no_grad():
                model(torch.randn(2, 8))
            assert calls == [model.transformer_blocks[3]]

            remove_hook()
            remove_hook()
            with torch.no_grad():
                model(torch.randn(2, 8))
            assert calls == [model.transformer_blocks[3]]
        finally:
            remove_hook()
            offloader.deactivate()

    def test_named_forward_hook_requires_module_name(self) -> None:
        model = _make_block_model()
        offloader = ModelOffloader.from_module(model)
        try:
            with pytest.raises(AttributeError):
                offloader.register_forward_hook(
                    "head.weight",
                    lambda _module, _args, _output: None,
                )
        finally:
            offloader.deactivate()

    @CUDA
    def test_hooks_installed_on_activate_removed_on_deactivate(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            strategy.activate("cuda")
            for block in m.transformer_blocks:
                assert len(block._forward_pre_hooks) > 0
            strategy.deactivate()
            for block in m.transformer_blocks:
                assert len(block._forward_pre_hooks) == 0
        finally:
            strategy.deactivate()

    @CUDA
    def test_hooks_removed_on_deactivate_drop(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        strategy.activate("cuda")
        blocks = list(m.transformer_blocks)
        strategy.deactivate()
        for block in blocks:
            assert len(block._forward_pre_hooks) == 0


# ---------------------------------------------------------------------------
# ModelOffloader-owned generic partition & immediate-release schedule
# ---------------------------------------------------------------------------


class TestOffloadPartitionResidency:
    def test_offload_paths_partition_from_resident_state(self) -> None:
        model = _make_block_model()
        model.embed.register_buffer("table", torch.randn(8))
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("transformer_blocks", "embed", "head")),
        )
        try:
            engine = residency_engine(offloader)
            assert engine.unit_paths == (
                "transformer_blocks.0",
                "transformer_blocks.1",
                "transformer_blocks.2",
                "transformer_blocks.3",
                "embed",
                "head",
            )
            assert set(engine.units[4].params) == {"weight"}
            assert set(engine.units[4].buffers) == {"table"}
            assert set(engine.units[5].params) == {"weight"}
            assert resident_instance(offloader) is None
        finally:
            offloader.deactivate()

    def test_capture_and_rebind_include_all_groups(self) -> None:
        source = _make_block_model(num_blocks=2)
        source.tail = nn.ModuleList([nn.Identity(), nn.Linear(8, 8, bias=False)]).requires_grad_(False)
        source.embed.register_buffer("scale", torch.ones(8))
        target = copy.deepcopy(source)
        for tensor in target.state_dict().values():
            tensor.zero_()

        config = OffloadConfig(
            offload_paths=("transformer_blocks", "tail", "embed"),
        )
        expected_bytes = sum(t.nbytes for t in source.state_dict().values())
        offloader = ModelOffloader.from_module(copy.deepcopy(source), config)
        assert offloader.cache_bytes == expected_bytes
        offloader.deactivate()

        store = ModelStore.from_module(source)
        assert store.cache_bytes == expected_bytes
        bound_offloader = store.bind(target, config)
        try:
            engine = residency_engine(bound_offloader)
            torch.testing.assert_close(target.state_dict(), source.state_dict())
            resident = resident_instance(bound_offloader)
            assert resident is not None
            assert set(resident.params) == {"head.weight"}
            assert engine.unit_paths == (
                "transformer_blocks.0",
                "transformer_blocks.1",
                "tail.1",
                "embed",
            )
            assert [u.module for u in engine.units] == [
                target.transformer_blocks[0],
                target.transformer_blocks[1],
                target.tail[1],
                target.embed,
            ]
            assert set(engine.units[3].buffers) == {"scale"}
        finally:
            bound_offloader.deactivate()

    def test_cpu_activation_stays_eager(self) -> None:
        model = _make_block_model()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("transformer_blocks", "embed", "head")),
        )
        try:
            with activated_model(offloader, "cpu"):
                with torch.inference_mode():
                    model(torch.randn(2, 8))
                assert not model._forward_hooks
                assert not model.embed._forward_hooks
                assert not model.head._forward_hooks
                assert not model.transformer_blocks[-1]._forward_hooks
        finally:
            offloader.deactivate()

    @CUDA
    def test_zero_budget_units_release_at_their_own_boundaries(self) -> None:
        model = _make_block_model(num_blocks=3)
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(
                offload_paths=("embed", "transformer_blocks", "head"),
                vram_budget_bytes=0,
            ),
        )
        engine = residency_engine(offloader)
        embed_entry_states: list[tuple[bool, int]] = []
        block_entry_states: list[tuple[bool, int]] = []
        head_entry_states: list[tuple[bool, int]] = []
        root_states: list[tuple[bool, int]] = []
        def _unit_state(unit_idx: int) -> tuple[bool, int]:
            return (
                engine._active is not None and engine._active.unit_idx == unit_idx,
                engine.pool.live_slot_count if engine.pool else 0,
            )

        embed_handle = model.embed.register_forward_pre_hook(
            lambda _m, _a: embed_entry_states.append(_unit_state(0))
        )
        block_handle = model.transformer_blocks[0].register_forward_pre_hook(
            lambda _m, _a: block_entry_states.append(_unit_state(1))
        )
        head_handle = model.head.register_forward_pre_hook(
            lambda _m, _a: head_entry_states.append(_unit_state(4))
        )
        root_handle = model.register_forward_hook(
            lambda _m, _a, _o: root_states.append(
                (engine._active is not None, engine.pool.live_slot_count if engine.pool else 0)
            )
        )
        try:
            with activated_model(offloader, "cuda"):
                value = torch.randn(2, 8, device="cuda")
                with torch.inference_mode():
                    first = model(value).clone()
                    second = model(value).clone()
                torch.cuda.synchronize()

                assert embed_entry_states == [(True, 1), (True, 1)]
                assert block_entry_states == [(True, 1), (True, 1)]
                assert head_entry_states == [(True, 1), (True, 1)]
                # Under budget=0, post-hook releases _active and pool evicts idle slots immediately.
                assert root_states == [(False, 0), (False, 0)]
                assert engine._active is None
                torch.testing.assert_close(second, first, rtol=0, atol=0)
        finally:
            embed_handle.remove()
            block_handle.remove()
            head_handle.remove()
            root_handle.remove()
            offloader.deactivate()

        assert not model._forward_hooks
        assert not model.embed._forward_hooks
        assert not model.head._forward_hooks
        assert not model.transformer_blocks[-1]._forward_hooks

    @CUDA
    def test_unit_records_its_forward_stream_before_release(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from piper_offload.target_lease import CudaTargetLease

        model = _make_block_model()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(
                offload_paths=("embed", "transformer_blocks"),
                vram_budget_bytes=0,
            ),
        )
        engine = residency_engine(offloader)
        recorded_streams: list[torch.cuda.Stream] = []
        original_record_stream = CudaTargetLease.record_stream

        def record_stream(lease: CudaTargetLease, stream: torch.cuda.Stream) -> None:
            recorded_streams.append(stream)
            original_record_stream(lease, stream)

        monkeypatch.setattr(CudaTargetLease, "record_stream", record_stream)
        side_stream = torch.cuda.Stream()
        try:
            with activated_model(offloader, "cuda"):
                with torch.cuda.stream(side_stream):
                    model.embed(torch.randn(2, 8, device="cuda"))

                assert recorded_streams[-1] == side_stream
                assert engine._active is None
        finally:
            offloader.deactivate()

    @CUDA
    def test_activation_inside_inference_mode_keeps_targets_mutable(self) -> None:
        model = _make_block_model(num_blocks=3)
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("embed", "transformer_blocks")),
        )
        engine = residency_engine(offloader)
        with torch.inference_mode():
            with activated_model(offloader, "cuda"):
                model(torch.randn(2, 8, device="cuda"))
                model(torch.randn(2, 8, device="cuda"))
                assert engine._working_set_state is WorkingSetState.ACQUIRED

    @CUDA
    def test_multiple_offload_paths_handle_skipped_stage(
        self,
    ) -> None:
        class MultiPathModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first_blocks = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(2)])
                self.second_blocks = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(2)])
                self.head = nn.Linear(8, 8, bias=False)
                self.run_first = True
                self.requires_grad_(False)

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                if self.run_first:
                    for block in self.first_blocks:
                        value = block(value)
                for block in self.second_blocks:
                    value = block(value)
                return self.head(value)

        model = MultiPathModel()
        reference = copy.deepcopy(model).cuda()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("first_blocks", "second_blocks")),
        )
        x = torch.randn(2, 8, device="cuda")
        try:
            with activated_model(offloader, "cuda"), torch.inference_mode():
                actual_full = model(x).clone()
                expected_full = reference(x)
                torch.testing.assert_close(actual_full, expected_full, rtol=1e-5, atol=1e-5)

                model.run_first = False
                reference.run_first = False
                actual_partial = model(x).clone()
                expected_partial = reference(x)
                torch.testing.assert_close(actual_partial, expected_partial, rtol=1e-5, atol=1e-5)
        finally:
            offloader.deactivate()

        assert not model._forward_hooks
        assert not model.first_blocks[-1]._forward_hooks
        assert not model.second_blocks[-1]._forward_hooks

    @CUDA
    def test_forward_failure_leaves_session_safe_to_deactivate(
        self,
    ) -> None:
        class FailingHead(nn.Linear):
            fail = True

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                if self.fail:
                    raise RuntimeError("simulated suffix failure")
                return super().forward(value)

        model = _make_block_model(num_blocks=3)
        model.head = FailingHead(8, 8, bias=False).requires_grad_(False)
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(
                offload_paths=("embed", "transformer_blocks"),
                vram_budget_bytes=0,
            ),
        )
        engine = residency_engine(offloader)
        try:
            offloader.activate("cuda")
            with pytest.raises(RuntimeError, match="simulated suffix failure"):
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
            assert engine._active is None
        finally:
            offloader.deactivate()

        assert not model._forward_hooks
        assert not model.embed._forward_hooks
        assert not model.transformer_blocks[-1]._forward_hooks

        model.head.fail = False
        try:
            with activated_model(offloader, "cuda"):
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
                assert engine._working_set_state is WorkingSetState.ACQUIRED
        finally:
            offloader.deactivate()

    @CUDA
    def test_hook_install_failure_rolls_back_activation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _make_block_model(num_blocks=3)
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("embed", "transformer_blocks")),
        )
        engine = residency_engine(offloader)
        original_register = model.transformer_blocks[-1].register_forward_hook

        def fail_register(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated tail hook failure")

        monkeypatch.setattr(model.transformer_blocks[-1], "register_forward_hook", fail_register)
        with pytest.raises(RuntimeError, match="simulated tail hook failure"):
            offloader.activate("cuda")

        assert offloader.active_device is None
        assert not engine.retains_resources
        assert not model.embed._forward_hooks
        assert not model.transformer_blocks[-1]._forward_hooks

        monkeypatch.setattr(model.transformer_blocks[-1], "register_forward_hook", original_register)
        try:
            with activated_model(offloader, "cuda"):
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
                assert engine._working_set_state is WorkingSetState.ACQUIRED
        finally:
            offloader.deactivate()


# ---------------------------------------------------------------------------
# Traversal-aware prefetch
# ---------------------------------------------------------------------------


class TestTraversalPrefetch:
    """The engine handles repeated forward and reverse traversals."""

    def _record_prefetches(
        self,
        engine: ResidencyEngine,
    ) -> list[int]:
        """Wrap the residency engine's lookahead submission to record unit indices."""
        recorded: list[int] = []
        original = engine._submit_lookahead

        def record(idx: int) -> None:
            recorded.append(idx)
            original(idx)

        engine._submit_lookahead = record  # type: ignore[method-assign]
        return recorded

    @CUDA
    def test_wraps_prefetch_at_iteration_boundary(self) -> None:
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(strategy)

        with activated_model(strategy, "cuda"):
            recorded = self._record_prefetches(engine)
            x = torch.randn(2, 8, device="cuda")
            m(x)  # iteration 1
            torch.cuda.synchronize()
            recorded.clear()
            m(x)  # iteration 2
            torch.cuda.synchronize()

        assert recorded == [1, 2, 3, 0], recorded

    @CUDA
    def test_continuous_backward_stays_backward(self) -> None:
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(strategy)

        with activated_model(strategy, "cuda"):
            recorded = self._record_prefetches(engine)
            x = torch.randn(2, 8, device="cuda")
            for idx in (3, 2, 1, 0):
                m.transformer_blocks[idx](x)
            torch.cuda.synchronize()

        assert recorded == [0, 1, 0, 3], recorded

    @CUDA
    def test_two_iterations_match_eager_baseline(self) -> None:
        torch.manual_seed(42)
        m_eager = _make_block_model(num_blocks=4, width=8).cuda()
        x = torch.randn(2, 8, device="cuda")
        with torch.no_grad():
            expected_1 = m_eager(x)
            expected_2 = m_eager(expected_1)

        torch.manual_seed(42)
        m_off = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader.from_module(
            m_off,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            with activated_model(strategy, "cuda"):
                with torch.no_grad():
                    got_1 = m_off(x)
                    got_2 = m_off(got_1)
                torch.cuda.synchronize()
            torch.testing.assert_close(got_1, expected_1, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(got_2, expected_2, atol=1e-5, rtol=1e-5)
        finally:
            strategy.deactivate()

    @CUDA
    def test_prefetch_resumes_after_a_skipped_block(self) -> None:
        torch.manual_seed(0)
        m = _make_block_model(num_blocks=4, width=8)
        reference = copy.deepcopy(m).cuda()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(strategy)
        x = torch.randn(2, 8, device="cuda")

        with activated_model(strategy, "cuda"), torch.no_grad():
            actual = x
            expected = x
            for idx in (0, 1, 3):
                actual = m.transformer_blocks[idx](actual)
                expected = reference.transformer_blocks[idx](expected)
            assert engine._active is not None and engine._active.unit_idx == 3
            lookahead = engine._lookahead
            assert lookahead is not None and lookahead.unit_idx == 0
            lookahead.future.result()
            torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @CUDA
    def test_lookahead_waits_while_a_foreground_transfer_runs(self) -> None:
        m = _make_block_model(num_blocks=2, width=8)
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(strategy)
        x = torch.randn(2, 8, device="cuda")

        with activated_model(strategy, "cuda"), torch.no_grad():
            with foreground_transfer("cuda"):
                m.transformer_blocks[0](x)
                lookahead = engine._lookahead
                assert lookahead is not None
                with pytest.raises(TimeoutError):
                    lookahead.future.result(timeout=0.2)
            lookahead.future.result(timeout=30)

    @CUDA
    def test_activation_inside_inference_mode_keeps_pooled_targets_mutable(self) -> None:
        torch.manual_seed(0)
        m = _make_block_model(num_blocks=4, width=8)
        reference = copy.deepcopy(m).cuda()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        x = torch.randn(2, 8, device="cuda")

        with torch.inference_mode():
            with activated_model(strategy, "cuda"):
                first = m(x).clone()
                second = m(x).clone()
            expected = reference(x)

        torch.testing.assert_close(first, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(second, first, rtol=0, atol=0)

    @CUDA
    def test_small_n_threshold_corner(self) -> None:
        m = _make_block_model(num_blocks=3, width=8)
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(strategy)

        with activated_model(strategy, "cuda"):
            recorded = self._record_prefetches(engine)
            x = torch.randn(2, 8, device="cuda")
            m(x)  # iteration 1
            torch.cuda.synchronize()
            recorded.clear()
            m(x)  # iteration 2
            torch.cuda.synchronize()

        assert recorded == [1, 2, 0], recorded


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


class TestForwardCorrectness:
    @CUDA
    def test_forward_matches_eager_baseline(self) -> None:
        torch.manual_seed(42)
        m = _make_block_model(num_blocks=4, width=8).cuda()
        x = torch.randn(2, 8, device="cuda")
        with torch.no_grad():
            expected = m(x)

        torch.manual_seed(42)
        m_off = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader.from_module(
            m_off,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            strategy.activate("cuda")
            with torch.no_grad():
                got = m_off(x)
            torch.cuda.synchronize()
            torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
        finally:
            strategy.deactivate()

    @CUDA
    def test_forward_after_deactivate_then_activate_cycle(self) -> None:
        torch.manual_seed(42)
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            x = torch.randn(2, 8, device="cuda")
            strategy.activate("cuda")
            with torch.no_grad():
                first = m(x)
            torch.cuda.synchronize()
            strategy.deactivate()

            strategy.activate("cuda")
            with torch.no_grad():
                second = m(x)
            torch.cuda.synchronize()
            torch.testing.assert_close(first, second)
            strategy.deactivate()
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_offload_paths_disables_unit_offloading(self) -> None:
        m = _make_block_model(num_blocks=4)
        strategy = ModelOffloader.from_module(m, OffloadConfig(offload_paths=()))
        try:
            assert not offloaded_units(strategy)
        finally:
            strategy.deactivate()

    def test_aliased_block_modules_deduplicate_to_one_unit(self) -> None:
        model = _make_block_model()
        shared = nn.Linear(8, 8, bias=False).requires_grad_(False)
        model.transformer_blocks = nn.ModuleList([shared, shared])

        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            assert len(offloaded_units(offloader)) == 1
            assert shared.weight.device.type == "cpu"
        finally:
            offloader.deactivate()

    def test_offload_paths_accept_standalone_module_and_reject_non_module(self) -> None:
        m = _make_block_model(num_blocks=4)
        standalone = ModelOffloader.from_module(m, OffloadConfig(offload_paths=("embed",)))
        try:
            assert residency_engine(standalone).unit_paths == ("embed",)
        finally:
            standalone.deactivate()

        m2 = _make_block_model(num_blocks=4)
        with pytest.raises(TypeError, match="Expected nn.Module or nn.ModuleList"):
            ModelOffloader.from_module(
                m2,
                OffloadConfig(offload_paths=("embed.weight",)),
            )

        with pytest.raises(ValueError, match="does not exist"):
            ModelOffloader.from_module(
                m2,
                OffloadConfig(offload_paths=("missing",)),
            )


# ---------------------------------------------------------------------------
# ResourceCache integration: cache_bytes is final at construction
# ---------------------------------------------------------------------------


class TestResourceCacheIntegration:
    def test_model_spec_propagates_offload_paths(self) -> None:
        from piper_offload import ModelSpec

        spec = ModelSpec(
            key="offload",
            estimated_cache_bytes=1024,
            factory=_make_block_model,
            config=OffloadConfig(
                offload_paths=("transformer_blocks", "embed", "head"),
            ),
        )

        offloader = spec.build_store()
        try:
            assert residency_engine(offloader).unit_paths == (
                "transformer_blocks.0",
                "transformer_blocks.1",
                "transformer_blocks.2",
                "transformer_blocks.3",
                "embed",
                "head",
            )
            assert resident_instance(offloader) is None
        finally:
            offloader.deactivate()

    def test_model_spec_reuses_single_model(self) -> None:
        from piper_offload import ModelCache, ModelSpec

        device = torch.device("cpu")
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return _make_block_model(num_blocks=4, width=8)

        cache = ModelCache()
        spec = ModelSpec(
            key="xformer",
            estimated_cache_bytes=1024,
            factory=factory,
            config=OffloadConfig(offload_paths=("transformer_blocks",)),
        )

        with cache.use(
            spec,
            device=device,
        ) as first_model:
            assert isinstance(first_model, nn.Module)
            assert all(not param.is_meta for param in first_model.parameters())
            assert all(not buffer.is_meta for buffer in first_model.buffers())
            x = torch.randn(2, 8)
            with torch.no_grad():
                _ = first_model(x)

        info = cache.info("xformer")
        assert info.cached
        assert info.cache_bytes is not None
        assert info.cache_bytes > 0
        assert info.lease_count == 0

        # Second sequential use returns the same cached model; factory is
        # not called again.
        with cache.use(
            spec,
            device=device,
        ) as second_model:
            assert second_model is first_model
        assert factory_calls == 1

        cache.clear()

    def test_model_spec_leases_the_same_offloader(self) -> None:
        from piper_offload import ResourceCache, ModelSpec

        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return _make_block_model()

        cache = ResourceCache(max_cache_bytes=10_000_000)
        spec = ModelSpec(
            key="xformer",
            estimated_cache_bytes=1024,
            factory=factory,
            config=OffloadConfig(offload_paths=("transformer_blocks",)),
        )

        with cache.lease(spec) as first_offloader:
            assert isinstance(first_offloader, ModelOffloader)
            with cache.lease(spec) as second_offloader:
                assert second_offloader is first_offloader
                assert cache.info("xformer").lease_count == 2

        assert factory_calls == 1
        cache.clear()

    def test_model_spec_rejects_nested_use(self) -> None:
        from piper_offload import ModelCache, ModelRuntimeInUseError, ModelSpec

        cache = ModelCache()
        spec = ModelSpec(
            key="xformer",
            estimated_cache_bytes=1024,
            factory=lambda: _make_block_model(num_blocks=4, width=8),
            config=OffloadConfig(offload_paths=("transformer_blocks",)),
        )

        with cache.use(spec, device="cpu"):
            with pytest.raises(ModelRuntimeInUseError, match="overlapping"):
                with cache.use(spec, device="cpu"):
                    pass

        assert cache.info("xformer").lease_count == 0
        cache.clear()

    def test_model_cache_deactivates_after_body_error(self) -> None:
        from piper_offload import ModelCache, ModelSpec

        cache = ModelCache()
        spec = ModelSpec(
            key="xformer",
            estimated_cache_bytes=1024,
            factory=lambda: _make_block_model(num_blocks=4, width=8),
            config=OffloadConfig(offload_paths=("transformer_blocks",)),
        )

        with pytest.raises(ValueError, match="caller failed"):
            with cache.use(spec, device="cpu"):
                raise ValueError("caller failed")

        assert cache.info("xformer").lease_count == 0
        # A second activation proves the cache deactivated the shared
        # offloader and released its single-runtime claim.
        with cache.use(spec, device="cpu"):
            pass

        cache.clear()

    def test_model_cache_is_an_unbounded_resource_cache(self) -> None:
        from piper_offload import ModelCache, ResourceCache

        cache = ModelCache()

        assert isinstance(cache, ResourceCache)
        assert cache.max_cache_bytes is None

        with pytest.raises(TypeError):
            ModelCache(max_cache_bytes=10_000_000)

    def test_model_cache_preserves_factory_split_file_mapping(self, tmp_path: Path) -> None:
        from piper_offload import ModelCache, ModelSpec

        path = tmp_path / "weight.bin"
        path.touch()
        seed = torch.from_file(str(path), shared=True, size=24, dtype=torch.float32)
        seed.copy_(torch.arange(24, dtype=torch.float32))
        del seed
        captured_storage_pointer = 0

        def factory() -> nn.Module:
            nonlocal captured_storage_pointer
            fused = torch.from_file(
                str(path),
                shared=False,
                size=24,
                dtype=torch.float32,
            ).reshape(3, 2, 4)
            captured_storage_pointer = fused.untyped_storage().data_ptr()
            model = nn.Module()
            model.q = nn.Parameter(fused[0], requires_grad=False)
            model.k = nn.Parameter(fused[1], requires_grad=False)
            model.v = nn.Parameter(fused[2], requires_grad=False)
            return model

        cache = ModelCache()
        spec = ModelSpec(key="mapped", factory=factory)

        with cache.use(spec, device="cpu") as model:
            params = (model.q, model.k, model.v)
            storage_pointers = {
                param.untyped_storage().data_ptr() for param in params
            }
            assert storage_pointers == {captured_storage_pointer}
            assert tuple(param.storage_offset() for param in params) == (0, 8, 16)
            assert all(not param.untyped_storage().resizable() for param in params)
            torch.testing.assert_close(
                torch.stack(params),
                torch.arange(24, dtype=torch.float32).reshape(3, 2, 4),
            )

        assert cache.info("mapped").cached
        assert cache.info("mapped").estimated_cache_bytes == 0
        assert cache.info("mapped").cache_bytes == 24 * torch.float32.itemsize
        cache.clear()

    def test_model_spec_rejects_trainable_factory_model(self) -> None:
        from piper_offload import ModelCache, ModelSpec

        cache = ModelCache()
        spec = ModelSpec(
            key="trainable",
            estimated_cache_bytes=1024,
            factory=lambda: nn.Linear(8, 8, bias=False),
        )

        with pytest.raises(ValueError, match="frozen"):
            with cache.use(spec, device="cpu"):
                pass

        assert not cache.info("trainable").cached
        cache.clear()


# ---------------------------------------------------------------------------
# Activate failure cleanup contract
# ---------------------------------------------------------------------------


class TestFailedSession:
    @staticmethod
    def _offloader_with_leftover() -> tuple[ModelOffloader, ResidencyEngine]:
        model = _make_block_model()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(offloader)
        engine._pin_lease = host_pin_manager.acquire([], pin=False)
        engine._working_set_state = WorkingSetState.CHANGING
        assert offloader.failed
        return offloader, engine

    def test_cache_retains_model_until_cleanup_succeeds(self, monkeypatch, request) -> None:
        from piper_offload import ModelCache, ModelSpec, ResourceLeasedError, ResourceTooLargeError

        cache = ModelCache()
        spec = ModelSpec(
            key="model",
            factory=_make_block_model,
            config=OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        with cache.lease(spec) as offloader:
            engine = residency_engine(offloader)
        request.addfinalizer(offloader.deactivate)

        def fail_close(_engine):
            raise RuntimeError("incomplete release")

        with monkeypatch.context() as patch:
            patch.setattr(ResidencyEngine, "_close_pin_lease", fail_close)
            with pytest.raises(RuntimeError, match="incomplete release"), cache.use(spec, device="cpu"):
                pin = engine._pin_lease = host_pin_manager.acquire([], pin=False)
                engine._working_set_state = WorkingSetState.CHANGING

            assert offloader.failed
            assert not pin.closed
            assert cache.info(spec.key).lease_count == 1
            assert cache.evict_bytes(cache.used_cache_bytes) == 0
            with pytest.raises(ResourceTooLargeError):
                cache.resize(0)
            with pytest.raises(ResourceLeasedError):
                cache.evict(spec.key)
            with pytest.raises(ResourceLeasedError):
                cache.unregister(spec.key)
            with pytest.raises(ResourceLeasedError):
                cache.register(spec, replace=True)
            with pytest.raises(ResourceLeasedError):
                cache.clear()
            with pytest.raises(ModelRuntimeFailedError), cache.use(spec, device="cpu"):
                pytest.fail("activation continued after failed cleanup")

        with cache.use(spec, device="cpu") as model:
            assert model is offloader.value
            assert pin.closed
            assert cache.info(spec.key).lease_count == 1
        cache.resize(0)
        assert not cache.info(spec.key).cached

    def test_activation_finishes_what_a_failed_session_kept(self) -> None:
        offloader, engine = self._offloader_with_leftover()
        leftover = engine._pin_lease
        assert leftover is not None

        with activated_model(offloader, "cpu"):
            assert leftover.closed
            assert not offloader.failed

    @pytest.mark.parametrize(
        ("error", "raised"),
        [(RuntimeError, ModelRuntimeFailedError), (Interrupt, Interrupt)],
        ids=["error", "interrupt"],
    )
    def test_activation_refuses_when_the_retry_fails_again(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: type[BaseException],
        raised: type[BaseException],
    ) -> None:
        offloader, engine = self._offloader_with_leftover()

        def still_failing(_engine: ResidencyEngine) -> None:
            raise error("release still failing")

        with monkeypatch.context() as patch:
            patch.setattr(ResidencyEngine, "_close_pin_lease", still_failing)
            with pytest.raises(raised) as caught:
                offloader.activate("cpu")
        if raised is ModelRuntimeFailedError:
            assert isinstance(caught.value.__cause__, RuntimeError)
        assert offloader.failed
        assert offloader.active_device is None
        assert engine.active_device is None

        offloader.deactivate()
        assert not offloader.failed
        assert engine._pin_lease is None
        with activated_model(offloader, "cpu"):
            pass

    @CUDA
    def test_recoverable_deactivation_error_keeps_adapter_activation_available(self) -> None:
        model = _make_block_model(num_blocks=3)
        lora = Adapter.from_state_dict(
            {
                f"transformer_blocks.{idx}.{name}": torch.randn(*shape)
                for idx in range(3)
                for name, shape in (("lora_A.weight", (1, 8)), ("lora_B.weight", (8, 1)))
            }
        )
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(offloader)
        x = torch.randn(2, 8, device="cuda")

        offloader.activate("cuda", adapters=[lora])
        with torch.no_grad():
            expected = model(x).clone()
        lookahead = engine._lookahead
        assert lookahead is not None
        lookahead.future.result()
        bad_prefetch: Future[None] = Future()
        bad_prefetch.set_exception(RuntimeError("simulated prefetch failure"))
        lookahead.future = bad_prefetch
        with pytest.raises(RuntimeError, match="simulated prefetch failure"):
            offloader.deactivate()

        assert not offloader.failed
        with activated_model(offloader, "cuda", adapters=[lora]), torch.no_grad():
            actual = model(x).clone()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestActivateFailureCleanup:
    @CUDA
    def test_partial_activate_failure_rolls_back_all_resources(self, monkeypatch) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(strategy)
        original_register_hooks = engine._register_unit_hooks

        def broken_register_hooks(*args, **kwargs):
            original_register_hooks(*args, **kwargs)
            raise RuntimeError("simulated activate failure")

        monkeypatch.setattr(engine, "_register_unit_hooks", broken_register_hooks)

        with pytest.raises(RuntimeError, match="simulated activate failure"):
            strategy.activate("cuda")

        assert engine.active_device is None
        assert not engine.retains_resources
        assert strategy.active_device is None

        monkeypatch.setattr(engine, "_register_unit_hooks", original_register_hooks)
        with activated_model(strategy, "cuda"):
            pass

    def test_interrupted_claim_is_given_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        offloader = ModelOffloader.from_module(
            _make_block_model(),
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )

        def interrupted(_offloader: ModelOffloader) -> bool:
            raise Interrupt

        with monkeypatch.context() as patch:
            patch.setattr(ModelOffloader, "failed", property(interrupted))
            with pytest.raises(Interrupt):
                offloader.activate("cpu")
        assert offloader.active_device is None
        with activated_model(offloader, "cpu"):
            pass


# ---------------------------------------------------------------------------
# Prefetch failure during deactivate
# ---------------------------------------------------------------------------


class TestPrefetchFailureOnDeactivate:
    @CUDA
    def test_prefetch_failure_propagates_after_cleanup(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        strategy.activate("cuda")
        engine = residency_engine(strategy)
        with torch.no_grad():
            m.transformer_blocks[0](torch.randn(2, 8, device="cuda"))
        lookahead = engine._lookahead
        assert lookahead is not None
        lookahead.future.result()
        bad_future: Future[None] = Future()
        bad_future.set_exception(RuntimeError("simulated prefetch failure"))
        lookahead.future = bad_future

        with pytest.raises(RuntimeError, match="simulated prefetch failure"):
            strategy.deactivate()

        assert not engine._unit_hooks
        assert engine._executor is None
        assert strategy.active_device is None

        with activated_model(strategy, "cpu"):
            pass

    @CUDA
    def test_failed_host_restore_keeps_the_block_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = _make_block_model(num_blocks=3)
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(strategy)
        first_block = m.transformer_blocks[0]
        install_host = HostModuleInstance.install_host
        failures = [Interrupt()]

        def install_host_failing_once(instance: HostModuleInstance) -> None:
            if failures and instance.module is first_block:
                raise failures.pop()
            install_host(instance)

        x = torch.randn(2, 8, device="cuda")
        with activated_model(strategy, "cuda"), torch.no_grad():
            hidden = m.transformer_blocks[0](x)
            monkeypatch.setattr(HostModuleInstance, "install_host", install_host_failing_once)
            with pytest.raises(Interrupt):
                m.transformer_blocks[1](hidden)
            active = engine._active
            assert active is not None and active.unit_idx == 0
            assert active.slot.checked_out
            assert engine.pool is not None
            idle = [s for s in engine.pool._slots if not s.checked_out]
            assert all(s.lease is not active.slot.lease for s in idle)

        assert all(param.device.type == "cpu" for param in m.parameters())


# ---------------------------------------------------------------------------
# Constructor leaves no GPU residency (cache_bytes is final, no pool yet)
# ---------------------------------------------------------------------------


class TestConstructedStateIsInactive:
    @CUDA
    def test_constructed_has_no_params_on_activation_device(self) -> None:
        m = _make_block_model(num_blocks=4, width=8)
        target = torch.device("cuda")
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            for p in m.parameters():
                assert p.device != target, f"constructed state leaked GPU residency: {p.shape}@{p.device}"
        finally:
            strategy.deactivate()

    def test_block_only_model_has_no_non_block_host(self) -> None:
        class BlockOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(4)])

        m = BlockOnly()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            assert resident_instance(strategy) is None
            assert strategy.cache_bytes > 0
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Buffer-only non-block module (e.g., a RoPE table)
# ---------------------------------------------------------------------------


class TestBufferOnlyNonBlock:
    @CUDA
    def test_buffer_only_non_block_module(self) -> None:
        class RopeTable(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.randn(8, 4))

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.rope = RopeTable()
                self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(4)])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            assert not m.rope.table.is_pinned()
            strategy.activate("cuda")
            assert m.rope.table.is_cuda
            strategy.deactivate()
            assert not m.rope.table.is_pinned()
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Shared storage handled inside one component
# ---------------------------------------------------------------------------


class TestSharedStorageLocalBehavior:
    def test_extra_name_for_block_param_is_allowed(
        self,
    ) -> None:
        block = nn.Linear(4, 4, bias=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([block, nn.Linear(4, 4, bias=False)])
                self.first_block = block

        m = M().requires_grad_(False)

        strat = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        strat.deactivate()

    def test_intra_block_tied_params_are_preserved(self) -> None:
        shared_0 = torch.randn(8, 8)
        shared_1 = torch.randn(8, 8)

        class TiedBlock(nn.Module):
            def __init__(self, shared: torch.Tensor):
                super().__init__()
                self.attn_q = nn.Linear(8, 8, bias=False)
                self.attn_k = nn.Linear(8, 8, bias=False)
                self.attn_q.weight = nn.Parameter(shared, requires_grad=False)
                self.attn_k.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([TiedBlock(shared_0), TiedBlock(shared_1)])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            for block in m.transformer_blocks:
                assert block.attn_q.weight is block.attn_k.weight
        finally:
            strategy.deactivate()

    def test_intra_block_tied_buffers_are_preserved(self) -> None:
        class TiedBufBlock(nn.Module):
            def __init__(self):
                super().__init__()
                shared = torch.randn(8)
                view_a = shared.view(8)
                view_b = shared.view(8)
                self.register_buffer("buf_a", view_a)
                self.register_buffer("buf_b", view_b)
                self.weight = nn.Parameter(torch.randn(2), requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([TiedBufBlock(), TiedBufBlock()])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            for block in m.transformer_blocks:
                assert block.buf_a is block.buf_b
                assert not block.buf_a.is_pinned()
        finally:
            strategy.deactivate()

    def test_intra_block_same_buffer_object_is_preserved(self) -> None:
        class TiedBufBlock(nn.Module):
            def __init__(self):
                super().__init__()
                shared = torch.randn(8)
                self.register_buffer("buf_a", shared)
                self.register_buffer("buf_b", shared)
                self.weight = nn.Parameter(torch.randn(2), requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([TiedBufBlock(), TiedBufBlock()])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            for block in m.transformer_blocks:
                assert block.buf_a is block.buf_b
                assert not block.buf_a.is_pinned()
        finally:
            strategy.deactivate()

    def test_non_block_internal_tied_works(self) -> None:
        embed = nn.Embedding(16, 8)
        head = nn.Linear(8, 16, bias=False)
        head.weight = embed.weight  # standard tie_weights() pattern

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = embed
                self.transformer_blocks = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(4)])
                self.head = head

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            assert m.embed.weight is m.head.weight
            non_block = resident_instance(strategy)
            assert non_block is not None
            assert set(non_block.params) == {"embed.weight", "head.weight"}
            assert len({id(non_block.params[name]) for name in non_block.params}) == 1
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Direct-parent state handling
# ---------------------------------------------------------------------------


class TestDirectParentStateHandled:
    def test_direct_frozen_param_on_root_is_host(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale_shift = nn.Parameter(torch.randn(4), requires_grad=False)
                self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(4)])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            assert not m.scale_shift.is_pinned()
        finally:
            strategy.deactivate()

    def test_nested_offload_paths_with_direct_root_param(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.root_param = nn.Parameter(torch.randn(4), requires_grad=False)
                self.encoder = nn.Module()
                self.encoder.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(4)])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("encoder.blocks",)),
        )
        try:
            assert not m.root_param.is_pinned()
        finally:
            strategy.deactivate()

    def test_direct_buffer_on_root_is_host(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.randn(8))
                self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(4)])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            assert not m.table.is_pinned()
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Block-internal buffers
# ---------------------------------------------------------------------------


class TestBlockBuffersHost:
    @staticmethod
    def _make_tied_buffer_model(
        num_blocks: int = 2,
        *,
        device: torch.device | str = "cpu",
    ) -> nn.Module:
        class TiedBufferBlock(nn.Module):
            def __init__(self):
                super().__init__()
                shared = torch.randn(8, device=device)
                self.register_buffer("buf_a", shared.view(8))
                self.register_buffer("buf_b", shared.view(8))
                self.weight = nn.Parameter(
                    torch.randn(2, device=device),
                    requires_grad=False,
                )

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(TiedBufferBlock() for _ in range(num_blocks))

        return M()

    def test_block_buffer_clone_is_host(self) -> None:
        class BlockWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.register_buffer("table", torch.randn(8))

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([BlockWithBuffer() for _ in range(4)])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            for block in m.transformer_blocks:
                assert not block.table.is_pinned(), (
                    "block buffer must be host for honest cache_bytes and to avoid silently-synchronous H2D copies"
                )
        finally:
            strategy.deactivate()

    def test_constructor_does_not_call_module_to(self) -> None:
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2), requires_grad=False)
                self.register_buffer("table", torch.randn(2))

            def to(self, *args, **kwargs):
                raise AssertionError("constructor must capture directly")

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([Block(), Block()])

            def to(self, *args, **kwargs):
                raise AssertionError("constructor must capture directly")

        m = M()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            for block in m.transformer_blocks:
                assert not block.weight.is_pinned()
                assert not block.table.is_pinned()
        finally:
            strategy.deactivate()

    @CUDA
    def test_cuda_origin_tied_block_buffers_stay_tied(self) -> None:
        m = self._make_tied_buffer_model(device="cuda")
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(strategy)
        try:
            for block in m.transformer_blocks:
                assert block.buf_a is block.buf_b
                assert not block.buf_a.is_pinned()

            strategy.activate("cuda")
            engine._before_unit_forward(0)
            resident = m.transformer_blocks[0]
            assert resident.buf_a is resident.buf_b
            assert resident.buf_a.is_cuda

            strategy.deactivate()
            for block in m.transformer_blocks:
                assert block.buf_a is block.buf_b
                assert block.buf_a.device.type == "cpu"
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Block layout check rejects heterogeneous block layouts
# ---------------------------------------------------------------------------


class TestUnitLayoutCompatibility:
    """Offloaded units in one list may differ in layout, shape, dtype,
    or parameter/buffer names because ElasticVramPool keys reusable GPU
    targets by per-unit layout signature."""

    @staticmethod
    def _signatures(offloader: ModelOffloader) -> list[object]:
        return [
            plan_target_signature(instance.resolve_load_plan())
            for instance in offloaded_units(offloader)
        ]

    def test_shape_mismatch_is_supported(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False), nn.Linear(8, 8, bias=False)])

        m = M().requires_grad_(False)
        offloader = _make_units_offloader(list(m.blocks))
        try:
            assert len(set(self._signatures(offloader))) == 2
        finally:
            offloader.deactivate()

    def test_dtype_mismatch_is_supported(self) -> None:
        b0 = nn.Linear(4, 4, bias=False).to(torch.float16).requires_grad_(False)
        b1 = nn.Linear(4, 4, bias=False).to(torch.bfloat16).requires_grad_(False)
        offloader = _make_units_offloader([b0, b1])
        try:
            assert len(set(self._signatures(offloader))) == 2
        finally:
            offloader.deactivate()

    def test_param_alias_topology_is_supported(self) -> None:
        class TiedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                shared = nn.Linear(4, 4, bias=False)
                shared.weight.requires_grad_(False)
                self.q = shared
                self.k = shared

        class UntiedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(4, 4, bias=False)
                self.k = nn.Linear(4, 4, bias=False)
                self.q.weight.requires_grad_(False)
                self.k.weight.requires_grad_(False)

        offloader = _make_units_offloader([TiedBlock(), UntiedBlock()])
        try:
            assert len(set(self._signatures(offloader))) == 2
        finally:
            offloader.deactivate()

    @CUDA
    def test_aliased_block_module_installs_single_hook(self) -> None:
        shared = nn.Linear(4, 4, bias=False).requires_grad_(False)
        offloader = _make_units_offloader([shared, shared])
        try:
            with activated_model(offloader, "cuda"):
                assert len(shared._forward_pre_hooks) == 1
        finally:
            offloader.deactivate()

    def test_buffer_shape_mismatch_is_supported(self) -> None:
        class Block(nn.Module):
            def __init__(self, buffer_size: int):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)
                self.register_buffer("table", torch.randn(buffer_size))

        offloader = _make_units_offloader([Block(4), Block(8)])
        try:
            assert len(set(self._signatures(offloader))) == 2
        finally:
            offloader.deactivate()

    def test_buffer_stride_mismatch_is_supported(self) -> None:
        class Block(nn.Module):
            def __init__(self, table: torch.Tensor):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)
                self.register_buffer("table", table)

        contiguous = torch.randn(2, 3)
        non_contiguous = torch.randn(3, 2).t()
        assert contiguous.shape == non_contiguous.shape
        assert contiguous.stride() != non_contiguous.stride()

        offloader = _make_units_offloader([Block(contiguous), Block(non_contiguous)])
        try:
            assert len(self._signatures(offloader)) == 2
        finally:
            offloader.deactivate()

    def test_heterogeneous_param_names_are_supported(self) -> None:
        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = nn.Parameter(torch.randn(4, 4), requires_grad=False)

        class B(nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = nn.Parameter(torch.randn(4, 4), requires_grad=False)

        offloader = _make_units_offloader([A(), B()])
        try:
            assert len(set(self._signatures(offloader))) == 2
            assert [set(u.params) for u in offloaded_units(offloader)] == [{"foo"}, {"bar"}]
        finally:
            offloader.deactivate()

    def test_trainable_block_param_raises(self) -> None:
        block_0 = nn.Linear(4, 4, bias=False).requires_grad_(False)
        block_1 = nn.Linear(4, 4, bias=False).requires_grad_(True)

        with pytest.raises(ValueError, match="frozen"):
            _make_units_offloader([block_0, block_1])

    def test_heterogeneous_buffer_names_are_supported(self) -> None:
        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)
                self.register_buffer("foo", torch.randn(4))

        class B(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)
                self.register_buffer("bar", torch.randn(4))

        offloader = _make_units_offloader([A(), B()])
        try:
            assert len(set(self._signatures(offloader))) == 2
            assert [set(u.buffers) for u in offloaded_units(offloader)] == [{"foo"}, {"bar"}]
        finally:
            offloader.deactivate()

    def test_validation_failure_leaves_model_unmutated(self) -> None:
        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = nn.Parameter(torch.randn(4, 4), requires_grad=False)

        class B(nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = nn.Parameter(torch.randn(4, 4), requires_grad=True)

        blocks = [A(), B()]
        original_params = [blocks[0].foo, blocks[1].bar]
        original_host = [p.is_pinned() for p in original_params]

        with pytest.raises(ValueError, match="frozen"):
            _make_units_offloader(blocks)

        assert blocks[0].foo is original_params[0]
        assert blocks[1].bar is original_params[1]
        for param, orig_pin in zip(
            original_params,
            original_host,
            strict=True,
        ):
            assert param.is_pinned() == orig_pin, "parameter pin status changed despite validation failure"


# ---------------------------------------------------------------------------
# Cleanup unwinding when a step fails during deactivate
# ---------------------------------------------------------------------------


class TestDeactivateCleanupUnwinding:
    @CUDA
    def test_resident_restore_failure_still_clears_unit_hooks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        engine = residency_engine(strategy)
        resident = resident_instance(strategy)
        assert resident is not None
        original_install_host = HostModuleInstance.install_host

        def broken_install_host(instance: HostModuleInstance) -> None:
            original_install_host(instance)
            if instance is resident:
                raise RuntimeError("simulated resident restore failure")

        monkeypatch.setattr(HostModuleInstance, "install_host", broken_install_host)
        try:
            strategy.activate("cuda")
            assert m.embed.weight.is_cuda

            with pytest.raises(RuntimeError, match="simulated resident restore failure"):
                strategy.deactivate()

            assert not m.embed.weight.is_pinned()
            assert not engine._unit_hooks
            assert engine.active_device is None
        finally:
            monkeypatch.setattr(HostModuleInstance, "install_host", original_install_host)
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Unit name selection & override routing
# ---------------------------------------------------------------------------


class TestUnitNameSelection:
    def test_units_expose_owned_local_names(self) -> None:
        m = _make_block_model()
        offloader = _make_units_offloader(list(m.transformer_blocks), blocks_path="transformer_blocks")
        try:
            assert [list(u.params) for u in offloaded_units(offloader)] == [["weight"]] * len(m.transformer_blocks)
            assert [list(u.buffers) for u in offloaded_units(offloader)] == [[]] * len(m.transformer_blocks)
        finally:
            offloader.deactivate()

    def test_partition_exposes_addressable_param_names(self) -> None:
        m = _make_block_model()
        offloader = _make_units_offloader(list(m.transformer_blocks), blocks_path="transformer_blocks")
        try:
            assert offloader.param_names == frozenset(
                {f"transformer_blocks.{i}.weight" for i in range(len(m.transformer_blocks))}
            )
        finally:
            offloader.deactivate()

    def test_store_binds_compatible_model_and_preserves_host_resident_weights(self) -> None:
        torch.manual_seed(0)
        prototype_blocks = [nn.Linear(8, 8, bias=False).requires_grad_(False) for _ in range(4)]
        prototype = _make_block_list_model(prototype_blocks, "transformer_blocks")
        target = _make_block_model()
        target_embed = target.embed.weight.detach().clone()

        store = ModelStore.from_module(prototype)
        offloader = store.bind(
            target,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            assert set(offloader._partition._unit_param_index) == {
                f"transformer_blocks.{i}.weight" for i in range(len(target.transformer_blocks))
            }
            for prototype_block, target_block in zip(
                prototype.transformer_blocks,
                target.transformer_blocks,
                strict=True,
            ):
                assert not target_block.weight.is_pinned()
                torch.testing.assert_close(target_block.weight, prototype_block.weight)
            torch.testing.assert_close(target.embed.weight, target_embed)
            assert not target.embed.weight.is_pinned()
        finally:
            offloader.deactivate()

    def test_model_offloader_exposes_all_managed_buffer_names(self) -> None:
        class BufferBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
                self.register_buffer("scale", torch.randn(2))

        class BufferModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.root_weight = nn.Parameter(
                    torch.randn(2, 2),
                    requires_grad=False,
                )
                self.register_buffer("root_scale", torch.randn(2))
                self.blocks = nn.ModuleList([BufferBlock(), BufferBlock()])

        model = BufferModel()
        strategy = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",)),
        )
        try:
            assert strategy.param_names == frozenset(
                {
                    "root_weight",
                    "blocks.0.weight",
                    "blocks.1.weight",
                }
            )
            assert strategy.buffer_names == frozenset(
                {
                    "root_scale",
                    "blocks.0.scale",
                    "blocks.1.scale",
                }
            )
            assert [set(u.buffers) for u in offloaded_units(strategy)] == [{"scale"}, {"scale"}]
            assert offloaded_units(strategy)[0].module is model.blocks[0]
        finally:
            strategy.deactivate()

    def test_partition_routes_parameter_override_by_name(self) -> None:
        m = _make_block_model()
        offloader = _make_units_offloader(list(m.transformer_blocks), blocks_path="transformer_blocks")

        def update(_param: nn.Parameter) -> None:
            pass

        transform = CallbackParameterTransform(update)

        try:
            _res, unit_ovrs = offloader._partition.route_overrides(
                {
                    "transformer_blocks.1.weight": ParameterOverride(
                        update=transform
                    )
                }
            )
            plans = [
                u.resolve_load_plan(ovr)
                for u, ovr in zip(offloaded_units(offloader), unit_ovrs, strict=True)
            ]
            assert plans[1].loads["weight"].update is transform
            assert plans[0].loads["weight"].update is None
        finally:
            offloader.deactivate()

    def test_partition_rejects_unknown_parameter_override_name(self) -> None:
        m = _make_block_model()
        offloader = _make_units_offloader(list(m.transformer_blocks), blocks_path="transformer_blocks")
        try:
            with pytest.raises(ValueError, match="unmanaged names"):
                offloader._partition.route_overrides(
                    {
                        "transformer_blocks.10.weight": ParameterOverride(
                            update=CallbackParameterTransform(lambda _param: None)
                        )
                    }
                )
        finally:
            offloader.deactivate()

    def test_model_offloader_partitions_unit_and_resident_names(
        self,
    ) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            non_block = resident_instance(strategy)
            assert non_block is not None

            assert [list(u.params) for u in offloaded_units(strategy)] == [["weight"]] * len(m.transformer_blocks)
            assert set(non_block.params) == {"embed.weight", "head.weight"}
            assert all(not name.startswith("transformer_blocks.") for name in non_block.params)
        finally:
            strategy.deactivate()


class TestResidencyEngineActivateTwice:
    @CUDA
    def test_double_activate_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        offloader = ModelOffloader.from_module(
            m,
            OffloadConfig(offload_paths=("transformer_blocks",)),
        )
        try:
            offloader.activate("cuda")
            engine = residency_engine(offloader)
            with pytest.raises(RuntimeError, match="already active"):
                engine.activate(torch.device("cuda"))
        finally:
            offloader.deactivate()


# ---------------------------------------------------------------------------
# Sparse offloaded unit lists
# ---------------------------------------------------------------------------


class TestSparseBlockStore:
    @staticmethod
    def _blocks(n: int = 4, width: int = 8) -> list[nn.Module]:
        blocks = [nn.Linear(width, width, bias=False) for _ in range(n)]
        for block in blocks:
            block.requires_grad_(False)
        return blocks

    def test_empty_blocks_are_skipped(self) -> None:
        real = self._blocks(n=2)
        for block in real:
            block.register_buffer("scale", torch.ones(1))
        mixed: list[nn.Module] = [nn.Module(), real[0], nn.Module(), real[1]]
        model = _make_block_list_model(mixed, "blocks")
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",)),
        )
        try:
            assert offloader.param_names == frozenset({"blocks.1.weight", "blocks.3.weight"})
            assert [u.module for u in offloaded_units(offloader)] == real
            assert offloader.buffer_names == frozenset({"blocks.1.scale", "blocks.3.scale"})
        finally:
            offloader.deactivate()

    def test_bind_rejects_unmanaged_nonempty_blocks(self) -> None:
        store_model = _make_block_list_model(self._blocks(n=3), "blocks")
        bind_model = _make_block_list_model(self._blocks(n=5), "blocks")
        store = ModelStore.from_module(store_model)
        with pytest.raises(ValueError, match="does not occupy"):
            store.bind(bind_model, OffloadConfig(offload_paths=("blocks",)))

    def test_bind_allows_empty_unoccupied_blocks(self) -> None:
        real = self._blocks(n=2)
        store_model = _make_block_list_model(real, "blocks")
        bind_model = _make_block_list_model(
            [*[copy.deepcopy(block) for block in real], nn.Module()],
            "blocks",
        )
        store = ModelStore.from_module(store_model)
        offloader = store.bind(bind_model, OffloadConfig(offload_paths=("blocks",)))
        try:
            assert offloader.param_names == frozenset({"blocks.0.weight", "blocks.1.weight"})
        finally:
            offloader.deactivate()

    def test_all_empty_blocks_raise(self) -> None:
        model = _make_block_list_model([nn.Module(), nn.Module()], "blocks")
        with pytest.raises(ValueError, match="no streamable blocks"):
            ModelOffloader.from_module(
                model,
                OffloadConfig(offload_paths=("blocks",)),
            )


class TestFrozenContract:
    def test_block_store_capture_and_bind_are_atomic_on_rejection(self) -> None:
        block_0 = nn.Linear(4, 4, bias=False).requires_grad_(False)
        block_1 = nn.Linear(4, 4, bias=False).requires_grad_(True)
        orig_weight = block_0.weight
        orig_ptr = block_0.weight.untyped_storage().data_ptr()
        model = _make_block_list_model([block_0, block_1], "blocks")

        with pytest.raises(ValueError, match="frozen"):
            ModelOffloader.from_module(
                model,
                OffloadConfig(offload_paths=("blocks",)),
            )

        assert block_0.weight is orig_weight
        assert block_0.weight.untyped_storage().data_ptr() == orig_ptr

        frozen_proto = _make_block_list_model(
            [nn.Linear(4, 4, bias=False).requires_grad_(False) for _ in range(2)],
            "blocks",
        )
        store = ModelStore.from_module(frozen_proto)
        with pytest.raises(ValueError, match="frozen"):
            store.bind(model, OffloadConfig(offload_paths=("blocks",)))

        assert block_0.weight is orig_weight
        assert block_0.weight.untyped_storage().data_ptr() == orig_ptr

    def test_model_offloader_and_partition_reject_without_partial_mutation(self) -> None:
        non_block_trainable = _make_block_model(num_blocks=2)
        orig_block_0 = non_block_trainable.transformer_blocks[0].weight
        orig_block_0_ptr = orig_block_0.untyped_storage().data_ptr()
        non_block_trainable.embed.weight.requires_grad_(True)
        config = OffloadConfig(
            offload_paths=("transformer_blocks", "head"),
        )
        with pytest.raises(ValueError, match="frozen"):
            ModelOffloader.from_module(non_block_trainable, config)
        assert non_block_trainable.transformer_blocks[0].weight is orig_block_0
        assert non_block_trainable.transformer_blocks[0].weight.untyped_storage().data_ptr() == orig_block_0_ptr

        in_block_trainable = _make_block_model(num_blocks=2)
        orig_embed = in_block_trainable.embed.weight
        orig_embed_ptr = orig_embed.untyped_storage().data_ptr()
        in_block_trainable.transformer_blocks[1].weight.requires_grad_(True)
        with pytest.raises(ValueError, match="frozen"):
            ModelOffloader.from_module(in_block_trainable, config)
        assert in_block_trainable.embed.weight is orig_embed
        assert in_block_trainable.embed.weight.untyped_storage().data_ptr() == orig_embed_ptr

        proto = _make_block_model(num_blocks=2)
        store = ModelStore.from_module(proto)
        bad_target = _make_block_model(num_blocks=2)
        bad_embed = bad_target.embed.weight
        bad_embed_ptr = bad_embed.untyped_storage().data_ptr()
        bad_head = bad_target.head.weight
        bad_head_ptr = bad_head.untyped_storage().data_ptr()
        bad_b0 = bad_target.transformer_blocks[0].weight
        bad_b0_ptr = bad_b0.untyped_storage().data_ptr()
        bad_target.transformer_blocks[1].weight.requires_grad_(True)

        with pytest.raises(ValueError, match="frozen"):
            store.bind(bad_target, config)

        assert bad_target.embed.weight is bad_embed
        assert bad_target.embed.weight.untyped_storage().data_ptr() == bad_embed_ptr
        assert bad_target.head.weight is bad_head
        assert bad_target.head.weight.untyped_storage().data_ptr() == bad_head_ptr
        assert bad_target.transformer_blocks[0].weight is bad_b0
        assert bad_target.transformer_blocks[0].weight.untyped_storage().data_ptr() == bad_b0_ptr


def test_adapter_mode_public_api_removed() -> None:
    import piper_offload
    from piper_offload import ModelCache, ModelSpec

    assert "AdapterMode" not in piper_offload.__all__
    assert not hasattr(piper_offload, "AdapterMode")

    model = _make_block_model(num_blocks=1)
    offloader = ModelOffloader.from_module(model)
    for bad_mode in ("merge", "routed"):
        with pytest.raises(TypeError, match="adapter_mode"):
            cast(Any, offloader.activate)("cpu", adapter_mode=bad_mode)
    assert offloader.active_device is None

    cache = ModelCache()
    spec = ModelSpec(key="model", estimated_cache_bytes=1024, factory=lambda: _make_block_model(num_blocks=1))
    for bad_mode in ("merge", "routed"):
        with pytest.raises(TypeError, match="adapter_mode"):
            with cast(Any, cache.use)(spec, device="cpu", adapter_mode=bad_mode):
                pass
