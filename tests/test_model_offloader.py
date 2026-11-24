"""Tests for the block-streaming machinery in ``piper_offload``.

Covers ``ModelOffloader`` (the public composite),
and ``BlockComponent`` (the per-block-list primitive).

CUDA-only tests gate on availability. CPU activation is pass-through
over the host-backed state.
"""

import copy
import threading
from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.utils.checkpoint
from torch import nn

from piper_offload import (
    BlockComponent,
    BlockComponentStore,
    BlockMode,
    Adapter,
    ModelOffloader,
    ModelRuntimeFailedError,
    HostComponent,
    ResourceBinding,
    ResourceStore,
)
from piper_offload.composite_component import CompositeComponent, CompositeComponentStore
from piper_offload.host_module import HostModuleInstance, HostModuleStore, ParameterOverride
from piper_offload._transfer_priority import foreground_transfer
from piper_offload._working_set import WorkingSetState
from piper_offload.pin_manager import host_pin_manager
from piper_offload.streaming_runtime import _BlockTarget, _plan_target_signature, _wait_for_prefetch

from tests.conftest import (
    CallbackParameterTransform,
    Interrupt,
    activated_model,
    host_component,
    block_components,
    transient_components,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_block_component(
    blocks: Sequence[nn.Module],
    *,
    blocks_path: str = "blocks",
    block_mode: BlockMode = "streaming",
) -> BlockComponent:
    model = _make_block_list_model(blocks, blocks_path)
    store = BlockComponentStore.from_module(
        model,
        blocks_path=blocks_path,
    )
    return store.bind(model, block_mode=block_mode)


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
            block_paths=["transformer_blocks"],
        )
        try:
            assert isinstance(strategy, ResourceStore)
            assert isinstance(strategy, ResourceBinding)
            assert callable(strategy.activate)
            assert callable(strategy.deactivate)
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Constructor pins; cache_bytes is final immediately
# ---------------------------------------------------------------------------


class TestConstructorPins:
    def test_constructor_captures_blocks(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        try:
            # Block weights are host via registry replacement.
            for block in m.transformer_blocks:
                assert not block.weight.is_pinned()
            # Non-block (embed/head) also host via composed HostComponent.
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
            block_paths=["transformer_blocks"],
        )
        try:
            for _ in range(2):
                with activated_model(strategy, "cuda"):
                    assert strategy.model is m
                    assert strategy.active_device == expected
                    assert block_components(strategy)[0]._active_device == expected
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
            block_paths=["transformer_blocks"],
        )
        try:
            assert strategy.active_device is None
            strategy.deactivate()
            strategy.deactivate()
            host_block_params = [block.weight for block in m_off.transformer_blocks]
            with activated_model(strategy, "cpu") as cpu_model:
                assert strategy.active_device == torch.device("cpu")
                assert all(s._active_device == torch.device("cpu") for s in block_components(strategy))
                assert all(not s._runtime.acquired for s in block_components(strategy))
                assert all(
                    block.weight is host
                    for block, host in zip(
                        m_off.transformer_blocks,
                        host_block_params,
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
            block_paths=["transformer_blocks"],
        )
        try:
            strategy.activate("cpu")

            def fail_activate(device: torch.device | str | None = None) -> None:
                del device
                raise AssertionError("component activation should not run")

            assert host_component(strategy) is not None
            monkeypatch.setattr(
                host_component(strategy),
                "activate",
                fail_activate,
            )
            with pytest.raises(RuntimeError, match=r"already.*active"):
                strategy.activate("cpu")
        finally:
            strategy.deactivate()


class TestBlockComponentBackendActivation:
    def test_direct_cpu_activation_uses_host_backed_weights(self) -> None:
        torch.manual_seed(42)
        m_eager = _make_block_model(num_blocks=4, width=8)
        m = _make_block_model(num_blocks=4, width=8)
        m.load_state_dict(m_eager.state_dict())

        x = torch.randn(2, 8)
        with torch.no_grad():
            expected = m_eager(x)

        streamer = _make_block_component(
            blocks=list(m.transformer_blocks),
        )
        try:
            host_params = [block.weight for block in m.transformer_blocks]
            with streamer.use("cpu"):
                assert streamer._active_device == torch.device("cpu")
                assert not streamer._runtime.acquired
                streamer.release()
                streamer.acquire()
                assert all(
                    block.weight is host
                    for block, host in zip(
                        m.transformer_blocks,
                        host_params,
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
            streamer.deactivate()

    @CUDA
    @pytest.mark.parametrize("block_mode", ["streaming", "resident"])
    def test_cuda_working_set_can_release_and_reacquire(
        self,
        block_mode: BlockMode,
    ) -> None:
        torch.manual_seed(42)
        eager_blocks = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(3)]).requires_grad_(False)
        blocks = copy.deepcopy(eager_blocks)
        value = torch.randn(2, 8)
        with torch.no_grad():
            expected = value
            for block in eager_blocks:
                expected = block(expected)

        streamer = _make_block_component(
            blocks=list(blocks),
            block_mode=block_mode,
        )
        runtime = streamer._runtime
        try:
            streamer.activate(torch.device("cuda"))
            assert runtime.acquired

            streamer.release()
            streamer.release()
            assert streamer._active_device is not None
            assert streamer._active_device.type == "cuda"
            assert streamer._active_runtime is runtime
            assert not runtime.acquired
            assert all(block.weight.device.type == "cpu" for block in blocks)

            streamer.acquire()
            streamer.acquire()
            assert runtime.acquired
            with torch.no_grad():
                actual = value.cuda()
                for block in blocks:
                    actual = block(actual)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual.cpu(), expected)
        finally:
            streamer.deactivate()

    def test_acquire_requires_active_session(self) -> None:
        block = nn.Linear(4, 4, bias=False).requires_grad_(False)
        streamer = _make_block_component(blocks=[block])
        streamer.release()
        with pytest.raises(RuntimeError, match="active session"):
            streamer.acquire()

    def test_already_active_block_tracks_the_forward_stream(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class Lease:
            def __init__(self) -> None:
                self.used: list[object] = []

            def record_stream(self, stream: object) -> None:
                self.used.append(stream)

        streamer = _make_block_component(
            blocks=[nn.Linear(4, 4, bias=False).requires_grad_(False)],
        )
        runtime = streamer._runtime
        lease = Lease()
        forward_stream = object()
        runtime._device = torch.device("cuda")
        runtime._active = _BlockTarget(0, cast(Any, lease), installed=True)
        monkeypatch.setattr(runtime, "_submit_prefetch", lambda _idx: None)
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda _device: forward_stream,
        )
        try:
            runtime._before_block_forward(0)
            assert lease.used == [forward_stream]
        finally:
            runtime._active = None
            runtime._device = None
            streamer.deactivate()

    def test_direct_cpu_double_activate_raises(self) -> None:
        m = _make_block_model(num_blocks=2)
        streamer = _make_block_component(
            blocks=list(m.transformer_blocks),
        )
        try:
            streamer.activate(torch.device("cpu"))
            with pytest.raises(RuntimeError, match="already.*active"):
                streamer.activate(torch.device("cpu"))
        finally:
            streamer.deactivate()


# ---------------------------------------------------------------------------
# Cleanup contract: deactivate restores CPU state, drop refs to free
# ---------------------------------------------------------------------------


class TestCleanup:
    @CUDA
    def test_deactivate_restores_cpu_state(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
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
            block_paths=["transformer_blocks"],
        )
        strategy.deactivate()
        strategy.deactivate()  # no error

    @CUDA
    def test_deactivate_ends_the_composite_session(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        strategy.activate("cuda")
        assert strategy._composite._active
        strategy.deactivate()
        assert not strategy._composite._active

    @CUDA
    def test_drop_strategy_without_deactivate_does_not_cycle(self) -> None:
        # Regression: BlockComponent's forward-pre-hook closure used to
        # capture `self`, creating a refcount cycle:
        #     layer → _forward_pre_hooks → closure → streamer →
        #     _blocks → layer
        # Refcount-based GC couldn't break it; only the periodic cycle
        # collector could. We use weakref.ref(self) in the closure to
        # break the cycle so dropping the strategy frees it
        # immediately even if the user skipped deactivate().
        import gc
        import weakref

        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        strategy.activate("cuda")  # installs hooks; no deactivate
        streamer = block_components(strategy)[0]
        streamer_ref = weakref.ref(streamer)

        # Disable cycle collector BEFORE dropping refs to prove the
        # cleanup is purely refcount-based, not cycle-collector-based.
        gc.disable()
        try:
            del strategy, streamer
            assert streamer_ref() is None
        finally:
            gc.enable()

    @CUDA
    def test_orphaned_hooks_noop_after_strategy_dropped(self) -> None:
        # When the strategy is dropped without deactivate, the hooks
        # remain installed on the model but the weakref inside is
        # dead. The hook must no-op cleanly so the model still works
        # for forward. (Slow path: blocks may be on GPU or CPU
        # depending on eviction state at drop-time.)
        torch.manual_seed(0)
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        strategy.activate("cuda")
        # Drop without deactivate.
        del strategy

        # Forward through the model — orphaned hooks should no-op.
        # (Some blocks resident on GPU, some CPU. PyTorch will
        # auto-promote inputs to match param device for each block.)
        x = torch.randn(2, 8, device="cpu")
        with torch.no_grad():
            # Should not raise. We don't assert correctness of the
            # output (params are in a mixed GPU/CPU state) —
            # just that the hooks don't crash.
            try:
                _ = m(x)
            except Exception as e:
                # If the cuda blocks fail because of input device
                # mismatch, that's expected. The hook itself must not
                # crash with AttributeError or NoneType errors.
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
            block_paths=["transformer_blocks"],
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
            block_paths=["transformer_blocks"],
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
            block_paths=["transformer_blocks"],
        )
        strategy.activate("cuda")
        blocks = list(m.transformer_blocks)
        strategy.deactivate()
        for block in blocks:
            assert len(block._forward_pre_hooks) == 0


# ---------------------------------------------------------------------------
# ModelOffloader-owned transient residency schedule
# ---------------------------------------------------------------------------


class TestTransientResidency:
    def test_transient_paths_partition_from_resident_state(self) -> None:
        model = _make_block_model()
        model.embed.register_buffer("table", torch.randn(8))
        offloader = ModelOffloader.from_module(
            model,
            block_paths=["transformer_blocks"],
            transient_paths=["embed", "head"],
        )
        try:
            components = dict(transient_components(offloader))
            assert components["embed"].param_names == {"embed.weight"}
            assert components["embed"].buffer_names == {"embed.table"}
            assert components["head"].param_names == {"head.weight"}
            assert host_component(offloader) is None
        finally:
            offloader.deactivate()

    def test_capture_and_rebind_include_all_groups(self) -> None:
        source = _make_block_model(num_blocks=2)
        source.tail = nn.ModuleList([nn.Identity(), nn.Linear(8, 8, bias=False)]).requires_grad_(False)
        source.embed.register_buffer("scale", torch.ones(8))
        target = copy.deepcopy(source)
        for tensor in target.state_dict().values():
            tensor.zero_()

        paths = {
            "block_paths": ["transformer_blocks"],
            "transient_block_paths": ["tail"],
            "transient_paths": ["embed"],
        }
        expected_bytes = sum(t.nbytes for t in source.state_dict().values())
        offloader = ModelOffloader.from_module(copy.deepcopy(source), **paths)
        assert offloader.cache_bytes == expected_bytes

        store = CompositeComponentStore.from_module(source, **paths)
        assert store.cache_bytes == expected_bytes
        bound = store.bind(target)
        torch.testing.assert_close(target.state_dict(), source.state_dict())
        assert bound.resident.param_names == {"head.weight"}
        (path, embed), = bound.transient
        assert path == "embed"
        assert embed.buffer_names == {"embed.scale"}
        persistent, tail = bound.blocks
        assert persistent.blocks == tuple(target.transformer_blocks)
        assert tail.blocks == (target.tail[1],)
        assert bound.transient_blocks == (tail,)

    def test_cpu_activation_stays_eager(self) -> None:
        model = _make_block_model()
        offloader = ModelOffloader.from_module(
            model,
            transient_block_paths=["transformer_blocks"],
            transient_paths=["embed", "head"],
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
    def test_paths_and_streaming_release_at_their_own_boundaries(self) -> None:
        model = _make_block_model(num_blocks=3)
        offloader = ModelOffloader.from_module(
            model,
            transient_block_paths=["transformer_blocks"],
            transient_paths=["embed", "head"],
        )
        components = dict(transient_components(offloader))
        embed = components["embed"]
        head = components["head"]
        runtime = block_components(offloader)[0]._runtime
        block_entry_states: list[tuple[bool, bool, bool]] = []
        head_entry_states: list[tuple[bool, bool, bool]] = []
        root_states: list[tuple[bool, bool, bool]] = []
        block_handle = model.transformer_blocks[0].register_forward_pre_hook(
            lambda _module, _args: block_entry_states.append(
                (embed._lease is not None, runtime.acquired, head._lease is not None)
            )
        )
        head_handle = model.head.register_forward_pre_hook(
            lambda _module, _args: head_entry_states.append(
                (embed._lease is not None, runtime.acquired, head._lease is not None)
            )
        )
        root_handle = model.register_forward_hook(
            lambda _module, _args, _output: root_states.append(
                (embed._lease is not None, runtime.acquired, head._lease is not None)
            )
        )
        try:
            with activated_model(offloader, "cuda"):
                value = torch.randn(2, 8, device="cuda")
                with torch.inference_mode():
                    first = model(value).clone()
                    second = model(value).clone()
                torch.cuda.synchronize()

                assert block_entry_states == [
                    (False, True, True),
                    (False, True, True),
                ]
                assert head_entry_states == [
                    (False, False, True),
                    (False, False, True),
                ]
                assert root_states == [
                    (False, False, False),
                    (False, False, False),
                ]
                assert embed._lease is not None
                assert runtime.acquired
                assert head._lease is not None
                torch.testing.assert_close(second, first, rtol=0, atol=0)
        finally:
            block_handle.remove()
            head_handle.remove()
            root_handle.remove()
            offloader.deactivate()

        assert not model._forward_hooks
        assert not model.embed._forward_hooks
        assert not model.head._forward_hooks
        assert not model.transformer_blocks[-1]._forward_hooks

    @CUDA
    def test_transient_path_records_its_forward_stream_before_release(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _make_block_model()
        offloader = ModelOffloader.from_module(
            model,
            block_paths=["transformer_blocks"],
            transient_paths=["embed"],
        )
        embed = dict(transient_components(offloader))["embed"]
        recorded_streams: list[torch.cuda.Stream] = []
        original_record_stream = embed.record_stream

        def record_stream(stream: torch.cuda.Stream) -> None:
            recorded_streams.append(stream)
            original_record_stream(stream)

        monkeypatch.setattr(embed, "record_stream", record_stream)
        side_stream = torch.cuda.Stream()
        try:
            with activated_model(offloader, "cuda"):
                with torch.cuda.stream(side_stream):
                    model.embed(torch.randn(2, 8, device="cuda"))

                assert recorded_streams[-1] == side_stream
                assert embed._lease is None
        finally:
            offloader.deactivate()

    @CUDA
    def test_activation_inside_inference_mode_keeps_targets_mutable(self) -> None:
        model = _make_block_model(num_blocks=3)
        offloader = ModelOffloader.from_module(
            model,
            transient_block_paths=["transformer_blocks"],
            transient_paths=["embed"],
        )
        embed = dict(transient_components(offloader))["embed"]
        runtime = block_components(offloader)[0]._runtime
        with torch.inference_mode():
            with activated_model(offloader, "cuda"):
                model(torch.randn(2, 8, device="cuda"))
                model(torch.randn(2, 8, device="cuda"))
                assert embed._lease is not None
                assert runtime.acquired

    @CUDA
    def test_persistent_and_transient_block_paths_schedule_independently(
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
        offloader = ModelOffloader.from_module(
            model,
            block_paths=["first_blocks"],
            transient_block_paths=["second_blocks"],
        )
        first, second = block_components(offloader)
        first_runtime = first._runtime
        second_runtime = second._runtime
        second_entry_states: list[tuple[bool, bool]] = []
        suffix_states: list[tuple[bool, bool]] = []
        second_handle = model.second_blocks[0].register_forward_pre_hook(
            lambda _module, _args: second_entry_states.append((first_runtime.acquired, second_runtime.acquired))
        )
        suffix_handle = model.head.register_forward_pre_hook(
            lambda _module, _args: suffix_states.append((first_runtime.acquired, second_runtime.acquired))
        )
        try:
            with activated_model(offloader, "cuda"):
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
                assert second_entry_states == [(True, True)]
                assert suffix_states == [(True, False)]
                assert first_runtime.acquired
                assert second_runtime.acquired

                model.run_first = False
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
                assert second_entry_states[-1] == (True, True)
                assert suffix_states[-1] == (True, False)
                assert first_runtime.acquired
                assert second_runtime.acquired
        finally:
            second_handle.remove()
            suffix_handle.remove()
            offloader.deactivate()

        assert not model._forward_hooks
        assert not model.first_blocks[-1]._forward_hooks
        assert not model.second_blocks[-1]._forward_hooks

    @CUDA
    def test_forward_failure_leaves_released_runtime_safe_to_deactivate(
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
            transient_block_paths=["transformer_blocks"],
            transient_paths=["embed"],
        )
        embed = dict(transient_components(offloader))["embed"]
        runtime = block_components(offloader)[0]._runtime
        try:
            offloader.activate("cuda")
            with pytest.raises(RuntimeError, match="simulated suffix failure"):
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
            assert not runtime.acquired
            assert embed._lease is None
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
                assert runtime.acquired
                assert embed._lease is not None
        finally:
            offloader.deactivate()

    @CUDA
    def test_failed_reacquire_is_retried_by_next_forward(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _make_block_model(num_blocks=3)
        offloader = ModelOffloader.from_module(
            model,
            transient_block_paths=["transformer_blocks"],
        )
        component = block_components(offloader)[0]
        runtime = component._runtime
        original_register_hooks = runtime._register_hooks
        failures = [RuntimeError("simulated reacquire failure")]

        def register_hooks_after_one_failure() -> None:
            if failures:
                raise failures.pop()
            original_register_hooks()

        value = torch.randn(2, 8, device="cuda")
        with activated_model(offloader, "cuda"), torch.inference_mode():
            expected = model(value).clone()
            monkeypatch.setattr(runtime, "_register_hooks", register_hooks_after_one_failure)
            # The forward computes, then its post-hook fails to reacquire
            # after staging block 0. That partial runtime must not count as
            # acquired, or the next forward would run without its hooks.
            with pytest.raises(RuntimeError, match="simulated reacquire failure"):
                model(value)
            assert not runtime.acquired
            assert component._pin_lease is None

            actual = model(value).clone()
            torch.cuda.synchronize()

            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert runtime.acquired

    @CUDA
    def test_hook_install_failure_rolls_back_activation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _make_block_model(num_blocks=3)
        offloader = ModelOffloader.from_module(
            model,
            transient_block_paths=["transformer_blocks"],
            transient_paths=["embed"],
        )
        embed = dict(transient_components(offloader))["embed"]
        runtime = block_components(offloader)[0]._runtime
        original_register = model.register_forward_hook

        def fail_register(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated root hook failure")

        monkeypatch.setattr(model, "register_forward_hook", fail_register)
        with pytest.raises(RuntimeError, match="simulated root hook failure"):
            offloader.activate("cuda")

        assert offloader.active_device is None
        assert not runtime.acquired
        assert embed._lease is None
        assert not model.embed._forward_hooks
        assert not model.transformer_blocks[-1]._forward_hooks

        monkeypatch.setattr(model, "register_forward_hook", original_register)
        try:
            with activated_model(offloader, "cuda"):
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
                assert runtime.acquired
                assert embed._lease is not None
        finally:
            offloader.deactivate()


# ---------------------------------------------------------------------------
# Traversal-aware prefetch
# ---------------------------------------------------------------------------


class TestTraversalPrefetch:
    """The runtime handles repeated forward and reverse traversals."""

    def _record_prefetches(
        self,
        streamer: BlockComponent,
    ) -> list[int]:
        """Wrap the block runtime's prefetch submission to record its index without disabling
        the actual prefetch (so on-GPU residency stays consistent)."""
        recorded: list[int] = []
        runtime = streamer._runtime
        original = runtime._submit_prefetch

        def record(idx: int) -> None:
            recorded.append(idx)
            original(idx)

        runtime._submit_prefetch = record  # type: ignore[method-assign]
        return recorded

    @CUDA
    def test_wraps_prefetch_at_iteration_boundary(self) -> None:
        # The second iteration's idx=0 hook must continue forward from the
        # previous iteration's final block.
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        streamer: BlockComponent = block_components(strategy)[0]

        with activated_model(strategy, "cuda"):
            recorded = self._record_prefetches(streamer)
            x = torch.randn(2, 8, device="cuda")
            m(x)  # iteration 1
            torch.cuda.synchronize()
            recorded.clear()
            m(x)  # iteration 2
            torch.cuda.synchronize()

        assert recorded == [1, 2, 3, 0], recorded

    @CUDA
    def test_transient_block_path_stops_prefetch_at_final_block(self) -> None:
        model = _make_block_model(num_blocks=4, width=8)
        offloader = ModelOffloader.from_module(
            model,
            transient_block_paths=["transformer_blocks"],
        )
        streamer = block_components(offloader)[0]

        with activated_model(offloader, "cuda"):
            recorded = self._record_prefetches(streamer)
            with torch.inference_mode():
                model(torch.randn(2, 8, device="cuda"))
            torch.cuda.synchronize()

        assert recorded == [1, 2, 3]

    @CUDA
    def test_continuous_backward_stays_backward(self) -> None:
        # Step-by-step reverse traversal (3, 2, 1, 0): each Δ=-1 is
        # below the wrap threshold, so direction inference yields
        # backward — not wrap-forward. Prefetch indices wrap modulo
        # N when the target falls below 0.
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        streamer: BlockComponent = block_components(strategy)[0]

        with activated_model(strategy, "cuda"):
            recorded = self._record_prefetches(streamer)
            x = torch.randn(2, 8, device="cuda")
            for idx in (3, 2, 1, 0):
                streamer.blocks[idx](x)
            torch.cuda.synchronize()

        # Per hook:
        #   idx=3 (last=-1 → forward init): prefetch 4 → wrap to 0
        #   idx=2 (last=3, Δ=-1 → backward): prefetch 1
        #   idx=1 (last=2, Δ=-1 → backward): prefetch 0
        #   idx=0 (last=1, Δ=-1 → backward): prefetch -1 → wrap to 3
        assert recorded == [0, 1, 0, 3], recorded

    @CUDA
    def test_two_iterations_match_eager_baseline(self) -> None:
        # Forward correctness: wrapped prefetch must not change which
        # block executes for which input. Two iterations through the
        # offloaded model must produce identical outputs to two
        # iterations of the same model on GPU without offloading.
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
            block_paths=["transformer_blocks"],
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
        # Running block 3 after block 1 skips the prefetched block 2. Its
        # lookahead target must return to the pool so block 3 can prefetch
        # block 0; otherwise every later block loads synchronously until
        # block 2 runs again.
        torch.manual_seed(0)
        m = _make_block_model(num_blocks=4, width=8)
        reference = copy.deepcopy(m).cuda()
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        streamer: BlockComponent = block_components(strategy)[0]
        runtime = streamer._runtime
        x = torch.randn(2, 8, device="cuda")

        with activated_model(strategy, "cuda"), torch.no_grad():
            actual = x
            expected = x
            for idx in (0, 1, 3):
                actual = streamer.blocks[idx](actual)
                expected = reference.transformer_blocks[idx](expected)
            assert runtime._active is not None and runtime._active.block_idx == 3
            lookahead = runtime._lookahead
            assert lookahead is not None and lookahead.block_idx == 0
            assert lookahead.prefetch is not None
            lookahead.prefetch.result()
            torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @CUDA
    def test_lookahead_waits_while_a_foreground_transfer_runs(self) -> None:
        # A collective issued inside a block, such as a context-parallel K/V
        # exchange, would otherwise share the host link with all of the next
        # block's weights (docs/streaming.md#yield-to-collectives).
        m = _make_block_model(num_blocks=2, width=8)
        strategy = ModelOffloader.from_module(m, block_paths=["transformer_blocks"])
        streamer: BlockComponent = block_components(strategy)[0]
        runtime = streamer._runtime
        x = torch.randn(2, 8, device="cuda")

        with activated_model(strategy, "cuda"), torch.no_grad():
            with foreground_transfer("cuda"):
                streamer.blocks[0](x)
                lookahead = runtime._lookahead
                assert lookahead is not None and lookahead.prefetch is not None
                with pytest.raises(TimeoutError):
                    lookahead.prefetch.result(timeout=0.2)
            lookahead.prefetch.result(timeout=30)

    @CUDA
    def test_activation_inside_inference_mode_keeps_pooled_targets_mutable(self) -> None:
        # Activation stages block 0 on the calling thread. The prefetch worker
        # later refills that pooled target outside the caller's inference
        # mode, which cannot update an inference tensor in place.
        torch.manual_seed(0)
        m = _make_block_model(num_blocks=4, width=8)
        reference = copy.deepcopy(m).cuda()
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
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
        # num_blocks=3 puts wrap_threshold at 1, the smallest meaningful
        # threshold (any |Δ|>1 wraps). Locks in current behavior at
        # this corner: forward continuation uses Δ=1 (no wrap), and
        # iteration boundary 2→0 has |Δ|=2>1 (wraps to forward).
        m = _make_block_model(num_blocks=3, width=8)
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        streamer: BlockComponent = block_components(strategy)[0]

        with activated_model(strategy, "cuda"):
            recorded = self._record_prefetches(streamer)
            x = torch.randn(2, 8, device="cuda")
            m(x)  # iteration 1
            torch.cuda.synchronize()
            recorded.clear()
            m(x)  # iteration 2
            torch.cuda.synchronize()

        # 3 blocks * 1 prefetch each:
        #   idx=0 (last=2, |Δ|=2>1 → wrap-forward): prefetch 1
        #   idx=1 (last=0, Δ=1 → forward):          prefetch 2
        #   idx=2 (last=1, Δ=1 → forward):          prefetch 3 → wrap to 0
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
            block_paths=["transformer_blocks"],
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
            block_paths=["transformer_blocks"],
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
    def test_empty_block_paths_disables_streaming(self) -> None:
        m = _make_block_model(num_blocks=4)
        strategy = ModelOffloader.from_module(m, block_paths=[])
        try:
            assert not block_components(strategy)
        finally:
            strategy.deactivate()

    def test_block_path_modes_must_be_disjoint(self) -> None:
        model = _make_block_model()
        with pytest.raises(
            ValueError,
            match="block_paths and transient_block_paths must be disjoint",
        ):
            ModelOffloader.from_module(
                model,
                block_paths=["transformer_blocks"],
                transient_block_paths=["transformer_blocks"],
            )

    def test_transient_block_paths_reject_aliased_modules(self) -> None:
        model = _make_block_model()
        shared = nn.Linear(8, 8, bias=False).requires_grad_(False)
        model.transformer_blocks = nn.ModuleList([shared, shared])

        with pytest.raises(
            ValueError,
            match="transient_block_paths does not support aliased block modules",
        ):
            ModelOffloader.from_module(
                model,
                transient_block_paths=["transformer_blocks"],
            )

        assert shared.weight.device.type == "cpu"

    def test_block_paths_resolving_to_non_modulelist_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(TypeError, match="nn.ModuleList"):
            ModelOffloader.from_module(
                m,
                block_paths=["embed"],  # an nn.Linear, not a ModuleList
            )


# ---------------------------------------------------------------------------
# ResourceCache integration: cache_bytes is final at construction
# ---------------------------------------------------------------------------


class TestResourceCacheIntegration:
    def test_model_spec_propagates_transient_paths(self) -> None:
        from piper_offload import ModelSpec

        spec = ModelSpec(
            key="transient",
            estimated_cache_bytes=1024,
            factory=_make_block_model,
            block_paths=("transformer_blocks",),
            transient_paths=("embed", "head"),
        )

        offloader = spec.build_store()
        try:
            assert [path for path, _component in transient_components(offloader)] == ["embed", "head"]
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
            block_paths=("transformer_blocks",),
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
            block_paths=("transformer_blocks",),
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
            block_paths=("transformer_blocks",),
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
            block_paths=("transformer_blocks",),
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
    def _offloader_with_leftover() -> tuple[ModelOffloader, BlockComponent, HostComponent]:
        model = _make_block_model()
        offloader = ModelOffloader.from_module(
            model,
            block_paths=["transformer_blocks"],
        )
        streamer = block_components(offloader)[0]
        resident = host_component(offloader)
        assert resident is not None
        # Stands in for a release that stopped partway, keeping the host
        # leases its unfinished transfers read.
        streamer._pin_lease = host_pin_manager.acquire([], pin=False)
        streamer._working_set_state = WorkingSetState.CHANGING
        assert offloader.failed
        return offloader, streamer, resident

    def test_cache_retains_model_until_cleanup_succeeds(self, monkeypatch, request) -> None:
        from piper_offload import ModelCache, ModelSpec, ResourceLeasedError, ResourceTooLargeError

        cache = ModelCache()
        spec = ModelSpec(key="model", factory=_make_block_model, block_paths=("transformer_blocks",))
        with cache.lease(spec) as offloader:
            streamer = block_components(offloader)[0]
        request.addfinalizer(offloader.deactivate)

        def fail_close(_component):
            raise RuntimeError("incomplete release")

        with monkeypatch.context() as patch:
            patch.setattr(BlockComponent, "_close_pin_lease", fail_close)
            with pytest.raises(RuntimeError, match="incomplete release"), cache.use(spec, device="cpu"):
                # Stand in for unfinished CUDA work. A retry must close its pin lease.
                pin = streamer._pin_lease = host_pin_manager.acquire([], pin=False)
                streamer._working_set_state = WorkingSetState.CHANGING

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
        offloader, streamer, _resident = self._offloader_with_leftover()
        leftover = streamer._pin_lease
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
        offloader, streamer, resident = self._offloader_with_leftover()

        def still_failing(_component: BlockComponent) -> None:
            raise error("release still failing")

        with monkeypatch.context() as patch:
            patch.setattr(BlockComponent, "_close_pin_lease", still_failing)
            with pytest.raises(raised) as caught:
                offloader.activate("cpu")
        # An interrupt during the retry propagates as itself; an ordinary
        # error becomes the failed-session error, chained to its cause.
        if raised is ModelRuntimeFailedError:
            assert isinstance(caught.value.__cause__, RuntimeError)
        assert offloader.failed
        assert offloader.active_device is None
        assert resident._active_device is None

        # The activation claim was released, so a later retry can succeed.
        offloader.deactivate()
        assert not offloader.failed
        assert streamer._pin_lease is None
        with activated_model(offloader, "cpu"):
            pass

    @CUDA
    def test_recoverable_deactivation_error_keeps_routed_activation_available(self) -> None:
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
            block_paths=["transformer_blocks"],
        )
        streamer = block_components(offloader)[0]
        x = torch.randn(2, 8, device="cuda")

        offloader.activate("cuda", adapters=[lora], adapter_mode="routed")
        with torch.no_grad():
            expected = model(x).clone()
        lookahead = streamer._runtime._lookahead
        assert lookahead is not None and lookahead.prefetch is not None
        lookahead.prefetch.result()
        bad_prefetch: Future[None] = Future()
        bad_prefetch.set_exception(RuntimeError("simulated prefetch failure"))
        lookahead.prefetch = bad_prefetch
        # Every target is synchronized and back on the host before this
        # surfaces, so the session ends cleanly despite the error.
        with pytest.raises(RuntimeError, match="simulated prefetch failure"):
            offloader.deactivate()

        assert not offloader.failed
        assert offloader._routed_lease is None
        with activated_model(offloader, "cuda", adapters=[lora], adapter_mode="routed"), torch.no_grad():
            actual = model(x).clone()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestActivateFailureCleanup:
    @CUDA
    def test_partial_activate_failure_rolls_back_other_components(self, monkeypatch) -> None:
        # If a streamer's activate raises, the offloader's deactivation
        # releases every component, and it releases its activation claim so
        # the same cached runtime remains reusable.
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        streamer = block_components(strategy)[0]
        runtime = streamer._runtime
        original_register_hooks = runtime._register_hooks

        def broken_register_hooks(*args, **kwargs):
            original_register_hooks(*args, **kwargs)
            raise RuntimeError("simulated activate failure")

        monkeypatch.setattr(runtime, "_register_hooks", broken_register_hooks)

        with pytest.raises(RuntimeError, match="simulated activate failure"):
            strategy.activate("cuda")

        assert not strategy._composite._active
        assert strategy.active_device is None

        monkeypatch.setattr(runtime, "_register_hooks", original_register_hooks)
        with activated_model(strategy, "cuda"):
            pass

    def test_interrupted_claim_is_given_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Asking whether a prior session failed is the first thing activation
        # does with its claim; an interrupt there must not keep the claim.
        offloader = ModelOffloader.from_module(
            _make_block_model(),
            block_paths=["transformer_blocks"],
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

    def test_failing_components_own_deactivate_runs(self) -> None:
        # If a component's activate raises midway, its own deactivate must
        # run too, not just its siblings', or its partial state leaks. The
        # offloader's deactivation unwinds every component exactly once.
        events: list[str] = []

        class _Recorder:
            def __init__(self, name: str, raise_on_activate: bool = False):
                self._name = name
                self._raise = raise_on_activate

            cache_bytes = 0
            failed = False

            def activate(self, device: torch.device | str | None = None, **kwargs: object) -> None:
                del device, kwargs
                events.append(f"activate:{self._name}")
                if self._raise:
                    # Simulate partial mutation before failing.
                    events.append(f"partial_state:{self._name}")
                    raise RuntimeError(f"{self._name} activate failed")

            def deactivate(self) -> None:
                events.append(f"deactivate:{self._name}")

        m = _make_block_model()
        strat = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        strat._composite = CompositeComponent(
            resident=None,
            blocks=[
                _Recorder("A"),
                _Recorder("B"),
                _Recorder("C", raise_on_activate=True),
            ],
        )

        with pytest.raises(RuntimeError, match="C activate failed"):
            strat.activate("cpu")

        # Activates ran in order until C raised.
        assert "activate:A" in events
        assert "activate:B" in events
        assert "activate:C" in events
        assert "partial_state:C" in events
        # C's own deactivate runs, and its siblings' follow in reverse order.
        assert "deactivate:C" in events
        assert "deactivate:B" in events
        assert "deactivate:A" in events
        # Reverse order on unwind.
        deact_events = [e for e in events if e.startswith("deactivate:")]
        assert deact_events == ["deactivate:C", "deactivate:B", "deactivate:A"]


# ---------------------------------------------------------------------------
# Prefetch failure during deactivate
# ---------------------------------------------------------------------------


class TestPrefetchFailureOnDeactivate:
    @CUDA
    def test_prefetch_failure_propagates_after_cleanup(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        strategy.activate("cuda")
        streamer = block_components(strategy)[0]
        # Running block 0 prefetches block 1; once that finishes, swap in a
        # failed prefetch so deactivation hits it.
        with torch.no_grad():
            m.transformer_blocks[0](torch.randn(2, 8, device="cuda"))
        lookahead = streamer._runtime._lookahead
        assert lookahead is not None and lookahead.prefetch is not None
        lookahead.prefetch.result()
        bad_future: Future[None] = Future()
        bad_future.set_exception(RuntimeError("simulated prefetch failure"))
        lookahead.prefetch = bad_future

        with pytest.raises(RuntimeError, match="simulated prefetch failure"):
            strategy.deactivate()

        # Even though we raised, cleanup completed.
        assert not streamer._runtime._hooks
        assert streamer._runtime._executor is None
        assert strategy.active_device is None

        with activated_model(strategy, "cpu"):
            pass

    def test_interrupted_wait_ends_after_the_worker_finishes(self) -> None:
        # Ctrl-C can end a prefetch wait while the worker still stages into
        # its block's target. Handing that target to another block then would
        # let two copies write it at once, so the interrupt must surface only
        # once the worker is done.
        future: Future[None] = Future()
        finish = threading.Timer(0.05, future.set_result, (None,))

        def interrupted_result(timeout: float | None = None) -> None:
            del timeout
            finish.start()
            raise Interrupt

        future.result = interrupted_result  # type: ignore[method-assign]
        with pytest.raises(Interrupt):
            _wait_for_prefetch(future)
        assert future.done()

    @CUDA
    def test_failed_host_restore_keeps_the_block_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Interrupting install_host() while switching away from a block can
        # leave its parameters on the GPU target. That target must stay with
        # the block rather than return to the pool, so deactivation finishes
        # the restore instead of leaving an inactive model on the GPU.
        m = _make_block_model(num_blocks=3)
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        streamer = block_components(strategy)[0]
        runtime = streamer._runtime
        first_block = streamer.blocks[0]
        install_host = HostModuleInstance.install_host
        failures = [Interrupt()]

        def install_host_failing_once(instance: HostModuleInstance) -> None:
            if failures and instance.module is first_block:
                raise failures.pop()
            install_host(instance)

        x = torch.randn(2, 8, device="cuda")
        with activated_model(strategy, "cuda"), torch.no_grad():
            hidden = streamer.blocks[0](x)
            monkeypatch.setattr(HostModuleInstance, "install_host", install_host_failing_once)
            with pytest.raises(Interrupt):
                streamer.blocks[1](hidden)
            active = runtime._active
            assert active is not None and active.block_idx == 0
            pooled = [lease for leases in runtime._pool._free.values() for lease in leases]
            assert all(lease is not active.lease for lease in pooled)

        assert all(param.device.type == "cpu" for param in m.parameters())


# ---------------------------------------------------------------------------
# Constructor leaves no GPU residency (cache_bytes is final, no pool yet)
# ---------------------------------------------------------------------------


class TestConstructedStateIsInactive:
    """Verifies the 'constructed but not active' state has no GPU
    footprint — the payoff of composer-owned stream/non-block partitioning.
    Without it, non-block siblings (embed, head, norms) would sit on
    the activation device permanently, defeating ResourceCache eviction."""

    @CUDA
    def test_constructed_has_no_params_on_activation_device(self) -> None:
        m = _make_block_model(num_blocks=4, width=8)
        target = torch.device("cuda")
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        try:
            for p in m.parameters():
                assert p.device != target, f"constructed state leaked GPU residency: {p.shape}@{p.device}"
        finally:
            strategy.deactivate()

    def test_block_only_model_has_no_non_block_host(self) -> None:
        # Edge case: model whose only top-level child IS the block list.
        # No non-block HostComponent in components; store bytes come from blocks only.
        class BlockOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(4)])

        m = BlockOnly()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        try:
            # No HostComponent component, just BlockComponent.
            assert host_component(strategy) is None
            assert strategy.cache_bytes > 0  # block bytes only
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
        target = torch.device("cuda")
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
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
            block_paths=["transformer_blocks"],
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
            block_paths=["transformer_blocks"],
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
            block_paths=["transformer_blocks"],
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
            block_paths=["transformer_blocks"],
        )
        try:
            for block in m.transformer_blocks:
                assert block.buf_a is block.buf_b
                assert not block.buf_a.is_pinned()
        finally:
            strategy.deactivate()

    def test_non_block_internal_tied_works(self) -> None:
        # Tied embed↔head WITHIN HostComponent: HostComponent handles
        # this via tensor-id dedup. Should not raise.
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
            block_paths=["transformer_blocks"],
        )
        try:
            assert m.embed.weight is m.head.weight
            non_block = host_component(strategy)
            assert non_block is not None
            assert non_block.param_names == {"embed.weight", "head.weight"}
            assert len({id(non_block._instance.params[name]) for name in non_block.param_names}) == 1
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
            block_paths=["transformer_blocks"],
        )
        try:
            assert not m.scale_shift.is_pinned()
        finally:
            strategy.deactivate()

    def test_nested_block_paths_with_direct_root_param(self) -> None:
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
            block_paths=["encoder.blocks"],
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
            block_paths=["transformer_blocks"],
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
            block_paths=["transformer_blocks"],
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
            block_paths=["transformer_blocks"],
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
            block_paths=["transformer_blocks"],
        )
        try:
            for block in m.transformer_blocks:
                assert block.buf_a is block.buf_b
                assert not block.buf_a.is_pinned()

            strategy.activate("cuda")
            resident = m.transformer_blocks[0]
            assert resident.buf_a is resident.buf_b
            assert resident.buf_a.is_cuda

            strategy.deactivate()
            for block in m.transformer_blocks:
                assert block.buf_a is block.buf_b
                # Released backing stays on CPU and may retain an idle pin.
                assert block.buf_a.device.type == "cpu"
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Block layout check rejects heterogeneous block layouts
# ---------------------------------------------------------------------------


class TestBlockLayoutCompatibility:
    """Blocks in one streamed group may differ in tensor *layout* — shape,
    dtype, quant format, storage tying, buffer shape/stride — because the
    morphing target pool keys reusable GPU targets by per-block layout
    signature, so a block never copies its bytes into a target shaped for
    a different layout. What blocks must still share is frozenness
    (``requires_grad=False``) and *structure*: the selected parameter and
    buffer *name* sets."""

    @staticmethod
    def _signatures(component: object) -> list[object]:
        return [
            _plan_target_signature(instance.resolve_load_plan())
            for instance in component._block_instances  # type: ignore[attr-defined]
        ]

    def test_shape_mismatch_is_supported(self) -> None:
        # Different weight shapes → distinct pool signatures, not a reject.
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False), nn.Linear(8, 8, bias=False)])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        component = _make_block_component(
            blocks=list(m.blocks),
        )
        assert len(set(self._signatures(component))) == 2

    def test_dtype_mismatch_is_supported(self) -> None:
        # Same shape, different dtype: separate signatures, so no silent
        # cross-dtype ``copy_``.
        b0 = nn.Linear(4, 4, bias=False).to(torch.float16)
        b1 = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
        for b in (b0, b1):
            for p in b.parameters():
                p.requires_grad = False
        component = _make_block_component(
            blocks=[b0, b1],
        )
        assert len(set(self._signatures(component))) == 2

    def test_param_alias_topology_is_supported(self) -> None:
        # Differing storage-tying topology → distinct signatures; each
        # block loads into a target allocated from its own instance.
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

        component = _make_block_component(
            blocks=[TiedBlock(), UntiedBlock()],
        )
        assert len(set(self._signatures(component))) == 2

    @CUDA
    def test_aliased_block_module_installs_single_hook(self) -> None:
        # A module aliased across blocks (a weight-shared layer appearing
        # multiple times in the block list) must get ONE forward-pre hook, not
        # one per alias — else a single shared forward would load then evict
        # each aliased block in turn, churning the GPU pool.
        shared = nn.Linear(4, 4, bias=False)
        shared.weight.requires_grad_(False)
        component = _make_block_component(
            blocks=[shared, shared],
        )
        with component.use("cuda"):
            assert len(shared._forward_pre_hooks) == 1

    def test_buffer_shape_mismatch_is_supported(self) -> None:
        class Block(nn.Module):
            def __init__(self, buffer_size: int):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(4, 4),
                    requires_grad=False,
                )
                self.register_buffer("table", torch.randn(buffer_size))

        component = _make_block_component(
            blocks=[Block(4), Block(8)],
        )
        assert len(set(self._signatures(component))) == 2

    def test_buffer_stride_mismatch_is_supported(self) -> None:
        class Block(nn.Module):
            def __init__(self, table: torch.Tensor):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(4, 4),
                    requires_grad=False,
                )
                self.register_buffer("table", table)

        contiguous = torch.randn(2, 3)
        non_contiguous = torch.randn(3, 2).t()
        assert contiguous.shape == non_contiguous.shape
        assert contiguous.stride() != non_contiguous.stride()

        # No reject. Capture clones buffers contiguous, so the two
        # collapse to one signature — what matters is the old hard layout
        # check no longer rejects the group.
        component = _make_block_component(
            blocks=[Block(contiguous), Block(non_contiguous)],
        )
        assert len(self._signatures(component)) == 2

    def test_param_name_mismatch_raises(self) -> None:
        # Differing selected param *names* is still a structural reject.
        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = nn.Parameter(torch.randn(4, 4), requires_grad=False)

        class B(nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = nn.Parameter(torch.randn(4, 4), requires_grad=False)

        with pytest.raises(ValueError, match="parameter names differ"):
            _make_block_component(
                blocks=[A(), B()],
            )

    def test_trainable_block_param_raises(self) -> None:
        block_0 = nn.Linear(4, 4, bias=False).requires_grad_(False)
        block_1 = nn.Linear(4, 4, bias=False).requires_grad_(True)

        with pytest.raises(ValueError, match="frozen"):
            _make_block_component(
                blocks=[block_0, block_1],
            )

    def test_buffer_name_mismatch_raises(self) -> None:
        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(4, 4),
                    requires_grad=False,
                )
                self.register_buffer("foo", torch.randn(4))

        class B(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(4, 4),
                    requires_grad=False,
                )
                self.register_buffer("bar", torch.randn(4))

        with pytest.raises(ValueError, match="buffer names differ"):
            _make_block_component(
                blocks=[A(), B()],
            )

    def test_validation_failure_leaves_model_unmutated(self) -> None:
        # Strong-exception-safety: structural validation runs before any
        # capture, so on a reject the user's Parameter objects keep their
        # identities and are not host.
        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = nn.Parameter(torch.randn(4, 4), requires_grad=False)

        class B(nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = nn.Parameter(torch.randn(4, 4), requires_grad=False)

        blocks = [A(), B()]
        original_params = [blocks[0].foo, blocks[1].bar]
        original_host = [p.is_pinned() for p in original_params]

        with pytest.raises(ValueError, match="parameter names differ"):
            _make_block_component(
                blocks=blocks,
            )

        assert blocks[0].foo is original_params[0]
        assert blocks[1].bar is original_params[1]
        for param, orig_pin in zip(
            original_params,
            original_host,
            strict=True,
        ):
            assert param.is_pinned() == orig_pin, "parameter pin status changed despite validation failure"


# ---------------------------------------------------------------------------
# Multi-component cleanup ordering (ExitStack semantics)
# ---------------------------------------------------------------------------


class TestMultiComponentCleanup:
    @CUDA
    def test_host_deactivate_failure_still_runs_streamer_deactivate(
        self,
        monkeypatch,
    ) -> None:
        # ExitStack continues unwinding callbacks even when one raises.
        # If non-block HostComponent raises during deactivate, streamers
        # earlier in unwind order have still been deactivated.

        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        original_deactivate = HostComponent.deactivate

        def broken_deactivate(component: HostComponent) -> None:
            original_deactivate(component)
            raise RuntimeError("simulated host deactivate failure")

        monkeypatch.setattr(HostComponent, "deactivate", broken_deactivate)
        try:
            strategy.activate("cuda")
            assert m.embed.weight.is_cuda  # type: ignore[union-attr]

            with pytest.raises(RuntimeError, match="simulated host deactivate failure"):
                strategy.deactivate()

            # HostComponent restored registry entries before raising, and streamers
            # were already unwound in LIFO order.
            assert not m.embed.weight.is_pinned()  # type: ignore[union-attr]
            assert not block_components(strategy)[0]._runtime._hooks
            assert not strategy._composite._active
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# BlockComponent name selection
# ---------------------------------------------------------------------------


class TestBlockNameSelection:
    def test_block_component_exposes_owned_block_local_names(self) -> None:
        m = _make_block_model()
        streamer = _make_block_component(
            blocks=list(m.transformer_blocks),
        )
        try:
            assert streamer.managed_param_names_by_block == [["weight"]] * len(m.transformer_blocks)
            assert streamer.managed_buffer_names_by_block == [[]] * len(m.transformer_blocks)
        finally:
            streamer.deactivate()

    def test_block_component_exposes_addressable_param_names(self) -> None:
        m = _make_block_model()
        streamer = _make_block_component(
            blocks=list(m.transformer_blocks),
            blocks_path="transformer_blocks",
        )
        try:
            assert streamer.param_names == {f"transformer_blocks.{i}.weight" for i in range(len(m.transformer_blocks))}
        finally:
            streamer.deactivate()

    def test_block_component_store_binds_compatible_model(self) -> None:
        torch.manual_seed(0)
        prototype = _make_block_model()
        target = _make_block_model()
        target_embed = target.embed.weight.detach().clone()

        store = BlockComponentStore.from_module(
            prototype,
            blocks_path="transformer_blocks",
        )
        streamer = store.bind(target)
        try:
            assert streamer.param_names == {
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
            streamer.deactivate()

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
            block_paths=["blocks"],
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
            streamer = block_components(strategy)[0]
            assert streamer.buffer_names == frozenset(
                {
                    "blocks.0.scale",
                    "blocks.1.scale",
                }
            )
            assert streamer.blocks[0] is model.blocks[0]
        finally:
            strategy.deactivate()

    def test_block_component_rejects_duplicate_buffer_names(self) -> None:
        block = nn.Module()
        block.register_buffer("scale", torch.ones(1))
        _, instance = HostModuleStore.capture(block)
        with pytest.raises(ValueError, match=r"duplicate block buffer name '3\.scale'"):
            BlockComponent([instance, instance], block_indices=[3, 3])

    def test_block_component_resolves_parameter_override_by_name(self) -> None:
        m = _make_block_model()
        streamer = _make_block_component(
            blocks=list(m.transformer_blocks),
            blocks_path="transformer_blocks",
        )
        def update(_param: nn.Parameter) -> None:
            pass

        transform = CallbackParameterTransform(update)

        try:
            plans = streamer._resolve_load_plans(
                {
                    "transformer_blocks.1.weight": ParameterOverride(
                        update=transform
                    )
                }
            )
            assert plans[1].loads["weight"].update is transform
            assert plans[0].loads["weight"].update is None
        finally:
            streamer.deactivate()

    def test_block_component_rejects_unknown_parameter_override_name(self) -> None:
        m = _make_block_model()
        streamer = _make_block_component(
            blocks=list(m.transformer_blocks),
            blocks_path="transformer_blocks",
        )
        try:
            with pytest.raises(ValueError, match="not owned by this block component"):
                streamer._resolve_load_plans(
                    {
                        "transformer_blocks.10.weight": ParameterOverride(
                            update=CallbackParameterTransform(lambda _param: None)
                        )
                    }
                )
        finally:
            streamer.deactivate()

    def test_model_offloader_partitions_streamed_and_host_names(
        self,
    ) -> None:
        m = _make_block_model()
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        try:
            streamer = block_components(strategy)[0]
            non_block = host_component(strategy)
            assert non_block is not None

            assert streamer.managed_param_names_by_block == [["weight"]] * len(m.transformer_blocks)
            assert non_block.param_names == {"embed.weight", "head.weight"}
            assert all(not name.startswith("transformer_blocks.") for name in non_block.param_names)
        finally:
            strategy.deactivate()


class TestBlockComponentActivateTwice:
    @CUDA
    def test_double_activate_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        offloader = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        try:
            offloader.activate("cuda")
            # Reach into the streamer and call activate again — the
            # composer's activate handles its own teardown ExitStack,
            # but the streamer itself must hard-guard against
            # double-install of forward-pre hooks.
            streamer = block_components(offloader)[0]
            with pytest.raises(RuntimeError, match="already.*active"):
                streamer.activate(torch.device("cuda"))
        finally:
            offloader.deactivate()


# ---------------------------------------------------------------------------
# Sparse streamed block groups
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
        store = BlockComponentStore.from_module(
            model,
            blocks_path="blocks",
        )
        assert store.block_indices == (1, 3)
        component = store.bind(model)
        assert list(component.blocks) == real
        assert store.buffer_names == component.buffer_names == {"blocks.1.scale", "blocks.3.scale"}

    def test_bind_rejects_unmanaged_nonempty_blocks(self) -> None:
        store_model = _make_block_list_model(self._blocks(n=3), "blocks")
        bind_model = _make_block_list_model(self._blocks(n=5), "blocks")
        store = BlockComponentStore.from_module(
            store_model,
            blocks_path="blocks",
        )
        with pytest.raises(ValueError, match="does not occupy"):
            store.bind(bind_model)

    def test_bind_allows_empty_unoccupied_blocks(self) -> None:
        real = self._blocks(n=2)
        model = _make_block_list_model([*real, nn.Module()], "blocks")
        store = BlockComponentStore.from_module(
            model,
            blocks_path="blocks",
        )
        assert store.block_indices == (0, 1)
        store.bind(model)

    def test_all_empty_blocks_raise(self) -> None:
        model = _make_block_list_model([nn.Module(), nn.Module()], "blocks")
        with pytest.raises(ValueError, match="no streamable blocks"):
            BlockComponentStore.from_module(
                model,
                blocks_path="blocks",
            )


class TestFrozenContract:
    def test_block_store_capture_and_bind_are_atomic_on_rejection(self) -> None:
        block_0 = nn.Linear(4, 4, bias=False).requires_grad_(False)
        block_1 = nn.Linear(4, 4, bias=False).requires_grad_(True)
        orig_weight = block_0.weight
        orig_ptr = block_0.weight.untyped_storage().data_ptr()
        model = _make_block_list_model([block_0, block_1], "blocks")

        with pytest.raises(ValueError, match="frozen"):
            BlockComponentStore.from_module(model, blocks_path="blocks")

        assert block_0.weight is orig_weight
        assert block_0.weight.untyped_storage().data_ptr() == orig_ptr

        frozen_proto = _make_block_list_model(
            [nn.Linear(4, 4, bias=False).requires_grad_(False) for _ in range(2)],
            "blocks",
        )
        store = BlockComponentStore.from_module(frozen_proto, blocks_path="blocks")
        with pytest.raises(ValueError, match="frozen"):
            store.bind(model)

        assert block_0.weight is orig_weight
        assert block_0.weight.untyped_storage().data_ptr() == orig_ptr

    def test_model_offloader_and_composite_store_reject_without_partial_mutation(self) -> None:
        from piper_offload.composite_component import CompositeComponentStore

        non_block_trainable = _make_block_model(num_blocks=2)
        orig_block_0 = non_block_trainable.transformer_blocks[0].weight
        orig_block_0_ptr = orig_block_0.untyped_storage().data_ptr()
        non_block_trainable.embed.weight.requires_grad_(True)
        with pytest.raises(ValueError, match="frozen"):
            ModelOffloader.from_module(
                non_block_trainable,
                block_paths=["transformer_blocks"],
                transient_paths=["head"],
            )
        assert non_block_trainable.transformer_blocks[0].weight is orig_block_0
        assert non_block_trainable.transformer_blocks[0].weight.untyped_storage().data_ptr() == orig_block_0_ptr

        in_block_trainable = _make_block_model(num_blocks=2)
        orig_embed = in_block_trainable.embed.weight
        orig_embed_ptr = orig_embed.untyped_storage().data_ptr()
        in_block_trainable.transformer_blocks[1].weight.requires_grad_(True)
        with pytest.raises(ValueError, match="frozen"):
            ModelOffloader.from_module(
                in_block_trainable,
                block_paths=["transformer_blocks"],
                transient_paths=["head"],
            )
        assert in_block_trainable.embed.weight is orig_embed
        assert in_block_trainable.embed.weight.untyped_storage().data_ptr() == orig_embed_ptr

        proto = _make_block_model(num_blocks=2)
        comp_store = CompositeComponentStore.from_module(
            proto,
            block_paths=["transformer_blocks"],
            transient_paths=["head"],
        )
        bad_target = _make_block_model(num_blocks=2)
        bad_embed = bad_target.embed.weight
        bad_embed_ptr = bad_embed.untyped_storage().data_ptr()
        bad_head = bad_target.head.weight
        bad_head_ptr = bad_head.untyped_storage().data_ptr()
        bad_b0 = bad_target.transformer_blocks[0].weight
        bad_b0_ptr = bad_b0.untyped_storage().data_ptr()
        bad_target.transformer_blocks[1].weight.requires_grad_(True)

        with pytest.raises(ValueError, match="frozen"):
            comp_store.bind(bad_target)

        assert bad_target.embed.weight is bad_embed
        assert bad_target.embed.weight.untyped_storage().data_ptr() == bad_embed_ptr
        assert bad_target.head.weight is bad_head
        assert bad_target.head.weight.untyped_storage().data_ptr() == bad_head_ptr
        assert bad_target.transformer_blocks[0].weight is bad_b0
        assert bad_target.transformer_blocks[0].weight.untyped_storage().data_ptr() == bad_b0_ptr
