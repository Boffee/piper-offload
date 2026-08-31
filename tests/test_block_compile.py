"""Opt-in ``torch.compile`` integration for declared block groups."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from piper_offload import (
    BlockCompileConfig,
    BlockMode,
    LoRA,
    ModelOffloader,
    ModelSpec,
)
from piper_offload.resident_runtime import ResidentBlockRuntime
from tests._block_compile_helpers import (
    _Block,
    _BlockModel,
    _make_offloader,
)
from tests.conftest import (
    activated_model,
    block_components,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _TwoGroupModel(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.first_blocks = nn.ModuleList([_Block(width), _Block(width)])
        self.second_blocks = nn.ModuleList([_Block(width), _Block(width)])
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.first_blocks:
            x = block(x)
        for block in self.second_blocks:
            x = block(x)
        return x


class _CompileSpy:
    """A lazy ``torch.compile`` stand-in that records construction/execution."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[tuple[Callable[..., object], dict[str, object]]] = []
        self.executions = 0
        self._events = events

    def __call__(
        self,
        fn: Callable[..., object],
        **kwargs: object,
    ) -> Callable[..., object]:
        self.calls.append((fn, kwargs))

        def compiled(*args: object, **call_kwargs: object) -> object:
            self.executions += 1
            if self._events is not None:
                self._events.append("compiled")
            return fn(*args, **call_kwargs)

        return compiled


class TestBlockCompileConfig:
    def test_defaults(self) -> None:
        config = BlockCompileConfig()

        assert config.dynamic is True
        assert config.fullgraph is False
        assert config.options is None

    def test_rolling_requires_fullgraph(self) -> None:
        with pytest.raises(ValueError, match="requires.*fullgraph=True"):
            _make_offloader(
                _BlockModel(),
                block_mode="rolling",
                block_compile=BlockCompileConfig(),
            )

    def test_rolling_requires_compile_config(self) -> None:
        with pytest.raises(ValueError, match="requires BlockCompileConfig"):
            _make_offloader(
                _BlockModel(),
                block_mode="rolling",
            )

    def test_compile_without_block_paths_is_unused(self) -> None:
        model = nn.Linear(4, 4, bias=False)
        offloader = ModelOffloader.from_module(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            assert not block_components(offloader)
        finally:
            offloader.deactivate()

    def test_model_spec_passes_config_to_bound_component(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        compiler_options = {"max_autotune": True}
        config = BlockCompileConfig(
            dynamic=None,
            fullgraph=True,
            options=compiler_options,
        )
        spec = ModelSpec(
            key="compiled",
            estimated_cache_bytes=1024,
            factory=_BlockModel,
            transient_block_paths=("blocks",),
            block_compile=config,
        )

        offloader = spec.build_store()
        try:
            block_component = block_components(offloader)[0]
            assert block_component.block_compile is config
            assert offloader._composite.transient_blocks == (block_component,)
            assert len(spy.calls) == 2
            assert all(
                kwargs
                == {
                    "backend": "inductor",
                    "dynamic": None,
                    "fullgraph": True,
                    "options": compiler_options,
                }
                for _fn, kwargs in spy.calls
            )
            forwarded_options = [kwargs["options"] for _fn, kwargs in spy.calls]
            assert all(options is not compiler_options for options in forwarded_options)
            assert len({id(options) for options in forwarded_options}) == len(forwarded_options)
        finally:
            offloader.deactivate()

    def test_model_spec_can_compile_resident_blocks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        spec = ModelSpec(
            key="resident-compiled",
            estimated_cache_bytes=1024,
            factory=_BlockModel,
            block_paths=("blocks",),
            block_mode="resident",
            block_compile=BlockCompileConfig(),
        )

        offloader = spec.build_store()
        try:
            components = block_components(offloader)
            assert len(components) == 1
            assert isinstance(
                components[0]._runtime,
                ResidentBlockRuntime,
            )
            assert len(spy.calls) == 2
        finally:
            offloader.deactivate()

    def test_resident_mode_applies_to_transient_block_paths(self) -> None:
        model = _TwoGroupModel()
        offloader = _make_offloader(
            model,
            block_paths=["first_blocks"],
            transient_block_paths=("second_blocks",),
            block_mode="resident",
        )
        try:
            resident, transient = block_components(offloader)
            assert isinstance(
                resident._runtime,
                ResidentBlockRuntime,
            )
            assert isinstance(
                transient._runtime,
                ResidentBlockRuntime,
            )
            assert offloader._composite.transient_blocks == (transient,)
        finally:
            offloader.deactivate()

    def test_compile_policy_does_not_change_cache_bytes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch, "compile", _CompileSpy())
        eager = _make_offloader(_BlockModel())
        compiled = _make_offloader(
            _BlockModel(),
            block_compile=BlockCompileConfig(),
        )
        try:
            assert compiled.cache_bytes == eager.cache_bytes
        finally:
            compiled.deactivate()
            eager.deactivate()

    def test_real_torch_compile_accepts_options_without_a_mode_conflict(self) -> None:
        offloader = _make_offloader(
            _BlockModel(num_blocks=1),
            block_compile=BlockCompileConfig(options={"max_autotune": False}),
        )
        try:
            assert block_components(offloader)[0].block_compile is not None
        finally:
            offloader.deactivate()

    def test_piper_convrot_compile_options_reach_every_block(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        convrot = pytest.importorskip("piper_kernels.linear.convrot")
        compiler_options = convrot.convrot_int8_compile_options({"max_autotune": True})
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)

        offloader = _make_offloader(
            _BlockModel(),
            block_compile=BlockCompileConfig(options=compiler_options),
        )
        try:
            assert len(spy.calls) == 2
            assert all(
                kwargs["options"] == compiler_options
                and kwargs["options"] is not compiler_options
                and "mode" not in kwargs
                for _fn, kwargs in spy.calls
            )
        finally:
            offloader.deactivate()


class TestCompiledForwardConstruction:
    def test_aliased_block_module_is_compiled_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        shared = _Block()
        model = _BlockModel(blocks=[shared, shared])

        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            assert len(spy.calls) == 1
        finally:
            offloader.deactivate()

    def test_one_config_applies_to_multiple_block_groups(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        model = _TwoGroupModel()
        config = BlockCompileConfig()

        offloader = _make_offloader(
            model,
            block_paths=["first_blocks", "second_blocks"],
            block_compile=config,
        )
        try:
            streamers = block_components(offloader)
            assert len(streamers) == 2
            assert all(streamer.block_compile is config for streamer in streamers)
            assert len(spy.calls) == 4
        finally:
            offloader.deactivate()


class TestCompiledForwardLifecycle:
    @pytest.mark.parametrize("block_mode", ["streaming", "resident"])
    def test_cpu_activation_remains_eager(
        self,
        monkeypatch: pytest.MonkeyPatch,
        block_mode: BlockMode,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        model = _BlockModel()
        offloader = _make_offloader(
            model,
            block_mode=block_mode,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(offloader, "cpu"):
                with torch.inference_mode():
                    model(torch.randn(2, 8))
                assert all("forward" not in block.__dict__ for block in model.blocks)
            assert spy.executions == 0
        finally:
            offloader.deactivate()

    @CUDA
    def test_resident_blocks_compile_without_prefetching(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        model = _BlockModel()
        offloader = _make_offloader(
            model,
            block_mode="resident",
            block_compile=BlockCompileConfig(),
        )
        streamer = block_components(offloader)[0]
        runtime = streamer._runtime
        try:
            assert isinstance(runtime, ResidentBlockRuntime)
            with activated_model(offloader, "cuda"):
                assert runtime.acquired
                assert len(runtime._leases) == len(model.blocks)
                leases = tuple(runtime._leases)
                assert not hasattr(runtime, "_executor")
                assert not hasattr(runtime, "_stream")
                assert not hasattr(runtime, "_hooks")
                assert all("forward" in block.__dict__ for block in model.blocks)
                assert all(param.device.type == "cuda" for param in model.parameters())
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
                    model(torch.randn(2, 8, device="cuda"))
                assert tuple(runtime._leases) == leases
            assert spy.executions == 4
            assert all("forward" not in block.__dict__ for block in model.blocks)
            assert all(param.device.type == "cpu" for param in model.parameters())
        finally:
            offloader.deactivate()

    @CUDA
    def test_cuda_installs_after_stream_hook_and_restores_descriptor_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        events: list[str] = []
        spy = _CompileSpy(events)
        monkeypatch.setattr(torch, "compile", spy)
        model = _BlockModel()
        assert all("forward" not in block.__dict__ for block in model.blocks)
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        streamer = block_components(offloader)[0]
        runtime = streamer._runtime
        original_before = runtime._before_block_forward

        def record_before(*args: object, **kwargs: object) -> None:
            events.append("stream")
            original_before(*args, **kwargs)

        monkeypatch.setattr(runtime, "_before_block_forward", record_before)
        try:
            with activated_model(offloader, "cuda"):
                assert streamer._active_runtime is runtime
                assert all("forward" in block.__dict__ for block in model.blocks)
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
            assert events == ["stream", "compiled", "stream", "compiled"]
            assert streamer._active_runtime is None
            assert all("forward" not in block.__dict__ for block in model.blocks)
        finally:
            offloader.deactivate()

    @CUDA
    def test_compiled_forwards_remain_installed_across_release_acquire(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        model = _BlockModel()
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        streamer = block_components(offloader)[0]
        runtime = streamer._runtime
        try:
            with activated_model(offloader, "cuda"):
                assert runtime.acquired
                assert all("forward" in block.__dict__ for block in model.blocks)

                streamer.release()
                assert not runtime.acquired
                assert all("forward" in block.__dict__ for block in model.blocks)

                streamer.acquire()
                assert runtime.acquired
                assert all("forward" in block.__dict__ for block in model.blocks)
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
            assert all("forward" not in block.__dict__ for block in model.blocks)
        finally:
            offloader.deactivate()

    @CUDA
    def test_named_forward_hook_wraps_compiled_block(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        model = _BlockModel()
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        calls: list[nn.Module] = []
        remove_hook = offloader.register_forward_hook(
            "blocks.1",
            lambda module, _args, _output: calls.append(module),
        )
        try:
            with activated_model(offloader, "cuda"):
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
                assert calls == [model.blocks[1]]

                remove_hook()
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
                assert calls == [model.blocks[1]]
        finally:
            remove_hook()
            offloader.deactivate()

    @CUDA
    def test_transient_block_path_retains_compiled_forwards(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        model = _BlockModel()
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
            transient_block_paths=("blocks",),
        )
        streamer = block_components(offloader)[0]
        runtime = streamer._runtime
        root_states: list[bool] = []
        remove_observer = offloader.register_forward_hook(
            "",
            lambda _module, _args, _output: root_states.append(runtime.acquired),
        )
        try:
            with activated_model(offloader, "cuda"):
                for _ in range(2):
                    with torch.inference_mode():
                        model(torch.randn(2, 8, device="cuda"))
                    assert runtime.acquired
                    assert all("forward" in block.__dict__ for block in model.blocks)
                assert root_states == [False, False]
            assert all("forward" not in block.__dict__ for block in model.blocks)
        finally:
            remove_observer()
            offloader.deactivate()

    @CUDA
    def test_existing_instance_forward_override_is_restored_verbatim(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        block = _Block()
        class_forward = block.forward

        def override(x: torch.Tensor) -> torch.Tensor:
            return class_forward(x) + 1

        block.forward = override
        original_override = block.__dict__["forward"]
        model = _BlockModel(blocks=[block])
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(offloader, "cuda"):
                assert block.__dict__["forward"] is not original_override
            assert block.__dict__["forward"] is original_override
        finally:
            offloader.deactivate()

    @CUDA
    @pytest.mark.parametrize("block_mode", ["streaming", "resident"])
    def test_activation_failure_restores_original_forwards(
        self,
        monkeypatch: pytest.MonkeyPatch,
        block_mode: BlockMode,
    ) -> None:
        monkeypatch.setattr(torch, "compile", _CompileSpy())
        model = _BlockModel()
        offloader = _make_offloader(
            model,
            block_mode=block_mode,
            block_compile=BlockCompileConfig(),
        )
        streamer = block_components(offloader)[0]
        compile_state = streamer._block_compile
        state_type = type(compile_state)
        original_install = state_type.install

        def broken_install(
            state: object,
            active_config: BlockCompileConfig | None,
        ) -> None:
            original_install(state, active_config)
            raise RuntimeError("simulated compiled-forward install failure")

        with monkeypatch.context() as install_patch:
            install_patch.setattr(state_type, "install", broken_install)
            with pytest.raises(RuntimeError, match="simulated compiled-forward"):
                offloader.activate("cuda")

        assert offloader.active_device is None
        assert not compile_state.installed
        assert all("forward" not in block.__dict__ for block in model.blocks)

        with activated_model(offloader, "cuda"):
            pass

    @CUDA
    def test_real_inductor_supports_dynamic_shapes_and_reactivation(self) -> None:
        torch.manual_seed(0)
        model = _BlockModel()
        inputs = [
            torch.randn(2, 8),
            torch.randn(3, 8),
            torch.randn(4, 8),
        ]
        with torch.inference_mode():
            expected = [model(x) for x in inputs]

        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(offloader, "cuda"):
                with torch.inference_mode():
                    actual = [model(x.cuda()).cpu() for x in inputs[:2]]
            with activated_model(offloader, "cuda"):
                with torch.inference_mode():
                    actual.append(model(inputs[2].cuda()).cpu())

            for actual_value, expected_value in zip(
                actual,
                expected,
                strict=True,
            ):
                torch.testing.assert_close(
                    actual_value,
                    expected_value,
                    rtol=1e-4,
                    atol=1e-5,
                )
        finally:
            offloader.deactivate()


class TestCompileFailureSemantics:
    @CUDA
    def test_compiler_failure_propagates_without_eager_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eager_calls = 0

        class CountingBlock(_Block):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                nonlocal eager_calls
                eager_calls += 1
                return super().forward(x)

        def failing_compile(
            _fn: Callable[..., object],
            **_kwargs: object,
        ) -> Callable[..., object]:
            def fail(*_args: object, **_call_kwargs: object) -> object:
                raise RuntimeError("simulated compiler failure")

            return fail

        monkeypatch.setattr(torch, "compile", failing_compile)
        model = _BlockModel(blocks=[CountingBlock()])
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(offloader, "cuda"):
                with pytest.raises(RuntimeError, match="simulated compiler"):
                    model(torch.randn(2, 8, device="cuda"))
            assert eager_calls == 0
        finally:
            offloader.deactivate()

    @CUDA
    def test_model_exception_propagates_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eager_calls = 0

        class FailingBlock(_Block):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                nonlocal eager_calls
                eager_calls += 1
                raise ValueError("model forward failed")

        monkeypatch.setattr(torch, "compile", _CompileSpy())
        model = _BlockModel(blocks=[FailingBlock()])
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(offloader, "cuda"):
                with pytest.raises(ValueError, match="model forward failed"):
                    model(torch.randn(2, 8, device="cuda"))
            assert eager_calls == 1
        finally:
            offloader.deactivate()


class TestCompiledLoRA:
    @CUDA
    def test_real_inductor_matches_eager_merge_mode(self) -> None:
        torch.manual_seed(0)
        eager_model = _BlockModel()
        compiled_model = _BlockModel()
        compiled_model.load_state_dict(eager_model.state_dict())
        lora = LoRA.from_state_dict(
            {
                "blocks.0.proj.lora_A.weight": torch.randn(2, 8),
                "blocks.0.proj.lora_B.weight": torch.randn(8, 2),
            }
        )
        x = torch.randn(2, 8, device="cuda")

        eager_offloader = _make_offloader(eager_model)
        try:
            with activated_model(
                eager_offloader,
                "cuda",
                loras=[lora],
                lora_strengths=[0.25],
                lora_mode="merge",
            ):
                with torch.inference_mode():
                    expected = eager_model(x).cpu()
        finally:
            eager_offloader.deactivate()

        compiled_offloader = _make_offloader(
            compiled_model,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(
                compiled_offloader,
                "cuda",
                loras=[lora],
                lora_strengths=[0.25],
                lora_mode="merge",
            ):
                with torch.inference_mode():
                    actual = compiled_model(x).cpu()
        finally:
            compiled_offloader.deactivate()

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    @CUDA
    @pytest.mark.parametrize("block_mode", ["streaming", "resident"])
    def test_routed_bypass_is_model_wide_and_temporary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        block_mode: BlockMode,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        model = _TwoGroupModel()
        lora = LoRA.from_state_dict(
            {
                "first_blocks.0.proj.lora_A.weight": torch.randn(2, 8),
                "first_blocks.0.proj.lora_B.weight": torch.randn(8, 2),
            }
        )
        offloader = _make_offloader(
            model,
            block_paths=["first_blocks", "second_blocks"],
            block_mode=block_mode,
            block_compile=BlockCompileConfig(),
        )
        x = torch.randn(2, 8, device="cuda")
        try:
            with activated_model(
                offloader,
                "cuda",
                loras=[lora],
                lora_mode="routed",
            ):
                assert all("forward" not in block.__dict__ for block in (*model.first_blocks, *model.second_blocks))
                with torch.inference_mode():
                    model(x)
            assert spy.executions == 0

            with activated_model(
                offloader,
                "cuda",
                lora_mode="routed",
            ):
                with torch.inference_mode():
                    model(x)
            assert spy.executions == 4

            with activated_model(
                offloader,
                "cuda",
                loras=[lora],
                lora_mode="merge",
            ):
                with torch.inference_mode():
                    model(x)
            assert spy.executions == 8
        finally:
            offloader.deactivate()
