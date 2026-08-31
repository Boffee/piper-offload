"""Single-target rolling compilation tests."""

import copy
from collections.abc import Callable, Iterator

import pytest
import torch
from torch import nn

from piper_offload import BlockCompileConfig, LoRA, register_adapter
from piper_offload.rolling_compile import (
    register_rolling_target,
    unregister_rolling_target,
)
from tests._block_compile_helpers import (
    _Block,
    _BlockModel,
    _make_offloader,
)
from tests.conftest import activated_model, streamed_components

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_ORDER_EVENTS: list[list[str]] = []


@pytest.fixture(autouse=True)
def _isolate_compiler_cache() -> Iterator[None]:
    """Keep normal and rolling backend variants below Dynamo's frame limit."""
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _compiled_output[T](
    model: nn.Module,
    config: BlockCompileConfig,
    forward: Callable[[], T],
    **activation: object,
) -> T:
    offloader = _make_offloader(model, block_compile=config)
    try:
        with activated_model(offloader, "cuda", **activation):
            with torch.inference_mode():
                return forward()
    finally:
        offloader.deactivate()


def _lora(num_blocks: int, width: int) -> LoRA:
    state: dict[str, torch.Tensor] = {}
    for block_idx in range(num_blocks):
        state[f"blocks.{block_idx}.proj.lora_A.weight"] = torch.randn(2, width)
        state[f"blocks.{block_idx}.proj.lora_B.weight"] = torch.randn(width, 2)
    return LoRA.from_state_dict(state)


def _torchao_mx_rolling_weight(kind: str) -> tuple[torch.Tensor, int, torch.dtype]:
    if kind in ("torchao-mxfp8", "torchao-mxfp4"):
        from tests.test_mx_adapter import _make_mx

        elem_dtype = torch.float8_e4m3fn if kind == "torchao-mxfp8" else torch.float4_e2m1fn_x2
        return (
            _make_mx(elem_dtype=elem_dtype, rows=128, cols=128, dynamic_activation=True),
            128,
            torch.bfloat16,
        )
    raise AssertionError(f"unknown TorchAO MX rolling quant test case {kind!r}")


def _torchao_rolling_weight(kind: str) -> tuple[torch.Tensor, int, torch.dtype]:
    if kind == "torchao-float8":
        from tests.test_float8_adapter import _make_float8

        return _make_float8(rows=64, cols=64, dynamic_activation=True), 64, torch.bfloat16
    if kind == "torchao-static-float8":
        from tests.test_static_float8_adapter import _make_static_float8

        return _make_static_float8(rows=64, cols=64), 64, torch.bfloat16
    if kind == "torchao-int8":
        from tests.test_int8_adapter import _make_int8

        return _make_int8(rows=64, cols=64, dynamic_activation=True), 64, torch.bfloat16
    if kind == "torchao-int4-tile":
        from tests.test_int4_tile_adapter import _make_int4_tile

        return _make_int4_tile(), 256, torch.bfloat16
    if kind.startswith("torchao-mx"):
        return _torchao_mx_rolling_weight(kind)
    if kind == "torchao-nvfp4":
        from tests.test_nvfp4_adapter import _make_nvfp4

        return _make_nvfp4(rows=64, cols=64, dynamic_activation=True), 64, torch.bfloat16
    raise AssertionError(f"unknown TorchAO rolling quant test case {kind!r}")


def _rolling_quant_weight(kind: str) -> tuple[torch.Tensor, int, torch.dtype]:
    if kind.startswith("torchao-"):
        return _torchao_rolling_weight(kind)
    if kind in ("quanto-qint8", "quanto-qfloat8"):
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        width = 64
        qtype = quanto.qint8 if kind == "quanto-qint8" else quanto.qfloat8_e4m3fn
        data = (
            torch.randint(-127, 128, (width, width), dtype=torch.int8)
            if qtype is quanto.qint8
            else torch.randn(width, width).to(qtype.dtype)
        )
        return (
            WeightQBytesTensor.create(
                qtype,
                0,
                (width, width),
                (width, 1),
                data,
                torch.rand(width, 1, dtype=torch.bfloat16).add_(0.01),
                None,
            ),
            width,
            torch.bfloat16,
        )
    if kind == "gguf-q4-0":
        gguf = pytest.importorskip("gguf")
        np = pytest.importorskip("numpy")
        from piper_offload.gguf_adapter import GGUFWeight

        width = 32
        quant_type = gguf.GGMLQuantizationType.Q4_0
        seed = torch.randint(0, torch.iinfo(torch.int32).max, ()).item()
        dense = np.random.default_rng(seed).standard_normal((width, width), dtype=np.float32)
        packed = torch.from_numpy(gguf.quantize(dense, quant_type))
        return GGUFWeight(packed, quant_type=int(quant_type)), width, torch.bfloat16
    raise AssertionError(f"unknown rolling quant test case {kind!r}")


def _rolling_quant_model(kind: str) -> tuple[_BlockModel, int, torch.dtype]:
    blocks: list[nn.Module] = []
    width = 0
    dtype = torch.bfloat16
    for _ in range(3):
        weight, width, dtype = _rolling_quant_weight(kind)
        block = _Block(width=width)
        block.proj.weight = nn.Parameter(weight, requires_grad=False)
        blocks.append(block)
    return _BlockModel(blocks=blocks), width, dtype


@torch.library.custom_op(
    "piper_offload_tests::rolling_reader",
    mutates_args=(),
)
def _rolling_reader(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if _ORDER_EVENTS:
        _ORDER_EVENTS[-1].append("reader")
    return torch.nn.functional.linear(value, weight)


@torch.library.register_fake("piper_offload_tests::rolling_reader")
def _rolling_reader_fake(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return value.new_empty((*value.shape[:-1], weight.shape[0]))


class _OrderedReaderBlock(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.piper_offload_tests.rolling_reader.default(
            x,
            self.weight,
        )


class _RuntimeStub:
    def wait_param(self, param_idx: int) -> None:
        del param_idx

    def rollover_param(self, param_idx: int) -> None:
        del param_idx


class TestRollingCompile:
    def test_lifecycle_ops_do_not_mutate_parameter_schema(self) -> None:
        for name in ("rolling_wait", "rolling_refill"):
            schema = getattr(torch.ops.piper_offload, name).default._schema
            assert not schema.is_mutable
            assert all(argument.alias_info is None for argument in schema.arguments)

    def test_active_slot_registration_rejects_collisions(self) -> None:
        param = nn.Parameter(torch.empty(1))
        first = _RuntimeStub()
        second = _RuntimeStub()
        register_rolling_target(first, [param])
        try:
            with pytest.raises(RuntimeError, match="registration collided"):
                register_rolling_target(second, [param])
        finally:
            unregister_rolling_target(first)

        register_rolling_target(second, [param])
        unregister_rolling_target(second)

    def test_rejects_unreviewed_external_adapter(self) -> None:
        from tests.test_tensor_adapter_registry import _PlainOverrideAdapter

        remove_adapter = register_adapter(_PlainOverrideAdapter)
        try:
            with pytest.raises(NotImplementedError, match="supports only"):
                _make_offloader(
                    _BlockModel(),
                    block_compile=BlockCompileConfig(rolling=True, fullgraph=True),
                )
        finally:
            remove_adapter()

    @CUDA
    @pytest.mark.parametrize(
        "quant_kind",
        [
            "torchao-float8",
            "torchao-static-float8",
            "torchao-int8",
            "torchao-int4-tile",
            "torchao-mxfp8",
            "torchao-mxfp4",
            "torchao-nvfp4",
            "quanto-qint8",
            "quanto-qfloat8",
            "gguf-q4-0",
        ],
    )
    def test_supported_quant_slots_match_compiled(self, quant_kind: str) -> None:
        torch.manual_seed(14)
        baseline_model, width, dtype = _rolling_quant_model(quant_kind)
        torch.manual_seed(14)
        rolling_model, rolling_width, rolling_dtype = _rolling_quant_model(quant_kind)
        assert (rolling_width, rolling_dtype) == (width, dtype)
        torch.manual_seed(15)
        x = torch.randn(32, width, device="cuda", dtype=dtype)
        activation: dict[str, object] = {}
        if quant_kind not in ("torchao-int4-tile", "gguf-q4-0"):
            activation.update(
                loras=[_lora(3, width)],
                lora_strengths=[0.125],
                lora_mode="merge",
                stochastic_rounding=False,
            )

        expected = _compiled_output(
            baseline_model,
            BlockCompileConfig(dynamic=False, fullgraph=True),
            lambda: [baseline_model(x).clone() for _ in range(2)],
            **activation,
        )
        actual = _compiled_output(
            rolling_model,
            BlockCompileConfig(dynamic=False, rolling=True, fullgraph=True),
            lambda: [rolling_model(x).clone() for _ in range(2)],
            **activation,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @CUDA
    def test_piper_convrot_structured_slots_match_compiled(self) -> None:
        convrot = pytest.importorskip("piper_kernels.linear.convrot")
        torch.manual_seed(10)

        blocks: list[nn.Module] = []
        for _ in range(3):
            block = _Block(width=64)
            qdata = torch.randint(-127, 128, (64, 64), dtype=torch.int8)
            scale = torch.rand(64, 1, dtype=torch.float32)
            block.proj.weight = nn.Parameter(
                convrot.ConvRotInt8Tensor.from_quantized(
                    qdata,
                    scale,
                    group_size=64,
                    logical_dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            blocks.append(block)

        baseline_model = _BlockModel(blocks=blocks)
        rolling_model = copy.deepcopy(baseline_model)
        lora = _lora(3, 64)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        compile_options = convrot.convrot_int8_compile_options()

        expected = _compiled_output(
            baseline_model,
            BlockCompileConfig(
                dynamic=False,
                fullgraph=True,
                options=compile_options,
            ),
            lambda: baseline_model(x).clone(),
            loras=[lora],
            lora_strengths=[0.125],
            lora_mode="merge",
            stochastic_rounding=False,
        )

        rolling_offloader = _make_offloader(
            rolling_model,
            block_compile=BlockCompileConfig(
                dynamic=False,
                rolling=True,
                fullgraph=True,
                options=compile_options,
            ),
        )
        try:
            with activated_model(
                rolling_offloader,
                "cuda",
                loras=[lora],
                lora_strengths=[0.125],
                lora_mode="merge",
                stochastic_rounding=False,
            ):
                assert len({id(block.proj.weight) for block in rolling_model.blocks}) == 1
                with torch.inference_mode():
                    actual = rolling_model(x).clone()
        finally:
            rolling_offloader.deactivate()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @CUDA
    def test_scheduler_orders_lifecycle_around_opaque_reader(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _BlockModel(
            blocks=[_OrderedReaderBlock(), _OrderedReaderBlock()],
        )
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(
                dynamic=False,
                rolling=True,
                fullgraph=True,
            ),
        )
        streamer = streamed_components(offloader)[0]
        runtime = streamer._rolling_runtime
        assert runtime is not None
        runtime_type = type(runtime)
        original_wait = runtime_type.wait_param
        original_rollover = runtime_type.rollover_param
        events: list[str] = []

        def tracked_wait(*args: object, **kwargs: object) -> None:
            events.append("wait")
            original_wait(*args, **kwargs)

        def tracked_rollover(*args: object, **kwargs: object) -> None:
            events.append("refill")
            original_rollover(*args, **kwargs)

        monkeypatch.setattr(runtime_type, "wait_param", tracked_wait)
        monkeypatch.setattr(runtime_type, "rollover_param", tracked_rollover)
        try:
            with activated_model(offloader, "cuda"):
                x = torch.randn(2, 8, device="cuda")
                _ORDER_EVENTS.append(events)
                with torch.inference_mode():
                    model(x)
                    torch.cuda.synchronize()
                events.clear()
                with torch.inference_mode():
                    model(x)
                    torch.cuda.synchronize()
                assert events == [
                    "wait",
                    "reader",
                    "refill",
                    "wait",
                    "reader",
                    "refill",
                ]
        finally:
            if _ORDER_EVENTS:
                assert _ORDER_EVENTS.pop() is events
            offloader.deactivate()

    @CUDA
    def test_reuses_one_target_and_matches_compiled_dynamic_forwards(
        self,
    ) -> None:
        torch.manual_seed(11)
        # More than Dynamo's default eight-entry recompile limit verifies that
        # every homogeneous block reuses the same rolling backend/graph key.
        baseline_model = _BlockModel(num_blocks=12)
        rolling_model = copy.deepcopy(baseline_model)
        inputs = [
            torch.randn(2, 8, device="cuda"),
            torch.randn(3, 8, device="cuda"),
        ]
        expected = _compiled_output(
            baseline_model,
            BlockCompileConfig(fullgraph=True),
            lambda: [baseline_model(x).clone() for x in inputs],
        )

        rolling_offloader = _make_offloader(
            rolling_model,
            block_compile=BlockCompileConfig(
                rolling=True,
                fullgraph=True,
            ),
        )
        streamer = streamed_components(rolling_offloader)[0]
        try:
            with activated_model(rolling_offloader, "cuda"):
                assert streamer._active_runtime is streamer._rolling_runtime
                assert not streamer._block_runtime.acquired
                assert len({id(block.proj.weight) for block in rolling_model.blocks}) == 1
                with torch.inference_mode():
                    actual = [rolling_model(x).clone() for x in inputs]
        finally:
            rolling_offloader.deactivate()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(block.proj.weight.device.type == "cpu" for block in rolling_model.blocks)

    @CUDA
    def test_release_reacquire_rebinds_rolling_target(self) -> None:
        torch.manual_seed(12)
        baseline_model = _BlockModel()
        rolling_model = copy.deepcopy(baseline_model)
        value = torch.randn(2, 8)
        with torch.inference_mode():
            expected = baseline_model(value).cuda()

        offloader = _make_offloader(
            rolling_model,
            block_compile=BlockCompileConfig(
                dynamic=False,
                rolling=True,
                fullgraph=True,
            ),
        )
        streamer = streamed_components(offloader)[0]
        runtime = streamer._rolling_runtime
        assert runtime is not None
        try:
            with activated_model(offloader, "cuda"):
                with torch.inference_mode():
                    first = rolling_model(value.cuda()).clone()
                torch.cuda.synchronize()

                streamer.release()
                assert streamer._active_device is not None
                assert streamer._active_device.type == "cuda"
                assert streamer._active_runtime is runtime
                assert not runtime.acquired
                assert all(
                    block.proj.weight.device.type == "cpu"
                    for block in rolling_model.blocks
                )

                streamer.acquire()
                assert runtime.acquired
                with torch.inference_mode():
                    second = rolling_model(value.cuda()).clone()
                torch.cuda.synchronize()
        finally:
            offloader.deactivate()

        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, first, rtol=0, atol=0)

    @CUDA
    def test_transient_block_path_reacquires_rolling_target(self) -> None:
        torch.manual_seed(13)
        baseline_model = _BlockModel()
        rolling_model = copy.deepcopy(baseline_model)
        value = torch.randn(2, 8)
        with torch.inference_mode():
            expected = baseline_model(value).cuda()

        offloader = _make_offloader(
            rolling_model,
            block_compile=BlockCompileConfig(
                dynamic=False,
                rolling=True,
                fullgraph=True,
            ),
            transient_block_paths=("blocks",),
        )
        runtime = streamed_components(offloader)[0]._rolling_runtime
        assert runtime is not None
        root_states: list[bool] = []
        remove_observer = offloader.register_forward_hook(
            "",
            lambda _module, _args, _output: root_states.append(runtime.acquired),
        )
        try:
            with activated_model(offloader, "cuda"):
                with torch.inference_mode():
                    first = rolling_model(value.cuda()).clone()
                    second = rolling_model(value.cuda()).clone()
                torch.cuda.synchronize()
                assert root_states == [False, False]
                assert runtime.acquired
        finally:
            remove_observer()
            offloader.deactivate()

        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, first, rtol=0, atol=0)

    @CUDA
    def test_transient_block_path_stops_rollover_at_final_block(self) -> None:
        model = _BlockModel(num_blocks=3)
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(
                dynamic=False,
                rolling=True,
                fullgraph=True,
            ),
            transient_block_paths=("blocks",),
        )
        runtime = streamed_components(offloader)[0]._rolling_runtime
        assert runtime is not None
        original_refill = runtime._refill
        refills: list[int] = []

        def record_refill(block_idx: int, param_idx: int) -> None:
            refills.append(block_idx)
            original_refill(block_idx, param_idx)

        runtime._refill = record_refill  # type: ignore[method-assign]
        try:
            with activated_model(offloader, "cuda"):
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
                torch.cuda.synchronize()
        finally:
            offloader.deactivate()

        assert refills == [1, 2]

    @CUDA
    def test_transient_rolling_target_survives_separate_activations(self) -> None:
        torch.manual_seed(14)
        baseline_model = _BlockModel()
        rolling_model = copy.deepcopy(baseline_model)
        value = torch.randn(2, 8, device="cuda")
        with torch.inference_mode():
            expected = baseline_model(value.cpu()).cuda()

        offloader = _make_offloader(
            rolling_model,
            block_compile=BlockCompileConfig(
                dynamic=False,
                rolling=True,
                fullgraph=True,
            ),
            transient_block_paths=("blocks",),
        )
        with torch.inference_mode():
            with activated_model(offloader, "cuda"):
                first = rolling_model(value).clone()
            with activated_model(offloader, "cuda"):
                second = rolling_model(value).clone()
        torch.cuda.synchronize()

        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, first, rtol=0, atol=0)

    @CUDA
    def test_routed_lora_selects_block_runtime_for_activation(self) -> None:
        model = _BlockModel()
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(rolling=True, fullgraph=True),
        )
        streamer = streamed_components(offloader)[0]
        try:
            with activated_model(
                offloader,
                "cuda",
                loras=[_lora(len(model.blocks), 8)],
                lora_mode="routed",
            ):
                assert streamer._active_runtime is streamer._block_runtime
                assert streamer._block_runtime.acquired
                assert streamer._rolling_runtime is not None
                assert not streamer._rolling_runtime.acquired
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
            assert streamer._active_runtime is None
        finally:
            offloader.deactivate()

    @CUDA
    def test_composes_after_piper_kernel_graph_rewrites(self) -> None:
        convrot = pytest.importorskip("piper_kernels.linear.convrot")
        torch.manual_seed(12)
        baseline_model = _BlockModel()
        rolling_model = copy.deepcopy(baseline_model)
        x = torch.randn(2, 8, device="cuda")
        compile_options = convrot.convrot_int8_compile_options()

        expected = _compiled_output(
            baseline_model,
            BlockCompileConfig(
                fullgraph=True,
                options=compile_options,
            ),
            lambda: baseline_model(x).clone(),
        )

        actual = _compiled_output(
            rolling_model,
            BlockCompileConfig(
                rolling=True,
                fullgraph=True,
                options=compile_options,
            ),
            lambda: rolling_model(x).clone(),
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @CUDA
    def test_matches_merge_mode_across_repeated_forwards(self) -> None:
        torch.manual_seed(13)
        baseline_model = _BlockModel(num_blocks=3)
        rolling_model = copy.deepcopy(baseline_model)
        lora = _lora(3, 8)
        x = torch.randn(2, 8, device="cuda")

        expected = _compiled_output(
            baseline_model,
            BlockCompileConfig(fullgraph=True),
            lambda: [baseline_model(x).clone() for _ in range(2)],
            loras=[lora],
            lora_strengths=[0.25],
            lora_mode="merge",
        )

        actual = _compiled_output(
            rolling_model,
            BlockCompileConfig(
                rolling=True,
                fullgraph=True,
            ),
            lambda: [rolling_model(x).clone() for _ in range(2)],
            loras=[lora],
            lora_strengths=[0.25],
            lora_mode="merge",
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
