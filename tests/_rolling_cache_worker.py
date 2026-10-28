"""Fresh-process checks for rolling graph persistence and runtime rebinding."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch._dynamo.utils import counters

from piper_offload import BlockCompileConfig
from piper_offload.rolling_compile import (
    register_rolling_target,
    rolling_inductor_backend,
    rolling_storage_tensors,
    unregister_rolling_target,
)
from tests._block_compile_helpers import _BlockModel, _make_offloader
from tests.conftest import activated_model


class _RepeatedReader(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 64), requires_grad=False)
        self.bias = nn.Parameter(torch.randn(64), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = torch.nn.functional.linear(value, self.weight, self.bias).relu()
        return torch.nn.functional.linear(first, self.weight, self.bias)


class _PoisoningRuntime:
    """Make a premature refill visible even without asynchronous CUDA copies."""

    def __init__(self, params: list[nn.Parameter]) -> None:
        self.params = params
        self.events: list[tuple[str, int]] = []

    def wait_param(self, param_idx: int) -> None:
        self.events.append(("wait", param_idx))

    def rollover_param(self, param_idx: int) -> None:
        self.events.append(("refill", param_idx))
        for storage in rolling_storage_tensors(self.params[param_idx]):
            storage.fill_(float("nan") if storage.is_floating_point() else 0)


def _run_cpu(reverse: bool, quantized: bool) -> None:
    block = _RepeatedReader().eval()
    if quantized:
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        block.weight = nn.Parameter(
            ConvRotInt8Tensor.from_hp(block.weight, group_size=64), requires_grad=False,
        )
    params = list(block.parameters())
    if reverse:
        params.reverse()
    runtime = _PoisoningRuntime(params)
    value = torch.randn(5, 64)
    with torch.inference_mode():
        expected = block(value)
        register_rolling_target(runtime, params)
        try:
            compiled = torch.compile(
                block.forward, backend=rolling_inductor_backend, fullgraph=True, dynamic=True,
            )
            actual = compiled(value)
            torch.testing.assert_close(actual, expected)
            assert sorted(runtime.events) == sorted([
                ("wait", 0), ("wait", 1), ("refill", 0), ("refill", 1),
            ])
            assert all(
                torch.isnan(storage).all()
                for param in params
                for storage in rolling_storage_tensors(param)
                if storage.is_floating_point()
            )
        finally:
            unregister_rolling_target(runtime)


class _FFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
        self.down = nn.Linear(128, 64, bias=False, dtype=torch.bfloat16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.gelu(self.up(value), approximate="tanh"))


def _run_cuda(quantized: bool) -> None:
    blocks = [_FFN() for _ in range(3)]
    options = None
    if quantized:
        from piper_kernels.fusions.convrot_int8_gelu_ffn import convrot_int8_gelu_ffn_compile_options
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        options = convrot_int8_gelu_ffn_compile_options()
        for block in blocks:
            for linear in (block.up, block.down):
                linear.weight = nn.Parameter(
                    ConvRotInt8Tensor.from_hp(linear.weight, group_size=64), requires_grad=False,
                )
    model = _BlockModel(blocks=blocks)
    baseline = copy.deepcopy(model)
    config = BlockCompileConfig(fullgraph=True, options=options)
    value = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
    streaming = _make_offloader(baseline, block_mode="streaming", block_compile=config)
    with activated_model(streaming, "cuda"), torch.inference_mode():
        expected = baseline(value).clone()

    # Count only rolling cache lookups, excluding the reference compilation.
    counters["inductor"].clear()
    rolling = _make_offloader(model, block_mode="rolling", block_compile=config)
    for _ in range(2):
        with activated_model(rolling, "cuda"), torch.inference_mode():
            actual = model(value).clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("mode", choices=("cpu", "cpu-int8", "cuda", "cuda-int8"))
    parser.add_argument("seed", type=int)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = []
    for round_idx in range(2):
        torch.compiler.reset()
        counters["inductor"].clear()
        torch.manual_seed(args.seed + round_idx)
        if args.mode.startswith("cpu"):
            _run_cpu(args.reverse, args.mode == "cpu-int8")
        else:
            _run_cuda(args.mode == "cuda-int8")
        results.append({
            key: counters["inductor"][f"fxgraph_cache_{key}"]
            for key in ("hit", "miss", "bypass")
        })
    args.result.write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    main()
