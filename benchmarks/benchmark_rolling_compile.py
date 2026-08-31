"""Compare ordinary compiled block prefetch with one-target rollover.

The synthetic block contains two projections so rollover has useful compute
after the first parameter becomes dead. Both modes use the same initialized
weights and input; the script verifies their outputs before reporting latency
and CUDA allocator residency.
"""

# ruff: noqa: T201 - benchmark CLI intentionally prints a report.

import argparse
import gc
import statistics
from dataclasses import dataclass

import torch
from torch import nn

from piper_offload import (
    BlockCompileConfig,
    BlockMode,
    ModelOffloader,
)

MIB = 1024**2


class _Block(nn.Module):
    def __init__(self, width: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.up = nn.Linear(width, width * 4, bias=False, dtype=dtype)
        self.down = nn.Linear(width * 4, width, bias=False, dtype=dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.down(torch.nn.functional.silu(self.up(value)))


class _Model(nn.Module):
    def __init__(
        self,
        *,
        blocks: int,
        width: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(width, dtype) for _ in range(blocks))
        self.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


@dataclass(frozen=True, slots=True)
class _Result:
    mode: str
    milliseconds: tuple[float, ...]
    allocated_bytes: int
    peak_bytes: int
    output: torch.Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("blocks", "width", "tokens", "repeats"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name} must be positive, got {value}")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _run(
    *,
    block_mode: BlockMode,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    input_tensor: torch.Tensor,
) -> _Result:
    torch.manual_seed(args.seed)
    model = _Model(blocks=args.blocks, width=args.width, dtype=dtype)
    offloader = ModelOffloader.from_module(
        model,
        block_paths=("blocks",),
        block_mode=block_mode,
        block_compile=BlockCompileConfig(fullgraph=True),
    )
    milliseconds: list[float] = []
    output: torch.Tensor | None = None
    try:
        offloader.activate(device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                output = model(input_tensor)
                torch.cuda.synchronize(device)

            torch.cuda.reset_peak_memory_stats(device)
            allocated_bytes = torch.cuda.memory_allocated(device)
            for _ in range(args.repeats):
                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record()
                output = model(input_tensor)
                finished.record()
                finished.synchronize()
                milliseconds.append(started.elapsed_time(finished))
            peak_bytes = torch.cuda.max_memory_allocated(device)

        assert output is not None
        host_output = output.cpu()
    finally:
        offloader.deactivate()

    return _Result(
        mode=block_mode,
        milliseconds=tuple(milliseconds),
        allocated_bytes=allocated_bytes,
        peak_bytes=peak_bytes,
        output=host_output,
    )


def _format_result(result: _Result) -> str:
    median = statistics.median(result.milliseconds)
    spread = max(result.milliseconds) - min(result.milliseconds)
    return (
        f"{result.mode:<15} {median:9.2f} ms "
        f"(range {spread:7.2f}), allocated={result.allocated_bytes / MIB:8.1f} "
        f"MiB, peak={result.peak_bytes / MIB:8.1f} MiB"
    )


def _main() -> None:
    args = _parse_args()
    _validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires an available CUDA device")
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)

    torch.manual_seed(args.seed + 1)
    input_tensor = torch.randn(
        args.tokens,
        args.width,
        dtype=dtype,
        device=device,
    )
    baseline = _run(
        block_mode="streaming",
        args=args,
        device=device,
        dtype=dtype,
        input_tensor=input_tensor,
    )
    gc.collect()
    torch.cuda.empty_cache()
    rolling = _run(
        block_mode="rolling",
        args=args,
        device=device,
        dtype=dtype,
        input_tensor=input_tensor,
    )
    torch.testing.assert_close(
        rolling.output,
        baseline.output,
        rtol=0,
        atol=0,
    )

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Configuration: {args.blocks} blocks, width={args.width}, tokens={args.tokens}, dtype={args.dtype}")
    print(_format_result(baseline))
    print(_format_result(rolling))
    baseline_median = statistics.median(baseline.milliseconds)
    rolling_median = statistics.median(rolling.milliseconds)
    print(f"Rolling / baseline latency: {rolling_median / baseline_median:.3f}x")
    print(f"Allocated memory saved: {(baseline.allocated_bytes - rolling.allocated_bytes) / MIB:.1f} MiB")


if __name__ == "__main__":
    _main()
