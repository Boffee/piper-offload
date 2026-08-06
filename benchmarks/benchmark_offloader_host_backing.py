"""Benchmark pinned copies versus adopted model streaming.

Unlike the raw transfer microbenchmark, this exercises the complete
``ModelOffloader`` block lifecycle: host-store construction, pooled GPU
targets, background prefetch, forward hooks, and actual Linear compute.

Run the same command on Linux, WSL2, and native Windows with matching PyTorch
and benchmark arguments for a platform comparison.
"""

# ruff: noqa: T201 - benchmark CLI intentionally prints a report.

import argparse
import gc
import platform
import statistics
import time
from dataclasses import dataclass

import torch
from torch import nn

from piper_offload import HostBacking, ModelOffloader, StreamConfig

GIB = 1024**3


class _BlockModel(nn.Module):
    def __init__(
        self,
        *,
        blocks: int,
        width: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Linear(width, width, bias=False, dtype=dtype)
            for _ in range(blocks)
        )
        self.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


@dataclass(frozen=True, slots=True)
class _Result:
    mode: HostBacking
    construction_ms: float
    cache_bytes: int
    forward_ms: tuple[float, ...]
    output: torch.Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--resident-blocks", type=int, default=1)
    parser.add_argument("--prefetch-blocks", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--order",
        choices=("pinned-first", "adopt-first"),
        default="pinned-first",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "blocks": args.blocks,
        "width": args.width,
        "batch_size": args.batch_size,
        "resident_blocks": args.resident_blocks,
        "repeats": args.repeats,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be positive, got {value}"
            )
    if args.prefetch_blocks < 0:
        raise ValueError(
            "--prefetch-blocks must be non-negative, "
            f"got {args.prefetch_blocks}"
        )
    if args.warmup < 0:
        raise ValueError(f"--warmup must be non-negative, got {args.warmup}")


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _build_offloader(
    mode: HostBacking,
    *,
    blocks: int,
    width: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[ModelOffloader, float]:
    torch.manual_seed(seed)
    model = _BlockModel(blocks=blocks, width=width, dtype=dtype)
    source_ptrs = {
        name: parameter.data_ptr()
        for name, parameter in model.named_parameters()
    }
    started = time.perf_counter()
    offloader = ModelOffloader.from_module(
        model,
        blocks_attr=("blocks",),
        host_backing=mode,
    )
    construction_ms = (time.perf_counter() - started) * 1000.0

    expected_pinned = mode == "pinned"
    actual = {parameter.is_pinned() for parameter in model.parameters()}
    if actual != {expected_pinned}:
        raise RuntimeError(
            f"{mode} construction produced unexpected is_pinned states {actual}"
        )
    if mode == "adopt":
        adopted_ptrs = {
            name: parameter.data_ptr()
            for name, parameter in model.named_parameters()
        }
        if adopted_ptrs != source_ptrs:
            raise RuntimeError("adopted construction copied source storage")
    return offloader, construction_ms


def _run_mode(
    mode: HostBacking,
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    input_tensor: torch.Tensor,
) -> _Result:
    offloader, construction_ms = _build_offloader(
        mode,
        blocks=args.blocks,
        width=args.width,
        dtype=dtype,
        seed=args.seed,
    )
    config = StreamConfig(
        num_resident_blocks=args.resident_blocks,
        num_prefetch_blocks=args.prefetch_blocks,
        cyclic=True,
    )
    values: list[float] = []
    output: torch.Tensor | None = None
    try:
        offloader.activate(device, stream_config=config)
        for _ in range(args.warmup):
            output = offloader.value(input_tensor)
            torch.cuda.synchronize(device)

        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = offloader.value(input_tensor)
            torch.cuda.synchronize(device)
            values.append((time.perf_counter() - started) * 1000.0)

        assert output is not None
        host_output = output.detach().cpu()
    finally:
        offloader.deactivate()

    return _Result(
        mode=mode,
        construction_ms=construction_ms,
        cache_bytes=offloader.cache_bytes,
        forward_ms=tuple(values),
        output=host_output,
    )


def _format_forward(result: _Result) -> str:
    median_ms = statistics.median(result.forward_ms)
    spread_ms = max(result.forward_ms) - min(result.forward_ms)
    gib_per_s = (result.cache_bytes / GIB) / (median_ms / 1000.0)
    return (
        f"{result.mode:<9} {median_ms:9.2f} ms "
        f"(range {spread_ms:6.2f} ms), "
        f"{gib_per_s:6.2f} GiB/s effective"
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
        args.batch_size,
        args.width,
        dtype=dtype,
        device=device,
    )
    order: tuple[HostBacking, HostBacking] = (
        ("pinned", "adopt")
        if args.order == "pinned-first"
        else ("adopt", "pinned")
    )

    print(f"Platform: {platform.platform()}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    print(
        "Configuration: "
        f"{args.blocks} blocks, {args.width}x{args.width} {args.dtype} weights, "
        f"batch={args.batch_size}, resident={args.resident_blocks}, "
        f"prefetch={args.prefetch_blocks}"
    )

    results: dict[HostBacking, _Result] = {}
    for mode in order:
        result = _run_mode(
            mode,
            args=args,
            device=device,
            dtype=dtype,
            input_tensor=input_tensor,
        )
        results[mode] = result
        gc.collect()

    pinned = results["pinned"]
    adopted = results["adopt"]
    torch.testing.assert_close(adopted.output, pinned.output)
    cache_mib = pinned.cache_bytes / 1024**2

    print(f"Host backing: {cache_mib:.1f} MiB")
    print("\nConstruction")
    print(f"pinned    {pinned.construction_ms:9.2f} ms")
    print(f"adopt     {adopted.construction_ms:9.2f} ms")
    print("\nSteady-state streamed forward")
    print(_format_forward(pinned))
    print(_format_forward(adopted))

    pinned_ms = statistics.median(pinned.forward_ms)
    adopted_ms = statistics.median(adopted.forward_ms)
    if adopted_ms < pinned_ms:
        print(f"adopt is {pinned_ms / adopted_ms:.2f}x faster")
    else:
        print(f"pinned is {adopted_ms / pinned_ms:.2f}x faster")
    print("Outputs match.")


if __name__ == "__main__":
    _main()
