"""Compare direct pageable H2D copies with a bounded pinned staging pool.

The benchmark models block streaming rather than timing only one copy:

* ``direct`` copies pageable CPU storage to a GPU target with
  ``non_blocking=True`` and lets CUDA perform its implicit staging.
* ``staged`` first copies into one of a bounded number of reusable pinned
  slots, then copies that slot to the GPU on a transfer stream.

Run this same file on native Linux, WSL2, and native Windows to compare the
platforms with identical PyTorch-level behavior.
"""

# ruff: noqa: PLR0915, T201 - benchmark CLI intentionally prints a detailed report.

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class _Workload:
    label: str
    run: Callable[[], None]
    measured_ms: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-mib", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--source-slots", type=int, default=4)
    parser.add_argument("--target-slots", type=int, default=2)
    parser.add_argument("--staging-slots", type=int, default=2)
    parser.add_argument("--compute-ms", type=float, default=5.0)
    parser.add_argument("--matmul-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "block_mib": args.block_mib,
        "blocks": args.blocks,
        "source_slots": args.source_slots,
        "target_slots": args.target_slots,
        "staging_slots": args.staging_slots,
        "matmul_size": args.matmul_size,
        "repeats": args.repeats,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}")
    if args.compute_ms < 0:
        raise ValueError(f"--compute-ms must be non-negative, got {args.compute_ms}")
    if args.warmup < 0:
        raise ValueError(f"--warmup must be non-negative, got {args.warmup}")


def _event_time_ms(stream: torch.cuda.Stream, operation: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        operation()
        end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _make_matmul_workload(
    device: torch.device,
    *,
    target_ms: float,
    matrix_size: int,
) -> _Workload:
    if target_ms == 0:
        return _Workload("none", lambda: None, 0.0)

    a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
    out = torch.empty_like(a)
    stream = torch.cuda.Stream(device=device)

    def one_matmul() -> None:
        torch.mm(a, b, out=out)

    for _ in range(3):
        with torch.cuda.stream(stream):
            one_matmul()
    stream.synchronize()
    one_ms = _event_time_ms(stream, one_matmul)
    matmuls = max(1, round(target_ms / one_ms))

    def run() -> None:
        for _ in range(matmuls):
            one_matmul()

    measured_ms = _event_time_ms(stream, run)
    return _Workload(
        label=f"{matmuls} x fp16 {matrix_size}x{matrix_size} matmul",
        run=run,
        measured_ms=measured_ms,
    )


def _run_pipeline(
    mode: str,
    *,
    sources: list[torch.Tensor],
    targets: list[torch.Tensor],
    staging: list[torch.Tensor],
    blocks: int,
    workload: _Workload,
    device: torch.device,
) -> float:
    transfer_stream = torch.cuda.Stream(device=device, priority=-1)
    compute_stream = torch.cuda.Stream(device=device)
    load_events = [torch.cuda.Event() for _ in range(blocks)]
    compute_done = [torch.cuda.Event() for _ in targets]
    staging_done: list[torch.cuda.Event | None] = [None for _ in staging]

    def enqueue_load(block: int) -> None:
        target_idx = block % len(targets)
        source = sources[block % len(sources)]
        if block >= len(targets):
            transfer_stream.wait_event(compute_done[target_idx])

        with torch.cuda.stream(transfer_stream):
            if mode == "direct":
                targets[target_idx].copy_(source, non_blocking=True)
            elif mode == "staged":
                staging_idx = block % len(staging)
                previous = staging_done[staging_idx]
                if previous is not None:
                    # The CPU must not overwrite a slot while a prior DMA is
                    # still reading it. Waiting here still permits the other
                    # slot's DMA and the current GPU compute to overlap.
                    previous.synchronize()
                staging[staging_idx].copy_(source)
                targets[target_idx].copy_(staging[staging_idx], non_blocking=True)
                staging_done[staging_idx] = load_events[block]
            else:
                raise ValueError(f"unknown pipeline mode {mode!r}")
            load_events[block].record()

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    enqueue_load(0)
    for block in range(blocks):
        target_idx = block % len(targets)
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(load_events[block])
            workload.run()
            compute_done[target_idx].record()
        if block + 1 < blocks:
            enqueue_load(block + 1)
    compute_stream.synchronize()
    transfer_stream.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _run_compute_only(
    workload: _Workload,
    *,
    blocks: int,
    device: torch.device,
) -> float:
    stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(blocks):
            workload.run()
    stream.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _run_primitive(
    operation: Callable[[], None],
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        operation()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def _format_result(name: str, values: list[float], total_bytes: int | None = None) -> str:
    median_ms = statistics.median(values)
    spread = max(values) - min(values)
    suffix = ""
    if total_bytes is not None:
        gib_per_s = (total_bytes / GIB) / (median_ms / 1000.0)
        suffix = f", {gib_per_s:6.2f} GiB/s effective"
    return f"{name:<28} {median_ms:9.2f} ms  (range {spread:6.2f} ms){suffix}"


def _main() -> None:
    args = _parse_args()
    _validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires an available CUDA device")

    torch.cuda.set_device(device)
    block_bytes = args.block_mib * MIB
    sources = [torch.empty(block_bytes, dtype=torch.uint8) for _ in range(args.source_slots)]
    for idx, source in enumerate(sources):
        source.fill_((idx + 1) % 256)
    targets = [torch.empty(block_bytes, dtype=torch.uint8, device=device) for _ in range(args.target_slots)]
    staging = [torch.empty(block_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(args.staging_slots)]
    workload = _make_matmul_workload(
        device,
        target_ms=args.compute_ms,
        matrix_size=args.matmul_size,
    )
    transfer_stream = torch.cuda.Stream(device=device)
    primitive_blocks = args.blocks
    primitive_bytes = primitive_blocks * block_bytes

    def pageable_h2d() -> None:
        with torch.cuda.stream(transfer_stream):
            for block in range(primitive_blocks):
                targets[0].copy_(sources[block % len(sources)], non_blocking=True)
        transfer_stream.synchronize()

    def pinned_h2d() -> None:
        with torch.cuda.stream(transfer_stream):
            for _block in range(primitive_blocks):
                targets[0].copy_(staging[0], non_blocking=True)
        transfer_stream.synchronize()

    def pageable_to_pinned() -> None:
        for block in range(primitive_blocks):
            staging[0].copy_(sources[block % len(sources)])

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    print(
        "Configuration: "
        f"{args.blocks} blocks x {args.block_mib} MiB, "
        f"{args.source_slots} pageable sources, {args.target_slots} GPU targets, "
        f"{args.staging_slots} pinned staging slots"
    )
    print(f"Synthetic compute: {workload.label} ({workload.measured_ms:.2f} ms/block measured)")

    print("\nIsolated operations")
    primitive_results = {
        "pageable -> GPU": _run_primitive(pageable_h2d, warmup=args.warmup, repeats=args.repeats),
        "pageable -> pinned": _run_primitive(
            pageable_to_pinned,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "pinned -> GPU": _run_primitive(pinned_h2d, warmup=args.warmup, repeats=args.repeats),
    }
    for name, values in primitive_results.items():
        print(_format_result(name, values, primitive_bytes))

    workloads = [
        ("copy-only", _Workload("none", lambda: None, 0.0)),
        ("copy + compute", workload),
    ]
    total_bytes = args.blocks * block_bytes
    print("\nStreamed pipelines")
    for scenario, scenario_workload in workloads:
        results = {"direct": [], "staged": []}
        for run_idx in range(args.warmup + args.repeats):
            order = ("direct", "staged") if run_idx % 2 == 0 else ("staged", "direct")
            for mode in order:
                elapsed_ms = _run_pipeline(
                    mode,
                    sources=sources,
                    targets=targets,
                    staging=staging,
                    blocks=args.blocks,
                    workload=scenario_workload,
                    device=device,
                )
                if run_idx >= args.warmup:
                    results[mode].append(elapsed_ms)

        direct_ms = statistics.median(results["direct"])
        staged_ms = statistics.median(results["staged"])
        print(f"{scenario}:")
        print("  " + _format_result("direct pageable", results["direct"], total_bytes))
        print("  " + _format_result("bounded pinned staging", results["staged"], total_bytes))
        if staged_ms < direct_ms:
            print(f"  staged is {direct_ms / staged_ms:.2f}x faster")
        else:
            print(f"  direct is {staged_ms / direct_ms:.2f}x faster")

    if workload.measured_ms > 0:
        compute_values = _run_primitive(
            lambda: _run_compute_only(workload, blocks=args.blocks, device=device),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        print("\nCompute-only reference")
        print(_format_result("compute only", compute_values))


if __name__ == "__main__":
    with torch.inference_mode():
        _main()
