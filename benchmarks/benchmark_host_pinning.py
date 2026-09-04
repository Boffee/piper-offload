"""Measure real block streaming with cold, retained, and competing host pins.

Compiler/kernel warmup is excluded. Cold means host registrations were cleared,
not a cold CUDA context or compiler. Two independently initialized model
backings remain alive; alternating sessions share a budget expressed as a
multiple of one model's page-rounded storage. Each measured session's final
output must equal that model/mode's warmed zero-budget reference exactly.
"""

# ruff: noqa: T201 - benchmark CLI prints progress and a result table.

import argparse
import contextlib
import gc
import json
import math
import mmap
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

from piper_offload import BlockCompileConfig, BlockMode, ModelOffloader, PinLease, host_pin_manager
from piper_offload._host_registration import RuntimeHostRegistration
from piper_offload.block_runtime import host_transfer_tensors
from piper_offload.streaming_runtime import StreamingBlockRuntime

GIB = 1024**3


@dataclass
class _PinningTotals:
    acquire_ms: float = 0.0
    register_calls: int = 0
    register_bytes: int = 0
    register_ms: float = 0.0
    unregister_calls: int = 0
    unregister_ms: float = 0.0


class _TimedRegistration(RuntimeHostRegistration):
    """Benchmark instrumentation around the actual native registration backend."""

    def __init__(self) -> None:
        self.totals = _PinningTotals()

    def register(self, pointer: int, size: int) -> bool:
        start = time.perf_counter()
        self.totals.register_calls += 1
        try:
            registered = super().register(pointer, size)
            if registered:
                self.totals.register_bytes += size
            return registered
        finally:
            self.totals.register_ms += (time.perf_counter() - start) * 1000

    def unregister(self, pointer: int) -> None:
        start = time.perf_counter()
        self.totals.unregister_calls += 1
        try:
            super().unregister(pointer)
        finally:
            self.totals.unregister_ms += (time.perf_counter() - start) * 1000


@contextlib.contextmanager
def _instrument_registration() -> Generator[_TimedRegistration]:
    # Only this standalone benchmark replaces the manager's backend, to time
    # native calls without adding instrumentation to the production hot path.
    if host_pin_manager.stats.active_leases or host_pin_manager.stats.registrations:
        raise RuntimeError("run the benchmark in a fresh process without existing pin leases")
    original_backend = host_pin_manager._backend
    original_budget = host_pin_manager.max_pinned_bytes
    backend = _TimedRegistration()
    host_pin_manager._backend = backend
    original_acquire = host_pin_manager.acquire

    def measured_acquire(tensors: Iterable[torch.Tensor]) -> PinLease:
        start = time.perf_counter()
        try:
            return original_acquire(tensors)
        finally:
            backend.totals.acquire_ms += (time.perf_counter() - start) * 1000

    try:
        # Exclude lazy runtime-library discovery from model registration cost.
        host_pin_manager.max_pinned_bytes = 2 * mmap.PAGESIZE
        source = torch.zeros(1)
        with host_pin_manager.acquire([source]):
            pass
        host_pin_manager.clear()
        host_pin_manager.max_pinned_bytes = 0
        with patch.object(host_pin_manager, "acquire", measured_acquire):
            yield backend
    finally:
        host_pin_manager.clear()
        host_pin_manager.max_pinned_bytes = original_budget
        host_pin_manager._backend = original_backend


def _weight(rows: int, cols: int, representation: str, dtype: torch.dtype) -> nn.Parameter:
    if representation == "dense":
        bound = 1 / math.sqrt(cols)
        data = torch.empty(rows, cols, dtype=dtype).uniform_(-bound, bound)
    else:
        from piper_kernels.linear.convrot import ConvRotInt8Tensor  # noqa: PLC0415

        # Construct valid synthetic packed weights directly, avoiding a dense
        # quantization scratch model. Scale metadata is part of the pin budget.
        data = ConvRotInt8Tensor.from_quantized(
            torch.randint(-127, 128, (rows, cols), dtype=torch.int8),
            torch.full((rows, 1), 1 / (127 * math.sqrt(cols)), dtype=torch.float32),
            group_size=64,
            logical_dtype=dtype,
        )
    return nn.Parameter(data, requires_grad=False)


class _Block(nn.Module):
    def __init__(self, width: int, representation: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.up = nn.Linear(width, width * 4, bias=False, device="meta", dtype=dtype)
        self.down = nn.Linear(width * 4, width, bias=False, device="meta", dtype=dtype)
        self.up.weight = _weight(width * 4, width, representation, dtype)
        self.down.weight = _weight(width, width * 4, representation, dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 0.1 * self.down(torch.nn.functional.silu(self.up(value)))


class _Model(nn.Module):
    def __init__(self, blocks: int, width: int, representation: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(width, representation, dtype) for _ in range(blocks))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


@dataclass
class _Subject:
    name: str
    offloader: ModelOffloader
    storage_bytes: int
    page_bytes: int
    allocations: int
    reference: torch.Tensor | None = None


def _build_subject(name: str, mode: BlockMode, representation: str, args: argparse.Namespace) -> _Subject:
    torch.manual_seed(args.seed + (name == "B"))
    model = _Model(args.blocks, args.width, representation, getattr(torch, args.dtype))
    offloader = ModelOffloader.from_module(
        model,
        block_paths=("blocks",),
        block_mode=mode,
        block_compile=BlockCompileConfig(fullgraph=True, dynamic=False),
    )
    # Benchmark-only inspection of the actual captured allocations, including
    # quant metadata. Merge page intervals without allocating a set per page.
    stores: dict[int, int] = {}
    for component in offloader._composite.blocks:
        plans = [instance.resolve_load_plan() for instance in component._block_instances]
        for tensor in host_transfer_tensors(plans):
            if tensor.numel():
                storage = tensor.untyped_storage()
                stores[storage.data_ptr()] = storage.nbytes()
    pages, prior_end = 0, 0
    for pointer, size in sorted(stores.items()):
        start, end = pointer // mmap.PAGESIZE, (pointer + size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        pages += max(0, end - max(start, prior_end))
        prior_end = max(prior_end, end)
    return _Subject(name, offloader, sum(stores.values()), pages * mmap.PAGESIZE, len(stores))


def _finish_work(subject: _Subject, device: torch.device) -> None:
    # GPU synchronization alone can miss a wraparound copy that the CPU
    # prefetch worker has not enqueued yet. Account for the complete traversal.
    for component in subject.offloader._composite.blocks:
        runtime = component._active_runtime
        if isinstance(runtime, StreamingBlockRuntime):
            for future in tuple(runtime._pending.values()):
                future.result()
    torch.cuda.synchronize(device)


def _warm_subject(subject: _Subject, value: torch.Tensor, repeats: int) -> None:
    try:
        subject.offloader.activate(value.device)
        with torch.inference_mode():
            for _ in range(repeats):
                output = subject.offloader.value(value)
                _finish_work(subject, value.device)
        subject.reference = output.cpu()
        if not torch.isfinite(subject.reference).all():
            raise RuntimeError("reference output is not finite")
    finally:
        subject.offloader.deactivate()


def _phase(operation: Callable[[], object], backend: _TimedRegistration) -> dict[str, float | int]:
    before = asdict(backend.totals)
    start = time.perf_counter()
    operation()
    elapsed = (time.perf_counter() - start) * 1000
    return {"ms": elapsed, **{name: value - before[name] for name, value in asdict(backend.totals).items()}}


def _session(
    subject: _Subject,
    value: torch.Tensor,
    forwards: int,
    backend: _TimedRegistration,
) -> dict[str, Any]:
    torch.cuda.synchronize(value.device)
    before_failures = host_pin_manager.stats.registration_failures
    memory_before = torch.cuda.memory_stats(value.device)
    output: torch.Tensor | None = None

    def activate() -> None:
        subject.offloader.activate(value.device)
        _finish_work(subject, value.device)

    def forward() -> None:
        nonlocal output
        output = subject.offloader.value(value)
        _finish_work(subject, value.device)

    try:
        activation = _phase(activate, backend)
        active = host_pin_manager.stats
        leases: list[PinLease] = [
            component._active_runtime._host_lease for component in subject.offloader._composite.blocks
        ]
        registered_bytes = sum(lease.registered_bytes for lease in leases)
        pageable_bytes = sum(lease.pageable_bytes for lease in leases)
        with torch.inference_mode():
            passes = [_phase(forward, backend) for _ in range(forwards)]
        peak_gpu_bytes = torch.cuda.max_memory_allocated(value.device)
    finally:
        deactivation = _phase(subject.offloader.deactivate, backend)
    assert output is not None
    assert subject.reference is not None
    torch.testing.assert_close(output.cpu(), subject.reference, rtol=0, atol=0)
    after = host_pin_manager.stats
    memory_after = torch.cuda.memory_stats(value.device)
    if after.active_leases or after.pinned_bytes > after.max_pinned_bytes:
        raise RuntimeError("session leaked an active lease or exceeded the pin budget")
    return {
        "subject": subject.name,
        "activation": activation,
        "forwards": passes,
        "deactivation": deactivation,
        "registered_bytes": registered_bytes,
        "pageable_bytes": pageable_bytes,
        "active_pinned_page_bytes": active.pinned_bytes,
        "idle_pinned_page_bytes": after.pinned_bytes,
        "registration_failures": after.registration_failures - before_failures,
        "peak_gpu_bytes": peak_gpu_bytes,
        "cuda_allocation_retries": memory_after["num_alloc_retries"] - memory_before["num_alloc_retries"],
        "cuda_ooms": memory_after["num_ooms"] - memory_before["num_ooms"],
    }


def _gpu_sample() -> str | None:
    executable = shutil.which("nvidia-smi")
    if torch.version.hip is not None or executable is None:
        return None
    result = subprocess.run(
        [executable, "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.free", "--format=csv"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _summarize(samples: list[dict[str, Any]]) -> dict[str, float]:
    median = statistics.median
    return {
        "activation_ms": median(sample["activation"]["ms"] for sample in samples),
        "acquire_ms": median(sample["activation"]["acquire_ms"] for sample in samples),
        "acquire_bookkeeping_ms": median(
            sample["activation"]["acquire_ms"]
            - sample["activation"]["register_ms"]
            - sample["activation"]["unregister_ms"]
            for sample in samples
        ),
        "register_ms": median(sample["activation"]["register_ms"] for sample in samples),
        "unregister_ms": median(sample["activation"]["unregister_ms"] for sample in samples),
        "register_calls": median(sample["activation"]["register_calls"] for sample in samples),
        "unregister_calls": median(sample["activation"]["unregister_calls"] for sample in samples),
        "forward_ms": median(passed["ms"] for sample in samples for passed in sample["forwards"]),
        "deactivation_ms": median(sample["deactivation"]["ms"] for sample in samples),
        "registered_gib": median(sample["registered_bytes"] / GIB for sample in samples),
        "registration_failures": sum(sample["registration_failures"] for sample in samples),
    }


def _run_case(
    subjects: list[_Subject],
    value: torch.Tensor,
    scenario: str,
    backend: _TimedRegistration,
    args: argparse.Namespace,
) -> dict[str, Any]:
    host_pin_manager.clear()
    before_gpu = _gpu_sample()
    torch.cuda.reset_peak_memory_stats(value.device)
    priming = (subjects[0], subjects[1], subjects[0]) if scenario == "alternating" else (subjects[0],)
    if scenario != "cold":
        for subject in priming:
            _session(subject, value, 1, backend)  # end with A most recently used
    samples = []
    clears = []
    for _ in range(args.repeats):
        if scenario == "cold":
            clears.append(_phase(host_pin_manager.clear, backend))
        order = (subjects[1], subjects[0]) if scenario == "alternating" else (subjects[0],)
        for subject in order:
            samples.append(_session(subject, value, args.forwards, backend))
    summary = _summarize(samples)
    return {
        "scenario": scenario,
        "budget_bytes": host_pin_manager.max_pinned_bytes,
        "summary": summary,
        "samples": samples,
        "cold_reset_phases": clears,
        "gpu_before": before_gpu,
        "gpu_after": _gpu_sample(),
    }


def _save(report: dict[str, Any], path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modes", nargs="+", choices=("streaming", "rolling"), default=["streaming", "rolling"])
    parser.add_argument("--representations", nargs="+", choices=("dense", "convrot-int8"), default=["dense"])
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--budget-fractions", type=float, nargs="+", default=[0, 0.5, 1, 2])
    parser.add_argument(
        "--scenarios", nargs="+", choices=("cold", "warm", "alternating"), default=["cold", "warm", "alternating"]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--forwards", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("blocks", "width", "tokens", "warmup", "repeats", "forwards"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if any(not math.isfinite(fraction) or fraction < 0 for fraction in args.budget_fractions):
        parser.error("--budget-fractions must be finite and non-negative")
    if "convrot-int8" in args.representations and args.width % 64:
        parser.error("ConvRot INT8 requires --width divisible by 64")
    return args


def _run_models(
    mode: BlockMode,
    representation: str,
    value: torch.Tensor,
    backend: _TimedRegistration,
    args: argparse.Namespace,
    report: dict[str, Any],
) -> None:
    print(f"Building two {representation} models for {mode}...", flush=True)
    subjects = [_build_subject(name, mode, representation, args) for name in ("A", "B")]
    try:
        print(
            f"Each model: {subjects[0].storage_bytes / GIB:.3f} GiB / {subjects[0].allocations} allocations",
            flush=True,
        )
        host_pin_manager.max_pinned_bytes = 0
        for subject in subjects:
            _warm_subject(subject, value, args.warmup)
        full_budget = max(subject.page_bytes for subject in subjects)
        print(
            "budget  scenario      pinned GiB    activate ms    native reg/unreg ms"
            "    calls reg/unreg    forward ms  deactivate ms"
        )
        for fraction in args.budget_fractions:
            host_pin_manager.clear()
            host_pin_manager.max_pinned_bytes = math.ceil(full_budget * fraction / mmap.PAGESIZE) * mmap.PAGESIZE
            for scenario in args.scenarios:
                result = _run_case(subjects, value, scenario, backend, args)
                result.update(
                    mode=mode,
                    representation=representation,
                    budget_fraction=fraction,
                    model_storage_bytes=subjects[0].storage_bytes,
                    model_page_bytes=full_budget,
                    model_allocations=subjects[0].allocations,
                )
                report["results"].append(result)
                _save(report, args.output)
                summary = result["summary"]
                print(
                    f"{fraction:5.2f}x  {scenario:12s} {summary['registered_gib']:9.3f}"
                    f" {summary['activation_ms']:14.2f}"
                    f" {summary['register_ms']:10.2f}/{summary['unregister_ms']:<9.2f}"
                    f" {summary['register_calls']:8.0f}/{summary['unregister_calls']:<8.0f}"
                    f" {summary['forward_ms']:12.2f} {summary['deactivation_ms']:14.2f}",
                    flush=True,
                )
    finally:
        for subject in subjects:
            subject.offloader.deactivate()
        host_pin_manager.clear()


def _main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA or HIP device")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    report: dict[str, Any] = {
        "started_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "config": {key: str(item) if isinstance(item, Path) else item for key, item in vars(args).items()},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "gpu": props.name,
            "gpu_free_bytes": free,
            "gpu_total_bytes": total,
            "gpu_before": _gpu_sample(),
        },
        "results": [],
    }
    print(f"{props.name}; PyTorch {torch.__version__}; {free / GIB:.1f} GiB GPU memory free", flush=True)
    print("Timings include completed CPU prefetch and GPU work in wall-clock milliseconds; compiler warmup excluded.")
    if report["environment"]["gpu_before"]:
        print(report["environment"]["gpu_before"], flush=True)
    torch.manual_seed(args.seed + 2)
    value = torch.randn(args.tokens, args.width, dtype=getattr(torch, args.dtype), device=device)
    with _instrument_registration() as backend:
        for representation in args.representations:
            for mode in args.modes:
                _run_models(mode, representation, value, backend, args, report)
                gc.collect()
                torch.cuda.empty_cache()
                torch.compiler.reset()
    report["finished_utc"] = datetime.now(UTC).isoformat()
    report["final_pin_stats"] = asdict(host_pin_manager.stats)
    _save(report, args.output)
    print("All outputs matched their zero-budget references; all registrations released.", flush=True)


if __name__ == "__main__":
    _main()
