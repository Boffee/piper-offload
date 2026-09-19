"""Measure two-model safetensors rotation on Linux with the existing loader.

Requires safetensors in the benchmark environment. Reports actual mapped
private memory separately from process RSS (which includes reclaimable pages).
Compiler and accelerator warmup use a zero pin budget and are not timed.
"""

# ruff: noqa: T201 - benchmark CLI emits a JSON report.

import argparse
import json
import mmap
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from piper_offload import BlockCompileConfig, ModelOffloader, host_pin_manager


class _Model(nn.Module):
    def __init__(self, width: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(blocks))
        self.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = torch.relu(block(value))
        return value


def _memory(paths: list[Path]) -> dict[str, object]:
    files = {str(path): {"Rss": 0, "Anonymous": 0, "Locked": 0} for path in paths}
    current = None
    with open("/proc/self/smaps", encoding="utf-8") as smaps:
        for line in smaps:
            fields = line.split()
            if "-" in fields[0]:
                current = files.get(fields[-1])
            elif current is not None and fields[0].rstrip(":") in current:
                current[fields[0].rstrip(":")] += int(fields[1]) * 1024
    with open("/proc/self/status", encoding="utf-8") as status:
        rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
    return {
        "process_rss_bytes": rss,
        "mappings": {Path(name).name: values for name, values in files.items()},
        "pins": asdict(host_pin_manager.stats),
    }


def _run_mode(mode: str, args: argparse.Namespace, directory: Path) -> list[dict[str, object]]:
    paths = [directory / f"{mode}-{index}.safetensors" for index in range(2)]
    subjects = []
    torch.manual_seed(117)
    value = torch.randn(2, args.width, device="cuda")
    for path in paths:
        model = _Model(args.width, args.blocks)
        save_file(model.state_dict(), str(path))
        with safe_open(path, framework="pt") as reader:
            names = reader.keys()
            model.load_state_dict({name: reader.get_tensor(name) for name in names}, assign=True)
        offloader = ModelOffloader.from_module(
            model,
            block_paths=[] if mode == "host" else ["blocks"],
            block_mode="streaming" if mode == "host" else mode,
            block_compile=BlockCompileConfig(fullgraph=True) if mode == "rolling" else None,
        )
        subjects.append(offloader)

    samples = []
    references = []
    try:
        host_pin_manager.max_pinned_bytes = 0
        for offloader in subjects:
            try:
                offloader.activate("cuda")
                with torch.inference_mode():
                    references.append(offloader.value(value).cpu())
            finally:
                offloader.deactivate()
        host_pin_manager.max_pinned_bytes = args.budget_mib * 1024**2
        for _ in range(args.repeats):
            for index, offloader in enumerate(subjects):
                before = _memory(paths)
                started = time.perf_counter()
                try:
                    offloader.activate("cuda")
                    torch.cuda.synchronize()
                    activation_ms = (time.perf_counter() - started) * 1000
                    active = _memory(paths)
                    with torch.inference_mode():
                        actual = offloader.value(value).cpu()
                finally:
                    started = time.perf_counter()
                    offloader.deactivate()
                    deactivation_ms = (time.perf_counter() - started) * 1000
                released = _memory(paths)
                torch.testing.assert_close(actual, references[index], rtol=0, atol=0)
                if args.check_release:
                    assert host_pin_manager.stats.pinned_bytes == 0
                    # Each safetensors tensor can leave two boundary pages.
                    for memory in released["mappings"].values():
                        assert memory["Anonymous"] <= 2 * args.blocks * mmap.PAGESIZE
                samples.append({
                    "mode": mode, "model": index,
                    "activation_ms": activation_ms, "deactivation_ms": deactivation_ms,
                    "before": before, "active": active, "released": released,
                })
    finally:
        for offloader in subjects:
            offloader.deactivate()
        host_pin_manager.clear()
        if mode == "rolling":
            torch.compiler.reset()
    return samples


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-mib", type=int, default=1024)
    parser.add_argument("--modes", nargs="+", choices=["streaming", "rolling", "resident", "host"],
                        default=["streaming", "resident", "host"])
    parser.add_argument("--check-release", action="store_true", help="assert the #117 release contract")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.width, args.blocks, args.repeats) <= 0 or args.budget_mib < 0:
        parser.error("dimensions/repeats must be positive and budget must be non-negative")
    if sys.platform != "linux" or not torch.cuda.is_available():
        parser.error("a Linux CUDA/HIP machine is required")
    original_budget = host_pin_manager.max_pinned_bytes
    if host_pin_manager.stats.registrations or host_pin_manager.stats.active_leases:
        raise RuntimeError("run this benchmark in a fresh process")
    report = {
        "platform": platform.platform(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "model_bytes": args.blocks * args.width**2 * 4,
        "budget_bytes": args.budget_mib * 1024**2,
        "samples": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="piper-mapped-rotation-") as directory:
            for mode in args.modes:
                report["samples"].extend(_run_mode(mode, args, Path(directory)))
    finally:
        host_pin_manager.max_pinned_bytes = original_budget
    output = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    _main()
