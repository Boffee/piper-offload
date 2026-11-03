"""Time piper's checkpoint read against the safetensors loaders of vLLM and SGLang.

Every loader reads the same files, from a cold or a warm page cache, until
every tensor byte is on the device: copied into one reused GPU buffer as large
as the largest tensor, so no loader needs VRAM for the whole checkpoint. With
``--device cpu`` the trial ends when every byte has been copied once on the
host: into piper's new copies, and for the rest into the same reused buffer.
That favours the rest twice over. The buffer's pages are already mapped,
where piper's copies fault in every page, and torch copies into it on all its
threads, where a GPU load's pageable copy goes through the CUDA driver's
staging buffer. Use it to compare cold reads; compare loads on a GPU. A copy
is needed because a warm mapping is already resident, so touching it would
time only the page-table updates. Only the loaders' read paths run. Model
code, sharding, fusion and tensor adapters are left out, so every time is a
lower bound on a real model load.

Loaders:

- ``safetensors``: ``safe_open`` and ``Tensor.copy_``, the reference.
- ``piper``: ``MappedCheckpoint`` and a pinning ``PinManager`` acquisition,
  whose owned copies fill with parallel positional reads, then pinned
  transfers under the lease.
- ``vllm``: ``safetensors_weights_iterator`` with the default strategy, which
  prefetches only on NFS or Lustre. ``vllm-prefetch`` forces the prefetch.
  ``vllm-multithread`` is ``multi_thread_safetensors_weights_iterator`` with
  the default loader's 8 threads. ``vllm-fastsafetensors`` reads through
  fastsafetensors, with GPU Direct Storage on a single GPU, and needs CUDA.
- ``sglang``: ``buffered_multi_thread_safetensors_weights_iterator`` with the
  default loader's 8 threads. ``sglang-prefetch`` is the single-threaded
  iterator with its 4-thread page-cache prefetch.

Each trial runs in a fresh process, so the loader meets the page-cache state
the parent set (``cold``: evicted, ``warm``: read through once) and nothing
pinned or mapped survives between trials. The kernel's read counter shows
what each trial actually read from disk. vLLM and SGLang pin their own
PyTorch, so ``--python ENGINE=PATH`` runs an engine's loaders under another
interpreter; a trial imports only what its loader needs.

After the load, the same process loads again. ``reload`` is what activating
the model a second time costs: reacquiring the still-registered copies for
piper, and rereading the now-warm page cache for the rest.

Without CUDA, piper's copies go to a backend that registers nothing, so its
``--device cpu`` time leaves out ``cudaHostRegister``.
"""

# ruff: noqa: T201 - benchmark CLI prints its report.

import argparse
import json
import statistics
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import _process_memory as process_memory
import torch

LOADERS = (
    "safetensors", "piper",
    "vllm", "vllm-prefetch", "vllm-multithread", "vllm-fastsafetensors",
    "sglang", "sglang-prefetch",
)
# Loaders that deliver straight to the GPU.
GPU_ONLY = frozenset({"vllm-fastsafetensors"})
# Engine loaders' DefaultModelLoader.DEFAULT_NUM_THREADS.
ENGINE_THREADS = 8
RESULT = "RESULT "

# No ``type`` statement or other 3.14 syntax: trials run under the engines' older interpreters.
Weights = Callable[[list[str]], Iterator[tuple[str, torch.Tensor]]]


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1).view(torch.uint8)


class _Sink:
    """One buffer, as large as the largest tensor, that every tensor is copied into in turn."""

    def __init__(self, device: torch.device, largest: int) -> None:
        self.device = device
        # Zeroed, so a host buffer's pages fault in before the timing rather than during the first copies.
        self.buffer = torch.zeros(largest, dtype=torch.uint8, device=device)

    def take(self, tensor: torch.Tensor) -> None:
        if tensor.device.type == "cpu" and tensor.numel():
            data = _bytes(tensor)
            self.buffer[: data.numel()].copy_(data)

    def finish(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class _Unregistered:
    """Accepts every range without registering it: a refused registration would stop piper's fill."""

    def register(self, pointer: int, size: int) -> bool:  # noqa: ARG002 - backend protocol
        return True

    def unregister(self, pointer: int) -> None:
        pass


def _safe_open_weights(files: list[str]) -> Iterator[tuple[str, torch.Tensor]]:
    from safetensors import safe_open  # noqa: PLC0415 - only the reference loader needs it

    for path in files:
        with safe_open(path, framework="pt") as reader:
            for name in reader.keys():  # noqa: SIM118 - safe_open is not a mapping
                yield name, reader.get_tensor(name)


def _weights(loader: str) -> Weights:
    """The named engine loader's weight iterator, imported now so the import stays out of the timings."""
    if loader == "safetensors":
        return _safe_open_weights
    if loader.startswith("vllm"):
        from vllm.model_executor.model_loader import weight_utils as vllm  # noqa: PLC0415 - vLLM's environment

        return {
            "vllm": lambda files: vllm.safetensors_weights_iterator(files, False),
            "vllm-prefetch": lambda files: vllm.safetensors_weights_iterator(
                files, False, safetensors_load_strategy="prefetch",
            ),
            "vllm-multithread": lambda files: vllm.multi_thread_safetensors_weights_iterator(
                files, False, max_workers=ENGINE_THREADS,
            ),
            "vllm-fastsafetensors": lambda files: vllm.fastsafetensors_weights_iterator(files, False),
        }[loader]
    from sglang.srt.model_loader import weight_utils as sglang  # noqa: PLC0415 - SGLang's environment

    return {
        "sglang": lambda files: sglang.buffered_multi_thread_safetensors_weights_iterator(
            files, max_workers=ENGINE_THREADS,
        ),
        "sglang-prefetch": lambda files: sglang.safetensors_weights_iterator(files, prefetch=True),
    }[loader]


def _engine(loader: str, files: list[str], sink: _Sink) -> Callable[[], dict[str, int]]:
    weights = _weights(loader)

    def load() -> dict[str, int]:
        for _name, tensor in weights(files):
            sink.take(tensor)
        sink.finish()
        return {}

    return load


def _piper(files: list[str], sink: _Sink) -> Callable[[], dict[str, int]]:
    from piper_offload import MappedCheckpoint, PinManager  # noqa: PLC0415 - piper's environment

    tensors: list[torch.Tensor] = []
    for path in files:
        reader = MappedCheckpoint(path)
        tensors += [
            tensor
            for name in reader.keys()  # noqa: SIM118 - the reader is not a mapping
            if (tensor := reader.get_tensor(name)).numel()
        ]
    manager = PinManager(backend=None if torch.cuda.is_available() else _Unregistered())

    def load() -> dict[str, int]:
        with manager.acquire(tensors) as lease:
            # On the CPU the filled copies are the result; copying them again would time a second read.
            if sink.device.type == "cuda":
                for tensor in tensors:
                    data = _bytes(tensor)
                    manager.transfer(sink.buffer[: data.numel()], data, non_blocking=True)
            # The lease must outlive the asynchronous transfers.
            sink.finish()
        return {"pageable_bytes": lease.pageable_bytes}

    return load


def _trial(loader: str, files: list[str], device: torch.device, largest: int) -> dict[str, float | int]:
    if device.type == "cuda":
        torch.empty(1, device=device)  # context creation stays out of the timings
    sink = _Sink(device, largest)
    load = _piper(files, sink) if loader == "piper" else _engine(loader, files, sink)
    before = process_memory.read_bytes()
    started = time.perf_counter()
    extra = load()
    loaded = time.perf_counter()
    read = process_memory.read_bytes() - before
    load()
    return {"load_s": loaded - started, "read_bytes": read, "reload_s": time.perf_counter() - loaded, **extra}


def _sizes(files: list[Path]) -> tuple[int, int]:
    """Tensor bytes in all files, and in the largest tensor, from the safetensors headers."""
    total = largest = 0
    for path in files:
        with open(path, "rb") as file:
            (length,) = struct.unpack("<Q", file.read(8))
            header = json.loads(file.read(length))
        header.pop("__metadata__", None)
        for entry in header.values():
            start, stop = entry["data_offsets"]
            total += stop - start
            largest = max(largest, stop - start)
    return total, largest


def _spawn(
    python: str, loader: str, files: list[Path], device: str, largest: int,
) -> dict[str, float | int] | None:
    """Run one trial in a fresh interpreter; its logs pass through and its result comes back on stdout."""
    command = [
        python, str(Path(__file__).resolve()), *map(str, files),
        "--trial", loader, "--device", device, "--largest", str(largest),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
    results = [line for line in completed.stdout.splitlines() if line.startswith(RESULT)]
    if completed.returncode or not results:
        print(completed.stdout[-4000:], end="")
        return None
    return json.loads(results[-1].removeprefix(RESULT))


def _run(args: argparse.Namespace, files: list[Path]) -> list[dict[str, float | int | str]]:
    total, largest = _sizes(files)
    print(f"{len(files)} file(s), {total / 1e9:.2f} GB of tensors, largest {largest / 2**20:.0f} MiB, {args.device}")
    loaders = [loader for loader in args.loaders if args.device != "cpu" or loader not in GPU_ONLY]
    if skipped := sorted(set(args.loaders) - set(loaders)):
        print(f"skipping {', '.join(skipped)}: they need CUDA")
    pythons = dict(args.python)
    results: list[dict[str, float | int | str]] = []
    for repeat in range(args.repeats):
        # Rotated each repeat, so a drift in the drive's speed does not always land on the same loaders.
        turn = repeat % len(loaders)
        for cache in args.cache:
            for loader in loaders[turn:] + loaders[:turn]:
                for path in files:
                    (process_memory.drop_from_cache if cache == "cold" else process_memory.fill_cache)(path)
                python = pythons.get(loader.partition("-")[0], sys.executable)
                result = _spawn(python, loader, files, args.device, largest)
                if result is None:
                    print(f"{loader:<22} {cache:<5} failed")
                    continue
                load_s, reload_s = float(result["load_s"]), float(result["reload_s"])
                pageable = int(result.get("pageable_bytes", 0))
                print(
                    f"{loader:<22} {cache:<5} load {load_s:7.2f} s {total / load_s / 1e9:6.2f} GB/s  "
                    f"disk {int(result['read_bytes']) / 1e9:6.2f} GB  "
                    f"reload {reload_s:7.3f} s {total / reload_s / 1e9:7.2f} GB/s"
                    + (f"  pageable {pageable / 1e9:.2f} GB" if pageable else ""),
                    flush=True,
                )
                results.append({"loader": loader, "cache": cache, "repeat": repeat, "bytes": total, **result})
    if args.repeats > 1:
        _summarize(results, loaders, args.cache, total)
    return results


def _summarize(
    results: list[dict[str, float | int | str]], loaders: list[str], caches: list[str], total: int,
) -> None:
    """Each loader's median load rate and range, and its median reload: single cold trials vary with the drive."""
    print("\nmedian load (range), median reload")
    for cache in caches:
        for loader in loaders:
            trials = [result for result in results if result["loader"] == loader and result["cache"] == cache]
            if not trials:
                continue
            rates = sorted(total / float(trial["load_s"]) / 1e9 for trial in trials)
            reload_s = statistics.median(float(trial["reload_s"]) for trial in trials)
            print(
                f"{loader:<22} {cache:<5} {statistics.median(rates):6.2f} GB/s ({rates[0]:.2f}-{rates[-1]:.2f})  "
                f"reload {reload_s * 1e3:8.1f} ms",
            )


def _engine_python(value: str) -> tuple[str, str]:
    engine, _, python = value.partition("=")
    if not python:
        raise argparse.ArgumentTypeError(f"expected ENGINE=PATH, got {value!r}")
    return engine, python


def _main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", type=Path, help="safetensors files, or directories of them")
    parser.add_argument("--loaders", nargs="+", choices=LOADERS, default=list(LOADERS))
    parser.add_argument("--cache", nargs="+", choices=("cold", "warm"), default=["cold", "warm"])
    parser.add_argument("--device", default="cuda:0", help="a CUDA device, or cpu to stop in host memory")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--python", type=_engine_python, action="append", default=[], metavar="ENGINE=PATH",
        help="interpreter for an engine's loaders (piper, safetensors, vllm, sglang); default: this one",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trial", choices=LOADERS, help=argparse.SUPPRESS)
    parser.add_argument("--largest", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    files = sorted(
        file for path in args.checkpoints for file in (path.glob("*.safetensors") if path.is_dir() else [path])
    )
    if args.trial is not None:
        result = _trial(args.trial, [str(file) for file in files], torch.device(args.device), args.largest)
        print(RESULT + json.dumps(result), flush=True)
        return
    results = _run(args, files)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    _main()
