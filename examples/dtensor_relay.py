"""Run two-rank DTensor initialization, all-gather and SUM all-reduce.

    python examples/dtensor_relay.py --device cpu
    python examples/dtensor_relay.py --shared-gpu --dtype bfloat16
    python examples/dtensor_relay.py

The default uses two physical GPUs. --shared-gpu maps both worker processes
to cuda:0 for correctness testing; it does not reduce single-GPU VRAM use.
"""

import argparse
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

from piper_offload.communication import register_relay_backend


def _worker(rank: int, store_path: str, device_type: str, shared_gpu: bool, dtype_name: str) -> None:
    if device_type == "cuda":
        torch.cuda.set_device(0 if shared_gpu else rank)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay",
        store=dist.FileStore(store_path, 2),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        # Construct from the existing group so tests may assign two logical
        # ranks to one GPU. Real deployments normally assign one GPU per rank.
        mesh = DeviceMesh.from_group(dist.group.WORLD, device_type)
        dtype = getattr(torch, dtype_name)
        full = torch.arange(7, dtype=dtype, device=device_type)
        for placement in (Replicate(), Shard(0)):
            source = full.clone() if rank == 0 else torch.full_like(full, -1)
            distributed = distribute_tensor(source, mesh, [placement])
            torch.testing.assert_close(distributed.full_tensor(), full)
        local = torch.full((4,), float(rank + 1), device=device_type, dtype=dtype)
        partial = DTensor.from_local(local, mesh, [Partial()], run_check=False)
        result = partial.redistribute(placements=[Replicate()]).to_local()
        torch.testing.assert_close(result, torch.full_like(result, 3))
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--shared-gpu", action="store_true", help="test both ranks on cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()
    if args.shared_gpu and args.device != "cuda":
        parser.error("--shared-gpu requires --device cuda")
    if args.device == "cuda":
        required = 1 if args.shared_gpu else 2
        if torch.cuda.device_count() < required:
            parser.error(f"requires {required} GPUs; use --shared-gpu or --device cpu for local testing")
    with TemporaryDirectory(prefix="piper-relay-") as directory:
        store_path = str(Path(directory) / "store")
        mp.spawn(_worker, args=(store_path, args.device, args.shared_gpu, args.dtype), nprocs=2)
    print("Both ranks passed DTensor initialization, all-gather, and SUM all-reduce.")  # noqa: T201


if __name__ == "__main__":
    main()
