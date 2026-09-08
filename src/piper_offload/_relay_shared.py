"""Same-machine relay: GPU -> shared host slot -> peer GPU.

Each rank owns two outgoing slots in one shared mapping. Slots are reused
across peer rounds, bounding storage independently of the number of pairs.
All-gather exchanges peers in cyclic rounds; broadcast and scatter visit each
receiver in turn. Gloo carries 8-byte ready/free signals per chunk and a 24-byte
metadata all-gather per copy collective. No CPU polling,
GPU IPC, cross-process CUDA events, or payload socket transfers are needed.
"""

import json
import mmap
import os
import socket
import tempfile
import uuid
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .pin_manager import PinManager

COPY_BUFFERS = 2


def shared_slot_bytes(nbytes: int, world_size: int) -> int:
    """Keep large DMA slots 4 KiB aligned while allowing tiny staging budgets."""
    capacity = nbytes // (world_size * COPY_BUFFERS)
    alignment = 4096 if capacity >= 4096 else 16
    return capacity // alignment * alignment


def _create_mapping(size: int) -> tuple[mmap.mmap, str]:
    if os.name == "nt":
        name = f"piper-relay-{uuid.uuid4().hex}"
        return mmap.mmap(-1, size, tagname=name), name
    directory = "/dev/shm" if Path("/dev/shm").is_dir() else None
    fd, name = tempfile.mkstemp(prefix="piper-relay-", dir=directory)
    try:
        # Reserve tmpfs capacity before mapping; ftruncate alone can defer an
        # out-of-space error until a fatal SIGBUS during a GPU/CPU transfer.
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(fd, 0, size)
        else:
            os.ftruncate(fd, size)
        return mmap.mmap(fd, size), name
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    finally:
        os.close(fd)


def _open_mapping(name: str, size: int) -> mmap.mmap:
    if os.name == "nt":
        return mmap.mmap(-1, size, tagname=name)
    with open(name, "r+b") as file:
        return mmap.mmap(file.fileno(), size)


def create_shared_buffer(
    store: dist.Store, rank: int, world_size: int, nbytes: int, timeout: timedelta,
) -> torch.Tensor:
    """Attach all ranks, then unlink POSIX names while mappings remain alive.

    frombuffer retains the mmap object through the tensor's storage. Never
    explicitly close it: outstanding pin-manager storage references must also
    keep the mapping alive if native unregistration fails.
    """
    store = dist.PrefixStore("piper_relay_shared", store)
    mapping = None
    name = ""
    if rank == 0:
        try:
            mapping, name = _create_mapping(nbytes)
            descriptor = {"name": name, "host": socket.gethostname(), "error": ""}
        except Exception as error:
            descriptor = {"error": str(error)}
        store.set("mapping", json.dumps(descriptor))
    try:
        descriptor = json.loads(store.get("mapping"))
        error_message = descriptor["error"]
        if not error_message:
            try:
                if descriptor["host"] != socket.gethostname():
                    raise RuntimeError("shared relay requires all ranks on the same machine")
                if rank != 0:
                    mapping = _open_mapping(descriptor["name"], nbytes)
            except Exception as error:
                error_message = str(error)
        store.set(str(rank), error_message)
        keys = [str(r) for r in range(world_size)]
        store.wait(keys, timeout)
        errors = [store.get(key).decode() for key in keys]
        if any(errors):
            raise RuntimeError(f"Could not initialize shared relay: {next(e for e in errors if e)}")
        assert mapping is not None
        return torch.frombuffer(mapping, dtype=torch.uint8)
    finally:
        if rank == 0 and name and os.name != "nt":
            Path(name).unlink(missing_ok=True)


class SharedRelay:
    buffer_count = COPY_BUFFERS

    def __init__(
        self, store: dist.Store, group: dist.ProcessGroupGloo, rank: int, world_size: int,
        nbytes: int, timeout: timedelta,
    ) -> None:
        self.buffer = create_shared_buffer(store, rank, world_size, nbytes, timeout)
        self.rank = rank
        self.world_size = world_size
        self.capacity = shared_slot_bytes(nbytes, world_size)
        self.group = group
        self.timeout = timeout
        self.broken = False
        self.device: torch.device | None = None
        self.download: torch.cuda.Stream | None = None
        self.upload: torch.cuda.Stream | None = None
        self.ready: list[torch.cuda.Event] = []
        self.uploaded: list[torch.cuda.Event] = []
        self._send_signal = torch.empty(1, dtype=torch.int64)
        self._recv_signal = torch.empty(1, dtype=torch.int64)
        self._pending: list[dist.Work] = []

    def local_buffer(self) -> torch.Tensor:
        start = self.rank * self.buffer_count * self.capacity
        return self.buffer[start:start + self.buffer_count * self.capacity]

    def _slot(self, rank: int, lane: int, width: int) -> torch.Tensor:
        start = (rank * self.buffer_count + lane) * self.capacity
        return self.buffer[start:start + width]

    def _prepare_streams(self, device: torch.device) -> None:
        if self.device != device:
            self.device = device
            self.download = torch.cuda.Stream(device=device)
            self.upload = torch.cuda.Stream(device=device)
            self.ready = [torch.cuda.Event() for _ in range(self.buffer_count)]
            self.uploaded = [torch.cuda.Event() for _ in range(self.buffer_count)]
        assert self.download is not None
        assert self.upload is not None
        current = torch.cuda.current_stream(device)
        self.download.wait_stream(current)
        self.upload.wait_stream(current)

    def _signal(self, send: int | None, receive: int | None, tag: int, sequence: int) -> None:
        # Post sends before waiting for receives, including cyclic peer rounds.
        # Completion precedes reusing these persistent control tensors.
        self._send_signal[0] = sequence
        if send is not None:
            self._pending.append(self.group.send([self._send_signal], send, tag))
        if receive is not None:
            self._pending.append(self.group.recv([self._recv_signal], receive, tag))
        for work in reversed(self._pending):
            work.wait(self.timeout)
        self._pending.clear()
        if receive is not None and self._recv_signal.item() != sequence:
            raise RuntimeError("shared relay control sequence mismatch")

    def _exchange(
        self, source: torch.Tensor | None, destination: torch.Tensor | None,
        send: int | None, receive: int | None, asynchronous: bool,
    ) -> None:
        if source is None and destination is None:
            return
        tensor = source if source is not None else destination
        assert tensor is not None
        length = tensor.numel()
        count = max(1, (length + self.capacity - 1) // self.capacity)

        def download(index: int) -> None:
            if source is None:
                return
            start = index * self.capacity
            width = min(self.capacity, length - start)
            with torch.cuda.stream(self.download) if asynchronous else nullcontext():
                self._slot(self.rank, index % self.buffer_count, width).copy_(
                    source[start:start + width], non_blocking=asynchronous,
                )
                if asynchronous:
                    self.ready[index % self.buffer_count].record(self.download)

        def release(index: int) -> None:
            if destination is not None and asynchronous:
                self.uploaded[index % self.buffer_count].synchronize()
            # The receiver acknowledges only after its GPU stops reading the
            # shared slot. This is the producer's permission to overwrite it.
            self._signal(receive, send, 1, index)

        for index in range(min(self.buffer_count, count)):
            download(index)
        for index in range(count):
            if index >= self.buffer_count:
                release(index - self.buffer_count)
                download(index)
            if source is not None and asynchronous:
                self.ready[index % self.buffer_count].synchronize()
            self._signal(send, receive, 0, index)
            if destination is not None:
                assert receive is not None
                start = index * self.capacity
                width = min(self.capacity, length - start)
                with torch.cuda.stream(self.upload) if asynchronous else nullcontext():
                    destination[start:start + width].copy_(
                        self._slot(receive, index % self.buffer_count, width), non_blocking=asynchronous,
                    )
                    if asynchronous:
                        self.uploaded[index % self.buffer_count].record(self.upload)
        for index in range(max(0, count - self.buffer_count), count):
            release(index)

    def copy(
        self, operation: str, sources: list[torch.Tensor], outputs: list[torch.Tensor],
        root: int, manager: PinManager,
    ) -> None:
        if self.broken:
            raise RuntimeError("shared relay failed previously; destroy and recreate the process group")
        device = outputs[0].device
        # Check rank agreement before any shared-slot writes. This also keeps
        # each collective separate from earlier peer rounds.
        metadata = torch.tensor([
            ("broadcast", "scatter", "allgather").index(operation), outputs[0].nbytes, root,
        ], dtype=torch.int64)
        gathered = [torch.empty_like(metadata) for _ in range(self.world_size)]
        self.group.allgather([gathered], [metadata]).wait(self.timeout)
        if any(not torch.equal(metadata, other) for other in gathered):
            raise ValueError("shared relay collective operation, payload bytes and root must match across ranks")
        sources = [t.reshape(-1).view(torch.uint8) for t in sources]
        outputs = [t.reshape(-1).view(torch.uint8) for t in outputs]
        with manager.acquire([self.buffer]) if device.type == "cuda" else nullcontext() as lease:
            asynchronous = lease is not None and lease.pageable_bytes == 0
            try:
                if asynchronous:
                    self._prepare_streams(device)
                self._rounds(operation, sources, outputs, root, asynchronous)
            except BaseException:
                # A failed exchange can leave unread ready/free messages. Do
                # not reuse its slots or control tensors in a later operation.
                self.broken = True
                raise
            finally:
                # Pin leases and source/destination storage must outlive DMA,
                # even if a peer fails after a ready message was published.
                if asynchronous:
                    assert self.download is not None
                    assert self.upload is not None
                    try:
                        self.download.synchronize()
                    finally:
                        self.upload.synchronize()
                if device.type == "cuda":
                    # Includes the local GPU contribution, which is copied on
                    # the caller's stream instead of passing through the relay.
                    torch.cuda.current_stream(device).synchronize()

    def _rounds(
        self, operation: str, sources: list[torch.Tensor], outputs: list[torch.Tensor],
        root: int, asynchronous: bool,
    ) -> None:
        if operation == "allgather":
            self._local_copy(outputs[self.rank], sources[0])
            for step in range(1, self.world_size):
                send, receive = (self.rank + step) % self.world_size, (self.rank - step) % self.world_size
                self._exchange(sources[0], outputs[receive], send, receive, asynchronous)
        else:
            if self.rank == root and operation == "scatter":
                self._local_copy(outputs[0], sources[root])
            for peer in range(self.world_size):
                if peer == root:
                    continue
                if self.rank == root:
                    source = sources[peer] if operation == "scatter" else sources[0]
                    self._exchange(source, None, peer, None, asynchronous)
                elif self.rank == peer:
                    self._exchange(None, outputs[0], None, root, asynchronous)

    @staticmethod
    def _local_copy(destination: torch.Tensor, source: torch.Tensor) -> None:
        # The local contribution never needs a trip through host memory.
        if destination.data_ptr() != source.data_ptr():
            destination.copy_(source)
