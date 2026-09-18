"""Same-machine relay: GPU -> shared host slot -> peer GPU.

Each rank owns two outgoing slots in one shared mapping. Slots are reused
across peer rounds, bounding storage independently of the number of pairs.
All-gather exchanges peers in cyclic rounds; broadcast and scatter visit each
receiver in turn. SUM publishes each original chunk once for all peers, then
accumulates in rank order on the GPU. Gloo carries 8-byte ready/free signals
per chunk and a 32-byte metadata all-gather per collective. No CPU polling,
GPU IPC, cross-process CUDA events, or payload socket transfers are needed.
"""

import json
import mmap
import os
import socket
import tempfile
import uuid
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from ._host_backing import HostBacking
from .host_memory import HostMemoryManager

BUFFERS_PER_RANK = 2
REDUCTION_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
COPY_DTYPES = (
    *REDUCTION_DTYPES,
    torch.float64,
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.complex64,
    torch.complex128,
)


def shared_slot_bytes(nbytes: int, world_size: int) -> int:
    """Keep large DMA slots 4 KiB aligned while allowing tiny staging budgets."""
    capacity = nbytes // (world_size * BUFFERS_PER_RANK)
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
    buffer_count = BUFFERS_PER_RANK

    def __init__(
        self, store: dist.Store, group: dist.ProcessGroupGloo, rank: int, world_size: int,
        nbytes: int, timeout: timedelta,
    ) -> None:
        self.buffer = create_shared_buffer(store, rank, world_size, nbytes, timeout)
        self._backings: tuple[HostBacking, ...] | None = None
        self.rank = rank
        self.world_size = world_size
        self._peers = tuple(
            ((rank + step) % world_size, (rank - step) % world_size)
            for step in range(1, world_size)
        )
        self.capacity = shared_slot_bytes(nbytes, world_size)
        self.group = group
        self.timeout = timeout
        self.broken = False
        self.device: torch.device | None = None
        self.download: torch.cuda.Stream | None = None
        self.upload: torch.cuda.Stream | None = None
        self.ready: list[torch.cuda.Event] = []
        self.uploaded: list[torch.cuda.Event] = []
        self._reduction_buffer: torch.Tensor | None = None
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
        # PyTorch's Gloo stubs omit its send/recv and collective bindings.
        self._send_signal[0] = sequence
        if send is not None:
            self._pending.append(self.group.send([self._send_signal], send, tag))  # type: ignore[attr-defined]
        if receive is not None:
            self._pending.append(self.group.recv([self._recv_signal], receive, tag))  # type: ignore[attr-defined]
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
        with closing(self._chunks(source, tensor.numel(), [(send, receive)], asynchronous)) as chunks:
            for index, start, width in chunks:
                if destination is not None:
                    assert receive is not None
                    destination[start:start + width].copy_(
                        self._slot(receive, index % self.buffer_count, width), non_blocking=asynchronous,
                    )

    def _chunks(
        self, source: torch.Tensor | None, length: int,
        peers: Sequence[tuple[int | None, int | None]], asynchronous: bool,
    ) -> Generator[tuple[int, int, int]]:
        """Yield published chunks on the upload stream; acknowledge every reader."""
        count = max(1, (length + self.capacity - 1) // self.capacity)
        receiving = any(receive is not None for _, receive in peers)

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
            if receiving and asynchronous:
                self.uploaded[index % self.buffer_count].synchronize()
            # The receiver acknowledges only after its GPU stops reading the
            # shared slot. This is the producer's permission to overwrite it.
            for send, receive in peers:
                self._signal(receive, send, 1, index)

        for index in range(min(self.buffer_count, count)):
            download(index)
        for index in range(count):
            if index >= self.buffer_count:
                release(index - self.buffer_count)
                download(index)
            if source is not None and asynchronous:
                self.ready[index % self.buffer_count].synchronize()
            for send, receive in peers:
                self._signal(send, receive, 0, index)
            start = index * self.capacity
            width = min(self.capacity, length - start)
            with torch.cuda.stream(self.upload) if asynchronous else nullcontext():
                yield index, start, width
                if receiving and asynchronous:
                    self.uploaded[index % self.buffer_count].record(self.upload)
        for index in range(max(0, count - self.buffer_count), count):
            release(index)

    def backings(self, manager: HostMemoryManager) -> tuple[HostBacking, ...]:
        """The shared buffer's handle, retained so idle pins survive between collectives."""
        if self._backings is None:
            # A MAP_SHARED mapping (tmpfs on Linux, anonymous on Windows) has
            # no copy-on-write, so a writable pin in place is safe; peers
            # write into it, so it must never be copied.
            self._backings = tuple(manager.capture([self.buffer], pin_in_place=True).values())
        return self._backings

    def copy(
        self, operation: str, sources: list[torch.Tensor], outputs: list[torch.Tensor],
        root: int, manager: HostMemoryManager,
    ) -> None:
        with self._collective(operation, outputs[0], root, manager) as asynchronous:
            sources = [t.reshape(-1).view(torch.uint8) for t in sources]
            outputs = [t.reshape(-1).view(torch.uint8) for t in outputs]
            self._rounds(operation, sources, outputs, root, asynchronous)

    def reduce(self, tensor: torch.Tensor, manager: HostMemoryManager) -> None:
        with self._collective("allreduce", tensor, -1, manager) as asynchronous:
            if self.world_size == 1 or tensor.numel() == 0:
                return
            if self._reduction_buffer is None or self._reduction_buffer.device != tensor.device:
                # One native-dtype receive chunk and one FP32 accumulator.
                # Both are reused on the upload stream, including across calls.
                self._reduction_buffer = torch.empty(3 * self.capacity, dtype=torch.uint8, device=tensor.device)
            buffer = self._reduction_buffer
            assert buffer is not None
            source = tensor.reshape(-1).view(torch.uint8)
            with closing(self._chunks(source, source.numel(), self._peers, asynchronous)) as chunks:
                for index, start, width in chunks:
                    output = source[start:start + width].view(tensor.dtype)
                    incoming = buffer[:width].view(tensor.dtype)
                    accumulator = buffer[
                        self.capacity:self.capacity + output.numel() * 4
                    ].view(torch.float32)
                    # All peers have published the original chunk. Accumulate
                    # in the same order everywhere, casting back only once.
                    for peer in range(self.world_size):
                        value = output
                        if peer != self.rank:
                            incoming.copy_(
                                self._slot(peer, index % self.buffer_count, width).view(tensor.dtype),
                                non_blocking=asynchronous,
                            )
                            value = incoming
                        if peer == 0:
                            accumulator.copy_(value)
                        else:
                            accumulator.add_(value)
                    output.copy_(accumulator)

    @contextmanager
    def _collective(
        self, operation: str, tensor: torch.Tensor, detail: int, manager: HostMemoryManager,
    ) -> Generator[bool]:
        if self.broken:
            raise RuntimeError("shared relay failed previously; destroy and recreate the process group")
        device = tensor.device
        # Check rank agreement before any shared-slot writes. This also keeps
        # each collective separate from earlier peer rounds.
        metadata = torch.tensor([
            ("broadcast", "scatter", "allgather", "allreduce").index(operation),
            tensor.nbytes,
            COPY_DTYPES.index(tensor.dtype),
            detail,
        ], dtype=torch.int64)
        gathered = [torch.empty_like(metadata) for _ in range(self.world_size)]
        self.group.allgather([gathered], [metadata]).wait(self.timeout)  # type: ignore[attr-defined]
        if any(not torch.equal(metadata, other) for other in gathered):
            raise ValueError("shared relay collective operation, payload bytes and root/dtype must match across ranks")
        with manager.acquire(self.backings(manager)) if device.type == "cuda" else nullcontext() as lease:
            asynchronous = lease is not None and lease.pinned
            try:
                if asynchronous:
                    self._prepare_streams(device)
                yield asynchronous
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
            for send, receive in self._peers:
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
