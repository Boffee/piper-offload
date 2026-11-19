"""Same-machine relay: GPU -> shared host slot -> peer GPU.

Each rank owns three outgoing slots in one shared mapping. Slots are reused
across peer rounds, bounding storage independently of the number of pairs.
All-gather and broadcast publish each chunk once for all peers; scatter visits
each receiver in turn. SUM publishes each original chunk once for all peers,
then accumulates in rank order on the GPU. Coalesced all-gather and SUM pack
their tensor lists into one continuous chunked transfer. Gloo carries 8-byte
ready/free signals per chunk and a 32-byte metadata all-gather per collective.
No CPU polling, GPU IPC, cross-process CUDA events, or payload socket
transfers are needed.
"""

import hashlib
import json
import mmap
import os
import socket
import struct
import tempfile
import uuid
from bisect import bisect_right
from collections.abc import Callable, Generator, Sequence
from contextlib import closing, contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .pin_manager import PinManager

BUFFERS_PER_RANK = 3
_MIN_PIPELINE_CHUNK_BYTES = 1024 * 1024
_TARGET_PIPELINE_CHUNKS = 6
_OPERATIONS = (
    "broadcast",
    "scatter",
    "allgather",
    "allreduce",
    "allgather_coalesced",
    "allreduce_coalesced",
)
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


def _shared_chunk_bytes(length: int, capacity: int) -> int:
    """Pick a deterministic 4 KiB-aligned chunk size up to capacity for full-duplex overlap.

    Depends only on ``(length, capacity)`` so ranks with different pinning
    outcomes choose identical chunk boundaries.
    """
    if length <= _MIN_PIPELINE_CHUNK_BYTES or capacity < 4096:
        return capacity
    min_chunks = (length + capacity - 1) // capacity
    desired_chunks = min(_TARGET_PIPELINE_CHUNKS, length // _MIN_PIPELINE_CHUNK_BYTES)
    chunks = max(min_chunks, desired_chunks)
    if chunks <= 1:
        return capacity
    return min(capacity, ((length + chunks - 1) // chunks + 4095) // 4096 * 4096)


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1).view(torch.uint8)


def _batch_signature(tensors: Sequence[torch.Tensor]) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for tensor in tensors:
        digest.update(struct.pack("<QQ", tensor.nbytes, COPY_DTYPES.index(tensor.dtype)))
    return struct.unpack("<q", digest.digest())[0]


def _pack_spans(lengths: Sequence[int], *, align: int = 1) -> tuple[list[tuple[int, int, int]], int]:
    """Pack non-empty byte lengths into ``(tensor_index, span_start, span_length)`` and total bytes."""
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for index, length in enumerate(lengths):
        if length == 0:
            continue
        if align > 1:
            offset = (offset + align - 1) // align * align
        spans.append((index, offset, length))
        offset += length
    return spans, offset


def _chunk_pieces(
    spans: Sequence[tuple[int, int, int]], span_ends: Sequence[int], start: int, end: int,
) -> list[tuple[int, int, int, int]]:
    """Return ``(tensor_index, slot_offset, tensor_offset, length)`` overlapping ``[start, end)``."""
    pieces: list[tuple[int, int, int, int]] = []
    index = bisect_right(span_ends, start)
    while index < len(spans):
        tensor_index, span_start, span_length = spans[index]
        if span_start >= end:
            break
        lo = max(start, span_start)
        hi = min(end, span_start + span_length)
        pieces.append((tensor_index, lo - start, lo - span_start, hi - lo))
        index += 1
    return pieces


def _make_slice_fill(
    source: torch.Tensor,
    asynchronous: bool,
) -> Callable[[torch.Tensor, int, int], None]:
    def fill(slot: torch.Tensor, start: int, width: int) -> None:
        slot.copy_(source[start:start + width], non_blocking=asynchronous)

    return fill


def _make_packed_fill(
    sources: Sequence[torch.Tensor],
    spans: Sequence[tuple[int, int, int]],
    span_ends: Sequence[int],
    asynchronous: bool,
) -> Callable[[torch.Tensor, int, int], None]:
    def fill(slot: torch.Tensor, start: int, width: int) -> None:
        for ti, slot_off, t_off, part_len in _chunk_pieces(spans, span_ends, start, start + width):
            slot[slot_off:slot_off + part_len].copy_(
                sources[ti][t_off:t_off + part_len], non_blocking=asynchronous,
            )

    return fill


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
        self.rank = rank
        self.world_size = world_size
        self._peers: tuple[tuple[int, int], ...] = tuple(
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
        self._recv_signals = torch.empty(max(1, world_size - 1), dtype=torch.int64)
        self._recv_slots = tuple(self._recv_signals[i:i + 1] for i in range(self._recv_signals.numel()))
        self._pending: list[dist.Work] = []

    def local_buffer(self) -> torch.Tensor:
        start = self.rank * self.buffer_count * self.capacity
        return self.buffer[start:start + self.buffer_count * self.capacity]

    def _slot(self, rank: int, lane: int, width: int) -> torch.Tensor:
        start = (rank * self.buffer_count + lane) * self.capacity
        return self.buffer[start:start + width]

    def _prepare_streams(self, device: torch.device) -> None:
        if self.device != device:
            download = torch.cuda.Stream(device=device)
            upload = torch.cuda.Stream(device=device)
            ready = [torch.cuda.Event() for _ in range(self.buffer_count)]
            uploaded = [torch.cuda.Event() for _ in range(self.buffer_count)]
            self.download = download
            self.upload = upload
            self.ready = ready
            self.uploaded = uploaded
            self.device = device
        assert self.download is not None
        assert self.upload is not None
        current = torch.cuda.current_stream(device)
        self.download.wait_stream(current)
        self.upload.wait_stream(current)

    def _signal(
        self, peers: Sequence[tuple[int | None, int | None]], tag: int, sequence: int,
    ) -> None:
        # Post all sends and receives before waiting, including across peers in
        # cyclic rounds and multi-rank reductions. Completion precedes reusing
        # these persistent control tensors.
        # PyTorch's Gloo stubs omit its send/recv and collective bindings.
        self._send_signal[0] = sequence
        receivers = 0
        try:
            for send, receive in peers:
                if send is not None:
                    self._pending.append(self.group.send([self._send_signal], send, tag))  # type: ignore[attr-defined]
                if receive is not None:
                    slot = self._recv_slots[receivers]
                    receivers += 1
                    self._pending.append(self.group.recv([slot], receive, tag))  # type: ignore[attr-defined]
            for work in reversed(self._pending):
                work.wait(self.timeout)
        finally:
            self._pending.clear()
        if receivers and any(self._recv_signals[i].item() != sequence for i in range(receivers)):
            raise RuntimeError("shared relay control sequence mismatch")

    def _transfer_single(
        self, source: torch.Tensor | None, destination: torch.Tensor | None,
        peers: Sequence[tuple[int | None, int | None]], sender: int | None, asynchronous: bool,
    ) -> None:
        tensor = source if source is not None else destination
        assert tensor is not None
        fill = None if source is None else _make_slice_fill(source, asynchronous)
        with closing(self._chunks(fill, tensor.numel(), peers, asynchronous)) as chunks:
            for index, start, width in chunks:
                if destination is not None:
                    assert sender is not None
                    destination[start:start + width].copy_(
                        self._slot(sender, index % self.buffer_count, width), non_blocking=asynchronous,
                    )

    def _chunks(
        self, fill: Callable[[torch.Tensor, int, int], None] | None, length: int,
        peers: Sequence[tuple[int | None, int | None]], asynchronous: bool,
    ) -> Generator[tuple[int, int, int]]:
        """Yield published chunks on the upload stream; acknowledge every reader."""
        chunk_bytes = _shared_chunk_bytes(length, self.capacity)
        count = max(1, (length + chunk_bytes - 1) // chunk_bytes)
        receiving = any(receive is not None for _, receive in peers)
        release_peers = [(receive, send) for send, receive in peers]

        def download(index: int) -> None:
            if fill is None:
                return
            start = index * chunk_bytes
            width = min(chunk_bytes, length - start)
            with torch.cuda.stream(self.download) if asynchronous else nullcontext():
                fill(self._slot(self.rank, index % self.buffer_count, width), start, width)
                if asynchronous:
                    self.ready[index % self.buffer_count].record(self.download)

        def release(index: int) -> None:
            if receiving and asynchronous:
                self.uploaded[index % self.buffer_count].synchronize()
            # The receiver acknowledges only after its GPU stops reading the
            # shared slot. This is the producer's permission to overwrite it.
            self._signal(release_peers, 1, index)

        for index in range(min(self.buffer_count, count)):
            download(index)
        for index in range(count):
            # Releasing chunk (index - (B - 1)) frees the exact lane (index + 1)
            # will fill while upload(index - 1) remains in flight on another lane.
            if index >= self.buffer_count - 1 and index + 1 < count:
                release(index - (self.buffer_count - 1))
                download(index + 1)
            if fill is not None and asynchronous:
                self.ready[index % self.buffer_count].synchronize()
            self._signal(peers, 0, index)
            start = index * chunk_bytes
            width = min(chunk_bytes, length - start)
            with torch.cuda.stream(self.upload) if asynchronous else nullcontext():
                yield index, start, width
                if receiving and asynchronous:
                    self.uploaded[index % self.buffer_count].record(self.upload)
        # All uploads execute in FIFO order on self.upload, so waiting for the
        # final chunk confirms every earlier peer slot read has finished.
        release(count - 1)

    def broadcast(self, tensor: torch.Tensor, root: int, manager: PinManager) -> None:
        with self._collective("broadcast", [tensor], root, manager) as asynchronous:
            if self.world_size == 1 or tensor.numel() == 0:
                return
            data = _bytes(tensor)
            if self.rank == root:
                peers: Sequence[tuple[int | None, int | None]] = [
                    (peer, None) for peer in range(self.world_size) if peer != root
                ]
                self._transfer_single(data, None, peers, None, asynchronous)
            else:
                self._transfer_single(None, data, [(None, root)], root, asynchronous)

    def scatter(
        self, sources: Sequence[torch.Tensor], output: torch.Tensor, root: int, manager: PinManager,
    ) -> None:
        with self._collective("scatter", [output], root, manager) as asynchronous:
            destination = _bytes(output)
            if self.rank == root:
                self._local_copy(destination, _bytes(sources[root]))
            if self.world_size == 1 or destination.numel() == 0:
                return
            for peer in range(self.world_size):
                if peer == root:
                    continue
                if self.rank == root:
                    self._transfer_single(_bytes(sources[peer]), None, [(peer, None)], None, asynchronous)
                elif self.rank == peer:
                    self._transfer_single(None, destination, [(None, root)], root, asynchronous)

    def allgather(
        self, sources: Sequence[torch.Tensor], outputs_list: Sequence[Sequence[torch.Tensor]], manager: PinManager,
    ) -> None:
        op = "allgather" if len(sources) == 1 else "allgather_coalesced"
        with self._collective(op, sources, -1, manager) as asynchronous:
            byte_sources = [_bytes(t) for t in sources]
            byte_outputs = [[_bytes(out) for out in outs] for outs in outputs_list]
            for src, outs in zip(byte_sources, byte_outputs, strict=True):
                self._local_copy(outs[self.rank], src)
            spans, total_bytes = _pack_spans([t.numel() for t in byte_sources])
            if self.world_size == 1 or total_bytes == 0:
                return
            span_ends = [start + length for _, start, length in spans]
            fill = _make_packed_fill(byte_sources, spans, span_ends, asynchronous)
            with closing(self._chunks(fill, total_bytes, self._peers, asynchronous)) as chunks:
                for index, start, width in chunks:
                    lane = index % self.buffer_count
                    pieces = _chunk_pieces(spans, span_ends, start, start + width)
                    for _, receive in self._peers:
                        peer_slot = self._slot(receive, lane, width)
                        for ti, slot_off, t_off, part_len in pieces:
                            byte_outputs[ti][receive][t_off:t_off + part_len].copy_(
                                peer_slot[slot_off:slot_off + part_len], non_blocking=asynchronous,
                            )

    def allreduce(self, tensors: Sequence[torch.Tensor], manager: PinManager) -> None:
        op = "allreduce" if len(tensors) == 1 else "allreduce_coalesced"
        with self._collective(op, tensors, -1, manager) as asynchronous:
            byte_sources = [_bytes(t) for t in tensors]
            # 16-byte alignment keeps every piece valid for native/FP32 views
            # across mixed-dtype chunks while fitting within 16-aligned slots.
            spans, total_bytes = _pack_spans([t.numel() for t in byte_sources], align=16)
            if self.world_size == 1 or total_bytes == 0:
                return
            buffer = self._reduction_buffer
            if buffer is None or buffer.device != tensors[0].device:
                # One native-dtype receive chunk and one FP32 accumulator.
                # Both are reused on the upload stream, including across calls.
                buffer = self._reduction_buffer = torch.empty(
                    3 * self.capacity, dtype=torch.uint8, device=tensors[0].device,
                )
            span_ends = [start + length for _, start, length in spans]
            fill = _make_packed_fill(byte_sources, spans, span_ends, asynchronous)
            with closing(self._chunks(fill, total_bytes, self._peers, asynchronous)) as chunks:
                for index, start, width in chunks:
                    lane = index % self.buffer_count
                    peer_slots = [
                        None if peer == self.rank else self._slot(peer, lane, width)
                        for peer in range(self.world_size)
                    ]
                    for ti, slot_off, t_off, part_len in _chunk_pieces(spans, span_ends, start, start + width):
                        dtype = tensors[ti].dtype
                        output = byte_sources[ti][t_off:t_off + part_len].view(dtype)
                        incoming = buffer[:part_len].view(dtype)
                        accumulator = buffer[
                            self.capacity:self.capacity + output.numel() * 4
                        ].view(torch.float32)
                        # All peers have published the original chunk. Accumulate
                        # in the same order everywhere, casting back only once.
                        for peer, peer_slot in enumerate(peer_slots):
                            value = output
                            if peer_slot is not None:
                                incoming.copy_(
                                    peer_slot[slot_off:slot_off + part_len].view(dtype),
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
        self, operation: str, tensors: Sequence[torch.Tensor], detail: int, manager: PinManager,
    ) -> Generator[bool]:
        if self.broken:
            raise RuntimeError("shared relay failed previously; destroy and recreate the process group")
        device = tensors[0].device
        # Check rank agreement before any shared-slot writes. This also keeps
        # each collective separate from earlier peer rounds.
        op_index = _OPERATIONS.index(operation)
        if operation in ("allgather_coalesced", "allreduce_coalesced"):
            metadata = torch.tensor([
                op_index,
                sum(t.nbytes for t in tensors),
                len(tensors),
                _batch_signature(tensors),
            ], dtype=torch.int64)
        else:
            metadata = torch.tensor([
                op_index,
                tensors[0].nbytes,
                COPY_DTYPES.index(tensors[0].dtype),
                detail,
            ], dtype=torch.int64)
        gathered = [torch.empty_like(metadata) for _ in range(self.world_size)]
        self.group.allgather([gathered], [metadata]).wait(self.timeout)  # type: ignore[attr-defined]
        if any(not torch.equal(metadata, other) for other in gathered):
            raise ValueError("shared relay collective operation, payload bytes and root/dtype must match across ranks")
        with manager.acquire([self.buffer]) if device.type == "cuda" else nullcontext() as lease:
            asynchronous = lease is not None and lease.pageable_bytes == 0
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
                    try:
                        if self.download is not None:
                            self.download.synchronize()
                    finally:
                        if self.upload is not None:
                            self.upload.synchronize()
                if device.type == "cuda":
                    # Includes the local GPU contribution, which is copied on
                    # the current stream instead of passing through the relay.
                    torch.cuda.current_stream(device).synchronize()

    @staticmethod
    def _local_copy(destination: torch.Tensor, source: torch.Tensor) -> None:
        # The local contribution never needs a trip through host memory.
        if destination.data_ptr() != source.data_ptr():
            destination.copy_(source)
