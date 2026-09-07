"""Experimental, blocking host-relay process group for DTensor inference.

Call ``register_relay_backend()`` in each worker before initializing a
``piper_relay`` process group. It supports SUM all-reduce in FP32, FP16 and
BF16, plus broadcast, scatter and all-gather of contiguous tensors. CPU Gloo
performs communication; Gloo never receives an accelerator tensor.

This first implementation stages the entire tensor and completes both device
copies before returning, including when ``async_op=True``. It establishes the
DTensor integration boundary; chunked staging and asynchronous progress are
separate follow-up work. No offloader, NCCL, or compiled extension is required.
"""

import threading
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import timedelta
from itertools import pairwise

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import AllgatherOptions

from .pin_manager import host_pin_manager

_BACKEND_NAME = "piper_relay"
_registration_lock = threading.Lock()
_REDUCTION_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_COPY_DTYPES = (
    *_REDUCTION_DTYPES,
    torch.float64, torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    torch.complex64, torch.complex128,
)


def register_relay_backend() -> None:
    """Register the experimental ``piper_relay`` backend in this process.

    Registration is idempotent and does not initialize an accelerator or a
    process group. Each spawned worker must call this function. Initialize
    with ``backend="piper_relay"`` after selecting the worker's accelerator
    with ``torch.cuda.set_device``; omit ``device_id`` from initialization.
    The Python process group does not implement eager accelerator connection.
    """
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("piper_relay requires a PyTorch build with CPU Gloo support")
    with _registration_lock:
        plugin = dist.Backend._plugins.get(_BACKEND_NAME.upper())
        if plugin is not None:
            if plugin.creator_fn is not _RelayProcessGroup:
                raise RuntimeError("A different backend is already registered as piper_relay")
            return
        dist.Backend.register_backend(
            _BACKEND_NAME,
            _RelayProcessGroup,
            devices=["cpu", "cuda"],
        )


class _RelayProcessGroup(dist.ProcessGroup):
    def __init__(self, store: dist.Store, rank: int, size: int, timeout: timedelta) -> None:
        # The rank/size overload constructs the Python trampoline. The
        # store/rank/size overload in PyTorch 2.13 cannot construct subclasses.
        super().__init__(rank, size)  # type: ignore[call-arg]
        self._collective_lock = threading.Lock()
        self._cpu_group = dist.ProcessGroupGloo(store, rank, size, timeout)
        # Give the base process group ownership of Gloo for shutdown and CPU
        # control operations. Deliberately do not register Gloo for CUDA.
        self._register_backend(
            torch.device("cpu"),
            dist.ProcessGroup.BackendType.GLOO,
            self._cpu_group,
        )

    def getBackendName(self) -> str:  # noqa: N802 -- PyTorch trampoline name
        return _BACKEND_NAME

    def allreduce(
        self,
        tensors: list[torch.Tensor],
        opts: dist.AllreduceOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.AllreduceOptions()
        tensor = _single_tensor(tensors)
        _validate_reduction([tensor], options)
        with self._collective_lock, torch.no_grad():
            self._reduce(tensor, options)
        return _CompletedWork([tensor])

    def allreduce_coalesced(
        self,
        tensors: list[torch.Tensor],
        opts: dist.AllreduceCoalescedOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.AllreduceCoalescedOptions()
        _validate_reduction(tensors, options)
        _validate_disjoint(tensors)
        # PyTorch's Python bindings do not accept coalesced options in the
        # single-tensor Gloo overload, despite their common C++ fields.
        single_options = dist.AllreduceOptions()
        single_options.reduceOp = options.reduceOp
        single_options.timeout = options.timeout
        single_options.asyncOp = options.asyncOp
        with self._collective_lock, torch.no_grad():
            for tensor in tensors:
                self._reduce(tensor, single_options)
        return _CompletedWork(tensors)

    def _reduce(self, tensor: torch.Tensor, options: dist.AllreduceOptions) -> None:
        with _host_staging([tensor], [tensor]) as (inputs, _outputs):
            host = inputs[0]
            # Promote on the CPU after download, so low-precision reduction
            # needs neither a full FP32 GPU temporary nor FP32 device copies.
            accumulator = host.float()
            self._cpu_group.allreduce([accumulator], options).wait()
            if accumulator is not host:
                host.copy_(accumulator)

    def broadcast(
        self,
        tensors: list[torch.Tensor],
        opts: dist.BroadcastOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.BroadcastOptions()
        tensor = _single_tensor(tensors)
        _validate_tensors([tensor])
        self._validate_root(options.rootRank)
        if options.rootTensor != 0:
            raise ValueError("piper_relay broadcast requires rootTensor=0")
        sources = [tensor] if self.rank() == options.rootRank else []
        with self._collective_lock, torch.no_grad(), _host_staging(sources, [tensor]) as (_inputs, outputs):
            self._cpu_group.broadcast([_bytes(outputs[0])], options).wait()
        return _CompletedWork([tensor])

    def scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: dist.ScatterOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.ScatterOptions()
        output = _single_tensor(output_tensors)
        self._validate_root(options.rootRank)
        sources: list[torch.Tensor] = []
        if self.rank() == options.rootRank:
            if len(input_tensors) != 1 or len(input_tensors[0]) != self.size():
                raise ValueError("piper_relay scatter root must provide one tensor per rank")
            sources = input_tensors[0]
        elif input_tensors:
            raise ValueError("piper_relay scatter inputs belong only on the root rank")
        _validate_tensors([output, *sources])
        for source_rank, tensor in enumerate(sources):
            _validate_pair(output, tensor, tensor.numel(), same_shape=True)
            # Gloo may copy the root's result while another rank's send still
            # reads its source. Only an exact alias of the root's own slice
            # is safe; enforce the same contract for CPU and accelerator use.
            if _overlap(output, tensor) and (
                source_rank != options.rootRank or output.data_ptr() != tensor.data_ptr()
            ):
                raise ValueError("piper_relay scatter output must not overlap inputs except its exact root slice")
        with self._collective_lock, torch.no_grad(), _host_staging(sources, [output]) as (inputs, outputs):
            cpu_inputs = [[_bytes(tensor) for tensor in inputs]] if sources else []
            self._cpu_group.scatter([_bytes(outputs[0])], cpu_inputs, options).wait()
        return _CompletedWork([output])

    def allgather(
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else AllgatherOptions()
        source = _single_tensor(input_tensors)
        if len(output_tensors) != 1 or len(output_tensors[0]) != self.size():
            raise ValueError("piper_relay all-gather requires one output tensor per rank")
        destinations = output_tensors[0]
        _validate_tensors([source, *destinations])
        _validate_disjoint(destinations)
        for output in destinations:
            _validate_pair(output, source, source.numel(), same_shape=True)
        with self._collective_lock, torch.no_grad(), _host_staging([source], destinations) as (inputs, outputs):
            self._cpu_group.allgather([[_bytes(tensor) for tensor in outputs]], [_bytes(inputs[0])], options).wait()
        return _CompletedWork(destinations, nested=True)

    def all_gather_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch parameter name
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        return self.all_gather_single_coalesced([output], [input], opts)

    def _allgather_base(
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch parameter name
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        return self.all_gather_single(output, input, opts)

    def all_gather_single_coalesced(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else AllgatherOptions()
        if len(output_tensors) != len(input_tensors) or not input_tensors:
            raise ValueError("piper_relay all-gather requires matching nonempty input/output lists")
        _validate_tensors([*input_tensors, *output_tensors])
        _validate_disjoint(output_tensors)
        for i, output in enumerate(output_tensors):
            for j, source in enumerate(input_tensors):
                if i != j and _overlap(output, source):
                    raise ValueError("piper_relay batched outputs must not overlap another input")
        # Validate the whole batch before starting communication or mutation.
        for output, source in zip(output_tensors, input_tensors, strict=True):
            _validate_pair(output, source, source.numel() * self.size())
        with self._collective_lock, torch.no_grad():
            for output, source in zip(output_tensors, input_tensors, strict=True):
                with _host_staging([source], [output]) as (inputs, outputs):
                    rows = _bytes(outputs[0]).reshape(self.size(), source.nbytes)
                    self._cpu_group.allgather([list(rows.unbind(0))], [_bytes(inputs[0])], options).wait()
        return _CompletedWork(output_tensors)

    def _validate_root(self, root: int) -> None:
        if not 0 <= root < self.size():
            raise ValueError("piper_relay root rank is outside the process group")


class _CompletedWork(dist.Work):
    """Expose an already completed collective to Python and c10d callers."""

    def __init__(self, tensors: list[torch.Tensor], *, nested: bool = False) -> None:
        super().__init__()
        self._tensors = tensors
        self._future: torch.futures.Future[list[torch.Tensor] | list[list[torch.Tensor]]] = torch.futures.Future()
        self._future.set_result([tensors] if nested else tensors)

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        del timeout
        return True

    def is_completed(self) -> bool:
        return True

    def is_success(self) -> bool:
        return True

    def get_future(self) -> torch.futures.Future[list[torch.Tensor] | list[list[torch.Tensor]]]:
        return self._future

    def result(self) -> list[torch.Tensor]:
        return self._tensors

    def synchronize(self) -> None:
        pass

    def block_current_stream(self) -> None:
        pass


def _validate_reduction(tensors: list[torch.Tensor], options: dist.AllreduceOptions) -> None:
    if options.reduceOp != dist.ReduceOp.SUM:
        raise NotImplementedError("piper_relay currently supports only SUM all-reduce")
    _validate_tensors(tensors)
    for tensor in tensors:
        if tensor.dtype not in _REDUCTION_DTYPES:
            raise NotImplementedError("piper_relay SUM all-reduce supports FP32, FP16 and BF16")


def _single_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) != 1:
        raise NotImplementedError("piper_relay requires exactly one local tensor for this collective")
    return tensors[0]


def _validate_tensors(tensors: Sequence[torch.Tensor]) -> None:
    if not tensors:
        raise ValueError("piper_relay requires a nonempty tensor list")
    for tensor in tensors:
        _validate_tensor(tensor)
        if tensor.device != tensors[0].device:
            raise ValueError("piper_relay requires one local device per collective")


def _validate_tensor(tensor: torch.Tensor) -> None:
    if type(tensor) is not torch.Tensor:
        raise TypeError("piper_relay expects a plain local tensor, not a tensor subclass")
    if tensor.device.type not in ("cpu", "cuda"):
        raise NotImplementedError("piper_relay supports only CPU and CUDA/HIP tensors")
    if tensor.dtype not in _COPY_DTYPES:
        raise NotImplementedError(f"piper_relay does not support dtype {tensor.dtype}")
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise NotImplementedError("piper_relay requires a contiguous strided tensor")


def _validate_pair(
    output: torch.Tensor, source: torch.Tensor, expected_numel: int, *, same_shape: bool = False,
) -> None:
    if output.dtype != source.dtype or output.numel() != expected_numel:
        raise ValueError("piper_relay output must have the expected element count and input dtype")
    if same_shape and output.shape != source.shape:
        raise ValueError("piper_relay list collectives require equal tensor shapes")


def _overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(left.nbytes and right.nbytes) and (
        left.data_ptr() < right.data_ptr() + right.nbytes
        and right.data_ptr() < left.data_ptr() + left.nbytes
    )


def _validate_disjoint(tensors: Sequence[torch.Tensor]) -> None:
    ordered = sorted((tensor for tensor in tensors if tensor.nbytes), key=lambda tensor: tensor.data_ptr())
    for left, right in pairwise(ordered):
        if _overlap(left, right):
            raise ValueError("piper_relay collective outputs must not overlap")


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Transport payload bits unchanged, including BF16 and integer metadata."""
    return tensor.reshape(-1).view(torch.uint8)


@contextmanager
def _host_staging(
    inputs: list[torch.Tensor], outputs: list[torch.Tensor],
) -> Generator[tuple[list[torch.Tensor], list[torch.Tensor]]]:
    # Callers validate devices and always supply at least one output.
    device = outputs[0].device
    if device.type == "cpu":
        yield inputs, outputs
        return

    tensors = {id(tensor): tensor for tensor in [*inputs, *outputs]}
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("piper_relay blocking transfers do not support CUDA graph capture")
        hosts = {
            key: torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
            for key, tensor in tensors.items()
        }
        # Pageable fallback is safe because every device copy is synchronous.
        # Keep all sources and destinations leased through communication and
        # final uploads; an exception exits without uploading partial results.
        with host_pin_manager.acquire(hosts.values()):
            for tensor in inputs:
                hosts[id(tensor)].copy_(tensor, non_blocking=False)
            yield [hosts[id(tensor)] for tensor in inputs], [hosts[id(tensor)] for tensor in outputs]
            for tensor in outputs:
                tensor.copy_(hosts[id(tensor)], non_blocking=False)


__all__ = ["register_relay_backend"]
