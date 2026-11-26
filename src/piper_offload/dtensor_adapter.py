"""Composing adapter for PyTorch ``DTensor`` (tensor-parallel) weights.

A ``DTensor`` is an *orthogonal outer wrapper*: a local shard plus a
``DeviceMesh`` and placements. Moving it across the CPU<->GPU boundary is
"move the local shard, then replay the wrapper". The local shard is itself
an ordinary tensor — plain, or a quantized subclass — so this adapter
**delegates all local-shard movement to whatever adapter the registry
already selects** for it, and adds only the distributed wrapper.

One adapter therefore composes with plain and Piper-subclass
adapters: a ``DTensor`` wrapping a ``ConvRotInt8Tensor`` reuses
``PiperConvRotInt8Adapter`` for the local shard, a ``DTensor`` wrapping a plain
tensor reuses ``RegularAdapter``, and so on — no per-quant ``DTensor``
variants.

Local shards must retain their tensor representation through ``.data``
because reconstruction reads it from the inner wrapper. ``capture_host``
rejects local ``nn.Parameter`` subclasses, which may carry state on the
object itself.

Scope: **movement plus shard-local additive merge, for frozen-inference tensor
parallelism.** Additive merge supports ordinary ``Replicate`` and contiguous
``Shard`` placements. It selects the host dense regions and, for rank-two
weights, LoRA factor regions needed by each rank before device staging, then
delegates the actual update to the local shard's adapter. This keeps
tensor-parallel concerns out of format-specific quant adapters.

Dense ``Replicate`` and contiguous ``Shard(dim)`` tensors across 1D and N-D
meshes can also be projected from a full plain host parameter or a supported
non-``nn.Parameter``-subclass quantized host parameter (such as Piper
``PiperNVFP4Tensor``, or 2-D Piper ``ConvRotInt8Tensor`` /
``ConvRotNVFP4Tensor`` via
:class:`~piper_offload.tensor_adapters.ShardProjectionTensorAdapter`). The
target DTensor supplies only its distributed layout and may use a meta local
tensor; host storage remains a zero-copy view or alias of the original host
allocation, including mmap-backed checkpoint storage. Non-canonical projected
shards (such as column-sharded quantized matrices) rest on CPU in a minimal
zero-allocation placeholder until CUDA activation materializes contiguous
``1/P`` device shards.

A DTensor merge is available only when its local shard's adapter
supports the corresponding factorized or dense merge operation; inference
without updates remains supported otherwise.

Identity and layout keys are taken from the **local shard** (delegated to
the inner adapter) plus a structural ``(mesh, placements)`` signature. This
sidesteps the two ``DTensor`` traps: its outer ``data_ptr()`` is ``0`` (which
would collapse tied-weight dedup) and its ``shape``/``numel`` are *global*
(which would over-account :meth:`cache_bytes` by the world size).

The resting (deactivated) weight stays a ``DTensor`` but on a CPU
``DeviceMesh`` (:func:`~piper_offload._dtensor.cpu_mesh_for`, cached on host
state during capture or projection) so its local shard stays on the host —
``from_local`` moves the local onto the mesh device (verified), so a CUDA mesh
would re-allocate GPU memory for the offloaded state. Keeping it a DTensor
(rather than a bare local shard) means a deactivated block's weight still
resolves to this adapter, so the store<->module bind and the resident GPU form
stay layout-consistent; :meth:`accepts_bind_target` ignores the mesh device type
so the CPU-resting and CUDA-resident forms match. The resident GPU parameter is
rebuilt on the original CUDA mesh in :meth:`gpu_param`.
"""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from ._dtensor import (
    Replicate,
    Shard,
    cpu_mesh_for,
    is_dtensor,
    local_shard,
    mesh_signature,
    placements_key,
    rebuild_dtensor,
    require_dtensor,
)
from .seeding import derive_seed
from .structured_adapter import _ProjectedStructuredShard
from .tensor_adapters import (
    DenseMergeTensorAdapter,
    DenseMergeValidationTensorAdapter,
    LogicalShapeTensorAdapter,
    LoRAMergeTensorAdapter,
    LoRAMergeValidationTensorAdapter,
    ShardProjectionTensorAdapter,
    TensorAdapter,
    accepts_bind_target,
    adapter_name,
)
from .tensor_adapters import (
    gpu_target_bytes as _gpu_target_bytes,
)


def _mesh_target_key(mesh: Any) -> tuple[object, ...]:  # noqa: ANN401
    """Target-pool mesh key preserving dim names and initialized process groups."""
    group_ids: list[int | None] = []
    for dim in range(mesh.ndim):
        try:
            group = mesh.get_group(dim)
        except Exception:
            group_ids.append(None)
        else:
            group_ids.append(id(group))
    return (mesh_signature(mesh), mesh.mesh_dim_names, tuple(group_ids))


@dataclass(slots=True)
class _DTensorHost:
    """Host state for a DTensor: the inner adapter's host local-shard
    state plus the distributed wrapper to replay on reconstruction. ``mesh`` is
    the target mesh — used as-is by :meth:`gpu_param` for the resident weight
    (which computes, so it must reuse the model's canonical mesh) — and
    ``cpu_mesh`` is cached via :func:`~piper_offload._dtensor.cpu_mesh_for` at
    capture/projection time for :meth:`cpu_param`. ``shape`` / ``stride`` are
    the original GLOBAL shape/stride, captured so the rebuilt DTensor reports
    the true global shape even for uneven shards (where ``from_local`` would
    otherwise infer ``local_size * world_size``), and
    ``inner_layout_signature`` is the inner adapter's layout signature used
    during recursive bind validation."""

    inner: TensorAdapter[Any, Any]
    inner_state: object
    inner_layout_signature: tuple
    mesh: object
    cpu_mesh: object
    placements: tuple[object, ...]
    shape: object
    stride: tuple[int, ...]


@dataclass(slots=True)
class _DTensorGpu:
    """GPU state for a DTensor: the inner adapter's GPU local-shard state."""

    inner_gpu: object


def _select(local: torch.Tensor) -> TensorAdapter[Any, Any]:
    # Lazy import: the registry imports this module, so importing
    # select_adapter at module load would be circular. A DTensor's local
    # shard is never itself a DTensor, so this never re-enters this adapter.
    from .tensor_adapter_registry import select_adapter  # noqa: PLC0415

    return select_adapter(local)


def _local_shape_and_offsets(
    global_shape: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    coordinate: tuple[int, ...],
    placements: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return this rank's contiguous shard shape and global offsets.

    Supports the common tensor-parallel placement vocabulary: replication and
    contiguous sharding. Applying placements from left to right mirrors
    DTensor's sharding semantics, including uneven and repeated sharding of one
    tensor dimension.
    """
    if not (len(mesh_shape) == len(coordinate) == len(placements)):
        raise ValueError(
            "DTensor layout requires one placement and coordinate per "
            f"mesh dimension; mesh shape is {mesh_shape}, coordinate is "
            f"{coordinate}, and placements are {placements}."
        )

    local_shape = list(global_shape)
    offsets = [0] * len(global_shape)
    for mesh_dim, (mesh_size, shard_coordinate, placement) in enumerate(
        zip(mesh_shape, coordinate, placements, strict=True)
    ):
        if isinstance(placement, Replicate):
            continue
        if type(placement) is not Shard:
            raise ValueError(
                "DTensor layout supports only Replicate and contiguous "
                f"Shard placements; mesh dimension {mesh_dim} uses "
                f"{placement!r}."
            )

        shard = cast(Any, placement)
        tensor_dim = shard.dim
        if tensor_dim < 0:
            tensor_dim += len(global_shape)
        if not 0 <= tensor_dim < len(global_shape):
            raise ValueError(
                f"DTensor merge placement {placement!r} refers to tensor "
                f"dimension {shard.dim}, but the target shape is "
                f"{global_shape}."
            )

        shard_size, relative_offset = Shard.local_shard_size_and_offset(
            local_shape[tensor_dim],
            mesh_size,
            shard_coordinate,
        )
        local_shape[tensor_dim] = int(shard_size)
        if shard_size == 0:
            # Match DTensor's checkpointing offset semantics: every empty
            # shard is anchored at the end of the original global dimension,
            # including when repeated sharding made an earlier local shard
            # empty.
            offsets[tensor_dim] = global_shape[tensor_dim]
        else:
            offsets[tensor_dim] += int(relative_offset)

    return tuple(local_shape), tuple(offsets)


def _logical_shape(
    tensor: torch.Tensor,
    adapter: TensorAdapter[Any, Any],
) -> tuple[int, ...]:
    if isinstance(adapter, LogicalShapeTensorAdapter):
        return adapter.logical_shape(tensor)
    return tuple(tensor.shape)


@dataclass(slots=True, frozen=True)
class _DTensorLayoutContext[InnerT]:
    """Validated global-to-local layout independent of merge capability."""

    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    offsets: tuple[int, ...]
    local: torch.Tensor
    inner: InnerT


type _DTensorLoRAMergeContext = _DTensorLayoutContext[
    LoRAMergeTensorAdapter[Any, Any]
]
type _DTensorDenseMergeContext = _DTensorLayoutContext[
    DenseMergeTensorAdapter[Any, Any]
]


def _localize_rounding_seed(
    seed: int | None,
    offsets: tuple[int, ...],
) -> int | None:
    """Keep replicas aligned while giving distinct shards distinct samples."""
    if seed is None:
        return None
    return derive_seed(seed, *offsets)


def _layout_context(
    target: torch.Tensor,
) -> _DTensorLayoutContext[TensorAdapter[Any, Any]]:
    """Validate a DTensor's global-to-local contiguous shard layout."""
    dt = require_dtensor(target)
    global_shape = tuple(dt.shape)
    coordinate = dt.device_mesh.get_coordinate()
    if coordinate is None:
        raise ValueError("DTensor merge cannot run on a rank outside the target device mesh.")
    local_shape, offsets = _local_shape_and_offsets(
        global_shape,
        tuple(dt.device_mesh.shape),
        tuple(coordinate),
        tuple(dt.placements),
    )

    local = local_shard(dt)
    inner = _select(local)
    actual_local_shape = _logical_shape(local, inner)
    if actual_local_shape != local_shape:
        raise ValueError(
            "DTensor merge computed a local shard shape that does not "
            "match the target representation: "
            f"computed={local_shape}, actual={actual_local_shape}, "
            f"global={global_shape}, placements={tuple(dt.placements)}."
        )

    return _DTensorLayoutContext(
        global_shape=global_shape,
        local_shape=local_shape,
        offsets=offsets,
        local=local,
        inner=inner,
    )


def _merge_context(target: torch.Tensor) -> _DTensorLoRAMergeContext:
    """Validate a DTensor LoRA merge target before any weight mutation."""
    context = _layout_context(target)
    if len(context.global_shape) != 2:
        raise ValueError(
            "DTensor LoRA merge requires a rank-two weight, got global shape "
            f"{context.global_shape}."
        )
    if not isinstance(context.inner, LoRAMergeTensorAdapter):
        raise ValueError(
            f"DTensor local shard adapter {adapter_name(context.inner)} does not support "
            "LoRA merge."
        )

    return cast(_DTensorLoRAMergeContext, context)


def _dense_merge_context(
    target: torch.Tensor,
) -> _DTensorDenseMergeContext:
    """Validate dense-merge support on this rank's local shard."""
    context = _layout_context(target)
    if not isinstance(context.inner, DenseMergeTensorAdapter):
        raise ValueError(
            f"DTensor local shard adapter {adapter_name(context.inner)} does "
            "not support dense parameter merge."
        )
    return cast(_DTensorDenseMergeContext, context)


def _local_dense_update(
    context: _DTensorLayoutContext[Any],
    update: torch.Tensor,
) -> torch.Tensor:
    """Accept a global or already-local dense update for this rank."""
    shape = tuple(update.shape)
    if shape == context.local_shape:
        return update.contiguous()
    if shape != context.global_shape:
        raise ValueError(
            "DTensor dense update must match either the global or local shape: "
            f"global={context.global_shape}, local={context.local_shape}, "
            f"update={shape}."
        )
    local = update
    for dim, (offset, size) in enumerate(
        zip(context.offsets, context.local_shape, strict=True)
    ):
        local = local.narrow(dim, offset, size)
    return local.contiguous()


def _local_lora_factors(
    context: _DTensorLoRAMergeContext,
    b: torch.Tensor,
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept global or already-local factors and return this rank's slices."""
    if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError(
            "DTensor LoRA factors must be rank two with matching inner "
            f"dimensions, got B={tuple(b.shape)}, A={tuple(a.shape)}."
        )

    factor_shape = (b.shape[0], a.shape[1])
    if factor_shape == context.local_shape:
        local_b = b
        local_a = a
    elif factor_shape == context.global_shape:
        out_offset, in_offset = context.offsets
        out_size, in_size = context.local_shape
        local_b = b.narrow(0, out_offset, out_size)
        local_a = a.narrow(1, in_offset, in_size)
    else:
        raise ValueError(
            "DTensor LoRA factors must match either the global or local "
            f"weight shape: global={context.global_shape}, "
            f"local={context.local_shape}, B={tuple(b.shape)}, "
            f"A={tuple(a.shape)}."
        )
    return local_b.contiguous(), local_a.contiguous()


def require_initialized_mesh(mesh: Any) -> None:  # noqa: ANN401
    """Validate that ``mesh`` targets CPU/CUDA and owns initialized groups."""
    if mesh.device_type not in ("cpu", "cuda"):
        raise ValueError(
            "DTensor projection target must use a CPU or CUDA/HIP device mesh; "
            f"got {mesh.device_type!r}."
        )
    for dim in range(mesh.ndim):
        try:
            mesh.get_group(dim)
        except Exception as exc:
            raise ValueError(
                "DTensor projection requires a DeviceMesh with initialized process groups."
            ) from exc


def _source_tensor_data(
    source_adapter: TensorAdapter[Any, Any],
    source_state: object,
    *,
    mode: str,
) -> torch.Tensor:
    source_cpu = source_adapter.cpu_param(source_state)
    if (
        issubclass(type(source_cpu), nn.Parameter)
        and type(source_cpu) is not nn.Parameter
    ) or issubclass(type(source_cpu.data), nn.Parameter):
        raise NotImplementedError(
            f"DTensor {mode} projection supports adapters whose local shard "
            "reconstructs without an nn.Parameter subclass wrapper."
        )
    return source_cpu.data


def _validate_dtensor_wrapping(
    local: torch.Tensor,
    cpu_mesh: Any,  # noqa: ANN401
    placements: tuple[object, ...],
    shape: Any,  # noqa: ANN401
    stride: tuple[int, ...],
    *,
    mode: str,
) -> None:
    """Preflight DTensor and Parameter wrapping before model mutation."""
    if type(local) is not torch.Tensor:
        try:
            # Validate during preflight that DTensor wrapping and Parameter
            # wrapping succeed for this tensor subclass (some subclasses lack
            # aten.view and would otherwise fail late during install_host after
            # earlier components mutated the shell).
            nn.Parameter(
                rebuild_dtensor(local, cpu_mesh, placements, shape, stride),
                requires_grad=False,
            )
        except (AssertionError, NotImplementedError, RuntimeError, IndexError) as exc:
            raise NotImplementedError(
                f"DTensor {mode} projection requires a local tensor representation "
                f"compatible with DTensor wrapping ({type(local).__name__}): {exc}"
            ) from exc
    if isinstance(local, _ProjectedStructuredShard):
        meta_tensor = local._meta_tensor
        try:
            nn.Parameter(
                rebuild_dtensor(meta_tensor, cpu_mesh, placements, shape, stride),
                requires_grad=False,
            )
        except (AssertionError, NotImplementedError, RuntimeError, IndexError) as exc:
            raise NotImplementedError(
                f"DTensor {mode} projection requires an active local tensor "
                f"representation compatible with DTensor wrapping ({type(meta_tensor).__name__}): {exc}"
            ) from exc


def _project_replicated_host(
    source_adapter: TensorAdapter[Any, Any],
    source_state: object,
    dt: Any,  # noqa: ANN401
    placements: tuple[object, ...],
) -> _DTensorHost:
    """Project an unsharded host backing onto an all-``Replicate`` DTensor."""
    source_data = _source_tensor_data(
        source_adapter,
        source_state,
        mode="Replicate",
    )
    if tuple(source_data.shape) != tuple(dt.shape):
        raise ValueError("DTensor projection source and target shapes must match.")
    target_local = local_shard(dt)
    inner_layout_signature = source_adapter.layout_signature(source_data)
    if not accepts_bind_target(
        source_adapter,
        source_state,
        inner_layout_signature,
        target_local,
    ):
        raise ValueError(
            "DTensor projection local target does not match the replicated source layout."
        )
    cpu_mesh = cpu_mesh_for(dt.device_mesh)
    stride = source_data.stride()
    _validate_dtensor_wrapping(
        source_data,
        cpu_mesh,
        placements,
        dt.shape,
        stride,
        mode="Replicate",
    )
    return _DTensorHost(
        inner=source_adapter,
        inner_state=source_state,
        inner_layout_signature=inner_layout_signature,
        mesh=dt.device_mesh,
        cpu_mesh=cpu_mesh,
        placements=placements,
        shape=dt.shape,
        stride=stride,
    )


def _project_sharded_host(
    source_adapter: TensorAdapter[Any, Any],
    source_state: object,
    dt: Any,  # noqa: ANN401
    placements: tuple[object, ...],
    local_shape: tuple[int, ...],
    offsets: tuple[int, ...],
) -> _DTensorHost:
    """Project a zero-copy view of a host parameter onto a sharded DTensor."""
    if not isinstance(source_adapter, ShardProjectionTensorAdapter):
        raise NotImplementedError(
            f"DTensor shard projection is unsupported for {adapter_name(source_adapter)}."
        )
    source_data = _source_tensor_data(
        source_adapter,
        source_state,
        mode="Shard",
    )
    if tuple(source_data.shape) != tuple(dt.shape):
        raise ValueError("DTensor projection source and target shapes must match.")
    inner_state, inner_layout_signature = source_adapter.project_shard(
        source_state,
        offsets=offsets,
        local_shape=local_shape,
    )
    target_local = local_shard(dt)
    if not accepts_bind_target(
        source_adapter,
        inner_state,
        inner_layout_signature,
        target_local,
    ):
        raise ValueError(
            "DTensor projection local target does not match the projected shard layout."
        )
    cpu_mesh = cpu_mesh_for(dt.device_mesh)
    stride = source_data.stride()
    local_cpu = source_adapter.cpu_param(inner_state).data
    _validate_dtensor_wrapping(
        local_cpu,
        cpu_mesh,
        placements,
        dt.shape,
        stride,
        mode="Shard",
    )
    return _DTensorHost(
        inner=source_adapter,
        inner_state=inner_state,
        inner_layout_signature=inner_layout_signature,
        mesh=dt.device_mesh,
        cpu_mesh=cpu_mesh,
        placements=placements,
        shape=dt.shape,
        stride=stride,
    )


class DTensorAdapter:
    """Movement and shard-local additive merges for tensor-parallel weights."""

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        return is_dtensor(t)

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple:
        # Include the GLOBAL shape/stride (matching layout_signature): two
        # DTensors can alias the same local shard yet carry different global
        # views, and tied-weight dedup reuses the first param's host state —
        # so without these, the second would rebuild with the wrong global
        # metadata.
        dt = require_dtensor(t)
        local = local_shard(dt)
        return (
            "dtensor",
            _select(local).tensor_id(local),
            tuple(dt.shape),
            dt.stride(),
            mesh_signature(dt.device_mesh),
            placements_key(dt.placements),
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple:
        # Include the GLOBAL shape/stride: gpu_param replays them, and for
        # uneven shards the local shard layout alone doesn't pin them (two
        # different global shapes can share a local shape on a rank), so a
        # pool target reused across such blocks would carry the wrong shape.
        # Tag with type(inner) so distinct inner adapters cannot collide.
        dt = require_dtensor(t)
        local = local_shard(dt)
        inner = _select(local)
        inner_type = (
            local._adapter_cls
            if isinstance(local, _ProjectedStructuredShard)
            else type(inner)
        )
        return (
            inner_type,
            inner.layout_signature(local),
            tuple(dt.shape),
            dt.stride(),
            _mesh_target_key(dt.device_mesh),
            placements_key(dt.placements),
        )

    @staticmethod
    def layout_signature_from_host(state: _DTensorHost) -> tuple:
        """Rebuild the target layout signature from captured or projected host state."""
        return (
            type(state.inner),
            state.inner_layout_signature,
            tuple(cast(Any, state.shape)),
            state.stride,
            _mesh_target_key(cast(Any, state.mesh)),
            placements_key(state.placements),
        )

    @staticmethod
    def capture_host(t: torch.Tensor) -> _DTensorHost:
        dt = require_dtensor(t)
        local = local_shard(dt)
        if issubclass(type(local), nn.Parameter):
            # A Parameter subclass may carry state that gpu_param would
            # lose through .data. Check actual inheritance because DTensor
            # can propagate its logical _is_param marker to a plain local tensor.
            raise NotImplementedError(
                f"DTensorAdapter cannot move a local shard that is itself an "
                f"nn.Parameter subclass ({type(local).__name__}); its quant "
                f"state would be stripped. Plain and structured-tensor local "
                f"shards are supported."
            )
        inner = _select(local)
        return _DTensorHost(
            inner=inner,
            inner_state=inner.capture_host(local),
            inner_layout_signature=inner.layout_signature(local),
            mesh=dt.device_mesh,
            cpu_mesh=cpu_mesh_for(dt.device_mesh),
            placements=tuple(dt.placements),
            shape=dt.shape,
            stride=dt.stride(),
        )

    @staticmethod
    def accepts_bind_target(
        state: _DTensorHost,
        target: torch.Tensor,
        *,
        layout_signature: object,
    ) -> bool:
        """Return whether ``target`` matches the DTensor layout and inner bind contract."""
        del layout_signature
        if not DTensorAdapter.matches(target):
            return False
        dt = require_dtensor(target)
        if (
            tuple(dt.shape) != tuple(cast(Any, state.shape))
            or mesh_signature(dt.device_mesh, include_device=False)
            != mesh_signature(cast(Any, state.mesh), include_device=False)
            or placements_key(dt.placements) != placements_key(state.placements)
        ):
            return False
        return accepts_bind_target(
            state.inner,
            state.inner_state,
            state.inner_layout_signature,
            local_shard(dt),
        )

    @staticmethod
    def project_host(
        source_adapter: TensorAdapter[Any, Any],
        source_state: object,
        target: torch.Tensor,
    ) -> _DTensorHost:
        """Project a full host parameter into a zero-copy ``DTensor`` view."""
        dt = require_dtensor(target)
        if isinstance(source_adapter, DTensorAdapter):
            raise NotImplementedError(
                "Cannot project a DTensor source onto another DTensor target."
            )
        require_initialized_mesh(dt.device_mesh)
        coordinate = dt.device_mesh.get_coordinate()
        if coordinate is None:
            raise ValueError(
                "DTensor projection cannot run on a rank outside the target device mesh."
            )
        placements = tuple(dt.placements)
        local_shape, offsets = _local_shape_and_offsets(
            tuple(dt.shape),
            tuple(dt.device_mesh.shape),
            tuple(coordinate),
            placements,
        )
        if all(isinstance(p, Replicate) for p in placements):
            return _project_replicated_host(
                source_adapter,
                source_state,
                dt,
                placements,
            )
        return _project_sharded_host(
            source_adapter,
            source_state,
            dt,
            placements,
            local_shape,
            offsets,
        )

    @staticmethod
    def storage_tensors(state: _DTensorHost) -> tuple[torch.Tensor, ...]:
        return state.inner.storage_tensors(state.inner_state)

    @staticmethod
    def cpu_param(state: _DTensorHost) -> nn.Parameter:
        # The resting weight stays a DTensor (so its adapter/layout matches the
        # store and a deactivated block is still a DTensor), using the cached
        # CPU mesh captured at construction so deactivation neither allocates
        # GPU memory nor requires an active distributed context. The local
        # aliases the inner adapter's host storage.
        with torch.inference_mode(False):
            local = state.inner.cpu_param(state.inner_state).data
            dt = rebuild_dtensor(
                local,
                state.cpu_mesh,
                state.placements,
                state.shape,
                state.stride,
            )
            return nn.Parameter(dt, requires_grad=False)

    @staticmethod
    def alloc_gpu(state: _DTensorHost, device: torch.device) -> _DTensorGpu:
        return _DTensorGpu(inner_gpu=state.inner.alloc_gpu(state.inner_state, device))

    @staticmethod
    def gpu_param(
        host: _DTensorHost,
        gpu_state: _DTensorGpu,
    ) -> nn.Parameter:
        # Reconstruct the local shard on the GPU via the inner adapter, then
        # replay the distributed wrapper. run_check=False: pure local rebuild,
        # never a collective. The local already lives on the mesh device, so
        # from_local aliases it (no copy) — the reused wrapper sees refills.
        with torch.inference_mode(False):
            local = host.inner.gpu_param(host.inner_state, gpu_state.inner_gpu).data
            dt = rebuild_dtensor(local, host.mesh, host.placements, host.shape, host.stride)
            return nn.Parameter(dt, requires_grad=False)

    @staticmethod
    def copy_to_gpu(src: _DTensorHost, dst: _DTensorGpu, *, non_blocking: bool = False) -> None:
        # Pure local-shard DMA; never a collective.
        src.inner.copy_to_gpu(src.inner_state, dst.inner_gpu, non_blocking=non_blocking)

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        """Return the DTensor's global logical shape."""
        return tuple(require_dtensor(t).shape)

    @staticmethod
    def merge_local_shape_and_offsets(
        target: torch.Tensor,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return this rank's local merge shape and global offsets."""
        context = _layout_context(target)
        return context.local_shape, context.offsets

    @staticmethod
    def validate_lora_merge(
        target: torch.Tensor,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Validate both DTensor placement and inner shard merge operation."""
        context = _merge_context(target)
        if isinstance(context.inner, LoRAMergeValidationTensorAdapter):
            context.inner.validate_lora_merge(
                context.local,
                rounding_seed=_localize_rounding_seed(
                    rounding_seed,
                    context.offsets,
                ),
            )

    @staticmethod
    def validate_dense_merge(
        target: torch.Tensor,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Validate placement and the local shard's dense merge operation."""
        context = _dense_merge_context(target)
        if isinstance(context.inner, DenseMergeValidationTensorAdapter):
            context.inner.validate_dense_merge(
                context.local,
                rounding_seed=_localize_rounding_seed(
                    rounding_seed,
                    context.offsets,
                ),
            )

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge local or global LoRA factors into this rank's weight shard.

        The optimized ``LoRATransform`` path supplies factors already localized
        to this rank. Generic callers may still supply factors matching the
        DTensor's global logical shape; those are sliced here as before. The
        resulting contiguous factors are delegated to the inner adapter. No
        collective is needed because the LoRA rank dimension is not sharded.
        """
        context = _merge_context(target)
        local_b, local_a = _local_lora_factors(context, b, a)
        context.inner.merge_lora_(
            context.local,
            local_b,
            local_a,
            alpha,
            beta=beta,
            rounding_seed=_localize_rounding_seed(
                rounding_seed,
                context.offsets,
            ),
        )

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a global or local full-rank update into this rank's shard."""
        context = _dense_merge_context(target)
        context.inner.merge_dense_(
            context.local,
            _local_dense_update(context, update),
            alpha,
            beta=beta,
            rounding_seed=_localize_rounding_seed(
                rounding_seed,
                context.offsets,
            ),
        )

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        # HostParam.compute_dtype and callers inspecting active parameters may
        # pass a DTensor (on CPU or CUDA) or an already unwrapped local shard.
        local = local_shard(t)
        return _select(local).compute_dtype(local)

    @staticmethod
    def cache_bytes(state: _DTensorHost) -> int:
        # Local shard bytes only — the DTensor's global numel would
        # over-account host memory by the world size.
        return state.inner.cache_bytes(state.inner_state)

    @staticmethod
    def gpu_target_bytes(state: _DTensorHost) -> int:
        return _gpu_target_bytes(state.inner, state.inner_state)
