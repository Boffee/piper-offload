"""Per-parameter CPU storage primitive.

Internal to the ``piper_offload`` subpackage. Shared by
:class:`ModelStore`, :class:`ModelOffloader` (whole-model bulk capture), and
:class:`BlockComponent` (per-block streaming). Both consumers reach this through
the same abstraction so the addition of new tensor types only requires writing
a new :class:`TensorAdapter`, not editing the consumers.

:class:`HostParam` pairs one frozen
:class:`nn.Parameter` with the adapter that handles its tensor type
plus the host state that adapter produced. Optional operations
such as in-place updates are exposed through adapter capability methods.
"""

from typing import Any, Self, cast

import torch
from torch import nn

from .dtensor_adapter import DTensorAdapter
from .tensor_adapter_registry import param_representation, select_adapter
from .tensor_adapters import (
    LogicalShapeTensorAdapter,
    TensorAdapter,
    accepts_bind_target,
    clone_host_state,
)


class HostParam:
    """Host storage for one frozen parameter, with GPU-load helpers.

    Construction requires ``requires_grad=False``, picks an adapter via
    :func:`select_adapter` based on the parameter's tensor type, then takes
    ownership of compatible pageable CPU allocations or non-resizable views
    and normalizes other sources into CPU storage when required.
    Model-bound callers create their own
    deactivated-state :class:`nn.Parameter` wrappers with
    :meth:`make_cpu_param`.

    The lifecycle methods (:meth:`make_cpu_param`,
    :meth:`allocate_gpu_storage`, :meth:`make_gpu_param`,
    :meth:`copy_to_gpu`) all dispatch through the adapter. This primitive works
    with the opaque state returned by :meth:`allocate_gpu_storage`; model-bound
    callers should normally use :class:`~piper_offload.host_module.HostModuleStore`
    and :class:`~piper_offload.host_module.HostModuleInstance`, which wrap active
    storage in target objects.

    Low-peak host construction behavior: compatible complete CPU allocations
    and views into non-resizable storage are retained directly, preserving file
    mappings without a copy. For plain
    ``torch.Tensor`` parameters, construction immediately repoints the source
    ``Parameter.data`` at the captured backing.
    This promptly releases replaced GPU or incompatible CPU storage before the
    owning store finishes. It also means host construction is not rollback-safe
    after copying has started: if a later parameter fails to capture, recovery
    of the partially constructed store/model is unsupported. Drop those
    references and rebuild from a fresh model instance. Tensor subclasses skip
    this optimization because ``.data =`` can drop wrapper state.

    A frozen plain floating-point meta parameter retains its shape, dtype, and
    stride without host backing or a cache reservation. It remains meta unless a
    merge-mode parameter value allocates and fills active device storage.
    Low-rank factors cannot materialize it.
    """

    __slots__ = (
        "_logical_shape",
        "_shape",
        "_target_layout",
        "adapter",
        "host_state",
    )

    def __init__(self, param: nn.Parameter) -> None:
        if param.requires_grad:
            raise ValueError(
                "Piper Offload is inference-only and requires frozen parameters "
                "(requires_grad=False)."
            )
        # Select the tensor that carries the parameter's representation.
        representation = param_representation(param)
        is_meta = representation.is_meta
        if is_meta:
            if type(representation) is not torch.Tensor:
                raise NotImplementedError(
                    "Meta parameters require a plain torch.Tensor representation; "
                    f"got {type(representation).__name__}."
                )
            if representation.layout is not torch.strided:
                raise ValueError(
                    "Meta parameters must use a strided layout; "
                    f"got {representation.layout}."
                )
            if not representation.is_floating_point():
                raise ValueError(
                    "Meta parameters must be floating-point; "
                    f"got {representation.dtype}."
                )
        adapter: TensorAdapter[Any, Any] = select_adapter(representation)
        # A meta tensor is the complete resting representation: it records
        # shape, dtype, and stride without owning physical host bytes.
        host_state = (
            representation.detach() if is_meta else adapter.capture_host(representation)
        )
        self._init_from_host_state(
            representation,
            adapter=adapter,
            host_state=host_state,
        )
        # Low-peak host construction optimization: release the original
        # source storage by repointing the source Parameter at the selected
        # host backing immediately. The assignment is an intentional
        # mutation of the caller's model before the owning store has finished
        # construction; see the class docstring for failure semantics.
        # Only safe for plain tensors; subclass wrappers (and Parameter
        # subclasses) can lose metadata or ignore .data assignment.
        if not is_meta and type(representation) is torch.Tensor:
            param.data = self.make_cpu_param().data

    @classmethod
    def project_dtensor(cls, source: HostParam, target: nn.Parameter) -> Self:
        """Project a full host source into a standard DTensor target layout.

        The target contributes only its global shape, mesh, placements, and
        rank-local layout. Its local storage is not retained. The physical
        source remains the captured host allocation, including any file
        mapping retained during capture.

        Projection supports frozen plain tensors with one ``Shard(0)``
        placement, or rank-two tensors with ``Shard(1)``. The local CPU shard
        is a view into the full source allocation, so construction allocates no
        rank-local host tensor.
        """
        if source.is_meta:
            raise ValueError("DTensor projection source must own physical host storage.")
        if target.requires_grad:
            raise ValueError(
                "DTensor projection supports frozen inference parameters only (requires_grad=False)."
            )

        representation = param_representation(target)
        adapter = select_adapter(representation)
        if not isinstance(adapter, DTensorAdapter):
            raise TypeError(
                "DTensor projection target must use a standard torch DTensor "
                f"representation; got {type(representation).__name__}."
            )
        host_state = adapter.project_host(
            source.adapter,
            source.host_state,
            representation,
        )
        projected = cls.__new__(cls)
        projected._init_from_host_state(
            representation,
            adapter=adapter,
            host_state=host_state,
        )
        return projected

    def _init_from_host_state(
        self,
        representation: torch.Tensor,
        *,
        adapter: TensorAdapter[Any, Any],
        host_state: object,
    ) -> None:
        """Install representation metadata and an already-captured host state."""
        self._shape = torch.Size(representation.shape)
        self.adapter = adapter
        self.host_state = host_state
        self._logical_shape = (
            adapter.logical_shape(representation)
            if isinstance(adapter, LogicalShapeTensorAdapter)
            else tuple(representation.shape)
        )
        self._target_layout = self._target_layout_from_adapter(
            adapter,
            representation,
            unresolved_meta=self.is_meta,
        )

    @staticmethod
    def _target_layout_from_adapter(
        adapter: TensorAdapter[Any, Any],
        tensor: torch.Tensor,
        *,
        unresolved_meta: bool | None = None,
    ) -> tuple[object, object]:
        signature: object = adapter.layout_signature(tensor)
        if unresolved_meta is None:
            unresolved_meta = tensor.is_meta
        if unresolved_meta:
            # Physical regular tensors are normalized to contiguous backing,
            # but meta parameters retain their declared layout while resting.
            # Keep differing unresolved meta layouts out of the same
            # streaming/rolling target pool. A ParameterValue load replaces
            # this signature with its physical source layout.
            signature = (
                "meta",
                signature,
                tensor.stride(),
                tensor.storage_offset(),
            )
        return (type(adapter), signature)

    @classmethod
    def rolling_slot_metadata_for(
        cls,
        param: nn.Parameter,
    ) -> tuple[type, torch.Size, tuple[object, object]]:
        """Adapter type, representation shape, and target layout before capture.

        Keeps :class:`HostParam` as the single owner of adapter dispatch and
        representation shape for pre-capture rolling validation.
        """
        representation = param_representation(param)
        adapter = select_adapter(representation)
        return (
            type(adapter),
            torch.Size(representation.shape),
            cls._target_layout_from_adapter(adapter, representation),
        )

    @classmethod
    def target_layout_for(cls, param: nn.Parameter) -> tuple[object, object]:
        """Opaque target-compatibility layout for ``param``.

        Used by pre-capture validation paths that must fail before
        :class:`HostParam` mutates the source parameter. Callers should
        compare the returned value for equality only.
        """
        return cls.rolling_slot_metadata_for(param)[2]

    def accepts_bind_target(self, param: nn.Parameter) -> bool:
        """Return whether ``param`` is a compatible bind target for this host store.

        Unrecognized or incompatible tensor types return ``False`` cleanly; malformed
        instances of an adapter's own tensor type (such as an unsupported transposed
        layout) may still raise during layout validation.
        """
        if param.requires_grad:
            return False
        representation = param_representation(param)
        if self.is_meta:
            host_representation = cast(torch.Tensor, self.host_state)
            return bool(
                type(representation) is torch.Tensor
                and representation.is_meta
                and representation.layout is torch.strided
                and tuple(representation.shape) == tuple(self.shape)
                and representation.dtype is host_representation.dtype
                and representation.stride() == host_representation.stride()
                and representation.storage_offset()
                == host_representation.storage_offset()
            )
        return accepts_bind_target(
            self.adapter,
            self.host_state,
            self._target_layout[1],
            representation,
        )

    @property
    def target_layout(self) -> tuple[object, object]:
        """Opaque target-compatibility layout for this host backing."""
        return self._target_layout

    @property
    def shape(self) -> torch.Size:
        """Shape of the source parameter representation at capture time."""
        return self._shape

    @property
    def logical_shape(self) -> tuple[int, ...]:
        """Dense logical shape represented by this host parameter."""
        return self._logical_shape

    @property
    def is_meta(self) -> bool:
        """Whether the resting parameter representation is meta."""
        return type(self.host_state) is torch.Tensor and self.host_state.is_meta

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """Enumerate existing CPU backing tensors without rebuilding wrappers.

        Includes tensor-valued metadata. Views and shared allocations are
        preserved; consumers own storage deduplication. Meta parameters have
        no physical backing and return an empty tuple.
        """
        if self.is_meta:
            return ()
        return self.adapter.storage_tensors(self.host_state)

    def make_cpu_param(self) -> nn.Parameter:
        """Build a frozen CPU :class:`nn.Parameter` wrapper over this host state.

        Each model-bound binding owns its own wrapper object, while the
        underlying host storage remains shared through this
        :class:`HostParam`.
        """
        if self.is_meta:
            return nn.Parameter(
                cast(torch.Tensor, self.host_state),
                requires_grad=False,
            )
        return self.adapter.cpu_param(self.host_state)

    def allocate_gpu_storage(self, device: torch.device) -> object:
        """Allocate empty GPU storage mirroring this host parameter's layout.

        Returns an opaque adapter-specific handle; pass it back to
        :meth:`make_gpu_param` and :meth:`copy_to_gpu`.
        """
        if self.is_meta:
            source = cast(torch.Tensor, self.host_state)
            return torch.empty_strided(
                tuple(source.shape),
                source.stride(),
                dtype=source.dtype,
                device=device,
            )
        return self.adapter.alloc_gpu(self.host_state, device)

    def make_gpu_param(self, gpu_state: object) -> nn.Parameter:
        """Build the frozen GPU-side :class:`nn.Parameter` for this host parameter.

        The adapter receives the paired host state to reconstruct
        structured tensor wrappers.
        """
        if self.is_meta:
            return nn.Parameter(cast(torch.Tensor, gpu_state), requires_grad=False)
        return self.adapter.gpu_param(self.host_state, gpu_state)

    def copy_to_gpu(self, gpu_state: object, *, non_blocking: bool = False) -> None:
        """Bulk DMA host bytes into pre-allocated GPU storage."""
        if self.is_meta:
            # The activation-scoped parameter-value hook populates this storage
            # directly. There are deliberately no zero bytes to transfer.
            return
        self.adapter.copy_to_gpu(self.host_state, gpu_state, non_blocking=non_blocking)

    def materialize(
        self,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> nn.Parameter:
        """Build and populate a one-shot representation on ``device``.

        Unlike calling ``.to(device)`` on :meth:`make_cpu_param`, this runs
        the complete adapter lifecycle. That distinction matters for outer
        tensor wrappers such as ``DTensor``: the local shard must move and the
        wrapper must then be rebuilt on its original device mesh.

        CPU materialization is a zero-copy wrapper over the host backing.
        CUDA materialization returns a parameter wrapping fresh adapter
        storage.
        """
        if self.is_meta:
            raise RuntimeError(
                "A meta parameter cannot be materialized without an "
                "active parameter value."
            )
        if device.type == "cpu":
            return self.make_cpu_param()
        if device.type != "cuda":
            raise ValueError(
                "HostParam materialization supports CPU and CUDA devices; "
                f"got {device}."
            )

        state = self.allocate_gpu_storage(device)
        param = self.make_gpu_param(state)
        self.copy_to_gpu(state, non_blocking=non_blocking)
        return param

    def clone_cpu_param(self) -> nn.Parameter:
        """Return an independent CPU clone of this representation.

        Unlike CPU :meth:`materialize`, this never aliases the immutable host
        backing. It is used when a parameter value becomes permanent model
        state and may subsequently be mutated by strength scaling.
        """
        if self.is_meta:
            raise RuntimeError(
                "A meta parameter cannot be cloned without an active "
                "parameter value."
            )
        representation = param_representation(self.make_cpu_param())
        cloned_state = clone_host_state(self.adapter, representation, self.host_state)
        return self.adapter.cpu_param(cloned_state)

    @property
    def compute_dtype(self) -> torch.dtype:
        """Logical compute dtype reported by this host parameter's adapter."""
        return self.adapter.compute_dtype(
            param_representation(self.make_cpu_param())
        )

    @property
    def cache_bytes(self) -> int:
        """Logical host bytes reported by this parameter's adapter."""
        if self.is_meta:
            return 0
        return self.adapter.cache_bytes(self.host_state)
