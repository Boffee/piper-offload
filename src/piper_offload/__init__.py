"""GPU memory management utilities -- model-agnostic, torch-only.

High-level API:

- :class:`ResourceCache` accepts structural :class:`ResourceSpec` implementations
  and leases reusable model, adapter, and object stores under a host-memory
  budget. :class:`ModelCache` specializes it with model-aware use: it composes
  a leased :class:`ModelSpec` with optional :class:`AdapterSpec` resources and
  activates the cached :class:`ModelOffloader`.
  :class:`ObjectSpec` caches general Python objects (tokenizers,
  processors, configs) in the same registry; by default they are
  charged zero bytes and live until explicitly evicted.

Lower-level resource bindings:

- :class:`ModelOffloader` -- whole-model host-RAM bulk cache when
  created by ``ModelOffloader.from_module(model)``, or configurable block
  residency when constructed with ``block_paths`` or
  ``transient_block_paths``. Block modes support optional adapter merge,
  opt-in forward-only block compilation for CUDA inference,
  path-selected transient pool lifetimes, trainable-parameter support,
  CUDA prefetch on a secondary stream, and
  activation checkpointing through autograd backward when block compilation
  is disabled. By default,
  trainable params are managed by
  :class:`PinnedComponent` and stay GPU-resident while active; set
  ``include_block_trainables=True`` to stream in-block trainable weights
  and materialize them only around ``optimizer.step()``. That step runs on
  the GPU via the ``optimizer_step()`` context; calling ``optimizer.step()``
  after deactivation instead runs the optimizer on CPU over the host-backed
  weights (state stays on host). On deactivation every managed trainable's
  ``.grad`` follows its ``.data`` to CPU, so the context-free CPU
  step works for both streamed and non-streamed trainables — keep such
  trainables in fp32.

- :class:`MpsWeights` -- whole-model CPU->MPS materializer. Use for
  frozen models that should become MPS-resident without retaining a
  separate CPU cache. Construction copies one managed tensor at a time
  and immediately replaces its module registry entry to keep peak host memory
  close to one model plus the current tensor.

The CUDA-oriented :class:`ModelOffloader` shares the underlying
per-parameter host storage from
:class:`~piper_offload.pinned_param.PinnedParam` (clone + pin
+ optional quanto ``WeightQBytesTensor`` decomposition, bitsandbytes
4-bit ``Params4bit`` (NF4/FP4) and 8-bit ``Int8Params`` (LLM.int8)
decomposition, GGUF packed weights, Piper ConvRot INT8 / NVFP4, TorchAO NVFP4 / MX
(MXFP8, MXFP4) / dynamic or calibrated-static scaled-FP8 / INT8 / INT4
(tile-packed) packed weights, and
tensor-parallel ``DTensor`` weights wrapping any of the above).

:class:`ModelOffloader` and :class:`MpsWeights` are cached resources that also
implement the :class:`ResourceBinding` Protocol. Each owns exactly one model
runtime and is reused sequentially.

``ModelOffloader.from_module(..., host_backing="adopt")`` adopts frozen
model state already in CPU RAM without copying it. Anonymous pageable and
file-backed/mmap tensors therefore share one path and retain their original
storage. Copies go directly into the same GPU targets; CUDA performs any
implicit staging. Unsupported adoption raises rather than materializing a
hidden copy. Capture completes for the entire adopted store before binding
changes any module registry, so adoption failures leave the supplied model
untouched. Adopted tensors and writable mmap contents must remain immutable for
the offloader's lifetime. The default ``"pinned"`` policy preserves the
full-bandwidth asynchronous path. Package resources make ``cache_bytes`` final
during construction.
``activate(device)`` then makes the resource usable on the requested device. For
:class:`ModelOffloader`, ``deactivate()`` returns managed tensors to
their configured host backing. For :class:`MpsWeights`,
construction has already materialized the model on MPS, so
``activate('mps')`` and ``deactivate()`` are lifecycle-only.

Pinned host-store construction intentionally optimizes peak host memory. For
plain ``torch.Tensor`` parameters, it may immediately repoint the source
``Parameter.data`` at each pinned clone so the original source storage can be
freed before all buffers finish. This avoids a temporary 2x host-memory peak
for CPU-origin models and promptly frees GPU storage for CUDA-origin models.
Adopted inference instead retains the existing CPU allocation. If pinned
construction raises after pinning has started, recovery of the partially
constructed resource/model is unsupported; drop those references and rebuild
from a fresh model instance.

:class:`ModelOffloader` composes:
  1. A resident :class:`PinnedComponent` for non-streamed state, including
     trainables skipped by block streaming.
  2. One :class:`PinnedComponent` per stateful path in ``transient_paths``.
  3. One :class:`BlockComponent` per path in ``block_paths`` or
     ``transient_block_paths`` when block residency is configured.
     ``block_mode`` selects resident, whole-block streaming, rolling, or
     automatic per-group rolling with streaming fallback independently of the
     paths' persistent or transient lifetime.

Optional adapter application is requested directly on
:meth:`ModelOffloader.activate` and resolved by installing post-copy hooks for
managed parameter targets. Unknown targets raise during activation unless the
adapter resource explicitly allows partial targets, in which case application
uses the intersection of adapter targets and model parameters. The
hooks run immediately after the owning component copies a base weight
from host storage to GPU, so block-streamed and non-block weights
use the same merge path. Merge eligibility is owned by the selected tensor
adapter: physical plain floating-point tensors support combined low-rank and
full-rank additive deltas; structured quantized wrappers can opt into staged
LoRA merge. Dense deltas for quantized wrappers are not yet supported. Frozen
plain floating-point meta tensors can instead be populated by parameter
values. Otherwise, use routed LoRA when the module exposes a compatible
logical Linear weight shape and compute dtype.

:meth:`Adapter.from_state_dict` interprets the exact ``.delta.weight`` and
``.delta.bias`` suffixes as full-rank additive updates and every other
non-LoRA entry as the complete value for an exact-name parameter. Parameter
values are merge-only and populate storage-free frozen plain floating-point
meta parameters; parameter deltas require existing physical parameters.

:class:`Adapter` owns immutable parameter-delta and parameter-value storage,
pinned by default or strictly adopted from existing CPU backing. Compatible
consumers read that backing directly and may overlap; routed hooks stage their
own per-forward device copies.

Downstream tensor subclasses can participate in pinning and movement without
adding format-specific dependencies here: implement the public
:class:`TensorAdapter` contract and register it during application startup with
:func:`register_adapter`. Registered adapters are used for both movement and
tied-storage identity. To additionally support ``host_backing="adopt"``,
implement :class:`AdoptableTensorAdapter`; its ``adopt_host()`` method returns
adapter state that aliases the retained source storage.

:class:`ResourceCache` manages cached backing stores with policy-driven
eviction, reference-counted leases, and transactional admission.
:class:`ModelCache` owns dependency leasing, adapter attachment, and device
activation. Each model offloader rejects overlapping use. Custom
:class:`EvictionPolicy`
implementations can replace the default LRU behavior. See its docstring
for design notes.

Compatibility
-------------
- **``torch.compile`` support is narrow.** Only declared block forwards
  configured through :class:`BlockCompileConfig` are supported, and only for
  CUDA inference. Ordinary block groups may be streamed or resident. External
  whole-model compilation, modules outside declared block groups, routed-LoRA
  activations, and compiled training remain unsupported.
  Experimental rolling compilation additionally requires frozen homogeneous
  blocks using a reviewed dense or quantized adapter, a full graph, and one
  shared target. Auto mode streams block groups that do not meet those static
  requirements.
- **Wrap before DDP/FSDP**, not after.
- **Coarse cache concurrency.** :class:`ResourceCache` serializes cache
  metadata and lease operations and releases its lock while caller code
  holds a lease. Model cache entries support one active use at a time; adapter
  backing may be shared.
"""

from .adapter import Adapter, AdapterMode, AdapterTarget
from .block_compile import BlockCompileConfig
from .block_component import BlockComponent, BlockComponentStore
from .block_mode import BlockMode
from .gguf_adapter import GGUFWeight
from .host_backing import HostBacking
from .lora import LoRAFactor, LoRATransform, ScaledLoRAFactor
from .merge import merge_adapter
from .model_cache import ModelCache
from .model_offloader import ModelOffloader, ModelRuntimeInUseError
from .mps_weights import MpsWeights
from .parameter_delta import ParameterDelta, ParameterDeltaTransform, ScaledParameterDelta
from .parameter_transform import ParameterTransform
from .parameter_value import (
    ParameterValue,
    ParameterValueTransform,
    ScaledParameterValue,
)
from .pinned_component import PinnedComponent, PinnedComponentStore
from .protocols import (
    ResourceBinding,
    ResourceSpec,
    ResourceStore,
)
from .resource_cache import (
    CacheError,
    DuplicateResourceKeyError,
    EvictionCandidate,
    EvictionContext,
    EvictionPolicy,
    EvictionPolicyError,
    LRUEvictionPolicy,
    ResourceCache,
    ResourceCachedError,
    ResourceInfo,
    ResourceLeasedError,
    ResourceNotRegisteredError,
    ResourceTooLargeError,
)
from .resource_specs import AdapterSpec, ModelSpec, ObjectSpec
from .seeding import derive_seed
from .tensor_adapter_registry import register_adapter
from .tensor_adapters import (
    AdoptableTensorAdapter,
    TensorAdapter,
)

__all__ = [
    "Adapter",
    "AdapterMode",
    "AdapterSpec",
    "AdapterTarget",
    "AdoptableTensorAdapter",
    "BlockCompileConfig",
    "BlockComponent",
    "BlockComponentStore",
    "BlockMode",
    "CacheError",
    "DuplicateResourceKeyError",
    "EvictionCandidate",
    "EvictionContext",
    "EvictionPolicy",
    "EvictionPolicyError",
    "GGUFWeight",
    "HostBacking",
    "LRUEvictionPolicy",
    "LoRAFactor",
    "LoRATransform",
    "ModelCache",
    "ModelOffloader",
    "ModelRuntimeInUseError",
    "ModelSpec",
    "MpsWeights",
    "ObjectSpec",
    "ParameterDelta",
    "ParameterDeltaTransform",
    "ParameterTransform",
    "ParameterValue",
    "ParameterValueTransform",
    "PinnedComponent",
    "PinnedComponentStore",
    "ResourceBinding",
    "ResourceCache",
    "ResourceCachedError",
    "ResourceInfo",
    "ResourceLeasedError",
    "ResourceNotRegisteredError",
    "ResourceSpec",
    "ResourceStore",
    "ResourceTooLargeError",
    "ScaledLoRAFactor",
    "ScaledParameterDelta",
    "ScaledParameterValue",
    "TensorAdapter",
    "derive_seed",
    "merge_adapter",
    "register_adapter",
]
