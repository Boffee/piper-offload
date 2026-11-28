"""Host caching and device activation for PyTorch models and adapters.

- :class:`ModelCache` uses :class:`ModelSpec` and :class:`AdapterSpec` to cache
  resources and coordinate model activation.
- :class:`ModelStore` and :class:`ModelStoreSpec` hold reusable flat host
  backing that binds to compatible models with :class:`OffloadConfig`.
- :class:`ModelOffloader` provides direct control over one bound model's
  lifecycle, including whole-model uploads, block streaming, and
  :meth:`~ModelOffloader.no_record_trace` for speculative passes.
- :class:`VramBudget`, :class:`VramReservation`, :class:`VramBudgetExhaustedError`,
  :class:`Uint8DualArena`, :class:`SlotSchema`, :class:`ExpertSlabPool`,
  :class:`KvChunkPool`, and :class:`DeviceTransferContext` coordinate shared
  per-device CUDA VRAM budgets, reusable async transfer streams, and global
  expert/KV slab residency across models and pools.
- :class:`ResourceCache` and :class:`ObjectSpec` support other cached resources.
- :class:`Adapter` holds reusable model updates; :func:`merge_adapter` applies
  them permanently.
- :class:`MappedCheckpoint` and :data:`host_pin_manager` provide checkpoint
  backing and host transfer controls.
- :class:`TensorAdapter` and :func:`register_adapter` integrate tensor formats.
- :class:`MpsWeights` materializes frozen CPU models on Apple MPS.

Each API's docstring owns its contract. The
`README <https://github.com/Boffee/piper-offload/blob/main/README.md>`_
provides installation, a working example, and links to the caller guides.
The `development guide
<https://github.com/Boffee/piper-offload/blob/main/docs/development.md>`_
maps implementation ownership and extension points.
"""

from ._host_registration import HostRegistrationError
from .adapter import Adapter, AdapterTarget
from .block_compile import BlockCompileConfig
from .checkpoint import CheckpointError, FileSlice, MappedCheckpoint, file_slice
from .device_transfer import DeviceTransferContext
from .dual_arena import (
    ArenaCapacityError,
    SlotSchema,
    Uint8DualArena,
)
from .expert_slab_pool import ExpertPoolStats, ExpertSlabPool, ExpertWave
from .gguf_parameter import GgufParameter
from .kv_chunk_pool import KvBatch, KvChunkLocation, KvChunkPool, KvPoolStats
from .lora import LoRAFactor, ScaledLoRAFactor
from .merge import merge_adapter
from .model_cache import ModelCache
from .model_offloader import ModelOffloader, ModelRuntimeFailedError, ModelRuntimeInUseError
from .model_store import ModelStore
from .mps_weights import MpsWeights
from .offload_config import OffloadConfig, auto_offload_paths
from .parameter_delta import ParameterDelta, ScaledParameterDelta
from .parameter_value import (
    ParameterValue,
    ScaledParameterValue,
)
from .pin_manager import PinLease, PinManager, PinStats, host_pin_manager
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
from .resource_specs import AdapterSpec, ModelSpec, ModelStoreSpec, ObjectSpec
from .seeding import derive_seed
from .tensor_adapter_registry import register_adapter, requires_activation
from .tensor_adapters import (
    TensorAdapter,
    transfer_,
)
from .trellis_parameter import TrellisParameter
from .vram_budget import (
    VramBudget,
    VramBudgetExhaustedError,
    VramReservation,
)

__all__ = [
    "Adapter",
    "AdapterSpec",
    "AdapterTarget",
    "ArenaCapacityError",
    "BlockCompileConfig",
    "CacheError",
    "CheckpointError",
    "DeviceTransferContext",
    "DuplicateResourceKeyError",
    "EvictionCandidate",
    "EvictionContext",
    "EvictionPolicy",
    "EvictionPolicyError",
    "ExpertPoolStats",
    "ExpertSlabPool",
    "ExpertWave",
    "FileSlice",
    "GgufParameter",
    "HostRegistrationError",
    "KvBatch",
    "KvChunkLocation",
    "KvChunkPool",
    "KvPoolStats",
    "LRUEvictionPolicy",
    "LoRAFactor",
    "MappedCheckpoint",
    "ModelCache",
    "ModelOffloader",
    "ModelRuntimeFailedError",
    "ModelRuntimeInUseError",
    "ModelSpec",
    "ModelStore",
    "ModelStoreSpec",
    "MpsWeights",
    "ObjectSpec",
    "OffloadConfig",
    "ParameterDelta",
    "ParameterValue",
    "PinLease",
    "PinManager",
    "PinStats",
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
    "SlotSchema",
    "TensorAdapter",
    "TrellisParameter",
    "Uint8DualArena",
    "VramBudget",
    "VramBudgetExhaustedError",
    "VramReservation",
    "auto_offload_paths",
    "derive_seed",
    "file_slice",
    "host_pin_manager",
    "merge_adapter",
    "register_adapter",
    "requires_activation",
    "transfer_",
]
