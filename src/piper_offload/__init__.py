"""Host caching and device activation for PyTorch models and adapters.

- :class:`ModelCache` uses :class:`ModelSpec` and :class:`AdapterSpec` to cache
  resources and coordinate model activation.
- :class:`ModelStore` and :class:`ModelStoreSpec` hold reusable flat host
  backing that binds to compatible models with :class:`OffloadConfig`.
- :class:`ModelOffloader` provides direct control over one bound model's
  lifecycle, including whole-model uploads, block streaming, and
  :meth:`~ModelOffloader.no_record_trace` for speculative passes.
- :class:`VramBudget`, :class:`VramReservation`, :class:`VramBudgetExhaustedError`,
  and :class:`DeviceTransferContext` coordinate shared per-device CUDA VRAM
  budgets and reusable async transfer streams/workers across models and pools.
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
from .adapter import Adapter, AdapterMode, AdapterTarget
from .block_compile import BlockCompileConfig
from .checkpoint import CheckpointError, FileSlice, MappedCheckpoint, file_slice
from .device_transfer import DeviceTransferContext
from .gguf_parameter import GgufParameter
from .lora import LoRAFactor, LoRATransform, ScaledLoRAFactor
from .merge import merge_adapter
from .model_cache import ModelCache
from .model_offloader import ModelOffloader, ModelRuntimeFailedError, ModelRuntimeInUseError
from .model_store import ModelStore
from .mps_weights import MpsWeights
from .offload_config import OffloadConfig, auto_offload_paths
from .parameter_delta import ParameterDelta, ParameterDeltaTransform, ScaledParameterDelta
from .parameter_transform import ParameterTransform
from .parameter_value import (
    ParameterValue,
    ParameterValueTransform,
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
    "AdapterMode",
    "AdapterSpec",
    "AdapterTarget",
    "BlockCompileConfig",
    "CacheError",
    "CheckpointError",
    "DeviceTransferContext",
    "DuplicateResourceKeyError",
    "EvictionCandidate",
    "EvictionContext",
    "EvictionPolicy",
    "EvictionPolicyError",
    "FileSlice",
    "GgufParameter",
    "HostRegistrationError",
    "LRUEvictionPolicy",
    "LoRAFactor",
    "LoRATransform",
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
    "ParameterDeltaTransform",
    "ParameterTransform",
    "ParameterValue",
    "ParameterValueTransform",
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
    "TensorAdapter",
    "TrellisParameter",
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
