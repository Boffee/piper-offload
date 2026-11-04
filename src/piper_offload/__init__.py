"""Host caching and device activation for PyTorch models and adapters.

- :class:`ModelCache` uses :class:`ModelSpec` and :class:`AdapterSpec` to cache
  resources and coordinate model activation.
- :class:`ModelOffloader` provides direct control over one model's lifecycle,
  including whole-model uploads and block streaming.
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
from .block_component import BlockComponent, BlockComponentStore
from .block_mode import BlockMode
from .checkpoint import CheckpointError, FileSlice, MappedCheckpoint, file_slice
from .gguf_parameter import GgufParameter
from .host_component import HostComponent, HostComponentStore
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
from .resource_specs import AdapterSpec, ModelSpec, ObjectSpec
from .seeding import derive_seed
from .tensor_adapter_registry import register_adapter, requires_activation
from .tensor_adapters import (
    TensorAdapter,
    transfer_,
)

__all__ = [
    "Adapter",
    "AdapterMode",
    "AdapterSpec",
    "AdapterTarget",
    "BlockCompileConfig",
    "BlockComponent",
    "BlockComponentStore",
    "BlockMode",
    "CacheError",
    "CheckpointError",
    "DuplicateResourceKeyError",
    "EvictionCandidate",
    "EvictionContext",
    "EvictionPolicy",
    "EvictionPolicyError",
    "FileSlice",
    "GgufParameter",
    "HostComponent",
    "HostComponentStore",
    "HostRegistrationError",
    "LRUEvictionPolicy",
    "LoRAFactor",
    "LoRATransform",
    "MappedCheckpoint",
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
    "derive_seed",
    "file_slice",
    "host_pin_manager",
    "merge_adapter",
    "register_adapter",
    "requires_activation",
    "transfer_",
]
