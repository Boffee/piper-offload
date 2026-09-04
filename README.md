# Piper Offload

A model-agnostic GPU/CPU memory manager for PyTorch. It caches reusable
model and adapter resources, preserves compatible file-backed CPU mappings,
and swaps independent models in and out of GPU memory.

Piper Offload is self-contained and library-friendly: it has no required
dependency beyond `torch`. Optional integrations support `bitsandbytes`,
`optimum.quanto`, `gguf`, `piper-kernels`, and `torchao` quantized models.

Requires Python 3.14 or newer and PyTorch 2.13.

## Installation

Install the base package from PyPI:

```bash
pip install piper-offload
```

Optional integrations are available individually through the `bnb`, `torchao`,
`gguf`, `quanto`, and `convrot` extras. Triton acceleration is a separate,
composable extra, while `all` includes every integration plus Triton:

```bash
pip install "piper-offload[all]"
```

The `triton` extra selects upstream `triton` on Linux and `triton-windows` on
64-bit Windows. Combine it with any individual quantization extra whose
optimized kernels you want; without it, those integrations retain their
portable fallback paths. The `all` extra includes this acceleration runtime.
The same installed Triton runtime enables Piper Kernels' ConvRot backend when
`convrot` is also selected. Windows execution requires Windows 10 or 11, a
supported NVIDIA GPU with a current driver, and the Visual C++ Redistributable
for Visual Studio 2015-2022; a separate CUDA toolkit or Visual Studio install
is not required.

## What's in here

| Module | Role |
|---|---|
| `resource_cache.py` | `ResourceCache`, eviction policy, cache metadata, and cache errors |
| `pin_manager.py` | `PinManager`, `PinLease`, `PinStats`, and the process-wide `host_pin_manager` for budgeted host registration |
| `model_cache.py` | `ModelCache` — model-aware `ResourceCache` with activation and adapter coordination |
| `resource_specs.py` | `ModelSpec`, `AdapterSpec`, `ObjectSpec` — standard frozen resource specifications |
| `protocols.py` | `ResourceSpec`, `ResourceStore`, `ResourceBinding` plug-in contracts |
| `block_compile.py` | `BlockCompileConfig` — opt-in Inductor policy for declared block forwards |
| `model_offloader.py` | `ModelOffloader` — cached single-model runtime for whole-model bulk CPU↔GPU or streamed block offload |
| `host_component.py` | `HostComponentStore`, `HostComponent` — lower-level reusable host backing storage plus lifecycle-only host component used by `ModelOffloader` |
| `block_component.py` | `BlockComponentStore`, `BlockComponent` — lower-level streamed backing storage plus per-block-list streaming component |
| `adapter.py` | `Adapter`, `AdapterTarget` — cached resources and their delta-or-value target union |
| `lora.py` | `LoRAFactor`, `ScaledLoRAFactor`, `LoRATransform` — low-rank data, merge, and routed execution |
| `parameter_delta.py` | `ParameterDelta`, `ScaledParameterDelta`, `ParameterDeltaTransform` — combined low-rank/full-rank additive updates |
| `parameter_value.py` | `ParameterValue`, `ScaledParameterValue`, `ParameterValueTransform` — meta-parameter data and materialization |
| `parameter_transform.py` | `ParameterTransform` — shared parameter-update protocol |
| `merge.py` | `merge_adapter()` — permanent in-place adapter application to base weights |
| `seeding.py` | `derive_seed()` — canonical stable unsigned 64-bit seed derivation from typed identity parts |
| `host_param.py` | `HostParam` — per-parameter capture primitive (handles plain tensors, quanto, GGUF, bitsandbytes, Piper ConvRot INT8 / NVFP4, DTensor, and TorchAO dynamic/static scaled-FP8 / INT8 / MX (MXFP8, MXFP4) / NVFP4 / INT4 tile-packed via adapters; see [Quantized weight support](#quantized-weight-support)) |
| `host_module.py` | Internal name-keyed host module storage plus concrete module bindings |
| `tensor_adapters.py`, `quanto_adapter.py`, `gguf_adapter.py`, `piper_convrot_int8_adapter.py`, `piper_convrot_nvfp4_adapter.py`, `nvfp4_adapter.py`, `mx_adapter.py`, `float8_adapter.py`, `static_float8_adapter.py`, `int8_adapter.py`, `int4_tile_adapter.py`, `dtensor_adapter.py` | Tensor adapter contracts/implementations and optional optimum-quanto / GGUF / Piper ConvRot / torchao / DTensor support |
| `torchao_structured_adapter.py` | Internal: shared `TorchaoStructuredAdapter` base for the TorchAO subclass adapters (scaled-FP8 / INT8 / MX / NVFP4 / INT4 tile-packed) — common capture/move/identity mechanics + per-format hooks; capabilities beyond inference movement (CPU round-trip, dequant/requant conversion, copy, and staged factorized/dense merge) are opted into per subclass |
| `dtensor_adapter.py` | Internal: `DTensorAdapter` for tensor-parallel `DTensor` weights — composes with other adapters by delegating local-shard movement and factorized/dense merge to the registry, then replaying the `(mesh, placements)` wrapper; frozen-inference scope (see `_dtensor.py`) |
| `tensor_adapter_registry.py` | Public external-adapter registration plus adapter dispatch and tensor-identity helpers |
| `module_names.py` | Internal name traversal and mutation helpers |
| `_quanto.py` | Internal: optimum-quanto optional-import + layout validation; consumed by `quanto_adapter.py` and `merge.py` |
| `_piper_convrot_int8.py` | Internal: Piper ConvRot INT8 optional-import, public-layout validation, and wrapper reconstruction; consumed by `piper_convrot_int8_adapter.py` |
| `_piper_convrot_nvfp4.py` | Internal: Piper ConvRot NVFP4 optional-import, public-layout and merge-capability validation, and wrapper reconstruction; consumed by `piper_convrot_nvfp4_adapter.py` |
| `_torchao_nvfp4.py` | Internal: TorchAO NVFP4 optional-import + layout validation and dequant/requant; consumed by `nvfp4_adapter.py` |
| `_torchao_mx.py` | Internal: TorchAO MX (MXFP8 / MXFP4) optional-import + layout validation, supported-dtype gate, and dequant/requant; consumed by `mx_adapter.py` |
| `_torchao_float8.py`, `_torchao_static_float8.py` | Internal: TorchAO dynamic/weight-only `Float8Tensor` and calibrated static `PrototypeFloat8Tensor` optional imports, layout validation, and dequant/requant; consumed by the corresponding FP8 adapters |
| `_torchao_int8.py` | Internal: TorchAO INT8 optional-import + layout validation and dequant/requant; consumed by `int8_adapter.py` |
| `_triton_*_lora.py` | Internal: format-specific CUDA LoRA merge kernels; tensor adapters select them only for validated raw layouts and otherwise use their reference merge |
| `_torchao_int4_tile.py` | Internal: TorchAO INT4 tile-packed (CUDA-native tinygemm) optional-import + layout validation; consumed by `int4_tile_adapter.py` |
| `_dtensor.py` | Internal: PyTorch `DTensor` optional-import + mesh/placements signatures and local-shard rewrap; consumed by `dtensor_adapter.py` |

## Why use this

You have multiple PyTorch models that don't all fit on GPU
simultaneously, and you want to swap them in and out efficiently
across many calls. Re-loading from disk every call is too slow
(seconds per gigabyte). Keeping all models resident on GPU is too
expensive. `torch.cuda.empty_cache()` plus `.to("meta")` gets you the
basics; a shared cache adds reusable host storage, managed activation, and
block streaming.

This library gives you:

1. **Cached resources** that retain reusable model or adapter state to host RAM.
2. **Activation lifecycles** that move one cached model onto a compute device.
3. **A resource cache** with reference-counted leases and optional byte-based
   eviction.
4. **An unbounded model cache** that leases model and adapter resources, owns
   their device lifecycle, and leaves mapped-page residency to the OS.

## When to use what

| Situation | Use |
|---|---|
| Most application code, especially multiple models or repeated calls | Use **`ModelCache`** with **`ModelSpec`** |
| Model too big for a CUDA GPU even when active | Use **`ModelSpec(..., block_paths=...)`** for automatic block streaming |
| Adapters reused across calls | Pass **`AdapterSpec`** entries through **`ModelCache.use()`** |
| Low-level/manual lifecycle for one model | Use **`ModelOffloader.from_module(model)`** directly |
| Component or resource development | Use the lower-level store/binding protocols and component stores directly |

## Quick start: cached model use

```python
import torch
from piper_offload import ModelCache, ModelSpec

cache = ModelCache()
model_spec = ModelSpec(
    key="main",
    factory=build_my_model,  # returns a fresh nn.Module
)
device = torch.device("cuda")

# First use builds and leases the runtime.
with cache.use(model_spec, device=device) as gpu_model:
    output = gpu_model(input_tensor)

with cache.use(model_spec, device=device) as gpu_model:
    output = gpu_model(input_tensor_2)
```

`ModelCache` uses an unbounded `ResourceCache`, adding model activation and
adapter coordination to its registry and lease API. `ModelSpec` factories
should build fresh modules. One model cache entry contains one
`ModelOffloader` and one model instance. Uses are sequential:
an overlapping activation raises `ModelRuntimeInUseError`. Applications that
need concurrent replicas must register separately constructed models under
distinct cache keys. Their file mappings can still share physical OS pages.
To release host memory, evict or clear inactive cache entries and drop
any escaped model references.

### Host backing

Model and adapter factories transfer ownership of compatible complete pageable
CPU allocations and non-empty views into non-resizable storage to the cached
resource. This preserves checkpoint mmap backing when loaders assign mapped
tensors or splits of mapped tensors directly into the model. Callers must not
mutate factory-produced tensors after construction. Device tensors, pinned CPU
tensors, partial views into ordinary resizable allocations, and incompatible
layouts are copied into pageable CPU allocations. Tensor adapters preserve
packed quantized data, scales, and reconstruction metadata.

Host storage is shared by bound wrappers and counted once per stored object.
Capture does not register memory for accelerated transfers. The old
construction-time backing modes have been removed.

`HostParam.storage_tensors()` and `HostBuffer.storage_tensors()` expose the
existing physical CPU tensors, including packed quantized data, scales, and
tensor-valued metadata. Enumeration makes no copies and preserves views;
DTensor delegates to its local shard, and meta parameters return an empty
tuple. Consumers must deduplicate underlying allocations themselves. Use
`tensor.untyped_storage()` for allocation addresses and sizes; a tensor's
`data_ptr()` and `nbytes` describe its view. Enumeration does not register or
pin memory.

### Optional host registration

`host_pin_manager` registers existing CPU storage in place under a separate
`max_pinned_bytes` budget. Its default budget is zero, and construction and
configuration do not initialize CUDA. Ordinary streaming and compiled rolling
acquire leases automatically with their CUDA working sets. Set the budget
before activation to enable pinning. CPU execution, resident blocks, and
non-block components do not acquire pin leases.

Deactivation releases the lease after transfers finish and leaves registrations
in the idle LRU. Reactivating the same backing reuses its retained registrations
without native register/unregister calls. `BlockComponent.release()` also
releases pin protection during a temporary working-set release; `acquire()`
reuses or registers backing again. This lets transient components share the
budget. Resolved replacement sources, quantized payloads and metadata, buffers,
and trainable optimizer backing all participate in the same lease.

Explicit leases are also available for custom transfers:

```python
import torch
from piper_offload import host_pin_manager

host_pin_manager.max_pinned_bytes = 4 * 1024**3
source = torch.randn(1024, 1024)
target = torch.empty_like(source, device="cuda")
copy_stream = torch.cuda.Stream()

with host_pin_manager.acquire([source]) as pins:
    pins.record_stream(copy_stream)  # record before enqueueing, including failure paths
    with torch.cuda.stream(copy_stream):
        target.copy_(source, non_blocking=True)
# Closing the lease waits for recorded streams; source may remain registered in the idle LRU.
```

For model backing, pass tensors from `HostParam.storage_tensors()` and
`HostBuffer.storage_tensors()`. Acquiring a lease protects existing
registrations and registers additional whole allocations when capacity allows.
Budget or supported runtime-capacity failures leave complete allocations
pageable. They remain pageable until all their active leases close, even if
another request arrives after capacity becomes available. A lease reports
`registered_bytes` and `pageable_bytes` for unique
requested allocations. `host_pin_manager.stats.pinned_bytes` instead counts
the union of covered OS pages, including shared boundary pages only once.

Released registrations enter an idle LRU. Budget pressure evicts idle entries;
active leases remain protected. Discarding a source tensor retires its
registration once active users finish. Storage remains alive until successful
unregistration, including after cleanup errors; `clear()` retries failed cleanup
and evicts idle entries. `ModelCache` retains host stores until explicit
eviction; unpinned mapped pages remain reclaimable by the OS.
Do not resize storage or register/unregister it outside the manager while it is
managed. Use views of one storage for aliases; distinct overlapping byte ranges
are rejected before mutation.

The backend binds the CUDA or HIP runtime already loaded by PyTorch. It clears
errors from handled registration failures and reports unexpected errors through
`HostRegistrationError`, including conflicting foreign registrations. CUDA
registration semantics follow the [CUDA memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html);
the corresponding HIP calls are documented in the
[HIP runtime API](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/hip__runtime__api_8h.html).

## Manual offloader lifecycle

Use `ModelOffloader` directly when you want explicit lifecycle control without
`ModelCache`.

```python
import torch
from piper_offload import ModelOffloader

model = build_my_model()
offload = ModelOffloader.from_module(model)
device = torch.device("cuda")

offload.activate(device)
try:
    output = offload.value(input_tensor)
finally:
    offload.deactivate()

del offload, model  # drop refs to free host memory
```

`ModelOffloader.from_module()` mutates the source model during
capture: frozen `nn.Parameter` registry entries get repointed at
Parameters wrapping CPU storage, trainable Parameter objects keep
their identity and point their `.data` at CPU storage, and buffers
are replaced with host copies. After construction, only access the bound
model while the offloader is active: call `activate(device)`, use
`offload.value`, and guarantee a matching `deactivate()` with `try`/`finally`.
For CUDA training, wrap `optimizer.step()` in
`offload.optimizer_step()` so trainable GPU updates are copied back to
the CPU cache before deactivation.
**Drop the offloader and model references to release host
memory** — there's no `close()`; resource cleanup is reference-drop + GC.

## Manual block streaming

For models too big to fit on GPU even when active. Streams transformer
blocks through a small GPU-resident window using forward-pre hooks
and a CUDA-stream-based async prefetcher.

```python
import torch
from piper_offload import ModelOffloader

# Construction captures and binds once; cache_bytes is final immediately.
# block_paths selects block groups; block_mode selects their residency strategy.
offload = ModelOffloader.from_module(
    model,
    block_paths=["transformer_blocks"],  # path(s) to the nn.ModuleList
)
device = torch.device("cuda")

offload.activate(device)
try:
    output = offload.value(input_tensor)
finally:
    offload.deactivate()

del offload, model  # drop refs to free host memory
```

Ordinary streaming owns one active block and one asynchronous lookahead target.
That fixed two-target window overlaps the next whole-block copy without
retaining blocks that a sequential traversal will reload anyway. Direction
changes and iteration wraparound are detected internally. Block groups are
selected by `block_paths` and `transient_block_paths`.
`block_mode="streaming"` is the default. `block_mode="resident"` keeps every
block target resident, while `block_mode="rolling"` uses one compiled target
refilled parameter-by-parameter. `block_mode="auto"` selects rolling for each
supported block group when full-graph compilation is configured and otherwise
uses streaming. With both path lists empty, the whole model is one bulk-host
component that activation copies to the GPU.
For heterogeneous block lists, execution still limits concurrency to the active
and lookahead blocks, while the morphing pool may park one reusable target per
distinct tensor-layout signature.

`ModelOffloader` only streams on CUDA. Activating the binding on
`cpu` is a pass-through over the already-installed CPU storage:
no target pool, no streaming hooks, no weight copies.
`adapter_mode="merge"` is CUDA-only; use routed LoRA mode for
CPU activation. Routed LoRA installs target-Linear hooks: a forward-PRE
hook copies that target's host-backed factors to the input device, and a
forward-POST hook applies the residual and releases those device copies.

### Optional transient residency

Inference workloads can release large modules and streamed pools as soon as
their traversals finish:

```python
offload = ModelOffloader.from_module(
    model,
    transient_block_paths=["transformer_blocks"],
    transient_paths=["input_embedder", "output_head"],
)
```

Each `transient_paths` module recursively owns a separate CUDA working set for
its non-streamed state. Its successful forward releases that set
immediately, before later model work. Every entry in `transient_block_paths` is
a block group whose working set releases after its final block, using the same
`block_mode` as ordinary groups. Ordinary `block_paths` working sets remain
resident for the activation. A successful root-model forward reacquires all
released components for the next invocation. A conditionally skipped component
remains acquired.

The release boundaries are deliberately explicit. A `transient_block_paths`
group releases after its final block and therefore cannot traverse that stack
again later in the same root call. For `transient_paths`, the named module's
own `forward` is the boundary, so its state cannot be used again afterward. A
skipped path remains acquired. `ModuleList`/`ModuleDict` containers are not
expanded for `transient_paths`, and ModelOffloader does not inspect repeated
calls, functional parameter access, aliases across component boundaries, or
autograd state. Paths must own disjoint state, and shared-storage aliases must
not cross component boundaries; these guarantees belong to the caller. CPU
activation remains eager and installs no scheduling hooks. Ordinary and
rolling transient block groups both stop at the final block instead of filling
block 0 immediately before release.

Every module object in a `transient_block_paths` group must be distinct. An
ordinary `block_paths` group may still reuse a module object, but a module hook
cannot distinguish which aliased list position just completed and therefore
cannot provide a safe early-release boundary.

### Optional block compilation

Repeated blocks can opt into forward-only Inductor compilation at runtime
construction:

```python
from piper_offload import BlockCompileConfig, ModelOffloader
from piper_kernels.linear.convrot import convrot_int8_compile_options

offload = ModelOffloader.from_module(
    model,
    block_paths=["transformer_blocks"],
    block_compile=BlockCompileConfig(
        dynamic=True,
        fullgraph=False,
        options=convrot_int8_compile_options({"max_autotune": True}),
    ),
)
```

`block_compile=None` (the default) preserves eager behavior. One configuration
applies to every `block_paths` and `transient_block_paths` group and has no
effect when neither is configured. `block_mode="streaming"` is the default.
Select `block_mode="resident"` to load every target once at activation while
retaining the same per-block compilation. No pool rotation, prefetch thread,
private copy stream, or block scheduling hooks are created:

```python
offload = ModelOffloader.from_module(
    model,
    block_paths=["transformer_blocks"],
    block_mode="resident",
    block_compile=BlockCompileConfig(),
)
```

The selected mode applies equally to `block_paths` and
`transient_block_paths`; the latter differs only in when its working set is
released and reacquired. The backend remains fixed to Inductor. Optional
backend settings can be supplied through `options`; the mapping is copied for
each block before it is forwarded to `torch.compile`. This is also the boundary
for compiler extensions such as Piper Kernels' ConvRot preparation-sharing and
activation-folding pass shown above. When no options are provided, compilation
uses Inductor's default mode. Piper Offload does not pass `mode` because PyTorch
treats `mode` and `options` as mutually exclusive.

Only each distinct block module's `forward` is compiled. Its module
`__call__` stays eager. For streamed groups, the forward-pre hook also stays
eager, so block activation and prefetch finish before compiled computation.
Compiled forwards are installed only for CUDA activations and the exact
original forwards are restored on deactivate or activation rollback. CPU
activation remains eager. The lazy compiled callables are retained by the
bound runtime, so later eligible activations can reuse their compiled graphs.

Compilation is inference-only in this initial implementation. Training remains
available without `block_compile`, but combining block compilation with
autograd is unsupported until it has dedicated correctness coverage.

Merge-mode adapter application remains compatible. If any routed LoRA is active, every
declared block runs eagerly for that activation because routed child-Linear
hooks stage parameters inside the block forward. The bypass is temporary:
a later activation with no routed LoRA uses compiled forwards again.
Selecting `adapter_mode="routed"` without supplying an adapter does not bypass
compilation.

Experimental rolling mode replaces ordinary whole-block targets with one
shared parameter target and requires full-graph compilation:

```python
offload = ModelOffloader.from_module(
    model,
    block_paths=["transformer_blocks"],
    block_mode="rolling",
    block_compile=BlockCompileConfig(fullgraph=True),
)

offload.activate("cuda")
try:
    output = model(inputs)
finally:
    offload.deactivate()
```

Use `block_mode="auto"` with the same full-graph configuration to select
rolling independently for each compatible block group and whole-block
streaming for the remaining groups. A component's `block_mode` property
reports the resolved mode.

After a parameter's final compiled-graph reader launches, a CUDA event lets a
private stream refill that same storage from the next block while the remainder
of the current block computes. Immediately before that parameter's first reader
in the next block, the compiled graph waits only for its corresponding refill;
it does not stall for every parameter in the block. The rollover pass is
appended after user/Piper Kernels post-grad graph passes, so its first/last-use
analysis sees their rewritten graph. Waits and refills are non-mutating ordered
host effects with late scheduler-only ordering edges. They neither consume
reader tensors nor model an immutable parameter as mutated, avoiding forced
intermediate materialization while preserving the ordinary fusion, memory
coalescing, and kernel-autotuning plan. For factors whose tensor formats
support merge-mode LoRA, planned updates run after every base refill on the
copy stream. GGUF sources converted to ConvRot targets use the target's merge
kernels; TorchAO INT4 tile-packed weights retain their existing no-merge
restriction.

Explicit rolling deliberately fails closed outside its tested contract: `fullgraph=True`,
frozen regular dense, TorchAO-family, Quanto, GGUF, or Piper ConvRot INT8 / NVFP4
parameters, homogeneous block layouts, distinct block modules, no tied streamed
parameters, and no streamed buffers. In auto mode, Bitsandbytes, DTensor,
unreviewed external adapters, and heterogeneous block layouts instead use the
existing morphing block-target pool. Structured logical weights are tracked
across every AOT-flattened storage input, and the refill is placed after the
last reader of any storage tensor.
Repeated resident traversal rolls the final block directly into block zero.
Transient streaming stops at the final block and reacquires a fresh block-0
target after the root model forward. Skipped or out-of-order traversal remains
correct through a foreground-refill fallback.

For the supported contract, the scheduler-only lifecycle edges preserve the
ordinary compiled compute kernels and their autotuning identity. The benchmark
checks output equivalence in addition to latency and memory; model-level
validation remains prudent after changing PyTorch, compiler extensions, or
custom kernels.

Use `python benchmarks/benchmark_rolling_compile.py` to compare steady-state
latency and CUDA allocator residency against ordinary compiled block prefetch
on the current GPU. The benchmark also checks output equivalence.

Compiler and backend failures propagate normally. Piper Offload never catches
a failed compiled invocation and retries that same block eagerly: with graph
breaks, earlier segments may already have executed, so retrying could duplicate
mutations. Dynamo's native graph-break and recompilation-limit behavior still
applies.

By default, trainable parameters (e.g. LoRA adapters) are managed by
the composed `HostComponent`: they move to GPU on CUDA activation and
back to CPU storage on deactivate. On CPU activation they stay in
the host-backed module state. Wrap CUDA optimizer updates in
`offload.optimizer_step()` so updated trainable bytes are copied back
to CPU storage before deactivation.

To let the selected block residency strategy own in-block trainable weights,
opt them into the block component:

```python
offload = ModelOffloader.from_module(
    model,
    block_paths=["transformer_blocks"],
    include_block_trainables=True,
)
```

During CUDA activation, trainable parameter `.data` follows the selected block
mode. In streaming mode it is GPU-resident while its block is resident, plus
during the optimizer update. CPU activation remains pass-through.
Gradients are not streamed; PyTorch owns `param.grad` normally.

### Adapter application

`ModelOffloader` supports optional per-parameter adapter updates through
activation arguments. Merge mode builds activation-scoped parameter overrides
for managed targets. Each component resolves those overrides into immutable
load plans before allocating CUDA storage. Unknown targets raise during
activation by default. An adapter that is
intentionally shared across separately loaded model components can set
`allow_partial_targets=True`; its merge and routed uses then apply only the
intersection of adapter targets and model parameters, including a valid no-op
when that intersection is empty. Present targets still receive the ordinary
shape and capability validation. Adapter target keys must
match the model's parameter names exactly; any remapping — stripping a
`diffusion_model.` prefix, inserting a PEFT `.base_layer.` segment — is
the caller's job when building the adapter state dict. Each planned update runs
immediately after the owning component copies its effective source from host
CPU storage to GPU, so both
block-streamed and non-block weights use the same merge path. Merge
compatibility is tensor-format-owned: physical plain floating-point tensors
accept combined low-rank and full-rank additive deltas; structured quantized
wrappers may independently opt into staged factorized and full-rank merges
whose implementations select a format-specific kernel or framework fallback.
Dense-only and mixed dense + LoRA updates use the full-rank capability; the
mixed form re-encodes a quantized base once. Plain float8 merge targets remain
unsupported. Frozen plain floating-point meta tensors can instead be populated
by dense or registered structured parameter values.
Routed mode remains factor-only and requires a compatible logical `nn.Linear`
shape and compute dtype.
`HostParam` remains a storage primitive; transforms ask the selected tensor
adapter for their required capabilities.

`Adapter.from_state_dict()` reserves the exact suffixes `.lora_A.weight` and
`.lora_B.weight` for low-rank factors, and `.delta.weight` and `.delta.bias`
for full-rank additive updates. `module.delta.weight` targets `module.weight`,
while `module.delta.bias` targets `module.bias`. LoRA and dense terms for the
same parameter form one `ParameterDelta`. Every other entry is a
`ParameterValue` whose key is the exact model parameter name and whose source
is the complete value for a plain floating-point meta parameter. The source may
be any physical representation with a registered tensor adapter and a floating
compute dtype. Its dtype, layout, quantization metadata, and bytes become the
active representation. Construct `Adapter(targets=...)` directly if a model
parameter name itself ends in one of the reserved suffixes.

```python
feature_adapter = Adapter.from_state_dict(
    {
        "projection.lora_A.weight": projection_a,
        "projection.lora_B.weight": projection_b,
        "projection.delta.weight": projection_dense_delta,
        "projection.delta.bias": projection_bias_delta,
        "guidance_embedder.weight": guidance_weight,
        "guidance_embedder.bias": guidance_bias,
    },
)
```

`Adapter.targets` is one immutable exact-name mapping. `AdapterTarget` is the
union `ParameterDelta | ParameterValue`, so every mapped target is already in
a valid, concrete form. A parameter delta contributes
`strength * (B @ A + dense)` with whichever low-rank and dense terms are
present. A parameter value materializes the complete value unchanged by
default. Pass `scale_parameter_values=True` to `Adapter.from_state_dict()` or
`AdapterSpec` when active adapter strength should scale those values. Explicit
non-unit scaling requires the value representation to support both
`dequantize()` and `merge_dense_()`. This setting affects only exact-name
parameter values; LoRA and dense deltas remain strength-scaled. A zero-strength
adapter is inactive under either policy. Directly constructed mappings can
select the policy per target with
`ParameterValue.from_tensor(..., scale_with_strength=True)`.

The resource-level API uses `Adapter`, `AdapterMode`, `AdapterSpec`, and
`merge_adapter`; activation arguments use the corresponding `adapter_*` names.

Parameter values are merge-only: routed mode rejects the request. For a meta
target, merge mode materializes the value according to its strength policy.
Only one active parameter value may own a target; repeated or competing values
are rejected. Parameter values do not apply to existing physical model
parameters: the model-side target remains a storage-free plain floating-point
meta placeholder. The value's backing owns the active dtype, layout, packed
storage, quantization metadata, logical compute dtype, and any DTensor
composition. The placeholder supplies only the target name, logical shape, and
alias group. Supplying `dtype=` while constructing an adapter can cast a dense
value, but cannot convert a structured value; it must either match the
representation's compute dtype or the caller must prequantize the value again.
The caller is responsible for tensor payload numerical validity. Piper
validates representation structure, shape, layout, dtype, and capabilities,
but does not perform finiteness preflight scans over parameter values,
additive deltas, or model quantization metadata during construction,
activation, or permanent merge. Scalar adapter strengths remain
finite-checked.

Exact replacement supports offload-capable physical representations with a
registered tensor adapter, floating compute dtype, and compatible logical-shape
metadata. This includes TorchAO INT4 tile-packed values even though that format
does not support additive merges. Only explicit non-unit strength scaling
needs both dequantization and dense merge support. That scaled subset mirrors
dense-delta support across Quanto, bitsandbytes, TorchAO, Piper ConvRot, and
supported DTensor compositions. `ParameterValue` does not define a second
quantization metadata schema.

A frozen plain floating-point meta parameter has no host backing and
contributes zero bytes to model cache accounting.
Without an active parameter value it stays meta and no CUDA slot is allocated;
executing a module that still references it is the caller's error. With a
parameter value, resident, streaming, rolling, and automatic block modes
allocate active storage—there is no model base storage to copy. Dense values
are filled in the placeholder's declared layout. Structured values allocate
from their own backing and copy the packed representation exactly. Unit
strength (or disabled strength scaling) therefore performs no
dequantize/requantize round trip. A non-unit strength reuses dense merge as
`W + (strength - 1) * W`, producing one terminal requantization. Permanent
merge follows the same rule and installs an independent representation rather
than aliasing immutable adapter backing. Deactivation restores the meta
parameter. Low-rank A/B factors cannot materialize a meta target. Rolling allocates the union
of slots needed by its homogeneous block group. If another block requires the
same rolling slot, inactive blocks may temporarily reference that already
allocated storage; its inactive contents are unspecified and consume no
additional VRAM.

The adapter request is scoped to one `activate()` call. Target lookup is
resolved and both parameter-delta and parameter-value compatibility are
preflighted during activation. `ParameterDeltaTransform` coordinates each
target's additive contributions; its low-rank term delegates to
`LoRATransform`. Parameter values use `ParameterValueTransform`.

Bias deltas use the same exact-name model as weight deltas:
`module.delta.bias` contributes to the existing physical `module.bias`
parameter. Routed mode remains low-rank factor-only and does not synthesize an
adapter bias for a bias-less module. Native Wan/ComfyUI `.diff` and `.diff_b`
parsing remains the caller's responsibility. A converter can map those entries
to canonical `.delta.weight` and `.delta.bias` keys. Exact parameter-name
entries represent complete values for meta model parameters, so a converter
must also remove checkpoint metadata and other non-adapter entries.

```python
import torch
from piper_offload import ModelOffloader, Adapter
from safetensors.torch import load_file

offload = ModelOffloader.from_module(
    model,
    block_paths=["transformer_blocks"],
    # Default: include_block_trainables=False
)
device = torch.device("cuda")

# Each Adapter owns immutable tensors shared by compatible uses.
lora_a = Adapter.from_state_dict(
    state_dict=load_file("lora_a.safetensors"),
)
lora_b = Adapter.from_state_dict(
    state_dict=load_file("lora_b.safetensors"),
)

offload.activate(
    device,
    adapters=[lora_a, lora_b],
    adapter_strengths=[0.8, 0.5],
    adapter_mode="merge",
)
try:
    output = offload.value(input_tensor)
finally:
    offload.deactivate()
```

Adapter entries are ordered contributions. Repeating the same `Adapter` applies it
again, and each occurrence uses the corresponding `adapter_strengths` value.

Quantized LoRA merge uses stochastic rounding by default so factor updates smaller
than one quantization step are not systematically rounded away. Opt into
deterministic round-to-nearest when exact deterministic codes are required:

```python
offload.activate(
    device,
    adapters=[lora_a],
    adapter_strengths=[0.8],
    adapter_mode="merge",
    stochastic_rounding=False,
)
```

Sampling is an internal quantized-merge detail. A scalar seed is derived from the
full parameter path and that transform's merge count, then used by backend-local
randomness without consuming PyTorch's global RNG. Reapplying a streamed merge
therefore uses a fresh deterministic sample each time. DTensor additionally
derives a seed from each shard's global offsets, while replicated ranks retain
matching samples. All low-rank contributions for a target are accumulated and
rounded once; when a dense term is present, it and every low-rank contribution
are accumulated and rounded together. Parameter values use plain floating-point
copies and scaling and do not consume the stochastic-rounding seed. Routed mode
ignores the option because it never requantizes the base.

`derive_seed(*parts)` is the public canonical derivation utility used by this
path. It accepts strings and unsigned 64-bit integers and is useful when an
external adapter needs a deterministic sub-seed:

```python
from piper_offload import derive_seed

local_seed = derive_seed(parent_seed, shard_offset)
```

Adapter delta and parameter-value tensors own pageable CPU backing. Compatible
allocations and mmap-backed split views transfer without a copy. `dtype=`
converts dense sources during capture; prequantized values retain their encoded
representation and must already have the requested compute dtype.

Block reload from pristine CPU storage automatically clears
the previous merge — no explicit unmerge step needed.

Pass `adapter_mode="routed"` as an alternative to the default merge mode.
Routed mode installs a forward hook pair on each matched
`nn.Linear` parent — `y = base(x) + alpha * (B * A * x + bias)` when a legacy
bias is present — instead of merging into the base weight. Its PRE hook
copies only that target's factor tensors from CPU storage to the
invocation's input device; its POST hook applies the residual and releases
those device tensors. Multiple LoRAs on one target are grouped into one hook
pair and summed independently. **Routed mode is
inference-only:** factors are frozen (`requires_grad=False`) and no gradient
flows to them. Adapter backing is immutable, so merge and routed uses may overlap
across model runtimes. Use routed mode when:

- The base weight is quantized or otherwise structured, but still exposes
  a logical `nn.Linear` weight shape and compute dtype, and its adapter
  does not support merge updates. `adapter_mode="routed"` works because it
  doesn't touch the base.
- You want to switch LoRAs frequently without re-streaming the underlying
  base weight or retaining the whole adapter on GPU.

Routed mode is restricted to `nn.Linear` parents. It handles tied
weights by hooking only the exact parent module named by the target, so
it never mutates shared storage. Packed formats whose parameter shape
differs from the logical matmul weight need a per-format route layer.

For a one-shot **permanent** application—bake the adapter into the model
weights and discard the adapter—use `merge_adapter`:

```python
from piper_offload import merge_adapter, Adapter

merge_adapter(
    model,
    [(Adapter.from_state_dict(state_dict=load_file("lora.safetensors")), 0.8)],
    stochastic_rounding=False,  # optional deterministic opt-out
)
```

This uses in-place arithmetic for plain fp/bf bases. Supported quantized
adapters use format-specific Triton kernels for factor-only and full-rank
updates on CUDA and retain their dequantize/requantize reference path as a
fallback. Formats without the required factorized merge capability need routed
LoRA instead; a dense update requires the full-rank capability. See [Quantized weight
support](#quantized-weight-support) for the full matrix. Unlike an
activation-scoped adapter request, this is not reversible. Unknown targets
raise, and all target names, factor shapes, and advertised merge
capabilities are preflighted before mutation. Multiple LoRAs for one
quantized parameter are packed into one staged low-rank update and the
weight is re-encoded once. If any dense term is present, all dense and LoRA
terms are instead packed into one staged full-rank update and the weight is
also re-encoded once. A meta target is permanently
materialized as one frozen CPU `Parameter`; tied aliases of the original meta
parameter are preserved.

For the default stochastic merge, each adapter first uses its existing upstream
recipe to compute the final data-dependent scales and other quantization
parameters, then samples only the terminal weight code between the two
neighboring values on that finalized grid. Exact endpoints and saturation
retain the upstream code. Exact-zero strengths are discarded before target
lookup or factor staging. Standard CUDA layouts use the same format-specific
Triton merge kernels for deterministic and stochastic rounding. Random samples
are keyed by logical element index, so launch geometry does not change the
result. The Torch and
Triton backends replay independently for a fixed seed but do not promise
byte-identical samples across implementations or Triton versions. Nested
bitsandbytes 4-bit scales still use the reference path because their final
effective scale is known only after double quantization. Piper ConvRot INT8
and NVFP4 forward the derived seed to `piper-kernels`' public `addmm_` or
`add_`, which owns each format's terminal-code selection.

This is one composable requantization pipeline per format rather than parallel
deterministic and stochastic implementations. Each concrete adapter's existing
`requantize(t, like=..., rounding_seed=None)` first constructs its normal
finalized representation; when `rounding_seed` is supplied, it then recodes
only terminal weight data against those stored parameters. Omitting the keyword
preserves the original deterministic bytes. The public structural conversion
protocol retains its deterministic minimum signature for downstream static
compatibility.

### Heterogeneous block lists

`block_paths` and `transient_block_paths` accept dotted paths for models with
multiple kinds of blocks (e.g. Flux's `transformer_blocks` +
`single_transformer_blocks`). Each path becomes its own streaming
group with its own target pool; the two lists are mutually exclusive. Blocks
within a group must share parameter and
buffer names plus trainability structure, but their shapes, dtypes, quantization
formats, alias topology, and buffer layouts may differ. The morphing pool keys
reusable targets by those layouts. For bespoke grouping, compose
`BlockComponentStore` instances directly:

```python
offload = ModelOffloader.from_module(
    model,
    block_paths=["transformer_blocks", "single_transformer_blocks"],
)
```

### Training streamed blocks

Training through a streamed block **requires activation checkpointing
on each block** — wrap call sites in
`torch.utils.checkpoint.checkpoint`, or call
`model.gradient_checkpointing_enable()` on a HuggingFace model.
Without it, `loss.backward()` raises:

```
RuntimeError: one of the variables needed for gradient computation
has been modified by an inplace operation
```

The reason is autograd's saved-tensor mechanism. A `Linear` saves a
reference to its weight tensor at forward time and records the
tensor's version counter. Streaming is a sequence of in-place `copy_`
writes into a fixed pool of GPU target tensors — every block load
bumps the target tensor's version, so by the time backward arrives at
an earlier block, the target has been overwritten and the version
mismatch raises.

Activation checkpointing sidesteps this. With checkpointing, the
block's internal forward runs under `no_grad` — no internal tensors
are saved for backward. When backward arrives, PyTorch re-runs the
block's forward with grad enabled, building a fresh autograd graph
whose saved references only live within that one block's
recompute-then-backward window. Target reuse outside that window is
safe because no autograd graph spans across reuses.

```python
import torch
from piper_offload import ModelOffloader

offload = ModelOffloader.from_module(
    model,
    block_paths=["transformer_blocks"],
)
device = torch.device("cuda")

model.gradient_checkpointing_enable()  # required for training
model.train()

offload.activate(device)
try:
    gpu_model = offload.value
    for batch in loader:
        loss = gpu_model(**batch).loss
        loss.backward()
        with offload.optimizer_step():
            optimizer.step()
        optimizer.zero_grad()
finally:
    offload.deactivate()
```

Checkpointing every streamed training block is the caller's
responsibility — `ModelOffloader` does not auto-detect or warn about its
absence. It matters most with `include_block_trainables=True`, where the
`.data` swap bypasses autograd's version-counter check, so missing
checkpointing can silently corrupt gradients. Verify every streamed
training block is checkpointed (HF `gradient_checkpointing_enable()` or
manual `torch.utils.checkpoint.checkpoint` wrapping).

Wrap CUDA optimizer updates so managed trainable weights are synced back
to CPU storage. With `include_block_trainables=True`, this also
materializes streamed trainable weights on GPU while a normal PyTorch
optimizer mutates them:

```python
offload.activate(device)
try:
    gpu_model = offload.value
    for batch in loader:
        loss = gpu_model(**batch).loss
        loss.backward()

        with offload.optimizer_step():
            optimizer.step()

        optimizer.zero_grad()
finally:
    offload.deactivate()
```

This boundary is not optimizer-specific. It runs whatever
`optimizer.step()` does, copies updated trainable data back to host
CPU storage, and leaves gradients on GPU.

## Cached model details

`ResourceCache` owns reusable-resource registration, accounting, leases, and
eviction. `ModelCache` uses it without a byte limit and adds dependency
leasing, adapter attachment, and device activation for model uses.

```python
from piper_offload import (
    AdapterSpec,
    ModelCache,
    ModelSpec,
)
from safetensors.torch import load_file

cache = ModelCache()
device = "cuda:0"

text_encoder = ModelSpec(
    key="text_encoder",
    factory=build_text_encoder,
)
diffusion_model = ModelSpec(
    key="diffusion_model",
    factory=build_diffusion_model,
    block_paths=("transformer_blocks",),
)
style_lora = AdapterSpec(
    key="style-lora",
    factory=lambda: load_file("style.safetensors"),
    dtype=torch.bfloat16,
)

with cache.use(text_encoder, device=device) as enc:
    embeddings = enc.encode(prompt)

with cache.use(
    diffusion_model,
    device=device,
    adapter_specs=[style_lora],
    adapter_strengths=[0.8],
    adapter_mode="routed",
) as model:
    latent = model(...)
```

For a parameter-value adapter, have the ordinary factory return the
exact-name mapping and activate it with `adapter_mode="merge"`:

```python
feature_adapter = AdapterSpec(
    key="feature-adapter",
    factory=load_parameter_values,
    dtype=torch.bfloat16,
)
```

The model cache leases adapter resources before the model resource. All leases
unwind in reverse order if construction or activation
fails. `adapter_strengths` defaults to `1.0` per adapter; when supplied, it must
have the same length as `adapter_specs`. Exact `0.0` and `-0.0` strengths are
inactive: they are filtered before target grouping or hook installation, and
`ModelCache` does not construct or lease their adapter resources. Merge and routed
uses may share one cached `Adapter` across model runtimes. LoRA-only adapters
may repeat within one use because each occurrence is an independent
contribution with its own strength; parameter values require a single active
owner per target.

For direct resource access, use a cache lease:

```python
with cache.lease(style_lora) as lora:  # auto-registers on first lease
    targets = lora.targets
```

> **Anti-pattern:** the factory should build a fresh model each call,
> not capture an externally-held one. With `factory=lambda:
> my_kept_model` the cache is no longer the sole owner of the model.
> Always have the factory build the model itself.

For a finite `ResourceCache`, custom `EvictionPolicy` implementations control
automatic eviction. The default is `LRUEvictionPolicy` for unleased stores.
The cache builds the eviction candidate set and byte context, then asks the
eviction policy to choose victims; `ResourceCache` still owns validation,
accounting, admission, and release. Policies are called under
the cache lock. `choose_victims()` must return unique keys from
`context.candidates` and enough bytes to satisfy
`context.bytes_to_free`; otherwise `ResourceCache` raises
`EvictionPolicyError` without evicting anything.

## Architecture

```
registration / cache admission
------------------------------
            ResourceSpec protocol
                    |
         ModelSpec / AdapterSpec / ObjectSpec
        |
        v
  +-------------+
  | ModelCache  |  unbounded ResourceCache with model-aware use
  +-------------+
        |
        +-- builds/retains -> ModelOffloader (one model, one runtime)
        |                    |
        |                    +-- HostComponent
        |                    |       |  resident non-block state
        |                    |       +-- HostParam(s)
        |                    |
        |                    +-- HostComponent(s)
        |                    |       |  transient path state
        |                    |       +-- HostParam(s)
        |                    |
        |                    +-- BlockComponent(s)
        |                            |
        |                            +-- HostParam(s)
        |
        +-- builds/retains -> Adapter (CPU-backed LoRA/parameter values)
        |
        +-- builds/retains -> custom ResourceStore

ModelCache.use(...)
-------------------
ModelCache
   |
   +-- lease AdapterSpec(s), then ModelSpec
   +-- ModelOffloader.activate(adapters=...) claims the model runtime
   +-- yield ModelOffloader.value
   +-- ModelOffloader.deactivate() removes adapter hooks + releases model
```

`ResourceSpec` is the structural registration contract: `key`,
`estimated_cache_bytes`, `build_store()`, and `value(store)`. The standard
specs are independent frozen dataclasses; custom specs can implement the
protocol without inheriting from them. `ResourceStore` is the backing-state
contract and reports `cache_bytes`. A cache lease protects that store from
eviction but does not create or activate a runtime.
`ResourceBinding` is the active-resource lifecycle contract: `value`,
`activate(device=None, **kwargs)`, and `deactivate()`.
`ModelOffloader` is both a cached `ResourceStore` and a `ResourceBinding`;
`Adapter` is an immutable cached `ResourceStore`. It exposes neither an active
lifecycle nor a model-like `value`; merge and routed hooks read its host
adapter backing directly.

A custom cached resource needs only one spec and one store:

```python
from dataclasses import dataclass

from piper_offload import ResourceSpec, ResourceStore


class MyStore:
    @property
    def cache_bytes(self) -> int: ...


@dataclass(frozen=True)
class MySpec:
    key: str
    estimated_cache_bytes: int

    def build_store(self) -> MyStore:
        return MyStore()

    def value(self, store: ResourceStore) -> MyStore:
        assert isinstance(store, MyStore)
        return store


spec: ResourceSpec[MyStore] = MySpec(
    key="my-resource", estimated_cache_bytes=...,
)

with cache.lease(spec) as store:
    ...
```

`BlockComponent` and `HostComponent` are composable
`activate`/`deactivate` lifecycle pieces (no `value` or `model`) that live
inside a top-level model runtime rather than acting as one themselves. Either
active component may `release()` and later `acquire()` its CUDA working set
without ending the activation session; activation acquires it immediately by
default. `ModelOffloader.register_forward_hook()` registers a native PyTorch
forward hook by fully-qualified module name and returns a caller-owned remover.
This lets higher-level runtimes coordinate component lifetimes at model
execution boundaries without adding policy to component internals.

`TensorAdapter` is the per-parameter extension point. Its base contract
only covers inference movement: host capture, storage enumeration, H2D copy, GPU wrapper rebuild,
cache bytes, logical compute dtype, and block-layout signatures. Extra
behaviors are explicit capabilities: CPU round-trip for optimizer-step
sync, `Parameter.data` swap for trainable streaming, shape-preserving
dequantize/requantize conversion, representation-preserving `copy_into`, and
adapter-owned staged factorized and dense merge. Plain bases implement these capabilities with native
`addmm_` and `add_`; structured bases own the representation-preserving
implementation. Conversion and copy capabilities do not implicitly advertise
merge support. Factorized merges with layout or value constraints can implement
`LoRAMergeValidationTensorAdapter`. Dense formats can implement
`DenseMergeTargetValidationTensorAdapter` for target-only constraints and
`DenseMergeValidationTensorAdapter` when they must inspect the staged update.
Target-only validation avoids materializing an update solely to check a weight
layout. `MergeLocalityTensorAdapter` lets composing wrappers expose one local
shape and global offset tuple for both factorized and dense staging. Permanent
merge validates every requested operation through these hooks before mutating
any weight; DTensor delegates validation to its local-shard adapter. The merge
and validation protocols include an optional
`rounding_seed: int | None = None` keyword. Downstream adapters should accept
that keyword even when they only implement deterministic rounding; omitting it
or passing `None` preserves deterministic behavior. An adapter that needs a
reproducible substream can derive one with the public `derive_seed()` utility.

Downstream tensor subclasses can provide their adapter without adding a
format-specific dependency to piper-offload.

Every adapter must implement `storage_tensors(state)` alongside
`capture_host()`. Return the captured state's existing plain CPU storage
tensors, including tensor-valued metadata that stays on the CPU. Omit absent
optional tensors, preserve shared allocations and views, and delegate through
composing wrappers instead of reconstructing them.

```python
from piper_offload import (
    TensorAdapter,
    register_adapter,
)


class MyTensorAdapter:
    # Implement the stateless TensorAdapter protocol.
    ...


remove_adapter = register_adapter(MyTensorAdapter)
```

Register adapters during application startup, before constructing models or
host resources. `DTensorAdapter` remains the outermost wrapper; registered
adapters are then checked newest-first before the remaining built-ins. This
lets a downstream adapter override a built-in `isinstance` match for a more
specific subclass, and also lets DTensor delegate a custom local shard through
the same registry. `register_adapter()` returns an idempotent removal callable
for tests and scoped integrations.

## Cached resource lifecycle

Cached resources own cache accounting. Host capture happens during construction
so `cache_bytes` is final once the store is built; leases protect resources
while they are used. `ModelOffloader` owns one exclusive activation lifecycle.
`Adapter` remains immutable host backing throughout its lease:

```
ModelOffloader: construct -> lease -> activate <-> deactivate -> release lease
Adapter:       construct -> lease -> read host updates -> release lease
```

`ModelOffloader.activate(device=...)` makes the model usable for compute on the
requested device. Merge updates stage factors when their base weight is loaded;
routed PRE hooks copy factors for one Linear invocation and routed POST hooks
release them after enqueueing the residual.
`ModelOffloader`, `MpsWeights`, `HostComponent`, and
`BlockComponent` require an explicit device. CUDA activation uses the
streaming/DMA path where applicable; CPU activation is pass-through over
host storage.
`deactivate()` releases transient device resources. Host backing remains cached
until its resource is evicted or otherwise released.

Construction transfers compatible complete pageable CPU allocations and
non-empty views into non-resizable storage, preserving checkpoint file mappings
through split parameters. Other sources are copied into pageable CPU storage.
For plain `torch.Tensor` parameters, the source `Parameter.data` may be
immediately repointed at the captured backing as soon as that host parameter is
created. This releases replaced source storage early and promptly frees GPU
storage for CUDA-origin models. Tensor subclasses such as quanto, GGUF, and
NVFP4 do not use this `.data` swap when it would lose wrapper state.

**There is no `close()`.** To release host memory, first let all
leases end, then evict or clear the cache entry. Python's refcount-based
GC frees host tensors once the cache and any escaped resource, binding, or
model references are gone.

**Failure semantics.** If construction raises after capture has started,
the model may already be partially repointed to host storage. Treat the
partially constructed resource/model as unrecoverable: drop those references
and rebuild from a fresh model instance. If `activate()` raises, the offloader
rolls back its active components and releases its activation claim, so a later
well-formed activation may retry the same cached resource. Routed LoRA
hook-registration failures remove any hooks already installed; permanent merge
validates all targets before mutation.
This is a low-level library; we don't guard against caller misuse.

## Compatibility

- **`torch.compile` support is deliberately narrow.** Use
  `block_compile=BlockCompileConfig(...)` to compile only declared block
  forwards during CUDA inference. `block_mode` selects resident, whole-block
  streaming, or rolling execution. External whole-model `torch.compile(model)`,
  `model.compile()`, compilation outside declared block groups, and compiled
  training remain unsupported. Routed LoRA
  temporarily bypasses compiled blocks. Compiler code/artifact caches and
  compiler-owned workspace are outside `ResourceCache.cache_bytes`; model
  eviction does not call process-global `torch.compiler.reset()`. Experimental
  `block_mode="rolling"` has the additional homogeneous/full-graph and
  adapter restrictions documented above. `block_mode="auto"` applies rolling
  only to groups satisfying those static restrictions and streams the rest.
- **Wrap before DDP/FSDP**, not after. Those wrappers manage parameter
  storage themselves and conflict with the registry-replacement pattern.
- **One runtime per cached model.** `ResourceCache` serializes resource
  construction and lease accounting, then releases its lock while a lease is
  held. Each cached `ModelOffloader` owns one model and rejects overlapping
  activation, including calls from different runners. Concurrent replicas
  require distinct `ModelSpec` keys and therefore distinct host storage.
- **Buffer mutations during CUDA activation are discarded** on
  `deactivate()`. CPU activation is pass-through over host-backed
  buffers, so CPU buffer mutations behave like ordinary module
  mutations. Suitable for inference of stateless modules; not suitable
  for models that need persistent buffer state across calls (BatchNorm
  running stats updated in training mode, RNN/SSM hidden state, KV
  cache).
- **Training requires activation checkpointing** on every streamed
  block (`model.gradient_checkpointing_enable()` for HF models, or
  manual `torch.utils.checkpoint.checkpoint` wrapping). Without it,
  `loss.backward()` raises an in-place modification error from
  autograd's saved-tensor check. See
  [Training streamed blocks](#training-streamed-blocks).

## Tied weights

`ModelOffloader` handles the standard `tie_weights()` pattern (one
`Parameter` referenced under multiple names) plus the rarer case of
distinct quanto wrappers around shared inner `_data` storage.

`ModelOffloader` is intended for ordinary transformer block lists where
the streamed block weights are independent. Shared storage is preserved
within one streamed block and within host state, including the standard
tied input-embedding/output-head pattern. It is unsupported across offload
ownership boundaries: between streamed and host state, or between distinct
streamed blocks or block groups. `ModelOffloader` does not prevalidate these
unusual layouts; omit `block_paths` and use whole-model offloading if that
sharing must be preserved.

## Quantized weight support

Every supported weight type can be offloaded (host ↔ GPU
movement). Additive-update support differs by type: factor-only LoRA and
full-rank dense deltas have separate adapter capabilities. **Routed** LoRA
(`adapter_mode="routed"` — a forward hook) is available for any of them
whose owning module is a logical `nn.Linear` with compatible shape and
dtype, no merge capability required.

| Weight type | Offload | LoRA merge | Dense / mixed merge |
|---|---|---|---|
| Plain floating-point tensor | ✓ | native `addmm_` | native `add_` |
| optimum-quanto qint8 / qfloat8 | ✓ | Triton; reference fallback | Triton; reference fallback |
| bitsandbytes NF4 / FP4 | ✓ | Triton; reference fallback | Triton; reference fallback |
| bitsandbytes int8 | ✓ | Triton; reference fallback | Triton; reference fallback |
| TorchAO scaled-FP8 | ✓ | Triton; reference fallback | Triton; reference fallback |
| TorchAO static-activation scaled-FP8 | ✓ | Triton; reference fallback | Triton; reference fallback |
| TorchAO INT8 | ✓ | Triton; reference fallback | Triton; reference fallback |
| TorchAO MX (MXFP8 / MXFP4) | ✓ | Triton; reference fallback † | Triton; reference fallback † |
| TorchAO NVFP4 | ✓ | Triton; reference fallback † | Triton; reference fallback † |
| GGUF → Piper ConvRot INT8 | ✓ | Piper `addmm_` | Piper `add_` |
| TorchAO INT4 tile-packed | ✓ | — routed only | — |
| Piper ConvRot INT8 | ✓ | Piper `addmm_` | Piper `add_` |
| Piper ConvRot NVFP4 | ✓ | Piper `addmm_` † | Piper `add_` † |
| DTensor (tensor-parallel shard) | ✓ | delegate to inner adapter ‡ | delegate to inner adapter ‡ |

Notes:

- **Stochastic rounding** is supported by every merge-capable built-in
  quantized adapter in the table. Standard CUDA layouts use fused Triton
  terminal-code selection; unsupported layouts and
  nested bitsandbytes 4-bit scales retain the dequantize/requantize reference
  path. Both preserve the same scale, calibration, packing, and wrapper
  metadata contract. Plain floating-point and DTensor-wrapped dense weights
  accept the option but have no quantization code to randomize.

- **Merging into a quantized base is lossy** because the updated value is
  re-encoded onto the quantization grid; choosing merge vs routed is the caller's
  accuracy/latency tradeoff, and is coarser the fewer bits the format has
  (e.g. MXFP4 / NVFP4 at 4 bits). Data-dependent weight scales are recomputed
  from the merged values. Formats that cannot safely encode with a zero scale,
  including Quanto qint8/qfloat8, floor exact-zero blocks to a small positive
  value; bitsandbytes int8 retains an exact zero scale for an all-zero row.
- **Triton dispatch is layout-conservative.** Each adapter checks its raw
  storage, scale metadata, compute dtype, and device before launching. A valid
  but nonstandard representation uses the reference merge when that format can
  re-encode the layout.
- **†** MX and NVFP4 store weights in a block-structured packed layout, so
  the standard re-encode (which produces the contiguous layout) cannot fill
  a transposed (non-contiguous `qdata`) target's storage; those raise a
  clear error pointing to routed LoRA. A transposed scaled-FP8 `PerGroup`
  weight is likewise unsupported because TorchAO only reconstructs groups
  along the last axis. `PerRow` and `PerTensor` scaled-FP8 transposes remain
  mergeable. int8 cannot be transposed.
- **‡** DTensor merge supports ordinary `Replicate` and contiguous `Shard`
  placements. Each rank selects its dense slice or the rows and columns needed
  from plain host-backed LoRA factors before device staging, then delegates the
  update to that shard's corresponding merge capability; no collective is
  required. Full adapter tensors remain in host memory. Unsupported factorized
  updates can use routed LoRA; dense updates require a dense-merge-capable
  local adapter. DTensor adapter tensors themselves are not accepted.
- **CPU round-trip** (D2H, for context-free CPU optimizer steps) and
  **trainable `Parameter.data` swap** are separate capabilities: plain
  tensors have both; quanto and both scaled-FP8 representations add CPU
  round-trip; the other quantized formats are movement + (where shown) merge
  only. See the per-format sections below.

## Quanto support

Quanto-quantized models (`optimum.quanto.WeightQBytesTensor`) are
handled correctly by both `ModelOffloader` modes. `HostParam` decomposes
the wrapper into its inner `_data` (int8/fp8) and `_scale` (fp16/fp32)
tensors, captures each, and reconstructs the quanto wrapper around the GPU
storage on activation.

Optimized `MarlinF8QBytesTensor` weights are first canonicalized to the
ordinary unpacked `WeightQBytesTensor` representation for host streaming,
so streamed execution is correct but does not use Marlin's packed matmul.
Direct merges into an existing Marlin weight use the reference
dequantize/addmm/requantize path and repack the result into that weight's
original physical storage.

A naive `param.data.clone()` on a quanto tensor silently
*dequantizes* it via the dispatch fallback — the explicit decomposition
is required for correctness.

LoRA on qint8 and the common E4M3/E5M2 qfloat8 bases uses a Triton merge on
CUDA that recomputes Quanto's absmax weight scale from the merged values.
Other layouts, including qfloat8 E4M3FNUZ, use the equivalent
dequantize/addmm/requantize path with the same scale policy. Exact-zero scale
blocks are repaired to a small positive value so qfloat8 never encodes a
`0 / 0` NaN. Both paths update the existing inner data and scale storage;
neither attempts native in-place `addmm_` on a `WeightQBytesTensor`. Use
`adapter_mode="routed"` when the base must remain untouched or adapters need
to switch without reloading it.

## GGUF source support

Piper Offload recognizes Diffusers `GGUFParameter` objects directly through
their existing packed tensor, `quant_type`, and `quant_shape` metadata; it does
not import or depend on Diffusers. Piper Engine can therefore load and remap a
GGUF model, normalize its GGUF linear modules to ordinary linear behavior, and
pass the model directly to `ModelOffloader`. No target policy, parameter
wrapping, or runtime option is required.

On activation, the host backing remains packed GGUF. The GPU target owns one
same-size packed staging buffer plus reusable ConvRot storage. Every load or
rolling refill copies the packed bytes and asks Piper Kernels to decode,
rotate, and requantize directly into that target; it never allocates the full
dense source weight. The active module parameter is a BF16
`ConvRotInt8Tensor`. Its group size is the largest of 256, 64, and 16 that
divides the logical input width, matching H3's group size 256 while retaining
smaller compatible matrices. Block compilation, rolling storage tracking,
LoRA merge, and dense/mixed merge use the existing ConvRot path without
GGUF-specific runtime branches. These updates are activation merges; permanent
merge cannot update the packed source representation because the ConvRot
target exists only while the offloader is active.

Direct conversion supports Piper Kernels' GGUF formats: F32, F16, BF16, Q4_0,
Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, and IQ4_XS.
The logical input width must be divisible by 16. Install the `gguf` extra;
direct conversion requires
`piper-kernels[convrot]>=0.7.0rc3`.

## Piper ConvRot INT8 support

Piper ConvRot weights
(`piper_kernels.linear.convrot.ConvRotInt8Tensor`) are handled when the
`convrot` optional extra is installed. `piper-kernels` owns the tensor semantics
plus reference and optimized execution backends; Piper Offload owns only the
built-in `PiperConvRotInt8Adapter`. `HostParam` captures the INT8 `qdata` and
float32 per-output `scale`, preserves `group_size` and the logical floating
`dtype`, and reconstructs the same wrapper around CUDA storage on activation.

The adapter remains frozen-only: it does not expose CPU round-trip or
trainable `Parameter.data` swap. Merge-mode LoRA delegates staged factors to
Piper's public in-place `ConvRotInt8Tensor.addmm_`; dense-only and mixed
dense + LoRA updates are combined once and delegated to
`ConvRotInt8Tensor.add_`. Both operations receive the optional reproducible
stochastic-rounding seed and preserve the wrapper and its storage identities.
Piper uses its optimized Triton backend on supported CUDA devices and its
portable reference backend elsewhere. Use routed LoRA when the base must
remain untouched. This integration requires Piper Kernels 0.7.0rc1 or newer.
The base package remains
importable without `piper-kernels`; use
`uv sync --extra convrot --group dev` and then
`pytest tests/test_piper_convrot_int8_adapter.py -q -rs` to exercise the
optional suite.

## Piper ConvRot NVFP4 support

Piper ConvRot NVFP4 weights
(`piper_kernels.linear.convrot.nvfp4.ConvRotNVFP4Tensor`) use a dedicated
adapter selected before the broader TorchAO NVFP4 adapter. It captures the same
packed E2M1 data, FP8 block scales, and optional global scales as ordinary
NVFP4 while additionally preserving the rotation group through identity,
pool-layout, host, device, and rolling reconstruction.

Merge-mode LoRA passes the staged factors, strength, and optional deterministic
rounding seed to Piper Kernels' public `ConvRotNVFP4Tensor.addmm_`. Dense-only
and mixed dense + LoRA updates are combined once and passed to its public
`add_`. Piper Kernels performs the appropriate rotation, updates the weight in
its stored rotated basis, recomputes weight-side NVFP4 scales, and refills the
existing packed storage while preserving activation calibration metadata.
This keeps rotation and quantization semantics out of Piper Offload. Use
routed LoRA to avoid the lossy 4-bit re-encode or when the packed target is
non-contiguous. Dense merge requires Piper Kernels 0.7.0rc1 or newer.

Install the `convrot` extra and run
`pytest tests/test_piper_convrot_nvfp4_adapter.py -q -rs` to exercise this
optional integration, including exact-SM120 forward coverage when available.

## TorchAO NVFP4 support

TorchAO NVFP4 weights
(`torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor`) are handled
when the `torchao` optional extra is installed.
`HostParam` captures the packed FP4 `qdata`, FP8 block `scale`,
optional per-tensor scales, and the TorchAO dispatch metadata, then
rebuilds the same concrete `NVFP4Tensor` subclass around GPU storage on
activation. Subclass identity also survives requantization and merge-mode LoRA.
The optional extra requires the package's supported TorchAO release; dynamic
NVFP4 matmul execution still depends on Blackwell-class CUDA hardware and the
matching PyTorch CUDA stack.
For uv-managed installs, this repo routes `torch` on Linux/Windows and
`torchao` on Linux through PyTorch's CUDA 13.0 wheel index. Windows uses
TorchAO's portable PyPI wheel because the CUDA 13.0 index does not publish a
Windows TorchAO wheel. Use
`uv sync --extra torchao --group dev` and then
`pytest tests/test_nvfp4_adapter.py -q -rs` to exercise the optional
TorchAO NVFP4 coverage.

Contiguous rank-two NVFP4 weights support a block-local Triton merge for
ordinary or swizzled scales. Single-level scaling needs one merge/pack pass;
two-level scaling first reduces the merged amax to a new global scale and
then recomputes and packs each block. Neither path materializes the dense
weight. Unsupported layouts use the existing `NVFP4Tensor.to_nvfp4`
reference path. Both preserve the wrapper and its dispatch metadata while
re-deriving the data-dependent scales. Like any merge into a quantized base,
NVFP4's 4-bit re-encoding is lossy.

The adapter does not opt into CPU round-trip or trainable
`Parameter.data` swap: the quant state lives in the wrapper object, not
its bytes, so NVFP4 weights stay frozen for streaming/training. Routed
Routed LoRA remains the non-destructive alternative when the target module is a
logical `nn.Linear` with compatible shape and compute dtype.

## TorchAO MX (MXFP8 / MXFP4) support

TorchAO MX (OCP microscaling) weights
(`torchao.prototype.mx_formats.mx_tensor.MXTensor`, created by
`quantize_(...)` with an MX inference config or directly via
`MXTensor.to_mx`) are handled when the `torchao` optional extra is
installed. A single adapter covers both
MXFP8 (`float8_e4m3fn` / `float8_e5m2`) and MXFP4
(`float4_e2m1fn_x2`), since TorchAO models them as the same `MXTensor`
subclass parameterized by `elem_dtype`. `HostParam` captures the packed
`qdata`, the E8M0 block `scale`, and the TorchAO dispatch metadata
(`elem_dtype`, `block_size`, `kernel_preference`, `act_quant_kwargs`,
`is_swizzled_scales`), then rebuilds the `MXTensor` wrapper around GPU
storage on activation. MXFP6 and any other MX element dtype are not
admitted; such a tensor falls through to a clear "no adapter" error
rather than being silently mishandled. MX matmul execution still
depends on Blackwell-class CUDA hardware and the matching PyTorch CUDA
stack. Use `uv sync --extra torchao --group dev` and then
`pytest tests/test_mx_adapter.py -q -rs` to exercise the coverage.

Standard blocksize-32 MX weights support a block-local Triton merge for
MXFP8 E4M3/E5M2 and packed MXFP4, including regular or swizzled scales and
TorchAO's FLOOR, RCEIL, CEIL, and EVEN scale modes when the mode is recorded
in `act_quant_kwargs`. Weight-only wrappers do not retain the mode and use
TorchAO's default FLOOR when re-encoded. Each kernel program updates and
packs one 32-element block without materializing the dense weight.
Unsupported layouts use the existing `MXTensor.to_mx` reference path.
MXFP4's grid makes a permanent merge much coarser than MXFP8. The adapter
does not opt into CPU round-trip or trainable `Parameter.data` swap: like
NVFP4, the wrapper's quant state lives in the object, so MX weights stay
frozen. Routed LoRA remains the non-destructive alternative.

## TorchAO scaled FP8 support

TorchAO scaled-fp8 weights (`torchao.quantization.Float8Tensor`, created
by `quantize_(..., Float8WeightOnlyConfig/Float8DynamicActivationFloat8WeightConfig)`)
are handled when the `torchao` optional extra is installed. `HostParam`
captures the fp8 `qdata` and fp32 `scale` tensors plus the TorchAO dispatch
metadata (`block_size`, `mm_config`, `kernel_preference`,
`act_quant_kwargs`), then rebuilds the `Float8Tensor` wrapper around GPU
storage on activation. Per-group, per-row, and per-tensor scale granularities are
supported; fp8 matmul execution requires SM89+ (Ada/Hopper or newer)
CUDA hardware.

Standard scaled-FP8 layouts use format-specific Triton merges on CUDA and
recompute the affected scales; unsupported layouts retain the public
`Float8Tensor.from_hp` reference path. The GPU representation is
byte-identical to the host one, so CPU round-trip is also available.
Trainable `Parameter.data` swap is not — scaled-FP8 weights stay frozen.

TorchAO's calibrated static-activation representation is handled separately
by `StaticFloat8Adapter`. It targets only
`torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor.PrototypeFloat8Tensor`
weights with per-tensor weight and activation quantization, and captures the FP8
`qdata`, weight `scale`, and checkpoint-provided `act_quant_scale`. All three
are included in identity, block-pool layout compatibility, cache accounting,
H2D/D2H movement, and wrapper reconstruction; the ordinary `Float8Tensor`
adapter remains the weight-only/dynamic path.

TorchAO 0.17 normally requires the activation scale rank to equal the input
rank. Piper Offload's static adapter installs a narrow `nn.Linear` dispatch
shim that flattens ordinary activations before static quantization and reshapes
the result afterwards. A checkpoint scalar (or any one-element scale layout)
therefore works unchanged for both 2-D and 3-D Linear inputs. LoRA merge uses
a format-specific Triton kernel pipeline on CUDA when Triton is available,
independently of `block_compile`. It fuses dequantization, the low-rank GEMM,
addition, and tile-level maximum collection before reducing the new per-tensor
weight scale and requantizing. CUDA installations without Triton use the same
raw storage through ordinary Torch operations; CPU merges retain the generic
adapter path. All paths copy only the re-encoded weight bytes and scale into
the target; the calibrated activation scale is preserved exactly. Routed LoRA
is supported as the non-destructive alternative. Output activation quantization
and non-per-tensor Prototype layouts are outside this adapter's contract and
are rejected explicitly.

## Failure modes

The cache and bindings surface failures as typed exceptions rather
than silent corruption.

| Exception | When |
|---|---|
| `ResourceTooLargeError` | Cache miss can't fit even after evicting all inactive entries. Exposes `required`, `used`, and `limit`. |
| `EvictionPolicyError` | Custom eviction policy returned duplicate/non-candidate victims or too few bytes |
| `ResourceLeasedError` | A cache mutation targets a currently leased entry |
| `ResourceCachedError` | `unregister(..., evict=False)` targets an entry with a built store |
| `ModelRuntimeInUseError` | Any caller overlaps activation of the same cached `ModelOffloader` |
| `DuplicateResourceKeyError` | `register()` is called for an existing key without `replace=True` |
| `ResourceNotRegisteredError` | `lease(str)` is called for an unknown key |

## State Inspection

Use `cache.used_cache_bytes` for logical backing accounting and
`cache.info(key)` for per-key state. `ModelCache.available_cache_bytes` is
`None` because it has no byte limit; release its inactive stores explicitly
with `evict()` or `clear()`.

An explicitly finite `ResourceCache` can change its budget while running:

```python
cache.resize(40 * 1024**3)
# Equivalently:
cache.max_cache_bytes = 40 * 1024**3
```

Growing the budget preserves cached entries. Shrinking evicts inactive entries
according to the configured eviction policy. If leased entries make the target
size impossible, resizing raises `ResourceTooLargeError` without changing the
previous budget or evicting entries.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
