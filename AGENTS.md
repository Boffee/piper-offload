# piper-offload

Keeps the state of several PyTorch models in host RAM and activates one at a
time on a GPU: whole-model uploads, or block streaming for models larger than
VRAM, with LoRA and quantized formats handled per parameter. Piper Engine is
the consumer and pins a released version. The README is the reference; this
file holds what agents get wrong without it.

## Layout

- Resources: `resource_cache.py`, `model_cache.py`, `resource_specs.py`,
  `protocols.py`. A cache of models and adapters with reference counting and
  optional byte-based eviction.
- Runtime: `model_offloader.py`, `host_component.py`, `block_component.py`,
  the streaming, rolling, and resident block runtimes, `block_compile.py`.
  Activation moves one cached model onto a device and back.
- Host state: `host_param.py`, `host_module.py`, `tensor_adapters.py` and the
  per-format adapters (quanto, GGUF, bitsandbytes, TorchAO, Piper ConvRot,
  DTensor), `tensor_adapter_registry.py`.
- Adapters: `adapter.py`, `lora.py`, `parameter_delta.py`,
  `parameter_value.py`, `parameter_transform.py`, `merge.py`.
- Host memory: `pin_manager.py`, `checkpoint.py`, `_host_registration.py`.
- Experimental DTensor: `communication.py`, `sequential.py`.

## Engineering rules

- Lead with the problem, not the fix: what goes wrong, for whom, how often,
  how badly, and where it shows up. Size the fix to that. A rare or
  hypothetical problem gets a note or a contract, not a mechanism, and
  finding the right problem is most of the work.
- Write only what the production logic needs. Future-proofing for a use we
  can foresee is good; flexibility for a use that will almost certainly
  never come is complexity with no return, and can be added the day it is
  needed.
- Contracts are stated in docstrings and the README, not enforced by
  defensive code. A runtime check is justified only where the failure it
  prevents would be silent or unsafe, and then it is one check at one
  boundary. A guard against a misuse the contract already forbids is a
  documentation change.
- Strong types; let pyright prove what it can. A loose type is contained at
  the boundary and carries a written reason.
- Fix the design, not the symptom. Breaking an internal contract is fine
  when the result is simpler.
- A change must not alter which weights are pinned, or any unrelated
  behavior, as a side effect.

## Contracts that bind callers

- Frozen host bytes are immutable while captured. Set `requires_grad`
  before capture; buffers are never copied back from the device. Factories
  take ownership of compatible CPU allocations and mapped views, and
  callers do not mutate factory-produced tensors afterwards.
- Piper never writes into a file mapping. Trainable parameters are copied
  out at capture, and `merge_adapter()` replaces a mapped target with an
  owned copy before merging.
- Every tensor adapter implements `storage_tensors()` alongside
  `capture_host()` and copies to the device through `transfer_()`.
- Every asynchronous host-to-device transfer runs under a pin lease, and
  `transfer_()` raises otherwise. A lease protects a transfer; it never
  decides pinning. Only transfers that repeat pin: streaming, rolling, and
  the relay. Resident and host uploads lease pageable.
- `max_pinned_bytes` caps pinned bytes only. Everything outside the pinned
  set is reclaimable by the OS, so there is no trim call; lower the cap or
  call `clear()`.
- Checkpoint storage with file provenance pins through an owned copy filled
  by positional reads; the mapping stays read-only page cache. The pinning
  mechanism is chosen statically per platform, never probed at runtime.
- Shared storage is preserved within one streamed block and within host
  state, not across a streamed block or block group boundary.
- Activation and deactivation are the caller's job. A failed activation
  leaves partial state whose only cleanup path is `deactivate()`, and a
  second activation before that raises.

## Vocabulary

One name per concept, taken from the code.

- A *resource* is a cached model or adapter, built from a *spec* by a
  *store* and used through a *binding*. The resource cache's
  reference-counted holds are *resource leases*; they are unrelated to pin
  leases.
- A *component* owns part of a model's host state: a *host component* for
  parameters uploaded whole, a *block component* for a streamed block list
  in one of three *block modes*, streaming, rolling, or resident. An
  activation is a *session*; a *load plan* is what a session transfers.
- *Host state* is a parameter's captured CPU representation, held by a
  `HostParam` and moved by its *tensor adapter*, the per-format protocol.
  An *Adapter* with a capital A is the resource: a set of LoRA factors,
  deltas, or values applied to a model. They are applied by *merge*, in
  place at load time, or *routed*, by hooks on every forward; `merge_adapter()`
  applies them permanently.
- Storage is *pinned* when the manager has *registered* it with the CUDA/HIP
  runtime, either *in place* or through a *copy*, the owned page-aligned
  region filled from a checkpoint file; unregistered storage is *pageable*.
  A *pin lease* protects a set of storages until it closes: a *session
  lease* lasts a component's activation, a *transfer lease* covers one
  transfer. The *budget* is `max_pinned_bytes`. A lease *closes*; an idle
  registration is *evicted*, which *unregisters* it and, for a copy, *frees*
  its region; a registration whose owning tensors are gone is *retired* and
  unregistered once no lease holds it. Storage an acquisition has *reserved*
  under the budget but not yet registered is *pending*; it *settles* by
  registering or being *discarded*, which takes the reservation back.

## Working in the repo

- Tests: `uv run pytest tests -q`. GPU tests need CUDA and skip without it.
- Lint and types: `uv run ruff check .` and `uv run pyright`. CI runs both
  on Linux and Windows.
- Reviews: a finding leads with what goes wrong, how badly, and where, then
  the fix, and is judged against the rules above. One that proposes a guard
  for a misuse the contract forbids is a documentation fix; one that
  proposes a capability needs a foreseeable use.
