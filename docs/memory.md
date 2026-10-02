# Host memory

Piper keeps reusable model state on the CPU and pins selected storage for
repeated GPU transfers. Caching a model, pinning its bytes, and keeping an
evicted copy on Windows are separate decisions.

| What is retained | Control | What it accounts for |
|---|---|---|
| Models and adapters | `ModelCache.evict()` / `clear()` | Cached resources; `ModelCache` is unbounded |
| CUDA/HIP registrations | `host_pin_manager.max_pinned_bytes` | Pinned OS pages, including pending reservations |
| Offered checkpoint copies on Windows | `host_pin_manager.max_offered_bytes` | Committed pages that Windows may discard |

These are not limits on total process memory. In particular, unpinned mapped
pages remain reclaimable by the OS while the model stays cached. The lower-level
`ResourceCache` supports byte-based resource eviction; see [models](models.md).

## Set the pin budget

```python
from piper_offload import host_pin_manager

host_pin_manager.max_pinned_bytes = 4 * 1024**3
```

The default is half of physical RAM, limited on Linux by the tightest cgroup
limit on the process or its ancestors, and rounded down to OS pages. Cgroup
discovery assumes `/sys/fs/cgroup`; set the budget explicitly for another
mount. If memory discovery fails, the default is zero with a warning.
Configuration does not initialize CUDA.

Set the budget to `0` to disable new registrations, or `None` to register up
to available CUDA/HIP runtime capacity. Reducing it evicts idle registrations
immediately; storage held by active leases stays registered until those leases
close.

A pin lease holds storage until a transfer finishes; it does not decide whether
to register that storage. Piper's built-in runtimes choose as follows:

| Transfer | Requests new registration |
|---|---|
| Streaming blocks, rolling blocks, experimental host relay | Yes: the transfer repeats |
| Resident blocks, host components, optimizer copy-back | No: pageable lease |
| Routed LoRA factors | No: pageable lease while hooks are installed |
| CPU execution | No lease |

A pageable lease can hold a registration that already exists; it requests no
new registrations. See [streaming](streaming.md) for block modes and
[adapters](adapters.md) for routed factors.

## Inspect and release memory

Inspect the manager's current accounting with:

```python
print(host_pin_manager.stats)
```

| Measurement | Meaning |
|---|---|
| `pinned_bytes` | Union of registered and pending OS pages |
| `copy_bytes` | Portion held by checkpoint copies, filling or registered |
| `registrations` / `idle_registrations` | Current registrations / those available for eviction |
| `active_leases` | Open pin leases, including pageable leases |
| `registration_failures` / `unregistration_failures` | Runtime registration and cleanup failure counts |
| `offered_bytes` | Windows offered commitment, separate from pinned bytes |

These statistics describe the manager, not total RSS, file-cache residency, or
the number of offered pages Windows has discarded.

Pinned bytes remaining after deactivation usually mean idle retention. When a
lease closes, its registrations enter an idle LRU for reuse, avoiding repeated
register/unregister calls. Budget pressure evicts idle registrations; active
leases protect both pinned and pageable storage.

`host_pin_manager.clear()` releases idle registrations and all offered copies,
while active leases remain valid. Bytes remaining after `clear()` can belong to
active leases or failed cleanup. There is no separate trim operation: lower the
pin budget or clear idle registrations. To release model and adapter backing,
evict inactive cache entries and drop references retained outside the cache;
see [models](models.md#releasing-and-inspecting-resources).

## Preserve checkpoint backing

Factories transfer ownership of compatible complete pageable CPU allocations
and non-empty views into non-resizable storage. This preserves mapped weights
assigned directly into a model. Device tensors, pinned CPU tensors, partial
views into ordinary resizable allocations, and incompatible layouts are copied
into pageable CPU storage. Quantized formats retain their packed bytes, scales,
and reconstruction metadata. Capture alone never registers memory.

Factory-produced tensors must not be mutated afterwards. Frozen host bytes are
immutable while captured; set `requires_grad` before capture. Trainable mapped
parameters are copied into owned memory at capture, and `merge_adapter()` copies
a mapped target before modifying it. Buffers are never copied back from the
device. **Piper never writes into a file mapping.**

[`MappedCheckpoint`](../src/piper_offload/checkpoint.py) reads safetensors and
GGUF files through a read-only mapping and records each storage's file slice.
Its tensors keep the mapping and file alive after the reader closes. The file
must remain unchanged while any tensor maps it. See [formats](formats.md) for
quantized format support.

For a checkpoint whose keys match the model, a factory can assign mapped
weights directly. Here `build_model_on_meta()` is the application's model
constructor, using meta parameters to avoid allocating initial weights:

```python
from piper_offload import MappedCheckpoint


def load_model():
    model = build_model_on_meta()
    with MappedCheckpoint("model.safetensors") as checkpoint:
        state = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
        model.load_state_dict(state, assign=True)
    return model.eval().requires_grad_(False)
```

Pass `load_model` as the `ModelSpec` factory. The same reading interface accepts
GGUF; its packed parameters also have the [execution requirements](formats.md#load-a-gguf-checkpoint)
described in the format guide.

Storage with this file provenance pins through an owned, page-aligned copy,
filled by parallel file reads. Transfers use that registered copy; the model's
CPU tensors and `state_dict()` still refer to the original mapping. Eviction
unregisters and frees the copy, or offers it on Windows as described below.
The mapping stays read-only page cache throughout.

Storage without file provenance registers in place. This includes private file
mappings from other loaders: registration can force private copies of their
pages that persist until the mapping is released. Preserving a mapping alone
therefore does not provide the same memory behavior as `MappedCheckpoint`.

## Optional Windows retention

`host_pin_manager.max_offered_bytes` defaults to zero. A positive budget enables
a Windows tier for evicted checkpoint copies; it has no effect elsewhere.
Measure model-switch time and disk reads with the budget both enabled and zero.
The tier is most useful when refilling would read from disk. A warm file cache
can make freeing and refilling comparably fast, and a working set that exceeds
RAM may leave no offered pages to reuse.

Only eviction to make room—for pin-budget pressure, a budget reduction, or a
runtime capacity retry—may offer a copy, and only after successful
unregistration. Offering removes its pages from the working set and permits
Windows to discard them. The copy remains committed and charges the separate
offered budget. Least recently offered copies are freed to admit new ones or
reduce that budget.

Offered pages are unreadable until reclaimed. A later acquisition reclaims the
copy, reuses intact spans, and refills discarded spans before registering it.
Spans are 16 MiB so one discarded page does not force a whole-copy refill.
Transfers never resolve to offered memory.

`clear()`, owner retirement, an oversized copy, or a surviving external view of
the copy causes it to be freed rather than retained. A failed offer also frees
it; a failed reclaim rebuilds it. Allocation failures free offered copies in
age order and retry allocation.

Discarded pages still consume commitment. The tier frees every offered copy
when Windows signals `MaximumCommitCondition`, watching while it retains
copies and checking before admission and allocation. It does not estimate
pressure from the current commit limit, which Windows can grow on demand. If
the event cannot be opened, a warning is emitted once and only the configured
budget limits retention. Setting the budget to zero releases offered commitment.

## Custom transfers

Custom asynchronous transfers need a lease that lasts until their host reads
finish. Use `transfer_()` so checkpoint sources resolve to their registered copy:

```python
import torch
from piper_offload import host_pin_manager, transfer_

source = torch.randn(1024, 1024)
target = torch.empty_like(source, device="cuda")
stream = torch.cuda.Stream()

# Request pinning for storage you will transfer repeatedly.
with host_pin_manager.acquire([source]) as lease:
    with torch.cuda.stream(stream):
        transfer_(target, source, non_blocking=True)
    stream.synchronize()
    print(lease.registered_bytes, lease.pageable_bytes)
```

The lease's `registered_bytes` and `pageable_bytes` count unique requested
storage bytes without page rounding. Use `pin=False` for one-time transfers;
it permits existing registrations but requests no new ones. The manager does
not synchronize GPU streams, so retaining and closing the lease at the right
time is the caller's job. `transfer_()` rejects asynchronous reads of registered
storage without a lease. A synchronous transfer completes under the manager's
lock and needs no lease.

For captured model backing, pass the plain tensors returned by
`HostParam.storage_tensors()` and `HostBuffer.storage_tensors()`. Enumeration
does not copy or pin; it includes quantized payloads and tensor metadata,
delegates DTensor to its local shard, and returns nothing for meta parameters.
For allocation sizes use `tensor.untyped_storage()`: a tensor's `data_ptr()`
and `nbytes` describe only its view. Deduplicate storages when counting them.

### Registration and failure behavior

Registrations cover whole storages. Views share their storage's registration;
distinct storages sharing a boundary page count that page once against the
budget. Use views of one storage for aliases: distinct overlapping byte ranges
are rejected. Do not resize storage or register or unregister it independently
while the manager owns it.

A storage left pageable stays pageable until all its active leases close, even
if capacity becomes available. Dropping its tensor owners retires a registration,
which is released after its remaining leases close. `BlockComponent.release()`
also closes its lease during a temporary working-set release; `acquire()` obtains
one again.

A runtime capacity failure evicts unrelated idle registrations and retries the
current registration. If capacity remains unavailable, the rest of that
acquisition stays pageable. An invalid-value refusal affects only that range;
later requests still run. Allocation or copy-fill failures also leave affected
storage pageable. Unexpected runtime errors, including foreign registrations,
propagate as `HostRegistrationError`; errors from prior GPU work are not hidden.
Failed unregistration retains the storage and its budget charge for a later
retry through `clear()` or budget pressure.

## Implementation notes

These ordering rules keep copy ownership safe and avoid slow registration or
needless discard under pressure:

- [`pin_manager.py`](../src/piper_offload/pin_manager.py) owns leases, page
  accounting, registration order, and the shared lock. The memory component
  owns allocation and released copies; it must not retain the manager.
  Ownership transfers to the manager at reservation and back only after
  successful unregistration.
- [`_host_memory.py`](../src/piper_offload/_host_memory.py) chooses the component
  once at import; pinning mechanisms are not probed at runtime. Copies use
  anonymous mappings on Linux and `VirtualAlloc` on Windows. Parallel fills
  use positional reads where available and a file handle per worker otherwise.
- [`_copy_memory.py`](../src/piper_offload/_copy_memory.py) touches a byte per
  page before registration. This brings pages into the working set without
  making `cudaHostRegister` fault them individually. The first unfilled copy
  is filled and registered before filling the rest, to detect exhausted runtime
  capacity before paying for all remaining reads.
- [`_copy_memory_windows.py`](../src/piper_offload/_copy_memory_windows.py)
  reclaims **all requested offered copies before any fill**: a fill's own
  memory demand could otherwise discard copies the acquisition was about to
  reuse. As reclaimed copies are yielded in request order, registration
  processes the ready request prefix, then that intact copy, without waiting
  for earlier fills. This preserves ready in-place requests' priority and locks
  intact pages before fills can push them out. Reservations follow request order; intact copies
  may obtain runtime capacity ahead of unfilled ones.
- Each offer starts as its own unregistration succeeds, overlapping offers
  with serial runtime cleanup. Preparation and release batches wait for and
  join workers only outside the shared lock: worker finalizers may need it.
  Reclaim readiness is published after its worker is joined; preparation is
  closed before pending copies are freed.
- Windows offers at normal offer priority. While the tier is enabled, every
  fill reads through the file cache at the lowest memory priority, including
  the first load before anything has been offered. Each worker first faults
  its destination slice at its normal priority, so only the file-cache pages
  receive the low priority. Otherwise the cache outranks offered copies, or
  the copy's own pages become the first to be trimmed. With the tier disabled,
  reads use normal priority to preserve warm refills. The native calls live in
  [`_host_memory_windows.py`](../src/piper_offload/_host_memory_windows.py).

[Back to the README](../README.md)
