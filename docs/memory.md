# Host memory

Piper keeps reusable model state on the CPU and uses a two-sub-tier host
memory hierarchy for repeated transfers. Caching a model, pinning bytes for
DMA, and holding a warm private pageable copy after pin eviction are separate
decisions controlled by independent budgets.

| What is retained | Control | What it accounts for |
|---|---|---|
| Models and adapters | [Cache budget and eviction](models.md#releasing-and-inspecting-resources) | Logical backing bytes in cached stores |
| Tier 2a pinned memory & CUDA/HIP registrations | `host_pin_manager.max_pinned_bytes` | In-place registered OS pages, Tier 2a pinned checkpoint slabs, and `PinnedBounceRing` reservations |
| Combined Tier 2a + Tier 2b managed host memory | `host_pin_manager.max_host_bytes` | Pinned bytes plus Tier 2b private pageable overflow slabs holding evicted checkpoint copies |

These are not limits on total process RSS. Unpinned file mappings and
allocations outside the managed arenas remain under OS page-cache management.

## Set host memory budgets

```python
from piper_offload import host_pin_manager

host_pin_manager.max_pinned_bytes = 4 * 1024**3
host_pin_manager.max_host_bytes = 6 * 1024**3
```

Defaults are derived from physical RAM, bounded on Linux by the tightest cgroup
memory limit (`memory.max` / `memory.limit_in_bytes`) on the process or its
ancestors, and rounded down to whole OS pages:

- **Linux:** `max_pinned_bytes` defaults to 70% of available memory and
  `max_host_bytes` defaults to 80%.
- **Windows:** `max_pinned_bytes` defaults to 45% of physical RAM (leaving headroom
  below Windows WDDM commit limits) and `max_host_bytes` defaults to 75%.

Cgroup discovery assumes `/sys/fs/cgroup`; set explicit budgets on hosts with a
custom mount. If memory discovery fails, defaults fall back to zero with a
warning. Configuration does not initialize CUDA.

Set `max_pinned_bytes` to `0` to disable registrations and the bounce ring, or
`None` to register up to available CUDA/HIP runtime capacity. Reducing
`max_pinned_bytes` evicts idle registrations immediately (demoting eligible
Tier 2a checkpoint copies into Tier 2b when `max_host_bytes` allows); active
leases remain pinned until those leases close.

Set `max_host_bytes` to `0` to disable Tier 2b private overflow retention
without affecting Tier 2a, or `None` to allow unbounded Tier 2b growth. Lowering
`max_host_bytes` immediately evicts least recently used idle Tier 2b copies.
`max_host_bytes` is enforced as a peak ceiling across all pinned and private
slab reservations: idle Tier 2b copies are evicted before allocating new pinned
memory when combined usage would exceed the cap, actively leased Tier 2b copies
reduce available pinned headroom within `max_host_bytes`, and simultaneous
demotion skips or drops copies rather than exceeding `max_host_bytes`.

A pin lease holds storage until a transfer finishes; it does not decide whether
to request registration. Piper's built-in runtimes choose as follows:

| Transfer | Requests new registration |
|---|---|
| Streamed or partially cached offloaded units, experimental host relay | Yes: the transfer repeats |
| `ExpertSlabPool` misses and `prefetch()` with `bypass=False` ([streaming](streaming.md#stream-keyed-slabs-expertslabpool)) | Yes: slots may evict and reload under shared pressure |
| `ExpertSlabPool` misses and `prefetch()` with `bypass=True` | No: pageable lease |
| Resident remainder and all-resident preloaded units (without a shared `VramBudget`) | No: pageable lease |
| CPU execution | No lease |

A pageable lease can hold an existing Tier 2a registration or warm Tier 2b copy;
it requests no new registrations. See [streaming](streaming.md) for offloading
and VRAM budget modes, and [adapters](adapters.md) for adapter usage.

## Inspect and release memory

Inspect the manager's current accounting with:

```python
print(host_pin_manager.stats)
```

| Measurement | Meaning |
|---|---|
| `pinned_bytes` | In-place registered OS pages + `copy_bytes` + `bounce_bytes` |
| `copy_bytes` | Reserved Tier 2a pinned slab arena bytes for checkpoint copies |
| `bounce_bytes` | Reserved `PinnedBounceRing` bytes for pipelined CUDA transfers |
| `private_copy_bytes` | Reserved Tier 2b private pageable slab arena bytes |
| `max_pinned_bytes` / `max_host_bytes` | Configured Tier 2a pin ceiling / combined Tier 2a + Tier 2b host ceiling |
| `registrations` / `idle_registrations` | Current pinned registrations / those available for eviction |
| `active_leases` | Open pin leases, including pageable and Tier 2b leases |
| `registration_failures` / `unregistration_failures` | Runtime registration and cleanup failure counts |

These statistics describe `host_pin_manager` arenas and registrations, not
total RSS or OS file-cache residency.

When a lease closes, its registrations stay in an idle LRU for reuse, and
Tier 2a slabs remain pre-pinned across evictions and refills to avoid repeated
`cudaHostRegister` / `cudaHostUnregister` calls. Dropping the last tensor owner
retires its entry and trims empty slabs once active leases close.

`host_pin_manager.clear()` unregisters idle in-place registrations, releases all
idle Tier 2a and Tier 2b copies, trims empty slabs, and closes the bounce ring,
while keeping active leases valid. Bytes remaining after `clear()` belong to
active leases or retained failed unregistrations.

## Preserve checkpoint backing

Factories transfer ownership of compatible complete pageable CPU allocations
and non-empty views into non-resizable storage. This preserves mapped weights
assigned directly into a model. Device tensors, pinned CPU tensors, partial
views into ordinary resizable allocations, and incompatible layouts are copied
into pageable CPU storage. Quantized formats retain their packed bytes, scales,
and reconstruction metadata. Capture alone never registers memory.

Factory-produced tensors must not be mutated afterwards. Models must be frozen
(`requires_grad=False`) before capture, and captured host bytes are immutable.
`merge_adapter()` copies a mapped target before modifying it. **Piper never
writes into a file mapping.**

[`MappedCheckpoint`](../src/piper_offload/checkpoint.py) reads safetensors and
GGUF files through a read-only mapping and records each storage's `FileSlice`
provenance. Its tensors keep the mapping and file alive after the reader
closes. The file must remain unchanged while any tensor maps it. See
[formats](formats.md) for quantized format support.

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

## Direct I/O, Tier 2a/2b arenas, and bounce streaming

Storage with `FileSlice` provenance is never pinned in place; its file mapping
stays untouched:

1. **Tier 2a pre-pinned slab arena:** Pinning a checkpoint storage allocates a
   span inside a chunked, page-aligned pinned slab (`HostSlabArena`, 128 MiB
   default slabs) and fills it through `DirectFileReader` (`O_DIRECT` on POSIX,
   `FILE_FLAG_NO_BUFFERING` on Windows, with automatic coalescing and buffered
   `preadv` fallback on filesystems such as `tmpfs`). Evicting an idle copy for
   another tensor reuses the existing registered slab without unregistration.
2. **Tier 2b private overflow arena:** When pin-budget pressure or capacity
   retry evicts an idle Tier 2a copy, the manager copies it into a private
   pageable `HostSlabArena` bounded under `max_host_bytes - pinned_bytes` (so
   Windows and Linux remain within physical RAM headroom). A later pinning
   acquisition promotes the copy back into Tier 2a in RAM without re-reading
   disk.
3. **Lazy `PinnedBounceRing`:** Transfers to CUDA from Tier 2b private copies,
   uncached checkpoint `FileSlice` tensors, or eligible pageable CPU tensors
   stream through a pre-pinned circular staging ring (`PinnedBounceRing`,
   disabled when `max_pinned_bytes == 0`). An uncached `FileSlice -> CUDA`
   transfer reads straight from disk through the bounce ring to the GPU without
   leaving extra bytes in host RAM.

Storage without file provenance registers in place so mutable shared relay
buffers and non-file tensors preserve in-place semantics.

## Custom transfers

Custom asynchronous transfers need a lease that lasts until their host reads
finish. Use `transfer_()` so Tier 2a, Tier 2b, and uncached `FileSlice`
sources use the optimal data path:

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
it holds existing registrations or Tier 2b copies, requests no new registrations,
and lets `transfer_()` stream uncached checkpoints through the bounce ring when
enabled. The manager does not synchronize GPU streams during asynchronous
transfers, so retaining and closing the lease at the right time is the caller's
job. `transfer_()` rejects asynchronous reads of registered storage without a
lease, while synchronous (`non_blocking=False`) transfers complete before
returning.

For captured model backing, pass the plain tensors returned by
`HostParam.storage_tensors()` and `HostBuffer.storage_tensors()`. Enumeration
does not copy or pin; it includes quantized payloads and tensor metadata,
delegates DTensor to its local shard, and returns nothing for meta parameters.
For allocation sizes use `tensor.untyped_storage()`: a tensor's `data_ptr()`
and `nbytes` describe only its view. Deduplicate storages when counting them.

### Registration and failure behavior

In-place registrations cover whole storages. Views share their storage's
registration; distinct in-place storages sharing a boundary page count that
page once against the budget. Use views of one storage for aliases: distinct
overlapping byte ranges are rejected. Do not resize storage or register or
unregister it independently while the manager owns it.

A storage left pageable stays pageable until all its active leases close, even
if capacity becomes available. Dropping its tensor owners retires an entry,
which releases after its remaining leases close. Using `vram_budget_bytes=0`
frees each offloaded unit's GPU target slot on unit exit without closing the
session's host pin lease; host pin leases close when the model deactivates.

A runtime capacity failure evicts unrelated idle registrations and retries the
current registration. If capacity remains unavailable, the rest of that
acquisition stays pageable. An invalid-value refusal
affects only that range; later requests still run. Allocation or Direct I/O read
failures also leave affected storage pageable. Unexpected runtime errors,
including foreign registrations, propagate as `HostRegistrationError`; errors
from prior GPU work are not hidden. Failed unregistration retains the range or
slab and its budget charge for a later retry through `clear()` or budget
pressure.

[Back to the README](../README.md)
