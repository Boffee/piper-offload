# Offloading and Streaming

Unit offloading runs models whose weights exceed GPU memory (or models whose
active weight working set should fit inside a bounded VRAM budget) by loading
declared units as they execute and retaining hot units up to `vram_budget_bytes`.
Start with the [README](../README.md) for an overview and [models](models.md) for
capture, caching, and activation.

## Select offloaded units

Set `OffloadConfig.offload_paths` to dotted paths resolving to registered
callable `nn.Module` submodules or nonempty `nn.ModuleList` objects of the root
model. Alternatively, call `auto_offload_paths(model, min_unit_bytes=16 * 1024 * 1024)`
to discover outermost `nn.ModuleList` containers and large callable standalone
modules whose unique storage meets `min_unit_bytes` without splitting tied
parameters or selecting non-callable containers (`nn.ModuleDict`,
`nn.ParameterList`, `nn.ParameterDict`). Everything outside `offload_paths`
remains resident for the activation.

This small model shows the required structure; the same setup applies to larger
models with `model.transformer.blocks` or standalone modules such as `embed`
and `head`:

```python
import torch
from torch import nn
from piper_offload import ModelCache, ModelSpec, OffloadConfig


class BlockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(8, 8) for _ in range(3))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def build_model():
    return BlockModel().eval().requires_grad_(False)


cache = ModelCache()
spec = ModelSpec(
    key="streaming-example",
    factory=build_model,
    config=OffloadConfig(offload_paths=("blocks",)),
)
with cache.use(spec, device="cuda") as model, torch.inference_mode():
    output = model(torch.randn(1, 8, device="cuda"))

del model
cache.clear()
```

Enter `cache.use()` before `torch.inference_mode()` so background lookahead
copies can stage weights cleanly. Keep inference mode around forward computation.

For multiple modules and block lists, pass
`OffloadConfig(offload_paths=("embed", "transformer_blocks", "head"))`.
Pass the same `OffloadConfig` to `ModelSpec`, `ModelOffloader.from_module()`,
or `ModelStore.bind()` when binding models directly.

Units within an `nn.ModuleList` may be completely heterogeneous — differing in
shapes, dtypes, quantization formats, alias topology, or parameter and buffer
names — because the VRAM pool keys reusable target allocations by structural
layout signature. Top-level repeated module aliases within one `ModuleList`
(for example looped shared blocks) are deduplicated to one unit; shared
stateless submodules (such as `nn.SiLU()`) and shared read-only buffers (such
as rotary tables) across distinct units are supported. Sharing parameter storage
or stateful/buffer submodule slots across offload boundaries is unsupported; use
whole-model resident offloading (`offload_paths=()`) when cross-boundary ties must
be preserved. Keep direct unit parameter/buffer accesses inside the unit's own
forward, or remember that invoking another offloaded unit returns the previous
unit to host storage. See
[`OffloadConfig`](../src/piper_offload/offload_config.py) and
[`ModelPartition`](../src/piper_offload/model_partition.py).

With `offload_paths=()`, activation uploads the whole model once without pinning.
CPU activation uses the captured CPU state directly: no VRAM pool, offloading
hooks, or weight copies.

## Control the VRAM budget

`vram_budget_bytes` caps the simultaneous **decoded GPU target bytes of
offloaded units only** (excluding non-offloaded resident weights, activations,
KV cache, and transient decode scratch; note packed formats such as GGUF and
Trellis decode to larger ConvRot INT8 GPU targets):

| `vram_budget_bytes` | CUDA behavior | When to use it |
|---|---|---|
| `None` (default) | Strict 2-slot streaming (one active + one lookahead slot), reusing same-signature allocations. | Minimize weight VRAM regardless of model size. |
| `0` | Releases the active unit's CUDA slot immediately on unit exit and disables async lookahead. | Tightest possible peak VRAM across non-overlapping stages. |
| `0 < budget < total` | Retains hot units up to `budget`, evicting furthest-future-use idle slots (Belady MIN); async lookahead runs when `active_bytes + next_bytes <= budget` (if `budget` is smaller than one active unit, forward still runs and releases the oversized slot on unit exit). | Accelerate multi-step or looped workloads within a fixed weight budget. |
| `budget >= total` | Without a shared `vram_budget`, preloads all offloaded units on activation using pageable (non-pinned) transfers and no lookahead thread; when `vram_budget` is also supplied, keeps units dynamically pooled and evictable with host pinning so peer allocations can reclaim space. | Weights fit cleanly in budget, avoiding host pinning and async lookahead when running standalone. |

Offload learns arbitrary execution orders (sequential, reverse, skipped, looped,
or multi-branch) via an online suffix predictor (`k <= 4`) and overlaps
computation with the predicted next unit's H2D copy when budget/ceiling permits.
To run speculative or draft passes (such as MTP) without overwriting the
predictor's main-pass history, wrap the draft forward in
`with offloader.no_record_trace():`. When adapters (LoRA or deltas) are
active, parked cached slots retain their merged GPU weights across cache hits
without re-merging or re-copying.

To choose a different budget per inference run without rebuilding `ModelStore`
or `ModelSpec`, pass `vram_budget_bytes=...` directly to
`ModelOffloader.activate(..., vram_budget_bytes=...)` or
`cache.use(..., vram_budget_bytes=...)`; omitted activations fall back to
`OffloadConfig.vram_budget_bytes`.

### Share VRAM and transfer streams across runtimes

When multiple offloaded models or external GPU pools (such as KV caches or
arenas) share one CUDA GPU:

- **`vram_budget` (`VramBudget`)**: Pass a shared per-GPU `VramBudget` to
  `ModelOffloader.activate(..., vram_budget=...)` or `cache.use(..., vram_budget=...)`
  (ignored on CPU). On CUDA activation, non-offloaded resident weights
  automatically reserve their actual device bytes (raising
  `VramBudgetExhaustedError` if they do not fit), and offloaded units register as
  an evictable tenant (`priority=10`, `yield_to_peers=True`). Idle weight slots
  yield both to lower-retention reservations/tenants and to equal-priority
  offloaded models sharing the same `VramBudget`, capped at
  `min(vram_budget_bytes, dynamic_ceiling)` (or `dynamic_ceiling` when
  `vram_budget_bytes is None`).
- **`transfer_context` (`DeviceTransferContext`)**: Pass a shared per-device
  `DeviceTransferContext(device)` to reuse one CUDA copy stream and lazy staging
  worker across concurrent models on the same GPU. Active sessions borrow the
  context (refusing `ctx.close()` until all borrowers deactivate).

## Stream keyed slabs (`ExpertSlabPool`)

When an MoE runtime (or another keyed collection of fixed-schema experts or
blocks) needs slot-level streaming without embedding `piper-offload` inside model
definitions, define a format-agnostic `SlotSchema`, pair a `Uint8DualArena` with
an `ExpertSlabPool`, and iterate yielded waves around the kernel call:

```python
import torch
from piper_offload import ExpertSlabPool, SlotSchema, Uint8DualArena

schema = SlotSchema(
    {"up_gate": max_up_bytes, "down": max_down_bytes},
    {"up_gate_scale": (torch.float32, (up_out,))},
)
with (
    Uint8DualArena(arena_bytes, device="cuda", budget=budget) as arena,
    ExpertSlabPool(arena, schema, transfer_context=transfer_context) as pool,
):
    pool.register_group("layer0", experts)  # frozen CPU plain-tensor mappings per group

    with pool.waves("layer0", active_expert_ids) as waves:
        for wave in waves:
            run_expert_kernel(
                wave.arena_views,     # field -> strided [max_slots, *shape] uint8 view
                wave.side_views,      # field -> strided [max_slots, *shape] typed pool
                                      #   (aligned stride(0), contiguous per-slot slice)
                wave.arena_bytes,     # 1-D uint8 arena for slot_offsets indexing
                wave.slot_offsets,    # field -> int64[max_slots] byte offsets in wave.arena_bytes
                wave.expert_to_slot,  # int32[num_experts] wave slots (-1 otherwise;
                                      #   all group experts on the zero-sync fast path)
                accumulate=wave.index > 0,
            )
```

Key behaviors and contracts of [`ExpertSlabPool`](../src/piper_offload/expert_slab_pool.py)
and [`Uint8DualArena`](../src/piper_offload/dual_arena.py):

- **Fixed aligned drawers:** Every low slot holds `schema.drawer_bytes` aligned
  bytes (requiring `arena.alignment >= schema.field_alignment` and
  `slot_stride % schema.field_alignment == 0`), while transfers copy only each
  expert's actual `src.nbytes`. Trailing bytes beyond `src.nbytes` are left
  uninitialized across slot reuse.
- **Wave staging & CUDA stream lifetime:** Cache hits execute first when
  available (`is_hit_wave=True`), while misses stage on the CUDA copy stream via
  `PinManager` and execute in subsequent accumulating waves. When free or
  transient slot headroom exists, the next miss wave stages ahead behind active
  execution; when all `max_slots` hold resident experts, subsequent miss waves
  stage once active wave slots finish and release. Pass `transfer_context` to
  share one CUDA copy stream across components on the same device, and pass
  `budget` (which `ExpertSlabPool` inherits from `arena.budget` by default; note
  the parameter name `budget` rather than `ModelOffloader`'s `vram_budget`) to
  coordinate `TypedSidePool` allocations inside `VramBudget`. Kernels reading a
  wave must launch on the CUDA stream active when fetching that wave, before
  requesting the next wave or exiting `with pool.waves(...)`.
- **Decode caching vs. prefill / speculative isolation:** Decode (`bypass=False`)
  uses W-TinyLFU frequency admission (`pin=True`), borrowing at most one resident
  slot per rejected cold miss wave so hot experts survive cold spikes. Prefill
  passes can set `bypass=True` to use a bounded transient window (default
  `max(1, max_slots // 4)`, configurable via `default_bypass_wave_slots` or
  `max_wave_slots`) with pageable leases (`pin=False`) instead of forcing new host
  registrations. Speculative passes can run drafts under
  `with pool.no_record_stats():` and promote accepted keys with
  `pool.commit_accepted(...)`. Call `pool.prefetch(...)` to warm misses ahead
  of execution without reclaiming high-end peers. When every expert in a group
  occupies a `READY` slot and either stats recording is off or total registered
  experts fit within `max_slots`, `waves()` returns one fast-path wave covering
  `0..num_experts-1` and skips CPU readbacks of non-empty integer `expert_ids`
  tensors (even all-`-1` tensors; all-sentinel Python iterables still yield 0
  waves). Out-of-range integer IDs (`< 0` or `>= num_experts`) are ignored as
  sentinels.
- **Host immutability, read-only waves & teardown:** `register_group()` requires
  frozen (`requires_grad=False`) CPU plain tensors (`torch.Tensor` or frozen
  `nn.Parameter`) and retains compatible storage immutably via
  `capture_host_tensor`. Treat all yielded `ExpertWave` tensors as strictly
  read-only. Close `pool` outside active `waves()` sessions and before closing
  `arena`.
- **Shared frontier negotiation:** `Uint8DualArena` coordinates low-end and
  high-end fixed slots across one frontier via `set_low_reclaimer` and
  `set_high_reclaimer` callbacks invoked outside the arena lock. On high-end
  pressure, `ExpertSlabPool` relocates resident frontier experts into lower
  idle or colder slots when possible, or otherwise evicts unleased frontier
  slots without disturbing active wave readers.

## Manage VRAM KV chunks (`KvChunkPool`)

`KvChunkPool` manages GPU-born quantized KV chunks on the descending high end of
a shared `Uint8DualArena` alongside typed `TypedSidePool` side pools described by
a `SlotSchema`. By default `KvChunkPool` inherits `arena.budget` for its side
pools (or accepts an explicit `budget=` argument).

```python
import torch

from piper_offload import KvChunkPool, SlotSchema, Uint8DualArena

kv_schema = SlotSchema(
    {"key_int8": (2, 1024, 64), "value_int8": (2, 1024, 64)},
    side_fields={"key_scale": (torch.float32, (2, 1024))},
)
with (
    Uint8DualArena(128 * 1024 * 1024, device="cuda", budget=budget) as arena,
    KvChunkPool(arena, kv_schema, max_slots=64) as kv_pool,
):
    with kv_pool.produce(["c0", "c1"]) as batch:
        freeze_kv_chunks(raw_k, raw_v, batch.arena_views, batch.side_views, batch.slot_ids)
    with kv_pool.read(["c0", "c1", None]) as batch:
        # None sentinels map to -1 in batch.slot_ids; duplicate keys share one slot.
        attn_out = decode_kv(query, batch.arena_views, batch.side_views, batch.slot_ids)
    kv_pool.discard(["c0"])
```

- **VRAM-only capacity (no host spilling):** `KvChunkPool` keeps all resident KV
  chunks on the device (`ArenaCapacityError` is raised if `produce()` cannot fit
  in the remaining high slots). It does not spill KV chunks to host RAM or copy
  live KV chunks during reclaim.
- **Persistent views, stream ordering, and teardown:** `arena_views`,
  `side_views`, `arena_bytes`, and `slot_offsets` stay invariant across the
  lifetime of the pool so `torch.compile(fullgraph=True)` graphs remain stable.
  Kernels populating or reading a yielded `KvBatch` must enqueue work on the
  CUDA stream active when entering `produce()` or `read()` before exiting its
  `with` block; `produce()` commits keys atomically on clean exit, while
  `read()` deduplicates repeated prefix keys in one `KvBatch` and tracks CUDA
  consumer streams until exit. Close `kv_pool` outside active `produce()` /
  `read()` sessions and before closing `arena`.
- **Non-blocking retirement and peer frontier sharing:** `discard(keys)` marks
  retired keys empty without blocking on CUDA streams (raising `RuntimeError`
  atomically if any key is currently producing or has active readers).
  Top-down slot reuse keeps discarded bottom slots near the arena frontier, where
  `ExpertSlabPool` can reclaim unleased `EMPTY` bottom slots on low-end pressure
  (stopping cleanly if a live `READY` or active chunk sits at the frontier).

## Yield to collectives

Weight copies and transfers of the `piper_relay` process group and CUDA
`ParallelExecutor` (`piper_parallel`) process groups share PCIe/host transfer
resources, and running both at once on the same GPU slows both. While a
`piper_relay` process group is active in the process or a CUDA `piper_parallel`
collective backend targets that GPU, lookahead streaming copies one tensor at a
time, pipelining large contiguous tensors in 2 MiB chunks with at most two chunks
(4 MiB) in flight on the copy stream and yielding between chunks (and between
smaller or non-contiguous tensors), so a collective issued inside a unit waits
for at most the queued chunks per active background copy stream rather than the
whole tensor or unit. When neither is active for that GPU, lookahead copies
enqueue directly without per-chunk or per-tensor stream synchronizations.

## Compile unit forwards

Compilation is optional. Add `block_compile=BlockCompileConfig()` to
`OffloadConfig` to compile each distinct offloaded unit's `forward`:

```python
from piper_offload import BlockCompileConfig, ModelSpec, OffloadConfig

spec = ModelSpec(
    key="compiled-example",
    factory=build_model,
    config=OffloadConfig(
        offload_paths=("blocks",),
        block_compile=BlockCompileConfig(),
    ),
)
```

Pass this spec to `cache.use()` as above. Original forwards are restored on
deactivation; later activations can reuse cached compiled callables. Without
a compile configuration, execution is eager. External whole-model
`torch.compile(model)` or `model.compile()` is unsupported.

The backend is Inductor. Defaults are `dynamic=True` and `fullgraph=False`;
`options` accepts Inductor settings and compiler extensions, with a separate
copy of the mapping for each unit. See
[`BlockCompileConfig`](../src/piper_offload/block_compile.py) for controls.
Adapters merge before execution and work with compilation. CPU compilation
operates on existing host weights and requires a working C++ toolchain.

[Back to the README](../README.md)
