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
| `budget >= total` | Preloads all offloaded units on activation using pageable (non-pinned) transfers and no lookahead thread. | Weights fit cleanly in budget, avoiding host pinning and async lookahead while keeping unit compilation. |

Offload learns arbitrary execution orders (sequential, reverse, skipped, looped,
or multi-branch) via an online suffix predictor (`k <= 4`) and overlaps
computation with the predicted next unit's H2D copy when budget/ceiling permits.
When merged adapters (LoRA or deltas) are active, parked cached slots retain
their merged GPU weights across cache hits without re-merging or re-copying.

To choose a different budget per inference run without rebuilding `ModelStore`
or `ModelSpec`, pass `vram_budget_bytes=...` directly to
`ModelOffloader.activate(..., vram_budget_bytes=...)` or
`cache.use(..., vram_budget_bytes=...)`; omitted activations fall back to
`OffloadConfig.vram_budget_bytes`.

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
Merge-mode adapters work with compilation; routed LoRA activations run units
eagerly. CPU compilation operates on existing host weights and requires a working
C++ toolchain.

[Back to the README](../README.md)
