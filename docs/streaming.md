# Block streaming

Block streaming runs models whose weights exceed GPU memory by loading blocks
as they execute. It trades transfer time for a smaller GPU working set. Start
with the [README](../README.md) for an overview and [models](models.md) for
capture, caching, and activation.

## Select block groups

Set `OffloadConfig.block_paths` to dotted paths resolving to nonempty
`nn.ModuleList` objects whose block entries are registered submodules of the
root model (unregistered modules in plain Python lists are rejected during
preflight). Each list becomes an independent group. The model's
forward calls its blocks normally, letting Piper's hooks load their weights.
Everything outside those groups stays resident for the activation by default.

This small model shows the required structure; the same setup applies to a
larger model with, for example, a block list at `model.transformer.blocks`:

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
    config=OffloadConfig(block_paths=("blocks",)),
)
with cache.use(spec, device="cuda") as model, torch.inference_mode():
    output = model(torch.randn(1, 8, device="cuda"))

del model
cache.clear()
```

Enter `cache.use()` before `torch.inference_mode()` so background copies can
refill the streamed targets. Keep inference mode around forward computation.

For multiple lists, use
`OffloadConfig(block_paths=("transformer_blocks", "single_transformer_blocks"))`.
Pass the same `OffloadConfig` to `ModelSpec`, `ModelOffloader.from_module()`,
or `ModelStore.bind()` when binding models directly.

Blocks within a group must share parameter and buffer names. Ordinary streaming
supports differing shapes, dtypes, quantization formats, alias topology, and
buffer layouts within that structure. Shared storage is preserved within host
state, including tied embeddings and output heads, or within one streamed block.
Sharing between host and streamed state, between streamed blocks, or between
block groups is unsupported. Use whole-model offloading when those ties must be
preserved. See [`OffloadConfig`](../src/piper_offload/offload_config.py) and
[`ModelStore`](../src/piper_offload/model_store.py).

The GPU must fit the resident state, block targets, activations, and temporary
workspace together. Ordinary streaming keeps an active block and a lookahead
target, overlapping computation with the next whole-block copy. Heterogeneous
groups may also retain reusable targets for distinct layouts, so two targets
are not a universal memory bound. Host pinning is configured separately; see
[memory](memory.md).

With no block paths, activation uploads the whole model. CPU activation uses
the captured CPU state directly: no block target pool, streaming hooks, or
weight copies. It can still use block compilation as described below.

## Choose a block mode

`block_mode` applies to every ordinary and transient block group.

| Mode | CUDA behavior | When to use it |
|---|---|---|
| `"streaming"` (default) | Copies whole blocks through an active/lookahead pool. | General streaming, including heterogeneous groups. |
| `"resident"` | Loads all block targets at activation. | All weights fit, but per-block compilation or transient group release is useful. |
| `"rolling"` | Refills one shared target parameter by parameter during compiled computation. | Supported workloads needing a smaller weight working set. |
| `"auto"` | Selects rolling per compatible group with full-graph compilation; otherwise uses streaming. | Try rolling while allowing unsupported groups to stream normally. |

Resident mode creates no prefetch thread, private copy stream, or block
scheduling hooks. Streaming handles direction changes and traversal wraparound.
Both streaming and rolling load skipped or out-of-order blocks in the
foreground; streaming then resumes prefetching from the block that ran.

### Yield to collectives

Weight copies and transfers of the experimental `piper_relay` process group and
CUDA `ParallelExecutor` (`piper_parallel`) process groups share PCIe/host
transfer resources, and running both at once on the same GPU slows both. While a
`piper_relay` process group is active in the process or a CUDA `piper_parallel`
collective backend targets that GPU, streaming copies the lookahead block one
tensor at a time, pipelining large contiguous tensors in 2 MiB chunks with at
most two chunks (4 MiB) in flight on the copy stream and yielding between chunks
(and between smaller or non-contiguous tensors), so a collective issued inside a
block waits for at most the queued chunks per active background copy stream
rather than the whole tensor or block. When neither is active for that GPU,
lookahead copies enqueue directly without per-chunk or per-tensor stream
synchronizations. Rolling refills cannot wait; use streaming for blocks that
issue collectives.

### Compile block forwards

Compilation is optional. Add one `BlockCompileConfig` to
the spec for all declared groups. Using the factory above:

```python
from piper_offload import BlockCompileConfig, ModelSpec, OffloadConfig

spec = ModelSpec(
    key="compiled-example",
    factory=build_model,
    config=OffloadConfig(
        block_paths=("blocks",),
        block_compile=BlockCompileConfig(),
    ),
)
```

Pass this spec to `cache.use()` as above. These examples use distinct keys
because changing a spec does not replace an already registered cache entry.

Only each distinct block's `forward` is compiled. Module calls and streaming
pre-hooks remain eager, so weight loading finishes before compiled computation.
Original forwards are restored on deactivation; later activations can reuse
cached compiled callables. Without a compile configuration, execution is eager.
External whole-model `torch.compile(model)` or `model.compile()`, and compilation
outside declared block groups, are unsupported.

The backend is Inductor. Defaults are `dynamic=True` and `fullgraph=False`;
`options` accepts Inductor settings and compiler extensions, with a separate
copy of the mapping for each block. There is no `mode` argument. See
[`BlockCompileConfig`](../src/piper_offload/block_compile.py) for the controls.
CPU compilation operates on existing host weights and requires a working C++
toolchain; on Windows, use an x64 MSVC developer shell.

Merge-mode adapters work with compilation. An activation with any routed LoRA
runs all declared blocks eagerly because its child-module hooks stage factors
inside the block forward. Selecting routed mode without an adapter does not
disable compilation. See [adapters](adapters.md).

Compiler errors propagate. Piper does not retry a failed compiled invocation
eagerly: earlier graph segments may already have executed, and retrying could
repeat mutations. PyTorch's graph-break and recompilation behavior still
applies.

Compiler artifact caches and workspace are outside resource byte accounting;
model eviction does not reset the process-wide compiler cache.

### Rolling requirements

Rolling requires full-graph compilation:

```python
spec = ModelSpec(
    key="rolling-example",
    factory=build_model,
    config=OffloadConfig(
        block_paths=("blocks",),
        block_mode="auto",  # use "rolling" to require it
        block_compile=BlockCompileConfig(fullgraph=True),
    ),
)
```

Rolling overlaps computation with transfers into reusable parameter storage.
The shared block target, resident state, activations, and workspace must still
fit together.

Rolling graphs can reuse PyTorch's standard Inductor disk cache after a compiler
reset or process restart. Custom compiler passes must also support caching.
PyTorch controls cache storage and lookup, including `TORCHINDUCTOR_CACHE_DIR`.

Rolling is experimental. Groups require distinct block modules, identical
parameter names/order and layouts, frozen managed parameters, no tied managed
parameters, no streamed buffers, and no zero-sized parameter slots. Supported
representations include regular dense, TorchAO-family, GGUF, trellis-coded,
and Piper ConvRot INT8/NVFP4. DTensor and unreviewed external tensor
adapters use ordinary streaming in auto mode; explicit rolling rejects
unsupported groups. Activation-specific parameter overrides must also be
compatible. See [formats](formats.md) for format and merge restrictions.

Auto falls back for unsupported groups or activation load plans; it does not
recover from compiler failures. Routed-LoRA activations use eager streaming
instead of rolling. On CPU, both modes use ordinary compilation without CUDA
rolling.

The [rolling benchmark](../benchmarks/benchmark_rolling_compile.py) compares
latency, CUDA memory, and output equivalence against compiled whole-block
streaming. Validate your model after changing PyTorch or compiler extensions.
Implementation pointers are in [development](development.md).

## Release working sets during a forward

For inference, transient paths release state before the rest of the model
finishes:

```python
spec = ModelSpec(
    key="transient-example",
    factory=build_model,
    config=OffloadConfig(transient_block_paths=("blocks",)),
)
```

For models with large non-streamed modules, use
`transient_paths=("input_embedder", "output_head")` with paths matching those
modules. A `transient_paths` module owns its non-streamed state recursively and
releases it after its own successful forward. A `transient_block_paths` group releases
its working set after its final block. The selected block mode still applies.
Ordinary `block_paths` retain their working sets throughout the activation;
the same path cannot appear in both block-path lists.

A successful root-model forward reacquires released components for the next
invocation. A skipped component stays acquired. If a forward raises after a
component released, or a transient module is called outside the root model,
the next root forward reacquires that component before running.

Choose boundaries after which that state is no longer used in the current
root forward. A transient block group cannot be traversed again later in that
call, and its module objects must be distinct. Paths must own disjoint state,
including shared-storage aliases. `ModuleList` and `ModuleDict` entries in `transient_paths` are not expanded
into child boundaries: the selected module's own forward is the release point.
Piper does not infer repeated calls, functional parameter access, or
cross-component aliases. These are caller contracts. CPU activation installs
no transient scheduling.
