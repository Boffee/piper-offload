# Weight formats

Piper's tensor adapters manage capture, movement, and reconstruction of each
supported weight representation. Offloading a format does not imply that it
supports additive updates. See [adapters](adapters.md) for applying
LoRA, deltas, and complete values, [streaming](streaming.md) for execution modes,
and the [README](../README.md) for the starting point.

## Choose a supported representation

The table lists the extras required to execute each representation. Reading a
GGUF checkpoint with `MappedCheckpoint` requires only the base install.
Install `piper-offload[all]` for every integration. The supported dependency
ranges are maintained in [pyproject.toml](../pyproject.toml). Native execution still requires the
format's compatible hardware and PyTorch stack.

The `torchao` extra also covers Piper ConvRot. On Linux, PyTorch supplies its matching
CUDA or ROCm Triton; the composable `triton` extra adds `triton-windows` on 64-bit Windows.
Most integrations have portable fallbacks without Triton; GGUF conversion and
trellis decoding require it. Windows GPU execution
requires Windows 10 or 11, a supported NVIDIA GPU and current driver, and the
Visual C++ Redistributable for Visual Studio 2015–2022. A separate CUDA toolkit
or Visual Studio installation is not required for eager GPU execution.

| Weight representation | Extra | LoRA merge | Dense / mixed merge |
|---|---|---|---|
| Plain floating point, excluding plain float8 | None | Native | Native |
| TorchAO scaled FP8 | `torchao` | Yes | Yes |
| TorchAO static-activation scaled FP8 | `torchao` | Yes | Yes |
| TorchAO INT8 | `torchao` | Yes | Yes |
| TorchAO MXFP8 / MXFP4 | `torchao` | Yes, layout restrictions | Yes, layout restrictions |
| TorchAO NVFP4 | `torchao` | Yes, layout restrictions | Yes, layout restrictions |
| TorchAO INT4 tile-packed | `torchao` | No | No |
| Piper ConvRot INT8 | `torchao` | Yes, matrices only | Yes, matrices only |
| Piper ConvRot NVFP4 | `torchao` | Yes, layout restrictions | Yes, layout restrictions |
| Packed GGUF → ConvRot INT8 | `gguf` | Activation only | Activation only |
| Trellis-coded → ConvRot INT8 | `torchao`, `triton` | Activation only | Activation only |
| DTensor | Depends on local shard | Delegates to local adapter | Delegates to local adapter |

Plain float8 storage can move but is not an additive merge target. Scaled FP8
wrappers have their own merge implementations. Routed LoRA requires no merge
capability, so it also works with tile-packed INT4 when the owning module is a
compatible logical `nn.Linear`.

[Broadcast additive updates](adapters.md#broadcast-additive-updates) are
supported by plain floating-point targets, excluding plain float8. Built-in
quantized and DTensor adapters require each source update to match the full
logical target shape and reject broadcast requests during preflight.

Quantized additive merges recompute data-dependent weight scales and preserve
activation calibration metadata. Optimized merges keep source scaling and
addition in FP32 until the terms are combined, avoiding intermediate overflow
or loss of small residuals. All merge-capable built-in quantized formats support
stochastic rounding. See [adapter rounding](adapters.md#choose-merge-or-routed-lora)
for configuration and reproducibility. Backend selection is documented with
each format's adapter.

## Layout limits

Transposed MX and NVFP4 packed targets cannot be re-encoded into their existing
storage; use routed LoRA. Scaled FP8 supports transposed per-row and per-tensor
scales, but not transposed per-group scales. TorchAO INT8 does not support
transposed weights; Piper ConvRot INT8 also rejects transposed matrices at
capture. Other layouts support merging only when the format can re-encode them.

Complete [parameter values](adapters.md#populate-optional-meta-parameters)
can copy supported physical representations even when additive merge is
unavailable. Scaling and mixed-update requirements belong to the
[composition contract](adapters.md#populate-optional-meta-parameters).

## Format details

**TorchAO FP8.** The weight-only and dynamic-activation path preserves
`Float8Tensor` metadata with per-group, per-row, or per-tensor scales. FP8
matmul requires compatible SM89+ CUDA hardware. The static-activation adapter
supports `PrototypeFloat8Tensor` with per-tensor weight and activation scales.
Its linear dispatch accommodates scalar or one-element checkpoint activation
scales for ordinary 2-D and 3-D inputs. Output activation quantization and other
Prototype layouts are unsupported. See
[float8_adapter.py](../src/piper_offload/float8_adapter.py) and
[static_float8_adapter.py](../src/piper_offload/static_float8_adapter.py).

**TorchAO MX and NVFP4.** MX supports E4M3/E5M2 MXFP8 and packed MXFP4; MXFP6
is unsupported. These formats' matmul paths depend on suitable Blackwell-class
hardware and CUDA software. Preserved metadata includes block layout, scale
swizzling, and dispatch settings. MX re-encoding uses the recorded scale mode
when available, otherwise TorchAO's default. See
[mx_adapter.py](../src/piper_offload/mx_adapter.py) and
[nvfp4_adapter.py](../src/piper_offload/nvfp4_adapter.py).

**Piper ConvRot.** INT8 and NVFP4 wrappers retain their rotation groups and
quantization metadata; Piper Kernels owns their arithmetic. Static-scale INT8
Conv3D weights also move, including activation scales, but require contiguous,
untransposed storage and accept neither LoRA nor dense updates. Their logical
shape is `[out, in, 3, 3, 3]`. See
[piper_convrot_int8_adapter.py](../src/piper_offload/piper_convrot_int8_adapter.py)
and [piper_convrot_nvfp4_adapter.py](../src/piper_offload/piper_convrot_nvfp4_adapter.py).

## Load a GGUF checkpoint

`MappedCheckpoint` reads little-endian GGUF versions 2 and 3. Use `keys()` to
list names, including entries whose encoding Offload cannot load. Request only
the names your model needs: requesting an unsupported descriptor or tensor
raises `CheckpointError`. GGUF `metadata()` returns `None`.

To inspect a supported weight without loading it into a model:

```python
from piper_offload import MappedCheckpoint

with MappedCheckpoint("model.gguf") as checkpoint:
    weight = checkpoint.get_slice("weight")
    print(weight.get_shape(), weight.get_dtype(), weight.get_nbytes())
```

The shape and dtype describe the logical tensor. The byte count measures its
checkpoint storage. For a quantized entry, `get_tensor()` returns a
`GgufParameter` with BF16 compute dtype and `as_tensor()` exposes its packed
bytes. F32, F16, and BF16 entries return ordinary tensors.

Load those entries with `model.load_state_dict(state, assign=True)` and freeze
the model before capture. Encoded rows must remain intact when splitting or
regrouping weights. Contiguous row splits keep their file provenance; copying
operations create independent storage. See [memory](memory.md) for the mapped
checkpoint loading pattern.

Packed GGUF parameters require Offload activation to execute;
`requires_activation(parameter)` lets loaders detect this. Host backing stays
packed; active weights use BF16 ConvRot INT8 without allocating a full dense
weight during conversion. The logical input width must be divisible by 16.

Direct conversion supports F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, and IQ4_XS. The `gguf` extra installs
TorchAO and Triton for this conversion. LoRA and dense updates apply to the
active ConvRot target; permanent merge cannot modify the packed GGUF source.

Externally supplied Diffusers `GGUFParameter` objects are recognized without a
Diffusers dependency. Their GGUF linear modules must first be normalized to
ordinary linear behavior. See [gguf_parameter.py](../src/piper_offload/gguf_parameter.py)
and [gguf_adapter.py](../src/piper_offload/gguf_adapter.py).

## Load trellis-coded weights

Trellis coding stores a ConvRot INT8 matrix at 3 to 6 bits per weight; Piper
Kernels owns the format and its encoder, `TrellisStorage.from_hp`. Wrap each
weight's stored tensors and block size, as `TrellisStorage` holds them, in a
`TrellisParameter` with the ConvRot group size, the logical dtype, and any
static activation scale. For per-row bitrates within a matrix, pass
`row_widths` (`uint8 [out]`, giving bytes per block for each row) with
row-concatenated 1-D `bitstreams` (`uint8` of at least
`(in / block_size) * sum(row_widths)` bytes); CPU construction and host capture
validate width bounds (`3` to `6` code bits per weight) and minimum bitstream
capacity, and variable-row decoding requires a Piper Kernels release that
accepts `row_widths` (added in Piper Kernels PR #320, after `0.8.0rc6`). Split or
regroup checkpoint storage first: the parameter supports only detach and clone.
Assign the parameters to the model and freeze it before capture.

Host backing stays packed, and `requires_activation(parameter)` is true. Each
refill transfers the packed tensors and decodes them into a reusable ConvRot
INT8 target, without allocating a dense weight; decoding requires a Triton
accelerator that Piper Kernels supports. LoRA and dense updates apply to the
active target; permanent merge cannot modify the packed source. See
[trellis_parameter.py](../src/piper_offload/trellis_parameter.py) and
[trellis_adapter.py](../src/piper_offload/trellis_adapter.py).

## DTensor and custom formats

DTensor support composes a distributed wrapper with the adapter for its local
shard. It targets frozen inference with ordinary `Replicate` and contiguous
`Shard` placements across 1D and N-D CPU or CUDA/HIP `DeviceMesh` topologies.
When binding an unsharded `ModelStore` onto a `DTensor`-sharded shell, plain
strided weights project zero-copy local views for any `Replicate` or contiguous
`Shard(dim)` placement. Non-`nn.Parameter`-subclass quantized weights whose
tensor subclass supports `DTensor` wrapping (such as TorchAO FP8, MX, NVFP4, or
Piper ConvRot) project zero-copy for all-`Replicate` targets. For `Shard(dim)`
placements across 1D and N-D `DeviceMesh` shells, zero-copy projection supports
TorchAO `Float8Tensor` (`PerTensor`, `PerRow` along any axis, and `PerGroup`
when shard boundaries align with the group size) and 2-D Piper
`ConvRotInt8Tensor` (`Shard(0)` on any non-empty split; `Shard(1)` when shard
offset and local width are aligned to `group_size`, preserving optional scalar
`act_per_tensor_scale`). Formats without shard projection (`MXTensor`,
`NVFP4Tensor`, `ConvRotNVFP4Tensor`, and 5-D Conv3D `ConvRotInt8Tensor`),
empty, misaligned, or transposed quantized shards, and `DTensor`-incompatible
tensor subclasses (such as TorchAO `Int8Tensor`, tile-packed INT4, or
`PrototypeFloat8Tensor`) are rejected cleanly with `layout mismatch` during bind
preflight. Non-canonical projected quantized shards (such as `Shard(1)`) keep
zero-copy host views under the single store allocation, report
`requires_activation(param) == True` while resting on CPU (disallowing direct
CPU execution and permanent `merge_adapter` updates), allocate contiguous
per-rank device buffers preserving block-pool compatibility across streaming
refills on CUDA activation, and support active LoRA and dense adapter merges on
`RowwiseParallel` layers, bias-free `nn.Linear` avoids PyTorch DTensor's linear
decomposition falling back to all-gathering `Shard(1)` quantized weights). Each
rank selects its portion of plain host-backed LoRA factors or dense deltas
before staging; merging requires no collective. Full adapter tensors remain in
host memory. DTensor adapter factors and deltas, `DTensor` registered buffers
against plain buffer stores, and `block_mode="rolling"` on projected DTensor
block groups are not accepted. Local shards must carry their tensor
representation independently of an `nn.Parameter` wrapper.
See [dtensor_adapter.py](../src/piper_offload/dtensor_adapter.py) for supported
host projections and reconstruction.

For experimental DTensor execution, [communication.py](../src/piper_offload/communication.py)
documents `piper_relay` collectives through host memory,
[sequential.py](../src/piper_offload/sequential.py) documents `SequentialExecutor`
for two ranks sharing one process and GPU, and
[parallel.py](../src/piper_offload/parallel.py) documents `ParallelExecutor`
and the `piper_parallel` backend for concurrent multi-rank execution over CPU,
direct CUDA peer copies, and bounded host-staged transfers. Their docstrings
cover setup and execution contracts. Block streaming yields host transfers to
`piper_relay` collectives and CUDA `ParallelExecutor` transfers via
`foreground_transfer`, as [streaming](streaming.md#yield-to-collectives)
describes; `SequentialExecutor` does not coordinate ordinary offloader
scheduling.

A custom tensor format integrates through `register_adapter()`. Implement the
capture, physical-storage enumeration, allocation, transfer, and reconstruction
protocol, then opt into the capabilities the format actually supports. The
authoritative contracts are in
[tensor_adapters.py](../src/piper_offload/tensor_adapters.py) and
[tensor_adapter_registry.py](../src/piper_offload/tensor_adapter_registry.py);
see [development](development.md) for contributing changes.
