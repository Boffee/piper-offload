# Weight formats

Piper's tensor adapters manage capture, movement, and reconstruction of each
supported weight representation. Offloading a format does not imply that it
supports training or additive updates. See [adapters](adapters.md) for applying
LoRA, deltas, and complete values, [streaming](streaming.md) for execution modes,
and the [README](../README.md) for the starting point.

## Choose a supported representation

All formats below support offloading. Install the corresponding optional extra;
the supported dependency ranges are maintained in
[pyproject.toml](../pyproject.toml). Native execution still requires the
format's compatible hardware and PyTorch stack.

The `torchao` extra also covers Piper ConvRot. The composable `triton` extra
selects upstream Triton on Linux and `triton-windows` on 64-bit Windows.
Most integrations have portable
fallbacks without Triton; GGUF conversion requires it. Windows GPU execution
requires Windows 10 or 11, a supported NVIDIA GPU and current driver, and the
Visual C++ Redistributable for Visual Studio 2015–2022. A separate CUDA toolkit
or Visual Studio installation is not required for eager GPU execution.

| Weight representation | Extra | LoRA merge | Dense / mixed merge |
|---|---|---|---|
| Plain floating point, excluding plain float8 | None | Native | Native |
| Quanto qint8 / qfloat8 | `quanto` | Yes | Yes |
| bitsandbytes NF4 / FP4 | `bnb` | Yes | Yes |
| bitsandbytes int8 | `bnb` | Yes | Yes |
| TorchAO scaled FP8 | `torchao` | Yes | Yes |
| TorchAO static-activation scaled FP8 | `torchao` | Yes | Yes |
| TorchAO INT8 | `torchao` | Yes | Yes |
| TorchAO MXFP8 / MXFP4 | `torchao` | Yes, layout restrictions | Yes, layout restrictions |
| TorchAO NVFP4 | `torchao` | Yes, layout restrictions | Yes, layout restrictions |
| TorchAO INT4 tile-packed | `torchao` | No | No |
| Piper ConvRot INT8 | `torchao` | Yes, matrices only | Yes, matrices only |
| Piper ConvRot NVFP4 | `torchao` | Yes, layout restrictions | Yes, layout restrictions |
| Packed GGUF → ConvRot INT8 | `gguf` | Activation only | Activation only |
| DTensor | Depends on local shard | Delegates to local adapter | Delegates to local adapter |

Plain float8 storage can move but is not an additive merge target. Scaled FP8
wrappers have their own merge implementations. Routed LoRA requires no merge
capability, so it also works with tile-packed INT4 when the owning module is a
compatible logical `nn.Linear`.

Quantized additive merges recompute data-dependent weight scales and preserve
activation calibration metadata. Standard CUDA layouts use format-specific
Triton kernels where available; other supported layouts use a reference
dequantize/requantize path. Piper ConvRot delegates merges to Piper Kernels.
All merge-capable built-in quantized formats support stochastic rounding.
Nested bitsandbytes 4-bit scales use the reference path.

## Training and layout limits

Quantized weights stay frozen. Only plain tensors support the `Parameter.data`
swap needed to preserve trainable parameter identity. Quanto and both scaled
FP8 adapters additionally support copying device state back to CPU; that
capability does **not** make their weights trainable. Other quantized adapters
and DTensor provide inference movement and the merges listed above. Separate
plain trainable parameters can coexist with a frozen quantized base; follow
the [training guidance](streaming.md).

Transposed MX and NVFP4 packed targets cannot be re-encoded into their existing
storage; use routed LoRA. Scaled FP8 supports transposed per-row and per-tensor
scales, but not transposed per-group scales. TorchAO INT8 does not support
transposed weights; Piper ConvRot INT8 also rejects transposed matrices at
capture. A valid nonstandard layout uses a reference merge only when that format
can re-encode the layout.

Complete [parameter values](adapters.md#populate-optional-meta-parameters)
have a broader use than additive updates: exact representation copying works
for supported physical values even when additive merge is unavailable.
Non-unit strength scaling requires the additional merge capabilities.

## Format details

**Quanto.** `WeightQBytesTensor` storage and scales are captured separately.
Marlin FP8 weights are canonicalized to ordinary unpacked Quanto weights for
offloading, so execution does not retain Marlin's packed matmul. A direct
permanent merge into an existing Marlin weight repacks its original storage.
See [quanto_adapter.py](../src/piper_offload/quanto_adapter.py).

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

`MappedCheckpoint` reads safetensors and GGUF through read-only file mappings.
GGUF F32, F16, and BF16 entries are ordinary tensors. Quantized entries are
`GgufParameter` objects with a logical matrix shape and BF16 compute dtype;
`as_tensor()` exposes packed bytes. `get_slice()` reports logical shape and
dtype, and `get_nbytes()` reports stored bytes. GGUF metadata is not exposed
as safetensors application metadata.

Load those entries with `model.load_state_dict(state, assign=True)` and freeze
the model before capture. Encoded rows must remain intact when splitting or
regrouping weights. Contiguous row splits keep their file provenance; copying
operations create independent storage. See [memory](memory.md) for the mapped
checkpoint loading pattern.

Packed GGUF parameters require Offload activation to execute;
`requires_activation(parameter)` lets loaders detect this. Each load transfers
the packed source to a reusable staging buffer, then decodes, rotates, and
requantizes directly into BF16 ConvRot INT8 storage without allocating the full
dense weight. The logical input width must be divisible by 16. Rotation uses
the largest compatible group size among 256, 64, and 16.

Direct conversion supports F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, and IQ4_XS. The `gguf` extra includes
TorchAO and Triton because this conversion requires both. LoRA and dense
updates apply to the active ConvRot target; permanent merge cannot modify the
packed GGUF source.

Externally supplied Diffusers `GGUFParameter` objects are recognized without a
Diffusers dependency. Their GGUF linear modules must first be normalized to
ordinary linear behavior. See [gguf_parameter.py](../src/piper_offload/gguf_parameter.py)
and [gguf_adapter.py](../src/piper_offload/gguf_adapter.py).

## DTensor and custom formats

DTensor support composes a distributed wrapper with the adapter for its local
shard. It targets frozen inference with ordinary `Replicate` and contiguous
`Shard` placements. Each rank selects its portion of plain host-backed LoRA
factors or dense deltas before staging; merging requires no collective. Full
adapter tensors remain in host memory. DTensor adapter factors and deltas are
not accepted, and bitsandbytes parameter subclasses cannot be local shards.
See [dtensor_adapter.py](../src/piper_offload/dtensor_adapter.py) for supported
host projections and reconstruction.

For experimental DTensor execution, [communication.py](../src/piper_offload/communication.py)
documents `piper_relay` collectives through host memory, and
[sequential.py](../src/piper_offload/sequential.py) documents `SequentialExecutor`
for two ranks sharing one process and GPU. Their docstrings cover setup and
execution contracts.

A custom tensor format integrates through `register_adapter()`. Implement the
capture, physical-storage enumeration, allocation, transfer, and reconstruction
protocol, then opt into the capabilities the format actually supports. The
authoritative contracts are in
[tensor_adapters.py](../src/piper_offload/tensor_adapters.py) and
[tensor_adapter_registry.py](../src/piper_offload/tensor_adapter_registry.py);
see [development](development.md) for contributing changes.
