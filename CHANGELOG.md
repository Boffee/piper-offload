# Changelog

All notable changes to Piper Offload are documented here. Versions follow the policy in
[VERSIONING.md](VERSIONING.md).

## [Unreleased]

### Added

- Add boolean opt-in stochastic rounding for quantized LoRA merges across Quanto, bitsandbytes,
  TorchAO FP8/INT8/MX/NVFP4, and DTensor-composed weights. Automatic per-target, per-merge seed
  selection remains internal to LoRA merge; deterministic rounding remains the default and Piper ConvRot
  INT8 reports stochastic merge as unsupported. Each quantized adapter composes both modes
  through its existing requantization method: deterministic representation construction followed
  by optional stochastic terminal-code recoding.
- Add public `derive_seed(*parts)` as the canonical stable unsigned 64-bit seed derivation
  utility for Piper Offload and downstream adapters.
- Treat exact-zero LoRA strengths as inactive before target lookup, factor staging, hook
  installation, or cache leasing.

### Fixed

- Recompute data-dependent Quanto qint8/qfloat8 weight scales after LoRA merges in both
  generic and CUDA paths, including safe exact-zero blocks.
- Merge TorchAO INT8 LoRA factors in the stored SmoothQuant/AWQ weight coordinates while
  preserving activation pre-scales and calibration metadata.
- Repair exact-zero TorchAO FP8 PerGroup blocks in the generic requantization path.
- Preflight adapter-specific quantized merge constraints before mutating any model weight.

## [0.1.0] - 2026-08-03

### Added

- Policy-driven resource and model caches with lease-aware LRU eviction and reusable model,
  LoRA, and general-object stores.
- Whole-model pinned-host offloading and asynchronous block streaming with configurable
  residency, prefetching, optional block compilation, and trainable-weight support.
- Activation-scoped merged and routed LoRA application, including format-specific CUDA merge
  kernels and tensor-parallel DTensor composition.
- Tensor adapters for dense, bitsandbytes, GGUF, optimum-quanto, Piper ConvRot INT8, and TorchAO
  scaled-FP8, INT8, MX, NVFP4, and INT4 tile-packed weights.
- Public tensor-adapter registration for downstream tensor subclasses.
- Python 3.14, PyTorch 2.13, TorchAO 0.18, Apache-2.0 licensing, and the Piper Offload package
  identity.

[Unreleased]: https://github.com/Boffee/piper-offload/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Boffee/piper-offload/releases/tag/v0.1.0
