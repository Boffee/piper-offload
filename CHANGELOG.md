# Changelog

All notable changes to Piper Offload are documented here. Versions follow the policy in
[VERSIONING.md](VERSIONING.md).

## [Unreleased]

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
