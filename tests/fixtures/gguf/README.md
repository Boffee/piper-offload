# GGUF test inputs

The fixtures were written with llama.cpp's Python `gguf` 0.19.0 writer, then
inspected with its reader. The expected offsets and byte counts in
`test_gguf_checkpoint.py` come from that inspection.

- `weights.gguf` holds a 64×64 Q4_0 matrix quantized from FP32 normal values
  (NumPy's default generator, seed 42), its rows in reverse order, and three
  small tensors stored in F32, F16, and BF16.
- `mixed-header.gguf` exercises 1024-byte alignment and every metadata value
  type, including nested arrays and UTF-8 strings. Its six entries use Q4_K,
  Q8_0, IQ3_S, F32, F16, and BF16. IQ3_S is listed but unsupported for loading.
- `unsupported-type.gguf` holds four F32 ones and four I8 ones. Offload can
  load the first tensor and reject the second.
- `two-tensors.gguf` and `two-tensors-be.gguf` contain two four-element F32
  tensors named `one` and `two` with 64-byte alignment. The tests use their
  little- and big-endian headers to check rejection paths.
