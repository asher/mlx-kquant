# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - unreleased

First public release: the standalone `kq.*` op layer plus a create-weights
toolchain that runs on a stock `mlx==0.31.2` wheel.

### Added
- `kq.*` op namespace: `dequantize`, `quantized_matmul`, `gather_qmm`,
  `quantize`, `load_gguf`, for all 10 GGUF K-quant / legacy codecs.
- `mlx_kquant.codec_geometry` - single source of truth for codec block geometry.
- `mlx_kquant.nn` - `kq.*`-backed `KQuantLinear` / `KQuantEmbedding` /
  `KQuantSwitchLinear` / `KQuantMultiLinear` modules, plus `install_kquant_modules`.
- `mlx_kquant.recipes` / `imatrix` / `quantize` - preset-driven create-weights
  pipeline with optional importance-matrix calibration (GPU encode).
- `mlx_kquant.loader` - load a kquant checkpoint into a stock mlx-lm model.
- `mlx-kquant` CLI (alias `mlxkq`): `quantize`, `calibrate-imatrix`, `lora`,
  `fuse`, `verify`, `run`, `inspect`. Requires the `[tools]` extra (except
  `inspect` and `verify --codecs`, which run on a base install).
- `mlx-kquant inspect` - print a checkpoint's per-tensor codec recipe (codec,
  bits, packed/logical shape) from the config + safetensors headers, no GPU.
- LoRA support for kquant bases (`mlx_kquant.mlx_lm_patch` + `docs/lora.md`),
  backed by a gradient-wrt-x `vjp` on the matmul / gather ops: attach / train /
  merge an adapter. `mlx-kquant run --adapter-path` attaches an adapter at runtime;
  `mlx-kquant fuse` re-encodes the merge to kquant (`--imatrix` carries the base's
  calibration through the re-encode) or, with `--dequantize`, writes a float
  checkpoint. DoRA is not supported. A standalone `patch_mlx_lm_load()` enables
  load-only interop (inference / eval, no LoRA).
- `docs/integration.md` - reference for building on the `kq.*` ops (wire-byte
  contract, on-disk format, codec table, the layer/loader/encode/interop seams).
- `docs/walkthrough.md` (end-to-end quantize -> run -> LoRA -> fuse -> inspect)
  and `docs/imatrix.md` (imatrix calibration: build, quantize, preserve on merge).
- CPU decode path for all 10 codecs (`stream=mx.cpu`) covering `dequantize` /
  `quantized_matmul` / `gather_qmm`, so the op tests run without a GPU.
- PEP 621 packaging, `py.typed`, ruff config, CI (lint + build + honest GPU-gated
  op tests), and a tag-triggered wheel-publish workflow.

### Notes
- `requires-python >= 3.10` (mlx 0.31.2 ships no cp39 wheel).
- macOS on Apple Silicon. Linux (CPU-only train/infer) is a contingent follow-up.

## [0.2.0] - planned
- CPU **encode** parity (`KQuantQuantize::eval_cpu`) -> quantize on Linux.
