# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Added
- **Fused MoE gather kernels** for quantized experts: `moe_glu_gather_kq` /
  `gather_qmv_kq` cover all 19 GGUF codecs (K-quant, legacy, IQ) plus
  mixed-codec shared experts; `moe_glu_gather` / `gather_qmv_bias` cover mxfp4.
  Each fuses the expert gather, dequant mat-vec, and GLU activation into one
  dispatch. Wide K-lane (NX=16/32) variants engage automatically for
  decode-scale two-stream GLU gathers.
- **Fused MoE router** (`moe_router_topk`): softmax + top-k + weight norm +
  shared-gate sigmoid in one dispatch; no-shexp routing-mix gather and
  per-expert-scale support; q8_0 odd-K fallback.
- **`sdpa_decode_gqa`**: tile-staged GQA decode SDPA kernel (head_dim
  64/128/256/512, GQA factor 2..16, attention sinks folded into the merge
  pass), with pair variants accepting verify widths (q_len 2..4, subject to
  gqa_factor * q_len <= 32).
- **Fused residual/rmsnorm glue ops** (`add_rmsnorm`, `rmsnorm_multi3`,
  `rmsnorm2_add`) with register-cached 4-wide reads, row-sized threadgroups,
  and scalar CPU eval paths.

### Changed
- `KQuantEmbedding` output dtype defaults to bf16 (was f16).
- q8_0 decode `qmv`/`qmv_fast` fuse the bias add.

## [0.2.1]

### Added
- **Verify mat-vec kernels (`mul_mv_ext` port)** for speculative-decode verify
  and batched MTP. Flat-with-M kernels ported from ggml's `mul_mv_ext` for all
  K-quant, legacy, and IQ codecs via a shared `kq_mv_ext_impl` template. Gated
  at M>=3 for non-IQ codecs (verify_qmv wins at M<=2) and all-M for IQ. Extends
  to M=12 for batch MTP verify.
- **Compiled vector SDPA kernel (`kq.sdpa_vector`)** for head_dim 256 and 512
  with GQA and strided KV support.
- **Synthetic all-codec matmul test** (`test_matmul_synth.py`) - GGUF-free
  quantized-matmul validation across all codecs.

## [0.2.0]

### Added
- IQ codecs are now full-featured alongside the K-quant/legacy codecs. All nine
  (`iq4_nl, iq4_xs, iq3_s, iq3_xxs, iq2_xxs, iq2_xs, iq2_s, iq1_s, iq1_m`) gained:
  - **NAX (tensor-core) matmul** for prefill, matching the K-quant kernels
    (decode `qmv` unchanged).
  - **`quantize` (encode)** — a scalar CPU port of ggml's `quantize_row_iq*`
    quantizers (grid search + inverse-index neighbour tables), so `kq.quantize`,
    the `convert` driver, and the `mlx-kquant quantize` CLI now produce every IQ
    codec. IQ encode is CPU-only (ggml has no GPU IQ quantizer) and is routed to
    a CPU stream internally. `iq2_xxs`, `iq2_xs`, and `iq1_s` require an
    importance matrix (mirroring ggml) and reject a missing one.

## [0.1.2]

### Fixed
- `mlx-kquant lora`/`chat` require `--model` and no longer fall back to an mlx-lm
  default model, potentially fetched from hf hub.
- expanded cli help

## [0.1.1]

Docs: PyPI install instructions, no code changes

## [0.1.0]

First public release. A C++/Metal extension for a stock `mlx==0.31.2` wheel that
adds the K-quant superblock and per-block integer codecs as native MLX ops
(`kq.dequantize` / `quantized_matmul` / `gather_qmm` / `quantize`), with Metal and
portable CPU paths for all ten codecs. On top of the ops, the `mlx-kquant` CLI
quantizes an HF / mlx-lm model into a K-quant MLX safetensors checkpoint and runs,
chats with, LoRA-fine-tunes, and fuses it, with importance-matrix calibration and
per-tensor recipe inspection. A loader runs those checkpoints on stock mlx-lm.

### Notes
- `requires-python >= 3.10` (mlx 0.31.2 ships no cp39 wheel).
- The GPU path is macOS 26 (Tahoe) or later on Apple Silicon (Metal); the NAX
  matmul kernel needs the Metal 4 SDK (`MetalPerformancePrimitives`). Linux is
  supported CPU-only - build against `mlx[cpu]==0.31.2`; model forwards there also
  need `MLX_DISABLE_COMPILE=1` (an upstream MLX CPU-JIT limitation under GCC, not
  mlx-kquant).
