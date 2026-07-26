# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- BM=32 double-buffered NAX qmm tile for the M<=32 batch-decode band, where
  the BM=64 tile wastes up to 75% of MMA issues on row padding. Rolled out
  codec by codec (q6_k, q8_0, q4_k, q5_k, q2_k/q3_k, then all 19 codecs)
  behind a measured per-codec policy; q3_k M12 +65% and q2_k M12 +110% at
  the wide decode shape.
- Name-suffixed `_db` double-buffered BM=64 qmm_t variant for the M33-64
  band, N-gated per codec (q6_k >= 16384; q8_0/q4_1/q5_1/q5_0 >= 8192):
  blanket double-buffering at BM=64 regressed M96+ and prefill, so only the
  band and the codecs that measured wins carry it.
- BM=128 qmm_t tile for M >= 193 when ceil(M/64) is even (odd ceilings add
  64 padding rows), with per-codec entry floors from a 19-codec paired ABBA
  (`benchmarks/bench_qmm_bm128_ab.py`): 193 for q6_k and the IQ grid codecs
  (+4-27%, the taller tile amortizes grid-dequant ALU over 2x rows), 449
  for the K-quants and byte-loaders that lose the padded M224 cell, 961 for
  the flat quartet. `KQ_NAX_BM128` is tri-state like `KQ_NAX_DB64`, plus a
  floor-dropping probe mode `2`.
- Flat-codec ushort loader fast paths: the q4_0/q4_1/q5_0/q5_1/q8_0/iq4_nl
  block loaders read each 32-weight block as aligned ushorts instead of 32
  single-byte device loads (q4_0 113 -> 306 GB/s at the M16 wide decode
  shape; q8_0 +15-25% at M48-128).
- Batch-decode routing levers documented in docs/kernels.md: `KQ_NAX_SMALL_BM`,
  `KQ_NAX_BM128`, `KQ_NAX_DB64`, `KQ_FORCE_QMM_MIN_M`, plus the two
  falsified-but-kept probe levers `KQ_NAX_SWIZZLE` (row-tile traversal
  swizzle; the M>64 band is per-threadgroup-bound, not DRAM-bound) and
  `KQ_MV_EXT_NR` (two-rows-per-thread mv_ext; no faster than shipped).

### Changed
- Transpose decode shapes above a per-codec crossover (M 6-9, with per-codec
  small-N/big-K floors, vocab-head N shifting q6_k to M >= 8) now route to
  the NAX qmm instead of riding the mat-vec paths to M 13; the mat-vec
  kernels decay past M~5 (q6_k M >= 9: 274-305 GB/s on qmm vs 221-254 cold).
- The q6_k/q5_k/q3_k/q2_k loaders drop their replay caches (scale/qh/pair/
  hmask), unifying each to one position-generic body; the caches only added
  register state, so the default BM=64 prefill path also speeds up (q5_k
  M2048 +13%).
- Metallib instantiation trim: `_db` variants exist only for the five
  policy-enabled codecs and the new tile macros (smallbm, db64, bm128) drop
  their float-x variants (float32 x is promoted to bf16 before the
  primitive). Release-over-release the metallib goes 80.6 MB (0.3.5) ->
  93.2 MB; the trim recovers 12.5 MB of what the new tiles would otherwise
  have added.
- Decode matmul width gate: M==2 now routes to the flat mat-vec
  (`kq_<codec>_mv_ext`) for every codec except q4_k and q8_0, where
  `verify_qmv` measures faster DRAM-cold. The K-quants and legacy codecs
  crater on the old M==2 route (q6_k 328 vs 536 GB/s cold), and mxfp4/nvfp4
  (no verify kernel) fell through to per-row qmv at half their M==1 rate
  (mv_ext is 5.6-7.1x there). M==2 is every B=2 batched decode step, not just
  draft-width-1 verify; qwen3.6-27B Q6_K B=2 decode gains ~6-12% aggregate
  tok/s end-to-end.

## [0.3.5]

### Added
- Fine-tiled qmv decode variant (`qmv_fast_fine`/`qmv_fine`, 2 output rows per
  threadgroup) for all matmul codecs, bit-exact, with per-codec dispatch-size
  defaults and a `KQ_QMV_FINE` override; recovers occupancy on mid-size decode
  matvecs (dense-llama 8B Q6_K decode ~+3% end-to-end).
- Fine-tiled MoE gather variants (2 output rows per threadgroup, 4x the
  threadgroups) for the tuned q6_k/q8_0 `gather_qmv`/`gather_qmv_mix` and the
  packed-mxfp4 `gather_qmv_bias`/`gather_qmv_mix_bias` kernels, plus the
  bias-fused q8_0 `qmv_bias` fine siblings; bit-exact, on by default for
  starved decode grids with a `KQ_GATHER_FINE` override (uniform-q6_k
  qwen3.6-35B decode ~+2.7%, gpt-oss-20b ~+1.25% end-to-end; the generic Ext
  codec-matrix equivalents measured E2E-neutral and are not instantiated).
- Tensor-op DSA indexer score GEMM: a Metal MMA (f16) path plus an i8mx
  quantized arm for the DeepSeek sparse-attention indexer, with a QAT emit
  helper (`dsa_indexer_qat_pack`) that packs pre-rotated, on-grid key rows,
  and an A/B benchmark (`benchmarks/bench_dsa_indexer_ab.py`).
- `gather_qmv_mix_bias`: packed-mxfp4 gathered down projection with the routing mix + expert bias folded in (f32 slot accumulation), replacing `gather_qmv_bias` + `(y * scores).sum(-2)` for gpt-oss

### Fixed
- `KQuantSwitchLinear` sorted-expert GEMM arm now also requires the default
  device to be the GPU, not just a present Metal backend: `expert_tile_map`/
  `gather_qmm_seg` are Metal-only, so a CPU default device (e.g.
  `KQUANT_FORCE_CPU`) correctly falls through to the per-expert loop instead
  of dispatching Metal-only ops.

## [0.3.4]

### Added
- Native-fp wire codecs `mxfp4` (GGML type 39: 17-byte blocks, e8m0 scale +
  16 two-halves nibble bytes) and `nvfp4` (type 40: 36-byte blocks, four
  ue4m3-scaled 16-value groups) as first-class kquant codecs: zero-copy GGUF
  wire bytes with scalar + NEON CPU decode/matvec, the full Metal ALU matmul
  and gather families, and the fused MoE GLU family (`moe_glu_gather_kq`,
  `gather_qmv_kq`, shexp/mix variants). Decode-only - GGUFs ship these
  tensors pre-quantized, so there is no encoder; `nvfp4` is verified at the
  kernel level against synthetic wire (no real-model artifact yet).
- `swiglu_clamp` activation + per-expert gate/up/down biases on the fused
  MoE ops (fp4 codecs only): the gpt-oss clamped SwiGLU runs as one dispatch
  per GLU and the biased down gather as another - bit-identical to the
  packed-mxfp4 `moe_glu_gather`/`gather_qmv_bias` pair on the same weights.
- `codec_has_matmul` / `codec_has_moe_glu` capability queries, so consumers
  gate GPU routing and fused installs off the registry instead of
  hard-coding codec lists.
- `KQuantSwitchLinear` sorted-GEMM prefill arm now serves biased expert
  stacks (bias applied after the segmented GEMM) - gpt-oss prefill takes
  the one-GEMM-per-expert-segment path instead of per-row gathers.
- `benchmarks/bench_native_fp_ab.py`: packed-vs-wire MoE A/B bench (decode +
  sorted-prefill units; gpt-oss/DeepSeek-V4 shapes; packed, packed-fused,
  wire, wire-fused(+bias) and CPU arms).

### Fixed
- E8M0/UE4M3 scale decode on Metal builds the float bits directly instead of
  fast-math `exp2`, which lands ulps off on some Metal compiler versions -
  keeps GPU dequant bit-exact with the CPU decoders and gguf-py on every
  toolchain, and matches MLX's own `fp8_e8m0` conversion.

## [0.3.3]

### Added
- DeepSeek-V4-Flash / GLM sparse attention, ported with modifications from
  omlx's `glm_moe_dsa` custom kernels (Apache-2.0; per-file OpenAI/Apple
  notices preserved). `dsa_sparse_attention` runs the sliding local window plus
  the indexer-selected pooled KV rows plus per-head attention sinks in one
  flash-softmax dispatch (f32 accumulation); the lightning indexer is
  `dsa_indexer_scores` (prefill GEMM), `dsa_indexer_score_decode` (fused
  decode-width scores), and `dsa_topk_indices` (2-pass radix arg-select). All
  accept `qL >= 1`, so decode, MTP verify (`qL = 2`), and prefill share them.
- `dsa_kv_qat` and `dsa_indexer_qat`: fused DeepSeek-V4 quantization-aware
  round-trips - per-64-block FP8-E4M3FN on the main-attention KV, and a 128-wide
  Hadamard transform then per-32-block FP4-E2M1 on the indexer activations -
  each bit-identical to the equivalent MLX graph.
- `silu_limit` activation for the fused K-quant MoE GLU gather: the clamped
  SwiGLU `silu(min(g, limit)) * clip(u, -limit, limit)` that DeepSeek-V4's
  `LimitedSwiGLU` needs, passed as a constant-buffer limit (dead-arg-stripped
  for the existing silu/gelu paths, so their kernels are unchanged).
- `moe_router_topk` gains `sqrtsoftplus` scoring (`sqrt(softplus(x))`),
  score-plus-bias ranked selection, and an optional per-expert routed scale -
  DeepSeek-V4 routing in one dispatch.
- `gather_qmm_seg` + `expert_tile_map`: expert-sorted MoE prefill as one GEMM
  per expert segment instead of per-row gathers, with the 64-row tile map built
  on the GPU from the sorted routing indices (no host sync). `KQuantSwitchLinear`
  takes this arm on large prefill batches, gated by `KQ_SWITCH_GEMM_MIN_ROWS`
  (default `512`). `nax_gather_enabled` reports whether the sorted-gather NAX
  leaf is reachable, so the arm defers to it on tensor-unit GPUs.
- `docs/kernels.md`: a capability-grouped reference for the fused and
  architecture-specific kernels beyond the four core codec ops.
- `sdpa_fa_verify` head_dim 512: 256-thread d-split kernel (gemma-4
  global-attention verify/decode, folds to 32 rows) + vectorized K/V staging.
- `sdpa_fa_verify` now takes a 64-row query tile (up from 32), so a GQA-16
  fold at `q_len` 4 stays on the matrix units instead of falling to the
  stock materialized path. `q_len` 1 is also accepted now, routing plain
  GQA decode through the same kernel.
- Non-NAX `gather_qmm_rhs`: steel simdgroup-mma GEMM for the sorted-rhs
  (SwitchGLU prefill) gather leaf on GPUs without tensor units, all 19
  codecs. Walks each row tile's per-expert segments and runs one full-K
  matmul per segment, so the sorted batch no longer decomposes into
  per-row `gather_qmv` calls. The row tile height adapts to the batch's
  rows-per-expert (BM 16/32/64 at M/E thresholds 40/384): every segment pays a
  full-tile mma pass, so a tile much taller than a segment wastes most of every
  matmul. This is a large speedup on MoE prefill shapes at big batches, and more
  at mid sizes where the adaptive tile kicks in.
  `KQ_DISABLE_GATHER_RHS_ALU=1` forces the old per-row path;
  `KQ_GATHER_RHS_BM` pins the tile height (retuning lever). On NAX machines
  the NAX leaf still takes precedence; the new kernel serves the cases NAX
  refuses (older macOS, `K % 64 != 0`, `KQ_DISABLE_NAX=1`).

### Changed
- iq2_xxs / iq3_xxs Ext MoE gathers stage their dequant LUTs in threadgroup
  memory, shared across the gather's lanes.
- q2_k / q3_k decode `qmm_t` weight loaders re-read each K-tile statelessly
  instead of carrying a deep decoded-register cache; q6_k keeps its shallow
  cache (its two-stream nibble decode is heavy enough that re-reading is
  slower).
- `KQuantMultiLinear` memoizes its gather index arrays across calls, so a decode
  loop reuses one constant pair instead of rebuilding the arange/broadcast graph
  every layer.

### Fixed
- `sdpa_fa_verify` read a lazily-strided query as packed: it trusted the
  query layout at graph-build time, so a non-contiguous q tile was streamed
  at the wrong stride. Layout is now checked at eval time.
- The sorted-rhs gather leaf mis-walked expert strides on the FIRST
  evaluation of a lazily-sliced weight stack (e.g. a fresh `w[:, :n]` view):
  the op trusted `w.flags().row_contiguous` at graph-build time, but flags of
  an unevaluated array are meaningless, so the compaction copy was skipped
  and the stride-less kernel read experts at the packed stride. Affected the
  NAX leaf too. The op now always inserts the `Contiguous` node when the rhs
  leaf is reachable (zero-copy at eval when already packed) and eval_gpu
  additionally gates on `w.flags().row_contiguous`.

## [0.3.2]

### Added
- `sdpa_fa_verify`: simdgroup-matrix speculative-verify attention for a
  GQA-folded query tile (G*q_len <= 32 rows, q_len 2..8, head_dim 256).
  Streams each contiguous KV split once through threadgroup-staged K/V tiles
  with S = Q@K^T and O += P@V on the matrix units, float32 accumulators, and a
  per-row offset-causal online softmax; reuses the `sdpa_decode_gqa`
  split-merge pass.
- `verify_zero_copy_views(items, no_alias=[])` and `zero_copy_view_count()`:
  post-load integrity check that arrays backed by a GGUF mapping still carry
  their wire dtype (integer reinterprets allowed) and that `no_alias` names
  own their buffers. Catches buffer donation into the file mapping: a donated
  dtype-changing copy leaves an array typed X over wire bytes typed Y, and
  the write is dropped on read-only maps. Metadata-only, O(#tensors).

### Changed
- `load_gguf` zero-copy mappings are now read-only shared (`PROT_READ` +
  `MAP_SHARED`) instead of writable private. Private mappings made the GPU
  write-fault every wired page, lazily copying the whole file into anonymous
  memory that can only be compressed, never dropped (a hidden full-model RAM
  copy on top of the page cache carrying the same bytes); read-only shared
  pages are wired in place and stay clean, evictable file cache.
  `KQ_GGUF_MMAP=private_rw` restores the old mode; `private_ro` is a
  diagnostic quadrant.

## [0.3.0]

### Added
- Fused MoE gather kernels: for quantized experts: `moe_glu_gather_kq` /
  `gather_qmv_kq` cover all 19 GGUF codecs (K-quant, legacy, IQ) plus
  mixed-codec shared experts; `moe_glu_gather` / `gather_qmv_bias` cover mxfp4.
  Each fuses the expert gather, dequant mat-vec, and GLU activation into one
  dispatch. Wide K-lane (NX=16/32) variants engage automatically for
  decode-scale two-stream GLU gathers.
- Fused MoE router: (`moe_router_topk`): softmax + top-k + weight norm +
  shared-gate sigmoid in one dispatch; no-shexp routing-mix gather and
  per-expert-scale support; q8_0 odd-K fallback.
- `sdpa_decode_gqa`: tile-staged GQA decode SDPA kernel (head_dim
  64/128/256/512, GQA factor 2..16, attention sinks folded into the merge
  pass), with pair variants accepting verify widths (q_len 2..4, subject to
  gqa_factor * q_len <= 32).
- Fused residual/rmsnorm glue ops (`add_rmsnorm`, `rmsnorm_multi3`,
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
