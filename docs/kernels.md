# Kernel reference

The `kq.*` namespace has two tiers. The **core codec ops** - `quantize`, `dequantize`,
`quantized_matmul`, `gather_qmm` - are the general K-quant surface, documented in the
[README](../README.md) and [integration.md](integration.md); everything a downstream project needs to
store and multiply K-quant weights is there. On top of them sits a set of **fused and
architecture-specific kernels**: decode/prefill fusions that collapse several ops into one dispatch,
and a cluster of sparse-attention kernels for the DeepSeek/GLM lightning-indexer attention. This page
catalogs that second tier.

Kernels are named for what they compute, not the model that first needed them. A fusion motivated by
one architecture (a norm layout, an activation, an attention shape) is written as a general kernel and
reused wherever the shape recurs; the entries below note the motivating regime. The one exception is
the `dsa_*` group, which implements a specific attention mechanism (DeepSeek-V4-Flash / GLM) end to
end and is scoped to it.

Each op's Python docstring carries the full argument contract, shape constraints, and dtype rules
(`help(kq.<name>)`); the one-liners here are a map, not a spec.

## Quantized matmul and MoE gather fusions

General K-quant matmul paths beyond the core `quantized_matmul` / `gather_qmm`, for the shapes those
two leave on the table (single-row decode, expert-sorted prefill, fused bias/mix).

- **`quantized_matmul_qmv_bias`** - `x @ dequant(w) + bias` with the bias add fused into the matvec.
  Decode-only (single row); `q8_0` for now, other codecs fall through to matmul-then-add.
- **`gather_qmv_kq`** - gathered matvec for an expert stack, one activation row per expert slot: the
  MoE down projection at decode. Takes an optional per-expert bias for the fp4 wire codecs
  (`mxfp4`/`nvfp4`, gpt-oss experts).
- **`gather_qmv_mix_kq`** / **`gather_qmv_mix_ns_kq`** - the down projection with the routing mix
  folded in - every slot accumulated in f32 weighted by its score, and (in the `mix` variant) a shared
  expert as the last slot - replacing a gather plus a weighted sum plus the shared-expert add. Shaped
  for DeepSeek-V3/V4-style shared-expert MoE.
- **`gather_qmv_bias`** - gathered matvec with a fused expert bias on MLX's packed mxfp4 layout (the
  counterpart to the K-quant gathers above for that codec).
- **LoRA epilogue** - `quantized_matmul`, `quantized_matmul_qmv_bias`, `gather_qmv_kq` and
  `gather_qmv_mix_ns_kq` accept `lora_a` / `lora_b` (plus `lora_rows`, and `lora_ids` / `lora_table`
  on the gathers) and add `rows * (x @ A) @ B` on their own output inside the primitive. Rows
  routes run one epilogue dispatch after the base route; the score-mixed gather forms `z = x @ A`
  in a kernel dispatched before the base gather (it reads only x and A, so the encoder overlaps the
  two) and applies `z @ B` in a barrier-free kernel after it, so the serialized cost behind the
  gather is one tiny dependent kernel. Codec independent, so a live adapter adds no graph ops at
  decode on any codec; `lora_table` remaps gathered ids (arena slot to expert, negative skips) and
  `lora_rows` scales rows (0 skips). f16/bf16 activations, rank x slots <= 512, inference only.
  `mlx_kquant.HAS_LORA_EPILOGUE` marks the build; `KQuantLinear(x, lora=(a_t, b_t, rows))` forwards.
  The epilogue kernels walk every LoRA operand as dense row-major memory; operands that evaluate
  to a strided view (an unevaluated array reports the default dense layout at op build, and
  `mx.repeat` of one value evaluates to a stride-0 broadcast) are copied into a dense temporary
  by a small `kq_lora_densify` dispatch at eval, on the primitive's own stream.
- **`gather_qmm_seg`** + **`expert_tile_map`** - expert-sorted MoE prefill as one GEMM per expert
  segment instead of per-row gathers. `expert_tile_map` builds the 64-row tile map on the GPU from the
  sorted routing indices (no host sync); `gather_qmm_seg` walks it. Gated by `KQ_SWITCH_GEMM_MIN_ROWS`
  (see [README](../README.md#environment-variables)).

On NAX GPUs, `quantized_matmul` transpose (decode-orientation) shapes route by row count M: the
mat-vec paths up to a per-codec crossover (M 6-9), a BM=32 double-buffered NAX tile through M 32,
the BM=64 tile above that with a double-buffered `_db` variant on the M 33-64 band at large N, and
a BM=128 tile from M 193 when ceil(M/64) is even. Every floor is a measured per-codec policy
(`kq_smallbm_policy` in `src/kquant_matmul.cpp`).

Tuning levers (defaults are right for normal use):

- `KQ_NAX_SMALL_BM` - small-M routing. `0` restores the old routing (mat-vec paths below M 13 and
  no BM=32 tile), `2` forces BM=32 for policy-excluded codecs, unset or `1` follows the per-codec
  policy.
- `KQ_NAX_BM128` - BM=128 band. `0` off, `1` forces the floor to M 193 for every codec, `2` drops
  the floor entirely (any even ceil(M/64), probing the M65-128 wash band), unset follows the
  per-codec entry floors (193/449/961 tiers, measured on M5 Max by
  `benchmarks/bench_qmm_bm128_ab.py`; re-run it before trusting them on new silicon).
- `KQ_NAX_DB64` - double-buffered M 33-64 band. `0` off, `1` drops the N floor, unset follows the
  per-codec N floors. Only the five policy-enabled codecs (q6_k, q8_0, q4_1, q5_1, q5_0) carry
  `_db` instantiations, so `1` is bounded by availability; probing another codec needs its
  instantiation restored and a metallib rebuild.
- `KQ_FORCE_QMM_MIN_M` - probe lever: routes transpose shapes with M at or above the value straight
  to the NAX qmm, bypassing the mat-vec route claims, for crossover measurement below M 13. Unset
  (off) by default.
- `KQ_NAX_SWIZZLE` - `1` enables the row-tile traversal swizzle (folds row-tiles into grid.x for
  SLC reuse of the weight band). Falsified on M5 Max, where the M>64 band is per-threadgroup-bound
  rather than DRAM-bound; kept as a probe for future silicon. Default off.
- `KQ_MV_EXT_NR` - `2` selects the two-rows-per-thread `mv_ext` variant (q6_k, M 5-12), which
  halves activation cache traffic but measured no faster than the shipped kernels. Kept as a probe
  for future silicon. Default `1` (shipped behavior).
- `KQ_QMM_SPLITK_NAX` - split-K on the NAX BM=32 tile; `0` disables the route, a value at or above
  `1` forces it and sets the target slice count. Unset takes the per-codec entry M in
  `kq_splitk_nax_min_m`, measured on M5 Max. Every codec with NAX kernels, M <= 32; read live per
  call, so both arms can share one process.
- `KQ_QMM_SPLITK` - the same lever for the plain small-M qmm, used when NAX is absent or disabled.
  Entry points come from a per-device table. K-quants, legacy quants and the IQ codecs, M <= 32.
- `KQ_MV_EXT_SB` / `KQ_MV_EXT_NX` / `KQ_MV_EXT_HD` - `mv_ext` activation-traffic experiments:
  shuffle-broadcast (`1`), wide nxpsg (`16`/`32`), half-precision chunk dots (`1`). q6_k M 4-12
  only. `HD` measured +4-5% at M 8; the rest flat to negative on M5 Max. Kept as probes. Default
  off.

`stq1_0` (structured-sparse ternary, llama.cpp PR #22836) ships the full ALU and NAX kernel set,
but only the pre-NAX floors are measured (M3 Max: plain split-K entry M 5). Its NAX policy is
inherited from `iq1_s` and needs M5-silicon calibration: `bm128_min_m`
(`benchmarks/bench_qmm_bm128_ab.py`), the NAX split-K entry in `kq_splitk_nax_min_m`
(`benchmarks/bench_verify_band_ab.py`), `kq_splitk_min_m_nax_alu`, and db64 candidacy (no `_db`
instantiation yet).

## MoE GLU

Fused gate/up expert matvecs with the GLU epilogue applied in the same dispatch, so each activation
load feeds both projections.

- **`moe_glu_gather_kq`** - fused MoE GLU gather for K-quant expert stacks: `act(gate) * up` in one
  decode-shaped dispatch. Bias-free for most codecs; the fp4 wire codecs (`mxfp4`/`nvfp4`) also take
  per-expert gate/up biases with the `swiglu_clamp` activation (gpt-oss experts).
- **`moe_glu_gather_shexp_kq`** - the same with the block's shared expert folded in as an extra slot.
- **`moe_glu_gather`** - the MLX packed-mxfp4 counterpart.
- **`moe_router_topk`** - the router in one dispatch: f32 scoring (`softmax`, or `sqrtsoftplus` for
  DeepSeek-V4), top-k with a min-index tie-break, optional bias-ranked selection, optional
  renormalization, and an optional per-expert scale.

The GLU activation is selected per model: plain SwiGLU/GELU, the clamped `silu_limit`
(`silu(min(g, limit)) * clip(u, -limit, limit)`) that DeepSeek-V4's `LimitedSwiGLU` needs, or
`swiglu_clamp` (gpt-oss clamped SwiGLU: biases added, sigmoid slope `alpha`, and a `(u + 1)` linear
term; requires the expert biases and is instantiated for `mxfp4`/`nvfp4` only).

## Attention

Scaled-dot-product variants for shapes stock MLX's fused allowlist excludes, plus the sparse
mechanism below.

- **`sdpa_vector`** - vector SDPA for large head dims (256, 512) - e.g. DeepSeek MLA - which MLX's
  fused vector path does not cover.
- **`sdpa_decode_gqa`** - decode/verify GQA tuned for long KV caches: the key axis splits into coarse
  chunks streamed through threadgroup-staged K/V tiles shared by the GQA group, so device memory reads
  the KV once per chunk. Optional `starts` (int32 `[B]`) restricts row b to keys `[starts[b], kL)` for
  left-padded batches, skipping fully padded-out chunks. Optional affine q8 K/V operands (scales and
  biases, bits 8, group 64) dequantize on the tile stage. `return_lse=True` adds per-row log-sum-exp.
- **`sdpa_decode_gqa_cascade`** - shared-prefix batched decode: every row attends one common prefix
  plus its own private suffix. The prefix is walked once for all rows on the matrix-unit tile, private
  suffixes run per row, one merge pass folds both; 1.6-4.2x over per-row calls at 14k-32k prefixes.
  qL 1-8 (verify width, end-aligned causal); takes `starts` and the q8 operands on either region.
- **`sdpa_decode_gqa_paged`** - sparse page-gather decode: attends only the K/V pages listed per
  (batch, kv-head), so cost tracks the selected keys rather than the cache length. The page unit is
  the staged tile height (32 rows at head dim 64/128, 16 at 256, 8 at 512); `tile_c` overrides it
  where instantiated (4-row pages at head dim 256, matching a 4-token block-sparse selection unit);
  takes `starts`.
- **`sdpa_prefill_block_sparse`** - block-sparse FA prefill over 4-row K/V pages at head dim 256:
  queries fold into 4-wide windows (with the GQA group, one MMA tile), and each window walks only
  its own page list with per-page membership bitmasks, so a prefill chunk pays for the selected
  blocks instead of the full key axis and no `[L, S]` mask is materialized. Causality inside the
  diagonal page is enforced in-kernel.
- **`sdpa_fa_verify`** - speculative-verify attention on the matrix units for a GQA-folded query tile.
  Head dims 64 through 512; `return_lse` as above.
- **`sdpa_fa_verify_kvarn`** - the same matrix-unit verify pass over a KVarN cache: sealed records
  dequantize at tile stage through the loaders `sdpa_decode_gqa_kvarn` uses, so the result matches
  `sdpa_fa_verify` over the materialized cache bit for bit. The verify-width route (q_len 2 to 8):
  the vector kernel's cost climbs steeply past two queries, the matrix tile prices extra rows at
  nearly zero. Head dims 128, 256 and 512; `n_attend` / `full_visibility` / `return_lse` as the
  decode op.

## DeepSeek/GLM sparse attention (DSA)

The DeepSeek-V4-Flash / GLM lightning-indexer attention: a lightweight indexer scores every pooled
(compressed) KV row against the query, a top-k select picks the rows to attend to, and sparse
attention runs the local sliding window plus those gathered rows in one pass. Ported, with
modifications, from omlx's `glm_moe_dsa` custom kernels (see the
[acknowledgement](../README.md#acknowledgements)). All six accept `qL >= 1`, so decode, MTP verify
(`qL = 2`), and prefill share them.

- **`dsa_indexer_scores`** - indexer relevance scores over a prefill query tile (steel GEMM):
  `out[b,0,m,n] = sum_h relu(q[b,h,m] . k[b,0,n]) * w[h,m]`.
- **`dsa_indexer_score_decode`** - the same for decode-width (`qL <= 4`) queries without materializing
  the per-head `[H, P]` scores. 4, 32 or 64 heads; fp32 head weights are read as-is.
- **`dsa_topk_indices`** - per-row top-k arg-select over the 16-bit scores (2-pass radix select). The
  selected index *set* matches a full sort; the order within a row does not.
- **`dsa_sparse_attention`** - the sliding local window plus the indexer-selected gathered rows plus
  per-head attention sinks, in one flash-softmax dispatch (f32 accumulation).
- **`dsa_kv_qat`** / **`dsa_indexer_qat`** - the fused quantization-aware round-trips DeepSeek-V4 does
  on its main-attention KV (per-64-block FP8-E4M3FN) and indexer activations (128-wide Hadamard then
  per-32-block FP4-E2M1), each bit-identical to the equivalent MLX graph. `dsa_kv_qat(...,
  f16_round=False)` drops the trailing fp16 round for the compressor emit path, whose pooled rows are
  quantized but never stored in the f16 KV cache.

Tuning levers (defaults are right for normal use):

- `KQ_DSA_BK` - key-tile width for `dsa_sparse_attention`, `128` or `256`. Default: `128` for top-k
  lists up to 128 entries, `256` for denser ones.
- `KQ_DSA_SPLIT` - `1`/`0` forces the split-KV decode route on/off. Default: auto, on for the
  small-grid decode/verify shapes where a single threadgroup would leave the GPU idle.

## Normalization fusions

- **`add_rmsnorm`** - fused post-norm residual `(residual + rms_norm(h, weight)) * scale`, all in f32.
- **`rmsnorm2_add`** - two independent RMS norms plus an add in one dispatch.
- **`rmsnorm_multi3`** - three RMS norms of one tensor sharing its mean-square reduction (the QK-norm
  plus a third head-norm shape).

## Introspection

- **`codecs`** - the list of supported codec names.
- **`metallib_loads`** / **`metallib_dir`** - whether the bundled metallib opened on the device, and
  where it lives.
- **`nax_available`** / **`nax_gather_enabled`** - whether the GPU exposes NAX tensor units, and
  whether the sorted-gather NAX GEMM leaf is reachable for a codec.
- **`cpu_neon_available`** - whether the arm64 NEON int8 GEMV path is compiled in.

## Feeder-loop primitives

The zero-copy arena buffers and shared-event stream primitives (`arena_alloc`, `event_signal` /
`event_wait`, `shared_event_*`, `zero_copy_view_count`, `verify_zero_copy_views`, `load_gguf`) support
a producer/consumer decode loop and are a separate subsystem; see
[docs/feeder/DESIGN.md](feeder/DESIGN.md).

- **`route_shed`** - routed-expert slot remap plus residency shed for streamed MoE decode: expert ids
  map to arena slots through a resident-slot table, non-resident experts are shed with their gate
  mass renormalized onto the kept ones, and the misses come back (ids and scores) for between-token
  prestaging. No host sync.
