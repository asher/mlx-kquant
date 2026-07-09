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
- **`gather_qmm_seg`** + **`expert_tile_map`** - expert-sorted MoE prefill as one GEMM per expert
  segment instead of per-row gathers. `expert_tile_map` builds the 64-row tile map on the GPU from the
  sorted routing indices (no host sync); `gather_qmm_seg` walks it. Gated by `KQ_SWITCH_GEMM_MIN_ROWS`
  (see [README](../README.md#environment-variables)).

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
  the KV once per chunk.
- **`sdpa_fa_verify`** - speculative-verify attention on the matrix units for a GQA-folded query tile.

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
  the per-head `[H, P]` scores.
- **`dsa_topk_indices`** - per-row top-k arg-select over the 16-bit scores (2-pass radix select). The
  selected index *set* matches a full sort; the order within a row does not.
- **`dsa_sparse_attention`** - the sliding local window plus the indexer-selected gathered rows plus
  per-head attention sinks, in one flash-softmax dispatch (f32 accumulation).
- **`dsa_kv_qat`** / **`dsa_indexer_qat`** - the fused quantization-aware round-trips DeepSeek-V4 does
  on its main-attention KV (per-64-block FP8-E4M3FN) and indexer activations (128-wide Hadamard then
  per-32-block FP4-E2M1), each bit-identical to the equivalent MLX graph.

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
