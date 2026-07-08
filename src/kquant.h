// Public C++ surface for mlx-kquant: the kq.* ops and their Primitive
// subclasses (toolchain self-checks, dequantize, quantized_matmul, gather_qmm,
// and quantize).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace mlx_kquant {

// ----------------------------- toolchain self-checks
// -----------------------------

// Directory containing the bundled mlx_kquant.metallib (colocated with
// _ext.so).
std::string metallib_dir();

// Load the bundled metallib via MLX's Metal device. Throws on failure.
// Returns false only on a non-Metal build.
bool metallib_loads();

// True when the GPU supports the NAX (tensor-core) matmul kernels (false on
// non-Metal builds and pre-NAX GPUs).
bool nax_available();

// True when gather_qmm's sorted-rhs NAX GEMM leaf can serve `kquant_type`
// here: NAX hardware present, the codec ships NAX kernels, and KQ_DISABLE_NAX
// is unset (read live). Callers with their own sorted-prefill arms (e.g. the
// gather_qmm_seg switch arm) defer to gather_qmm when this holds.
bool nax_gather_enabled(const std::string& kquant_type);

// ----------------------------- ops -----------------------------

// Dequantize GGUF K-quant wire bytes `w` (uint8, last dim a multiple of the
// codec's bytes_per_block) into a float array. `scales` is a vestigial
// placeholder - K-quant scales live inside `w`; the argument exists only for
// buffer-signature parity with the Metal kernel and is otherwise ignored.
// group_size/bits are derived from `kquant_type`. Default output dtype is
// float16.
mx::array dequantize(
    const mx::array& w,
    const mx::array& scales,
    const std::string& kquant_type,
    std::optional<mx::Dtype> dtype = std::nullopt,
    mx::StreamOrDevice s = {});

// Quantized matmul: x (float) @ dequant(w). `w` is uint8 K-quant wire bytes,
// `scales` a vestigial placeholder (see dequantize). transpose=true means
// `w` is laid out [N, K] (the row-major weight convention). Output dtype is
// x.dtype(), except float32 x is promoted to bfloat16.
// group_size/bits are derived from `kquant_type`.
mx::array quantized_matmul(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    bool transpose = true,
    mx::StreamOrDevice s = {});

// Bias-fused quantized matmul: x (float) @ dequant(w) + bias, fusing the add
// into the same kernel dispatch instead of a separate elementwise op.
// Decode-only: x must carry exactly one row (x.shape(-2) == 1 after
// flattening leading batch dims), matching how KQuantLinear is called during
// B=1 token-at-a-time decode -- throws outside that shape. Only kquant_type
// "q8_0" is wired so far. transpose is always true (the only regime
// KQuantLinear uses); group_size/bits are derived from kquant_type. Callers
// should fall back to quantized_matmul() + a separate add for every other
// shape (prefill, verify/MTP) or codec.
mx::array quantized_matmul_qmv_bias(
    mx::array x,
    mx::array w,
    mx::array scales,
    mx::array bias,
    const std::string& kquant_type,
    mx::StreamOrDevice s = {});

// Gather (mixture-of-experts) quantized matmul: for each output row, select an
// expert weight matrix via `rhs_indices` and an x row via `lhs_indices`, then
// matmul. `w` is uint8 K-quant wire bytes shaped (n_experts, N, bytes_per_row);
// `scales` a vestigial placeholder (see dequantize). Either index defaults to a
// plain arange when omitted. `sorted_indices` enables an optimized path when
// the (defaulted) indices are sorted. Output dtype follows quantized_matmul.
// group_size/bits are derived from `kquant_type`.
mx::array gather_qmm(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    std::optional<mx::array> lhs_indices = std::nullopt,
    std::optional<mx::array> rhs_indices = std::nullopt,
    bool transpose = true,
    bool sorted_indices = false,
    mx::StreamOrDevice s = {});

// Build the gather_qmm_seg tile map from expert-sorted routing indices, on
// GPU (no host sync). Returns {map, counts}: map uint32 [cap, 3] rows of
// (expert, row_start, num_rows) tiling each segment into 64-row tiles where
// only the last tile of a segment can be partial; counts uint32 [1] = the
// number of valid tiles. The cap is a shape-only upper bound; slots past
// counts are uninitialized and tile order is unspecified. Metal-only.
std::vector<mx::array>
expert_tile_map(mx::array indices, int n_experts, mx::StreamOrDevice s = {});

// Segment-walking gather GEMM for expert-sorted MoE prefill. x is [R, K] rows
// pre-sorted by expert; w is uint8 wire bytes (n_experts, N, bytes_per_row);
// map / counts are the expert_tile_map outputs (or host-built equivalents):
// uint32 [*, 3] tile rows of (expert, row_start, num_rows <= 64), tiles
// never spanning experts. One dispatch computes
// y[row_start : row_start + num_rows] = x_seg @ dequant(w[expert]).T per tile
// at tiled-GEMM throughput; partial tiles skip dead row fragments in the MMA
// so ragged segments cost ~ceil(num_rows / 8) row fragments. Rows not
// covered by the valid tiles are left unwritten (the caller's map must cover
// every row exactly once). Output dtype follows quantized_matmul. Metal-only.
mx::array gather_qmm_seg(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    mx::array map,
    mx::array counts,
    mx::StreamOrDevice s = {});

// Quantize (encode) a float weight tensor `w` into GGUF K-quant wire bytes.
// Returns {wq (uint8), scales_placeholder (uint8, shape [1])}: K-quant scales
// live inline in `wq`, so the second output is a vestigial placeholder kept for
// signature parity with dequantize/quantized_matmul. Optional `imatrix` (a 1-D
// float32 importance vector of length K = w.shape(-1)) steers the encoder.
// group_size/bits are derived from `kquant_type`. Runs on CPU or Metal. The
// last dim of `w` must be a multiple of the codec's weights_per_block.
std::vector<mx::array> quantize(
    const mx::array& w,
    const std::string& kquant_type,
    const std::optional<mx::array>& imatrix = std::nullopt,
    mx::StreamOrDevice s = {});

// Vector scaled-dot-product attention for large head dims (e.g. 512) that stock
// MLX's fused vector allowlist {64,96,128,256} excludes. q/k/v are float
// [B, n_q_heads, qL, D] / [B, n_kv_heads, kL, D] (GQA: n_q_heads % n_kv_heads
// == 0); q is made row-contiguous, k/v are read in place via their strides.
// `causal` applies an offset causal mask (query row i attends keys <= kL - qL +
// i). Returns the attention output [B, n_q_heads, qL, D]. Metal-only.
mx::array sdpa_vector(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    bool causal = true,
    mx::StreamOrDevice s = {});

// Decode-time (qL == 1) GQA attention tuned for long KV: fixed coarse
// contiguous key splits plus threadgroup-staged K/V tiles shared across the
// GQA group, so device memory reads the KV once per kv-head. Optional
// per-q-head attention sinks (an extra softmax logit with no value row).
// q [B, n_q_heads, 1, D], k/v [B, n_kv_heads, kL, D] with contiguous D;
// head/seq strides are read in place. head_dim 64 only; gqa_factor 2..8.
// `splits` 0 picks the default (32); `tile_c` is the staged tile height
// (32 or 16). Metal-only.
mx::array sdpa_decode_gqa(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    const std::optional<mx::array>& sinks = std::nullopt,
    int splits = 0,
    int tile_c = 32,
    mx::StreamOrDevice s = {});

// Speculative-verify attention on the GPU matrix units for a GQA-folded query
// tile. The caller folds q [B, Hq, qL, D] -> [B, Hkv, G*qL, D] (kv-major
// heads) and passes the original qL, so folded row r is causally clamped to
// key <= kL - qL + (r % qL). One 32-row query tile streams each contiguous KV
// split once (S = Q @ K^T and O += P @ V on simdgroup_matrix, float32
// accumulators, online softmax per row); the per-split partials merge through
// the shared kq_sdpa_gqa 2-pass reduction. Requires B == 1, q heads == kv
// heads (folded), G*qL <= 32, qL in [2, 8], head_dim 256, kL >= qL. k/v are
// read in place via their head/seq strides (head_dim contiguous). `splits` 0
// picks the default. Metal-only.
mx::array sdpa_fa_verify(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    int q_len,
    int splits = 0,
    mx::StreamOrDevice s = {});

// Fused MoE GLU gather on the MLX packed mxfp4 layout: gate and up expert
// matvecs (sharing each activation load), expert biases, and the clamped
// SwiGLU epilogue out = (min(g, limit) * sigmoid(alpha * g)) * (clip(u,
// +-limit) + 1) in one dispatch. Decode-shaped: x [T, K] (one row per token,
// shared across that token's R expert slots), indices [T, R]; weights uint32
// [E, N, K/8] + scales uint8 [E, N, K/32] (group 32, 4-bit), biases [E, N].
// Returns [T, R, N] in x.dtype. Metal-only.
mx::array moe_glu_gather(
    mx::array x,
    mx::array gate_w,
    mx::array gate_scales,
    mx::array gate_bias,
    mx::array up_w,
    mx::array up_scales,
    mx::array up_bias,
    mx::array indices,
    float alpha = 1.702f,
    float limit = 7.0f,
    mx::StreamOrDevice s = {});

// Gathered matvec with the expert bias fused, same packed-mxfp4 layout as
// moe_glu_gather. x [T, R, K] (one row per expert slot), indices [T, R].
// Returns [T, R, N] in x.dtype. Metal-only.
mx::array gather_qmv_bias(
    mx::array x,
    mx::array w,
    mx::array scales,
    mx::array bias,
    mx::array indices,
    mx::StreamOrDevice s = {});

// K-quant counterpart of moe_glu_gather: gate and up expert matvecs on GGUF
// wire bytes (n_experts, out_dims, bytes_per_row) sharing each activation
// load, with the GLU epilogue act(g) * u fused (act: "silu", "gelu" or
// "silu_limit" -- the latter clamps g from above and u to +-limit before
// silu(g) * u, deepseek-v4 LimitedSwiGLU semantics, and requires limit > 0).
// No biases. x [T, K], indices [T, R]. Returns [T, R, N]. Metal-only;
// requires K % 256 == 0 and a codec with the fused kernel wired (full GGUF
// matrix).
mx::array moe_glu_gather_kq(
    mx::array x,
    mx::array gate_w,
    mx::array up_w,
    const std::string& kquant_type,
    mx::array indices,
    const std::string& act = "silu",
    float limit = 0.0f,
    mx::StreamOrDevice s = {});

// DeepSeek-V4-Flash sparse attention: sliding local window + gathered
// indexer-selected pooled rows + per-head sinks in one dispatch (flash
// online softmax, f32 accumulation). q [B, 64, qL, 512] (qL >= 1: decode,
// MTP verify and prefill all use this kernel), local_kv [B, 1, localL, 512]
// (K == V shared latent), pooled [B, P, 512], topk_indices [B, 1, qL, N]
// uint32 (invalid/masked slots < 0 or >= the causal pooled horizon
// (q_offset + pos + 1) / compress_ratio contribute nothing), sinks [64].
// Returns [B, 64, qL, 512]. Metal-only.
mx::array dsa_sparse_attention(
    mx::array q,
    mx::array local_kv,
    mx::array pooled,
    mx::array topk_indices,
    mx::array sinks,
    float scale,
    int q_offset,
    int compress_ratio,
    int local_window,
    mx::StreamOrDevice s = {});

// DeepSeek-V4-Flash lightning-indexer relevance scores (steel GEMM):
// out[b, 0, m, n] = sum_h relu(q[b, h, m, :] . k[b, 0, n, :]) * w_h_m.
// q [B, H, M, 128] (H 32 or 64), k [B, 1, N, 128]; weights either
// [B, M, H] ("lh" layout) or [B, H, M, 1]. Requires M % 64 == 0 and
// N % 64 == 0 (pad or fall back Python-side; decode pads the single query
// row to 64 and keeps row 0). causal masks n > causal_q_offset + m with
// -inf (causal_q_offset -1 means N - M); with skip_causal_future_store the
// fully-masked tiles are left unwritten instead (only valid when the
// consumer never reads them, e.g. a causal_valid_prefix top-k).
// Returns [B, 1, M, N]. Metal-only.
mx::array dsa_indexer_scores(
    mx::array queries,
    mx::array keys,
    mx::array weights,
    bool causal = true,
    int unused_causal_prefix_topk = 0,
    bool skip_causal_future_store = false,
    int causal_q_offset = -1,
    mx::StreamOrDevice s = {});

// Per-row top-k arg-select over 16-bit float scores (2-pass radix over the
// orderable bit transform; one threadgroup per row). scores [B, 1, L, K]
// fp16/bf16, topk in {512, 2048}, K >= topk. The selected set matches a
// full sort; intra-row order does not (threshold ties admitted in scan
// order). bucketed emits >threshold before ==threshold deterministically.
// causal_valid_prefix clamps each row's scan to K - L + row%L + 1 entries
// (scores laid out with queries as trailing rows) and emits the identity
// prefix when the valid range fits inside topk. Returns [B, 1, L, topk]
// uint32. Metal-only.
mx::array dsa_topk_indices(
    mx::array scores,
    int topk,
    bool bucketed = false,
    bool causal_valid_prefix = false,
    mx::StreamOrDevice s = {});

// DeepSeek-V4-Flash indexer activation QAT round-trip, fused: 128-wide
// Hadamard (mlx hadamard_n butterfly order and 1/sqrt(128) scale, exactly)
// then the per-32-block FP4-E2M1 round-trip (scale 2^ceil(log2(amax/6)),
// amax floor FLT_MIN*6, clamp +-6, tie-to-even threshold ladder). x is any
// shape with a trailing dim of 128; returns the same shape and dtype.
// Bit-compatible with the mx.hadamard_transform + compiled fp4-core chain.
// Metal-only.
mx::array dsa_indexer_qat(mx::array x, mx::StreamOrDevice s = {});

// K-quant gathered matvec (down projection), same wire layout. x [T, R, K]
// (one row per expert slot), indices [T, R]. Returns [T, R, N]. Metal-only.
mx::array gather_qmv_kq(
    mx::array x,
    mx::array w,
    const std::string& kquant_type,
    mx::array indices,
    mx::StreamOrDevice s = {});

// moe_glu_gather_kq with the block's shared expert folded in as one extra
// slot: shexp_gate_w / shexp_up_w are single-expert 2-D wire-byte tensors
// [N, bytes_per_row(shexp codec)] shape-matched to one expert stack row.
// shexp_kquant_type defaults to kquant_type; a different codec (mixed-codec
// blocks, UD-style upcast shexp) must be q6_k or q8_0.
// Returns [T, R + 1, N]; the last slot is the shared expert.
mx::array moe_glu_gather_shexp_kq(
    mx::array x,
    mx::array gate_w,
    mx::array up_w,
    mx::array shexp_gate_w,
    mx::array shexp_up_w,
    const std::string& kquant_type,
    mx::array indices,
    const std::string& act = "silu",
    const std::string& shexp_kquant_type = "",
    mx::StreamOrDevice s = {});

// Down projection with the routing mix folded in: x [T, S, K] (slot S-1 =
// shared expert), indices [T, S-1], scores [T, S] (routed weights then the
// sigmoid shared-expert gate). Accumulates all S slots in f32 and returns
// [T, N] -- replaces gather + (y * scores).sum + shared add. Metal-only.
mx::array gather_qmv_mix_kq(
    mx::array x,
    mx::array w,
    mx::array shexp_w,
    const std::string& kquant_type,
    mx::array indices,
    mx::array scores,
    const std::string& shexp_kquant_type = "",
    mx::StreamOrDevice s = {});

// No-shared-expert routing mix (gemma-style MoE): x [T, S, K], indices
// [T, S], scores [T, S]; every slot gathers from the expert stack and the
// score-weighted sum accumulates in f32. Returns [T, N]. Metal-only.
mx::array gather_qmv_mix_ns_kq(
    mx::array x,
    mx::array w,
    const std::string& kquant_type,
    mx::array indices,
    mx::array scores,
    mx::StreamOrDevice s = {});

// Router top-k in one dispatch: softmax over E logit columns in f32, pick
// the top_k largest (min-index tie-break), optionally renormalize the picked
// probabilities (norm_topk_prob; equals softmax over the selected raw logits
// -- the gemma router semantics). With shared_gate (qwen3-next), logits are
// [T, E + 1], column E is the shared-expert gate logit and its sigmoid lands
// in the last scores slot; without, logits are [T, E] and scores are
// [T, top_k]. Optional per_expert_scale ([E] float) multiplies each picked
// score by its expert's scale (gemma). Returns {indices [T, top_k] uint32,
// scores float32} -- exactly the inputs moe_glu_gather_shexp_kq /
// gather_qmv_mix*_kq consume. E <= 1024, top_k <= 16. Metal-only.
std::vector<mx::array> moe_router_topk(
    mx::array logits,
    int top_k,
    bool norm_topk_prob,
    bool shared_gate = true,
    const std::optional<mx::array>& per_expert_scale = std::nullopt,
    mx::StreamOrDevice s = {});

// Fused residual + RMSNorm glue (one dispatch each; see kq_norm_fused.h).
// rms_norm(x, w) = w * x * rsqrt(mean(x^2) + eps) over the last axis, all
// math in f32, matching mx::fast::rms_norm. float16/bfloat16 only; weights
// (and the optional scale) must match the activation dtype. Metal-only.

// out = (residual + rms_norm(h, weight)) * scale; scale is an optional
// size-1 array (gemma-4's layer_scalar), 1.0 when absent.
mx::array add_rmsnorm(
    mx::array h,
    mx::array residual,
    mx::array weight,
    float eps,
    const std::optional<mx::array>& scale = std::nullopt,
    mx::StreamOrDevice s = {});

// {rms_norm(x, w0), rms_norm(x, w1), rms_norm(x, w2)} sharing one
// mean-square reduction of x.
std::vector<mx::array> rmsnorm_multi3(
    mx::array x,
    mx::array w0,
    mx::array w1,
    mx::array w2,
    float eps,
    mx::StreamOrDevice s = {});

// out = rms_norm(a, wa) + rms_norm(b, wb).
mx::array rmsnorm2_add(
    mx::array a,
    mx::array wa,
    mx::array b,
    mx::array wb,
    float eps,
    mx::StreamOrDevice s = {});

// ----------------------------- primitives -----------------------------

// Dequantize a single uint8 K-quant wire-byte tensor. Inference-only:
// jvp/vjp/vmap inherit the base-class throwing defaults.
class KQuantDequantize : public mx::Primitive {
 public:
  explicit KQuantDequantize(
      mx::Stream stream,
      std::string kquant_type,
      int group_size,
      int bits)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        group_size_(group_size),
        bits_(bits) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantDequantize";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
  int group_size_;
  int bits_;
};

// Quantized matmul. vjp implements only the gradient wrt x (re-dispatch with
// the transpose flipped) so LoRA can train on a frozen kquant base; jvp/vmap
// and the gradient wrt the quantized weights/scales inherit the base-class
// throwing defaults. Split-k perf paths (qmm_splitk / qvm_split_k) are gated
// off - they depend on the unexported strided_reduce_general_dispatch; plain
// qmm/qvm produce identical results with less parallelism.
class KQuantMatmul : public mx::Primitive {
 public:
  explicit KQuantMatmul(
      mx::Stream stream,
      std::string kquant_type,
      int group_size,
      int bits,
      bool transpose)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        group_size_(group_size),
        bits_(bits),
        transpose_(transpose) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantMatmul";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

  // Gradient wrt x only (arg 0): re-dispatch the matmul with the transpose
  // flipped. The quantized weights/scales are frozen - arg 1/2 throw.
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;

 private:
  std::string kquant_type_;
  int group_size_;
  int bits_;
  bool transpose_;
};

// Bias-fused decode-only (M=1) qmv/qmv_fast dispatch -- see
// quantized_matmul_qmv_bias() above for the shape contract eval_gpu enforces.
// Inference-only, like the SDPA primitive below: no vjp override, so a grad
// request throws via the base-class default (KQuantLinear always freezes
// weight/scales/bias, so this is never exercised in practice).
class KQuantQmvBias : public mx::Primitive {
 public:
  explicit KQuantQmvBias(
      mx::Stream stream,
      std::string kquant_type,
      int group_size,
      int bits)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        group_size_(group_size),
        bits_(bits) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantQmvBias";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
  int group_size_;
  int bits_;
};

// Vector SDPA for large head dims. Inference-only: jvp/vjp/vmap inherit the
// base-class throwing defaults. eval_cpu throws (Metal-only kernel).
class KQuantSDPA : public mx::Primitive {
 public:
  explicit KQuantSDPA(mx::Stream stream, float scale, bool causal)
      : mx::Primitive(stream), scale_(scale), causal_(causal) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantSDPA";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  float scale_;
  bool causal_;
};

// Decode-time GQA attention (see sdpa_decode_gqa). Sinks presence is encoded
// in the input count (q, k, v[, sinks]). Inference-only.
class KQuantSDPAGQA : public mx::Primitive {
 public:
  explicit KQuantSDPAGQA(mx::Stream stream, float scale, int splits, int tile_c)
      : mx::Primitive(stream),
        scale_(scale),
        splits_(splits),
        tile_c_(tile_c) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantSDPAGQA";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  float scale_;
  int splits_;
  int tile_c_;
};

// Simdgroup-matrix FA verify attention (see sdpa_fa_verify). Inference-only.
class KQuantSDPAFAVerify : public mx::Primitive {
 public:
  explicit KQuantSDPAFAVerify(
      mx::Stream stream,
      float scale,
      int q_len,
      int splits)
      : mx::Primitive(stream), scale_(scale), q_len_(q_len), splits_(splits) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantSDPAFAVerify";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  float scale_;
  int q_len_;
  int splits_;
};

// Fused MoE GLU gather (see moe_glu_gather). Inference-only.
class KQuantMoEGLU : public mx::Primitive {
 public:
  explicit KQuantMoEGLU(mx::Stream stream, float alpha, float limit)
      : mx::Primitive(stream), alpha_(alpha), limit_(limit) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantMoEGLU";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  float alpha_;
  float limit_;
};

// Gathered matvec with fused expert bias (see gather_qmv_bias).
// Inference-only.
class KQuantGatherQMVBias : public mx::Primitive {
 public:
  explicit KQuantGatherQMVBias(mx::Stream stream) : mx::Primitive(stream) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantGatherQMVBias";
  }
  bool is_equivalent(const mx::Primitive& other) const override;
};

// DeepSeek-V4-Flash sparse attention (see dsa_sparse_attention).
// Inference-only, Metal-only.
class KQDsaSparseAttention : public mx::Primitive {
 public:
  explicit KQDsaSparseAttention(
      mx::Stream stream,
      float scale,
      int q_offset,
      int compress_ratio,
      int local_window)
      : mx::Primitive(stream),
        scale_(scale),
        q_offset_(q_offset),
        compress_ratio_(compress_ratio),
        local_window_(local_window) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQDsaSparseAttention";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  float scale_;
  int q_offset_;
  int compress_ratio_;
  int local_window_;
};

// DeepSeek-V4-Flash indexer relevance scores (see dsa_indexer_scores).
// Inference-only, Metal-only.
class KQDsaIndexerScores : public mx::Primitive {
 public:
  explicit KQDsaIndexerScores(
      mx::Stream stream,
      bool causal,
      bool weights_lh,
      int unused_causal_prefix_topk,
      bool skip_causal_future_store,
      int causal_q_offset)
      : mx::Primitive(stream),
        causal_(causal),
        weights_lh_(weights_lh),
        unused_causal_prefix_topk_(unused_causal_prefix_topk),
        skip_causal_future_store_(skip_causal_future_store),
        causal_q_offset_(causal_q_offset) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQDsaIndexerScores";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  bool causal_;
  bool weights_lh_;
  int unused_causal_prefix_topk_;
  bool skip_causal_future_store_;
  int causal_q_offset_;
};

// DeepSeek-V4-Flash indexer top-k select (see dsa_topk_indices).
// Inference-only, Metal-only.
class KQDsaTopKIndices : public mx::Primitive {
 public:
  explicit KQDsaTopKIndices(
      mx::Stream stream,
      int topk,
      bool bucketed,
      bool causal_valid_prefix)
      : mx::Primitive(stream),
        topk_(topk),
        bucketed_(bucketed),
        causal_valid_prefix_(causal_valid_prefix) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQDsaTopKIndices";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int topk_;
  bool bucketed_;
  bool causal_valid_prefix_;
};

// DeepSeek-V4-Flash fused indexer QAT round-trip (see dsa_indexer_qat).
// Inference-only, Metal-only.
class KQDsaIndexerQat : public mx::Primitive {
 public:
  explicit KQDsaIndexerQat(mx::Stream stream) : mx::Primitive(stream) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQDsaIndexerQat";
  }
  bool is_equivalent(const mx::Primitive& other) const override;
};

// K-quant fused MoE GLU gather (see moe_glu_gather_kq). Inference-only.
class KQuantMoEGLUKQ : public mx::Primitive {
 public:
  explicit KQuantMoEGLUKQ(
      mx::Stream stream,
      std::string kquant_type,
      std::string act,
      float limit = 0.0f)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        act_(std::move(act)),
        limit_(limit) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantMoEGLUKQ";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
  std::string act_;
  float limit_;
};

// K-quant gathered matvec (see gather_qmv_kq). Inference-only.
class KQuantGatherQMVKQ : public mx::Primitive {
 public:
  explicit KQuantGatherQMVKQ(mx::Stream stream, std::string kquant_type)
      : mx::Primitive(stream), kquant_type_(std::move(kquant_type)) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantGatherQMVKQ";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
};

// K-quant fused MoE GLU gather with shared-expert slot (see
// moe_glu_gather_shexp_kq). Inference-only.
class KQuantMoEGLUShexpKQ : public mx::Primitive {
 public:
  explicit KQuantMoEGLUShexpKQ(
      mx::Stream stream,
      std::string kquant_type,
      std::string act,
      std::string shexp_type)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        act_(std::move(act)),
        shexp_type_(std::move(shexp_type)) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantMoEGLUShexpKQ";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
  std::string act_;
  std::string shexp_type_;
};

// K-quant gathered matvec with routing mix folded in (see gather_qmv_mix_kq).
// Inference-only.
class KQuantGatherQMVMixKQ : public mx::Primitive {
 public:
  explicit KQuantGatherQMVMixKQ(
      mx::Stream stream,
      std::string kquant_type,
      std::string shexp_type)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        shexp_type_(std::move(shexp_type)) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantGatherQMVMixKQ";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
  std::string shexp_type_;
};

// No-shared-expert routing mix (see gather_qmv_mix_ns_kq). Inference-only.
class KQuantGatherQMVMixNSKQ : public mx::Primitive {
 public:
  explicit KQuantGatherQMVMixNSKQ(mx::Stream stream, std::string kquant_type)
      : mx::Primitive(stream), kquant_type_(std::move(kquant_type)) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantGatherQMVMixNSKQ";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
};

// Router softmax + top-k + score epilogue (see moe_router_topk). Two
// outputs (indices, scores). Inference-only.
class KQuantMoERouterTopK : public mx::Primitive {
 public:
  explicit KQuantMoERouterTopK(
      mx::Stream stream,
      int top_k,
      bool norm,
      bool shared,
      bool has_pes)
      : mx::Primitive(stream),
        top_k_(top_k),
        norm_(norm),
        shared_(shared),
        has_pes_(has_pes) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantMoERouterTopK";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int top_k_;
  bool norm_;
  bool shared_;
  bool has_pes_;
};

// Fused (residual + rms_norm(h, w)) * scale (see add_rmsnorm).
// Inference-only.
class KQuantAddRMSNorm : public mx::Primitive {
 public:
  explicit KQuantAddRMSNorm(mx::Stream stream, float eps, bool has_scale)
      : mx::Primitive(stream), eps_(eps), has_scale_(has_scale) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantAddRMSNorm";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  float eps_;
  bool has_scale_;
};

// Three rms_norms of one tensor sharing the reduction (see rmsnorm_multi3).
// Inference-only.
class KQuantRMSNormMulti3 : public mx::Primitive {
 public:
  explicit KQuantRMSNormMulti3(mx::Stream stream, float eps)
      : mx::Primitive(stream), eps_(eps) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantRMSNormMulti3";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  float eps_;
};

// rms_norm(a, wa) + rms_norm(b, wb) in one dispatch (see rmsnorm2_add).
// Inference-only.
class KQuantRMSNorm2Add : public mx::Primitive {
 public:
  explicit KQuantRMSNorm2Add(mx::Stream stream, float eps)
      : mx::Primitive(stream), eps_(eps) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantRMSNorm2Add";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  float eps_;
};

// Gather (MoE) quantized matmul. vjp implements only the gradient wrt x (a
// per-expert gather with the transpose flipped, scatter-added back to the
// source x rows) so LoRA can train on a frozen kquant base; jvp/vmap and the
// gradient wrt the weights/scales/indices inherit the base-class throwing
// defaults. The gather_qmm_rhs fast path (the only function-constant kernel) is
// deferred - it requires right_sorted_ == true, i.e. rhs_indices defaulted +
// sorted_indices, which callers never do (they always pass rhs_indices
// explicitly). The dispatch falls through to gather_qmm / gather_qmv, which is
// correctness-preserving for all inputs.
class KQuantGatherQMM : public mx::Primitive {
 public:
  explicit KQuantGatherQMM(
      mx::Stream stream,
      std::string kquant_type,
      int group_size,
      int bits,
      bool transpose,
      bool left_sorted,
      bool right_sorted)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        group_size_(group_size),
        bits_(bits),
        transpose_(transpose),
        left_sorted_(left_sorted),
        right_sorted_(right_sorted) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantGatherQMM";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

  // Gradient wrt x only (arg 0): gather the cotangent against the experts with
  // the transpose flipped, then scatter-add the rows back via lhs_indices. The
  // quantized weights/scales (arg 1/2) and the indices (arg 3/4) throw.
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;

 private:
  std::string kquant_type_;
  int group_size_;
  int bits_;
  bool transpose_;
  bool left_sorted_;
  bool right_sorted_;
};

// Tile-map builder (see expert_tile_map). Multi-output: {map, counts}.
// Inference-only, Metal-only.
class KQuantExpertTileMap : public mx::Primitive {
 public:
  explicit KQuantExpertTileMap(mx::Stream stream, int n_experts)
      : mx::Primitive(stream), n_experts_(n_experts) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantExpertTileMap";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int n_experts_;
};

// Segment-walking gather GEMM (see gather_qmm_seg). Inference-only, Metal-only:
// eval_cpu and the default vjp/jvp throw.
class KQuantGatherQMMSeg : public mx::Primitive {
 public:
  explicit KQuantGatherQMMSeg(
      mx::Stream stream,
      std::string kquant_type,
      int group_size,
      int bits)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        group_size_(group_size),
        bits_(bits) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantGatherQMMSeg";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
  int group_size_;
  int bits_;
};

// Encode a float weight tensor into K-quant wire bytes. Multi-output primitive:
// outputs[0] = wq (uint8), outputs[1] = scales placeholder (uint8, shape [1]).
// Whether an imatrix is used is derived from inputs.size() (1 = w only, 2 = w +
// imatrix). Inference-only: jvp/vjp/vmap throw. Runs on CPU or Metal.
class KQuantQuantize : public mx::Primitive {
 public:
  explicit KQuantQuantize(
      mx::Stream stream,
      std::string kquant_type,
      int group_size,
      int bits)
      : mx::Primitive(stream),
        kquant_type_(std::move(kquant_type)),
        group_size_(group_size),
        bits_(bits) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantQuantize";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  std::string kquant_type_;
  int group_size_;
  int bits_;
};

// ---------------- shared-event stream primitives ----------------
// Building blocks for the feeder decode loop (docs/feeder/DESIGN.md): a
// process-wide MTLSharedEvent registry with a host-side API, and two ops that
// encode a signal/wait against a registered event directly on the Metal
// command stream, so a single command buffer can hand control to a CPU feeder
// mid-stream and resume when it answers. Registry handles are opaque
// non-zero integers; every call throws on an unknown handle, and all of it
// throws on CPU-only builds (there is no event to encode).

// Create/destroy a shared event. Events start at signaled value 0.
uint64_t shared_event_create();
void shared_event_destroy(uint64_t handle);

// Host side: set the event's value (the GPU-visible signal, including the
// UINT64_MAX poison that releases any encoded wait), read it, or block until
// it reaches `value` (timeout_ms < 0 waits forever; returns false on
// timeout).
void shared_event_set(uint64_t handle, uint64_t value);
uint64_t shared_event_read(uint64_t handle);
bool shared_event_wait(uint64_t handle, uint64_t value, int64_t timeout_ms);

// Writable zero-copy host buffer: page-aligned host memory wrapped no-copy
// in a Metal shared-storage buffer, as a uint8 array of `shape` plus the raw
// base address (for the Python binding's writable memoryview over the same
// bytes; valid exactly as long as the array lives). The CPU-write -> GPU-read
// ordering contract is the caller's, via the event ops below.
std::pair<mx::array, uintptr_t> arena_alloc(const mx::Shape& shape);

// Stream side: identity ops on `x` that encode an MTLSharedEvent signal/wait
// at their position in the graph's evaluation order. The returned array
// aliases x and MUST be threaded into downstream compute (or evaluated
// explicitly) - an unused output is dead code and encodes nothing. eval_cpu
// is a plain identity so CPU placements remain legal no-ops.
mx::array event_signal(
    const mx::array& x,
    uint64_t handle,
    uint64_t value,
    mx::StreamOrDevice s = {});
mx::array event_wait(
    const mx::array& x,
    uint64_t handle,
    uint64_t value,
    mx::StreamOrDevice s = {});

class KQuantEventSignal : public mx::Primitive {
 public:
  explicit KQuantEventSignal(mx::Stream stream, uint64_t handle, uint64_t value)
      : mx::Primitive(stream), handle_(handle), value_(value) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantEventSignal";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  uint64_t handle_;
  uint64_t value_;
};

class KQuantEventWait : public mx::Primitive {
 public:
  explicit KQuantEventWait(mx::Stream stream, uint64_t handle, uint64_t value)
      : mx::Primitive(stream), handle_(handle), value_(value) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

  const char* name() const override {
    return "KQuantEventWait";
  }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  uint64_t handle_;
  uint64_t value_;
};

} // namespace mlx_kquant
