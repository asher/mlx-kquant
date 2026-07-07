// KQDsaIndexerScores / KQDsaTopKIndices primitives: the DeepSeek-V4-Flash
// lightning indexer -- a steel GEMM producing per-(query, pooled-token)
// relevance scores (sum_h relu(q_h . k) * w_h) and a one-threadgroup-per-row
// 2-pass radix arg-select picking the top-k score columns. The selected
// indices feed dsa_sparse_attention's gathered-pooled tiles.
// Ported from omlx glm_moe_dsa dsa_indexer.cpp; the kq build fetches both
// kernels from the AOT metallib with function constants (300 causal,
// 301 weights-lh, 302 bucketed emission). Inference-only (no CPU eval).
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/ops.h" // astype, contiguous, result_type
#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "../metal/mlx/backend/metal/kernels/kq_dsa_params.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

#ifdef _METAL_

void KQDsaIndexerScores::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& weights = inputs[2];
  auto& out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  constexpr int bm = 64;
  constexpr int bn = 64;
  constexpr int bk = 16;
  constexpr int wm = 2;
  constexpr int wn = 2;

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int M = q.shape(2);
  const int N = k.shape(2);
  const int D = q.shape(3);
  const int tiles_m = (M + bm - 1) / bm;
  const int tiles_n = (N + bn - 1) / bn;

  mlx::steel::GEMMParams params{
      /* const int M = */ M,
      /* const int N = */ N,
      /* const int K = */ D,
      /* const int lda = */ D,
      /* const int ldb = */ D,
      /* const int ldd = */ N,
      /* const int tiles_n = */ tiles_n,
      /* const int tiles_m = */ tiles_m,
      /* const int64_t batch_stride_a = */ int64_t(H) * M * D,
      /* const int64_t batch_stride_b = */ int64_t(N) * D,
      /* const int64_t batch_stride_d = */ int64_t(M) * N,
      /* const int swizzle_log = */ 0,
      /* const int gemm_k_iterations_aligned = */ D / bk,
      /* const int batch_ndim = */ 1};

  bool do_causal = causal_;
  bool use_weights_lh = weights_lh_;
  mx::metal::MTLFCList func_consts = {
      {&do_causal, MTL::DataType::DataTypeBool, 300},
      {&use_weights_lh, MTL::DataType::DataTypeBool, 301},
  };

  const std::string kname = "kq_dsa_indexer_score_" + kq_type_string(q.dtype()) +
      "_bm64_bn64_bk16_wm2_wn2";
  const std::string hash_name = kname + "_causal_" +
      (do_causal ? 't' : 'n') + "_wlh_" + (use_weights_lh ? 't' : 'n');

  auto kernel = kq_get_kernel(d, kname, hash_name, func_consts);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  ce.set_input_array(q, 0);
  ce.set_input_array(k, 1);
  ce.set_input_array(weights, 2);
  ce.set_output_array(out, 3);
  ce.set_bytes(params, 4);
  ce.set_bytes(H, 5);
  ce.set_bytes(unused_causal_prefix_topk_, 6);
  ce.set_bytes(skip_causal_future_store_, 7);
  ce.set_bytes(causal_q_offset_, 8);

  MTL::Size group_dims(wm * wn * 32, 1, 1);
  MTL::Size grid_dims(tiles_n, tiles_m, B);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQDsaTopKIndices::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  const auto& scores = inputs[0];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  constexpr int threads = 1024;

  const int B = scores.shape(0);
  const int L = scores.shape(2);
  const int K = scores.shape(3);
  const int rows = B * L;

  const std::string kname = "kq_dsa_topk_indices_" +
      kq_type_string(scores.dtype()) + "_topk" + std::to_string(topk_) +
      "_t" + std::to_string(threads);

  bool bucketed = bucketed_;
  mx::metal::MTLFCList func_consts = {
      {&bucketed, MTL::DataType::DataTypeBool, 302},
  };
  const std::string hash_name =
      kname + "_bucketed_" + (bucketed ? 't' : 'n');

  auto kernel = kq_get_kernel(d, kname, hash_name, func_consts);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  KQDsaTopKParams params{
      /* int rows = */ rows,
      /* int L = */ L,
      /* int K = */ K,
      /* int topk = */ topk_,
      /* bool causal_valid_prefix = */ causal_valid_prefix_};

  ce.set_input_array(scores, 0);
  ce.set_output_array(out, 1);
  ce.set_bytes(params, 2);

  MTL::Size group_dims(threads, 1, 1);
  MTL::Size grid_dims(rows, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQDsaIndexerScores::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_indexer_scores] requires a Metal build.");
}

void KQDsaTopKIndices::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_topk_indices] requires a Metal build.");
}

#endif

void KQDsaIndexerScores::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_indexer_scores] has no CPU implementation.");
}

void KQDsaTopKIndices::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_topk_indices] has no CPU implementation.");
}

std::vector<mx::Shape> KQDsaIndexerScores::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  return {mx::Shape{q.shape(0), 1, q.shape(2), k.shape(2)}};
}

std::vector<mx::Shape> KQDsaTopKIndices::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& scores = inputs[0];
  return {mx::Shape{scores.shape(0), 1, scores.shape(2), topk_}};
}

bool KQDsaIndexerScores::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQDsaIndexerScores&>(other);
  return causal_ == o.causal_ && weights_lh_ == o.weights_lh_ &&
      unused_causal_prefix_topk_ == o.unused_causal_prefix_topk_ &&
      skip_causal_future_store_ == o.skip_causal_future_store_ &&
      causal_q_offset_ == o.causal_q_offset_;
}

bool KQDsaTopKIndices::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQDsaTopKIndices&>(other);
  return topk_ == o.topk_ && bucketed_ == o.bucketed_ &&
      causal_valid_prefix_ == o.causal_valid_prefix_;
}

mx::array dsa_indexer_scores(
    mx::array queries,
    mx::array keys,
    mx::array weights,
    bool causal,
    int unused_causal_prefix_topk,
    bool skip_causal_future_store,
    int causal_q_offset,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (queries.ndim() != 4 || keys.ndim() != 4 ||
      (weights.ndim() != 3 && weights.ndim() != 4)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_indexer_scores] expected q/k rank 4 and weights "
        << "rank 3 or 4, got " << queries.shape() << ", " << keys.shape()
        << ", " << weights.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  const bool weights_lh = weights.ndim() == 3;
  bool weights_match = false;
  if (weights_lh) {
    weights_match = weights.shape(1) == queries.shape(2) &&
        weights.shape(2) == queries.shape(1);
  } else {
    weights_match = weights.shape(1) == queries.shape(1) &&
        weights.shape(2) == queries.shape(2) && weights.shape(3) == 1;
  }
  if (queries.shape(0) != keys.shape(0) ||
      queries.shape(0) != weights.shape(0) || !weights_match ||
      keys.shape(1) != 1 || queries.shape(3) != keys.shape(3)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_indexer_scores] incompatible q, k, weights "
        << "shapes: " << queries.shape() << ", " << keys.shape() << ", "
        << weights.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if ((queries.shape(1) != 32 && queries.shape(1) != 64) ||
      queries.shape(3) != 128) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_indexer_scores] expected 32 or 64 indexer heads "
        << "of dim 128, got " << queries.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (queries.shape(2) % 64 != 0 || keys.shape(2) % 64 != 0 ||
      keys.shape(2) < 64) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_indexer_scores] M and N must be positive "
        << "multiples of 64, got M " << queries.shape(2) << ", N "
        << keys.shape(2) << ".";
    throw std::invalid_argument(msg.str());
  }
  if (unused_causal_prefix_topk < 0) {
    throw std::invalid_argument(
        "[mlx_kquant.dsa_indexer_scores] unused_causal_prefix_topk must be "
        "non-negative.");
  }
  if (causal_q_offset < -1) {
    throw std::invalid_argument(
        "[mlx_kquant.dsa_indexer_scores] causal_q_offset must be -1 or "
        "non-negative.");
  }

  auto final_type =
      mx::result_type(std::vector<mx::array>{queries, keys, weights});
  if (final_type != mx::float16 && final_type != mx::bfloat16) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_indexer_scores] expected fp16 or bf16 inputs, "
        << "got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  // Unconditional contiguous: flags()/strides() on lazy graph arrays are
  // unreliable before eval (a broadcast input can read as row-contiguous),
  // and Contiguous no-op-donates at eval when the input already is.
  auto q = mx::contiguous(mx::astype(queries, final_type, s), false, s);
  auto k = mx::contiguous(mx::astype(keys, final_type, s), false, s);
  auto w = mx::contiguous(mx::astype(weights, final_type, s), false, s);

  mx::Shape out_shape{q.shape(0), 1, q.shape(2), k.shape(2)};
  std::vector<mx::array> inputs = {
      std::move(q), std::move(k), std::move(w)};
  return mx::array(
      std::move(out_shape),
      final_type,
      std::make_shared<KQDsaIndexerScores>(
          s,
          causal,
          weights_lh,
          unused_causal_prefix_topk,
          skip_causal_future_store,
          causal_q_offset),
      std::move(inputs));
}

mx::array dsa_topk_indices(
    mx::array scores,
    int topk,
    bool bucketed,
    bool causal_valid_prefix,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (scores.ndim() != 4 || scores.shape(1) != 1) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_topk_indices] expected scores with shape "
        << "[B, 1, L, K], got " << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk != 512 && topk != 2048) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_topk_indices] topk must be 512 or 2048, got "
        << topk << ".";
    throw std::invalid_argument(msg.str());
  }
  if (scores.shape(3) < topk) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_topk_indices] scores row length "
        << scores.shape(3) << " is smaller than topk " << topk << ".";
    throw std::invalid_argument(msg.str());
  }
  if (scores.dtype() != mx::float16 && scores.dtype() != mx::bfloat16) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_topk_indices] expected fp16 or bf16 scores, got "
        << scores.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  // See dsa_indexer_scores: pre-eval flags are unreliable, contiguous is
  // a no-op at eval when the input already is.
  auto sc = mx::contiguous(scores, false, s);

  mx::Shape out_shape{sc.shape(0), 1, sc.shape(2), topk};
  std::vector<mx::array> inputs = {std::move(sc)};
  return mx::array(
      std::move(out_shape),
      mx::uint32,
      std::make_shared<KQDsaTopKIndices>(
          s, topk, bucketed, causal_valid_prefix),
      std::move(inputs));
}

} // namespace mlx_kquant
