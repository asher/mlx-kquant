// KQDsaSparseAttention primitive: DeepSeek-V4-Flash sparse attention -- one
// dispatch covering the sliding local window plus the indexer-selected pooled
// rows, with per-head fp32 attention sinks seeding the flash online softmax.
// Ported from omlx glm_moe_dsa deepseek_v4_sparse_attention; the kq dispatch
// additionally accepts qL == 1 (decode) and qL == 2 (MTP verify) -- omlx
// gated those out only because it lacked a decode-time indexer. K == V (the
// V4 shared latent). Inference-only (no CPU eval).
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/ops.h" // astype, result_type
#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#include "../metal/mlx/backend/metal/kernels/kq_dsa_params.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

#ifdef _METAL_

void KQDsaSparseAttention::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  const auto& q = inputs[0];
  const auto& local_kv = inputs[1];
  const auto& pooled = inputs[2];
  const auto& topk = inputs[3];
  const auto& sinks = inputs[4];
  auto& o = outputs[0];

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int qL = q.shape(2);
  const int localL = local_kv.shape(2);
  const int pooledL = pooled.shape(1);
  const int topkN = topk.shape(3);

  // Row-contiguous output with explicit strides (the kernel stores through
  // O_strides; mirrors the omlx primitive).
  int64_t str_oD = 1;
  int64_t str_oL = o.shape(3);
  int64_t str_oH = o.shape(2) * str_oL;
  int64_t str_oB = o.shape(1) * str_oH;
  size_t data_size = o.shape(0) * str_oB;
  mx::array::Flags flags{
      /* bool contiguous = */ 1,
      /* bool row_contiguous = */ 1,
      /* bool col_contiguous = */ 0,
  };
  o.set_data(
      mx::allocator::malloc(o.nbytes()),
      data_size,
      {str_oB, str_oH, str_oL, str_oD},
      flags);

  auto bcast_stride = [](const mx::array& a, int axis) -> int64_t {
    return a.shape(axis) == 1 ? 0 : a.strides(axis);
  };

  KQDsaSparseAttentionParams params{
      /* int B = */ B,
      /* int H = */ H,
      /* int qL = */ qL,
      /* int localL = */ localL,
      /* int pooledL = */ pooledL,
      /* int topk = */ topkN,
      /* int local_window = */ local_window_,
      /* int compress_ratio = */ compress_ratio_,
      /* int q_offset = */ q_offset_,

      /* float scale = */ scale_,

      /* int64_t Q_strides[3] = */
      {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t Local_strides[3] = */
      {local_kv.strides(0), bcast_stride(local_kv, 1), local_kv.strides(2)},
      /* int64_t Pooled_strides[2] = */
      {pooled.strides(0), pooled.strides(1)},
      /* int64_t Topk_strides[3] = */
      {topk.strides(0), bcast_stride(topk, 1), topk.strides(2)},
      /* int64_t O_strides[3] = */
      {o.strides(0), o.strides(1), o.strides(2)}};

  const std::string kname = "kq_dsa_sparse_attention_" +
      kq_type_string(q.dtype()) + "_bk256_dc32_h64_d512_wm8";
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(q, 0);
  ce.set_input_array(local_kv, 1);
  ce.set_input_array(pooled, 2);
  ce.set_input_array(topk, 3);
  ce.set_input_array(sinks, 4);
  ce.set_output_array(o, 5);
  ce.set_bytes(params, 6);

  MTL::Size grid_dims(qL, B, 1);
  MTL::Size group_dims(32, 8, 1); // WM = 8 simdgroups
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQDsaSparseAttention::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_sparse_attention] requires a Metal build.");
}

#endif

void KQDsaSparseAttention::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_sparse_attention] has no CPU implementation.");
}

std::vector<mx::Shape> KQDsaSparseAttention::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQDsaSparseAttention::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQDsaSparseAttention&>(other);
  return scale_ == o.scale_ && q_offset_ == o.q_offset_ &&
      compress_ratio_ == o.compress_ratio_ &&
      local_window_ == o.local_window_;
}

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
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (q.ndim() != 4 || local_kv.ndim() != 4 || pooled.ndim() != 3 ||
      topk_indices.ndim() != 4 || sinks.ndim() != 1) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_sparse_attention] incompatible ranks: "
        << q.shape() << ", " << local_kv.shape() << ", " << pooled.shape()
        << ", " << topk_indices.shape() << ", " << sinks.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (q.shape(0) != local_kv.shape(0) || q.shape(0) != pooled.shape(0) ||
      q.shape(0) != topk_indices.shape(0) || q.shape(1) != 64 ||
      q.shape(3) != 512 || local_kv.shape(1) != 1 ||
      local_kv.shape(3) != 512 || pooled.shape(2) != 512 ||
      topk_indices.shape(1) != 1 || topk_indices.shape(2) != q.shape(2) ||
      sinks.shape(0) != q.shape(1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_sparse_attention] incompatible shapes: "
        << q.shape() << ", " << local_kv.shape() << ", " << pooled.shape()
        << ", " << topk_indices.shape() << ", " << sinks.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk_indices.dtype() != mx::uint32) {
    throw std::invalid_argument(
        "[mlx_kquant.dsa_sparse_attention] topk_indices must be uint32.");
  }
  if (q.shape(2) < 1 || local_kv.shape(2) < q.shape(2) ||
      pooled.shape(1) <= 0 || topk_indices.shape(3) <= 0) {
    throw std::invalid_argument(
        "[mlx_kquant.dsa_sparse_attention] needs qL >= 1, localL >= qL and "
        "non-empty pooled/topk.");
  }
  if (q_offset < 0 || compress_ratio <= 0 || local_window <= 0) {
    throw std::invalid_argument(
        "[mlx_kquant.dsa_sparse_attention] q_offset/compress_ratio/"
        "local_window out of range.");
  }

  auto final_type =
      mx::result_type(std::vector<mx::array>{q, local_kv, pooled});
  if (final_type != mx::float16 && final_type != mx::bfloat16) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_sparse_attention] expected fp16 or bf16 inputs, "
        << "got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  // The kernel reads through explicit strides but requires unit stride on
  // the last (feature) axis of every tensor. strides()/flags() on lazy
  // graph arrays are unreliable before eval, so force contiguity instead
  // of checking -- Contiguous no-op-donates at eval when the input already
  // is (eval_gpu still reads the real strides for the param block).
  auto q_cast = mx::contiguous(mx::astype(q, final_type, s), false, s);
  auto local_cast =
      mx::contiguous(mx::astype(local_kv, final_type, s), false, s);
  auto pooled_cast =
      mx::contiguous(mx::astype(pooled, final_type, s), false, s);
  auto sinks_cast =
      mx::contiguous(mx::astype(sinks, final_type, s), false, s);
  topk_indices = mx::contiguous(topk_indices, false, s);

  mx::Shape out_shape{
      q_cast.shape(0), q_cast.shape(1), q_cast.shape(2), q_cast.shape(3)};
  std::vector<mx::array> inputs = {
      std::move(q_cast),
      std::move(local_cast),
      std::move(pooled_cast),
      std::move(topk_indices),
      std::move(sinks_cast)};
  return mx::array(
      std::move(out_shape),
      final_type,
      std::make_shared<KQDsaSparseAttention>(
          s, scale, q_offset, compress_ratio, local_window),
      std::move(inputs));
}

} // namespace mlx_kquant
