// Attention with int8 Q K^T and bfloat16 P V at head dim 128 (see
// kq_sdpa_qk8_nax.h for the kernels, adapted from MLX's attention_nax; MIT,
// see mlx_kquant/licenses/mlx-LICENSE). GPU only: the ops refuse a CPU
// stream when the call is built, and sdpa_qk8 any GPU without NAX units.
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string, kq_unit_stride_gpu

#include "mlx/allocator.h"
#include "mlx/ops.h"
#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel, kq_is_nax_available
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

const std::string kInputsOp = "[mlx_kquant.sdpa_qk8_inputs] ";
const std::string kQK8Op = "[mlx_kquant.sdpa_qk8] ";
constexpr int kQK8HeadDim = 128;
constexpr int kQK8BlockQ = 64;
constexpr int kQK8BlockK = 32;

void check_gpu_stream(const mx::Stream& s, const std::string& op) {
  if (s.device != mx::Device::gpu) {
    throw std::invalid_argument(
        op + "runs on the GPU only; the stream is on the CPU.");
  }
}

// q and k [B, heads, L, 128] with q's heads a multiple of k's, one batch.
void check_qk(
    const mx::array& q,
    const mx::array& k,
    const std::string& op,
    const char* qn,
    const char* kn) {
  if (q.ndim() != 4 || k.ndim() != 4) {
    throw std::invalid_argument(
        op + qn + " and " + kn + " must be 4-D [B, heads, L, 128].");
  }
  if (q.shape(3) != kQK8HeadDim || k.shape(3) != kQK8HeadDim) {
    throw std::invalid_argument(op + "only head dim 128 is supported.");
  }
  if (k.shape(0) != q.shape(0)) {
    throw std::invalid_argument(
        op + qn + " and " + kn + " must share a batch size.");
  }
  if (k.shape(1) < 1 || q.shape(1) % k.shape(1) != 0) {
    throw std::invalid_argument(
        op + "the heads of " + qn + " must be a multiple of " + kn + "'s.");
  }
}

} // namespace

#ifdef _METAL_

void KQuantSDPAQK8Inputs::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  for (auto& o : outputs) {
    o.set_data(mx::allocator::malloc(o.nbytes()));
  }
  // Rows of 128 are read through their strides with a unit last stride.
  const mx::array q = kq_unit_stride_gpu(inputs[0], s);
  const mx::array k = kq_unit_stride_gpu(inputs[1], s);
  const mx::array& mean = inputs[2];
  auto kernel = kq_get_kernel(
      d, std::string("kq_sdpa_qk8_rows_") + kq_type_string(q.dtype()));
  auto& ce = mx::metal::get_command_encoder(s);
  auto rows = [&](const mx::array& x,
                  float mul,
                  int has_mean,
                  mx::array& codes,
                  mx::array& scales) {
    const int L = x.shape(2);
    if (x.size() == 0) {
      return;
    }
    std::vector<int64_t> strides = {x.strides(0), x.strides(1), x.strides(2)};
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(x, 0);
    ce.set_input_array(mean, 1);
    ce.set_output_array(codes, 2);
    ce.set_output_array(scales, 3);
    ce.set_vector_bytes(strides, 4);
    ce.set_bytes(L, 5);
    ce.set_bytes(mul, 6);
    ce.set_bytes(has_mean, 7);
    const size_t width = static_cast<size_t>(L) * 32;
    ce.dispatch_threads(
        MTL::Size(width, x.shape(1), x.shape(0)),
        MTL::Size(width < 256 ? width : 256, 1, 1));
  };
  rows(q, scale_, 0, outputs[0], outputs[1]);
  rows(k, 1.0f, 1, outputs[2], outputs[3]);
}

void KQuantSDPAQK8::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  if (!kq_is_nax_available()) {
    throw std::runtime_error(
        kQK8Op +
        "requires a GPU with tensor-op (NAX) units, M5 or later; check "
        "nax_available().");
  }
  // Q, K and V are read through their batch, head and row strides with a
  // unit head-dim stride, from boundaries of four elements.
  const mx::array q =
      kq_strides_aligned_gpu(inputs[0], s, "sdpa_qk8 q8", 4 * 1);
  const mx::array k =
      kq_strides_aligned_gpu(inputs[1], s, "sdpa_qk8 k8", 4 * 1);
  const mx::array v = kq_strides_aligned_gpu(
      inputs[2], s, "sdpa_qk8 v", 4 * static_cast<int>(inputs[2].itemsize()));
  const mx::array& sq = inputs[3];
  const mx::array& sk = inputs[4];
  auto& o = outputs[0];

  // [B, H, qL, D] stored as [B, qL, H, D], as MLX's attention stores it, so
  // the caller's transpose to [B, qL, H * D] is a view.
  const int64_t str_h = o.shape(3);
  const int64_t str_l = static_cast<int64_t>(o.shape(1)) * str_h;
  const int64_t str_b = static_cast<int64_t>(o.shape(2)) * str_l;
  mx::array::Flags flags{
      /* bool contiguous = */ 1,
      /* bool row_contiguous = */ 0,
      /* bool col_contiguous = */ 0,
  };
  o.set_data(
      mx::allocator::malloc(o.nbytes()),
      o.size(),
      {str_b, str_h, str_l, 1},
      flags);
  if (o.size() == 0) {
    return;
  }

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int qL = q.shape(2);
  const int kL = k.shape(2);
  const int NQ = (qL + kQK8BlockQ - 1) / kQK8BlockQ;
  const int NK = (kL + kQK8BlockK - 1) / kQK8BlockK;
  const int NQ_aligned = qL / kQK8BlockQ;
  const int NK_aligned = kL / kQK8BlockK;
  mlx::steel::AttnParams params{
      /* int B = */ B,
      /* int H = */ H,
      /* int D = */ kQK8HeadDim,
      /* int qL = */ qL,
      /* int kL = */ kL,
      /* int gqa_factor = */ H / k.shape(1),
      /* float scale = */ 1.0f,
      /* int NQ = */ NQ,
      /* int NK = */ NK,
      /* int NQ_aligned = */ NQ_aligned,
      /* int NK_aligned = */ NK_aligned,
      /* int qL_rem = */ qL - NQ_aligned * kQK8BlockQ,
      /* int kL_rem = */ kL - NK_aligned * kQK8BlockK,
      /* int qL_off = */ kL - qL,
      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {str_b, str_h, str_l}};

  bool align_q = (qL % kQK8BlockQ) == 0;
  bool align_k = (kL % kQK8BlockK) == 0;
  mx::metal::MTLFCList func_consts = {
      {&align_q, MTL::DataType::DataTypeBool, 200},
      {&align_k, MTL::DataType::DataTypeBool, 201}};
  const std::string kname =
      std::string("kq_sdpa_qk8_nax_") + kq_type_string(v.dtype());
  const std::string hash_name = kname + "_align_q_" + (align_q ? 't' : 'n') +
      "_align_k_" + (align_k ? 't' : 'n');
  auto kernel = kq_get_kernel(d, kname, hash_name, func_consts);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(q, 0);
  ce.set_input_array(k, 1);
  ce.set_input_array(v, 2);
  ce.set_output_array(o, 3);
  ce.set_bytes(params, 4);
  ce.set_input_array(sq, 5);
  ce.set_input_array(sk, 6);
  ce.dispatch_threadgroups(MTL::Size(NQ, H, B), MTL::Size(32, 4, 1));
}

#else // !_METAL_

void KQuantSDPAQK8Inputs::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(kInputsOp + "requires a Metal build.");
}

void KQuantSDPAQK8::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(kQK8Op + "requires a Metal build.");
}

#endif

void KQuantSDPAQK8Inputs::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(kInputsOp + "has no CPU implementation.");
}

void KQuantSDPAQK8::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(kQK8Op + "has no CPU implementation.");
}

std::vector<mx::Shape> KQuantSDPAQK8Inputs::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  return {
      q.shape(),
      {q.shape(0), q.shape(1), q.shape(2)},
      k.shape(),
      {k.shape(0), k.shape(1), k.shape(2)}};
}

std::vector<mx::Shape> KQuantSDPAQK8::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

std::tuple<mx::array, mx::array, mx::array, mx::array>
sdpa_qk8_inputs(mx::array q, mx::array k, float scale, mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  check_gpu_stream(s, kInputsOp);
  check_qk(q, k, kInputsOp, "q", "k");
  if (q.dtype() != mx::bfloat16 || k.dtype() != mx::bfloat16) {
    throw std::invalid_argument(kInputsOp + "q and k must be bfloat16.");
  }
  if (!std::isfinite(scale)) {
    throw std::invalid_argument(kInputsOp + "scale must be finite.");
  }
  // K's mean over keys in k's dtype, widened: a shift of K leaves the
  // softmax unchanged, so its rounding does not reach the attention.
  auto mean = mx::contiguous(
      mx::astype(mx::mean(k, 2, false, s), mx::float32, s), false, s);
  auto shapes = std::vector<mx::Shape>{
      q.shape(),
      {q.shape(0), q.shape(1), q.shape(2)},
      k.shape(),
      {k.shape(0), k.shape(1), k.shape(2)}};
  auto outs = mx::array::make_arrays(
      std::move(shapes),
      {mx::int8, mx::float32, mx::int8, mx::float32},
      std::make_shared<KQuantSDPAQK8Inputs>(s, scale),
      {std::move(q), std::move(k), std::move(mean)});
  return {outs[0], outs[1], outs[2], outs[3]};
}

mx::array sdpa_qk8(
    mx::array q8,
    mx::array q_scale,
    mx::array k8,
    mx::array k_scale,
    mx::array v,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  check_gpu_stream(s, kQK8Op);
  check_qk(q8, k8, kQK8Op, "q8", "k8");
  if (q8.dtype() != mx::int8 || k8.dtype() != mx::int8) {
    throw std::invalid_argument(kQK8Op + "q8 and k8 must be int8.");
  }
  if (k8.shape(2) < 1) {
    throw std::invalid_argument(kQK8Op + "k8 needs at least one key row.");
  }
  const mx::Shape sq_shape = {q8.shape(0), q8.shape(1), q8.shape(2)};
  if (q_scale.shape() != sq_shape) {
    throw std::invalid_argument(kQK8Op + "q_scale must be [B, H, qL].");
  }
  const mx::Shape sk_shape = {k8.shape(0), k8.shape(1), k8.shape(2)};
  if (k_scale.shape() != sk_shape) {
    throw std::invalid_argument(kQK8Op + "k_scale must be [B, Hk, kL].");
  }
  if (v.ndim() != 4 || v.shape() != k8.shape()) {
    throw std::invalid_argument(kQK8Op + "v must have the shape of k8.");
  }
  if (v.dtype() != mx::bfloat16) {
    throw std::invalid_argument(kQK8Op + "v must be bfloat16.");
  }
  // Checked after the arguments, so the argument checks also run on GPUs
  // without NAX.
  if (!nax_available()) {
    throw std::runtime_error(
        kQK8Op +
        "requires a GPU with tensor-op (NAX) units, M5 or later; check "
        "nax_available().");
  }
  auto f32 = [&](mx::array a) {
    return mx::contiguous(mx::astype(a, mx::float32, s), false, s);
  };
  auto shape = q8.shape();
  return mx::array(
      std::move(shape),
      mx::bfloat16,
      std::make_shared<KQuantSDPAQK8>(s),
      {std::move(q8),
       std::move(k8),
       std::move(v),
       f32(std::move(q_scale)),
       f32(std::move(k_scale))});
}

} // namespace mlx_kquant
