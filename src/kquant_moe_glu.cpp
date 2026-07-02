// Fused MoE gather ops. Two families sharing the decode shape (one activation
// row per gathered expert row), all inference-only (no CPU eval):
//   mxfp4 packed layout: moe_glu_gather (gate + up + biases + clamped-SwiGLU
//     in one dispatch) and gather_qmv_bias.
//   K-quant wire bytes: moe_glu_gather_kq (gate + up + silu/gelu GLU, no
//     biases) and gather_qmv_kq, per-codec kernels (q6_k, q8_0 wired).
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_codec.h" // codec_by_name
#include "kquant_internal.h" // kq_type_string

#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

#ifdef _METAL_

void KQuantMoEGLU::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& gw = inputs[0];
  const auto& gs = inputs[1];
  const auto& gb = inputs[2];
  const auto& uw = inputs[3];
  const auto& us = inputs[4];
  const auto& ub = inputs[5];
  const auto& x = inputs[6];
  const auto& indices = inputs[7];

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = gw.shape(1);
  int K = x.shape(-1);
  float alpha = alpha_;
  float limit = limit_;

  std::string kname = "kq_moe_glu_gather_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(gw, 0);
  ce.set_input_array(gs, 1);
  ce.set_input_array(gb, 2);
  ce.set_input_array(uw, 3);
  ce.set_input_array(us, 4);
  ce.set_input_array(ub, 5);
  ce.set_input_array(x, 6);
  ce.set_input_array(indices, 7);
  ce.set_output_array(out, 8);
  ce.set_bytes(K, 9);
  ce.set_bytes(N, 10);
  ce.set_bytes(alpha, 11);
  ce.set_bytes(limit, 12);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / 8, R, T);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantGatherQMVBias::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& w = inputs[0];
  const auto& sc = inputs[1];
  const auto& b = inputs[2];
  const auto& x = inputs[3];
  const auto& indices = inputs[4];

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = w.shape(1);
  int K = x.shape(-1);

  std::string kname = "kq_gather_qmv_bias_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(w, 0);
  ce.set_input_array(sc, 1);
  ce.set_input_array(b, 2);
  ce.set_input_array(x, 3);
  ce.set_input_array(indices, 4);
  ce.set_output_array(out, 5);
  ce.set_bytes(K, 6);
  ce.set_bytes(N, 7);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / 8, R, T);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantMoEGLUKQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& gw = inputs[0];
  const auto& uw = inputs[1];
  const auto& x = inputs[2];
  const auto& indices = inputs[3];

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = gw.shape(1);
  int K = x.shape(-1);

  std::string kname = "kq_" + kquant_type_ + "_moe_glu_gather_" + act_ + "_" +
      kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(gw, 0);
  ce.set_input_array(uw, 1);
  ce.set_input_array(x, 2);
  ce.set_input_array(indices, 3);
  ce.set_output_array(out, 4);
  ce.set_bytes(K, 5);
  ce.set_bytes(N, 6);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / 8, R, T);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantGatherQMVKQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& w = inputs[0];
  const auto& x = inputs[1];
  const auto& indices = inputs[2];

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = w.shape(1);
  int K = x.shape(-1);

  std::string kname =
      "kq_" + kquant_type_ + "_gather_qmv_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(w, 0);
  ce.set_input_array(x, 1);
  ce.set_input_array(indices, 2);
  ce.set_output_array(out, 3);
  ce.set_bytes(K, 4);
  ce.set_bytes(N, 5);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / 8, R, T);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantMoEGLUShexpKQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& gw = inputs[0];
  const auto& uw = inputs[1];
  const auto& sgw = inputs[2];
  const auto& suw = inputs[3];
  const auto& x = inputs[4];
  const auto& indices = inputs[5];

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = gw.shape(1);
  int K = x.shape(-1);

  std::string kname = "kq_" + kquant_type_ + "_moe_glu_gather_shexp_" + act_ +
      "_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(gw, 0);
  ce.set_input_array(uw, 1);
  ce.set_input_array(sgw, 2);
  ce.set_input_array(suw, 3);
  ce.set_input_array(x, 4);
  ce.set_input_array(indices, 5);
  ce.set_output_array(out, 6);
  ce.set_bytes(K, 7);
  ce.set_bytes(N, 8);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / 8, R + 1, T);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantGatherQMVMixKQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& w = inputs[0];
  const auto& sw = inputs[1];
  const auto& x = inputs[2];
  const auto& indices = inputs[3];
  const auto& scores = inputs[4];

  int T = x.shape(0);
  int S = x.shape(1);
  int N = w.shape(1);
  int K = x.shape(-1);
  (void)scores;

  std::string kname =
      "kq_" + kquant_type_ + "_gather_qmv_mix_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(w, 0);
  ce.set_input_array(sw, 1);
  ce.set_input_array(x, 2);
  ce.set_input_array(indices, 3);
  ce.set_input_array(scores, 4);
  ce.set_output_array(out, 5);
  ce.set_bytes(K, 6);
  ce.set_bytes(N, 7);
  ce.set_bytes(S, 8);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / 8, 1, T);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantMoEGLU::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.moe_glu_gather] requires Metal.");
}

void KQuantGatherQMVBias::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.gather_qmv_bias] requires Metal.");
}

void KQuantMoEGLUKQ::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.moe_glu_gather_kq] requires Metal.");
}

void KQuantGatherQMVKQ::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.gather_qmv_kq] requires Metal.");
}

void KQuantMoEGLUShexpKQ::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.moe_glu_gather_shexp_kq] requires Metal.");
}

void KQuantGatherQMVMixKQ::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.gather_qmv_mix_kq] requires Metal.");
}

#endif

void KQuantMoEGLU::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.moe_glu_gather] has no CPU implementation.");
}

void KQuantGatherQMVBias::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_bias] has no CPU implementation.");
}

bool KQuantMoEGLU::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMoEGLU&>(other);
  return alpha_ == o.alpha_ && limit_ == o.limit_;
}

bool KQuantGatherQMVBias::is_equivalent(const mx::Primitive&) const {
  return true;
}

void KQuantMoEGLUKQ::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.moe_glu_gather_kq] has no CPU implementation.");
}

void KQuantGatherQMVKQ::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_kq] has no CPU implementation.");
}

bool KQuantMoEGLUKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMoEGLUKQ&>(other);
  return kquant_type_ == o.kquant_type_ && act_ == o.act_;
}

bool KQuantGatherQMVKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMVKQ&>(other);
  return kquant_type_ == o.kquant_type_;
}

std::vector<mx::Shape> KQuantMoEGLUKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[3].shape(0), inputs[3].shape(1), inputs[0].shape(1)}};
}

void KQuantMoEGLUShexpKQ::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.moe_glu_gather_shexp_kq] has no CPU implementation.");
}

void KQuantGatherQMVMixKQ::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_mix_kq] has no CPU implementation.");
}

bool KQuantMoEGLUShexpKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMoEGLUShexpKQ&>(other);
  return kquant_type_ == o.kquant_type_ && act_ == o.act_;
}

bool KQuantGatherQMVMixKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMVMixKQ&>(other);
  return kquant_type_ == o.kquant_type_;
}

std::vector<mx::Shape> KQuantMoEGLUShexpKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[5].shape(0), inputs[5].shape(1) + 1, inputs[0].shape(1)}};
}

std::vector<mx::Shape> KQuantGatherQMVMixKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[2].shape(0), inputs[0].shape(1)}};
}

std::vector<mx::Shape> KQuantGatherQMVKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[2].shape(0), inputs[2].shape(1), inputs[0].shape(1)}};
}

std::vector<mx::Shape> KQuantMoEGLU::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[7].shape(0), inputs[7].shape(1), inputs[0].shape(1)}};
}

std::vector<mx::Shape> KQuantGatherQMVBias::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[4].shape(0), inputs[4].shape(1), inputs[0].shape(1)}};
}

namespace {

// Shared validation for one packed-mxfp4 expert stack (w, scales, bias).
void check_expert_stack(
    const char* op,
    const mx::array& w,
    const mx::array& sc,
    const mx::array& b,
    int K) {
  if (w.ndim() != 3 || w.dtype() != mx::uint32) {
    throw std::invalid_argument(
        std::string(op) + " expert weights must be uint32 [E, N, K/8].");
  }
  if (sc.ndim() != 3 || sc.dtype() != mx::uint8) {
    throw std::invalid_argument(
        std::string(op) + " expert scales must be uint8 [E, N, K/32].");
  }
  if (w.shape(2) * 8 != K || sc.shape(2) * 32 != K) {
    throw std::invalid_argument(
        std::string(op) + " weight/scale trailing dims do not match K.");
  }
  if (w.shape(0) != sc.shape(0) || w.shape(1) != sc.shape(1)) {
    throw std::invalid_argument(
        std::string(op) + " weight/scale expert dims do not match.");
  }
  if (b.ndim() != 2 || b.shape(0) != w.shape(0) || b.shape(1) != w.shape(1)) {
    throw std::invalid_argument(std::string(op) + " bias must be [E, N].");
  }
  if (w.shape(1) % 8 != 0) {
    throw std::invalid_argument(
        std::string(op) + " N must be a multiple of 8.");
  }
  if (K % 32 != 0) {
    throw std::invalid_argument(
        std::string(op) + " K must be a multiple of 32.");
  }
  if (!w.flags().row_contiguous || !sc.flags().row_contiguous) {
    throw std::invalid_argument(
        std::string(op) + " weights/scales must be row-contiguous.");
  }
}

mx::array prep_bias(const mx::array& b, mx::StreamOrDevice s) {
  return mx::contiguous(mx::astype(b, mx::float32, s), false, s);
}

mx::array prep_indices(const mx::array& idx, mx::StreamOrDevice s) {
  return mx::contiguous(mx::astype(idx, mx::uint32, s), false, s);
}

// Codecs with the fused kq GLU/gather kernels wired (kq_moe_glu_kq.h). Grows
// with the codec matrix; unsupported codecs must fall back to the stock path
// per-tensor (callers gate on this via ops throwing invalid_argument).
bool codec_has_moe_glu(const std::string& kquant_type) {
  return kquant_type == "q6_k" || kquant_type == "q8_0";
}

// Shared validation for one K-quant wire-byte expert stack.
void check_kq_expert_stack(
    const char* op,
    const mx::array& w,
    const std::string& kquant_type,
    int K) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr || !codec_has_moe_glu(kquant_type)) {
    throw std::invalid_argument(
        std::string(op) + " codec '" + kquant_type +
        "' has no fused MoE kernel.");
  }
  if (w.ndim() != 3 || w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        std::string(op) +
        " expert weights must be uint8 [E, N, bytes_per_row].");
  }
  const int64_t bpr =
      (int64_t)K / codec->weights_per_block * codec->bytes_per_block;
  if (K % codec->weights_per_block != 0 || w.shape(2) != bpr) {
    throw std::invalid_argument(
        std::string(op) + " weight trailing dim does not match K for '" +
        kquant_type + "'.");
  }
  if (K % 256 != 0) {
    throw std::invalid_argument(
        std::string(op) + " requires K % 256 == 0 (fast-path mapping).");
  }
  if (w.shape(1) % 8 != 0) {
    throw std::invalid_argument(
        std::string(op) + " N must be a multiple of 8.");
  }
  if (!w.flags().row_contiguous) {
    throw std::invalid_argument(
        std::string(op) + " weights must be row-contiguous.");
  }
}

// Shared validation for one single-expert (2-D) K-quant wire-byte tensor;
// must match the expert stack's row shape exactly.
void check_kq_shexp_row(
    const char* op,
    const mx::array& w,
    const std::string& kquant_type,
    int K,
    int N) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (w.ndim() != 2 || w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        std::string(op) +
        " shared-expert weights must be uint8 [N, bytes_per_row].");
  }
  const int64_t bpr =
      (int64_t)K / codec->weights_per_block * codec->bytes_per_block;
  if (w.shape(0) != N || w.shape(1) != bpr) {
    throw std::invalid_argument(
        std::string(op) +
        " shared-expert weights must match the expert stack row shape.");
  }
  if (!w.flags().row_contiguous) {
    throw std::invalid_argument(
        std::string(op) + " shared-expert weights must be row-contiguous.");
  }
}

mx::array prep_scores(const mx::array& sc, mx::StreamOrDevice s) {
  return mx::contiguous(mx::astype(sc, mx::float32, s), false, s);
}

} // namespace

mx::array moe_glu_gather(
    mx::array x,
    mx::array gate_w,
    mx::array gate_scales,
    mx::array gate_bias,
    mx::array up_w,
    mx::array up_scales,
    mx::array up_bias,
    mx::array indices,
    float alpha,
    float limit,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  if (x.ndim() != 2) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather] x must be 2-D [T, K].");
  }
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0)) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather] indices must be [T, R].");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather] x must be float16 or bfloat16.");
  }
  int K = x.shape(1);
  check_expert_stack(
      "[mlx_kquant.moe_glu_gather]", gate_w, gate_scales, gate_bias, K);
  check_expert_stack(
      "[mlx_kquant.moe_glu_gather]", up_w, up_scales, up_bias, K);
  if (gate_w.shape(0) != up_w.shape(0) || gate_w.shape(1) != up_w.shape(1)) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather] gate/up expert shapes must match.");
  }

  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  mx::Shape out_shape = {x.shape(0), indices.shape(1), gate_w.shape(1)};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantMoEGLU>(s, alpha, limit),
      {std::move(gate_w),
       std::move(gate_scales),
       prep_bias(gate_bias, s),
       std::move(up_w),
       std::move(up_scales),
       prep_bias(up_bias, s),
       std::move(x_c),
       prep_indices(indices, s)});
}

mx::array gather_qmv_bias(
    mx::array x,
    mx::array w,
    mx::array scales,
    mx::array bias,
    mx::array indices,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  if (x.ndim() != 3) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_bias] x must be 3-D [T, R, K].");
  }
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0) ||
      indices.shape(1) != x.shape(1)) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_bias] indices must be [T, R] matching x.");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_bias] x must be float16 or bfloat16.");
  }
  int K = x.shape(2);
  check_expert_stack("[mlx_kquant.gather_qmv_bias]", w, scales, bias, K);

  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  mx::Shape out_shape = {x.shape(0), x.shape(1), w.shape(1)};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVBias>(s),
      {std::move(w),
       std::move(scales),
       prep_bias(bias, s),
       std::move(x_c),
       prep_indices(indices, s)});
}

mx::array moe_glu_gather_kq(
    mx::array x,
    mx::array gate_w,
    mx::array up_w,
    const std::string& kquant_type,
    mx::array indices,
    const std::string& act,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  if (x.ndim() != 2) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] x must be 2-D [T, K].");
  }
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0)) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] indices must be [T, R].");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] x must be float16 or bfloat16.");
  }
  if (act != "silu" && act != "gelu") {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] act must be 'silu' or 'gelu'.");
  }
  int K = x.shape(1);
  check_kq_expert_stack(
      "[mlx_kquant.moe_glu_gather_kq]", gate_w, kquant_type, K);
  check_kq_expert_stack("[mlx_kquant.moe_glu_gather_kq]", up_w, kquant_type, K);
  if (gate_w.shape(0) != up_w.shape(0) || gate_w.shape(1) != up_w.shape(1)) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] gate/up expert shapes must match.");
  }

  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  mx::Shape out_shape = {x.shape(0), indices.shape(1), gate_w.shape(1)};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantMoEGLUKQ>(s, kquant_type, act),
      {std::move(gate_w),
       std::move(up_w),
       std::move(x_c),
       prep_indices(indices, s)});
}

mx::array gather_qmv_kq(
    mx::array x,
    mx::array w,
    const std::string& kquant_type,
    mx::array indices,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  if (x.ndim() != 3) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_kq] x must be 3-D [T, R, K].");
  }
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0) ||
      indices.shape(1) != x.shape(1)) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_kq] indices must be [T, R] matching x.");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_kq] x must be float16 or bfloat16.");
  }
  int K = x.shape(2);
  check_kq_expert_stack("[mlx_kquant.gather_qmv_kq]", w, kquant_type, K);

  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  mx::Shape out_shape = {x.shape(0), x.shape(1), w.shape(1)};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVKQ>(s, kquant_type),
      {std::move(w), std::move(x_c), prep_indices(indices, s)});
}

mx::array moe_glu_gather_shexp_kq(
    mx::array x,
    mx::array gate_w,
    mx::array up_w,
    mx::array shexp_gate_w,
    mx::array shexp_up_w,
    const std::string& kquant_type,
    mx::array indices,
    const std::string& act,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.moe_glu_gather_shexp_kq]";
  if (x.ndim() != 2) {
    throw std::invalid_argument(std::string(op) + " x must be 2-D [T, K].");
  }
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0)) {
    throw std::invalid_argument(std::string(op) + " indices must be [T, R].");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " x must be float16 or bfloat16.");
  }
  if (act != "silu" && act != "gelu") {
    throw std::invalid_argument(
        std::string(op) + " act must be 'silu' or 'gelu'.");
  }
  int K = x.shape(1);
  check_kq_expert_stack(op, gate_w, kquant_type, K);
  check_kq_expert_stack(op, up_w, kquant_type, K);
  if (gate_w.shape(0) != up_w.shape(0) || gate_w.shape(1) != up_w.shape(1)) {
    throw std::invalid_argument(
        std::string(op) + " gate/up expert shapes must match.");
  }
  int N = gate_w.shape(1);
  check_kq_shexp_row(op, shexp_gate_w, kquant_type, K, N);
  check_kq_shexp_row(op, shexp_up_w, kquant_type, K, N);

  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  mx::Shape out_shape = {x.shape(0), indices.shape(1) + 1, N};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantMoEGLUShexpKQ>(s, kquant_type, act),
      {std::move(gate_w),
       std::move(up_w),
       std::move(shexp_gate_w),
       std::move(shexp_up_w),
       std::move(x_c),
       prep_indices(indices, s)});
}

mx::array gather_qmv_mix_kq(
    mx::array x,
    mx::array w,
    mx::array shexp_w,
    const std::string& kquant_type,
    mx::array indices,
    mx::array scores,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.gather_qmv_mix_kq]";
  if (x.ndim() != 3) {
    throw std::invalid_argument(std::string(op) + " x must be 3-D [T, S, K].");
  }
  int S = x.shape(1);
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0) ||
      indices.shape(1) != S - 1) {
    throw std::invalid_argument(
        std::string(op) + " indices must be [T, S - 1].");
  }
  if (scores.ndim() != 2 || scores.shape(0) != x.shape(0) ||
      scores.shape(1) != S) {
    throw std::invalid_argument(std::string(op) + " scores must be [T, S].");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " x must be float16 or bfloat16.");
  }
  int K = x.shape(2);
  check_kq_expert_stack(op, w, kquant_type, K);
  check_kq_shexp_row(op, shexp_w, kquant_type, K, w.shape(1));

  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  mx::Shape out_shape = {x.shape(0), w.shape(1)};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVMixKQ>(s, kquant_type),
      {std::move(w),
       std::move(shexp_w),
       std::move(x_c),
       prep_indices(indices, s),
       prep_scores(scores, s)});
}

} // namespace mlx_kquant
