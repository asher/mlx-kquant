// Fused MoE gather ops. Two families sharing the decode shape (one activation
// row per gathered expert row), all inference-only (no CPU eval):
//   mxfp4 packed layout: moe_glu_gather (gate + up + biases + clamped-SwiGLU
//     in one dispatch) and gather_qmv_bias.
//   K-quant wire bytes: moe_glu_gather_kq (gate + up + GLU epilogue;
//     optional biases with act "swiglu_clamp", mxfp4/nvfp4 only) and
//     gather_qmv_kq, per-codec kernels (full codec matrix).
#include <cstdio>
#include <cstdlib>
#include <set>
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

namespace {

// Tuned q6_k/q8_0 kernels stride K in 256-wide steps; the generic Ext-trait
// kernels only need whole blocks. q6_k is structurally K % 256 == 0; q8_0 at
// K % 256 != 0 dispatches the generic "q8_0_ext" instantiations.
inline std::string kq_gather_stem(const std::string& t, int K) {
  if (t == "q8_0" && K % 256 != 0) {
    return "q8_0_ext";
  }
  return t;
}

// Decode-scale launches underfill the GPU at the default mapping (8 K-lanes
// x 8 output rows per threadgroup); widening the K-lane split (NX = 16/32
// lanes per output row) multiplies threadgroups and shortens each K-chain.
// Measured at t = 1 (gemma-a4b q4_k K=2816, qwen3-next q6_k K=2048): only
// the two-stream GLU gathers (gate+up / shexp) profit -- sharing each
// activation chunk across two weight streams keeps per-thread work high
// enough that the extra threadgroups come free (nx16 +9% / +30% incl. the
// tuned-q6_k swap; nx32 mixed, so the auto pick caps at 16). Single-stream
// gathers (qmv / mix / mix_ns) are flat-to-negative under widening at every
// measured shape and stay NX = 8. KQ_MOE_NX=8|16|32 forces a width for ALL
// ops (A/B and tests); rows = output rows across the whole dispatch.
inline int kq_moe_pick_nx(int64_t rows, int K, bool two_stream) {
  // Re-read per call only when the variable exists at all (interleaved A/B
  // flips it in-process); unset costs one static check.
  static const bool has_env = std::getenv("KQ_MOE_NX") != nullptr;
  if (has_env) {
    const char* e = std::getenv("KQ_MOE_NX");
    const int v = e == nullptr ? 0 : std::atoi(e);
    if (v == 8 || v == 16 || v == 32) {
      return v;
    }
  }
  if (two_stream && K / 16 >= 32 && (rows * 8) / 64 < 2048) {
    return 16;
  }
  return 8;
}

inline const char* kq_nx_suffix(int nx) {
  return nx == 32 ? "_nx32" : (nx == 16 ? "_nx16" : "");
}

// KQ_MOE_NX_LOG=1: print each fused-MoE kernel name once (dispatch audit).
inline void kq_moe_log_kname(const std::string& kname) {
  static const bool log = std::getenv("KQ_MOE_NX_LOG") != nullptr;
  if (!log) {
    return;
  }
  static std::set<std::string> seen;
  if (seen.insert(kname).second) {
    std::fprintf(stderr, "[kq_moe] %s\n", kname.c_str());
  }
}

// Wide-NX variants exist only on the generic Ext kernels; tuned q6_k/q8_0
// uniform dispatches reroute to their "_ext" stems when a wide NX is picked.
inline std::string kq_gather_stem_nx(const std::string& t, int K, int nx) {
  std::string stem = kq_gather_stem(t, K);
  if (nx > 8 && (stem == "q6_k" || stem == "q8_0")) {
    stem += "_ext";
  }
  return stem;
}

} // namespace

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
  kq_moe_log_kname(kname);
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
  kq_moe_log_kname(kname);
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

  // Biased (swiglu_clamp) layout: gw, uw, gb, ub, x, indices.
  const bool biased = inputs.size() == 6;
  const auto& gw = inputs[0];
  const auto& uw = inputs[1];
  const auto& x = inputs[biased ? 4 : 2];
  const auto& indices = inputs.back();

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = gw.shape(1);
  int K = x.shape(-1);

  const int nx = kq_moe_pick_nx((int64_t)N * R * T, K, true);
  std::string kname = "kq_" + kq_gather_stem_nx(kquant_type_, K, nx) +
      "_moe_glu_gather_" + (biased ? "bias_" : "") + act_ + kq_nx_suffix(nx) +
      "_" + kq_type_string(x.dtype());
  kq_moe_log_kname(kname);
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(gw, 0);
  ce.set_input_array(uw, 1);
  if (biased) {
    ce.set_input_array(inputs[2], 2);
    ce.set_input_array(inputs[3], 3);
    ce.set_input_array(x, 4);
    ce.set_input_array(indices, 5);
    ce.set_output_array(out, 6);
    ce.set_bytes(K, 7);
    ce.set_bytes(N, 8);
    ce.set_bytes(limit_, 9);
    ce.set_bytes(alpha_, 10);
  } else {
    ce.set_input_array(x, 2);
    ce.set_input_array(indices, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(K, 5);
    ce.set_bytes(N, 6);
    ce.set_bytes(limit_, 7);
  }
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / (64 / nx), R, T);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantGatherQMVKQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // Biased layout: w, b, x, indices.
  const bool biased = inputs.size() == 4;
  const auto& w = inputs[0];
  const auto& x = inputs[biased ? 2 : 1];
  const auto& indices = inputs.back();

  int T = indices.shape(0);
  int R = indices.shape(1);
  int N = w.shape(1);
  int K = x.shape(-1);

  const int nx = kq_moe_pick_nx((int64_t)N * R * T, K, false);
  std::string kname = "kq_" + kq_gather_stem_nx(kquant_type_, K, nx) +
      "_gather_qmv" + (biased ? "_bias" : "") + kq_nx_suffix(nx) + "_" +
      kq_type_string(x.dtype());
  kq_moe_log_kname(kname);
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(w, 0);
  if (biased) {
    ce.set_input_array(inputs[1], 1);
    ce.set_input_array(x, 2);
    ce.set_input_array(indices, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(K, 5);
    ce.set_bytes(N, 6);
  } else {
    ce.set_input_array(x, 1);
    ce.set_input_array(indices, 2);
    ce.set_output_array(out, 3);
    ce.set_bytes(K, 4);
    ce.set_bytes(N, 5);
  }
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / (64 / nx), R, T);
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

  // Mixed shexp codecs dispatch the generic "_sx_" instantiations.
  const int nx = kq_moe_pick_nx((int64_t)N * (R + 1) * T, K, true);
  const std::string stem = shexp_type_ == kquant_type_
      ? kq_gather_stem_nx(kquant_type_, K, nx)
      : kquant_type_ + "_sx_" + shexp_type_;
  std::string kname = "kq_" + stem + "_moe_glu_gather_shexp_" + act_ +
      kq_nx_suffix(nx) + "_" + kq_type_string(x.dtype());
  kq_moe_log_kname(kname);
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
  // Signature parity with the plain kernels: the shexp epilogue takes the
  // limit arg too (dead for silu/gelu; no shexp silu_limit instantiations).
  const float limit = 0.0f;
  ce.set_bytes(limit, 9);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / (64 / nx), R + 1, T);
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

  const int nx = kq_moe_pick_nx((int64_t)N * T, K, false);
  const std::string stem = shexp_type_ == kquant_type_
      ? kq_gather_stem_nx(kquant_type_, K, nx)
      : kquant_type_ + "_sx_" + shexp_type_;
  std::string kname = "kq_" + stem + "_gather_qmv_mix" + kq_nx_suffix(nx) +
      "_" + kq_type_string(x.dtype());
  kq_moe_log_kname(kname);
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
  MTL::Size grid_dims(N / (64 / nx), 1, T);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantGatherQMVMixNSKQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& w = inputs[0];
  const auto& x = inputs[1];
  const auto& indices = inputs[2];
  const auto& scores = inputs[3];

  int T = x.shape(0);
  int S = x.shape(1);
  int N = w.shape(1);
  int K = x.shape(-1);

  // mix_ns is generic for every codec (no tuned variants) -- plain names.
  const int nx = kq_moe_pick_nx((int64_t)N * T, K, false);
  std::string kname = "kq_" + kquant_type_ + "_gather_qmv_mix_ns" +
      kq_nx_suffix(nx) + "_" + kq_type_string(x.dtype());
  kq_moe_log_kname(kname);
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(w, 0);
  ce.set_input_array(x, 1);
  ce.set_input_array(indices, 2);
  ce.set_input_array(scores, 3);
  ce.set_output_array(out, 4);
  ce.set_bytes(K, 5);
  ce.set_bytes(N, 6);
  ce.set_bytes(S, 7);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(N / (64 / nx), 1, T);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantMoERouterTopK::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& indices = outputs[0];
  auto& scores = outputs[1];
  indices.set_data(mx::allocator::malloc(indices.nbytes()));
  scores.set_data(mx::allocator::malloc(scores.nbytes()));

  const auto& logits = inputs[0];
  int SHARED = shared_ ? 1 : 0;
  int HAS_PES = has_pes_ ? 1 : 0;
  int HAS_BIAS = has_bias_ ? 1 : 0;
  int SCORING = scoring_;
  float SCALE = scale_;
  int T = logits.shape(0);
  int E = logits.shape(1) - SHARED;
  int R = top_k_;
  int NORM = norm_ ? 1 : 0;
  // pes/bias must be bound buffers even when unused; logits stands in.
  const auto& pes = has_pes_ ? inputs[1] : inputs[0];
  const auto& bias = has_bias_ ? inputs[1 + (has_pes_ ? 1 : 0)] : inputs[0];

  std::string kname = "kq_moe_router_topk_" + kq_type_string(logits.dtype());
  kq_moe_log_kname(kname);
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(logits, 0);
  ce.set_output_array(indices, 1);
  ce.set_output_array(scores, 2);
  ce.set_bytes(E, 3);
  ce.set_bytes(R, 4);
  ce.set_bytes(NORM, 5);
  ce.set_bytes(SHARED, 6);
  ce.set_input_array(pes, 7);
  ce.set_bytes(HAS_PES, 8);
  ce.set_input_array(bias, 9);
  ce.set_bytes(HAS_BIAS, 10);
  ce.set_bytes(SCORING, 11);
  ce.set_bytes(SCALE, 12);
  MTL::Size group_dims(256, 1, 1);
  MTL::Size grid_dims(T, 1, 1);
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

void KQuantGatherQMVMixNSKQ::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.gather_qmv_mix_ns_kq] requires Metal.");
}

void KQuantMoERouterTopK::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.moe_router_topk] requires Metal.");
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
  return kquant_type_ == o.kquant_type_ && act_ == o.act_ &&
      limit_ == o.limit_ && alpha_ == o.alpha_;
}

bool KQuantGatherQMVKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMVKQ&>(other);
  return kquant_type_ == o.kquant_type_;
}

std::vector<mx::Shape> KQuantMoEGLUKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  // Indices are the last input in both the unbiased and biased layouts.
  const auto& idx = inputs.back();
  return {{idx.shape(0), idx.shape(1), inputs[0].shape(1)}};
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
  return kquant_type_ == o.kquant_type_ && act_ == o.act_ &&
      shexp_type_ == o.shexp_type_;
}

bool KQuantGatherQMVMixKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMVMixKQ&>(other);
  return kquant_type_ == o.kquant_type_ && shexp_type_ == o.shexp_type_;
}

std::vector<mx::Shape> KQuantMoEGLUShexpKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[5].shape(0), inputs[5].shape(1) + 1, inputs[0].shape(1)}};
}

std::vector<mx::Shape> KQuantGatherQMVMixKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[2].shape(0), inputs[0].shape(1)}};
}

void KQuantGatherQMVMixNSKQ::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_mix_ns_kq] has no CPU implementation.");
}

bool KQuantGatherQMVMixNSKQ::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMVMixNSKQ&>(other);
  return kquant_type_ == o.kquant_type_;
}

std::vector<mx::Shape> KQuantGatherQMVMixNSKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[1].shape(0), inputs[0].shape(1)}};
}

void KQuantMoERouterTopK::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.moe_router_topk] has no CPU implementation.");
}

bool KQuantMoERouterTopK::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMoERouterTopK&>(other);
  return top_k_ == o.top_k_ && norm_ == o.norm_ && shared_ == o.shared_ &&
      has_pes_ == o.has_pes_ && has_bias_ == o.has_bias_ &&
      scoring_ == o.scoring_ && scale_ == o.scale_;
}

std::vector<mx::Shape> KQuantMoERouterTopK::output_shapes(
    const std::vector<mx::array>& inputs) {
  int T = inputs[0].shape(0);
  return {{T, top_k_}, {T, top_k_ + (shared_ ? 1 : 0)}};
}

std::vector<mx::Shape> KQuantGatherQMVKQ::output_shapes(
    const std::vector<mx::array>& inputs) {
  // Indices are the last input in both the unbiased and biased layouts.
  const auto& idx = inputs.back();
  return {{idx.shape(0), idx.shape(1), inputs[0].shape(1)}};
}

std::vector<mx::Shape> KQuantMoEGLU::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[7].shape(0), inputs[7].shape(1), inputs[0].shape(1)}};
}

std::vector<mx::Shape> KQuantGatherQMVBias::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {{inputs[4].shape(0), inputs[4].shape(1), inputs[0].shape(1)}};
}

// Codecs with the fused kq GLU/gather kernels wired (kq_moe_glu_kq.h):
// tuned kernels for q6_k/q8_0, generic Ext-trait kernels for the rest of the
// codec matrix. Unsupported codecs must fall back to the stock path
// per-tensor (callers gate on this via ops throwing invalid_argument).
bool codec_has_moe_glu(const std::string& t) {
  return t == "q2_k" || t == "q3_k" || t == "q4_k" || t == "q5_k" ||
      t == "q6_k" || t == "q8_0" || t == "q4_0" || t == "q4_1" || t == "q5_0" ||
      t == "q5_1" || t == "iq4_nl" || t == "iq4_xs" || t == "iq3_s" ||
      t == "iq3_xxs" || t == "iq2_xxs" || t == "iq2_xs" || t == "iq2_s" ||
      t == "iq1_s" || t == "iq1_m" || t == "mxfp4" || t == "nvfp4";
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

// Mixed-codec shared experts instantiate only the UD-style upcast combos:
// shexp codec == expert codec, or q6_k / q8_0 over anything.
bool shexp_combo_has_kernel(
    const std::string& kquant_type,
    const std::string& shexp_type) {
  return codec_has_moe_glu(kquant_type) && codec_has_moe_glu(shexp_type) &&
      (shexp_type == kquant_type || shexp_type == "q6_k" ||
       shexp_type == "q8_0");
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
  // Whole blocks suffice: the generic Ext-trait kernels stride K in
  // 16-weight chunks; the tuned q6_k/q8_0 kernels additionally need
  // K % 256 == 0 and kq_gather_stem falls back to the generic q8_0_ext
  // instantiations otherwise (q6_k's superblock IS 256).
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
// must match the expert stack's row shape (in its OWN codec's wire bytes).
void check_kq_shexp_row(
    const char* op,
    const mx::array& w,
    const std::string& kquant_type,
    int K,
    int N) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        std::string(op) + " unknown shared-expert codec '" + kquant_type +
        "'.");
  }
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
    float limit,
    const std::optional<mx::array>& gate_bias,
    const std::optional<mx::array>& up_bias,
    float alpha,
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
  if (act != "silu" && act != "gelu" && act != "silu_limit" &&
      act != "swiglu_clamp") {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] act must be 'silu', 'gelu', "
        "'silu_limit' or 'swiglu_clamp'.");
  }
  if (act == "silu_limit" && !(limit > 0.0f)) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] act 'silu_limit' requires limit > 0.");
  }
  const bool biased = gate_bias.has_value() || up_bias.has_value();
  if (act == "swiglu_clamp") {
    if (!gate_bias.has_value() || !up_bias.has_value() || !(limit > 0.0f) ||
        !(alpha > 0.0f)) {
      throw std::invalid_argument(
          "[mlx_kquant.moe_glu_gather_kq] act 'swiglu_clamp' requires "
          "gate_bias, up_bias, limit > 0 and alpha > 0.");
    }
    if (kquant_type != "mxfp4" && kquant_type != "nvfp4") {
      throw std::invalid_argument(
          "[mlx_kquant.moe_glu_gather_kq] biased kernels are instantiated "
          "for mxfp4/nvfp4 only.");
    }
  } else if (biased) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] gate_bias/up_bias require act "
        "'swiglu_clamp'.");
  }
  int K = x.shape(1);
  check_kq_expert_stack(
      "[mlx_kquant.moe_glu_gather_kq]", gate_w, kquant_type, K);
  check_kq_expert_stack("[mlx_kquant.moe_glu_gather_kq]", up_w, kquant_type, K);
  if (gate_w.shape(0) != up_w.shape(0) || gate_w.shape(1) != up_w.shape(1)) {
    throw std::invalid_argument(
        "[mlx_kquant.moe_glu_gather_kq] gate/up expert shapes must match.");
  }
  if (biased) {
    for (const auto* b : {&*gate_bias, &*up_bias}) {
      if (b->ndim() != 2 || b->shape(0) != gate_w.shape(0) ||
          b->shape(1) != gate_w.shape(1)) {
        throw std::invalid_argument(
            "[mlx_kquant.moe_glu_gather_kq] biases must be [E, N] matching "
            "the expert stacks.");
      }
    }
  }

  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  mx::Shape out_shape = {x.shape(0), indices.shape(1), gate_w.shape(1)};
  std::vector<mx::array> op_inputs = {std::move(gate_w), std::move(up_w)};
  if (biased) {
    op_inputs.push_back(prep_bias(*gate_bias, s));
    op_inputs.push_back(prep_bias(*up_bias, s));
  }
  op_inputs.push_back(std::move(x_c));
  op_inputs.push_back(prep_indices(indices, s));
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantMoEGLUKQ>(s, kquant_type, act, limit, alpha),
      std::move(op_inputs));
}

mx::array gather_qmv_kq(
    mx::array x,
    mx::array w,
    const std::string& kquant_type,
    mx::array indices,
    const std::optional<mx::array>& bias,
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
  if (bias.has_value()) {
    if (kquant_type != "mxfp4" && kquant_type != "nvfp4") {
      throw std::invalid_argument(
          "[mlx_kquant.gather_qmv_kq] biased kernels are instantiated for "
          "mxfp4/nvfp4 only.");
    }
    if (bias->ndim() != 2 || bias->shape(0) != w.shape(0) ||
        bias->shape(1) != w.shape(1)) {
      throw std::invalid_argument(
          "[mlx_kquant.gather_qmv_kq] bias must be [E, N] matching the "
          "expert stack.");
    }
  }

  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  mx::Shape out_shape = {x.shape(0), x.shape(1), w.shape(1)};
  std::vector<mx::array> op_inputs = {std::move(w)};
  if (bias.has_value()) {
    op_inputs.push_back(prep_bias(*bias, s));
  }
  op_inputs.push_back(std::move(x_c));
  op_inputs.push_back(prep_indices(indices, s));
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVKQ>(s, kquant_type),
      std::move(op_inputs));
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
    const std::string& shexp_kquant_type,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.moe_glu_gather_shexp_kq]";
  const std::string shexp_type =
      shexp_kquant_type.empty() ? kquant_type : shexp_kquant_type;
  if (!shexp_combo_has_kernel(kquant_type, shexp_type)) {
    throw std::invalid_argument(
        std::string(op) + " no fused kernel for expert codec '" + kquant_type +
        "' with shared-expert codec '" + shexp_type + "'.");
  }
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
  check_kq_shexp_row(op, shexp_gate_w, shexp_type, K, N);
  check_kq_shexp_row(op, shexp_up_w, shexp_type, K, N);

  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  mx::Shape out_shape = {x.shape(0), indices.shape(1) + 1, N};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantMoEGLUShexpKQ>(s, kquant_type, act, shexp_type),
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
    const std::string& shexp_kquant_type,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.gather_qmv_mix_kq]";
  const std::string shexp_type =
      shexp_kquant_type.empty() ? kquant_type : shexp_kquant_type;
  if (!shexp_combo_has_kernel(kquant_type, shexp_type)) {
    throw std::invalid_argument(
        std::string(op) + " no fused kernel for expert codec '" + kquant_type +
        "' with shared-expert codec '" + shexp_type + "'.");
  }
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
  check_kq_shexp_row(op, shexp_w, shexp_type, K, w.shape(1));

  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  mx::Shape out_shape = {x.shape(0), w.shape(1)};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVMixKQ>(s, kquant_type, shexp_type),
      {std::move(w),
       std::move(shexp_w),
       std::move(x_c),
       prep_indices(indices, s),
       prep_scores(scores, s)});
}

mx::array gather_qmv_mix_ns_kq(
    mx::array x,
    mx::array w,
    const std::string& kquant_type,
    mx::array indices,
    mx::array scores,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.gather_qmv_mix_ns_kq]";
  if (x.ndim() != 3) {
    throw std::invalid_argument(std::string(op) + " x must be 3-D [T, S, K].");
  }
  int S = x.shape(1);
  if (indices.ndim() != 2 || indices.shape(0) != x.shape(0) ||
      indices.shape(1) != S) {
    throw std::invalid_argument(std::string(op) + " indices must be [T, S].");
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

  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  mx::Shape out_shape = {x.shape(0), w.shape(1)};
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVMixNSKQ>(s, kquant_type),
      {std::move(w),
       std::move(x_c),
       prep_indices(indices, s),
       prep_scores(scores, s)});
}

std::vector<mx::array> moe_router_topk(
    mx::array logits,
    int top_k,
    bool norm_topk_prob,
    bool shared_gate,
    const std::optional<mx::array>& per_expert_scale,
    const std::optional<mx::array>& bias,
    const std::string& scoring,
    float scale,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.moe_router_topk]";
  const int shared = shared_gate ? 1 : 0;
  int scoring_i;
  if (scoring == "softmax") {
    scoring_i = 0;
  } else if (scoring == "sqrtsoftplus") {
    scoring_i = 1;
  } else {
    throw std::invalid_argument(
        std::string(op) + " scoring must be softmax or sqrtsoftplus.");
  }
  if (scoring_i == 1 && !norm_topk_prob) {
    throw std::invalid_argument(
        std::string(op) + " sqrtsoftplus requires norm_topk_prob.");
  }
  if (logits.ndim() != 2) {
    throw std::invalid_argument(
        std::string(op) + " logits must be 2-D [T, E + shared_gate].");
  }
  auto dt = logits.dtype();
  if (dt != mx::float32 && dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " logits must be float32/float16/bfloat16.");
  }
  int E = logits.shape(1) - shared;
  if (top_k < 1 || top_k > 16 || top_k > E) {
    throw std::invalid_argument(
        std::string(op) + " top_k must be in [1, min(E, 16)].");
  }
  if (E > 1024) {
    throw std::invalid_argument(std::string(op) + " requires E <= 1024.");
  }

  auto l_c =
      logits.flags().row_contiguous ? logits : mx::contiguous(logits, false, s);
  int T = l_c.shape(0);
  std::vector<mx::array> inputs = {std::move(l_c)};
  if (per_expert_scale.has_value()) {
    const auto& pes = *per_expert_scale;
    if (pes.ndim() != 1 || pes.shape(0) != E) {
      throw std::invalid_argument(
          std::string(op) + " per_expert_scale must be 1-D [E].");
    }
    inputs.push_back(prep_scores(pes, s));
  }
  if (bias.has_value()) {
    const auto& b = *bias;
    if (b.ndim() != 1 || b.shape(0) != E) {
      throw std::invalid_argument(std::string(op) + " bias must be 1-D [E].");
    }
    inputs.push_back(prep_scores(b, s));
  }
  return mx::array::make_arrays(
      {{T, top_k}, {T, top_k + shared}},
      {mx::uint32, mx::float32},
      std::make_shared<KQuantMoERouterTopK>(
          s,
          top_k,
          norm_topk_prob,
          shared_gate,
          per_expert_scale.has_value(),
          bias.has_value(),
          scoring_i,
          scale),
      std::move(inputs));
}

} // namespace mlx_kquant
