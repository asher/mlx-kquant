// Fused deepseek4 hyper-connection glue ops for the single-token decode
// route (see kq_hc_glue.h for the kernel shapes). Four streams (hc_mult 4)
// baked; the gmlx caller gates on that. GPU only, like the dsa ops.
#include <stdexcept>
#include <string>

#include "kquant.h"
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

constexpr int HC = 4;
constexpr int MIX = (2 + HC) * HC;

mx::array prep_hc_act(
    const mx::array& x,
    const char* op,
    const char* what,
    mx::Stream s) {
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must be float16 or bfloat16.");
  }
  return mx::contiguous(x, false, s);
}

mx::array prep_hc_f32(
    const mx::array& a,
    const char* op,
    const char* what,
    mx::Stream s) {
  if (a.dtype() != mx::float32) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must be float32.");
  }
  return mx::contiguous(a, false, s);
}

// [..., 4, D] stream tensor: validates the stream axis and the D limits the
// kernels assume, returns D.
int check_streams(const mx::array& x, const char* op, const char* what) {
  if (x.ndim() < 2 || x.shape(-2) != HC) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must be [..., 4, D].");
  }
  int D = x.shape(-1);
  if (D % 8 != 0 || D > 8 * 1024) {
    throw std::invalid_argument(
        std::string(op) + " D must be a multiple of 8 and at most 8192.");
  }
  return D;
}

void check_fn(const mx::array& fn, int D, const char* op) {
  if (fn.ndim() != 2 || fn.shape(0) != MIX || fn.shape(1) != HC * D) {
    throw std::invalid_argument(
        std::string(op) + " fn must be [24, 4 * D] float32.");
  }
}

} // namespace

#ifdef _METAL_

void KQuantHcFrontReduce::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
  const auto& x = inputs[0];
  int D = x.shape(-1);
  int rows = int(x.size() / (HC * D));

  std::string kname = "kq_hc_front_reduce_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(x, 0);
  ce.set_input_array(inputs[1], 1);
  ce.set_output_array(outputs[0], 2);
  ce.set_output_array(outputs[1], 3);
  ce.set_bytes(D, 4);
  ce.dispatch_threadgroups(
      MTL::Size(rows * (MIX + 1), 1, 1), MTL::Size(256, 1, 1));
}

void KQuantHcFrontExpandReduce::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
  const auto& resid = inputs[1];
  int D = resid.shape(-1);
  int rows = int(resid.size() / (HC * D));

  std::string kname =
      "kq_hc_front_expand_reduce_" + kq_type_string(resid.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(inputs[0], 0);
  ce.set_input_array(resid, 1);
  ce.set_input_array(inputs[2], 2);
  ce.set_input_array(inputs[3], 3);
  ce.set_input_array(inputs[4], 4);
  ce.set_output_array(outputs[0], 5);
  ce.set_output_array(outputs[1], 6);
  ce.set_output_array(outputs[2], 7);
  ce.set_bytes(D, 8);
  ce.dispatch_threadgroups(
      MTL::Size(rows * (MIX + 1), 1, 1), MTL::Size(256, 1, 1));
}

void KQuantHcSinkhornCollapse::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
  const auto& x = inputs[0];
  int D = x.shape(-1);
  int rows = int(x.size() / (HC * D));

  std::string kname = "kq_hc_sinkhorn_collapse_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(x, 0);
  ce.set_input_array(inputs[1], 1);
  ce.set_input_array(inputs[2], 2);
  ce.set_input_array(inputs[3], 3);
  ce.set_input_array(inputs[4], 4);
  ce.set_input_array(inputs[5], 5);
  ce.set_output_array(outputs[0], 6);
  ce.set_output_array(outputs[1], 7);
  ce.set_output_array(outputs[2], 8);
  ce.set_bytes(D, 9);
  ce.set_bytes(iters_, 10);
  ce.set_bytes(hc_eps_, 11);
  ce.set_bytes(norm_eps_, 12);
  ce.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
}

void KQuantHcExpand::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  const auto& resid = inputs[1];
  int D = resid.shape(-1);
  int rows = int(resid.size() / (HC * D));

  std::string kname = "kq_hc_expand_" + kq_type_string(resid.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(inputs[0], 0);
  ce.set_input_array(resid, 1);
  ce.set_input_array(inputs[2], 2);
  ce.set_input_array(inputs[3], 3);
  ce.set_output_array(out, 4);
  ce.set_bytes(D, 5);
  ce.dispatch_threadgroups(MTL::Size(rows * 2, 1, 1), MTL::Size(256, 1, 1));
}

#else // !_METAL_

void KQuantHcFrontReduce::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.hc_front_reduce] requires Metal.");
}

void KQuantHcFrontExpandReduce::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.hc_front_expand_reduce] requires Metal.");
}

void KQuantHcSinkhornCollapse::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.hc_sinkhorn_collapse] requires Metal.");
}

void KQuantHcExpand::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.hc_expand] requires Metal.");
}

#endif

void KQuantHcFrontReduce::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.hc_front_reduce] has no CPU implementation.");
}

void KQuantHcFrontExpandReduce::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.hc_front_expand_reduce] has no CPU implementation.");
}

void KQuantHcSinkhornCollapse::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.hc_sinkhorn_collapse] has no CPU implementation.");
}

void KQuantHcExpand::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.hc_expand] has no CPU implementation.");
}

bool KQuantHcSinkhornCollapse::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantHcSinkhornCollapse&>(other);
  return iters_ == o.iters_ && hc_eps_ == o.hc_eps_ && norm_eps_ == o.norm_eps_;
}

std::vector<mx::array>
hc_front_reduce(mx::array x, mx::array fn, mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.hc_front_reduce]";
  int D = check_streams(x, op, "x");
  check_fn(fn, D, op);
  auto x_c = prep_hc_act(x, op, "x", s);
  auto fn_c = prep_hc_f32(fn, op, "fn", s);

  auto lead = x.shape();
  lead.pop_back();
  lead.pop_back();
  auto mix_shape = lead;
  mix_shape.push_back(MIX);
  auto ssq_shape = lead;
  ssq_shape.push_back(1);
  return mx::array::make_arrays(
      {std::move(mix_shape), std::move(ssq_shape)},
      {mx::float32, mx::float32},
      std::make_shared<KQuantHcFrontReduce>(s),
      {std::move(x_c), std::move(fn_c)});
}

std::vector<mx::array> hc_front_expand_reduce(
    mx::array x_sub,
    mx::array resid,
    mx::array post,
    mx::array comb,
    mx::array fn,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.hc_front_expand_reduce]";
  int D = check_streams(resid, op, "resid");
  check_fn(fn, D, op);
  if (x_sub.shape(-1) != D || x_sub.size() != resid.size() / HC) {
    throw std::invalid_argument(
        std::string(op) + " x_sub must be [..., D] matching resid.");
  }
  if (x_sub.dtype() != resid.dtype()) {
    throw std::invalid_argument(
        std::string(op) + " x_sub and resid dtypes must match.");
  }
  int64_t rows = resid.size() / (int64_t(HC) * D);
  if (int64_t(post.size()) != rows * HC ||
      int64_t(comb.size()) != rows * HC * HC) {
    throw std::invalid_argument(
        std::string(op) + " post/comb must be [..., 4] / [..., 4, 4].");
  }
  auto xs_c = prep_hc_act(x_sub, op, "x_sub", s);
  auto r_c = prep_hc_act(resid, op, "resid", s);
  auto p_c = prep_hc_f32(post, op, "post", s);
  auto c_c = prep_hc_f32(comb, op, "comb", s);
  auto fn_c = prep_hc_f32(fn, op, "fn", s);

  auto lead = resid.shape();
  lead.pop_back();
  lead.pop_back();
  auto mix_shape = lead;
  mix_shape.push_back(MIX);
  auto ssq_shape = lead;
  ssq_shape.push_back(1);
  return mx::array::make_arrays(
      {resid.shape(), std::move(mix_shape), std::move(ssq_shape)},
      {resid.dtype(), mx::float32, mx::float32},
      std::make_shared<KQuantHcFrontExpandReduce>(s),
      {std::move(xs_c),
       std::move(r_c),
       std::move(p_c),
       std::move(c_c),
       std::move(fn_c)});
}

std::vector<mx::array> hc_sinkhorn_collapse(
    mx::array x,
    mx::array mixes_raw,
    mx::array sumsq,
    mx::array scale,
    mx::array base,
    mx::array w,
    int iters,
    float hc_eps,
    float norm_eps,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.hc_sinkhorn_collapse]";
  int D = check_streams(x, op, "x");
  int64_t rows = x.size() / (int64_t(HC) * D);
  if (int64_t(mixes_raw.size()) != rows * MIX ||
      int64_t(sumsq.size()) != rows) {
    throw std::invalid_argument(
        std::string(op) + " mixes_raw/sumsq must be [..., 24] / [..., 1].");
  }
  if (scale.size() != 3 || int(base.size()) != MIX) {
    throw std::invalid_argument(
        std::string(op) + " scale must be [3] and base [24].");
  }
  if (w.ndim() != 1 || w.shape(0) != D || w.dtype() != x.dtype()) {
    throw std::invalid_argument(
        std::string(op) + " w must be [D] in the activation dtype.");
  }
  if (iters < 1) {
    throw std::invalid_argument(std::string(op) + " iters must be >= 1.");
  }
  auto x_c = prep_hc_act(x, op, "x", s);
  auto m_c = prep_hc_f32(mixes_raw, op, "mixes_raw", s);
  auto q_c = prep_hc_f32(sumsq, op, "sumsq", s);
  auto sc_c = prep_hc_f32(scale, op, "scale", s);
  auto b_c = prep_hc_f32(base, op, "base", s);
  auto w_c = mx::contiguous(w, false, s);

  auto lead = x.shape();
  lead.pop_back();
  lead.pop_back();
  auto col_shape = lead;
  col_shape.push_back(D);
  auto post_shape = lead;
  post_shape.push_back(HC);
  auto comb_shape = lead;
  comb_shape.push_back(HC);
  comb_shape.push_back(HC);
  return mx::array::make_arrays(
      {std::move(col_shape), std::move(post_shape), std::move(comb_shape)},
      {x.dtype(), mx::float32, mx::float32},
      std::make_shared<KQuantHcSinkhornCollapse>(s, iters, hc_eps, norm_eps),
      {std::move(x_c),
       std::move(m_c),
       std::move(q_c),
       std::move(sc_c),
       std::move(b_c),
       std::move(w_c)});
}

mx::array hc_expand(
    mx::array x,
    mx::array resid,
    mx::array post,
    mx::array comb,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.hc_expand]";
  int D = check_streams(resid, op, "resid");
  if (x.shape(-1) != D || x.size() != resid.size() / HC ||
      x.dtype() != resid.dtype()) {
    throw std::invalid_argument(
        std::string(op) + " x must be [..., D] matching resid.");
  }
  int64_t rows = resid.size() / (int64_t(HC) * D);
  if (int64_t(post.size()) != rows * HC ||
      int64_t(comb.size()) != rows * HC * HC) {
    throw std::invalid_argument(
        std::string(op) + " post/comb must be [..., 4] / [..., 4, 4].");
  }
  auto x_c = prep_hc_act(x, op, "x", s);
  auto r_c = prep_hc_act(resid, op, "resid", s);
  auto p_c = prep_hc_f32(post, op, "post", s);
  auto c_c = prep_hc_f32(comb, op, "comb", s);

  return mx::array(
      resid.shape(),
      resid.dtype(),
      std::make_shared<KQuantHcExpand>(s),
      {std::move(x_c), std::move(r_c), std::move(p_c), std::move(c_c)});
}

} // namespace mlx_kquant
