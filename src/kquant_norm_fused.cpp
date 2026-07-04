// Fused residual + RMSNorm glue ops. Transformer layer glue at decode is a
// chain of tiny dependent dispatches (norms, residual adds, per-layer
// scalars); each op here replaces one recurring pattern with a single
// dispatch (see kq_norm_fused.h for the kernel shapes). The scalar CPU evals
// mirror the kernels' f32-accumulate / one-round-at-write semantics;
// rms_norm semantics match mx::fast::rms_norm.
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

// Row-contiguous activation with a supported dtype, or throw.
mx::array
prep_act(const mx::array& x, const char* op, const char* what, mx::Stream s) {
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must be float16 or bfloat16.");
  }
  return x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
}

// 1-D [D] weight matching the activation dtype, or throw.
mx::array prep_norm_weight(
    const mx::array& w,
    const mx::array& x,
    int D,
    const char* op,
    const char* what,
    mx::Stream s) {
  if (w.ndim() != 1 || w.shape(0) != D) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must be 1-D [" + std::to_string(D) +
        "].");
  }
  if (w.dtype() != x.dtype()) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " dtype must match the activations.");
  }
  return w.flags().row_contiguous ? w : mx::contiguous(w, false, s);
}

} // namespace

#ifdef _METAL_

namespace {

// One threadgroup per row, sized so each thread covers 4 contiguous elements
// (KQ_NORM_NREADS); rows wider than 4096 loop strided inside the kernel.
MTL::Size kq_norm_group_dims(int D) {
  int threads = (D + 3) / 4;
  threads = ((threads + 31) / 32) * 32;
  if (threads > 1024) {
    threads = 1024;
  }
  return MTL::Size(threads, 1, 1);
}

} // namespace

void KQuantAddRMSNorm::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& h = inputs[0];
  const auto& residual = inputs[1];
  const auto& w = inputs[2];
  // scale must be a bound buffer even when unused; h stands in.
  const auto& lscale = has_scale_ ? inputs[3] : inputs[0];
  int HAS_SCALE = has_scale_ ? 1 : 0;

  int D = h.shape(-1);
  int T = int(h.size() / D);

  std::string kname = "kq_add_rmsnorm_" + kq_type_string(h.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(h, 0);
  ce.set_input_array(residual, 1);
  ce.set_input_array(w, 2);
  ce.set_input_array(lscale, 3);
  ce.set_output_array(out, 4);
  ce.set_bytes(D, 5);
  ce.set_bytes(eps_, 6);
  ce.set_bytes(HAS_SCALE, 7);
  MTL::Size group_dims = kq_norm_group_dims(D);
  MTL::Size grid_dims(T, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantRMSNormMulti3::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }

  const auto& x = inputs[0];
  int D = x.shape(-1);
  int T = int(x.size() / D);

  std::string kname = "kq_rmsnorm_multi3_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(x, 0);
  ce.set_input_array(inputs[1], 1);
  ce.set_input_array(inputs[2], 2);
  ce.set_input_array(inputs[3], 3);
  ce.set_output_array(outputs[0], 4);
  ce.set_output_array(outputs[1], 5);
  ce.set_output_array(outputs[2], 6);
  ce.set_bytes(D, 7);
  ce.set_bytes(eps_, 8);
  MTL::Size group_dims = kq_norm_group_dims(D);
  MTL::Size grid_dims(T, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantRMSNorm2Add::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& a = inputs[0];
  int D = a.shape(-1);
  int T = int(a.size() / D);

  std::string kname = "kq_rmsnorm2_add_" + kq_type_string(a.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(a, 0);
  ce.set_input_array(inputs[1], 1);
  ce.set_input_array(inputs[2], 2);
  ce.set_input_array(inputs[3], 3);
  ce.set_output_array(out, 4);
  ce.set_bytes(D, 5);
  ce.set_bytes(eps_, 6);
  MTL::Size group_dims = kq_norm_group_dims(D);
  MTL::Size grid_dims(T, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantAddRMSNorm::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.add_rmsnorm] requires Metal.");
}

void KQuantRMSNormMulti3::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.rmsnorm_multi3] requires Metal.");
}

void KQuantRMSNorm2Add::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.rmsnorm2_add] requires Metal.");
}

#endif

namespace {

// Row mean-square -> 1/sqrt(ms + eps), f32 accumulate (kernel pass 1).
template <typename T>
float row_inv_rms(const T* row, int D, float eps) {
  float ss = 0;
  for (int i = 0; i < D; i++) {
    const float x = static_cast<float>(row[i]);
    ss += x * x;
  }
  return 1.0f / std::sqrt(ss / static_cast<float>(D) + eps);
}

// Dispatch a per-dtype functor over the f16/bf16 activation dtypes the
// factories admit.
template <typename F>
void norm_cpu_dispatch(mx::Dtype dt, const char* op, F&& run) {
  if (dt == mx::float16) {
    run(static_cast<mx::float16_t*>(nullptr));
  } else if (dt == mx::bfloat16) {
    run(static_cast<mx::bfloat16_t*>(nullptr));
  } else {
    throw std::runtime_error(
        std::string(op) + " only float16/bfloat16 inputs are supported.");
  }
}

} // namespace

void KQuantAddRMSNorm::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& h = inputs[0];
  const auto& residual = inputs[1];
  const auto& w = inputs[2];
  const auto& lscale = has_scale_ ? inputs[3] : inputs[0];

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(h);
  encoder.set_input_array(residual);
  encoder.set_input_array(w);
  encoder.set_input_array(lscale);
  encoder.set_output_array(out);
  encoder.dispatch([out = mx::array::unsafe_weak_copy(out),
                    h = mx::array::unsafe_weak_copy(h),
                    residual = mx::array::unsafe_weak_copy(residual),
                    w = mx::array::unsafe_weak_copy(w),
                    lscale = mx::array::unsafe_weak_copy(lscale),
                    eps = eps_,
                    has_scale = has_scale_]() mutable {
    const int D = h.shape(-1);
    const int64_t T = h.size() / D;
    norm_cpu_dispatch(h.dtype(), "[mlx_kquant.add_rmsnorm]", [&](auto* tag) {
      using DT = std::remove_pointer_t<decltype(tag)>;
      const DT* hp = h.data<DT>();
      const DT* rp = residual.data<DT>();
      const DT* wp = w.data<DT>();
      const float sc =
          has_scale ? static_cast<float>(lscale.data<DT>()[0]) : 1.0f;
      DT* op = out.data<DT>();
      for (int64_t t = 0; t < T; t++) {
        const DT* hrow = hp + t * D;
        const DT* rrow = rp + t * D;
        DT* orow = op + t * D;
        const float inv = row_inv_rms(hrow, D, eps);
        for (int i = 0; i < D; i++) {
          const float o = static_cast<float>(rrow[i]) +
              static_cast<float>(wp[i]) * static_cast<float>(hrow[i]) * inv;
          orow[i] = static_cast<DT>(o * sc);
        }
      }
    });
  });
}

void KQuantRMSNormMulti3::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }

  auto& encoder = mx::cpu::get_command_encoder(stream());
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  for (auto& out : outputs) {
    encoder.set_output_array(out);
  }
  encoder.dispatch([x = mx::array::unsafe_weak_copy(inputs[0]),
                    w0 = mx::array::unsafe_weak_copy(inputs[1]),
                    w1 = mx::array::unsafe_weak_copy(inputs[2]),
                    w2 = mx::array::unsafe_weak_copy(inputs[3]),
                    o0 = mx::array::unsafe_weak_copy(outputs[0]),
                    o1 = mx::array::unsafe_weak_copy(outputs[1]),
                    o2 = mx::array::unsafe_weak_copy(outputs[2]),
                    eps = eps_]() mutable {
    const int D = x.shape(-1);
    const int64_t T = x.size() / D;
    norm_cpu_dispatch(x.dtype(), "[mlx_kquant.rmsnorm_multi3]", [&](auto* tag) {
      using DT = std::remove_pointer_t<decltype(tag)>;
      const DT* xp = x.data<DT>();
      const DT* w0p = w0.data<DT>();
      const DT* w1p = w1.data<DT>();
      const DT* w2p = w2.data<DT>();
      DT* o0p = o0.data<DT>();
      DT* o1p = o1.data<DT>();
      DT* o2p = o2.data<DT>();
      for (int64_t t = 0; t < T; t++) {
        const DT* xrow = xp + t * D;
        const float inv = row_inv_rms(xrow, D, eps);
        for (int i = 0; i < D; i++) {
          const float n = static_cast<float>(xrow[i]) * inv;
          o0p[t * D + i] = static_cast<DT>(static_cast<float>(w0p[i]) * n);
          o1p[t * D + i] = static_cast<DT>(static_cast<float>(w1p[i]) * n);
          o2p[t * D + i] = static_cast<DT>(static_cast<float>(w2p[i]) * n);
        }
      }
    });
  });
}

void KQuantRMSNorm2Add::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& encoder = mx::cpu::get_command_encoder(stream());
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  encoder.dispatch([a = mx::array::unsafe_weak_copy(inputs[0]),
                    wa = mx::array::unsafe_weak_copy(inputs[1]),
                    b = mx::array::unsafe_weak_copy(inputs[2]),
                    wb = mx::array::unsafe_weak_copy(inputs[3]),
                    out = mx::array::unsafe_weak_copy(out),
                    eps = eps_]() mutable {
    const int D = a.shape(-1);
    const int64_t T = a.size() / D;
    norm_cpu_dispatch(a.dtype(), "[mlx_kquant.rmsnorm2_add]", [&](auto* tag) {
      using DT = std::remove_pointer_t<decltype(tag)>;
      const DT* ap = a.data<DT>();
      const DT* wap = wa.data<DT>();
      const DT* bp = b.data<DT>();
      const DT* wbp = wb.data<DT>();
      DT* op = out.data<DT>();
      for (int64_t t = 0; t < T; t++) {
        const DT* arow = ap + t * D;
        const DT* brow = bp + t * D;
        DT* orow = op + t * D;
        const float inva = row_inv_rms(arow, D, eps);
        const float invb = row_inv_rms(brow, D, eps);
        for (int i = 0; i < D; i++) {
          const float o =
              static_cast<float>(wap[i]) * static_cast<float>(arow[i]) * inva +
              static_cast<float>(wbp[i]) * static_cast<float>(brow[i]) * invb;
          orow[i] = static_cast<DT>(o);
        }
      }
    });
  });
}

bool KQuantAddRMSNorm::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantAddRMSNorm&>(other);
  return eps_ == o.eps_ && has_scale_ == o.has_scale_;
}

bool KQuantRMSNormMulti3::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantRMSNormMulti3&>(other);
  return eps_ == o.eps_;
}

bool KQuantRMSNorm2Add::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantRMSNorm2Add&>(other);
  return eps_ == o.eps_;
}

std::vector<mx::Shape> KQuantAddRMSNorm::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

std::vector<mx::Shape> KQuantRMSNormMulti3::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape(), inputs[0].shape(), inputs[0].shape()};
}

std::vector<mx::Shape> KQuantRMSNorm2Add::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

mx::array add_rmsnorm(
    mx::array h,
    mx::array residual,
    mx::array weight,
    float eps,
    const std::optional<mx::array>& scale,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.add_rmsnorm]";
  if (h.ndim() < 1 || h.shape(-1) < 1) {
    throw std::invalid_argument(std::string(op) + " h must have a last axis.");
  }
  if (residual.shape() != h.shape() || residual.dtype() != h.dtype()) {
    throw std::invalid_argument(
        std::string(op) + " residual must match h in shape and dtype.");
  }
  int D = h.shape(-1);
  auto h_c = prep_act(h, op, "h", s);
  auto r_c = prep_act(residual, op, "residual", s);
  auto w_c = prep_norm_weight(weight, h, D, op, "weight", s);

  std::vector<mx::array> inputs = {
      std::move(h_c), std::move(r_c), std::move(w_c)};
  if (scale.has_value()) {
    const auto& sc = *scale;
    if (sc.size() != 1) {
      throw std::invalid_argument(std::string(op) + " scale must be size 1.");
    }
    if (sc.dtype() != h.dtype()) {
      throw std::invalid_argument(
          std::string(op) + " scale dtype must match the activations.");
    }
    inputs.push_back(
        sc.flags().row_contiguous ? sc : mx::contiguous(sc, false, s));
  }
  return mx::array(
      h.shape(),
      h.dtype(),
      std::make_shared<KQuantAddRMSNorm>(s, eps, scale.has_value()),
      std::move(inputs));
}

std::vector<mx::array> rmsnorm_multi3(
    mx::array x,
    mx::array w0,
    mx::array w1,
    mx::array w2,
    float eps,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.rmsnorm_multi3]";
  if (x.ndim() < 1 || x.shape(-1) < 1) {
    throw std::invalid_argument(std::string(op) + " x must have a last axis.");
  }
  int D = x.shape(-1);
  auto x_c = prep_act(x, op, "x", s);
  auto w0_c = prep_norm_weight(w0, x, D, op, "w0", s);
  auto w1_c = prep_norm_weight(w1, x, D, op, "w1", s);
  auto w2_c = prep_norm_weight(w2, x, D, op, "w2", s);

  return mx::array::make_arrays(
      {x.shape(), x.shape(), x.shape()},
      {x.dtype(), x.dtype(), x.dtype()},
      std::make_shared<KQuantRMSNormMulti3>(s, eps),
      {std::move(x_c), std::move(w0_c), std::move(w1_c), std::move(w2_c)});
}

mx::array rmsnorm2_add(
    mx::array a,
    mx::array wa,
    mx::array b,
    mx::array wb,
    float eps,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.rmsnorm2_add]";
  if (a.ndim() < 1 || a.shape(-1) < 1) {
    throw std::invalid_argument(std::string(op) + " a must have a last axis.");
  }
  if (b.shape() != a.shape() || b.dtype() != a.dtype()) {
    throw std::invalid_argument(
        std::string(op) + " b must match a in shape and dtype.");
  }
  int D = a.shape(-1);
  auto a_c = prep_act(a, op, "a", s);
  auto b_c = prep_act(b, op, "b", s);
  auto wa_c = prep_norm_weight(wa, a, D, op, "wa", s);
  auto wb_c = prep_norm_weight(wb, a, D, op, "wb", s);

  return mx::array(
      a.shape(),
      a.dtype(),
      std::make_shared<KQuantRMSNorm2Add>(s, eps),
      {std::move(a_c), std::move(wa_c), std::move(b_c), std::move(wb_c)});
}

} // namespace mlx_kquant
