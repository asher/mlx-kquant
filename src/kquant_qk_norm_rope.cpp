// Fused per-head RMSNorm and RoPE of the DiT attention q and k (see
// kq_qk_norm_rope.h for the kernel). On the GPU one dispatch reads q and k
// through their strides and writes both in the [B, H, L, D] attention
// layout. CPU streams, and head dims without a kernel, run the same math as
// a composition of stock ops: f32 throughout, one round at the end.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/ops.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

// Head dims with a kernel; each is a multiple of 32 so a 16-lane group
// holds an even number of elements per lane.
bool qknr_kernel_dim(int D) {
  return D == 64 || D == 96 || D == 128 || D == 256;
}

// f32 rms_norm over the last axis, then the pair rotation, rounded once to
// the input dtype and written as [B, H, L, D].
mx::array qknr_compose(
    const mx::array& x,
    const mx::array& w,
    const mx::array& cos,
    const mx::array& sin,
    float eps,
    mx::Stream s) {
  const int B = x.shape(0);
  const int L = x.shape(1);
  const int H = x.shape(2);
  const int D = x.shape(3);
  const int P = D / 2;
  auto xf = mx::astype(x, mx::float32, s);
  auto ms = mx::mean(mx::square(xf, s), -1, true, s);
  auto inv = mx::rsqrt(mx::add(ms, mx::array(eps, mx::float32), s), s);
  auto y = mx::multiply(mx::multiply(xf, inv, s), w, s);
  y = mx::reshape(y, {B, L, H, P, 2}, s);
  auto y0 = mx::reshape(
      mx::slice(y, {0, 0, 0, 0, 0}, {B, L, H, P, 1}, s), {B, L, H, P}, s);
  auto y1 = mx::reshape(
      mx::slice(y, {0, 0, 0, 0, 1}, {B, L, H, P, 2}, s), {B, L, H, P}, s);
  auto c = mx::reshape(cos, {1, L, 1, P}, s);
  auto sn = mx::reshape(sin, {1, L, 1, P}, s);
  auto o0 = mx::subtract(mx::multiply(y0, c, s), mx::multiply(y1, sn, s), s);
  auto o1 = mx::add(mx::multiply(y0, sn, s), mx::multiply(y1, c, s), s);
  auto o = mx::reshape(mx::stack({o0, o1}, -1, s), {B, L, H, D}, s);
  o = mx::astype(o, x.dtype(), s);
  return mx::contiguous(mx::transpose(o, {0, 2, 1, 3}, s), false, s);
}

} // namespace

#ifdef _METAL_

namespace {

// Mirror of the Metal-side KQQkNormRopeParams (kq_qk_norm_rope.h); the
// layouts must match field for field.
struct KQQkNormRopeParams {
  int64_t q_b;
  int64_t q_l;
  int64_t q_h;
  int64_t q_d;
  int64_t k_b;
  int64_t k_l;
  int64_t k_h;
  int64_t k_d;
  int32_t L;
  int32_t H;
  float eps;
};

// Elements per vector load: a lane holds D / 16 elements and loads them 4
// at a time when that count is a multiple of 4, else 2 (head dim 96).
int qknr_vec_width(int D) {
  return (D / 16) % 4 == 0 ? 4 : 2;
}

// True when x can take vector loads: a unit d stride, and every head row
// start aligned to the vector width. Size-1 axes never move the start.
bool qknr_vec_ok(const mx::array& x, int vw) {
  if (x.strides(3) != 1) {
    return false;
  }
  for (int ax = 0; ax < 3; ++ax) {
    if (x.shape(ax) > 1 && x.strides(ax) % vw != 0) {
      return false;
    }
  }
  return x.offset() % (int64_t(vw) * x.itemsize()) == 0;
}

// Simdgroups per threadgroup: a (b, l) row holds 2 * H head rows, two per
// simdgroup pass. Prefer the count that splits the passes evenly.
int qknr_simdgroups(int H) {
  const int passes = H;
  for (int n : {5, 6, 4, 3, 8, 2}) {
    if (passes % n == 0) {
      return n;
    }
  }
  return 4;
}

} // namespace

void KQuantQkNormRope::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const int B = q.shape(0);
  const int L = q.shape(1);
  const int H = q.shape(2);
  const int D = q.shape(3);
  if (q.size() == 0) {
    return;
  }

  const int vw = qknr_vec_width(D);
  bool vec = qknr_vec_ok(q, vw) && qknr_vec_ok(k, vw);
  KQQkNormRopeParams p{
      q.strides(0),
      q.strides(1),
      q.strides(2),
      q.strides(3),
      k.strides(0),
      k.strides(1),
      k.strides(2),
      k.strides(3),
      L,
      H,
      eps_};

  const std::string kname =
      "kq_qk_norm_rope_" + kq_type_string(q.dtype()) + "_" + std::to_string(D);
  mx::metal::MTLFCList func_consts = {
      {&vec, MTL::DataType::DataTypeBool, 350},
  };
  auto kernel =
      kq_get_kernel(d, kname, kname + (vec ? "_vec" : "_strided"), func_consts);
  const int cap = std::min(int(kernel->maxTotalThreadsPerThreadgroup()), 1024);
  const int nsg = std::max(1, std::min(qknr_simdgroups(H), cap / 32));

  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(q, 0);
  ce.set_input_array(k, 1);
  ce.set_input_array(inputs[2], 2);
  ce.set_input_array(inputs[3], 3);
  ce.set_input_array(inputs[4], 4);
  ce.set_input_array(inputs[5], 5);
  ce.set_output_array(outputs[0], 6);
  ce.set_output_array(outputs[1], 7);
  ce.set_bytes(p, 8);
  ce.dispatch_threadgroups(
      MTL::Size(size_t(B) * L, 1, 1), MTL::Size(nsg * 32, 1, 1));
}

#else // !_METAL_

void KQuantQkNormRope::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.qk_norm_rope] requires Metal.");
}

#endif

void KQuantQkNormRope::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.qk_norm_rope] CPU streams build the composed ops.");
}

bool KQuantQkNormRope::is_equivalent(const mx::Primitive& other) const {
  return eps_ == static_cast<const KQuantQkNormRope&>(other).eps_;
}

std::vector<mx::Shape> KQuantQkNormRope::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& q = inputs[0];
  mx::Shape shape = {q.shape(0), q.shape(2), q.shape(1), q.shape(3)};
  return {shape, shape};
}

std::pair<mx::array, mx::array> qk_norm_rope(
    mx::array q,
    mx::array k,
    mx::array q_w,
    mx::array k_w,
    mx::array cos,
    mx::array sin,
    float eps,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const std::string op = "[mlx_kquant.qk_norm_rope]";
  if (q.ndim() != 4) {
    throw std::invalid_argument(op + " q must be [B, L, H, D].");
  }
  if (k.shape() != q.shape() || k.dtype() != q.dtype()) {
    throw std::invalid_argument(op + " k must match q in shape and dtype.");
  }
  const auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(op + " q and k must be float16 or bfloat16.");
  }
  const int B = q.shape(0);
  const int L = q.shape(1);
  const int H = q.shape(2);
  const int D = q.shape(3);
  if (D < 2 || D % 2 != 0) {
    throw std::invalid_argument(op + " head dim must be even.");
  }
  for (const auto* w : {&q_w, &k_w}) {
    if (w->ndim() != 1 || w->shape(0) != D ||
        !mx::issubdtype(w->dtype(), mx::floating)) {
      throw std::invalid_argument(
          op + " q_w and k_w must be floating [" + std::to_string(D) + "].");
    }
  }
  for (const auto* t : {&cos, &sin}) {
    if (t->ndim() != 2 || t->shape(0) != L || t->shape(1) != D / 2 ||
        !mx::issubdtype(t->dtype(), mx::floating)) {
      throw std::invalid_argument(
          op + " cos and sin must be floating [L, D / 2] = [" +
          std::to_string(L) + ", " + std::to_string(D / 2) + "].");
    }
  }

  // The weights and tables are small: cast to f32 and pack them. q and k
  // stay as they are; the kernel reads them through their strides.
  auto f32 = [&](const mx::array& a) {
    return mx::contiguous(mx::astype(a, mx::float32, s), false, s);
  };
  auto qw_c = f32(q_w);
  auto kw_c = f32(k_w);
  auto cos_c = f32(cos);
  auto sin_c = f32(sin);

  if (s.device.type == mx::Device::DeviceType::cpu || !qknr_kernel_dim(D)) {
    return {
        qknr_compose(q, qw_c, cos_c, sin_c, eps, s),
        qknr_compose(k, kw_c, cos_c, sin_c, eps, s)};
  }
  mx::Shape shape = {B, H, L, D};
  auto outs = mx::array::make_arrays(
      {shape, shape},
      {dt, dt},
      std::make_shared<KQuantQkNormRope>(s, eps),
      {std::move(q),
       std::move(k),
       std::move(qw_c),
       std::move(kw_c),
       std::move(cos_c),
       std::move(sin_c)});
  return {outs[0], outs[1]};
}

} // namespace mlx_kquant
