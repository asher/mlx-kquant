// Chunked KDA (per-key-channel gated delta rule) prefill. The Metal path
// runs the NAX tile kernel in kq_kda_chunk_nax.h, one threadgroup per
// (batch, head) with the state resident across 32-token chunks and takes
// the within-chunk cumulative log gate; the CPU path is the token-sequential
// recurrence in fp32 on the per-token log gate.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel, kq_is_nax_available
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

constexpr int kKdaChunk = 32;
constexpr int kKdaDim = 128;

template <typename T>
void kda_chunk_cpu_impl(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& lg,
    const mx::array& beta,
    const mx::array& state,
    mx::array& y,
    mx::array& state_out) {
  const int B = q.shape(0);
  const int Tn = q.shape(1);
  const int H = q.shape(2);
  const int D = q.shape(3);
  const T* qp = q.data<T>();
  const T* kp = k.data<T>();
  const T* vp = v.data<T>();
  const float* gp = lg.data<float>();
  const float* bp = beta.data<float>();
  const float* sp = state.data<float>();
  T* yp = y.data<T>();
  float* so = state_out.data<float>();
  std::vector<float> S(static_cast<size_t>(D) * D);
  std::vector<float> kk(D), qq(D), gg(D);
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const size_t sbase = (static_cast<size_t>(b) * H + h) * D * D;
      for (size_t i = 0; i < S.size(); ++i) {
        S[i] = sp[sbase + i];
      }
      for (int t = 0; t < Tn; ++t) {
        const size_t row = (static_cast<size_t>(b) * Tn + t) * H + h;
        const size_t off = row * D;
        for (int d = 0; d < D; ++d) {
          kk[d] = static_cast<float>(kp[off + d]);
          qq[d] = static_cast<float>(qp[off + d]);
          gg[d] = std::exp(gp[off + d]);
        }
        const float bt = bp[row];
        for (int dv = 0; dv < D; ++dv) {
          float* s = S.data() + static_cast<size_t>(dv) * D;
          float kv = 0.0f;
          for (int d = 0; d < D; ++d) {
            s[d] *= gg[d];
            kv += s[d] * kk[d];
          }
          const float delta = (static_cast<float>(vp[off + dv]) - kv) * bt;
          float o = 0.0f;
          for (int d = 0; d < D; ++d) {
            s[d] += kk[d] * delta;
            o += s[d] * qq[d];
          }
          yp[off + dv] = static_cast<T>(o);
        }
      }
      for (size_t i = 0; i < S.size(); ++i) {
        so[sbase + i] = S[i];
      }
    }
  }
}

} // namespace

void KQuantKdaChunk::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& y = outputs[0];
  auto& so = outputs[1];
  y.set_data(mx::allocator::malloc(y.nbytes()));
  so.set_data(mx::allocator::malloc(so.nbytes()));
  auto& encoder = mx::cpu::get_command_encoder(stream());
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(y);
  encoder.set_output_array(so);
  encoder.dispatch([q = mx::array::unsafe_weak_copy(inputs[0]),
                    k = mx::array::unsafe_weak_copy(inputs[1]),
                    v = mx::array::unsafe_weak_copy(inputs[2]),
                    lg = mx::array::unsafe_weak_copy(inputs[3]),
                    beta = mx::array::unsafe_weak_copy(inputs[4]),
                    state = mx::array::unsafe_weak_copy(inputs[5]),
                    y = mx::array::unsafe_weak_copy(y),
                    so = mx::array::unsafe_weak_copy(so)]() mutable {
    switch (q.dtype()) {
      case mx::float16:
        kda_chunk_cpu_impl<mx::float16_t>(q, k, v, lg, beta, state, y, so);
        break;
      case mx::bfloat16:
        kda_chunk_cpu_impl<mx::bfloat16_t>(q, k, v, lg, beta, state, y, so);
        break;
      case mx::float32:
        kda_chunk_cpu_impl<float>(q, k, v, lg, beta, state, y, so);
        break;
      default:
        throw std::runtime_error(
            "[mlx_kquant.kda_chunk] unsupported dtype on the CPU path.");
    }
  });
}

#ifdef _METAL_

void KQuantKdaChunk::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  if (!kq_is_nax_available()) {
    throw std::runtime_error(
        "[mlx_kquant.kda_chunk] requires tensor-op (NAX) hardware; check "
        "mlx_kquant.nax_available() and use the sequential kernel elsewhere.");
  }
  auto& y = outputs[0];
  auto& so = outputs[1];
  y.set_data(mx::allocator::malloc(y.nbytes()));
  so.set_data(mx::allocator::malloc(so.nbytes()));

  const auto& q = inputs[0];
  const int B = q.shape(0);
  const int T = q.shape(1);
  const int H = q.shape(2);
  if (q.dtype() != mx::float16 && q.dtype() != mx::bfloat16) {
    throw std::runtime_error(
        "[mlx_kquant.kda_chunk] the Metal kernel takes float16 or bfloat16 "
        "q, k and v.");
  }

  std::string kname = "kq_kda_chunk_nax_" + kq_type_string(q.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  for (int i = 0; i < 6; ++i) {
    ce.set_input_array(inputs[i], i);
  }
  ce.set_output_array(y, 6);
  ce.set_output_array(so, 7);
  ce.set_bytes(T, 8);
  ce.set_bytes(H, 9);
  MTL::Size group_dims(256, 1, 1);
  MTL::Size grid_dims(B * H, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantKdaChunk::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.kda_chunk] requires a Metal build.");
}

#endif

std::vector<mx::Shape> KQuantKdaChunk::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape(), inputs[5].shape()};
}

bool KQuantKdaChunk::is_equivalent(const mx::Primitive&) const {
  return true;
}

std::vector<mx::array> kda_chunk(
    mx::array q,
    mx::array k,
    mx::array v,
    mx::array log_g,
    mx::array beta,
    mx::array state,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.kda_chunk]";
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || log_g.ndim() != 4) {
    throw std::invalid_argument(
        std::string(op) + " q, k, v and log_g must be [B, T, H, D].");
  }
  const int B = q.shape(0);
  const int T = q.shape(1);
  const int H = q.shape(2);
  const int D = q.shape(3);
  if (D != kKdaDim) {
    throw std::invalid_argument(
        std::string(op) + " only head_dim 128 is supported.");
  }
  if (k.shape() != q.shape() || v.shape() != q.shape() ||
      log_g.shape() != q.shape()) {
    throw std::invalid_argument(
        std::string(op) + " q, k, v and log_g must share one shape.");
  }
  if (beta.ndim() != 3 || beta.shape(0) != B || beta.shape(1) != T ||
      beta.shape(2) != H) {
    throw std::invalid_argument(std::string(op) + " beta must be [B, T, H].");
  }
  if (state.ndim() != 4 || state.shape(0) != B || state.shape(1) != H ||
      state.shape(2) != D || state.shape(3) != D) {
    throw std::invalid_argument(
        std::string(op) + " state must be [B, H, D, D].");
  }
  auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16 && dt != mx::float32) {
    throw std::invalid_argument(
        std::string(op) + " q must be float16, bfloat16 or float32.");
  }
  if (k.dtype() != dt || v.dtype() != dt) {
    throw std::invalid_argument(
        std::string(op) + " q, k and v must share a dtype.");
  }
  if (T < 1) {
    throw std::invalid_argument(std::string(op) + " T must be positive.");
  }

  // Pad the sequence to whole chunks: zero k, v and beta leave the state
  // untouched and zero log g carries the decay through unchanged.
  const int Tp = ((T + kKdaChunk - 1) / kKdaChunk) * kKdaChunk;
  auto padt = [&](mx::array a) {
    if (Tp == T) {
      return a;
    }
    return mx::pad(
        a, {1}, {0}, {Tp - T}, mx::array(0, a.dtype()), "constant", s);
  };
  auto q_c = mx::contiguous(padt(q), false, s);
  auto k_c = mx::contiguous(padt(k), false, s);
  auto v_c = mx::contiguous(padt(v), false, s);
  auto lg = padt(mx::astype(log_g, mx::float32, s));
  // The Metal kernel takes the cumulative log gate within each chunk; the
  // CPU recurrence takes the per-token log gate.
  auto gam = lg;
  if (s.device == mx::Device::gpu) {
    gam = mx::reshape(lg, {B, Tp / kKdaChunk, kKdaChunk, H, D}, s);
    gam = mx::cumsum(gam, 2, false, true, s);
    gam = mx::reshape(gam, {B, Tp, H, D}, s);
  }
  gam = mx::contiguous(gam, false, s);
  auto beta_c =
      mx::contiguous(padt(mx::astype(beta, mx::float32, s)), false, s);
  auto state_c = mx::contiguous(mx::astype(state, mx::float32, s), false, s);

  auto prim = std::make_shared<KQuantKdaChunk>(s);
  auto outs = mx::array::make_arrays(
      {q_c.shape(), state_c.shape()},
      {dt, mx::float32},
      prim,
      {q_c, k_c, v_c, gam, beta_c, state_c});
  auto y = outs[0];
  if (Tp != T) {
    y = mx::slice(y, {0, 0, 0, 0}, {B, T, H, D}, {1, 1, 1, 1}, s);
  }
  return {y, outs[1]};
}

} // namespace mlx_kquant
