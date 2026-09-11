// KDA prefill ops: the chunked recurrence and the fused short conv.
//
// Chunked KDA (per-key-channel gated delta rule) prefill. The Metal path
// runs the NAX tile kernel in kq_kda_chunk_nax.h, one threadgroup per
// (batch, head) with the state resident across 32-token chunks; the CPU
// path is the token-sequential recurrence in fp32. Both take the per-token
// log gate.

#include <algorithm>
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

// gate_p is the fp32 log gate, or with `gated` the pre-activation in T
// with a_scale [H], dt_bias [H * D] and lb (see kda_chunk_gated). Rows at
// and past t_real are padding and leave the state untouched.
template <typename T>
void kda_chunk_cpu_impl(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& gate,
    const mx::array& beta,
    const mx::array& state,
    mx::array& y,
    mx::array& state_out,
    int t_real,
    bool gated,
    float lb,
    const float* asc,
    const float* dtb) {
  const int B = q.shape(0);
  const int Tn = std::min<int>(q.shape(1), t_real);
  const int Tp = q.shape(1);
  const int H = q.shape(2);
  const int D = q.shape(3);
  const T* qp = q.data<T>();
  const T* kp = k.data<T>();
  const T* vp = v.data<T>();
  const float* gp = gated ? nullptr : gate.data<float>();
  const T* ap = gated ? gate.data<T>() : nullptr;
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
        const size_t row = (static_cast<size_t>(b) * Tp + t) * H + h;
        const size_t off = row * D;
        for (int d = 0; d < D; ++d) {
          kk[d] = static_cast<float>(kp[off + d]);
          qq[d] = static_cast<float>(qp[off + d]);
          if (gated) {
            const float z =
                asc[h] * (static_cast<float>(ap[off + d]) + dtb[h * D + d]);
            gg[d] = std::exp(lb / (1.0f + std::exp(-z)));
          } else {
            gg[d] = std::exp(gp[off + d]);
          }
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
      for (int t = Tn; t < Tp; ++t) {
        const size_t off = ((static_cast<size_t>(b) * Tp + t) * H + h) * D;
        for (int dv = 0; dv < D; ++dv) {
          yp[off + dv] = static_cast<T>(0.0f);
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
  const bool gated = gated_;
  const int t_real = t_real_;
  const float lb = lb_;
  auto asc_a = gated ? inputs[6] : inputs[3];
  auto dtb_a = gated ? inputs[7] : inputs[3];
  encoder.dispatch([q = mx::array::unsafe_weak_copy(inputs[0]),
                    k = mx::array::unsafe_weak_copy(inputs[1]),
                    v = mx::array::unsafe_weak_copy(inputs[2]),
                    lg = mx::array::unsafe_weak_copy(inputs[3]),
                    beta = mx::array::unsafe_weak_copy(inputs[4]),
                    state = mx::array::unsafe_weak_copy(inputs[5]),
                    asc_a = mx::array::unsafe_weak_copy(asc_a),
                    dtb_a = mx::array::unsafe_weak_copy(dtb_a),
                    y = mx::array::unsafe_weak_copy(y),
                    so = mx::array::unsafe_weak_copy(so),
                    t_real,
                    gated,
                    lb]() mutable {
    const float* asc = gated ? asc_a.data<float>() : nullptr;
    const float* dtb = gated ? dtb_a.data<float>() : nullptr;
    switch (q.dtype()) {
      case mx::float16:
        kda_chunk_cpu_impl<mx::float16_t>(
            q, k, v, lg, beta, state, y, so, t_real, gated, lb, asc, dtb);
        break;
      case mx::bfloat16:
        kda_chunk_cpu_impl<mx::bfloat16_t>(
            q, k, v, lg, beta, state, y, so, t_real, gated, lb, asc, dtb);
        break;
      case mx::float32:
        kda_chunk_cpu_impl<float>(
            q, k, v, lg, beta, state, y, so, t_real, gated, lb, asc, dtb);
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

  std::string kname = std::string("kq_kda_chunk_nax_") +
      (gated_ ? "gated_" : "") + kq_type_string(q.dtype());
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
  // a_scale and dt_bias must be bound even on the log_g form; beta and
  // the state stand in.
  ce.set_input_array(gated_ ? inputs[6] : inputs[4], 10);
  ce.set_input_array(gated_ ? inputs[7] : inputs[5], 11);
  ce.set_bytes(lb_, 12);
  ce.set_bytes(t_real_, 13);
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

bool KQuantKdaChunk::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantKdaChunk&>(other);
  return t_real_ == o.t_real_ && gated_ == o.gated_ && lb_ == o.lb_;
}

namespace {

// Shared validation and padding for both gate forms; gate is log_g (fp32
// on the op, any float dtype in) or the pre-activation in the q dtype.
std::vector<mx::array> kda_chunk_impl(
    mx::array q,
    mx::array k,
    mx::array v,
    mx::array gate,
    mx::array beta,
    mx::array state,
    bool gated,
    mx::array a_scale,
    mx::array dt_bias,
    float lb,
    mx::Stream s) {
  const char* op =
      gated ? "[mlx_kquant.kda_chunk_gated]" : "[mlx_kquant.kda_chunk]";
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || gate.ndim() != 4) {
    throw std::invalid_argument(
        std::string(op) + " q, k, v and the gate must be [B, T, H, D].");
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
      gate.shape() != q.shape()) {
    throw std::invalid_argument(
        std::string(op) + " q, k, v and the gate must share one shape.");
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
  if (gated) {
    if (gate.dtype() != dt) {
      throw std::invalid_argument(
          std::string(op) + " the gate pre-activation must share the q dtype.");
    }
    if (a_scale.ndim() != 1 || a_scale.shape(0) != H) {
      throw std::invalid_argument(std::string(op) + " a_scale must be [H].");
    }
    if (dt_bias.size() != static_cast<size_t>(H) * D) {
      throw std::invalid_argument(
          std::string(op) + " dt_bias must hold H * 128 values.");
    }
  }
  if (T < 1) {
    throw std::invalid_argument(std::string(op) + " T must be positive.");
  }

  // Pad the sequence to whole chunks: zero k, v and beta leave the state
  // untouched; a zero log gate (or the gated kernel's T_real mask)
  // carries the decay through unchanged.
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
  auto g_c = gated
      ? mx::contiguous(padt(gate), false, s)
      : mx::contiguous(padt(mx::astype(gate, mx::float32, s)), false, s);
  auto beta_c =
      mx::contiguous(padt(mx::astype(beta, mx::float32, s)), false, s);
  auto state_c = mx::contiguous(mx::astype(state, mx::float32, s), false, s);
  std::vector<mx::array> ins = {q_c, k_c, v_c, g_c, beta_c, state_c};
  if (gated) {
    ins.push_back(
        mx::contiguous(mx::astype(a_scale, mx::float32, s), false, s));
    ins.push_back(mx::contiguous(
        mx::reshape(mx::astype(dt_bias, mx::float32, s), {H * D}, s),
        false,
        s));
  }

  auto prim = std::make_shared<KQuantKdaChunk>(s, T, gated, lb);
  auto outs = mx::array::make_arrays(
      {q_c.shape(), state_c.shape()}, {dt, mx::float32}, prim, ins);
  auto y = outs[0];
  if (Tp != T) {
    y = mx::slice(y, {0, 0, 0, 0}, {B, T, H, D}, {1, 1, 1, 1}, s);
  }
  return {y, outs[1]};
}

} // namespace

std::vector<mx::array> kda_chunk(
    mx::array q,
    mx::array k,
    mx::array v,
    mx::array log_g,
    mx::array beta,
    mx::array state,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  return kda_chunk_impl(
      q, k, v, log_g, beta, state, false, log_g, log_g, 0.0f, s);
}

std::vector<mx::array> kda_chunk_gated(
    mx::array q,
    mx::array k,
    mx::array v,
    mx::array a,
    mx::array a_scale,
    mx::array dt_bias,
    mx::array beta,
    mx::array state,
    float lb,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  return kda_chunk_impl(q, k, v, a, beta, state, true, a_scale, dt_bias, lb, s);
}

// ---- kda_conv -------------------------------------------------------------

namespace {

template <typename T>
void kda_conv_cpu_impl(
    const mx::array& x,
    const mx::array& state,
    const mx::array& w,
    mx::array& y,
    mx::array& state_out,
    int K,
    int D,
    float scale,
    float eps) {
  const int B = x.shape(0);
  const int Tn = x.shape(1);
  const int C = x.shape(2);
  const int NS = K - 1;
  const T* xp = x.data<T>();
  const T* sp = state.data<T>();
  const T* wp = w.data<T>();
  T* yp = y.data<T>();
  T* so = state_out.data<T>();
  std::vector<float> row(C);
  for (int b = 0; b < B; ++b) {
    const T* xb = xp + static_cast<size_t>(b) * Tn * C;
    const T* sb = sp + static_cast<size_t>(b) * NS * C;
    for (int t = 0; t < Tn; ++t) {
      for (int c = 0; c < C; ++c) {
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
          const int m = t - NS + j;
          const T* src = (m < 0) ? sb + static_cast<size_t>(NS + m) * C
                                 : xb + static_cast<size_t>(m) * C;
          acc = std::fma(
              static_cast<float>(wp[static_cast<size_t>(c) * K + j]),
              static_cast<float>(src[c]),
              acc);
        }
        row[c] = acc / (1.0f + std::exp(-acc));
      }
      if (scale != 0.0f) {
        for (int h0 = 0; h0 < C; h0 += D) {
          float ss = 0.0f;
          for (int c = h0; c < h0 + D; ++c) {
            ss = std::fma(row[c], row[c], ss);
          }
          const float f = scale / std::sqrt(ss / static_cast<float>(D) + eps);
          for (int c = h0; c < h0 + D; ++c) {
            row[c] *= f;
          }
        }
      }
      T* yrow = yp + (static_cast<size_t>(b) * Tn + t) * C;
      for (int c = 0; c < C; ++c) {
        yrow[c] = static_cast<T>(row[c]);
      }
    }
    T* sob = so + static_cast<size_t>(b) * NS * C;
    for (int j = 0; j < NS; ++j) {
      const int m = Tn - NS + j;
      const T* src = (m < 0) ? sb + static_cast<size_t>(NS + m) * C
                             : xb + static_cast<size_t>(m) * C;
      for (int c = 0; c < C; ++c) {
        sob[static_cast<size_t>(j) * C + c] = src[c];
      }
    }
  }
}

} // namespace

void KQuantKdaConv::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& y = outputs[0];
  auto& so = outputs[1];
  y.set_data(mx::allocator::malloc(y.nbytes()));
  so.set_data(mx::allocator::malloc(so.nbytes()));
  const int K = inputs[2].shape(1);
  auto& encoder = mx::cpu::get_command_encoder(stream());
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(y);
  encoder.set_output_array(so);
  encoder.dispatch([x = mx::array::unsafe_weak_copy(inputs[0]),
                    state = mx::array::unsafe_weak_copy(inputs[1]),
                    w = mx::array::unsafe_weak_copy(inputs[2]),
                    y = mx::array::unsafe_weak_copy(y),
                    so = mx::array::unsafe_weak_copy(so),
                    K,
                    D = head_dim_,
                    scale = scale_,
                    eps = eps_]() mutable {
    switch (x.dtype()) {
      case mx::float16:
        kda_conv_cpu_impl<mx::float16_t>(x, state, w, y, so, K, D, scale, eps);
        break;
      case mx::bfloat16:
        kda_conv_cpu_impl<mx::bfloat16_t>(x, state, w, y, so, K, D, scale, eps);
        break;
      default:
        throw std::runtime_error(
            "[mlx_kquant.kda_conv] unsupported dtype on the CPU path.");
    }
  });
}

#ifdef _METAL_

void KQuantKdaConv::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& y = outputs[0];
  auto& so = outputs[1];
  y.set_data(mx::allocator::malloc(y.nbytes()));
  so.set_data(mx::allocator::malloc(so.nbytes()));

  const auto& x = inputs[0];
  const int B = x.shape(0);
  const int T = x.shape(1);
  const int C = x.shape(2);
  const int K = inputs[2].shape(1);
  const int nrows = B * T * (C / head_dim_);

  std::string kname = "kq_kda_conv_" + kq_type_string(x.dtype()) + "_" +
      std::to_string(head_dim_ / 32);
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(inputs[0], 0);
  ce.set_input_array(inputs[1], 1);
  ce.set_input_array(inputs[2], 2);
  ce.set_output_array(y, 3);
  ce.set_output_array(so, 4);
  ce.set_bytes(T, 5);
  ce.set_bytes(C, 6);
  ce.set_bytes(K, 7);
  ce.set_bytes(scale_, 8);
  ce.set_bytes(eps_, 9);
  ce.set_bytes(nrows, 10);
  MTL::Size group_dims(256, 1, 1);
  MTL::Size grid_dims((nrows + 7) / 8, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantKdaConv::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.kda_conv] requires a Metal build.");
}

#endif

std::vector<mx::Shape> KQuantKdaConv::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape(), inputs[1].shape()};
}

bool KQuantKdaConv::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantKdaConv&>(other);
  return head_dim_ == o.head_dim_ && scale_ == o.scale_ && eps_ == o.eps_;
}

std::vector<mx::array> kda_conv(
    mx::array x,
    mx::array state,
    mx::array w,
    int head_dim,
    float scale,
    float eps,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.kda_conv]";
  if (x.ndim() != 3) {
    throw std::invalid_argument(std::string(op) + " x must be [B, T, C].");
  }
  const int B = x.shape(0);
  const int T = x.shape(1);
  const int C = x.shape(2);
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " x must be float16 or bfloat16.");
  }
  if (w.ndim() == 3 && w.shape(2) == 1) {
    w = mx::reshape(w, {w.shape(0), w.shape(1)}, s);
  }
  if (w.ndim() != 2 || w.shape(0) != C) {
    throw std::invalid_argument(
        std::string(op) + " w must be [C, K] or [C, K, 1].");
  }
  const int K = w.shape(1);
  if (K < 2 || K > 8) {
    throw std::invalid_argument(std::string(op) + " K must be 2 to 8 taps.");
  }
  if (state.ndim() != 3 || state.shape(0) != B || state.shape(1) != K - 1 ||
      state.shape(2) != C) {
    throw std::invalid_argument(
        std::string(op) + " state must be [B, K - 1, C].");
  }
  if (head_dim != 64 && head_dim != 128 && head_dim != 256) {
    throw std::invalid_argument(
        std::string(op) + " head_dim must be 64, 128 or 256.");
  }
  if (C % head_dim != 0) {
    throw std::invalid_argument(
        std::string(op) + " C must be a multiple of head_dim.");
  }
  if (T < 1) {
    throw std::invalid_argument(std::string(op) + " T must be positive.");
  }
  auto x_c = mx::contiguous(x, false, s);
  auto st_c = mx::contiguous(mx::astype(state, dt, s), false, s);
  auto w_c = mx::contiguous(mx::astype(w, dt, s), false, s);
  auto prim = std::make_shared<KQuantKdaConv>(s, head_dim, scale, eps);
  return mx::array::make_arrays(
      {x_c.shape(), st_c.shape()}, {dt, dt}, prim, {x_c, st_c, w_c});
}

} // namespace mlx_kquant
