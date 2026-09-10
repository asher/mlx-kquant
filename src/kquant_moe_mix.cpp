// Fused unsort and score mix for sorted-prefill MoE (see gather_mix): the
// Metal kernel in kq_moe_mix.h, and a CPU loop with the same f32
// accumulate and one round at the write.

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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

template <typename T>
void gather_mix_cpu_impl(
    const mx::array& y,
    const mx::array& inv_order,
    const mx::array& scores,
    mx::array& out) {
  const int N = y.shape(1);
  const int Tn = scores.shape(0);
  const int K = scores.shape(1);
  const T* yp = y.data<T>();
  const uint32_t* io = inv_order.data<uint32_t>();
  const float* sc = scores.data<float>();
  T* op = out.data<T>();
  std::vector<float> acc(N);
  for (int t = 0; t < Tn; ++t) {
    for (int c = 0; c < N; ++c) {
      acc[c] = 0.0f;
    }
    for (int s = 0; s < K; ++s) {
      const float w = sc[static_cast<size_t>(t) * K + s];
      const T* row =
          yp + static_cast<size_t>(io[static_cast<size_t>(t) * K + s]) * N;
      for (int c = 0; c < N; ++c) {
        acc[c] = std::fma(w, static_cast<float>(row[c]), acc[c]);
      }
    }
    T* orow = op + static_cast<size_t>(t) * N;
    for (int c = 0; c < N; ++c) {
      orow[c] = static_cast<T>(acc[c]);
    }
  }
}

} // namespace

void KQuantGatherMix::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  auto& encoder = mx::cpu::get_command_encoder(stream());
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(out);
  encoder.dispatch([y = mx::array::unsafe_weak_copy(inputs[0]),
                    io = mx::array::unsafe_weak_copy(inputs[1]),
                    sc = mx::array::unsafe_weak_copy(inputs[2]),
                    out = mx::array::unsafe_weak_copy(out)]() mutable {
    if (y.dtype() == mx::float16) {
      gather_mix_cpu_impl<mx::float16_t>(y, io, sc, out);
    } else {
      gather_mix_cpu_impl<mx::bfloat16_t>(y, io, sc, out);
    }
  });
}

#ifdef _METAL_

void KQuantGatherMix::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  const auto& y = inputs[0];
  const int N = y.shape(1);
  const int Tn = inputs[2].shape(0);
  const int K = inputs[2].shape(1);
  std::string kname = "kq_moe_mix_" + kq_type_string(y.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(inputs[0], 0);
  ce.set_input_array(inputs[1], 1);
  ce.set_input_array(inputs[2], 2);
  ce.set_output_array(out, 3);
  ce.set_bytes(N, 4);
  ce.set_bytes(K, 5);
  int threads = N / 4;
  threads = ((threads + 31) / 32) * 32;
  if (threads > 256) {
    threads = 256;
  }
  MTL::Size group_dims(threads, 1, 1);
  MTL::Size grid_dims(Tn, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantGatherMix::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.gather_mix] requires a Metal build.");
}

#endif

mx::array gather_mix(
    mx::array y,
    mx::array inv_order,
    mx::array scores,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.gather_mix]";
  if (y.ndim() != 2) {
    throw std::invalid_argument(std::string(op) + " y must be [rows, N].");
  }
  const int rows = y.shape(0);
  const int N = y.shape(1);
  if (N % 4 != 0) {
    throw std::invalid_argument(
        std::string(op) + " N must be a multiple of 4.");
  }
  auto dt = y.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " y must be float16 or bfloat16.");
  }
  if (scores.ndim() != 2) {
    throw std::invalid_argument(std::string(op) + " scores must be [T, k].");
  }
  const int Tn = scores.shape(0);
  const int K = scores.shape(1);
  if (inv_order.size() != static_cast<size_t>(Tn) * K ||
      static_cast<int>(inv_order.size()) != rows) {
    throw std::invalid_argument(
        std::string(op) +
        " inv_order must hold T * k entries, one per row of y.");
  }
  if (inv_order.dtype() != mx::uint32 && inv_order.dtype() != mx::int32) {
    throw std::invalid_argument(
        std::string(op) + " inv_order must be uint32 or int32.");
  }
  auto y_c = mx::contiguous(y, false, s);
  auto io = mx::contiguous(
      mx::reshape(mx::astype(inv_order, mx::uint32, s), {rows}, s), false, s);
  auto sc = mx::contiguous(mx::astype(scores, mx::float32, s), false, s);
  return mx::array(
      {Tn, N}, dt, std::make_shared<KQuantGatherMix>(s), {y_c, io, sc});
}

} // namespace mlx_kquant
