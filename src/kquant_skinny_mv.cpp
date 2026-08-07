// Skinny matmul: y = x @ w.T for token widths 1..16 against small-N,
// large-K nn.Linear-layout weights. MLX's steel GEMM leaves the GEMV fast
// path at M >= 2 and runs these shapes far below their bytes (router
// gates, indexer weight projections, hyper-connection mixes at
// speculative verify widths); the kernel keeps the GEMV shape with all M
// row accumulators in registers (see kq_skinny_mv.h). The CPU eval
// mirrors the kernel's f32-accumulate / one-round-at-write semantics.
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

#ifdef _METAL_

void KQuantSkinnyMV::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& x = inputs[0];
  const auto& w = inputs[1];
  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = w.shape(0);
  int T = int(x.size() / (int64_t(M) * K));

  std::string kname = "kq_skinny_mv_" + kq_type_string(x.dtype()) + "_" +
      kq_type_string(w.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(x, 0);
  ce.set_input_array(w, 1);
  ce.set_output_array(out, 2);
  ce.set_bytes(M, 3);
  ce.set_bytes(N, 4);
  ce.set_bytes(K, 5);
  MTL::Size group_dims(32 * 8, 1, 1);
  MTL::Size grid_dims((N + 7) / 8, T, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantSkinnyMV::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.skinny_matmul] requires Metal.");
}

#endif

namespace {

// Dispatch a functor over the admitted (x dtype, w dtype) pairs.
template <typename F>
void skinny_cpu_dispatch(mx::Dtype xt, mx::Dtype wt, F&& run) {
  auto with_x = [&](auto* xtag) {
    using XT = std::remove_pointer_t<decltype(xtag)>;
    if (wt == mx::float32) {
      run(static_cast<XT*>(nullptr), static_cast<float*>(nullptr));
    } else {
      run(static_cast<XT*>(nullptr), static_cast<XT*>(nullptr));
    }
  };
  if (xt == mx::float16) {
    with_x(static_cast<mx::float16_t*>(nullptr));
  } else if (xt == mx::bfloat16) {
    with_x(static_cast<mx::bfloat16_t*>(nullptr));
  } else {
    with_x(static_cast<float*>(nullptr));
  }
}

} // namespace

void KQuantSkinnyMV::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& x = inputs[0];
  const auto& w = inputs[1];

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_output_array(out);
  encoder.dispatch([x = mx::array::unsafe_weak_copy(x),
                    w = mx::array::unsafe_weak_copy(w),
                    out = mx::array::unsafe_weak_copy(out)]() mutable {
    const int K = x.shape(-1);
    const int M = x.shape(-2);
    const int N = w.shape(0);
    const int64_t T = x.size() / (int64_t(M) * K);
    skinny_cpu_dispatch(x.dtype(), w.dtype(), [&](auto* xtag, auto* wtag) {
      using XT = std::remove_pointer_t<decltype(xtag)>;
      using WT = std::remove_pointer_t<decltype(wtag)>;
      const bool out_f32 =
          std::is_same_v<XT, float> || std::is_same_v<WT, float>;
      const XT* xp = x.data<XT>();
      const WT* wp = w.data<WT>();
      for (int64_t t = 0; t < T; t++) {
        for (int m = 0; m < M; m++) {
          const XT* xrow = xp + (t * M + m) * K;
          for (int n = 0; n < N; n++) {
            const WT* wrow = wp + int64_t(n) * K;
            float acc = 0;
            for (int k = 0; k < K; k++) {
              acc += static_cast<float>(xrow[k]) * static_cast<float>(wrow[k]);
            }
            const int64_t o = (t * M + m) * N + n;
            if (out_f32) {
              out.data<float>()[o] = acc;
            } else {
              out.data<XT>()[o] = static_cast<XT>(acc);
            }
          }
        }
      }
    });
  });
}

bool KQuantSkinnyMV::is_equivalent(const mx::Primitive&) const {
  return true;
}

std::vector<mx::Shape> KQuantSkinnyMV::output_shapes(
    const std::vector<mx::array>& inputs) {
  auto shape = inputs[0].shape();
  shape.back() = inputs[1].shape(0);
  return {shape};
}

mx::array skinny_matmul(mx::array x, mx::array w, mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.skinny_matmul]";
  if (x.ndim() < 2) {
    throw std::invalid_argument(
        std::string(op) + " x must have at least 2 axes.");
  }
  if (w.ndim() != 2) {
    throw std::invalid_argument(std::string(op) + " w must be 2-D [N, K].");
  }
  int K = x.shape(-1);
  int M = x.shape(-2);
  if (M < 1 || M > 16) {
    throw std::invalid_argument(
        std::string(op) + " x rows (second-to-last axis) must be 1..16.");
  }
  if (K % 4 != 0 || w.shape(1) != K) {
    throw std::invalid_argument(
        std::string(op) + " K must match w and be a multiple of 4.");
  }
  auto xt = x.dtype();
  auto wt = w.dtype();
  if (xt != mx::float16 && xt != mx::bfloat16 && xt != mx::float32) {
    throw std::invalid_argument(
        std::string(op) + " x must be float16, bfloat16, or float32.");
  }
  if (wt != xt && wt != mx::float32) {
    throw std::invalid_argument(
        std::string(op) + " w dtype must match x or be float32.");
  }
  auto out_dtype = (xt == mx::float32 || wt == mx::float32) ? mx::float32 : xt;
  // Unconditional: flags() on a lazy array are not yet valid, so a
  // build-time row_contiguous check can skip a needed copy. Contiguous is
  // a zero-copy passthrough at eval when the input already is.
  auto x_c = mx::contiguous(x, false, s);
  auto w_c = mx::contiguous(w, false, s);

  auto shape = x.shape();
  shape.back() = w.shape(0);
  return mx::array(
      std::move(shape),
      out_dtype,
      std::make_shared<KQuantSkinnyMV>(s),
      {std::move(x_c), std::move(w_c)});
}

} // namespace mlx_kquant
