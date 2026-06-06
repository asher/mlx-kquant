// KQuantMatmul primitive: x @ dequant(w). GPU dispatch ports the KQuant arm of
// mlx/backend/metal/quantized.cpp:QuantizedMatmul::eval_gpu (:1566-1713) plus
// its leaf kernels (qmm / qmm_nax / qvm / qmv) and dispatch helpers, with the
// extension-specific changes established in M1:
//   * kernels fetched from OUR bundled metallib via d.get_kernel(name, lib);
//   * row-contiguity guaranteed by the op (contiguous_copy_gpu is unexported);
//   * kernel-name type tokens via kq_type_string (not type_to_name).
// Two further deviations, both correctness-preserving:
//   * is_nax_available() is a hidden symbol -> replicated as kq_is_nax_available;
//   * the split-k paths (qmm_splitk / qvm_split_k) need the unexported
//     strided_reduce_general_dispatch -> gated off; plain qmm/qvm run instead.
//   * kquant never carries biases, so the bias buffer is dropped throughout.
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_internal.h"

#include "mlx/allocator.h"
#include "mlx/utils.h" // env::enable_tf32

#ifdef _METAL_
#include "kquant_metal_internal.h" // shared dispatch helpers
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h" // concatenate
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

#ifdef _METAL_

namespace {

using mx::Stream;
// array / Device / CommandEncoder and the shared dispatch helpers
// (kq_is_nax_available, kquant_qmv_bn, qmv_fast_k_align, codec_has_matmul,
// get_qmv_batch_limit, add_strides_and_shapes, kq_get_kernel) come from
// kquant_metal_internal.h.

// quantized.cpp:545-650 (kquant path, no biases).
void qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  int B = out.size() / M / N;
  int wm = 2, wn = 2, bm = 64, bn = 64, bk = 64;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  bool aligned = N % bn == 0;
  bool batched = B > 1;
  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + (transpose ? "qmm_t_nax_" : "qmm_n_nax_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  add_strides_and_shapes(ce, B <= 1, x, w, scales, c);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// quantized.cpp:760-866 (kquant path, no biases). qmm_splitk gated off upstream.
void qmm(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  // Upstream gate is `(env::enable_tf32() || x.dtype() != float32)`, but the op
  // promotes float32 x to bfloat16 before constructing this primitive (just as
  // mlx-core does), so x.dtype() is never float32 here — the enable_tf32 branch
  // is always short-circuited. Dropping it avoids the unexported env::get_var.
  if (kq_is_nax_available() && transpose && (K % 64 == 0) &&
      (x.dtype() != mx::float32) && codec_has_matmul(kquant_type)) {
    return qmm_nax(
        x, w, scales, out, transpose, group_size, bits, M, N, K, d, s,
        kquant_type);
  }

  int B = out.size() / M / N;
  int wm = 2, wn = 2;
  int bm = 64;
  int bn = transpose ? 64 : 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  bool aligned = N % bn == 0;
  bool batched = B > 1;
  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + (transpose ? "qmm_t_" : "qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  add_strides_and_shapes(ce, B <= 1, x, w, scales, c);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// quantized.cpp:490-543 (kquant path, no biases).
void qvm(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  int B = out.size() / M / N;
  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + "qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  add_strides_and_shapes(ce, B <= 1, x, w, scales, c++);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// quantized.cpp:294-358 (kquant path, no biases).
void qmv(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  int B = out.size() / M / N;
  int bn = kquant_qmv_bn(kquant_type);
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string type_string = kq_type_string(x.dtype());
  bool fast = (N % bn == 0) && (K % qmv_fast_k_align() == 0);
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + (fast ? "qmv_fast_" : "qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  add_strides_and_shapes(ce, B <= 1, x, w, scales, c);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// quantized.cpp:1516-1561. The qmv_quad branch (K==64/128) is unreachable for
// kquant — eval_gpu throws for that case first — so this always routes to qmv.
void dispatch_qmv(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  qmv(x, w, scales, out, group_size, bits, M, N, K, d, s, kquant_type);
}

} // namespace

#endif // _METAL_

std::vector<mx::Shape> KQuantMatmul::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const KQuantCodec* codec = codec_by_name(kquant_type_);
  int weights_per_row =
      (w.shape(-1) / codec->bytes_per_block) * codec->weights_per_block;
  int N = transpose_ ? w.shape(-2) : weights_per_row;
  auto shape = x.shape();
  shape.back() = N;
  return {shape};
}

bool KQuantMatmul::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMatmul&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_ && transpose_ == o.transpose_;
}

void KQuantMatmul::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant] quantized_matmul has no CPU implementation; run on the GPU "
      "stream (the default device).");
}

#ifdef _METAL_

void KQuantMatmul::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // inputs are row-contiguous (ensured by the op): x, w (uint8), scales.
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& scales = inputs[2];

  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? static_cast<int>(x.size()) / K : x.shape(-2);
  int N = out.shape(-1);

  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;

  // KQuant special cases (quantized.cpp:1591-1615).
  if (!transpose_ && M < vector_limit) {
    qmm(x, w, scales, out, transpose_, group_size_, bits_, M, N, K, d, s,
        kquant_type_);
    return;
  }
  // quantized.cpp:1610 routes K in {64,128} (M<vector_limit, transpose) to a
  // quad-optimized qmv kernel; no kquant qmv_quad kernel exists (the fork throws
  // here too). Plain qmv is correct for any K that is a multiple of the codec
  // group size — K-quants (gs=256) can never reach K=64/128, only the legacy
  // gs=32 codecs can, and only on weights whose input dim is exactly 64/128
  // (none in standard transformers). Fall through to dispatch_qmv below rather
  // than throw; qmv_quad would be a perf-only kernel for an essentially-dead
  // path. See docs/progress-extension-refactor.log qmv_quad survey.

  if (M >= vector_limit) {
    // Upstream uses qmm_splitk for (transpose && B==1); gated off here (it needs
    // strided_reduce_general_dispatch). Plain qmm is correct, just less parallel.
    qmm(x, w, scales, out, transpose_, group_size_, bits_, M, N, K, d, s,
        kquant_type_);
    return;
  }

  if (transpose_) {
    dispatch_qmv(x, w, scales, out, group_size_, bits_, M, N, K, d, s,
                 kquant_type_);
    return;
  }

  // Upstream routes K>=1024 to qvm_split_k; gated off here. Plain qvm is correct.
  qvm(x, w, scales, out, group_size_, bits_, M, N, K, d, s, kquant_type_);
}

#else

void KQuantMatmul::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant] quantized_matmul has no GPU implementation.");
}

#endif // _METAL_

} // namespace mlx_kquant
