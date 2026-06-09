// KQuantGatherQMM primitive: mixture-of-experts quantized matmul. GPU dispatch
// ports the KQuant arm of mlx/backend/metal/quantized.cpp:GatherQMM::eval_gpu
// (:1715-1854) plus its gather leaf kernels (gather_qmm / gather_qmm_nax /
// gather_qmv) at :652-1163, with these extension-specific changes:
//   * kernels fetched from OUR bundled metallib via kq_get_kernel;
//   * row-contiguity guaranteed by the op (contiguous_copy_gpu is unexported);
//   * kernel-name type tokens via kq_type_string (not type_to_name);
//   * is_nax_available() replicated as kq_is_nax_available (hidden symbol);
//   * kquant carries no biases, so the bias buffer is dropped throughout.
//
// The gather_qmm_rhs fast path (quantized.cpp:1748-1768) IS implemented here as
// gather_qmm_rhs_nax — it is the only function-constant kernel (align_M/N/K at
// constant ids 200/201/202). It requires right_sorted_ == true, which holds
// when lhs_indices is defaulted AND sorted_indices is requested. mlx-lm's
// SwitchGLU sorts tokens by expert (do_sort when indices.size>=64) and passes
// rhs_indices only, so right_sorted_ == do_sort: MoE PREFILL takes this sorted
// per-expert GEMM (≈6-8x faster than B separate gather_qmv vector-matmuls),
// while decode (top_k<64 -> no sort -> B<16) falls through to gather_qmv,
// exactly as the fork does. An earlier revision deferred this on a swapped
// left/right_sorted premise, which left MoE prefill ~3.3x slower than the fork.
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_internal.h"

#include "mlx/allocator.h"

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
// array / Device / CommandEncoder and the shared dispatch helpers come from
// kquant_metal_internal.h.

// quantized.cpp:652-758 (kquant path, no biases).
void gather_qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const array& lhs_indices,
    const array& rhs_indices,
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
  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) +
          (transpose ? "gather_qmm_t_nax_" : "gather_qmm_n_nax_"),
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
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_input_array(lhs_indices, c++);
  ce.set_input_array(rhs_indices, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  c = add_strides_and_shapes(ce, false, x, w, scales, c);
  add_gather_strides_and_shapes(ce, lhs_indices, rhs_indices, c);

  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// quantized.cpp:986-1094 (kquant path, no biases).
void gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const array& lhs_indices,
    const array& rhs_indices,
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
  // enable_tf32 dropped (always short-circuited; op promotes f32 x -> bf16).
  if (kq_is_nax_available() && transpose && (K % 64 == 0) &&
      (x.dtype() != mx::float32) && codec_has_matmul(kquant_type)) {
    return gather_qmm_nax(
        x,
        w,
        scales,
        lhs_indices,
        rhs_indices,
        out,
        transpose,
        group_size,
        bits,
        M,
        N,
        K,
        d,
        s,
        kquant_type);
  }

  int B = out.size() / M / N;
  int wm = 2, wn = 2, bm = 32, bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  bool aligned = N % 32 == 0;
  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) +
          (transpose ? "gather_qmm_t_" : "gather_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_input_array(lhs_indices, c++);
  ce.set_input_array(rhs_indices, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  c = add_strides_and_shapes(ce, false, x, w, scales, c);
  add_gather_strides_and_shapes(ce, lhs_indices, rhs_indices, c);

  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// quantized.cpp:1096-1163 (kquant path, no biases).
void gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const array& lhs_indices,
    const array& rhs_indices,
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
  int k_align = qmv_fast_k_align();
  bool fast = (N % bn == 0) && (K % k_align == 0);
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) +
          (fast ? "gather_qmv_fast_" : "gather_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_input_array(lhs_indices, c++);
  ce.set_input_array(rhs_indices, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  c = add_strides_and_shapes(ce, false, x, w, scales, c);
  add_gather_strides_and_shapes(ce, lhs_indices, rhs_indices, c);

  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// quantized.cpp:1225-1360 (kquant path: NAX-only, no biases). The sorted-rhs
// fast path: x rows are pre-sorted by expert (lhs_indices defaulted), so a
// single batched GEMM walks contiguous per-expert row blocks, switching the
// weight matrix per row-block from the sorted `indices`. M here is the TOTAL
// row count (x.size()/K), NOT x.shape(-2)==1. Unlike the other gather leaves
// this passes no index strides — it requires row-contiguous x / indices (the
// caller guards this) and bakes align_M/N/K into func consts 200/201/202.
// kquant has no non-NAX gather_qmm_rhs kernel, so this is only ever reached
// when the NAX gate holds (mirrors gather_qmm_rhs()'s unconditional NAX route).
void gather_qmm_rhs_nax(
    const array& x,
    const array& w,
    const array& scales,
    const array& indices,
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
  int bm = 64, bn = 64, bk = 64, wm = 2, wn = 2;
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) +
          (transpose ? "gather_qmm_rhs_nax_nt_" : "gather_qmm_rhs_nax_nn_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn);

  mx::metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  std::string hash_name;
  hash_name.reserve(128);
  mx::concatenate(
      hash_name,
      kname,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  auto kernel = kq_get_kernel(d, kname, hash_name, func_consts);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);

  int c = 0;
  ce.set_input_array(x, c++);
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(indices, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(M, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(K, c++);

  ce.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace

#endif // _METAL_

std::vector<mx::Shape> KQuantGatherQMM::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& lhs_indices = inputs[3];
  const KQuantCodec* codec = codec_by_name(kquant_type_);
  int weights_per_row =
      (w.shape(-1) / codec->bytes_per_block) * codec->weights_per_block;
  int N = transpose_ ? w.shape(-2) : weights_per_row;
  auto shape = lhs_indices.shape();
  shape.push_back(x.shape(-2));
  shape.push_back(N);
  return {shape};
}

bool KQuantGatherQMM::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMM&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_ && transpose_ == o.transpose_ &&
      left_sorted_ == o.left_sorted_ && right_sorted_ == o.right_sorted_;
}

void KQuantGatherQMM::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant] gather_qmm has no CPU implementation; run on the GPU stream "
      "(the default device).");
}

#ifdef _METAL_

void KQuantGatherQMM::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // inputs are row-contiguous (ensured by the op):
  //   x (float), w (uint8), scales, lhs_indices (uint32), rhs_indices (uint32).
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& scales = inputs[2];
  const auto& lhs_indices = inputs[3];
  const auto& rhs_indices = inputs[4];

  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;
  int E = w.size() / w.shape(-1) / w.shape(-2);
  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;

  // Sorted-rhs fast path (mirrors quantized.cpp:1744-1768). When the expert
  // (rhs) indices are sorted and lhs was defaulted (right_sorted_), and the
  // batch is large enough to amortize a per-expert GEMM, route to the NAX
  // gather_qmm_rhs kernel instead of B separate gather_qmv vector-matmuls.
  // kquant has no non-NAX rhs kernel, so the NAX gate must hold. We
  // additionally require x and the rhs indices to already be row-contiguous
  // with one x row per output row (x.size()/K == B): the op keeps broadcast
  // index strides for the strided leaves, whereas this kernel takes no strides.
  // The SwitchGLU sort path satisfies all of this; any case that does not falls
  // through to the (correct, slower) gather_qmv / gather_qmm leaves below.
  bool kquant_rhs_ok = kq_is_nax_available() && transpose_ && (K % 64 == 0) &&
      (x.dtype() != mx::float32) && codec_has_matmul(kquant_type_);
  if (M == 1 && B >= 16 && right_sorted_ && (B / E >= 4) && kquant_rhs_ok &&
      x.flags().row_contiguous && rhs_indices.flags().row_contiguous &&
      (x.size() / K == static_cast<size_t>(B))) {
    gather_qmm_rhs_nax(
        x,
        w,
        scales,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        /*M=*/static_cast<int>(x.size() / K),
        N,
        K,
        d,
        s,
        kquant_type_);
    return;
  }

  if (M >= vector_limit) {
    gather_qmm(
        x,
        w,
        scales,
        lhs_indices,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        kquant_type_);
    return;
  }

  if (transpose_) {
    gather_qmv(
        x,
        w,
        scales,
        lhs_indices,
        rhs_indices,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        kquant_type_);
    return;
  }

  // KQuant has no dedicated gather_qvm kernel; route through gather_qmm_n.
  gather_qmm(
      x,
      w,
      scales,
      lhs_indices,
      rhs_indices,
      out,
      transpose_,
      group_size_,
      bits_,
      M,
      N,
      K,
      d,
      s,
      kquant_type_);
}

#else

void KQuantGatherQMM::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant] gather_qmm has no GPU implementation.");
}

#endif // _METAL_

} // namespace mlx_kquant
