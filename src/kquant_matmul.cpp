// KQuantMatmul primitive: x @ dequant(w). The GPU path dispatches the leaf
// kernels (qmm / qmm_nax / qvm / qmv) from the bundled metallib via
// d.get_kernel(name, lib); the op guarantees row-contiguity before dispatch and
// kernel-name type tokens come from kq_type_string. NAX (tensor-core)
// availability is probed via kq_is_nax_available. The split-k paths
// (qmm_splitk / qvm_split_k) are omitted - plain qmm/qvm produce identical
// results with less parallelism. KQuantMatmul itself never carries a bias (a
// separate elementwise add is fine off the decode-latency-critical path); the
// decode-only bias-fused fast path lives in the KQuantQmvBias primitive below
// (qmv_bias), which reuses this file's qmv dispatch helpers.
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_cpu_decode.h"
#include "kquant_internal.h"

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h" // elem_to_loc
#include "mlx/backend/cpu/encoder.h"
#include "mlx/types/half_types.h"
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

// NAX (tensor-core) GEMM dispatch for the quantized matmul (no biases).
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

// Tiled quantized GEMM (no biases). The split-k variant is omitted; plain qmm
// is correct with less parallelism.
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
  // The op promotes float32 x to bfloat16 before constructing this primitive,
  // so x.dtype() is never float32 here and the NAX path stays eligible on the
  // dtype axis with no tf32 gate.
  if (kq_is_nax_available() && transpose && (K % 64 == 0) &&
      (x.dtype() != mx::float32) && codec_has_nax(kquant_type)) {
    return qmm_nax(
        x,
        w,
        scales,
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

// Vector-times-matrix quantized kernel dispatch (no biases).
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

// Matrix-times-vector quantized kernel dispatch (no biases).
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

// Bias-fused qmv/qmv_fast dispatch: decode-only (M=1, non-batched) fast path
// for a KQuantLinear whose GGUF weight carries a real linear bias. The caller
// (KQuantQmvBias::eval_gpu, gated by the M=1 shape contract
// quantized_matmul_qmv_bias() enforces at the op level) guarantees M=1 --
// unlike qmv() above, there is no batched-B or M>1 fallback here because the
// only caller (KQuantLinear) never batches and only takes this path at
// decode.
void qmv_bias(
    const array& x,
    const array& w,
    const array& scales,
    const array& bias,
    array& out,
    int group_size,
    int bits,
    int N,
    int K,
    Device& d,
    const Stream& s,
    const std::string& kquant_type) {
  int bn = kquant_qmv_bn(kquant_type);
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(1, (N + bn - 1) / bn, 1);

  std::string type_string = kq_type_string(x.dtype());
  bool fast = (N % bn == 0) && (K % qmv_fast_k_align() == 0);
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + (fast ? "qmv_fast_bias_" : "qmv_bias_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_batch_0");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_input_array(bias, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  add_strides_and_shapes(ce, /*non_batched=*/true, x, w, scales, c);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// Verify-shaped small-M matmul. One threadgroup per
// N-tile (grid_dims.x = 1) reads each weight tile once and dots it against all
// M activation rows, amortizing the dominant weight read; the per-row qmv would
// re-read it M times (M on grid_dims.x). Non-batched only; M (= vm) in
// [2, verify_qmv_max_rows()], codec in codec_has_verify_qmv. Bit-for-bit
// identical to running qmv per row.
void verify_qmv(
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
  int bn = kquant_qmv_bn(kquant_type);
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);

  // Weight rows small enough at modest K stay L2-resident, so the default
  // verify tiling (8 output rows / threadgroup => few threadgroups) can starve
  // the GPU of occupancy vs the per-row qmv (M x more threadgroups) without
  // repaying it in saved DRAM traffic. The finer variant emits 2 output rows /
  // threadgroup (4x the threadgroups). Bit-exact vs the default. q8_0 measured
  // a win at idle so it is fine by default; q6_k measured NEUTRAL under
  // saturation (the verify forward's real condition), so it stays coarse by
  // default. Only q8_0 and q6_k have the fine kernel instantiated.
  // KQ_VERIFY_FINE=1 forces fine for both, KQ_VERIFY_FINE=0 forces coarse for
  // both (A/B lever).
  static const int verify_fine = []() {
    const char* e = std::getenv("KQ_VERIFY_FINE");
    return e != nullptr ? std::atoi(e) : -1; // -1 = per-codec default
  }();
  bool codec_has_fine = (kquant_type == "q8_0" || kquant_type == "q6_k");
  bool default_fine = (kquant_type == "q8_0"); // q6_k neutral -> coarse default
  bool use_fine = codec_has_fine &&
      (verify_fine == 1 || (verify_fine != 0 && default_fine));
  std::string verify_kname = "verify_qmv_";
  int rows_per_tg =
      bn; // default kernel emits bn (= num_simdgroups*RPS) rows/tg
  if (use_fine) {
    verify_kname = "verify_qmv_fine_";
    rows_per_tg = 2; // num_simdgroups(2) * results_per_simdgroup(1)
  }
  MTL::Size grid_dims(1, (N + rows_per_tg - 1) / rows_per_tg, 1);

  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + verify_kname,
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_batch_0");

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++); // in_vec_size
  ce.set_bytes(N, c++); // out_vec_size
  ce.set_bytes(M, c++); // vm (activation-row count)
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// Flat-with-M verify mat-vec (kq_<codec>_mv_ext): port of ggml mul_mv_ext_q4x4.
// One output row per thread, M (= vm) register accumulators, nypsg=4 output
// rows in parallel per simdgroup, nxpsg=8-lane K-reduction; the weight row
// streams once and is dotted against all M activation columns. M in [2, 12]
// selects the kernel (r1ptg is compile-time). All wired codecs (the kname
// prefix carries the codec); same call args as verify_qmv.
void verify_mv_ext(
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
  constexpr int nsg = 2;
  constexpr int nxpsg = 8;
  constexpr int rows_per_tg = (32 / nxpsg) * nsg; // nypsg * nsg = 8
  MTL::Size group_dims(32, nsg, 1);
  MTL::Size grid_dims((N + rows_per_tg - 1) / rows_per_tg, 1, 1);

  std::string type_string = kq_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  mx::concatenate(
      kname,
      kq_kname_prefix(kquant_type) + "mv_ext_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_m",
      M);

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(scales, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++); // in_vec_size
  ce.set_bytes(N, c++); // out_vec_size
  ce.set_bytes(M, c++); // vm (== r1ptg)
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

// The qmv_quad branch (K==64/128) is unreachable for kquant - eval_gpu throws
// for that case first - so this always routes to qmv.
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
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs: x, w (uint8), scales placeholder (ignored). Matrix-contiguous by
  // the op, so the M x K / weight rows are dense; leading (batch) dims are
  // walked via elem_to_loc.
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_output_array(out);
  encoder.dispatch([out = mx::array::unsafe_weak_copy(out),
                    x = mx::array::unsafe_weak_copy(x),
                    w = mx::array::unsafe_weak_copy(w),
                    transpose_ = transpose_,
                    kquant_type = kquant_type_]() mutable {
    int K = x.shape(-1);
    int M = x.ndim() > 1 ? x.shape(-2) : 1;
    int N = out.shape(-1);
    int batch_size =
        static_cast<int>(x.size() / (static_cast<std::size_t>(K) * M));
    std::size_t w_batch_els =
        w.ndim() > 2 ? static_cast<std::size_t>(w.shape(-1)) * w.shape(-2) : 0;
    auto run = [&](auto* tag) {
      using T = std::remove_pointer_t<decltype(tag)>;
      for (int i = 0; i < batch_size; i++) {
        kquant_qmm_cpu<T>(
            out.data<T>() + static_cast<std::size_t>(i) * M * N,
            x.data<T>() +
                elem_to_loc64(
                    static_cast<int64_t>(i) * M * K, x.shape(), x.strides()),
            w.data<uint8_t>() +
                elem_to_loc64(
                    static_cast<int64_t>(i) * static_cast<int64_t>(w_batch_els),
                    w.shape(),
                    w.strides()),
            M,
            N,
            K,
            transpose_,
            kquant_type);
      }
    };
    auto dt = x.dtype();
    if (dt == mx::float32) {
      run(static_cast<float*>(nullptr));
    } else if (dt == mx::float16) {
      run(static_cast<mx::float16_t*>(nullptr));
    } else if (dt == mx::bfloat16) {
      run(static_cast<mx::bfloat16_t*>(nullptr));
    } else {
      throw std::runtime_error(
          "[mlx_kquant] quantized_matmul: only float32/float16/bfloat16 inputs "
          "are supported.");
    }
  });
}

std::vector<mx::array> KQuantMatmul::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  // primals = {x, w (wire bytes), scales placeholder}. Only the gradient wrt x
  // is defined: dL/dx = cotan @ dequant(w) with the transpose flipped, i.e. the
  // same quantized matmul run the other way. The quantized base is frozen (the
  // LoRA use case), so the weight/scale branches throw.
  std::vector<mx::array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(quantized_matmul(
          cotangents[0],
          primals[1],
          primals[2],
          kquant_type_,
          !transpose_,
          stream()));
    } else if (arg == 1) {
      throw std::invalid_argument(
          "[mlx_kquant] quantized_matmul vjp: no gradient wrt the quantized "
          "weights (the kquant base is frozen).");
    } else {
      throw std::invalid_argument(
          "[mlx_kquant] quantized_matmul vjp: no gradient wrt scales.");
    }
  }
  return vjps;
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

  // KQuant special cases.
  if (!transpose_ && M < vector_limit) {
    qmm(x,
        w,
        scales,
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
  // There is no kquant qmv_quad kernel for K in {64,128}. Plain qmv is correct
  // for any K that is a multiple of the codec group size - K-quants (gs=256)
  // can never reach K=64/128, and the legacy gs=32 codecs only could on weights
  // whose input dim is exactly 64/128 (none in standard transformers). Fall
  // through to dispatch_qmv below rather than throw; a qmv_quad kernel would be
  // a perf-only path for an essentially-dead case.

  if (M >= vector_limit) {
    // For transpose shapes in the mv_ext M-range, the weight-read-amortizing
    // kernel beats qmm's under-utilised BM=64 tile at small M. Let those fall
    // through to the mv_ext check below; vector_limit was calibrated for
    // qmv-vs-qmm, not mv_ext-vs-qmm.
    if (!(transpose_ && non_batched && M >= 2 && M <= 12)) {
      qmm(x,
          w,
          scales,
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
  }

  if (transpose_) {
    // The verify width goes through the flat-with-M mat-vec (kq_<codec>_mv_ext,
    // a port of ggml mul_mv_ext): one output row per thread with M register
    // accumulators + nypsg parallel rows per simdgroup, vs verify_qmv's
    // [MAX_VM][RPS] block that stays occupancy-exposed under saturation.
    // Measured flat with M (matching llama) and bit-exact (fp-noise) vs
    // verify_qmv and per-row qmv. On by default for the codecs in
    // mv_ext_default_on; KQ_VERIFY_EXT=0 forces the verify_qmv path, =1 forces
    // mv_ext for any codec that has the kernel (A/B lever).
    static const int verify_ext = []() {
      const char* e = std::getenv("KQ_VERIFY_EXT");
      return e != nullptr ? std::atoi(e) : -1; // -1 = per-codec default
    }();
    // Every wired codec now has an mv_ext kernel: q8_0, the five K-quants, the
    // four legacy non-K (q4_0/q4_1/q5_0/q5_1), and all nine IQ. Validated
    // bit-exact, so default-on == has-kernel.
    const bool codec_has_mv_ext = kquant_type_ == "q8_0" ||
        kquant_type_ == "q2_k" || kquant_type_ == "q3_k" ||
        kquant_type_ == "q4_k" || kquant_type_ == "q5_k" ||
        kquant_type_ == "q6_k" || kquant_type_ == "q4_0" ||
        kquant_type_ == "q4_1" || kquant_type_ == "q5_0" ||
        kquant_type_ == "q5_1" || kquant_type_ == "iq4_nl" ||
        kquant_type_ == "iq4_xs" || kquant_type_ == "iq3_s" ||
        kquant_type_ == "iq3_xxs" || kquant_type_ == "iq2_xxs" ||
        kquant_type_ == "iq2_xs" || kquant_type_ == "iq2_s" ||
        kquant_type_ == "iq1_s" || kquant_type_ == "iq1_m";
    const bool mv_ext_default_on = codec_has_mv_ext;
    // Width gate for the DEFAULT path (the A/B force-on KQ_VERIFY_EXT=1 ignores
    // it). Measured DRAM-fresh: for non-IQ codecs verify_qmv (tuned for small
    // M, MAX_VM accumulators) ties or beats mv_ext at M==2 and mv_ext only pays
    // at M>=3, so non-IQ falls back to verify_qmv at M==2 (no regression at the
    // rarely-used draft-width-1 case). IQ has no verify_qmv kernel, so mv_ext
    // (vs per-row qmv) is a clear win at every M>=2 and stays on.
    const bool is_iq = kquant_type_.rfind("iq", 0) == 0;
    const bool mv_ext_width_ok = is_iq || M >= 3;
    // 32-weight blocks (legacy + q8_0 + iq4_nl) align K to 32; the 256-weight
    // super-block codecs (K-quants + the other IQ) align to 256. Pull the
    // modulus from the codec geometry rather than hard-coding per codec.
    const KQuantCodec* mv_ext_codec = codec_by_name(kquant_type_);
    const int mv_ext_k_align =
        mv_ext_codec ? mv_ext_codec->weights_per_block : 256;
    if (codec_has_mv_ext && non_batched && M >= 2 && M <= 12 &&
        (K % mv_ext_k_align == 0) &&
        (verify_ext == 1 ||
         (verify_ext != 0 && mv_ext_default_on && mv_ext_width_ok))) {
      verify_mv_ext(
          x, w, scales, out, group_size_, bits_, M, N, K, d, s, kquant_type_);
      return;
    }
    // Verify / small-batch regime: amortize the weight read across the M rows
    // instead of re-reading it per row (qmv puts M on grid_dims.x). Falls back
    // to qmv outside the supported codec/shape/row-count envelope.
    // KQ_DISABLE_VERIFY_QMV=1 forces the per-row qmv path (A/B harness lever).
    static const bool verify_disabled = []() {
      const char* e = std::getenv("KQ_DISABLE_VERIFY_QMV");
      return e != nullptr && e[0] == '1';
    }();
    int bn = kquant_qmv_bn(kquant_type_);
    bool verify_ok = !verify_disabled && non_batched && M >= 2 &&
        M <= verify_qmv_max_rows() && (N % bn == 0) &&
        (K % qmv_fast_k_align() == 0) && codec_has_verify_qmv(kquant_type_);
    if (verify_ok) {
      verify_qmv(
          x, w, scales, out, group_size_, bits_, M, N, K, d, s, kquant_type_);
      return;
    }
    dispatch_qmv(
        x, w, scales, out, group_size_, bits_, M, N, K, d, s, kquant_type_);
    return;
  }

  // The split-k qvm variant is omitted here; plain qvm is correct.
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

std::vector<mx::Shape> KQuantQmvBias::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  auto shape = x.shape();
  shape.back() = w.shape(-2); // transpose=true only: w is [N, K]
  return {shape};
}

bool KQuantQmvBias::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantQmvBias&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_;
}

void KQuantQmvBias::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs: x, w (uint8), scales placeholder (ignored), bias. The op-level
  // shape contract (quantized_matmul_qmv_bias) guarantees M=1 and
  // non-batched, so this needs no batch loop, unlike KQuantMatmul::eval_cpu.
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& bias = inputs[3];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(bias);
  encoder.set_output_array(out);
  encoder.dispatch([out = mx::array::unsafe_weak_copy(out),
                    x = mx::array::unsafe_weak_copy(x),
                    w = mx::array::unsafe_weak_copy(w),
                    bias = mx::array::unsafe_weak_copy(bias),
                    kquant_type = kquant_type_]() mutable {
    int K = x.shape(-1);
    int N = out.shape(-1);
    auto run = [&](auto* tag) {
      using T = std::remove_pointer_t<decltype(tag)>;
      kquant_qmm_cpu<T>(
          out.data<T>(),
          x.data<T>(),
          w.data<uint8_t>(),
          1,
          N,
          K,
          /*transpose=*/true,
          kquant_type);
      const T* bias_ptr = bias.data<T>();
      T* out_ptr = out.data<T>();
      for (int i = 0; i < N; i++) {
        out_ptr[i] = static_cast<T>(
            static_cast<float>(out_ptr[i]) + static_cast<float>(bias_ptr[i]));
      }
    };
    auto dt = x.dtype();
    if (dt == mx::float32) {
      run(static_cast<float*>(nullptr));
    } else if (dt == mx::float16) {
      run(static_cast<mx::float16_t*>(nullptr));
    } else if (dt == mx::bfloat16) {
      run(static_cast<mx::bfloat16_t*>(nullptr));
    } else {
      throw std::runtime_error(
          "[mlx_kquant] quantized_matmul_qmv_bias: only float32/float16/"
          "bfloat16 inputs are supported.");
    }
  });
}

#ifdef _METAL_

void KQuantQmvBias::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& scales = inputs[2];
  const auto& bias = inputs[3];

  int K = x.shape(-1);
  int N = out.shape(-1);

  qmv_bias(
      x, w, scales, bias, out, group_size_, bits_, N, K, d, s, kquant_type_);
}

#else

void KQuantQmvBias::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant] quantized_matmul_qmv_bias has no GPU implementation.");
}

#endif // _METAL_

} // namespace mlx_kquant
