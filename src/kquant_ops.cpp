#include <dlfcn.h>

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "kquant.h"
#include "kquant_codec.h"

#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_is_nax_available / codec_has_nax
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

// Matrix-contiguity helper. A "matrix-contiguous" tensor only needs
// its LAST TWO dims densely packed (stride(-2)==shape(-1) && stride(-1)==1); a
// strided LEADING/batch dim is fine because the qmv/qmm/gather kernels walk it
// via the strides passed in add_strides_and_shapes. Using full
// mx::contiguous() here instead would force a copy of the whole tensor whenever
// the leading dim is strided - e.g. the gate/up halves of a fused MoE
// gate_up_exps tensor are slices [..., :H, :] / [..., H:, :] whose expert
// (leading) stride is 2*H rows: matrix-contiguous but NOT row_contiguous, so
// full contiguify copies ~142 MB PER gather call (the decode MoE 5x
// regression).
inline mx::array kq_ensure_row_contiguous_matrix(
    const mx::array& x,
    mx::StreamOrDevice s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == static_cast<int64_t>(x.shape(-1)) && stride_1 == 1) {
      return x;
    }
  }
  return mx::contiguous(x, false, s);
}

} // namespace

// ----------------------------- ops -----------------------------

mx::array dequantize(
    const mx::array& w,
    const mx::array& scales,
    const std::string& kquant_type,
    std::optional<mx::Dtype> dtype,
    mx::StreamOrDevice s_) {
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.dequantize] w must be uint8 (raw GGUF wire bytes).");
  }
  if (w.ndim() < 1) {
    throw std::invalid_argument(
        "[mlx_kquant.dequantize] w must have at least 1 dimension.");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.dequantize] Unknown kquant_type: '" + kquant_type + "'.");
  }
  if (w.shape(-1) % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dequantize] Last dim (" << w.shape(-1)
        << ") must be a multiple of bytes_per_block (" << codec->bytes_per_block
        << ") for codec '" << kquant_type << "'.";
    throw std::invalid_argument(msg.str());
  }

  mx::Dtype out_type = dtype.value_or(mx::float16);
  if (!mx::issubdtype(out_type, mx::floating)) {
    throw std::invalid_argument(
        "[mlx_kquant.dequantize] Only real floating output dtypes are "
        "supported.");
  }

  auto s = mx::to_stream(s_);

  auto out_shape = w.shape();
  out_shape.back() =
      (w.shape(-1) / codec->bytes_per_block) * codec->weights_per_block;

  // Row-contiguize at the op level so eval_gpu can assume dense inputs.
  auto w_c = w.flags().row_contiguous ? w : mx::contiguous(w, false, s);

  // For every kquant codec, group_size == weights_per_block (32 or 256).
  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantDequantize>(
          s, kquant_type, codec->weights_per_block, codec->bits),
      {w_c, scales});
}

mx::array quantized_matmul(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    bool transpose,
    mx::StreamOrDevice s_) {
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul] w must be uint8 (raw GGUF wire bytes).");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul] Unknown kquant_type: '" + kquant_type +
        "'.");
  }
  if (x.ndim() > 2 && w.ndim() > 2) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul] batched (>2D) weights are not supported; "
        "use gather_qmm for mixture-of-experts.");
  }

  // Expand w's quantized geometry from the codec block layout.
  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul] KQuant weight last dim ("
        << w_bytes_per_row << " bytes) is not a whole number of "
        << codec->bytes_per_block << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int weights_per_row =
      (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  int w_inner_dims = transpose ? weights_per_row : w.shape(-2);
  int w_outer_dims = transpose ? w.shape(-2) : weights_per_row;
  if (w_inner_dims != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized matrix inner dim ("
        << w_inner_dims << ") for codec '" << codec->name
        << "' with transpose=" << std::boolalpha << transpose << ".";
    throw std::invalid_argument(msg.str());
  }

  // Output dtype: x.dtype(), with float32 promoted to bfloat16 (the NAX matmul
  // path emits bf16 accumulation).
  mx::Dtype out_type = (x.dtype() == mx::float32) ? mx::bfloat16 : x.dtype();
  if (!mx::issubdtype(out_type, mx::floating)) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul] Only real floating x dtypes are "
        "supported.");
  }

  auto s = mx::to_stream(s_);

  // Cast x to the output dtype, then matrix-row-contiguize x / w / scales at
  // the op level so eval_gpu can assume dense inputs.
  auto x_c = kq_ensure_row_contiguous_matrix(mx::astype(x, out_type, s), s);
  auto w_c = kq_ensure_row_contiguous_matrix(w, s);
  auto scales_c = kq_ensure_row_contiguous_matrix(scales, s);

  auto out_shape = x_c.shape();
  out_shape.back() = w_outer_dims;

  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantMatmul>(
          s, kquant_type, codec->weights_per_block, codec->bits, transpose),
      {x_c, w_c, scales_c});
}

mx::array quantized_matmul_qmv_bias(
    mx::array x,
    mx::array w,
    mx::array scales,
    mx::array bias,
    const std::string& kquant_type,
    mx::StreamOrDevice s_) {
  if (kquant_type != "q8_0") {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_bias] only kquant_type 'q8_0' is "
        "wired so far, got '" +
        kquant_type + "'.");
  }
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_bias] w must be uint8 (raw GGUF "
        "wire bytes).");
  }
  if (w.ndim() != 2) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_bias] w must be 2D (this op is not "
        "batched); use gather_qmm for mixture-of-experts.");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);

  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_bias] KQuant weight last dim ("
        << w_bytes_per_row << " bytes) is not a whole number of "
        << codec->bytes_per_block << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int weights_per_row =
      (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  int N = w.shape(-2); // transpose=true only: w is [N, K]
  if (weights_per_row != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_bias] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized matrix "
        << "inner dim (" << weights_per_row << ") for codec '" << codec->name
        << "'.";
    throw std::invalid_argument(msg.str());
  }
  if (bias.ndim() != 1 || bias.shape(0) != N) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_bias] bias must be 1D with "
        << "length " << N << " (the output dim), got shape " << bias.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  int64_t total_rows = x.size() / x.shape(-1);
  if (total_rows != 1) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_bias] decode-only: x must carry "
        << "exactly one row (got " << total_rows << "). Use "
        << "quantized_matmul() + a separate add for prefill/verify shapes.";
    throw std::invalid_argument(msg.str());
  }

  mx::Dtype out_type = (x.dtype() == mx::float32) ? mx::bfloat16 : x.dtype();
  if (!mx::issubdtype(out_type, mx::floating)) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_bias] Only real floating x dtypes "
        "are supported.");
  }

  auto s = mx::to_stream(s_);

  auto x_c = kq_ensure_row_contiguous_matrix(mx::astype(x, out_type, s), s);
  auto w_c = kq_ensure_row_contiguous_matrix(w, s);
  auto scales_c = kq_ensure_row_contiguous_matrix(scales, s);
  auto bias_c = mx::astype(bias, out_type, s);
  bias_c =
      bias_c.flags().row_contiguous ? bias_c : mx::contiguous(bias_c, false, s);

  auto out_shape = x_c.shape();
  out_shape.back() = N;

  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantQmvBias>(
          s, kquant_type, codec->weights_per_block, codec->bits),
      {x_c, w_c, scales_c, bias_c});
}

namespace {

// When indices are omitted, default to a flat arange over the leading (batch)
// dims.
mx::array kq_indices_or_default(
    const std::optional<mx::array>& indices,
    const mx::array& x,
    mx::StreamOrDevice s) {
  if (indices.has_value()) {
    return indices.value();
  }
  mx::Shape shape(x.shape().begin(), x.shape().end() - 2);
  int total = 1;
  for (auto dim : shape) {
    total *= dim;
  }
  return mx::reshape(mx::arange(total, mx::uint32, s), std::move(shape), s);
}

} // namespace

mx::array gather_qmm(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    std::optional<mx::array> lhs_indices_,
    std::optional<mx::array> rhs_indices_,
    bool transpose,
    bool sorted_indices,
    mx::StreamOrDevice s_) {
  // No indices at all -> a plain (non-gathered) quantized matmul.
  if (!lhs_indices_ && !rhs_indices_) {
    return quantized_matmul(
        std::move(x),
        std::move(w),
        std::move(scales),
        kquant_type,
        transpose,
        s_);
  }

  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm] w must be uint8 (raw GGUF wire bytes).");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm] Unknown kquant_type: '" + kquant_type + "'.");
  }
  if (x.ndim() < 2) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm] x must have at least two dimensions but got "
        << "shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Expand w's quantized geometry from the codec block layout. w is
  // [..., out_dims, bytes_per_row]; the last dim is whole K-quant blocks.
  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm] KQuant weight last dim (" << w_bytes_per_row
        << " bytes) is not a whole number of " << codec->bytes_per_block
        << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int weights_per_row =
      (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  int w_inner_dims = transpose ? weights_per_row : w.shape(-2);
  int w_outer_dims = transpose ? w.shape(-2) : weights_per_row;
  if (w_inner_dims != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized matrix inner dim ("
        << w_inner_dims << ") for codec '" << codec->name
        << "' with transpose=" << std::boolalpha << transpose << ".";
    throw std::invalid_argument(msg.str());
  }

  // Output dtype: x.dtype(), with float32 promoted to bfloat16.
  mx::Dtype out_type = (x.dtype() == mx::float32) ? mx::bfloat16 : x.dtype();
  if (!mx::issubdtype(out_type, mx::floating)) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm] Only real floating x dtypes are supported.");
  }

  auto s = mx::to_stream(s_);

  // Default + broadcast the indices, then cast to uint32.
  mx::array lhs_indices = kq_indices_or_default(lhs_indices_, x, s);
  mx::array rhs_indices = kq_indices_or_default(rhs_indices_, w, s);
  auto bc = mx::broadcast_arrays({lhs_indices, rhs_indices}, s);
  lhs_indices = mx::astype(bc[0], mx::uint32, s);
  rhs_indices = mx::astype(bc[1], mx::uint32, s);

  // Full output shape: lhs_indices batch dims, then [M, w_outer_dims].
  auto out_shape = lhs_indices.shape();
  out_shape.push_back(x.shape(-2));
  out_shape.push_back(w_outer_dims);

  // left_sorted_ / right_sorted_: a sort guarantee holds for the side whose
  // index was *defaulted* (broadcast from the other), so each "sorted" flag is
  // gated on the OPPOSITE index being
  // absent. left_sorted tracks the lhs (x) ordering -> gated on rhs absent;
  // right_sorted tracks the rhs (w/expert) ordering -> gated on lhs absent.
  // The MoE call passes rhs_indices only, so right_sorted_ == sorted_indices,
  // which arms the gather_qmm_rhs fast path in eval_gpu.
  bool left_sorted = sorted_indices && !rhs_indices_;
  bool right_sorted = sorted_indices && !lhs_indices_;

  // Cast x to the output dtype; row-contiguize x / w / scales at the op level.
  //
  // w needs FULL row-contiguity ONLY when a gather_qmm_rhs prefill leaf is
  // reachable (right_sorted && transpose && codec has a fused matmul kernel):
  // those kernels assume dense expert packing and take NO w strides. Every
  // other leaf (qmv decode / qmm) walks w's strides, so matrix-contiguity (a
  // strided LEADING/expert dim, no copy) suffices and is what keeps the fused
  // gate_up_exps slice from being copied wholesale on every DECODE gather - the
  // 5x-sensitive hot path. x / scales always take the cheaper matrix check: the
  // eval_gpu rhs gate already requires x.row_contiguous (skips the rhs leaf
  // otherwise, routing to a strided-safe leaf), and scales is a (1,)
  // placeholder.
  bool rhs_reachable = right_sorted && transpose && codec->has_matmul_kernel;
  auto x_c = kq_ensure_row_contiguous_matrix(mx::astype(x, out_type, s), s);
  // Always insert the Contiguous node when the rhs leaf is reachable: flags
  // of an UNEVALUATED array (e.g. a fresh gate/up slice view) are not
  // meaningful at op-build time, so a `w.flags().row_contiguous ? w : ...`
  // short-circuit here skipped the compaction on the slice's first eval and
  // handed the stride-less rhs kernels an expert stride they mis-walk.
  // Contiguous::eval aliases zero-copy when w turns out packed anyway.
  auto w_c = rhs_reachable ? mx::contiguous(w, false, s)
                           : kq_ensure_row_contiguous_matrix(w, s);
  auto scales_c = kq_ensure_row_contiguous_matrix(scales, s);

  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantGatherQMM>(
          s,
          kquant_type,
          codec->weights_per_block,
          codec->bits,
          transpose,
          left_sorted,
          right_sorted),
      {x_c, w_c, scales_c, std::move(lhs_indices), std::move(rhs_indices)});
}

std::vector<mx::array> expert_tile_map(
    mx::array indices,
    int n_experts,
    mx::StreamOrDevice s_) {
  if (indices.ndim() != 1 || indices.dtype() != mx::uint32) {
    std::ostringstream msg;
    msg << "[mlx_kquant.expert_tile_map] indices must be 1-D uint32 but got "
        << "shape " << indices.shape() << " dtype " << indices.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  int rows = indices.shape(0);
  if (rows < 1 || n_experts < 1) {
    throw std::invalid_argument(
        "[mlx_kquant.expert_tile_map] indices must be non-empty and "
        "n_experts positive.");
  }

  auto s = mx::to_stream(s_);
  auto idx_c = mx::contiguous(indices, false, s);

  int cap = std::max(rows / 64 + std::min(n_experts, rows), 1);
  std::vector<mx::Shape> shapes = {{cap, 3}, {1}};
  std::vector<mx::Dtype> dtypes = {mx::uint32, mx::uint32};
  return mx::array::make_arrays(
      std::move(shapes),
      dtypes,
      std::make_shared<KQuantExpertTileMap>(s, n_experts),
      {idx_c});
}

namespace {

void check_seg_map(
    const mx::array& m, const char* which, const std::string& op) {
  if (m.ndim() != 2 || m.shape(1) != 3 || m.dtype() != mx::uint32 ||
      m.shape(0) < 1) {
    std::ostringstream msg;
    msg << "[mlx_kquant." << op << "] " << which
        << " must be uint32 [n_tiles >= 1, 3] (expert, row_start, num_rows) "
        << "but got shape " << m.shape() << " dtype " << m.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

mx::array gather_qmm_seg(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    mx::array map,
    mx::array counts,
    mx::StreamOrDevice s_) {
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_seg] w must be uint8 (raw GGUF wire bytes).");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_seg] Unknown kquant_type: '" + kquant_type +
        "'.");
  }
  if (x.ndim() != 2) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_seg] x must be [rows, K] but got shape "
        << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (w.ndim() != 3) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_seg] w must be "
        << "(n_experts, N, bytes_per_row) but got shape " << w.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  check_seg_map(map, "map", "gather_qmm_seg");
  if (counts.ndim() != 1 || counts.shape(0) != 1 ||
      counts.dtype() != mx::uint32) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_seg] counts must be uint32 [1] but got "
        << "shape " << counts.shape() << " dtype " << counts.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_seg] KQuant weight last dim ("
        << w_bytes_per_row << " bytes) is not a whole number of "
        << codec->bytes_per_block << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int weights_per_row =
      (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  if (weights_per_row != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_seg] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized row width ("
        << weights_per_row << ") for codec '" << codec->name << "'.";
    throw std::invalid_argument(msg.str());
  }

  mx::Dtype out_type = (x.dtype() == mx::float32) ? mx::bfloat16 : x.dtype();
  if (!mx::issubdtype(out_type, mx::floating)) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_seg] Only real floating x dtypes are "
        "supported.");
  }

  auto s = mx::to_stream(s_);
  auto x_c = kq_ensure_row_contiguous_matrix(mx::astype(x, out_type, s), s);
  // The kernel computes each expert's base pointer as expert * N * K_w, so w
  // must be fully row-contiguous (no strided expert dim).
  auto w_c = mx::contiguous(w, false, s);
  auto scales_c = kq_ensure_row_contiguous_matrix(scales, s);
  auto map_c = mx::contiguous(map, false, s);
  auto cnt_c = mx::contiguous(counts, false, s);

  mx::Shape out_shape{x.shape(0), w.shape(1)};
  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantGatherQMMSeg>(
          s, kquant_type, codec->weights_per_block, codec->bits),
      {x_c, w_c, scales_c, std::move(map_c), std::move(cnt_c)});
}

std::vector<mx::array> quantize(
    const mx::array& w,
    const std::string& kquant_type,
    const std::optional<mx::array>& imatrix,
    mx::StreamOrDevice s_) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.quantize] Unknown kquant_type: '" + kquant_type + "'.");
  }
  if (!codec->has_encode) {
    throw std::invalid_argument(
        "[mlx_kquant.quantize] Encode is not supported for codec '" +
        kquant_type + "'.");
  }
  if (!mx::issubdtype(w.dtype(), mx::floating)) {
    throw std::invalid_argument(
        "[mlx_kquant.quantize] w must be a real floating tensor (float32, "
        "float16, or bfloat16).");
  }
  if (w.ndim() < 1) {
    throw std::invalid_argument(
        "[mlx_kquant.quantize] w must have at least 1 dimension.");
  }
  if (w.shape(-1) % codec->weights_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantize] Last dim (" << w.shape(-1)
        << ") must be a multiple of weights_per_block ("
        << codec->weights_per_block << ") for codec '" << kquant_type << "'.";
    throw std::invalid_argument(msg.str());
  }
  if (imatrix.has_value()) {
    const auto& im = *imatrix;
    if (im.dtype() != mx::float32) {
      throw std::invalid_argument(
          "[mlx_kquant.quantize] imatrix must be float32.");
    }
    if (im.ndim() != 1 || im.shape(-1) != w.shape(-1)) {
      std::ostringstream msg;
      msg << "[mlx_kquant.quantize] imatrix shape must be [K]=(" << w.shape(-1)
          << ",) but got " << im.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }
  if (codec->requires_imatrix && !imatrix.has_value()) {
    throw std::invalid_argument(
        "[mlx_kquant.quantize] codec '" + kquant_type +
        "' requires an importance matrix (imatrix); ggml rejects encoding it "
        "without one.");
  }

  auto s = mx::to_stream(s_);
  // IQ encode is CPU-only (ggml has no GPU IQ quantizer); force the op onto the
  // CPU stream UNCONDITIONALLY so KQuantQuantize::eval_cpu runs (it needs a CPU
  // command encoder) even if the caller passed stream=gpu -- there is no GPU IQ
  // encoder to honor. All nine IQ codecs -- and no K-quant -- start with "iq".
  if (kquant_type.rfind("iq", 0) == 0) {
    s = mx::default_stream(mx::Device::cpu);
  }

  auto wq_shape = w.shape();
  wq_shape.back() =
      (w.shape(-1) / codec->weights_per_block) * codec->bytes_per_block;
  mx::Shape s_shape = {1};

  // Row-contiguize w (+ imatrix) at the op level; eval_gpu trusts contiguity.
  auto w_c = w.flags().row_contiguous ? w : mx::contiguous(w, false, s);
  std::vector<mx::array> inputs = {w_c};
  if (imatrix.has_value()) {
    auto im_c = imatrix->flags().row_contiguous
        ? *imatrix
        : mx::contiguous(*imatrix, false, s);
    inputs.push_back(im_c);
  }

  return mx::array::make_arrays(
      {std::move(wq_shape), std::move(s_shape)},
      {mx::uint8, mx::uint8},
      std::make_shared<KQuantQuantize>(
          s, kquant_type, codec->weights_per_block, codec->bits),
      inputs);
}

// ----------------------------- toolchain self-checks
// -----------------------------

// Locate the directory of this shared object (where mlx_kquant.metallib lives).
std::string metallib_dir() {
  static std::string dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&metallib_dir), &info)) {
      throw std::runtime_error(
          "mlx_kquant: unable to resolve binary directory");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return dir;
}

bool metallib_loads() {
#ifdef _METAL_
  auto& d = mx::metal::device(mx::Device::gpu);
  // get_library throws if <dir>/mlx_kquant.metallib is missing or unreadable.
  d.get_library("mlx_kquant", metallib_dir());
  return true;
#else
  return false;
#endif
}

bool nax_available() {
#ifdef _METAL_
  return kq_is_nax_available();
#else
  return false;
#endif
}

bool nax_gather_enabled(const std::string& kquant_type) {
#ifdef _METAL_
  return kq_is_nax_available() && codec_has_nax(kquant_type);
#else
  (void)kquant_type;
  return false;
#endif
}

} // namespace mlx_kquant
