#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_internal.h"
#include "kquant_lora_epilogue.h"

#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_is_nax_available / codec_has_nax
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

// Matrix-contiguity. A "matrix-contiguous" tensor only needs its LAST TWO
// dims densely packed (stride(-2)==shape(-1) && stride(-1)==1); a strided
// LEADING/batch dim is fine because the qmv/qmm/gather kernels walk it via
// the strides passed in add_strides_and_shapes. Using full mx::contiguous()
// instead would force a copy of the whole tensor whenever the leading dim is
// strided - e.g. the gate/up halves of a fused MoE gate_up_exps tensor are
// slices [..., :H, :] / [..., H:, :] whose expert (leading) stride is 2*H
// rows: matrix-contiguous but NOT row_contiguous, so full contiguify copies
// ~142 MB PER gather call (the decode MoE 5x regression). w and scales go
// through kq_matrix_dense (kquant_dense.cpp), which reads the strides of an
// array that has memory and leaves any other array to a node that decides
// at eval.

// x always gets an unconditional Contiguous node: build-time strides of an
// unevaluated array are meaningless, so a stride check here can hand the
// kernels a lazy transpose whose rows they mis-walk as dense (rows past the
// first read wrong memory). Contiguous::eval aliases zero-copy whenever the
// input lands row-contiguous.
inline mx::array kq_dense_x(const mx::array& x, mx::StreamOrDevice s) {
  return mx::contiguous(x, false, s);
}

// Codecs whose plain transposed quantized_matmul runs as dequantize then the
// stock matmul on a GPU without NAX before M5, from a floor in the rows of
// x. The qmm_t kernel decodes each weight tile once per 64-row tile of x;
// the dequantize pass decodes the weight once, so it pays from the floor,
// where the stock matmul outruns the in-loop decode. Floors measured on M3
// Max, and applied to the whole class.
struct KqDequantizeRow {
  const char* codec;
  int min_m;
};

constexpr KqDequantizeRow kDequantizeRows[] = {
    {"nvfp4", 2048},
    {"ptq1_0", 2048},
    {"q3_k", 2048},
    {"q4_k", 16384},
    {"q5_k", 2048},
    {"q6_k", 2048},
};

// The measured range of the route. The dense copy of the weight lives while
// the call runs, so N x K caps it at 24576 x 12288 x 2 bytes; a vocabulary
// head keeps the kernel.
constexpr int kDequantizeMinK = 3840;
constexpr int kDequantizeMaxK = 12288;
constexpr int kDequantizeMinN = 3840;
constexpr int kDequantizeMaxN = 24576;

// The row for a codec and shape, the chip class aside: min_m, min_k, max_k,
// min_n and max_n, or empty.
std::vector<int>
kq_dequantize_shape_row(const std::string& kquant_type, int M, int N, int K) {
  if (K < kDequantizeMinK || K > kDequantizeMaxK || N < kDequantizeMinN ||
      N > kDequantizeMaxN) {
    return {};
  }
  for (const auto& r : kDequantizeRows) {
    if (kquant_type == r.codec && M >= r.min_m) {
      return {
          r.min_m,
          kDequantizeMinK,
          kDequantizeMaxK,
          kDequantizeMinN,
          kDequantizeMaxN};
    }
  }
  return {};
}

// True for the chip classes the route runs on by default: an architecture
// string applegpu_g<gen><suffix> with a two digit generation from 13 (M1)
// to 16 (M4), a one letter suffix other than p (phones), and no NAX. Any
// other string keeps the kernel.
bool kq_dequantize_class(const std::string& arch, bool nax) {
  if (nax) {
    return false;
  }
  const std::string head = "applegpu_g";
  if (arch.size() != head.size() + 3 || arch.compare(0, head.size(), head)) {
    return false;
  }
  const char d0 = arch[head.size()];
  const char d1 = arch[head.size() + 1];
  const char suffix = arch[head.size() + 2];
  if (!std::isdigit(static_cast<unsigned char>(d0)) ||
      !std::isdigit(static_cast<unsigned char>(d1)) ||
      !std::islower(static_cast<unsigned char>(suffix)) || suffix == 'p') {
    return false;
  }
  const int gen = (d0 - '0') * 10 + (d1 - '0');
  return gen >= 13 && gen <= 16;
}

// Whether a plain transposed call on a 2-D weight on a GPU stream takes the
// route: never under off or while KQ_QMM_TILE pins the tile, on any GPU
// under force, on the class under auto, and then only for a codec row and
// shape of the table.
bool kq_dequantize_route_takes(
    bool on_class,
    int mode,
    bool pinned,
    const std::string& kquant_type,
    int M,
    int N,
    int K) {
  if (mode == kDequantOff || pinned) {
    return false;
  }
  if (mode != kDequantForce && !on_class) {
    return false;
  }
  return !kq_dequantize_shape_row(kquant_type, M, N, K).empty();
}

#ifdef _METAL_
// This GPU's class verdict, read once per process. NAX first.
bool kq_dequantize_class_here() {
  static const bool here = []() {
    if (kq_is_nax_available()) {
      return false;
    }
    auto& d = mx::metal::device(mx::Device::gpu);
    return kq_dequantize_class(d.get_architecture(), false);
  }();
  return here;
}
#endif

// Dequantize then the stock matmul, carried by KQuantDequantRoute for the
// vjp of the kernel path.
mx::array kq_dequantize_matmul(
    const mx::array& x,
    const mx::array& w,
    const mx::array& scales,
    const std::string& kquant_type,
    mx::Dtype out_type,
    mx::Stream s);

} // namespace

std::vector<int> alu_dequantize_row(
    const std::string& arch,
    bool nax,
    const std::string& kquant_type,
    int M,
    int N,
    int K) {
  if (!kq_dequantize_class(arch, nax)) {
    return {};
  }
  return kq_dequantize_shape_row(kquant_type, M, N, K);
}

bool dequantize_route_takes(
    const std::string& arch,
    bool nax,
    const std::string& mode,
    bool pinned,
    const std::string& kquant_type,
    int M,
    int N,
    int K) {
  int m = kDequantAuto;
  if (mode == "off") {
    m = kDequantOff;
  } else if (mode == "force") {
    m = kDequantForce;
  } else if (mode != "auto") {
    throw std::invalid_argument(
        "[mlx_kquant] dequantize route mode must be auto, off or force, got '" +
        mode + "'.");
  }
  return kq_dequantize_route_takes(
      kq_dequantize_class(arch, nax), m, pinned, kquant_type, M, N, K);
}

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

  // Row-major bytes for the kernel. A w without memory is decided at eval.
  auto w_c = kq_row_major(w, s);

  kq_check_weight_base_at_build(w, kquant_type, s, "dequantize");

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
    const std::optional<mx::array>& lora_a,
    const std::optional<mx::array>& lora_b,
    const std::optional<mx::array>& lora_rows,
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
  require_matmul_codec(*codec, "[mlx_kquant.quantized_matmul]");
  // A stack of one matrix is that matrix; a stack of more would run only
  // w[0].
  if (w.ndim() > 2) {
    int64_t mats = 1;
    for (int i = 0; i < w.ndim() - 2; i++) {
      mats *= w.shape(i);
    }
    if (mats != 1) {
      throw std::invalid_argument(
          "[mlx_kquant.quantized_matmul] batched (>2D) weights are not "
          "supported; use gather_qmm for mixture-of-experts.");
    }
    w = mx::reshape(w, {w.shape(-2), w.shape(-1)}, s_);
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
  const bool lora =
      lora_a.has_value() || lora_b.has_value() || lora_rows.has_value();
  if (lora && w.ndim() != 2) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul] the LoRA epilogue needs a 2-D w.");
  }
  // Rows of x, which the leading dims give for any inner dim.
  int rows = 1;
  for (int i = 0; i + 1 < static_cast<int>(x.ndim()); i++) {
    rows *= x.shape(i);
  }
  // An inner dim of 0 makes every sum empty, so the result is zeros. The
  // kernels divide by the inner dim. The LoRA arguments pass the checks of
  // a full call first.
  if (w_inner_dims == 0) {
    if (lora) {
      std::vector<mx::array> checked;
      kq_lora_prep(
          "[mlx_kquant.quantized_matmul]",
          lora_a,
          lora_b,
          std::nullopt,
          std::nullopt,
          lora_rows,
          out_type,
          w_inner_dims,
          w_outer_dims,
          0,
          {rows},
          1,
          s,
          checked);
    }
    auto shape = x.shape();
    shape.back() = w_outer_dims;
    return mx::zeros(shape, out_type, s);
  }
  kq_check_weight_base_at_build(w, kquant_type, s, "quantized_matmul");

#ifdef _METAL_
  if (transpose && !lora && w.ndim() == 2 && s.device == mx::Device::gpu) {
    const bool here = kq_dequantize_class_here();
    if (kq_dequantize_route_takes(
            here,
            kq_dequant_route_mode(),
            kq_qmm_tile_pinned(),
            kquant_type,
            rows,
            w_outer_dims,
            w_inner_dims)) {
      if (kq_qmm_log_on()) {
        std::fprintf(
            stderr,
            "[kq dequant] %s M=%d N=%d K=%d\n",
            kquant_type.c_str(),
            rows,
            w_outer_dims,
            w_inner_dims);
      }
      return kq_dequantize_matmul(x, w, scales, kquant_type, out_type, s);
    }
  }
#endif

  // Cast x to the output dtype, then matrix-row-contiguize x / w / scales at
  // the op level so eval_gpu can assume dense inputs.
  auto x_c = kq_dense_x(mx::astype(x, out_type, s), s);
  auto w_c = kq_matrix_dense(w, s);
  auto scales_c = kq_matrix_dense(scales, s);

  auto out_shape = x_c.shape();
  out_shape.back() = w_outer_dims;

  std::vector<mx::array> op_inputs = {x_c, w_c, scales_c};
  int lora_flags = 0;
  if (lora) {
    // The epilogue walks x as dense [rows, K] rows, which the Contiguous
    // node of x gives it.
    lora_flags = kq_lora_prep(
        "[mlx_kquant.quantized_matmul]",
        lora_a,
        lora_b,
        std::nullopt,
        std::nullopt,
        lora_rows,
        out_type,
        w_inner_dims,
        w_outer_dims,
        0,
        {rows},
        1,
        s,
        op_inputs);
  }
  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantMatmul>(
          s,
          kquant_type,
          codec->weights_per_block,
          codec->bits,
          transpose,
          lora_flags),
      std::move(op_inputs));
}

namespace {

mx::array kq_dequantize_matmul(
    const mx::array& x,
    const mx::array& w,
    const mx::array& scales,
    const std::string& kquant_type,
    mx::Dtype out_type,
    mx::Stream s) {
  auto w_f = dequantize(w, scales, kquant_type, out_type, s);
  auto y = mx::stop_gradient(
      mx::matmul(mx::astype(x, out_type, s), mx::transpose(w_f, s), s), s);
  auto shape = y.shape();
  return mx::array(
      std::move(shape),
      out_type,
      std::make_shared<KQuantDequantRoute>(s, kquant_type),
      {y, x, w, scales});
}

} // namespace

void KQuantDequantRoute::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  outputs[0].copy_shared_buffer(inputs[0]);
}

void KQuantDequantRoute::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  outputs[0].copy_shared_buffer(inputs[0]);
}

std::vector<mx::array> KQuantDequantRoute::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  // primals = {y (stop_gradient), x, w, scales}, as KQuantMatmul::vjp: x
  // gets the quantized matmul the other way, w and scales throw.
  std::vector<mx::array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(cotangents[0]);
    } else if (arg == 1) {
      auto dx = quantized_matmul(
          cotangents[0],
          primals[2],
          primals[3],
          kquant_type_,
          false,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          stream());
      vjps.push_back(mx::astype(dx, primals[1].dtype(), stream()));
    } else if (arg == 2) {
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

std::vector<mx::array> KQuantDequantRoute::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::invalid_argument(
      "[Primitive::jvp] Not implemented for KQuantMatmul.");
}

std::pair<std::vector<mx::array>, std::vector<int>> KQuantDequantRoute::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::invalid_argument(
      "[Primitive::vmap] Not implemented for KQuantMatmul.");
}

mx::array quantized_matmul_qmv_bias(
    mx::array x,
    mx::array w,
    mx::array scales,
    mx::array bias,
    const std::string& kquant_type,
    const std::optional<mx::array>& lora_a,
    const std::optional<mx::array>& lora_b,
    const std::optional<mx::array>& lora_rows,
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
  kq_check_weight_base_at_build(w, kquant_type, s, "quantized_matmul_qmv_bias");

  auto x_c = kq_dense_x(mx::astype(x, out_type, s), s);
  auto w_c = kq_matrix_dense(w, s);
  auto scales_c = kq_matrix_dense(scales, s);
  auto bias_c = kq_row_major(mx::astype(bias, out_type, s), s);

  auto out_shape = x_c.shape();
  out_shape.back() = N;

  std::vector<mx::array> op_inputs = {x_c, w_c, scales_c, bias_c};
  const int lora_flags = kq_lora_prep(
      "[mlx_kquant.quantized_matmul_qmv_bias]",
      lora_a,
      lora_b,
      std::nullopt,
      std::nullopt,
      lora_rows,
      out_type,
      weights_per_row,
      N,
      0,
      {1},
      1,
      s,
      op_inputs);
  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantQmvBias>(
          s, kquant_type, codec->weights_per_block, codec->bits, lora_flags),
      std::move(op_inputs));
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
  // No indices on one matrix -> a plain quantized matmul. A stack of
  // matrices takes the gathered path below, which runs each one.
  int64_t w_mats = 1;
  for (int i = 0; i < w.ndim() - 2; i++) {
    w_mats *= w.shape(i);
  }
  if (!lhs_indices_ && !rhs_indices_ && w_mats == 1) {
    return quantized_matmul(
        std::move(x),
        std::move(w),
        std::move(scales),
        kquant_type,
        transpose,
        std::nullopt,
        std::nullopt,
        std::nullopt,
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
  require_matmul_codec(*codec, "[mlx_kquant.gather_qmm]");
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
  kq_check_weight_base_at_build(w, kquant_type, s, "gather_qmm");

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
  auto x_c = kq_dense_x(mx::astype(x, out_type, s), s);
  // Always insert the Contiguous node when the rhs leaf is reachable: flags
  // of an UNEVALUATED array (e.g. a fresh gate/up slice view) are not
  // meaningful at op-build time, so a `w.flags().row_contiguous ? w : ...`
  // short-circuit here skipped the compaction on the slice's first eval and
  // handed the stride-less rhs kernels an expert stride they mis-walk.
  // Contiguous::eval aliases zero-copy when w turns out packed anyway.
  auto w_c =
      rhs_reachable ? mx::contiguous(w, false, s) : kq_matrix_dense(w, s);
  auto scales_c = kq_matrix_dense(scales, s);

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
    int tile_rows,
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
  if (tile_rows != 64 && tile_rows != 128) {
    throw std::invalid_argument(
        "[mlx_kquant.expert_tile_map] tile_rows must be 64 or 128.");
  }

  auto s = mx::to_stream(s_);
  auto idx_c = mx::contiguous(indices, false, s);

  int cap = std::max(rows / tile_rows + std::min(n_experts, rows), 1);
  std::vector<mx::Shape> shapes = {{cap, 3}, {1}};
  std::vector<mx::Dtype> dtypes = {mx::uint32, mx::uint32};
  return mx::array::make_arrays(
      std::move(shapes),
      dtypes,
      std::make_shared<KQuantExpertTileMap>(s, n_experts, tile_rows),
      {idx_c});
}

namespace {

void check_seg_map(
    const mx::array& m,
    const char* which,
    const std::string& op) {
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
    int tile_rows,
    mx::StreamOrDevice s_) {
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_seg] w must be uint8 (raw GGUF wire bytes).");
  }
  if (tile_rows != 64 && tile_rows != 128) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_seg] tile_rows must be 64 or 128.");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_seg] Unknown kquant_type: '" + kquant_type +
        "'.");
  }
  require_matmul_codec(*codec, "[mlx_kquant.gather_qmm_seg]");
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
  kq_check_weight_base_at_build(w, kquant_type, s, "gather_qmm_seg");
  auto x_c = kq_dense_x(mx::astype(x, out_type, s), s);
  // The kernel computes each expert's base pointer as expert * N * K_w, so w
  // must be fully row-contiguous (no strided expert dim).
  auto w_c = mx::contiguous(w, false, s);
  auto scales_c = kq_matrix_dense(scales, s);
  auto map_c = mx::contiguous(map, false, s);
  auto cnt_c = mx::contiguous(counts, false, s);

  mx::Shape out_shape{x.shape(0), w.shape(1)};
  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantGatherQMMSeg>(
          s, kquant_type, codec->weights_per_block, codec->bits, tile_rows),
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
  // The IQ, stq1_0 and Prism encoders are CPU-only; force the CPU stream so
  // eval_cpu runs even when the caller passed stream=gpu.
  if (kquant_type.rfind("iq", 0) == 0 || kquant_type == "stq1_0" ||
      kquant_type == "pq2_0" || kquant_type == "q2_0" ||
      kquant_type == "ptq1_0") {
    s = mx::default_stream(mx::Device::cpu);
  }

  auto wq_shape = w.shape();
  wq_shape.back() =
      (w.shape(-1) / codec->weights_per_block) * codec->bytes_per_block;
  mx::Shape s_shape = {1};

  // Row-major w and imatrix for the encoder. An operand without memory is
  // decided at eval.
  std::vector<mx::array> inputs = {kq_row_major(w, s)};
  if (imatrix.has_value()) {
    inputs.push_back(kq_row_major(*imatrix, s));
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

std::vector<std::string> metallib_kernels() {
#ifdef _METAL_
  auto& d = mx::metal::device(mx::Device::gpu);
  MTL::Library* lib = d.get_library("mlx_kquant", metallib_dir());
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  NS::Array* names = lib->functionNames();
  std::vector<std::string> out;
  out.reserve(names->count());
  for (NS::UInteger i = 0; i < names->count(); ++i) {
    out.emplace_back(names->object<NS::String>(i)->utf8String());
  }
  std::sort(out.begin(), out.end());
  return out;
#else
  return {};
#endif
}

int max_threadgroup_memory() {
#ifdef _METAL_
  auto& d = mx::metal::device(mx::Device::gpu);
  return static_cast<int>(d.mtl_device()->maxThreadgroupMemoryLength());
#else
  return 0;
#endif
}

bool probe_kernels() {
#ifdef KQ_PROBE_KERNELS
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
