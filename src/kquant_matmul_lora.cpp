// Quantized matmul with a low-rank term in the same reduction (see
// kq_quantized_nax_lora.h for the kernel).
//
//   y = x @ W^T + (strength * (x @ A)) @ B
//
// On a NAX GPU, for the codecs with a kernel, from kLoraMinRows rows and
// with K a whole number of the codec's K tiles, stock ops compute
// z = strength * (x @ A) and one dispatch computes the product and adds
// z @ B in its accumulator, with y as its only output buffer. Every other
// case builds the same sum from quantized_matmul and stock ops.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_internal.h"

#include "mlx/allocator.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "kquant_metal_internal.h"
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

const char* kLoraOp = "[mlx_kquant.quantized_matmul_lora]";

// Column tile of the kernel. A band starts on a column tile.
constexpr int kLoraBn = 64;
// Largest rank of one band the kernel runs.
constexpr int kLoraMaxRank = 256;
// Most bands the kernel looks through per threadgroup.
constexpr int kLoraMaxBands = 16;

// K tile of the kernel for a codec and an activation dtype, 0 where the op
// builds the composition.
int lora_kernel_bk(const std::string& t, mx::Dtype dt) {
  if (t == "q4_k" && dt == mx::bfloat16) {
    return 128;
  }
  return 0;
}

bool lora_tune_live() {
  static const bool live = kq_env_lever("KQ_TUNE_LIVE", 0) != 0;
  return live;
}

// KQ_QMM_LORA=0 builds the composition on every call.
bool lora_on() {
  static const bool cached = kq_env_flag("KQ_QMM_LORA", 1) != 0;
  return lora_tune_live() ? kq_env_flag("KQ_QMM_LORA", 1) != 0 : cached;
}

// Rows from which the kernel runs. KQ_QMM_LORA_MIN_ROWS replaces the bound
// with a value of at least 1. A value below 1 keeps the bound, with a
// warning.
constexpr int kLoraMinRows = 1024;
int lora_min_rows() {
  static const int cached =
      kq_env_lever_from("KQ_QMM_LORA_MIN_ROWS", kLoraMinRows, 1);
  return lora_tune_live()
      ? kq_env_lever_from("KQ_QMM_LORA_MIN_ROWS", kLoraMinRows, 1)
      : cached;
}

bool lora_log() {
  static const bool cached = std::getenv("KQ_QMM_LOG") != nullptr;
  return lora_tune_live() ? std::getenv("KQ_QMM_LOG") != nullptr : cached;
}

mx::array last_axis_slice(const mx::array& a, int lo, int hi, mx::Stream s) {
  mx::Shape start(a.ndim(), 0);
  mx::Shape stop = a.shape();
  start.back() = lo;
  stop.back() = hi;
  return mx::slice(a, std::move(start), std::move(stop), s);
}

// base + z @ B over the bands. With no band table every column takes
// every rank. With one, a band's columns take its ranks and its block of
// B, and a column outside every band takes none.
mx::array lora_add(
    const mx::array& base,
    const mx::array& z,
    const mx::array& b,
    const std::vector<int>& bands,
    mx::Stream s) {
  auto add_block =
      [&](const mx::array& y, const mx::array& zz, const mx::array& bb) {
        if (zz.ndim() < 2) {
          // One row: the 2-D call on that row, with its bits.
          auto row = mx::addmm(
              mx::expand_dims(y, 0, s),
              mx::expand_dims(zz, 0, s),
              bb,
              1.0f,
              1.0f,
              s);
          return mx::squeeze(row, 0, s);
        }
        return mx::addmm(y, zz, bb, 1.0f, 1.0f, s);
      };
  if (bands.empty()) {
    return add_block(base, z, b);
  }
  const int N = base.shape(-1);
  std::vector<mx::array> parts;
  int col = 0;
  for (size_t i = 0; i < bands.size(); i += 4) {
    const int cs = bands[i];
    const int ce = bands[i + 1];
    const int z0 = bands[i + 2];
    const int rb = bands[i + 3];
    if (cs > col) {
      parts.push_back(last_axis_slice(base, col, cs, s));
    }
    auto y = last_axis_slice(base, cs, ce, s);
    if (rb > 0) {
      auto bb = mx::slice(b, {z0, cs}, {z0 + rb, ce}, s);
      y = add_block(y, last_axis_slice(z, z0, z0 + rb, s), bb);
    }
    parts.push_back(std::move(y));
    col = ce;
  }
  if (col < N) {
    parts.push_back(last_axis_slice(base, col, N, s));
  }
  return parts.size() == 1 ? parts[0]
                           : mx::concatenate(std::move(parts), -1, s);
}

// The sum from quantized_matmul and stock ops, given z. The op builds this
// where it has no kernel, and the derivatives of the kernel are the
// derivatives of this.
mx::array lora_composed(
    const mx::array& x,
    const mx::array& w,
    const mx::array& z,
    const mx::array& b,
    const std::string& kquant_type,
    const std::vector<int>& bands,
    mx::Stream s) {
  auto scales = mx::zeros({1}, mx::uint8, s);
  auto base = quantized_matmul(
      x,
      w,
      scales,
      kquant_type,
      true,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      s);
  return lora_add(base, z, b, bands, s);
}

} // namespace

#ifdef _METAL_

void KQuantMatmulLora::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  const auto& x = inputs[0];
  // A weight whose rows are dense binds as it is, at its own offset. Any
  // other view is copied to dense rows first. z and B bind as they are
  // when they are row-major.
  const mx::array w = kq_matrix_dense_gpu(inputs[1], s);
  const mx::array z = kq_row_major_gpu(inputs[2], s);
  const mx::array b = kq_row_major_gpu(inputs[3], s);
  auto& out = outputs[0];
  kq_check_weight_base(w, kquant_type_, "quantized_matmul_lora");
  out.set_data(mx::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  const int K = x.shape(-1);
  const int N = w.shape(0);
  const int R = b.shape(0);
  // x is dense rows, so its batch dims fold into M.
  const int M = static_cast<int>(x.size() / K);

  const int bm = 128;
  const int bk = lora_kernel_bk(kquant_type_, x.dtype());
  const bool aligned = N % kLoraBn == 0;
  const std::string kname = kq_kname_prefix(kquant_type_) + "qmm_t_nax_lora_" +
      kq_type_string(x.dtype()) + "_bm" + std::to_string(bm) + "_bn64_bk" +
      std::to_string(bk) + "_wm2_wn2_alN_" + (aligned ? "true" : "false");

  // With no band table, one band over every column and every rank.
  std::vector<int> bands = bands_;
  if (bands.empty()) {
    bands = {0, N, 0, R};
  }
  const int n_bands = static_cast<int>(bands.size() / 4);

  MTL::Size group_dims(32, 2, 2);
  MTL::Size grid_dims((N + kLoraBn - 1) / kLoraBn, (M + bm - 1) / bm, 1);
  if (lora_log()) {
    std::fprintf(
        stderr,
        "[kq qmm] %s M=%d N=%d K=%d R=%d bands=%d grid=%lu,%lu,%lu\n",
        kname.c_str(),
        M,
        N,
        K,
        R,
        n_bands,
        static_cast<unsigned long>(grid_dims.width),
        static_cast<unsigned long>(grid_dims.height),
        static_cast<unsigned long>(grid_dims.depth));
  }

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(x, c++);
  ce.set_input_array(z, c++);
  ce.set_input_array(b, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  ce.set_bytes(R, c++);
  ce.set_bytes(n_bands, c++);
  ce.set_vector_bytes(bands, c++);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantMatmulLora::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(std::string(kLoraOp) + " requires Metal.");
}

#endif

void KQuantMatmulLora::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      std::string(kLoraOp) + " CPU streams build the composed ops.");
}

// The kernel computes the function of the composition, so its derivatives
// are the composition's: with respect to x, z and B, with the quantized
// weight frozen as in quantized_matmul. z is built from x, A and the
// strength by stock ops before this node, so their derivatives follow
// from the one with respect to z.
namespace {

constexpr int kLoraArgOfInput[] = {0, -1, 1, 2};

void lora_check_argnums(const std::vector<int>& argnums, const char* what) {
  for (auto arg : argnums) {
    if (arg == 1) {
      throw std::invalid_argument(
          std::string("[mlx_kquant] quantized_matmul_lora ") + what +
          " wrt the quantized weight (the kquant base is frozen).");
    }
  }
}

} // namespace

std::vector<mx::array> KQuantMatmulLora::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  lora_check_argnums(argnums, "vjp: no gradient");
  auto fun = [&](const std::vector<mx::array>& in) {
    return std::vector<mx::array>{lora_composed(
        in[0], primals[1], in[1], in[2], kquant_type_, bands_, stream())};
  };
  auto vjps = mx::vjp(
                  fun,
                  std::vector<mx::array>{primals[0], primals[2], primals[3]},
                  cotangents)
                  .second;
  std::vector<mx::array> out;
  for (auto arg : argnums) {
    out.push_back(vjps[kLoraArgOfInput[arg]]);
  }
  return out;
}

std::vector<mx::array> KQuantMatmulLora::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
  lora_check_argnums(argnums, "jvp: no derivative");
  // A tangent of zero for the float inputs that are not in argnums.
  std::vector<mx::array> tan = {
      mx::zeros_like(primals[0], stream()),
      mx::zeros_like(primals[2], stream()),
      mx::zeros_like(primals[3], stream())};
  for (size_t i = 0; i < argnums.size(); i++) {
    tan[kLoraArgOfInput[argnums[i]]] = tangents[i];
  }
  auto fun = [&](const std::vector<mx::array>& in) {
    return std::vector<mx::array>{lora_composed(
        in[0], primals[1], in[1], in[2], kquant_type_, bands_, stream())};
  };
  return mx::jvp(
             fun,
             std::vector<mx::array>{primals[0], primals[2], primals[3]},
             tan)
      .second;
}

bool KQuantMatmulLora::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMatmulLora&>(other);
  return kquant_type_ == o.kquant_type_ && bands_ == o.bands_;
}

std::vector<mx::Shape> KQuantMatmulLora::output_shapes(
    const std::vector<mx::array>& inputs) {
  auto shape = inputs[0].shape();
  shape.back() = inputs[1].shape(0);
  return {shape};
}

mx::array quantized_matmul_lora(
    mx::array x,
    mx::array w,
    const std::string& kquant_type,
    mx::array lora_a,
    mx::array lora_b,
    const std::optional<mx::array>& strength,
    const std::vector<int>& bands,
    mx::StreamOrDevice s_) {
  const std::string op = kLoraOp;
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        op + " Unknown kquant_type: '" + kquant_type + "'.");
  }
  if (w.dtype() != mx::uint8 || w.ndim() != 2) {
    throw std::invalid_argument(
        op + " w must be 2-D uint8 (raw GGUF wire bytes).");
  }
  if (w.shape(-1) % codec->bytes_per_block != 0) {
    throw std::invalid_argument(
        op + " weight last dim (" + std::to_string(w.shape(-1)) +
        " bytes) is not a whole number of " +
        std::to_string(codec->bytes_per_block) + "-byte " + codec->name +
        " blocks.");
  }
  const int K =
      (w.shape(-1) / codec->bytes_per_block) * codec->weights_per_block;
  const int N = w.shape(0);
  if (x.ndim() < 1 || x.shape(-1) != K) {
    throw std::invalid_argument(
        op + " x last dim does not match the weight inner dim (" +
        std::to_string(K) + ").");
  }
  if (!mx::issubdtype(x.dtype(), mx::floating)) {
    throw std::invalid_argument(
        op + " Only real floating x dtypes are supported.");
  }
  if (lora_a.ndim() != 2 || lora_a.shape(0) != K ||
      !mx::issubdtype(lora_a.dtype(), mx::floating)) {
    throw std::invalid_argument(
        op + " lora_a must be a float array [K, R] with K " +
        std::to_string(K) + ".");
  }
  const int R = lora_a.shape(1);
  if (lora_b.ndim() != 2 || lora_b.shape(0) != R || lora_b.shape(1) != N ||
      !mx::issubdtype(lora_b.dtype(), mx::floating)) {
    throw std::invalid_argument(
        op + " lora_b must be a float array [R, N] with R " +
        std::to_string(R) + " and N " + std::to_string(N) + ".");
  }
  if (strength.has_value()) {
    const auto& st = *strength;
    const bool shape_ok =
        st.size() == 1 || (st.ndim() == 1 && st.shape(0) == R);
    if (!shape_ok || !mx::issubdtype(st.dtype(), mx::floating)) {
      throw std::invalid_argument(
          op +
          " strength must be a float array of one element or of shape "
          "[R] with R " +
          std::to_string(R) + ".");
    }
  }
  if (bands.size() % 4 != 0) {
    throw std::invalid_argument(
        op + " bands must hold rows of (col_start, col_end, z_start, rank).");
  }
  int prev_end = 0;
  for (size_t i = 0; i < bands.size(); i += 4) {
    const int cs = bands[i];
    const int ce = bands[i + 1];
    const int z0 = bands[i + 2];
    const int rb = bands[i + 3];
    if (cs < prev_end || ce <= cs || ce > N) {
      throw std::invalid_argument(
          op +
          " bands must be in column order, disjoint, not empty and "
          "within the " +
          std::to_string(N) + " columns: band " + std::to_string(i / 4) +
          " covers columns " + std::to_string(cs) + " to " +
          std::to_string(ce) + ".");
    }
    if (z0 < 0 || rb < 0 || rb > R - z0) {
      throw std::invalid_argument(
          op + " band " + std::to_string(i / 4) + " takes ranks " +
          std::to_string(z0) + " to " + std::to_string(z0) + " + " +
          std::to_string(rb) + " of " + std::to_string(R) + ".");
    }
    prev_end = ce;
  }
  auto s = mx::to_stream(s_);

  // float32 x runs in bfloat16, as in quantized_matmul.
  const mx::Dtype out_type =
      x.dtype() == mx::float32 ? mx::bfloat16 : x.dtype();
  const int64_t rows = K > 0 ? static_cast<int64_t>(x.size()) / K : 0;
  auto scales = mx::zeros({1}, mx::uint8, s);
  if (rows == 0 || N == 0 || R == 0) {
    // Nothing to add: the base product, with its shape and dtype.
    return quantized_matmul(
        x,
        w,
        scales,
        kquant_type,
        true,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        s);
  }

  const int bk = lora_kernel_bk(kquant_type, out_type);
  bool fused = s.device == mx::Device::gpu && nax_available() && lora_on() &&
      bk > 0 && rows >= lora_min_rows();
  if (bands.empty()) {
    fused = fused && R <= kLoraMaxRank;
  } else {
    fused = fused && bands.size() / 4 <= kLoraMaxBands;
    for (size_t i = 0; i < bands.size(); i += 4) {
      fused = fused && bands[i] % kLoraBn == 0 && bands[i + 3] <= kLoraMaxRank;
    }
  }

  // x gets a Contiguous node: the strides of an unevaluated array mean
  // nothing at build time. The weight and B go in as they are and the eval
  // looks at their strides.
  auto x_c = fused ? mx::contiguous(mx::astype(x, out_type, s), false, s)
                   : mx::astype(x, out_type, s);
  auto z = mx::matmul(x_c, mx::astype(lora_a, out_type, s), s);
  if (strength.has_value()) {
    z = mx::multiply(z, mx::astype(*strength, out_type, s), s);
  }
  auto b = mx::astype(lora_b, out_type, s);
  if (!fused) {
    return lora_composed(x, w, z, b, kquant_type, bands, s);
  }

  kq_check_weight_base_at_build(w, kquant_type, s, "quantized_matmul_lora");
  auto out_shape = x.shape();
  out_shape.back() = N;
  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantMatmulLora>(s, kquant_type, bands),
      {std::move(x_c), std::move(w), std::move(z), std::move(b)});
}

} // namespace mlx_kquant
