// Quantized matmul with a residual epilogue (see kq_quantized_nax_resid.h
// for the kernel).
//
//   y = residual + gate * (x @ W^T)
//
// On a NAX GPU, for the codecs with a kernel (q4_k, q5_k, q6_k and q8_0 on
// bfloat16), from kResidMinRows rows, one dispatch computes the product
// and stores residual + gate * product, rounded once. Every other case
// builds the same sum from quantized_matmul and stock ops.
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

const char* kResidOp = "[mlx_kquant.quantized_matmul_residual]";

// Column tile of the kernel.
constexpr int kResidBn = 64;
// Sizes the op refuses. The kernel offsets a row tile by a 32-bit product
// of 64 rows and a row length (K or N), which wraps from 2^25, and holds
// the row count of a batch in an int.
constexpr int64_t kResidMaxCols = int64_t{1} << 25;
constexpr int64_t kResidMaxRows = int64_t{1} << 31;

// K tile of the kernel for a codec and an activation dtype, 0 where the op
// builds the composition.
int resid_kernel_bk(const std::string& t, mx::Dtype dt) {
  if (dt != mx::bfloat16) {
    return 0;
  }
  if (t == "q4_k" || t == "q5_k" || t == "q6_k") {
    return 128;
  }
  if (t == "q8_0") {
    return 64;
  }
  return 0;
}

bool resid_tune_live() {
  static const bool live = kq_env_lever("KQ_TUNE_LIVE", 0) != 0;
  return live;
}

// KQ_QMM_RESID: 0 = the composition on every call, 2 = the kernel wherever
// it can run, unset, 1 or any other value = the kernel from kResidMinRows
// rows.
int resid_mode() {
  static const int cached = kq_env_lever("KQ_QMM_RESID", 1);
  return resid_tune_live() ? kq_env_lever("KQ_QMM_RESID", 1) : cached;
}

// Rows from which the kernel runs under the default mode.
constexpr int64_t kResidMinRows = 1024;

bool resid_log() {
  static const bool cached = std::getenv("KQ_QMM_LOG") != nullptr;
  return resid_tune_live() ? std::getenv("KQ_QMM_LOG") != nullptr : cached;
}

// The sum from quantized_matmul and stock ops. The op builds this where it
// has no kernel, and the derivatives of the kernel are the derivatives of
// this. residual and gate are in the output dtype.
mx::array resid_composed(
    const mx::array& x,
    const mx::array& w,
    const mx::array& residual,
    const std::optional<mx::array>& gate,
    const std::string& kquant_type,
    mx::Stream s) {
  auto scales = mx::zeros({1}, mx::uint8, s);
  auto y = quantized_matmul(
      x,
      w,
      scales,
      kquant_type,
      true,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      s);
  if (gate.has_value()) {
    y = mx::multiply(*gate, y, s);
  }
  return mx::add(residual, y, s);
}

} // namespace

#ifdef _METAL_

void KQuantMatmulResid::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  const mx::array x =
      kq_x_aligned_gpu(inputs[0], s, "quantized_matmul_residual x");
  // A weight whose rows are dense binds as it is, at its own offset. Any
  // other view is copied to dense rows first. The residual and the gate
  // bind as they are when they are row-major.
  const mx::array w = kq_matrix_dense_gpu(inputs[1], s);
  const mx::array r = kq_row_major_gpu(inputs[2], s);
  auto& out = outputs[0];
  kq_check_weight_base(w, kquant_type_, "quantized_matmul_residual");
  out.set_data(mx::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  const bool gated = inputs.size() > 3;
  const mx::array g = gated ? kq_row_major_gpu(inputs[3], s) : r;

  const int K = x.shape(-1);
  const int N = w.shape(0);
  // A gate per batch: the batch dims run in the grid and M is the rows of
  // one batch. Else every row folds into M.
  const int64_t rows = static_cast<int64_t>(x.size() / K);
  const bool per_batch = gated && g.size() > static_cast<size_t>(N);
  const int B = per_batch ? static_cast<int>(g.size() / N) : 1;
  const int M = static_cast<int>(rows / B);
  const int g_stride = per_batch ? N : 0;

  const int max_k = kq_nax_large_max_k(kquant_type_);
  const int bm = max_k > 0 && K > max_k ? 64 : 128;
  const int bk = resid_kernel_bk(kquant_type_, x.dtype());
  const bool aligned = N % kResidBn == 0;
  const std::string kname = kq_kname_prefix(kquant_type_) + "qmm_t_nax_resid_" +
      kq_type_string(x.dtype()) + "_bm" + std::to_string(bm) + "_bn64_bk" +
      std::to_string(bk) + "_wm2_wn2_alN_" + (aligned ? "true" : "false") +
      "_gate_" + (gated ? "true" : "false");

  MTL::Size group_dims(32, 2, 2);
  MTL::Size grid_dims(
      (N + kResidBn - 1) / kResidBn,
      (static_cast<size_t>(M) + bm - 1) / bm,
      static_cast<size_t>(B));
  if (resid_log()) {
    std::fprintf(
        stderr,
        "[kq qmm] %s M=%d N=%d K=%d B=%d grid=%lu,%lu,%lu tg=%lu,%lu,%lu xa=%d\n",
        kname.c_str(),
        M,
        N,
        K,
        B,
        static_cast<unsigned long>(grid_dims.width),
        static_cast<unsigned long>(grid_dims.height),
        static_cast<unsigned long>(grid_dims.depth),
        static_cast<unsigned long>(group_dims.width),
        static_cast<unsigned long>(group_dims.height),
        static_cast<unsigned long>(group_dims.depth),
        kq_x_align(x));
  }

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(x, c++);
  ce.set_input_array(r, c++);
  ce.set_input_array(g, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  ce.set_bytes(g_stride, c++);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantMatmulResid::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(std::string(kResidOp) + " requires Metal.");
}

#endif

void KQuantMatmulResid::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      std::string(kResidOp) + " CPU streams build the composed ops.");
}

// The kernel computes the function of the composition, so its transforms
// are the composition's, with the quantized weight frozen as in
// quantized_matmul.
namespace {

std::optional<mx::array> resid_gate_of(const std::vector<mx::array>& in) {
  return in.size() > 3 ? std::optional<mx::array>(in[3]) : std::nullopt;
}

} // namespace

std::vector<mx::array> KQuantMatmulResid::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  for (auto arg : argnums) {
    if (arg == 1) {
      throw std::invalid_argument(
          "[mlx_kquant] quantized_matmul_residual vjp: no gradient wrt the "
          "quantized weight (the kquant base is frozen).");
    }
  }
  // The inputs other than the weight, in their order.
  std::vector<mx::array> at;
  for (size_t i = 0; i < primals.size(); i++) {
    if (i != 1) {
      at.push_back(primals[i]);
    }
  }
  auto fun = [&](const std::vector<mx::array>& in) {
    std::optional<mx::array> g =
        in.size() > 2 ? std::optional<mx::array>(in[2]) : std::nullopt;
    return std::vector<mx::array>{
        resid_composed(in[0], primals[1], in[1], g, kquant_type_, stream())};
  };
  auto vjps = mx::vjp(fun, at, cotangents).second;
  std::vector<mx::array> out;
  for (auto arg : argnums) {
    out.push_back(vjps[arg == 0 ? 0 : arg - 1]);
  }
  return out;
}

// jvp and vmap are the transforms of the composition. A tangent or a batch
// of x or of the weight reaches quantized_matmul, which has neither
// transform and refuses by its own name, as it does where the op builds
// the composition.
std::vector<mx::array> KQuantMatmulResid::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
  auto fun = [&](const std::vector<mx::array>& in) {
    std::vector<mx::array> full = primals;
    for (size_t i = 0; i < argnums.size(); i++) {
      full[argnums[i]] = in[i];
    }
    return std::vector<mx::array>{resid_composed(
        full[0],
        full[1],
        full[2],
        resid_gate_of(full),
        kquant_type_,
        stream())};
  };
  std::vector<mx::array> at;
  for (auto arg : argnums) {
    at.push_back(primals[arg]);
  }
  return mx::jvp(fun, at, tangents).second;
}

std::pair<std::vector<mx::array>, std::vector<int>> KQuantMatmulResid::vmap(
    const std::vector<mx::array>& inputs,
    const std::vector<int>& axes) {
  auto fun = [&](const std::vector<mx::array>& in) {
    return std::vector<mx::array>{resid_composed(
        in[0], in[1], in[2], resid_gate_of(in), kquant_type_, stream())};
  };
  return {mx::vmap(fun, axes, {0})(inputs), {0}};
}

bool KQuantMatmulResid::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMatmulResid&>(other);
  return kquant_type_ == o.kquant_type_;
}

std::vector<mx::Shape> KQuantMatmulResid::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[2].shape()};
}

mx::array quantized_matmul_residual(
    mx::array x,
    mx::array w,
    const std::string& kquant_type,
    mx::array residual,
    const std::optional<mx::array>& gate,
    mx::StreamOrDevice s_) {
  const std::string op = kResidOp;
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        op + " Unknown kquant_type: '" + kquant_type + "'.");
  }
  require_matmul_codec(*codec, op);
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
  auto out_shape = x.shape();
  out_shape.back() = N;
  if (residual.shape() != out_shape ||
      !mx::issubdtype(residual.dtype(), mx::floating)) {
    throw std::invalid_argument(
        op +
        " residual must be a float array of the output shape, x's "
        "shape with N " +
        std::to_string(N) + " last.");
  }
  // A gate of [N], or of x's batch dims, then 1, then N.
  if (gate.has_value()) {
    const auto& g = *gate;
    mx::Shape batch_shape = out_shape;
    bool ok = mx::issubdtype(g.dtype(), mx::floating);
    if (g.ndim() == 1) {
      ok = ok && g.shape(0) == N;
    } else if (x.ndim() >= 2) {
      batch_shape[batch_shape.size() - 2] = 1;
      ok = ok && g.shape() == batch_shape;
    } else {
      ok = false;
    }
    if (!ok) {
      throw std::invalid_argument(
          op +
          " gate must be a float array of shape [N] or of x's batch "
          "dims, then 1, then N, with N " +
          std::to_string(N) + ".");
    }
  }
  if (K >= kResidMaxCols || N >= kResidMaxCols) {
    throw std::invalid_argument(
        op + " K and N must each be below " + std::to_string(kResidMaxCols) +
        " (2^25): K " + std::to_string(K) + ", N " + std::to_string(N) + ".");
  }
  const int64_t rows = K > 0 ? static_cast<int64_t>(x.size()) / K : 0;
  if (rows >= kResidMaxRows) {
    throw std::invalid_argument(
        op + " x must hold fewer than " + std::to_string(kResidMaxRows) +
        " (2^31) rows: it holds " + std::to_string(rows) + ".");
  }
  auto s = mx::to_stream(s_);

  // float32 x runs in bfloat16, as in quantized_matmul. The residual and
  // the gate run in the output dtype.
  const mx::Dtype out_type =
      x.dtype() == mx::float32 ? mx::bfloat16 : x.dtype();
  auto r = mx::astype(residual, out_type, s);
  std::optional<mx::array> g;
  if (gate.has_value()) {
    g = mx::astype(*gate, out_type, s);
  }

  const int bk = resid_kernel_bk(kquant_type, out_type);
  const int mode = resid_mode();
  const bool fused = s.device == mx::Device::gpu && nax_available() &&
      mode != 0 && bk > 0 && K > 0 && N > 0 && K % bk == 0 &&
      (mode == 2 || rows >= kResidMinRows);
  if (!fused) {
    return resid_composed(x, w, r, g, kquant_type, s);
  }

  kq_check_weight_base_at_build(w, kquant_type, s, "quantized_matmul_residual");
  std::vector<mx::array> in = {
      mx::astype(x, out_type, s), std::move(w), std::move(r)};
  if (g.has_value()) {
    in.push_back(*g);
  }
  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantMatmulResid>(s, kquant_type),
      std::move(in));
}

} // namespace mlx_kquant
