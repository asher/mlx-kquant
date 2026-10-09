// 3x3 convolution, stride 1, padding 1, channels last (see kq_conv_nax.h
// for the kernel).
//
//   y = conv(x, w) + bias
//
// On a NAX GPU, for bfloat16 with C a multiple of 16, from kConvMinRows
// output pixels, one dispatch computes y as an implicit GEMM with f32 sums,
// rounded once. Every other case builds the same convolution from MLX's
// conv_general on the flipped kernel (its implicit GEMM, f32 sums, no
// Winograd transform) and a stock add of the bias.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kquant.h"
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

const char* kConvOp = "[mlx_kquant.conv2d_3x3]";

// Output pixels from which the kernel runs under the default mode.
constexpr int64_t kConvMinRows = 1024;
// The kernel holds element offsets of x and y in an int.
constexpr int64_t kConvMaxElems = int64_t{1} << 31;

bool conv_tune_live() {
  static const bool live = kq_env_lever("KQ_TUNE_LIVE", 0) != 0;
  return live;
}

// KQ_CONV_NAX: 0 = the composition on every call, 2 = the kernel at every
// pixel count, unset, 1 or any other value = from kConvMinRows pixels.
int conv_mode() {
  static const int cached = kq_env_lever("KQ_CONV_NAX", 1);
  return conv_tune_live() ? kq_env_lever("KQ_CONV_NAX", 1) : cached;
}

bool conv_log() {
  static const bool cached = std::getenv("KQ_QMM_LOG") != nullptr;
  return conv_tune_live() ? std::getenv("KQ_QMM_LOG") != nullptr : cached;
}

// The composition: MLX's convolution on the flipped kernel with flip set,
// which on the GPU takes the implicit GEMM and not the Winograd transform,
// then the bias.
mx::array conv_composed(
    const mx::array& x,
    const mx::array& w,
    const std::optional<mx::array>& bias,
    mx::Stream s) {
  auto rev = mx::array({2, 1, 0});
  auto wf = mx::take(mx::take(w, rev, 1, s), rev, 2, s);
  auto y = mx::conv_general(
      x, wf, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, 1, true, s);
  if (bias.has_value()) {
    y = mx::add(y, *bias, s);
  }
  return y;
}

} // namespace

#ifdef _METAL_

void KQuantConv3x3::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  const mx::array x = kq_row_major_gpu(inputs[0], s);
  const mx::array w = kq_row_major_gpu(inputs[1], s);
  const bool has_bias = inputs.size() > 2;
  const mx::array b = has_bias ? kq_row_major_gpu(inputs[2], s) : w;
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  const int H = x.shape(1);
  const int W = x.shape(2);
  const int C = x.shape(3);
  const int O = w.shape(0);
  const int M = x.shape(0) * H * W;
  const int hb = has_bias ? 1 : 0;

  // Column tile: 128, or 64 where 64 pads O less. KQ_CONV_TILE
  // (bm,bn,wm,wn) pins a tile the metallib carries, read once per process
  // unless KQ_TUNE_LIVE=1.
  int bm = 128, bn = 128, wm = 4, wn = 4;
  const int pad128 = (O + 127) / 128 * 128;
  const int pad64 = (O + 63) / 64 * 64;
  if (pad64 < pad128) {
    bn = 64;
    wn = 2;
  }
  static const char* tile_cached = std::getenv("KQ_CONV_TILE");
  const char* tile_env =
      conv_tune_live() ? std::getenv("KQ_CONV_TILE") : tile_cached;
  if (tile_env != nullptr && tile_env[0] != '\0') {
    if (std::sscanf(tile_env, "%d,%d,%d,%d", &bm, &bn, &wm, &wn) != 4) {
      throw std::invalid_argument(
          std::string(kConvOp) + " KQ_CONV_TILE=" + tile_env +
          " is not bm,bn,wm,wn.");
    }
  }
  const std::string kname = "kq_conv3x3_nax_" + kq_type_string(x.dtype()) +
      "_bm" + std::to_string(bm) + "_bn" + std::to_string(bn) + "_wm" +
      std::to_string(wm) + "_wn" + std::to_string(wn);
  if (tile_env != nullptr && tile_env[0] != '\0') {
    kq_require_kernel(d, std::string("KQ_CONV_TILE=") + tile_env, kname);
  }

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims(
      (O + bn - 1) / bn, (static_cast<size_t>(M) + bm - 1) / bm, 1);
  if (conv_log()) {
    std::fprintf(
        stderr,
        "[kq qmm] %s M=%d N=%d K=%d grid=%lu,%lu,%lu\n",
        kname.c_str(),
        M,
        O,
        9 * C,
        static_cast<unsigned long>(grid_dims.width),
        static_cast<unsigned long>(grid_dims.height),
        static_cast<unsigned long>(grid_dims.depth));
  }

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  int c = 0;
  ce.set_input_array(x, c++);
  ce.set_input_array(w, c++);
  ce.set_input_array(b, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(H, c++);
  ce.set_bytes(W, c++);
  ce.set_bytes(C, c++);
  ce.set_bytes(O, c++);
  ce.set_bytes(M, c++);
  ce.set_bytes(hb, c++);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantConv3x3::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(std::string(kConvOp) + " requires Metal.");
}

#endif

void KQuantConv3x3::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      std::string(kConvOp) + " CPU streams build the composed ops.");
}

namespace {

std::optional<mx::array> conv_bias_of(const std::vector<mx::array>& in) {
  return in.size() > 2 ? std::optional<mx::array>(in[2]) : std::nullopt;
}

} // namespace

// The kernel computes the function of the composition, so its transforms
// are the composition's.
std::vector<mx::array> KQuantConv3x3::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  auto fun = [&](const std::vector<mx::array>& in) {
    return std::vector<mx::array>{
        conv_composed(in[0], in[1], conv_bias_of(in), stream())};
  };
  auto vjps = mx::vjp(fun, primals, cotangents).second;
  std::vector<mx::array> out;
  for (auto arg : argnums) {
    out.push_back(vjps[arg]);
  }
  return out;
}

std::vector<mx::array> KQuantConv3x3::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
  auto fun = [&](const std::vector<mx::array>& in) {
    std::vector<mx::array> full = primals;
    for (size_t i = 0; i < argnums.size(); i++) {
      full[argnums[i]] = in[i];
    }
    return std::vector<mx::array>{
        conv_composed(full[0], full[1], conv_bias_of(full), stream())};
  };
  std::vector<mx::array> at;
  for (auto arg : argnums) {
    at.push_back(primals[arg]);
  }
  return mx::jvp(fun, at, tangents).second;
}

std::pair<std::vector<mx::array>, std::vector<int>> KQuantConv3x3::vmap(
    const std::vector<mx::array>& inputs,
    const std::vector<int>& axes) {
  auto fun = [&](const std::vector<mx::array>& in) {
    return std::vector<mx::array>{
        conv_composed(in[0], in[1], conv_bias_of(in), stream())};
  };
  return {mx::vmap(fun, axes, {0})(inputs), {0}};
}

bool KQuantConv3x3::is_equivalent(const mx::Primitive&) const {
  return true;
}

std::vector<mx::Shape> KQuantConv3x3::output_shapes(
    const std::vector<mx::array>& inputs) {
  auto shape = inputs[0].shape();
  shape.back() = inputs[1].shape(0);
  return {shape};
}

mx::array conv2d_3x3(
    mx::array x,
    mx::array w,
    const std::optional<mx::array>& bias,
    mx::StreamOrDevice s_) {
  const std::string op = kConvOp;
  if (x.ndim() != 4 || !mx::issubdtype(x.dtype(), mx::floating)) {
    throw std::invalid_argument(
        op + " x must be a float array [B, H, W, C] (channels last).");
  }
  const int C = x.shape(3);
  if (w.ndim() != 4 || w.shape(1) != 3 || w.shape(2) != 3 || w.shape(3) != C ||
      !mx::issubdtype(w.dtype(), mx::floating)) {
    throw std::invalid_argument(
        op + " w must be a float array [O, 3, 3, C] with C " +
        std::to_string(C) + ".");
  }
  const int O = w.shape(0);
  if (bias.has_value() &&
      (bias->ndim() != 1 || bias->shape(0) != O ||
       !mx::issubdtype(bias->dtype(), mx::floating))) {
    throw std::invalid_argument(
        op + " bias must be a float array [O] with O " + std::to_string(O) +
        ".");
  }
  auto s = mx::to_stream(s_);
  const mx::Dtype dt = x.dtype();
  auto wc = mx::astype(w, dt, s);
  std::optional<mx::array> bc;
  if (bias.has_value()) {
    bc = mx::astype(*bias, dt, s);
  }
  const int64_t pixels = static_cast<int64_t>(x.size()) / std::max(C, 1);
  const int mode = conv_mode();
  const bool fused = s.device == mx::Device::gpu && nax_available() &&
      mode != 0 && dt == mx::bfloat16 && C > 0 && C % 16 == 0 && O > 0 &&
      static_cast<int64_t>(x.size()) < kConvMaxElems &&
      pixels * O < kConvMaxElems && (mode == 2 || pixels >= kConvMinRows);
  if (!fused) {
    return conv_composed(x, wc, bc, s);
  }
  auto out_shape = x.shape();
  out_shape.back() = O;
  std::vector<mx::array> in = {std::move(x), std::move(wc)};
  if (bc.has_value()) {
    in.push_back(*bc);
  }
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantConv3x3>(s),
      std::move(in));
}

} // namespace mlx_kquant
