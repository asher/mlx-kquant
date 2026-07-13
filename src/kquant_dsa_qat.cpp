// KQDsaIndexerQat primitive: the DeepSeek-V4-Flash indexer activation QAT
// round-trip (128-wide Hadamard + per-32-block FP4-E2M1 round-trip) fused
// into one kernel. Applies to indexer queries and compressed-pool rows;
// bit-compatible with the mx.hadamard_transform + fp4-core graph it
// replaces (see kq_dsa_qat.h). Inference-only (no CPU eval).
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/ops.h" // contiguous
#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

#ifdef _METAL_

void KQDsaIndexerQat::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  const auto& x = inputs[0];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  constexpr int rows_per_tg = 32;
  const int rows = int(x.size() / 128);
  // mlx's hadamard_transform default scale for n=128.
  const float scale = 1.0f / std::sqrt(128.0f);

  const std::string kname = "kq_dsa_indexer_qat_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname, kname, {});
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  ce.set_input_array(x, 0);
  ce.set_output_array(out, 1);
  ce.set_bytes(rows, 2);
  ce.set_bytes(scale, 3);

  MTL::Size group_dims(256, 1, 1);
  MTL::Size grid_dims((rows + rows_per_tg - 1) / rows_per_tg, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQDsaIndexerQatQuant::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  const auto& x = inputs[0];
  auto& codes = outputs[0];
  auto& scales = outputs[1];
  codes.set_data(mx::allocator::malloc(codes.nbytes()));
  scales.set_data(mx::allocator::malloc(scales.nbytes()));

  constexpr int rows_per_tg = 32;
  const int rows = int(x.size() / 128);
  // mlx's hadamard_transform default scale for n=128.
  const float scale = 1.0f / std::sqrt(128.0f);

  const std::string kname =
      "kq_dsa_indexer_qat_quant_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname, kname, {});
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  ce.set_input_array(x, 0);
  ce.set_output_array(codes, 1);
  ce.set_output_array(scales, 2);
  ce.set_bytes(rows, 3);
  ce.set_bytes(scale, 4);

  MTL::Size group_dims(256, 1, 1);
  MTL::Size grid_dims((rows + rows_per_tg - 1) / rows_per_tg, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQDsaKvQat::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  const auto& x = inputs[0];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const int D = x.shape(-1);
  const int rows = int(x.size() / D);
  const int n_rot = n_rot_;

  const std::string kname = "kq_dsa_kv_qat_" + kq_type_string(x.dtype());
  auto kernel = kq_get_kernel(d, kname, kname, {});
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  ce.set_input_array(x, 0);
  ce.set_output_array(out, 1);
  ce.set_bytes(D, 2);
  ce.set_bytes(n_rot, 3);

  MTL::Size group_dims(256, 1, 1);
  MTL::Size grid_dims(rows, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQDsaIndexerQat::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_indexer_qat] requires a Metal build.");
}

void KQDsaIndexerQatQuant::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_indexer_qat_quant] requires a Metal build.");
}

void KQDsaKvQat::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.dsa_kv_qat] requires a Metal build.");
}

#endif

void KQDsaIndexerQat::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_indexer_qat] has no CPU implementation.");
}

std::vector<mx::Shape> KQDsaIndexerQat::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQDsaIndexerQat::is_equivalent(const mx::Primitive&) const {
  return true;
}

void KQDsaIndexerQatQuant::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_indexer_qat_quant] has no CPU implementation.");
}

std::vector<mx::Shape> KQDsaIndexerQatQuant::output_shapes(
    const std::vector<mx::array>& inputs) {
  auto scales_shape = inputs[0].shape();
  scales_shape.back() = 4;
  return {inputs[0].shape(), std::move(scales_shape)};
}

bool KQDsaIndexerQatQuant::is_equivalent(const mx::Primitive&) const {
  return true;
}

void KQDsaKvQat::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_kv_qat] has no CPU implementation.");
}

std::vector<mx::Shape> KQDsaKvQat::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQDsaKvQat::is_equivalent(const mx::Primitive& other) const {
  return n_rot_ == static_cast<const KQDsaKvQat&>(other).n_rot_;
}

mx::array dsa_kv_qat(mx::array x, int n_rot, mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (x.ndim() < 1) {
    throw std::invalid_argument(
        "[mlx_kquant.dsa_kv_qat] x must have rank >= 1.");
  }
  const int D = x.shape(-1);
  if (n_rot < 0 || n_rot > D || (D - n_rot) % 64 != 0 || D == n_rot) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_kv_qat] needs 0 <= n_rot < D and "
        << "(D - n_rot) % 64 == 0 (64-wide fp8 blocks), got D = " << D
        << ", n_rot = " << n_rot << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != mx::float16 && x.dtype() != mx::bfloat16 &&
      x.dtype() != mx::float32) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_kv_qat] expected fp16/bf16/fp32 input, got "
        << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto xc = mx::contiguous(x, false, s);
  auto out_shape = xc.shape();
  std::vector<mx::array> inputs = {std::move(xc)};
  return mx::array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<KQDsaKvQat>(s, n_rot),
      std::move(inputs));
}

mx::array dsa_indexer_qat(mx::array x, mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (x.ndim() < 1 || x.shape(-1) != 128) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_indexer_qat] expected a trailing dim of 128 "
        << "(the V4 indexer head dim), got shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != mx::float16 && x.dtype() != mx::bfloat16 &&
      x.dtype() != mx::float32) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_indexer_qat] expected fp16/bf16/fp32 input, got "
        << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Pre-eval flags are unreliable; contiguous is a no-op at eval when the
  // input already is.
  auto xc = mx::contiguous(x, false, s);

  auto out_shape = xc.shape();
  std::vector<mx::array> inputs = {std::move(xc)};
  return mx::array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<KQDsaIndexerQat>(s),
      std::move(inputs));
}

std::vector<mx::array> dsa_indexer_qat_quant(
    mx::array x,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (x.ndim() < 1 || x.shape(-1) != 128) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_indexer_qat_quant] expected a trailing dim of "
        << "128 (the V4 indexer head dim), got shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != mx::float16 && x.dtype() != mx::bfloat16 &&
      x.dtype() != mx::float32) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dsa_indexer_qat_quant] expected fp16/bf16/fp32 "
        << "input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Pre-eval flags are unreliable; contiguous is a no-op at eval when the
  // input already is.
  auto xc = mx::contiguous(x, false, s);

  auto scales_shape = xc.shape();
  scales_shape.back() = 4;
  std::vector<mx::Shape> shapes = {xc.shape(), std::move(scales_shape)};
  std::vector<mx::Dtype> dtypes = {mx::int8, mx::float32};
  return mx::array::make_arrays(
      std::move(shapes),
      dtypes,
      std::make_shared<KQDsaIndexerQatQuant>(s),
      {std::move(xc)});
}

} // namespace mlx_kquant
