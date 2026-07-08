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

#else // !_METAL_

void KQDsaIndexerQat::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.dsa_indexer_qat] requires a Metal build.");
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

} // namespace mlx_kquant
