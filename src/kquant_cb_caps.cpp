// Runtime access to MLX's command-buffer split caps (max ops and MB per
// buffer). The MLX_MAX_*_PER_BUFFER env knobs latch at Metal device init,
// but decode wants coarse buffers (submission otherwise blocks on
// in-flight drain) while deep prefill needs fine ones (a giant buffer
// holds every layer's transients live at once and can exhaust GPU
// memory), so serving flips the caps per phase.
//
// This is the only TU that includes device.h with access relaxed. The
// pinned mlx wheel ships the exact header this compiles against, so the
// member offsets match the wheel's libmlx; set_cb_caps still refuses to
// write when the current values read implausible, as a layout tripwire.
#include <stdexcept>
#include <utility>

#ifdef _METAL_
#define private public
#include "mlx/backend/metal/device.h"
#undef private
#endif

#include "mlx/device.h"

namespace mlx_kquant {

#ifdef _METAL_

namespace {

bool plausible(int ops, int mb) {
  return ops > 0 && ops <= (1 << 30) && mb > 0 && mb <= (1 << 30);
}

} // namespace

std::pair<int, int> get_cb_caps() {
  auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
  auto [ops, mb] = d.get_max_ops_mb_per_buffer();
  return {ops, mb};
}

std::pair<int, int> set_cb_caps(int max_ops, int max_mb) {
  if (!plausible(max_ops, max_mb)) {
    throw std::invalid_argument(
        "[mlx_kquant.set_cb_caps] caps must be in [1, 2^30].");
  }
  auto& d = mlx::core::metal::device(mlx::core::Device::gpu);
  auto [prev_ops, prev_mb] = d.get_max_ops_mb_per_buffer();
  if (!plausible(prev_ops, prev_mb)) {
    throw std::runtime_error(
        "[mlx_kquant.set_cb_caps] current caps read implausible; mlx "
        "device layout drift, refusing to write.");
  }
  d.max_ops_per_buffer_ = max_ops;
  d.max_mb_per_buffer_ = max_mb;
  return {prev_ops, prev_mb};
}

#else // !_METAL_

std::pair<int, int> get_cb_caps() {
  throw std::runtime_error("[mlx_kquant.get_cb_caps] requires Metal.");
}

std::pair<int, int> set_cb_caps(int, int) {
  throw std::runtime_error("[mlx_kquant.set_cb_caps] requires Metal.");
}

#endif

} // namespace mlx_kquant
