// MiniMax-H3 VSA block-sparse attention (see kq_vsa_nax.h for the kernel,
// ported from FastVideo). GPU only, on tensor-op (NAX) hardware; the op
// refuses a CPU stream and any other GPU when the call is built.
// FastVideo is Apache-2.0: see mlx_kquant/licenses/FastVideo-LICENSE.
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/allocator.h"
#include "mlx/ops.h"
#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel, kq_is_nax_available
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

const std::string kVsaOp = "[mlx_kquant.vsa_block_sparse] ";
constexpr int kVsaHeadDim = 128;
constexpr int kVsaTile = 64;

bool vsa_int_dtype(mx::Dtype dt) {
  return mx::issubdtype(dt, mx::integer);
}

} // namespace

#ifdef _METAL_

void KQuantVSABlockSparse::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  if (!kq_is_nax_available()) {
    throw std::runtime_error(
        kVsaOp + "requires tensor-op (NAX) hardware; check nax_available().");
  }
  const auto& q = inputs[0];
  const auto& block_idx = inputs[3];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  int heads = q.shape(1);
  int n_tiles = inputs[5].shape(0);
  int n_prefix = n_prefix_;
  int n_video = n_tiles - n_prefix;
  int k_max = block_idx.shape(2);
  float scale = scale_;

  // Prefix rows, pad rows and the rows of a query tile with no key tile
  // read zero: the kernel writes only the live rows it attends.
  auto& ce = mx::metal::get_command_encoder(s);
  const uint32_t n16 = static_cast<uint32_t>(out.nbytes() / 16);
  ce.set_compute_pipeline_state(kq_get_kernel(d, "kq_vsa_zero"));
  ce.set_output_array(out, 0);
  ce.set_bytes(n16, 1);
  ce.dispatch_threads(
      MTL::Size(n16, 1, 1), MTL::Size(n16 < 1024 ? n16 : 1024, 1, 1));

  std::string kname = "kq_vsa_nax_" + kq_type_string(q.dtype()) + "_" +
      std::to_string(tile_) + "_bq" + std::to_string(query_rows_);
  auto kernel = kq_get_kernel(d, kname);
  ce.set_compute_pipeline_state(kernel);
  for (int i = 0; i < 6; ++i) {
    ce.set_input_array(inputs[i], i);
  }
  ce.set_output_array(out, 6);
  ce.set_bytes(k_max, 7);
  ce.set_bytes(n_prefix, 8);
  ce.set_bytes(n_video, 9);
  ce.set_bytes(heads, 10);
  ce.set_bytes(scale, 11);
  MTL::Size group_dims((query_rows_ / 16) * 32, 1, 1);
  MTL::Size grid_dims(tile_ / query_rows_, n_video, heads);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantVSABlockSparse::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(kVsaOp + "requires a Metal build.");
}

#endif

void KQuantVSABlockSparse::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(kVsaOp + "has no CPU implementation.");
}

std::vector<mx::Shape> KQuantVSABlockSparse::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQuantVSABlockSparse::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantVSABlockSparse&>(other);
  return n_prefix_ == o.n_prefix_ && tile_ == o.tile_ &&
      query_rows_ == o.query_rows_ && scale_ == o.scale_;
}

mx::array vsa_block_sparse(
    mx::array q,
    mx::array k,
    mx::array v,
    mx::array block_idx,
    mx::array block_num,
    mx::array variable_block_sizes,
    int num_prefix_tiles,
    float scale,
    int tile_size,
    int query_rows,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  if (s.device != mx::Device::gpu) {
    throw std::invalid_argument(
        kVsaOp + "runs on the GPU only; the stream is on the CPU.");
  }
  if (q.ndim() != 3 || q.shape(1) < 1 || k.shape() != q.shape() ||
      v.shape() != q.shape()) {
    throw std::invalid_argument(
        kVsaOp +
        "q, k and v must share one shape [S_pad, heads, D] with heads >= 1.");
  }
  const int padded = q.shape(0);
  const int heads = q.shape(1);
  if (q.shape(2) != kVsaHeadDim) {
    throw std::invalid_argument(kVsaOp + "only head dim 128 is supported.");
  }
  if (q.dtype() != mx::bfloat16 || k.dtype() != mx::bfloat16 ||
      v.dtype() != mx::bfloat16) {
    throw std::invalid_argument(kVsaOp + "q, k and v must be bfloat16.");
  }
  if (tile_size != kVsaTile) {
    throw std::invalid_argument(kVsaOp + "only tile size 64 is supported.");
  }
  if (query_rows != 16 && query_rows != 32 && query_rows != 64) {
    throw std::invalid_argument(kVsaOp + "query_rows must be 16, 32 or 64.");
  }
  if (!std::isfinite(scale)) {
    throw std::invalid_argument(kVsaOp + "scale must be finite.");
  }
  if (variable_block_sizes.ndim() != 1 ||
      !vsa_int_dtype(variable_block_sizes.dtype())) {
    throw std::invalid_argument(
        kVsaOp + "variable_block_sizes must be a 1-D integer array.");
  }
  const int n_tiles = variable_block_sizes.shape(0);
  if (n_tiles < 1 || static_cast<int64_t>(n_tiles) * tile_size != padded) {
    throw std::invalid_argument(
        kVsaOp +
        "q must hold len(variable_block_sizes) * tile_size rows, got " +
        std::to_string(padded) + " rows for " + std::to_string(n_tiles) +
        " tiles.");
  }
  if (num_prefix_tiles < 0 || num_prefix_tiles > n_tiles) {
    throw std::invalid_argument(
        kVsaOp + "num_prefix_tiles must be in [0, number of tiles].");
  }
  const int n_video = n_tiles - num_prefix_tiles;
  if (block_idx.ndim() != 3 || block_idx.shape(0) != heads ||
      block_idx.shape(1) != n_video) {
    throw std::invalid_argument(
        kVsaOp + "block_idx must be [heads, video tiles, k].");
  }
  if (block_num.ndim() != 2 || block_num.shape(0) != heads ||
      block_num.shape(1) != n_video) {
    throw std::invalid_argument(
        kVsaOp + "block_num must be [heads, video tiles].");
  }
  if (!vsa_int_dtype(block_idx.dtype()) || !vsa_int_dtype(block_num.dtype())) {
    throw std::invalid_argument(
        kVsaOp + "block_idx and block_num must be integer arrays.");
  }
  // The kernel computes routing indices and the row offsets inside a 32-row
  // key block (up to 32 * heads * 128 elements) in 32 bits, and the zero fill
  // counts 16-byte words in 32 bits.
  if (block_idx.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      heads > std::numeric_limits<int>::max() / (32 * kVsaHeadDim) ||
      q.nbytes() / 16 > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(kVsaOp + "q or block_idx is too large.");
  }
  // Checked after the arguments, so the argument checks also run on GPUs
  // without NAX.
  if (!nax_available()) {
    throw std::runtime_error(
        kVsaOp +
        "requires a GPU with tensor-op (NAX) units, M5 or later; check "
        "nax_available().");
  }
  if (n_video == 0 || block_idx.shape(2) == 0) {
    return mx::zeros(q.shape(), q.dtype(), s);
  }
  auto as_i32 = [&](const mx::array& a) {
    return mx::contiguous(mx::astype(a, mx::int32, s), false, s);
  };
  // Routing wider than int32 is clipped to [lo, hi] before the cast, so a
  // value out of range stays out of range instead of wrapping into it.
  auto routing_i32 = [&](mx::array a, int lo, int hi) {
    if (a.dtype() == mx::int64) {
      a = mx::clip(
          a,
          mx::array(static_cast<int64_t>(lo), mx::int64),
          mx::array(static_cast<int64_t>(hi), mx::int64),
          s);
    } else if (a.dtype() == mx::uint32 || a.dtype() == mx::uint64) {
      a = mx::minimum(a, mx::array(hi, a.dtype()), s);
    }
    return as_i32(a);
  };
  const int k_max = block_idx.shape(2);
  std::vector<mx::array> inputs = {
      mx::contiguous(q, false, s),
      mx::contiguous(k, false, s),
      mx::contiguous(v, false, s),
      routing_i32(block_idx, -1, n_tiles),
      routing_i32(block_num, 0, k_max),
      as_i32(variable_block_sizes)};
  auto shape = q.shape();
  auto prim = std::make_shared<KQuantVSABlockSparse>(
      s, num_prefix_tiles, tile_size, query_rows, scale);
  return mx::array(
      std::move(shape), mx::bfloat16, std::move(prim), std::move(inputs));
}

} // namespace mlx_kquant
