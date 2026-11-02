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
const std::string kVsaTileSumsOp = "[mlx_kquant.vsa_attention] tile sums: ";
constexpr int kVsaHeadDim = 128;
constexpr int kVsaTile = 64;
// Default query rows: 32 below this many video tiles, 64 from it. On M5 Max
// 32 is faster at 32 video tiles, 64 ties at 108 and wins at 660, and 70
// is the midpoint.
constexpr int kVsaRows64MinVideoTiles = 70;

bool vsa_int_dtype(mx::Dtype dt) {
  return mx::issubdtype(dt, mx::integer);
}

// The identity, on the buffer of its input. A shapeless replay calls
// output_shapes, which refuses, so a shapeless compiled function cannot hold
// a graph that passes through this node: it would replay the traced geometry
// at other shapes.
class KQuantVSAFixedShapes : public mx::UnaryPrimitive {
 public:
  explicit KQuantVSAFixedShapes(mx::Stream stream)
      : mx::UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    out.copy_shared_buffer(inputs[0]);
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    out.copy_shared_buffer(inputs[0]);
  }

  const char* name() const override {
    return "KQuantVSAFixedShapes";
  }
  bool is_equivalent(const mx::Primitive&) const override {
    return true;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    throw std::invalid_argument(
        "[mlx_kquant.vsa_attention] A shapeless compiled function cannot hold "
        "this call: it would replay the traced geometry at other shapes. "
        "Compile one function per geometry, with fixed shapes.");
  }
};

// The float32 sum of each 64-row tile of a tiled bfloat16 array, [tiles * 64,
// H, D] to [tiles, H, D], through MLX's col_reduce_looped with a bfloat16
// input (kq_vsa_pool.metal). For x.astype(float32).reshape(tiles, 64, H,
// D).sum(1) MLX runs that template at float32 with the arguments eval_gpu
// encodes, so the sums have MLX's bits.
class KQuantVSATileSums : public mx::UnaryPrimitive {
 public:
  explicit KQuantVSATileSums(mx::Stream stream) : mx::UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error(kVsaTileSumsOp + "has no CPU kernel.");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override;

  const char* name() const override {
    return "KQuantVSATileSums";
  }
  bool is_equivalent(const mx::Primitive&) const override {
    return true;
  }
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    const auto& x = inputs[0];
    return {{x.shape(0) / kVsaTile, x.shape(1), x.shape(2)}};
  }
};

#ifdef _METAL_

void KQuantVSATileSums::eval_gpu(
    const std::vector<mx::array>& inputs,
    mx::array& out) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  const auto& x = inputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // strided_reduce_looped's arguments for this column reduce: the 64 rows of
  // a tile, H * D apart, summed for each of the tiles, one grid row a tile.
  const size_t reduction_size = kVsaTile;
  const int64_t reduction_stride = int64_t{x.shape(1)} * x.shape(2);
  const int n_tiles = x.shape(0) / kVsaTile;
  const mx::Shape shape = {n_tiles};
  const mx::Strides strides = {kVsaTile * reduction_stride};
  const int ndim = 1;
  const mx::Shape reduce_shape = {kVsaTile};
  const mx::Strides reduce_strides = {reduction_stride};
  const int reduce_ndim = 1;
  const size_t non_col_reductions = 1;
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kq_get_kernel(d, "kq_vsa_tile_sums_bfloat16"));
  ce.set_input_array(x, 0);
  ce.set_output_array(out, 1);
  ce.set_bytes(reduction_size, 2);
  ce.set_bytes(reduction_stride, 3);
  ce.set_vector_bytes(shape, 4);
  ce.set_vector_bytes(strides, 5);
  ce.set_bytes(ndim, 6);
  ce.set_vector_bytes(reduce_shape, 7);
  ce.set_vector_bytes(reduce_strides, 8);
  ce.set_bytes(reduce_ndim, 9);
  ce.set_bytes(non_col_reductions, 10);
  // 256 threads per 32 columns (BN), as MLX dispatches col_reduce_looped.
  constexpr int64_t kBN = 32;
  MTL::Size grid_dims(256 * ((reduction_stride + kBN - 1) / kBN), n_tiles, 1);
  ce.dispatch_threads(grid_dims, MTL::Size(256, 1, 1));
}

#else // !_METAL_

void KQuantVSATileSums::eval_gpu(const std::vector<mx::array>&, mx::array&) {
  throw std::runtime_error(kVsaTileSumsOp + "requires a Metal build.");
}

#endif

} // namespace

mx::array vsa_fixed_shapes(const mx::array& x, mx::StreamOrDevice s) {
  return mx::array(
      x.shape(),
      x.dtype(),
      std::make_shared<KQuantVSAFixedShapes>(mx::to_stream(s)),
      {x});
}

mx::array vsa_tile_sums(const mx::array& x, mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  if (x.ndim() != 3 || x.dtype() != mx::bfloat16 ||
      x.shape(0) % kVsaTile != 0) {
    throw std::invalid_argument(
        kVsaTileSumsOp + "x must be bfloat16 [tiles * 64, heads, D].");
  }
  const int n_tiles = x.shape(0) / kVsaTile;
  // MLX's own ops on a CPU stream, for an empty x, and past 32-bit indices,
  // where MLX takes the 64-bit variant of the kernel: the same bits.
  if (s.device != mx::Device::gpu || x.size() == 0 ||
      x.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    auto wide = mx::reshape(
        mx::astype(x, mx::float32, s),
        {n_tiles, kVsaTile, x.shape(1), x.shape(2)},
        s);
    return mx::sum(wide, 1, false, s);
  }
  return mx::array(
      {n_tiles, x.shape(1), x.shape(2)},
      mx::float32,
      std::make_shared<KQuantVSATileSums>(s),
      {mx::contiguous(x, false, s)});
}

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
  // With no video tile every row stays zero.
  if (n_video == 0) {
    return;
  }

  std::string kname =
      std::string(gated_ ? "kq_vsa_nax_gated_" : "kq_vsa_nax_") +
      kq_type_string(q.dtype()) + "_" + std::to_string(tile_) + "_bq" +
      std::to_string(query_rows_);
  auto kernel = kq_get_kernel(d, kname);
  ce.set_compute_pipeline_state(kernel);
  for (int i = 0; i < 6; ++i) {
    // With k 0, block_idx is empty and never read: block_num takes its slot.
    ce.set_input_array(i == 3 && k_max == 0 ? inputs[4] : inputs[i], i);
  }
  if (gated_) {
    ce.set_input_array(inputs[6], 12);
    ce.set_input_array(inputs[7], 13);
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

// Called when a shapeless compile replays the graph at new input shapes,
// which skips the checks in vsa_block_sparse, so other shapes are refused.
std::vector<mx::Shape> KQuantVSABlockSparse::output_shapes(
    const std::vector<mx::array>& inputs) {
  bool same = inputs.size() == input_shapes_.size();
  for (size_t i = 0; same && i < inputs.size(); ++i) {
    same = inputs[i].shape() == input_shapes_[i];
  }
  if (!same) {
    throw std::invalid_argument(
        kVsaOp +
        "got input shapes it was not built for, as in a shapeless compile. "
        "Compile without shapeless=True.");
  }
  return {inputs[0].shape()};
}

bool KQuantVSABlockSparse::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantVSABlockSparse&>(other);
  return n_prefix_ == o.n_prefix_ && tile_ == o.tile_ &&
      query_rows_ == o.query_rows_ && scale_ == o.scale_ && gated_ == o.gated_;
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
    std::optional<int> query_rows,
    std::optional<mx::array> coarse,
    std::optional<mx::array> gate,
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
  if (query_rows && *query_rows != 16 && *query_rows != 32 &&
      *query_rows != 64) {
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
  if (coarse.has_value() != gate.has_value()) {
    throw std::invalid_argument(
        kVsaOp + "coarse and gate are given together or not at all.");
  }
  if (gate && (gate->shape() != q.shape() || gate->dtype() != mx::bfloat16)) {
    throw std::invalid_argument(
        kVsaOp + "gate must be bfloat16 with the shape of q.");
  }
  if (coarse &&
      (coarse->shape() != mx::Shape{heads, n_tiles, kVsaHeadDim} ||
       coarse->dtype() != mx::float32)) {
    throw std::invalid_argument(
        kVsaOp + "coarse must be float32 [heads, tiles, 128].");
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
  // A call with no video tile or no key tile also builds the primitive, so
  // a shapeless compile traced on it refuses other shapes.
  auto as_i32 = [&](const mx::array& a) {
    return mx::contiguous(mx::astype(a, mx::int32, s), false, s);
  };
  // Integers wider than int32 are clipped to [lo, hi] before the cast, so a
  // value out of range stays out of range instead of wrapping into it. The
  // routing bounds do not depend on the shapes: the kernel skips tile indices
  // past the last tile and caps block_num at k.
  auto clip_i32 = [&](mx::array a, int lo, int hi) {
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
  constexpr int kInt32Max = std::numeric_limits<int>::max();
  const int rows =
      query_rows.value_or(n_video < kVsaRows64MinVideoTiles ? 32 : 64);
  std::vector<mx::array> inputs = {
      mx::contiguous(q, false, s),
      mx::contiguous(k, false, s),
      mx::contiguous(v, false, s),
      clip_i32(block_idx, -1, kInt32Max),
      clip_i32(block_num, 0, kInt32Max),
      clip_i32(variable_block_sizes, 0, tile_size)};
  if (gate) {
    inputs.push_back(mx::contiguous(*coarse, false, s));
    inputs.push_back(mx::contiguous(*gate, false, s));
  }
  std::vector<mx::Shape> input_shapes;
  for (const auto& in : inputs) {
    input_shapes.push_back(in.shape());
  }
  auto shape = q.shape();
  auto prim = std::make_shared<KQuantVSABlockSparse>(
      s,
      num_prefix_tiles,
      tile_size,
      rows,
      scale,
      gate.has_value(),
      std::move(input_shapes));
  return mx::array(
      std::move(shape), mx::bfloat16, std::move(prim), std::move(inputs));
}

} // namespace mlx_kquant
