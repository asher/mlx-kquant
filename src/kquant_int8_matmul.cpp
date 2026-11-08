// int8 row quantization and the int8 x int8 matmul on the tensor-op (NAX)
// units (see kq_int8_gemm_nax.h and kq_int8_quant.metal). GPU only: both
// ops refuse a CPU stream when the call is built, and int8_matmul any GPU
// without NAX units.
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string, kq_*_gpu layout helpers

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

const std::string kQuantOp = "[mlx_kquant.int8_quantize_rows] ";
const std::string kMatmulOp = "[mlx_kquant.int8_matmul] ";
constexpr int kKStep = 32; // K of one tensor-op step
constexpr int kBlock = 128; // rows and columns of an output block
constexpr int64_t kMaxRows = (int64_t{1} << 31) - 256;
// Grid limits: Metal counts a grid axis's threads in 32 bits and runs only
// the remainder mod 2^32, so a grid must stay under 2^32 threads. The
// quantizer runs ceil(units / 8) threadgroups of 256 threads on one axis,
// int8_matmul 4 threadgroups of 256 a column block; each limit is the
// largest value whose grid is under 2^32 threads.
constexpr int64_t kMaxQuantUnits = (int64_t{1} << 27) - 8;
constexpr int64_t kMaxN = (int64_t{1} << 29) - 128;
// The longest int32 sum of 127 x 127 products, a multiple of 32:
// 127^2 x 133120 = 2147092480 < 2^31.
constexpr int kMaxGroupLen = 133120;
// The grouped kernel stages its block's row scales, 128 x G floats, in
// threadgroup memory up to this size (G <= 32), half of the 32 KiB a
// threadgroup may hold; larger G reads them from device memory.
constexpr size_t kScaleStageBytes = 16384;
constexpr int kSimdgroups = 8;
constexpr int kSwizzle = 2;

void check_gpu_stream(const mx::Stream& s, const std::string& op) {
  if (s.device != mx::Device::gpu) {
    throw std::invalid_argument(
        op + "runs on the GPU only; the stream is on the CPU.");
  }
}

void check_group_size(const std::optional<int>& gs, const std::string& op) {
  if (gs && (*gs < kKStep || *gs % kKStep != 0)) {
    throw std::invalid_argument(
        op + "group_size must be None or a multiple of 32 from 32 up.");
  }
}

// Scales a row: 1 without a group, ceil(K / group_size) with one.
int scale_groups(int K, const std::optional<int>& gs) {
  return gs ? (K + *gs - 1) / *gs : 1;
}

// x's shape with its last dim replaced by n.
mx::Shape last_dim(const mx::Shape& shape, int n) {
  mx::Shape out = shape;
  out.back() = n;
  return out;
}

} // namespace

#ifdef _METAL_

void KQuantInt8QuantizeRows::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  for (auto& o : outputs) {
    o.set_data(mx::allocator::malloc(o.nbytes()));
  }
  const auto& x_in = inputs[0];
  const int K = x_in.shape(-1);
  const size_t rows = x_in.size() / K;
  const int G = scale_groups(K, group_size_);
  if (rows == 0) {
    return;
  }
  const int gs = group_size_ && *group_size_ < K ? *group_size_ : K;
  // Four columns at a time when every group starts on a boundary of four
  // elements; x is then bound on one (copied when it is not).
  const bool vec = K % 4 == 0;
  const mx::array x = vec ? kq_x_aligned_gpu(
                                x_in,
                                s,
                                "int8_quantize_rows x",
                                4 * static_cast<int>(x_in.itemsize()))
                          : kq_row_major_gpu(x_in, s);
  // One simdgroup a (row, group) unit, eight units a threadgroup.
  const int64_t units = static_cast<int64_t>(rows) * G;
  auto kernel = kq_get_kernel(
      d,
      std::string("kq_int8_quantize_rows_") + (vec ? "vec_" : "") +
          kq_type_string(x.dtype()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(x, 0);
  ce.set_output_array(outputs[0], 1);
  ce.set_output_array(outputs[1], 2);
  ce.set_bytes(K, 3);
  ce.set_bytes(gs, 4);
  ce.set_bytes(G, 5);
  ce.set_bytes(units, 6);
  ce.dispatch_threadgroups(
      MTL::Size(static_cast<size_t>((units + 7) / 8), 1, 1),
      MTL::Size(256, 1, 1));
}

namespace {

// KqInt8MMParams of kq_int8_gemm_nax.h.
struct Int8MMParams {
  int M;
  int N;
  int G;
  int tiles_m;
};

// The last tile of `tile_rows` rows of the row-major [rows, K] int8
// operand a, zero padded to tile_rows rows, in a buffer the command buffer
// keeps alive. a is bound on a 16-byte boundary and K is a multiple of 32,
// so every row starts on one.
mx::array pad_last_tile(
    const mx::array& a,
    int rows,
    int tile_rows,
    int K,
    const mx::Stream& s) {
  auto& d = mx::metal::device(s.device);
  auto& ce = mx::metal::get_command_encoder(s);
  const int first = ((rows - 1) / tile_rows) * tile_rows;
  const int live = rows - first;
  mx::array t({tile_rows, K}, mx::int8, nullptr, {});
  t.set_data(mx::allocator::malloc(t.nbytes()));
  ce.set_compute_pipeline_state(kq_get_kernel(d, "kq_int8_pad_rows"));
  ce.set_input_array(a, 0, static_cast<int64_t>(first) * K);
  ce.set_output_array(t, 1);
  ce.set_bytes(K, 2);
  ce.set_bytes(live, 3);
  const int cols = K / 16;
  ce.dispatch_threads(
      MTL::Size(cols, tile_rows, 1), MTL::Size(cols < 256 ? cols : 256, 1, 1));
  ce.add_temporary(t);
  return t;
}

} // namespace

void KQuantInt8Matmul::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  if (!kq_is_nax_available()) {
    throw std::runtime_error(
        kMatmulOp +
        "requires a GPU with tensor-op (NAX) units, M5 or later; check "
        "nax_available().");
  }
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  // Codes are read as row-major runs from 16-byte boundaries (copied when
  // they are not); the scales row-major.
  const mx::array x8 = kq_x_aligned_gpu(inputs[0], s, "int8_matmul x8", 16);
  const mx::array w8 = kq_x_aligned_gpu(inputs[1], s, "int8_matmul w8", 16);
  const mx::array sa = kq_row_major_gpu(inputs[2], s);
  const mx::array sb = kq_row_major_gpu(inputs[3], s);

  const int K = x8.shape(-1);
  const int M = static_cast<int>(x8.size() / K);
  const int N = w8.shape(0);
  const bool grouped = group_size_ && *group_size_ < K;
  const int G = grouped ? scale_groups(K, group_size_) : 1;
  const int tiles_n = (N + kBlock - 1) / kBlock;
  Int8MMParams p{
      /* M = */ M,
      /* N = */ N,
      /* G = */ G,
      /* tiles_m = */ (M + kBlock - 1) / kBlock};

  // The ragged last row and column blocks read zero-padded copies.
  const mx::array At = M % kBlock ? pad_last_tile(x8, M, kBlock, K, s) : x8;
  const mx::array Bt = N % kBlock ? pad_last_tile(w8, N, kBlock, K, s) : w8;

  const std::string kname = std::string("kq_int8_mm_") +
      (grouped ? "grouped_" : "") + "nax_" + kq_type_string(out.dtype());
  auto& ce = mx::metal::get_command_encoder(s);
  // K, whether N is a multiple of the block, and for the grouped kernel the
  // group size and where the row scales live are function constants: one
  // pipeline per combination.
  int fc_k = K;
  bool align_n = N % kBlock == 0;
  int fc_gs = grouped ? *group_size_ : K;
  const size_t scale_bytes = static_cast<size_t>(kBlock) * G * sizeof(float);
  bool tg_scales = grouped && scale_bytes <= kScaleStageBytes;
  mx::metal::MTLFCList func_consts = {
      {&fc_k, MTL::DataType::DataTypeInt, 310},
      {&align_n, MTL::DataType::DataTypeBool, 312}};
  std::string hash_name =
      kname + "_k" + std::to_string(K) + "_align_n_" + (align_n ? 't' : 'n');
  if (grouped) {
    func_consts.push_back({&fc_gs, MTL::DataType::DataTypeInt, 311});
    func_consts.push_back({&tg_scales, MTL::DataType::DataTypeBool, 313});
    hash_name +=
        "_g" + std::to_string(fc_gs) + "_tg_" + (tg_scales ? 't' : 'n');
  }
  ce.set_compute_pipeline_state(
      kq_get_kernel(d, kname, hash_name, func_consts));
  ce.set_input_array(x8, 0);
  ce.set_input_array(w8, 1);
  ce.set_input_array(sa, 2);
  ce.set_input_array(sb, 3);
  ce.set_output_array(out, 4);
  ce.set_input_array(At, 5);
  ce.set_input_array(Bt, 6);
  ce.set_bytes(p, 7);
  if (grouped) {
    ce.set_threadgroup_memory_length(
        tg_scales ? (scale_bytes + 15) / 16 * 16 : 16, 0);
  }
  const int gy = (p.tiles_m + (1 << kSwizzle) - 1) >> kSwizzle;
  ce.dispatch_threadgroups(
      MTL::Size(tiles_n << kSwizzle, gy, 1), MTL::Size(kSimdgroups * 32, 1, 1));
}

#else // !_METAL_

void KQuantInt8QuantizeRows::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(kQuantOp + "requires a Metal build.");
}

void KQuantInt8Matmul::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(kMatmulOp + "requires a Metal build.");
}

#endif

void KQuantInt8QuantizeRows::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(kQuantOp + "has no CPU implementation.");
}

void KQuantInt8Matmul::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(kMatmulOp + "has no CPU implementation.");
}

std::vector<mx::Shape> KQuantInt8QuantizeRows::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& x = inputs[0];
  return {
      x.shape(), last_dim(x.shape(), scale_groups(x.shape(-1), group_size_))};
}

std::vector<mx::Shape> KQuantInt8Matmul::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {last_dim(inputs[0].shape(), inputs[1].shape(0))};
}

std::tuple<mx::array, mx::array> int8_quantize_rows(
    mx::array x,
    std::optional<int> group_size,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  check_gpu_stream(s, kQuantOp);
  if (x.ndim() < 1 || x.shape(-1) < 1) {
    throw std::invalid_argument(
        kQuantOp + "x must have at least one dim, its last (K) at least 1.");
  }
  if (x.dtype() != mx::bfloat16 && x.dtype() != mx::float16 &&
      x.dtype() != mx::float32) {
    throw std::invalid_argument(
        kQuantOp + "x must be bfloat16, float16 or float32.");
  }
  check_group_size(group_size, kQuantOp);
  const int64_t units = static_cast<int64_t>(x.size() / x.shape(-1)) *
      scale_groups(x.shape(-1), group_size);
  if (units > kMaxQuantUnits) {
    throw std::invalid_argument(
        kQuantOp +
        "x must have at most 2^27 - 8 groups, rows times ceil(K / group_size); "
        "quantize it in row chunks.");
  }
  auto shapes = std::vector<mx::Shape>{
      x.shape(), last_dim(x.shape(), scale_groups(x.shape(-1), group_size))};
  auto outs = mx::array::make_arrays(
      std::move(shapes),
      {mx::int8, mx::float32},
      std::make_shared<KQuantInt8QuantizeRows>(s, group_size),
      {std::move(x)});
  return {outs[0], outs[1]};
}

mx::array int8_matmul(
    mx::array x8,
    mx::array x_scales,
    mx::array w8,
    mx::array w_scales,
    std::optional<int> group_size,
    mx::Dtype out_dtype,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  check_gpu_stream(s, kMatmulOp);
  if (x8.ndim() < 1 || x8.dtype() != mx::int8) {
    throw std::invalid_argument(
        kMatmulOp + "x8 must be int8 with at least one dim.");
  }
  const int K = x8.shape(-1);
  if (K < kKStep || K % kKStep != 0) {
    throw std::invalid_argument(
        kMatmulOp +
        "K, the last dim of x8 and w8, must be a positive multiple of 32; "
        "pad both with zero columns to the next multiple of 32.");
  }
  check_group_size(group_size, kMatmulOp);
  if ((group_size && *group_size < K ? *group_size : K) > kMaxGroupLen) {
    throw std::invalid_argument(
        kMatmulOp +
        "a group, or K without one, must be at most 133120 columns, so its "
        "int32 sum cannot wrap; pass a smaller group_size.");
  }
  if (w8.ndim() != 2 || w8.dtype() != mx::int8) {
    throw std::invalid_argument(kMatmulOp + "w8 must be int8 [N, K].");
  }
  if (w8.shape(1) != K) {
    throw std::invalid_argument(
        kMatmulOp + "w8's last dim must equal x8's (K).");
  }
  // M is held in 32 bits, with a block of headroom for the tile arithmetic
  // (row0 + 128, M + 127); N is capped by the grid (kMaxN).
  if (static_cast<int64_t>(x8.size() / K) >= kMaxRows) {
    throw std::invalid_argument(
        kMatmulOp + "x8 must have fewer than 2^31 - 256 rows.");
  }
  if (w8.shape(0) > kMaxN) {
    throw std::invalid_argument(
        kMatmulOp + "w8 must have at most 2^29 - 128 rows (N).");
  }
  if (x_scales.dtype() != mx::float32 ||
      x_scales.shape() != last_dim(x8.shape(), scale_groups(K, group_size))) {
    throw std::invalid_argument(
        kMatmulOp +
        "x_scales must be float32 [..., G] with x8's leading dims, G = 1 "
        "without a group_size and ceil(K / group_size) with one.");
  }
  if (w_scales.dtype() != mx::float32 || w_scales.ndim() != 1 ||
      w_scales.shape(0) != w8.shape(0)) {
    throw std::invalid_argument(kMatmulOp + "w_scales must be float32 [N].");
  }
  if (out_dtype != mx::bfloat16 && out_dtype != mx::float16 &&
      out_dtype != mx::float32) {
    throw std::invalid_argument(
        kMatmulOp + "out_dtype must be bfloat16, float16 or float32.");
  }
  // Checked after the arguments, so the argument checks also run on GPUs
  // without NAX.
  if (!nax_available()) {
    throw std::runtime_error(
        kMatmulOp +
        "requires a GPU with tensor-op (NAX) units, M5 or later; check "
        "nax_available().");
  }
  auto shape = last_dim(x8.shape(), w8.shape(0));
  return mx::array(
      std::move(shape),
      out_dtype,
      std::make_shared<KQuantInt8Matmul>(s, group_size, out_dtype),
      {std::move(x8), std::move(w8), std::move(x_scales), std::move(w_scales)});
}

} // namespace mlx_kquant
