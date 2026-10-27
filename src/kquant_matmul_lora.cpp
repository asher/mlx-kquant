// Quantized matmul with a low-rank term in the same reduction (see
// kq_quantized_nax_lora.h for the kernel).
//
//   y = x @ W^T + (strength * (x @ A)) @ B
//
// On a NAX GPU, for the codecs with a kernel (q4_k, q5_k, q6_k and q8_0 on
// bfloat16), from kLoraMinRows rows, on the shapes of lora_shape_rule,
// stock ops compute z = strength * (x @ A) and one dispatch computes the
// product and adds z @ B in its accumulator, with y as its only output
// buffer. Every other case builds the same sum from quantized_matmul and
// stock ops.
#include <algorithm>
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
// Column tile of a simdgroup. A band ends on one, or at N.
constexpr int kLoraSn = 32;
// Largest rank of one band the kernel runs.
constexpr int kLoraMaxRank = 256;
// Most bands the kernel looks through per threadgroup.
constexpr int kLoraMaxBands = 16;
// Sizes the op refuses. The kernel offsets a row tile by a 32-bit product
// of 64 rows and a row length (K, N or R), which wraps from 2^25, and
// holds the row count in an int.
constexpr int64_t kLoraMaxCols = int64_t{1} << 25;
constexpr int64_t kLoraMaxRows = int64_t{1} << 31;

// K tile of the kernel for a codec and an activation dtype, 0 where the op
// builds the composition.
int lora_kernel_bk(const std::string& t, mx::Dtype dt) {
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

bool lora_tune_live() {
  static const bool live = kq_env_lever("KQ_TUNE_LIVE", 0) != 0;
  return live;
}

// KQ_QMM_LORA: 0 = the composition on every call, 2 = the kernel wherever
// it can run, unset, 1 or any other value = the shape rule
// (lora_shape_rule).
int lora_mode() {
  static const int cached = kq_env_lever("KQ_QMM_LORA", 1);
  return lora_tune_live() ? kq_env_lever("KQ_QMM_LORA", 1) : cached;
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

// The shapes on which the kernel runs under the default mode: those where
// it measured faster than the unfused ops. The kernel removes the second
// pass over y, a cost in proportion to N. It waits for z, a cost in
// proportion to K that the unfused product does not pay, since x @ A runs
// beside it. So the kernel is faster where N is at least K, and its share
// of the call falls as K grows: above kLoraRuleK the two forms tie. From N
// of kLoraRuleWideN the kernel runs at every row count and rank it takes.
// Where N is nearer K the gain is about one percent: it holds from
// kLoraRuleRows rows and up to a rank of kLoraRuleRank, since the wait for
// z grows with the rank. rank is the most ranks a column takes.
constexpr int kLoraRuleK = 4096;
constexpr int kLoraRuleRows = 4096;
constexpr int kLoraRuleWideN = 8192;
constexpr int kLoraRuleRank = 128;
bool lora_shape_rule(int K, int N, int64_t rows, int rank) {
  return K <= kLoraRuleK && N >= K &&
      (N >= kLoraRuleWideN || (rows >= kLoraRuleRows && rank <= kLoraRuleRank));
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

// c + a @ b by the stock op, with c copied to dense memory first when it
// has one column. On the GPU the stock op reads a first argument of one
// column and several rows at a step that is not its row stride, so a
// column of a wider array comes out wrong (docs/upstream-defects.md). A
// dense c is passed through with no copy, and a c of two or more columns
// is read in place, as before.
mx::array
lora_addmm(mx::array c, const mx::array& a, const mx::array& b, mx::Stream s) {
  if (c.shape(-1) == 1) {
    c = mx::contiguous(c, false, s);
  }
  return mx::addmm(std::move(c), a, b, 1.0f, 1.0f, s);
}

// The base product and z on their way into the banded sum: the identity,
// on the buffer of its input, under every transform. The stock Slice ops
// that cut the bands have no shape rule, so a shapeless compiled function
// cannot hold the banded sum. Each array that a Slice cuts first passes
// through this node, so the node comes before every Slice in the order of
// a trace and refuses by the name of the op.
class LoraBandedInput : public mx::UnaryPrimitive {
 public:
  explicit LoraBandedInput(mx::Stream stream) : mx::UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    out.copy_shared_buffer(inputs[0]);
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    out.copy_shared_buffer(inputs[0]);
  }

  const char* name() const override {
    return "KQuantLoraBandedInput";
  }
  bool is_equivalent(const mx::Primitive&) const override {
    return true;
  }

  std::vector<mx::array> vjp(
      const std::vector<mx::array>&,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>&,
      const std::vector<mx::array>&) override {
    return {cotangents[0]};
  }
  std::vector<mx::array> jvp(
      const std::vector<mx::array>&,
      const std::vector<mx::array>& tangents,
      const std::vector<int>&) override {
    return {tangents[0]};
  }
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override {
    return {{inputs[0]}, axes};
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>&) override {
    throw std::invalid_argument(
        std::string(kLoraOp) +
        " A shapeless compiled function cannot hold a call with bands that "
        "builds the unfused ops. Compile it with fixed shapes, or trace it "
        "on a call that runs the kernel.");
  }
};

// base + z @ B over the bands. With no band table every column takes
// every rank. With one, a band's columns take its ranks and its block of
// B, and a column outside every band takes none.
mx::array lora_add(
    const mx::array& base_,
    const mx::array& z_,
    const mx::array& b,
    const std::vector<int>& bands,
    mx::Stream s) {
  if (bands.empty()) {
    return lora_addmm(base_, z_, b, s);
  }
  auto banded_input = [&s](const mx::array& a) {
    return mx::array(
        a.shape(), a.dtype(), std::make_shared<LoraBandedInput>(s), {a});
  };
  const mx::array base = banded_input(base_);
  const mx::array z = banded_input(z_);
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
      y = lora_addmm(std::move(y), last_axis_slice(z, z0, z0 + rb, s), bb, s);
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
  const mx::array x = kq_x_aligned_gpu(inputs[0], s, "quantized_matmul_lora x");
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
  // M may be within a row tile of the largest int.
  MTL::Size grid_dims(
      (N + kLoraBn - 1) / kLoraBn, (static_cast<size_t>(M) + bm - 1) / bm, 1);
  if (lora_log()) {
    std::fprintf(
        stderr,
        "[kq qmm] %s M=%d N=%d K=%d R=%d bands=%d grid=%lu,%lu,%lu tg=%lu,%lu,%lu xa=%d\n",
        kname.c_str(),
        M,
        N,
        K,
        R,
        n_bands,
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

// The kernel computes the function of the composition, so its vjp is the
// composition's: with respect to x, z and B, with the quantized weight
// frozen as in quantized_matmul. z is built from x, A and the strength by
// stock ops before this node, so their gradients follow from the one with
// respect to z.
namespace {

constexpr int kLoraArgOfInput[] = {0, -1, 1, 2};

void lora_check_argnums(const std::vector<int>& argnums) {
  for (auto arg : argnums) {
    if (arg == 1) {
      throw std::invalid_argument(
          "[mlx_kquant] quantized_matmul_lora vjp: no gradient wrt the "
          "quantized weight (the kquant base is frozen).");
    }
  }
}

} // namespace

std::vector<mx::array> KQuantMatmulLora::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  lora_check_argnums(argnums);
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

// jvp and vmap are the transforms of the composition, over the inputs that
// carry a tangent or a batch axis. The others enter as constants. So a
// tangent or a batch of z or B (of lora_a, lora_b or the strength) runs on
// stock ops, and one of x or of the weight reaches quantized_matmul, which
// has neither transform and refuses by its own name, as it does where the
// op builds the composition.
std::vector<mx::array> KQuantMatmulLora::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
  auto fun = [&](const std::vector<mx::array>& in) {
    std::vector<mx::array> full = primals;
    for (size_t i = 0; i < argnums.size(); i++) {
      full[argnums[i]] = in[i];
    }
    return std::vector<mx::array>{lora_composed(
        full[0], full[1], full[2], full[3], kquant_type_, bands_, stream())};
  };
  std::vector<mx::array> at;
  for (auto arg : argnums) {
    at.push_back(primals[arg]);
  }
  return mx::jvp(fun, at, tangents).second;
}

std::pair<std::vector<mx::array>, std::vector<int>> KQuantMatmulLora::vmap(
    const std::vector<mx::array>& inputs,
    const std::vector<int>& axes) {
  auto fun = [&](const std::vector<mx::array>& in) {
    return std::vector<mx::array>{lora_composed(
        in[0], in[1], in[2], in[3], kquant_type_, bands_, stream())};
  };
  return {mx::vmap(fun, axes, {0})(inputs), {0}};
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
    const std::optional<std::vector<int>>& bands_,
    mx::StreamOrDevice s_) {
  const std::string op = kLoraOp;
  // No table: every column takes every rank, which the code below holds as
  // an empty vector. A table with no row: no column takes a rank.
  const bool no_band = bands_.has_value() && bands_->empty();
  const std::vector<int> bands = bands_.value_or(std::vector<int>{});
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
  if (K >= kLoraMaxCols || N >= kLoraMaxCols || R >= kLoraMaxCols) {
    throw std::invalid_argument(
        op + " K, N and R must each be below " + std::to_string(kLoraMaxCols) +
        " (2^25): K " + std::to_string(K) + ", N " + std::to_string(N) +
        ", R " + std::to_string(R) + ".");
  }
  const int64_t rows = K > 0 ? static_cast<int64_t>(x.size()) / K : 0;
  if (rows >= kLoraMaxRows) {
    throw std::invalid_argument(
        op + " x must hold fewer than " + std::to_string(kLoraMaxRows) +
        " (2^31) rows: it holds " + std::to_string(rows) + ".");
  }
  if (strength.has_value()) {
    const auto& st = *strength;
    // 0-D, [1] or [R]. Any other shape would broadcast z to more dims.
    const bool shape_ok = st.ndim() == 0 ||
        (st.ndim() == 1 && (st.shape(0) == 1 || st.shape(0) == R));
    if (!shape_ok || !mx::issubdtype(st.dtype(), mx::floating)) {
      throw std::invalid_argument(
          op +
          " strength must be a float array of no dim, of shape [1] or of "
          "shape [R] with R " +
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
  auto scales = mx::zeros({1}, mx::uint8, s);
  if (rows == 0 || N == 0 || R == 0 || no_band) {
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
        std::nullopt,
        std::nullopt,
        std::nullopt,
        s);
  }

  // The kernel walks K in whole K tiles, so a K off the tile runs the
  // composition (q8_0 only: the K-quant codecs hold K in blocks of 256).
  const int bk = lora_kernel_bk(kquant_type, out_type);
  const int mode = lora_mode();
  // The most ranks a column takes: every rank with no table, the rank of
  // the largest band with one.
  int col_rank = bands.empty() ? R : 0;
  for (size_t i = 0; i < bands.size(); i += 4) {
    col_rank = std::max(col_rank, bands[i + 3]);
  }
  bool fused = s.device == mx::Device::gpu && nax_available() && mode != 0 &&
      bk > 0 && K % bk == 0 && rows >= lora_min_rows() &&
      (mode == 2 || lora_shape_rule(K, N, rows, col_rank));
  if (bands.empty()) {
    fused = fused && R <= kLoraMaxRank;
  } else {
    fused = fused && bands.size() / 4 <= kLoraMaxBands;
    for (size_t i = 0; i < bands.size(); i += 4) {
      // A band that ended inside a simdgroup's columns would put z times
      // zero on the columns after it, which is not zero for a z that is
      // not finite.
      const bool end_ok = bands[i + 1] % kLoraSn == 0 || bands[i + 1] == N;
      fused = fused && bands[i] % kLoraBn == 0 && end_ok &&
          bands[i + 3] <= kLoraMaxRank;
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
