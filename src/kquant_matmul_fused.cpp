// Quantized matmuls with an epilogue fused into the store (see
// kq_quantized_nax_epi.h for the kernels).
//
// GLU product: y = act(x @ Wg^T) * (x @ Wu^T) over two packed weights of
// one codec and one shape. On a NAX GPU, for the codecs with a GLU kernel,
// from kGluMinRows rows and with K a whole number of the codec's K tiles,
// one dispatch computes the product with y as its only output buffer.
// Every other case builds the same product from quantized_matmul and stock
// ops.
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
#include "mlx/compile.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "kquant_metal_internal.h"
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

const char* kGluOp = "[mlx_kquant.quantized_matmul_glu]";

// Activation kinds, the values of the kernel's function constant.
int glu_act_kind(const std::string& activation) {
  if (activation == "silu") {
    return 0;
  }
  if (activation == "gelu_tanh") {
    return 1;
  }
  throw std::invalid_argument(
      std::string(kGluOp) + " activation must be 'silu' or 'gelu_tanh', got '" +
      activation + "'.");
}

// The activation of the composition, built the way mlx.nn builds silu and
// gelu_approx: one compiled elementwise function of the same ops in the
// same order, with the constants a Python scalar gives. The composition
// then has the bits and the peak memory of nn.silu(g) * u and
// nn.gelu_approx(g) * u: the compiled function writes into g's buffer,
// where separate ops hold a buffer per step.
std::vector<mx::array> glu_silu(const std::vector<mx::array>& in) {
  const auto& x = in[0];
  return {mx::multiply(x, mx::sigmoid(x))};
}

std::vector<mx::array> glu_gelu_tanh(const std::vector<mx::array>& in) {
  // 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))
  const auto& x = in[0];
  const auto dt = x.dtype();
  auto x3 = mx::power(x, mx::array(3, dt));
  auto sum = mx::add(x, mx::multiply(mx::array(0.044715f, dt), x3));
  auto t = mx::tanh(mx::multiply(mx::array(0.7978845608028654f, dt), sum));
  return {mx::multiply(
      mx::multiply(mx::array(0.5f, dt), x), mx::add(mx::array(1, dt), t))};
}

mx::array glu_activation(const mx::array& g, int act, mx::Stream s) {
  // Never destroyed: a compiled function erases its cache entry when it
  // goes, and at process exit the cache can go first.
  static const auto* silu = new auto(mx::compile(glu_silu, true));
  static const auto* gelu = new auto(mx::compile(glu_gelu_tanh, true));
  // The ops of a compiled function run on the default stream.
  mx::StreamContext on_stream(s);
  return (act == 1 ? *gelu : *silu)({g})[0];
}

// K tile of the GLU kernel for a codec and an activation dtype, 0 where
// the op builds the composition. The q8_0 decoder has a K tile of 64 only,
// and its kernel runs on bfloat16 only: on float16 it measured slower than
// the composition.
int glu_kernel_bk(const std::string& t, mx::Dtype dt) {
  if (t == "q4_k" || t == "q5_k" || t == "q6_k") {
    return 128;
  }
  if (t == "q8_0" && dt == mx::bfloat16) {
    return 64;
  }
  return 0;
}

bool glu_tune_live() {
  static const bool live = kq_env_lever("KQ_TUNE_LIVE", 0) != 0;
  return live;
}

// KQ_GLU=0 builds the composition on every call.
bool glu_on() {
  static const bool cached = kq_env_flag("KQ_GLU", 1) != 0;
  return glu_tune_live() ? kq_env_flag("KQ_GLU", 1) != 0 : cached;
}

// Rows from which the fused kernel runs. The kernel is measured against
// the composition from 1056 rows up; fewer rows build the composition.
// KQ_GLU_MIN_ROWS replaces the bound.
constexpr int kGluMinRows = 1024;
int glu_min_rows() {
  static const int cached = kq_env_lever("KQ_GLU_MIN_ROWS", kGluMinRows);
  return glu_tune_live() ? kq_env_lever("KQ_GLU_MIN_ROWS", kGluMinRows)
                         : cached;
}

// The op runs the two-walk kernel on a row tile of 128 and a K tile of
// 128. KQ_GLU_BM and KQ_GLU_BK (64 or 128), KQ_GLU_WALKS=0 (the one-walk
// kernel) and KQ_GLU_DUAL128=1 select probe kernels, which exist in a
// build with KQ_PROBE_KERNELS=ON only. All are read once per process
// unless KQ_TUNE_LIVE=1.
int glu_lever(const char* name) {
  const int v = kq_env_lever(name, -1);
  return v == 64 || v == 128 ? v : -1;
}
int glu_bm_lever() {
  static const int cached = glu_lever("KQ_GLU_BM");
  return glu_tune_live() ? glu_lever("KQ_GLU_BM") : cached;
}
int glu_bk_lever() {
  static const int cached = glu_lever("KQ_GLU_BK");
  return glu_tune_live() ? glu_lever("KQ_GLU_BK") : cached;
}
bool glu_walks() {
  static const bool cached = kq_env_flag("KQ_GLU_WALKS", 1) != 0;
  return glu_tune_live() ? kq_env_flag("KQ_GLU_WALKS", 1) != 0 : cached;
}
// The one-walk K tile of 128 on two unpadded Ws buffers.
bool glu_dual128() {
  static const bool cached = kq_env_flag("KQ_GLU_DUAL128", 0) != 0;
  return glu_tune_live() ? kq_env_flag("KQ_GLU_DUAL128", 0) != 0 : cached;
}

bool glu_log() {
  static const bool cached = std::getenv("KQ_QMM_LOG") != nullptr;
  return glu_tune_live() ? std::getenv("KQ_QMM_LOG") != nullptr : cached;
}

} // namespace

#ifdef _METAL_

void KQuantMatmulFused::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  const auto& x = inputs[0];
  // A weight whose rows are dense binds as it is, at its own offset, so a
  // row band of a wider weight is read in place. Any other view is copied
  // to dense rows first.
  const mx::array wg = kq_matrix_dense_gpu(inputs[1], s);
  const mx::array wu = kq_matrix_dense_gpu(inputs[2], s);
  auto& out = outputs[0];
  kq_check_weight_base(wg, kquant_type_, "quantized_matmul_glu");
  kq_check_weight_base(wu, kquant_type_, "quantized_matmul_glu");
  out.set_data(mx::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  const int K = x.shape(-1);
  const int N = wg.shape(0);
  // x is dense rows, so its batch dims fold into M.
  const int M = static_cast<int>(x.size() / K);

  int bm = 128;
  int bk = glu_kernel_bk(kquant_type_, x.dtype());
  if (const int v = glu_bm_lever(); v > 0) {
    bm = v;
  }
  if (const int v = glu_bk_lever(); v > 0 && v <= bk && K % v == 0) {
    bk = v;
  }
  const int bn = 64;

  const bool aligned = N % bn == 0;
  bool saturate = x.dtype() == mx::float16;
  const std::string kname = kq_kname_prefix(kquant_type_) + "qmm_t_nax_glu_" +
      kq_type_string(x.dtype()) + "_bm" + std::to_string(bm) + "_bn64_bk" +
      std::to_string(bk) +
      (glu_walks()                      ? "w"
           : bk == 128 && glu_dual128() ? "d"
                                        : "") +
      "_wm2_wn2_alN_" + (aligned ? "true" : "false");
  const std::string hash_name =
      kname + "_act" + std::to_string(act_) + (saturate ? "_sat" : "");
  mx::metal::MTLFCList func_consts = {
      {&act_, MTL::DataType::DataTypeInt, 360},
      {&saturate, MTL::DataType::DataTypeBool, 361},
  };
  MTL::Size group_dims(32, 2, 2);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);
  if (glu_log()) {
    std::fprintf(
        stderr,
        "[kq qmm] %s M=%d N=%d K=%d grid=%lu,%lu,%lu\n",
        kname.c_str(),
        M,
        N,
        K,
        static_cast<unsigned long>(grid_dims.width),
        static_cast<unsigned long>(grid_dims.height),
        static_cast<unsigned long>(grid_dims.depth));
  }

  auto kernel = kq_get_kernel(d, kname, hash_name, func_consts);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  int c = 0;
  ce.set_input_array(wg, c++);
  ce.set_input_array(wu, c++);
  ce.set_input_array(x, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(M, c++);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantMatmulFused::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(std::string(kGluOp) + " requires Metal.");
}

#endif

void KQuantMatmulFused::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      std::string(kGluOp) + " CPU streams build the composed ops.");
}

bool KQuantMatmulFused::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantMatmulFused&>(other);
  return kquant_type_ == o.kquant_type_ && act_ == o.act_;
}

std::vector<mx::Shape> KQuantMatmulFused::output_shapes(
    const std::vector<mx::array>& inputs) {
  auto shape = inputs[0].shape();
  shape.back() = inputs[1].shape(0);
  return {shape};
}

mx::array quantized_matmul_glu(
    mx::array x,
    mx::array w_gate,
    mx::array w_up,
    const std::string& kquant_type,
    const std::string& activation,
    mx::StreamOrDevice s_) {
  const std::string op = kGluOp;
  const int act = glu_act_kind(activation);
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        op + " Unknown kquant_type: '" + kquant_type + "'.");
  }
  for (const auto* w : {&w_gate, &w_up}) {
    if (w->dtype() != mx::uint8 || w->ndim() != 2) {
      throw std::invalid_argument(
          op + " w_gate and w_up must be 2-D uint8 (raw GGUF wire bytes).");
    }
  }
  if (w_gate.shape() != w_up.shape()) {
    throw std::invalid_argument(op + " w_gate and w_up must have one shape.");
  }
  if (w_gate.shape(-1) % codec->bytes_per_block != 0) {
    throw std::invalid_argument(
        op + " weight last dim (" + std::to_string(w_gate.shape(-1)) +
        " bytes) is not a whole number of " +
        std::to_string(codec->bytes_per_block) + "-byte " + codec->name +
        " blocks.");
  }
  const int K =
      (w_gate.shape(-1) / codec->bytes_per_block) * codec->weights_per_block;
  if (x.ndim() < 1 || x.shape(-1) != K) {
    throw std::invalid_argument(
        op + " x last dim does not match the weight inner dim (" +
        std::to_string(K) + ").");
  }
  if (!mx::issubdtype(x.dtype(), mx::floating)) {
    throw std::invalid_argument(
        op + " Only real floating x dtypes are supported.");
  }
  auto s = mx::to_stream(s_);

  // K must be a whole number of the codec's K tiles (of the lever's tile
  // when KQ_GLU_BK selects a smaller one).
  // float32 x runs in bfloat16, as in quantized_matmul.
  const mx::Dtype out_type =
      x.dtype() == mx::float32 ? mx::bfloat16 : x.dtype();
  int bk = glu_kernel_bk(kquant_type, out_type);
  if (const int v = glu_bk_lever(); v > 0 && v <= bk) {
    bk = v;
  }
  const int64_t rows = K > 0 ? static_cast<int64_t>(x.size()) / K : 0;
  bool fused = s.device == mx::Device::gpu && nax_available() && glu_on() &&
      bk > 0 && K % bk == 0 && x.ndim() >= 2 && rows >= glu_min_rows() &&
      rows > 0 && w_gate.shape(0) > 0;
  if (!fused) {
    auto scales = mx::zeros({1}, mx::uint8, s);
    auto g = quantized_matmul(
        x,
        w_gate,
        scales,
        kquant_type,
        true,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        s);
    auto u = quantized_matmul(
        x,
        w_up,
        scales,
        kquant_type,
        true,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        s);
    return mx::multiply(glu_activation(g, act, s), u, s);
  }

  kq_check_weight_base_at_build(w_gate, kquant_type, s, "quantized_matmul_glu");
  kq_check_weight_base_at_build(w_up, kquant_type, s, "quantized_matmul_glu");
  // x gets a Contiguous node: the strides of an unevaluated array mean
  // nothing at build time. The weights go in as they are and the eval
  // looks at their strides. A Contiguous node would copy a row band of a
  // wider weight on every call, since its buffer is larger than the band.
  auto x_c = mx::contiguous(mx::astype(x, out_type, s), false, s);
  auto out_shape = x.shape();
  out_shape.back() = w_gate.shape(0);
  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantMatmulFused>(s, kquant_type, act),
      {std::move(x_c), std::move(w_gate), std::move(w_up)});
}

} // namespace mlx_kquant
