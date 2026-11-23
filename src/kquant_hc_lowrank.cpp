// Fused qwen4exp low-rank hyper-connection ops for the short-row decode
// route (see kq_hc_lowrank.h for the kernel shapes and the eager reference
// they mirror). The three ops take q8_0 down and up wire and a float32
// inject. hc_lowrank_mix is the whole mixer as one op, and takes each weight
// in any codec of hc_lowrank_mix_codecs (kq_hc_ext.h). Four streams baked.
// GPU only, like the dsa ops.
#include <algorithm>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

constexpr int HC = 4;
constexpr int Q8_GROUP = 32;
constexpr int Q8_BLOCK_BYTES = 34;
constexpr int MAX_LR = 512; // KQ_HCLR_MAX_LRBLK * 32

// x [..., 4, D]: validates the stream axis and the alignment the kernels
// assume against the half dtype `hd`; returns D.
int check_hclr_streams(const mx::array& x, mx::Dtype hd, const char* op) {
  if (x.ndim() < 2 || x.shape(-2) != HC) {
    throw std::invalid_argument(std::string(op) + " needs [..., 4, D].");
  }
  int D = x.shape(-1);
  if (D % 64 != 0 || D > 8 * 1024) {
    throw std::invalid_argument(
        std::string(op) + " D must be a multiple of 64 and at most 8192.");
  }
  auto xd = x.dtype();
  if (xd != mx::float32 && xd != mx::float16 && xd != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " streams must be float32, float16 or bfloat16.");
  }
  if (hd != mx::float16 && hd != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + " the half dtype must be float16 or bfloat16.");
  }
  if (xd != mx::float32 && xd != hd) {
    throw std::invalid_argument(
        std::string(op) + " half streams must match the half dtype.");
  }
  return D;
}

void check_hclr_wire(
    const mx::array& w,
    int rows,
    int K,
    const char* op,
    const char* what) {
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must be uint8 q8_0 wire.");
  }
  int64_t row_bytes = int64_t(K) / Q8_GROUP * Q8_BLOCK_BYTES;
  if (w.ndim() != 2 || w.shape(0) != rows || w.shape(1) != row_bytes) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must be [" + std::to_string(rows) +
        ", " + std::to_string(row_bytes) + "] q8_0 wire bytes.");
  }
}

mx::array prep_c(const mx::array& a, mx::Stream s) {
  return mx::contiguous(a, false, s);
}

// The codecs of the hc_lowrank_mix kernels (kq_hc_ext.metal).
const std::vector<std::string>& hc_mix_codecs() {
  static const std::vector<std::string> v = {
      "q2_k",    "q3_k",   "q4_k",  "q5_k",   "q6_k",   "q8_0",   "q4_0",
      "q4_1",    "q5_0",   "q5_1",  "iq4_nl", "iq4_xs", "iq3_s",  "iq3_xxs",
      "iq2_xxs", "iq2_xs", "iq2_s", "iq1_s",  "iq1_m",  "stq1_0", "pq2_0",
      "q2_0",    "ptq1_0", "mxfp4", "f32",    "f16",    "bf16"};
  return v;
}

// A mixer weight's codec: its kquant type for uint8 wire (q8_0 when none
// is named), else f32, f16 or bf16 by dtype.
std::string hc_codec(const mx::array& w, const std::string& t, const char* op) {
  std::string c;
  if (w.dtype() == mx::uint8) {
    c = t.empty() ? "q8_0" : t;
  } else if (w.dtype() == mx::float32) {
    c = "f32";
  } else if (w.dtype() == mx::float16) {
    c = "f16";
  } else if (w.dtype() == mx::bfloat16) {
    c = "bf16";
  }
  const auto& v = hc_mix_codecs();
  if (c.empty() || std::find(v.begin(), v.end(), c) == v.end()) {
    throw std::invalid_argument(
        std::string(op) + " no kernel reads a weight of codec '" + c +
        "' (dtype or kquant type).");
  }
  return c;
}

// Values a block of codec c (16 for float weights, whose kernels read
// 16 values a block).
int hc_block(const std::string& c) {
  if (c == "f32" || c == "f16" || c == "bf16") {
    return 16;
  }
  return codec_by_name(c)->weights_per_block;
}

// w is [rows, n] values of codec c.
void check_hc_weight(
    const mx::array& w,
    const std::string& c,
    int rows,
    int n,
    const char* op,
    const char* what) {
  if (n % hc_block(c) != 0) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " rows of " + std::to_string(n) +
        " values do not fill " + c + " blocks.");
  }
  int64_t cols = n;
  if (w.dtype() == mx::uint8) {
    const KQuantCodec* k = codec_by_name(c);
    cols = int64_t(n) / k->weights_per_block * k->bytes_per_block;
  }
  if (w.ndim() != 2 || w.shape(0) != rows || w.shape(1) != cols) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must be [" + std::to_string(rows) +
        ", " + std::to_string(cols) + "] for codec " + c + ".");
  }
}

} // namespace

std::vector<std::string> hc_lowrank_mix_codecs() {
  return hc_mix_codecs();
}

#ifdef _METAL_

namespace {

// The dispatches of the three kernels, shared by the three ops and by
// hc_lowrank_mix so that a row gets the same bits from either.
void hclr_norm_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& h,
    const mx::array& gamma,
    mx::array& xn,
    float eps) {
  int D = h.shape(-1);
  int rows = int(h.size() / (int64_t(HC) * D));
  auto kernel = kq_get_kernel(
      d,
      "kq_hc_lowrank_norm_" + kq_type_string(h.dtype()) + "_" +
          kq_type_string(gamma.dtype()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(h, 0);
  ce.set_input_array(gamma, 1);
  ce.set_output_array(xn, 2);
  ce.set_bytes(D, 3);
  ce.set_bytes(eps, 4);
  ce.dispatch_threadgroups(MTL::Size(HC, rows, 1), MTL::Size(256, 1, 1));
}

// Rows of xn a front threadgroup takes, reading each down weight once for
// them: 2 above HC_FRONT_RB2_ROWS rows, else 1, or KQ_HC_FRONT_RB (1, 2 or
// 4). The kernel is latency bound, so fewer threadgroups cost more than the
// reads save until the grid is wide. A row's bits do not depend on it.
constexpr int HC_FRONT_RB2_ROWS = 12;

void hclr_front_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& xn,
    const mx::array& w_down,
    const mx::array& w_inject,
    mx::array& lo,
    mx::array& inj) {
  int D = xn.shape(-1);
  int rows = int(xn.size() / (int64_t(HC) * D));
  int LR = lo.shape(-1);
  int rb =
      kq_env_lever_from("KQ_HC_FRONT_RB", rows > HC_FRONT_RB2_ROWS ? 2 : 1, 1);
  rb = rb >= 4 ? 4 : rb >= 2 ? 2 : 1;
  while (rb > 1 && rb / 2 >= rows) {
    rb /= 2;
  }
  auto kernel = kq_get_kernel(
      d,
      "kq_hc_lowrank_front_r" + std::to_string(rb) + "_" +
          kq_type_string(xn.dtype()) + "_" + kq_type_string(lo.dtype()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(xn, 0);
  ce.set_input_array(w_down, 1);
  ce.set_input_array(w_inject, 2);
  ce.set_output_array(lo, 3);
  ce.set_output_array(inj, 4);
  ce.set_bytes(D, 5);
  ce.set_bytes(LR, 6);
  ce.set_bytes(rows, 7);
  ce.dispatch_threadgroups(
      MTL::Size(LR / 8 + 1, (rows + rb - 1) / rb, 1), MTL::Size(64, 1, 1));
}

// Epilogue forms. KQ_HC_EPI_LPR=4, 8 or 16 sets the lanes the simdgroup
// kernel gives the up rows of a d and 0 picks the one-lane kernel (read per
// call); the two give different bits. The default is 8 on GPUs with the
// tensor-op units (the GPUs the simdgroup kernel was timed on) and 0
// elsewhere. KQ_HC_EPI_RB (1, 2, 4 or 8) caps the rows a threadgroup of the
// simdgroup kernel takes; a row's bits do not depend on it.
constexpr int HC_EPI_LPR = 8;
constexpr int HC_EPI_RB = 4;
inline int hc_epi_lpr() {
  const int unset = kq_is_nax_available() ? HC_EPI_LPR : 0;
  const int lpr = kq_env_lever_from("KQ_HC_EPI_LPR", unset, 0);
  return lpr == 0 || lpr == 4 || lpr == 8 || lpr == 16 ? lpr : unset;
}

// With inv and gamma, xn holds the streams h and the kernel rebuilds the
// normalized values from them (the folded-norm front's form).
void hclr_epilogue_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& lo_in,
    const mx::array& w_up,
    const mx::array& xn,
    mx::array& mixed,
    const mx::array* inv = nullptr,
    const mx::array* gamma = nullptr) {
  int D = xn.shape(-1);
  int rows = int(xn.size() / (int64_t(HC) * D));
  int LR = lo_in.shape(-1);
  // The simdgroup kernel reads lo four values a load.
  const mx::array lo = kq_x_aligned_gpu(lo_in, s, "hc_lowrank_epilogue lo", 8);
  const int lpr = hc_epi_lpr();
  int rb = kq_env_lever_from("KQ_HC_EPI_RB", HC_EPI_RB, 1);
  rb = rb >= 8 ? 8 : rb >= 4 ? 4 : rb >= 2 ? 2 : 1;
  const size_t lo_bytes = mx::size_of(lo.dtype());
  while (rb > 1 && (rb / 2 >= rows || rb * LR * lo_bytes > 16384)) {
    rb /= 2;
  }
  const std::string types =
      kq_type_string(xn.dtype()) + "_" + kq_type_string(lo.dtype());
  std::string kname = "kq_hc_lowrank_epilogue_" + types;
  if (lpr != 0) {
    kname = std::string("kq_hc_lowrank_epilogue_") + (inv ? "fn_" : "") + "mr" +
        std::to_string(lpr) + "_r" + std::to_string(rb) + "_" + types;
  }
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(lo, 0);
  ce.set_input_array(w_up, 1);
  ce.set_input_array(xn, 2);
  ce.set_output_array(mixed, 3);
  ce.set_bytes(D, 4);
  ce.set_bytes(LR, 5);
  ce.set_bytes(rows, 6);
  if (inv) {
    ce.set_input_array(*inv, 7);
    ce.set_input_array(*gamma, 8);
  }
  if (lpr == 0) {
    // One lane per (row, d) output; see the kernel comment for the mapping.
    int64_t outs = int64_t(rows) * D;
    ce.dispatch_threadgroups(
        MTL::Size((outs + 255) / 256, 1, 1), MTL::Size(256, 1, 1));
  } else {
    ce.set_threadgroup_memory_length(
        size_t(std::min(rb, rows)) * LR * lo_bytes, 0);
    ce.dispatch_threadgroups(
        MTL::Size(D / (32 / lpr) / 4, (rows + rb - 1) / rb, 1),
        MTL::Size(32, 4, 1));
  }
}

// The folded-norm front (kq_hc_lowrank_front_fn): float32 streams, q8_0
// down, float32 inject. KQ_HC_FRONT_FN_RB (1, 2, 4 or 8) sets the rows a
// threadgroup takes (default 2 above HC_FRONT_FN_RB2_ROWS rows, else 1) and
// KQ_HC_FRONT_FN_NO (4 or 8) its down outputs; a row's bits depend on
// neither. KQ_HC_FRONT_FN_SPS (1, 2 or 4, RB 1 or 2 with 4 outputs) sets
// the simdgroups on a stream: 2 up to HC_FRONT_FN_SPS2_ROWS rows, where it
// is faster, else 1. Each count gives other bits.
constexpr int HC_FRONT_FN_RB2_ROWS = 3;
constexpr int HC_FRONT_FN_SPS2_ROWS = 1;
constexpr int HC_EPI_MMA_ROWS = 2;
constexpr int HC_FRONT_FN_NO = 4;

void hclr_front_fn_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& h,
    const mx::array& gamma,
    const mx::array& w_down,
    const mx::array& w_inject,
    mx::array& lo,
    mx::array& inj,
    mx::array& inv,
    float eps) {
  int D = h.shape(-1);
  int rows = int(h.size() / (int64_t(HC) * D));
  int LR = lo.shape(-1);
  auto bind = [&](auto& ce) {
    ce.set_input_array(h, 0);
    ce.set_input_array(gamma, 1);
    ce.set_input_array(w_down, 2);
    ce.set_input_array(w_inject, 3);
    ce.set_output_array(lo, 4);
    ce.set_output_array(inj, 5);
    ce.set_output_array(inv, 6);
    ce.set_bytes(D, 7);
    ce.set_bytes(LR, 8);
    ce.set_bytes(rows, 9);
    ce.set_bytes(eps, 10);
  };
  // Up to 16 rows take the MMA form, all rows one operand
  // (kq_hc_lowrank_front_mma) under KQ_HC_FRONT_MMA=1. It ties the lane
  // form at 8 rows and loses below, so it is off. The two give different
  // bits.
  if (rows <= 16 && kq_env_lever_from("KQ_HC_FRONT_MMA", 0, 0) != 0) {
    auto kernel = kq_get_kernel(
        d,
        std::string("kq_hc_lowrank_front_mma_f") + (rows <= 8 ? "1" : "2") +
            "_" + kq_type_string(gamma.dtype()));
    auto& ce = mx::metal::get_command_encoder(s);
    ce.set_compute_pipeline_state(kernel);
    bind(ce);
    ce.dispatch_threadgroups(
        MTL::Size(LR / 8 + 1, 1, 1), MTL::Size(32 * 8, 1, 1));
    return;
  }
  int rb = kq_env_lever_from(
      "KQ_HC_FRONT_FN_RB", rows > HC_FRONT_FN_RB2_ROWS ? 2 : 1, 1);
  rb = rb >= 8 ? 8 : rb >= 4 ? 4 : rb >= 2 ? 2 : 1;
  while (rb > 1 && rb / 2 >= rows) {
    rb /= 2;
  }
  const int no =
      kq_env_lever_from("KQ_HC_FRONT_FN_NO", HC_FRONT_FN_NO, 1) >= 8 ? 8 : 4;
  int sps = kq_env_lever_from(
      "KQ_HC_FRONT_FN_SPS", rows <= HC_FRONT_FN_SPS2_ROWS ? 2 : 1, 1);
  sps = (no != 4 || rb > 2) ? 1 : sps >= 4 ? 4 : sps >= 2 ? 2 : 1;
  auto kernel = kq_get_kernel(
      d,
      "kq_hc_lowrank_front_fn_r" + std::to_string(rb) + "_o" +
          std::to_string(no) + (sps > 1 ? "_s" + std::to_string(sps) : "") +
          "_" + kq_type_string(gamma.dtype()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(h, 0);
  ce.set_input_array(gamma, 1);
  ce.set_input_array(w_down, 2);
  ce.set_input_array(w_inject, 3);
  ce.set_output_array(lo, 4);
  ce.set_output_array(inj, 5);
  ce.set_output_array(inv, 6);
  ce.set_bytes(D, 7);
  ce.set_bytes(LR, 8);
  ce.set_bytes(rows, 9);
  ce.set_bytes(eps, 10);
  ce.dispatch_threadgroups(
      MTL::Size(LR / no + 1, (rows + rb - 1) / rb, 1),
      MTL::Size(32, HC * sps, 1));
}

// The kernels of kq_hc_ext.h, for weights of any codec of
// hc_lowrank_mix_codecs. Front: 32 / HC_EXT_NX down rows a simdgroup, four
// simdgroups. Inject: a simdgroup per inject row. Epilogue: HC_EXT_L lanes
// per output, four rows a threadgroup.
constexpr int HC_EXT_NX = 16;
constexpr int HC_EXT_L = 4;

void hclr_front_ext_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& xn,
    const mx::array& w_down,
    const std::string& codec,
    mx::array& lo) {
  int D = xn.shape(-1);
  int rows = int(xn.size() / (int64_t(HC) * D));
  int K = HC * D;
  int LR = lo.shape(-1);
  auto kernel = kq_get_kernel(
      d,
      "kq_hc_front_ext_" + codec + "_" + kq_type_string(xn.dtype()) + "_" +
          kq_type_string(lo.dtype()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(xn, 0);
  ce.set_input_array(w_down, 1);
  ce.set_output_array(lo, 2);
  ce.set_bytes(K, 3);
  ce.set_bytes(LR, 4);
  ce.set_bytes(rows, 5);
  ce.dispatch_threadgroups(
      MTL::Size(LR / (4 * (32 / HC_EXT_NX)), (rows + 3) / 4, 1),
      MTL::Size(32, 4, 1));
}

void hclr_inject_ext_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& xn,
    const mx::array& w_inject,
    const std::string& codec,
    mx::array& inj) {
  int D = xn.shape(-1);
  int rows = int(xn.size() / (int64_t(HC) * D));
  int K = HC * D;
  auto kernel = kq_get_kernel(
      d,
      "kq_hc_inject_ext_" + codec + "_" + kq_type_string(xn.dtype()) + "_" +
          kq_type_string(inj.dtype()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(xn, 0);
  ce.set_input_array(w_inject, 1);
  ce.set_output_array(inj, 2);
  ce.set_bytes(K, 3);
  ce.set_bytes(rows, 4);
  ce.dispatch_threadgroups(
      MTL::Size(1, (rows + 3) / 4, 1), MTL::Size(32, HC, 1));
}

void hclr_epilogue_ext_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& lo,
    const mx::array& w_up,
    const std::string& codec,
    const mx::array& xn,
    mx::array& mixed) {
  int D = xn.shape(-1);
  int rows = int(xn.size() / (int64_t(HC) * D));
  int LR = lo.shape(-1);
  auto kernel = kq_get_kernel(
      d,
      "kq_hc_epilogue_ext_" + codec + "_" + kq_type_string(xn.dtype()) + "_" +
          kq_type_string(lo.dtype()));
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(lo, 0);
  ce.set_input_array(w_up, 1);
  ce.set_input_array(xn, 2);
  ce.set_output_array(mixed, 3);
  ce.set_bytes(D, 4);
  ce.set_bytes(LR, 5);
  ce.set_bytes(rows, 6);
  ce.set_threadgroup_memory_length(
      size_t(std::min(4, rows)) * LR * mx::size_of(lo.dtype()), 0);
  ce.dispatch_threadgroups(
      MTL::Size(D / (32 / HC_EXT_L) / 4, (rows + 3) / 4, 1),
      MTL::Size(32, 4, 1));
}

} // namespace

void KQuantHcLowrankNorm::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  hclr_norm_gpu(
      mx::metal::device(s.device), s, inputs[0], inputs[1], out, eps_);
}

void KQuantHcLowrankFront::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
  hclr_front_gpu(
      mx::metal::device(s.device),
      s,
      inputs[0],
      inputs[1],
      inputs[2],
      outputs[0],
      outputs[1]);
}

void KQuantHcLowrankEpilogue::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  hclr_epilogue_gpu(
      mx::metal::device(s.device), s, inputs[0], inputs[1], inputs[2], out);
}

// The mixer in one primitive. q8_0 down with a float32 inject takes the
// fused front, q8_0 up the simdgroup epilogue, and other codecs the
// kernels of kq_hc_ext.h. xn and lo are temporaries of the command buffer,
// and an operand that is row-major is bound as it is.
void KQuantHcLowrankMix::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
  const mx::array h = kq_row_major_gpu(inputs[0], s);
  const mx::array gamma = kq_row_major_gpu(inputs[1], s);
  const mx::array wd = kq_row_major_gpu(inputs[2], s);
  const mx::array wi = kq_row_major_gpu(inputs[3], s);
  const mx::array wu = kq_row_major_gpu(inputs[4], s);
  const int D = h.shape(-1);
  const int rows = int(h.size() / (int64_t(HC) * D));
  // Float32 streams with q8_0 down and up and a float32 inject fold the
  // norm into the front: two dispatches, no xn. KQ_HC_MIX_FN=0 keeps the
  // three kernels (the bits of the three ops).
  if (h.dtype() == mx::float32 && down_type_ == "q8_0" &&
      inject_type_ == "f32" && up_type_ == "q8_0" && hc_epi_lpr() != 0 &&
      kq_env_lever_from("KQ_HC_MIX_FN", 1, 0) != 0) {
    mx::array lo({rows, int(wd.shape(0))}, gamma.dtype(), nullptr, {});
    lo.set_data(mx::allocator::malloc(lo.nbytes()));
    mx::array inv({rows, HC}, mx::float32, nullptr, {});
    inv.set_data(mx::allocator::malloc(inv.nbytes()));
    hclr_front_fn_gpu(d, s, h, gamma, wd, wi, lo, outputs[1], inv, eps_);
    // KQ_HC_EPI_MMA: fewest rows that take the MMA epilogue
    // (kq_hc_lowrank_epilogue_mma), 0 for never. Its bits differ from the
    // lane epilogue's.
    const int LR = wd.shape(0);
    const int mma_rows = kq_env_lever_from("KQ_HC_EPI_MMA", HC_EPI_MMA_ROWS, 0);
    if (mma_rows > 0 && rows >= mma_rows && rows <= 16 && LR % 64 == 0 &&
        D % 8 == 0) {
      auto kernel = kq_get_kernel(
          d,
          std::string("kq_hc_lowrank_epilogue_mma_f") +
              (rows <= 8 ? "1" : "2") + "_" + kq_type_string(gamma.dtype()));
      auto& ce = mx::metal::get_command_encoder(s);
      ce.set_compute_pipeline_state(kernel);
      ce.set_input_array(lo, 0);
      ce.set_input_array(wu, 1);
      ce.set_input_array(h, 2);
      ce.set_output_array(outputs[0], 3);
      ce.set_bytes(D, 4);
      ce.set_bytes(LR, 5);
      ce.set_bytes(rows, 6);
      ce.set_input_array(inv, 7);
      ce.set_input_array(gamma, 8);
      ce.dispatch_threadgroups(MTL::Size(D / 8, 1, 1), MTL::Size(32 * 4, 1, 1));
    } else {
      hclr_epilogue_gpu(d, s, lo, wu, h, outputs[0], &inv, &gamma);
    }
    auto& ce = mx::metal::get_command_encoder(s);
    ce.add_temporary(lo);
    ce.add_temporary(inv);
    return;
  }
  mx::array xn(h.shape(), h.dtype(), nullptr, {});
  xn.set_data(mx::allocator::malloc(xn.nbytes()));
  mx::array lo({rows, int(wd.shape(0))}, gamma.dtype(), nullptr, {});
  lo.set_data(mx::allocator::malloc(lo.nbytes()));
  hclr_norm_gpu(d, s, h, gamma, xn, eps_);
  if (down_type_ == "q8_0" && inject_type_ == "f32") {
    hclr_front_gpu(d, s, xn, wd, wi, lo, outputs[1]);
  } else {
    hclr_front_ext_gpu(d, s, xn, wd, down_type_, lo);
    hclr_inject_ext_gpu(d, s, xn, wi, inject_type_, outputs[1]);
  }
  if (up_type_ == "q8_0") {
    hclr_epilogue_gpu(d, s, lo, wu, xn, outputs[0]);
  } else {
    hclr_epilogue_ext_gpu(d, s, lo, wu, up_type_, xn, outputs[0]);
  }
  auto& ce = mx::metal::get_command_encoder(s);
  ce.add_temporary(xn);
  ce.add_temporary(lo);
}

#else // !_METAL_

void KQuantHcLowrankMix::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.hc_lowrank_mix] requires Metal.");
}

void KQuantHcLowrankNorm::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.hc_lowrank_norm] requires Metal.");
}

void KQuantHcLowrankFront::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.hc_lowrank_front] requires Metal.");
}

void KQuantHcLowrankEpilogue::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.hc_lowrank_epilogue] requires Metal.");
}

#endif

void KQuantHcLowrankNorm::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.hc_lowrank_norm] has no CPU implementation.");
}

void KQuantHcLowrankFront::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.hc_lowrank_front] has no CPU implementation.");
}

void KQuantHcLowrankEpilogue::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.hc_lowrank_epilogue] has no CPU implementation.");
}

void KQuantHcLowrankMix::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.hc_lowrank_mix] has no CPU implementation.");
}

bool KQuantHcLowrankMix::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantHcLowrankMix&>(other);
  return eps_ == o.eps_ && down_type_ == o.down_type_ &&
      inject_type_ == o.inject_type_ && up_type_ == o.up_type_;
}

bool KQuantHcLowrankNorm::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantHcLowrankNorm&>(other);
  return eps_ == o.eps_;
}

mx::array hc_lowrank_norm(
    mx::array h,
    mx::array gamma,
    float eps,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.hc_lowrank_norm]";
  int D = check_hclr_streams(h, gamma.dtype(), op);
  if (gamma.ndim() != 1 || gamma.shape(0) != HC * D) {
    throw std::invalid_argument(std::string(op) + " gamma must be [4 * D].");
  }
  auto h_c = prep_c(h, s);
  auto g_c = prep_c(gamma, s);
  return mx::array(
      h.shape(),
      h.dtype(),
      std::make_shared<KQuantHcLowrankNorm>(s, eps),
      {std::move(h_c), std::move(g_c)});
}

std::vector<mx::array> hc_lowrank_front(
    mx::array xn,
    mx::array w_down,
    mx::array w_inject,
    mx::Dtype lo_dtype,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.hc_lowrank_front]";
  int D = check_hclr_streams(xn, lo_dtype, op);
  int K = HC * D;
  if (w_down.ndim() != 2) {
    throw std::invalid_argument(std::string(op) + " w_down must be 2-d.");
  }
  int LR = w_down.shape(0);
  if (LR % 32 != 0 || LR > MAX_LR) {
    throw std::invalid_argument(
        std::string(op) + " lowrank must be a multiple of 32, at most 512.");
  }
  check_hclr_wire(w_down, LR, K, op, "w_down");
  if (w_inject.dtype() != mx::float32 || w_inject.ndim() != 2 ||
      w_inject.shape(0) != HC || w_inject.shape(1) != K) {
    throw std::invalid_argument(
        std::string(op) + " w_inject must be float32 [4, 4 * D].");
  }

  auto x_c = prep_c(xn, s);
  auto wd_c = prep_c(w_down, s);
  auto wi_c = prep_c(w_inject, s);

  auto lead = xn.shape();
  lead.pop_back();
  lead.pop_back();
  auto lo_shape = lead;
  lo_shape.push_back(LR);
  auto inj_shape = lead;
  inj_shape.push_back(HC);
  return mx::array::make_arrays(
      {std::move(lo_shape), std::move(inj_shape)},
      {lo_dtype, mx::float32},
      std::make_shared<KQuantHcLowrankFront>(s),
      {std::move(x_c), std::move(wd_c), std::move(wi_c)});
}

mx::array hc_lowrank_epilogue(
    mx::array lo,
    mx::array w_up,
    mx::array xn,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.hc_lowrank_epilogue]";
  if (lo.ndim() < 1 ||
      (lo.dtype() != mx::float16 && lo.dtype() != mx::bfloat16)) {
    throw std::invalid_argument(
        std::string(op) + " lo must be [..., LR] float16/bfloat16.");
  }
  int D = check_hclr_streams(xn, lo.dtype(), op);
  int64_t rows = xn.size() / (int64_t(HC) * D);
  int LR = lo.shape(-1);
  if (LR % 32 != 0 || LR > MAX_LR || int64_t(lo.size()) != rows * LR) {
    throw std::invalid_argument(
        std::string(op) +
        " lo must be [..., LR], LR a multiple of 32, at "
        "most 512, rows matching xn.");
  }
  check_hclr_wire(w_up, HC * D, LR, op, "w_up");

  auto lo_c = prep_c(lo, s);
  auto wu_c = prep_c(w_up, s);
  auto x_c = prep_c(xn, s);

  auto lead = xn.shape();
  lead.pop_back();
  lead.pop_back();
  auto out_shape = lead;
  out_shape.push_back(D);
  return mx::array(
      std::move(out_shape),
      xn.dtype(),
      std::make_shared<KQuantHcLowrankEpilogue>(s),
      {std::move(lo_c), std::move(wu_c), std::move(x_c)});
}

std::vector<mx::array> hc_lowrank_mix(
    mx::array h,
    mx::array gamma,
    mx::array w_down,
    mx::array w_inject,
    mx::array w_up,
    float eps,
    const std::string& down_type,
    const std::string& inject_type,
    const std::string& up_type,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.hc_lowrank_mix]";
  int D = check_hclr_streams(h, gamma.dtype(), op);
  int K = HC * D;
  if (gamma.ndim() != 1 || gamma.shape(0) != K) {
    throw std::invalid_argument(std::string(op) + " gamma must be [4 * D].");
  }
  if (w_down.ndim() != 2) {
    throw std::invalid_argument(std::string(op) + " w_down must be 2-d.");
  }
  int LR = w_down.shape(0);
  if (LR % 32 != 0 || LR > MAX_LR) {
    throw std::invalid_argument(
        std::string(op) + " lowrank must be a multiple of 32, at most 512.");
  }
  const std::string cd = hc_codec(w_down, down_type, op);
  const std::string ci = hc_codec(w_inject, inject_type, op);
  const std::string cu = hc_codec(w_up, up_type, op);
  check_hc_weight(w_down, cd, LR, K, op, "w_down");
  check_hc_weight(w_inject, ci, HC, K, op, "w_inject");
  check_hc_weight(w_up, cu, K, LR, op, "w_up");
  // The inject output takes the dtype of the eager matmul: the streams'
  // for wire, else the promotion of the streams' and the weight's.
  const mx::Dtype inj_dtype = w_inject.dtype() == mx::uint8
      ? h.dtype()
      : mx::promote_types(h.dtype(), w_inject.dtype());

  auto lead = h.shape();
  lead.pop_back();
  lead.pop_back();
  auto mixed_shape = lead;
  mixed_shape.push_back(D);
  auto inj_shape = lead;
  inj_shape.push_back(HC);
  return mx::array::make_arrays(
      {std::move(mixed_shape), std::move(inj_shape)},
      {h.dtype(), inj_dtype},
      std::make_shared<KQuantHcLowrankMix>(s, eps, cd, ci, cu),
      {std::move(h),
       std::move(gamma),
       std::move(w_down),
       std::move(w_inject),
       std::move(w_up)});
}

} // namespace mlx_kquant
