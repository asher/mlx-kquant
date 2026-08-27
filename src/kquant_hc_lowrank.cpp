// Fused qwen4exp low-rank hyper-connection ops for the short-row decode
// route (see kq_hc_lowrank.h for the kernel shapes and the eager reference
// they mirror). Four streams baked; q8_0 down/up wire only for now -- the
// dot loops route through the shared q8_0 helpers so other 32-block codecs
// are a mechanical extension. GPU only, like the dsa ops.
#include <stdexcept>
#include <string>

#include "kquant.h"
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

} // namespace

#ifdef _METAL_

void KQuantHcLowrankNorm::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  const auto& h = inputs[0];
  const auto& gamma = inputs[1];
  int D = h.shape(-1);
  int rows = int(h.size() / (int64_t(HC) * D));

  std::string kname = "kq_hc_lowrank_norm_" + kq_type_string(h.dtype()) + "_" +
      kq_type_string(gamma.dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(h, 0);
  ce.set_input_array(gamma, 1);
  ce.set_output_array(out, 2);
  ce.set_bytes(D, 3);
  ce.set_bytes(eps_, 4);
  ce.dispatch_threadgroups(MTL::Size(HC, rows, 1), MTL::Size(256, 1, 1));
}

void KQuantHcLowrankFront::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
  const auto& xn = inputs[0];
  int D = xn.shape(-1);
  int rows = int(xn.size() / (int64_t(HC) * D));
  int LR = outputs[0].shape(-1);

  std::string kname = "kq_hc_lowrank_front_" + kq_type_string(xn.dtype()) +
      "_" + kq_type_string(outputs[0].dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(xn, 0);
  ce.set_input_array(inputs[1], 1);
  ce.set_input_array(inputs[2], 2);
  ce.set_output_array(outputs[0], 3);
  ce.set_output_array(outputs[1], 4);
  ce.set_bytes(D, 5);
  ce.set_bytes(LR, 6);
  ce.dispatch_threadgroups(MTL::Size(LR / 8 + 1, rows, 1), MTL::Size(64, 1, 1));
}

void KQuantHcLowrankEpilogue::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  const auto& xn = inputs[2];
  int D = xn.shape(-1);
  int rows = int(xn.size() / (int64_t(HC) * D));
  int LR = inputs[0].shape(-1);

  std::string kname = "kq_hc_lowrank_epilogue_" + kq_type_string(xn.dtype()) +
      "_" + kq_type_string(inputs[0].dtype());
  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(inputs[0], 0);
  ce.set_input_array(inputs[1], 1);
  ce.set_input_array(xn, 2);
  ce.set_output_array(out, 3);
  ce.set_bytes(D, 4);
  ce.set_bytes(LR, 5);
  ce.dispatch_threadgroups(MTL::Size(D / 2, rows, 1), MTL::Size(64, 1, 1));
}

#else // !_METAL_

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

} // namespace mlx_kquant
