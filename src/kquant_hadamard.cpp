// Signed block Walsh-Hadamard rotation of activation rows, the run-time
// half of a Hadamard-folded weight (PrismML's Ternary Bonsai GGUFs store
// W H diag(s) and expect y = W (H (s * x)) at run time), alone or fused
// onto the gated activation that produces the row. One dispatch per call;
// the CPU evals mirror the kernels' f32 math and rounding points.
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

constexpr int kHadamardNT = 256;

bool hadamard_block_supported(int block) {
  return block == 256 || block == 512 || block == 1024 || block == 2048 ||
      block == 4096;
}

// Source index of permuted position p: tiled [rep, nk, hd] -> grouped
// [nk, rep, hd] along the row.
inline int perm_source(int p, int rep, int nk, int hd) {
  const int span = rep * hd;
  const int k = p / span;
  const int rem = p - k * span;
  const int r = rem / hd;
  const int h = rem - r * hd;
  return (r * nk + k) * hd + h;
}

// Unnormalized butterflies over each block-wide chunk of v[K], in place.
void fwht_blocks(float* v, int K, int block) {
  for (int b = 0; b < K; b += block) {
    float* c = v + b;
    for (int i = 1; i < block; i *= 2) {
      for (int j = 0; j < block; j += 2 * i) {
        for (int k = 0; k < i; k++) {
          const float a = c[j + k];
          const float d = c[j + k + i];
          c[j + k] = a + d;
          c[j + k + i] = a - d;
        }
      }
    }
  }
}

float sigmoid_f32(float g) {
  const float e = std::exp(-std::fabs(g));
  const float p = 1.0f / (1.0f + e);
  return g < 0.0f ? e * p : p;
}

// Activation dtype, block and width checks shared by the rotation ops.
void check_rotation_args(
    const mx::array& x,
    int block,
    const std::optional<mx::array>& signs,
    const char* op) {
  if (x.ndim() < 1 || x.shape(-1) < 1) {
    throw std::invalid_argument(std::string(op) + " x must have a last axis.");
  }
  const auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16 && dt != mx::float32) {
    throw std::invalid_argument(
        std::string(op) + " x must be float16, bfloat16 or float32.");
  }
  if (!hadamard_block_supported(block)) {
    throw std::invalid_argument(
        std::string(op) +
        " block must be one of 256, 512, 1024, 2048, 4096; got " +
        std::to_string(block) + ".");
  }
  const int K = x.shape(-1);
  if (K % block != 0) {
    throw std::invalid_argument(
        std::string(op) + " last axis " + std::to_string(K) +
        " is not a multiple of block " + std::to_string(block) + ".");
  }
  if (signs.has_value()) {
    const auto& sg = *signs;
    if (sg.ndim() != 1 || sg.shape(0) != K || sg.dtype() != mx::float32) {
      throw std::invalid_argument(
          std::string(op) + " signs must be float32 [" + std::to_string(K) +
          "].");
    }
  }
}

// An operand that must match x in shape and dtype.
void check_like_x(
    const mx::array& a,
    const mx::array& x,
    const char* op,
    const char* what) {
  if (a.shape() != x.shape() || a.dtype() != x.dtype()) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must match x in shape and dtype.");
  }
}

template <typename F>
void hadamard_cpu_dispatch(mx::Dtype dt, F&& run) {
  if (dt == mx::float16) {
    run(static_cast<mx::float16_t*>(nullptr));
  } else if (dt == mx::bfloat16) {
    run(static_cast<mx::bfloat16_t*>(nullptr));
  } else {
    run(static_cast<float*>(nullptr));
  }
}

} // namespace

#ifdef _METAL_

namespace {

// The rotation kernels run KQ_HADAMARD_NT threads per threadgroup; a
// pipeline admitting fewer would run a partial group silently.
template <typename K>
void require_group_width(K kernel, const std::string& kname, const char* op) {
  if (int(kernel->maxTotalThreadsPerThreadgroup()) < kHadamardNT) {
    throw std::runtime_error(
        std::string(op) + " pipeline " + kname + " admits fewer than " +
        std::to_string(kHadamardNT) + " threads per threadgroup.");
  }
}

} // namespace

void KQuantHadamard::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& x = inputs[0];
  // signs must be a bound buffer even when unused; x stands in.
  const auto& signs = has_signs_ ? inputs[1] : inputs[0];
  const int K = x.shape(-1);
  const int n_blk = K / block_;
  const int rows = int(x.size() / K);
  const int has_signs = has_signs_ ? 1 : 0;

  std::string kname =
      "kq_hadamard_" + kq_type_string(x.dtype()) + "_" + std::to_string(block_);
  auto kernel = kq_get_kernel(d, kname);
  require_group_width(kernel, kname, "[mlx_kquant.hadamard_rotate]");
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(x, 0);
  ce.set_input_array(signs, 1);
  ce.set_output_array(out, 2);
  ce.set_bytes(n_blk, 3);
  ce.set_bytes(has_signs, 4);
  ce.set_bytes(perm_rep_, 5);
  ce.set_bytes(perm_nk_, 6);
  ce.set_bytes(perm_hd_, 7);
  MTL::Size group_dims(kHadamardNT, 1, 1);
  MTL::Size grid_dims(size_t(rows) * n_blk, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

void KQuantGluHadamard::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  const auto& x = inputs[0];
  const auto& gate = inputs[1];
  const auto& signs = has_signs_ ? inputs[2] : inputs[0];
  const int K = x.shape(-1);
  const int n_blk = K / block_;
  const int rows = int(x.size() / K);
  const int has_signs = has_signs_ ? 1 : 0;

  std::string kname = std::string("kq_glu_hadamard_") +
      (sigmoid_ ? "sigmoid_" : "silu_") + kq_type_string(x.dtype()) + "_" +
      std::to_string(block_);
  auto kernel = kq_get_kernel(d, kname);
  require_group_width(kernel, kname, "[mlx_kquant.glu_hadamard]");
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(x, 0);
  ce.set_input_array(gate, 1);
  ce.set_input_array(signs, 2);
  ce.set_output_array(out, 3);
  ce.set_bytes(n_blk, 4);
  ce.set_bytes(has_signs, 5);
  MTL::Size group_dims(kHadamardNT, 1, 1);
  MTL::Size grid_dims(size_t(rows) * n_blk, 1, 1);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantHadamard::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.hadamard_rotate] requires Metal.");
}

void KQuantGluHadamard::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.glu_hadamard] requires Metal.");
}

#endif

void KQuantHadamard::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& x = inputs[0];
  const auto& signs = has_signs_ ? inputs[1] : inputs[0];

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(signs);
  encoder.set_output_array(out);
  encoder.dispatch([out = mx::array::unsafe_weak_copy(out),
                    x = mx::array::unsafe_weak_copy(x),
                    signs = mx::array::unsafe_weak_copy(signs),
                    block = block_,
                    has_signs = has_signs_,
                    rep = perm_rep_,
                    nk = perm_nk_,
                    hd = perm_hd_]() mutable {
    const int K = x.shape(-1);
    const int64_t rows = x.size() / K;
    const float scale = 1.0f / std::sqrt(float(block));
    const float* sp = has_signs ? signs.data<float>() : nullptr;
    std::vector<float> buf(K);
    hadamard_cpu_dispatch(x.dtype(), [&](auto* tag) {
      using DT = std::remove_pointer_t<decltype(tag)>;
      const DT* xp = x.data<DT>();
      DT* op = out.data<DT>();
      for (int64_t t = 0; t < rows; t++) {
        const DT* xrow = xp + t * K;
        DT* orow = op + t * K;
        for (int p = 0; p < K; p++) {
          const int src = hd > 0 ? perm_source(p, rep, nk, hd) : p;
          const float s = sp ? sp[p] : 1.0f;
          buf[p] = static_cast<float>(xrow[src]) * s * scale;
        }
        fwht_blocks(buf.data(), K, block);
        for (int p = 0; p < K; p++) {
          orow[p] = static_cast<DT>(buf[p]);
        }
      }
    });
  });
}

bool KQuantHadamard::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantHadamard&>(other);
  return block_ == o.block_ && has_signs_ == o.has_signs_ &&
      perm_rep_ == o.perm_rep_ && perm_nk_ == o.perm_nk_ &&
      perm_hd_ == o.perm_hd_;
}

std::vector<mx::Shape> KQuantHadamard::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

mx::array hadamard_rotate(
    mx::array x,
    const std::optional<mx::array>& signs,
    int block,
    const std::optional<std::tuple<int, int, int>>& perm,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.hadamard_rotate]";
  check_rotation_args(x, block, signs, op);
  const auto dt = x.dtype();
  const int K = x.shape(-1);
  int rep = 0, nk = 0, hd = 0;
  if (perm.has_value()) {
    std::tie(rep, nk, hd) = *perm;
    if (rep < 1 || nk < 1 || hd < 1 || rep * nk * hd != K) {
      throw std::invalid_argument(
          std::string(op) + " perm (rep, nk, hd) must multiply to " +
          std::to_string(K) + ".");
    }
  }
  std::vector<mx::array> inputs = {mx::contiguous(x, false, s)};
  if (signs.has_value()) {
    inputs.push_back(mx::contiguous(*signs, false, s));
  }
  return mx::array(
      x.shape(),
      dt,
      std::make_shared<KQuantHadamard>(
          s, block, signs.has_value(), rep, nk, hd),
      std::move(inputs));
}

void KQuantGluHadamard::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  const auto& x = inputs[0];
  const auto& gate = inputs[1];
  const auto& signs = has_signs_ ? inputs[2] : inputs[0];

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(gate);
  encoder.set_input_array(signs);
  encoder.set_output_array(out);
  encoder.dispatch([x = mx::array::unsafe_weak_copy(x),
                    gate = mx::array::unsafe_weak_copy(gate),
                    signs = mx::array::unsafe_weak_copy(signs),
                    out = mx::array::unsafe_weak_copy(out),
                    block = block_,
                    has_signs = has_signs_,
                    sigmoid = sigmoid_]() mutable {
    const int K = x.shape(-1);
    const int64_t rows = x.size() / K;
    const float scale = 1.0f / std::sqrt(float(block));
    const float* sp = has_signs ? signs.data<float>() : nullptr;
    std::vector<float> buf(K);
    hadamard_cpu_dispatch(x.dtype(), [&](auto* tag) {
      using DT = std::remove_pointer_t<decltype(tag)>;
      for (int64_t t = 0; t < rows; t++) {
        const DT* xrow = x.data<DT>() + t * K;
        const DT* grow = gate.data<DT>() + t * K;
        for (int p = 0; p < K; p++) {
          const float g = static_cast<float>(grow[p]);
          const float a = sigmoid ? sigmoid_f32(g) : g * sigmoid_f32(g);
          buf[p] =
              a * static_cast<float>(xrow[p]) * (sp ? sp[p] : 1.0f) * scale;
        }
        fwht_blocks(buf.data(), K, block);
        DT* orow = out.data<DT>() + t * K;
        for (int p = 0; p < K; p++) {
          orow[p] = static_cast<DT>(buf[p]);
        }
      }
    });
  });
}

bool KQuantGluHadamard::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGluHadamard&>(other);
  return block_ == o.block_ && has_signs_ == o.has_signs_ &&
      sigmoid_ == o.sigmoid_;
}

std::vector<mx::Shape> KQuantGluHadamard::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

mx::array glu_hadamard(
    mx::array x,
    mx::array gate,
    const std::optional<mx::array>& signs,
    int block,
    const std::string& activation,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.glu_hadamard]";
  check_rotation_args(x, block, signs, op);
  check_like_x(gate, x, op, "gate");
  if (activation != "silu" && activation != "sigmoid") {
    throw std::invalid_argument(
        std::string(op) +
        " activation must be \"silu\" or \"sigmoid\"; got \"" + activation +
        "\".");
  }
  std::vector<mx::array> inputs = {
      mx::contiguous(x, false, s), mx::contiguous(gate, false, s)};
  if (signs.has_value()) {
    inputs.push_back(mx::contiguous(*signs, false, s));
  }
  return mx::array(
      x.shape(),
      x.dtype(),
      std::make_shared<KQuantGluHadamard>(
          s, block, signs.has_value(), activation == "sigmoid"),
      std::move(inputs));
}

} // namespace mlx_kquant
