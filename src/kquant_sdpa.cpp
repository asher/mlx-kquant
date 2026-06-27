// KQuantSDPA primitive: vector scaled-dot-product attention for large head dims
// (e.g. 512) that stock MLX's fused vector allowlist excludes. The GPU path
// dispatches the two-pass kernels (kq_sdpa_vector_2pass_1 / _2) from the
// bundled metallib. q is row-contiguous; k/v are read in place via their
// head/seq strides so a strided KV-cache prefix needs no copy. Inference-only
// (no CPU eval).
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

namespace {

using mx::array;
using mx::Stream;
using mx::metal::Device;

// Number of key-blocks to split the reduction across. Mirrors MLX's own
// sdpa_vector_2pass heuristic: more blocks only when there are enough
// simdgroups per kv-head (n_simds = gqa_factor * qL) to justify the extra
// partials.
int kq_sdpa_blocks(int N, int n_simds, Device& d) {
  char devc = d.get_architecture().back();
  int blocks;
  if (devc == 's') {
    blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) {
        blocks = 128;
      } else if (N <= 32768) {
        blocks = 256;
      } else if (N <= 65536) {
        blocks = 512;
      } else {
        blocks = 1024;
      }
    }
  } else if (devc == 'd') {
    blocks = 128;
    if (n_simds <= 2 && N > 8192) {
      blocks = 256;
    } else if (n_simds >= 6) {
      if (N >= 16384 && N < 65536) {
        blocks = 512;
      } else if (N >= 65536) {
        blocks = 1024;
      }
    }
  } else {
    blocks = (n_simds >= 4) ? 64 : 32;
  }
  return blocks;
}

} // namespace

void KQuantSDPA::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // q row-contiguous [B, Hq, qL, D]; k/v [B, Hkv, kL, D], D contiguous.
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];

  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int qL = q.shape(2);
  int D = q.shape(3);
  int n_kv_heads = k.shape(1);
  int kL = k.shape(2);
  int gqa_factor = n_q_heads / n_kv_heads;
  int n_simds = gqa_factor * qL;
  int blocks = kq_sdpa_blocks(kL, n_simds, d);

  size_t k_head_stride =
      static_cast<size_t>(k.shape(1) == 1 ? k.strides(0) : k.strides(1));
  size_t k_seq_stride = static_cast<size_t>(k.strides(2));
  size_t v_head_stride =
      static_cast<size_t>(v.shape(1) == 1 ? v.strides(0) : v.strides(1));
  size_t v_seq_stride = static_cast<size_t>(v.strides(2));
  float scale = scale_;

  // Per-block partials + running max/sum, reduced by pass 2.
  mx::Shape part_shape = {B, n_q_heads, qL, blocks, D};
  mx::Shape red_shape = {B, n_q_heads, qL, blocks};
  array partials(part_shape, q.dtype(), nullptr, {});
  array sums(red_shape, mx::float32, nullptr, {});
  array maxs(red_shape, mx::float32, nullptr, {});
  partials.set_data(mx::allocator::malloc(partials.nbytes()));
  sums.set_data(mx::allocator::malloc(sums.nbytes()));
  maxs.set_data(mx::allocator::malloc(maxs.nbytes()));

  auto& ce = mx::metal::get_command_encoder(s);
  ce.add_temporary(partials);
  ce.add_temporary(sums);
  ce.add_temporary(maxs);

  std::string ts = kq_type_string(q.dtype());
  bool causal = causal_;
  mx::metal::MTLFCList fc = {
      {&causal, MTL::DataType::DataTypeBool, 0},
      {&blocks, MTL::DataType::DataTypeInt, 1},
  };

  // Pass 1: each (kv-head, batch, block) threadgroup computes a partial output.
  {
    std::string kname =
        "kq_sdpa_vector_2pass_1_" + ts + "_" + std::to_string(D);
    std::string hash =
        kname + (causal ? "_c1" : "_c0") + "_b" + std::to_string(blocks);
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(q, 0);
    ce.set_input_array(k, 1);
    ce.set_input_array(v, 2);
    ce.set_output_array(partials, 3);
    ce.set_output_array(sums, 4);
    ce.set_output_array(maxs, 5);
    ce.set_bytes(kL, 6);
    ce.set_bytes(k_head_stride, 7);
    ce.set_bytes(k_seq_stride, 8);
    ce.set_bytes(v_head_stride, 9);
    ce.set_bytes(v_seq_stride, 10);
    ce.set_bytes(scale, 11);
    MTL::Size group_dims(32, gqa_factor, qL);
    MTL::Size grid_dims(n_kv_heads, B, blocks);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 2: reduce per-block partials into the final output.
  {
    std::string kname =
        "kq_sdpa_vector_2pass_2_" + ts + "_" + std::to_string(D);
    std::string hash = kname + "_b" + std::to_string(blocks);
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(partials, 0);
    ce.set_input_array(sums, 1);
    ce.set_input_array(maxs, 2);
    ce.set_output_array(out, 3);
    MTL::Size group_dims(1024, 1, 1);
    MTL::Size grid_dims(B * n_q_heads, qL, 1);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }
}

#else // !_METAL_

void KQuantSDPA::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.sdpa_vector] requires a Metal build.");
}

#endif

void KQuantSDPA::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_vector] has no CPU implementation.");
}

std::vector<mx::Shape> KQuantSDPA::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQuantSDPA::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantSDPA&>(other);
  return scale_ == o.scale_ && causal_ == o.causal_;
}

mx::array sdpa_vector(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    bool causal,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] q, k, v must be 4-D [B, heads, L, D].");
  }
  int D = q.shape(-1);
  if (v.shape(-1) != D) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] value head dim must equal query head dim.");
  }
  if (D != 256 && D != 512) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] only head_dim 256 or 512 is supported.");
  }
  auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] q must be float16 or bfloat16.");
  }
  if (k.dtype() != dt || v.dtype() != dt) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] q, k, v must share a dtype.");
  }
  int n_q_heads = q.shape(1);
  int n_kv_heads = k.shape(1);
  if (n_kv_heads == 0 || n_q_heads % n_kv_heads != 0) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] n_q_heads must be a multiple of n_kv_heads.");
  }
  int qL = q.shape(2);
  int gqa_factor = n_q_heads / n_kv_heads;
  // pass-1 threadgroup is 32 * gqa_factor * qL threads; cap at the Metal max.
  if (32 * gqa_factor * qL > 1024) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] gqa_factor * qL exceeds the 32-wide limit.");
  }
  if (qL > k.shape(2)) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] query length exceeds key length.");
  }

  // q small -> contiguize if needed (cheap). k/v: only the last (head) dim must
  // be contiguous; head/seq strides are read in place, so a strided KV-cache
  // prefix is passed through without a copy.
  auto q_c = q.flags().row_contiguous ? q : mx::contiguous(q, false, s);
  auto k_c = k.strides().back() == 1 ? k : mx::contiguous(k, false, s);
  auto v_c = v.strides().back() == 1 ? v : mx::contiguous(v, false, s);

  auto out_shape = q_c.shape();
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantSDPA>(s, scale, causal),
      {std::move(q_c), std::move(k_c), std::move(v_c)});
}

} // namespace mlx_kquant
