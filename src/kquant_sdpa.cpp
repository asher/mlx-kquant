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
    // Register-heavy pipeline: some GPUs cap it below the dispatch width, and
    // Metal turns an oversized dispatch into silent garbage, not an error.
    const size_t tg = size_t(32) * gqa_factor * qL;
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_vector] threadgroup of " + std::to_string(tg) +
          " threads exceeds this GPU's pipeline limit (" +
          std::to_string(kernel->maxTotalThreadsPerThreadgroup()) + ").");
    }
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

void KQuantSDPAGQA::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  const bool sinks = inputs.size() == 4;

  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int qL = q.shape(2);
  int D = q.shape(3);
  int n_kv_heads = k.shape(1);
  int kL = k.shape(2);
  int gqa_factor = n_q_heads / n_kv_heads;
  // Auto splits: coarse buckets (a per-kL value would mint a new pipeline
  // specialization every decode step). Measured on M5 Max: more splits win as
  // depth grows; ~512-1024 keys per chunk is the sweet spot.
  int splits = splits_;
  if (splits == 0) {
    splits = kL <= 8192 ? 16 : kL <= 24576 ? 32 : kL <= 49152 ? 64 : 128;
  }

  size_t k_head_stride =
      static_cast<size_t>(k.shape(1) == 1 ? k.strides(0) : k.strides(1));
  size_t k_seq_stride = static_cast<size_t>(k.strides(2));
  size_t v_head_stride =
      static_cast<size_t>(v.shape(1) == 1 ? v.strides(0) : v.strides(1));
  size_t v_seq_stride = static_cast<size_t>(v.strides(2));
  float scale = scale_;

  // Coarse per-split partials (float32) + running max/sum, merged by pass 2.
  mx::Shape part_shape = {B, n_q_heads, qL, splits, D};
  mx::Shape red_shape = {B, n_q_heads, qL, splits};
  array partials(part_shape, mx::float32, nullptr, {});
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
  bool has_sinks = sinks;
  mx::metal::MTLFCList fc = {
      {&splits, MTL::DataType::DataTypeInt, 2},
      {&has_sinks, MTL::DataType::DataTypeBool, 3},
  };

  // Pass 1: one threadgroup per (kv-head, batch, split); the whole GQA group
  // (and, at verify width, every query pair -- the threadgroup z axis) shares
  // each staged K/V tile. qL > 1 dispatches the _p2 (two queries per
  // simdgroup) instantiation.
  {
    std::string kname = "kq_sdpa_gqa_2pass_1_" + ts + "_" + std::to_string(D) +
        "_c" + std::to_string(tile_c_) + (qL > 1 ? "_p2" : "");
    std::string hash = kname + "_s" + std::to_string(splits);
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    // Register-heavy pipeline: some GPUs cap it below the dispatch width, and
    // Metal turns an oversized dispatch into silent garbage, not an error.
    const size_t tg =
        size_t(32) * gqa_factor * (qL > 1 ? size_t((qL + 1) / 2) : 1);
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_decode_gqa] threadgroup of " + std::to_string(tg) +
          " threads exceeds this GPU's pipeline limit (" +
          std::to_string(kernel->maxTotalThreadsPerThreadgroup()) + ").");
    }
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
    ce.set_bytes(qL, 12);
    MTL::Size group_dims(32, gqa_factor, qL > 1 ? (qL + 1) / 2 : 1);
    MTL::Size grid_dims(n_kv_heads, B, splits);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 2: merge the per-split partials; sinks fold into the denominator.
  // Grid z is the query axis.
  {
    std::string kname = "kq_sdpa_gqa_2pass_2_" + ts + "_" + std::to_string(D);
    std::string hash =
        kname + "_s" + std::to_string(splits) + (has_sinks ? "_k1" : "_k0");
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(partials, 0);
    ce.set_input_array(sums, 1);
    ce.set_input_array(maxs, 2);
    // Metal wants every buffer bound; without sinks, rebind sums as a dummy
    // (the read is compiled out via the function constant).
    ce.set_input_array(sinks ? inputs[3] : sums, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(n_q_heads, 5);
    MTL::Size group_dims(32, 1, 1);
    MTL::Size grid_dims(n_q_heads, B, qL);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }
}

#else // !_METAL_

void KQuantSDPA::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.sdpa_vector] requires a Metal build.");
}

void KQuantSDPAGQA::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_decode_gqa] requires a Metal build.");
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

void KQuantSDPAGQA::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_decode_gqa] has no CPU implementation.");
}

std::vector<mx::Shape> KQuantSDPAGQA::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQuantSDPAGQA::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantSDPAGQA&>(other);
  return scale_ == o.scale_ && splits_ == o.splits_ && tile_c_ == o.tile_c_;
}

mx::array sdpa_decode_gqa(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    const std::optional<mx::array>& sinks,
    int splits,
    int tile_c,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] q, k, v must be 4-D [B, heads, L, D].");
  }
  int D = q.shape(-1);
  if ((D != 64 && D != 128 && D != 256 && D != 512) || v.shape(-1) != D ||
      k.shape(-1) != D) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] only head_dim 64/128/256/512 is "
        "supported.");
  }
  int qL = q.shape(2);
  if (qL < 1 || qL > 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] query length must be 1 (decode) "
        "to 4 (speculative-verify width).");
  }
  auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] q must be float16 or bfloat16.");
  }
  if (k.dtype() != dt || v.dtype() != dt) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] q, k, v must share a dtype.");
  }
  int n_q_heads = q.shape(1);
  int n_kv_heads = k.shape(1);
  if (n_kv_heads == 0 || n_q_heads % n_kv_heads != 0) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] n_q_heads must be a multiple of "
        "n_kv_heads.");
  }
  int gqa_factor = n_q_heads / n_kv_heads;
  if (gqa_factor > 16) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] gqa_factor must be <= 16.");
  }
  // Pass-1 threadgroup is 32 * gqa_factor * ceil(qL / 2) threads (Metal max
  // 1024).
  if (gqa_factor * ((qL + 1) / 2) > 32) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] gqa_factor * ceil(query length / 2) "
        "must be <= 32 (1024-thread threadgroup).");
  }
  if (splits < 0 || splits > 128) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] splits must be in [0, 128].");
  }
  if (tile_c == 0) {
    tile_c = D <= 128 ? 32 : D == 256 ? 16 : 8;
  }
  // Instantiated (D, C) pairs: threadgroup K+V tiles cap at 16 KB so two
  // threadgroups co-reside per core (D=64/128: C 32/16; 256: 16/8; 512: 8).
  const bool tile_ok = (D <= 128 && (tile_c == 32 || tile_c == 16)) ||
      (D == 256 && (tile_c == 16 || tile_c == 8)) || (D == 512 && tile_c == 8);
  if (!tile_ok) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] tile_c not instantiated for this "
        "head_dim (0 picks the default).");
  }
  if (k.shape(2) < qL) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] key length must be >= query length.");
  }

  auto q_c = q.flags().row_contiguous ? q : mx::contiguous(q, false, s);
  auto k_c = k.strides().back() == 1 ? k : mx::contiguous(k, false, s);
  auto v_c = v.strides().back() == 1 ? v : mx::contiguous(v, false, s);

  std::vector<mx::array> inputs = {
      std::move(q_c), std::move(k_c), std::move(v_c)};
  if (sinks.has_value()) {
    auto sk = *sinks;
    if (sk.size() != static_cast<size_t>(n_q_heads)) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_decode_gqa] sinks must have n_q_heads elements.");
    }
    sk = mx::astype(mx::reshape(sk, {n_q_heads}, s), mx::float32, s);
    inputs.push_back(mx::contiguous(sk, false, s));
  }

  auto out_shape = q.shape();
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantSDPAGQA>(s, scale, splits, tile_c),
      std::move(inputs));
}

} // namespace mlx_kquant
