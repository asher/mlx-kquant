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

// Eval-time layout guard. The op builders wrap q in mx::contiguous (layout
// flags are undefined on unevaluated inputs, so build-time checks can miss),
// and k/v are read in place via strides but need a contiguous head_dim; by
// the time eval_gpu runs the flags are real, so anything non-conforming here
// is a bug upstream -- fail loudly instead of reading garbage.
void kq_sdpa_check_layout(
    const char* op,
    const array& q,
    const array& k,
    const array& v) {
  if (!q.flags().row_contiguous) {
    throw std::runtime_error(
        std::string("[mlx_kquant.") + op + "] q must be row-contiguous.");
  }
  if (k.strides().back() != 1 || v.strides().back() != 1) {
    throw std::runtime_error(
        std::string("[mlx_kquant.") + op +
        "] k/v head_dim must be contiguous.");
  }
}

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
  kq_sdpa_check_layout("sdpa_vector", q, k, v);

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
  const bool write_lse = return_lse_;
  if (write_lse) {
    outputs[1].set_data(mx::allocator::malloc(outputs[1].nbytes()));
  }

  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  const bool sinks = has_sinks_;
  const bool starts = has_starts_;
  const bool kv_q8 = has_kv_q8_;
  const bool paged = paged_;
  const size_t starts_idx = 3 + (sinks ? 1 : 0);
  const size_t qkv_idx = starts_idx + (starts ? 1 : 0);
  kq_sdpa_check_layout("sdpa_decode_gqa", q, k, v);

  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int qL = q.shape(2);
  int D = q.shape(3);
  int n_kv_heads = k.shape(1);
  int kL = k.shape(2);
  int gqa_factor = n_q_heads / n_kv_heads;
  // Auto splits: coarse buckets (a per-kL value would mint a new pipeline
  // specialization every decode step). Measured on M5 Max: more splits win as
  // depth grows; ~512-1024 keys per chunk is the sweet spot. The paged walk
  // rides the pages operand at the end of the input list and buckets on the
  // SELECTED key count, not the cache depth.
  int n_pages = 0;
  if (paged) {
    n_pages = static_cast<int>(inputs.back().shape(2));
  }
  int splits = splits_;
  if (splits == 0) {
    const int span = paged ? n_pages * tile_c_ : kL;
    splits = span <= 8192 ? 16 : span <= 24576 ? 32 : span <= 49152 ? 64 : 128;
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
  bool has_starts = starts;
  bool has_kv_q8 = kv_q8;
  bool has_lse = write_lse;
  bool has_cascade = false;
  bool has_paged = paged;
  mx::metal::MTLFCList fc = {
      {&splits, MTL::DataType::DataTypeInt, 2},
      {&has_sinks, MTL::DataType::DataTypeBool, 3},
      {&has_starts, MTL::DataType::DataTypeBool, 4},
      {&has_kv_q8, MTL::DataType::DataTypeBool, 5},
      {&has_lse, MTL::DataType::DataTypeBool, 6},
      {&has_cascade, MTL::DataType::DataTypeBool, 7},
      {&has_paged, MTL::DataType::DataTypeBool, 8},
  };

  // Pass 1: one threadgroup per (kv-head, batch, split); the whole GQA group
  // (and, at verify width, every query pair -- the threadgroup z axis) shares
  // each staged K/V tile. qL > 1 dispatches the _p2 (two queries per
  // simdgroup) instantiation.
  {
    std::string kname = "kq_sdpa_gqa_2pass_1_" + ts + "_" + std::to_string(D) +
        "_c" + std::to_string(tile_c_) + (qL > 1 ? "_p2" : "");
    std::string hash = kname + "_s" + std::to_string(splits) +
        (has_starts ? "_st1" : "_st0") + (has_kv_q8 ? "_q8" : "") +
        (paged ? "_pg" : "");
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
    // Metal wants every buffer bound; without starts, rebind sums as a dummy
    // (the read is compiled out via the function constant).
    ce.set_input_array(starts ? inputs[starts_idx] : sums, 13);
    // Quantized-KV scale/bias operands (dummies when compiled out). Scale
    // strides are shared per K/V side (scales and biases are congruent).
    size_t ks_head_stride = 0, ks_seq_stride = 0;
    size_t vs_head_stride = 0, vs_seq_stride = 0;
    if (kv_q8) {
      const auto& ksc = inputs[qkv_idx];
      const auto& vsc = inputs[qkv_idx + 2];
      ks_head_stride = static_cast<size_t>(
          ksc.shape(1) == 1 ? ksc.strides(0) : ksc.strides(1));
      ks_seq_stride = static_cast<size_t>(ksc.strides(2));
      vs_head_stride = static_cast<size_t>(
          vsc.shape(1) == 1 ? vsc.strides(0) : vsc.strides(1));
      vs_seq_stride = static_cast<size_t>(vsc.strides(2));
    }
    for (int i = 0; i < 4; i++) {
      ce.set_input_array(kv_q8 ? inputs[qkv_idx + i] : sums, 14 + i);
    }
    ce.set_bytes(ks_head_stride, 18);
    ce.set_bytes(ks_seq_stride, 19);
    ce.set_bytes(vs_head_stride, 20);
    ce.set_bytes(vs_seq_stride, 21);
    // Page list (dummy when the paged walk is compiled out).
    ce.set_input_array(paged ? inputs.back() : sums, 22);
    ce.set_bytes(n_pages, 23);
    MTL::Size group_dims(32, gqa_factor, qL > 1 ? (qL + 1) / 2 : 1);
    MTL::Size grid_dims(n_kv_heads, B, splits);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 2: merge the per-split partials; sinks fold into the denominator.
  // Grid z is the query axis.
  {
    std::string kname = "kq_sdpa_gqa_2pass_2_" + ts + "_" + std::to_string(D);
    std::string hash = kname + "_s" + std::to_string(splits) +
        (has_sinks ? "_k1" : "_k0") + (write_lse ? "_lse" : "");
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
    if (write_lse) {
      ce.set_output_array(outputs[1], 6);
    } else {
      ce.set_input_array(sums, 6);
    }
    // Cascade compiled out: dummy second-set bindings.
    const int czero = 0;
    for (int i = 7; i <= 9; i++) {
      ce.set_input_array(sums, i);
    }
    ce.set_bytes(czero, 10);
    ce.set_bytes(czero, 11);
    MTL::Size group_dims(32, 1, 1);
    MTL::Size grid_dims(n_q_heads, B, qL);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }
}

void KQuantSDPAFAVerify::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  const bool write_lse = return_lse_;
  if (write_lse) {
    outputs[1].set_data(mx::allocator::malloc(outputs[1].nbytes()));
  }

  // q is the GQA-folded query tile [B, Hkv, n_rows, D], row-contiguous;
  // k/v [B, Hkv, kL, D] are read in place via their head/seq strides.
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  kq_sdpa_check_layout("sdpa_fa_verify", q, k, v);

  int B = q.shape(0);
  int n_kv_heads = k.shape(1);
  int n_rows = q.shape(2);
  int D = q.shape(3);
  int kL = k.shape(2);
  int q_len = q_len_;
  // Same coarse split buckets as sdpa_decode_gqa (a per-kL value would mint
  // a new pipeline specialization every decode step).
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

  // Coarse per-split partials (float32) + running max/sum; the folded row
  // axis flattens identically to the unfolded [B, Hq, qL] layout, so the
  // kq_sdpa_gqa merge pass is reused unchanged.
  mx::Shape part_shape = {B, n_kv_heads, n_rows, splits, D};
  mx::Shape red_shape = {B, n_kv_heads, n_rows, splits};
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
  bool has_sinks = false;
  bool has_lse = write_lse;
  bool has_cascade = false;
  bool no_q8 = false;
  mx::metal::MTLFCList fc = {
      {&splits, MTL::DataType::DataTypeInt, 2},
      {&has_sinks, MTL::DataType::DataTypeBool, 3},
      {&no_q8, MTL::DataType::DataTypeBool, 5},
      {&has_lse, MTL::DataType::DataTypeBool, 6},
      {&has_cascade, MTL::DataType::DataTypeBool, 7},
  };

  // Pass 1: one threadgroup per (kv-head, batch, split) streams its key
  // chunk through the simdgroup-matrix tile. head_dim 256 runs a simdgroup
  // per 8 tile rows ((BQ/8)*32 threads); 512 runs the 256-thread d-split
  // variant (BQ 32 only, capped at the op).
  {
    const int bq = n_rows <= 32 ? 32 : 64;
    std::string kname = "kq_sdpa_fa_verify_2pass_1_" + ts + "_" +
        std::to_string(D) + "_bq" + std::to_string(bq);
    std::string hash = kname + "_s" + std::to_string(splits);
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    // Register-heavy pipeline: some GPUs cap it below the dispatch width, and
    // Metal turns an oversized dispatch into silent garbage, not an error.
    const size_t tg = D == 512 ? 256 : (bq / 8) * 32;
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_fa_verify] threadgroup of " + std::to_string(tg) +
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
    ce.set_bytes(q_len, 12);
    ce.set_bytes(n_rows, 13);
    // q8 operand slots: compiled out (fc 5 false); bind dummies.
    size_t zero = 0;
    for (int i = 0; i < 4; i++) {
      ce.set_input_array(sums, 14 + i);
    }
    ce.set_bytes(zero, 18);
    ce.set_bytes(zero, 19);
    ce.set_bytes(zero, 20);
    ce.set_bytes(zero, 21);
    MTL::Size group_dims(32, tg / 32, 1);
    MTL::Size grid_dims(n_kv_heads, B, splits);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 2: the shared kq_sdpa_gqa merge; grid z is the folded row axis.
  {
    std::string kname = "kq_sdpa_gqa_2pass_2_" + ts + "_" + std::to_string(D);
    std::string hash = kname + "_s" + std::to_string(splits) + "_k0" +
        (write_lse ? "_lse" : "");
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(partials, 0);
    ce.set_input_array(sums, 1);
    ce.set_input_array(maxs, 2);
    // No sinks: rebind sums as a dummy (the read is compiled out via the
    // function constant).
    ce.set_input_array(sums, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(n_kv_heads, 5);
    if (write_lse) {
      ce.set_output_array(outputs[1], 6);
    } else {
      ce.set_input_array(sums, 6);
    }
    // Cascade compiled out: dummy second-set bindings.
    const int czero = 0;
    for (int i = 7; i <= 9; i++) {
      ce.set_input_array(sums, i);
    }
    ce.set_bytes(czero, 10);
    ce.set_bytes(czero, 11);
    MTL::Size group_dims(32, 1, 1);
    MTL::Size grid_dims(n_kv_heads, B, n_rows);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }
}

void KQuantSDPACascade::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  const bool write_lse = return_lse_;
  if (write_lse) {
    outputs[1].set_data(mx::allocator::malloc(outputs[1].nbytes()));
  }

  const auto& q = inputs[0];
  const auto& qf = inputs[1];
  const auto& k_sh = inputs[2];
  const auto& v_sh = inputs[3];
  const auto& k_pr = inputs[4];
  const auto& v_pr = inputs[5];
  const bool starts = has_starts_;
  const bool kv_q8 = has_kv_q8_;
  // Optional trailing inputs: [starts], then the eight q8 scale/bias
  // arrays (shared k/v pairs first, then private).
  const int q8_base = 6 + int(starts);
  kq_sdpa_check_layout("sdpa_decode_gqa_cascade", q, k_pr, v_pr);
  kq_sdpa_check_layout("sdpa_decode_gqa_cascade", qf, k_sh, v_sh);

  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int D = q.shape(3);
  int n_kv_heads = k_sh.shape(1);
  int gqa_factor = n_q_heads / n_kv_heads;
  int P = k_sh.shape(2);
  int Sp = k_pr.shape(2);
  int qL = q.shape(2);
  int n_rows = B * gqa_factor * qL;
  float scale = scale_;

  int s_sh = splits_shared_;
  if (s_sh == 0) {
    s_sh = P <= 8192 ? 16 : P <= 24576 ? 32 : P <= 49152 ? 64 : 128;
  }
  int s_pr = splits_priv_;
  if (s_pr == 0) {
    s_pr = Sp <= 8192 ? 16 : Sp <= 24576 ? 32 : Sp <= 49152 ? 64 : 128;
  }

  std::string ts = kq_type_string(q.dtype());
  auto& ce = mx::metal::get_command_encoder(s);

  // Private-region partials (decode layout) + shared-region partials (fa
  // folded layout); one merge pass folds both.
  mx::Shape p1_shape = {B, n_q_heads, qL, s_pr, D};
  mx::Shape r1_shape = {B, n_q_heads, qL, s_pr};
  array partials1(p1_shape, mx::float32, nullptr, {});
  array sums1(r1_shape, mx::float32, nullptr, {});
  array maxs1(r1_shape, mx::float32, nullptr, {});
  mx::Shape p2_shape = {1, n_kv_heads, n_rows, s_sh, D};
  mx::Shape r2_shape = {1, n_kv_heads, n_rows, s_sh};
  array partials2(p2_shape, mx::float32, nullptr, {});
  array sums2(r2_shape, mx::float32, nullptr, {});
  array maxs2(r2_shape, mx::float32, nullptr, {});
  for (array* a : {&partials1, &sums1, &maxs1, &partials2, &sums2, &maxs2}) {
    a->set_data(mx::allocator::malloc(a->nbytes()));
    ce.add_temporary(*a);
  }

  // Pass 1a: private suffixes through the decode-gqa kernel (per-row grid,
  // starts honored).
  {
    bool f = false;
    bool has_starts = starts;
    bool q8 = kv_q8;
    mx::metal::MTLFCList fc = {
        {&s_pr, MTL::DataType::DataTypeInt, 2},
        {&f, MTL::DataType::DataTypeBool, 3},
        {&has_starts, MTL::DataType::DataTypeBool, 4},
        {&q8, MTL::DataType::DataTypeBool, 5},
        {&f, MTL::DataType::DataTypeBool, 8},
    };
    std::string kname = "kq_sdpa_gqa_2pass_1_" + ts + "_" + std::to_string(D) +
        "_c" + std::to_string(tile_c_) + (qL > 1 ? "_p2" : "");
    std::string hash = kname + "_s" + std::to_string(s_pr) +
        (has_starts ? "_st1" : "_st0") + (q8 ? "_q8" : "") + "_casc";
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    const size_t tg =
        size_t(32) * gqa_factor * (qL > 1 ? size_t((qL + 1) / 2) : 1);
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_decode_gqa_cascade] threadgroup of " +
          std::to_string(tg) + " threads exceeds this GPU's pipeline limit.");
    }
    size_t k_head_stride = static_cast<size_t>(
        k_pr.shape(1) == 1 ? k_pr.strides(0) : k_pr.strides(1));
    size_t k_seq_stride = static_cast<size_t>(k_pr.strides(2));
    size_t v_head_stride = static_cast<size_t>(
        v_pr.shape(1) == 1 ? v_pr.strides(0) : v_pr.strides(1));
    size_t v_seq_stride = static_cast<size_t>(v_pr.strides(2));
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(q, 0);
    ce.set_input_array(k_pr, 1);
    ce.set_input_array(v_pr, 2);
    ce.set_output_array(partials1, 3);
    ce.set_output_array(sums1, 4);
    ce.set_output_array(maxs1, 5);
    ce.set_bytes(Sp, 6);
    ce.set_bytes(k_head_stride, 7);
    ce.set_bytes(k_seq_stride, 8);
    ce.set_bytes(v_head_stride, 9);
    ce.set_bytes(v_seq_stride, 10);
    ce.set_bytes(scale, 11);
    ce.set_bytes(qL, 12);
    ce.set_input_array(starts ? inputs[6] : sums1, 13);
    size_t zero = 0;
    if (kv_q8) {
      const auto& ksc = inputs[q8_base + 4];
      const auto& kbi = inputs[q8_base + 5];
      const auto& vsc = inputs[q8_base + 6];
      const auto& vbi = inputs[q8_base + 7];
      ce.set_input_array(ksc, 14);
      ce.set_input_array(kbi, 15);
      ce.set_input_array(vsc, 16);
      ce.set_input_array(vbi, 17);
      size_t ks_head = static_cast<size_t>(
          ksc.shape(1) == 1 ? ksc.strides(0) : ksc.strides(1));
      size_t ks_seq = static_cast<size_t>(ksc.strides(2));
      size_t vs_head = static_cast<size_t>(
          vsc.shape(1) == 1 ? vsc.strides(0) : vsc.strides(1));
      size_t vs_seq = static_cast<size_t>(vsc.strides(2));
      ce.set_bytes(ks_head, 18);
      ce.set_bytes(ks_seq, 19);
      ce.set_bytes(vs_head, 20);
      ce.set_bytes(vs_seq, 21);
    } else {
      for (int i = 0; i < 4; i++) {
        ce.set_input_array(sums1, 14 + i);
      }
      ce.set_bytes(zero, 18);
      ce.set_bytes(zero, 19);
      ce.set_bytes(zero, 20);
      ce.set_bytes(zero, 21);
    }
    const int pzero = 0;
    ce.set_input_array(sums1, 22);
    ce.set_bytes(pzero, 23);
    MTL::Size group_dims(32, gqa_factor, qL > 1 ? (qL + 1) / 2 : 1);
    MTL::Size grid_dims(n_kv_heads, B, s_pr);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 1b: the shared prefix through the fa row-tile kernel -- one KV walk
  // serves all B*gqa folded rows.
  {
    bool f = false;
    bool q8 = kv_q8;
    mx::metal::MTLFCList fc = {
        {&s_sh, MTL::DataType::DataTypeInt, 2},
        {&f, MTL::DataType::DataTypeBool, 3},
        {&q8, MTL::DataType::DataTypeBool, 5},
    };
    const int bq = n_rows <= 32 ? 32 : 64;
    std::string kname = "kq_sdpa_fa_verify_2pass_1_" + ts + "_" +
        std::to_string(D) + "_bq" + std::to_string(bq);
    std::string hash =
        kname + "_s" + std::to_string(s_sh) + (q8 ? "_q8" : "") + "_casc";
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    const size_t tg = D == 512 ? 256 : (bq / 8) * 32;
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_decode_gqa_cascade] threadgroup of " +
          std::to_string(tg) + " threads exceeds this GPU's pipeline limit.");
    }
    size_t k_head_stride = static_cast<size_t>(
        k_sh.shape(1) == 1 ? k_sh.strides(0) : k_sh.strides(1));
    size_t k_seq_stride = static_cast<size_t>(k_sh.strides(2));
    size_t v_head_stride = static_cast<size_t>(
        v_sh.shape(1) == 1 ? v_sh.strides(0) : v_sh.strides(1));
    size_t v_seq_stride = static_cast<size_t>(v_sh.strides(2));
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(qf, 0);
    ce.set_input_array(k_sh, 1);
    ce.set_input_array(v_sh, 2);
    ce.set_output_array(partials2, 3);
    ce.set_output_array(sums2, 4);
    ce.set_output_array(maxs2, 5);
    ce.set_bytes(P, 6);
    ce.set_bytes(k_head_stride, 7);
    ce.set_bytes(k_seq_stride, 8);
    ce.set_bytes(v_head_stride, 9);
    ce.set_bytes(v_seq_stride, 10);
    ce.set_bytes(scale, 11);
    const int q_len_shared = 1; // unclamped: verify rows see the whole prefix
    ce.set_bytes(q_len_shared, 12);
    ce.set_bytes(n_rows, 13);
    size_t zero = 0;
    if (kv_q8) {
      const auto& ksc = inputs[q8_base + 0];
      const auto& kbi = inputs[q8_base + 1];
      const auto& vsc = inputs[q8_base + 2];
      const auto& vbi = inputs[q8_base + 3];
      ce.set_input_array(ksc, 14);
      ce.set_input_array(kbi, 15);
      ce.set_input_array(vsc, 16);
      ce.set_input_array(vbi, 17);
      size_t ks_head = static_cast<size_t>(
          ksc.shape(1) == 1 ? ksc.strides(0) : ksc.strides(1));
      size_t ks_seq = static_cast<size_t>(ksc.strides(2));
      size_t vs_head = static_cast<size_t>(
          vsc.shape(1) == 1 ? vsc.strides(0) : vsc.strides(1));
      size_t vs_seq = static_cast<size_t>(vsc.strides(2));
      ce.set_bytes(ks_head, 18);
      ce.set_bytes(ks_seq, 19);
      ce.set_bytes(vs_head, 20);
      ce.set_bytes(vs_seq, 21);
    } else {
      for (int i = 0; i < 4; i++) {
        ce.set_input_array(sums2, 14 + i);
      }
      ce.set_bytes(zero, 18);
      ce.set_bytes(zero, 19);
      ce.set_bytes(zero, 20);
      ce.set_bytes(zero, 21);
    }
    MTL::Size group_dims(32, tg / 32, 1);
    MTL::Size grid_dims(n_kv_heads, 1, s_sh);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 2: one merge over both partial sets (the LSE merge).
  {
    bool f = false;
    bool t = true;
    bool has_lse = write_lse;
    mx::metal::MTLFCList fc = {
        {&s_pr, MTL::DataType::DataTypeInt, 2},
        {&f, MTL::DataType::DataTypeBool, 3},
        {&has_lse, MTL::DataType::DataTypeBool, 6},
        {&t, MTL::DataType::DataTypeBool, 7},
    };
    std::string kname = "kq_sdpa_gqa_2pass_2_" + ts + "_" + std::to_string(D);
    std::string hash = kname + "_s" + std::to_string(s_pr) + "_k0_casc" +
        std::to_string(s_sh) + (write_lse ? "_lse" : "");
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(partials1, 0);
    ce.set_input_array(sums1, 1);
    ce.set_input_array(maxs1, 2);
    ce.set_input_array(sums1, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(n_q_heads, 5);
    if (write_lse) {
      ce.set_output_array(outputs[1], 6);
    } else {
      ce.set_input_array(sums1, 6);
    }
    ce.set_input_array(partials2, 7);
    ce.set_input_array(sums2, 8);
    ce.set_input_array(maxs2, 9);
    ce.set_bytes(s_sh, 10);
    ce.set_bytes(gqa_factor, 11);
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

void KQuantSDPAFAVerify::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_fa_verify] requires a Metal build.");
}

void KQuantSDPACascade::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_decode_gqa_cascade] requires a Metal build.");
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
  // Unconditional: layout flags are undefined on unevaluated inputs (a
  // lazy strided view can read as row-contiguous here), and Contiguous
  // decides at eval time, sharing the buffer when q is already packed.
  auto q_c = mx::contiguous(q, false, s);
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
  const auto& qs = inputs[0].shape();
  if (return_lse_) {
    return {qs, {qs[0], qs[1], qs[2]}};
  }
  return {qs};
}

bool KQuantSDPAGQA::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantSDPAGQA&>(other);
  return scale_ == o.scale_ && splits_ == o.splits_ && tile_c_ == o.tile_c_ &&
      has_sinks_ == o.has_sinks_ && has_starts_ == o.has_starts_ &&
      has_kv_q8_ == o.has_kv_q8_ && return_lse_ == o.return_lse_ &&
      paged_ == o.paged_;
}

static std::vector<mx::array> sdpa_decode_gqa_impl(
    bool return_lse,
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    const std::optional<mx::array>& sinks,
    int splits,
    int tile_c,
    const std::optional<mx::array>& starts,
    const std::optional<mx::array>& k_scales,
    const std::optional<mx::array>& k_biases,
    const std::optional<mx::array>& v_scales,
    const std::optional<mx::array>& v_biases,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  const int n_qkv = int(k_scales.has_value()) + int(k_biases.has_value()) +
      int(v_scales.has_value()) + int(v_biases.has_value());
  const bool kv_q8 = n_qkv == 4;
  if (n_qkv != 0 && n_qkv != 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] quantized KV needs all four of "
        "k_scales/k_biases/v_scales/v_biases.");
  }

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] q, k, v must be 4-D [B, heads, L, D].");
  }
  int D = q.shape(-1);
  if ((D != 64 && D != 128 && D != 256 && D != 512) ||
      (!kv_q8 && (v.shape(-1) != D || k.shape(-1) != D))) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] only head_dim 64/128/256/512 is "
        "supported.");
  }
  if (kv_q8) {
    // mlx affine wire, bits 8 / group 64: packed uint32 words, one
    // scale/bias per 64-element group ([B, Hkv, S, D/64], q's dtype).
    if (k.dtype() != mx::uint32 || v.dtype() != mx::uint32 ||
        k.shape(-1) != D / 4 || v.shape(-1) != D / 4) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_decode_gqa] quantized k/v must be uint32 wire "
          "with last dim head_dim / 4 (bits 8).");
    }
    for (const auto& a : {*k_scales, *k_biases, *v_scales, *v_biases}) {
      if (a.dtype() != q.dtype() || a.ndim() != 4 || a.shape(-1) != D / 64 ||
          a.shape(2) != k.shape(2)) {
        throw std::invalid_argument(
            "[mlx_kquant.sdpa_decode_gqa] quantized KV scales/biases must "
            "be [B, n_kv_heads, S, head_dim / 64] in q's dtype (group 64).");
      }
    }
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
  if (!kv_q8 && (k.dtype() != dt || v.dtype() != dt)) {
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

  // Unconditional: layout flags are undefined on unevaluated inputs (a
  // lazy strided view can read as row-contiguous here), and Contiguous
  // decides at eval time, sharing the buffer when q is already packed.
  auto q_c = mx::contiguous(q, false, s);
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
  if (starts.has_value()) {
    auto st = *starts;
    if (st.size() != static_cast<size_t>(q.shape(0))) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_decode_gqa] starts must have one element per "
          "batch row.");
    }
    if (st.dtype() != mx::int32) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_decode_gqa] starts must be int32.");
    }
    st = mx::reshape(st, {q.shape(0)}, s);
    inputs.push_back(mx::contiguous(st, false, s));
  }
  if (kv_q8) {
    for (const auto& a : {*k_scales, *k_biases, *v_scales, *v_biases}) {
      inputs.push_back(
          a.strides().back() == 1 ? a : mx::contiguous(a, false, s));
    }
  }

  auto prim = std::make_shared<KQuantSDPAGQA>(
      s,
      scale,
      splits,
      tile_c,
      sinks.has_value(),
      starts.has_value(),
      kv_q8,
      return_lse);
  auto out_shape = q.shape();
  if (return_lse) {
    mx::Shape lse_shape = {q.shape(0), q.shape(1), q.shape(2)};
    return mx::array::make_arrays(
        {std::move(out_shape), std::move(lse_shape)},
        {dt, mx::float32},
        std::move(prim),
        std::move(inputs));
  }
  return {
      mx::array(std::move(out_shape), dt, std::move(prim), std::move(inputs))};
}

mx::array sdpa_decode_gqa(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    const std::optional<mx::array>& sinks,
    int splits,
    int tile_c,
    const std::optional<mx::array>& starts,
    const std::optional<mx::array>& k_scales,
    const std::optional<mx::array>& k_biases,
    const std::optional<mx::array>& v_scales,
    const std::optional<mx::array>& v_biases,
    mx::StreamOrDevice s_) {
  return sdpa_decode_gqa_impl(
      false,
      std::move(q),
      std::move(k),
      std::move(v),
      scale,
      sinks,
      splits,
      tile_c,
      starts,
      k_scales,
      k_biases,
      v_scales,
      v_biases,
      s_)[0];
}

mx::array sdpa_decode_gqa_paged(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    mx::array pages,
    int splits,
    const std::optional<mx::array>& starts,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.sdpa_decode_gqa_paged] ";
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument(std::string(op) + "q/k/v must be 4-D.");
  }
  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int qL = q.shape(2);
  int D = q.shape(3);
  int n_kv_heads = k.shape(1);
  if (qL != 1) {
    throw std::invalid_argument(
        std::string(op) + "query length must be 1 (decode).");
  }
  if (D != 64 && D != 128 && D != 256 && D != 512) {
    throw std::invalid_argument(
        std::string(op) + "only head_dim 64/128/256/512 is supported.");
  }
  auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + "q must be float16 or bfloat16.");
  }
  if (k.dtype() != dt || v.dtype() != dt) {
    throw std::invalid_argument(
        std::string(op) + "q, k, v must share a dtype (no quantized KV).");
  }
  if (n_kv_heads == 0 || n_q_heads % n_kv_heads != 0) {
    throw std::invalid_argument(
        std::string(op) + "n_q_heads must be a multiple of n_kv_heads.");
  }
  int gqa_factor = n_q_heads / n_kv_heads;
  if (gqa_factor > 16) {
    throw std::invalid_argument(std::string(op) + "gqa_factor must be <= 16.");
  }
  if (splits < 0 || splits > 128) {
    throw std::invalid_argument(
        std::string(op) + "splits must be in [0, 128].");
  }
  // Page unit is the head dim's staged tile height.
  const int tile_c = D <= 128 ? 32 : D == 256 ? 16 : 8;
  if (pages.dtype() != mx::int32 || pages.ndim() != 3 || pages.shape(0) != B ||
      pages.shape(1) != n_kv_heads || pages.shape(2) < 1) {
    throw std::invalid_argument(
        std::string(op) +
        "pages must be int32 [B, n_kv_heads, n_pages] with n_pages >= 1 "
        "(page indices into the key axis; page size = " +
        std::to_string(tile_c) + " rows at this head_dim).");
  }

  auto q_c = mx::contiguous(q, false, s);
  auto k_c = k.strides().back() == 1 ? k : mx::contiguous(k, false, s);
  auto v_c = v.strides().back() == 1 ? v : mx::contiguous(v, false, s);
  auto p_c = mx::contiguous(pages, false, s);

  auto prim = std::make_shared<KQuantSDPAGQA>(
      s,
      scale,
      splits,
      tile_c,
      /*has_sinks=*/false,
      /*has_starts=*/starts.has_value(),
      /*has_kv_q8=*/false,
      /*return_lse=*/false,
      /*paged=*/true);
  auto out_shape = q.shape();
  std::vector<mx::array> inputs = {
      std::move(q_c), std::move(k_c), std::move(v_c)};
  if (starts.has_value()) {
    auto st = *starts;
    if (st.size() != static_cast<size_t>(B)) {
      throw std::invalid_argument(
          std::string(op) + "starts must have one element per batch row.");
    }
    if (st.dtype() != mx::int32) {
      throw std::invalid_argument(std::string(op) + "starts must be int32.");
    }
    st = mx::reshape(st, {B}, s);
    inputs.push_back(mx::contiguous(st, false, s));
  }
  inputs.push_back(std::move(p_c));
  return mx::array(
      std::move(out_shape), dt, std::move(prim), std::move(inputs));
}

std::vector<mx::array> sdpa_decode_gqa_lse(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    const std::optional<mx::array>& sinks,
    int splits,
    int tile_c,
    const std::optional<mx::array>& starts,
    const std::optional<mx::array>& k_scales,
    const std::optional<mx::array>& k_biases,
    const std::optional<mx::array>& v_scales,
    const std::optional<mx::array>& v_biases,
    mx::StreamOrDevice s_) {
  return sdpa_decode_gqa_impl(
      true,
      std::move(q),
      std::move(k),
      std::move(v),
      scale,
      sinks,
      splits,
      tile_c,
      starts,
      k_scales,
      k_biases,
      v_scales,
      v_biases,
      s_);
}

void KQuantSDPAFAVerify::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_fa_verify] has no CPU implementation.");
}

void KQuantSDPACascade::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_decode_gqa_cascade] has no CPU implementation.");
}

std::vector<mx::Shape> KQuantSDPACascade::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& qs = inputs[0].shape();
  if (return_lse_) {
    return {qs, {qs[0], qs[1], qs[2]}};
  }
  return {qs};
}

bool KQuantSDPACascade::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantSDPACascade&>(other);
  return scale_ == o.scale_ && splits_shared_ == o.splits_shared_ &&
      splits_priv_ == o.splits_priv_ && tile_c_ == o.tile_c_ &&
      has_starts_ == o.has_starts_ && return_lse_ == o.return_lse_ &&
      has_kv_q8_ == o.has_kv_q8_;
}

std::vector<mx::array> sdpa_decode_gqa_cascade(
    mx::array q,
    mx::array k_shared,
    mx::array v_shared,
    mx::array k_priv,
    mx::array v_priv,
    float scale,
    const std::optional<mx::array>& starts,
    int splits_shared,
    int splits_priv,
    int tile_c,
    bool return_lse,
    const std::optional<mx::array>& k_shared_scales,
    const std::optional<mx::array>& k_shared_biases,
    const std::optional<mx::array>& v_shared_scales,
    const std::optional<mx::array>& v_shared_biases,
    const std::optional<mx::array>& k_priv_scales,
    const std::optional<mx::array>& k_priv_biases,
    const std::optional<mx::array>& v_priv_scales,
    const std::optional<mx::array>& v_priv_biases,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  const char* op = "[mlx_kquant.sdpa_decode_gqa_cascade] ";

  const int n_q8 = int(k_shared_scales.has_value()) +
      int(k_shared_biases.has_value()) + int(v_shared_scales.has_value()) +
      int(v_shared_biases.has_value()) + int(k_priv_scales.has_value()) +
      int(k_priv_biases.has_value()) + int(v_priv_scales.has_value()) +
      int(v_priv_biases.has_value());
  const bool kv_q8 = n_q8 == 8;
  if (n_q8 != 0 && n_q8 != 8) {
    throw std::invalid_argument(
        std::string(op) +
        "quantized KV needs all eight scale/bias arrays (shared and "
        "private, k and v).");
  }

  if (q.ndim() != 4 || k_shared.ndim() != 4 || v_shared.ndim() != 4 ||
      k_priv.ndim() != 4 || v_priv.ndim() != 4) {
    throw std::invalid_argument(
        std::string(op) + "q, k, v must be 4-D [B, heads, L, D].");
  }
  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int D = q.shape(3);
  int n_kv_heads = k_shared.shape(1);
  auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        std::string(op) + "q must be float16 or bfloat16.");
  }
  if (kv_q8) {
    if (D == 512) {
      throw std::invalid_argument(
          std::string(op) + "quantized KV is not supported at head_dim 512.");
    }
    // mlx affine wire, bits 8 / group 64: packed uint32 words, one
    // scale/bias per 64-element group in q's dtype.
    for (const auto& a : {k_shared, v_shared, k_priv, v_priv}) {
      if (a.dtype() != mx::uint32 || a.shape(3) != D / 4) {
        throw std::invalid_argument(
            std::string(op) +
            "quantized k/v must be uint32 wire with last "
            "dim head_dim / 4 (bits 8).");
      }
    }
    const mx::array* sb[8] = {
        &*k_shared_scales,
        &*k_shared_biases,
        &*v_shared_scales,
        &*v_shared_biases,
        &*k_priv_scales,
        &*k_priv_biases,
        &*v_priv_scales,
        &*v_priv_biases};
    for (int i = 0; i < 8; i++) {
      const auto& ref = i < 4 ? k_shared : k_priv;
      if (sb[i]->dtype() != dt || sb[i]->ndim() != 4 ||
          sb[i]->shape(0) != ref.shape(0) || sb[i]->shape(1) != n_kv_heads ||
          sb[i]->shape(2) != ref.shape(2) || sb[i]->shape(3) != D / 64) {
        throw std::invalid_argument(
            std::string(op) +
            "quantized KV scales/biases must be "
            "[B, n_kv_heads, S, head_dim / 64] in q's dtype (group 64), "
            "matching their region.");
      }
    }
  } else {
    for (const auto& a : {k_shared, v_shared, k_priv, v_priv}) {
      if (a.dtype() != dt || a.shape(3) != D) {
        throw std::invalid_argument(
            std::string(op) + "k/v must share q's dtype and head_dim.");
      }
    }
  }
  if (D != 64 && D != 128 && D != 256 && D != 512) {
    throw std::invalid_argument(
        std::string(op) + "head_dim must be 64, 128, 256 or 512.");
  }
  int qL = q.shape(2);
  if (qL < 1 || qL > 8) {
    throw std::invalid_argument(std::string(op) + "q_len must be in [1, 8].");
  }
  if (k_shared.shape(0) != 1 || v_shared.shape(0) != 1 ||
      v_shared.shape(1) != n_kv_heads ||
      v_shared.shape(2) != k_shared.shape(2)) {
    throw std::invalid_argument(
        std::string(op) +
        "shared k/v must be [1, n_kv_heads, P, D] with matching P.");
  }
  if (k_priv.shape(0) != B || v_priv.shape(0) != B ||
      k_priv.shape(1) != n_kv_heads || v_priv.shape(1) != n_kv_heads ||
      v_priv.shape(2) != k_priv.shape(2)) {
    throw std::invalid_argument(
        std::string(op) +
        "private k/v must be [B, n_kv_heads, Sp, D] with matching Sp.");
  }
  if (k_priv.shape(2) < 1 || k_shared.shape(2) < 1) {
    throw std::invalid_argument(
        std::string(op) + "both key regions must be non-empty.");
  }
  if (n_q_heads % n_kv_heads != 0) {
    throw std::invalid_argument(
        std::string(op) + "q heads must be a multiple of kv heads.");
  }
  int gqa_factor = n_q_heads / n_kv_heads;
  if (gqa_factor > 16) {
    throw std::invalid_argument(std::string(op) + "gqa factor must be <= 16.");
  }
  int n_rows = B * gqa_factor * qL;
  int max_rows = D == 512 ? 32 : 64;
  if (n_rows > max_rows) {
    throw std::invalid_argument(
        std::string(op) + "B * gqa * q_len must be <= " +
        std::to_string(max_rows) + " at head_dim " + std::to_string(D) + ".");
  }
  if (qL > 1 && gqa_factor * ((qL + 1) / 2) > 32) {
    throw std::invalid_argument(
        std::string(op) + "gqa * ceil(q_len/2) must be <= 32.");
  }
  const int tile_default = D <= 128 ? 32 : D == 256 ? 16 : 8;
  if (tile_c == 0) {
    tile_c = tile_default;
  }
  const bool tile_ok = (D <= 128 && (tile_c == 32 || tile_c == 16)) ||
      (D == 256 && (tile_c == 16 || tile_c == 8)) || (D == 512 && tile_c == 8);
  if (!tile_ok) {
    throw std::invalid_argument(
        std::string(op) + "tile_c not instantiated for this head_dim.");
  }
  if (qL > 1 && tile_c != tile_default) {
    // the verify-width (_p2) private kernel exists at the default tile only
    throw std::invalid_argument(
        std::string(op) + "q_len > 1 requires the default tile_c.");
  }
  if (splits_shared < 0 || splits_shared > 128 || splits_priv < 0 ||
      splits_priv > 128) {
    throw std::invalid_argument(
        std::string(op) + "splits must be in [0, 128].");
  }

  auto q_c = mx::contiguous(q, false, s);
  // kv-head-major fold for the shared row-tile pass, query axis innermost:
  // row = (b*gqa + g)*qL + t.
  auto q_folded = mx::contiguous(
      mx::reshape(
          mx::transpose(
              mx::reshape(q_c, {B, n_kv_heads, gqa_factor, qL, D}, s),
              {1, 0, 2, 3, 4},
              s),
          {1, n_kv_heads, n_rows, D},
          s),
      false,
      s);
  auto contig_kv = [&](const mx::array& a) {
    return a.strides().back() == 1 ? a : mx::contiguous(a, false, s);
  };

  std::vector<mx::array> inputs = {
      std::move(q_c),
      std::move(q_folded),
      contig_kv(k_shared),
      contig_kv(v_shared),
      contig_kv(k_priv),
      contig_kv(v_priv)};
  if (starts.has_value()) {
    auto st = *starts;
    if (st.size() != static_cast<size_t>(B)) {
      throw std::invalid_argument(
          std::string(op) + "starts must have one element per batch row.");
    }
    if (st.dtype() != mx::int32) {
      throw std::invalid_argument(std::string(op) + "starts must be int32.");
    }
    st = mx::reshape(st, {B}, s);
    inputs.push_back(mx::contiguous(st, false, s));
  }
  if (kv_q8) {
    for (const mx::array* a :
         {&*k_shared_scales,
          &*k_shared_biases,
          &*v_shared_scales,
          &*v_shared_biases,
          &*k_priv_scales,
          &*k_priv_biases,
          &*v_priv_scales,
          &*v_priv_biases}) {
      inputs.push_back(contig_kv(*a));
    }
  }

  auto prim = std::make_shared<KQuantSDPACascade>(
      s,
      scale,
      splits_shared,
      splits_priv,
      tile_c,
      starts.has_value(),
      return_lse,
      kv_q8);
  auto out_shape = q.shape();
  if (return_lse) {
    mx::Shape lse_shape = {B, n_q_heads, qL};
    return mx::array::make_arrays(
        {std::move(out_shape), std::move(lse_shape)},
        {dt, mx::float32},
        std::move(prim),
        std::move(inputs));
  }
  return {
      mx::array(std::move(out_shape), dt, std::move(prim), std::move(inputs))};
}

std::vector<mx::Shape> KQuantSDPAFAVerify::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& qs = inputs[0].shape();
  if (return_lse_) {
    return {qs, {qs[0], qs[1], qs[2]}};
  }
  return {qs};
}

bool KQuantSDPAFAVerify::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantSDPAFAVerify&>(other);
  return scale_ == o.scale_ && q_len_ == o.q_len_ && splits_ == o.splits_ &&
      return_lse_ == o.return_lse_;
}

static std::vector<mx::array> sdpa_fa_verify_impl(
    bool return_lse,
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    int q_len,
    int splits,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] q, k, v must be 4-D [B, heads, L, D].");
  }
  int D = q.shape(-1);
  if ((D != 64 && D != 128 && D != 256 && D != 512) || k.shape(-1) != D ||
      v.shape(-1) != D) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] only head_dim 64, 128, 256 or 512 is "
        "supported.");
  }
  auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] q must be float16 or bfloat16.");
  }
  if (k.dtype() != dt || v.dtype() != dt) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] q, k, v must share a dtype.");
  }
  if (q.shape(0) != 1 || k.shape(0) != 1 || v.shape(0) != 1) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] batch size must be 1.");
  }
  if (q.shape(1) != k.shape(1) || v.shape(1) != k.shape(1)) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] q must be GQA-folded: q heads must "
        "equal kv heads.");
  }
  if (v.shape(2) != k.shape(2)) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] k and v must share a key length.");
  }
  if (q_len < 1 || q_len > 8) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] q_len must be in [1, 8].");
  }
  int n_rows = q.shape(2);
  // The 64-row tile exists for the register-resident head dims (64-256);
  // the 512 d-split kernel is fixed at the 32-row tile.
  int max_rows = D == 512 ? 32 : 64;
  if (n_rows < q_len || n_rows > max_rows || n_rows % q_len != 0) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] folded rows must be a multiple of q_len "
        "and <= " +
        std::to_string(max_rows) + " at head_dim " + std::to_string(D) + ".");
  }
  if (k.shape(2) < q_len) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] key length must be >= q_len.");
  }
  if (splits < 0 || splits > 128) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] splits must be in [0, 128].");
  }

  // Unconditional: layout flags are undefined on unevaluated inputs (a
  // lazy strided view can read as row-contiguous here), and Contiguous
  // decides at eval time, sharing the buffer when q is already packed.
  auto q_c = mx::contiguous(q, false, s);
  auto k_c = k.strides().back() == 1 ? k : mx::contiguous(k, false, s);
  auto v_c = v.strides().back() == 1 ? v : mx::contiguous(v, false, s);

  auto prim =
      std::make_shared<KQuantSDPAFAVerify>(s, scale, q_len, splits, return_lse);
  auto out_shape = q_c.shape();
  std::vector<mx::array> inputs = {
      std::move(q_c), std::move(k_c), std::move(v_c)};
  if (return_lse) {
    mx::Shape lse_shape = {out_shape[0], out_shape[1], out_shape[2]};
    return mx::array::make_arrays(
        {std::move(out_shape), std::move(lse_shape)},
        {dt, mx::float32},
        std::move(prim),
        std::move(inputs));
  }
  return {
      mx::array(std::move(out_shape), dt, std::move(prim), std::move(inputs))};
}

mx::array sdpa_fa_verify(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    int q_len,
    int splits,
    mx::StreamOrDevice s_) {
  return sdpa_fa_verify_impl(
      false,
      std::move(q),
      std::move(k),
      std::move(v),
      scale,
      q_len,
      splits,
      s_)[0];
}

std::vector<mx::array> sdpa_fa_verify_lse(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    int q_len,
    int splits,
    mx::StreamOrDevice s_) {
  return sdpa_fa_verify_impl(
      true, std::move(q), std::move(k), std::move(v), scale, q_len, splits, s_);
}

} // namespace mlx_kquant
