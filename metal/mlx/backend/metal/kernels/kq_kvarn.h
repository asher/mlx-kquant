// KVarN KV-cache quantizer kernels (BeeLlama variant).
//
// One record covers a 128-token group of one kv-head slice (128 dims):
// 16-iteration log-domain Sinkhorn variance normalization with
// best-imbalance selection, per-row asymmetric RTN, LSB-first row
// bitstream packing, and three fp16 axis vectors (scale, zp, other).
// K tiles are [row=dim][col=token], V tiles are [row=token][col=dim].
//
// Numerics pin the fp32 GPU path of beellama.cpp (ggml-cuda/kvarn.cu):
// fp32 accumulators with sequential per-thread accumulation order,
// metal::round (half away from zero) for codes, half() (nearest even)
// for the axis vectors. tests/kvarn_ref.py is the oracle.
//
// The Sinkhorn tile (128x128 fp32 = 64 KB) exceeds Apple threadgroup
// memory, so only the per-axis stats live in threadgroup storage and the
// fp16 tile is re-read from device per std pass, like the CUDA lowshmem
// route. The source is fp16 either way, so re-reading loses nothing.

#pragma once

#include "mlx/backend/metal/kernels/kq_kvarn_params.h"

using namespace metal;

template <typename T>
inline float kq_kvarn_tile_at(const device T* xt, int r, int c, int vside) {
  return vside ? float(xt[r * KQ_KVARN_GROUP + c])
               : float(xt[c * KQ_KVARN_GROUP + r]);
}

// Unbiased sample std of one tile row (row_pass) or column, normalized by
// the current scales. Sequential accumulation order matches the oracle.
template <typename T>
inline float kq_kvarn_std(
    const device T* xt,
    int vside,
    threadgroup const float* s_col,
    threadgroup const float* s_row,
    int fixed,
    bool row_pass) {
  float sum = 0.0f;
  float sum_sq = 0.0f;
  for (int i = 0; i < KQ_KVARN_GROUP; ++i) {
    const int r = row_pass ? fixed : i;
    const int c = row_pass ? i : fixed;
    const float v = kq_kvarn_tile_at(xt, r, c, vside) / (s_col[c] * s_row[r]);
    sum += v;
    sum_sq += v * v;
  }
  const float mean = sum / 128.0f;
  return precise::sqrt(max((sum_sq - 128.0f * mean * mean) / 127.0f, 0.0f));
}

// Max and min of a 128-wide threadgroup array (4 simdgroups), broadcast to
// every thread. Exact regardless of reduction order.
inline void kq_kvarn_minmax128(
    threadgroup const float* vals,
    threadgroup float* scratch,
    uint lid,
    uint simd_gid,
    uint simd_lid,
    thread float& lo,
    thread float& hi) {
  const float v = vals[lid];
  const float smax = simd_max(v);
  const float smin = simd_min(v);
  if (simd_lid == 0) {
    scratch[simd_gid] = smax;
    scratch[4 + simd_gid] = smin;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  hi = max(max(scratch[0], scratch[1]), max(scratch[2], scratch[3]));
  lo = min(min(scratch[4], scratch[5]), min(scratch[6], scratch[7]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One threadgroup of 128 threads per record: thread i owns column stat i,
// row stat i, and RTN row i. X is the rotated fp16 stage, tile-contiguous
// [n_tiles, 128 tokens, 128 dims]; CODES/AXES are record-contiguous.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(128)]] void kq_kvarn_quantize(
    const device T* X [[buffer(0)]],
    device uint32_t* CODES [[buffer(1)]],
    device half* AXES [[buffer(2)]],
    const constant int& bits [[buffer(3)]],
    const constant int& vside [[buffer(4)]],
    const constant int& iters [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float log_s_col[KQ_KVARN_GROUP];
  threadgroup float log_s_row[KQ_KVARN_GROUP];
  threadgroup float s_col[KQ_KVARN_GROUP];
  threadgroup float s_row[KQ_KVARN_GROUP];
  threadgroup float best_col[KQ_KVARN_GROUP];
  threadgroup float best_row[KQ_KVARN_GROUP];
  threadgroup float col_std[KQ_KVARN_GROUP];
  threadgroup float row_std[KQ_KVARN_GROUP];
  threadgroup float scratch[8];

  const device T* xt = X + size_t(tgid) * KQ_KVARN_GROUP * KQ_KVARN_GROUP;
  const int i = int(lid);

  log_s_col[i] = 0.0f;
  log_s_row[i] = 0.0f;
  s_col[i] = 1.0f;
  s_row[i] = 1.0f;
  best_col[i] = 1.0f;
  best_row[i] = 1.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  col_std[i] = kq_kvarn_std(xt, vside, s_col, s_row, i, false);
  row_std[i] = kq_kvarn_std(xt, vside, s_col, s_row, i, true);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float cmin, cmax, rmin, rmax;
  kq_kvarn_minmax128(col_std, scratch, lid, simd_gid, simd_lid, cmin, cmax);
  kq_kvarn_minmax128(row_std, scratch, lid, simd_gid, simd_lid, rmin, rmax);
  float imb_best = cmax / max(cmin, 1e-8f) + rmax / max(rmin, 1e-8f);

  for (int iter = 0; iter < iters; ++iter) {
    const float sc = clamp(col_std[i], 1e-3f, 1e3f);
    log_s_col[i] = clamp(log_s_col[i] + precise::log(sc), -0.3f, 10.0f);
    s_col[i] = precise::exp(log_s_col[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    row_std[i] = kq_kvarn_std(xt, vside, s_col, s_row, i, true);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sr = clamp(row_std[i], 1e-3f, 1e3f);
    log_s_row[i] = clamp(log_s_row[i] + precise::log(sr), -0.3f, 10.0f);
    s_row[i] = precise::exp(log_s_row[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    col_std[i] = kq_kvarn_std(xt, vside, s_col, s_row, i, false);
    row_std[i] = kq_kvarn_std(xt, vside, s_col, s_row, i, true);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    kq_kvarn_minmax128(col_std, scratch, lid, simd_gid, simd_lid, cmin, cmax);
    kq_kvarn_minmax128(row_std, scratch, lid, simd_gid, simd_lid, rmin, rmax);
    const float imb = cmax / max(cmin, 1e-8f) + rmax / max(rmin, 1e-8f);
    if (imb <= imb_best) {
      imb_best = imb;
      best_col[i] = s_col[i];
      best_row[i] = s_row[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Per-row asymmetric RTN on the best-imbalance normalization, packed as
  // an LSB-first bitstream (row r occupies words [r*4*bits, (r+1)*4*bits)).
  const int qmax = (1 << bits) - 1;
  const float brow = best_row[i];
  float lo = INFINITY;
  float hi = -INFINITY;
  for (int c = 0; c < KQ_KVARN_GROUP; ++c) {
    const float v = kq_kvarn_tile_at(xt, i, c, vside) / (best_col[c] * brow);
    lo = min(lo, v);
    hi = max(hi, v);
  }
  const float scale = max((hi - lo) / float(qmax), 1e-10f);

  device uint32_t* row_out =
      CODES + size_t(tgid) * 4 * bits * KQ_KVARN_GROUP + size_t(i) * 4 * bits;
  ulong acc = 0;
  int nbits = 0;
  int w = 0;
  for (int c = 0; c < KQ_KVARN_GROUP; ++c) {
    const float v = kq_kvarn_tile_at(xt, i, c, vside) / (best_col[c] * brow);
    const float q = clamp(metal::round((v - lo) / scale), 0.0f, float(qmax));
    acc |= ulong(uint(q)) << nbits;
    nbits += bits;
    if (nbits >= 32) {
      row_out[w++] = uint32_t(acc);
      acc >>= 32;
      nbits -= 32;
    }
  }

  device half* axes_out = AXES + size_t(tgid) * 3 * KQ_KVARN_GROUP;
  axes_out[i] = half(brow * scale);
  axes_out[KQ_KVARN_GROUP + i] = half(brow * lo);
  axes_out[2 * KQ_KVARN_GROUP + i] = half(best_col[i]);
}

// Dequantize records back to the rotated fp16/bf16 group [token, dim].
// One threadgroup of 128 threads per record; thread t owns token t.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(128)]] void kq_kvarn_dequant(
    const device uint32_t* CODES [[buffer(0)]],
    const device half* AXES [[buffer(1)]],
    device T* X [[buffer(2)]],
    const constant int& bits [[buffer(3)]],
    const constant int& vside [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  const device uint32_t* rec = CODES + size_t(tgid) * 4 * bits * KQ_KVARN_GROUP;
  const device half* axes = AXES + size_t(tgid) * 3 * KQ_KVARN_GROUP;
  device T* out = X + size_t(tgid) * KQ_KVARN_GROUP * KQ_KVARN_GROUP +
      size_t(lid) * KQ_KVARN_GROUP;

  const int t = int(lid);
  const uint mask = (1u << bits) - 1u;
  for (int d = 0; d < KQ_KVARN_GROUP; ++d) {
    const int row = vside ? t : d;
    const int col = vside ? d : t;
    const int bit_offset = (row * KQ_KVARN_GROUP + col) * bits;
    const int word = bit_offset >> 5;
    const int sh = bit_offset & 31;
    uint v = rec[word] >> sh;
    if (sh + bits > 32) {
      v |= rec[word + 1] << (32 - sh);
    }
    const float q = float(v & mask);
    const float scale = float(axes[row]);
    const float zp = float(axes[KQ_KVARN_GROUP + row]);
    const float other = float(axes[2 * KQ_KVARN_GROUP + col]);
    out[d] = T((q * scale + zp) * other);
  }
}
