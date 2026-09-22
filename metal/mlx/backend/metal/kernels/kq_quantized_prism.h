// PQ2_0 and PTQ1_0 (PrismML/llama.cpp, ggml types 142 and 143): 128-weight
// blocks with one fp16 scale, ported from the fork's Metal kernels
// (ggml-metal/kernels/dequantize.h and mul_mv.metal, MIT; see licenses/).
//
// PQ2_0, 34 bytes [fp16 d][32 x u8 qs]: element j is bits (j%4)*2 of byte
// j/4, code - 1 in {-1, 0, +1, +2}. The block stride is 2 mod 4, so only
// half and byte loads are alignment-safe.
//
// PTQ1_0, 28 bytes [24 x u8 qs][2 x u8 qh][fp16 d]: five base-3 trits per
// qs byte, four per qh byte, trit - 1. Element e < 80 is trit e>>4 of
// qs[e&15]; 80 <= e < 120 is trit (e-80)>>3 of qs[16+((e-80)&7)]; e >= 120
// is trit (e-120)>>1 of qh[(e-120)&1]. Trit n of byte b is
//   floor(3^(n+1) u) - 3 floor(3^n u),  u = b / 256,
// exact in fp32 (3^5 * 255 < 2^24), which keeps the decode in the float
// pipe. The mat-vecs collapse sum_n t_n y_n over a byte into floor terms
// whose coefficients depend only on the activations, so each byte costs a
// few floors and fmas and no integer work.

MLX_MTL_CONST int KQ_PQ2_0_SUPERBLOCK = 128;
MLX_MTL_CONST int KQ_PQ2_0_BLOCK_BYTES = 34;
MLX_MTL_CONST int KQ_PQ2_0_QS_OFFSET = 2;

MLX_MTL_CONST int KQ_PTQ1_0_SUPERBLOCK = 128;
MLX_MTL_CONST int KQ_PTQ1_0_BLOCK_BYTES = 28;
MLX_MTL_CONST int KQ_PTQ1_0_QH_OFFSET = 24;
MLX_MTL_CONST int KQ_PTQ1_0_D_OFFSET = 26;

// ================================ PQ2_0 =================================

METAL_FUNC float4 kq_pq2_0_peel(uint8_t b) {
  return float4(
             float(b & 3),
             float((b >> 2) & 3),
             float((b >> 4) & 3),
             float(b >> 6)) -
      1.0f;
}

inline void kq_pq2_0_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const float d = float(*(const device half*)(block));
  const device uint8_t* qs = block + KQ_PQ2_0_QS_OFFSET + 4 * il;
#pragma unroll
  for (short k = 0; k < 4; ++k) {
    reg[k] = kq_pq2_0_peel(qs[k]) * d;
  }
}

template <typename T>
METAL_FUNC void kq_pq2_0_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_PQ2_0_SUPERBLOCK;
  const int j = gid - sb_id * KQ_PQ2_0_SUPERBLOCK;
  const device uint8_t* sb =
      w + static_cast<int64_t>(sb_id) * KQ_PQ2_0_BLOCK_BYTES;
  const float d = float(*(const device half*)(sb));
  const uint8_t b = sb[KQ_PQ2_0_QS_OFFSET + (j >> 2)];
  const short code = (b >> (2 * (j & 3))) & 3;
  out[gid] = T(d * float(code - 1));
}

struct KqPq2_0Ext {
  MLX_MTL_CONST int superblock = KQ_PQ2_0_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_PQ2_0_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_pq2_0_deq_chunk16(block, il, reg);
  }
};

// Half2 pair decode of one lane's 16 codes, held as one uint whose bits
// 2j..2j+1 are code j. Codes j and j+8 sit 16 bits apart, so one mask and
// one or put them into the mantissas of a half2 with value 1024 + 4^j c,
// and one half2 fma turns that into (c - 1) / 16 exactly: the 1/16 keeps
// the half accumulator away from overflow and the -1 is the codec offset.
// Pairs 5..7 read the word shifted down by 10 bits so their codes also
// land below the 1024 mantissa bit.
METAL_FUNC void kq_pq2_0_decode_pairs(uint wv, thread half2* h) {
#pragma unroll
  for (short j = 0; j < 8; ++j) {
    const short jj = j < 5 ? j : j - 5;
    const uint src = j < 5 ? wv : wv >> 10;
    const uint p = (src & (0x00030003u << (2 * jj))) | 0x64006400u;
    const half scale = half(1.0f / float(16 << (2 * jj)));
    h[j] =
        fma(as_type<half2>(p),
            half2(scale),
            half2(-1024.0h * scale - half(1.0f / 16.0f)));
  }
}

// M=1 mat-vec: eight lanes per block, four blocks per simdgroup pass, 16
// contiguous weights per lane decoded as half2 pairs. Two half MACs per
// instruction against the activation pair (y_j, y_j+8) keep the decode
// near the load-only rate.
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_pq2_0_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_PQ2_0_SUPERBLOCK, "PQ2_0 requires gs=128");
  static_assert(bits == 2, "PQ2_0 requires bits=2");
  constexpr int num_simdgroups = 2;
  constexpr int blocks_per_pass = 4;
  // fp32 activations keep an fp32 pair accumulator; half inputs stay in
  // the half pipe.
  typedef metal::conditional_t<metal::is_same_v<T, float>, float2, half2> A2;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int nb = in_vec_size / KQ_PQ2_0_SUPERBLOCK;
  const int row_bytes = nb * KQ_PQ2_0_BLOCK_BYTES;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const short ix = simd_lid >> 3;
  const short it = simd_lid & 7;
  U result[results_per_simdgroup] = {0};
  for (int ib = ix; ib < nb; ib += blocks_per_pass) {
    const device T* xb = x + ib * KQ_PQ2_0_SUPERBLOCK + it * 16;
    const vec<T, 8> lo = *(const device vec<T, 8>*)(xb);
    const vec<T, 8> hi = *(const device vec<T, 8>*)(xb + 8);
    A2 y2[8];
#pragma unroll
    for (short j = 0; j < 8; ++j) {
      y2[j] = A2(float(lo[j]), float(hi[j]));
    }
    // A static row loop lets the compiler interleave the rows' loads; the
    // tail threadgroup recomputes its last row and drops it at the store.
#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(min(out_row + row, out_vec_size - 1)) *
              row_bytes +
          ib * KQ_PQ2_0_BLOCK_BYTES;
      const U d16 = 16.0f * U(float(*(const device half*)(sb)));
      const device ushort* qs =
          (const device ushort*)(sb + KQ_PQ2_0_QS_OFFSET + 4 * it);
      half2 h[8];
      kq_pq2_0_decode_pairs(uint(qs[0]) | (uint(qs[1]) << 16), h);
      A2 acc = A2(0, 0);
#pragma unroll
      for (short j = 0; j < 8; ++j) {
        acc = fma(A2(h[j]), y2[j], acc);
      }
      result[row] += d16 * (float(acc.x) + float(acc.y));
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

// Verify-shaped mat-vec (M = vm in 2..KQ_PRISM_MAX_VM activation rows, see
// kq_q8_0_verify_qmv_impl): the same lane geometry as the M=1 kernel, each
// row's pairs decoded once per block and dotted against every activation
// row. Non-batched only; bit-identical to the M=1 kernel per row.
MLX_MTL_CONST int KQ_PRISM_MAX_VM = 8;

template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_pq2_0_verify_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& vm,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_PQ2_0_SUPERBLOCK, "PQ2_0 requires gs=128");
  static_assert(bits == 2, "PQ2_0 requires bits=2");
  constexpr int num_simdgroups = 2;
  constexpr int blocks_per_pass = 4;
  typedef metal::conditional_t<metal::is_same_v<T, float>, float2, half2> A2;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int nb = in_vec_size / KQ_PQ2_0_SUPERBLOCK;
  const int row_bytes = nb * KQ_PQ2_0_BLOCK_BYTES;
  const short ix = simd_lid >> 3;
  const short it = simd_lid & 7;
  U result[KQ_PRISM_MAX_VM][results_per_simdgroup] = {{0}};
  for (int ib = ix; ib < nb; ib += blocks_per_pass) {
    half2 h[results_per_simdgroup][8];
    U d16[results_per_simdgroup];
#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(min(out_row + row, out_vec_size - 1)) *
              row_bytes +
          ib * KQ_PQ2_0_BLOCK_BYTES;
      d16[row] = 16.0f * U(float(*(const device half*)(sb)));
      const device ushort* qs =
          (const device ushort*)(sb + KQ_PQ2_0_QS_OFFSET + 4 * it);
      kq_pq2_0_decode_pairs(uint(qs[0]) | (uint(qs[1]) << 16), h[row]);
    }
#pragma unroll
    for (int m = 0; m < KQ_PRISM_MAX_VM; m++) {
      if (m < vm) {
        const device T* xb =
            x + m * in_vec_size + ib * KQ_PQ2_0_SUPERBLOCK + it * 16;
        const vec<T, 8> lo = *(const device vec<T, 8>*)(xb);
        const vec<T, 8> hi = *(const device vec<T, 8>*)(xb + 8);
        A2 y2[8];
#pragma unroll
        for (short j = 0; j < 8; ++j) {
          y2[j] = A2(float(lo[j]), float(hi[j]));
        }
#pragma unroll
        for (int row = 0; row < results_per_simdgroup; row++) {
          A2 acc = A2(0, 0);
#pragma unroll
          for (short j = 0; j < 8; ++j) {
            acc = fma(A2(h[row][j]), y2[j], acc);
          }
          result[m][row] += d16[row] * (float(acc.x) + float(acc.y));
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < KQ_PRISM_MAX_VM; m++) {
    if (m < vm) {
      for (int row = 0; row < results_per_simdgroup; row++) {
        U r = simd_sum(result[m][row]);
        if (simd_lid == 0 && row < active_rows) {
          y[m * out_vec_size + out_row + row] = static_cast<T>(r);
        }
      }
    }
  }
}

// ================================ PTQ1_0 ================================

inline void kq_ptq1_0_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  // il 0..4 are the 16-byte chunk at trit il; il 5 and 6 are the 8-byte
  // chunk at trits 2(il-5) and 2(il-5)+1, each byte read once for both;
  // il 7 is that chunk at trit 4 followed by the qh tail.
  const float pow3f[6] = {1.0f, 3.0f, 9.0f, 27.0f, 81.0f, 243.0f};
  const float d = float(*(const device half*)(block + KQ_PTQ1_0_D_OFFSET));
  if (il < 5) {
    const device uint8_t* qs = block;
    const float c0 = pow3f[il];
    const float c1 = 3.0f * c0;
#pragma unroll
    for (short k = 0; k < 16; ++k) {
      const float u = float(qs[k]) * (1.0f / 256.0f);
      reg[k / 4][k % 4] = d * (floor(c1 * u) - 3.0f * floor(c0 * u) - 1.0f);
    }
  } else if (il < 7) {
    const device uint8_t* qs = block + 16;
    const float c0 = pow3f[2 * (il - 5)];
    const float c1 = 3.0f * c0;
    const float c2 = 3.0f * c1;
#pragma unroll
    for (short k = 0; k < 8; ++k) {
      const float u = float(qs[k]) * (1.0f / 256.0f);
      const float g0 = floor(c0 * u);
      const float g1 = floor(c1 * u);
      const float g2 = floor(c2 * u);
      reg[k / 4][k % 4] = d * (g1 - 3.0f * g0 - 1.0f);
      reg[(k + 8) / 4][(k + 8) % 4] = d * (g2 - 3.0f * g1 - 1.0f);
    }
  } else {
    const device uint8_t* qs = block + 16;
#pragma unroll
    for (short k = 0; k < 8; ++k) {
      const float u = float(qs[k]) * (1.0f / 256.0f);
      reg[k / 4][k % 4] =
          d * (floor(243.0f * u) - 3.0f * floor(81.0f * u) - 1.0f);
    }
    const device uint8_t* qh = block + KQ_PTQ1_0_QH_OFFSET;
#pragma unroll
    for (short k = 0; k < 8; ++k) {
      const float c0 = pow3f[k >> 1];
      const float u = float(qh[k & 1]) * (1.0f / 256.0f);
      reg[(k + 8) / 4][(k + 8) % 4] =
          d * (floor(3.0f * c0 * u) - 3.0f * floor(c0 * u) - 1.0f);
    }
  }
}

template <typename T>
METAL_FUNC void kq_ptq1_0_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const float pow3f[6] = {1.0f, 3.0f, 9.0f, 27.0f, 81.0f, 243.0f};
  const int sb_id = gid / KQ_PTQ1_0_SUPERBLOCK;
  const int e = gid - sb_id * KQ_PTQ1_0_SUPERBLOCK;
  const device uint8_t* sb =
      w + static_cast<int64_t>(sb_id) * KQ_PTQ1_0_BLOCK_BYTES;
  const float d = float(*(const device half*)(sb + KQ_PTQ1_0_D_OFFSET));
  uint8_t b;
  short n;
  if (e < 80) {
    b = sb[e & 15];
    n = e >> 4;
  } else if (e < 120) {
    const int t = e - 80;
    b = sb[16 + (t & 7)];
    n = t >> 3;
  } else {
    const int t = e - 120;
    b = sb[KQ_PTQ1_0_QH_OFFSET + (t & 1)];
    n = t >> 1;
  }
  const float u = float(b) * (1.0f / 256.0f);
  const float p = pow3f[n];
  out[gid] = T(d * (floor(3.0f * p * u) - 3.0f * floor(p * u) - 1.0f));
}

struct KqPtq1_0Ext {
  MLX_MTL_CONST int superblock = KQ_PTQ1_0_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_PTQ1_0_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_ptq1_0_deq_chunk16(block, il, reg);
  }
};

// M=1 mat-vec: eight lanes per block, four blocks per simdgroup pass. Lane
// it owns whole bytes, so the block's 26 payload bytes are read once:
//   qs[2it], qs[2it+1]  -> elements 16n + m for n in 0..4
//   qs[16 + it]         -> elements 80 + 8n + it for n in 0..4
//   qh[it & 1]          -> element 120 + it, trit it >> 1
// With g_k = floor(3^k u), trit n is g_{n+1} - 3 g_n and g_0 = 0, so
//   sum_n t_n y_n = sum_{k=1..4} g_k (y_{k-1} - 3 y_k) + g_5 y_4;
// the lane stages those coefficients once per block and reuses them for
// every row. The -1 offset is one subtraction of sum y per block.
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_ptq1_0_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_PTQ1_0_SUPERBLOCK, "PTQ1_0 requires gs=128");
  static_assert(bits == 1, "PTQ1_0 requires bits=1");
  constexpr int num_simdgroups = 2;
  constexpr int blocks_per_pass = 4;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int nb = in_vec_size / KQ_PTQ1_0_SUPERBLOCK;
  const int row_bytes = nb * KQ_PTQ1_0_BLOCK_BYTES;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const short ix = simd_lid >> 3;
  const short it = simd_lid & 7;
  const float pow3f[4] = {1.0f, 3.0f, 9.0f, 27.0f};
  const U ph = pow3f[it >> 1];
  U result[results_per_simdgroup] = {0};
  for (int ib = ix; ib < nb; ib += blocks_per_pass) {
    const device T* xb = x + ib * KQ_PTQ1_0_SUPERBLOCK;
    // c[0..4], c[5..9]: the two 16-byte-chunk bytes; c[10..14]: the 8-byte
    // chunk byte; c[15]: the qh activation.
    U c[16];
    U sumy = 0;
#pragma unroll
    for (short k = 0; k < 2; ++k) {
      const short m = 2 * it + k;
      U yv[5];
#pragma unroll
      for (short n = 0; n < 5; ++n) {
        yv[n] = U(xb[16 * n + m]);
        sumy += yv[n];
      }
#pragma unroll
      for (short n = 0; n < 4; ++n) {
        c[5 * k + n] = yv[n] - 3.0f * yv[n + 1];
      }
      c[5 * k + 4] = yv[4];
    }
    {
      U yv[5];
#pragma unroll
      for (short n = 0; n < 5; ++n) {
        yv[n] = U(xb[80 + 8 * n + it]);
        sumy += yv[n];
      }
#pragma unroll
      for (short n = 0; n < 4; ++n) {
        c[10 + n] = yv[n] - 3.0f * yv[n + 1];
      }
      c[14] = yv[4];
    }
    c[15] = U(xb[120 + it]);
    sumy += c[15];
#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(min(out_row + row, out_vec_size - 1)) *
              row_bytes +
          ib * KQ_PTQ1_0_BLOCK_BYTES;
      const U d = U(float(*(const device half*)(sb + KQ_PTQ1_0_D_OFFSET)));
      U acc = 0;
#pragma unroll
      for (short k = 0; k < 3; ++k) {
        const U u = U(sb[k < 2 ? 2 * it + k : 16 + it]) * (1.0f / 256.0f);
        const short cb = 5 * k;
        acc += floor(3.0f * u) * c[cb + 0];
        acc += floor(9.0f * u) * c[cb + 1];
        acc += floor(27.0f * u) * c[cb + 2];
        acc += floor(81.0f * u) * c[cb + 3];
        acc += floor(243.0f * u) * c[cb + 4];
      }
      {
        const U u = U(sb[KQ_PTQ1_0_QH_OFFSET + (it & 1)]) * (1.0f / 256.0f);
        acc += (floor(3.0f * ph * u) - 3.0f * floor(ph * u)) * c[15];
      }
      result[row] += d * (acc - sumy);
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

// Verify-shaped mat-vec (M = vm in 2..KQ_PRISM_MAX_VM): the M=1 lane
// geometry, but the lane decodes its 16 trits to values once per row and
// block (the trit chain is the codec's cost, so it is paid once, not per
// activation row) and dots them against every activation row.
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_ptq1_0_verify_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& vm,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_PTQ1_0_SUPERBLOCK, "PTQ1_0 requires gs=128");
  static_assert(bits == 1, "PTQ1_0 requires bits=1");
  constexpr int num_simdgroups = 2;
  constexpr int blocks_per_pass = 4;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int nb = in_vec_size / KQ_PTQ1_0_SUPERBLOCK;
  const int row_bytes = nb * KQ_PTQ1_0_BLOCK_BYTES;
  const short ix = simd_lid >> 3;
  const short it = simd_lid & 7;
  const float pow3f[4] = {1.0f, 3.0f, 9.0f, 27.0f};
  const U ph = pow3f[it >> 1];
  U result[KQ_PRISM_MAX_VM][results_per_simdgroup] = {{0}};
  for (int ib = ix; ib < nb; ib += blocks_per_pass) {
    // t[row][0..4], [5..9]: bytes qs[2it], qs[2it+1] (elements 16n + m);
    // [10..14]: qs[16 + it] (elements 80 + 8n + it); [15]: the qh trit
    // (element 120 + it). Values are trit - 1.
    U t[results_per_simdgroup][16];
    U d[results_per_simdgroup];
#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(min(out_row + row, out_vec_size - 1)) *
              row_bytes +
          ib * KQ_PTQ1_0_BLOCK_BYTES;
      d[row] = U(float(*(const device half*)(sb + KQ_PTQ1_0_D_OFFSET)));
#pragma unroll
      for (short k = 0; k < 3; ++k) {
        const U u = U(sb[k < 2 ? 2 * it + k : 16 + it]) * (1.0f / 256.0f);
        const float p3[5] = {3.0f, 9.0f, 27.0f, 81.0f, 243.0f};
        U g = 0;
#pragma unroll
        for (short n = 0; n < 5; ++n) {
          const U gn = floor(p3[n] * u);
          t[row][5 * k + n] = gn - 3.0f * g - 1.0f;
          g = gn;
        }
      }
      {
        const U u = U(sb[KQ_PTQ1_0_QH_OFFSET + (it & 1)]) * (1.0f / 256.0f);
        t[row][15] = floor(3.0f * ph * u) - 3.0f * floor(ph * u) - 1.0f;
      }
    }
#pragma unroll
    for (int m = 0; m < KQ_PRISM_MAX_VM; m++) {
      if (m < vm) {
        const device T* xb = x + m * in_vec_size + ib * KQ_PTQ1_0_SUPERBLOCK;
        U xs[16];
#pragma unroll
        for (short k = 0; k < 2; ++k) {
#pragma unroll
          for (short n = 0; n < 5; ++n) {
            xs[5 * k + n] = U(xb[16 * n + 2 * it + k]);
          }
        }
#pragma unroll
        for (short n = 0; n < 5; ++n) {
          xs[10 + n] = U(xb[80 + 8 * n + it]);
        }
        xs[15] = U(xb[120 + it]);
#pragma unroll
        for (int row = 0; row < results_per_simdgroup; row++) {
          U acc = 0;
#pragma unroll
          for (short i = 0; i < 16; ++i) {
            acc = fma(t[row][i], xs[i], acc);
          }
          result[m][row] += d[row] * acc;
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < KQ_PRISM_MAX_VM; m++) {
    if (m < vm) {
      for (int row = 0; row < results_per_simdgroup; row++) {
        U r = simd_sum(result[m][row]);
        if (simd_lid == 0 && row < active_rows) {
          y[m * out_vec_size + out_row + row] = static_cast<T>(r);
        }
      }
    }
  }
}

// ======================= Shared loader and kernels =======================

// ALU tile producer: a 32-weight sub-block is chunks 2s and 2s+1 of
// deq_chunk16, and one thread produces either a whole chunk or an 8-run of
// one, so every kq_qmm_* tile shape maps onto the chunk contract.
template <
    typename Ext,
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqPrismBlockLoader {
  MLX_MTL_CONST int weights_per_block = Ext::superblock;
  MLX_MTL_CONST int bytes_per_block = Ext::block_bytes;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;
  static_assert(BCOLS == sub_block_size, "Prism loader requires BCOLS==32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");
  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;
  static_assert(n_reads % 8 == 0, "vector loader needs whole 8-runs");

  const int src_ld;
  const int row_bytes;
  const int tile_stride;
  const short fixed_sub_block_idx;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device uint8_t* src;
  short sub_block_idx;

  KqPrismBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0) thread
      : src_ld(src_ld_),
        row_bytes(src_ld_* bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS*(src_ld_* bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id* SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() const thread {
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
    if constexpr (n_reads % 16 == 0) {
#pragma unroll
      for (short t = 0; t < n_reads / 16; ++t) {
        const short chunk = (sb * sub_block_size + bj + 16 * t) >> 4;
        float4x4 reg;
        Ext::deq_chunk16(src, chunk, reg);
#pragma unroll
        for (short k = 0; k < 4; ++k) {
          *(threadgroup vec<T, 4>*)(dst + 16 * t + 4 * k) = vec<T, 4>(reg[k]);
        }
      }
    } else {
#pragma unroll
      for (short t = 0; t < n_reads / 8; ++t) {
        const short o = sb * sub_block_size + bj + 8 * t;
        const short h = (o >> 3) & 1;
        float4x4 reg;
        Ext::deq_chunk16(src, o >> 4, reg);
        *(threadgroup vec<T, 4>*)(dst + 8 * t) = vec<T, 4>(reg[2 * h]);
        *(threadgroup vec<T, 4>*)(dst + 8 * t + 4) = vec<T, 4>(reg[2 * h + 1]);
      }
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() thread {
    if (reduction_dim == 1) {
      sub_block_idx++;
      if (sub_block_idx == sub_blocks_per_block) {
        sub_block_idx = 0;
        src += bytes_per_block;
      }
    } else {
      src += tile_stride;
    }
  }
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
using KqPq2_0BlockLoader = KqPrismBlockLoader<
    KqPq2_0Ext,
    T,
    BROWS,
    BCOLS,
    dst_ld,
    reduction_dim,
    tgp_size>;

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
using KqPtq1_0BlockLoader = KqPrismBlockLoader<
    KqPtq1_0Ext,
    T,
    BROWS,
    BCOLS,
    dst_ld,
    reduction_dim,
    tgp_size>;

// The per-codec kernel set over the shared impls: qmv_fast is qmv (the
// fast/K-alignment split is a host-side name gate only).
#define KQ_PRISM_DEFINE_KERNELS(CODEC, SB, BITS, EXT, LOADER)                \
  template <typename T, int group_size, int bits>                            \
  [[kernel]] void kq_##CODEC##_dequantize(                                   \
      const device uint8_t* w,                                               \
      const device uint8_t* /* scales */,                                    \
      device T* out,                                                         \
      const constant uint& num_weights,                                      \
      uint gid [[thread_position_in_grid]]) {                                \
    static_assert(group_size == SB, #CODEC " kernel requires gs=" #SB);      \
    static_assert(bits == BITS, #CODEC " kernel requires bits=" #BITS);      \
    kq_##CODEC##_dequantize_impl<T>(w, out, num_weights, gid);               \
  }                                                                          \
                                                                             \
  template <typename T, short r1ptg, short nsg, short nxpsg>                 \
  [[kernel]] void kq_##CODEC##_mv_ext(                                       \
      const device uint8_t* w,                                               \
      const device uint8_t* /* scales */,                                    \
      const device T* x,                                                     \
      device T* y,                                                           \
      const constant int& in_vec_size,                                       \
      const constant int& out_vec_size,                                      \
      const constant int& /* vm */,                                          \
      uint3 tgpig [[threadgroup_position_in_grid]],                          \
      ushort tiisg [[thread_index_in_simdgroup]],                            \
      ushort sgitg [[simdgroup_index_in_threadgroup]]) {                     \
    kq_mv_ext_impl<T, EXT, r1ptg, nsg, nxpsg>(                               \
        w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);            \
  }                                                                          \
                                                                             \
  template <                                                                 \
      typename T,                                                            \
      int group_size,                                                        \
      int bits,                                                              \
      int results_per_simdgroup = 2>                                         \
  METAL_FUNC void kq_##CODEC##_qmv_fast_impl(                                \
      const device uint8_t* w,                                               \
      const device T* x,                                                     \
      device T* y,                                                           \
      const constant int& in_vec_size,                                       \
      const constant int& out_vec_size,                                      \
      uint3 tid,                                                             \
      uint simd_gid,                                                         \
      uint simd_lid) {                                                       \
    kq_##CODEC##_qmv_impl<T, group_size, bits, results_per_simdgroup>(       \
        w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);        \
  }                                                                          \
                                                                             \
  template <                                                                 \
      typename T,                                                            \
      int group_size,                                                        \
      int bits,                                                              \
      bool aligned_N,                                                        \
      bool batched>                                                          \
  [[kernel]] void kq_##CODEC##_qmm_t(                                        \
      const device uint8_t* w,                                               \
      const device uint8_t* /* scales */,                                    \
      const device T* x,                                                     \
      device T* y,                                                           \
      const constant int& K,                                                 \
      const constant int& N,                                                 \
      const constant int& M,                                                 \
      const constant int& x_batch_ndims,                                     \
      const constant int* x_shape,                                           \
      const constant int64_t* x_strides,                                     \
      const constant int& w_batch_ndims,                                     \
      const constant int* w_shape,                                           \
      const constant int64_t* w_strides,                                     \
      const constant int64_t* /* s_strides */,                               \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint lid [[thread_index_in_threadgroup]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    if constexpr (batched) {                                                 \
      kq_adjust_matrix_offsets<T>(                                           \
          x,                                                                 \
          w,                                                                 \
          y,                                                                 \
          M * N,                                                             \
          x_batch_ndims,                                                     \
          x_shape,                                                           \
          x_strides,                                                         \
          w_batch_ndims,                                                     \
          w_shape,                                                           \
          w_strides,                                                         \
          tid);                                                              \
    }                                                                        \
    static_assert(group_size == SB, #CODEC " requires gs=" #SB);             \
    static_assert(bits == BITS, #CODEC " requires bits=" #BITS);             \
    constexpr int BM = 64, BK = 32, BN = 64;                                 \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                         \
    threadgroup T Xs[BM * BK_padded];                                        \
    threadgroup T Ws[BN * BK_padded];                                        \
    using LoaderW = LOADER<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;      \
    kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(                        \
        w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);          \
  }                                                                          \
                                                                             \
  template <                                                                 \
      typename T,                                                            \
      int group_size,                                                        \
      int bits,                                                              \
      bool aligned_N,                                                        \
      int small_bm = 0>                                                      \
  [[kernel]] void kq_##CODEC##_qmm_t_splitk(                                 \
      const device uint8_t* w,                                               \
      const device uint8_t* /* scales */,                                    \
      const device T* x,                                                     \
      device T* y,                                                           \
      const constant int& K,                                                 \
      const constant int& N,                                                 \
      const constant int& M,                                                 \
      const constant int& k_partition_size,                                  \
      const constant int& split_k_partition_stride,                          \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint lid [[thread_index_in_threadgroup]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    static_assert(group_size == SB, #CODEC " requires gs=" #SB);             \
    static_assert(bits == BITS, #CODEC " requires bits=" #BITS);             \
    constexpr int BM = small_bm ? small_bm : 32;                             \
    constexpr int BK = 32, BN = small_bm ? 64 : 32;                          \
    constexpr int WM = BM == 8 ? 1 : 2, WN = 4 / WM;                         \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                         \
    threadgroup T Xs[BM * BK_padded];                                        \
    threadgroup T Ws[BN * BK_padded];                                        \
    using LoaderW = LOADER<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;      \
    const int k_start = tid.z * k_partition_size;                            \
    x += k_start;                                                            \
    auto wl = w;                                                             \
    wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block; \
    y += tid.z * static_cast<int64_t>(split_k_partition_stride);             \
    kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN, WM, WN>(                \
        wl,                                                                  \
        x,                                                                   \
        y,                                                                   \
        Xs,                                                                  \
        Ws,                                                                  \
        K,                                                                   \
        N,                                                                   \
        M,                                                                   \
        k_partition_size,                                                    \
        tid,                                                                 \
        lid,                                                                 \
        simd_gid,                                                            \
        simd_lid);                                                           \
  }                                                                          \
                                                                             \
  template <typename T, int group_size, int bits, bool batched>              \
  [[kernel]] void kq_##CODEC##_qmm_n(                                        \
      const device uint8_t* w,                                               \
      const device uint8_t* /* scales */,                                    \
      const device T* x,                                                     \
      device T* y,                                                           \
      const constant int& K,                                                 \
      const constant int& N,                                                 \
      const constant int& M,                                                 \
      const constant int& x_batch_ndims,                                     \
      const constant int* x_shape,                                           \
      const constant int64_t* x_strides,                                     \
      const constant int& w_batch_ndims,                                     \
      const constant int* w_shape,                                           \
      const constant int64_t* w_strides,                                     \
      const constant int64_t* /* s_strides */,                               \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint lid [[thread_index_in_threadgroup]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    if constexpr (batched) {                                                 \
      kq_adjust_matrix_offsets<T>(                                           \
          x,                                                                 \
          w,                                                                 \
          y,                                                                 \
          M * N,                                                             \
          x_batch_ndims,                                                     \
          x_shape,                                                           \
          x_strides,                                                         \
          w_batch_ndims,                                                     \
          w_shape,                                                           \
          w_strides,                                                         \
          tid);                                                              \
    }                                                                        \
    static_assert(group_size == SB, #CODEC " requires gs=" #SB);             \
    static_assert(bits == BITS, #CODEC " requires bits=" #BITS);             \
    constexpr int BM = 64, BK = 32, BN = 32;                                 \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                         \
    constexpr int BN_padded = (BN + 16 / sizeof(T));                         \
    threadgroup T Xs[BM * BK_padded];                                        \
    threadgroup T Ws[BK * BN_padded];                                        \
    using LoaderW = LOADER<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;      \
    kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(                                   \
        w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);             \
  }                                                                          \
                                                                             \
  template <typename T, int group_size, int bits, bool batched>              \
  [[kernel]] void kq_##CODEC##_qmv_fast(                                     \
      const device uint8_t* w,                                               \
      const device uint8_t* /* scales */,                                    \
      const device T* x,                                                     \
      device T* y,                                                           \
      const constant int& in_vec_size,                                       \
      const constant int& out_vec_size,                                      \
      const constant int& x_batch_ndims,                                     \
      const constant int* x_shape,                                           \
      const constant int64_t* x_strides,                                     \
      const constant int& w_batch_ndims,                                     \
      const constant int* w_shape,                                           \
      const constant int64_t* w_strides,                                     \
      const constant int64_t* /* s_strides */,                               \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    if constexpr (batched) {                                                 \
      int batch_M = x_shape[x_batch_ndims];                                  \
      kq_adjust_matrix_offsets<T>(                                           \
          x,                                                                 \
          w,                                                                 \
          y,                                                                 \
          out_vec_size * batch_M,                                            \
          x_batch_ndims,                                                     \
          x_shape,                                                           \
          x_strides,                                                         \
          w_batch_ndims,                                                     \
          w_shape,                                                           \
          w_strides,                                                         \
          tid);                                                              \
    }                                                                        \
    kq_##CODEC##_qmv_fast_impl<T, group_size, bits>(                         \
        w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);        \
  }                                                                          \
                                                                             \
  template <typename T, int group_size, int bits, bool batched>              \
  [[kernel]] void kq_##CODEC##_qmv_fast_fine(                                \
      const device uint8_t* w,                                               \
      const device uint8_t* /* scales */,                                    \
      const device T* x,                                                     \
      device T* y,                                                           \
      const constant int& in_vec_size,                                       \
      const constant int& out_vec_size,                                      \
      const constant int& /* x_batch_ndims */,                               \
      const constant int* /* x_shape */,                                     \
      const constant int64_t* /* x_strides */,                               \
      const constant int& /* w_batch_ndims */,                               \
      const constant int* /* w_shape */,                                     \
      const constant int64_t* /* w_strides */,                               \
      const constant int64_t* /* s_strides */,                               \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    kq_##CODEC##_qmv_fast_impl<T, group_size, bits, 1>(                      \
        w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);        \
  }                                                                          \
                                                                             \
  template <typename T, int group_size, int bits, bool batched>              \
  [[kernel]] void kq_##CODEC##_verify_qmv(                                   \
      const device uint8_t* w,                                               \
      const device uint8_t* /* scales */,                                    \
      const device T* x,                                                     \
      device T* y,                                                           \
      const constant int& in_vec_size,                                       \
      const constant int& out_vec_size,                                      \
      const constant int& vm,                                                \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    kq_##CODEC##_verify_qmv_impl<T, group_size, bits>(                       \
        w, x, y, in_vec_size, out_vec_size, vm, tid, simd_gid, simd_lid);    \
  }                                                                          \
                                                                             \
  template <typename T, int group_size, int bits, bool batched>              \
  [[kernel]] void kq_##CODEC##_qmv(                                          \
      const device uint8_t* w,                                               \
      const device uint8_t* /* scales */,                                    \
      const device T* x,                                                     \
      device T* y,                                                           \
      const constant int& in_vec_size,                                       \
      const constant int& out_vec_size,                                      \
      const constant int& x_batch_ndims,                                     \
      const constant int* x_shape,                                           \
      const constant int64_t* x_strides,                                     \
      const constant int& w_batch_ndims,                                     \
      const constant int* w_shape,                                           \
      const constant int64_t* w_strides,                                     \
      const constant int64_t* /* s_strides */,                               \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    if constexpr (batched) {                                                 \
      int batch_M = x_shape[x_batch_ndims];                                  \
      kq_adjust_matrix_offsets<T>(                                           \
          x,                                                                 \
          w,                                                                 \
          y,                                                                 \
          out_vec_size * batch_M,                                            \
          x_batch_ndims,                                                     \
          x_shape,                                                           \
          x_strides,                                                         \
          w_batch_ndims,                                                     \
          w_shape,                                                           \
          w_strides,                                                         \
          tid);                                                              \
    }                                                                        \
    kq_##CODEC##_qmv_impl<T, group_size, bits>(                              \
        w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);        \
  }                                                                          \
                                                                             \
  template <typename T, int group_size, int bits, bool batched>              \
  [[kernel]] void kq_##CODEC##_qmv_fine(                                     \
      const device uint8_t* w,                                               \
      const device uint8_t* /* scales */,                                    \
      const device T* x,                                                     \
      device T* y,                                                           \
      const constant int& in_vec_size,                                       \
      const constant int& out_vec_size,                                      \
      const constant int& /* x_batch_ndims */,                               \
      const constant int* /* x_shape */,                                     \
      const constant int64_t* /* x_strides */,                               \
      const constant int& /* w_batch_ndims */,                               \
      const constant int* /* w_shape */,                                     \
      const constant int64_t* /* w_strides */,                               \
      const constant int64_t* /* s_strides */,                               \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]) {                         \
    kq_##CODEC##_qmv_impl<T, group_size, bits, 1>(                           \
        w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);        \
  }

KQ_PRISM_DEFINE_KERNELS(pq2_0, 128, 2, KqPq2_0Ext, KqPq2_0BlockLoader)
KQ_PRISM_DEFINE_KERNELS(ptq1_0, 128, 1, KqPtq1_0Ext, KqPtq1_0BlockLoader)

// Register-resident MMA verify (kq_verify_mma.h). PQ2_0: lane L owns the
// two 16-code words at bytes 8L..8L+7 of qs; fragment (j, j+8) of a word
// is the pair of code j in each half, masked into the mantissas of a
// half2 (1024 * 4^j + code) and scaled back by 4^-j, which is exact in
// half. Codes 5..7 come from the word shifted down by ten bits.
struct KqPq2_0Mma {
  static constant constexpr int block_k = KQ_PQ2_0_SUPERBLOCK;
  static constant constexpr int block_bytes = KQ_PQ2_0_BLOCK_BYTES;
  static constant constexpr int d_offset = 0;
  static METAL_FUNC int perm(int f, int col) {
    return 32 * (col / 2) + 16 * (f / 8) + (f % 8) + 8 * (col & 1);
  }
  template <int NT>
  static METAL_FUNC void block(
      thread const KqVmmaRows<NT>& rows,
      int boff,
      short L,
      short fm,
      const threadgroup half* xb,
      thread simdgroup_half8x8 (&acc)[NT]) {
    uint wv[NT][2];
    for (short t = 0; t < NT; ++t) {
      const packed_ushort4 v =
          *(const device packed_ushort4*)(rows.p[t] + boff +
                                          KQ_PQ2_0_QS_OFFSET + 8 * L);
      wv[t][0] = uint(v.x) | (uint(v.y) << 16);
      wv[t][1] = uint(v.z) | (uint(v.w) << 16);
    }
    for (short h = 0; h < 2; ++h) {
      for (short j = 0; j < 8; ++j) {
        const short jj = j < 5 ? j : j - 5;
        const uint mask = 0x00030003u << (2 * jj);
        const half scale = half(1.0f / float(1 << (2 * jj)));
        const half off = -half(1024.0f / float(1 << (2 * jj))) - 1.0h;
        half2 a[NT];
        for (short t = 0; t < NT; ++t) {
          const uint src = j < 5 ? wv[t][h] : (wv[t][h] >> 10);
          a[t] =
              fma(as_type<half2>((src & mask) | 0x64006400u),
                  half2(scale),
                  half2(off));
        }
        kq_vmma_step<NT>(xb + 8 * perm(8 * h + j, fm), a, acc);
      }
    }
  }
};

// PTQ1_0: lane L owns byte pair (8p + 2L, 8p + 2L + 1) of qs for p < 3,
// decoded as a half2 base-3 recurrence u = b / 256, t = floor(3u),
// u = 3u - t (every step exact in half); fragment 5p + level is that
// level's trit pair. Fragment 15 is qh: lane L takes level L of (qh0, qh1).
struct KqPtq1_0Mma {
  static constant constexpr int block_k = KQ_PTQ1_0_SUPERBLOCK;
  static constant constexpr int block_bytes = KQ_PTQ1_0_BLOCK_BYTES;
  static constant constexpr int d_offset = KQ_PTQ1_0_D_OFFSET;
  static METAL_FUNC int perm(int f, int col) {
    if (f == 15) {
      return 120 + 2 * (col / 2) + (col & 1);
    }
    const int byte = 8 * (f / 5) + 2 * (col / 2) + (col & 1);
    const int level = f % 5;
    return byte < 16 ? 16 * level + byte : 80 + 8 * level + (byte - 16);
  }
  template <int NT>
  static METAL_FUNC void block(
      thread const KqVmmaRows<NT>& rows,
      int boff,
      short L,
      short fm,
      const threadgroup half* xb,
      thread simdgroup_half8x8 (&acc)[NT]) {
    for (short p = 0; p < 3; ++p) {
      half2 u[NT];
      for (short t = 0; t < NT; ++t) {
        const ushort v =
            *(const device ushort*)(rows.p[t] + boff + 8 * p + 2 * L);
        u[t] = half2(half(v & 0xFFu), half(v >> 8)) * half2(1.0h / 256.0h);
      }
      for (short lv = 0; lv < 5; ++lv) {
        half2 a[NT];
        for (short t = 0; t < NT; ++t) {
          const half2 tt = floor(u[t] * half2(3.0h));
          u[t] = fma(half2(3.0h), u[t], -tt);
          a[t] = tt - half2(1.0h);
        }
        kq_vmma_step<NT>(xb + 8 * perm(5 * p + lv, fm), a, acc);
      }
    }
    half2 a[NT];
    for (short t = 0; t < NT; ++t) {
      const ushort v =
          *(const device ushort*)(rows.p[t] + boff + KQ_PTQ1_0_QH_OFFSET);
      half2 u = half2(half(v & 0xFFu), half(v >> 8)) * half2(1.0h / 256.0h);
      half2 tt = floor(u * half2(3.0h));
      for (short lv = 0; lv < L; ++lv) {
        u = fma(half2(3.0h), u, -tt);
        tt = floor(u * half2(3.0h));
      }
      a[t] = tt - half2(1.0h);
    }
    kq_vmma_step<NT>(xb + 8 * perm(15, fm), a, acc);
  }
};

KQ_DEFINE_VERIFY_MMA_KERNEL(pq2_0, KqPq2_0Mma, 2)
KQ_DEFINE_VERIFY_MMA_KERNEL(ptq1_0, KqPtq1_0Mma, 1)
