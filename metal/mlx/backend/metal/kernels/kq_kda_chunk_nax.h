// Chunked gated delta rule (KDA: a per-key-channel decay) prefill on the
// tensor-op (NAX) tiles of MLX's steel attention (MIT; see
// mlx_kquant/licenses/). One threadgroup per (batch, head) walks the
// sequence in 32-token chunks with the [Dv, Dk] state resident in fp32
// fragment registers, 16 value rows per simdgroup, so the T-sequential
// recurrence
//
//   S_t = S_{t-1} diag(g_t) + beta_t (v_t - S_{t-1} diag(g_t) k_t) k_t^T
//   o_t = S_t q_t
//
// runs as matrix products per chunk. With gamma_i the cumulative log gate
// from the chunk start and S_0 the incoming state, a chunk computes
//
//   A_ij = beta_i sum_d k_i k_j exp(gamma_i - gamma_j)   (i > j)
//   P_ij =        sum_d q_i k_j exp(gamma_i - gamma_j)   (i >= j)
//   T    = (I + A)^-1
//   U    = T (beta * (V - Khat S_0^T))     Khat = k exp(gamma)
//   O    = Qhat S_0^T + P U                Qhat = q exp(gamma)
//   S_C  = S_0 diag(exp(gamma_C)) + U^T Ktil   Ktil = k exp(gamma_C - gamma)
//
// exp(gamma_i - gamma_j) is formed through the chunk's middle row as
// exp(gamma_i - gamma_15) * exp(gamma_15 - gamma_j): each factor stays
// within exp(-16 min log g) (exp(80) at the GLM gate floor of -5, inside
// fp32 and bf16 range but not fp16, so the staged operands are bf16
// whatever the activation type), and the entries that pair a row before 16
// with a column past 15 (i < j, which the causal masks drop by select) are
// the only ones that can overflow. Every product is formed transposed
// (value rows as M) so no tile is one fragment wide on N, which the mlx
// 0.32.1 tile_matmad_nax helper mishandles. The state enters the
// V - S Khat^T and S Qhat^T products rounded to bf16 while its fp32 copy
// stays in registers, Z and U are rounded to bf16 as well, and
// diag(beta) T^T and P^T enter as bf16 hi/lo pairs: the output and the
// state land within about 4e-3 relative of the sequential recurrence and
// the error does not grow with the sequence length. The per-chunk critical
// path (one threadgroup per head, 40 cores) is what bounds the kernel, so
// the solve runs in registers and the staged operands go through
// threadgroup memory once. q, k, v [B, T, H, D] in T; log_g, beta, state
// fp32; T % 32 == 0 (the host pads); D == 128.

#pragma once

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;
using namespace mlx::steel;

namespace kq_kda {

constant constexpr int C = 32; // tokens per chunk
constant constexpr int D = 128; // Dk == Dv
constant constexpr int SLD = D + 8; // staged tile row stride (bf16 elements)
constant constexpr int MLD = C + 4; // 32 x 32 fp32 matrix row stride

using ffrag = BaseNAXFrag::dtype_frag_t<float>;

template <typename T>
using tfrag = BaseNAXFrag::dtype_frag_t<T>;

template <typename T>
METAL_FUNC void
split_hi_lo(const thread ffrag& f, thread tfrag<T>& hi, thread tfrag<T>& lo) {
  hi = tfrag<T>(f);
  lo = tfrag<T>(f - ffrag(hi));
}

// A 16 x 16 fragment of a staged tile (row stride SLD) as two 8-byte loads
// per lane: the lane's 4 columns of rows fm and fm + 8.
template <typename T>
METAL_FUNC void load_frag4(thread tfrag<T>& f, const threadgroup T* p) {
  using v4 = metal::vec<T, 4>;
  const short2 sc = BaseNAXFrag::get_coord();
  p += sc.y * SLD + sc.x;
  f = tfrag<T>(
      *reinterpret_cast<const threadgroup v4*>(p),
      *reinterpret_cast<const threadgroup v4*>(p + 8 * SLD));
}

// C[16 x 32] += A[16 x 16] * B^T with B the two 16-row blocks of a staged
// [32 x 128] tile at k offset kk * 16.
template <typename T>
METAL_FUNC void mma_bt(
    thread ffrag& c0,
    thread ffrag& c1,
    const thread tfrag<T>& a,
    const threadgroup T* bt,
    const short kk) {
  tfrag<T> b0, b1;
  load_frag4(b0, bt + kk * 16);
  load_frag4(b1, bt + 16 * SLD + kk * 16);
  BaseNAXFrag::mma(
      c0,
      c1,
      a,
      metal::bool_constant<false>{},
      b0,
      b1,
      metal::bool_constant<true>{});
}

// C[16 x 32] = S[16 x 128] (rounded to bf16) * B^T over a staged [32 x 128]
// tile, on two accumulator chains so consecutive tensor ops overlap.
template <typename T, typename STile>
METAL_FUNC void mma_s_bt(
    thread ffrag& c0,
    thread ffrag& c1,
    const thread STile& S,
    const threadgroup T* bt) {
  ffrag d0 = ffrag(0), d1 = ffrag(0);
  c0 = ffrag(0);
  c1 = ffrag(0);
  STEEL_PRAGMA_UNROLL
  for (short kk = 0; kk < 8; kk += 2) {
    const tfrag<T> a0 = tfrag<T>(S.frag_at(0, kk));
    const tfrag<T> a1 = tfrag<T>(S.frag_at(0, kk + 1));
    mma_bt<T>(c0, c1, a0, bt, kk);
    mma_bt<T>(d0, d1, a1, bt, kk + 1);
  }
  c0 += d0;
  c1 += d1;
}

} // namespace kq_kda

// GATED: buffer 3 holds the gate pre-activation a [B, T, H, D] in T and the
// per-token log gate is lb * sigmoid(a_scale[h] * (a + dt_bias[h, d])),
// zero on the padded rows past T_real (the log_g form pads with zeros).
template <typename T, bool GATED>
[[kernel, max_total_threads_per_threadgroup(256)]] void kq_kda_chunk_nax(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device void* gate_in [[buffer(3)]],
    const device float* beta [[buffer(4)]],
    const device float* state_in [[buffer(5)]],
    device T* y [[buffer(6)]],
    device float* state_out [[buffer(7)]],
    const constant int& T_len [[buffer(8)]],
    const constant int& H [[buffer(9)]],
    const device float* a_scale [[buffer(10)]],
    const device float* dt_bias [[buffer(11)]],
    const constant float& lb [[buffer(12)]],
    const constant int& T_real [[buffer(13)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  using namespace kq_kda;
  using ST = bfloat16_t;
  using tf = tfrag<ST>;
  using stile_t = NAXTile<float, 1, 8>; // 16 value rows x 128 key channels

  threadgroup ST st0[C * SLD];
  threadgroup ST st1[C * SLD];
  threadgroup float mA[C * MLD]; // A^T, then diag(beta) T^T
  threadgroup float mP[C * MLD]; // P^T
  threadgroup float bet[C];
  threadgroup float gC[D];

  const int b = int(tgid.x) / H;
  const int h = int(tgid.x) % H;
  const int HD = H * D;
  const int sg = int(simd_group_id);
  const int dv0 = sg * 16;

  const device T* qb = q + (size_t(b) * T_len * H + h) * D;
  const device T* kb = k + (size_t(b) * T_len * H + h) * D;
  const device T* vb = v + (size_t(b) * T_len * H + h) * D;
  const device float* gb = static_cast<const device float*>(gate_in) +
      (size_t(b) * T_len * H + h) * D;
  const device T* ab =
      static_cast<const device T*>(gate_in) + (size_t(b) * T_len * H + h) * D;
  const device float* betb = beta + size_t(b) * T_len * H + h;
  device T* yb = y + (size_t(b) * T_len * H + h) * D;
  const size_t soff = (size_t(b) * H + h) * D * D + size_t(dv0) * D;

  stile_t S;
  S.load(state_in + soff, D);

  // Staging threads: key channel d, rows 16 * hrow .. + 15.
  const int d = int(tid) & (D - 1);
  const int hrow = int(tid) >> 7;
  // Staged tiles reinterpreted inside the fp32 matrices: 16 x SLD of ST.
  threadgroup ST* mAt = reinterpret_cast<threadgroup ST*>(mA);
  threadgroup ST* mPt = reinterpret_cast<threadgroup ST*>(mP);

  const short2 sc = BaseNAXFrag::get_coord(); // (fn, fm)

  for (int t0 = 0; t0 < T_len; t0 += C) {
    // --- chunk constants and the thread's 16 rows in registers ----------
    if (tid < uint(C)) {
      bet[tid] = betb[size_t(t0 + tid) * H];
    }
    const device float* gr = gb + size_t(t0) * HD + d;
    const device T* ar = ab + size_t(t0) * HD + d;
    const float asc = GATED ? a_scale[h] : 0.0f;
    const float dtb = GATED ? dt_bias[h * D + d] : 0.0f;
    const device T* kr = kb + size_t(t0) * HD + d;
    const device T* qr = qb + size_t(t0) * HD + d;
    // gamma = the within-chunk prefix sum of the per-token log gate, read
    // for all 32 rows of this channel (the two staging threads of a
    // channel share the lines); the thread keeps its own 16.
    float gam[16];
    float gacc = 0.0f;
    float g15 = 0.0f;
    STEEL_PRAGMA_UNROLL
    for (int r = 0; r < C; ++r) {
      if (GATED) {
        const float z = asc * (float(ar[r * HD]) + dtb);
        const float lg = lb / (1.0f + metal::exp(-z));
        gacc += (t0 + r < T_real) ? lg : 0.0f;
      } else {
        gacc += gr[r * HD];
      }
      if (r == 15) {
        g15 = gacc;
      }
      if ((r >> 4) == hrow) {
        gam[r & 15] = gacc;
      }
    }
    const float g31 = gacc;
    if (hrow == 0) {
      gC[d] = metal::exp(g31);
    }
    // e15 = exp(gamma - gamma_15), eg = exp(gamma), ex = exp(gamma_31 -
    // gamma); ex is at most 1, e15 and 1 / e15 at most exp(-16 min log g).
    float kv[16], qv[16], e15[16], eg[16], ex[16];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 16; ++i) {
      const int r = hrow * 16 + i;
      const float g = gam[i];
      kv[i] = float(kr[r * HD]);
      qv[i] = float(qr[r * HD]);
      e15[i] = metal::exp(g - g15);
      eg[i] = metal::exp(g);
      ex[i] = metal::exp(g31 - g);
    }

    // --- phase 1: A^T and P^T against the reference row 15 -------------
    // st0 = ky = k exp(gamma_15 - gamma), st1 = kx = k exp(gamma -
    // gamma_15), qx = q exp(gamma - gamma_15) in the bytes of mA (rows
    // 0..15) and mP (rows 16..31). Entries pairing a row before 16 with a
    // column past 15 can overflow; the masks drop them by select.
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 16; ++i) {
      const int r = hrow * 16 + i;
      st0[r * SLD + d] = ST(kv[i] / e15[i]);
      st1[r * SLD + d] = ST(kv[i] * e15[i]);
      (hrow == 0 ? mAt : mPt)[i * SLD + d] = ST(qv[i] * e15[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ffrag c0 = ffrag(0), c1 = ffrag(0);
    if (sg < 2) {
      // A^T rows 16 sg .. + 15: ky against kx^T, [16 x 32].
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < 8; ++kk) {
        tf a;
        load_frag4(a, st0 + sg * 16 * SLD + kk * 16);
        mma_bt<ST>(c0, c1, a, st1, kk);
      }
    } else if (sg < 4) {
      // P^T rows 16 (sg - 2) .. + 15: ky against qx^T, [16 x 32].
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < 8; ++kk) {
        tf a, b0, b1;
        load_frag4(a, st0 + (sg - 2) * 16 * SLD + kk * 16);
        load_frag4(b0, mAt + kk * 16);
        load_frag4(b1, mPt + kk * 16);
        BaseNAXFrag::mma(
            c0,
            c1,
            a,
            metal::bool_constant<false>{},
            b0,
            b1,
            metal::bool_constant<true>{});
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Park the products as fp32 [32 x 32] with the causal masks folded
    // in: A^T[r][s] = beta_s A'^T[r][s] for s > r, else 0; P^T[r][s] for
    // s >= r, else 0 (r the key row, s the query column).
    if (sg < 4) {
      const bool is_a = sg < 2;
      threadgroup float* dst = (is_a ? mA : mP) + (sg & 1) * 16 * MLD;
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < 2; ++nn) {
        const thread ffrag& cf = nn == 0 ? c0 : c1;
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
          const int rl = sc.y + (e >> 2) * 8;
          const int r = (sg & 1) * 16 + rl;
          const int s = nn * 16 + sc.x + (e & 3);
          const float val = cf[e];
          dst[rl * MLD + s] =
              is_a ? (s > r ? bet[s] * val : 0.0f) : (s >= r ? val : 0.0f);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Solve (I + A^T) X = I by back substitution in registers: lane p of
    // column col holds X[s][col] for s = p + 8 m; the 8 lanes split each
    // row's sum and reduce with shuffles.
    const int col = int(tid) >> 3;
    const int p = int(tid) & 7;
    float xr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    STEEL_PRAGMA_UNROLL
    for (int r = C - 1; r >= 0; --r) {
      float acc = 0.0f;
      STEEL_PRAGMA_UNROLL
      for (int m = 0; m < 4; ++m) {
        const int s = p + 8 * m;
        if (s > r) {
          acc += mA[r * MLD + s] * xr[m];
        }
      }
      acc += simd_shuffle_xor(acc, 1);
      acc += simd_shuffle_xor(acc, 2);
      acc += simd_shuffle_xor(acc, 4);
      if (p == (r & 7)) {
        xr[r >> 3] = (r == col ? 1.0f : 0.0f) - acc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- phase 2: mA = diag(beta) T^T, Khat = k exp(gamma) -> st0, Qhat =
    // q exp(gamma) -> st1 ------------------------------------------------
    STEEL_PRAGMA_UNROLL
    for (int m = 0; m < 4; ++m) {
      const int s = p + 8 * m;
      mA[s * MLD + col] = bet[s] * xr[m];
    }
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 16; ++i) {
      const int r = hrow * 16 + i;
      st0[r * SLD + d] = ST(kv[i] * eg[i]);
      st1[r * SLD + d] = ST(qv[i] * eg[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Z^T = V^T - S Khat^T and the S Qhat^T part of O^T, [16 x 32] per
    // simdgroup; the two state products overlap.
    ffrag z0, z1, o0, o1;
    mma_s_bt<ST>(z0, z1, S, st0);
    mma_s_bt<ST>(o0, o1, S, st1);
    {
      ffrag vt0, vt1;
      const device T* vp = vb + size_t(t0) * HD + dv0;
      BaseNAXFrag::load(vt0, vp, Int<1>{}, HD);
      BaseNAXFrag::load(vt1, vp + 16 * HD, Int<1>{}, HD);
      z0 = vt0 - z0;
      z1 = vt1 - z1;
    }

    // U^T = Z^T (diag(beta) T^T), [16 x 32]: Z rounded to bf16, the solved
    // matrix as a hi/lo pair.
    ffrag u0 = ffrag(0), u1 = ffrag(0);
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < 2; ++kk) {
      tf b0h, b0l, b1h, b1l;
      ffrag bf0, bf1;
      const tf zh = tf(kk == 0 ? z0 : z1);
      BaseNAXFrag::load(bf0, mA + kk * 16 * MLD, Int<MLD>{}, Int<1>{});
      BaseNAXFrag::load(bf1, mA + kk * 16 * MLD + 16, Int<MLD>{}, Int<1>{});
      split_hi_lo<ST>(bf0, b0h, b0l);
      split_hi_lo<ST>(bf1, b1h, b1l);
      BaseNAXFrag::mma(
          u0,
          u1,
          zh,
          metal::bool_constant<false>{},
          b0h,
          b1h,
          metal::bool_constant<false>{});
      BaseNAXFrag::mma(
          u0,
          u1,
          zh,
          metal::bool_constant<false>{},
          b0l,
          b1l,
          metal::bool_constant<false>{});
    }
    // Every simdgroup is past Khat before st0 takes Ktil below.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- phase 3: Ktil = k exp(gamma_31 - gamma) -> st0 while O^T
    // finishes with U^T P^T; O^T parks as [token][dv] in st1 ------------
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 16; ++i) {
      const int r = hrow * 16 + i;
      st0[r * SLD + d] = ST(kv[i] * ex[i]);
    }
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < 2; ++kk) {
      tf b0h, b0l, b1h, b1l;
      ffrag bf0, bf1;
      const tf uh = tf(kk == 0 ? u0 : u1);
      BaseNAXFrag::load(bf0, mP + kk * 16 * MLD, Int<MLD>{}, Int<1>{});
      BaseNAXFrag::load(bf1, mP + kk * 16 * MLD + 16, Int<MLD>{}, Int<1>{});
      split_hi_lo<ST>(bf0, b0h, b0l);
      split_hi_lo<ST>(bf1, b1h, b1l);
      BaseNAXFrag::mma(
          o0,
          o1,
          uh,
          metal::bool_constant<false>{},
          b0h,
          b1h,
          metal::bool_constant<false>{});
      BaseNAXFrag::mma(
          o0,
          o1,
          uh,
          metal::bool_constant<false>{},
          b0l,
          b1l,
          metal::bool_constant<false>{});
    }
    BaseNAXFrag::store(o0, st1 + dv0, Int<1>{}, Int<SLD>{});
    BaseNAXFrag::store(o1, st1 + 16 * SLD + dv0, Int<1>{}, Int<SLD>{});
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // y out in whole rows.
    {
      device T* yp = yb + size_t(t0) * HD + d;
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < 16; ++i) {
        const int r = hrow * 16 + i;
        yp[size_t(r) * HD] = T(st1[r * SLD + d]);
      }
    }

    // S = S diag(exp(gamma_31)) + U^T Ktil.
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < 8; ++kk) {
      thread ffrag& sf = S.frag_at(0, kk);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 8; ++e) {
        sf[e] *= gC[kk * 16 + sc.x + (e & 3)];
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < 2; ++kk) {
      const tf uh = tf(kk == 0 ? u0 : u1);
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < 8; nn += 2) {
        tf b0, b1;
        load_frag4(b0, st0 + kk * 16 * SLD + nn * 16);
        load_frag4(b1, st0 + kk * 16 * SLD + (nn + 1) * 16);
        BaseNAXFrag::mma(
            S.frag_at(0, nn),
            S.frag_at(0, nn + 1),
            uh,
            metal::bool_constant<false>{},
            b0,
            b1,
            metal::bool_constant<false>{});
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  S.store(state_out + soff, D);
}
