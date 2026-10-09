// NAX 3x3 convolution, stride 1, padding 1, as an implicit GEMM.
//
//   y[p, o] = bias[o] + sum_{dy, dx, c} x[p + (dy - 1, dx - 1), c] * w[o, dy,
//   dx, c]
//
// x [B, H, W, C] and y [B, H, W, O] channels last, w [O, 3, 3, C] (rows of
// 9 C), bias [O], all in one float dtype. Rows of the GEMM are the M = B H W
// output pixels, columns the O output channels, and K the 9 C taps and
// channels, walked tap by tap, 32 channels a step. A threadgroup owns one
// [BM, BN] tile of y. Each simdgroup loads its A tile straight from x: a row
// is one output pixel, and its 32 columns are 32 channels of the input
// pixel under the tap, or zeros where the tap falls outside the image. The B
// tile loads from w as in a GEMM with B transposed. Sums are f32 and y is
// rounded once at the store, with the bias added in f32.
//
// C is a multiple of 16, so a 16-channel fragment column is all inside C or
// all past it. M times C and M times O stay below 2^31.
#pragma once

#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

using namespace mlx::steel;

template <typename T, const int BM, const int BN, const int WM, const int WN>
METAL_FUNC void kq_conv3x3_nax_impl(
    const device T* x,
    const device T* w,
    const device T* bias,
    device T* y,
    const constant int& H,
    const constant int& W,
    const constant int& C,
    const constant int& O,
    const constant int& M,
    const constant int& has_bias,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  using Frag = typename NAXTile<T, TM, TK>::NAXFrag_t;
  constexpr short ER = Frag::kElemRows;
  constexpr short EC = Frag::kElemCols;
  constexpr short RJ = Frag::kElemRowsJump;
  constexpr short NR = TM * ER;

  const int y_row = int(tid.y) * BM;
  const int y_col = int(tid.x) * BN;
  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);
  if (y_row + tm >= M) {
    return;
  }
  const short sgp_sm = short(min(int(SM), M - (y_row + tm)));
  const short sgp_sn = short(clamp(O - (y_col + tn), 0, int(SN)));
  if (sgp_sn <= 0) {
    return;
  }

  // The rows of this thread: pixel, its row and column in the image.
  const short2 sc = Frag::get_coord();
  const int hw = H * W;
  int pix[NR];
  int py[NR];
  int px[NR];
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < ER; i++) {
      const short r = mm * ER + i;
      const int p = y_row + tm + mm * 16 + sc.y + i * RJ;
      pix[r] = p;
      if (p < M) {
        const int q = p % hw;
        py[r] = q / W;
        px[r] = q % W;
      } else {
        // Every tap of a pixel past M falls outside the image.
        py[r] = -4;
        px[r] = -4;
      }
    }
  }

  const int ldw = 9 * C;
  const device T* wt = w + int64_t(y_col + tn) * ldw;

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  const bool n_full = sgp_sn == SN;

  STEEL_PRAGMA_NO_UNROLL
  for (short tap = 0; tap < 9; tap++) {
    const short dy = tap / 3 - 1;
    const short dx = tap % 3 - 1;
    // Element offset of the input pixel under the tap, -1 outside.
    int off[NR];
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < NR; r++) {
      const int yy = py[r] + dy;
      const int xx = px[r] + dx;
      const bool in = yy >= 0 && yy < H && xx >= 0 && xx < W;
      off[r] = in ? (pix[r] + dy * W + dx) * C + sc.x : -1;
    }
    const device T* wk = wt + tap * C;

    STEEL_PRAGMA_NO_UNROLL
    for (int c0 = 0; c0 < C; c0 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<T, TN, TK> Btile;

      volatile int compiler_barrier;

      const short ck = short(min(int(SK), C - c0));
      STEEL_PRAGMA_UNROLL
      for (short mm = 0; mm < TM; mm++) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; kk++) {
          thread auto& f = Atile.frag_at(mm, kk);
          const bool col_in = kk * 16 < ck;
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < ER; i++) {
            const int o = off[mm * ER + i];
            if (col_in && o >= 0) {
              const device T* src = x + o + c0 + kk * 16;
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < EC; j++) {
                f[i * EC + j] = src[j];
              }
            } else {
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < EC; j++) {
                f[i * EC + j] = T(0);
              }
            }
          }
        }
      }

      if (n_full && ck == SK) {
        Btile.load(wk + c0, ldw);
      } else {
        Btile.load_safe(wk + c0, ldw, short2(ck, sgp_sn));
      }

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<false>{},
          Btile,
          metal::bool_constant<true>{});

      (void)compiler_barrier;
    }
  }

  device T* ys = y + int64_t(y_row + tm) * O + y_col + tn;
  const bool full = n_full && sgp_sm == SM;
  if (has_bias) {
    NAXTile<float, TM, TN> Ctile;
    const device T* bs = bias + y_col + tn;
    if (n_full) {
      Ctile.load(bs, 0);
    } else {
      Ctile.load_safe(bs, 0, short2(sgp_sn, SM));
    }
    constexpr short NE = decltype(Dtile)::kElemsPerFrag;
    constexpr short NF = decltype(Dtile)::kNumFrags;
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < NF; ++f) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < NE; ++e) {
        Dtile.val_frags[f][e] += Ctile.val_frags[f][e];
      }
    }
  }
  if (full) {
    Dtile.store(ys, O);
  } else {
    Dtile.store_safe(ys, O, short2(sgp_sn, sgp_sm));
  }
}

template <typename T, int BM, int BN, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void kq_conv3x3_nax(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* y [[buffer(3)]],
    const constant int& H [[buffer(4)]],
    const constant int& W [[buffer(5)]],
    const constant int& C [[buffer(6)]],
    const constant int& O [[buffer(7)]],
    const constant int& M [[buffer(8)]],
    const constant int& has_bias [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  kq_conv3x3_nax_impl<T, BM, BN, WM, WN>(
      x, w, bias, y, H, W, C, O, M, has_bias, tid, simd_gid);
}
