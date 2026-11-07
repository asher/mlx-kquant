// Attention with int8 Q K^T and 16-bit P V at head dim 128 on the tensor-op
// (NAX) units, a port of MLX's attention_nax (MIT, steel_attention_nax.h;
// see mlx_kquant/licenses/), and the row quantization of Q and K it takes.
//
// kq_sdpa_qk8_rows quantizes rows of 128: one simdgroup a row, lane l holding
// columns 4l to 4l + 3. f = x * mul, less the row's head mean when one is
// given; m = max(max |f|, 1e-30); the row is clamp(rint(f * (127 / m)),
// -127, 127) in int8 and its scale m / 127.
//
// kq_sdpa_qk8_nax keeps MLX's tile: 64 query rows a threadgroup as four
// simdgroups of 16 rows, a key block of 32, Q loaded again for every key
// block, K and V read from device memory by each simdgroup. Q and K arrive
// in int8 with one float scale a row; Q carries the softmax scale and K has
// its mean over keys removed, which shifts every score of a query row by one
// constant that the softmax drops. S = Q K^T runs on the int8 tensor ops into
// int32 and is scaled into float by the K scale. The softmax runs in base 2,
// lean: the Q scale enters the exponent through one fma, O is rescaled only
// on a block in which a row maximum of the simdgroup moved, and each thread
// keeps its own row sums until the end. P V is MLX's: the float
// probabilities against V in T.
//
// SQ [B, H, qL] and SK [B, Hk, kL] are row-major. Function constants 200
// and 201 say whether qL is a multiple of 64 and kL of 32. Otherwise the
// last query block loads its live Q rows and scales, zero past them, and
// stores only its live rows, and the last key block loads its live K and V
// rows and scales, zero past them, and masks its columns past kL before the
// row maximum. With both constants true none of that code runs.

#pragma once

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/utils.h"

constant bool kq_qk8_align_q [[function_constant(200)]];
constant bool kq_qk8_align_k [[function_constant(201)]];

namespace kq_sdpa_qk8 {

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

// MLX's NAXFrag::mma with the precision flag given: C[16 x 32] += A B with
// B as two 16 x 16 fragments.
template <
    typename CType,
    typename AType,
    typename BType,
    bool TB,
    bool RELAX,
    typename Frag = mlx::steel::BaseNAXFrag>
METAL_FUNC void mma(
    thread typename Frag::template dtype_frag_t<CType>& c0,
    thread typename Frag::template dtype_frag_t<CType>& c1,
    const thread typename Frag::template dtype_frag_t<AType>& a,
    const thread typename Frag::template dtype_frag_t<BType>& b0,
    const thread typename Frag::template dtype_frag_t<BType>& b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      TB,
      RELAX,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ca =
      op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto cb =
      op.template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto cc = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ca)>,
      metal::remove_addrspace_t<decltype(cb)>,
      CType>();
  constexpr short E = Frag::kElemsPerFrag;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < E; i++) {
    ca[i] = a[i];
  }
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < E; i++) {
    cb[i] = b0[i];
    cb[E + i] = b1[i];
  }
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < E; i++) {
    cc[i] = c0[i];
    cc[E + i] = c1[i];
  }
  op.run(ca, cb, cc);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < E; i++) {
    c0[i] = cc[i];
    c1[i] = cc[E + i];
  }
}

} // namespace kq_sdpa_qk8

// x [B, H, L, 128] through its batch, head and row strides; q [B, H, L, 128]
// and s [B, H, L] row-major. mean [B, H, 128] when has_mean. The grid is
// (32 L, H, B) threads.
template <typename T>
[[kernel]] void kq_sdpa_qk8_rows(
    const device T* x [[buffer(0)]],
    const device float* mean [[buffer(1)]],
    device int8_t* q [[buffer(2)]],
    device float* s [[buffer(3)]],
    const constant int64_t* strides [[buffer(4)]],
    const constant int& L [[buffer(5)]],
    const constant float& mul [[buffer(6)]],
    const constant int& has_mean [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = gid.x >> 5;
  const ulong head = ulong(gid.z) * grid.y + gid.y;
  const ulong out_row = head * ulong(L) + row;
  const device T* xr =
      x + gid.z * strides[0] + gid.y * strides[1] + row * strides[2] + lane * 4;
  float4 f = float4(xr[0], xr[1], xr[2], xr[3]) * mul;
  if (has_mean != 0) {
    const device float* mr = mean + head * 128 + lane * 4;
    f -= float4(mr[0], mr[1], mr[2], mr[3]);
  }
  float m = metal::max(
      metal::max(metal::abs(f.x), metal::abs(f.y)),
      metal::max(metal::abs(f.z), metal::abs(f.w)));
  m = metal::max(metal::simd_max(m), 1e-30f);
  const float inv = 127.f / m;
  const float4 r = metal::clamp(metal::rint(f * inv), -127.f, 127.f);
  *(device char4*)(q + out_row * 128 + lane * 4) = char4(r);
  if (lane == 0) {
    s[out_row] = m / 127.f;
  }
}

// clang-format off
template <typename T>
[[kernel, max_total_threads_per_threadgroup(128)]] void kq_sdpa_qk8_nax(
    const device int8_t* Q [[buffer(0)]],
    const device int8_t* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant mlx::steel::AttnParams* params [[buffer(4)]],
    const device float* SQ [[buffer(5)]],
    const device float* SK [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on
  using namespace mlx::steel;
  using namespace kq_sdpa_qk8;
  using AccumType = float;
  constexpr int BQ = 64;
  constexpr int BK = 32;
  constexpr int BD = 128;
  constexpr short kU = 16;
  constexpr int TD = BD / kU;
  constexpr short TK = BK / kU;
  (void)simd_lane_id;

  ulong3 tidl{tid.x, tid.y, tid.z};
  const ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  const short tm = kU * simd_group_id;

  Q += tidl.z * params->Q_strides[0] + tidl.y * params->Q_strides[1] +
      (tidl.x * BQ + tm) * params->Q_strides[2];
  K += tidl.z * params->K_strides[0] + kv_head_idx * params->K_strides[1];
  V += tidl.z * params->V_strides[0] + kv_head_idx * params->V_strides[1];
  O += tidl.z * params->O_strides[0] + tidl.y * params->O_strides[1] +
      (tidl.x * BQ + tm) * params->O_strides[2];
  const int H = params->H;
  const int Hk = H / params->gqa_factor;
  SQ += (tidl.z * H + tidl.y) * ulong(params->qL) + tidl.x * BQ + tm;
  SK += (tidl.z * Hk + kv_head_idx) * ulong(params->kL);

  const int ldq = int(params->Q_strides[2]);
  const int ldk = int(params->K_strides[2]);
  const int ldv = int(params->V_strides[2]);
  const int ldo = int(params->O_strides[2]);

  using otile_t = NAXTile<AccumType, 1, TD>;
  using stile_t = NAXTile<AccumType, 1, TK>;
  using itile_t = NAXTile<int, 1, TK>;
  otile_t Otile;
  Otile.clear();

  constexpr short kRowsPT = otile_t::kRowsPerThread;
  const short2 sc = otile_t::NAXFrag_t::get_coord();
  const short sm = sc.y;
  const short sn = sc.x;

  const bool is_last_q = int(tid.x) == params->NQ_aligned;
  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  // The scale of this thread's two query rows, in base 2.
  metal::vec<AccumType, kRowsPT> qs;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    if (!kq_qk8_align_q && is_last_q) {
      qs[i] = sm + i * 8 < lim_rows_q ? SQ[sm + i * 8] * 1.44269504089f : 0.f;
    } else {
      qs[i] = SQ[sm + i * 8] * 1.44269504089f;
    }
  }

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  const int kb_lim = params->NK;

  for (int kb = 0; kb < kb_lim; kb++) {
    const bool is_last_k = kb == params->NK_aligned;

    // S = Q K^T in int32.
    itile_t Si;
    Si.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
      for (short id = 0; id < TD; id++) {
        NAXTile<int8_t, 1, 1> Qtile;
        NAXTile<int8_t, 2, 1> Ktile;
        if (!kq_qk8_align_q && is_last_q) {
          Qtile.load_rows(Q + id * kU, ldq, lim_rows_q);
        } else {
          Qtile.load(Q + id * kU, ldq);
        }
        if (!kq_qk8_align_k && is_last_k) {
          Ktile.load_rows(
              K + ik * kU * ldk + id * kU, ldk, short(lim_rows_k - ik * kU));
        } else {
          Ktile.load(K + ik * kU * ldk + id * kU, ldk);
        }
        kq_sdpa_qk8::mma<int, int8_t, int8_t, true, false>(
            Si.frag_at(0, ik),
            Si.frag_at(0, ik + 1),
            Qtile.frag_at(0, 0),
            Ktile.frag_at(0, 0),
            Ktile.frag_at(1, 0));
      }
    }

    // Into float: S * sk. The Q scale enters with the exponent.
    stile_t Stile;
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      float4 ks;
      if (!kq_qk8_align_k && is_last_k) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < 4; jj++) {
          const short c = ik * kU + sn + jj;
          ks[jj] = c < lim_rows_k ? SK[c] : 0.f;
        }
      } else {
        ks = *(const device float4*)(SK + ik * kU + sn);
      }
      thread auto& fi = Si.frag_at(0, ik);
      thread auto& fs = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
          const short loc = ii * stile_t::kFragThrCols + jj;
          fs[loc] = float(fi[loc]) * ks[jj];
        }
      }
    }

    // The columns past kL of the last key block.
    if (!kq_qk8_align_k && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const short col_pos = ik * kU + sn;
        thread auto& fs = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const short loc = ii * stile_t::kFragThrCols + jj;
            fs[loc] = (col_pos + jj) < lim_rows_k ? fs[loc] : neg_inf;
          }
        }
      }
    }

    // The online softmax, in base 2.
    {
      metal::vec<AccumType, kRowsPT> bmax;
      metal::vec<AccumType, kRowsPT> new_max;
      int moved = 0;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        bmax[i] = Limits<AccumType>::finite_min;
      }
      Stile.template row_reduce<MaxOp>(bmax);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        new_max[i] = metal::max(max_score[i], bmax[i] * qs[i]);
        moved |= int(new_max[i] != max_score[i]);
      }
      if (simd_max(moved) != 0) {
        metal::vec<AccumType, kRowsPT> factor;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
          factor[i] = fast::exp2(max_score[i] - new_max[i]);
          max_score[i] = new_max[i];
          sum_score[i] = sum_score[i] * factor[i];
        }
        Otile.template row_bin_op<MulOp>(factor);
      }
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fs = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const short loc = ii * stile_t::kFragThrCols + jj;
            const float e = metal::fma(qs[ii], fs[loc], -max_score[ii]);
            fs[loc] = fast::exp2(e);
            sum_score[ii] += fs[loc];
          }
        }
      }
    }

    simdgroup_barrier(mem_flags::mem_none);

    // O += P V.
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id += 2) {
      if (id == 4) {
        threadgroup_barrier(mem_flags::mem_none);
      }
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;
        if (!kq_qk8_align_k && is_last_k) {
          Vtile.load_rows(
              V + ik * kU * ldv + id * kU, ldv, short(lim_rows_k - ik * kU));
        } else {
          Vtile.load(V + ik * kU * ldv + id * kU, ldv);
        }
        otile_t::NAXFrag_t::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            Stile.frag_at(0, ik),
            metal::false_type{},
            Vtile.frag_at(0, 0),
            Vtile.frag_at(0, 1),
            metal::false_type{});
      }
    }

    K += BK * ldk;
    V += BK * ldv;
    SK += BK;
  }

  threadgroup_barrier(mem_flags::mem_none);

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    sum_score[i] += simd_shuffle_xor(sum_score[i], ushort(1));
    sum_score[i] += simd_shuffle_xor(sum_score[i], ushort(8));
  }
  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }
  Otile.template row_bin_op<MulOp>(rcp);

  if (!kq_qk8_align_q && is_last_q) {
    if (lim_rows_q <= 0) {
      return;
    }
    Otile.store_rows(O, ldo, lim_rows_q);
  } else {
    Otile.store(O, ldo);
  }
}
