// Scalar CPU encoders for the IQ GGUF codecs. Faithful ports of ggml's
// quantize_row_iq*_impl (llama.cpp, MIT; see mlx_kquant/licenses/
// llama.cpp-LICENSE): same scale search, accumulation order, grid lookups, and
// rounding, so the wire bytes match llama-quantize for the deterministic
// codecs. Built -ffp-contract=off (CMakeLists.txt) so the float reductions are
// not FMA-fused -- the byte-exactness depends on it. CPU-only: kq.quantize
// routes IQ requests to a CPU stream at the op level.
//
// E0 lands the two deterministic codecs (iq4_nl, iq4_xs), which share ggml's
// quantize_row_iq4_nl_impl. The grid codecs (iq3/iq2/iq1) and their inverse-
// index machinery follow in later phases.
#include "kquant_iq_encode.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "kquant_cpu_encode_util.h" // write_f16, kq_nearest_int (anon-ns, per-TU)
#include "kquant_iq_tables.h" // kvalues_iq4nl

#include "mlx/types/half_types.h"

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

// ggml GROUP_MAX_EPS (ggml-quants.c): the below-which-treat-as-zero amax floor.
// iq4 uses the base value; the lower-bit codecs override it in later phases.
constexpr float GROUP_MAX_EPS = 1e-15f;

// ggml best_index_int8 (ggml-quants.c:28): nearest codebook entry to x by
// binary search over the sorted `val` table, tie-breaking toward the lower
// index.
inline int best_index_int8(int n, const int8_t* val, float x) {
  if (x <= val[0]) {
    return 0;
  }
  if (x >= val[n - 1]) {
    return n - 1;
  }
  int ml = 0, mu = n - 1;
  while (mu - ml > 1) {
    int mav = (ml + mu) / 2;
    if (x < val[mav]) {
      mu = mav;
    } else {
      ml = mav;
    }
  }
  return x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

// Port of ggml quantize_row_iq4_nl_impl (ggml-quants.c:4894), writing one
// super-block straight into the GGUF wire layout. super_block_size is 32 for
// iq4_nl (one 32-weight block: [fp16 d][qs[16]]) or 256 for iq4_xs (eight
// blocks: [fp16 d][u16 scales_h][scales_l[4]][qs[128]]). values is
// kvalues_iq4nl.
template <typename T>
void quantize_iq4_block(
    int super_block_size,
    const T* x,
    uint8_t* block,
    const float* quant_weights,
    int ntry) {
  constexpr int block_size = 32;
  const int8_t* values = kvalues_iq4nl;
  const int nsb = super_block_size / block_size;

  float sigma2 = 0.f;
  for (int j = 0; j < super_block_size; ++j) {
    float v = float(x[j]);
    sigma2 += v * v;
  }
  sigma2 *= 2.f / float(super_block_size);

  uint8_t L[256];
  float scales[8];
  float weight[block_size];

  float max_scale = 0.f, amax_scale = 0.f;
  for (int ib = 0; ib < nsb; ++ib) {
    const T* xb = x + ib * block_size;
    uint8_t* Lb = L + ib * block_size;
    if (quant_weights) {
      const float* qw = quant_weights + ib * block_size;
      for (int j = 0; j < block_size; ++j) {
        float xbj = float(xb[j]);
        weight[j] = qw[j] * std::sqrt(sigma2 + xbj * xbj);
      }
    } else {
      for (int j = 0; j < block_size; ++j) {
        float xbj = float(xb[j]);
        weight[j] = xbj * xbj;
      }
    }
    float amax = 0.f, max = 0.f;
    for (int j = 0; j < block_size; ++j) {
      float ax = std::fabs(float(xb[j]));
      if (ax > amax) {
        amax = ax;
        max = float(xb[j]);
      }
    }
    if (amax < GROUP_MAX_EPS) {
      scales[ib] = 0.f;
      continue;
    }
    float d = ntry > 0 ? -max / float(values[0]) : max / float(values[0]);
    float id = 1.f / d;
    float sumqx = 0.f, sumq2 = 0.f;
    for (int j = 0; j < block_size; ++j) {
      float al = id * float(xb[j]);
      int l = best_index_int8(16, values, al);
      Lb[j] = uint8_t(l);
      float q = float(values[l]);
      float w = weight[j];
      sumqx += w * q * float(xb[j]);
      sumq2 += w * q * q;
    }
    d = sumq2 > 0.f ? sumqx / sumq2 : 0.f;
    float best = d * sumqx;
    for (int itry = -ntry; itry <= ntry; ++itry) {
      id = (float(itry) + float(values[0])) / max;
      sumqx = sumq2 = 0.f;
      for (int j = 0; j < block_size; ++j) {
        float al = id * float(xb[j]);
        int l = best_index_int8(16, values, al);
        float q = float(values[l]);
        float w = weight[j];
        sumqx += w * q * float(xb[j]);
        sumq2 += w * q * q;
      }
      if (sumq2 > 0.f && sumqx * sumqx > best * sumq2) {
        d = sumqx / sumq2;
        best = d * sumqx;
      }
    }
    scales[ib] = d;
    float abs_d = std::fabs(d);
    if (abs_d > amax_scale) {
      amax_scale = abs_d;
      max_scale = d;
    }
  }

  uint8_t* q4;
  if (nsb > 1) {
    // iq4_xs: super-scale + per-sub-block 6-bit scales (4 low nibbles in
    // scales_l, 2 high bits in scales_h), then re-quantize each sub-block.
    uint8_t* scales_l = block + 4;
    q4 = block + 8;
    uint16_t scales_h = 0;
    for (int i = 0; i < 4; ++i) {
      scales_l[i] = 0;
    }
    float d = -max_scale / 32.f;
    write_f16(block, d);
    float id = d ? 1.f / d : 0.f;
    for (int ib = 0; ib < nsb; ++ib) {
      int l = kq_nearest_int(id * scales[ib]);
      l = std::max(-32, std::min(31, l));
      float dl = d * float(l);
      float idl = dl ? 1.f / dl : 0.f;
      uint8_t* Lb = L + ib * block_size;
      const T* xb = x + ib * block_size;
      for (int j = 0; j < block_size; ++j) {
        Lb[j] = uint8_t(best_index_int8(16, values, idl * float(xb[j])));
      }
      l += 32;
      uint8_t l_l = uint8_t(l & 0xf);
      uint8_t l_h = uint8_t(l >> 4);
      if (ib % 2 == 0) {
        scales_l[ib / 2] = l_l;
      } else {
        scales_l[ib / 2] |= uint8_t(l_l << 4);
      }
      scales_h |= uint16_t(uint16_t(l_h) << (2 * (ib % 8)));
    }
    block[2] = uint8_t(scales_h & 0xff);
    block[3] = uint8_t(scales_h >> 8);
  } else {
    // iq4_nl: single block, scale written straight to the fp16 d slot.
    q4 = block + 2;
    write_f16(block, scales[0]);
    if (ntry > 0) {
      float id = scales[0] ? 1.f / scales[0] : 0.f;
      for (int j = 0; j < super_block_size; ++j) {
        L[j] = uint8_t(best_index_int8(16, values, id * float(x[j])));
      }
    }
  }

  // Pack levels into nibbles: low half of each 32-group in the low nibble.
  for (int i = 0; i < super_block_size / 32; ++i) {
    for (int j = 0; j < 16; ++j) {
      q4[16 * i + j] = uint8_t(L[32 * i + j] | (L[32 * i + 16 + j] << 4));
    }
  }
}

} // namespace

template <typename T>
void kquant_iq_quantize_dispatch(
    const T* w,
    uint8_t* out,
    std::size_t num_weights,
    const std::string& kquant_type,
    const float* imatrix,
    std::size_t K) {
  // ggml's production iq4 quantizers use ntry=7.
  if (kquant_type == "iq4_nl") {
    constexpr int wpb = 32, bpb = 18;
    std::size_t nblocks = num_weights / wpb;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const float* qw = imatrix ? imatrix + ((b * wpb) % K) : nullptr;
      quantize_iq4_block<T>(wpb, w + b * wpb, out + b * bpb, qw, 7);
    }
  } else if (kquant_type == "iq4_xs") {
    constexpr int wpb = 256, bpb = 136;
    std::size_t nblocks = num_weights / wpb;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const float* qw = imatrix ? imatrix + ((b * wpb) % K) : nullptr;
      quantize_iq4_block<T>(wpb, w + b * wpb, out + b * bpb, qw, 7);
    }
  } else {
    throw std::runtime_error(
        "[mlx_kquant] quantize: IQ encode not implemented for codec: " +
        kquant_type);
  }
}

template void kquant_iq_quantize_dispatch<float>(
    const float*,
    uint8_t*,
    std::size_t,
    const std::string&,
    const float*,
    std::size_t);
template void kquant_iq_quantize_dispatch<mx::float16_t>(
    const mx::float16_t*,
    uint8_t*,
    std::size_t,
    const std::string&,
    const float*,
    std::size_t);
template void kquant_iq_quantize_dispatch<mx::bfloat16_t>(
    const mx::bfloat16_t*,
    uint8_t*,
    std::size_t,
    const std::string&,
    const float*,
    std::size_t);

} // namespace mlx_kquant
