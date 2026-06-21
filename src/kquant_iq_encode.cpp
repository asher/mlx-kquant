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
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "kquant_cpu_encode_util.h" // write_f16, kq_nearest_int, kq_make_qp_quants
#include "kquant_iq_encode_tables.h" // kgrid_2bit_*, kgrid_1bit_2048, kgrid_256/512
#include "kquant_iq_tables.h" // kvalues_iq4nl

#include "mlx/types/half_types.h"

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

// ggml GROUP_MAX_EPS (ggml-quants.c): the below-which-treat-as-zero amax floor.
// iq4, iq2_xxs and iq2_xs use the base value; the other low-bit codecs use a
// looser per-codec floor.
constexpr float GROUP_MAX_EPS = 1e-15f;
constexpr float GROUP_MAX_EPS_IQ2_S = 1e-8f;
constexpr float GROUP_MAX_EPS_IQ3_XXS = 1e-8f;
constexpr float GROUP_MAX_EPS_IQ1_S = 1e-12f;
constexpr float GROUP_MAX_EPS_IQ1_M = 1e-7f;
constexpr float IQ1S_DELTA = 0.125f; // ggml IQ1S_DELTA == IQ1M_DELTA

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

// ---- IQ grid inverse-index machinery (ggml iq2xs_init_impl / iq3xs_init_impl)
//
// The grid IQ codecs search a lattice. ggml precomputes per grid: the expanded
// grid (each entry's NVALS values as 2l+1), an inverse map
// kmap[packed_index] -> grid index (negative sentinel for off-grid points), and
// a flat kneighbours list giving the nearest grid points for every off-grid
// packed index. We build the same three arrays from the compact init grids
// (kquant_iq_encode_tables.h). NB: this expanded grid is the SEARCH
// representation (odd 2l+1 values), distinct from the decode tables in
// kquant_iq_tables.h, which hold the reconstruction values. iq2 family: uint64
// grid, 8 values, 2 bits each; iq3 family: uint32 grid, 4 values, 3 bits each.

template <typename GridT, int NVALS, int BITS>
struct IqGridIndex {
  std::vector<GridT> grid; // expanded grid: NVALS bytes of 2l+1 per entry
  std::vector<int> kmap; // packed index -> grid index, or -(offset+1)
  std::vector<uint16_t> kneighbours; // [count, idx...] slice per off-grid slot
};

template <typename GridT, int NVALS, int BITS>
void build_iq_grid_index(
    const uint16_t* kgrid,
    int grid_size,
    int kmap_size,
    int nwant,
    IqGridIndex<GridT, NVALS, BITS>& out) {
  constexpr int mask = (1 << BITS) - 1;

  // Expand the compact grid into the 2l+1 search representation.
  out.grid.resize(grid_size);
  for (int k = 0; k < grid_size; ++k) {
    auto* pos = reinterpret_cast<int8_t*>(&out.grid[k]);
    for (int i = 0; i < NVALS; ++i) {
      int l = (kgrid[k] >> (BITS * i)) & mask;
      pos[i] = int8_t(2 * l + 1);
    }
  }

  // Invert: kmap[packed index of grid entry] = entry index; -1 otherwise. The
  // packed index recovered here equals kgrid[k] (the compact entry), which the
  // table guarantees is < kmap_size.
  out.kmap.assign(kmap_size, -1);
  for (int i = 0; i < grid_size; ++i) {
    const auto* aux8 = reinterpret_cast<const uint8_t*>(&out.grid[i]);
    int index = 0;
    for (int k = 0; k < NVALS; ++k) {
      int q = (aux8[k] - 1) / 2;
      index |= (q << (BITS * k));
    }
    if (index < 0 || index >= kmap_size) {
      throw std::runtime_error(
          "[mlx_kquant] quantize: IQ grid index out of range");
    }
    out.kmap[index] = i;
  }

  // For each off-grid packed index, collect its nearest grid points (within
  // `nwant` distinct squared distances). ggml ties on grid index (a total
  // order), so the result is deterministic without its OpenMP parallelism.
  std::vector<std::vector<uint16_t>> nbr(kmap_size);
  std::vector<std::pair<int, int>> dist2(grid_size);
  int8_t pos[NVALS];
  std::size_t total = 0;
  for (int i = 0; i < kmap_size; ++i) {
    if (out.kmap[i] >= 0) {
      continue;
    }
    for (int k = 0; k < NVALS; ++k) {
      int l = (i >> (BITS * k)) & mask;
      pos[k] = int8_t(2 * l + 1);
    }
    for (int j = 0; j < grid_size; ++j) {
      const auto* pg = reinterpret_cast<const int8_t*>(&out.grid[j]);
      int d2 = 0;
      for (int k = 0; k < NVALS; ++k) {
        int diff = pg[k] - pos[k];
        d2 += diff * diff;
      }
      dist2[j] = {d2, j};
    }
    std::sort(dist2.begin(), dist2.end()); // by distance, then grid index
    int d2 = dist2[0].first, nhave = 1;
    for (int j = 0; j < grid_size; ++j) {
      if (dist2[j].first > d2) {
        if (nhave == nwant) {
          break;
        }
        d2 = dist2[j].first;
        ++nhave;
      }
      nbr[i].push_back(uint16_t(dist2[j].second));
    }
    total += 1 + nbr[i].size();
  }

  // Flatten into [count, idx...] slices; the kmap sentinel for an off-grid slot
  // is -(offset+1), so the encoder recovers the slice via `kneighbours - kmap -
  // 1`.
  out.kneighbours.resize(total);
  int counter = 0;
  for (int i = 0; i < kmap_size; ++i) {
    if (out.kmap[i] >= 0) {
      continue;
    }
    out.kmap[i] = -(counter + 1);
    out.kneighbours[counter++] = uint16_t(nbr[i].size());
    for (uint16_t idx : nbr[i]) {
      out.kneighbours[counter++] = idx;
    }
  }
}

using Iq2Index = IqGridIndex<uint64_t, 8, 2>;

// iq2 family (ggml iq2_data gindex): 0=iq2_xxs(256), 1=iq2_xs(512),
// 2=iq1_s/iq1_m(2048, shared), 3=iq2_s(1024). All share kmap_size 43692. Built
// once per gindex on first use; the returned reference is read-only thereafter.
const Iq2Index& iq2_index(int gindex) {
  static Iq2Index idx[4];
  static std::once_flag flag[4];
  std::call_once(flag[gindex], [gindex] {
    constexpr int kmap_size = 43692;
    switch (gindex) {
      case 0:
        build_iq_grid_index(kgrid_2bit_256, 256, kmap_size, 2, idx[0]);
        break;
      case 1:
        build_iq_grid_index(kgrid_2bit_512, 512, kmap_size, 2, idx[1]);
        break;
      case 2:
        build_iq_grid_index(kgrid_1bit_2048, 2048, kmap_size, 3, idx[2]);
        break;
      case 3:
        build_iq_grid_index(kgrid_2bit_1024, 1024, kmap_size, 1, idx[3]);
        break;
    }
  });
  return idx[gindex];
}

// ggml iq2_find_best_neighbour (ggml-quants.c:3198): of an off-grid point's
// candidate neighbours, pick the one minimising weighted squared error to the
// scaled lattice point, and write its (value-1)/2 levels into L.
int iq2_find_best_neighbour(
    const uint16_t* neighbours,
    const uint64_t* grid,
    const float* xval,
    const float* weight,
    float scale,
    int8_t* L) {
  int num_neighbors = neighbours[0];
  float best_d2 = FLT_MAX;
  int grid_index = -1;
  for (int j = 1; j <= num_neighbors; ++j) {
    const auto* pg = reinterpret_cast<const int8_t*>(grid + neighbours[j]);
    float d2 = 0;
    for (int i = 0; i < 8; ++i) {
      float q = pg[i];
      float diff = scale * q - xval[i];
      d2 += weight[i] * diff * diff;
    }
    if (d2 < best_d2) {
      best_d2 = d2;
      grid_index = neighbours[j];
    }
  }
  const auto* pg = reinterpret_cast<const int8_t*>(grid + grid_index);
  for (int i = 0; i < 8; ++i) {
    L[i] = (pg[i] - 1) / 2;
  }
  return grid_index;
}

// Port of ggml quantize_row_iq2_xxs_impl (ggml-quants.c:3222) for one QK_K=256
// super-block -> block_iq2_xxs wire layout ([fp16 d][u16 qs[32]] = 66 bytes).
// imatrix-required: quant_weights is non-null (enforced at the op level).
template <typename T>
void quantize_iq2_xxs_block(
    const T* x,
    uint8_t* block,
    const float* quant_weights) {
  constexpr int kMaxQ = 3;
  const Iq2Index& gi = iq2_index(0);
  const uint64_t* kgrid_q2xs = gi.grid.data();
  const int* kmap_q2xs = gi.kmap.data();
  const uint16_t* kneighbors_q2xs = gi.kneighbours.data();

  float scales[8]; // QK_K/32
  float weight[32];
  float xval[32];
  int8_t L[32];
  int8_t Laux[32];
  float waux[32];
  uint8_t block_signs[4];
  uint32_t q2[16]; // 2*(QK_K/32)
  std::memset(q2, 0, sizeof(q2));

  float sumx2 = 0;
  for (int i = 0; i < 256; ++i) {
    float v = float(x[i]);
    sumx2 += v * v;
  }
  float sigma2 = sumx2 / 256.0f;

  float max_scale = 0;
  for (int ib = 0; ib < 8; ++ib) {
    const T* xb = x + 32 * ib;
    const float* qw = quant_weights + 32 * ib;
    for (int i = 0; i < 32; ++i) {
      float v = float(xb[i]);
      weight[i] = qw[i] * std::sqrt(sigma2 + v * v);
    }
    for (int i = 0; i < 32; ++i) {
      waux[i] = std::sqrt(weight[i]);
    }
    for (int k = 0; k < 4; ++k) {
      int nflip = 0;
      uint8_t s = 0;
      for (int i = 0; i < 8; ++i) {
        float v = float(xb[8 * k + i]);
        if (v >= 0) {
          xval[8 * k + i] = v;
        } else {
          xval[8 * k + i] = -v;
          ++nflip;
          s |= (1 << i);
        }
      }
      if (nflip % 2) {
        int imin = 0;
        float v0 = float(xb[8 * k]);
        float mn = weight[8 * k] * v0 * v0;
        for (int i = 1; i < 8; ++i) {
          float vi = float(xb[8 * k + i]);
          float ax = weight[8 * k + i] * vi * vi;
          if (ax < mn) {
            mn = ax;
            imin = i;
          }
        }
        xval[8 * k + imin] = -xval[8 * k + imin];
        s ^= (1 << imin);
      }
      block_signs[k] = s & 127;
    }
    float mx = xval[0];
    for (int i = 1; i < 32; ++i) {
      mx = std::max(mx, xval[i]);
    }
    if (mx < GROUP_MAX_EPS) {
      scales[ib] = 0;
      std::memset(L, 0, 32);
      continue;
    }
    float scale = kq_make_qp_quants<32>(
        xval, weight, kMaxQ + 1, (uint8_t*)L, GROUP_MAX_EPS);
    float eff_max = scale * kMaxQ;
    if (eff_max <= 0) {
      scales[ib] = 0;
      std::memset(L, 0, 32);
      continue;
    }
    float best = 0;
    for (int is = -6; is <= 6; ++is) {
      float id = (2 * kMaxQ - 1 + is * 0.1f) / eff_max;
      float this_scale = 1 / id;
      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < 8; ++i) {
          int l = kq_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
          Laux[8 * k + i] = int8_t(std::max(0, std::min(kMaxQ - 1, l)));
        }
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i) {
          u |= (Laux[8 * k + i] << 2 * i);
        }
        int grid_index = kmap_q2xs[u];
        if (grid_index < 0) {
          const uint16_t* neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
          grid_index = iq2_find_best_neighbour(
              neighbours,
              kgrid_q2xs,
              xval + 8 * k,
              waux + 8 * k,
              this_scale,
              Laux + 8 * k);
        }
      }
      float sumqx = 0, sumq2 = 0;
      for (int i = 0; i < 32; ++i) {
        float w = weight[i];
        float q = 2 * Laux[i] + 1;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
      }
      if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
        scale = sumqx / sumq2;
        best = scale * sumqx;
        std::memcpy(L, Laux, 32);
      }
    }
    if (scale > 0) {
      float id = 1 / scale;
      for (int k = 0; k < 4; ++k) {
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i) {
          int l = kq_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
          l = std::max(0, std::min(kMaxQ - 1, l));
          u |= (l << 2 * i);
        }
        int grid_index = kmap_q2xs[u];
        if (grid_index < 0) {
          const uint16_t* neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
          grid_index = iq2_find_best_neighbour(
              neighbours,
              kgrid_q2xs,
              xval + 8 * k,
              waux + 8 * k,
              scale,
              L + 8 * k);
        }
        const auto* pg =
            reinterpret_cast<const int8_t*>(kgrid_q2xs + grid_index);
        for (int i = 0; i < 8; ++i) {
          L[8 * k + i] = (pg[i] - 1) / 2;
        }
      }
      float sumqx = 0, sumq2 = 0;
      for (int i = 0; i < 32; ++i) {
        float w = weight[i];
        float q = 2 * L[i] + 1;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
      }
      if (sumq2 > 0) {
        scale = sumqx / sumq2;
      }
    }
    if (scale < 0) {
      // Should never happen; flip scale positive (the wire encodes an unsigned
      // scale) and flip the quant signs to match.
      scale = -scale;
      for (int k = 0; k < 4; ++k) {
        block_signs[k] = (~block_signs[k]) & 127;
      }
    }
    for (int k = 0; k < 4; ++k) {
      uint16_t u = 0;
      for (int i = 0; i < 8; ++i) {
        u |= (L[8 * k + i] << 2 * i);
      }
      int grid_index = kmap_q2xs[u];
      if (grid_index < 0) {
        throw std::runtime_error(
            "[mlx_kquant] quantize: iq2_xxs produced a point not on the grid");
      }
      q2[2 * ib + 0] |= ((uint32_t)grid_index << 8 * k);
      q2[2 * ib + 1] |= ((uint32_t)block_signs[k] << 7 * k);
    }
    scales[ib] = scale;
    max_scale = std::max(max_scale, scale);
  }

  if (!max_scale) {
    write_f16(block, 0.0f);
    std::memset(block + 2, 0, 64);
    return;
  }

  float d = max_scale / 31;
  write_f16(block, d);
  float id = 1 / d;
  for (int ib = 0; ib < 8; ++ib) {
    int l = kq_nearest_int(0.5f * (id * scales[ib] - 1));
    l = std::max(0, std::min(15, l));
    q2[2 * ib + 1] |= ((uint32_t)l << 28);
  }
  std::memcpy(block + 2, q2, 64);
}

// Port of ggml quantize_row_iq2_xs_impl (ggml-quants.c:3400) for one QK_K=256
// super-block -> block_iq2_xs ([fp16 d][u16 qs[32]][u8 scales[8]] = 74 bytes).
// 16 sixteen-weight sub-blocks (two 8-wide groups each), 4-bit scales stored
// separately. imatrix-required (quant_weights non-null, enforced at the op
// level).
template <typename T>
void quantize_iq2_xs_block(
    const T* x,
    uint8_t* block,
    const float* quant_weights) {
  constexpr int kMaxQ = 3;
  const Iq2Index& gi = iq2_index(1);
  const uint64_t* kgrid_q2xs = gi.grid.data();
  const int* kmap_q2xs = gi.kmap.data();
  const uint16_t* kneighbors_q2xs = gi.kneighbours.data();

  float scales[16]; // QK_K/16
  float weight[16];
  float xval[16];
  int8_t L[16];
  int8_t Laux[16];
  float waux[16];
  bool is_on_grid[2];
  bool is_on_grid_aux[2];
  uint8_t block_signs[2];
  uint16_t q2[32]; // 2*(QK_K/16)
  uint8_t scales_out[8]; // QK_K/32
  std::memset(q2, 0, sizeof(q2));
  std::memset(scales_out, 0, sizeof(scales_out));

  float sumx2 = 0;
  for (int i = 0; i < 256; ++i) {
    float v = float(x[i]);
    sumx2 += v * v;
  }
  float sigma2 = sumx2 / 256.0f;

  float max_scale = 0;
  for (int ib = 0; ib < 16; ++ib) { // QK_K/16
    const T* xb = x + 16 * ib;
    const float* qw = quant_weights + 16 * ib;
    for (int i = 0; i < 16; ++i) {
      float v = float(xb[i]);
      weight[i] = qw[i] * std::sqrt(sigma2 + v * v);
    }
    for (int i = 0; i < 16; ++i) {
      waux[i] = std::sqrt(weight[i]);
    }
    for (int k = 0; k < 2; ++k) {
      int nflip = 0;
      uint8_t s = 0;
      for (int i = 0; i < 8; ++i) {
        float v = float(xb[8 * k + i]);
        if (v >= 0) {
          xval[8 * k + i] = v;
        } else {
          xval[8 * k + i] = -v;
          ++nflip;
          s |= (1 << i);
        }
      }
      if (nflip % 2) {
        int imin = 0;
        float v0 = float(xb[8 * k]);
        float mn = weight[8 * k] * v0 * v0;
        for (int i = 1; i < 8; ++i) {
          float vi = float(xb[8 * k + i]);
          float ax = weight[8 * k + i] * vi * vi;
          if (ax < mn) {
            mn = ax;
            imin = i;
          }
        }
        xval[8 * k + imin] = -xval[8 * k + imin];
        s ^= (1 << imin);
      }
      block_signs[k] = s & 127;
    }
    float mx = xval[0];
    for (int i = 1; i < 16; ++i) {
      mx = std::max(mx, xval[i]);
    }
    std::memset(L, 0, 16);
    if (mx < GROUP_MAX_EPS) {
      scales[ib] = 0;
      continue;
    }
    float best = 0;
    float scale = mx / (2 * kMaxQ - 1);
    is_on_grid[0] = is_on_grid[1] = true;
    for (int is = -9; is <= 9; ++is) {
      float id = (2 * kMaxQ - 1 + is * 0.1f) / mx;
      float this_scale = 1 / id;
      for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < 8; ++i) {
          int l = kq_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
          Laux[8 * k + i] = int8_t(std::max(0, std::min(kMaxQ - 1, l)));
        }
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i) {
          u |= (Laux[8 * k + i] << 2 * i);
        }
        int grid_index = kmap_q2xs[u];
        is_on_grid_aux[k] = true;
        if (grid_index < 0) {
          is_on_grid_aux[k] = false;
          const uint16_t* neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
          grid_index = iq2_find_best_neighbour(
              neighbours,
              kgrid_q2xs,
              xval + 8 * k,
              waux + 8 * k,
              this_scale,
              Laux + 8 * k);
        }
      }
      float sumqx = 0, sumq2 = 0;
      for (int i = 0; i < 16; ++i) {
        float w = weight[i];
        float q = 2 * Laux[i] + 1;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
      }
      if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
        scale = sumqx / sumq2;
        best = scale * sumqx;
        for (int i = 0; i < 16; ++i) {
          L[i] = Laux[i];
        }
        for (int k = 0; k < 2; ++k) {
          is_on_grid[k] = is_on_grid_aux[k];
        }
      }
    }
    int n_not_ongrid = 0;
    for (int k = 0; k < 2; ++k) {
      if (!is_on_grid[k]) {
        ++n_not_ongrid;
      }
    }
    if (n_not_ongrid > 0 && scale > 0) {
      float id = 1 / scale;
      for (int k = 0; k < 2; ++k) {
        if (is_on_grid[k]) {
          continue;
        }
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i) {
          int l = kq_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
          l = std::max(0, std::min(kMaxQ - 1, l));
          u |= (l << 2 * i);
          L[8 * k + i] = int8_t(l);
        }
        int grid_index = kmap_q2xs[u];
        if (grid_index < 0) {
          const uint16_t* neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
          grid_index = iq2_find_best_neighbour(
              neighbours,
              kgrid_q2xs,
              xval + 8 * k,
              waux + 8 * k,
              scale,
              L + 8 * k);
        }
      }
      float sumqx = 0, sumq2 = 0;
      for (int i = 0; i < 16; ++i) {
        float w = weight[i];
        float q = 2 * L[i] + 1;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
      }
      if (sumq2 > 0) {
        scale = sumqx / sumq2;
      }
    }
    if (scale < 0) {
      scale = -scale;
      for (int k = 0; k < 2; ++k) {
        block_signs[k] = (~block_signs[k]) & 127;
      }
    }
    for (int k = 0; k < 2; ++k) {
      uint16_t u = 0;
      for (int i = 0; i < 8; ++i) {
        u |= (L[8 * k + i] << 2 * i);
      }
      int grid_index = kmap_q2xs[u];
      if (grid_index < 0) {
        throw std::runtime_error(
            "[mlx_kquant] quantize: iq2_xs produced a point not on the grid");
      }
      q2[2 * ib + k] = uint16_t(grid_index | (block_signs[k] << 9));
    }
    scales[ib] = scale;
    max_scale = std::max(max_scale, scale);
  }

  if (!max_scale) {
    write_f16(block, 0.0f);
    std::memset(block + 2, 0, 64 + 8); // qs + scales
    return;
  }

  float d = max_scale / 31;
  write_f16(block, d);
  float id = 1 / d;
  for (int ib = 0; ib < 16; ++ib) {
    int l = kq_nearest_int(0.5f * (id * scales[ib] - 1));
    l = std::max(0, std::min(15, l));
    if (ib % 2 == 0) {
      scales_out[ib / 2] = uint8_t(l);
    } else {
      scales_out[ib / 2] |= uint8_t(l << 4);
    }
  }
  std::memcpy(block + 2, q2, 64);
  std::memcpy(block + 66, scales_out, 8);
}

// Port of ggml quantize_row_iq2_s_impl (ggml-quants.c:5070) for one QK_K=256
// super-block -> block_iq2_s ([fp16 d][u8 qs[64]][u8 qh[8]][u8 scales[8]] = 82
// bytes). Graceful (imatrix-optional): the fallback weight is 0.25*sigma2+xb^2
// with sigma2 = 2*sumx2/QK_K. Unlike iq2_xxs/iq2_xs the signs are full 8-bit
// (no parity fixup, no &127), and the stored super-scale carries a 0.9875
// fudge.
template <typename T>
void quantize_iq2_s_block(
    const T* x,
    uint8_t* block,
    const float* quant_weights) {
  constexpr int kMaxQ = 3;
  const Iq2Index& gi = iq2_index(3);
  const uint64_t* kgrid_q2xs = gi.grid.data();
  const int* kmap_q2xs = gi.kmap.data();
  const uint16_t* kneighbors_q2xs = gi.kneighbours.data();

  float scales[16]; // QK_K/16
  float weight[16];
  float xval[16];
  int8_t L[16];
  int8_t Laux[16];
  float waux[16];
  bool is_on_grid[2];
  bool is_on_grid_aux[2];
  uint8_t block_signs[2];
  uint8_t qs_out[64]; // QK_K/4: [0..31] grid low byte, [32..63] signs
  uint8_t qh_out[8]; // QK_K/32
  uint8_t scales_out[8]; // QK_K/32
  std::memset(qs_out, 0, sizeof(qs_out));
  std::memset(qh_out, 0, sizeof(qh_out));
  std::memset(scales_out, 0, sizeof(scales_out));

  float sumx2 = 0;
  for (int i = 0; i < 256; ++i) {
    float v = float(x[i]);
    sumx2 += v * v;
  }
  float sigma2 = 2.0f * sumx2 / 256.0f;

  float max_scale = 0;
  for (int ib = 0; ib < 16; ++ib) { // QK_K/16
    const T* xb = x + 16 * ib;
    if (quant_weights) {
      const float* qw = quant_weights + 16 * ib;
      for (int i = 0; i < 16; ++i) {
        float v = float(xb[i]);
        weight[i] = qw[i] * std::sqrt(sigma2 + v * v);
      }
    } else {
      for (int i = 0; i < 16; ++i) {
        float v = float(xb[i]);
        weight[i] = 0.25f * sigma2 + v * v;
      }
    }
    for (int i = 0; i < 16; ++i) {
      waux[i] = std::sqrt(weight[i]);
    }
    for (int k = 0; k < 2; ++k) {
      uint8_t s = 0;
      for (int i = 0; i < 8; ++i) {
        float v = float(xb[8 * k + i]);
        if (v >= 0) {
          xval[8 * k + i] = v;
        } else {
          xval[8 * k + i] = -v;
          s |= (1 << i);
        }
      }
      block_signs[k] = s;
    }
    float mx = xval[0];
    for (int i = 1; i < 16; ++i) {
      mx = std::max(mx, xval[i]);
    }
    std::memset(L, 0, 16);
    if (mx < GROUP_MAX_EPS_IQ2_S) {
      scales[ib] = 0;
      continue;
    }
    float best = 0;
    float scale = mx / (2 * kMaxQ - 1);
    is_on_grid[0] = is_on_grid[1] = true;
    for (int is = -9; is <= 9; ++is) {
      float id = (2 * kMaxQ - 1 + is * 0.1f) / mx;
      float this_scale = 1 / id;
      for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < 8; ++i) {
          int l = kq_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
          Laux[8 * k + i] = int8_t(std::max(0, std::min(kMaxQ - 1, l)));
        }
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i) {
          u |= (Laux[8 * k + i] << 2 * i);
        }
        int grid_index = kmap_q2xs[u];
        is_on_grid_aux[k] = true;
        if (grid_index < 0) {
          is_on_grid_aux[k] = false;
          const uint16_t* neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
          grid_index = iq2_find_best_neighbour(
              neighbours,
              kgrid_q2xs,
              xval + 8 * k,
              waux + 8 * k,
              this_scale,
              Laux + 8 * k);
        }
      }
      float sumqx = 0, sumq2 = 0;
      for (int i = 0; i < 16; ++i) {
        float w = weight[i];
        float q = 2 * Laux[i] + 1;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
      }
      if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
        scale = sumqx / sumq2;
        best = scale * sumqx;
        for (int i = 0; i < 16; ++i) {
          L[i] = Laux[i];
        }
        for (int k = 0; k < 2; ++k) {
          is_on_grid[k] = is_on_grid_aux[k];
        }
      }
    }
    int n_not_ongrid = 0;
    for (int k = 0; k < 2; ++k) {
      if (!is_on_grid[k]) {
        ++n_not_ongrid;
      }
    }
    if (n_not_ongrid > 0 && scale > 0) {
      float id = 1 / scale;
      for (int k = 0; k < 2; ++k) {
        if (is_on_grid[k]) {
          continue;
        }
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i) {
          int l = kq_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
          l = std::max(0, std::min(kMaxQ - 1, l));
          u |= (l << 2 * i);
          L[8 * k + i] = int8_t(l);
        }
        int grid_index = kmap_q2xs[u];
        if (grid_index < 0) {
          const uint16_t* neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
          grid_index = iq2_find_best_neighbour(
              neighbours,
              kgrid_q2xs,
              xval + 8 * k,
              waux + 8 * k,
              scale,
              L + 8 * k);
        }
      }
      float sumqx = 0, sumq2 = 0;
      for (int i = 0; i < 16; ++i) {
        float w = weight[i];
        float q = 2 * L[i] + 1;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
      }
      if (sumq2 > 0) {
        scale = sumqx / sumq2;
      }
    }
    if (scale < 0) {
      scale = -scale;
      for (int k = 0; k < 2; ++k) {
        block_signs[k] = ~block_signs[k];
      }
    }
    for (int k = 0; k < 2; ++k) {
      uint16_t u = 0;
      for (int i = 0; i < 8; ++i) {
        u |= (L[8 * k + i] << 2 * i);
      }
      int grid_index = kmap_q2xs[u];
      if (grid_index < 0) {
        throw std::runtime_error(
            "[mlx_kquant] quantize: iq2_s produced a point not on the grid");
      }
      const int i8 = 2 * ib + k;
      qs_out[i8] = uint8_t(grid_index & 255);
      qh_out[i8 / 4] |= uint8_t((grid_index >> 8) << 2 * (i8 % 4));
      qs_out[32 + i8] = block_signs[k]; // QK_K/8 = 32
    }
    scales[ib] = scale;
    max_scale = std::max(max_scale, scale);
  }

  if (!max_scale) {
    write_f16(block, 0.0f);
    std::memset(block + 2, 0, 64 + 8 + 8); // qs + qh + scales
    return;
  }

  float d = max_scale / 31;
  write_f16(block, d * 0.9875f);
  float id = 1 / d;
  for (int ib = 0; ib < 16; ++ib) {
    int l = kq_nearest_int(0.5f * (id * scales[ib] - 1));
    l = std::max(0, std::min(15, l));
    if (ib % 2 == 0) {
      scales_out[ib / 2] = uint8_t(l);
    } else {
      scales_out[ib / 2] |= uint8_t(l << 4);
    }
  }
  std::memcpy(block + 2, qs_out, 64);
  std::memcpy(block + 66, qh_out, 8);
  std::memcpy(block + 74, scales_out, 8);
}

using Iq3Index = IqGridIndex<uint32_t, 4, 3>;

// iq3 family: grid 256 (iq3_xxs) or 512 (iq3_s); kmap_size 4096; nwant 2 (256)
// or 3 (512). Built once per grid on first use.
const Iq3Index& iq3_index(int grid_size) {
  static Iq3Index idx[2];
  static std::once_flag flag[2];
  int gindex = grid_size == 256 ? 0 : 1;
  std::call_once(flag[gindex], [gindex, grid_size] {
    constexpr int kmap_size = 4096;
    if (grid_size == 256) {
      build_iq_grid_index(kgrid_256, 256, kmap_size, 2, idx[0]);
    } else {
      build_iq_grid_index(kgrid_512, 512, kmap_size, 3, idx[1]);
    }
  });
  return idx[gindex];
}

// ggml iq3_find_best_neighbour (ggml-quants.c:3842): the uint32-grid, 4-value
// analogue of iq2_find_best_neighbour.
int iq3_find_best_neighbour(
    const uint16_t* neighbours,
    const uint32_t* grid,
    const float* xval,
    const float* weight,
    float scale,
    int8_t* L) {
  int num_neighbors = neighbours[0];
  float best_d2 = FLT_MAX;
  int grid_index = -1;
  for (int j = 1; j <= num_neighbors; ++j) {
    const auto* pg = reinterpret_cast<const int8_t*>(grid + neighbours[j]);
    float d2 = 0;
    for (int i = 0; i < 4; ++i) {
      float q = pg[i];
      float diff = scale * q - xval[i];
      d2 += weight[i] * diff * diff;
    }
    if (d2 < best_d2) {
      best_d2 = d2;
      grid_index = neighbours[j];
    }
  }
  const auto* pg = reinterpret_cast<const int8_t*>(grid + grid_index);
  for (int i = 0; i < 4; ++i) {
    L[i] = (pg[i] - 1) / 2;
  }
  return grid_index;
}

// Port of ggml quantize_row_iq3_xxs_impl (ggml-quants.c:3866, grid_size==256
// path) for one QK_K=256 super-block -> block_iq3_xxs ([fp16 d][u8 qs[96]] = 98
// bytes). qs = 64 grid indices then a u32[8] scales+signs table. 3-bit lattice
// (kMaxQ=8); graceful (imatrix-optional, fallback weight xb^2).
template <typename T>
void quantize_iq3_xxs_block(
    const T* x,
    uint8_t* block,
    const float* quant_weights) {
  constexpr int kMaxQ = 8;
  const Iq3Index& gi = iq3_index(256);
  const uint32_t* kgrid_q3xs = gi.grid.data();
  const int* kmap_q3xs = gi.kmap.data();
  const uint16_t* kneighbors_q3xs = gi.kneighbours.data();

  float scales[8]; // QK_K/32
  float weight[32];
  float xval[32];
  int8_t L[32];
  int8_t Laux[32];
  float waux[32];
  bool is_on_grid[8];
  bool is_on_grid_aux[8];
  uint8_t block_signs[8];
  uint8_t grid_idx[64]; // q3[8*ib+k]
  uint32_t scales_and_signs[8]; // QK_K/32
  std::memset(grid_idx, 0, sizeof(grid_idx));
  std::memset(scales_and_signs, 0, sizeof(scales_and_signs));

  float sumx2 = 0;
  for (int i = 0; i < 256; ++i) {
    float v = float(x[i]);
    sumx2 += v * v;
  }
  float sigma2 = 2.0f * sumx2 / 256.0f;

  float max_scale = 0;
  for (int ib = 0; ib < 8; ++ib) { // QK_K/32
    const T* xb = x + 32 * ib;
    if (quant_weights) {
      const float* qw = quant_weights + 32 * ib;
      for (int i = 0; i < 32; ++i) {
        float v = float(xb[i]);
        weight[i] = qw[i] * std::sqrt(sigma2 + v * v);
      }
    } else {
      for (int i = 0; i < 32; ++i) {
        float v = float(xb[i]);
        weight[i] = v * v;
      }
    }
    for (int i = 0; i < 32; ++i) {
      waux[i] = std::sqrt(weight[i]);
    }
    for (int k = 0; k < 4; ++k) {
      int nflip = 0;
      uint8_t s = 0;
      for (int i = 0; i < 8; ++i) {
        float v = float(xb[8 * k + i]);
        if (v >= 0) {
          xval[8 * k + i] = v;
        } else {
          xval[8 * k + i] = -v;
          ++nflip;
          s |= (1 << i);
        }
      }
      if (nflip % 2) {
        int imin = 0;
        float v0 = float(xb[8 * k]);
        float mn = weight[8 * k] * v0 * v0;
        for (int i = 1; i < 8; ++i) {
          float vi = float(xb[8 * k + i]);
          float ax = weight[8 * k + i] * vi * vi;
          if (ax < mn) {
            mn = ax;
            imin = i;
          }
        }
        xval[8 * k + imin] = -xval[8 * k + imin];
        s ^= (1 << imin);
      }
      block_signs[k] = s & 127;
    }
    float mx = xval[0];
    for (int i = 1; i < 32; ++i) {
      mx = std::max(mx, xval[i]);
    }
    std::memset(L, 0, 32);
    if (mx < GROUP_MAX_EPS_IQ3_XXS) {
      scales[ib] = 0;
      continue;
    }
    float best = 0;
    float scale = mx / (2 * kMaxQ - 1);
    for (int k = 0; k < 8; ++k) {
      is_on_grid[k] = true;
    }
    for (int is = -15; is <= 15; ++is) {
      float id = (2 * kMaxQ - 1 + is * 0.2f) / mx;
      float this_scale = 1 / id;
      for (int k = 0; k < 8; ++k) {
        for (int i = 0; i < 4; ++i) {
          int l = kq_nearest_int(0.5f * (id * xval[4 * k + i] - 1));
          Laux[4 * k + i] = int8_t(std::max(0, std::min(kMaxQ - 1, l)));
        }
        uint16_t u = 0;
        for (int i = 0; i < 4; ++i) {
          u |= (Laux[4 * k + i] << 3 * i);
        }
        int grid_index = kmap_q3xs[u];
        is_on_grid_aux[k] = true;
        if (grid_index < 0) {
          is_on_grid_aux[k] = false;
          const uint16_t* neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
          grid_index = iq3_find_best_neighbour(
              neighbours,
              kgrid_q3xs,
              xval + 4 * k,
              waux + 4 * k,
              this_scale,
              Laux + 4 * k);
        }
      }
      float sumqx = 0, sumq2 = 0;
      for (int i = 0; i < 32; ++i) {
        float w = weight[i];
        float q = 2 * Laux[i] + 1;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
      }
      if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
        scale = sumqx / sumq2;
        best = scale * sumqx;
        for (int i = 0; i < 32; ++i) {
          L[i] = Laux[i];
        }
        for (int k = 0; k < 8; ++k) {
          is_on_grid[k] = is_on_grid_aux[k];
        }
      }
    }
    int n_not_ongrid = 0;
    for (int k = 0; k < 8; ++k) {
      if (!is_on_grid[k]) {
        ++n_not_ongrid;
      }
    }
    if (n_not_ongrid > 0 && scale > 0) {
      float id = 1 / scale;
      for (int k = 0; k < 8; ++k) {
        if (is_on_grid[k]) {
          continue;
        }
        uint16_t u = 0;
        for (int i = 0; i < 4; ++i) {
          int l = kq_nearest_int(0.5f * (id * xval[4 * k + i] - 1));
          l = std::max(0, std::min(kMaxQ - 1, l));
          u |= (l << 3 * i);
        }
        int grid_index = kmap_q3xs[u];
        if (grid_index < 0) {
          const uint16_t* neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
          grid_index = iq3_find_best_neighbour(
              neighbours,
              kgrid_q3xs,
              xval + 4 * k,
              waux + 4 * k,
              scale,
              L + 4 * k);
        }
        const auto* pg =
            reinterpret_cast<const int8_t*>(kgrid_q3xs + grid_index);
        for (int i = 0; i < 4; ++i) {
          L[4 * k + i] = (pg[i] - 1) / 2;
        }
      }
      float sumqx = 0, sumq2 = 0;
      for (int i = 0; i < 32; ++i) {
        float w = weight[i];
        float q = 2 * L[i] + 1;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
      }
      if (sumq2 > 0) {
        scale = sumqx / sumq2;
      }
    }
    if (scale < 0) {
      scale = -scale;
      for (int k = 0; k < 4; ++k) {
        block_signs[k] = (~block_signs[k]) & 127;
      }
    }
    for (int k = 0; k < 8; ++k) {
      uint16_t u = 0;
      for (int i = 0; i < 4; ++i) {
        u |= (L[4 * k + i] << 3 * i);
      }
      int grid_index = kmap_q3xs[u];
      if (grid_index < 0) {
        throw std::runtime_error(
            "[mlx_kquant] quantize: iq3_xxs produced a point not on the grid");
      }
      grid_idx[8 * ib + k] = uint8_t(grid_index);
    }
    scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) |
        (block_signs[2] << 14) | (block_signs[3] << 21);
    scales[ib] = scale;
    max_scale = std::max(max_scale, scale);
  }

  if (!max_scale) {
    write_f16(block, 0.0f);
    std::memset(block + 2, 0, 96);
    return;
  }

  float d = max_scale / 31;
  write_f16(block, d * 1.0125f);
  float id = 1 / d;
  for (int ib = 0; ib < 8; ++ib) {
    int l = kq_nearest_int(0.5f * (id * scales[ib] - 1));
    l = std::max(0, std::min(15, l));
    scales_and_signs[ib] |= ((uint32_t)l << 28);
  }
  std::memcpy(block + 2, grid_idx, 64);
  std::memcpy(block + 66, scales_and_signs, 32);
}

// Port of ggml quantize_row_iq3_s_impl (ggml-quants.c:4097, block_size=32) for
// one QK_K=256 super-block -> block_iq3_s ([fp16 d][u8 qs[64]][u8 qh[8]][u8
// signs[32]][u8 scales[4]] = 110 bytes). 3-bit, grid 512 (9-bit index split
// qs/qh). Graceful (fallback weight xb^2); full 8-bit signs; *1.033 fudge. The
// qs/signs cursors only advance on non-zero sub-blocks while qh uses the
// absolute group index -- replicated verbatim for byte-exactness.
template <typename T>
void quantize_iq3_s_block(
    const T* x,
    uint8_t* block,
    const float* quant_weights) {
  constexpr int kMaxQ = 8;
  const Iq3Index& gi = iq3_index(512);
  const uint32_t* kgrid_q3xs = gi.grid.data();
  const int* kmap_q3xs = gi.kmap.data();
  const uint16_t* kneighbors_q3xs = gi.kneighbours.data();

  float scales[8]; // QK_K/32
  float weight[32];
  float xval[32];
  int8_t L[32];
  int8_t Laux[32];
  float waux[32];
  bool is_on_grid[8];
  bool is_on_grid_aux[8];
  uint8_t block_signs[8];
  uint8_t qs_out[64]; // QK_K/4
  uint8_t qh_out[8]; // QK_K/32
  uint8_t signs_out[32]; // QK_K/8
  uint8_t scales_out[4]; // QK_K/64
  std::memset(qs_out, 0, sizeof(qs_out));
  std::memset(qh_out, 0, sizeof(qh_out));
  std::memset(signs_out, 0, sizeof(signs_out));
  std::memset(scales_out, 0, sizeof(scales_out));

  constexpr int bs4 = 8; // block_size/4
  constexpr int bs8 = 4; // block_size/8

  float sumx2 = 0;
  for (int i = 0; i < 256; ++i) {
    float v = float(x[i]);
    sumx2 += v * v;
  }
  float sigma2 = 2.0f * sumx2 / 256.0f;

  int qs_pos = 0, signs_pos = 0;
  float max_scale = 0;
  for (int ib = 0; ib < 8; ++ib) { // QK_K/block_size
    const T* xb = x + 32 * ib;
    if (quant_weights) {
      const float* qw = quant_weights + 32 * ib;
      for (int i = 0; i < 32; ++i) {
        float v = float(xb[i]);
        weight[i] = qw[i] * std::sqrt(sigma2 + v * v);
      }
    } else {
      for (int i = 0; i < 32; ++i) {
        float v = float(xb[i]);
        weight[i] = v * v;
      }
    }
    for (int i = 0; i < 32; ++i) {
      waux[i] = std::sqrt(weight[i]);
    }
    for (int k = 0; k < bs8; ++k) {
      uint8_t s = 0;
      for (int i = 0; i < 8; ++i) {
        float v = float(xb[8 * k + i]);
        if (v >= 0) {
          xval[8 * k + i] = v;
        } else {
          xval[8 * k + i] = -v;
          s |= (1 << i);
        }
      }
      block_signs[k] = s;
    }
    float mx = xval[0];
    for (int i = 1; i < 32; ++i) {
      mx = std::max(mx, xval[i]);
    }
    std::memset(L, 0, 32);
    if (!mx) { // ggml: if (!max) -- exact-zero floor, no GROUP_MAX_EPS
      scales[ib] = 0;
      continue;
    }
    float best = 0;
    float scale = mx / (2 * kMaxQ - 1);
    for (int k = 0; k < bs4; ++k) {
      is_on_grid[k] = false;
    }
    for (int is = -9; is <= 9; ++is) {
      float id = (2 * kMaxQ - 1 + is * 0.2f) / mx;
      float this_scale = 1 / id;
      for (int k = 0; k < bs4; ++k) {
        for (int i = 0; i < 4; ++i) {
          int l = kq_nearest_int(0.5f * (id * xval[4 * k + i] - 1));
          Laux[4 * k + i] = int8_t(std::max(0, std::min(kMaxQ - 1, l)));
        }
        uint16_t u = 0;
        for (int i = 0; i < 4; ++i) {
          u |= (Laux[4 * k + i] << 3 * i);
        }
        int grid_index = kmap_q3xs[u];
        is_on_grid_aux[k] = true;
        if (grid_index < 0) {
          is_on_grid_aux[k] = false;
          const uint16_t* neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
          grid_index = iq3_find_best_neighbour(
              neighbours,
              kgrid_q3xs,
              xval + 4 * k,
              waux + 4 * k,
              this_scale,
              Laux + 4 * k);
        }
      }
      float sumqx = 0, sumq2 = 0;
      for (int i = 0; i < 32; ++i) {
        float w = weight[i];
        float q = 2 * Laux[i] + 1;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
      }
      if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
        scale = sumqx / sumq2;
        best = scale * sumqx;
        for (int i = 0; i < 32; ++i) {
          L[i] = Laux[i];
        }
        for (int k = 0; k < bs4; ++k) {
          is_on_grid[k] = is_on_grid_aux[k];
        }
      }
    }
    int n_not_ongrid = 0;
    for (int k = 0; k < bs4; ++k) {
      if (!is_on_grid[k]) {
        ++n_not_ongrid;
      }
    }
    if (n_not_ongrid > 0 && scale > 0) {
      float id = 1 / scale;
      for (int k = 0; k < bs4; ++k) {
        // ggml leaves the on-grid skip commented out: re-quantise every group.
        uint16_t u = 0;
        for (int i = 0; i < 4; ++i) {
          int l = kq_nearest_int(0.5f * (id * xval[4 * k + i] - 1));
          l = std::max(0, std::min(kMaxQ - 1, l));
          u |= (l << 3 * i);
        }
        int grid_index = kmap_q3xs[u];
        if (grid_index < 0) {
          const uint16_t* neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
          grid_index = iq3_find_best_neighbour(
              neighbours,
              kgrid_q3xs,
              xval + 4 * k,
              waux + 4 * k,
              scale,
              L + 4 * k);
        }
        const auto* pg =
            reinterpret_cast<const int8_t*>(kgrid_q3xs + grid_index);
        for (int i = 0; i < 4; ++i) {
          L[4 * k + i] = (pg[i] - 1) / 2;
        }
      }
      float sumqx = 0, sumq2 = 0;
      for (int i = 0; i < 32; ++i) {
        float w = weight[i];
        float q = 2 * L[i] + 1;
        sumqx += w * xval[i] * q;
        sumq2 += w * q * q;
      }
      if (sumq2 > 0) {
        scale = sumqx / sumq2;
      }
    }
    if (scale < 0) {
      scale = -scale;
      for (int k = 0; k < bs8; ++k) {
        block_signs[k] = ~block_signs[k];
      }
    }
    for (int k = 0; k < bs4; ++k) {
      uint16_t u = 0;
      for (int i = 0; i < 4; ++i) {
        u |= (L[4 * k + i] << 3 * i);
      }
      int grid_index = kmap_q3xs[u];
      if (grid_index < 0) {
        throw std::runtime_error(
            "[mlx_kquant] quantize: iq3_s produced a point not on the grid");
      }
      qs_out[qs_pos + k] = uint8_t(grid_index & 255);
      qh_out[(ib * bs4 + k) / 8] |=
          uint8_t((grid_index >> 8) << ((ib * bs4 + k) % 8));
    }
    qs_pos += bs4;
    for (int k = 0; k < bs8; ++k) {
      signs_out[signs_pos + k] = block_signs[k];
    }
    signs_pos += bs8;
    scales[ib] = scale;
    max_scale = std::max(max_scale, scale);
  }

  if (!max_scale) {
    write_f16(block, 0.0f);
    std::memset(block + 2, 0, 64 + 8 + 32 + 4);
    return;
  }

  float d = max_scale / 31;
  write_f16(block, d * 1.033f);
  float id = 1 / d;
  for (int ib = 0; ib < 8; ib += 2) {
    int l1 = kq_nearest_int(0.5f * (id * scales[ib + 0] - 1));
    l1 = std::max(0, std::min(15, l1));
    int l2 = kq_nearest_int(0.5f * (id * scales[ib + 1] - 1));
    l2 = std::max(0, std::min(15, l2));
    scales_out[ib / 2] = uint8_t(l1 | (l2 << 4));
  }
  std::memcpy(block + 2, qs_out, 64);
  std::memcpy(block + 66, qh_out, 8);
  std::memcpy(block + 74, signs_out, 32);
  std::memcpy(block + 106, scales_out, 4);
}

// ggml iq1_sort_helper (ggml-quants.c:4428): ascending sort of (value, index)
// pairs by the leading float. Used via std::qsort (libc, matching ggml's qsort
// tie-ordering) for the iq1 exhaustive boundary search.
int iq1_sort_helper(const void* left, const void* right) {
  const float* l = reinterpret_cast<const float*>(left);
  const float* r = reinterpret_cast<const float*>(right);
  return *l < *r ? -1 : *l > *r ? 1 : 0;
}

// ggml iq1_find_best_neighbour2 (ggml-quants.c:4371): of an off-grid point's
// candidates, pick the one minimising weighted squared error to the scaled
// value-table point xg[(grid-1)/2], with a full-grid fallback if the candidate
// list is empty. Writes (grid-1)/2 levels into L.
int iq1_find_best_neighbour2(
    const uint16_t* neighbours,
    const uint64_t* grid,
    const float* xval,
    const float* weight,
    float scale,
    const float* xg,
    int8_t* L,
    int ngrid) {
  int num_neighbors = neighbours[0];
  float best_score = FLT_MAX;
  int grid_index = -1;
  for (int j = 1; j <= num_neighbors; ++j) {
    const auto* pg = reinterpret_cast<const int8_t*>(grid + neighbours[j]);
    float d2 = 0;
    for (int i = 0; i < 8; ++i) {
      float q = xg[(pg[i] - 1) / 2];
      float w = weight[i];
      float diff = scale * q - xval[i];
      d2 += w * diff * diff;
    }
    if (d2 < best_score) {
      best_score = d2;
      grid_index = neighbours[j];
    }
  }
  if (grid_index < 0) {
    for (int i = 0; i < ngrid; ++i) {
      const auto* grid_i = reinterpret_cast<const int8_t*>(grid + i);
      float d2 = 0;
      for (int j = 0; j < 8; ++j) {
        float w = weight[j];
        float q = xg[(grid_i[j] - 1) / 2];
        float diff = scale * q - xval[i];
        d2 += w * diff * diff;
      }
      if (d2 < best_score) {
        best_score = d2;
        grid_index = i;
      }
    }
  }
  const auto* pg = reinterpret_cast<const int8_t*>(grid + grid_index);
  for (int i = 0; i < 8; ++i) {
    L[i] = (pg[i] - 1) / 2;
  }
  return grid_index;
}

// Port of ggml quantize_row_iq1_s_impl (ggml-quants.c:4436) for one QK_K=256
// super-block -> block_iq1_s ([fp16 d][u8 qs[32]][u16 qh[8]] = 50 bytes). 1-bit
// (values {-1,0,1}+delta); the per-32 scale comes from an exhaustive
// two-boundary SSD search over the sorted weights. imatrix-required; *1.125
// scale fudge.
template <typename T>
void quantize_iq1_s_block(
    const T* x,
    uint8_t* block,
    const float* quant_weights) {
  const Iq2Index& gi = iq2_index(2);
  const uint64_t* kgrid_q2xs = gi.grid.data();
  const int* kmap_q2xs = gi.kmap.data();
  const uint16_t* kneighbors_q2xs = gi.kneighbours.data();

  constexpr int block_size = 32; // IQ1S_BLOCK_SIZE
  const float x_p[3] = {-1 + IQ1S_DELTA, IQ1S_DELTA, 1 + IQ1S_DELTA};
  const float x_m[3] = {-1 - IQ1S_DELTA, -IQ1S_DELTA, 1 - IQ1S_DELTA};

  float scales[8]; // QK_K/block_size
  float weight[32];
  float xbf[32];
  float sumx[33];
  float sumw[33];
  float pairs[64]; // 2*block_size
  int8_t L[32];
  uint16_t index[4]; // block_size/8
  int8_t shifts[8];
  int* idx = reinterpret_cast<int*>(pairs + 1);

  uint8_t qs_out[32]; // QK_K/8
  uint16_t qh_out[8]; // QK_K/32
  std::memset(qs_out, 0, sizeof(qs_out));
  std::memset(qh_out, 0, sizeof(qh_out));

  float sumx2 = 0;
  for (int i = 0; i < 256; ++i) {
    float v = float(x[i]);
    sumx2 += v * v;
  }
  float sigma2 = 2.0f * sumx2 / 256.0f;

  float max_scale = 0;
  for (int ib = 0; ib < 8; ++ib) { // QK_K/block_size
    const T* xb = x + block_size * ib;
    const float* qw = quant_weights + block_size * ib;
    for (int i = 0; i < block_size; ++i) {
      xbf[i] = float(xb[i]);
    }
    for (int i = 0; i < block_size; ++i) {
      weight[i] = qw[i] * std::sqrt(sigma2 + xbf[i] * xbf[i]);
    }
    float mx = std::fabs(xbf[0]);
    for (int i = 1; i < block_size; ++i) {
      mx = std::max(mx, std::fabs(xbf[i]));
    }
    if (mx < GROUP_MAX_EPS_IQ1_S) {
      scales[ib] = 0;
      shifts[ib] = 1;
      std::memset(L, 1, block_size);
      continue;
    }
    for (int j = 0; j < block_size; ++j) {
      pairs[2 * j] = xbf[j];
      idx[2 * j] = j;
    }
    std::qsort(pairs, block_size, 2 * sizeof(float), iq1_sort_helper);
    sumx[0] = sumw[0] = 0;
    for (int j = 0; j < block_size; ++j) {
      int i = idx[2 * j];
      sumx[j + 1] = sumx[j] + weight[i] * xbf[i];
      sumw[j + 1] = sumw[j] + weight[i];
    }
    float best_score = -FLT_MAX, scale = mx;
    int besti1 = -1, besti2 = -1, best_shift = 0;
    for (int i1 = 0; i1 <= block_size; ++i1) {
      for (int i2 = i1; i2 <= block_size; ++i2) {
        float sumqx = (sumx[i1] - sumx[0]) * x_p[0] +
            (sumx[i2] - sumx[i1]) * x_p[1] +
            (sumx[block_size] - sumx[i2]) * x_p[2];
        float sumq2 = (sumw[i1] - sumw[0]) * x_p[0] * x_p[0] +
            (sumw[i2] - sumw[i1]) * x_p[1] * x_p[1] +
            (sumw[block_size] - sumw[i2]) * x_p[2] * x_p[2];
        if (sumq2 > 0 && sumqx * sumqx > best_score * sumq2) {
          scale = sumqx / sumq2;
          best_score = scale * sumqx;
          besti1 = i1;
          besti2 = i2;
          best_shift = 1;
        }
        sumqx = (sumx[i1] - sumx[0]) * x_m[0] + (sumx[i2] - sumx[i1]) * x_m[1] +
            (sumx[block_size] - sumx[i2]) * x_m[2];
        sumq2 = (sumw[i1] - sumw[0]) * x_m[0] * x_m[0] +
            (sumw[i2] - sumw[i1]) * x_m[1] * x_m[1] +
            (sumw[block_size] - sumw[i2]) * x_m[2] * x_m[2];
        if (sumq2 > 0 && sumqx * sumqx > best_score * sumq2) {
          scale = sumqx / sumq2;
          best_score = scale * sumqx;
          besti1 = i1;
          besti2 = i2;
          best_shift = -1;
        }
      }
    }
    if (besti1 < 0 || besti2 < 0 || best_shift == 0) {
      scales[ib] = 0;
      shifts[ib] = 1;
      std::memset(L, 1, block_size);
      continue;
    }
    for (int j = 0; j < besti1; ++j) {
      L[idx[2 * j]] = 0;
    }
    for (int j = besti1; j < besti2; ++j) {
      L[idx[2 * j]] = 1;
    }
    for (int j = besti2; j < block_size; ++j) {
      L[idx[2 * j]] = 2;
    }
    if (scale < 0) {
      for (int j = 0; j < block_size; ++j) {
        L[j] = 2 - L[j];
      }
      scale = -scale;
      best_shift = -best_shift;
    }
    bool all_on_grid = true;
    const float* xx = best_shift == 1 ? x_p : x_m;
    for (int k = 0; k < block_size / 8; ++k) {
      uint16_t u = 0;
      for (int j = 0; j < 8; ++j) {
        u |= (L[8 * k + j] << 2 * j);
      }
      int grid_index = kmap_q2xs[u];
      if (grid_index < 0) {
        all_on_grid = false;
        const uint16_t* neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
        grid_index = iq1_find_best_neighbour2(
            neighbours,
            kgrid_q2xs,
            xbf + 8 * k,
            weight + 8 * k,
            scale,
            xx,
            L + 8 * k,
            2048);
      }
      index[k] = uint16_t(grid_index);
    }
    if (!all_on_grid) {
      float sumqx = 0, sumq2 = 0;
      for (int k = 0; k < block_size / 8; ++k) {
        const auto* pg = reinterpret_cast<const int8_t*>(kgrid_q2xs + index[k]);
        for (int j = 0; j < 8; ++j) {
          float w = weight[8 * k + j];
          float q = xx[(pg[j] - 1) / 2];
          sumqx += w * q * xbf[8 * k + j];
          sumq2 += w * q * q;
        }
      }
      if (sumqx > 0 && sumq2 > 0) {
        scale = sumqx / sumq2;
      }
    }
    uint16_t h = 0;
    for (int k = 0; k < block_size / 8; ++k) {
      qs_out[(block_size / 8) * ib + k] = uint8_t(index[k] & 255);
      h |= (index[k] >> 8) << 3 * k;
    }
    qh_out[ib] = h;
    scales[ib] = scale;
    shifts[ib] = int8_t(best_shift);
    max_scale = std::max(max_scale, scale);
  }

  if (!max_scale) {
    write_f16(block, 0.0f);
    std::memset(block + 2, 0, 32 + 16); // qs + qh
    return;
  }

  float d = max_scale / 15;
  write_f16(block, d * 1.125f);
  float id = 1 / d;
  for (int ib = 0; ib < 8; ++ib) {
    int l = kq_nearest_int(0.5f * (id * scales[ib] - 1));
    l = std::max(0, std::min(7, l));
    if (shifts[ib] == -1) {
      l |= 8;
    }
    qh_out[ib] |= (l << 12);
  }
  std::memcpy(block + 2, qs_out, 32);
  std::memcpy(block + 34, qh_out, 16);
}

// Port of ggml quantize_row_iq1_m_impl (ggml-quants.c:4620) for one QK_K=256
// super-block -> block_iq1_m ([u8 qs[32]][u8 qh[16]][u8 scales[8]] = 56 bytes,
// no top-level d -- the fp16 super-scale is split across the scales' high
// nibbles). 16-weight blocks; a 4-way (+/-, +/-) sign search per block;
// graceful (fallback weight xb^2); *1.1125 scale fudge.
template <typename T>
void quantize_iq1_m_block(
    const T* x,
    uint8_t* block,
    const float* quant_weights) {
  const Iq2Index& gi = iq2_index(2);
  const uint64_t* kgrid_q2xs = gi.grid.data();
  const int* kmap_q2xs = gi.kmap.data();
  const uint16_t* kneighbors_q2xs = gi.kneighbours.data();

  constexpr int block_size = 16; // IQ1M_BLOCK_SIZE
  const float x_p[3] = {-1 + IQ1S_DELTA, IQ1S_DELTA, 1 + IQ1S_DELTA};
  const float x_m[3] = {-1 - IQ1S_DELTA, -IQ1S_DELTA, 1 - IQ1S_DELTA};
  const uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};

  float scales[16]; // QK_K/block_size
  float weight[16];
  float xbf[16];
  float pairs[32]; // 2*block_size
  int8_t L[16];
  uint16_t index[2]; // block_size/8
  int8_t shifts[16];
  int* idx = reinterpret_cast<int*>(pairs + 1);
  float sumqx[4], sumq2[4];

  uint8_t qs_out[32]; // QK_K/8
  uint8_t qh_out[16]; // QK_K/16
  uint16_t sc[4]; // QK_K/32 bytes, viewed as u16[4]
  std::memset(qs_out, 0, sizeof(qs_out));
  std::memset(qh_out, 0, sizeof(qh_out));
  std::memset(sc, 0, sizeof(sc));

  float sumx2 = 0;
  for (int i = 0; i < 256; ++i) {
    float v = float(x[i]);
    sumx2 += v * v;
  }
  float sigma2 = 2.0f * sumx2 / 256.0f;

  float max_scale = 0;
  for (int ib = 0; ib < 16; ++ib) { // QK_K/block_size
    const T* xb = x + block_size * ib;
    for (int i = 0; i < block_size; ++i) {
      xbf[i] = float(xb[i]);
    }
    if (quant_weights) {
      const float* qw = quant_weights + block_size * ib;
      for (int i = 0; i < block_size; ++i) {
        weight[i] = qw[i] * std::sqrt(sigma2 + xbf[i] * xbf[i]);
      }
    } else {
      for (int i = 0; i < block_size; ++i) {
        weight[i] = xbf[i] * xbf[i];
      }
    }
    float mx = std::fabs(xbf[0]);
    for (int i = 1; i < block_size; ++i) {
      mx = std::max(mx, std::fabs(xbf[i]));
    }
    if (mx < GROUP_MAX_EPS_IQ1_M) {
      scales[ib] = 0;
      shifts[ib] = 0;
      std::memset(L, 1, block_size);
      continue;
    }
    for (int j = 0; j < block_size; ++j) {
      pairs[2 * j] = xbf[j];
      idx[2 * j] = j;
    }
    std::qsort(pairs, block_size, 2 * sizeof(float), iq1_sort_helper);
    float best_score = -FLT_MAX, scale = mx;
    int besti1 = -1, besti2 = -1, best_k = -1;
    for (int i1 = 0; i1 <= block_size; ++i1) {
      for (int i2 = i1; i2 <= block_size; ++i2) {
        std::memset(sumqx, 0, 4 * sizeof(float));
        std::memset(sumq2, 0, 4 * sizeof(float));
        for (int j = 0; j < i1; ++j) {
          int i = idx[2 * j];
          if (i < block_size / 2) {
            sumqx[0] += weight[i] * x_p[0] * xbf[i];
            sumqx[1] += weight[i] * x_p[0] * xbf[i];
            sumqx[2] += weight[i] * x_m[0] * xbf[i];
            sumqx[3] += weight[i] * x_m[0] * xbf[i];
            sumq2[0] += weight[i] * x_p[0] * x_p[0];
            sumq2[1] += weight[i] * x_p[0] * x_p[0];
            sumq2[2] += weight[i] * x_m[0] * x_m[0];
            sumq2[3] += weight[i] * x_m[0] * x_m[0];
          } else {
            sumqx[0] += weight[i] * x_p[0] * xbf[i];
            sumqx[2] += weight[i] * x_p[0] * xbf[i];
            sumqx[1] += weight[i] * x_m[0] * xbf[i];
            sumqx[3] += weight[i] * x_m[0] * xbf[i];
            sumq2[0] += weight[i] * x_p[0] * x_p[0];
            sumq2[2] += weight[i] * x_p[0] * x_p[0];
            sumq2[1] += weight[i] * x_m[0] * x_m[0];
            sumq2[3] += weight[i] * x_m[0] * x_m[0];
          }
        }
        for (int j = i1; j < i2; ++j) {
          int i = idx[2 * j];
          if (i < block_size / 2) {
            sumqx[0] += weight[i] * x_p[1] * xbf[i];
            sumqx[1] += weight[i] * x_p[1] * xbf[i];
            sumqx[2] += weight[i] * x_m[1] * xbf[i];
            sumqx[3] += weight[i] * x_m[1] * xbf[i];
            sumq2[0] += weight[i] * x_p[1] * x_p[1];
            sumq2[1] += weight[i] * x_p[1] * x_p[1];
            sumq2[2] += weight[i] * x_m[1] * x_m[1];
            sumq2[3] += weight[i] * x_m[1] * x_m[1];
          } else {
            sumqx[0] += weight[i] * x_p[1] * xbf[i];
            sumqx[2] += weight[i] * x_p[1] * xbf[i];
            sumqx[1] += weight[i] * x_m[1] * xbf[i];
            sumqx[3] += weight[i] * x_m[1] * xbf[i];
            sumq2[0] += weight[i] * x_p[1] * x_p[1];
            sumq2[2] += weight[i] * x_p[1] * x_p[1];
            sumq2[1] += weight[i] * x_m[1] * x_m[1];
            sumq2[3] += weight[i] * x_m[1] * x_m[1];
          }
        }
        for (int j = i2; j < block_size; ++j) {
          int i = idx[2 * j];
          if (i < block_size / 2) {
            sumqx[0] += weight[i] * x_p[2] * xbf[i];
            sumqx[1] += weight[i] * x_p[2] * xbf[i];
            sumqx[2] += weight[i] * x_m[2] * xbf[i];
            sumqx[3] += weight[i] * x_m[2] * xbf[i];
            sumq2[0] += weight[i] * x_p[2] * x_p[2];
            sumq2[1] += weight[i] * x_p[2] * x_p[2];
            sumq2[2] += weight[i] * x_m[2] * x_m[2];
            sumq2[3] += weight[i] * x_m[2] * x_m[2];
          } else {
            sumqx[0] += weight[i] * x_p[2] * xbf[i];
            sumqx[2] += weight[i] * x_p[2] * xbf[i];
            sumqx[1] += weight[i] * x_m[2] * xbf[i];
            sumqx[3] += weight[i] * x_m[2] * xbf[i];
            sumq2[0] += weight[i] * x_p[2] * x_p[2];
            sumq2[2] += weight[i] * x_p[2] * x_p[2];
            sumq2[1] += weight[i] * x_m[2] * x_m[2];
            sumq2[3] += weight[i] * x_m[2] * x_m[2];
          }
        }
        for (int k = 0; k < 4; ++k) {
          if (sumq2[k] > 0 && sumqx[k] * sumqx[k] > best_score * sumq2[k]) {
            scale = sumqx[k] / sumq2[k];
            best_score = scale * sumqx[k];
            besti1 = i1;
            besti2 = i2;
            best_k = k;
          }
        }
      }
    }
    if (besti1 < 0 || besti2 < 0 || best_k < 0) {
      scales[ib] = 0;
      shifts[ib] = 0;
      std::memset(L, 1, block_size);
      continue;
    }
    for (int j = 0; j < besti1; ++j) {
      L[idx[2 * j]] = 0;
    }
    for (int j = besti1; j < besti2; ++j) {
      L[idx[2 * j]] = 1;
    }
    for (int j = besti2; j < block_size; ++j) {
      L[idx[2 * j]] = 2;
    }
    if (scale < 0) {
      for (int j = 0; j < block_size; ++j) {
        L[j] = 2 - L[j];
      }
      scale = -scale;
      best_k = best_k == 0 ? 3 : best_k == 1 ? 2 : best_k == 2 ? 1 : 0;
    }
    bool all_on_grid = true;
    const float* xx;
    for (int k = 0; k < block_size / 8; ++k) {
      if (k == 0) {
        xx = best_k < 2 ? x_p : x_m;
      } else {
        xx = best_k % 2 == 0 ? x_p : x_m;
      }
      uint16_t u = 0;
      for (int j = 0; j < 8; ++j) {
        u |= (L[8 * k + j] << 2 * j);
      }
      int grid_index = kmap_q2xs[u];
      if (grid_index < 0) {
        all_on_grid = false;
        const uint16_t* neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
        grid_index = iq1_find_best_neighbour2(
            neighbours,
            kgrid_q2xs,
            xbf + 8 * k,
            weight + 8 * k,
            scale,
            xx,
            L + 8 * k,
            2048);
      }
      index[k] = uint16_t(grid_index);
    }
    if (!all_on_grid) {
      float sumqx_f = 0, sumq2_f = 0;
      for (int k = 0; k < block_size / 8; ++k) {
        if (k == 0) {
          xx = best_k < 2 ? x_p : x_m;
        } else {
          xx = best_k % 2 == 0 ? x_p : x_m;
        }
        const auto* pg = reinterpret_cast<const int8_t*>(kgrid_q2xs + index[k]);
        for (int j = 0; j < 8; ++j) {
          float w = weight[8 * k + j];
          float q = xx[(pg[j] - 1) / 2];
          sumqx_f += w * q * xbf[8 * k + j];
          sumq2_f += w * q * q;
        }
      }
      if (sumqx_f > 0 && sumq2_f > 0) {
        scale = sumqx_f / sumq2_f;
      }
    }
    qs_out[2 * ib + 0] = uint8_t(index[0] & 255);
    qs_out[2 * ib + 1] = uint8_t(index[1] & 255);
    qh_out[ib] = uint8_t((index[0] >> 8) | ((index[1] >> 8) << 4));
    scales[ib] = scale;
    shifts[ib] = int8_t(best_k);
    max_scale = std::max(max_scale, scale);
  }

  if (!max_scale) {
    std::memset(block, 0, 32 + 16 + 8); // qs + qh + scales
    return;
  }

  float d = max_scale / 15;
  float id = 1 / d;
  float sumqx_f = 0, sumq2_f = 0;
  for (int ib = 0; ib < 16; ++ib) {
    int l = kq_nearest_int(0.5f * (id * scales[ib] - 1));
    l = std::max(0, std::min(7, l));
    sc[ib / 4] |= (l << 3 * (ib % 4));
    qh_out[ib] |= masks[shifts[ib]];
    const T* xb = x + block_size * ib;
    for (int i = 0; i < block_size; ++i) {
      xbf[i] = float(xb[i]);
    }
    if (quant_weights) {
      const float* qw = quant_weights + block_size * ib;
      for (int i = 0; i < block_size; ++i) {
        weight[i] = qw[i] * std::sqrt(sigma2 + xbf[i] * xbf[i]);
      }
    } else {
      for (int i = 0; i < block_size; ++i) {
        weight[i] = xbf[i] * xbf[i];
      }
    }
    const float* xx;
    for (int k = 0; k < block_size / 8; ++k) {
      if (k == 0) {
        xx = shifts[ib] < 2 ? x_p : x_m;
      } else {
        xx = shifts[ib] % 2 == 0 ? x_p : x_m;
      }
      const auto* pg = reinterpret_cast<const int8_t*>(
          kgrid_q2xs + qs_out[2 * ib + k] +
          ((qh_out[ib] << (8 - 4 * k)) & 0x700));
      for (int j = 0; j < 8; ++j) {
        float w = weight[8 * k + j];
        float q = xx[(pg[j] - 1) / 2] * (2 * l + 1);
        sumqx_f += w * q * xbf[8 * k + j];
        sumq2_f += w * q * q;
      }
    }
  }
  if (sumq2_f > 0) {
    d = sumqx_f / sumq2_f;
  }
  _Float16 hf = static_cast<_Float16>(d * 1.1125f);
  uint16_t su16;
  std::memcpy(&su16, &hf, sizeof(su16));
  sc[0] |= ((su16 & 0x000f) << 12);
  sc[1] |= ((su16 & 0x00f0) << 8);
  sc[2] |= ((su16 & 0x0f00) << 4);
  sc[3] |= ((su16 & 0xf000) << 0);
  std::memcpy(block, qs_out, 32);
  std::memcpy(block + 32, qh_out, 16);
  std::memcpy(block + 48, sc, 8);
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
  } else if (kquant_type == "iq2_xxs") {
    constexpr int wpb = 256, bpb = 66;
    std::size_t nblocks = num_weights / wpb;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const float* qw = imatrix ? imatrix + ((b * wpb) % K) : nullptr;
      quantize_iq2_xxs_block<T>(w + b * wpb, out + b * bpb, qw);
    }
  } else if (kquant_type == "iq2_xs") {
    constexpr int wpb = 256, bpb = 74;
    std::size_t nblocks = num_weights / wpb;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const float* qw = imatrix ? imatrix + ((b * wpb) % K) : nullptr;
      quantize_iq2_xs_block<T>(w + b * wpb, out + b * bpb, qw);
    }
  } else if (kquant_type == "iq2_s") {
    constexpr int wpb = 256, bpb = 82;
    std::size_t nblocks = num_weights / wpb;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const float* qw = imatrix ? imatrix + ((b * wpb) % K) : nullptr;
      quantize_iq2_s_block<T>(w + b * wpb, out + b * bpb, qw);
    }
  } else if (kquant_type == "iq3_xxs") {
    constexpr int wpb = 256, bpb = 98;
    std::size_t nblocks = num_weights / wpb;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const float* qw = imatrix ? imatrix + ((b * wpb) % K) : nullptr;
      quantize_iq3_xxs_block<T>(w + b * wpb, out + b * bpb, qw);
    }
  } else if (kquant_type == "iq3_s") {
    constexpr int wpb = 256, bpb = 110;
    std::size_t nblocks = num_weights / wpb;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const float* qw = imatrix ? imatrix + ((b * wpb) % K) : nullptr;
      quantize_iq3_s_block<T>(w + b * wpb, out + b * bpb, qw);
    }
  } else if (kquant_type == "iq1_s") {
    constexpr int wpb = 256, bpb = 50;
    std::size_t nblocks = num_weights / wpb;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const float* qw = imatrix ? imatrix + ((b * wpb) % K) : nullptr;
      quantize_iq1_s_block<T>(w + b * wpb, out + b * bpb, qw);
    }
  } else if (kquant_type == "iq1_m") {
    constexpr int wpb = 256, bpb = 56;
    std::size_t nblocks = num_weights / wpb;
    for (std::size_t b = 0; b < nblocks; ++b) {
      const float* qw = imatrix ? imatrix + ((b * wpb) % K) : nullptr;
      quantize_iq1_m_block<T>(w + b * wpb, out + b * bpb, qw);
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
