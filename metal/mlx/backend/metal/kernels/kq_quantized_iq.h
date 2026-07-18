// IQ (importance-quantization) GGUF block formats: load geometry, dequant,
// qmv and qmm kernels. Decode math derives from ggml (llama.cpp, MIT); the
// qmv/qmm kernel structure is adapted from MLX (MIT). See mlx_kquant/licenses/.
// Included from kq_quantized.h after the shared machinery; not standalone.
// IQ4_XS: 136 bytes/256 weights. [fp16 d][u16 scales_h][u8
// scales_l[4]][qs[128]]. Symmetric (no min), signed 6-bit per-32 scale (ls -
// 32), kvalues LUT.

MLX_MTL_CONST int KQ_IQ4_XS_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_IQ4_XS_BLOCK_BYTES = 136;
MLX_MTL_CONST int KQ_IQ4_XS_SCALESL_OFFSET = 4;
MLX_MTL_CONST int KQ_IQ4_XS_QS_OFFSET = 8;

template <typename T>
METAL_FUNC void kq_iq4_xs_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_IQ4_XS_SUPERBLOCK;
  const int within = gid - sb_id * KQ_IQ4_XS_SUPERBLOCK;
  const int ib = within / 32;
  const int j = within % 32;
  const device uint8_t* sb =
      w + static_cast<int64_t>(sb_id) * KQ_IQ4_XS_BLOCK_BYTES;
  const float d = float(*(const device half*)sb);
  const uint16_t scales_h = uint16_t(sb[2]) | (uint16_t(sb[3]) << 8);
  const device uint8_t* scales_l = sb + KQ_IQ4_XS_SCALESL_OFFSET;
  const int ls = ((scales_l[ib / 2] >> (4 * (ib & 1))) & 0xf) |
      (((scales_h >> (2 * ib)) & 3) << 4);
  const float dl = d * float(ls - 32);
  const device uint8_t* qs = sb + KQ_IQ4_XS_QS_OFFSET + ib * 16;
  const int nib = (j < 16) ? (int(qs[j]) & 0xf) : (int(qs[j - 16]) >> 4);
  out[gid] = T(dl * float(kvalues_iq4nl[nib]));
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_iq4_xs_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_IQ4_XS_SUPERBLOCK, "IQ4_XS kernel requires gs=256");
  static_assert(bits == 4, "IQ4_XS kernel requires bits=4");
  kq_iq4_xs_dequantize_impl<T>(w, out, num_weights, gid);
}

// iq4_xs flat-with-M verify mat-vec: kq_mv_ext_impl (see kq_quantized.h) +
// chunk dequant. Mirrors kq_iq4_xs_dequantize_impl over 16 contiguous weights
// (natural order [il*16, il*16+16)). chunk il -> ib32 = il/2, half = il&1
// (low/high nibbles of qs[0..15] of that ib32). signed 6-bit per-32 scale (ls -
// 32), LUT.
inline void kq_iq4_xs_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const int ib = il / 2; // ib32 within the super-block
  const int shift = (il & 1) ? 4 : 0; // low or high nibble
  const float d = float(*(const device half*)block);
  const uint16_t scales_h = uint16_t(block[2]) | (uint16_t(block[3]) << 8);
  const device uint8_t* scales_l = block + KQ_IQ4_XS_SCALESL_OFFSET;
  const int ls = ((scales_l[ib / 2] >> (4 * (ib & 1))) & 0xf) |
      (((scales_h >> (2 * ib)) & 3) << 4);
  const float dl = d * float(ls - 32);
  const device uint8_t* qs = block + KQ_IQ4_XS_QS_OFFSET + ib * 16;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int nib = (int(qs[i]) >> shift) & 0xf;
    reg[i / 4][i % 4] = dl * float(kvalues_iq4nl[nib]);
  }
}

struct KqIq4_xsExt {
  MLX_MTL_CONST int superblock = KQ_IQ4_XS_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_IQ4_XS_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_iq4_xs_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_iq4_xs_mv_ext(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size, // K
    const constant int& out_vec_size, // N
    const constant int& /* vm */, // == r1ptg
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  kq_mv_ext_impl<T, KqIq4_xsExt, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

// IQ3_XXS: 98 bytes/256 weights. [fp16 d][qs[64] grid idx][gas[32]]. Each ib32
// gas u32 holds the 4 sign indices (low 28 bits, via ksigns LUT) + a top-4-bit
// scale. The *0.5 is folded into the block scale on this (dequant) path.

MLX_MTL_CONST int KQ_IQ3_XXS_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_IQ3_XXS_BLOCK_BYTES = 98;
MLX_MTL_CONST int KQ_IQ3_XXS_QS_OFFSET = 2;
MLX_MTL_CONST int KQ_IQ3_XXS_GAS_OFFSET = 66;

template <typename T>
METAL_FUNC void kq_iq3_xxs_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_IQ3_XXS_SUPERBLOCK;
  const int within = gid - sb_id * KQ_IQ3_XXS_SUPERBLOCK;
  const int ib32 = within / 32;
  const int p = within % 32;
  const int l = p / 8;
  const int sub = p % 8;
  const device uint8_t* sb =
      w + static_cast<int64_t>(sb_id) * KQ_IQ3_XXS_BLOCK_BYTES;
  const float d = float(*(const device half*)sb);
  const device uint8_t* qs = sb + KQ_IQ3_XXS_QS_OFFSET + ib32 * 8;
  const device uint8_t* gas = sb + KQ_IQ3_XXS_GAS_OFFSET + ib32 * 4;
  const uint32_t aux32 = uint32_t(gas[0]) | (uint32_t(gas[1]) << 8) |
      (uint32_t(gas[2]) << 16) | (uint32_t(gas[3]) << 24);
  const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
  const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
  const int qi = (sub < 4) ? int(qs[2 * l]) : int(qs[2 * l + 1]);
  const int bytej = (sub < 4) ? sub : (sub - 4);
  const uint8_t gb = (iq3xxs_grid[qi] >> (8 * bytej)) & 0xff;
  const float sgn = (signs & kmask_iq2xs[sub]) ? -1.0f : 1.0f;
  out[gid] = T(db * float(gb) * sgn);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_iq3_xxs_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_IQ3_XXS_SUPERBLOCK, "IQ3_XXS kernel requires gs=256");
  static_assert(bits == 3, "IQ3_XXS kernel requires bits=3");
  kq_iq3_xxs_dequantize_impl<T>(w, out, num_weights, gid);
}

// iq3_xxs flat-with-M verify mat-vec: kq_mv_ext_impl + chunk dequant. Mirrors
// kq_iq3_xxs_dequantize_impl over 16 contiguous weights (natural order
// [il*16, il*16+16)). chunk il -> ib32 = il/2; the 16 weights span the two
// grid-groups l = (il&1)*2 + i/8, byte sub = i%8.
inline void kq_iq3_xxs_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const int ib32 = il / 2;
  const int lbase = (il & 1) * 2;
  const float d = float(*(const device half*)block);
  const device uint8_t* qs = block + KQ_IQ3_XXS_QS_OFFSET + ib32 * 8;
  const device uint8_t* gas = block + KQ_IQ3_XXS_GAS_OFFSET + ib32 * 4;
  const uint32_t aux32 = uint32_t(gas[0]) | (uint32_t(gas[1]) << 8) |
      (uint32_t(gas[2]) << 16) | (uint32_t(gas[3]) << 24);
  const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int l = lbase + i / 8;
    const int sub = i % 8;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
    const int qi = (sub < 4) ? int(qs[2 * l]) : int(qs[2 * l + 1]);
    const int bytej = (sub < 4) ? sub : (sub - 4);
    const uint8_t gb = (iq3xxs_grid[qi] >> (8 * bytej)) & 0xff;
    const float sgn = (signs & kmask_iq2xs[sub]) ? -1.0f : 1.0f;
    reg[i / 4][i % 4] = db * float(gb) * sgn;
  }
}

struct KqIq3_xxsExt {
  MLX_MTL_CONST int superblock = KQ_IQ3_XXS_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_IQ3_XXS_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_iq3_xxs_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_iq3_xxs_mv_ext(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size, // K
    const constant int& out_vec_size, // N
    const constant int& /* vm */, // == r1ptg
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  kq_mv_ext_impl<T, KqIq3_xxsExt, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

// IQ3_S: 110 bytes/256 weights. [fp16 d][qs[64]][qh[8]][signs[32]][scales[4]].
// 9-bit grid index (qs low 8 + qh bit 8), signs from block bytes, 4-bit scale
// -> 2*s+1. No min.

MLX_MTL_CONST int KQ_IQ3_S_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_IQ3_S_BLOCK_BYTES = 110;
MLX_MTL_CONST int KQ_IQ3_S_QS_OFFSET = 2;
MLX_MTL_CONST int KQ_IQ3_S_QH_OFFSET = 66;
MLX_MTL_CONST int KQ_IQ3_S_SIGNS_OFFSET = 74;
MLX_MTL_CONST int KQ_IQ3_S_SCALES_OFFSET = 106;

template <typename T>
METAL_FUNC void kq_iq3_s_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_IQ3_S_SUPERBLOCK;
  const int within = gid - sb_id * KQ_IQ3_S_SUPERBLOCK;
  const int s = within / 32;
  const int p = within % 32;
  const int l = p / 8;
  const int sub = p % 8;
  const device uint8_t* sb =
      w + static_cast<int64_t>(sb_id) * KQ_IQ3_S_BLOCK_BYTES;
  const float d = float(*(const device half*)sb);
  const device uint8_t* qs = sb + KQ_IQ3_S_QS_OFFSET + s * 8;
  const device uint8_t* qh = sb + KQ_IQ3_S_QH_OFFSET;
  const device uint8_t* signs = sb + KQ_IQ3_S_SIGNS_OFFSET + s * 4;
  const device uint8_t* scales = sb + KQ_IQ3_S_SCALES_OFFSET;
  const int sc_nib = (scales[s / 2] >> (4 * (s & 1))) & 0xf;
  const float db = d * float(1 + 2 * sc_nib);
  const int qhbit = (sub < 4) ? ((int(qh[s]) << (8 - 2 * l)) & 256)
                              : ((int(qh[s]) << (7 - 2 * l)) & 256);
  const int qi =
      (sub < 4) ? (int(qs[2 * l]) | qhbit) : (int(qs[2 * l + 1]) | qhbit);
  const int bytej = (sub < 4) ? sub : (sub - 4);
  const uint8_t gb = (iq3s_grid[qi] >> (8 * bytej)) & 0xff;
  const float sgn = (signs[l] & kmask_iq2xs[sub]) ? -1.0f : 1.0f;
  out[gid] = T(db * float(gb) * sgn);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_iq3_s_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_IQ3_S_SUPERBLOCK, "IQ3_S kernel requires gs=256");
  static_assert(bits == 3, "IQ3_S kernel requires bits=3");
  kq_iq3_s_dequantize_impl<T>(w, out, num_weights, gid);
}

// iq3_s flat-with-M verify mat-vec: kq_mv_ext_impl + chunk dequant. Mirrors
// kq_iq3_s_dequantize_impl over 16 contiguous weights (natural order
// [il*16, il*16+16)). chunk il -> s = il/2; l = (il&1)*2 + i/8, byte sub = i%8.
// 9-bit grid index (qs low 8 + qh bit 8), signs from block bytes, 4-bit scale.
inline void kq_iq3_s_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const int s = il / 2;
  const int lbase = (il & 1) * 2;
  const float d = float(*(const device half*)block);
  const device uint8_t* qs = block + KQ_IQ3_S_QS_OFFSET + s * 8;
  const device uint8_t* qh = block + KQ_IQ3_S_QH_OFFSET;
  const device uint8_t* signs = block + KQ_IQ3_S_SIGNS_OFFSET + s * 4;
  const device uint8_t* scales = block + KQ_IQ3_S_SCALES_OFFSET;
  const int sc_nib = (scales[s / 2] >> (4 * (s & 1))) & 0xf;
  const float db = d * float(1 + 2 * sc_nib);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int l = lbase + i / 8;
    const int sub = i % 8;
    const int qhbit = (sub < 4) ? ((int(qh[s]) << (8 - 2 * l)) & 256)
                                : ((int(qh[s]) << (7 - 2 * l)) & 256);
    const int qi =
        (sub < 4) ? (int(qs[2 * l]) | qhbit) : (int(qs[2 * l + 1]) | qhbit);
    const int bytej = (sub < 4) ? sub : (sub - 4);
    const uint8_t gb = (iq3s_grid[qi] >> (8 * bytej)) & 0xff;
    const float sgn = (signs[l] & kmask_iq2xs[sub]) ? -1.0f : 1.0f;
    reg[i / 4][i % 4] = db * float(gb) * sgn;
  }
}

struct KqIq3_sExt {
  MLX_MTL_CONST int superblock = KQ_IQ3_S_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_IQ3_S_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_iq3_s_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_iq3_s_mv_ext(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size, // K
    const constant int& out_vec_size, // N
    const constant int& /* vm */, // == r1ptg
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  kq_mv_ext_impl<T, KqIq3_sExt, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

// IQ2_XXS: 66 bytes/256 weights. [fp16 d][qs[32] u16]. Each ib32 (8 bytes) = 4
// grid-index bytes + a u32 with 4 7-bit sign indices (low 28, via ksigns LUT) +
// a top-4-bit scale. The grid is uint64 (one entry = the full 8-weight group),
// so byte j of the group is (grid >> 8*j). db folds (0.5+scale)*0.25 here.

MLX_MTL_CONST int KQ_IQ2_XXS_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_IQ2_XXS_BLOCK_BYTES = 66;
MLX_MTL_CONST int KQ_IQ2_XXS_QS_OFFSET = 2;

template <typename T>
METAL_FUNC void kq_iq2_xxs_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_IQ2_XXS_SUPERBLOCK;
  const int within = gid - sb_id * KQ_IQ2_XXS_SUPERBLOCK;
  const int ib32 = within / 32;
  const int p = within % 32;
  const int l = p / 8;
  const int j = p % 8;
  const device uint8_t* sb =
      w + static_cast<int64_t>(sb_id) * KQ_IQ2_XXS_BLOCK_BYTES;
  const float d = float(*(const device half*)sb);
  const device uint8_t* qs = sb + KQ_IQ2_XXS_QS_OFFSET + ib32 * 8;
  const uint32_t signbits = uint32_t(qs[4]) | (uint32_t(qs[5]) << 8) |
      (uint32_t(qs[6]) << 16) | (uint32_t(qs[7]) << 24);
  const float db = d * (0.5f + float(signbits >> 28)) * 0.25f;
  const uint8_t signs = ksigns_iq2xs[(signbits >> (7 * l)) & 127];
  const uint8_t gb = (iq2xxs_grid[qs[l]] >> (8 * j)) & 0xff;
  const float sgn = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
  out[gid] = T(db * float(gb) * sgn);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_iq2_xxs_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_IQ2_XXS_SUPERBLOCK, "IQ2_XXS kernel requires gs=256");
  static_assert(bits == 2, "IQ2_XXS kernel requires bits=2");
  kq_iq2_xxs_dequantize_impl<T>(w, out, num_weights, gid);
}

// iq2_xxs flat-with-M verify mat-vec: kq_mv_ext_impl + chunk dequant. Mirrors
// kq_iq2_xxs_dequantize_impl over 16 contiguous weights (natural order
// [il*16, il*16+16)). chunk il -> ib32 = il/2; l = (il&1)*2 + i/8, byte j =
// i%8.
inline void kq_iq2_xxs_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const int ib32 = il / 2;
  const int lbase = (il & 1) * 2;
  const float d = float(*(const device half*)block);
  const device uint8_t* qs = block + KQ_IQ2_XXS_QS_OFFSET + ib32 * 8;
  const uint32_t signbits = uint32_t(qs[4]) | (uint32_t(qs[5]) << 8) |
      (uint32_t(qs[6]) << 16) | (uint32_t(qs[7]) << 24);
  const float db = d * (0.5f + float(signbits >> 28)) * 0.25f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int l = lbase + i / 8;
    const int j = i % 8;
    const uint8_t signs = ksigns_iq2xs[(signbits >> (7 * l)) & 127];
    const uint8_t gb = (iq2xxs_grid[qs[l]] >> (8 * j)) & 0xff;
    const float sgn = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
    reg[i / 4][i % 4] = db * float(gb) * sgn;
  }
}

struct KqIq2_xxsExt {
  MLX_MTL_CONST int superblock = KQ_IQ2_XXS_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_IQ2_XXS_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_iq2_xxs_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_iq2_xxs_mv_ext(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size, // K
    const constant int& out_vec_size, // N
    const constant int& /* vm */, // == r1ptg
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  kq_mv_ext_impl<T, KqIq2_xxsExt, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

// IQ2_XS: 74 bytes/256 weights. [fp16 d][qs[32] u16][scales[8]]. Each qs u16 =
// 9-bit grid idx + 7-bit sign idx (via ksigns LUT). scales: two 4-bit per byte,
// low nibble for the first half of the ib32, high nibble for the second; db
// folds (0.5+s)*0.25. uint64 grid.

MLX_MTL_CONST int KQ_IQ2_XS_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_IQ2_XS_BLOCK_BYTES = 74;
MLX_MTL_CONST int KQ_IQ2_XS_QS_OFFSET = 2;
MLX_MTL_CONST int KQ_IQ2_XS_SCALES_OFFSET = 66;

template <typename T>
METAL_FUNC void kq_iq2_xs_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_IQ2_XS_SUPERBLOCK;
  const int within = gid - sb_id * KQ_IQ2_XS_SUPERBLOCK;
  const int ib32 = within / 32;
  const int p = within % 32;
  const int l = p / 8;
  const int j = p % 8;
  const device uint8_t* sb =
      w + static_cast<int64_t>(sb_id) * KQ_IQ2_XS_BLOCK_BYTES;
  const float d = float(*(const device half*)sb);
  const device uint8_t* qp = sb + KQ_IQ2_XS_QS_OFFSET + ib32 * 8 + l * 2;
  const uint q = uint(qp[0]) | (uint(qp[1]) << 8);
  const uint8_t sc = sb[KQ_IQ2_XS_SCALES_OFFSET + ib32];
  const int sc_nib = (l < 2) ? (sc & 0xf) : (sc >> 4);
  const float db = d * (0.5f + float(sc_nib)) * 0.25f;
  const uint8_t signs = ksigns_iq2xs[q >> 9];
  const uint8_t gb = (iq2xs_grid[q & 511] >> (8 * j)) & 0xff;
  const float sgn = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
  out[gid] = T(db * float(gb) * sgn);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_iq2_xs_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_IQ2_XS_SUPERBLOCK, "IQ2_XS kernel requires gs=256");
  static_assert(bits == 2, "IQ2_XS kernel requires bits=2");
  kq_iq2_xs_dequantize_impl<T>(w, out, num_weights, gid);
}

// iq2_xs flat-with-M verify mat-vec: kq_mv_ext_impl + chunk dequant. Mirrors
// kq_iq2_xs_dequantize_impl over 16 contiguous weights (natural order
// [il*16, il*16+16)). chunk il -> ib32 = il/2; l = (il&1)*2 + i/8, byte j =
// i%8. Each qs u16 = 9-bit grid idx + 7-bit sign idx; scale nibble by l (l<2
// low).
inline void kq_iq2_xs_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const int ib32 = il / 2;
  const int lbase = (il & 1) * 2;
  const float d = float(*(const device half*)block);
  const uint8_t sc = block[KQ_IQ2_XS_SCALES_OFFSET + ib32];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int l = lbase + i / 8;
    const int j = i % 8;
    const device uint8_t* qp = block + KQ_IQ2_XS_QS_OFFSET + ib32 * 8 + l * 2;
    const uint q = uint(qp[0]) | (uint(qp[1]) << 8);
    const int sc_nib = (l < 2) ? (sc & 0xf) : (sc >> 4);
    const float db = d * (0.5f + float(sc_nib)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q >> 9];
    const uint8_t gb = (iq2xs_grid[q & 511] >> (8 * j)) & 0xff;
    const float sgn = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
    reg[i / 4][i % 4] = db * float(gb) * sgn;
  }
}

struct KqIq2_xsExt {
  MLX_MTL_CONST int superblock = KQ_IQ2_XS_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_IQ2_XS_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_iq2_xs_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_iq2_xs_mv_ext(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size, // K
    const constant int& out_vec_size, // N
    const constant int& /* vm */, // == r1ptg
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  kq_mv_ext_impl<T, KqIq2_xsExt, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

// IQ2_S: 82 bytes/256 weights. [fp16 d][qs[32]][signs[32]][qh[8]][scales[8]].
// 10-bit grid index (qs low 8 + 2 bits from qh), signs from block bytes,
// scales like IQ2_XS (two 4-bit per byte -> (0.5+s)*0.25). uint64 grid.

MLX_MTL_CONST int KQ_IQ2_S_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_IQ2_S_BLOCK_BYTES = 82;
MLX_MTL_CONST int KQ_IQ2_S_QS_OFFSET = 2;
MLX_MTL_CONST int KQ_IQ2_S_SIGNS_OFFSET = 34;
MLX_MTL_CONST int KQ_IQ2_S_QH_OFFSET = 66;
MLX_MTL_CONST int KQ_IQ2_S_SCALES_OFFSET = 74;

template <typename T>
METAL_FUNC void kq_iq2_s_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_IQ2_S_SUPERBLOCK;
  const int within = gid - sb_id * KQ_IQ2_S_SUPERBLOCK;
  const int ib32 = within / 32;
  const int p = within % 32;
  const int l = p / 8;
  const int j = p % 8;
  const device uint8_t* sb =
      w + static_cast<int64_t>(sb_id) * KQ_IQ2_S_BLOCK_BYTES;
  const float d = float(*(const device half*)sb);
  const device uint8_t* qs = sb + KQ_IQ2_S_QS_OFFSET;
  const device uint8_t* qh = sb + KQ_IQ2_S_QH_OFFSET;
  const device uint8_t* signs = sb + KQ_IQ2_S_SIGNS_OFFSET;
  const uint8_t sc = sb[KQ_IQ2_S_SCALES_OFFSET + ib32];
  const int sc_nib = (l < 2) ? (sc & 0xf) : (sc >> 4);
  const float db = d * (0.5f + float(sc_nib)) * 0.25f;
  const int high2 = (int(qh[ib32]) << (8 - 2 * l)) & 0x300;
  const int qi = int(qs[ib32 * 4 + l]) | high2;
  const uint8_t signs_byte = signs[ib32 * 4 + l];
  const uint8_t gb = (iq2s_grid[qi] >> (8 * j)) & 0xff;
  const float sgn = (signs_byte & kmask_iq2xs[j]) ? -1.0f : 1.0f;
  out[gid] = T(db * float(gb) * sgn);
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_iq2_s_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_IQ2_S_SUPERBLOCK, "IQ2_S kernel requires gs=256");
  static_assert(bits == 2, "IQ2_S kernel requires bits=2");
  kq_iq2_s_dequantize_impl<T>(w, out, num_weights, gid);
}

// iq2_s flat-with-M verify mat-vec: kq_mv_ext_impl + chunk dequant. Mirrors
// kq_iq2_s_dequantize_impl over 16 contiguous weights (natural order
// [il*16, il*16+16)). chunk il -> ib32 = il/2; l = (il&1)*2 + i/8, byte j =
// i%8. 10-bit grid index (qs low 8 + 2 bits from qh), signs from block bytes.
inline void kq_iq2_s_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const int ib32 = il / 2;
  const int lbase = (il & 1) * 2;
  const float d = float(*(const device half*)block);
  const device uint8_t* qs = block + KQ_IQ2_S_QS_OFFSET;
  const device uint8_t* qh = block + KQ_IQ2_S_QH_OFFSET;
  const device uint8_t* signs = block + KQ_IQ2_S_SIGNS_OFFSET;
  const uint8_t sc = block[KQ_IQ2_S_SCALES_OFFSET + ib32];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int l = lbase + i / 8;
    const int j = i % 8;
    const int sc_nib = (l < 2) ? (sc & 0xf) : (sc >> 4);
    const float db = d * (0.5f + float(sc_nib)) * 0.25f;
    const int high2 = (int(qh[ib32]) << (8 - 2 * l)) & 0x300;
    const int qi = int(qs[ib32 * 4 + l]) | high2;
    const uint8_t signs_byte = signs[ib32 * 4 + l];
    const uint8_t gb = (iq2s_grid[qi] >> (8 * j)) & 0xff;
    const float sgn = (signs_byte & kmask_iq2xs[j]) ? -1.0f : 1.0f;
    reg[i / 4][i % 4] = db * float(gb) * sgn;
  }
}

struct KqIq2_sExt {
  MLX_MTL_CONST int superblock = KQ_IQ2_S_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_IQ2_S_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_iq2_s_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_iq2_s_mv_ext(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size, // K
    const constant int& out_vec_size, // N
    const constant int& /* vm */, // == r1ptg
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  kq_mv_ext_impl<T, KqIq2_sExt, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

// IQ1_S: 50 bytes/256 weights. [fp16 d][qs[32]][qh[8] u16]. Eight 32-blocks;
// each qh u16 carries a 3-bit scale (bits 12..14), a block sign (bit 15) folded
// into a +/-0.125 delta, and 3 high bits per 8-group of the 11-bit grid index.
// The grid stores SIGNED int8 -> y = dl*(grid + delta). uint64 grid, shared
// with IQ1_M.

MLX_MTL_CONST int KQ_IQ1_S_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_IQ1_S_BLOCK_BYTES = 50;
MLX_MTL_CONST int KQ_IQ1_S_QS_OFFSET = 2;
MLX_MTL_CONST int KQ_IQ1_S_QH_OFFSET = 34;

template <typename T>
METAL_FUNC void kq_iq1_s_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_IQ1_S_SUPERBLOCK;
  const int within = gid - sb_id * KQ_IQ1_S_SUPERBLOCK;
  const int ib = within / 32;
  const int l = (within % 32) / 8;
  const int j = within % 8;
  const device uint8_t* sb =
      w + static_cast<int64_t>(sb_id) * KQ_IQ1_S_BLOCK_BYTES;
  const float d = float(*(const device half*)sb);
  const device uint8_t* qhp = sb + KQ_IQ1_S_QH_OFFSET + ib * 2;
  const uint qh = uint(qhp[0]) | (uint(qhp[1]) << 8);
  const uint8_t qs = sb[KQ_IQ1_S_QS_OFFSET + ib * 4 + l];
  const float dl = d * float(2 * int((qh >> 12) & 7) + 1);
  const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
  const uint grid_idx = uint(qs) | (((qh >> (3 * l)) & 7) << 8);
  const int8_t gv =
      as_type<int8_t>(uint8_t((iq1s_grid[grid_idx] >> (8 * j)) & 0xff));
  out[gid] = T(dl * (float(gv) + delta));
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_iq1_s_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_IQ1_S_SUPERBLOCK, "IQ1_S kernel requires gs=256");
  static_assert(bits == 1, "IQ1_S kernel requires bits=1");
  kq_iq1_s_dequantize_impl<T>(w, out, num_weights, gid);
}

// iq1_s flat-with-M verify mat-vec: kq_mv_ext_impl + chunk dequant. Mirrors
// kq_iq1_s_dequantize_impl over 16 contiguous weights (natural order
// [il*16, il*16+16)). chunk il -> ib = il/2; l = (il&1)*2 + i/8, byte j = i%8.
// 11-bit grid index (qs + 3 high bits per 8-group), SIGNED grid, +/-delta.
inline void kq_iq1_s_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const int ib = il / 2;
  const int lbase = (il & 1) * 2;
  const float d = float(*(const device half*)block);
  const device uint8_t* qhp = block + KQ_IQ1_S_QH_OFFSET + ib * 2;
  const uint qh = uint(qhp[0]) | (uint(qhp[1]) << 8);
  const float dl = d * float(2 * int((qh >> 12) & 7) + 1);
  const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int l = lbase + i / 8;
    const int j = i % 8;
    const uint8_t qs = block[KQ_IQ1_S_QS_OFFSET + ib * 4 + l];
    const uint grid_idx = uint(qs) | (((qh >> (3 * l)) & 7) << 8);
    const int8_t gv =
        as_type<int8_t>(uint8_t((iq1s_grid[grid_idx] >> (8 * j)) & 0xff));
    reg[i / 4][i % 4] = dl * (float(gv) + delta);
  }
}

struct KqIq1_sExt {
  MLX_MTL_CONST int superblock = KQ_IQ1_S_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_IQ1_S_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_iq1_s_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_iq1_s_mv_ext(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size, // K
    const constant int& out_vec_size, // N
    const constant int& /* vm */, // == r1ptg
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  kq_mv_ext_impl<T, KqIq1_sExt, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

// IQ1_M: 56 bytes/256 weights. [qs[32]][qh[16]][scales[8]]. NO super-block d --
// the fp16 scale is rebuilt from the top nibbles of the four uint16 scale
// words. Each 32-block splits into two 16-halves: per-half 3-bit scale (from
// word sc[ib/2]), per-half sign (qh bit 0x08/0x80 -> +/-0.125 delta), and a
// 3-bit grid-index high part per 8-group (qh<<8 / qh<<4). SIGNED uint64 grid
// (shared with IQ1_S).

MLX_MTL_CONST int KQ_IQ1_M_SUPERBLOCK = 256;
MLX_MTL_CONST int KQ_IQ1_M_BLOCK_BYTES = 56;
MLX_MTL_CONST int KQ_IQ1_M_QS_OFFSET = 0;
MLX_MTL_CONST int KQ_IQ1_M_QH_OFFSET = 32;
MLX_MTL_CONST int KQ_IQ1_M_SCALES_OFFSET = 48;

template <typename T>
METAL_FUNC void kq_iq1_m_dequantize_impl(
    const device uint8_t* w,
    device T* out,
    const constant uint& num_weights,
    uint gid) {
  if (gid >= num_weights) {
    return;
  }
  const int sb_id = gid / KQ_IQ1_M_SUPERBLOCK;
  const int within = gid - sb_id * KQ_IQ1_M_SUPERBLOCK;
  const int ib = within / 32;
  const int l = (within % 32) / 8; // 0..3
  const int j = within % 8;
  const device uint8_t* sb =
      w + static_cast<int64_t>(sb_id) * KQ_IQ1_M_BLOCK_BYTES;
  const device uint8_t* scp = sb + KQ_IQ1_M_SCALES_OFFSET;
  // Rebuild the fp16 super-block scale from four scattered top nibbles.
  const ushort sc0 = ushort(scp[0]) | (ushort(scp[1]) << 8);
  const ushort sc1 = ushort(scp[2]) | (ushort(scp[3]) << 8);
  const ushort sc2 = ushort(scp[4]) | (ushort(scp[5]) << 8);
  const ushort sc3 = ushort(scp[6]) | (ushort(scp[7]) << 8);
  const ushort scale_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) |
      ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
  const float d = float(as_type<half>(scale_u16));
  // Per-16-half 3-bit scale: field base 0 (l<2) or 3 (l>=2), +6 for the odd ib.
  const device uint8_t* swp = scp + (ib / 2) * 2;
  const uint sc_word = uint(swp[0]) | (uint(swp[1]) << 8);
  const int shift = 6 * (ib & 1) + ((l < 2) ? 0 : 3);
  const float dl = d * float(2 * int((sc_word >> shift) & 7) + 1);
  // Grid-index high bits + sign delta from the per-half qh byte.
  const uint8_t qh = sb[KQ_IQ1_M_QH_OFFSET + ib * 2 + l / 2];
  const int hshift = (l & 1) ? 4 : 8;
  const uint grid_idx = uint(sb[KQ_IQ1_M_QS_OFFSET + ib * 4 + l]) |
      ((uint(qh) << hshift) & 0x700);
  const float delta = (qh & ((l & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
  const int8_t gv =
      as_type<int8_t>(uint8_t((iq1s_grid[grid_idx] >> (8 * j)) & 0xff));
  out[gid] = T(dl * (float(gv) + delta));
}

template <typename T, int group_size, int bits>
[[kernel]] void kq_iq1_m_dequantize(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    device T* out,
    const constant uint& num_weights,
    uint gid [[thread_position_in_grid]]) {
  static_assert(
      group_size == KQ_IQ1_M_SUPERBLOCK, "IQ1_M kernel requires gs=256");
  static_assert(bits == 1, "IQ1_M kernel requires bits=1");
  kq_iq1_m_dequantize_impl<T>(w, out, num_weights, gid);
}

// iq1_m flat-with-M verify mat-vec: kq_mv_ext_impl + chunk dequant. Mirrors
// kq_iq1_m_dequantize_impl over 16 contiguous weights (natural order
// [il*16, il*16+16)). chunk il -> ib = il/2; l = (il&1)*2 + i/8, byte j = i%8.
// No super-block d (rebuilt from four scale-word top nibbles); per-16-half
// scale + sign; SIGNED grid (shared with iq1_s).
inline void kq_iq1_m_deq_chunk16(
    const device uint8_t* block,
    short il,
    thread float4x4& reg) {
  const int ib = il / 2;
  const int lbase = (il & 1) * 2;
  const device uint8_t* scp = block + KQ_IQ1_M_SCALES_OFFSET;
  const ushort sc0 = ushort(scp[0]) | (ushort(scp[1]) << 8);
  const ushort sc1 = ushort(scp[2]) | (ushort(scp[3]) << 8);
  const ushort sc2 = ushort(scp[4]) | (ushort(scp[5]) << 8);
  const ushort sc3 = ushort(scp[6]) | (ushort(scp[7]) << 8);
  const ushort scale_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) |
      ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
  const float d = float(as_type<half>(scale_u16));
  const device uint8_t* swp = scp + (ib / 2) * 2;
  const uint sc_word = uint(swp[0]) | (uint(swp[1]) << 8);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int l = lbase + i / 8;
    const int j = i % 8;
    const int shift = 6 * (ib & 1) + ((l < 2) ? 0 : 3);
    const float dl = d * float(2 * int((sc_word >> shift) & 7) + 1);
    const uint8_t qh = block[KQ_IQ1_M_QH_OFFSET + ib * 2 + l / 2];
    const int hshift = (l & 1) ? 4 : 8;
    const uint grid_idx = uint(block[KQ_IQ1_M_QS_OFFSET + ib * 4 + l]) |
        ((uint(qh) << hshift) & 0x700);
    const float delta = (qh & ((l & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
    const int8_t gv =
        as_type<int8_t>(uint8_t((iq1s_grid[grid_idx] >> (8 * j)) & 0xff));
    reg[i / 4][i % 4] = dl * (float(gv) + delta);
  }
}

struct KqIq1_mExt {
  MLX_MTL_CONST int superblock = KQ_IQ1_M_SUPERBLOCK;
  MLX_MTL_CONST int block_bytes = KQ_IQ1_M_BLOCK_BYTES;
  static METAL_FUNC void
  deq_chunk16(const device uint8_t* block, short il, thread float4x4& reg) {
    kq_iq1_m_deq_chunk16(block, il, reg);
  }
};

template <typename T, short r1ptg, short nsg, short nxpsg>
[[kernel]] void kq_iq1_m_mv_ext(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size, // K
    const constant int& out_vec_size, // N
    const constant int& /* vm */, // == r1ptg
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  kq_mv_ext_impl<T, KqIq1_mExt, r1ptg, nsg, nxpsg>(
      w, x, y, in_vec_size, out_vec_size, tgpig, tiisg, sgitg);
}

// ===================== IQ4_XS matmul / gather / qmv =====================

// Lane `simd_lid` owns one 8-weight group (sub-block lid/4, half lid%4) of
// every superblock; simd_sum reduces the 32 lanes. bn=4 -> 2 rows/simdgroup.
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq4_xs_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_IQ4_XS_SUPERBLOCK, "IQ4_XS requires gs=256");
  static_assert(bits == 4, "IQ4_XS requires bits=4");
  constexpr int num_simdgroups = 2;
  constexpr int vpt = 8;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int row_bytes =
      in_vec_size * KQ_IQ4_XS_BLOCK_BYTES / KQ_IQ4_XS_SUPERBLOCK;
  const int nb = in_vec_size / KQ_IQ4_XS_SUPERBLOCK;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const int s = simd_lid / 4; // sub-block
  const int o = (simd_lid % 4) * 8; // offset within sub-block
  const bool is_high = o >= 16;
  const int byte0 = is_high ? (o - 16) : o;
  U result[results_per_simdgroup] = {0};
  for (int ib = 0; ib < nb; ib++) {
    U xt[vpt];
#pragma unroll
    for (int i = 0; i < vpt; i++) {
      xt[i] = U(x[ib * KQ_IQ4_XS_SUPERBLOCK + simd_lid * vpt + i]);
    }
    for (int row = 0; row < active_rows; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(out_row + row) * row_bytes +
          ib * KQ_IQ4_XS_BLOCK_BYTES;
      const U d = U(float(*(const device half*)sb));
      const uint16_t scales_h = uint16_t(sb[2]) | (uint16_t(sb[3]) << 8);
      const device uint8_t* scales_l = sb + KQ_IQ4_XS_SCALESL_OFFSET;
      const int ls = ((scales_l[s / 2] >> (4 * (s & 1))) & 0xf) |
          (((scales_h >> (2 * s)) & 3) << 4);
      const U dl = d * U(ls - 32);
      const device uint8_t* qs = sb + KQ_IQ4_XS_QS_OFFSET + s * 16 + byte0;
      U partial = 0;
#pragma unroll
      for (int i = 0; i < vpt; i++) {
        const int nib = is_high ? (qs[i] >> 4) : (qs[i] & 0xf);
        partial += xt[i] * U(kvalues_iq4nl[nib]);
      }
      result[row] += dl * partial;
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq4_xs_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_iq4_xs_qmv_impl<T, group_size, bits, results_per_simdgroup>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Decodes n_reads weights at tile column `bj` of sub-block `sb` per call; no
// odd-sub-block caching (correctness over the q4_k vectorization).
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqIq4_xsBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ4_XS_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ4_XS_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;
  static_assert(BCOLS == sub_block_size, "IQ4_XS loader requires BCOLS==32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");
  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

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

  KqIq4_xsBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() const {
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
    const float d = float(*(const device half*)src);
    const uint16_t scales_h = uint16_t(src[2]) | (uint16_t(src[3]) << 8);
    const device uint8_t* scales_l = src + KQ_IQ4_XS_SCALESL_OFFSET;
    const int ls = ((scales_l[sb / 2] >> (4 * (sb & 1))) & 0xf) |
        (((scales_h >> (2 * sb)) & 3) << 4);
    const float dl = d * float(ls - 32);
    const device uint8_t* qs = src + KQ_IQ4_XS_QS_OFFSET + sb * 16;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = bj + i;
      const bool is_high = p >= 16;
      const int b = is_high ? (p - 16) : p;
      const int nib = is_high ? (qs[b] >> 4) : (qs[b] & 0xf);
      dst[i] = T(dl * float(kvalues_iq4nl[nib]));
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
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

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_iq4_xs_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ4_XS_SUPERBLOCK, "IQ4_XS requires gs=256");
  static_assert(bits == 4, "IQ4_XS requires bits=4");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq4_xsBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_iq4_xs_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(group_size == KQ_IQ4_XS_SUPERBLOCK, "IQ4_XS requires gs=256");
  static_assert(bits == 4, "IQ4_XS requires bits=4");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq4_xsBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq4_xs_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ4_XS_SUPERBLOCK, "IQ4_XS requires gs=256");
  static_assert(bits == 4, "IQ4_XS requires bits=4");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW =
      KqIq4_xsBlockLoader<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq4_xs_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq4_xs_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv_fast: 1 result per simdgroup -> 2 output rows per
// threadgroup (vs 4 in the default), doubling the threadgroup count for
// the same N. Dispatched (non-batched, N even) when the default tiling
// leaves the GPU short of threadgroups at decode-size N. Bit-exact vs the
// default variant (same per-row lane fold + simd_sum).
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq4_xs_qmv_fast_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq4_xs_qmv_fast_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq4_xs_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq4_xs_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv (see kq_iq4_xs_qmv_fast_fine): 2 output rows per
// threadgroup.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq4_xs_qmv_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq4_xs_qmv_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ===================== IQ3_XXS matmul / gather / qmv =====================

// Grid bytes are magnitudes (<128); signs come from the per-block "gas" word
// (high nibble = scale, low bits index ksigns). 0.5 factor folded into db to
// match the dequant path (bit-equivalent to the mat-vec post-sum *0.5).
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq3_xxs_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_IQ3_XXS_SUPERBLOCK, "IQ3_XXS requires gs=256");
  static_assert(bits == 3, "IQ3_XXS requires bits=3");
  constexpr int num_simdgroups = 2;
  constexpr int vpt = 8;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int row_bytes =
      in_vec_size * KQ_IQ3_XXS_BLOCK_BYTES / KQ_IQ3_XXS_SUPERBLOCK;
  const int nb = in_vec_size / KQ_IQ3_XXS_SUPERBLOCK;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const int s = simd_lid / 4; // sub-block
  const int l = simd_lid % 4; // l-group
  U result[results_per_simdgroup] = {0};
  for (int ib = 0; ib < nb; ib++) {
    U xt[vpt];
#pragma unroll
    for (int i = 0; i < vpt; i++) {
      xt[i] = U(x[ib * KQ_IQ3_XXS_SUPERBLOCK + simd_lid * vpt + i]);
    }
    for (int row = 0; row < active_rows; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(out_row + row) * row_bytes +
          ib * KQ_IQ3_XXS_BLOCK_BYTES;
      const U d = U(float(*(const device half*)sb));
      const device uint8_t* qs = sb + KQ_IQ3_XXS_QS_OFFSET + s * 8;
      const device uint8_t* gas = sb + KQ_IQ3_XXS_GAS_OFFSET + s * 4;
      const uint aux32 = uint(gas[0]) | (uint(gas[1]) << 8) |
          (uint(gas[2]) << 16) | (uint(gas[3]) << 24);
      const U db = d * (U(0.5f) + U(aux32 >> 28)) * U(0.5f);
      const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
      const uint g1 = iq3xxs_grid[qs[2 * l]];
      const uint g2 = iq3xxs_grid[qs[2 * l + 1]];
      U partial = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        partial += xt[j] * U((g1 >> (8 * j)) & 0xff) *
            ((signs & kmask_iq2xs[j]) ? U(-1) : U(1));
        partial += xt[j + 4] * U((g2 >> (8 * j)) & 0xff) *
            ((signs & kmask_iq2xs[j + 4]) ? U(-1) : U(1));
      }
      result[row] += db * partial;
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq3_xxs_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_iq3_xxs_qmv_impl<T, group_size, bits, results_per_simdgroup>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqIq3_xxsBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ3_XXS_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ3_XXS_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;
  static_assert(BCOLS == sub_block_size, "IQ3_XXS loader requires BCOLS==32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");
  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

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

  KqIq3_xxsBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() const {
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
    const float d = float(*(const device half*)src);
    const device uint8_t* qs = src + KQ_IQ3_XXS_QS_OFFSET + sb * 8;
    const device uint8_t* gas = src + KQ_IQ3_XXS_GAS_OFFSET + sb * 4;
    const uint aux32 = uint(gas[0]) | (uint(gas[1]) << 8) |
        (uint(gas[2]) << 16) | (uint(gas[3]) << 24);
    const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = bj + i;
      const int l = p / 8;
      const int jpos = p % 8;
      const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
      const uint g = iq3xxs_grid[jpos < 4 ? qs[2 * l] : qs[2 * l + 1]];
      const int jj = jpos < 4 ? jpos : jpos - 4;
      const float gb = float((g >> (8 * jj)) & 0xff);
      const float sgn = (signs & kmask_iq2xs[jpos]) ? -1.f : 1.f;
      dst[i] = T(db * gb * sgn);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
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

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_iq3_xxs_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ3_XXS_SUPERBLOCK, "IQ3_XXS requires gs=256");
  static_assert(bits == 3, "IQ3_XXS requires bits=3");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq3_xxsBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_iq3_xxs_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(group_size == KQ_IQ3_XXS_SUPERBLOCK, "IQ3_XXS requires gs=256");
  static_assert(bits == 3, "IQ3_XXS requires bits=3");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq3_xxsBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq3_xxs_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ3_XXS_SUPERBLOCK, "IQ3_XXS requires gs=256");
  static_assert(bits == 3, "IQ3_XXS requires bits=3");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW =
      KqIq3_xxsBlockLoader<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq3_xxs_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq3_xxs_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv_fast: 1 result per simdgroup -> 2 output rows per
// threadgroup (vs 4 in the default), doubling the threadgroup count for
// the same N. Dispatched (non-batched, N even) when the default tiling
// leaves the GPU short of threadgroups at decode-size N. Bit-exact vs the
// default variant (same per-row lane fold + simd_sum).
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq3_xxs_qmv_fast_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq3_xxs_qmv_fast_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq3_xxs_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq3_xxs_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv (see kq_iq3_xxs_qmv_fast_fine): 2 output rows per
// threadgroup.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq3_xxs_qmv_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq3_xxs_qmv_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ===================== IQ3_S matmul / gather / qmv =====================

// 9-bit grid index (qs low 8 + qh high bit), per-block sign bytes, scale =
// d*(1+2*nibble) (no 0.5 factor).
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq3_s_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_IQ3_S_SUPERBLOCK, "IQ3_S requires gs=256");
  static_assert(bits == 3, "IQ3_S requires bits=3");
  constexpr int num_simdgroups = 2;
  constexpr int vpt = 8;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int row_bytes =
      in_vec_size * KQ_IQ3_S_BLOCK_BYTES / KQ_IQ3_S_SUPERBLOCK;
  const int nb = in_vec_size / KQ_IQ3_S_SUPERBLOCK;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const int s = simd_lid / 4; // sub-block
  const int l = simd_lid % 4; // l-group
  U result[results_per_simdgroup] = {0};
  for (int ib = 0; ib < nb; ib++) {
    U xt[vpt];
#pragma unroll
    for (int i = 0; i < vpt; i++) {
      xt[i] = U(x[ib * KQ_IQ3_S_SUPERBLOCK + simd_lid * vpt + i]);
    }
    for (int row = 0; row < active_rows; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(out_row + row) * row_bytes +
          ib * KQ_IQ3_S_BLOCK_BYTES;
      const U d = U(float(*(const device half*)sb));
      const device uint8_t* scales = sb + KQ_IQ3_S_SCALES_OFFSET;
      const U db = d * U(1 + 2 * ((scales[s / 2] >> (4 * (s & 1))) & 0xf));
      const uint qh = sb[KQ_IQ3_S_QH_OFFSET + s];
      const device uint8_t* qs = sb + KQ_IQ3_S_QS_OFFSET + s * 8;
      const uint8_t signs = sb[KQ_IQ3_S_SIGNS_OFFSET + s * 4 + l];
      const uint i1 = qs[2 * l] | ((qh << (8 - 2 * l)) & 256);
      const uint i2 = qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256);
      const uint g1 = iq3s_grid[i1];
      const uint g2 = iq3s_grid[i2];
      U partial = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        partial += xt[j] * U((g1 >> (8 * j)) & 0xff) *
            ((signs & kmask_iq2xs[j]) ? U(-1) : U(1));
        partial += xt[j + 4] * U((g2 >> (8 * j)) & 0xff) *
            ((signs & kmask_iq2xs[j + 4]) ? U(-1) : U(1));
      }
      result[row] += db * partial;
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq3_s_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_iq3_s_qmv_impl<T, group_size, bits, results_per_simdgroup>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqIq3_sBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ3_S_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ3_S_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;
  static_assert(BCOLS == sub_block_size, "IQ3_S loader requires BCOLS==32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");
  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

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

  KqIq3_sBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() const {
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
    const float d = float(*(const device half*)src);
    const device uint8_t* scales = src + KQ_IQ3_S_SCALES_OFFSET;
    const float db =
        d * float(1 + 2 * ((scales[sb / 2] >> (4 * (sb & 1))) & 0xf));
    const uint qh = src[KQ_IQ3_S_QH_OFFSET + sb];
    const device uint8_t* qs = src + KQ_IQ3_S_QS_OFFSET + sb * 8;
    const device uint8_t* sg = src + KQ_IQ3_S_SIGNS_OFFSET + sb * 4;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = bj + i;
      const int l = p / 8;
      const int jpos = p % 8;
      const uint8_t signs = sg[l];
      const uint idx = (jpos < 4)
          ? (qs[2 * l] | ((qh << (8 - 2 * l)) & 256))
          : (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256));
      const uint g = iq3s_grid[idx];
      const int jj = jpos < 4 ? jpos : jpos - 4;
      const float gb = float((g >> (8 * jj)) & 0xff);
      const float sgn = (signs & kmask_iq2xs[jpos]) ? -1.f : 1.f;
      dst[i] = T(db * gb * sgn);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
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

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_iq3_s_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ3_S_SUPERBLOCK, "IQ3_S requires gs=256");
  static_assert(bits == 3, "IQ3_S requires bits=3");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq3_sBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_iq3_s_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(group_size == KQ_IQ3_S_SUPERBLOCK, "IQ3_S requires gs=256");
  static_assert(bits == 3, "IQ3_S requires bits=3");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq3_sBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq3_s_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ3_S_SUPERBLOCK, "IQ3_S requires gs=256");
  static_assert(bits == 3, "IQ3_S requires bits=3");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW =
      KqIq3_sBlockLoader<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq3_s_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq3_s_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv_fast: 1 result per simdgroup -> 2 output rows per
// threadgroup (vs 4 in the default), doubling the threadgroup count for
// the same N. Dispatched (non-batched, N even) when the default tiling
// leaves the GPU short of threadgroups at decode-size N. Bit-exact vs the
// default variant (same per-row lane fold + simd_sum).
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq3_s_qmv_fast_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq3_s_qmv_fast_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq3_s_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq3_s_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv (see kq_iq3_s_qmv_fast_fine): 2 output rows per
// threadgroup.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq3_s_qmv_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq3_s_qmv_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ===================== IQ2_XXS matmul / gather / qmv =====================

// Grid is uint64 (one entry = the full 8-weight group). Signs/scale come from
// the per-ib32 u32 (top 4 bits = scale, low 28 index ksigns). (0.5+scale)*0.25
// folded into db to match the dequant path.
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq2_xxs_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_IQ2_XXS_SUPERBLOCK, "IQ2_XXS requires gs=256");
  static_assert(bits == 2, "IQ2_XXS requires bits=2");
  constexpr int num_simdgroups = 2;
  constexpr int vpt = 8;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int row_bytes =
      in_vec_size * KQ_IQ2_XXS_BLOCK_BYTES / KQ_IQ2_XXS_SUPERBLOCK;
  const int nb = in_vec_size / KQ_IQ2_XXS_SUPERBLOCK;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const int s = simd_lid / 4; // sub-block
  const int l = simd_lid % 4; // l-group (one 8-weight group)
  U result[results_per_simdgroup] = {0};
  for (int ib = 0; ib < nb; ib++) {
    U xt[vpt];
#pragma unroll
    for (int i = 0; i < vpt; i++) {
      xt[i] = U(x[ib * KQ_IQ2_XXS_SUPERBLOCK + simd_lid * vpt + i]);
    }
    for (int row = 0; row < active_rows; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(out_row + row) * row_bytes +
          ib * KQ_IQ2_XXS_BLOCK_BYTES;
      const U d = U(float(*(const device half*)sb));
      const device uint8_t* qs = sb + KQ_IQ2_XXS_QS_OFFSET + s * 8;
      const uint signbits = uint(qs[4]) | (uint(qs[5]) << 8) |
          (uint(qs[6]) << 16) | (uint(qs[7]) << 24);
      const U db = d * (U(0.5f) + U(signbits >> 28)) * U(0.25f);
      const uint8_t signs = ksigns_iq2xs[(signbits >> (7 * l)) & 127];
      const uint64_t g = iq2xxs_grid[qs[l]];
      U partial = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        partial += xt[j] * U((g >> (8 * j)) & 0xff) *
            ((signs & kmask_iq2xs[j]) ? U(-1) : U(1));
      }
      result[row] += db * partial;
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq2_xxs_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_iq2_xxs_qmv_impl<T, group_size, bits, results_per_simdgroup>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqIq2_xxsBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ2_XXS_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ2_XXS_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;
  static_assert(BCOLS == sub_block_size, "IQ2_XXS loader requires BCOLS==32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");
  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

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

  KqIq2_xxsBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() const {
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
    const float d = float(*(const device half*)src);
    const device uint8_t* qs = src + KQ_IQ2_XXS_QS_OFFSET + sb * 8;
    const uint signbits = uint(qs[4]) | (uint(qs[5]) << 8) |
        (uint(qs[6]) << 16) | (uint(qs[7]) << 24);
    const float db = d * (0.5f + float(signbits >> 28)) * 0.25f;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = bj + i;
      const int l = p / 8;
      const int j = p % 8;
      const uint8_t signs = ksigns_iq2xs[(signbits >> (7 * l)) & 127];
      const uint64_t g = iq2xxs_grid[qs[l]];
      const float gb = float((g >> (8 * j)) & 0xff);
      const float sgn = (signs & kmask_iq2xs[j]) ? -1.f : 1.f;
      dst[i] = T(db * gb * sgn);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
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

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_iq2_xxs_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ2_XXS_SUPERBLOCK, "IQ2_XXS requires gs=256");
  static_assert(bits == 2, "IQ2_XXS requires bits=2");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq2_xxsBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_iq2_xxs_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(group_size == KQ_IQ2_XXS_SUPERBLOCK, "IQ2_XXS requires gs=256");
  static_assert(bits == 2, "IQ2_XXS requires bits=2");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq2_xxsBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_xxs_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ2_XXS_SUPERBLOCK, "IQ2_XXS requires gs=256");
  static_assert(bits == 2, "IQ2_XXS requires bits=2");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW =
      KqIq2_xxsBlockLoader<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_xxs_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq2_xxs_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv_fast: 1 result per simdgroup -> 2 output rows per
// threadgroup (vs 4 in the default), doubling the threadgroup count for
// the same N. Dispatched (non-batched, N even) when the default tiling
// leaves the GPU short of threadgroups at decode-size N. Bit-exact vs the
// default variant (same per-row lane fold + simd_sum).
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_xxs_qmv_fast_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq2_xxs_qmv_fast_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_xxs_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq2_xxs_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv (see kq_iq2_xxs_qmv_fast_fine): 2 output rows per
// threadgroup.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_xxs_qmv_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq2_xxs_qmv_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ===================== IQ2_XS matmul / gather / qmv =====================

// Grid is uint64 (one entry = the full 8-weight group). Each qs u16 = 9-bit
// grid idx + 7-bit sign idx (ksigns). scales: two 4-bit per byte, low nibble
// for the first half of the ib32, high for the second; (0.5+s)*0.25 into db.
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq2_xs_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_IQ2_XS_SUPERBLOCK, "IQ2_XS requires gs=256");
  static_assert(bits == 2, "IQ2_XS requires bits=2");
  constexpr int num_simdgroups = 2;
  constexpr int vpt = 8;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int row_bytes =
      in_vec_size * KQ_IQ2_XS_BLOCK_BYTES / KQ_IQ2_XS_SUPERBLOCK;
  const int nb = in_vec_size / KQ_IQ2_XS_SUPERBLOCK;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const int s = simd_lid / 4; // sub-block
  const int l = simd_lid % 4; // l-group (one 8-weight group)
  U result[results_per_simdgroup] = {0};
  for (int ib = 0; ib < nb; ib++) {
    U xt[vpt];
#pragma unroll
    for (int i = 0; i < vpt; i++) {
      xt[i] = U(x[ib * KQ_IQ2_XS_SUPERBLOCK + simd_lid * vpt + i]);
    }
    for (int row = 0; row < active_rows; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(out_row + row) * row_bytes +
          ib * KQ_IQ2_XS_BLOCK_BYTES;
      const U d = U(float(*(const device half*)sb));
      const device uint8_t* qp = sb + KQ_IQ2_XS_QS_OFFSET + s * 8 + l * 2;
      const uint q = uint(qp[0]) | (uint(qp[1]) << 8);
      const uint8_t sc = sb[KQ_IQ2_XS_SCALES_OFFSET + s];
      const int sc_nib = (l < 2) ? (sc & 0xf) : (sc >> 4);
      const U db = d * (U(0.5f) + U(sc_nib)) * U(0.25f);
      const uint8_t signs = ksigns_iq2xs[q >> 9];
      const uint64_t g = iq2xs_grid[q & 511];
      U partial = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        partial += xt[j] * U((g >> (8 * j)) & 0xff) *
            ((signs & kmask_iq2xs[j]) ? U(-1) : U(1));
      }
      result[row] += db * partial;
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq2_xs_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_iq2_xs_qmv_impl<T, group_size, bits, results_per_simdgroup>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqIq2_xsBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ2_XS_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ2_XS_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;
  static_assert(BCOLS == sub_block_size, "IQ2_XS loader requires BCOLS==32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");
  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

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

  KqIq2_xsBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() const {
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
    const float d = float(*(const device half*)src);
    const device uint8_t* qs = src + KQ_IQ2_XS_QS_OFFSET + sb * 8;
    const uint8_t sc = src[KQ_IQ2_XS_SCALES_OFFSET + sb];
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = bj + i;
      const int l = p / 8;
      const int j = p % 8;
      const int sc_nib = (l < 2) ? (sc & 0xf) : (sc >> 4);
      const float db = d * (0.5f + float(sc_nib)) * 0.25f;
      const device uint8_t* qp = qs + l * 2;
      const uint q = uint(qp[0]) | (uint(qp[1]) << 8);
      const uint8_t signs = ksigns_iq2xs[q >> 9];
      const uint64_t g = iq2xs_grid[q & 511];
      const float gb = float((g >> (8 * j)) & 0xff);
      const float sgn = (signs & kmask_iq2xs[j]) ? -1.f : 1.f;
      dst[i] = T(db * gb * sgn);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
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

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_iq2_xs_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ2_XS_SUPERBLOCK, "IQ2_XS requires gs=256");
  static_assert(bits == 2, "IQ2_XS requires bits=2");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq2_xsBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_iq2_xs_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(group_size == KQ_IQ2_XS_SUPERBLOCK, "IQ2_XS requires gs=256");
  static_assert(bits == 2, "IQ2_XS requires bits=2");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq2_xsBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_xs_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ2_XS_SUPERBLOCK, "IQ2_XS requires gs=256");
  static_assert(bits == 2, "IQ2_XS requires bits=2");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW =
      KqIq2_xsBlockLoader<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_xs_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq2_xs_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv_fast: 1 result per simdgroup -> 2 output rows per
// threadgroup (vs 4 in the default), doubling the threadgroup count for
// the same N. Dispatched (non-batched, N even) when the default tiling
// leaves the GPU short of threadgroups at decode-size N. Bit-exact vs the
// default variant (same per-row lane fold + simd_sum).
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_xs_qmv_fast_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq2_xs_qmv_fast_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_xs_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq2_xs_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv (see kq_iq2_xs_qmv_fast_fine): 2 output rows per
// threadgroup.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_xs_qmv_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq2_xs_qmv_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ===================== IQ2_S matmul / gather / qmv =====================

// Grid is uint64 (one entry = the full 8-weight group). 10-bit grid index (qs
// low 8 + 2 bits from qh), signs from the block's own sign bytes, scales like
// IQ2_XS (two 4-bit per byte -> (0.5+s)*0.25 into db).
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq2_s_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_IQ2_S_SUPERBLOCK, "IQ2_S requires gs=256");
  static_assert(bits == 2, "IQ2_S requires bits=2");
  constexpr int num_simdgroups = 2;
  constexpr int vpt = 8;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int row_bytes =
      in_vec_size * KQ_IQ2_S_BLOCK_BYTES / KQ_IQ2_S_SUPERBLOCK;
  const int nb = in_vec_size / KQ_IQ2_S_SUPERBLOCK;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const int s = simd_lid / 4; // sub-block
  const int l = simd_lid % 4; // l-group (one 8-weight group)
  U result[results_per_simdgroup] = {0};
  for (int ib = 0; ib < nb; ib++) {
    U xt[vpt];
#pragma unroll
    for (int i = 0; i < vpt; i++) {
      xt[i] = U(x[ib * KQ_IQ2_S_SUPERBLOCK + simd_lid * vpt + i]);
    }
    for (int row = 0; row < active_rows; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(out_row + row) * row_bytes +
          ib * KQ_IQ2_S_BLOCK_BYTES;
      const U d = U(float(*(const device half*)sb));
      const device uint8_t* qs = sb + KQ_IQ2_S_QS_OFFSET + s * 4;
      const device uint8_t* sg = sb + KQ_IQ2_S_SIGNS_OFFSET + s * 4;
      const uint qh = sb[KQ_IQ2_S_QH_OFFSET + s];
      const uint8_t sc = sb[KQ_IQ2_S_SCALES_OFFSET + s];
      const int sc_nib = (l < 2) ? (sc & 0xf) : (sc >> 4);
      const U db = d * (U(0.5f) + U(sc_nib)) * U(0.25f);
      const uint idx = qs[l] | ((qh << (8 - 2 * l)) & 0x300);
      const uint8_t signs_byte = sg[l];
      const uint64_t g = iq2s_grid[idx];
      U partial = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        partial += xt[j] * U((g >> (8 * j)) & 0xff) *
            ((signs_byte & kmask_iq2xs[j]) ? U(-1) : U(1));
      }
      result[row] += db * partial;
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq2_s_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_iq2_s_qmv_impl<T, group_size, bits, results_per_simdgroup>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqIq2_sBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ2_S_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ2_S_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;
  static_assert(BCOLS == sub_block_size, "IQ2_S loader requires BCOLS==32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");
  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

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

  KqIq2_sBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() const {
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
    const float d = float(*(const device half*)src);
    const device uint8_t* qs = src + KQ_IQ2_S_QS_OFFSET + sb * 4;
    const device uint8_t* sg = src + KQ_IQ2_S_SIGNS_OFFSET + sb * 4;
    const uint qh = src[KQ_IQ2_S_QH_OFFSET + sb];
    const uint8_t sc = src[KQ_IQ2_S_SCALES_OFFSET + sb];
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = bj + i;
      const int l = p / 8;
      const int j = p % 8;
      const int sc_nib = (l < 2) ? (sc & 0xf) : (sc >> 4);
      const float db = d * (0.5f + float(sc_nib)) * 0.25f;
      const uint idx = qs[l] | ((qh << (8 - 2 * l)) & 0x300);
      const uint8_t signs = sg[l];
      const uint64_t g = iq2s_grid[idx];
      const float gb = float((g >> (8 * j)) & 0xff);
      const float sgn = (signs & kmask_iq2xs[j]) ? -1.f : 1.f;
      dst[i] = T(db * gb * sgn);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
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

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_iq2_s_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ2_S_SUPERBLOCK, "IQ2_S requires gs=256");
  static_assert(bits == 2, "IQ2_S requires bits=2");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq2_sBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_iq2_s_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(group_size == KQ_IQ2_S_SUPERBLOCK, "IQ2_S requires gs=256");
  static_assert(bits == 2, "IQ2_S requires bits=2");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq2_sBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_s_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ2_S_SUPERBLOCK, "IQ2_S requires gs=256");
  static_assert(bits == 2, "IQ2_S requires bits=2");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW =
      KqIq2_sBlockLoader<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_s_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq2_s_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv_fast: 1 result per simdgroup -> 2 output rows per
// threadgroup (vs 4 in the default), doubling the threadgroup count for
// the same N. Dispatched (non-batched, N even) when the default tiling
// leaves the GPU short of threadgroups at decode-size N. Bit-exact vs the
// default variant (same per-row lane fold + simd_sum).
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_s_qmv_fast_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq2_s_qmv_fast_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_s_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq2_s_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv (see kq_iq2_s_qmv_fast_fine): 2 output rows per
// threadgroup.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq2_s_qmv_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq2_s_qmv_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ===================== IQ1_S matmul / gather / qmv =====================

// uint64 grid with SIGNED int8 entries. Each 32-block: 3-bit scale (qh bits
// 12..14 -> 2s+1), a block sign (qh bit 15) folded into a +/-0.125 delta, and 3
// high index bits per 8-group (qh >> 3l). y = dl*(grid + delta); no sign array,
// no post-sum factor.
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq1_s_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_IQ1_S_SUPERBLOCK, "IQ1_S requires gs=256");
  static_assert(bits == 1, "IQ1_S requires bits=1");
  constexpr int num_simdgroups = 2;
  constexpr int vpt = 8;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int row_bytes =
      in_vec_size * KQ_IQ1_S_BLOCK_BYTES / KQ_IQ1_S_SUPERBLOCK;
  const int nb = in_vec_size / KQ_IQ1_S_SUPERBLOCK;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const int s = simd_lid / 4; // sub-block
  const int l = simd_lid % 4; // l-group (one 8-weight group)
  U result[results_per_simdgroup] = {0};
  for (int ib = 0; ib < nb; ib++) {
    U xt[vpt];
#pragma unroll
    for (int i = 0; i < vpt; i++) {
      xt[i] = U(x[ib * KQ_IQ1_S_SUPERBLOCK + simd_lid * vpt + i]);
    }
    for (int row = 0; row < active_rows; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(out_row + row) * row_bytes +
          ib * KQ_IQ1_S_BLOCK_BYTES;
      const U d = U(float(*(const device half*)sb));
      const device uint8_t* qhp = sb + KQ_IQ1_S_QH_OFFSET + s * 2;
      const uint qh = uint(qhp[0]) | (uint(qhp[1]) << 8);
      const uint8_t qs = sb[KQ_IQ1_S_QS_OFFSET + s * 4 + l];
      const U dl = d * U(2 * int((qh >> 12) & 7) + 1);
      const U delta = (qh & 0x8000) ? U(-0.125f) : U(0.125f);
      const uint idx = uint(qs) | (((qh >> (3 * l)) & 7) << 8);
      const uint64_t g = iq1s_grid[idx];
      U partial = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int8_t gv = as_type<int8_t>(uint8_t((g >> (8 * j)) & 0xff));
        partial += xt[j] * (U(gv) + delta);
      }
      result[row] += dl * partial;
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq1_s_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_iq1_s_qmv_impl<T, group_size, bits, results_per_simdgroup>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqIq1_sBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ1_S_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ1_S_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;
  static_assert(BCOLS == sub_block_size, "IQ1_S loader requires BCOLS==32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");
  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

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

  KqIq1_sBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() const {
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
    const float d = float(*(const device half*)src);
    const device uint8_t* qhp = src + KQ_IQ1_S_QH_OFFSET + sb * 2;
    const uint qh = uint(qhp[0]) | (uint(qhp[1]) << 8);
    const device uint8_t* qs = src + KQ_IQ1_S_QS_OFFSET + sb * 4;
    const float dl = d * float(2 * int((qh >> 12) & 7) + 1);
    const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = bj + i;
      const int l = p / 8;
      const int j = p % 8;
      const uint idx = uint(qs[l]) | (((qh >> (3 * l)) & 7) << 8);
      const int8_t gv =
          as_type<int8_t>(uint8_t((iq1s_grid[idx] >> (8 * j)) & 0xff));
      dst[i] = T(dl * (float(gv) + delta));
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
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

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_iq1_s_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ1_S_SUPERBLOCK, "IQ1_S requires gs=256");
  static_assert(bits == 1, "IQ1_S requires bits=1");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq1_sBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_iq1_s_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(group_size == KQ_IQ1_S_SUPERBLOCK, "IQ1_S requires gs=256");
  static_assert(bits == 1, "IQ1_S requires bits=1");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq1_sBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq1_s_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ1_S_SUPERBLOCK, "IQ1_S requires gs=256");
  static_assert(bits == 1, "IQ1_S requires bits=1");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW =
      KqIq1_sBlockLoader<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq1_s_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq1_s_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv_fast: 1 result per simdgroup -> 2 output rows per
// threadgroup (vs 4 in the default), doubling the threadgroup count for
// the same N. Dispatched (non-batched, N even) when the default tiling
// leaves the GPU short of threadgroups at decode-size N. Bit-exact vs the
// default variant (same per-row lane fold + simd_sum).
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq1_s_qmv_fast_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq1_s_qmv_fast_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq1_s_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq1_s_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv (see kq_iq1_s_qmv_fast_fine): 2 output rows per
// threadgroup.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq1_s_qmv_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq1_s_qmv_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ===================== IQ1_M matmul / gather / qmv =====================

// Like IQ1_S but no super-block d (rebuilt from four scattered scale-word top
// nibbles) and per-16-half structure: per-half 3-bit scale (word sc[s/2]) and
// per-half sign (qh bit 0x08/0x80 -> +/-0.125 delta); grid-index high part is
// qh<<8 (even l) / qh<<4 (odd l). SIGNED uint64 grid (shared with IQ1_S).
template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq1_m_qmv_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(group_size == KQ_IQ1_M_SUPERBLOCK, "IQ1_M requires gs=256");
  static_assert(bits == 1, "IQ1_M requires bits=1");
  constexpr int num_simdgroups = 2;
  constexpr int vpt = 8;
  typedef float U;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, out_vec_size - out_row);
  const int row_bytes =
      in_vec_size * KQ_IQ1_M_BLOCK_BYTES / KQ_IQ1_M_SUPERBLOCK;
  const int nb = in_vec_size / KQ_IQ1_M_SUPERBLOCK;
  x += tid.x * in_vec_size;
  y += tid.x * out_vec_size;
  const int s = simd_lid / 4; // sub-block
  const int l = simd_lid % 4; // l-group (one 8-weight group)
  U result[results_per_simdgroup] = {0};
  for (int ib = 0; ib < nb; ib++) {
    U xt[vpt];
#pragma unroll
    for (int i = 0; i < vpt; i++) {
      xt[i] = U(x[ib * KQ_IQ1_M_SUPERBLOCK + simd_lid * vpt + i]);
    }
    for (int row = 0; row < active_rows; row++) {
      const device uint8_t* sb = w +
          static_cast<int64_t>(out_row + row) * row_bytes +
          ib * KQ_IQ1_M_BLOCK_BYTES;
      const device uint8_t* scp = sb + KQ_IQ1_M_SCALES_OFFSET;
      const ushort sc0 = ushort(scp[0]) | (ushort(scp[1]) << 8);
      const ushort sc1 = ushort(scp[2]) | (ushort(scp[3]) << 8);
      const ushort sc2 = ushort(scp[4]) | (ushort(scp[5]) << 8);
      const ushort sc3 = ushort(scp[6]) | (ushort(scp[7]) << 8);
      const ushort scale_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) |
          ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
      const U d = U(float(as_type<half>(scale_u16)));
      const device uint8_t* swp = scp + (s / 2) * 2;
      const uint sc_word = uint(swp[0]) | (uint(swp[1]) << 8);
      const int shift = 6 * (s & 1) + ((l < 2) ? 0 : 3);
      const U dl = d * U(2 * int((sc_word >> shift) & 7) + 1);
      const uint8_t qh = sb[KQ_IQ1_M_QH_OFFSET + s * 2 + l / 2];
      const int hshift = (l & 1) ? 4 : 8;
      const uint idx = uint(sb[KQ_IQ1_M_QS_OFFSET + s * 4 + l]) |
          ((uint(qh) << hshift) & 0x700);
      const U delta = (qh & ((l & 1) ? 0x80 : 0x08)) ? U(-0.125f) : U(0.125f);
      const uint64_t g = iq1s_grid[idx];
      U partial = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int8_t gv = as_type<int8_t>(uint8_t((g >> (8 * j)) & 0xff));
        partial += xt[j] * (U(gv) + delta);
      }
      result[row] += dl * partial;
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U r = simd_sum(result[row]);
    if (simd_lid == 0 && row < active_rows) {
      y[out_row + row] = static_cast<T>(r);
    }
  }
}

template <typename T, int group_size, int bits, int results_per_simdgroup = 2>
METAL_FUNC void kq_iq1_m_qmv_fast_impl(
    const device uint8_t* w,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  kq_iq1_m_qmv_impl<T, group_size, bits, results_per_simdgroup>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size>
struct KqIq1_mBlockLoader {
  MLX_MTL_CONST int weights_per_block = KQ_IQ1_M_SUPERBLOCK;
  MLX_MTL_CONST int bytes_per_block = KQ_IQ1_M_BLOCK_BYTES;
  MLX_MTL_CONST int sub_block_size = 32;
  MLX_MTL_CONST int sub_blocks_per_block = weights_per_block / sub_block_size;
  static_assert(BCOLS == sub_block_size, "IQ1_M loader requires BCOLS==32.");
  static_assert(
      (BCOLS * BROWS) % tgp_size == 0,
      "tgp_size must evenly divide BCOLS * BROWS.");
  MLX_MTL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
  MLX_MTL_CONST short TCOLS = BCOLS / n_reads;

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

  KqIq1_mBlockLoader(
      const device uint8_t* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      int col_in_block = 0)
      : src_ld(src_ld_),
        row_bytes(src_ld_ * bytes_per_block / weights_per_block),
        tile_stride(
            reduction_dim
                ? 0
                : BROWS * (src_ld_ * bytes_per_block / weights_per_block)),
        fixed_sub_block_idx(
            reduction_dim == 0 ? (col_in_block / sub_block_size) : 0),
        thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj((thread_idx % TCOLS) * n_reads),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * (src_ld_ * bytes_per_block / weights_per_block)),
        sub_block_idx(0) {}

  void load_unsafe() const {
    const short sb = (reduction_dim == 0) ? fixed_sub_block_idx : sub_block_idx;
    const device uint8_t* scp = src + KQ_IQ1_M_SCALES_OFFSET;
    const ushort sc0 = ushort(scp[0]) | (ushort(scp[1]) << 8);
    const ushort sc1 = ushort(scp[2]) | (ushort(scp[3]) << 8);
    const ushort sc2 = ushort(scp[4]) | (ushort(scp[5]) << 8);
    const ushort sc3 = ushort(scp[6]) | (ushort(scp[7]) << 8);
    const ushort scale_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) |
        ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
    const float d = float(as_type<half>(scale_u16));
    const device uint8_t* swp = scp + (sb / 2) * 2;
    const uint sc_word = uint(swp[0]) | (uint(swp[1]) << 8);
    const device uint8_t* qhp = src + KQ_IQ1_M_QH_OFFSET + sb * 2;
    const device uint8_t* qs = src + KQ_IQ1_M_QS_OFFSET + sb * 4;
#pragma unroll
    for (short i = 0; i < n_reads; i++) {
      const int p = bj + i;
      const int l = p / 8;
      const int j = p % 8;
      const int shift = 6 * (sb & 1) + ((l < 2) ? 0 : 3);
      const float dl = d * float(2 * int((sc_word >> shift) & 7) + 1);
      const uint8_t qh = qhp[l / 2];
      const int hshift = (l & 1) ? 4 : 8;
      const uint idx = uint(qs[l]) | ((uint(qh) << hshift) & 0x700);
      const float delta = (qh & ((l & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
      const int8_t gv =
          as_type<int8_t>(uint8_t((iq1s_grid[idx] >> (8 * j)) & 0xff));
      dst[i] = T(dl * (float(gv) + delta));
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (bi >= src_tile_dim.y) {
#pragma unroll
      for (short i = 0; i < n_reads; i++) {
        dst[i] = T(0);
      }
      return;
    }
    load_unsafe();
  }

  void next() {
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

template <typename T, int group_size, int bits, bool aligned_N, bool batched>
[[kernel]] void kq_iq1_m_qmm_t(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ1_M_SUPERBLOCK, "IQ1_M requires gs=256");
  static_assert(bits == 1, "IQ1_M requires bits=1");
  constexpr int BM = 64, BK = 32, BN = 64;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq1_mBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool aligned_N>
[[kernel]] void kq_iq1_m_qmm_t_splitk(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    const constant int& split_k_partition_stride,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(group_size == KQ_IQ1_M_SUPERBLOCK, "IQ1_M requires gs=256");
  static_assert(bits == 1, "IQ1_M requires bits=1");
  constexpr int BM = 32, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using LoaderW =
      KqIq1_mBlockLoader<T, BN, BK, BK_padded, 1, 2 * 2 * SIMD_SIZE>;
  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = w;
  wl += (k_start / LoaderW::weights_per_block) * LoaderW::bytes_per_block;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);
  kq_qmm_t_impl<T, LoaderW, aligned_N, BM, BK, BN>(
      wl,
      x,
      y,
      Xs,
      Ws,
      K,
      N,
      M,
      k_partition_size,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq1_m_qmm_n(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  static_assert(group_size == KQ_IQ1_M_SUPERBLOCK, "IQ1_M requires gs=256");
  static_assert(bits == 1, "IQ1_M requires bits=1");
  constexpr int BM = 64, BK = 32, BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];
  using LoaderW =
      KqIq1_mBlockLoader<T, BK, BN, BN_padded, 0, 2 * 2 * SIMD_SIZE>;
  kq_qmm_n_impl<T, LoaderW, BM, BK, BN>(
      w, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq1_m_qmv_fast(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq1_m_qmv_fast_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv_fast: 1 result per simdgroup -> 2 output rows per
// threadgroup (vs 4 in the default), doubling the threadgroup count for
// the same N. Dispatched (non-batched, N even) when the default tiling
// leaves the GPU short of threadgroups at decode-size N. Bit-exact vs the
// default variant (same per-row lane fold + simd_sum).
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq1_m_qmv_fast_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq1_m_qmv_fast_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq1_m_qmv(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  if constexpr (batched) {
    int batch_M = x_shape[x_batch_ndims];
    kq_adjust_matrix_offsets<T>(
        x,
        w,
        y,
        out_vec_size * batch_M,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        tid);
  }
  kq_iq1_m_qmv_impl<T, group_size, bits>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Finer-tiled qmv (see kq_iq1_m_qmv_fast_fine): 2 output rows per
// threadgroup.
template <typename T, int group_size, int bits, bool batched>
[[kernel]] void kq_iq1_m_qmv_fine(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* /* s_strides */,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  kq_iq1_m_qmv_impl<T, group_size, bits, 1>(
      w, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// Threadgroup-staged decode LUTs. The iq2/iq3_xxs chunk dequant reads the
// constant-address grid and sign tables with a data-dependent index per
// weight byte, and at gather-MoE occupancy those scattered constant-cache
// reads bound the kernel ahead of DRAM. Staging the tables into threadgroup
// memory once per threadgroup removes that latency; kmask_iq2xs is 1 << j
// and folds away. Codecs without LUTs keep bytes = 0 and the passthrough
// three-argument deq.
template <typename Codec>
struct KqTgLuts {
  MLX_MTL_CONST int bytes = 0;
  static METAL_FUNC void stage(threadgroup uint8_t*, ushort, ushort) {}
  static METAL_FUNC void deq_chunk16(
      const device uint8_t* block,
      short il,
      thread float4x4& reg,
      const threadgroup uint8_t*) {
    Codec::deq_chunk16(block, il, reg);
  }
};

template <>
struct KqTgLuts<KqIq2_xxsExt> {
  MLX_MTL_CONST int bytes = 2048 + 128; // u64 grid[256] | ksigns[128]
  static METAL_FUNC void
  stage(threadgroup uint8_t* dst, ushort lin, ushort n_threads) {
    threadgroup uint32_t* d32 = reinterpret_cast<threadgroup uint32_t*>(dst);
    const constant uint32_t* grid32 =
        reinterpret_cast<const constant uint32_t*>(iq2xxs_grid);
    for (int i = lin; i < 512; i += n_threads) {
      d32[i] = grid32[i];
    }
    const constant uint32_t* signs32 =
        reinterpret_cast<const constant uint32_t*>(ksigns_iq2xs);
    for (int i = lin; i < 32; i += n_threads) {
      d32[512 + i] = signs32[i];
    }
  }
  static METAL_FUNC void deq_chunk16(
      const device uint8_t* block,
      short il,
      thread float4x4& reg,
      const threadgroup uint8_t* luts) {
    const threadgroup uint64_t* grid =
        reinterpret_cast<const threadgroup uint64_t*>(luts);
    const threadgroup uint8_t* ksigns = luts + 2048;
    const int ib32 = il / 2;
    const int lbase = (il & 1) * 2;
    const float d = float(*(const device half*)block);
    const device uint8_t* qs = block + KQ_IQ2_XXS_QS_OFFSET + ib32 * 8;
    const uint32_t signbits = uint32_t(qs[4]) | (uint32_t(qs[5]) << 8) |
        (uint32_t(qs[6]) << 16) | (uint32_t(qs[7]) << 24);
    const float db = d * (0.5f + float(signbits >> 28)) * 0.25f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int l = lbase + i / 8;
      const int j = i % 8;
      const uint8_t signs = ksigns[(signbits >> (7 * l)) & 127];
      const uint8_t gb = (grid[qs[l]] >> (8 * j)) & 0xff;
      const float sgn = (signs & (1 << j)) ? -1.0f : 1.0f;
      reg[i / 4][i % 4] = db * float(gb) * sgn;
    }
  }
};

template <>
struct KqTgLuts<KqIq3_xxsExt> {
  MLX_MTL_CONST int bytes = 1024 + 128; // u32 grid[256] | ksigns[128]
  static METAL_FUNC void
  stage(threadgroup uint8_t* dst, ushort lin, ushort n_threads) {
    threadgroup uint32_t* d32 = reinterpret_cast<threadgroup uint32_t*>(dst);
    for (int i = lin; i < 256; i += n_threads) {
      d32[i] = iq3xxs_grid[i];
    }
    const constant uint32_t* signs32 =
        reinterpret_cast<const constant uint32_t*>(ksigns_iq2xs);
    for (int i = lin; i < 32; i += n_threads) {
      d32[256 + i] = signs32[i];
    }
  }
  static METAL_FUNC void deq_chunk16(
      const device uint8_t* block,
      short il,
      thread float4x4& reg,
      const threadgroup uint8_t* luts) {
    const threadgroup uint32_t* grid =
        reinterpret_cast<const threadgroup uint32_t*>(luts);
    const threadgroup uint8_t* ksigns = luts + 1024;
    const int ib32 = il / 2;
    const int lbase = (il & 1) * 2;
    const float d = float(*(const device half*)block);
    const device uint8_t* qs = block + KQ_IQ3_XXS_QS_OFFSET + ib32 * 8;
    const device uint8_t* gas = block + KQ_IQ3_XXS_GAS_OFFSET + ib32 * 4;
    const uint32_t aux32 = uint32_t(gas[0]) | (uint32_t(gas[1]) << 8) |
        (uint32_t(gas[2]) << 16) | (uint32_t(gas[3]) << 24);
    const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int l = lbase + i / 8;
      const int sub = i % 8;
      const uint8_t signs = ksigns[(aux32 >> (7 * l)) & 127];
      const int qi = (sub < 4) ? int(qs[2 * l]) : int(qs[2 * l + 1]);
      const int bytej = (sub < 4) ? sub : (sub - 4);
      const uint8_t gb = (grid[qi] >> (8 * bytej)) & 0xff;
      const float sgn = (signs & (1 << sub)) ? -1.0f : 1.0f;
      reg[i / 4][i % 4] = db * float(gb) * sgn;
    }
  }
};
