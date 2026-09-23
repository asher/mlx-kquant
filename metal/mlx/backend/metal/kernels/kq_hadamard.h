// Signed block Walsh-Hadamard rotation of activation rows (see
// hadamard_rotate in kquant.h), alone or fused onto the op that produces
// the row. One threadgroup of NT threads per (row, block): each thread owns
// NE = N / NT values at stride NT, applies the optional sign and the
// 1/sqrt(N) scale on load, and runs the butterflies in three stages:
// distances below the simd width through simd_shuffle_xor, distances up to
// NT through threadgroup memory, and the rest between the thread's own
// registers. All math is f32 with one round to T at the store. Grid:
// (rows * blocks_per_row, 1, 1) threadgroups.
//
// The optional permute (perm_hd > 0) reads the row in grouped V-head order
// (tiled [rep, nk, hd] -> grouped [nk, rep, hd]) so the sign and the
// transform see the permuted row; the sign vector is indexed by the
// permuted position.
//
// kq_glu_hadamard produces the row in registers first, act(gate) * x, and
// rotates it in the same threadgroup.

#define KQ_HADAMARD_NT 256

// Butterflies over one block held as reg[j] = row[j * NT + tid], in place.
template <int N, int NT>
inline void
kq_fwht_regs(thread float* reg, threadgroup float* shmem, const int tid) {
  constexpr int NW = 32;
  constexpr int NE = N / NT;
  for (int i = 1; i < NW; i *= 2) {
    for (int j = 0; j < NE; j++) {
      const float v = reg[j];
      const float v2 = simd_shuffle_xor(v, ushort(i));
      reg[j] = (tid & i) == 0 ? v2 + v : v2 - v;
    }
  }

  for (int i = NW; i < NT; i *= 2) {
    for (int j = 0; j < NE; j++) {
      shmem[j * NT + tid] = reg[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < NE; j++) {
      const float v = reg[j];
      const float v2 = shmem[j * NT + (tid ^ i)];
      reg[j] = (tid & i) == 0 ? v2 + v : v2 - v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int i = NT; i < N; i *= 2) {
    const int step = i / NT;
    for (int j = 0; j < NE; j += 2 * step) {
      for (int k = 0; k < step; k++) {
        const float a = reg[j + k];
        const float b = reg[j + k + step];
        reg[j + k] = a + b;
        reg[j + k + step] = a - b;
      }
    }
  }
}

// sigmoid(g) without overflow for large |g|.
inline float kq_sigmoid(const float g) {
  const float e = metal::exp(-metal::abs(g));
  const float p = 1.0f / (1.0f + e);
  return g < 0.0f ? e * p : p;
}

template <int N, int NT, typename T>
[[kernel]] void kq_hadamard(
    const device T* x [[buffer(0)]],
    const device float* signs [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant int& n_blk [[buffer(3)]],
    const constant int& has_signs [[buffer(4)]],
    const constant int& perm_rep [[buffer(5)]],
    const constant int& perm_nk [[buffer(6)]],
    const constant int& perm_hd [[buffer(7)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]) {
  constexpr int NW = 32;
  constexpr int NE = N / NT;
  static_assert(N % NT == 0 && NT % NW == 0, "block must tile the group");

  threadgroup float shmem[N];

  const int row = int(tgpig) / n_blk;
  const int blk = int(tgpig) - row * n_blk;
  const int K = n_blk * N;
  const int tid = int(sgitg) * NW + int(tiisg);
  const float scale = 1.0f / metal::sqrt(float(N));

  const device T* xrow = x + int64_t(row) * K;
  device T* orow = out + int64_t(row) * K + blk * N;

  float reg[NE];
  for (int j = 0; j < NE; j++) {
    const int p = blk * N + j * NT + tid;
    int src = p;
    if (perm_hd > 0) {
      const int span = perm_rep * perm_hd;
      const int k = p / span;
      const int rem = p - k * span;
      const int r = rem / perm_hd;
      const int h = rem - r * perm_hd;
      src = (r * perm_nk + k) * perm_hd + h;
    }
    const float s = has_signs ? signs[p] : 1.0f;
    reg[j] = float(xrow[src]) * s * scale;
  }

  kq_fwht_regs<N, NT>(reg, shmem, tid);

  for (int j = 0; j < NE; j++) {
    orow[j * NT + tid] = T(reg[j]);
  }
}

// out = H(s * (act(gate) * x)), act silu (ACT 0, the swiglu that feeds a
// down projection) or sigmoid (ACT 1, an attention output gate).
template <int N, int NT, typename T, int ACT>
[[kernel]] void kq_glu_hadamard(
    const device T* x [[buffer(0)]],
    const device T* gate [[buffer(1)]],
    const device float* signs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& n_blk [[buffer(4)]],
    const constant int& has_signs [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]) {
  constexpr int NW = 32;
  constexpr int NE = N / NT;
  static_assert(N % NT == 0 && NT % NW == 0, "block must tile the group");

  threadgroup float shmem[N];

  const int row = int(tgpig) / n_blk;
  const int blk = int(tgpig) - row * n_blk;
  const int K = n_blk * N;
  const int tid = int(sgitg) * NW + int(tiisg);
  const float scale = 1.0f / metal::sqrt(float(N));

  const device T* xrow = x + int64_t(row) * K;
  const device T* grow = gate + int64_t(row) * K;

  float reg[NE];
  for (int j = 0; j < NE; j++) {
    const int p = blk * N + j * NT + tid;
    const float g = float(grow[p]);
    const float a = ACT == 0 ? g * kq_sigmoid(g) : kq_sigmoid(g);
    reg[j] = a * float(xrow[p]) * (has_signs ? signs[p] : 1.0f) * scale;
  }

  kq_fwht_regs<N, NT>(reg, shmem, tid);

  device T* orow = out + int64_t(row) * K + blk * N;
  for (int j = 0; j < NE; j++) {
    orow[j * NT + tid] = T(reg[j]);
  }
}
