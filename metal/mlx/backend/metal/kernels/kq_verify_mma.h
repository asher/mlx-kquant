#include <metal_simdgroup_matrix>

// Register-resident simdgroup-MMA verify kernels for the M <= 8 band.
//
// Each of the 8 simdgroups owns NT tiles of 8 weight rows. A block of
// every row is decoded straight into 8x8 A fragments (rows n, cols k) by
// lane-owned bytes, the activations are staged once per K chunk as B^T
// (rows k, cols m) in threadgroup memory in natural k order, and one
// simdgroup_multiply_accumulate per fragment accumulates D (rows n, cols
// m). The codes decode exactly in half. The activations stay in their own
// dtype in threadgroup memory, both operands widen to float in the
// fragment, and each block sums in float; the block scale is applied to it
// in float. The KQ_VERIFY_MMA_HALF kernels (HALF, name suffix _h) stage
// the activations as half and sum each block in half, whose range stops
// at 65504; for them, a codec whose code magnitudes could carry a block's
// sum out of half range decodes scaled-down codes and restores the factor
// through d_scale. Split-K over grid z writes U partials that
// kquant_qmm_splitk_accum folds.
//
// The k order inside a block is a fixed per-codec permutation
// (Codec::perm) chosen so each lane's fragment elements are the code pairs
// it can extract cheapest; the B fragment of fragment f reads the staged
// row perm(f, fm), so the dot is unchanged. Fragment layout is MLX steel's
// BaseMMAFrag<T, 8, 8>: lane l holds row fm = (l/4 & 4) + (l/2 % 4) and
// columns fn = (l/4 & 2) * 2 + (l % 2) * 2, fn + 1 of every operand.
//
// Codec contract:
//   block_k, block_bytes, d_offset, d_scale (factor on the block scale)
//   perm(f, col): element index of fragment f (0..block_k/8-1), column col
//   block<NT>(rows, boff, L, fm, xb, acc): decode block boff of each row
//     and accumulate its block_k/8 fragments through kq_vmma_step; L =
//     fn / 2 is the lane's column pair, xb the staged B base of this block
//     (fragment f at xb + 8 * perm(f, fm)).
//   has_min, m_offset (optional, block_k 32, float kernels only): the
//     block also has a half min m at m_offset, and w = d * code + m. The
//     staging pass then sums each block's activations per column in float
//     into threadgroup memory, and each block adds m times that sum to its
//     scaled float sum. Codecs without has_min compile without the sums.

MLX_MTL_CONST int KQ_VMMA_THREADS = 256;
MLX_MTL_CONST int KQ_VMMA_NSG = KQ_VMMA_THREADS / 32;
MLX_MTL_CONST int KQ_VMMA_KC = 512;

template <int NT>
struct KqVmmaRows {
  const device uint8_t* p[NT];
};

// Staged activation type X and accumulator type A: T and float, or half
// and half for the KQ_VERIFY_MMA_HALF kernels.
template <typename T, bool HALF>
struct KqVmmaTypes {
  using X = metal::conditional_t<HALF, half, T>;
  using A = metal::conditional_t<HALF, half, float>;
};

// Codec::has_min, false when the codec does not declare it.
template <typename>
struct KqVmmaVoid {
  using type = void;
};
template <typename Codec, typename = void>
struct KqVmmaHasMin {
  static constant constexpr bool value = false;
};
template <typename Codec>
struct KqVmmaHasMin<
    Codec,
    typename KqVmmaVoid<decltype(Codec::has_min)>::type> {
  static constant constexpr bool value = Codec::has_min;
};

// Staged activation elements per threadgroup: the K chunk, and for a codec
// with a min the per-block column sums (KC / 32 blocks of 8 floats) after
// it.
template <typename T, typename Codec, bool HALF>
constexpr int kq_vmma_stage_elems() {
  using X = typename KqVmmaTypes<T, HALF>::X;
  return KQ_VMMA_KC * 8 +
      (KqVmmaHasMin<Codec>::value
           ? KQ_VMMA_KC / 32 * 8 * int(sizeof(float) / sizeof(X))
           : 0);
}

// One fragment step: the lane's two staged activations at bsrc and its
// decoded code pair a[t] of each row tile, widened to A.
template <int NT, typename X, typename A>
METAL_FUNC void kq_vmma_step(
    const threadgroup X* bsrc,
    thread const half2 (&a)[NT],
    thread simdgroup_matrix<A, 8, 8> (&acc)[NT]) {
  simdgroup_matrix<A, 8, 8> B;
  const vec<X, 2> b = *(const threadgroup vec<X, 2>*)bsrc;
  reinterpret_cast<thread vec<A, 2>&>(B.thread_elements()) =
      vec<A, 2>(static_cast<A>(b.x), static_cast<A>(b.y));
  for (short t = 0; t < NT; ++t) {
    simdgroup_matrix<A, 8, 8> W;
    reinterpret_cast<thread vec<A, 2>&>(W.thread_elements()) =
        vec<A, 2>(static_cast<A>(a[t].x), static_cast<A>(a[t].y));
    simdgroup_multiply_accumulate(acc[t], W, B, acc[t]);
  }
}

template <typename T, typename Codec, int NT, typename U = T, bool HALF = false>
METAL_FUNC void kq_verify_mma_impl(
    const device uint8_t* w,
    const device T* x,
    device U* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& k_partition_size,
    threadgroup typename KqVmmaTypes<T, HALF>::X* XsT,
    uint3 tid,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  using X = typename KqVmmaTypes<T, HALF>::X;
  using A = typename KqVmmaTypes<T, HALF>::A;
  constexpr int KC = KQ_VMMA_KC;
  constexpr int BK = Codec::block_k;
  constexpr bool MIN = KqVmmaHasMin<Codec>::value;
  static_assert(!MIN || (BK == 32 && !HALF), "min codecs: float, 32-wide");
  // Column sums of each staged block, after the chunk.
  threadgroup float* Xsum = (threadgroup float*)(XsT + KC * 8);
  const int row_bytes = (K / BK) * Codec::block_bytes;
  const short qid = simd_lid / 4;
  const short fm = (qid & 4) + ((simd_lid / 2) % 4);
  const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
  const short L = fn / 2;
  const int n = tid.x * (KQ_VMMA_NSG * 8 * NT) + simd_gid * 8 * NT + fm;
  KqVmmaRows<NT> rows;
  for (short t = 0; t < NT; ++t) {
    rows.p[t] = w + static_cast<int64_t>(min(n + 8 * t, N - 1)) * row_bytes;
  }
  const int kbeg = tid.z * k_partition_size;
  const int kend = kbeg + k_partition_size;
  float2 c[NT];
  for (short t = 0; t < NT; ++t) {
    c[t] = float2(0.0f);
  }
  for (int kc = kbeg; kc < kend; kc += KC) {
    const int kn = min(KC, kend - kc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = lid; kk < kn; kk += KQ_VMMA_THREADS) {
      const device T* xe = x + kc + kk;
      vec<X, 8> v;
      for (short m = 0; m < 8; ++m) {
        if constexpr (HALF) {
          v[m] = m < M ? half(float(xe[m * K])) : half(0);
        } else {
          v[m] = m < M ? xe[m * K] : T(0);
        }
      }
      *(threadgroup vec<X, 8>*)(XsT + kk * 8) = v;
      if constexpr (MIN) {
        // A simdgroup stages 32 consecutive k, one whole block, since kn
        // and the chunk start are multiples of 32.
        float4 s0 = float4(float(v[0]), float(v[1]), float(v[2]), float(v[3]));
        float4 s1 = float4(float(v[4]), float(v[5]), float(v[6]), float(v[7]));
        s0 = simd_sum(s0);
        s1 = simd_sum(s1);
        if (simd_lid == 0) {
          *(threadgroup float4*)(Xsum + (kk / 32) * 8) = s0;
          *(threadgroup float4*)(Xsum + (kk / 32) * 8 + 4) = s1;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kb = 0; kb < kn; kb += BK) {
      const int boff = ((kc + kb) / BK) * Codec::block_bytes;
      const threadgroup X* xb = XsT + kb * 8 + fn;
      simdgroup_matrix<A, 8, 8> acc[NT];
      for (short t = 0; t < NT; ++t) {
        acc[t] = make_filled_simdgroup_matrix<A, 8, 8>(A(0));
      }
      Codec::template block<NT>(rows, boff, L, fm, xb, acc);
      for (short t = 0; t < NT; ++t) {
        const float d = Codec::d_scale *
            float(*(const device half*)(rows.p[t] + boff + Codec::d_offset));
        c[t] += d *
            float2(reinterpret_cast<thread vec<A, 2>&>(
                acc[t].thread_elements()));
        if constexpr (MIN) {
          const float mn =
              float(*(const device half*)(rows.p[t] + boff + Codec::m_offset));
          c[t] += mn * *(const threadgroup float2*)(Xsum + (kb / 32) * 8 + fn);
        }
      }
    }
  }
  for (short t = 0; t < NT; ++t) {
    const int nt = n + 8 * t;
    if (nt < N) {
      if (fn < M) {
        y[fn * N + nt] = static_cast<U>(c[t].x);
      }
      if (fn + 1 < M) {
        y[(fn + 1) * N + nt] = static_cast<U>(c[t].y);
      }
    }
  }
}

// Entry point: qmm_t_splitk's buffer layout, grid (ceil(N / (64 NT)), 1,
// splits), 256 threads.
#define KQ_DEFINE_VERIFY_MMA_KERNEL(CODEC, TRAITS, NT)                   \
  template <                                                             \
      typename T,                                                        \
      int group_size,                                                    \
      int bits,                                                          \
      typename U = T,                                                    \
      bool HALF = false>                                                 \
  [[kernel]] void kq_##CODEC##_verify_mma(                               \
      const device uint8_t* w,                                           \
      const device uint8_t* /* scales */,                                \
      const device T* x,                                                 \
      device U* y,                                                       \
      const constant int& K,                                             \
      const constant int& N,                                             \
      const constant int& M,                                             \
      const constant int& k_partition_size,                              \
      const constant int& split_k_partition_stride,                      \
      uint3 tid [[threadgroup_position_in_grid]],                        \
      uint lid [[thread_index_in_threadgroup]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                  \
      uint simd_lid [[thread_index_in_simdgroup]]) {                     \
    static_assert(group_size == TRAITS::block_k, #CODEC " block width"); \
    threadgroup typename KqVmmaTypes<T, HALF>::X                         \
        XsT[kq_vmma_stage_elems<T, TRAITS, HALF>()];                     \
    kq_verify_mma_impl<T, TRAITS, NT, U, HALF>(                          \
        w,                                                               \
        x,                                                               \
        y + tid.z * static_cast<int64_t>(split_k_partition_stride),      \
        K,                                                               \
        N,                                                               \
        M,                                                               \
        k_partition_size,                                                \
        XsT,                                                             \
        tid,                                                             \
        lid,                                                             \
        simd_gid,                                                        \
        simd_lid);                                                       \
  }
