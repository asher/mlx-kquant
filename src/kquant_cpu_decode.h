// Scalar CPU decode path for the kq.* ops: GGUF wire bytes -> float for all 10
// codecs, plus a dequant-then-matmul. No Metal dependency, so it builds and
// runs on any platform with the stock mlx wheel; it is also the correctness
// oracle the GPU kernels are A/B'd against. Entry points are template-declared
// here and explicitly instantiated for float / float16_t / bfloat16_t in the
// .cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace mlx_kquant {

// Run fn(begin, end) over a partition of [0, n_items) on the shared CPU
// worker pool (the calling thread participates). Falls back to a single
// inline call when the pool is disabled (KQ_CPU_THREADS=1) or n_items is
// small. NOT reentrant: fn must not itself call kq_parallel_for.
void kq_parallel_for(
    std::size_t n_items,
    const std::function<void(std::size_t, std::size_t)>& fn);

// Same, with a minimum part size: jobs shorter than one `grain` of items run
// inline (waking the pool costs more than they're worth), and no part is
// split finer than `grain`. Use when one item is cheap (e.g. per-element
// dtype casts) so the partitioning reflects actual work.
void kq_parallel_for(
    std::size_t n_items,
    std::size_t grain,
    const std::function<void(std::size_t, std::size_t)>& fn);

// Worker count the pool was built with (KQ_CPU_THREADS env override, else
// std::thread::hardware_concurrency()).
int kq_cpu_threads();

// Dequantize `num_weights` weights of `kquant_type` wire bytes `w` into `out`.
// `w` must hold whole blocks and `num_weights` be a multiple of the codec's
// weights_per_block. Throws on an unknown codec.
template <typename T>
void kquant_dequantize_dispatch(
    const uint8_t* w,
    T* out,
    std::size_t num_weights,
    const std::string& kquant_type);

// kquant_dequantize_dispatch with the block range split across the worker
// pool. Same contract; bit-identical output (per-block decode is independent).
template <typename T>
void kquant_dequantize_parallel(
    const uint8_t* w,
    T* out,
    std::size_t num_weights,
    const std::string& kquant_type);

// result[M, N] = x[M, K] @ dequant(w), accumulating in float. transpose_w=true
// means `w` decodes to [N, K] (the row-major weight convention); otherwise
// [K, N]. `w` is contiguous wire bytes for the full matrix.
template <typename T>
void kquant_qmm_cpu(
    T* result,
    const T* x,
    const uint8_t* w,
    int M,
    int N,
    int K,
    bool transpose_w,
    const std::string& kquant_type);

// Small-M ceiling for the fused decode-then-dot GEMV path inside
// kquant_qmm_cpu; larger M dequantizes once and runs a GEMM instead.
constexpr int kQmvFusedMaxM = 16;

// One GEMV in a batched (MoE decode) call: out[m, N] = x[m, K] @ dequant(w).T
// with per-task row count m <= kQmvFusedMaxM and contiguous [N, K] wire bytes.
template <typename T>
struct KQmvTask {
  T* out; // [m, N]
  const T* x; // [m, K]
  const uint8_t* w; // wire bytes for one [N, K] weight matrix
  int m;
};

// Run a batch of same-codec, same-[N, K], transpose_w=true fused GEMVs as ONE
// parallel job over all (task, output-row) work items, instead of one
// thread-pool job per task. This is the MoE gather decode shape: top-k tiny
// matvecs per call, where per-task pool wake/teardown dominates. Activation
// rows referenced by multiple tasks (same x pointer) are staged/quantized
// once per call.
template <typename T>
void kquant_qmm_cpu_batch(
    const KQmvTask<T>* tasks,
    int n_tasks,
    int N,
    int K,
    const std::string& kquant_type);

} // namespace mlx_kquant
