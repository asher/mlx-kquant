// Scalar CPU decode path for the kq.* ops: GGUF wire bytes -> float for all 10
// codecs, plus a dequant-then-matmul. No Metal dependency, so it builds and
// runs on any platform with the stock mlx wheel; it is also the correctness
// oracle the GPU kernels are A/B'd against. Entry points are template-declared
// here and explicitly instantiated for float / float16_t / bfloat16_t in the
// .cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlx_kquant {

// Dequantize `num_weights` weights of `kquant_type` wire bytes `w` into `out`.
// `w` must hold whole blocks and `num_weights` be a multiple of the codec's
// weights_per_block. Throws on an unknown codec.
template <typename T>
void kquant_dequantize_dispatch(
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

} // namespace mlx_kquant
