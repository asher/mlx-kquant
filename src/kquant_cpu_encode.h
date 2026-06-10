// Scalar CPU encode path for kq.quantize: float weights -> GGUF wire bytes for
// all ten codecs (q8_0 + the four legacy block codecs + the five K-quant
// superblocks). Ported from the Metal encode kernels (kq_quantized_encode.h):
// the flat codecs and q6_k are byte-identical to the GPU encoder; the four
// codecs that reduce sigma2 (q2_k/q4_k/q5_k, and q3_k under an imatrix) can
// differ by an ULP-tied level but are numerically equivalent. Validated against
// the gguf-py reference. No Metal dependency, so it builds and runs on any
// platform with the stock mlx wheel. The entry point is template-declared here
// and explicitly instantiated for float / float16_t / bfloat16_t in the .cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlx_kquant {

// Encode `num_weights` weights from `w` into `kquant_type` wire bytes `out`.
// `num_weights` must be a multiple of the codec's weights_per_block, and `out`
// must hold the corresponding whole blocks. `imatrix` (length `K`, or nullptr)
// steers the K-quant superblock codecs; the flat codecs ignore it. `K` is the
// logical row width in weights, for the imatrix wraparound when a matrix's rows
// are encoded in one contiguous call. Throws on an unknown codec.
template <typename T>
void kquant_quantize_dispatch(
    const T* w,
    uint8_t* out,
    std::size_t num_weights,
    const std::string& kquant_type,
    const float* imatrix,
    std::size_t K);

} // namespace mlx_kquant
