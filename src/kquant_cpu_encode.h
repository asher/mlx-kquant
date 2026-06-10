// Scalar CPU encode path for kq.quantize: float weights -> GGUF wire bytes for
// the flat codecs (q8_0 + the four legacy block codecs); the K-quant superblock
// codecs are encoded on the GPU only for now (the dispatch throws for them on
// CPU). Ported from the Metal encode kernels (kq_quantized_encode.h) so the CPU
// output is byte-identical to the GPU encoder, and validated against the
// gguf-py reference. No Metal dependency, so it builds and runs on any platform
// with the stock mlx wheel. The entry point is template-declared here and
// explicitly instantiated for float / float16_t / bfloat16_t in the .cpp.
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
// are encoded in one contiguous call. Throws on an unknown codec, and on a
// K-quant codec until the CPU K-quant encoders land.
template <typename T>
void kquant_quantize_dispatch(
    const T* w,
    uint8_t* out,
    std::size_t num_weights,
    const std::string& kquant_type,
    const float* imatrix,
    std::size_t K);

} // namespace mlx_kquant
