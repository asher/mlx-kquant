// Scalar CPU encode path for the IQ codecs (kq.quantize): float weights -> GGUF
// IQ wire bytes. ggml has no GPU IQ quantizer, so this is CPU-only; kq.quantize
// forces an IQ request onto a CPU stream at the op level (kquant_ops.cpp). Each
// body is a faithful port of ggml's quantize_row_iq*_impl (llama.cpp, MIT; see
// mlx_kquant/licenses/llama.cpp-LICENSE) -- same scale search, accumulation
// order, and grid lookups -- so the output is byte-exact against llama-quantize
// for the deterministic codecs. This translation unit is built
// -ffp-contract=off to keep that byte-exactness (see CMakeLists.txt). The entry
// point is template-declared here and explicitly instantiated for float /
// float16_t / bfloat16_t in the .cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlx_kquant {

// Encode `num_weights` weights from `w` into `kquant_type` IQ wire bytes `out`.
// `num_weights` must be a multiple of the codec's weights_per_block. `imatrix`
// (length `K`, or nullptr) is the per-column importance vector; it is sliced
// per super-block with column wraparound at `K` (matching the K-quant
// encoders). The codecs ggml marks imatrix-required reject a nullptr at the op
// level before reaching here. Throws on an IQ codec whose encoder is not yet
// implemented.
template <typename T>
void kquant_iq_quantize_dispatch(
    const T* w,
    uint8_t* out,
    std::size_t num_weights,
    const std::string& kquant_type,
    const float* imatrix,
    std::size_t K);

} // namespace mlx_kquant
