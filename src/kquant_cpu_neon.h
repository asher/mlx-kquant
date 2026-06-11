// arm64 NEON-dotprod int8 GEMV kernels for the CPU decode path. The fused
// small-M matmul quantizes each activation row once to an int8 "q8" layout,
// then dots weight wire bytes straight against it with vdotq_s32 instead of
// decoding every block to f32 first. Per-codec kernels exist for the hot
// decode codecs (q4_k / q5_k / q6_k / q8_0); every other codec - and every
// non-arm64 or non-dotprod build - falls back to the portable scalar path.
//
// Numerics: the activation q8 quantization is lossy by design (the same
// trade ggml makes), so matmul results differ from the f32 decode-then-dot
// path at tolerance level. The f32 dequantize() decode is untouched and
// stays bit-exact.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mlx_kquant {

// One int8 fused-GEMV kernel: a per-codec activation quantizer plus a
// whole-row vec-dot of wire bytes against the quantized activations.
struct KQNeonKernel {
  // Dot one weight row (`nblocks` codec blocks of contiguous wire bytes)
  // against one pre-quantized activation row.
  float (*vec_dot)(const uint8_t* w_row, const void* act_row, int nblocks);
  // Bytes per codec block in the q8 activation layout (row stride =
  // nblocks * act_block_bytes; rows stay 4-byte aligned).
  std::size_t act_block_bytes;
  // Quantize one float activation row of `k` elements (a whole number of
  // codec blocks) into the codec's q8 layout.
  void (*quantize_act_row)(const float* x, void* act, int k);
};

#ifdef KQ_CPU_NEON_TU
// Kernel for `codec`, or nullptr when the codec has no int8 kernel, the CPU
// lacks the dotprod extension (Linux aarch64 hwcap check), or KQ_CPU_NEON=0.
const KQNeonKernel* kq_neon_kernel(const std::string& codec);

// True when the NEON int8 kernels can run here (build + CPU + kill switch);
// per-codec coverage still goes through kq_neon_kernel.
bool kq_cpu_neon_available();
#else
// Non-arm64 builds: the TU is excluded, the scalar path always runs.
inline const KQNeonKernel* kq_neon_kernel(const std::string&) {
  return nullptr;
}

inline bool kq_cpu_neon_available() {
  return false;
}
#endif

} // namespace mlx_kquant
