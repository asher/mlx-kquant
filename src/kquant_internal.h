// Internal helpers shared across the kq primitive eval paths. Not exported via
// the Python bindings.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mx = mlx::core;

namespace mlx_kquant {

// Map a dtype to the C++ type token the kq Metal kernels are instantiated with:
//   float32 -> "float", float16 -> "float16_t", bfloat16 -> "bfloat16_t".
// Throws on other dtypes.
std::string kq_type_string(mx::Dtype d);

// KVarN key/value widths the record codec supports.
inline bool kq_kvarn_bits_ok(int bits) {
  return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
      bits == 8;
}

// Kernel-name prefix for a codec: "kquant_<codec>_".
inline std::string kq_kname_prefix(const std::string& kquant_type) {
  return "kquant_" + kquant_type + "_";
}

// Byte alignment the GPU kernels need at the start of a packed weight: the
// widest load any kernel of the codec makes through the weight pointer.
// q4_k and q5_k read 16-byte words, iq4_xs and iq1_m 8-byte words, and the
// codecs at 4 read 4-byte words. mxfp4 and nvfp4 read bytes. A start off
// the value is undefined for Metal loads, and on M5 Max the kernels return
// wrong values or NaN from every odd start, and from starts 2 mod 4 on the
// codecs at 4 or more. The CPU kernels accept any start. Each value divides
// the codec's block size, so a weight that starts on a block boundary of an
// aligned buffer passes.
inline int kq_weight_base_align(const std::string& kquant_type) {
  const std::string& t = kquant_type;
  if (t == "q4_k" || t == "q5_k") {
    return 16;
  }
  if (t == "iq4_xs" || t == "iq1_m") {
    return 8;
  }
  if (t == "q2_k" || t == "q4_1" || t == "q5_1" || t == "ptq1_0") {
    return 4;
  }
  if (t == "mxfp4" || t == "nvfp4") {
    return 1;
  }
  return 2;
}

// GPU eval guard: throws std::invalid_argument when packed weight w of
// codec kquant_type starts off kq_weight_base_align. op names the calling
// op in the message. An eval runs it before allocating its output.
inline void kq_check_weight_base(
    const mx::array& w,
    const std::string& kquant_type,
    const char* op) {
  const int align = kq_weight_base_align(kquant_type);
  const uintptr_t off = reinterpret_cast<uintptr_t>(w.data<uint8_t>()) % align;
  if (off != 0) {
    throw std::invalid_argument(
        std::string("[mlx_kquant] ") + op + ": the " + kquant_type +
        " weight starts " + std::to_string(off) +
        (off == 1 ? " byte" : " bytes") + " past a " + std::to_string(align) +
        "-byte boundary. The GPU kernels for " + kquant_type +
        " need it to start on a " + std::to_string(align) +
        "-byte boundary. Copy it into a new buffer, for example with "
        "mx.array(np.array(w)).");
  }
}

// Op-builder form of kq_check_weight_base, so that the common case raises
// from the op call instead of from an eval with other work in flight. It
// checks a weight that already has memory and is row-contiguous, on a GPU
// stream. A weight that is not yet evaluated, or that the op copies to make
// it contiguous, is left to the eval check.
inline void kq_check_weight_base_at_build(
    const mx::array& w,
    const std::string& kquant_type,
    const mx::Stream& s,
    const char* op) {
  if (s.device == mx::Device::gpu &&
      w.status() != mx::array::Status::unscheduled &&
      w.flags().row_contiguous) {
    kq_check_weight_base(w, kquant_type, op);
  }
}

// 64-bit variant of mx::elem_to_loc (mlx/backend/common/utils.h), whose linear
// index parameter is a 32-bit int. That truncates offsets into tensors whose
// flattened element count exceeds 2^31 - e.g. a stacked MoE expert weight whose
// per-expert byte span is itself > 2 GB. This mirrors the stock per-dimension
// loop with a 64-bit index so the decomposition stays exact for large tensors.
inline int64_t elem_to_loc64(
    int64_t elem,
    const mx::Shape& shape,
    const mx::Strides& strides) {
  int64_t loc = 0;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    const int64_t dim = static_cast<int64_t>(shape[i]);
    const int64_t q = elem / dim;
    loc += (elem - q * dim) * strides[i];
    elem = q;
  }
  return loc;
}

} // namespace mlx_kquant
