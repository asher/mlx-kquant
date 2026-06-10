// C++ GGUF loader for the mlx-kquant extension: a fast K-quant load path that
// reads tensor wire bytes via a single C++ memcpy from gguflib's mmap
// (~15 GB/s) and decodes all GGUF KV metadata, so callers need no separate
// Python GGUF reader on the load path.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "mlx/array.h"

namespace mx = mlx::core;

namespace mlx_kquant {

// A decoded GGUF metadata value. GGUF KV value types collapse to
// Python-friendly forms: integer scalars (any width, plus bool) -> int64; float
// scalars (f32/f64) -> double; strings -> std::string; 1-D arrays -> the
// matching vector. (GGUF allows only one level of array nesting.)
using GgufMetaValue = std::variant<
    std::monostate,
    int64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>,
    std::vector<double>,
    std::vector<std::string>>;

struct GgufLoadResult {
  // Tensor arrays. K-quant tensors: uint8 wire bytes in MLX axis order
  // (innermost-last), last dim byte-packed, plus a vestigial 1-byte uint8
  // "<prefix>.scales" placeholder each (signature parity with the kq ops).
  // F32/F16/BF16/I8/I16/I32 tensors: native dtype, MLX axis order.
  std::unordered_map<std::string, mx::array> arrays;
  // K-quant tensor name -> codec name ("q4_k", ...). Only quantized tensors;
  // order preserved from the file.
  std::vector<std::pair<std::string, std::string>> codecs;
  // Decoded GGUF KV metadata (general.architecture, tokenizer.*, rope.*, ...).
  // Order preserved from the file.
  std::vector<std::pair<std::string, GgufMetaValue>> metadata;
  // Logical tensor shapes in GGUF native (innermost-first) order - i.e. the
  // same order gguf-py's ReaderTensor.shape exposes - for config-synth
  // geometry probes. Logical element dims, NOT byte-packed.
  std::vector<std::pair<std::string, std::vector<int64_t>>> tensor_shapes;
};

// Load a GGUF file. Throws std::runtime_error / std::invalid_argument on a
// missing/malformed file or an unsupported tensor type.
//
// When `zero_copy` is true (default), tensor arrays are constructed as no-copy
// views over gguflib's mmap (one Metal `newBufferWithBytesNoCopy` per tensor,
// over a page-aligned window, sliced to the tensor's bytes). The mmap is kept
// alive by a shared_ptr to the gguf_ctx captured in each array's deleter, so it
// is munmap'd only when the last viewing array is freed. Any tensor that can't
// be wrapped no-copy (unaligned window, >INT32_MAX elements) falls back to the
// eager-memcpy constructor. When `zero_copy` is false every tensor is memcpy'd
// (the original ~15 GB/s path) and the mmap is closed before returning.
GgufLoadResult load_gguf(const std::string& path, bool zero_copy = true);

} // namespace mlx_kquant
