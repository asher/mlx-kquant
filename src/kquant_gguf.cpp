// GGUF loader implementation: metadata walk, get_shape, direct-dtype tensors,
// and the K-quant wire-byte load path. gguflib is the antirez/gguf-tools
// single-file parser (FetchContent-pinned in CMake); it mmaps the file, so
// tensor.weights_data points into the mmap. By default each tensor is a NO-COPY
// view over that mmap (see try_zero_copy_array); with zero_copy=false it is
// memcpy'd out via the array(It, shape, dtype) constructor (one memcpy,
// ~15 GB/s).
#include <algorithm>
#include <climits>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include <unistd.h> // getpagesize

#include "kquant_codec.h"
#include "kquant_gguf.h"

#include "mlx/allocator.h"
#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "mlx/types/half_types.h"

extern "C" {
#include <gguflib.h>
}

namespace mlx_kquant {

namespace {

constexpr int kGgufArrayHeaderSize = 12; // 4-byte elem type + 8-byte length.

// Live zero-copy tensor ranges: the tensor's exact wire bytes and dtype (NOT
// the enclosing page window -- adjacent tensors can share a page, and exact
// ranges keep the dtype unambiguous). Registered when a no-copy view is
// built, erased by the view's buffer deleter. Backs verify_zero_copy_views().
struct ZcRange {
  uintptr_t base;
  size_t bytes;
  mx::Dtype dtype;
};
std::mutex g_zc_mutex;
std::vector<ZcRange> g_zc_ranges;

void zc_register(uintptr_t base, size_t bytes, mx::Dtype dtype) {
  std::lock_guard<std::mutex> lock(g_zc_mutex);
  g_zc_ranges.push_back({base, bytes, dtype});
}

void zc_unregister(uintptr_t base) {
  std::lock_guard<std::mutex> lock(g_zc_mutex);
  for (auto it = g_zc_ranges.begin(); it != g_zc_ranges.end(); ++it) {
    if (it->base == base) {
      g_zc_ranges.erase(it);
      return;
    }
  }
}

// Local dtype name map (mlx::core::dtype_to_string is not exported from the
// wheel dylib).
const char* zc_dtype_name(mx::Dtype d) {
  if (d == mx::float32)
    return "float32";
  if (d == mx::float16)
    return "float16";
  if (d == mx::bfloat16)
    return "bfloat16";
  if (d == mx::float64)
    return "float64";
  if (d == mx::uint8)
    return "uint8";
  if (d == mx::uint16)
    return "uint16";
  if (d == mx::uint32)
    return "uint32";
  if (d == mx::uint64)
    return "uint64";
  if (d == mx::int8)
    return "int8";
  if (d == mx::int16)
    return "int16";
  if (d == mx::int32)
    return "int32";
  if (d == mx::int64)
    return "int64";
  if (d == mx::bool_)
    return "bool";
  if (d == mx::complex64)
    return "complex64";
  return "unknown";
}

// Build a no-copy mx.array viewing `nbytes` of `dtype`/`shape` starting at `wd`
// (a pointer into gguflib's mmap). Wraps a page-aligned window enclosing the
// tensor in a Metal shared-storage buffer via allocator::make_buffer (=
// newBufferWithBytesNoCopy), then slices/reshapes to the tensor - all no-copy.
// The window array's deleter releases that buffer and drops a ref to the
// captured gguf_ctx, so the mmap survives exactly as long as some viewing array
// does. Returns nullopt when a no-copy wrap isn't possible (unaligned window or
// >INT32_MAX elements past the page-aligned base); the caller then memcpy's.
//
// Alignment reasoning: gguflib mmaps at a page-aligned base and GGUF tensor
// data sits at a 32-byte-aligned file offset, so `wd` is 32-aligned -> win_off
// is a multiple of every dtype's itemsize (1/2/4/8) and win_base is
// page-aligned (the pointer-alignment newBufferWithBytesNoCopy requires).
std::optional<mx::array> try_zero_copy_array(
    const void* wd,
    size_t nbytes,
    mx::Dtype dtype,
    const mx::Shape& shape,
    const std::shared_ptr<gguf_ctx>& ctx_holder) {
  if (wd == nullptr || nbytes == 0) {
    return std::nullopt;
  }
  const size_t isz = mx::size_of(dtype);
  if (nbytes % isz != 0) {
    return std::nullopt;
  }

  const auto addr = reinterpret_cast<uintptr_t>(wd);
  const size_t page = static_cast<size_t>(getpagesize());
  const uintptr_t win_base = addr & ~(static_cast<uintptr_t>(page) - 1);
  const size_t win_off = static_cast<size_t>(addr - win_base);
  if (win_off % isz != 0) {
    return std::nullopt; // 32-byte GGUF alignment should make this unreachable.
  }

  const size_t win_bytes = win_off + nbytes;
  const size_t win_elems = win_bytes / isz;
  const size_t off_elems = win_off / isz;
  const size_t num_elems = nbytes / isz;
  if (win_elems > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return std::nullopt; // mx::Shape elements are int32.
  }

  mx::allocator::Buffer buf =
      mx::allocator::make_buffer(reinterpret_cast<void*>(win_base), win_bytes);
  if (buf.ptr() == nullptr) {
    return std::nullopt; // no-copy wrap rejected (e.g. alignment).
  }

  zc_register(addr, nbytes, dtype);
  mx::Deleter del = [ctx_holder, addr](mx::allocator::Buffer b) {
    zc_unregister(addr);
    mx::allocator::release(b);
  };
  mx::array window(buf, mx::Shape{static_cast<int>(win_elems)}, dtype, del);
  mx::array view = mx::slice(
      window,
      mx::Shape{static_cast<int>(off_elems)},
      mx::Shape{static_cast<int>(off_elems + num_elems)});
  return mx::reshape(view, shape);
}

// Map a GGUF tensor type to one of the extension's supported codecs, or
// nullptr for non-quantized / unsupported types.
const KQuantCodec* gguf_type_to_kquant_codec(uint32_t t) {
  switch (t) {
    case GGUF_TYPE_Q4_0:
      return codec_by_name("q4_0");
    case GGUF_TYPE_Q4_1:
      return codec_by_name("q4_1");
    case GGUF_TYPE_Q5_0:
      return codec_by_name("q5_0");
    case GGUF_TYPE_Q5_1:
      return codec_by_name("q5_1");
    case GGUF_TYPE_Q8_0:
      return codec_by_name("q8_0");
    case GGUF_TYPE_Q2_K:
      return codec_by_name("q2_k");
    case GGUF_TYPE_Q3_K:
      return codec_by_name("q3_k");
    case GGUF_TYPE_Q4_K:
      return codec_by_name("q4_k");
    case GGUF_TYPE_Q5_K:
      return codec_by_name("q5_k");
    case GGUF_TYPE_Q6_K:
      return codec_by_name("q6_k");
    case GGUF_TYPE_IQ4_NL:
      return codec_by_name("iq4_nl");
    case GGUF_TYPE_IQ4_XS:
      return codec_by_name("iq4_xs");
    case GGUF_TYPE_IQ3_S:
      return codec_by_name("iq3_s");
    case GGUF_TYPE_IQ3_XXS:
      return codec_by_name("iq3_xxs");
    case GGUF_TYPE_IQ2_XXS:
      return codec_by_name("iq2_xxs");
    case GGUF_TYPE_IQ2_XS:
      return codec_by_name("iq2_xs");
    case GGUF_TYPE_IQ2_S:
      return codec_by_name("iq2_s");
    case GGUF_TYPE_IQ1_S:
      return codec_by_name("iq1_s");
    case GGUF_TYPE_IQ1_M:
      return codec_by_name("iq1_m");
    // GGML type numbers (ggml.h): MXFP4=39, NVFP4=40. Literals so this
    // compiles against a gguflib whose enum predates them (an unpatched
    // parser would have already stopped at the tensor anyway).
    case 39:
      return codec_by_name("mxfp4");
    case 40:
      return codec_by_name("nvfp4");
    default:
      return nullptr;
  }
}

// Non-quantized GGUF tensor types we pass through with their native dtype. BF16
// is kept as native bfloat16 (not down-converted to float16).
std::optional<mx::Dtype> gguf_type_to_dtype(uint32_t t) {
  switch (t) {
    case GGUF_TYPE_F32:
      return mx::float32;
    case GGUF_TYPE_F16:
      return mx::float16;
    case GGUF_TYPE_BF16:
      return mx::bfloat16;
    case GGUF_TYPE_I8:
      return mx::int8;
    case GGUF_TYPE_I16:
      return mx::int16;
    case GGUF_TYPE_I32:
      return mx::int32;
    default:
      return std::nullopt;
  }
}

// MLX axis order = reverse of GGUF (innermost-first -> innermost-last).
mx::Shape mlx_shape(const gguf_tensor& tensor) {
  mx::Shape shape;
  for (int i = static_cast<int>(tensor.ndim) - 1; i >= 0; --i) {
    shape.push_back(static_cast<mx::ShapeElem>(tensor.dim[i]));
  }
  return shape;
}

std::string strip_weight_suffix(const std::string& name) {
  constexpr std::string_view suffix = ".weight";
  if (name.size() > suffix.size() &&
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return name.substr(0, name.size() - suffix.size());
  }
  return name;
}

// Build a native-dtype array from a tensor's wire bytes: a no-copy mmap view
// when zero_copy is set and the wrap succeeds, else the copying constructor.
mx::array make_direct_array(
    const gguf_tensor& tensor,
    mx::Dtype dtype,
    const mx::Shape& shape,
    bool zero_copy,
    const std::shared_ptr<gguf_ctx>& ctx_holder) {
  const void* wd = tensor.weights_data;
  if (wd == nullptr) {
    throw std::runtime_error(
        "[load_gguf] NULL tensor data pointer for " +
        std::string(tensor.name, tensor.namelen));
  }
  if (zero_copy) {
    if (auto a =
            try_zero_copy_array(wd, tensor.bsize, dtype, shape, ctx_holder)) {
      return *a;
    }
  }
  if (dtype == mx::float32) {
    return mx::array(static_cast<const float*>(wd), shape, mx::float32);
  }
  if (dtype == mx::float16) {
    return mx::array(static_cast<const mx::float16_t*>(wd), shape, mx::float16);
  }
  if (dtype == mx::bfloat16) {
    return mx::array(
        static_cast<const mx::bfloat16_t*>(wd), shape, mx::bfloat16);
  }
  if (dtype == mx::int8) {
    return mx::array(static_cast<const int8_t*>(wd), shape, mx::int8);
  }
  if (dtype == mx::int16) {
    return mx::array(static_cast<const int16_t*>(wd), shape, mx::int16);
  }
  if (dtype == mx::int32) {
    return mx::array(static_cast<const int32_t*>(wd), shape, mx::int32);
  }
  throw std::runtime_error("[load_gguf] unhandled direct dtype");
}

// Load one block-quantized tensor as uint8 wire bytes (MLX order, byte-packed
// last dim) + a 1-byte scales placeholder. All block codecs - K-quants and the
// native-fp codecs mxfp4/nvfp4 - route here via the registry geometry; a
// Python loader may still de-interleave the native-fp wire bytes into MLX's
// packed layout downstream (gguf-mlx packed mode).
void load_block_tensor(
    GgufLoadResult& res,
    const gguf_tensor& tensor,
    int weights_per_block,
    int bytes_per_block,
    const std::string& codec_name,
    const std::string& name,
    bool zero_copy,
    const std::shared_ptr<gguf_ctx>& ctx_holder) {
  mx::Shape logical = mlx_shape(tensor);
  if (logical.empty()) {
    throw std::runtime_error(
        "[load_gguf] block tensor " + name + " has no dimensions");
  }
  auto last_dim = logical.back();
  if (last_dim % weights_per_block != 0) {
    std::ostringstream msg;
    msg << "[load_gguf] block tensor " << name << " last dim " << last_dim
        << " is not divisible by weights_per_block " << weights_per_block
        << " for codec " << codec_name;
    throw std::runtime_error(msg.str());
  }
  auto bytes_per_row = (last_dim / weights_per_block) * bytes_per_block;
  mx::Shape packed = logical;
  packed.back() = bytes_per_row;

  size_t total_bytes = std::accumulate(
      packed.begin(),
      packed.end(),
      static_cast<size_t>(1),
      std::multiplies<size_t>());
  if (total_bytes != tensor.bsize) {
    std::ostringstream msg;
    msg << "[load_gguf] block tensor " << name << " (" << codec_name
        << ") computed byte size " << total_bytes
        << " does not match tensor.bsize " << tensor.bsize;
    throw std::runtime_error(msg.str());
  }
  if (tensor.weights_data == nullptr) {
    throw std::runtime_error(
        "[load_gguf] NULL tensor data pointer for " + name);
  }

  mx::array packed_arr = [&]() -> mx::array {
    if (zero_copy) {
      if (auto a = try_zero_copy_array(
              tensor.weights_data,
              tensor.bsize,
              mx::uint8,
              packed,
              ctx_holder)) {
        return *a;
      }
    }
    return mx::array(
        static_cast<const uint8_t*>(tensor.weights_data), packed, mx::uint8);
  }();
  res.arrays.emplace(name, std::move(packed_arr));

  uint8_t zero = 0;
  res.arrays.emplace(
      strip_weight_suffix(name) + ".scales",
      mx::array(&zero, mx::Shape{1}, mx::uint8));

  res.codecs.emplace_back(name, codec_name);
}

// Byte size of a fixed-width GGUF scalar value type (0 for STRING/ARRAY).
uint64_t gguf_scalar_size(uint32_t t) {
  switch (t) {
    case GGUF_VALUE_TYPE_UINT8:
    case GGUF_VALUE_TYPE_INT8:
    case GGUF_VALUE_TYPE_BOOL:
      return 1;
    case GGUF_VALUE_TYPE_UINT16:
    case GGUF_VALUE_TYPE_INT16:
      return 2;
    case GGUF_VALUE_TYPE_UINT32:
    case GGUF_VALUE_TYPE_INT32:
    case GGUF_VALUE_TYPE_FLOAT32:
      return 4;
    case GGUF_VALUE_TYPE_UINT64:
    case GGUF_VALUE_TYPE_INT64:
    case GGUF_VALUE_TYPE_FLOAT64:
      return 8;
    default:
      return 0;
  }
}

// Read a fixed-width integer (or bool) from a (possibly unaligned) pointer,
// widening to int64. memcpy avoids unaligned-load UB.
int64_t read_int_at(const char* p, uint32_t t) {
  switch (t) {
    case GGUF_VALUE_TYPE_UINT8:
    case GGUF_VALUE_TYPE_BOOL: {
      uint8_t v;
      std::memcpy(&v, p, 1);
      return static_cast<int64_t>(v);
    }
    case GGUF_VALUE_TYPE_INT8: {
      int8_t v;
      std::memcpy(&v, p, 1);
      return static_cast<int64_t>(v);
    }
    case GGUF_VALUE_TYPE_UINT16: {
      uint16_t v;
      std::memcpy(&v, p, 2);
      return static_cast<int64_t>(v);
    }
    case GGUF_VALUE_TYPE_INT16: {
      int16_t v;
      std::memcpy(&v, p, 2);
      return static_cast<int64_t>(v);
    }
    case GGUF_VALUE_TYPE_UINT32: {
      uint32_t v;
      std::memcpy(&v, p, 4);
      return static_cast<int64_t>(v);
    }
    case GGUF_VALUE_TYPE_INT32: {
      int32_t v;
      std::memcpy(&v, p, 4);
      return static_cast<int64_t>(v);
    }
    case GGUF_VALUE_TYPE_UINT64: {
      uint64_t v;
      std::memcpy(&v, p, 8);
      return static_cast<int64_t>(v);
    }
    case GGUF_VALUE_TYPE_INT64: {
      int64_t v;
      std::memcpy(&v, p, 8);
      return v;
    }
    default:
      return 0;
  }
}

double read_float_at(const char* p, uint32_t t) {
  if (t == GGUF_VALUE_TYPE_FLOAT32) {
    float v;
    std::memcpy(&v, p, 4);
    return static_cast<double>(v);
  }
  double v;
  std::memcpy(&v, p, 8); // FLOAT64
  return v;
}

// Decode one GGUF KV value into a GgufMetaValue, advancing ctx->off past the
// value bytes (gguflib's gguf_get_key relies on the caller advancing ctx->off
// past each value).
void read_metadata_value(
    gguf_ctx* ctx,
    uint32_t type,
    gguf_value* val,
    GgufMetaValue& out) {
  switch (type) {
    case GGUF_VALUE_TYPE_UINT8:
    case GGUF_VALUE_TYPE_INT8:
    case GGUF_VALUE_TYPE_UINT16:
    case GGUF_VALUE_TYPE_INT16:
    case GGUF_VALUE_TYPE_UINT32:
    case GGUF_VALUE_TYPE_INT32:
    case GGUF_VALUE_TYPE_UINT64:
    case GGUF_VALUE_TYPE_INT64:
      out = read_int_at(reinterpret_cast<const char*>(val), type);
      ctx->off += gguf_scalar_size(type);
      break;
    case GGUF_VALUE_TYPE_FLOAT32:
    case GGUF_VALUE_TYPE_FLOAT64:
      out = read_float_at(reinterpret_cast<const char*>(val), type);
      ctx->off += gguf_scalar_size(type);
      break;
    case GGUF_VALUE_TYPE_BOOL:
      out = static_cast<bool>(val->boolval);
      ctx->off += 1;
      break;
    case GGUF_VALUE_TYPE_STRING:
      out =
          std::string(val->string.string, static_cast<size_t>(val->string.len));
      ctx->off += sizeof(gguf_string) + val->string.len;
      break;
    case GGUF_VALUE_TYPE_ARRAY: {
      ctx->off += kGgufArrayHeaderSize; // skip elem-type + length header
      char* data = reinterpret_cast<char*>(val) + kGgufArrayHeaderSize;
      uint64_t n = val->array.len;
      uint32_t et = val->array.type;
      switch (et) {
        case GGUF_VALUE_TYPE_STRING: {
          std::vector<std::string> strs;
          strs.reserve(n);
          for (uint64_t i = 0; i < n; ++i) {
            auto* s = reinterpret_cast<gguf_string*>(data);
            strs.emplace_back(s->string, static_cast<size_t>(s->len));
            uint64_t adv = sizeof(gguf_string) + s->len;
            data += adv;
            ctx->off += adv;
          }
          out = std::move(strs);
          break;
        }
        case GGUF_VALUE_TYPE_FLOAT32:
        case GGUF_VALUE_TYPE_FLOAT64: {
          std::vector<double> v;
          v.reserve(n);
          uint64_t es = gguf_scalar_size(et);
          for (uint64_t i = 0; i < n; ++i) {
            v.push_back(read_float_at(data + i * es, et));
          }
          ctx->off += n * es;
          out = std::move(v);
          break;
        }
        case GGUF_VALUE_TYPE_ARRAY:
          throw std::invalid_argument(
              "[load_gguf] only one level of nested arrays is supported");
        default: { // integer / bool element types
          uint64_t es = gguf_scalar_size(et);
          if (es == 0) {
            throw std::runtime_error(
                "[load_gguf] unsupported array element type");
          }
          std::vector<int64_t> v;
          v.reserve(n);
          for (uint64_t i = 0; i < n; ++i) {
            v.push_back(read_int_at(data + i * es, et));
          }
          ctx->off += n * es;
          out = std::move(v);
          break;
        }
      }
      break;
    }
    default:
      throw std::runtime_error("[load_gguf] unexpected metadata value type");
  }
}

} // namespace

GgufLoadResult load_gguf(const std::string& path, bool zero_copy /* = true */) {
  {
    std::ifstream f(path.c_str());
    if (!f.good()) {
      throw std::invalid_argument("[load_gguf] failed to open " + path);
    }
  }

  // shared_ptr (not unique_ptr): zero-copy tensor arrays capture it in their
  // deleters so the mmap outlives this function and is gguf_close'd (munmap'd)
  // only when the last viewing array is freed. With zero_copy=false no array
  // captures it, so it is closed when this local ref drops on return.
  // Read-only mapping (default PROT_READ/MAP_SHARED; see KQ_GGUF_MMAP in
  // cmake/patch-gguflib.cmake): tensor pages stay clean, evictable file-cache
  // pages and the source file can never be written through a zero-copy view.
  std::shared_ptr<gguf_ctx> ctx(gguf_open_ro(path.c_str()), gguf_close);
  if (!ctx) {
    throw std::runtime_error("[load_gguf] gguf_open_ro failed for " + path);
  }

  GgufLoadResult res;

  // 1. metadata KVs (must be fully consumed before tensor infos so gguflib
  //    positions ctx->off at the tensor-info section).
  gguf_key key;
  while (gguf_get_key(ctx.get(), &key)) {
    std::string key_name(key.name, key.namelen);
    GgufMetaValue mv;
    read_metadata_value(ctx.get(), key.type, key.val, mv);
    res.metadata.emplace_back(std::move(key_name), std::move(mv));
  }

  // 2. tensors.
  gguf_tensor tensor;
  while (gguf_get_tensor(ctx.get(), &tensor)) {
    std::string name(tensor.name, tensor.namelen);

    // Logical shape in GGUF native order (matches gguf-py ReaderTensor.shape).
    std::vector<int64_t> native_shape;
    native_shape.reserve(tensor.ndim);
    for (uint32_t i = 0; i < tensor.ndim; ++i) {
      native_shape.push_back(static_cast<int64_t>(tensor.dim[i]));
    }
    res.tensor_shapes.emplace_back(name, std::move(native_shape));

    if (const KQuantCodec* codec = gguf_type_to_kquant_codec(tensor.type)) {
      load_block_tensor(
          res,
          tensor,
          codec->weights_per_block,
          codec->bytes_per_block,
          codec->name,
          name,
          zero_copy,
          ctx);
    } else if (auto dtype = gguf_type_to_dtype(tensor.type)) {
      res.arrays.emplace(
          name,
          make_direct_array(tensor, *dtype, mlx_shape(tensor), zero_copy, ctx));
    } else {
      std::ostringstream msg;
      msg << "[load_gguf] tensor " << name << " has unsupported type "
          << tensor.type
          << " (not a K-quant / MXFP4 / NVFP4 codec or a "
             "F32/F16/BF16/I8/I16/I32 dtype)";
      throw std::runtime_error(msg.str());
    }
  }

  return res;
}

size_t zero_copy_view_count() {
  std::lock_guard<std::mutex> lock(g_zc_mutex);
  return g_zc_ranges.size();
}

std::vector<std::string> verify_zero_copy_views(
    const std::vector<std::pair<std::string, mx::array>>& items,
    const std::vector<std::string>& no_alias) {
  std::vector<ZcRange> ranges;
  {
    std::lock_guard<std::mutex> lock(g_zc_mutex);
    ranges = g_zc_ranges;
  }
  std::sort(
      ranges.begin(), ranges.end(), [](const ZcRange& a, const ZcRange& b) {
        return a.base < b.base;
      });
  std::unordered_set<std::string> must_own(no_alias.begin(), no_alias.end());

  // Tensor ranges never overlap, so the candidate is the last range starting
  // at or before p.
  auto find_range = [&](uintptr_t p) -> const ZcRange* {
    auto it = std::upper_bound(
        ranges.begin(), ranges.end(), p, [](uintptr_t v, const ZcRange& r) {
          return v < r.base;
        });
    if (it == ranges.begin()) {
      return nullptr;
    }
    --it;
    return p < it->base + it->bytes ? &*it : nullptr;
  };

  std::vector<std::string> problems;
  for (const auto& [name, arr] : items) {
    if (!arr.is_available()) {
      problems.push_back(name + ": unevaluated (mx.eval the arrays first)");
      continue;
    }
    const auto p = reinterpret_cast<uintptr_t>(arr.data<char>());
    const ZcRange* r = find_range(p);
    if (r == nullptr) {
      continue; // owned buffer
    }
    if (must_own.count(name)) {
      problems.push_back(
          name + ": transformed param still aliases the mapping");
      continue;
    }
    // Integer<->integer reinterprets (uint8 wire viewed as uint32 for packed
    // layouts) are legitimate; any float-involved dtype change on mapped
    // bytes is the donation-corruption signature.
    const bool both_int = mx::issubdtype(arr.dtype(), mx::integer) &&
        mx::issubdtype(r->dtype, mx::integer);
    if (arr.dtype() != r->dtype && !both_int) {
      problems.push_back(
          name + ": mapped view dtype " + zc_dtype_name(arr.dtype()) +
          " != wire dtype " + zc_dtype_name(r->dtype) +
          " (buffer donation through the mapping?)");
    }
  }
  return problems;
}

} // namespace mlx_kquant
