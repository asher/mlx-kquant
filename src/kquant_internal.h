// Internal helpers shared across the kq primitive eval paths. Not exported via
// the Python bindings.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mx = mlx::core;

namespace mlx_kquant {

// Map a dtype to the C++ type token the kq Metal kernels are instantiated with:
//   float32 -> "float", float16 -> "float16_t", bfloat16 -> "bfloat16_t".
// Throws on other dtypes.
std::string kq_type_string(mx::Dtype d);

// Integer value of the environment lever `name`. Absent or empty reads as
// `unset`; 0, false, off and no read as 0; 1, true, on and yes read as 1;
// any other integer reads as itself. Any other value reads as `unset`, with
// a warning on stderr the first time the lever holds it.
int kq_env_lever(const char* name, int unset);

// kq_env_lever for a lever whose set values are 0 and 1: any other integer
// also reads as `unset`, with the same warning.
int kq_env_flag(const char* name, int unset);

// kq_env_lever for a lever whose set values start at `lowest`: a smaller
// integer also reads as `unset`, with the same warning.
int kq_env_lever_from(const char* name, int unset, int lowest);

// Index in `choices` (lower case) of the environment lever `name`, read in
// any case. Absent or empty reads as `unset`; any other value reads as
// `unset`, with the same warning.
int kq_env_choice(
    const char* name,
    const std::vector<std::string>& choices,
    int unset);

// K ceiling of the codec's large-M NAX tile: above it the plain qmm_t
// runs the BM=64 tile in place of BM=128. 0 = no ceiling.
int kq_nax_large_max_k(const std::string& kquant_type);

// True when KQ_QMM_TILE pins the plain qmm_t tile, which also keeps a call
// off the dequantize route. Metal builds only.
bool kq_qmm_tile_pinned();

// KQ_DEQUANT_ROUTE: auto (the chip classes of the dequantize route), off
// (never) or force (on any GPU, for the codecs and shapes of the route).
enum KqDequantRoute { kDequantAuto = 0, kDequantOff = 1, kDequantForce = 2 };

// The KQ_DEQUANT_ROUTE mode, read once per process unless KQ_TUNE_LIVE=1.
// Metal builds only.
int kq_dequant_route_mode();

// True when KQ_QMM_LOG is set, read as the qmm tiles read it. Metal builds
// only.
bool kq_qmm_log_on();

// Byte offset of x's start modulo 16, for KQ_QMM_LOG. A buffer starts on a
// 16-byte boundary, so this is the alignment the kernels see.
inline int kq_x_align(const mx::array& x) {
  return static_cast<int>(x.offset() % 16);
}

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
// codecs at 4 read 4-byte words. mxfp4 and mxfp8 read bytes. A start off
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
  if (t == "q2_k" || t == "q4_1" || t == "q5_1" || t == "ptq1_0" ||
      t == "nvfp4") {
    return 4;
  }
  if (t == "mxfp4" || t == "mxfp8") {
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

// True when the last two dims of a are laid out as dense rows: the last
// dim has stride 1 and the dim before it has the row length as its stride.
// The matmul kernels walk a weight that way and take the strides of any
// leading dims. The answer means something only for an array with memory.
inline bool kq_matrix_rows_dense(const mx::array& a) {
  const int nd = static_cast<int>(a.ndim());
  if (nd == 0) {
    return true;
  }
  if (nd < 2) {
    return a.strides()[0] == 1;
  }
  return a.strides()[nd - 1] == 1 &&
      a.strides()[nd - 2] == static_cast<int64_t>(a.shape(nd - 1));
}

// Op-builder form. An x with memory comes back as it is when its matrix
// rows are dense, and as a row-major copy otherwise. An x without memory
// has no strides to read yet, so it gets a KQuantMatrixDense node, which
// decides at eval.
mx::array kq_matrix_dense(const mx::array& x, const mx::Stream& s);

// The kernel mode of an out_scale argument (kq_out_scale.h), 0 for none: 1
// for shape (1,), 2 for (N,), and for a gather over n_mats > 0 weight
// matrices, 3 for (n_mats, 1) and 4 for (n_mats, N). what names the
// argument in the errors.
int kq_out_scale_mode(
    const std::optional<mx::array>& os,
    const std::string& kquant_type,
    bool transpose,
    int N,
    int64_t n_mats,
    const char* op,
    const char* what = "out_scale");

// The cotangent of a quantized_matmul with out_scale s, times s, for the
// transposed matmul of its vjp. An NVFP4 scale (5e-5 to 4.4e-4) puts s * dy
// of a training step below the float16 normal range, so a float16 cotangent
// gives a bfloat16 product.
mx::array kq_out_scaled_cotangent(
    const mx::array& cot,
    const mx::array& s,
    const mx::Stream& st);

// Op-builder form for an operand a kernel reads as one row-major run: an x
// with memory comes back as it is when it is row-major, and as a row-major
// copy otherwise. An x without memory gets a KQuantMatrixDense node that
// decides at eval, and passes a row-major x through with its buffer.
mx::array kq_row_major(const mx::array& x, const mx::Stream& s);

// kq_row_major for a weight a kernel reads as one row-major run, in place
// of a Contiguous node. MLX's Contiguous copies a row-major input whose
// buffer is larger than the input, such as a leading-axis slice of an
// expert stack or a dense prefix of a larger buffer; this passes such an
// input through. An unevaluated weight of more than 2^32 elements keeps
// the Contiguous node, since the eval-time copy indexes 32-bit elements.
mx::array kq_row_major_weight(const mx::array& w, const mx::Stream& s);

// Eval form for a GPU primitive: a itself when its matrix rows are dense,
// else a row-major copy made by one dispatch on s, which the command buffer
// keeps alive. a has a 1-, 2- or 4-byte dtype and at most 8 dims.
mx::array kq_matrix_dense_gpu(const mx::array& a, const mx::Stream& s);

// A row-major copy of a in a buffer the command buffer keeps alive,
// whatever the layout of a. The dtype and dim limits are those above.
mx::array kq_dense_copy_gpu(const mx::array& a, const mx::Stream& s);

// The same for an operand a kernel reads as one row-major run: a itself
// when it is row-major, else the row-major copy. An op whose operand is
// unevaluated on most calls, such as an activation, binds the result of
// this in its eval and adds no node for the operand.
mx::array kq_row_major_gpu(const mx::array& a, const mx::Stream& s);

// An operand as kernels that read it as vectors need it: a itself when it
// is row-major and starts on an align-byte boundary, else the row-major
// copy, whose buffer starts on a 256-byte boundary. Metal defines a vector
// load only at an address aligned to the vector's size, and a row-major
// view can start at any element of its buffer. The ops bind activations
// and float operands they read as vectors through this, with align the
// widest load (16 bytes unless noted at the call). what names the op and
// operand for KQ_QMM_LOG, which prints "[kq align] <what> xin=<n>" for
// each copy, n being the byte offset of a's start modulo align.
mx::array kq_x_aligned_gpu(
    const mx::array& a,
    const mx::Stream& s,
    const char* what,
    int align = 16);

// The same for an operand a kernel reads with a unit stride in its last
// dim and the operand's own strides in the dims before it, such as a KV
// cache: a itself when its last stride is 1 and its start and the stride
// of each dim longer than 1 fall on align-byte boundaries, else the
// row-major copy.
mx::array kq_strides_aligned_gpu(
    const mx::array& a,
    const mx::Stream& s,
    const char* what,
    int align = 16);

// The same for an operand a kernel reads with a unit stride in its last
// dim and the operand's own strides in the dims before it: a itself when
// the last stride is 1, else the row-major copy.
mx::array kq_unit_stride_gpu(const mx::array& a, const mx::Stream& s);

// Eval guard for an operand an op does not copy, such as an expert stack:
// throws std::invalid_argument(what + " must be row-contiguous.") when a
// is not row-major. An op builder can run it only on an array that has
// memory, since any other array reports the default dense strides.
inline void kq_check_row_major(const mx::array& a, const std::string& what) {
  if (!a.flags().row_contiguous) {
    throw std::invalid_argument(what + " must be row-contiguous.");
  }
}

// True when a has memory, so that its strides and flags are its own.
inline bool kq_has_layout(const mx::array& a) {
  return a.status() != mx::array::Status::unscheduled;
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

// Eval form for a CPU primitive, called inside its dispatched task: the
// elements of a in row-major order, a's own memory when a is row-major and
// else a copy held by store.
template <typename T>
const T* kq_row_major_cpu(const mx::array& a, std::vector<T>& store) {
  const T* p = a.data<T>();
  if (a.flags().row_contiguous) {
    return p;
  }
  const int64_t n = static_cast<int64_t>(a.size());
  store.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; i++) {
    store[static_cast<size_t>(i)] = p[elem_to_loc64(i, a.shape(), a.strides())];
  }
  return store.data();
}

} // namespace mlx_kquant
