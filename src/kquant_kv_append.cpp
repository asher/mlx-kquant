// KV cache append without a copy of the cache. The output shares the cache
// buffer and one small kernel writes the new rows into it, where a slice
// update copies the whole buffer whenever a pending step still holds it.
// This is the one kq op that writes through a shared buffer; the caller
// keeps every other holder off the written rows (see kv_append in kquant.h).
#include <cstring>
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_internal.h"

#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

void kq_kv_append_check(int64_t offset, int64_t n, int64_t cap) {
  if (offset < 0 || offset + n > cap) {
    throw std::invalid_argument(
        "[mlx_kquant.kv_append] rows [offset, offset + x.shape[2]) must lie "
        "within buf.");
  }
}

} // namespace

#ifdef _METAL_

void KQuantKVAppend::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& out = outputs[0];
  const auto& buf = inputs[0];
  const auto& x = inputs[1];
  if (buf.size() == 0 || buf.flags().row_contiguous) {
    out.copy_shared_buffer(buf);
  } else {
    kq_dense_copy_into_gpu(buf, out, s);
  }
  if (x.size() == 0) {
    return;
  }
  // KqKvAppendArgs in kq_kv_append.h.
  struct {
    int shape[4];
    int64_t strides[4];
    int64_t cap;
    int64_t offset;
    int64_t size;
  } args{};
  for (int i = 0; i < 4; i++) {
    args.shape[i] = x.shape(i);
    args.strides[i] = x.strides()[i];
  }
  args.cap = out.shape(2);
  args.offset =
      param_slot_ >= 0 ? kq_param_value(inputs[2], param_slot_) : offset_;
  args.size = static_cast<int64_t>(x.size());
  kq_kv_append_check(args.offset, x.shape(2), args.cap);

  auto& d = mx::metal::device(s.device);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(
      kq_get_kernel(d, "kq_kv_append_" + std::to_string(x.itemsize())));
  ce.set_input_array(x, 0);
  ce.set_output_array(out, 1);
  ce.set_bytes(args, 2);
  if (param_slot_ >= 0) {
    // The low word of the 64-bit offset; the high word stays 0. A replay
    // writes only offsets the rows fit at.
    kq_param_site(2, 56, param_slot_, 1, 0, 0, args.cap - x.shape(2));
  }
  const size_t tg = x.size() < 256 ? x.size() : 256;
  ce.dispatch_threads(MTL::Size(x.size(), 1, 1), MTL::Size(tg, 1, 1));
}

#else // !_METAL_

void KQuantKVAppend::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.kv_append] GPU eval requires a Metal build.");
}

#endif // _METAL_

void KQuantKVAppend::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  const auto& buf = inputs[0];
  auto& out = outputs[0];
  const int64_t offset = param_slot_ >= 0
      ? int64_t(kq_param_value(inputs[2], param_slot_))
      : offset_;
  kq_kv_append_check(offset, inputs[1].shape(2), out.shape(2));
  const bool shared = buf.flags().row_contiguous;
  if (shared) {
    out.copy_shared_buffer(buf);
  } else {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(inputs[0]);
  encoder.set_input_array(inputs[1]);
  encoder.set_output_array(out);
  encoder.dispatch([buf = mx::array::unsafe_weak_copy(inputs[0]),
                    x = mx::array::unsafe_weak_copy(inputs[1]),
                    out = mx::array::unsafe_weak_copy(out),
                    shared,
                    offset]() mutable {
    const size_t item = out.itemsize();
    char* op = out.data<char>();
    if (!shared) {
      const char* bp = buf.data<char>();
      const int64_t n = static_cast<int64_t>(buf.size());
      for (int64_t i = 0; i < n; i++) {
        const int64_t loc = elem_to_loc64(i, buf.shape(), buf.strides());
        std::memcpy(op + i * item, bp + loc * item, item);
      }
    }
    const int64_t H = x.shape(1);
    const int64_t n = x.shape(2);
    const int64_t D = x.shape(3);
    const int64_t cap = out.shape(2);
    const char* xp = x.data<char>();
    const int64_t total = static_cast<int64_t>(x.size());
    for (int64_t i = 0; i < total; i++) {
      int64_t rem = i;
      const int64_t d = rem % D;
      rem /= D;
      const int64_t t = rem % n;
      rem /= n;
      const int64_t h = rem % H;
      const int64_t b = rem / H;
      const int64_t dst = ((b * H + h) * cap + offset + t) * D + d;
      std::memcpy(
          op + dst * item,
          xp + elem_to_loc64(i, x.shape(), x.strides()) * item,
          item);
    }
  });
}

mx::array kv_append(
    mx::array buf,
    mx::array x,
    int64_t offset,
    mx::StreamOrDevice s_,
    const std::optional<mx::array>& params,
    int offset_slot) {
  auto s = mx::to_stream(s_);
  const std::string op = "[mlx_kquant.kv_append]";
  if (params &&
      (params->dtype() != mx::int32 || params->ndim() != 1 || offset_slot < 0 ||
       offset_slot >= params->size())) {
    throw std::invalid_argument(
        op + " params must be an int32 vector holding offset_slot.");
  }
  // The offset is read on the host when the append is encoded.
  if (params && !params->is_available()) {
    throw std::invalid_argument(op + " params must be evaluated.");
  }
  if (buf.ndim() != 4 || x.ndim() != 4) {
    throw std::invalid_argument(
        op + " buf and x must both be [B, H, rows, D].");
  }
  if (buf.dtype() != x.dtype()) {
    throw std::invalid_argument(op + " x must have the dtype of buf.");
  }
  if (buf.itemsize() != 1 && buf.itemsize() != 2 && buf.itemsize() != 4) {
    throw std::invalid_argument(op + " dtype must be 1, 2 or 4 bytes wide.");
  }
  if (buf.shape(0) != x.shape(0) || buf.shape(1) != x.shape(1) ||
      buf.shape(3) != x.shape(3)) {
    throw std::invalid_argument(op + " x must match buf in axes 0, 1 and 3.");
  }
  if (offset < 0 || offset + x.shape(2) > buf.shape(2)) {
    throw std::invalid_argument(
        op + " rows [offset, offset + x.shape[2]) must lie within buf.");
  }
  if (x.size() > UINT32_MAX) {
    throw std::invalid_argument(
        op + " x must have at most 4294967295 elements.");
  }
  // Read before the move: argument evaluation order is unspecified, and GCC
  // moves buf first.
  auto shape = buf.shape();
  auto dtype = buf.dtype();
  std::vector<mx::array> inputs = {std::move(buf), std::move(x)};
  if (params) {
    inputs.push_back(*params);
  }
  return mx::array(
      std::move(shape),
      dtype,
      std::make_shared<KQuantKVAppend>(s, offset, params ? offset_slot : -1),
      std::move(inputs));
}

std::optional<uintptr_t> buffer_id(const mx::array& a) {
  // An evaluated array has its memory even while its kernel is in flight.
  if (a.status() == mx::array::Status::unscheduled ||
      a.data_shared_ptr() == nullptr) {
    return std::nullopt;
  }
  const void* ptr = a.buffer().ptr();
  if (ptr == nullptr) {
    return std::nullopt;
  }
  return reinterpret_cast<uintptr_t>(ptr);
}

} // namespace mlx_kquant
