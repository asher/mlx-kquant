// Shared-event stream primitives for the feeder decode loop (see
// docs/feeder/DESIGN.md and the declarations in kquant.h).
//
// The GPU-side encoding mirrors what MLX's own event backend does: end the
// active compute encoder, then encode the signal/wait on the underlying
// command buffer. Metal orders an encoded signal after all work encoded
// before it in the buffer and stalls everything after an encoded wait, which
// is exactly the handoff edge the feeder needs. Because a wait's satisfying
// signal always comes from the HOST (and a signal's consumer is the host
// feeder), MLX splitting or committing command buffers between the pair can
// serialize the stream but never deadlock it.
//
// Events live in a process-wide registry and stay retained until
// shared_event_destroy, so an event can outlive any command buffer that
// references it.

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "kquant.h"

#include "mlx/ops.h"
#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

#ifdef _METAL_

namespace {

std::mutex g_event_mutex;
std::unordered_map<uint64_t, NS::SharedPtr<MTL::SharedEvent>>& event_registry() {
  static auto* reg =
      new std::unordered_map<uint64_t, NS::SharedPtr<MTL::SharedEvent>>();
  return *reg;
}

MTL::SharedEvent* lookup_event(uint64_t handle, const char* op) {
  std::lock_guard<std::mutex> lock(g_event_mutex);
  auto& reg = event_registry();
  auto it = reg.find(handle);
  if (it == reg.end()) {
    throw std::invalid_argument(
        std::string("[mlx_kquant.") + op + "] Unknown shared event handle " +
        std::to_string(handle) + ".");
  }
  return it->second.get();
}

} // namespace

uint64_t shared_event_create() {
  auto& d = mx::metal::device(mx::Device::gpu);
  auto evt = NS::TransferPtr(d.mtl_device()->newSharedEvent());
  if (!evt) {
    throw std::runtime_error(
        "[mlx_kquant.shared_event_create] newSharedEvent failed.");
  }
  static uint64_t next_handle = 1;
  std::lock_guard<std::mutex> lock(g_event_mutex);
  uint64_t handle = next_handle++;
  event_registry().emplace(handle, std::move(evt));
  return handle;
}

void shared_event_destroy(uint64_t handle) {
  std::lock_guard<std::mutex> lock(g_event_mutex);
  event_registry().erase(handle);
}

void shared_event_set(uint64_t handle, uint64_t value) {
  lookup_event(handle, "shared_event_set")->setSignaledValue(value);
}

uint64_t shared_event_read(uint64_t handle) {
  return lookup_event(handle, "shared_event_read")->signaledValue();
}

bool shared_event_wait(uint64_t handle, uint64_t value, int64_t timeout_ms) {
  auto* evt = lookup_event(handle, "shared_event_wait");
  uint64_t ms = timeout_ms < 0 ? uint64_t(-1) : uint64_t(timeout_ms);
  return evt->waitUntilSignaledValue(value, ms);
}

void KQuantEventSignal::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto* evt = lookup_event(handle_, "event_signal");
  auto& enc = mx::metal::get_command_encoder(stream());
  enc.end_encoding();
  enc.get_command_buffer()->encodeSignalEvent(evt, value_);
  outputs[0].copy_shared_buffer(inputs[0]);
}

void KQuantEventWait::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto* evt = lookup_event(handle_, "event_wait");
  auto& enc = mx::metal::get_command_encoder(stream());
  enc.end_encoding();
  enc.get_command_buffer()->encodeWait(evt, value_);
  outputs[0].copy_shared_buffer(inputs[0]);
}

#else // !_METAL_

namespace {
[[noreturn]] void throw_no_metal(const char* op) {
  throw std::runtime_error(
      std::string("[mlx_kquant.") + op +
      "] Shared events require a Metal build.");
}
} // namespace

uint64_t shared_event_create() {
  throw_no_metal("shared_event_create");
}
void shared_event_destroy(uint64_t) {
  throw_no_metal("shared_event_destroy");
}
void shared_event_set(uint64_t, uint64_t) {
  throw_no_metal("shared_event_set");
}
uint64_t shared_event_read(uint64_t) {
  throw_no_metal("shared_event_read");
}
bool shared_event_wait(uint64_t, uint64_t, int64_t) {
  throw_no_metal("shared_event_wait");
}

void KQuantEventSignal::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw_no_metal("event_signal");
}

void KQuantEventWait::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw_no_metal("event_wait");
}

#endif // _METAL_

// CPU placement is a legal no-op: there is no command stream to gate, and
// the feeder pairs its edges with host-side signal/wait instead.
void KQuantEventSignal::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  outputs[0].copy_shared_buffer(inputs[0]);
}

void KQuantEventWait::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  outputs[0].copy_shared_buffer(inputs[0]);
}

std::vector<mx::Shape> KQuantEventSignal::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

std::vector<mx::Shape> KQuantEventWait::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQuantEventSignal::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const KQuantEventSignal&>(other);
  return handle_ == o.handle_ && value_ == o.value_;
}

bool KQuantEventWait::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const KQuantEventWait&>(other);
  return handle_ == o.handle_ && value_ == o.value_;
}

namespace {

mx::array make_event_op(
    const mx::array& x,
    uint64_t handle,
    uint64_t value,
    mx::StreamOrDevice s_,
    bool is_signal) {
  if (handle == 0) {
    throw std::invalid_argument(
        std::string("[mlx_kquant.") +
        (is_signal ? "event_signal" : "event_wait") +
        "] handle must be a live shared_event_create handle.");
  }
  auto s = mx::to_stream(s_);
  std::shared_ptr<mx::Primitive> prim;
  if (is_signal) {
    prim = std::make_shared<KQuantEventSignal>(s, handle, value);
  } else {
    prim = std::make_shared<KQuantEventWait>(s, handle, value);
  }
  return mx::array(x.shape(), x.dtype(), std::move(prim), {x});
}

} // namespace

mx::array event_signal(
    const mx::array& x,
    uint64_t handle,
    uint64_t value,
    mx::StreamOrDevice s) {
  return make_event_op(x, handle, value, s, true);
}

mx::array event_wait(
    const mx::array& x,
    uint64_t handle,
    uint64_t value,
    mx::StreamOrDevice s) {
  return make_event_op(x, handle, value, s, false);
}

} // namespace mlx_kquant
