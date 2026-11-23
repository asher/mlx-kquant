// cb_commit_mark: a commit boundary a graph carries with it.
//
// mlx ends a command buffer after the primitive whose encode took the
// buffer past the device caps (ops or MB), so a caller shapes the buffers
// of one eval only through those two numbers. A speculative round wants a
// different shape: a short first buffer, so the GPU starts while the host
// still encodes, then everything else in one buffer, so the host is never
// held at mlx's limit on command buffers in flight. With the caps set high
// for the eval (set_cb_caps), each mark in the graph is one boundary.
//
// The mark is an identity on its input (the output shares the buffer). On
// the GPU its eval sets the encoder's op count past the cap when the
// buffer holds work, and mlx's own eval loop then ends and commits the
// buffer the way it does at a cap, with its task accounting. Nothing is
// committed here.
#include <atomic>
#include <memory>
#include <vector>

#ifdef _METAL_
#define private public
#include "mlx/backend/metal/device.h"
#undef private
#endif

#include "mlx/array.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include "kquant.h"

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

std::atomic<uint64_t> g_cb_marks{0};

class KQuantCbCommitMark : public mx::UnaryPrimitive {
 public:
  explicit KQuantCbCommitMark(mx::Stream stream) : mx::UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    out.copy_shared_buffer(inputs[0]);
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    out.copy_shared_buffer(inputs[0]);
#ifdef _METAL_
    auto& d = mx::metal::device(stream().device);
    auto& ce = mx::metal::get_command_encoder(stream());
    if (ce.buffer_ops_ > 0) {
      auto [ops, mb] = d.get_max_ops_mb_per_buffer();
      if (ce.buffer_ops_ <= ops) {
        ce.buffer_ops_ = ops + 1;
      }
      g_cb_marks.fetch_add(1, std::memory_order_relaxed);
    }
#endif
  }
  const char* name() const override {
    return "KQuantCbCommitMark";
  }
  bool is_equivalent(const mx::Primitive& other) const override {
    return dynamic_cast<const KQuantCbCommitMark*>(&other) != nullptr;
  }
};

} // namespace

mx::array cb_commit_mark(const mx::array& x, mx::StreamOrDevice s) {
  return mx::array(
      x.shape(),
      x.dtype(),
      std::make_shared<KQuantCbCommitMark>(mx::to_stream(s)),
      {x});
}

uint64_t cb_commit_marks() {
  return g_cb_marks.load(std::memory_order_relaxed);
}

} // namespace mlx_kquant
