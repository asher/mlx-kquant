// Writable zero-copy host buffers for the feeder loop (docs/feeder/DESIGN.md).
//
// arena_alloc gives the feeder the memory both sides of the handoff share: a
// page-aligned host allocation wrapped no-copy in a Metal shared-storage
// buffer (allocator::make_buffer, the same mechanism the read-only GGUF
// tensor views use), returned as a uint8 mx.array plus the raw base address
// so the binding can hand Python a writable memoryview over the same bytes.
// The CPU feeder preads expert bytes straight into that memoryview and
// host-signals; kernels encoded after the paired event_wait then read the
// array. Ordering and coherence come from the event edge - nothing here
// synchronizes on its own.

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include "kquant.h"

namespace mx = mlx::core;

namespace mlx_kquant {

std::pair<mx::array, uintptr_t> arena_alloc(const mx::Shape& shape) {
  size_t nbytes = 1;
  for (auto d : shape) {
    if (d <= 0) {
      throw std::invalid_argument(
          "[mlx_kquant.arena_alloc] shape dims must be positive.");
    }
    nbytes *= static_cast<size_t>(d);
  }
  const size_t page = static_cast<size_t>(getpagesize());
  const size_t alloc_bytes = (nbytes + page - 1) & ~(page - 1);
  void* ptr = nullptr;
  if (posix_memalign(&ptr, page, alloc_bytes) != 0 || ptr == nullptr) {
    throw std::runtime_error(
        "[mlx_kquant.arena_alloc] failed to allocate " +
        std::to_string(alloc_bytes) + " bytes.");
  }
  mx::allocator::Buffer buf = mx::allocator::make_buffer(ptr, alloc_bytes);
  if (buf.ptr() == nullptr) {
    std::free(ptr);
    throw std::runtime_error(
        "[mlx_kquant.arena_alloc] no-copy buffer wrap rejected (needs the "
        "Metal allocator).");
  }
  mx::Deleter del = [ptr](mx::allocator::Buffer b) {
    mx::allocator::release(b);
    std::free(ptr);
  };
  mx::array arr(buf, shape, mx::uint8, del);
  return {std::move(arr), reinterpret_cast<uintptr_t>(ptr)};
}

} // namespace mlx_kquant
