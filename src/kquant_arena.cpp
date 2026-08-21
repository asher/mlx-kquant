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

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

std::pair<mx::array, uintptr_t> arena_alloc(
    const mx::Shape& shape,
    mx::Dtype dtype) {
  if (dtype != mx::uint8 && dtype != mx::uint16 && dtype != mx::uint32 &&
      dtype != mx::uint64) {
    throw std::invalid_argument(
        "[mlx_kquant.arena_alloc] dtype must be an unsigned integer type.");
  }
  size_t nbytes = mx::size_of(dtype);
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
  mx::array arr(buf, shape, dtype, del);
  return {std::move(arr), reinterpret_cast<uintptr_t>(ptr)};
}

#ifdef _METAL_

// mlx 0.31 exposed its queue-attached MTL::ResidencySet and these helpers
// piggybacked on it; 0.32 split residency into size-capped sets budgeted
// by set_wired_limit, with no raw handle out and the command queues
// private. The arena keeps its own standalone set instead: created once
// against MLX's MTL device with a standing requestResidency, mutations
// applied at residency_commit(). The arena's host pages stay pinned by
// mlock either way; this set keeps their GPU mappings resident without
// consuming MLX's wired-limit budget, which the arena must not depend on
// (streaming-mode servers run with that budget at zero). metal-cpp is
// header-only, so the additions run entirely in this TU.
static MTL::ResidencySet* kq_residency_set() {
  static MTL::ResidencySet* rs = []() -> MTL::ResidencySet* {
    if (__builtin_available(macOS 15, *)) {
      auto& d = mx::metal::device(mx::Device(mx::Device::gpu));
      auto* desc = MTL::ResidencySetDescriptor::alloc()->init();
      NS::Error* error = nullptr;
      auto* set = d.mtl_device()->newResidencySet(desc, &error);
      desc->release();
      if (set != nullptr) {
        set->requestResidency();
      }
      return set;
    }
    return nullptr;
  }();
  return rs;
}

bool residency_insert(const mx::array& a) {
  const void* ptr = a.buffer().ptr();
  if (ptr == nullptr) {
    return false;
  }
  auto* rs = kq_residency_set();
  if (rs == nullptr) {
    return false;
  }
  rs->addAllocation(static_cast<MTL::Buffer*>(const_cast<void*>(ptr)));
  return true;
}

bool residency_commit() {
  auto* rs = kq_residency_set();
  if (rs == nullptr) {
    return false;
  }
  rs->commit();
  rs->requestResidency();
  return true;
}

bool residency_erase(const mx::array& a) {
  const void* ptr = a.buffer().ptr();
  if (ptr == nullptr) {
    return false;
  }
  auto* rs = kq_residency_set();
  if (rs == nullptr) {
    return false;
  }
  rs->removeAllocation(static_cast<MTL::Buffer*>(const_cast<void*>(ptr)));
  return true;
}

#else

bool residency_insert(const mx::array&) {
  return false;
}

bool residency_commit() {
  return false;
}

bool residency_erase(const mx::array&) {
  return false;
}

#endif

} // namespace mlx_kquant
