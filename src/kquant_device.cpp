// GPU core count for the occupancy policies that scale with it. Probed once
// from the IOKit accelerator entry; KQ_GPU_CORES overrides it and is read per
// call so one process can A/B a policy.
#include <cstdlib>

#include "kquant.h"

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#endif

namespace mlx_kquant {

namespace {

int probe_gpu_core_count() {
#ifdef __APPLE__
  io_iterator_t it = 0;
  if (IOServiceGetMatchingServices(
          kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &it) !=
      KERN_SUCCESS) {
    return 0;
  }
  int cores = 0;
  for (io_object_t entry = IOIteratorNext(it); entry != 0;
       entry = IOIteratorNext(it)) {
    if (cores == 0) {
      CFTypeRef v = IORegistryEntryCreateCFProperty(
          entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
      if (v != nullptr) {
        int n = 0;
        if (CFGetTypeID(v) == CFNumberGetTypeID() &&
            CFNumberGetValue(
                static_cast<CFNumberRef>(v), kCFNumberIntType, &n) &&
            n > 0) {
          cores = n;
        }
        CFRelease(v);
      }
    }
    IOObjectRelease(entry);
  }
  IOObjectRelease(it);
  return cores;
#else
  return 0;
#endif
}

} // namespace

int gpu_core_count() {
  const char* e = std::getenv("KQ_GPU_CORES");
  if (e != nullptr) {
    const int n = std::atoi(e);
    if (n > 0) {
      return n;
    }
  }
  static const int probed = probe_gpu_core_count();
  return probed;
}

} // namespace mlx_kquant
