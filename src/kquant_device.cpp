// GPU core count for the occupancy policies that scale with it. Probed once
// from the IOKit accelerator entry; KQ_GPU_CORES overrides it and is read per
// call so one process can A/B a policy. Also the environment lever parser
// (kq_env_lever, kq_env_flag).
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

#include "kquant.h"
#include "kquant_internal.h"

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

// Warns once per lever and value that `value` is not one `name` takes.
void warn_env_lever(const char* name, const char* value, const char* takes) {
  static std::mutex mu;
  static std::set<std::string> seen;
  std::lock_guard<std::mutex> lock(mu);
  if (seen.insert(std::string(name) + "=" + value).second) {
    std::fprintf(
        stderr,
        "[mlx_kquant] %s=%s is not %s; %s keeps its default.\n",
        name,
        value,
        takes,
        name);
  }
}

int parse_env_lever(const char* name, int unset, bool flag) {
  const char* e = std::getenv(name);
  if (e == nullptr || e[0] == '\0') {
    return unset;
  }
  std::string v;
  for (const char* c = e; *c != '\0'; ++c) {
    v += static_cast<char>(std::tolower(static_cast<unsigned char>(*c)));
  }
  if (v == "false" || v == "off" || v == "no") {
    return 0;
  }
  if (v == "true" || v == "on" || v == "yes") {
    return 1;
  }
  const char* takes = flag ? "0, 1, true, false, on, off, yes or no"
                           : "an integer, true, false, on, off, yes or no";
  char* end = nullptr;
  errno = 0;
  const long n = std::strtol(e, &end, 10);
  if (end == e || *end != '\0' || errno != 0 || n < INT_MIN || n > INT_MAX ||
      (flag && n != 0 && n != 1)) {
    warn_env_lever(name, e, takes);
    return unset;
  }
  return static_cast<int>(n);
}

} // namespace

int kq_env_lever(const char* name, int unset) {
  return parse_env_lever(name, unset, false);
}

int kq_env_flag(const char* name, int unset) {
  return parse_env_lever(name, unset, true);
}

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
