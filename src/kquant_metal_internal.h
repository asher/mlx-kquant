// Metal-side dispatch helpers shared between KQuantMatmul (kquant_matmul.cpp),
// KQuantQmvBias (kquant_matmul.cpp), and KQuantGatherQMM (kquant_gather.cpp).
// Header-inline so all translation units share a single definition; compiled
// only where _METAL_ is defined. Kernels are fetched from the bundled
// metallib via kq_get_kernel, NAX availability is probed via
// kq_is_nax_available, and kq_collapse_contiguous_dims collapses adjacent
// contiguous index axes. Most kquant paths carry no bias buffer; the q8_0
// decode-only qmv/qmv_fast kernels are the exception (see KQuantQmvBias).
#pragma once

#ifdef _METAL_

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "kquant.h" // metallib_dir
#include "kquant_codec.h" // codec_by_name
#include "kquant_internal.h" // kq_qmm_log_on

#include "mlx/array.h" // Shape, Strides
#include "mlx/backend/metal/device.h"

namespace mx = mlx::core;

namespace mlx_kquant {

using mx::array;
using Device = mx::metal::Device;
using CommandEncoder = mx::metal::CommandEncoder;

// Probe whether the GPU supports NAX (tensor-core) matmul. metal::device is
// exported and get_architecture()/get_architecture_gen() are inline header
// methods, so the check runs here directly. Cached process-wide.
inline bool kq_is_nax_available() {
#ifdef MLX_METAL_NO_NAX
  return false;
#else
  static bool available = []() {
    bool can_use_nax = false;
    if (__builtin_available(
            macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    auto& d = mx::metal::device(mx::Device::gpu);
    auto arch = d.get_architecture().back();
    auto gen = d.get_architecture_gen();
    can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
    return can_use_nax;
  }();
  return available;
#endif
}

// quantized.cpp:52-58
inline int kquant_qmv_bn(const std::string& kquant_type) {
  if (kquant_type == "q2_k" || kquant_type == "q3_k" || kquant_type == "q4_k" ||
      kquant_type == "q5_k" || kquant_type == "iq4_xs" ||
      kquant_type == "iq3_s" || kquant_type == "iq3_xxs" ||
      kquant_type == "iq2_xxs" || kquant_type == "iq2_xs" ||
      kquant_type == "iq2_s" || kquant_type == "iq1_s" ||
      kquant_type == "iq1_m" || kquant_type == "stq1_0" ||
      kquant_type == "pq2_0" || kquant_type == "q2_0" ||
      kquant_type == "ptq1_0") {
    return 4;
  }
  return 8;
}

// Rows per threadgroup of the M=1 qmv_fast kernel: kquant_qmv_bn, except
// ptq1_0, whose qmv_fast runs four rows per simdgroup
// (KQ_PRISM_DEFINE_KERNELS FAST_RPS).
inline int kquant_qmv_fast_bn(const std::string& kquant_type) {
  return kquant_type == "ptq1_0" ? 8 : kquant_qmv_bn(kquant_type);
}

// Output rows and simdgroups per threadgroup of the split-K M=1 qmv. Must
// match KQ_Q6_K_SPLITK_RPS and KQ_Q6_K_SPLITK_NSG in kq_quantized_kquants.h.
constexpr int kq_q6_k_splitk_rps = 4;
constexpr int kq_q6_k_splitk_nsg = 8;

// Output rows and simdgroups per threadgroup of the split-K M=1 q8_0 qmv,
// kq_q8_0_qmv_splitk in kq_quantized.h.
constexpr int kq_q8_0_splitk_rps = 2;
constexpr int kq_q8_0_splitk_nsg = 8;

// Simdgroups per threadgroup of the split-K M=1 qmv at row width K: one per
// 256-weight step (q8_0) or superblock pair (q6_k), at most the tile's nsg.
// The kernels sum only the partials of the simdgroups that launched and
// hold work, so the count does not change the output. KQ_QMV_SPLITK_TRIM=0
// (read live) dispatches the full nsg at every K.
int kq_qmv_splitk_sgs(bool q6, int K);

// quantized.cpp:63-65 (kquant branch). KQuant blocks are 32 or 256 weights.
inline int qmv_fast_k_align() {
  return 256;
}

inline bool codec_has_matmul(const std::string& kquant_type) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  return codec != nullptr && codec->has_matmul_kernel;
}

// Gates the NAX (tensor-core) dispatch per codec (has_nax_kernel in the
// codec table); codecs without a NAX kernel route to the ALU kernels.
inline bool codec_has_nax(const std::string& kquant_type) {
  // KQ_DISABLE_NAX=1 forces the ALU qmm/gather path (A/B harness lever). Read
  // live (not cached) so a single process can toggle NAX between calls; only
  // reached on the NAX-eligible prefill path, so the getenv cost is negligible.
  const char* e = std::getenv("KQ_DISABLE_NAX");
  if (e != nullptr && e[0] == '1') {
    return false;
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  return codec != nullptr && codec->has_nax_kernel;
}

// Codecs with qmv_fast_fine/qmv_fine instantiations (2 output rows per
// threadgroup vs kquant_qmv_bn rows, multiplying the threadgroup count for
// the same N). Explicit allow-list like codec_has_verify_qmv below: the
// dispatch may only route to kernels actually instantiated in
// kq_quantized.metal.
inline bool codec_has_qmv_fine(const std::string& kquant_type) {
  return kquant_type == "q8_0" || kquant_type == "q4_0" ||
      kquant_type == "q4_1" || kquant_type == "q5_0" || kquant_type == "q5_1" ||
      kquant_type == "q2_k" || kquant_type == "q3_k" || kquant_type == "q4_k" ||
      kquant_type == "q5_k" || kquant_type == "q6_k" ||
      kquant_type == "iq4_nl" || kquant_type == "mxfp4" ||
      kquant_type == "nvfp4" || kquant_type == "iq4_xs" ||
      kquant_type == "iq3_xxs" || kquant_type == "iq3_s" ||
      kquant_type == "iq2_xxs" || kquant_type == "iq2_xs" ||
      kquant_type == "iq2_s" || kquant_type == "iq1_s" ||
      kquant_type == "iq1_m" || kquant_type == "stq1_0" ||
      kquant_type == "pq2_0" || kquant_type == "q2_0" ||
      kquant_type == "ptq1_0";
}

// Default fine-tiling ceiling per codec: the qmv dispatch uses the fine
// (2 rows per threadgroup) variant when M == 1, B == 1, and N <= this value.
// Calibrated on M5 Max via thermally-paired A/B (mlx-quant-lab
// tools/gemv-fine-ab.py): fine wins on occupancy-starved layer-size
// dispatches and loses at embedding-size N, where the grid already
// saturates and finer tiling only repeats per-threadgroup x reads. 0 =
// stay coarse by default (measured negative or not yet calibrated).
// KQ_QMV_FINE=1/0 overrides in either direction.
inline int kquant_qmv_fine_default_max_n(const std::string& kquant_type) {
  // Uniform wins across layer shapes; lm_head-size N excluded.
  if (kquant_type == "q6_k" || kquant_type == "q4_k" || kquant_type == "q5_k" ||
      kquant_type == "q5_0" || kquant_type == "iq4_xs" ||
      kquant_type == "iq3_s" || kquant_type == "iq3_xxs" ||
      kquant_type == "iq2_s" || kquant_type == "iq1_m") {
    return 16384;
  }
  // Wins through mid-N; measured negative at the largest layer shape.
  if (kquant_type == "q3_k" || kquant_type == "iq4_nl") {
    return 8192;
  }
  if (kquant_type == "q5_1") {
    return 4096;
  }
  // Only kv-size dispatches reproducibly win.
  if (kquant_type == "q8_0" || kquant_type == "q2_k" || kquant_type == "q4_0" ||
      kquant_type == "q4_1") {
    return 2048;
  }
  // mxfp4/nvfp4/iq2_xxs/iq2_xs/iq1_s: no synthetic encoder, so no paired
  // A/B yet -- coarse by default until calibrated on real wire data.
  // stq1_0 is encodable but unmeasured; coarse for now.
  return 0;
}

// The ceilings above were calibrated on a 40-core GPU. A part with fewer
// cores saturates its grid at a proportionally smaller N, so when
// KQ_GPU_CORES names the part's core count the ceiling scales with it.
// Unset, the calibrated value applies on every part (the scaling is not
// yet measured on a smaller GPU).
constexpr int kq_fine_reference_cores = 40;
inline int kquant_qmv_fine_max_n(const std::string& kquant_type) {
  const int base = kquant_qmv_fine_default_max_n(kquant_type);
  if (base == 0 || std::getenv("KQ_GPU_CORES") == nullptr) {
    return base;
  }
  const int cores = gpu_core_count();
  if (cores <= 0 || cores == kq_fine_reference_cores) {
    return base;
  }
  const long scaled = static_cast<long>(base) * cores / kq_fine_reference_cores;
  const int rounded = static_cast<int>(scaled / 256 * 256);
  return std::min(std::max(rounded, 512), 32768);
}

// Codecs with a verify_qmv kernel (the small-M weight-read-amortizing leaf).
// Kept as an explicit allow-list so the dispatch only routes to a kernel that
// was actually instantiated in kq_quantized.metal; new codecs are added here
// once their verify_qmv instantiation lands.
inline bool codec_has_verify_qmv(const std::string& kquant_type) {
  return kquant_type == "q6_k" || kquant_type == "q8_0" ||
      kquant_type == "q4_k" || kquant_type == "q5_k" || kquant_type == "q5_1" ||
      kquant_type == "q3_k" || kquant_type == "q2_k" || kquant_type == "q4_0" ||
      kquant_type == "q4_1" || kquant_type == "q5_0" ||
      kquant_type == "pq2_0" || kquant_type == "q2_0" ||
      kquant_type == "ptq1_0";
}

// Largest activation-row count (M) the verify_qmv kernels are instantiated for;
// must match MAX_VM in kq_*_verify_qmv_impl. M above this falls back to qmv.
inline int verify_qmv_max_rows() {
  return 8;
}

// Codecs with a register-resident MMA verify kernel (kq_verify_mma.h) and
// the output rows one of its threadgroups covers: 8 simdgroups of 8-row
// tiles, two tiles each for q4_0, q4_1 and q8_0 and one for pq2_0, q2_0
// and ptq1_0; 0 for codecs without the kernel. M <= 8 only.
inline int codec_verify_mma_rows(const std::string& kquant_type) {
  if (kquant_type == "q4_0" || kquant_type == "q4_1" || kquant_type == "q8_0") {
    return 128;
  }
  if (kquant_type == "pq2_0" || kquant_type == "q2_0" ||
      kquant_type == "ptq1_0") {
    return 64;
  }
  return 0;
}

// Codecs with a register-fed NAX verify kernel (kq_verify_nax.h) and the K
// its loop consumes per iteration (units per iteration times the unit
// width); 0 for codecs without the kernel. NAX GPUs, M <= 8 only. q4_0
// and q4_1 list their two-block kernels; the host runs the eight-block
// kernels (256) where K is a multiple of 256 (kq_verify_nax_call_kstep).
inline int codec_verify_nax_kstep(const std::string& kquant_type) {
  if (kquant_type == "pq2_0" || kquant_type == "q2_0" ||
      kquant_type == "q8_0" || kquant_type == "nvfp4") {
    return 128;
  }
  if (kquant_type == "q4_0" || kquant_type == "q4_1") {
    return 64;
  }
  if (kquant_type == "q4_k" || kquant_type == "iq4_xs" ||
      kquant_type == "iq4_nl") {
    return 256;
  }
  if (kquant_type == "q5_k" || kquant_type == "q6_k" || kquant_type == "q3_k" ||
      kquant_type == "q2_k") {
    return 128;
  }
  return 0;
}

// quantized.cpp:133-175
inline int get_qmv_batch_limit(int D, int O, Device& d) {
  auto arch_size = d.get_architecture().back();
  auto arch_gen = d.get_architecture_gen();
  if (arch_gen == 13 || arch_gen == 14) {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 14;
        } else if (D <= 4096 && O <= 4096) {
          return 10;
        } else {
          return 6;
        }
    }
  } else {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 18;
        } else if (D <= 4096 && O <= 4096) {
          return 12;
        } else {
          return 10;
        }
    }
  }
}

// quantized.cpp:177-205, bias buffer dropped (kquant has no biases). Used by
// both the plain matmul leaf fns and the gather leaf fns.
inline int add_strides_and_shapes(
    CommandEncoder& ce,
    bool skip,
    const array& x,
    const array& w,
    const array& scales,
    int offset) {
  if (skip) {
    return 0;
  }
  int x_batch_ndims = x.ndim() - 2;
  int w_batch_ndims = w.ndim() - 2;
  ce.set_bytes(x_batch_ndims, offset++);
  ce.set_vector_bytes(x.shape(), offset++);
  ce.set_vector_bytes(x.strides(), offset++);
  ce.set_bytes(w_batch_ndims, offset++);
  ce.set_vector_bytes(w.shape(), offset++);
  ce.set_vector_bytes(w.strides(), offset++);
  ce.set_vector_bytes(scales.strides(), offset++);
  return offset;
}

// Collapse adjacent contiguous axes of the index tensors. The kernel's
// flat-offset math is identical whether or not the dims are collapsed; this
// just reduces the loop-trip count.
inline std::tuple<mx::Shape, std::vector<mx::Strides>>
kq_collapse_contiguous_dims(
    const mx::Shape& shape,
    const std::vector<mx::Strides>& strides) {
  const int64_t size_cap = std::numeric_limits<int32_t>::max();

  // Mark axes with -1 separators; collapse runs between separators.
  mx::Shape to_collapse;
  if (shape.size() > 0) {
    if (shape[0] != 1) {
      to_collapse.push_back(0);
    }
    int64_t size = shape[0];
    for (int i = 1; i < static_cast<int>(shape.size()); i++) {
      bool contiguous = true;
      size *= shape[i];
      for (const auto& st : strides) {
        if (st[i] * shape[i] != st[i - 1] || size > size_cap) {
          contiguous = false;
          size = shape[i];
          break;
        }
      }
      if (!contiguous) {
        to_collapse.push_back(-1);
      }
      if (shape[i] != 1) {
        to_collapse.push_back(i);
      }
    }
    to_collapse.push_back(-1);
  }

  mx::Shape out_shape;
  std::vector<mx::Strides> out_strides(strides.size());
  for (int i = 0;;) {
    while (i < static_cast<int>(to_collapse.size()) && to_collapse[i] == -1) {
      ++i;
    }
    if (i == static_cast<int>(to_collapse.size())) {
      break;
    }
    int current_shape = shape[to_collapse[i]];
    int k = i;
    while (to_collapse[++k] != -1) {
      current_shape *= shape[to_collapse[k]];
    }
    out_shape.push_back(current_shape);
    for (int j = 0; j < static_cast<int>(strides.size()); j++) {
      out_strides[j].push_back(strides[j][to_collapse[k - 1]]);
    }
    i = k + 1;
  }

  if (!shape.empty() && out_shape.empty()) {
    out_shape.push_back(1);
    for (auto& out_stride : out_strides) {
      out_stride.push_back(0);
    }
  }
  return std::make_tuple(out_shape, out_strides);
}

// Append the collapsed index ndims/shape/strides for the gather leaf kernels.
inline int add_gather_strides_and_shapes(
    CommandEncoder& ce,
    const array& lhs_indices,
    const array& rhs_indices,
    int offset) {
  auto [shape, strides] = kq_collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  int ndims = shape.size();
  ce.set_bytes(ndims, offset++);
  ce.set_vector_bytes(shape, offset++);
  ce.set_vector_bytes(strides[0], offset++);
  ce.set_vector_bytes(strides[1], offset++);
  return offset;
}

// One KQ_QMM_LOG line for a gather or MoE dispatch: the kernel name, M and
// N of out's rows, K of x's last dim, the grid, the threadgroup size and xa.
void kq_log_dispatch(
    const std::string& kname,
    const array& x,
    const array& out,
    const MTL::Size& grid,
    const MTL::Size& group);

// True when the bundled metallib carries a function named kname. Looks the
// name up without building a pipeline, so a tile override can raise with
// the kernel name before any dispatch.
inline bool kq_has_kernel(Device& d, const std::string& kname) {
  static MTL::Library* lib = d.get_library("mlx_kquant", metallib_dir());
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto fn = NS::TransferPtr(lib->newFunction(
      NS::String::string(kname.c_str(), NS::UTF8StringEncoding)));
  return fn.get() != nullptr;
}

// The error for a lever setting that selects a kernel the metallib does not
// carry, naming the setting and the kernel.
inline std::invalid_argument kq_missing_kernel(
    const std::string& setting,
    const std::string& kname) {
  return std::invalid_argument(
      "[mlx_kquant] " + setting + " selects kernel " + kname +
      ", which is not instantiated.");
}

// Raises kq_missing_kernel when the metallib does not carry kname.
inline void kq_require_kernel(
    Device& d,
    const std::string& setting,
    const std::string& kname) {
  if (!kq_has_kernel(d, kname)) {
    throw kq_missing_kernel(setting, kname);
  }
}

// Function constant 370 (kq_kquant_word in kq_quantized_kquants.h): the
// q2_k, q4_k and q5_k chunk decodes form four codes per 32-bit word.
// KQ_KQUANT_WORD=0 restores the per-byte loop (read once per process unless
// KQ_TUNE_LIVE=1, since every dispatch asks). The constant has no default,
// so a function that declares it builds only with a value, and
// kq_get_kernel gives it one: the function's constant table in the library
// says whether it declares it. Functions without the chunk decode do not.
constexpr int kq_fc_kquant_word = 370;
bool kq_kquant_word_on();

// Whether the codec has BK=128 NAX kernels (a pair or half loader).
bool kq_nax_has_bk128(const std::string& t);

// Whether the library function kname exists, and whether it declares the
// constant named `fc`. Reads the unspecialized function once.
struct KqFunctionInfo {
  bool exists = false;
  bool word = false;
};
inline KqFunctionInfo
kq_function_info(MTL::Library* lib, const std::string& kname, const char* fc) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto fn = NS::TransferPtr(lib->newFunction(
      NS::String::string(kname.c_str(), NS::UTF8StringEncoding)));
  KqFunctionInfo info;
  if (fn.get() == nullptr) {
    return info;
  }
  info.exists = true;
  NS::Dictionary* fcs = fn->functionConstantsDictionary();
  info.word = fcs != nullptr &&
      fcs->object(NS::String::string(fc, NS::UTF8StringEncoding)) != nullptr;
  return info;
}

// The pipelines of one kq_get_kernel key: whether the function declares
// the decode constant, and its pipeline for each value of the constant (a
// function without it has one pipeline, in slot 1).
struct KqPipelines {
  bool word = false;
  MTL::ComputePipelineState* ps[2] = {nullptr, nullptr};
};

// Fetch a kq kernel from the bundled metallib, specialized via an MTLFCList.
// hash_name must encode the func-const values so each specialization gets a
// distinct pipeline-state cache entry (d.get_kernel(name, lib, hash, consts)).
// A function that declares the decode constant gets the lever's value
// appended, and the "_kw0" suffix keys its off value; KQ_QMM_LOG prints
// "[kq decode] byte <key>" for each fetch with it off. A missing function
// raises kq_missing_kernel when the caller names the lever setting that
// selected it.
//
// The first fetch of a key reads the function's constants and builds the
// pipeline; both are cached here by key, under a lock, since engines on
// separate threads fetch concurrently. A later fetch costs one locked map
// lookup, which takes the place of the device's own lookup. The lever value
// picks the pipeline on every fetch, so a live flip switches forms. The
// pipelines live for the process: the device keeps them, and the
// "mlx_kquant" library is never cleared.
inline MTL::ComputePipelineState* kq_get_kernel(
    Device& d,
    const std::string& kname,
    const std::string& hash_name,
    const mx::metal::MTLFCList& func_consts,
    const char* setting = nullptr) {
  static MTL::Library* lib = d.get_library("mlx_kquant", metallib_dir());
  static std::shared_mutex mu;
  static std::unordered_map<std::string, KqPipelines> cache;
  const std::string& key = hash_name.empty() ? kname : hash_name;
  const int on = kq_kquant_word_on() ? 1 : 0;
  MTL::ComputePipelineState* ps = nullptr;
  bool byte = false;
  {
    std::shared_lock<std::shared_mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) {
      const int i = it->second.word ? on : 1;
      ps = it->second.ps[i];
      byte = i == 0;
    }
  }
  if (ps == nullptr) {
    std::unique_lock<std::shared_mutex> lock(mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
      const KqFunctionInfo info =
          kq_function_info(lib, kname, "kq_kquant_word");
      if (!info.exists) {
        if (setting != nullptr) {
          throw kq_missing_kernel(setting, kname);
        }
        // The device raises its load error; nothing is cached.
        return d.get_kernel(kname, lib, key, func_consts);
      }
      it = cache.emplace(key, KqPipelines{info.word, {nullptr, nullptr}}).first;
    }
    KqPipelines& p = it->second;
    const int i = p.word ? on : 1;
    if (p.ps[i] == nullptr) {
      if (p.word) {
        static const bool values[2] = {false, true};
        mx::metal::MTLFCList fc = func_consts;
        fc.emplace_back(
            &values[i], MTL::DataType::DataTypeBool, kq_fc_kquant_word);
        p.ps[i] = d.get_kernel(kname, lib, i ? key : key + "_kw0", fc);
      } else {
        p.ps[i] = d.get_kernel(kname, lib, key, func_consts);
      }
    }
    ps = p.ps[i];
    byte = i == 0;
  }
  if (byte && kq_qmm_log_on()) {
    std::fprintf(stderr, "[kq decode] byte %s_kw0\n", key.c_str());
  }
  return ps;
}

// Fetch a kq kernel with no function constants of its own.
inline MTL::ComputePipelineState* kq_get_kernel(
    Device& d,
    const std::string& kname) {
  return kq_get_kernel(d, kname, kname, {});
}

// The same, for a kernel a lever setting selects: a name the metallib does
// not carry raises with the setting.
inline MTL::ComputePipelineState*
kq_get_kernel(Device& d, const std::string& kname, const char* setting) {
  return kq_get_kernel(d, kname, kname, {}, setting);
}

} // namespace mlx_kquant

#endif // _METAL_
