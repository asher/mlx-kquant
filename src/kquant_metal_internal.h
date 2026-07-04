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

#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "kquant.h" // metallib_dir
#include "kquant_codec.h" // codec_by_name

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
      kquant_type == "iq1_m") {
    return 4;
  }
  return 8;
}

// quantized.cpp:63-65 (kquant branch). KQuant blocks are 32 or 256 weights.
inline int qmv_fast_k_align() {
  return 256;
}

inline bool codec_has_matmul(const std::string& kquant_type) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  return codec != nullptr && codec->has_matmul_kernel;
}

// Gates the NAX (tensor-core) dispatch. IQ codecs ship ALU-only, so this is
// false for them and their qmm/gather route to the ALU kernels.
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

// Codecs with a verify_qmv kernel (the small-M weight-read-amortizing leaf).
// Kept as an explicit allow-list so the dispatch only routes to a kernel that
// was actually instantiated in kq_quantized.metal; new codecs are added here
// once their verify_qmv instantiation lands.
inline bool codec_has_verify_qmv(const std::string& kquant_type) {
  return kquant_type == "q6_k" || kquant_type == "q8_0" ||
      kquant_type == "q4_k" || kquant_type == "q5_k" || kquant_type == "q5_1" ||
      kquant_type == "q3_k" || kquant_type == "q2_k" || kquant_type == "q4_0" ||
      kquant_type == "q4_1" || kquant_type == "q5_0";
}

// Largest activation-row count (M) the verify_qmv kernels are instantiated for;
// must match MAX_VM in kq_*_verify_qmv_impl. M above this falls back to qmv.
inline int verify_qmv_max_rows() {
  return 8;
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

// Fetch a kq kernel from the bundled metallib.
//
// The library handle is resolved exactly once and cached in a function-local
// static: get_library() is a locked map lookup keyed by name, and
// metallib_dir() returns a fresh std::string copy of the .so directory. The GPU
// Device is a process singleton and the handle it owns lives for the process,
// so caching the raw pointer is safe. Only the per-kname get_kernel() lookup
// remains per call.
inline MTL::ComputePipelineState* kq_get_kernel(
    Device& d,
    const std::string& kname) {
  static MTL::Library* lib = d.get_library("mlx_kquant", metallib_dir());
  return d.get_kernel(kname, lib);
}

// Func-constant variant for kernels specialized via an MTLFCList (the only kq
// consumer is gather_qmm_rhs_nax: align_M/N/K at constant ids 200/201/202).
// hash_name must encode the func-const values so each specialization gets a
// distinct pipeline-state cache entry (d.get_kernel(name, lib, hash, consts)).
inline MTL::ComputePipelineState* kq_get_kernel(
    Device& d,
    const std::string& kname,
    const std::string& hash_name,
    const mx::metal::MTLFCList& func_consts) {
  static MTL::Library* lib = d.get_library("mlx_kquant", metallib_dir());
  return d.get_kernel(kname, lib, hash_name, func_consts);
}

} // namespace mlx_kquant

#endif // _METAL_
