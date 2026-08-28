// LoRA epilogue for the kq matmul primitives: out += fac * (x @ A[e]) @ B[e]
// as a second dispatch on the primitive's own output, after whatever base
// route (qmv, verify, split-K, NAX, qmm, gathered) wrote it. Codec
// independent, so a live GGUF LoRA adapter costs zero extra graph ops on
// every codec. The op layer validates and appends the operands to the
// primitive's inputs (kq_lora_prep); eval reads them back through
// kq_lora_view and hands them to the Metal (or CPU) epilogue.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "mlx/mlx.h"

#include "mlx/backend/cpu/encoder.h"
#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

// Kernel FLAGS bits (mirrored in kq_lora_epilogue.h) plus the host-only
// presence bit the primitives store.
constexpr int KQ_LORA_HAS_IDS = 1;
constexpr int KQ_LORA_HAS_TABLE = 2;
constexpr int KQ_LORA_HAS_FAC = 4;
constexpr int KQ_LORA_PRESENT = 8;
constexpr int KQ_LORA_KERNEL_FLAGS = 7;
// Threadgroup memory for the rank-r intermediates: slots * rank floats.
constexpr int KQ_LORA_MAX_Z = 512;
constexpr int KQ_LORA_MAX_S = 16;

// Validate the optional LoRA operands of a kq matmul op, contiguize them and
// append them to `inputs` in the fixed order a, b, [ids], [table], [fac].
// Returns the flags (0 when no adapter is given). n_experts == 0 is the
// dense case (a [K, r], b [r, N], no ids/table); otherwise a [E, K, r] and
// b [E, r, N] with E the expert-stack size. `rows` is the shape ids and fac
// must carry ([R] dense, [T, S] gathered); `slots` is S (1 for the
// per-row kernels) for the threadgroup-memory bound.
int kq_lora_prep(
    const char* op,
    const std::optional<mx::array>& a,
    const std::optional<mx::array>& b,
    const std::optional<mx::array>& ids,
    const std::optional<mx::array>& table,
    const std::optional<mx::array>& fac,
    mx::Dtype dtype,
    int K,
    int N,
    int n_experts,
    const mx::Shape& rows,
    int slots,
    mx::StreamOrDevice s,
    std::vector<mx::array>& inputs);

// How many inputs kq_lora_prep appended for `flags`.
inline int kq_lora_input_count(int flags) {
  if (!(flags & KQ_LORA_PRESENT)) {
    return 0;
  }
  return 2 + ((flags & KQ_LORA_HAS_IDS) ? 1 : 0) +
      ((flags & KQ_LORA_HAS_TABLE) ? 1 : 0) +
      ((flags & KQ_LORA_HAS_FAC) ? 1 : 0);
}

struct KqLoraView {
  std::optional<mx::array> a;
  std::optional<mx::array> b;
  std::optional<mx::array> ids;
  std::optional<mx::array> table;
  std::optional<mx::array> fac;
  int flags = 0;
  int rank = 0;
};

// The operands appended by kq_lora_prep, read back from a primitive's inputs
// starting at `base` (the count of the op's own inputs). Layout is not
// checked here: the epilogues walk every operand as dense row-major memory,
// and an operand's strides are only known once it is evaluated (a lazy
// broadcast such as mx.repeat of a single value evaluates to a stride-0
// view), so the eval-time constructors below densify what needs it.
KqLoraView
kq_lora_view(const std::vector<mx::array>& inputs, size_t base, int flags);

// Eval-time views: every operand that is not row-contiguous is copied into
// a dense temporary on the primitive's stream (released with the command
// buffer / encoder), so the epilogue kernels never read a strided view.
KqLoraView kq_lora_view_cpu(
    const std::vector<mx::array>& inputs,
    size_t base,
    int flags,
    mx::cpu::CommandEncoder& encoder,
    const mx::Stream& s);
// CPU capture helper: densifies via kq_lora_view_cpu, registers every operand
// as an encoder input and returns weak copies in kq_lora_view order, so the
// dispatch lambda reads them back with kq_lora_view(lora, 0, flags).
std::vector<mx::array> kq_lora_capture_cpu(
    const std::vector<mx::array>& inputs,
    size_t base,
    int flags,
    mx::cpu::CommandEncoder& encoder,
    const mx::Stream& s);
#ifdef _METAL_
KqLoraView kq_lora_view_gpu(
    const std::vector<mx::array>& inputs,
    size_t base,
    int flags,
    const mx::Stream& s);
#endif

// CPU epilogue (the per-row kernel's semantics) for the dense ops' eval_cpu.
template <typename T>
void kq_lora_epilogue_rows_cpu(
    T* out,
    const T* x,
    const KqLoraView& v,
    int R,
    int K,
    int N) {
  const int rank = v.rank;
  const T* a = v.a->data<T>();
  const T* b = v.b->data<T>();
  const uint32_t* ids = v.ids ? v.ids->data<uint32_t>() : nullptr;
  const int32_t* table = v.table ? v.table->data<int32_t>() : nullptr;
  const float* fac = v.fac ? v.fac->data<float>() : nullptr;
  std::vector<float> z(rank);
  for (int r = 0; r < R; r++) {
    int e = ids ? static_cast<int>(ids[r]) : 0;
    if (table) {
      e = table[e];
      if (e < 0) {
        continue;
      }
    }
    const float f = fac ? fac[r] : 1.0f;
    if (f == 0.0f) {
      continue;
    }
    const T* xrow = x + static_cast<int64_t>(r) * K;
    const T* ae = a + static_cast<int64_t>(e) * K * rank;
    const T* be = b + static_cast<int64_t>(e) * rank * N;
    for (int j = 0; j < rank; j++) {
      float acc = 0.0f;
      for (int k = 0; k < K; k++) {
        acc += static_cast<float>(xrow[k]) *
            static_cast<float>(ae[static_cast<int64_t>(k) * rank + j]);
      }
      z[j] = acc;
    }
    T* orow = out + static_cast<int64_t>(r) * N;
    for (int n = 0; n < N; n++) {
      float acc = 0.0f;
      for (int j = 0; j < rank; j++) {
        acc += z[j] * static_cast<float>(be[static_cast<int64_t>(j) * N + n]);
      }
      orow[n] = static_cast<T>(static_cast<float>(orow[n]) + f * acc);
    }
  }
}

#ifdef _METAL_
// out [R, N] += per-row delta; x [R, K] row-contiguous.
void kq_lora_epilogue_rows_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& x,
    const KqLoraView& v,
    mx::array& out,
    int R,
    int K,
    int N);

// Mix LoRA in two dispatches around the base gather. kq_lora_mix_z_gpu goes
// BEFORE the base dispatch (its inputs are x and A only, so it overlaps the
// gather) and returns the float32 z scratch [T*S, rank] (a command-encoder
// temporary); kq_lora_mix_apply_gpu goes after and adds the mixed delta to
// out [T, N]. x [T, S, K], scores [T, S] float32.
mx::array kq_lora_mix_z_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& x,
    const KqLoraView& v,
    int T,
    int S,
    int K);

void kq_lora_mix_apply_gpu(
    mx::metal::Device& d,
    const mx::Stream& s,
    const mx::array& z,
    const mx::array& scores,
    const KqLoraView& v,
    mx::array& out,
    int T,
    int S,
    int N);
#endif

} // namespace mlx_kquant
