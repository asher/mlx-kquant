// CPU decoders for the 10 GGUF codecs + dequant-then-matmul. The per-block
// decode is scalar and bit-exact per-codec against the gguf-py reference; no
// Metal deps, so this builds and runs on any platform with the stock mlx
// wheel. The matmul wrapper is performance-tuned but portable: a shared
// worker pool parallelizes over output rows / blocks, small-M matmuls run a
// fused decode-one-block-then-dot loop (no full-matrix scratch), and large-M
// matmuls dequantize once then run a GEMM — through Accelerate where
// available (KQ_USE_ACCELERATE), else a threaded scalar loop. The block
// decode math derives from ggml (llama.cpp, MIT) - see
// mlx_kquant/licenses/llama.cpp-LICENSE.
#include "kquant_cpu_decode.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include "kquant_codec.h"

#include "mlx/types/half_types.h" // float16_t, bfloat16_t

#ifdef KQ_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

inline float read_f16(const uint8_t* ptr) {
  _Float16 tmp;
  std::memcpy(&tmp, ptr, sizeof(_Float16));
  return static_cast<float>(tmp);
}

template <typename T>
void dequantize_q8_0(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 34;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
    T* dst = out + b * block_weights;
    for (int i = 0; i < block_weights; i++) {
      dst[i] = static_cast<T>(d * static_cast<float>(qs[i]));
    }
  }
}

template <typename T>
void dequantize_q4_0(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 18;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qs = block + 2;
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      int x0 = (qs[j] & 0x0F) - 8;
      int x1 = (qs[j] >> 4) - 8;
      dst[j] = static_cast<T>(d * static_cast<float>(x0));
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1));
    }
  }
}

template <typename T>
void dequantize_q4_1(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 20;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float m = read_f16(block + 2);
    const uint8_t* qs = block + 4;
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      int x0 = qs[j] & 0x0F;
      int x1 = qs[j] >> 4;
      dst[j] = static_cast<T>(d * static_cast<float>(x0) + m);
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1) + m);
    }
  }
}

template <typename T>
void dequantize_q5_0(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 22;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qh_bytes = block + 2;
    uint32_t qh = static_cast<uint32_t>(qh_bytes[0]) |
        (static_cast<uint32_t>(qh_bytes[1]) << 8) |
        (static_cast<uint32_t>(qh_bytes[2]) << 16) |
        (static_cast<uint32_t>(qh_bytes[3]) << 24);
    const uint8_t* qs = block + 6;
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      int xh_0 = ((qh >> j) << 4) & 0x10;
      int xh_1 = (qh >> (j + 12)) & 0x10;
      int x0 = (qs[j] & 0x0F) | xh_0;
      int x1 = (qs[j] >> 4) | xh_1;
      dst[j] = static_cast<T>(d * static_cast<float>(x0 - 16));
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1 - 16));
    }
  }
}

template <typename T>
void dequantize_q5_1(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 24;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float m = read_f16(block + 2);
    const uint8_t* qh_bytes = block + 4;
    const uint8_t* qs = block + 8;
    uint32_t qh;
    std::memcpy(&qh, qh_bytes, 4);
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      uint8_t xh_0 = ((qh >> j) << 4) & 0x10;
      uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
      uint8_t x0 = (qs[j] & 0x0F) | xh_0;
      uint8_t x1 = (qs[j] >> 4) | xh_1;
      dst[j] = static_cast<T>(d * static_cast<float>(x0) + m);
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1) + m);
    }
  }
}

inline void unpack_q4k_scales(
    const uint8_t* scales_packed,
    float* sc,
    float* mn,
    float d,
    float dmin) {
  for (int i = 0; i < 8; i++) {
    uint8_t raw_sc, raw_m;
    if (i < 4) {
      raw_sc = scales_packed[i] & 0x3F;
      raw_m = scales_packed[i + 4] & 0x3F;
    } else {
      raw_sc =
          (scales_packed[i + 4] & 0x0F) | ((scales_packed[i - 4] >> 6) << 4);
      raw_m = (scales_packed[i + 4] >> 4) | ((scales_packed[i] >> 6) << 4);
    }
    sc[i] = d * static_cast<float>(raw_sc);
    mn[i] = dmin * static_cast<float>(raw_m);
  }
}

template <typename T>
void dequantize_q4_k(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 144;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float dmin = read_f16(block + 2);
    const uint8_t* scales_packed = block + 4;
    const uint8_t* qs = block + 16;

    float sc[8], mn[8];
    unpack_q4k_scales(scales_packed, sc, mn, d, dmin);

    T* dst = out + b * block_weights;
    for (int g = 0; g < 4; g++) {
      for (int i = 0; i < 32; i++) {
        dst[(2 * g) * 32 + i] = static_cast<T>(
            sc[2 * g] * static_cast<float>(qs[g * 32 + i] & 0x0F) - mn[2 * g]);
        dst[(2 * g + 1) * 32 + i] = static_cast<T>(
            sc[2 * g + 1] * static_cast<float>(qs[g * 32 + i] >> 4) -
            mn[2 * g + 1]);
      }
    }
  }
}

template <typename T>
void dequantize_q5_k(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 176;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float dmin = read_f16(block + 2);
    const uint8_t* scales_packed = block + 4;
    const uint8_t* qh = block + 16;
    const uint8_t* qs = block + 48;

    float sc[8], mn[8];
    unpack_q4k_scales(scales_packed, sc, mn, d, dmin);

    T* dst = out + b * block_weights;
    for (int g = 0; g < 4; g++) {
      for (int i = 0; i < 32; i++) {
        uint8_t lo0 = qs[g * 32 + i] & 0x0F;
        uint8_t lo1 = qs[g * 32 + i] >> 4;
        uint8_t hi0 = (qh[i] >> (2 * g)) & 1;
        uint8_t hi1 = (qh[i] >> (2 * g + 1)) & 1;
        dst[(2 * g) * 32 + i] = static_cast<T>(
            sc[2 * g] * static_cast<float>(lo0 | (hi0 << 4)) - mn[2 * g]);
        dst[(2 * g + 1) * 32 + i] = static_cast<T>(
            sc[2 * g + 1] * static_cast<float>(lo1 | (hi1 << 4)) -
            mn[2 * g + 1]);
      }
    }
  }
}

template <typename T>
void dequantize_q6_k(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 210;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    const uint8_t* ql_base = block;
    const uint8_t* qh_base = block + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
    float d = read_f16(block + 208);

    T* dst = out + b * block_weights;
    for (int half = 0; half < 2; half++) {
      const uint8_t* ql = ql_base + half * 64;
      const uint8_t* qh = qh_base + half * 32;
      const int8_t* sc = scales + half * 8;
      T* out_half = dst + half * 128;

      for (int l = 0; l < 32; l++) {
        int is0 = l / 16;
        int8_t q1 =
            static_cast<int8_t>((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) -
            32;
        int8_t q2 = static_cast<int8_t>(
                        (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) -
            32;
        int8_t q3 =
            static_cast<int8_t>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int8_t q4 =
            static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) -
            32;
        out_half[l] = static_cast<T>(
            d * static_cast<float>(sc[is0]) * static_cast<float>(q1));
        out_half[l + 32] = static_cast<T>(
            d * static_cast<float>(sc[is0 + 2]) * static_cast<float>(q2));
        out_half[l + 64] = static_cast<T>(
            d * static_cast<float>(sc[is0 + 4]) * static_cast<float>(q3));
        out_half[l + 96] = static_cast<T>(
            d * static_cast<float>(sc[is0 + 6]) * static_cast<float>(q4));
      }
    }
  }
}

inline void unpack_q3k_scales(const uint8_t* s, int32_t* sc) {
  for (int k = 0; k < 4; k++) {
    sc[k] = static_cast<int32_t>(s[k] & 0x0F) |
        (static_cast<int32_t>((s[8 + k]) & 0x03) << 4);
    sc[k + 4] = static_cast<int32_t>(s[k + 4] & 0x0F) |
        (static_cast<int32_t>((s[8 + k] >> 2) & 0x03) << 4);
    sc[k + 8] = static_cast<int32_t>((s[k] >> 4) & 0x0F) |
        (static_cast<int32_t>((s[8 + k] >> 4) & 0x03) << 4);
    sc[k + 12] = static_cast<int32_t>((s[k + 4] >> 4) & 0x0F) |
        (static_cast<int32_t>((s[8 + k] >> 6) & 0x03) << 4);
  }
  for (int i = 0; i < 16; i++) {
    sc[i] -= 32;
  }
}

template <typename T>
void dequantize_q3_k(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 110;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    const uint8_t* hmask = block;
    const uint8_t* qs_full = block + 32;
    const uint8_t* scales_packed = block + 96;
    float d = read_f16(block + 108);

    int32_t sc[16];
    unpack_q3k_scales(scales_packed, sc);

    T* dst = out + b * block_weights;
    int out_idx = 0;
    for (int outer_half = 0; outer_half < 2; outer_half++) {
      const uint8_t* qs_chunk = qs_full + outer_half * 32;
      for (int shift_idx = 0; shift_idx < 4; shift_idx++) {
        int shift = shift_idx * 2;
        uint8_t m = 1 << (outer_half * 4 + shift_idx);
        int is_left = outer_half * 8 + shift_idx * 2;
        float dl_left = d * static_cast<float>(sc[is_left]);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l] >> shift) & 3;
          int h = (hmask[l] & m) ? 0 : 4;
          dst[out_idx++] = static_cast<T>(dl_left * static_cast<float>(q2 - h));
        }
        float dl_right = d * static_cast<float>(sc[is_left + 1]);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l + 16] >> shift) & 3;
          int h = (hmask[l + 16] & m) ? 0 : 4;
          dst[out_idx++] =
              static_cast<T>(dl_right * static_cast<float>(q2 - h));
        }
      }
    }
  }
}

template <typename T>
void dequantize_q2_k(const uint8_t* w, T* out, std::size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 84;
  std::size_t num_blocks = num_weights / block_weights;
  for (std::size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    const uint8_t* scales_raw = block;
    const uint8_t* qs_full = block + 16;
    float d = read_f16(block + 80);
    float dmin = read_f16(block + 82);

    T* dst = out + b * block_weights;
    int out_idx = 0;
    int is_idx = 0;
    for (int outer_half = 0; outer_half < 2; outer_half++) {
      const uint8_t* qs_chunk = qs_full + outer_half * 32;
      for (int shift_idx = 0; shift_idx < 4; shift_idx++) {
        int shift = shift_idx * 2;
        uint8_t sc_byte_left = scales_raw[is_idx++];
        float dl_left = d * static_cast<float>(sc_byte_left & 0x0F);
        float ml_left = dmin * static_cast<float>(sc_byte_left >> 4);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l] >> shift) & 3;
          dst[out_idx++] =
              static_cast<T>(dl_left * static_cast<float>(q2) - ml_left);
        }
        uint8_t sc_byte_right = scales_raw[is_idx++];
        float dl_right = d * static_cast<float>(sc_byte_right & 0x0F);
        float ml_right = dmin * static_cast<float>(sc_byte_right >> 4);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l + 16] >> shift) & 3;
          dst[out_idx++] =
              static_cast<T>(dl_right * static_cast<float>(q2) - ml_right);
        }
      }
    }
  }
}

// --------------------------------------------------------------------------
// Shared CPU worker pool
// --------------------------------------------------------------------------
//
// MLX's CPU command encoder runs each primitive's lambda on a single
// per-stream scheduler thread, so without this pool every kq.* CPU op is
// single-threaded. The pool is lazily created on first use and lives for the
// process; the dispatching thread participates in every job, so
// KQ_CPU_THREADS=1 (or a single-core box) degrades to plain inline execution
// with no thread traffic. Not reentrant by design — callers below only ever
// invoke it from the scheduler thread, never from inside a worker.

class KQThreadPool {
 public:
  static KQThreadPool& get() {
    static KQThreadPool pool;
    return pool;
  }

  int n_threads() const {
    return static_cast<int>(workers_.size()) + 1; // + calling thread
  }

  // Run fn(part) for part in [0, n_parts), distributing parts across the
  // workers and the calling thread. Blocks until every part has finished.
  void parallel(int n_parts, const std::function<void(int)>& fn) {
    if (n_parts <= 0) {
      return;
    }
    if (n_parts == 1 || workers_.empty()) {
      for (int p = 0; p < n_parts; p++) {
        fn(p);
      }
      return;
    }
    {
      std::lock_guard<std::mutex> lk(m_);
      job_fn_ = fn; // copy: workers may outlive the caller's frame otherwise
      job_parts_ = n_parts;
      job_next_.store(0, std::memory_order_relaxed);
      job_done_.store(0, std::memory_order_relaxed);
      generation_++;
      cv_.notify_all();
    }
    run_parts(); // calling thread participates
    // Wait until every part has run AND every woken worker has left
    // run_parts — a straggler that exhausted the part counter must not still
    // be touching job state when the next job's setup rewrites it.
    std::unique_lock<std::mutex> lk(m_);
    done_cv_.wait(
        lk, [&] { return job_done_.load() >= job_parts_ && in_flight_ == 0; });
  }

 private:
  KQThreadPool() {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    if (const char* env = std::getenv("KQ_CPU_THREADS")) {
      int v = std::atoi(env);
      if (v > 0) {
        n = v;
      }
    }
    if (n < 1) {
      n = 1;
    }
    if (n > 64) {
      n = 64;
    }
    for (int i = 0; i < n - 1; i++) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  ~KQThreadPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
      cv_.notify_all();
    }
    for (auto& t : workers_) {
      t.join();
    }
  }

  void run_parts() {
    int p;
    while ((p = job_next_.fetch_add(1)) < job_parts_) {
      job_fn_(p);
      job_done_.fetch_add(1);
    }
  }

  void worker_loop() {
    std::uint64_t seen = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
        if (stop_) {
          return;
        }
        seen = generation_;
        in_flight_++; // under m_: ordered against the caller's job setup
      }
      run_parts();
      {
        std::lock_guard<std::mutex> lk(m_);
        in_flight_--;
        done_cv_.notify_all();
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::function<void(int)> job_fn_;
  int job_parts_{0};
  std::atomic<int> job_next_{0};
  std::atomic<int> job_done_{0};
  int in_flight_{0}; // workers currently inside run_parts (guarded by m_)
  std::uint64_t generation_{0};
  bool stop_{false};
};

} // namespace

int kq_cpu_threads() {
  return KQThreadPool::get().n_threads();
}

void kq_parallel_for(
    std::size_t n_items,
    const std::function<void(std::size_t, std::size_t)>& fn) {
  if (n_items == 0) {
    return;
  }
  auto& pool = KQThreadPool::get();
  std::size_t parts = static_cast<std::size_t>(pool.n_threads());
  if (parts > n_items) {
    parts = n_items;
  }
  if (parts <= 1) {
    fn(0, n_items);
    return;
  }
  pool.parallel(static_cast<int>(parts), [&](int p) {
    std::size_t begin = n_items * p / parts;
    std::size_t end = n_items * (p + 1) / parts;
    if (begin < end) {
      fn(begin, end);
    }
  });
}

namespace {

// Function-pointer resolver for the float32 block decoders, so the hot matmul
// loops pay the codec-name string compare once per call instead of per block.
using DequantFnF32 = void (*)(const uint8_t*, float*, std::size_t);

DequantFnF32 dequant_fn_f32(const std::string& t) {
  if (t == "q8_0") {
    return &dequantize_q8_0<float>;
  } else if (t == "q4_0") {
    return &dequantize_q4_0<float>;
  } else if (t == "q4_1") {
    return &dequantize_q4_1<float>;
  } else if (t == "q5_0") {
    return &dequantize_q5_0<float>;
  } else if (t == "q5_1") {
    return &dequantize_q5_1<float>;
  } else if (t == "q4_k") {
    return &dequantize_q4_k<float>;
  } else if (t == "q5_k") {
    return &dequantize_q5_k<float>;
  } else if (t == "q6_k") {
    return &dequantize_q6_k<float>;
  } else if (t == "q3_k") {
    return &dequantize_q3_k<float>;
  } else if (t == "q2_k") {
    return &dequantize_q2_k<float>;
  }
  return nullptr;
}

// Multi-accumulator dot so the compiler can vectorize without -ffast-math
// (a single float accumulator forbids reassociation). n is a whole number of
// codec blocks, always a multiple of 8.
inline float dot_block(const float* a, const float* b, int n) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  float s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;
  for (int i = 0; i < n; i += 8) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4];
    s5 += a[i + 5] * b[i + 5];
    s6 += a[i + 6] * b[i + 6];
    s7 += a[i + 7] * b[i + 7];
  }
  return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

// Small-M GEMV ceiling for the fused path; larger M goes through
// dequantize-once + GEMM, which amortizes the decode across rows instead.
constexpr int kMaxFusedM = 16;

// Fused decode-then-dot over output rows [n0, n1) for transpose_w=true
// (w decodes to [N, K] row-major, the weight convention of every model
// matmul). One 256-weight block is decoded into a stack buffer and
// immediately dotted against all M activation rows — wire bytes are read
// once and no [N, K] scratch is ever materialized, which is what makes
// memory-bound decode honest.
void qmv_fused_rows(
    float* outf, // [M, N]
    const float* xf, // [M, K]
    const uint8_t* w,
    int M,
    int N,
    int K,
    std::size_t row_bytes,
    int block_weights,
    std::size_t bytes_per_block,
    DequantFnF32 fn,
    std::size_t n0,
    std::size_t n1) {
  float buf[256]; // max weights_per_block across codecs
  float acc[kMaxFusedM];
  const int nblocks = K / block_weights;
  for (std::size_t n = n0; n < n1; n++) {
    const uint8_t* wr = w + n * row_bytes;
    for (int m = 0; m < M; m++) {
      acc[m] = 0.0f;
    }
    for (int b = 0; b < nblocks; b++) {
      fn(wr + b * bytes_per_block, buf, block_weights);
      const float* xb = xf + static_cast<std::size_t>(b) * block_weights;
      for (int m = 0; m < M; m++) {
        acc[m] +=
            dot_block(xb + static_cast<std::size_t>(m) * K, buf, block_weights);
      }
    }
    for (int m = 0; m < M; m++) {
      outf[static_cast<std::size_t>(m) * N + n] = acc[m];
    }
  }
}

template <typename T>
void convert_to_f32(const T* src, float* dst, std::size_t n) {
  kq_parallel_for(n, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; i++) {
      dst[i] = static_cast<float>(src[i]);
    }
  });
}

template <typename T>
void convert_from_f32(const float* src, T* dst, std::size_t n) {
  kq_parallel_for(n, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; i++) {
      dst[i] = static_cast<T>(src[i]);
    }
  });
}

} // namespace

template <typename T>
void kquant_dequantize_dispatch(
    const uint8_t* w,
    T* out,
    std::size_t num_weights,
    const std::string& kquant_type) {
  if (kquant_type == "q8_0") {
    dequantize_q8_0(w, out, num_weights);
  } else if (kquant_type == "q4_0") {
    dequantize_q4_0(w, out, num_weights);
  } else if (kquant_type == "q4_1") {
    dequantize_q4_1(w, out, num_weights);
  } else if (kquant_type == "q5_0") {
    dequantize_q5_0(w, out, num_weights);
  } else if (kquant_type == "q5_1") {
    dequantize_q5_1(w, out, num_weights);
  } else if (kquant_type == "q4_k") {
    dequantize_q4_k(w, out, num_weights);
  } else if (kquant_type == "q5_k") {
    dequantize_q5_k(w, out, num_weights);
  } else if (kquant_type == "q6_k") {
    dequantize_q6_k(w, out, num_weights);
  } else if (kquant_type == "q3_k") {
    dequantize_q3_k(w, out, num_weights);
  } else if (kquant_type == "q2_k") {
    dequantize_q2_k(w, out, num_weights);
  } else {
    throw std::runtime_error(
        "[mlx_kquant] dequantize: unsupported codec: " + kquant_type);
  }
}

template <typename T>
void kquant_dequantize_parallel(
    const uint8_t* w,
    T* out,
    std::size_t num_weights,
    const std::string& kquant_type) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::runtime_error(
        "[mlx_kquant] dequantize: unsupported codec: " + kquant_type);
  }
  const std::size_t bw = codec->weights_per_block;
  const std::size_t bb = codec->bytes_per_block;
  const std::size_t n_blocks = num_weights / bw;
  kq_parallel_for(n_blocks, [&](std::size_t b0, std::size_t b1) {
    kquant_dequantize_dispatch(
        w + b0 * bb, out + b0 * bw, (b1 - b0) * bw, kquant_type);
  });
}

template <typename T>
void kquant_qmm_cpu(
    T* result,
    const T* x,
    const uint8_t* w,
    int M,
    int N,
    int K,
    bool transpose_w,
    const std::string& kquant_type) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::runtime_error(
        "[mlx_kquant] quantized_matmul: unsupported codec: " + kquant_type);
  }
  const int w_rows = transpose_w ? N : K;
  const int w_cols = transpose_w ? K : N;
  const std::size_t row_bytes =
      (static_cast<std::size_t>(w_cols) / codec->weights_per_block) *
      codec->bytes_per_block;
  DequantFnF32 fn = dequant_fn_f32(kquant_type);

  // Stage activations (and, for half outputs, the result) in float32. The
  // accumulation dtype is float either way; this just hoists the per-element
  // casts out of the inner loops.
  const std::size_t x_els = static_cast<std::size_t>(M) * K;
  const std::size_t out_els = static_cast<std::size_t>(M) * N;
  std::unique_ptr<float[]> xf_store;
  const float* xf;
  if constexpr (std::is_same_v<T, float>) {
    xf = x;
  } else {
    xf_store.reset(new float[x_els]);
    convert_to_f32(x, xf_store.get(), x_els);
    xf = xf_store.get();
  }
  std::unique_ptr<float[]> of_store;
  float* of;
  if constexpr (std::is_same_v<T, float>) {
    of = result;
  } else {
    of_store.reset(new float[out_els]);
    of = of_store.get();
  }

  if (transpose_w && M <= kMaxFusedM && fn != nullptr) {
    // Decode/GEMV shape: fused decode-then-dot, parallel over output rows.
    // No scratch matrix; wire bytes are read exactly once.
    kq_parallel_for(
        static_cast<std::size_t>(N), [&](std::size_t n0, std::size_t n1) {
          qmv_fused_rows(
              of,
              xf,
              w,
              M,
              N,
              K,
              row_bytes,
              codec->weights_per_block,
              codec->bytes_per_block,
              fn,
              n0,
              n1);
        });
  } else {
    // Prefill/GEMM shape: dequantize the weight matrix once (parallel over
    // rows, uninitialized scratch — every element is overwritten), then GEMM.
    std::unique_ptr<float[]> w_dec(
        new float[static_cast<std::size_t>(w_rows) * w_cols]);
    kq_parallel_for(
        static_cast<std::size_t>(w_rows), [&](std::size_t r0, std::size_t r1) {
          for (std::size_t r = r0; r < r1; r++) {
            kquant_dequantize_dispatch(
                w + r * row_bytes,
                w_dec.get() + r * w_cols,
                w_cols,
                kquant_type);
          }
        });

#ifdef KQ_USE_ACCELERATE
    // Accelerate's sgemm engages the AMX/SME matrix units — far past what
    // scalar (or even NEON) per-core loops can reach for M x N x K work.
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        transpose_w ? CblasTrans : CblasNoTrans,
        M,
        N,
        K,
        1.0f,
        xf,
        K,
        w_dec.get(),
        w_cols,
        0.0f,
        of,
        N);
#else
    kq_parallel_for(
        static_cast<std::size_t>(N), [&](std::size_t n0, std::size_t n1) {
          for (int m = 0; m < M; m++) {
            const float* xm = xf + static_cast<std::size_t>(m) * K;
            for (std::size_t n = n0; n < n1; n++) {
              float acc = 0.0f;
              if (transpose_w) {
                const float* wn = w_dec.get() + n * K;
                for (int k = 0; k < K; k++) {
                  acc += xm[k] * wn[k];
                }
              } else {
                for (int k = 0; k < K; k++) {
                  acc += xm[k] * w_dec[static_cast<std::size_t>(k) * N + n];
                }
              }
              of[static_cast<std::size_t>(m) * N + n] = acc;
            }
          }
        });
#endif
  }

  if constexpr (!std::is_same_v<T, float>) {
    convert_from_f32(of, result, out_els);
  }
}

// Explicit instantiations for the float types the eval paths dispatch over.
template void kquant_dequantize_dispatch<float>(
    const uint8_t*,
    float*,
    std::size_t,
    const std::string&);
template void kquant_dequantize_dispatch<mx::float16_t>(
    const uint8_t*,
    mx::float16_t*,
    std::size_t,
    const std::string&);
template void kquant_dequantize_dispatch<mx::bfloat16_t>(
    const uint8_t*,
    mx::bfloat16_t*,
    std::size_t,
    const std::string&);

template void kquant_dequantize_parallel<float>(
    const uint8_t*,
    float*,
    std::size_t,
    const std::string&);
template void kquant_dequantize_parallel<mx::float16_t>(
    const uint8_t*,
    mx::float16_t*,
    std::size_t,
    const std::string&);
template void kquant_dequantize_parallel<mx::bfloat16_t>(
    const uint8_t*,
    mx::bfloat16_t*,
    std::size_t,
    const std::string&);

template void kquant_qmm_cpu<float>(
    float*,
    const float*,
    const uint8_t*,
    int,
    int,
    int,
    bool,
    const std::string&);
template void kquant_qmm_cpu<mx::float16_t>(
    mx::float16_t*,
    const mx::float16_t*,
    const uint8_t*,
    int,
    int,
    int,
    bool,
    const std::string&);
template void kquant_qmm_cpu<mx::bfloat16_t>(
    mx::bfloat16_t*,
    const mx::bfloat16_t*,
    const uint8_t*,
    int,
    int,
    int,
    bool,
    const std::string&);

} // namespace mlx_kquant
