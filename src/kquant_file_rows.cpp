// Rows of a table read straight from a file (see file_rows in kquant.h).
// The read is a node on a CPU stream, so a caller builds its graph before
// the row ids exist and the reads start when they do. Rows are read in
// parallel: a cold row is one SSD read, and a decode step wants tens of them.
// Reads skip the page cache where the OS allows (F_NOCACHE): a prefill
// touches one page per row, and a cache filled at drive speed makes the
// kernel compress other memory. Such a read stays on the direct path only
// when the file offset and the destination are both page-aligned, so each
// row is read as the whole pages it sits in, into an aligned buffer.
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#include "kquant.h"
#include "kquant_internal.h"

#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

std::atomic<uint64_t> g_failures{0};

struct RowJob {
  int fd;
  const int64_t* ids;
  uint8_t* out;
  int64_t n;
  int64_t offset;
  int64_t row_bytes;
  int64_t rows;
  int64_t file_size;
  int64_t page;
  int64_t chunk;
};

// Rows [lo, hi) of the job. A row that cannot be read is zeros and counts
// as a failure: this runs on a scheduler thread, where a throw would end
// the process.
void read_rows(const RowJob& j, int64_t lo, int64_t hi) {
  const int64_t mask = ~(j.page - 1);
  const int64_t cap = ((j.row_bytes + j.page - 1) & mask) + j.page;
  void* mem = nullptr;
  if (::posix_memalign(
          &mem, static_cast<size_t>(j.page), static_cast<size_t>(cap)) != 0) {
    mem = nullptr;
  }
  uint8_t* bounce = static_cast<uint8_t*>(mem);
  for (int64_t i = lo; i < hi; i++) {
    uint8_t* dst = j.out + i * j.row_bytes;
    const int64_t id = j.ids[i];
    bool ok = bounce != nullptr && id >= 0 && id < j.rows;
    if (ok) {
      const int64_t off = j.offset + id * j.row_bytes;
      const int64_t a = off & mask;
      const int64_t b =
          std::min((off + j.row_bytes + j.page - 1) & mask, j.file_size);
      int64_t done = 0;
      while (ok && done < b - a) {
        const ssize_t got = ::pread(
            j.fd,
            bounce + done,
            static_cast<size_t>(b - a - done),
            static_cast<off_t>(a + done));
        if (got > 0) {
          done += got;
        } else if (got == 0 || errno != EINTR) {
          ok = false;
        }
      }
      if (ok) {
        std::memcpy(dst, bounce + (off - a), static_cast<size_t>(j.row_bytes));
      }
    }
    if (!ok) {
      std::memset(dst, 0, static_cast<size_t>(j.row_bytes));
      g_failures.fetch_add(1, std::memory_order_relaxed);
    }
  }
  std::free(mem);
}

void read_chunk(void* ctx, size_t c) {
  const RowJob& j = *static_cast<const RowJob*>(ctx);
  const int64_t lo = static_cast<int64_t>(c) * j.chunk;
  read_rows(j, lo, std::min(j.n, lo + j.chunk));
}

void read_all(RowJob& j) {
  // Reads wait on the SSD, not the CPU, so more of them are in flight
  // than there are cores.
  constexpr int64_t kMaxChunks = 32;
  constexpr int64_t kMinChunk = 2;
  const int64_t chunks =
      std::max<int64_t>(1, std::min(kMaxChunks, j.n / kMinChunk));
  j.chunk = (j.n + chunks - 1) / chunks;
  if (chunks == 1) {
    read_rows(j, 0, j.n);
    return;
  }
#ifdef __APPLE__
  dispatch_apply_f(
      static_cast<size_t>(chunks),
      dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
      &j,
      read_chunk);
#else
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(chunks));
  for (int64_t c = 0; c < chunks; c++) {
    pool.emplace_back(read_chunk, &j, static_cast<size_t>(c));
  }
  for (auto& t : pool) {
    t.join();
  }
#endif
}

} // namespace

KQuantFileRows::Fd::~Fd() {
  if (fd >= 0) {
    ::close(fd);
  }
}

uint64_t file_rows_failures() {
  return g_failures.load(std::memory_order_relaxed);
}

mx::array file_rows(
    mx::array ids,
    const std::string& path,
    int64_t offset,
    int64_t row_bytes,
    int64_t rows,
    mx::StreamOrDevice s_) {
  auto s = std::holds_alternative<std::monostate>(s_)
      ? mx::default_stream(mx::Device::cpu)
      : mx::to_stream(s_);
  if (s.device != mx::Device::cpu) {
    throw std::invalid_argument(
        "[mlx_kquant.file_rows] reads on a CPU stream; pass one, or none.");
  }
  if (offset < 0 || row_bytes <= 0 || rows < 0) {
    throw std::invalid_argument(
        "[mlx_kquant.file_rows] offset and rows must be at least 0 and "
        "row_bytes at least 1.");
  }
  if (mx::issubdtype(ids.dtype(), mx::floating) || ids.dtype() == mx::bool_) {
    throw std::invalid_argument(
        "[mlx_kquant.file_rows] ids must be an integer array.");
  }
  auto fd = std::make_shared<KQuantFileRows::Fd>();
  fd->fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd->fd < 0) {
    throw std::runtime_error(
        "[mlx_kquant.file_rows] cannot open " + path + ": " +
        std::strerror(errno));
  }
#ifdef F_NOCACHE
  ::fcntl(fd->fd, F_NOCACHE, 1);
#endif
  const off_t size = ::lseek(fd->fd, 0, SEEK_END);
  fd->size = static_cast<int64_t>(size);
  if (size < 0 || rows > (static_cast<int64_t>(size) - offset) / row_bytes) {
    throw std::invalid_argument(
        "[mlx_kquant.file_rows] " + path + " ends before row " +
        std::to_string(rows) + " of " + std::to_string(row_bytes) +
        " bytes from offset " + std::to_string(offset) + ".");
  }
  auto shape = ids.shape();
  shape.push_back(static_cast<int>(row_bytes));
  auto flat = mx::contiguous(mx::astype(ids, mx::int64, s), false, s);
  return mx::array(
      std::move(shape),
      mx::uint8,
      std::make_shared<KQuantFileRows>(
          s, std::move(fd), offset, row_bytes, rows),
      {std::move(flat)});
}

void KQuantFileRows::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(inputs[0]);
  encoder.set_output_array(out);
  encoder.dispatch([ids = mx::array::unsafe_weak_copy(inputs[0]),
                    out = mx::array::unsafe_weak_copy(out),
                    fd = fd_,
                    offset = offset_,
                    row_bytes = row_bytes_,
                    rows = rows_]() mutable {
    RowJob j{
        fd->fd,
        ids.data<int64_t>(),
        out.data<uint8_t>(),
        static_cast<int64_t>(ids.size()),
        offset,
        row_bytes,
        rows,
        fd->size,
        static_cast<int64_t>(::sysconf(_SC_PAGESIZE)),
        0};
    if (j.n > 0) {
      read_all(j);
    }
  });
}

void KQuantFileRows::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.file_rows] reads on a CPU stream.");
}

} // namespace mlx_kquant
