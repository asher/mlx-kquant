// KV cache append: writes x [B, H, n, D] (any strides) into rows
// [offset, offset + n) of axis 2 of a row-major [B, H, cap, D] buffer that
// the output shares with the cache. One thread per element of x.

struct KqKvAppendArgs {
  int shape[4];
  int64_t strides[4];
  int64_t cap;
  int64_t offset;
  int64_t size;
};

template <typename T>
[[kernel]] void kq_kv_append(
    const device T* x [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant KqKvAppendArgs& a [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= a.size) {
    return;
  }
  const int64_t D = a.shape[3];
  const int64_t n = a.shape[2];
  const int64_t H = a.shape[1];
  int64_t rem = gid;
  const int64_t d = rem % D;
  rem /= D;
  const int64_t t = rem % n;
  rem /= n;
  const int64_t h = rem % H;
  const int64_t b = rem / H;
  const int64_t src =
      b * a.strides[0] + h * a.strides[1] + t * a.strides[2] + d * a.strides[3];
  out[((b * H + h) * a.cap + a.offset + t) * D + d] = x[src];
}
