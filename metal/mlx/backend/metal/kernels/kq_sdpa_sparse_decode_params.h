// Host/device-shared parameter block for kq_sdpa_sparse_decode.h. Included
// by the Metal TU (after utils.h, which supplies int64_t) and via a relative
// path by src/kquant_sdpa_sparse_decode.cpp: keep it include-free.
#pragma once

struct KQSdpaSparseDecodeParams {
  int B; ///< Batch
  int H; ///< Query heads
  int Hp; ///< H rounded up to the 8-head tile (partials layout)
  int L; ///< Query positions
  int W; ///< Window rows (all listed first)
  int P; ///< Pool rows (the index list selects among them)
  int N; ///< Listed pool rows per query
  int n_splits; ///< Key splits (grid x)
  int keys_per_split; ///< ceil((W + N) / n_splits)
  int has_sinks;
  int has_win_mask;
  int has_sel_mask;
  int direct; ///< One split: the split kernel normalizes and writes O
  float scale_log2; ///< softmax scale times log2(e)

  int64_t q_strides[3]; ///< B, H, L (D contiguous)
  int64_t win_strides[2]; ///< B, row
  int64_t pool_strides[2]; ///< B, row
  int64_t idx_strides[2]; ///< B, L (N contiguous)
  int64_t win_mask_strides[2]; ///< B (0 = broadcast), L (W contiguous)
  int64_t sel_mask_strides[2]; ///< B (0 = broadcast), L (N contiguous)
  int64_t o_strides[3]; ///< B, H, L
};
