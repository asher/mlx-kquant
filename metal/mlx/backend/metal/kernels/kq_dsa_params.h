// Host/device-shared parameter blocks for the kq DSA kernels
// (kq_dsa_sparse_attn.h / kq_dsa_indexer.h; ported from omlx glm_moe_dsa).
// Included by the Metal TUs (after utils.h, which supplies int64_t) and via
// relative paths by src/kquant_dsa_*.cpp -- keep it include-free so both
// compilers accept it, and keep field order/layout in sync with all consumers.
#pragma once

struct KQDsaSparseAttentionParams {
  int B; ///< Batch size
  int H; ///< Query heads
  int qL; ///< Query sequence length
  int localL; ///< Local rotating KV length
  int pooledL; ///< Number of pooled KV tokens
  int topk; ///< Number of selected pooled tokens per query
  int local_window; ///< Sliding-window size for local KV
  int compress_ratio; ///< Pooled compression ratio
  int q_offset; ///< Absolute query offset before this chunk

  float scale; ///< Attention scale

  int64_t Q_strides[3]; ///< Query strides (B, H, L, D = 1)
  int64_t Local_strides[3]; ///< Local KV strides (B, 1, L, D = 1)
  int64_t Pooled_strides[2]; ///< Pooled KV strides (B, L, D = 1)
  int64_t Topk_strides[3]; ///< Top-k strides (B, 1, L, topk = 1)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

// Parameters for the kq DSA radix top-k select (kq_dsa_topk_indices_16bit).
// Layout matches omlx's OMLXDSATopKParams field-for-field.
struct KQDsaTopKParams {
  int rows; ///< B * L score rows
  int L; ///< Query length (per-row causal position = row % L)
  int K; ///< Scores per row
  int topk; ///< Selection count (also baked into the kernel template)
  bool causal_valid_prefix; ///< Clamp the scan to the causal prefix
};
