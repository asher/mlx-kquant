// Shared by the sdpa_paged_rows kernels of kq_sdpa.h (simdgroup matrices)
// and kq_sdpa_nax.h (tensor ops).

#pragma once

// Paged attention for several queries, each with its own page list (the
// verify block of a block-sparse attention model: every query selected its
// own top-k pages). Mirrored in kquant_sdpa.cpp; layouts must match field
// for field. Key row r reads keys / values when r < base_len and row
// r - base_len of the tails otherwise, so a cache kept as a base and a
// decode tail is read in place.
struct KQPagedRowsMeta {
  int N; // key capacity: rows at or past it are never read
  int n_pages; // pages per query
  int q_len; // queries per batch row
  int base_len; // first key row of the tails
  int page_heads; // 1: a page list per query, else per (kv head, query)
  int tail_page; // 1: each list ends with page ends / PR, implied
  float scale;
  int win; // shared-tile form: queries a window (0 in the per-query form)
  int gqa; // query heads per kv head
  int pad0;
  ulong k_batch;
  ulong k_head;
  ulong k_seq;
  ulong v_batch;
  ulong v_head;
  ulong v_seq;
  ulong kt_batch;
  ulong kt_head;
  ulong kt_seq;
  ulong vt_batch;
  ulong vt_head;
  ulong vt_seq;
};

// Pages the shared-tile forms mark and compact at a time.
#define KQ_PR_UNION_MAX 512
