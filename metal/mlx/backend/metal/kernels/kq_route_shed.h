// Routed-expert slot remap + residency shed (gpu-dispatch front end).
// One thread per token row: R <= 64 entries of serial work per thread, so
// occupancy is irrelevant and launch cost dominates. Semantics and f32
// accumulation order must stay bit-identical to KQuantRouteShed::eval_cpu.

#define KQ_ROUTE_SHED_MAX_R 64

[[kernel]] void kq_route_shed(
    const device uint32_t* indices [[buffer(0)]],
    const device float* scores [[buffer(1)]],
    const device int32_t* slot_table [[buffer(2)]],
    device uint32_t* slots [[buffer(3)]],
    device float* mix [[buffer(4)]],
    device int32_t* miss_ids [[buffer(5)]],
    device float* miss_scores [[buffer(6)]],
    const constant int& R [[buffer(7)]],
    const constant int& E [[buffer(8)]],
    const constant int& T [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  if (int(tid) >= T) {
    return;
  }
  const device uint32_t* id = indices + tid * R;
  const device float* sc = scores + tid * R;
  device uint32_t* sl = slots + tid * R;
  device float* mo = mix + tid * R;
  device int32_t* mi = miss_ids + tid * R;
  device float* ms = miss_scores + tid * R;

  // Pass 1: residency, mass sums, first kept slot. Ascending-r f32
  // accumulation order is part of the op contract (CPU parity).
  int32_t slot_of[KQ_ROUTE_SHED_MAX_R];
  float s_all = 0.0f;
  float s_kept = 0.0f;
  int32_t first_kept = 0;
  bool have_kept = false;
  for (int r = 0; r < R; r++) {
    const int32_t e = int32_t(id[r]);
    const int32_t slot = (e >= 0 && e < E) ? slot_table[e] : -1;
    slot_of[r] = slot;
    const float sv = sc[r];
    s_all += sv;
    if (slot >= 0) {
      s_kept += sv;
      if (!have_kept) {
        first_kept = slot;
        have_kept = true;
      }
    }
  }
  const float renorm = s_kept > 0.0f ? s_all / s_kept : 0.0f;

  // Pass 2: slots + mix; collect misses into registers.
  int32_t m_id[KQ_ROUTE_SHED_MAX_R];
  float m_sc[KQ_ROUTE_SHED_MAX_R];
  int n_miss = 0;
  for (int r = 0; r < R; r++) {
    if (slot_of[r] >= 0) {
      sl[r] = uint32_t(slot_of[r]);
      mo[r] = sc[r] * renorm;
    } else {
      sl[r] = uint32_t(first_kept);
      mo[r] = 0.0f;
      m_id[n_miss] = int32_t(id[r]);
      m_sc[n_miss] = sc[r];
      n_miss++;
    }
  }
  // Misses front-packed in descending score order (prestage priority);
  // stable insertion sort, matching the CPU eval.
  for (int i = 1; i < n_miss; i++) {
    const int32_t vi = m_id[i];
    const float vs = m_sc[i];
    int j = i - 1;
    while (j >= 0 && m_sc[j] < vs) {
      m_id[j + 1] = m_id[j];
      m_sc[j + 1] = m_sc[j];
      j--;
    }
    m_id[j + 1] = vi;
    m_sc[j + 1] = vs;
  }
  for (int r = 0; r < n_miss; r++) {
    mi[r] = m_id[r];
    ms[r] = m_sc[r];
  }
  for (int r = n_miss; r < R; r++) {
    mi[r] = -1;
    ms[r] = 0.0f;
  }
}
