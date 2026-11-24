// Routed GLU codec pairs with fused gather kernels: the gate stack in the
// first codec and the up stack in the second (KqGluPair). kq_moe_glu_kq.metal
// instantiates every pair's kernels from this list and
// moe_glu_pair_has_kernel answers from it, so one line adds a pair.
// X(gate codec, gate traits, up codec, up traits)
#pragma once

#define KQ_GLU_SPLIT_PAIRS(X)                   \
  X(iq2_xxs, KqIq2_xxsExt, stq1_0, KqStq1_0Ext) \
  X(stq1_0, KqStq1_0Ext, iq2_xxs, KqIq2_xxsExt)
