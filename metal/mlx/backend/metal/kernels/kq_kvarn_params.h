// KVarN record geometry shared by the kernels (kq_kvarn.h, kq_sdpa.h) and
// the host (src/kquant_kvarn.cpp, src/kquant_sdpa.cpp). Included by the
// Metal TUs and via relative paths from src/, so it stays include-free.
#pragma once

// Tokens per group and dims per slice.
#define KQ_KVARN_GROUP 128
#define KQ_KVARN_GROUP_SHIFT 7
// Code words per 128-value row per bit (128 / 32).
#define KQ_KVARN_ROW_WORDS_PER_BIT 4
// Code words per slice record per bit (KQ_KVARN_GROUP rows).
#define KQ_KVARN_SLICE_WORDS_PER_BIT 512
// fp16 axes per slice record: scale, zp, other, KQ_KVARN_GROUP each.
#define KQ_KVARN_AXES_PER_SLICE 384
