# Teach the FetchContent'd antirez/gguf-tools parser about the GGML tensor
# types added after its pinned ref (which only knows types up to BF16=30).
#
# Without this, `gguf_get_tensor` hits its `if (*type >= GGUF_TYPE_COUNT) return 0;`
# guard on the first tensor of a newer codec and TRUNCATES the tensor list — a
# GGUF whose experts are MXFP4 (type 39) would load with every tensor at/after
# the first expert silently missing. The tensor data offset is read straight
# from the file, so the only thing the parser needs is each type's block
# geometry (items-per-block / bytes-per-block) and a COUNT past the new types.
#
# Block geometry (from ggml-common.h): MXFP4 = 32 vals / 17 B (1 E8M0 byte + 16
# nibble bytes); NVFP4 = 64 vals / 36 B (4 UE4M3 sub-scales + 32 nibble bytes);
# Q1_0 = 128 vals / 18 B; TQ1_0/TQ2_0 = 256 vals / 54,66 B. Types 31-33 and
# 36-38 are removed/unused ggml slots — placeholder entries keep the table
# positional (indexed by type number) without claiming support.
#
# Run via FetchContent PATCH_COMMAND with -DGGUFLIB_DIR=<SOURCE_DIR>. Idempotent:
# guarded on the marker symbol so a re-configure is a no-op.

set(_h "${GGUFLIB_DIR}/gguflib.h")
set(_c "${GGUFLIB_DIR}/gguflib.c")

file(READ "${_h}" _H)
if(NOT _H MATCHES "GGUF_TYPE_MXFP4")
  string(REPLACE
    "    GGUF_TYPE_BF16 = 30,\n    GGUF_TYPE_COUNT,"
    "    GGUF_TYPE_BF16 = 30,\n    GGUF_TYPE_TQ1_0 = 34,\n    GGUF_TYPE_TQ2_0 = 35,\n    GGUF_TYPE_MXFP4 = 39,\n    GGUF_TYPE_NVFP4 = 40,\n    GGUF_TYPE_Q1_0 = 41,\n    GGUF_TYPE_COUNT,"
    _H "${_H}")
  file(WRITE "${_h}" "${_H}")
  message(STATUS "gguflib: patched gguflib.h (MXFP4/NVFP4/Q1_0/TQ types; COUNT=42)")
endif()

file(READ "${_c}" _C)
if(NOT _C MATCHES "\"mxfp4\"")
  string(REPLACE
    "    {\"bf16\", 1, 2},\n}"
    "    {\"bf16\", 1, 2},\n    {\"removed_31\", 0, 0},\n    {\"removed_32\", 0, 0},\n    {\"removed_33\", 0, 0},\n    {\"tq1_0\", 256, 54},\n    {\"tq2_0\", 256, 66},\n    {\"removed_36\", 0, 0},\n    {\"removed_37\", 0, 0},\n    {\"removed_38\", 0, 0},\n    {\"mxfp4\", 32, 17},\n    {\"nvfp4\", 64, 36},\n    {\"q1_0\", 128, 18},\n}"
    _C "${_C}")
  file(WRITE "${_c}" "${_C}")
  message(STATUS "gguflib: patched gguflib.c (fp-codec + Q1_0/TQ type-features)")
endif()
