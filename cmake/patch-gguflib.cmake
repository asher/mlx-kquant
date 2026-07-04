# Teach the FetchContent'd antirez/gguf-tools parser about the GGML tensor
# types added after its pinned ref (which only knows types up to BF16=30).
#
# Without this, `gguf_get_tensor` hits its `if (*type >= GGUF_TYPE_COUNT) return 0;`
# guard on the first tensor of a newer codec and TRUNCATES the tensor list - a
# GGUF whose experts are MXFP4 (type 39) would load with every tensor at/after
# the first expert silently missing. The tensor data offset is read straight
# from the file, so the only thing the parser needs is each type's block
# geometry (items-per-block / bytes-per-block) and a COUNT past the new types.
#
# Block geometry (from ggml-common.h): MXFP4 = 32 vals / 17 B (1 E8M0 byte + 16
# nibble bytes); NVFP4 = 64 vals / 36 B (4 UE4M3 sub-scales + 32 nibble bytes);
# Q1_0 = 128 vals / 18 B; TQ1_0/TQ2_0 = 256 vals / 54,66 B. Types 31-33 and
# 36-38 are removed/unused ggml slots - placeholder entries keep the table
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

# Correct two upstream IQ block-geometry errors (antirez's table predates the
# final ggml IQ layout): IQ1_S is 50 B / 256 (it had 110), and IQ4_NL is the
# ONLY flat IQ codec at 18 B / 32 (it had 256/50 = IQ1_S's geometry). The IQ4_NL
# error made gguflib mis-size every IQ4_NL tensor, so kq.load_gguf aborted on a
# bsize mismatch vs the codec registry's correct (32,18).
file(READ "${_c}" _C)
if(NOT _C MATCHES "iq4_nl\", 32, 18")
  string(REPLACE "{\"iq1_s\", 256, 110}" "{\"iq1_s\", 256, 50}" _C "${_C}")
  string(REPLACE "{\"iq4_nl\", 256, 50}" "{\"iq4_nl\", 32, 18}" _C "${_C}")
  file(WRITE "${_c}" "${_C}")
  message(STATUS "gguflib: corrected IQ1_S/IQ4_NL block geometry")
endif()

# Add a read-only open path. The parser is upstream a read/write GGUF *editor*:
# gguf_open() uses O_RDWR + mmap(PROT_READ|PROT_WRITE, MAP_SHARED), so a no-copy
# array viewing a tensor aliases a writable, disk-backed page -- a fused in-place
# op or a stray buffer-donation then mutates the SOURCE FILE. Consumers that only
# read existing files must use gguf_open_ro(): O_RDONLY + PROT_READ + MAP_SHARED,
# so tensor pages stay clean, evictable file-cache pages and a stray write
# faults loudly instead of reaching the file. Metal wraps read-only shared pages
# fine via newBufferWithBytesNoCopy (llama.cpp maps weights the same way). The
# historical mode, PROT_READ|PROT_WRITE + MAP_PRIVATE, made the GPU write-fault
# every touched page: the first forward lazily copied the whole file into
# anonymous memory (compress-only, never droppable), doubling the model's
# standing cost; under memory pressure that turns a graceful slowdown into a
# compress/refault storm. KQ_GGUF_MMAP=private_rw restores that mode as a kill
# switch; KQ_GGUF_MMAP=private_ro is a diagnostic quadrant. gguf_open() (and
# thus the gguf_create() write path, which calls it) is left untouched.
# gguf_remap() branches on the new ctx->ro.
file(READ "${_h}" _H)
if(NOT _H MATCHES "gguf_open_ro")
  string(REPLACE
    "    uint64_t alignment;             // File data alignment. Default: 32 bytes.\n} gguf_ctx;"
    "    uint64_t alignment;             // File data alignment. Default: 32 bytes.\n    int ro;                         // Read-only mapping (see KQ_GGUF_MMAP).\n} gguf_ctx;"
    _H "${_H}")
  string(REPLACE
    "gguf_ctx *gguf_open(const char *filename);\ngguf_ctx *gguf_create(const char *filename, int flags);"
    "gguf_ctx *gguf_open(const char *filename);\ngguf_ctx *gguf_open_ro(const char *filename);\ngguf_ctx *gguf_create(const char *filename, int flags);"
    _H "${_H}")
  file(WRITE "${_h}" "${_H}")
  message(STATUS "gguflib: patched gguflib.h (read-only open: gguf_open_ro)")
endif()

file(READ "${_c}" _C)
if(NOT _C MATCHES "gguf_open_ro")
  string(REPLACE
    "    void *mapped = mmap(0,sb.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,ctx->fd,0);"
    "    /* Read-only contexts default to PROT_READ + MAP_SHARED: weights stay\n     * clean, evictable file-cache pages, and Metal wraps read-only shared\n     * pages fine (llama.cpp maps weights the same way). The historical mode,\n     * PROT_READ|PROT_WRITE + MAP_PRIVATE, made the GPU write-fault every\n     * touched page, lazily copying the whole file into anonymous memory that\n     * can only be compressed, never dropped. KQ_GGUF_MMAP=private_rw restores\n     * it; KQ_GGUF_MMAP=private_ro is a diagnostic. The writer path keeps\n     * PROT_READ|PROT_WRITE + MAP_SHARED to build new files. */\n    int _prot = PROT_READ|PROT_WRITE;\n    int _flags = MAP_SHARED;\n    if (ctx->ro) {\n        const char *_m = getenv(\"KQ_GGUF_MMAP\");\n        if (_m && !strcmp(_m,\"private_rw\")) { _flags = MAP_PRIVATE; }\n        else if (_m && !strcmp(_m,\"private_ro\")) { _prot = PROT_READ; _flags = MAP_PRIVATE; }\n        else { _prot = PROT_READ; }\n    }\n    void *mapped = mmap(0,sb.st_size,_prot,_flags,ctx->fd,0);"
    _C "${_C}")
  string(REPLACE
    "void gguf_rewind(gguf_ctx *ctx) {"
    "/* Like gguf_open() but opens an existing file read-only (O_RDONLY; mapping\n * mode per KQ_GGUF_MMAP, default PROT_READ + MAP_SHARED) so a no-copy tensor\n * view can never be written back to disk. */\ngguf_ctx *gguf_open_ro(const char *filename) {\n    int fd = open(filename,O_RDONLY);\n    if (fd == -1) return NULL;\n    gguf_ctx *ctx = calloc(1, sizeof(*ctx));\n    if (!ctx) { close(fd); return NULL; }\n    ctx->fd = fd;\n    ctx->ro = 1;\n    ctx->alignment = 32;\n    ctx->data_off = 0;\n    if (gguf_remap(ctx) == 0) {\n        gguf_close(ctx);\n        return NULL;\n    }\n    gguf_rewind(ctx);\n    return ctx;\n}\n\nvoid gguf_rewind(gguf_ctx *ctx) {"
    _C "${_C}")
  file(WRITE "${_c}" "${_C}")
  message(STATUS "gguflib: patched gguflib.c (read-only open: gguf_open_ro)")
endif()
