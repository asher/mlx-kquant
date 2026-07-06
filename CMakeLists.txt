cmake_minimum_required(VERSION 3.27)

project(mlx_kquant LANGUAGES CXX C)

# ----------------------------- Setup -----------------------------
# C++20: MLX 0.31.2's public headers (device.h, stream.h) use defaulted
# comparison operators, a C++20 feature. Apple Clang accepts them as an
# extension under -std=c++17, but GCC (the Linux toolchain) rejects them, so a
# C++17 standard breaks the Metal-free Linux build. 20 builds cleanly on both.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
# Pick the right interpreter. mlx.extension.CMakeBuild invokes cmake WITHOUT
# forwarding the building interpreter, so CMake's FindPython can otherwise latch
# onto an unrelated system/uv Python that lacks mlx & nanobind (then
# `python -m mlx --cmake-dir` returns empty and find_package(MLX) fails).
# Precedence: an explicit -DPython_EXECUTABLE (e.g. via CMAKE_ARGS) wins; else an
# active virtualenv ($VIRTUAL_ENV); else CMake's normal search.
if(NOT Python_EXECUTABLE AND DEFINED ENV{VIRTUAL_ENV})
  set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
endif()
set(Python_FIND_VIRTUALENV FIRST)

find_package(
  Python 3.9
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# NOTE: do NOT splice ${MLX_CXX_FLAGS} into CMAKE_CXX_FLAGS - it is a CMake list
# (-DACCELERATE_NEW_LAPACK;-D_METAL_), and embedding the ';' into the flags
# *string* corrupts every compile command (`/bin/sh: -D_METAL_: command not
# found`). MLXConfig already attaches these as INTERFACE_COMPILE_OPTIONS on the
# imported `mlx` target, so -D_METAL_ propagates correctly to anything that
# links `mlx` (below).

# ----------------------------- gguflib (GGUF parser) -----------------------------
# antirez/gguf-tools single-file GGUF parser, for the C++ kq.load_gguf loader.
# PIN: a ref new enough to define GGUF_TYPE_Q5_0/Q5_1/Q2_K..Q6_K + correct
# K-quant block sizes (older refs only know Q4_0/Q4_1/Q8_0).
include(FetchContent)
FetchContent_Declare(
  gguflib
  GIT_REPOSITORY https://github.com/antirez/gguf-tools/
  GIT_TAG fdfafbed766db0a1e9019b07994cd88f133d1aab
  # The pinned ref only knows GGML types up to BF16 (30); a tensor of any newer
  # codec trips gguf_get_tensor's `>= GGUF_TYPE_COUNT` guard and truncates the
  # tensor list. Teach the parser the post-BF16 type block-geometry (incl. the
  # MXFP4/NVFP4 float codecs MLX has native kernels for). Idempotent; see script.
  PATCH_COMMAND ${CMAKE_COMMAND} -DGGUFLIB_DIR=<SOURCE_DIR>
                -P ${CMAKE_CURRENT_LIST_DIR}/cmake/patch-gguflib.cmake)
FetchContent_MakeAvailable(gguflib)
add_library(gguflib STATIC ${gguflib_SOURCE_DIR}/fp16.c
                           ${gguflib_SOURCE_DIR}/gguflib.c)
# Hidden visibility so the gguf_* symbols stay LOCAL to whichever image links
# them whole-archive (below) and never interpose / collide at runtime with the
# identically-named gguf_* symbols that libmlx also exports.
set_target_properties(gguflib PROPERTIES POSITION_INDEPENDENT_CODE ON
                                         C_VISIBILITY_PRESET hidden)
# gguflib uses assert() to reject malformed tensor headers (e.g. ndim > 8).
# -DNDEBUG (release builds) would compile those out, leaving out-of-bounds
# reads/writes unguarded when loading untrusted GGUF files. Force NDEBUG off.
target_compile_options(gguflib PRIVATE -UNDEBUG)

# Whole-archive flag for the patched gguflib (used on both linking images below):
# pull all its (hidden) gguf_* objects into the image so they win over libmlx's
# exported copy. Apple ld spells this -force_load; GNU ld / lld use the
# --whole-archive / --no-whole-archive bracket.
if(APPLE)
  set(KQ_WHOLE_ARCHIVE_GGUFLIB "-Wl,-force_load,$<TARGET_FILE:gguflib>")
else()
  set(KQ_WHOLE_ARCHIVE_GGUFLIB
      "-Wl,--whole-archive,$<TARGET_FILE:gguflib>,--no-whole-archive")
endif()

# ----------------------------- C++ library -----------------------------
add_library(mlx_kquant_ext)

target_sources(
  mlx_kquant_ext
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/kquant_codec.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_cpu_decode.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_cpu_encode.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_iq_encode.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_ops.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_dequantize.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_matmul.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_sdpa.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_moe_glu.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_norm_fused.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_gather.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_encode.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_event.cpp
         ${CMAKE_CURRENT_LIST_DIR}/src/kquant_gguf.cpp)

# The IQ CPU encoders must match llama-quantize bit-for-bit on the deterministic
# codecs, so this TU is compiled with FMA contraction off (the float reductions
# in the scale search are contraction-sensitive). Directory-scoped, so it applies
# both to the mlx_kquant_ext library and to the nanobind module that recompiles
# this PUBLIC source.
set_source_files_properties(
  ${CMAKE_CURRENT_LIST_DIR}/src/kquant_iq_encode.cpp
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

target_include_directories(mlx_kquant_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src)
# gguflib include/link are PUBLIC: target_sources above lists the .cpp as PUBLIC,
# so it is also compiled into the `_ext` nanobind module (consumer), which then
# needs gguflib.h on its include path and the gguf_* symbols at link time.
target_include_directories(mlx_kquant_ext PUBLIC ${gguflib_SOURCE_DIR})
target_link_libraries(mlx_kquant_ext PUBLIC mlx)
# Apple: route the CPU matmul's large-M GEMM through Accelerate (engages the
# AMX/SME matrix units). PUBLIC so the nanobind module, which re-compiles the
# PUBLIC sources above, sees the same define and link. Other platforms keep
# the portable threaded-scalar fallback.
if(APPLE)
  target_compile_definitions(mlx_kquant_ext PUBLIC KQ_USE_ACCELERATE)
  target_link_libraries(mlx_kquant_ext PUBLIC "-framework Accelerate")
endif()
# arm64: NEON-dotprod int8 GEMV kernels for the fused small-M CPU matmul.
# The TU is compiled only on arm64/aarch64 hosts; execution is further gated
# at runtime (KQ_CPU_NEON=0 kill switch; Linux aarch64 hwcap dotprod check).
# Every other target keeps the portable scalar path (the header stubs
# kq_neon_kernel to nullptr when KQ_CPU_NEON_TU is undefined).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  target_sources(mlx_kquant_ext
                 PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/kquant_cpu_neon.cpp)
  target_compile_definitions(mlx_kquant_ext PUBLIC KQ_CPU_NEON_TU)
  if(NOT APPLE)
    # Linux toolchains may default to an armv8.0 baseline without dotprod;
    # enable it for this one TU - dotprod instructions only execute after the
    # runtime hwcap check passes.
    set_source_files_properties(
      ${CMAKE_CURRENT_LIST_DIR}/src/kquant_cpu_neon.cpp
      PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
  endif()
endif()
# libmlx bundles its OWN (older) gguflib and exports its gguf_* symbols. A plain
# link can bind our gguf_get_tensor()/gguf_open() calls to MLX's copy, whose
# GGUF_TYPE_COUNT predates the MXFP4/NVFP4 float codecs - so a tensor of type
# >= 31 trips its `>= GGUF_TYPE_COUNT` guard and the tensor list silently
# truncates at the first such tensor. Linking our patched gguflib whole-archive
# pulls its objects into this image as hidden-visibility definitions, so every
# gguf_* call resolves in-image to the patched parser instead of to libmlx.
target_link_libraries(mlx_kquant_ext PUBLIC gguflib)
target_link_options(mlx_kquant_ext PRIVATE ${KQ_WHOLE_ARCHIVE_GGUFLIB})

# ----------------------------- Metal library -----------------------------
# The kq_* kernels compiled into a single mlx_kquant.metallib.
# The repo's metal/ dir is searched BEFORE the stock-wheel include tree so the
# repo's kq_*.h + quantized_utils.h (which adds load_vector) win;
# steel/gemm/*, utils.h, etc. fall through to the stock wheel headers.
if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
    mlx_kquant_metallib
    TITLE
    mlx_kquant
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/metal/kq_quantized.metal
    ${CMAKE_CURRENT_LIST_DIR}/metal/kq_quantized_nax.metal
    ${CMAKE_CURRENT_LIST_DIR}/metal/kq_quantized_encode.metal
    ${CMAKE_CURRENT_LIST_DIR}/metal/kq_sdpa.metal
    ${CMAKE_CURRENT_LIST_DIR}/metal/kq_moe_glu.metal
    ${CMAKE_CURRENT_LIST_DIR}/metal/kq_moe_glu_kq.metal
    ${CMAKE_CURRENT_LIST_DIR}/metal/kq_norm_fused.metal
    INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}/metal
    ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mlx_kquant_ext mlx_kquant_metallib)
endif()

# ----------------------------- Python bindings -----------------------------
nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE mlx_kquant_ext)
# kquant_gguf.cpp is a PUBLIC source (above), so it is also compiled into this
# module and the LTO link keeps its own gguf_* references - force_load the
# patched gguflib here too so they don't fall back to libmlx's exported copy
# (see the mlx_kquant_ext force_load note for the full rationale).
target_link_libraries(_ext PRIVATE gguflib)
target_link_options(_ext PRIVATE ${KQ_WHOLE_ARCHIVE_GGUFLIB})

if(BUILD_SHARED_LIBS)
  if(APPLE)
    # @loader_path finds the co-located libmlx_kquant_ext.dylib (same package
    # dir); the second rpath resolves @rpath/libmlx.dylib against the *user's*
    # installed mlx wheel at runtime. _ext.so lives in site-packages/mlx_kquant/,
    # libmlx in site-packages/mlx/lib/, so ../mlx/lib reaches it. (The absolute
    # build-tree rpath MLXConfig adds works only on the build machine; this is
    # what makes a redistributed wheel find the ABI-pinned mlx==0.31.2. delocate
    # must therefore *exclude* libmlx from vendoring - see [tool.cibuildwheel].)
    target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
    target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path/../mlx/lib)
  else()
    # ELF equivalent of @loader_path: $ORIGIN is the dir of _ext.so. The
    # co-located libmlx_kquant_ext.so sits beside it; ../mlx/lib reaches the
    # user's installed mlx wheel's libmlx.so (auditwheel must *exclude* libmlx.so
    # from vendoring, mirroring the macOS delocate exclude).
    target_link_options(_ext PRIVATE "-Wl,-rpath,\$ORIGIN")
    target_link_options(_ext PRIVATE "-Wl,-rpath,\$ORIGIN/../mlx/lib")
  endif()
endif()
