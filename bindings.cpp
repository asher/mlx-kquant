#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_cpu_neon.h"
#include "kquant_gguf.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Convert a decoded GGUF metadata value to a Python object: scalars to
// int/float/bool/str, arrays to lists, monostate to None.
nb::object meta_to_py(const mlx_kquant::GgufMetaValue& v) {
  return std::visit(
      [](auto&& x) -> nb::object {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return nb::none();
        } else {
          return nb::cast(x);
        }
      },
      v);
}

} // namespace

NB_MODULE(_ext, m) {
  m.doc() =
      "mlx-kquant: standalone GGUF K-quant ops for MLX (custom Metal kernels).";

  // --- toolchain self-checks ---
  m.def(
      "codecs",
      &mlx_kquant::codec_names,
      "Return the list of supported K-quant codec names.");

  m.def(
      "metallib_dir",
      &mlx_kquant::metallib_dir,
      "Directory holding the bundled mlx_kquant.metallib.");

  m.def(
      "metallib_loads",
      &mlx_kquant::metallib_loads,
      "Load the bundled metallib via the Metal device (toolchain self-check).");

  m.def(
      "cpu_neon_available",
      &mlx_kquant::kq_cpu_neon_available,
      "True when the arm64 NEON int8 CPU GEMV kernels can run here (arm64 "
      "build with the dotprod extension, not disabled via KQ_CPU_NEON=0).");

  m.def(
      "nax_available",
      &mlx_kquant::nax_available,
      "True when the GPU supports the NAX (tensor-core) matmul kernels.");

  m.def(
      "nax_gather_enabled",
      &mlx_kquant::nax_gather_enabled,
      "kquant_type"_a,
      "True when gather_qmm's sorted-rhs NAX GEMM leaf can serve this codec "
      "here: NAX hardware present, the codec ships NAX kernels, and "
      "KQ_DISABLE_NAX is unset (read live). Sorted-prefill callers defer to "
      "gather_qmm when this holds.");

  m.def(
      "codec_has_moe_glu",
      &mlx_kquant::codec_has_moe_glu,
      "kquant_type"_a,
      "True when this codec has the fused MoE GLU/gather kernel family "
      "(kq.moe_glu_gather_kq and friends).");

  m.def(
      "codec_has_matmul",
      [](const std::string& kquant_type) {
        const auto* codec = mlx_kquant::codec_by_name(kquant_type);
        return codec != nullptr && codec->has_matmul_kernel;
      },
      "kquant_type"_a,
      "True when this codec ships Metal matmul kernels (qmv/qmm/gather). "
      "CPU-only wire codecs return False; their matmuls must stay on the "
      "CPU stream.");

  // --- ops ---
  m.def(
      "dequantize",
      &mlx_kquant::dequantize,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Dequantize GGUF K-quant wire bytes to a float array.

        Args:
            w (array): uint8 wire bytes; last dim a multiple of the codec's
                bytes_per_block.
            scales (array): vestigial placeholder (K-quant scales live inside
                ``w``); ignored by the kernel.
            kquant_type (str): codec name, e.g. ``"q4_k"``, ``"q8_0"``.
            dtype (Dtype, optional): output float dtype. Default ``float16``.

        Returns:
            array: the dequantized weights.
      )");

  m.def(
      "quantized_matmul",
      &mlx_kquant::quantized_matmul,
      "x"_a,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "transpose"_a = true,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Quantized matmul: ``x @ dequant(w)`` for GGUF K-quant weights.

        Args:
            x (array): float activations.
            w (array): uint8 K-quant wire bytes (laid out [N, K] when
                transpose=True).
            scales (array): vestigial placeholder; ignored by the kernel.
            kquant_type (str): codec name, e.g. ``"q4_k"``.
            transpose (bool): whether ``w`` is transposed ([N, K]). Default True.

        Returns:
            array: the matmul result (x.dtype, float32 promoted to bfloat16).
      )");

  m.def(
      "quantized_matmul_qmv_bias",
      &mlx_kquant::quantized_matmul_qmv_bias,
      "x"_a,
      "w"_a,
      "scales"_a,
      "bias"_a,
      "kquant_type"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Bias-fused quantized matmul: ``x @ dequant(w) + bias`` for GGUF
        K-quant weights, fusing the add into the matmul kernel dispatch.

        Decode-only: ``x`` must carry exactly one row (``x.shape[-2] == 1``
        after flattening leading batch dims) -- raises otherwise. Only
        ``kquant_type="q8_0"`` is wired so far. ``transpose`` is always True
        (the only regime this is used for). For any other shape or codec, use
        ``quantized_matmul`` followed by a separate ``+ bias``.

        Args:
            x (array): float activations, exactly one row.
            w (array): uint8 K-quant wire bytes, laid out [N, K].
            scales (array): vestigial placeholder; ignored by the kernel.
            bias (array): 1D, length N (the output dim).
            kquant_type (str): codec name; only ``"q8_0"`` is wired so far.

        Returns:
            array: the matmul-plus-bias result (x.dtype, float32 promoted to
            bfloat16).
      )");

  m.def(
      "sdpa_vector",
      &mlx_kquant::sdpa_vector,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "causal"_a = true,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Vector scaled-dot-product attention for large head dims (256, 512) that
        stock MLX's fused vector allowlist excludes.

        Args:
            q (array): queries [B, n_q_heads, qL, D], float16/bfloat16.
            k (array): keys [B, n_kv_heads, kL, D]; head/seq strided is fine
                (read in place), D dim must be contiguous.
            v (array): values [B, n_kv_heads, kL, D].
            scale (float): query scale (typically 1/sqrt(D)).
            causal (bool): apply an offset causal mask. Default True.

        Returns:
            array: attention output [B, n_q_heads, qL, D].
      )");

  m.def(
      "sdpa_decode_gqa",
      [](mx::array q,
         mx::array k,
         mx::array v,
         float scale,
         const std::optional<mx::array>& sinks,
         int splits,
         int tile_c,
         const std::optional<mx::array>& starts,
         const std::optional<mx::array>& k_scales,
         const std::optional<mx::array>& k_biases,
         const std::optional<mx::array>& v_scales,
         const std::optional<mx::array>& v_biases,
         bool return_lse,
         mx::StreamOrDevice s) -> nb::object {
        if (return_lse) {
          auto outs = mlx_kquant::sdpa_decode_gqa_lse(
              std::move(q),
              std::move(k),
              std::move(v),
              scale,
              sinks,
              splits,
              tile_c,
              starts,
              k_scales,
              k_biases,
              v_scales,
              v_biases,
              s);
          return nb::make_tuple(outs[0], outs[1]);
        }
        return nb::cast(mlx_kquant::sdpa_decode_gqa(
            std::move(q),
            std::move(k),
            std::move(v),
            scale,
            sinks,
            splits,
            tile_c,
            starts,
            k_scales,
            k_biases,
            v_scales,
            v_biases,
            s));
      },
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "sinks"_a = nb::none(),
      "splits"_a = 0,
      "tile_c"_a = 0,
      "starts"_a = nb::none(),
      "k_scales"_a = nb::none(),
      "k_biases"_a = nb::none(),
      "v_scales"_a = nb::none(),
      "v_biases"_a = nb::none(),
      nb::kw_only(),
      "return_lse"_a = false,
      "stream"_a = nb::none(),
      R"(
        Decode/verify GQA attention tuned for long KV caches: the key axis
        is split into a fixed number of coarse contiguous chunks and each
        chunk is streamed through threadgroup-staged K/V tiles shared by the
        whole GQA group, so device memory reads the KV once per kv-head. At
        qL 2..4 (speculative-verify width) every query also shares the staged
        tiles, causally clamped to its own trailing position. With `starts`,
        batch row b attends keys [starts[b], kL) -- a left-padded batched KV
        cache -- and fully padded-out key chunks are skipped, not staged.
        With k_scales/k_biases/v_scales/v_biases (all four), k and v are
        mlx affine-quantized wire (uint32, bits 8, group 64) and dequant is
        fused into the tile stage.

        Args:
            q (array): queries [B, n_q_heads, qL, D], float16/bfloat16;
                qL in 1..4, D in {64, 128, 256, 512}.
            k (array): keys [B, n_kv_heads, kL, D]; head/seq strided is fine
                (read in place), the head_dim must be contiguous.
            v (array): values [B, n_kv_heads, kL, D].
            scale (float): query scale (typically 1/sqrt(D)).
            sinks (array, optional): per-q-head attention sinks, shape
                [n_q_heads] -- an extra softmax logit with no value row.
            splits (int): key-axis split count; 0 picks the default.
            tile_c (int): staged tile height, 8/16/32; 0 (default) picks by
                head_dim (32 up to D=128, 16 at D=256, 8 at D=512).
            starts (array, optional): per-batch-row key start offsets,
                int32 [B], each in [0, kL - qL]; row b attends [starts[b],
                kL). Out-of-range values read as an empty row (zero output).

        Returns:
            array: attention output [B, n_q_heads, qL, D]. With
            ``return_lse=True``, a tuple ``(out, lse)`` where lse
            [B, n_q_heads, qL] float32 is the natural-log softmax
            normalizer per query row (the merge weight for combining
            attention over disjoint key regions).
      )");

  m.def(
      "sdpa_prefill_block_sparse",
      &mlx_kquant::sdpa_prefill_block_sparse,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "pages"_a,
      "pmask"_a,
      "counts"_a,
      "offset"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Block-sparse FA prefill over QSA-selected 4-row key pages. Queries
        fold into windows of 4 (with the GQA group); window w walks ONLY
        pages[w, :counts[w]] through a simdgroup-matrix FA tile. A key
        counts for a query when its page's pmask bit for that query is set
        (the builder sets bits only for blocks complete at the query), or
        when it lies in the query's own incomplete tail block (causal).
        The page list must include each window's tail-span blocks.

        Args:
            q (array): queries [1, n_q_heads, L, D]; n_q_heads must be
                12 * n_kv_heads, D must be 256, L a multiple of 4.
            k (array): keys [1, n_kv_heads, S, D] (full cache view).
            v (array): values [1, n_kv_heads, S, D].
            scale (float): query scale (typically 1/sqrt(D)).
            pages (array): int32 [L / 4, max_pages] page indices per
                window, padded with -1 past counts[w].
            pmask (array): uint16 [L / 4, max_pages] membership bits
                (bit i = query i of the window selected the page).
            counts (array): int32 [L / 4] live page count per window.
            offset (int): global position of query row 0 (S - L for a
                standard prefill chunk).

        Returns:
            array: attention output [1, n_q_heads, L, D].
      )");

  m.def(
      "sdpa_fa_verify",
      [](mx::array q,
         mx::array k,
         mx::array v,
         float scale,
         int q_len,
         int splits,
         bool return_lse,
         mx::StreamOrDevice s) -> nb::object {
        if (return_lse) {
          auto outs = mlx_kquant::sdpa_fa_verify_lse(
              std::move(q),
              std::move(k),
              std::move(v),
              scale,
              q_len,
              splits,
              s);
          return nb::make_tuple(outs[0], outs[1]);
        }
        return nb::cast(mlx_kquant::sdpa_fa_verify(
            std::move(q), std::move(k), std::move(v), scale, q_len, splits, s));
      },
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "q_len"_a,
      "splits"_a = 0,
      nb::kw_only(),
      "return_lse"_a = false,
      "stream"_a = nb::none(),
      R"(
        Speculative-verify attention on the GPU matrix units for a GQA-folded
        query tile. Fold the GQA group into the query rows first --
        q [1, Hq, q_len, D] reshaped to [1, Hkv, G*q_len, D] with kv-major
        heads -- and pass the original q_len: folded row r is causally
        clamped to key <= kL - q_len + (r % q_len). The query tile (32 rows,
        or 64 for oversized folds such as gqa16 x q_len 4) streams each
        contiguous KV split once, computing S = Q @ K^T and O += P @ V on
        simdgroup_matrix with float32 accumulators and a per-row online
        softmax; per-split partials are merged by the same reduction pass as
        ``sdpa_decode_gqa``.

        Args:
            q (array): folded queries [1, n_kv_heads, G*q_len, D],
                float16/bfloat16; D = 64, 128, 256 or 512; G*q_len <= 64
                except <= 32 at D=512.
            k (array): keys [1, n_kv_heads, kL, D]; head/seq strided is fine
                (read in place), the head_dim must be contiguous.
            v (array): values [1, n_kv_heads, kL, D].
            scale (float): query scale (typically 1/sqrt(D)).
            q_len (int): pre-fold query length (1..8); sets each folded
                row's causal clamp. q_len 1 is plain GQA decode on the
                matrix units (every folded row attends the full KV).
            splits (int): key-axis split count; 0 picks the default.

        Returns:
            array: attention output [1, n_kv_heads, G*q_len, D]. With
            ``return_lse=True``, a tuple ``(out, lse)`` where lse
            [1, n_kv_heads, G*q_len] float32 is the natural-log softmax
            normalizer per folded row (cascade merge weight).
      )");

  m.def(
      "sdpa_decode_gqa_paged",
      &mlx_kquant::sdpa_decode_gqa_paged,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "pages"_a,
      "splits"_a = 0,
      "tile_c"_a = 0,
      "starts"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Sparse page-gather decode attention: attend ONLY the key/value
        pages listed per (batch, kv-head), walking the selected pages
        through the decode kernel instead of the full cache. The page
        unit is the head dim's staged tile height: 32 rows at head_dim
        64/128, 16 at 256, 8 at 512; tile_c=4 selects a 4-row page at
        head_dim 256 (block-sparse attention with a 4-token selection
        unit). Optional starts (int32 [B])
        restricts row b to keys [starts[b], N) for left-padded batches.

        Args:
            q (array): queries [B, n_q_heads, 1, D], float16/bfloat16.
            k (array): keys [B, n_kv_heads, S, D] (full cache view).
            v (array): values [B, n_kv_heads, S, D].
            scale (float): query scale (typically 1/sqrt(D)).
            pages (array): int32 [B, n_kv_heads, n_pages] page indices in
                [0, ceil(S / page_size)); no duplicates. The final partial
                page is tail-clamped to S automatically.
            splits (int): key-axis split count; 0 buckets by the SELECTED
                key count.
            tile_c (int): page size in rows; 0 picks the head dim's
                default. 4 is instantiated at head_dim 256 only.

        Returns:
            array: attention output [B, n_q_heads, 1, D].
      )");

  m.def(
      "sdpa_decode_gqa_cascade",
      [](mx::array q,
         mx::array k_shared,
         mx::array v_shared,
         mx::array k_priv,
         mx::array v_priv,
         float scale,
         const std::optional<mx::array>& starts,
         int splits_shared,
         int splits_priv,
         int tile_c,
         bool return_lse,
         const std::optional<mx::array>& k_shared_scales,
         const std::optional<mx::array>& k_shared_biases,
         const std::optional<mx::array>& v_shared_scales,
         const std::optional<mx::array>& v_shared_biases,
         const std::optional<mx::array>& k_priv_scales,
         const std::optional<mx::array>& k_priv_biases,
         const std::optional<mx::array>& v_priv_scales,
         const std::optional<mx::array>& v_priv_biases,
         mx::StreamOrDevice s) -> nb::object {
        auto outs = mlx_kquant::sdpa_decode_gqa_cascade(
            std::move(q),
            std::move(k_shared),
            std::move(v_shared),
            std::move(k_priv),
            std::move(v_priv),
            scale,
            starts,
            splits_shared,
            splits_priv,
            tile_c,
            return_lse,
            k_shared_scales,
            k_shared_biases,
            v_shared_scales,
            v_shared_biases,
            k_priv_scales,
            k_priv_biases,
            v_priv_scales,
            v_priv_biases,
            s);
        if (return_lse) {
          return nb::make_tuple(outs[0], outs[1]);
        }
        return nb::cast(outs[0]);
      },
      "q"_a,
      "k_shared"_a,
      "v_shared"_a,
      "k_priv"_a,
      "v_priv"_a,
      "scale"_a,
      "starts"_a = nb::none(),
      "splits_shared"_a = 0,
      "splits_priv"_a = 0,
      "tile_c"_a = 0,
      nb::kw_only(),
      "return_lse"_a = false,
      "k_shared_scales"_a = nb::none(),
      "k_shared_biases"_a = nb::none(),
      "v_shared_scales"_a = nb::none(),
      "v_shared_biases"_a = nb::none(),
      "k_priv_scales"_a = nb::none(),
      "k_priv_biases"_a = nb::none(),
      "v_priv_scales"_a = nb::none(),
      "v_priv_biases"_a = nb::none(),
      "stream"_a = nb::none(),
      R"(
        Fused shared-prefix (cascade) decode attention: every batch row
        attends one COMMON prefix, stored once, plus its own private
        suffix. The shared region is walked ONCE for all B*gqa query rows
        on the matrix-unit row tile; the private region runs per row (with
        optional left-pad ``starts``); both partial sets fold through a
        single merge pass. Equivalent to ``sdpa_decode_gqa`` over the
        concatenated KV, reading the prefix once instead of B times.

        Args:
            q (array): queries [B, n_q_heads, qL, D], float16/bfloat16;
                qL in [1, 8] (verify width: end-aligned causal on the
                private suffix, full shared visibility); D in
                {64, 128, 256, 512}; gqa <= 16; B*gqa*qL <= 64 (<= 32 at
                D=512); gqa*ceil(qL/2) <= 32 at qL > 1.
            k_shared (array): shared prefix keys [1, n_kv_heads, P, D].
            v_shared (array): shared prefix values [1, n_kv_heads, P, D].
            k_priv (array): private suffix keys [B, n_kv_heads, Sp, D],
                Sp >= 1.
            v_priv (array): private suffix values [B, n_kv_heads, Sp, D].
            scale (float): query scale (typically 1/sqrt(D)).
            starts (array, optional): int32 [B] per-row private-region key
                start offsets (left-padded private suffixes).
            splits_shared (int): shared-region split count; 0 = default.
            splits_priv (int): private-region split count; 0 = default.
            tile_c (int): private-pass staged tile height; 0 picks by
                head_dim.
            k_shared_scales ... v_priv_biases (array, optional): quantized
                KV (mlx affine wire, bits 8 / group 64). Pass all eight and
                both k/v slabs bind as packed uint32 words ([.., S, D/4])
                with scales/biases [.., S, D/64] in q's dtype; dequant
                happens at tile stage. Not supported at D=512.

        Returns:
            array: attention output [B, n_q_heads, 1, D]. With
            ``return_lse=True``, a tuple ``(out, lse)``.
      )");

  m.def(
      "moe_glu_gather",
      &mlx_kquant::moe_glu_gather,
      "x"_a,
      "gate_w"_a,
      "gate_scales"_a,
      "gate_bias"_a,
      "up_w"_a,
      "up_scales"_a,
      "up_bias"_a,
      "indices"_a,
      "alpha"_a = 1.702f,
      "limit"_a = 7.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused MoE GLU gather on the MLX packed mxfp4 layout: gate and up
        expert matvecs (sharing each activation load), expert biases, and the
        clamped-SwiGLU epilogue
        ``(min(g, limit) * sigmoid(alpha * g)) * (clip(u, -limit, limit) + 1)``
        in one dispatch. Decode-shaped: one activation row per token, shared
        across that token's expert slots.

        Args:
            x (array): activations [T, K], float16/bfloat16.
            gate_w (array): packed gate weights uint32 [E, N, K/8].
            gate_scales (array): E8M0 group scales uint8 [E, N, K/32].
            gate_bias (array): gate biases [E, N].
            up_w / up_scales / up_bias: same layout for the up projection.
            indices (array): expert indices [T, R].
            alpha (float): sigmoid slope. Default 1.702.
            limit (float): activation clamp. Default 7.0.

        Returns:
            array: activated hidden states [T, R, N] in x.dtype.
      )");

  m.def(
      "gather_qmv_bias",
      &mlx_kquant::gather_qmv_bias,
      "x"_a,
      "w"_a,
      "scales"_a,
      "bias"_a,
      "indices"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Gathered matvec with the expert bias fused, on the MLX packed mxfp4
        layout (see moe_glu_gather). One activation row per expert slot.

        Args:
            x (array): activations [T, R, K], float16/bfloat16.
            w (array): packed weights uint32 [E, N, K/8].
            scales (array): E8M0 group scales uint8 [E, N, K/32].
            bias (array): biases [E, N].
            indices (array): expert indices [T, R].

        Returns:
            array: output [T, R, N] in x.dtype.
      )");

  m.def(
      "gather_qmv_mix_bias",
      &mlx_kquant::gather_qmv_mix_bias,
      "x"_a,
      "w"_a,
      "scales"_a,
      "bias"_a,
      "indices"_a,
      "scores"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        gather_qmv_bias with the routing mix folded in: each routed slot's
        matvec + expert bias is accumulated in f32 weighted by its score,
        replacing gather_qmv_bias + (y * scores).sum(-2).

        Args:
            x (array): activations [T, S, K], float16/bfloat16.
            w (array): packed weights uint32 [E, N, K/8].
            scales (array): E8M0 group scales uint8 [E, N, K/32].
            bias (array): biases [E, N].
            indices (array): expert indices [T, S].
            scores (array): mix weights [T, S]; cast to float32.

        Returns:
            array: mixed output [T, N] in x.dtype.
      )");

  m.def(
      "dsa_sparse_attention",
      &mlx_kquant::dsa_sparse_attention,
      "q"_a,
      "local_kv"_a,
      "pooled"_a,
      "topk_indices"_a,
      "sinks"_a,
      "scale"_a,
      "q_offset"_a,
      "compress_ratio"_a,
      "local_window"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        DeepSeek-V4-Flash sparse attention: sliding local window + gathered
        indexer-selected pooled rows + per-head attention sinks in one
        dispatch (flash online softmax, f32 accumulation). Ported from omlx
        glm_moe_dsa; qL >= 1, so decode, MTP verify (qL = 2) and prefill all
        run this kernel.

        Args:
            q (array): queries [B, 64, qL, 512], float16/bfloat16.
            local_kv (array): sliding-window KV [B, 1, localL, 512]
                (K == V shared latent), temporal order, localL >= qL.
            pooled (array): compressed pooled rows [B, P, 512].
            topk_indices (array): uint32 [B, 1, qL, N] pooled-row indices;
                slots >= the causal horizon (q_offset + pos + 1) /
                compress_ratio are masked out kernel-side.
            sinks (array): per-head sink logits [64].
            scale (float): attention scale (1/sqrt(512)).
            q_offset (int): absolute position of the chunk start.
            compress_ratio (int): pooled compression ratio.
            local_window (int): sliding-window size (128).

        Returns:
            array: attention output [B, 64, qL, 512] in the input dtype.
      )");

  m.def(
      "dsa_indexer_scores",
      &mlx_kquant::dsa_indexer_scores,
      "queries"_a,
      "keys"_a,
      "weights"_a,
      "causal"_a = true,
      "unused_causal_prefix_topk"_a = 0,
      "skip_causal_future_store"_a = false,
      "causal_q_offset"_a = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        DeepSeek-V4-Flash lightning-indexer relevance scores (steel GEMM):
        out[b, 0, m, n] = sum_h relu(q[b, h, m] . k[b, 0, n]) * w[h, m].
        Ported from omlx glm_moe_dsa. Feed the result to dsa_topk_indices to
        pick the pooled rows dsa_sparse_attention gathers.

        Args:
            queries (array): [B, H, M, 128], H 32 or 64, M % 64 == 0,
                float16/bfloat16. Decode pads the single query row to 64
                and keeps output row 0.
            keys (array): pooled indexer keys [B, 1, N, 128], N % 64 == 0.
            weights (array): per-head query weights, [B, M, H] (lh layout)
                or [B, H, M, 1].
            causal (bool): mask n > causal_q_offset + m with -inf.
            unused_causal_prefix_topk (int): skip writing tiles whose rows
                all fall inside a causal prefix of this many keys (they are
                identity-selected by a causal_valid_prefix top-k).
            skip_causal_future_store (bool): leave fully-masked future tiles
                unwritten instead of storing -inf (pair with a
                causal_valid_prefix top-k that never reads them).
            causal_q_offset (int): absolute position of query row 0; -1
                means N - M.

        Returns:
            array: scores [B, 1, M, N] in the input dtype.
      )");

  m.def(
      "dsa_topk_indices",
      &mlx_kquant::dsa_topk_indices,
      "scores"_a,
      "topk"_a,
      "bucketed"_a = false,
      "causal_valid_prefix"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Per-row top-k arg-select over 16-bit float scores (2-pass radix
        select, one threadgroup per row). Ported from omlx glm_moe_dsa.
        The selected index set matches a full sort; the order within a row
        does not (ties at the threshold are admitted in scan order) --
        dsa_sparse_attention is order-insensitive.

        Args:
            scores (array): [B, 1, L, K], float16/bfloat16, K >= topk.
            topk (int): 512 or 2048.
            bucketed (bool): deterministic bucketed emission (>threshold
                entries before ==threshold entries).
            causal_valid_prefix (bool): clamp each row's scan to its causal
                prefix K - L + (row % L) + 1 and emit the identity prefix
                when it fits inside topk.

        Returns:
            array: uint32 indices [B, 1, L, topk].
      )");

  m.def(
      "dsa_indexer_score_decode",
      &mlx_kquant::dsa_indexer_score_decode,
      "queries"_a,
      "keys"_a,
      "weights"_a,
      "q_offset"_a,
      "ratio"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Decode-width lightning-indexer scores, fused:
        out[b, 0, j, p] = sum_h relu(q[b, h, j] . k[b, p]) * w[b, j, h]
        for qL <= 4 query rows without materializing the [H, P] per-head
        scores. Selection-equivalent to the inline path when any positive
        global scale is folded out. Pooled visibility follows
        PoolingCache.make_mask(qL, q_offset): row p is visible to query j
        iff p < (q_offset + j + 1) // ratio, and every row is visible when
        qL == 1; invisible rows score the dtype's finite min.

        Args:
            queries (array): [B, 64, qL, 128], qL in [1, 4],
                float16/bfloat16.
            keys (array): the pooled indexer key cache [B, P, 128].
            weights (array): per-head query weights [B, qL, 64].
            q_offset (int): absolute position of query row 0's step
                (make_mask's ``offset``).
            ratio (int): pooled compression ratio.

        Returns:
            array: scores [B, 1, qL, P] shaped for dsa_topk_indices.
      )");

  m.def(
      "dsa_indexer_qat",
      &mlx_kquant::dsa_indexer_qat,
      "x"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        DeepSeek-V4-Flash indexer activation QAT round-trip, fused: the
        128-wide Hadamard transform (mlx hadamard_transform's butterfly
        order and 1/sqrt(128) scale, bit-exactly) followed by the
        per-32-block FP4-E2M1 round-trip (scale 2^ceil(log2(amax/6)) with
        an FLT_MIN*6 amax floor, clamp to +-6, tie-to-even rounding).
        One kernel in place of the multi-pass hadamard + quantize chain.

        Args:
            x (array): any shape with a trailing dim of 128,
                float16/bfloat16/float32.

        Returns:
            array: same shape and dtype as ``x``.
      )");

  m.def(
      "dsa_indexer_qat_quant",
      &mlx_kquant::dsa_indexer_qat_quant,
      "x"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Emit variant of dsa_indexer_qat: same fused Hadamard + FP4-E2M1
        quantization, returning the quantized wire form instead of the
        dequantized round-trip. codes * scales (each scale covering its
        32-block) reproduces dsa_indexer_qat(x) bit-exactly, except
        negatives snapped to zero re-dequantize as +0.0 where the
        round-trip stores -0.0 (value-equal; scores unaffected). Feed to
        dsa_indexer_scores_q.

        Args:
            x (array): any shape with a trailing dim of 128,
                float16/bfloat16/float32.

        Returns:
            tuple(array, array): codes int8 (x's shape; E2M1 values
            doubled, in [-12, 12]) and scales float32 (x's shape with the
            trailing 128 replaced by 4; per-32-block power-of-two scale
            pre-folded as scale * 0.5).
      )");

  m.def(
      "dsa_indexer_qat_pack",
      &mlx_kquant::dsa_indexer_qat_pack,
      "x"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Pack variant of dsa_indexer_qat_quant WITHOUT the Hadamard: for
        rows that are already rotated and on the E2M1 grid (e.g. pooled
        indexer keys cached as the fp16 output of dsa_indexer_qat). Same
        wire form as dsa_indexer_qat_quant; on on-grid rows the pack is a
        fixed point (codes * scales == x bit-exactly, with the same
        +0.0-for--0.0 caveat). A block whose max is exactly 3*2^k
        re-derives scale 2^(k-1) where the in-graph quant may have chosen
        2^k (the original scale is not recoverable from on-grid values);
        codes double and every downstream dequant / dsa_indexer_scores_q
        result is bit-identical either way.

        Args:
            x (array): any shape with a trailing dim of 128, already
                Hadamard-rotated on-grid rows; float16/bfloat16/float32.

        Returns:
            tuple(array, array): codes int8 and scales float32, as
            dsa_indexer_qat_quant.
      )");

  m.def(
      "dsa_indexer_scores_q",
      &mlx_kquant::dsa_indexer_scores_q,
      "codes_q"_a,
      "scales_q"_a,
      "codes_k"_a,
      "scales_k"_a,
      "weights"_a,
      "causal"_a = true,
      "unused_causal_prefix_topk"_a = 0,
      "skip_causal_future_store"_a = false,
      "causal_q_offset"_a = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        dsa_indexer_scores on pre-quantized operands (dsa_indexer_qat_quant
        wire form) via int8 tensor-op MMA: each 128-dim dot runs as four
        K=32 int8 x int8 segments accumulated in int32 (exact), rescaled by
        the segment's scale pair. Scores are bit-identical to
        dsa_indexer_scores on the dequantized codes. Tensor-op hardware
        only (no fallback; dequantize and use dsa_indexer_scores instead).

        Args:
            codes_q (array): [B, H, M, 128] int8, H 32 or 64, M % 64 == 0.
            scales_q (array): [B, H, M, 4] float32.
            codes_k (array): [B, 1, N, 128] int8, N % 64 == 0.
            scales_k (array): [B, 1, N, 4] float32.
            weights (array): [B, M, H] (lh layout) or [B, H, M, 1],
                float16/bfloat16/float32.
            causal (bool): as in dsa_indexer_scores.
            unused_causal_prefix_topk (int): as in dsa_indexer_scores.
            skip_causal_future_store (bool): as in dsa_indexer_scores.
            causal_q_offset (int): as in dsa_indexer_scores.

        Returns:
            array: scores [B, 1, M, N]; bfloat16 for bfloat16 weights,
            else float16.
      )");

  m.def(
      "dsa_kv_qat",
      &mlx_kquant::dsa_kv_qat,
      "x"_a,
      "n_rot"_a,
      nb::kw_only(),
      "f16_round"_a = true,
      "stream"_a = nb::none(),
      R"(
        DeepSeek-V4-Flash main-attention KV QAT round-trip, fused: the
        per-64-block FP8-E4M3FN round-trip (scale 2^ceil(log2(amax/448))
        with a 1e-4 amax floor, clamp to +-448, ties-to-even) on the
        leading D - n_rot dims, the trailing n_rot RoPE dims fp8-exempt,
        then the whole row rounded through fp16 (the f16 KV-cache step).
        One kernel in place of the split + fp8-core + concat + astype
        chain, bit-identically.

        With ``f16_round`` false the fp16 step is dropped: the fp8 result
        stays in the storage dtype and the RoPE tail is copied through
        unchanged. That is the compressor emit-path form, where the pooled
        row is quantized but never passes through the f16 KV cache; it
        replaces the split + fp8-core + concat chain on its own.

        Args:
            x (array): any shape with trailing dim D,
                (D - n_rot) % 64 == 0; float16/bfloat16/float32.
            n_rot (int): trailing RoPE dims excluded from the fp8 step.
            f16_round (bool): apply the trailing fp16 round. Default True.

        Returns:
            array: same shape and dtype as ``x``.
      )");

  m.def(
      "moe_glu_gather_kq",
      &mlx_kquant::moe_glu_gather_kq,
      "x"_a,
      "gate_w"_a,
      "up_w"_a,
      "kquant_type"_a,
      "indices"_a,
      "act"_a = "silu",
      "limit"_a = 0.0f,
      "gate_bias"_a = nb::none(),
      "up_bias"_a = nb::none(),
      "alpha"_a = 0.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused MoE GLU gather for K-quant expert stacks: gate and up expert
        matvecs share each activation load and the GLU epilogue act(g) * u is
        applied in the same dispatch. Decode-shaped.

        Args:
            x (array): activations [T, K], float16/bfloat16. K % 256 == 0.
            gate_w (array): uint8 wire bytes (n_experts, N, bytes_per_row).
            up_w (array): uint8 wire bytes, same shape as gate_w.
            kquant_type (str): codec with a fused kernel (full GGUF matrix).
            indices (array): expert indices [T, R].
            act (str): 'silu' (default), 'gelu' (tanh approx), 'silu_limit'
                (silu(min(g, limit)) * clip(u, -limit, limit) -- deepseek-v4
                LimitedSwiGLU; requires limit > 0) or 'swiglu_clamp'
                (gpt-oss clamped SwiGLU: biases added, g clamped from above,
                u clamped both sides, sigmoid slope alpha and a (u + 1)
                linear term; requires gate_bias/up_bias, limit > 0 and
                alpha > 0; mxfp4/nvfp4 only).
            limit (float): clamp bound for 'silu_limit'/'swiglu_clamp'.
            gate_bias (array, optional): per-(expert, out_dim) bias [E, N],
                'swiglu_clamp' only.
            up_bias (array, optional): same shape, 'swiglu_clamp' only.
            alpha (float): sigmoid slope for 'swiglu_clamp'.

        Returns:
            array: activated hidden states [T, R, N] in x.dtype.
      )");

  m.def(
      "gather_qmv_kq",
      &mlx_kquant::gather_qmv_kq,
      "x"_a,
      "w"_a,
      "kquant_type"_a,
      "indices"_a,
      "bias"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Gathered matvec for K-quant expert stacks (the MoE down projection).
        One activation row per expert slot.

        Args:
            x (array): activations [T, R, K], float16/bfloat16. K % 256 == 0.
            w (array): uint8 wire bytes (n_experts, N, bytes_per_row).
            kquant_type (str): codec with a fused kernel (full GGUF matrix).
            indices (array): expert indices [T, R].
            bias (array, optional): per-(expert, out_dim) bias [E, N] added
                to each gathered row (mxfp4/nvfp4 only).

        Returns:
            array: output [T, R, N] in x.dtype.
      )");

  m.def(
      "moe_glu_gather_shexp_kq",
      &mlx_kquant::moe_glu_gather_shexp_kq,
      "x"_a,
      "gate_w"_a,
      "up_w"_a,
      "shexp_gate_w"_a,
      "shexp_up_w"_a,
      "kquant_type"_a,
      "indices"_a,
      "act"_a = "silu",
      "shexp_kquant_type"_a = "",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        moe_glu_gather_kq with the block's shared expert folded in as one
        extra slot (the last), fed by single-expert 2-D wire-byte tensors
        row-shape-matched to the expert stack.

        Args:
            x (array): activations [T, K], float16/bfloat16. K % 256 == 0.
            gate_w (array): uint8 wire bytes (n_experts, N, bytes_per_row).
            up_w (array): uint8 wire bytes, same shape as gate_w.
            shexp_gate_w (array): uint8 wire bytes (N, bytes_per_row).
            shexp_up_w (array): uint8 wire bytes (N, bytes_per_row).
            kquant_type (str): expert codec with a fused kernel.
            indices (array): expert indices [T, R].
            act (str): 'silu' (default) or 'gelu' (tanh approx).
            shexp_kquant_type (str): shared-expert codec; '' (default) =
                kquant_type. Mixed combos must be q6_k or q8_0.

        Returns:
            array: activated hidden states [T, R + 1, N] in x.dtype.
      )");

  m.def(
      "gather_qmv_mix_kq",
      &mlx_kquant::gather_qmv_mix_kq,
      "x"_a,
      "w"_a,
      "shexp_w"_a,
      "kquant_type"_a,
      "indices"_a,
      "scores"_a,
      "shexp_kquant_type"_a = "",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Gathered down projection with the routing mix folded in: every slot
        (the last being the shared expert) is accumulated in f32 weighted by
        its score, replacing gather + (y * scores).sum + shared add.

        Args:
            x (array): activations [T, S, K], float16/bfloat16. K % 256 == 0.
            w (array): uint8 wire bytes (n_experts, N, bytes_per_row).
            shexp_w (array): uint8 wire bytes (N, bytes_per_row).
            kquant_type (str): expert codec with a fused kernel.
            indices (array): expert indices [T, S - 1].
            scores (array): mix weights [T, S]; cast to float32.
            shexp_kquant_type (str): shared-expert codec; '' (default) =
                kquant_type. Mixed combos must be q6_k or q8_0.

        Returns:
            array: mixed output [T, N] in x.dtype.
      )");

  m.def(
      "gather_qmv_mix_ns_kq",
      &mlx_kquant::gather_qmv_mix_ns_kq,
      "x"_a,
      "w"_a,
      "kquant_type"_a,
      "indices"_a,
      "scores"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Gathered down projection with the routing mix folded in, no shared
        expert: each of the S routed slots is accumulated in f32 weighted by
        its score, replacing gather + (y * scores).sum.

        Args:
            x (array): activations [T, S, K], float16/bfloat16.
            w (array): uint8 wire bytes (n_experts, N, bytes_per_row).
            kquant_type (str): expert codec with a fused kernel.
            indices (array): expert indices [T, S].
            scores (array): mix weights [T, S]; cast to float32.

        Returns:
            array: mixed output [T, N] in x.dtype.
      )");

  m.def(
      "moe_router_topk",
      &mlx_kquant::moe_router_topk,
      "logits"_a,
      "top_k"_a,
      "norm_topk_prob"_a = true,
      "shared_gate"_a = true,
      "per_expert_scale"_a = nb::none(),
      "bias"_a = nb::none(),
      "scoring"_a = "softmax",
      "scale"_a = 1.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Router top-k in one dispatch: f32 scoring over the first E columns
        (softmax, or sqrtsoftplus = sqrt(softplus(x)) for deepseek-v4),
        top_k selection (min-index tie-break) optionally ranked by
        score + bias, optional renormalization, an optional per-expert
        scale applied to the picked scores, a uniform scale on emitted
        routed scores, and (when shared_gate) the sigmoid of column E (the
        shared-expert gate logit) in the last scores slot.

        Args:
            logits (array): router logits [T, E + shared_gate]; E <= 1024.
            top_k (int): experts per token, <= 16.
            norm_topk_prob (bool): renormalize picked scores. Required for
                sqrtsoftplus (renorm carries the model's 1e-20 guard).
            shared_gate (bool): logits carry a trailing shared-gate column.
            per_expert_scale (array, optional): [E] multiplier on picked
                scores, applied after renormalization; cast to float32.
            bias (array, optional): [E] selection bias
                (e_score_correction_bias); emitted scores stay unbiased.
            scoring (str): "softmax" or "sqrtsoftplus".
            scale (float): uniform multiplier on emitted routed scores
                (routed_scaling_factor).

        Returns:
            tuple: (indices [T, top_k] uint32,
            scores [T, top_k + shared_gate] float32).
      )");

  m.def(
      "add_rmsnorm",
      &mlx_kquant::add_rmsnorm,
      "h"_a,
      "residual"_a,
      "weight"_a,
      "eps"_a,
      "scale"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused post-norm residual: (residual + rms_norm(h, weight)) * scale
        in one dispatch, all math in f32.

        Args:
            h (array): [..., D], float16/bfloat16.
            residual (array): same shape and dtype as h.
            weight (array): [D] norm weight, same dtype as h.
            eps (float): rms_norm epsilon.
            scale (array, optional): size-1 epilogue scalar, same dtype as
                h; 1.0 when absent.

        Returns:
            array: same shape and dtype as h.
      )");

  m.def(
      "rmsnorm_multi3",
      &mlx_kquant::rmsnorm_multi3,
      "x"_a,
      "w0"_a,
      "w1"_a,
      "w2"_a,
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Three rms_norms of one tensor in one dispatch, sharing the
        mean-square reduction of x.

        Args:
            x (array): [..., D], float16/bfloat16.
            w0 (array): [D] norm weight, same dtype as x.
            w1 (array): [D] norm weight, same dtype as x.
            w2 (array): [D] norm weight, same dtype as x.
            eps (float): rms_norm epsilon.

        Returns:
            tuple: (rms_norm(x, w0), rms_norm(x, w1), rms_norm(x, w2)).
      )");

  m.def(
      "rmsnorm2_add",
      &mlx_kquant::rmsnorm2_add,
      "a"_a,
      "wa"_a,
      "b"_a,
      "wb"_a,
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused branch merge: rms_norm(a, wa) + rms_norm(b, wb) in one
        dispatch, all math in f32.

        Args:
            a (array): [..., D], float16/bfloat16.
            wa (array): [D] norm weight, same dtype as a.
            b (array): same shape and dtype as a.
            wb (array): [D] norm weight, same dtype as a.
            eps (float): rms_norm epsilon.

        Returns:
            array: same shape and dtype as a.
      )");

  m.def(
      "hc_front_reduce",
      &mlx_kquant::hc_front_reduce,
      "x"_a,
      "fn"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Hyper-connection front reduction for the fused M=1 decode route:
        the 24 mix dots of x against fn plus the row sum of squares
        (deferred rms factor). hc_mult 4 only.

        Args:
            x (array): [..., 4, D] streams, float16/bfloat16, D % 8 == 0,
                D <= 8192.
            fn (array): [24, 4 * D] float32 mix matrix.

        Returns:
            tuple: (mixes_raw f32 [..., 24], sumsq f32 [..., 1]).
      )");

  m.def(
      "hc_front_expand_reduce",
      &mlx_kquant::hc_front_expand_reduce,
      "x_sub"_a,
      "resid"_a,
      "post"_a,
      "comb"_a,
      "fn"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        The previous cycle's hc_expand fused ahead of the front reduction:
        one dispatch expands (x_sub, resid, post, comb) to h, writes it,
        and reduces the mix dots and sum of squares of h.

        Args:
            x_sub (array): [..., D] sublayer output.
            resid (array): [..., 4, D] residual streams.
            post (array): [..., 4] float32.
            comb (array): [..., 4, 4] float32.
            fn (array): [24, 4 * D] float32 mix matrix.

        Returns:
            tuple: (h [..., 4, D], mixes_raw f32 [..., 24],
            sumsq f32 [..., 1]); h is bit-identical to the unfused expand.
      )");

  m.def(
      "hc_sinkhorn_collapse",
      &mlx_kquant::hc_sinkhorn_collapse,
      "x"_a,
      "mixes_raw"_a,
      "sumsq"_a,
      "scale"_a,
      "base"_a,
      "w"_a,
      "iters"_a,
      "hc_eps"_a,
      "norm_eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Sinkhorn mix normalization plus stream collapse with the sublayer
        RMSNorm folded into the single output rounding. The deferred front
        rms factor enters through sumsq and multiplies the three scales.

        Args:
            x (array): [..., 4, D] streams, float16/bfloat16.
            mixes_raw (array): [..., 24] float32 from the front reduction.
            sumsq (array): [..., 1] float32 row sum of squares.
            scale (array): [3] float32 pre/post/comb scales.
            base (array): [24] float32 mix biases.
            w (array): [D] sublayer norm weight, same dtype as x.
            iters (int): sinkhorn iterations.
            hc_eps (float): sinkhorn epsilon.
            norm_eps (float): rms_norm epsilon.

        Returns:
            tuple: (collapsed [..., D], post f32 [..., 4],
            comb f32 [..., 4, 4]).
      )");

  m.def(
      "hc_expand",
      &mlx_kquant::hc_expand,
      "x"_a,
      "resid"_a,
      "post"_a,
      "comb"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Expand the sublayer output back to four streams:
        out[i] = post[i] * x + sum_j comb[j][i] * resid[j].

        Args:
            x (array): [..., D] sublayer output.
            resid (array): [..., 4, D] residual streams.
            post (array): [..., 4] float32.
            comb (array): [..., 4, 4] float32.

        Returns:
            array: [..., 4, D], dtype of resid.
      )");

  m.def(
      "hc_lowrank_norm",
      &mlx_kquant::hc_lowrank_norm,
      "h"_a,
      "gamma"_a,
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Grouped rms norm for the low-rank hyper-connection (qwen4exp):
        one statistic per stream, gamma gain over all streams. One
        dispatch, materializing xn once for the front and epilogue.

        Args:
            h (array): [..., 4, D] residual streams, float32 or the gamma
                dtype. D % 64 == 0, D <= 8192.
            gamma (array): [4 * D] norm gain, float16/bfloat16.
            eps (float): rms epsilon.

        Returns:
            array: xn [..., 4, D] in the h dtype.
      )");

  m.def(
      "hc_lowrank_front",
      &mlx_kquant::hc_lowrank_front,
      "xn"_a,
      "w_down"_a,
      "w_inject"_a,
      "lo_dtype"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused low-rank hyper-connection front (qwen4exp): the q8_0 down
        qmv over xn with silu(x / 4), plus the float32 inject dots
        2 * sigmoid(x / 4). One dispatch.

        Args:
            xn (array): [..., 4, D] normed streams (hc_lowrank_norm
                output), float32 or lo_dtype. D % 64 == 0, D <= 8192.
            w_down (array): [LR, 4 * D / 32 * 34] uint8 q8_0 wire.
                LR % 32 == 0, LR <= 512.
            w_inject (array): [4, 4 * D] float32.
            lo_dtype (Dtype): float16 or bfloat16; the down qmv output
                rounds here before silu (the kq qmv promotion).

        Returns:
            tuple: (lo [..., LR] lo_dtype, inj f32 [..., 4]).
      )");

  m.def(
      "hc_lowrank_epilogue",
      &mlx_kquant::hc_lowrank_epilogue,
      "lo"_a,
      "w_up"_a,
      "xn"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused low-rank hyper-connection epilogue (qwen4exp): the q8_0 up
        qmv of lo, sigmoid gate against xn, mean over the 4 streams. One
        dispatch.

        Args:
            lo (array): [..., LR] float16/bfloat16 (hc_lowrank_front
                output).
            w_up (array): [4 * D, LR / 32 * 34] uint8 q8_0 wire.
            xn (array): [..., 4, D] normed streams (hc_lowrank_norm
                output).

        Returns:
            array: mixed [..., D] in the xn dtype.
      )");

  m.def(
      "get_cb_caps",
      &mlx_kquant::get_cb_caps,
      R"(
        Read MLX's live command-buffer split caps.

        Returns:
            tuple: (max_ops_per_buffer, max_mb_per_buffer).
      )");

  m.def(
      "set_cb_caps",
      &mlx_kquant::set_cb_caps,
      "max_ops"_a,
      "max_mb"_a,
      R"(
        Set MLX's command-buffer split caps at runtime. The env knobs
        latch at device init; decode wants coarse buffers, deep prefill
        fine ones, so servers flip these per phase.

        Args:
            max_ops (int): ops per command buffer, in [1, 2^30].
            max_mb (int): MB per command buffer, in [1, 2^30].

        Returns:
            tuple: the previous (max_ops, max_mb).
      )");

  m.def(
      "skinny_matmul",
      &mlx_kquant::skinny_matmul,
      "x"_a,
      "w"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        y = x @ w.T at token widths 1..16 against a small-N, large-K
        weight in nn.Linear layout, f32 accumulate. Fills the GEMV-to-GEMM
        cliff MLX hits at M >= 2 on these shapes.

        Args:
            x (array): [..., M, K], 1 <= M <= 16, K a multiple of 4;
                float16/bfloat16/float32.
            w (array): [N, K] weight, dtype matching x or float32.

        Returns:
            array: [..., M, N]; float32 when either operand is, else the
            x dtype.
      )");

  m.def(
      "route_shed",
      &mlx_kquant::route_shed,
      "indices"_a,
      "scores"_a,
      "slot_table"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        GPU-side routed-expert slot remap + residency shed for streamed MoE
        decode. Remaps routed expert ids to arena slots via slot_table
        (expert id -> slot, negative = non-resident), sheds every
        non-resident expert (a lazy graph cannot demand-read disk),
        renormalizes kept gate weights mass-preserving
        (score * S_all / S_kept), and reports misses for between-token
        prestaging. Shed entries keep a valid slot (the row's first kept
        slot, else 0) with a zero mix weight, so downstream gather kernels
        need no changes. All-miss rows return an all-zero mix.

        Args:
            indices (array): expert indices [T, R], uint32. R <= 64.
            scores (array): gate scores [T, R], float32.
            slot_table (array): [n_experts], int32; slot index or negative.

        Returns:
            tuple: (slots u32 [T, R], mix f32 [T, R],
            miss_ids i32 [T, R] front-packed descending-score with -1 pad,
            miss_scores f32 [T, R] aligned with 0 pad).
      )");

  m.def(
      "gather_qmm",
      &mlx_kquant::gather_qmm,
      "x"_a,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "lhs_indices"_a = nb::none(),
      "rhs_indices"_a = nb::none(),
      "transpose"_a = true,
      "sorted_indices"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Gather (mixture-of-experts) quantized matmul for GGUF K-quant weights.

        Args:
            x (array): float activations, at least 2-D.
            w (array): uint8 K-quant wire bytes shaped
                (n_experts, out_dims, bytes_per_row).
            scales (array): vestigial placeholder; ignored by the kernel.
            kquant_type (str): codec name, e.g. ``"q4_k"``.
            lhs_indices (array, optional): uint32 indices selecting x rows.
                Defaults to a plain arange.
            rhs_indices (array, optional): uint32 indices selecting expert
                weight matrices. Defaults to a plain arange.
            transpose (bool): whether each expert ``w`` is transposed
                ([out, in]). Default True.
            sorted_indices (bool): hint that the defaulted indices are sorted.

        Returns:
            array: the gathered matmul result (x.dtype, float32 -> bfloat16).
      )");

  m.def(
      "expert_tile_map",
      &mlx_kquant::expert_tile_map,
      "indices"_a,
      "n_experts"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Build the gather_qmm_seg tile map from expert-sorted routing indices,
        entirely on GPU (no host sync).

        Args:
            indices (array): 1-D uint32 expert ids, sorted ascending.
            n_experts (int): total expert count (sizes the map bound).

        Returns:
            tuple: (map, counts) -- uint32 [cap, 3] tile table of
            (expert, row_start, num_rows) tiling each segment into 64-row
            tiles where only the last tile of a segment can be partial, plus
            uint32 [1] valid-tile count. Slots past the count are
            uninitialized; tile order is unspecified. Metal-only.
      )");

  m.def(
      "gather_qmm_seg",
      &mlx_kquant::gather_qmm_seg,
      "x"_a,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "map"_a,
      "counts"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Segment-walking gather GEMM for expert-sorted MoE prefill.

        Args:
            x (array): float activations [rows, K], rows pre-sorted by expert.
            w (array): uint8 K-quant wire bytes shaped
                (n_experts, out_dims, bytes_per_row).
            scales (array): vestigial placeholder; ignored by the kernel.
            kquant_type (str): codec name, e.g. ``"iq2_xxs"``.
            map (array): uint32 [n_tiles, 3] rows of
                (expert, row_start, num_rows), num_rows <= 64. Only the last
                tile of a segment may be partial; its dead row fragments are
                skipped in the MMA.
            counts (array): uint32 [1] valid-tile count. Valid tiles never
                span experts and must cover every x row exactly once
                (uncovered output rows are left unwritten). Use
                expert_tile_map to build both.

        Returns:
            array: [rows, out_dims] (x.dtype, float32 -> bfloat16). Metal-only.
      )");

  m.def(
      "quantize",
      &mlx_kquant::quantize,
      "w"_a,
      "kquant_type"_a,
      "imatrix"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Encode a float weight tensor into GGUF K-quant wire bytes (CPU or Metal).

        Args:
            w (array): float weights; last dim a multiple of the codec's
                weights_per_block.
            kquant_type (str): codec name, e.g. ``"q4_k"``, ``"q8_0"``.
            imatrix (array, optional): 1-D float32 importance vector of length
                K = ``w.shape[-1]`` to steer the encoder.

        Returns:
            tuple[array, array]: ``(wq, scales)`` where ``wq`` is the uint8 wire
            bytes and ``scales`` is a vestigial uint8 placeholder of shape [1]
            (K-quant scales live inside ``wq``).
      )");

  // --- GGUF loader ---
  m.def(
      "load_gguf",
      [](const std::string& path, bool zero_copy) {
        mlx_kquant::GgufLoadResult res = mlx_kquant::load_gguf(path, zero_copy);

        nb::dict arrays;
        for (auto& [name, arr] : res.arrays) {
          arrays[nb::str(name.c_str())] = nb::cast(arr);
        }
        nb::dict codecs;
        for (auto& [name, codec] : res.codecs) {
          codecs[nb::str(name.c_str())] = nb::str(codec.c_str());
        }
        nb::dict metadata;
        for (auto& [key, value] : res.metadata) {
          metadata[nb::str(key.c_str())] = meta_to_py(value);
        }
        nb::dict shapes;
        for (auto& [name, dims] : res.tensor_shapes) {
          shapes[nb::str(name.c_str())] = nb::cast(dims);
        }
        return nb::make_tuple(arrays, codecs, metadata, shapes);
      },
      "path"_a,
      "zero_copy"_a = true,
      R"(
        Load a GGUF file's tensors and metadata directly from gguflib's mmap.

        With ``zero_copy=True`` (default) each tensor array is a no-copy view
        over the mmap (a Metal newBufferWithBytesNoCopy per tensor, kept mapped
        until the last viewing array is freed) - no per-tensor allocation or
        byte-copy. With ``zero_copy=False`` every tensor is memcpy'd out of the
        mmap (~15 GB/s). Tensors that can't be wrapped no-copy fall back to the
        copy path transparently.

        Args:
            path (str): GGUF file path.
            zero_copy (bool): view the mmap instead of copying (default True).

        Returns:
            tuple[dict, dict, dict, dict]: ``(arrays, codecs, metadata, shapes)``:
              - arrays: tensor name -> mx.array. K-quant tensors are uint8 wire
                bytes (MLX axis order, byte-packed last dim) each with a 1-byte
                ``<prefix>.scales`` placeholder; F32/F16/BF16/I8/I16/I32 tensors
                keep their native dtype.
              - codecs: K-quant tensor name -> codec name ("q4_k", ...).
              - metadata: GGUF KV key -> decoded value (int/float/bool/str/list).
              - shapes: tensor name -> logical shape (GGUF native, innermost-first
                order; matches gguf-py ReaderTensor.shape).
      )");

  m.def(
      "zero_copy_view_count",
      &mlx_kquant::zero_copy_view_count,
      "Number of live zero-copy GGUF tensor views (registered mmap ranges).");

  m.def(
      "verify_zero_copy_views",
      [](nb::list items, std::vector<std::string> no_alias) {
        std::vector<std::pair<std::string, mx::array>> pairs;
        pairs.reserve(nb::len(items));
        for (nb::handle h : items) {
          auto t = nb::cast<nb::sequence>(h);
          pairs.emplace_back(
              nb::cast<std::string>(t[0]), nb::cast<mx::array>(t[1]));
        }
        return mlx_kquant::verify_zero_copy_views(pairs, no_alias);
      },
      "items"_a,
      "no_alias"_a = std::vector<std::string>{},
      R"(
        Check (name, array) pairs against the live zero-copy GGUF mappings.

        Returns a list of problem strings, one per violation: an array whose
        buffer sits inside a mapped GGUF tensor range but whose dtype differs
        from the wire dtype recorded at load (integer-to-integer reinterprets
        allowed), or an array named in ``no_alias`` that aliases any mapping
        (loader transforms must produce owned buffers). An empty list means
        clean. This detects buffer donation into the file mapping: a donated
        dtype-changing copy leaves an array typed X over wire bytes typed Y,
        and the write is dropped on read-only shared mappings. Arrays must be
        evaluated first. Metadata-only; no tensor data is read.

        Args:
            items (list[tuple[str, array]]): named arrays to check, e.g.
                ``mlx.utils.tree_flatten(model.parameters())``.
            no_alias (list[str]): names that must not alias any mapping.

        Returns:
            list[str]: problem descriptions; empty when clean.
      )");

  // --- feeder-loop support: writable zero-copy buffers + shared events ---

  m.def(
      "arena_alloc",
      [](const std::vector<int>& shape, int itemsize) {
        mlx::core::Dtype dt = mlx::core::uint8;
        switch (itemsize) {
          case 1:
            break;
          case 2:
            dt = mlx::core::uint16;
            break;
          case 4:
            dt = mlx::core::uint32;
            break;
          case 8:
            dt = mlx::core::uint64;
            break;
          default:
            throw std::invalid_argument(
                "[mlx_kquant.arena_alloc] itemsize must be 1, 2, 4 or 8.");
        }
        auto [arr, addr] = mlx_kquant::arena_alloc(
            mlx::core::Shape(shape.begin(), shape.end()), dt);
        PyObject* mv = PyMemoryView_FromMemory(
            reinterpret_cast<char*>(addr),
            static_cast<Py_ssize_t>(arr.nbytes()),
            PyBUF_WRITE);
        if (mv == nullptr) {
          throw nb::python_error();
        }
        return nb::make_tuple(arr, nb::steal(mv));
      },
      "shape"_a,
      "itemsize"_a = 1,
      R"(
        Allocate a page-aligned host buffer wrapped zero-copy as a Metal
        shared-storage unsigned-integer array (dtype uint8/16/32/64 per
        ``itemsize``; wider itemsizes let a >2 GiB slot fit int32 shape dims).

        Returns (array, memoryview): the same bytes seen from both sides,
        the memoryview always byte-addressed over the full allocation.
        The writable memoryview is the CPU feeder's window (os.preadv into
        slices of it reads disk straight into GPU-visible memory); the array
        is what kernels consume. The memoryview is valid only while the array
        is alive - hold them together. Writes become safely visible to GPU
        work encoded after an event_wait whose value the writer signals
        (shared_event_set) after writing; nothing else orders them.
      )");

  m.def(
      "residency_insert",
      &mlx_kquant::residency_insert,
      "a"_a,
      "Stage ``a``'s underlying Metal buffer for the device residency set "
      "(wired for the buffer's lifetime once residency_commit runs), so "
      "command buffers stop re-wiring its pages on every use. The array "
      "must have materialized data (evaluate first). False on non-Metal "
      "builds or missing data.");

  m.def(
      "residency_commit",
      &mlx_kquant::residency_commit,
      "Commit staged residency_insert additions and request residency. "
      "False on non-Metal builds.");

  m.def(
      "residency_erase",
      &mlx_kquant::residency_erase,
      "a"_a,
      "Stage removal of ``a``'s buffer from the residency set (call before "
      "dropping a member buffer; takes effect at the next commit). False "
      "on non-Metal builds or missing data.");

  // --- shared-event stream primitives (feeder loop) ---

  m.def(
      "shared_event_create",
      &mlx_kquant::shared_event_create,
      "Create a process-wide MTLSharedEvent (signaled value 0); returns an "
      "opaque handle for the other shared_event_*/event_* calls. Metal-only.");

  m.def(
      "shared_event_destroy",
      &mlx_kquant::shared_event_destroy,
      "handle"_a,
      "Release a shared event. Live command buffers keep their own "
      "reference; encoding against a destroyed handle throws.");

  m.def(
      "shared_event_set",
      &mlx_kquant::shared_event_set,
      "handle"_a,
      "value"_a,
      "Host-side signal: set the event's value. Releases any encoded "
      "event_wait on a value <= the one set; setting 2**64 - 1 is the "
      "poison that un-wedges every wait on the event (watchdog recovery).");

  m.def(
      "shared_event_read",
      &mlx_kquant::shared_event_read,
      "handle"_a,
      "Current signaled value of the event (non-blocking).");

  m.def(
      "shared_event_wait",
      &mlx_kquant::shared_event_wait,
      "handle"_a,
      "value"_a,
      "timeout_ms"_a = -1,
      nb::call_guard<nb::gil_scoped_release>(),
      "Block the calling thread until the event reaches ``value``; releases "
      "the GIL. ``timeout_ms < 0`` waits forever. Returns False on timeout.");

  m.def(
      "event_signal",
      &mlx_kquant::event_signal,
      "x"_a,
      "handle"_a,
      "value"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Identity op on ``x`` that encodes an MTLSharedEvent signal at its
        position in evaluation order: the event reaches ``value`` only after
        all GPU work encoded before it (e.g. the router whose output the
        feeder is about to read) has completed. The result aliases ``x`` and
        must feed downstream compute or be evaluated - an unused output
        encodes nothing. On the CPU stream it is a plain identity.
      )");

  m.def(
      "event_wait",
      &mlx_kquant::event_wait,
      "x"_a,
      "handle"_a,
      "value"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Identity op on ``x`` that encodes an MTLSharedEvent wait at its
        position in evaluation order: GPU work encoded after it (e.g. an
        expert kernel reading staged weights) stalls until the event reaches
        ``value`` - normally from shared_event_set on the feeder thread, or
        the 2**64 - 1 poison to drain a wedged buffer. The result aliases
        ``x`` and must feed downstream compute or be evaluated. On the CPU
        stream it is a plain identity.
      )");
}
