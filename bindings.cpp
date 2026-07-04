#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
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
      &mlx_kquant::sdpa_decode_gqa,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "sinks"_a = nb::none(),
      "splits"_a = 0,
      "tile_c"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Decode/verify GQA attention tuned for long KV caches: the key axis
        is split into a fixed number of coarse contiguous chunks and each
        chunk is streamed through threadgroup-staged K/V tiles shared by the
        whole GQA group, so device memory reads the KV once per kv-head. At
        qL 2..4 (speculative-verify width) every query also shares the staged
        tiles, causally clamped to its own trailing position.

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

        Returns:
            array: attention output [B, n_q_heads, qL, D].
      )");

  m.def(
      "sdpa_fa_verify",
      &mlx_kquant::sdpa_fa_verify,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "q_len"_a,
      "splits"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Speculative-verify attention on the GPU matrix units for a GQA-folded
        query tile. Fold the GQA group into the query rows first --
        q [1, Hq, q_len, D] reshaped to [1, Hkv, G*q_len, D] with kv-major
        heads -- and pass the original q_len: folded row r is causally
        clamped to key <= kL - q_len + (r % q_len). The 32-row query tile
        streams each contiguous KV split once, computing S = Q @ K^T and
        O += P @ V on simdgroup_matrix with float32 accumulators and a
        per-row online softmax; per-split partials are merged by the same
        reduction pass as ``sdpa_decode_gqa``.

        Args:
            q (array): folded queries [1, n_kv_heads, G*q_len, D],
                float16/bfloat16; G*q_len <= 32, D = 256.
            k (array): keys [1, n_kv_heads, kL, D]; head/seq strided is fine
                (read in place), the head_dim must be contiguous.
            v (array): values [1, n_kv_heads, kL, D].
            scale (float): query scale (typically 1/sqrt(D)).
            q_len (int): pre-fold query length (2..8); sets each folded
                row's causal clamp.
            splits (int): key-axis split count; 0 picks the default.

        Returns:
            array: attention output [1, n_kv_heads, G*q_len, D].
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
      "moe_glu_gather_kq",
      &mlx_kquant::moe_glu_gather_kq,
      "x"_a,
      "gate_w"_a,
      "up_w"_a,
      "kquant_type"_a,
      "indices"_a,
      "act"_a = "silu",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused MoE GLU gather for K-quant expert stacks: gate and up expert
        matvecs share each activation load and the GLU epilogue act(g) * u is
        applied in the same dispatch. No biases. Decode-shaped.

        Args:
            x (array): activations [T, K], float16/bfloat16. K % 256 == 0.
            gate_w (array): uint8 wire bytes (n_experts, N, bytes_per_row).
            up_w (array): uint8 wire bytes, same shape as gate_w.
            kquant_type (str): codec with a fused kernel (full GGUF matrix).
            indices (array): expert indices [T, R].
            act (str): 'silu' (default) or 'gelu' (tanh approx).

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
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Router top-k in one dispatch: f32 softmax over the first E columns,
        top_k selection (min-index tie-break), optional renormalization, an
        optional per-expert scale applied to the picked scores, and (when
        shared_gate) the sigmoid of column E (the shared-expert gate logit)
        in the last scores slot.

        Args:
            logits (array): router logits [T, E + shared_gate]; E <= 1024.
            top_k (int): experts per token, <= 16.
            norm_topk_prob (bool): renormalize picked probabilities.
            shared_gate (bool): logits carry a trailing shared-gate column.
            per_expert_scale (array, optional): [E] multiplier on picked
                scores, applied after renormalization; cast to float32.

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
}
