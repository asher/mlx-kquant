#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "kquant.h"
#include "kquant_codec.h"
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
}
