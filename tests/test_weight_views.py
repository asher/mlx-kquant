"""Views of a packed weight and of the other operands a kernel reads dense.

quantized_matmul and the ops built on it walk the last two dims of a
packed weight as dense rows. A view with other strides has to give the
values of its contiguous copy, whether the view has memory when the op is
built or not. An unevaluated view reports dense strides whatever it
becomes, so the op decides for it at eval, in a KQuantMatrixDense node. A
weight that has memory and dense rows adds no node. The same holds for the
bytes dequantize decodes, the floats quantize encodes and the bias of the
fused mat-vec.

The cases run on the default device, so the suite covers the GPU and, under
KQUANT_FORCE_CPU, the CPU stream.
"""

from __future__ import annotations

import re

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

N, K = 96, 512
CODECS = ["q4_k", "q6_k", "q8_0", "q4_0", "q4_1"]
# 1 row takes qmv, 4 the verify kernel, 64 and 1100 the matmul tiles.
ROWS = [1, 4, 64, 1100]
SCALES = mx.zeros((1,), dtype=mx.uint8)
_W = {}


def _wire(codec, n, k, seed):
    key = (codec, n, k, seed)
    if key not in _W:
        wf = mx.random.normal((n, k), key=mx.random.key(seed)) * 0.1
        w, _ = kq.quantize(wf, codec)
        mx.eval(w)
        _W[key] = w
    return _W[key]


def _x(m, k, dtype=mx.bfloat16):
    x = (mx.random.normal((m, k), key=mx.random.key(m)) * 0.5).astype(dtype)
    mx.eval(x)
    return x


def _dense(view):
    """A copy of a view in its own row-major buffer."""
    return mx.array(np.ascontiguousarray(np.array(view)))


def _views(codec):
    """name -> a function that builds a fresh [N, K bytes] view."""
    w = _wire(codec, 2 * N, K, 1)
    w2 = _wire(codec, 2 * N, K, 2)
    kb = w.shape[1]
    wide = mx.concatenate([w, w2], axis=1)
    stack = mx.stack([w, w2])
    turned = _dense(mx.swapaxes(w[:N], 0, 1))
    flipped = _dense(w[:N, ::-1])
    mx.eval(wide, stack, turned, flipped)
    return {
        # Rows of the right length that run backwards in memory.
        "reversed_cols": lambda: flipped[:, ::-1],
        # Rows that are not dense: every other row, from row 0 and row 1.
        "even_rows": lambda: w[0::2],
        "odd_rows": lambda: w[1::2],
        # Dense rows that start inside a larger buffer.
        "row_band": lambda: w[N // 2 : N // 2 + N],
        "picked": lambda: stack[1][:N],
        # A column band of a wider buffer: the row stride is the wider row.
        "left_cols": lambda: wide[:N, :kb],
        "right_cols": lambda: wide[:N, kb:],
        # Rows along the stride-1 axis of the buffer.
        "transposed": lambda: mx.swapaxes(turned, 0, 1),
        "whole": lambda: w[:N],
    }


def _nodes(*outputs):
    """Names of the primitives in the graph of the outputs."""
    import tempfile

    with tempfile.NamedTemporaryFile("r", suffix=".dot") as f:
        mx.export_to_dot(f.name, *outputs)
        text = f.read()
    return sorted(re.findall(r'label ="([A-Za-z0-9_]+)"', text))


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("codec", CODECS)
def test_quantized_matmul_weight_views(codec, state):
    views = _views(codec)
    for m in ROWS:
        x = _x(m, K)
        for name, make in views.items():
            ref = kq.quantized_matmul(x, _dense(make()), SCALES, codec, True)
            view = make()
            if state == "evaluated":
                mx.eval(view)
            y = kq.quantized_matmul(x, view, SCALES, codec, True)
            mx.eval(y, ref)
            assert mx.array_equal(y, ref).item(), (codec, m, name, state)


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("codec", ["q4_k", "q8_0", "q4_0", "q4_1"])
def test_multi_weight_views(codec, dtype, state):
    """1 row runs quantized_matmul per weight, 4 and 6 rows the joined
    verify kernel where there is one."""
    views = _views(codec)
    for m in (1, 4, 6):
        x = _x(m, K, dtype)
        for pair in (("even_rows", "odd_rows"), ("row_band", "left_cols")):
            makes = [views[name] for name in pair] + [views["whole"]]
            ref = kq.quantized_matmul_multi(x, [_dense(f()) for f in makes], codec)
            ws = [f() for f in makes]
            if state == "evaluated":
                mx.eval(ws)
            ys = kq.quantized_matmul_multi(x, ws, codec)
            mx.eval(ys, ref)
            for y, r in zip(ys, ref, strict=True):
                assert mx.array_equal(y, r).item(), (codec, m, pair, state)


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("m", [64, 1100])
@pytest.mark.parametrize("codec", ["q4_k", "q8_0"])
def test_glu_weight_views(codec, m, dtype, state):
    """64 rows build the composition on every device, 1100 rows run the
    fused kernel where there is one."""
    both = _wire(codec, 2 * N, 768, 5)
    x = _x(m, 768, dtype)
    pairs = {
        "interleaved": (lambda: both[0::2], lambda: both[1::2]),
        "bands": (lambda: both[:N], lambda: both[N:]),
    }
    for activation in ("silu", "gelu_tanh"):
        for name, (gate, up) in pairs.items():
            ref = kq.quantized_matmul_glu(
                x, _dense(gate()), _dense(up()), codec, activation
            )
            wg, wu = gate(), up()
            if state == "evaluated":
                mx.eval(wg, wu)
            y = kq.quantized_matmul_glu(x, wg, wu, codec, activation)
            mx.eval(y, ref)
            assert mx.array_equal(y, ref).item(), (codec, m, name, activation)


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
def test_qmv_bias_weight_views(dtype, state):
    """The weight of the bias-fused mat-vec, which runs q8_0."""
    x = _x(1, K, dtype)
    b = (mx.random.normal((N,), key=mx.random.key(9)) * 4).astype(dtype)
    mx.eval(b)
    for name, make in _views("q8_0").items():
        ref = kq.quantized_matmul_qmv_bias(x, _dense(make()), SCALES, b, "q8_0")
        view = make()
        if state == "evaluated":
            mx.eval(view)
        y = kq.quantized_matmul_qmv_bias(x, view, SCALES, b, "q8_0")
        mx.eval(y, ref)
        assert mx.array_equal(y, ref).item(), (name, state)


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("sorted_indices", [False, True], ids=["unsorted", "sorted"])
@pytest.mark.parametrize("rows", [6, 300])
def test_gather_qmm_weight_views(rows, sorted_indices, state):
    """An expert stack that is every other row of each expert. The rows
    of a matrix are dense and the matrices are not, with and without the
    sorted_indices promise."""
    experts = 4
    stack = _dense(_wire("q8_0", experts * 2 * N, K, 12).reshape(experts, 2 * N, -1))
    mx.eval(stack)
    rng = np.random.default_rng(rows)
    rhs = mx.array(np.sort(rng.integers(0, experts, size=rows)).astype(np.uint32))
    x = _x(rows, K, mx.float16).reshape(rows, 1, K)

    def call(w):
        return kq.gather_qmm(
            x,
            w,
            SCALES,
            "q8_0",
            rhs_indices=rhs,
            transpose=True,
            sorted_indices=sorted_indices,
        )

    for make in (lambda: stack[:, 0::2], lambda: stack[:, 1::2]):
        ref = call(_dense(make()))
        view = make()
        if state == "evaluated":
            mx.eval(view)
        y = call(view)
        mx.eval(y, ref)
        assert mx.array_equal(y, ref).item()


def test_weight_with_memory_adds_no_node():
    """The call gmlx makes in decode: an evaluated weight with dense rows.
    Its graph is the cast-free x layout node and the matmul, and nothing
    for the weight."""
    w = _wire("q4_k", 2 * N, K, 1)
    x = _x(4, K)
    assert _nodes(kq.quantized_matmul(x, w, SCALES, "q4_k", True)) == [
        "Contiguous",
        "KQuantMatmul",
    ]
    # A row band with memory has dense rows too.
    band = w[N // 2 : N // 2 + N]
    mx.eval(band)
    assert _nodes(kq.quantized_matmul(x, band, SCALES, "q4_k", True)) == [
        "Contiguous",
        "KQuantMatmul",
    ]


def test_lazy_weight_gets_the_eval_time_node():
    w = _wire("q4_k", 2 * N, K, 1)
    x = _x(4, K)
    names = _nodes(kq.quantized_matmul(x, w[0::2], SCALES, "q4_k", True))
    assert names.count("KQuantMatrixDense") == 1
    assert "KQuantMatmul" in names


def test_dense_lazy_view_keeps_its_buffer():
    """A lazy row band is passed through, not copied: the call allocates the
    output and nothing the size of the weight."""
    n, k = 2048, 2048
    w = _wire("q8_0", 2 * n, k, 7)
    x = _x(4, k)
    y = kq.quantized_matmul(x, w[n:], SCALES, "q8_0", True)
    mx.eval(y)
    del y
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    y = kq.quantized_matmul(x, w[n:], SCALES, "q8_0", True)
    mx.eval(y)
    grown = mx.get_peak_memory() - base
    # The band is 4.4 MB. The output and x layout node are a few KB.
    assert grown < w.nbytes // 8, (grown, w.nbytes)


# dequantize decodes its input as one row-major run of bytes, quantize
# encodes its float input the same way, and the bias of the fused mat-vec
# is read as dense elements. A view passes through when it is row-major in
# all of its dims and is copied otherwise, lazy or not.


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("codec", CODECS)
def test_dequantize_weight_views(codec, state):
    views = _views(codec)
    w = _wire(codec, 2 * N, K, 1)
    views["reversed"] = lambda: w[::-1][:N]
    for name, make in views.items():
        ref = kq.dequantize(_dense(make()), SCALES, codec, dtype=mx.float32)
        view = make()
        if state == "evaluated":
            mx.eval(view)
        y = kq.dequantize(view, SCALES, codec, dtype=mx.float32)
        mx.eval(y, ref)
        assert mx.array_equal(y, ref).item(), (codec, name, state)


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
def test_dequantize_and_quantize_3d_views(state):
    """Every other matrix of a 3-D input. Each matrix has dense rows, and
    the run of bytes or floats as a whole is not row-major."""
    wf = mx.random.normal((8, 16, K), key=mx.random.key(3)) * 0.1
    w, _ = kq.quantize(wf, "q8_0")
    mx.eval(wf, w)
    for step in (slice(0, None, 2), slice(1, None, 2), slice(None, None, -1)):
        ref = kq.dequantize(_dense(w[step]), SCALES, "q8_0", dtype=mx.float32)
        view = w[step]
        if state == "evaluated":
            mx.eval(view)
        y = kq.dequantize(view, SCALES, "q8_0", dtype=mx.float32)
        mx.eval(y, ref)
        assert mx.array_equal(y, ref).item(), (step, state)
        ref, _ = kq.quantize(_dense(wf[step]), "q8_0")
        view = wf[step]
        if state == "evaluated":
            mx.eval(view)
        y, _ = kq.quantize(view, "q8_0")
        mx.eval(y, ref)
        assert mx.array_equal(y, ref).item(), (step, state)


def test_dequantize_nodes():
    """A weight with memory adds no node. A lazy view gets the node that
    decides at eval."""
    w = _wire("q4_k", 2 * N, K, 1)
    assert _nodes(kq.dequantize(w, SCALES, "q4_k")) == ["KQuantDequantize"]
    names = _nodes(kq.dequantize(w[0::2], SCALES, "q4_k"))
    assert names.count("KQuantMatrixDense") == 1


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16], ids=["f32", "f16"])
@pytest.mark.parametrize("codec", ["q4_k", "q8_0"])
def test_quantize_float_views(codec, dtype, state):
    wf = (mx.random.normal((2 * N, 2 * K), key=mx.random.key(3)) * 0.1).astype(dtype)
    turned = _dense(mx.swapaxes(wf[:N, :K], 0, 1))
    im2 = mx.abs(mx.random.normal((2 * K,), key=mx.random.key(4))) + 0.5
    mx.eval(wf, turned, im2)
    views = {
        "even_rows": lambda: wf[0::2, :K],
        "row_band": lambda: wf[N // 2 : N // 2 + N, :K],
        "even_cols": lambda: wf[:N, 0::2],
        "reversed": lambda: wf[::-1][:N, K:],
        "transposed": lambda: mx.swapaxes(turned, 0, 1),
    }
    for name, make in views.items():
        # q4_k weighs its error by an importance vector.
        for imatrix in (None, lambda: im2[0::2]) if codec == "q4_k" else (None,):
            ref_im = None if imatrix is None else _dense(imatrix())
            ref, _ = kq.quantize(_dense(make()), codec, ref_im)
            view = make()
            im = None if imatrix is None else imatrix()
            if state == "evaluated":
                mx.eval(view)
                if im is not None:
                    mx.eval(im)
            y, _ = kq.quantize(view, codec, im)
            mx.eval(y, ref)
            assert mx.array_equal(y, ref).item(), (codec, name, state)


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
def test_qmv_bias_views(dtype, state):
    # The bias-fused mat-vec runs q8_0 weights.
    w = _wire("q8_0", 2 * N, K, 1)[:N]
    mx.eval(w)
    x = _x(1, K, dtype)
    b2 = (mx.random.normal((2 * N,), key=mx.random.key(9)) * 4).astype(dtype)
    mx.eval(b2)
    for name, make in {
        "every_other": lambda: b2[0::2],
        "reversed": lambda: b2[::-1][:N],
        "band": lambda: b2[N // 2 : N // 2 + N],
    }.items():
        # numpy has no bfloat16, and float32 holds both dtypes exactly.
        dense = _dense(make().astype(mx.float32)).astype(dtype)
        ref = kq.quantized_matmul_qmv_bias(x, w, SCALES, dense, "q8_0")
        bias = make()
        if state == "evaluated":
            mx.eval(bias)
        y = kq.quantized_matmul_qmv_bias(x, w, SCALES, bias, "q8_0")
        mx.eval(y, ref)
        assert mx.array_equal(y, ref).item(), (name, state)
