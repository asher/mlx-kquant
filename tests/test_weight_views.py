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
CODECS = ["q4_k", "q6_k", "q8_0", "q4_0"]
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
    mx.eval(wide, stack, turned)
    return {
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
