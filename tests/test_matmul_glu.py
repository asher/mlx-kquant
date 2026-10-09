"""quantized_matmul_glu against a float64 reference.

The reference is act(x @ Wg^T) * (x @ Wu^T) in float64 on the CPU, with the
exact kq.dequantize weights and x as the op receives it. The fused kernel
runs the activation and the product in float32 and rounds the activation
once and the product once, so it should be no less accurate than the
composition, which rounds each matmul to the activation dtype first.

The op runs the kernel from MIN_ROWS rows, for the codecs of GLU_CODECS
(q8_0 on bfloat16 only) and an inner dim that is a whole number of the
codec's K tiles, and builds the composition otherwise. In-process cases
cover both sides of the row bound, each codec, partial row tiles, both
parities of the row-tile count and counts above 32767. Inputs come
dense, as a lazy transposed view, as a strided view and with one and two
batch dims. The kernel below MIN_ROWS and the probe tiles run in a fresh
interpreter, since the levers are read once per process.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_kquant._ext import _probe_kernels
from qmm_probe import force_cpu

import mlx_kquant as kq

R1 = {mx.bfloat16: 6e-3, mx.float16: 1e-3}
DTYPES = [mx.bfloat16, mx.float16]
# Gate of the composition on the CPU stream, as in test_cpu_neon.
CPU_REL = 5e-2
K = 768
# Rows from which the op runs the fused kernel (kGluMinRows).
MIN_ROWS = 1024
# Codecs with a fused kernel, and the K tile each runs.
GLU_CODECS = {"q4_k": 128, "q5_k": 128, "q6_k": 128, "q8_0": 64}
OTHER_CODECS = ["q5_k", "q6_k", "q8_0"]
# Codecs whose kernel runs on bfloat16 only. On float16 the op builds the
# composition, which measured faster there.
BF16_ONLY = {"q8_0"}
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

_W = {}


def _nax():
    return not force_cpu() and kq.nax_available()


def _fused(rows=MIN_ROWS, codec="q4_k", k=K, dtype=mx.bfloat16):
    tile = GLU_CODECS.get(codec, 0)
    if codec in BF16_ONLY and dtype != mx.bfloat16:
        tile = 0
    return _nax() and rows >= MIN_ROWS and tile > 0 and k % tile == 0


def _weights(codec, n, k, seed=0):
    """Two packed weights [n, k] of one codec and their float64 values."""
    key = (codec, n, k, seed)
    if key not in _W:
        out = []
        for i in range(2):
            wf = (
                mx.random.normal(
                    (n, k), key=mx.random.key(1000 * seed + 17 * n + k + i)
                )
                * 0.1
            )
            w, s = kq.quantize(wf, codec)
            mx.eval(w, s)
            wd = np.array(kq.dequantize(w, s, codec, dtype=mx.float32)).astype(
                np.float64
            )
            out.append((w, wd))
        _W[key] = out
    return _W[key]


def _x(shape, dtype, seed, scale=0.5):
    x = (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(dtype)
    mx.eval(x)
    return x


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _act(g, activation):
    if activation == "gelu_tanh":
        return 0.5 * g * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (g + 0.044715 * g**3)))
    with np.errstate(over="ignore"):
        return g / (1.0 + np.exp(-g))


def _ref(x, wg, wu, activation="silu"):
    x64 = _f64(x)
    return _act(x64 @ wg.T, activation) * (x64 @ wu.T)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _composed(x, wg, wu, codec, activation="silu"):
    """The form a caller writes without the op. Where the op has no fused
    kernel it builds this, bit for bit."""
    s = mx.zeros((1,), dtype=mx.uint8)
    g = kq.quantized_matmul(x, wg, s, codec, True)
    u = kq.quantized_matmul(x, wu, s, codec, True)
    act = nn.gelu_approx if activation == "gelu_tanh" else nn.silu
    return act(g) * u


def _check(x, n, dtype, activation="silu", codec="q4_k", k=K):
    (wg, wg64), (wu, wu64) = _weights(codec, n, k)
    y = kq.quantized_matmul_glu(x, wg, wu, codec, activation)
    mx.eval(y)
    assert y.dtype == dtype
    assert y.shape == tuple(x.shape[:-1]) + (n,)
    ref = _ref(x, wg64, wu64, activation)
    got = _f64(y)
    assert np.isfinite(got).all()
    err = _rel(got, ref)
    comp = _composed(x, wg, wu, codec, activation)
    mx.eval(comp)
    err_c = _rel(_f64(comp), ref)
    if not _fused(x.size // k, codec, k, dtype):
        # The op is the composition here. On the CPU a few rows run the int8
        # GEMV path, which is good to about 1e-2 per matmul.
        assert mx.array_equal(y, comp).item()
        assert err <= (R1[dtype] if _nax() else CPU_REL), err
        return err, err_c
    assert err <= R1[dtype], (err, R1[dtype])
    # No less accurate than the composition, which runs the same NAX tiles.
    assert err <= 1.1 * err_c + 1e-7, (err, err_c)
    return err, err_c


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize(
    "m", [1, 17, 64, 129, 577, 1023, 1024, 1025, 1039, 1040, 1100, 1151, 1152]
)
@pytest.mark.parametrize("n", [64, 96, 200])
def test_glu_rows_and_columns(m, n, dtype):
    # Below MIN_ROWS the op is the composition. From there up: a last row
    # tile of 1 row, 15 and 16 rows (a partial and a whole simdgroup
    # fragment), 76 rows, 127 rows and none.
    _check(_x([m, K], dtype, m * 7 + n), n, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("m", [4127, 4128, 4160, 4224])
def test_glu_image_rows(m, dtype):
    # 4127, 4128 and 4160 leave a partial last row tile of 128, 4224 none.
    _check(_x([m, K], dtype, m), 128, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("m", [32768, 32769, 33792])
def test_glu_rows_above_32767(m, dtype):
    (wg, wg64), (wu, wu64) = _weights("q4_k", 96, 256)
    x = _x([m, 256], dtype, m)
    y = kq.quantized_matmul_glu(x, wg, wu, "q4_k")
    mx.eval(y)
    ref = _ref(x, wg64, wu64)
    got = _f64(y)
    assert _rel(got, ref) <= R1[dtype]
    # The first and last row tiles, where a 16-bit row span would wrap.
    for rows in (slice(0, 256), slice(m - 300, m)):
        assert _rel(got[rows], ref[rows]) <= R1[dtype]


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_glu_activations(activation, dtype):
    _check(_x([1100, K], dtype, 5, scale=2.0), 96, dtype, activation)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("layout", ["batch1", "batch2", "transposed", "strided"])
def test_glu_layouts(layout, dtype):
    m, n = 1030, 96
    if layout == "batch1":
        x = _x([1, m, K], dtype, 3)
    elif layout == "batch2":
        x = _x([2, m, K], dtype, 4)
    elif layout == "transposed":
        # A lazy transposed view: the op sees an unevaluated swapaxes.
        x = mx.swapaxes(_x([K, m], dtype, 5), 0, 1)
    else:
        x = _x([m, 2 * K], dtype, 6)[:, ::2]
    _check(x, n, dtype)


def test_glu_float32_x_runs_in_bfloat16():
    (wg, wg64), (wu, wu64) = _weights("q4_k", 96, K)
    x = _x([1030, K], mx.float32, 9)
    y = kq.quantized_matmul_glu(x, wg, wu, "q4_k")
    mx.eval(y)
    assert y.dtype == mx.bfloat16
    ref = _ref(x.astype(mx.bfloat16), wg64, wu64)
    assert _rel(_f64(y), ref) <= R1[mx.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
def test_glu_row_slices_of_one_weight(dtype):
    """Gate and up as the two row bands of one fused weight."""
    n = 96
    (wg, wg64), (wu, wu64) = _weights("q4_k", n, K)
    both = mx.concatenate([wg, wu], axis=0)
    mx.eval(both)
    x = _x([1100, K], dtype, 11)
    y = kq.quantized_matmul_glu(x, both[:n], both[n:], "q4_k")
    y2 = kq.quantized_matmul_glu(x, wg, wu, "q4_k")
    mx.eval(y, y2)
    assert _rel(_f64(y), _ref(x, wg64, wu64)) <= R1[dtype]
    assert mx.array_equal(y, y2).item()


def _settle():
    """Active memory once the buffers of finished evals are released."""
    mx.synchronize()
    last, same = -1, 0
    for _ in range(400):
        mx.clear_cache()
        cur = mx.get_active_memory()
        same = same + 1 if cur == last else 0
        last = cur
        if same >= 4:
            break
        time.sleep(0.005)
    return last


def _peak_outputs(build, out_bytes):
    """MLX peak of one evaluated call over the memory live before it, in
    output buffers. The call runs once first, so pipelines and constants
    are in place."""
    y = build()
    mx.eval(y)
    del y
    base = _settle()
    mx.reset_peak_memory()
    y = build()
    mx.eval(y)
    peak = mx.get_peak_memory() - base
    del y
    return peak / out_bytes


def test_glu_row_bands_are_read_in_place():
    """Two row bands of one weight on the fused path: the call allocates
    its output and no copy of a band. The two bands are half an output."""
    if not _fused():
        pytest.skip("the fused kernel needs a NAX GPU")
    n, k, m = 2048, 1024, 1100
    (wg, _), (wu, _) = _weights("q4_k", n, k)
    both = mx.concatenate([wg, wu], axis=0)
    mx.eval(both)
    x = _x([m, k], mx.bfloat16, 41)
    out_bytes = m * n * 2
    assert 2 * wg.nbytes > 0.5 * out_bytes
    for lazy in (True, False):
        bands = (both[:n], both[n:])
        if not lazy:
            mx.eval(*bands)

        def build(bands=bands, lazy=lazy):
            g, u = (both[:n], both[n:]) if lazy else bands
            return kq.quantized_matmul_glu(x, g, u, "q4_k")

        peak = _peak_outputs(build, out_bytes)
        assert 0.99 <= peak <= 1.05, (lazy, peak)
    y = kq.quantized_matmul_glu(x, both[:n], both[n:], "q4_k")
    ref = kq.quantized_matmul_glu(x, wg, wu, "q4_k")
    mx.eval(y, ref)
    assert mx.array_equal(y, ref).item()


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_glu_composition_is_the_stock_form(activation, dtype):
    """A codec with no fused kernel: the op has the bits and the peak of
    the stock activation on one matmul times the other. That is two
    outputs with nn.silu, which writes into its input, and three with
    nn.gelu_approx, which does not."""
    n, k, m = 2048, 1024, 1100
    (wg, _), (wu, _) = _weights("q3_k", n, k)
    x = _x([m, k], dtype, 47)
    y = kq.quantized_matmul_glu(x, wg, wu, "q3_k", activation)
    comp = _composed(x, wg, wu, "q3_k", activation)
    mx.eval(y, comp)
    assert mx.array_equal(y, comp).item()
    out_bytes = m * n * 2
    op = _peak_outputs(
        lambda: kq.quantized_matmul_glu(x, wg, wu, "q3_k", activation), out_bytes
    )
    stock = _peak_outputs(lambda: _composed(x, wg, wu, "q3_k", activation), out_bytes)
    assert abs(op - stock) <= 0.05, (op, stock)
    assert op <= (3.05 if activation == "gelu_tanh" else 2.05), op


def _node_kinds(*outputs):
    """Kinds of the primitives in the graph of the outputs. A compiled
    function is one node, named after the ops it holds."""
    import tempfile

    with tempfile.NamedTemporaryFile("r", suffix=".dot") as f:
        mx.export_to_dot(f.name, *outputs)
        text = f.read()
    names = re.findall(r'label ="([A-Za-z0-9_]+)"', text)
    return sorted("Compiled" if n.startswith("Compiled") else n for n in names)


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_glu_unfused_ops_follow_the_compile_switch(activation):
    """The unfused path builds its activation as mlx.nn does in either
    state of mx's compile switch: one compiled function while the switch is
    on and separate ops while it is off, whatever the state was at an
    earlier call. 64 rows build the unfused ops on every device."""
    (wg, _), (wu, _) = _weights("q4_k", 96, K)
    x = _x([64, K], mx.bfloat16, 71)
    try:
        for on in (True, False, True, False):
            (mx.enable_compile if on else mx.disable_compile)()
            y = kq.quantized_matmul_glu(x, wg, wu, "q4_k", activation)
            ref = _composed(x, wg, wu, "q4_k", activation)
            kinds = _node_kinds(y)
            assert kinds == _node_kinds(ref), (on, kinds)
            assert ("Compiled" in kinds) == on, (on, kinds)
            mx.eval(y, ref)
            assert mx.array_equal(y, ref).item(), on
    finally:
        mx.enable_compile()


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_glu_unfused_ops_run_on_the_stream_of_the_call(activation):
    """An unfused call on a stream that is not the default one has the bits
    of the stock form built on that stream. For silu the two devices round
    the activation differently at this input, so the bits tell where it ran.
    For gelu_tanh they agree here, and that case checks the value alone."""
    if force_cpu() or mx.default_device() != mx.gpu:
        pytest.skip("the CPU is the default device")
    (wg, _), (wu, _) = _weights("q3_k", 96, K)
    x = _x([64, K], mx.bfloat16, 61)
    y = kq.quantized_matmul_glu(x, wg, wu, "q3_k", activation, stream=mx.cpu)
    with mx.stream(mx.cpu):
        ref = _composed(x, wg, wu, "q3_k", activation)
    mx.eval(y, ref)
    assert mx.array_equal(y, ref).item()


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("state", ["lazy", "evaluated"])
def test_glu_strided_weight_views(state, dtype):
    """Weights whose rows are not dense: every other row of one weight, and
    a column band of a wider one. The fused path copies them at eval."""
    n = 96
    (wg, wg64), (wu, wu64) = _weights("q4_k", n, K)
    rows = mx.stack([wg, wu], axis=1).reshape(2 * n, -1)
    cols = mx.concatenate([wg, wu], axis=1)
    mx.eval(rows, cols)
    kb = wg.shape[1]
    x = _x([1100, K], dtype, 43)
    ref = kq.quantized_matmul_glu(x, wg, wu, "q4_k")
    for make in (
        lambda: (rows[0::2], rows[1::2]),
        lambda: (cols[:, :kb], cols[:, kb:]),
    ):
        g, u = make()
        if state == "evaluated":
            mx.eval(g, u)
        y = kq.quantized_matmul_glu(x, g, u, "q4_k")
        mx.eval(y, ref)
        assert mx.array_equal(y, ref).item()
    assert _rel(_f64(ref), _ref(x, wg64, wu64)) <= R1[dtype]


def test_glu_float16_store_saturates():
    """Products past the half range store 65504 with the sign kept."""
    n = 64
    (wg, wg64), (wu, wu64) = _weights("q4_k", n, K)
    x = _x([1088, K], mx.float16, 13, scale=200.0)
    ref = _ref(x, wg64, wu64)
    assert (np.abs(ref) > 65504.0).mean() > 0.05
    # The activation alone stays in range, so the clipped product is exact.
    assert np.abs(_act(_f64(x) @ wg64.T, "silu")).max() < 65504.0
    y = kq.quantized_matmul_glu(x, wg, wu, "q4_k")
    mx.eval(y)
    got = _f64(y)
    if not _fused():
        # The composition overflows to inf where the fused store clamps.
        ok = np.isfinite(got)
        assert _rel(got[ok], ref[ok]) <= 2e-2
        return
    assert np.isfinite(got).all()
    clipped = np.clip(ref, -65504.0, 65504.0)
    assert _rel(got, clipped) <= R1[mx.float16]
    over = np.abs(ref) > 66000.0
    assert (np.abs(got[over]) == 65504.0).all()
    assert (np.sign(got[over]) == np.sign(ref[over])).all()


def test_glu_bfloat16_result_above_the_float16_range():
    """The float16 store saturates at 65504. bfloat16 holds the products
    past it, and the op returns them."""
    n = 64
    (wg, wg64), (wu, wu64) = _weights("q4_k", n, K)
    x = _x([1088, K], mx.bfloat16, 13, scale=200.0)
    ref = _ref(x, wg64, wu64)
    assert (np.abs(ref) > 65504.0).mean() > 0.05
    y = kq.quantized_matmul_glu(x, wg, wu, "q4_k")
    mx.eval(y)
    got = _f64(y)
    assert np.isfinite(got).all()
    assert _rel(got, ref) <= (R1[mx.bfloat16] if _nax() else CPU_REL)
    over = np.abs(ref) > 70000.0
    assert over.any() and (np.abs(got[over]) > 65536.0).all()


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("codec", ["q4_k", "q6_k"])
def test_glu_nan_and_inf_pass(codec, activation, dtype):
    """A NaN or an infinity in x gives NaN and inf where the unfused ops
    give them. The float16 store saturates finite values only."""
    n, m = 96, 1100
    (wg, _), (wu, _) = _weights(codec, n, K)
    xn = np.array(_x([m, K], dtype, 31).astype(mx.float32))
    # Rows of the first, a middle and the partial last row tile.
    xn[3, 5] = np.nan
    xn[200, 700] = np.inf
    xn[511, 0] = -np.inf
    xn[640, 100] = np.inf
    xn[640, 101] = -np.inf
    xn[1090, 767] = -np.inf
    xn[1099, 300] = np.nan
    x = mx.array(xn).astype(dtype)
    mx.eval(x)
    y = kq.quantized_matmul_glu(x, wg, wu, codec, activation)
    comp = _composed(x, wg, wu, codec, activation)
    mx.eval(y, comp)
    got = np.array(y.astype(mx.float32))
    ref = np.array(comp.astype(mx.float32))
    assert np.isnan(ref).any() and np.isinf(ref).any()
    assert (np.isnan(got) == np.isnan(ref)).all()
    assert (np.isinf(got) == np.isinf(ref)).all()
    inf = np.isinf(ref)
    assert (got[inf] == ref[inf]).all()
    ok = np.isfinite(ref)
    assert _rel(got[ok].astype(np.float64), ref[ok].astype(np.float64)) <= 2 * R1[dtype]


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("m", [1023, 1024, 1039, 1100, 1152])
@pytest.mark.parametrize("n", [64, 200])
@pytest.mark.parametrize("codec", OTHER_CODECS)
def test_glu_codecs_rows_and_columns(codec, n, m, dtype):
    _check(_x([m, K], dtype, m * 7 + n), n, dtype, codec=codec)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("codec", OTHER_CODECS)
def test_glu_codecs_image_rows(codec, dtype):
    _check(_x([4128, K], dtype, 4128), 128, dtype, codec=codec)


@pytest.mark.parametrize("codec", OTHER_CODECS)
def test_glu_codecs_gelu(codec):
    _check(
        _x([1100, K], mx.bfloat16, 5, scale=2.0), 96, mx.bfloat16, "gelu_tanh", codec
    )


@pytest.mark.parametrize("codec", OTHER_CODECS)
def test_glu_codecs_rows_above_32767(codec):
    m = 32769
    (wg, wg64), (wu, wu64) = _weights(codec, 96, 256)
    x = _x([m, 256], mx.bfloat16, m)
    y = kq.quantized_matmul_glu(x, wg, wu, codec)
    mx.eval(y)
    ref = _ref(x, wg64, wu64)
    got = _f64(y)
    for rows in (slice(0, m), slice(0, 256), slice(m - 300, m)):
        assert _rel(got[rows], ref[rows]) <= R1[mx.bfloat16]


@pytest.mark.parametrize("k", [832, 800])
def test_glu_q8_0_inner_dim_rule(k):
    """q8_0 has a K tile of 64: 832 is 13 tiles and runs the kernel, 800 is
    not a whole number of tiles and builds the composition."""
    assert _fused(1100, "q8_0", k) == (_nax() and k == 832)
    _check(_x([1100, k], mx.bfloat16, k), 96, mx.bfloat16, codec="q8_0", k=k)


def test_glu_q8_0_float16_composes():
    """q8_0 on float16 is the composition at every row count."""
    assert not _fused(4128, "q8_0", K, mx.float16)
    _check(_x([4128, K], mx.float16, 23), 96, mx.float16, codec="q8_0")


@pytest.mark.parametrize("codec", ["q3_k", "q4_0", "q2_0", "pq2_0"])
def test_glu_codec_without_kernel_composes(codec):
    (wg, wg64), (wu, wu64) = _weights(codec, 96, K)
    x = _x([1100, K], mx.bfloat16, 15)
    y = kq.quantized_matmul_glu(x, wg, wu, codec)
    mx.eval(y)
    assert _rel(_f64(y), _ref(x, wg64, wu64)) <= R1[mx.bfloat16]
    comp = _composed(x, wg, wu, codec)
    assert mx.array_equal(y, comp).item()


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("m", [1023, 1100])
@pytest.mark.parametrize(
    "dtype", [mx.bfloat16, mx.float16, mx.float32], ids=["bf16", "f16", "f32"]
)
def test_glu_gradient_is_the_gradient_of_the_composition(dtype, m, activation):
    """mx.grad wrt x on both sides of the row bound. The fused kernel has
    the vjp of the unfused ops, bit for bit on float16 and bfloat16 x.

    float32 x is cast to bfloat16 before the fused node, so the node sums
    the gate term and the up term of the gradient in bfloat16 and the
    unfused ops sum them in float32. The two differ by the rounding of that
    one sum. bfloat16 keeps 8 significant bits, so one rounding moves a
    value by at most 2^-8 = 3.9e-3 of itself and by about 1.7e-3 in the
    root mean square, and the difference measures 1.8e-3. The bound is
    6e-3, the bound on a bfloat16 result against float64."""
    n = 96
    (wg, _), (wu, _) = _weights("q4_k", n, K)
    x = _x([m, K], dtype, 53)
    c = mx.random.normal((m, n), key=mx.random.key(59))
    mx.eval(c)

    def loss(f):
        return lambda t: (f(t).astype(mx.float32) * c).sum()

    g_op = mx.grad(
        loss(lambda t: kq.quantized_matmul_glu(t, wg, wu, "q4_k", activation))
    )(x)
    g_ref = mx.grad(loss(lambda t: _composed(t, wg, wu, "q4_k", activation)))(x)
    mx.eval(g_op, g_ref)
    assert g_op.shape == x.shape and g_op.dtype == x.dtype
    ref = _f64(g_ref)
    assert np.isfinite(ref).all() and np.abs(ref).max() > 0
    if dtype == mx.float32 and _fused(m):
        rel = _rel(_f64(g_op), ref)
        assert rel <= R1[mx.bfloat16], rel
    else:
        assert mx.array_equal(g_op, g_ref).item()


def test_glu_jvp_and_vmap_do_not_depend_on_the_row_count():
    """quantized_matmul has no jvp and no vmap, so both raise for the
    composition. The fused kernel raises the same way."""
    (wg, _), (wu, _) = _weights("q4_k", 96, K)
    seen = {"jvp": [], "vmap": []}
    for m in (1023, 1100):
        x = _x([m, K], mx.bfloat16, 61)

        def f(t):
            return kq.quantized_matmul_glu(t, wg, wu, "q4_k")

        with pytest.raises(ValueError) as e:
            mx.eval(mx.jvp(f, [x], [x])[1])
        seen["jvp"].append(str(e.value))
        with pytest.raises(ValueError) as e:
            mx.eval(mx.vmap(f)(x))
        seen["vmap"].append(type(e.value))
    assert seen["jvp"][0] == seen["jvp"][1], seen["jvp"]
    assert "KQuantMatmul" in seen["jvp"][0]


def test_glu_compiled_function_with_both_activations():
    """Two calls that differ in the activation alone stay two nodes when a
    compiled function holds both."""
    (wg, _), (wu, _) = _weights("q4_k", 96, K)
    x = _x([1100, K], mx.bfloat16, 67)

    def both(t):
        a = kq.quantized_matmul_glu(t, wg, wu, "q4_k", "silu")
        b = kq.quantized_matmul_glu(t, wg, wu, "q4_k", "gelu_tanh")
        # mx.compile merges equivalent nodes but never an output of the
        # function, so the two calls feed the outputs. The casts are exact.
        return a.astype(mx.float32), b.astype(mx.float32)

    a, b = mx.compile(both)(x)
    ra, rb = both(x)
    mx.eval(a, b, ra, rb)
    assert mx.array_equal(a, ra).item()
    assert mx.array_equal(b, rb).item()
    assert not mx.array_equal(ra, rb).item()


def test_glu_compiles_and_repeats():
    (wg, _), (wu, _) = _weights("q4_k", 96, K)
    x = _x([1100, K], mx.bfloat16, 17)
    f = mx.compile(lambda t: kq.quantized_matmul_glu(t, wg, wu, "q4_k"))
    a = f(x)
    b = f(x)
    c = kq.quantized_matmul_glu(x, wg, wu, "q4_k")
    mx.eval(a, b, c)
    assert mx.array_equal(a, c).item()
    assert mx.array_equal(b, c).item()


def test_glu_argument_errors():
    (wg, _), (wu, _) = _weights("q4_k", 96, K)
    x = _x([4, K], mx.bfloat16, 19)
    with pytest.raises(ValueError, match="activation"):
        kq.quantized_matmul_glu(x, wg, wu, "q4_k", "relu")
    with pytest.raises(ValueError, match="one shape"):
        kq.quantized_matmul_glu(x, wg, wu[:64], "q4_k")
    with pytest.raises(ValueError, match="inner dim"):
        kq.quantized_matmul_glu(x[:, :256], wg, wu, "q4_k")
    with pytest.raises(ValueError, match="Unknown kquant_type"):
        kq.quantized_matmul_glu(x, wg, wu, "q4_z")
    with pytest.raises(ValueError, match="uint8"):
        kq.quantized_matmul_glu(x, wg.astype(mx.float32), wu, "q4_k")


def test_glu_kernels_in_metallib():
    if force_cpu():
        pytest.skip("Metal kernel names only")
    names = kq.metallib_kernels()
    if not names:
        pytest.skip("no metallib loaded")
    glu = {n for n in names if "_qmm_t_nax_glu_" in n}
    # The tile the op runs for each codec.
    want = {
        f"kquant_{c}_qmm_t_nax_glu_{t}_bm128_bn64_bk{bk}w_wm2_wn2_alN_{al}"
        for c, bk in GLU_CODECS.items()
        for t in ("float16_t", "bfloat16_t")
        for al in ("true", "false")
        if t == "bfloat16_t" or c not in BF16_ONLY
    }
    assert want <= glu, sorted(want - glu)
    # A release build carries those and no other GLU kernel. A build with
    # KQ_PROBE_KERNELS=ON carries the probe tiles too, and the module that
    # reads their levers reports the option.
    if _probe_kernels():
        assert glu > want
    else:
        assert glu == want, sorted(glu - want)
    # Every GLU kernel is a tile of a known form of one of those codecs: K
    # tile 64 or 128, with d (two unpadded Ws buffers) or w (two K walks).
    pat = re.compile(
        r"^kquant_(" + "|".join(GLU_CODECS) + r")_qmm_t_nax_glu_"
        r"(bfloat16_t|float16_t)_bm(64|128)_bn64"
        r"_bk(64|128|128d|64w|128w)_wm2_wn2_alN_(true|false)$"
    )
    bad = sorted(n for n in glu if not pat.match(n))
    assert not bad, bad[:5]


def _built_tiles(codec="q4_k"):
    """(BM, K tile) of every GLU kernel of a codec in the metallib."""
    tiles = set()
    head = f"kquant_{codec}_qmm_t_nax_glu_bfloat16_t_"
    for n in kq.metallib_kernels():
        if n.startswith(head) and n.endswith("alN_true"):
            f = n.split("_")
            bm = next(p for p in f if p.startswith("bm"))[2:]
            bk = next(p for p in f if p.startswith("bk"))[2:]
            tiles.add((int(bm), bk))
    return sorted(tiles)


_CHILD = r"""
import hashlib, json, os, sys
import numpy as np
spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["tests_dir"])
import mlx.core as mx
import mlx_kquant as kq
import test_matmul_glu as t

out = []
for m, n, dt in spec["cells"]:
    dtype = getattr(mx, dt)
    (wg, wg64), (wu, wu64) = t._weights(spec["codec"], n, t.K)
    x = t._x([m, t.K], dtype, m * 7 + n)
    row = {"m": m, "n": n, "dtype": dt}
    for bm, bk in spec["tiles"]:
        os.environ["KQ_GLU_BM"] = str(bm)
        os.environ["KQ_GLU_BK"] = bk.rstrip("dw")
        os.environ["KQ_GLU_DUAL128"] = "1" if bk.endswith("d") else "0"
        os.environ["KQ_GLU_WALKS"] = "1" if bk.endswith("w") else "0"
        y = kq.quantized_matmul_glu(x, wg, wu, spec["codec"])
        mx.eval(y)
        got = t._f64(y)
        row[f"{bm},{bk}"] = {
            "rel": t._rel(got, t._ref(x, wg64, wu64)),
            "hash": hashlib.sha256(np.array(y.astype(mx.float32))).hexdigest(),
        }
    out.append(row)
print("RESULT " + json.dumps(out))
"""


@pytest.mark.parametrize("codec", list(GLU_CODECS))
def test_glu_kernel_below_min_rows_and_tiles_agree(codec):
    """The kernel at any row count, and one set of bits per kernel form."""
    if not _nax():
        pytest.skip("the fused kernel needs a NAX GPU")
    # Every tile in the metallib. One K walk: K tile 64 on two Ws buffers,
    # 128 on one, 128d on two unpadded ones. Two K walks (64w, 128w) round
    # the gate activation to the output dtype before the product, so those
    # bits form their own group.
    tiles = _built_tiles(codec)
    assert (128, f"{GLU_CODECS[codec]}w") in tiles
    rows = (1, 16, 17, 63, 65, 127, 129, 300, 449, 577, 4128)
    cells = [
        (m, n, dt)
        for m in (rows if codec == "q4_k" else rows[::3])
        for n in (64, 96, 200)
        for dt in ("bfloat16", "float16")
    ]
    spec = {"tests_dir": TESTS_DIR, "tiles": tiles, "cells": cells, "codec": codec}
    env = dict(os.environ, KQ_TUNE_LIVE="1", KQ_QMM_LOG="1", KQ_GLU_MIN_ROWS="1")
    p = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(spec)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    rows = json.loads(line[len("RESULT ") :])
    logged = {
        ln.split()[2]
        for ln in p.stderr.splitlines()
        if ln.startswith("[kq qmm] ") and "_glu_" in ln
    }
    for bm, bk in tiles:
        assert any(f"_bm{bm}_bn64_bk{bk}_" in n for n in logged), (
            bm,
            bk,
            sorted(logged),
        )
    for row in rows:
        tol = R1[getattr(mx, row["dtype"])]
        hashes = {False: set(), True: set()}
        for bm, bk in tiles:
            r = row[f"{bm},{bk}"]
            assert r["rel"] <= tol, (row["m"], row["n"], bm, bk, r["rel"])
            hashes[bk.endswith("w")].add(r["hash"])
        for walks, group in hashes.items():
            # A release build has no one-walk tile, so that group is empty.
            assert len(group) <= 1, (row["m"], row["n"], row["dtype"], walks)


_PROBE = r"""
import json, os, sys
spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["tests_dir"])
import mlx.core as mx
import mlx_kquant as kq
import test_matmul_glu as t

out = []
for i, c in enumerate(spec["calls"]):
    dtype = getattr(mx, c["dtype"])
    act = c.get("activation", "silu")
    k = c.get("k", t.K)
    (wg, wg64), (wu, wu64) = t._weights(c["codec"], c["n"], k)
    x = t._x([c["m"], k], dtype, 7 * c["m"] + c["n"])
    os.environ.update(c.get("env", {}))
    print(f"[probe] {i}", file=sys.stderr, flush=True)
    res = {"error": None}
    try:
        y = kq.quantized_matmul_glu(x, wg, wu, c["codec"], act)
        mx.eval(y)
        res["rel"] = t._rel(t._f64(y), t._ref(x, wg64, wu64, act))
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    out.append(res)
print("RESULT " + json.dumps(out))
"""


def _glu_probe(calls, env=None):
    """Run GLU calls in a fresh interpreter that keeps no KQ_ variable but
    those of env and of each call. Per call: glu (the GLU kernels that
    KQ_QMM_LOG named), plain (the other kernels), stderr (its other
    lines), error and rel."""
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    child_env.update({"KQ_QMM_LOG": "1", **(env or {})})
    spec = {"tests_dir": TESTS_DIR, "calls": calls}
    p = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(spec)],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert p.returncode == 0, p.stderr[-3000:]
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    results = json.loads(line[len("RESULT ") :])
    cur = None
    for ln in p.stderr.splitlines():
        if ln.startswith("[probe] "):
            cur = results[int(ln.split()[1])]
            cur.update(glu=[], plain=[], stderr=[])
        elif cur is None:
            continue
        elif ln.startswith("[kq qmm] "):
            name = ln.split()[2]
            cur["glu" if "_glu_" in name else "plain"].append(name)
        else:
            cur["stderr"].append(ln)
    return results


def test_glu_min_rows_lever_starts_at_one():
    """KQ_GLU_MIN_ROWS moves the row bound. A value below 1 keeps the
    bound of 1024 rows, with one warning per value."""
    if not _nax():
        pytest.skip("the fused kernel needs a NAX GPU")
    call = {"codec": "q4_k", "m": 600, "n": 96, "dtype": "bfloat16"}
    values = [
        ("600", True, False),
        ("601", False, False),
        ("1", True, False),
        ("0", False, True),
        ("0", False, False),
        ("-4", False, True),
        ("false", False, True),
    ]
    calls = [dict(call, env={"KQ_GLU_MIN_ROWS": v}) for v, _, _ in values]
    res = _glu_probe(calls, env={"KQ_TUNE_LIVE": "1"})
    for (v, fused, warns), r in zip(values, res, strict=True):
        assert r["error"] is None, (v, r["error"])
        assert r["rel"] <= R1[mx.bfloat16], (v, r["rel"])
        assert len(r["glu"]) == (1 if fused else 0), (v, r["glu"])
        assert len(r["plain"]) == (0 if fused else 2), (v, r["plain"])
        warned = [ln for ln in r["stderr"] if f"KQ_GLU_MIN_ROWS={v} is not" in ln]
        assert len(warned) == warns, (v, r["stderr"])
        if warns:
            assert "at least 1" in warned[0]


def test_glu_tile_levers_belong_to_a_probe_build():
    """A release build reads no tile lever. A probe build raises for a tile
    the codec has no kernel for, with the levers and the kernel name."""
    if not _nax():
        pytest.skip("the fused kernel needs a NAX GPU")
    call = {"codec": "q5_k", "m": 1100, "n": 96, "dtype": "bfloat16"}
    env = {"KQ_GLU_BM": "64", "KQ_GLU_WALKS": "0"}
    if not _probe_kernels():
        # A release build reads none of the four.
        env.update({"KQ_GLU_BK": "64", "KQ_GLU_DUAL128": "1"})
    (r,) = _glu_probe([call], env=env)
    head = "kquant_q5_k_qmm_t_nax_glu_bfloat16_t_"
    if _probe_kernels():
        # q5_k has no one-walk kernel in any build.
        assert r["error"] is not None and r["error"].startswith("ValueError")
        assert "KQ_GLU_BM=64" in r["error"] and "KQ_GLU_WALKS=0" in r["error"]
        assert head + "bm64_bn64_bk128_wm2_wn2_alN_false" in r["error"]
        assert "not instantiated" in r["error"]
    else:
        assert r["error"] is None, r["error"]
        assert r["glu"] == [head + "bm128_bn64_bk128w_wm2_wn2_alN_false"]
        assert r["rel"] <= R1[mx.bfloat16]


def _glu_name(codec, dtype, n):
    """The kernel the op runs for a codec, an activation dtype and N."""
    aligned = "true" if n % 64 == 0 else "false"
    return (
        f"kquant_{codec}_qmm_t_nax_glu_{dtype}_t_bm128_bn64"
        f"_bk{GLU_CODECS[codec]}w_wm2_wn2_alN_{aligned}"
    )


@pytest.mark.parametrize("codec", list(GLU_CODECS))
def test_glu_kernel_that_runs(codec):
    """KQ_QMM_LOG names the kernel of each call. From 1024 rows a codec
    with a kernel for the dtype runs it, once, and no matmul tile: for both
    dtypes, N on and off the column tile and both activations. At 1023
    rows, and for q8_0 on float16, the two matmuls run and no GLU kernel."""
    if not _nax():
        pytest.skip("the fused kernel needs a NAX GPU")
    calls = [
        {"codec": codec, "m": m, "n": n, "dtype": dt, "activation": act}
        for dt in ("bfloat16", "float16")
        for m, n, act in (
            (1100, 64, "silu"),
            (1100, 96, "silu"),
            (1100, 64, "gelu_tanh"),
            (1100, 96, "gelu_tanh"),
            (1024, 96, "silu"),
            (1023, 96, "silu"),
        )
    ]
    res = _glu_probe(calls)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] <= R1[getattr(mx, c["dtype"])], (c, r["rel"])
        has_kernel = c["dtype"] == "bfloat16" or codec not in BF16_ONLY
        if c["m"] >= MIN_ROWS and has_kernel:
            assert r["glu"] == [_glu_name(codec, c["dtype"], c["n"])], (c, r["glu"])
            assert r["plain"] == [], (c, r["plain"])
        else:
            assert r["glu"] == [], (c, r["glu"])
            assert len(r["plain"]) == 2, (c, r["plain"])


def test_glu_lever_off_runs_the_unfused_ops():
    """KQ_GLU=0: no GLU kernel at any row count, for any codec."""
    if not _nax():
        pytest.skip("the fused kernel needs a NAX GPU")
    calls = [
        {"codec": codec, "m": 1100, "n": 96, "dtype": "bfloat16"}
        for codec in GLU_CODECS
    ]
    off = _glu_probe(calls, env={"KQ_GLU": "0"})
    on = _glu_probe(calls, env={"KQ_GLU": "1"})
    for c, r0, r1 in zip(calls, off, on, strict=True):
        assert r0["error"] is None and r1["error"] is None
        assert r0["glu"] == [] and len(r0["plain"]) == 2, (c, r0)
        assert r1["glu"] == [_glu_name(c["codec"], "bfloat16", 96)], (c, r1)
        assert r0["rel"] <= R1[mx.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("codec", ["q4_k", "q6_k"])
def test_glu_gelu_below_saturation(codec, dtype):
    """gelu_tanh on gate sums of about 1.4, where the tanh is far from 1
    and the cubic term of its argument shows in the result."""
    _check(_x([1100, K], dtype, 71, scale=0.5), 96, dtype, "gelu_tanh", codec)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("n", [1, 17, 63])
def test_glu_columns_below_one_tile(n, dtype):
    """N below the column tile of 64: one partial tile."""
    _check(_x([1100, K], dtype, 73 + n), n, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
def test_glu_columns_above_32767(dtype):
    """N past 32767 with a partial last row tile and N off the column
    tile, where a 16-bit column span would wrap. The reference covers the
    first and the last columns."""
    n, k, m = 32833, 256, 1100
    (wg, wg64), (wu, wu64) = _weights("q4_k", n, k)
    x = _x([m, k], dtype, 77)
    y = kq.quantized_matmul_glu(x, wg, wu, "q4_k")
    mx.eval(y)
    assert y.shape == (m, n)
    x64 = _f64(x)
    for cols in (slice(0, 256), slice(n - 300, n)):
        ref = _act(x64 @ wg64[cols].T, "silu") * (x64 @ wu64[cols].T)
        got = _f64(y[:, cols])
        # All rows, and the partial last row tile alone.
        for rows in (slice(0, m), slice(1024, m)):
            assert _rel(got[rows], ref[rows]) <= R1[dtype], (cols, rows)
    del y


def test_glu_float16_activation_saturates_beside_a_small_up():
    """A float16 gate activation past the half range is stored as 65504, so
    the product with a small up sum is finite: 65504 times the up sum. The
    gate weight reads the first two K blocks and the up weight the third,
    so x sets the size of each sum."""
    m, n = 1100, 96
    if not _fused(m, dtype=mx.float16):
        pytest.skip("the unfused ops return inf here")
    kg = 512
    wf = mx.random.normal((2, n, K), key=mx.random.key(83)) * 0.1
    cols = mx.arange(K)
    wg, _ = kq.quantize(mx.where(cols < kg, wf[0], 0.0), "q4_k")
    wu, _ = kq.quantize(mx.where(cols < kg, 0.0, wf[1]), "q4_k")
    s = mx.zeros((1,), dtype=mx.uint8)
    wg64, wu64 = (_f64(kq.dequantize(w, s, "q4_k", dtype=mx.float32)) for w in (wg, wu))
    big = mx.random.uniform(-3e4, 3e4, (m, K), key=mx.random.key(85))
    small = mx.random.normal((m, K), key=mx.random.key(87)) * 0.1
    x = mx.where(cols < kg, big, small).astype(mx.float16)
    mx.eval(x, wg, wu)
    g = _f64(x) @ wg64.T
    u = _f64(x) @ wu64.T
    over = (g > 66000.0) & (np.abs(u) < 0.9) & (np.abs(u) > 1e-3)
    assert over.mean() > 0.02
    y = kq.quantized_matmul_glu(x, wg, wu, "q4_k")
    mx.eval(y)
    got = _f64(y)
    assert np.isfinite(got[over]).all()
    assert _rel(got[over], 65504.0 * u[over]) <= 2 * R1[mx.float16]


def test_glu_zero_size_inputs():
    """No rows, no output columns and an inner dim of 0 give an array of
    the composed shape."""
    (wg, _), (wu, _) = _weights("q4_k", 96, K)
    for shape in ([0, K], [2, 0, K], [0, 3, K]):
        y = kq.quantized_matmul_glu(mx.zeros(shape, dtype=mx.bfloat16), wg, wu, "q4_k")
        mx.eval(y)
        assert y.shape == (*shape[:-1], 96) and y.dtype == mx.bfloat16
    x = _x([1100, K], mx.bfloat16, 89)
    y = kq.quantized_matmul_glu(x, wg[:0], wu[:0], "q4_k")
    mx.eval(y)
    assert y.shape == (1100, 0) and y.dtype == mx.bfloat16
    w0 = mx.zeros((96, 0), dtype=mx.uint8)
    y = kq.quantized_matmul_glu(mx.zeros((1100, 0), dtype=mx.float16), w0, w0, "q4_k")
    mx.eval(y)
    assert y.shape == (1100, 96) and y.dtype == mx.float16
    assert not y.astype(mx.float32).any().item()
