"""quantized_matmul_residual: y = residual + gate * (x @ W^T).

On a NAX GPU the op runs one kernel for q4_k, q5_k, q6_k and q8_0 weights
on bfloat16 from 1024 rows: it walks K into f32 sums and stores residual +
gate * sum, rounded once. Everywhere else it builds the sum from
quantized_matmul and stock ops.

- R1: relative Frobenius error against the float64 sum, at most 6e-3 on
  bfloat16 and 1e-3 on float16.
- R2: against the model of the kernel, with the weight rounded to the
  activation dtype, exact sums and one rounding of y. The kernel sits at
  the error of that one rounding.
- Where the op has no kernel, y is the composition a caller writes, bit
  for bit, and every column of it is held against the float64 sum (COL).

Under KQUANT_FORCE_CPU the op runs the composition on the CPU stream, and
the tests that name a Metal kernel skip.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest
from qmm_probe import force_cpu
from test_operand_views import _dense, _strided
from test_weight_base import BASE_ALIGN, _shift

import mlx_kquant as kq

R1 = {mx.bfloat16: 6e-3, mx.float16: 1e-3}
# Gate of the composition on the CPU stream, as in test_cpu_neon.
CPU_REL = 5e-2
COL = {mx.bfloat16: 5e-2, mx.float16: 2e-2}
CPU_COL = 0.2
K = 768
# Rows from which the op runs the kernel (kResidMinRows).
MIN_ROWS = 1024
# Codecs with a kernel on bfloat16, the K tile of each kernel, and the K
# above which the kernel takes 64 rows a tile in place of 128 (the K
# ceiling of the codec's large-row tile in quantized_matmul).
KERNEL_BK = {"q4_k": 128, "q5_k": 128, "q6_k": 128, "q8_0": 64}
MAX_K = {"q4_k": 10240, "q5_k": 0, "q6_k": 0, "q8_0": 10240}
CODECS = list(KERNEL_BK)
GATES = ["none", "n", "batch"]
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

_W = {}


def _nax():
    return not force_cpu() and kq.nax_available()


def _fused(rows, codec="q4_k", dtype=mx.bfloat16, k=K):
    """Where the kernel runs under the default lever."""
    if not _nax() or codec not in KERNEL_BK or dtype != mx.bfloat16:
        return False
    return k % KERNEL_BK[codec] == 0 and rows >= MIN_ROWS


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _rounded(v, dtype):
    return _f64(mx.array(v.astype(np.float32)).astype(dtype))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _col(a, b):
    """The largest error of a column of a against b, over the size of a
    mean column of b. Rows are the leading axis."""
    mean = np.linalg.norm(b) / np.sqrt(b.shape[1]) + 1e-30
    return float(np.linalg.norm(a - b, axis=0).max() / mean)


def _weight(codec, n, k, seed=0):
    """A packed weight [n, k] and its float64 values."""
    key = (codec, n, k, seed)
    if key not in _W:
        wf = mx.random.normal((n, k), key=mx.random.key(1000 * seed + 17 * n + k)) * 0.1
        w, s = kq.quantize(wf, codec)
        mx.eval(w, s)
        _W[key] = (w, _f64(kq.dequantize(w, s, codec, dtype=mx.float32)))
    return _W[key]


def _x(shape, dtype, seed, scale=0.5):
    x = (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(dtype)
    mx.eval(x)
    return x


def _gate(kind, xshape, n, dtype, seed):
    """No gate, a gate [n], or one gate row per batch of x."""
    if kind == "none":
        return None
    shape = (n,) if kind == "n" else tuple(xshape[:-2]) + (1, n)
    g = (mx.random.uniform(shape=shape, key=mx.random.key(seed)) * 3 - 1.5).astype(
        dtype
    )
    mx.eval(g)
    return g


def _ref(x, wd, r, g, model=None):
    """The float64 sum, rows flattened. model=dtype rounds the weight to
    that dtype, as the kernel reads it."""
    out = mx.bfloat16 if x.dtype == mx.float32 else x.dtype
    n, k = wd.shape
    x64 = _f64(x.astype(out)).reshape(-1, k)
    if model is not None:
        wd = _rounded(wd, model)
    prod = (x64 @ wd.T).reshape(tuple(x.shape[:-1]) + (n,))
    if g is not None:
        prod = _f64(g.astype(out)) * prod
    return (_f64(r.astype(out)) + prod).reshape(-1, n)


def _composed(x, w, r, g, codec="q4_k"):
    """The form a caller writes without the op. Where the op has no kernel
    it builds this, bit for bit."""
    out = mx.bfloat16 if x.dtype == mx.float32 else x.dtype
    y = kq.quantized_matmul(x, w, mx.zeros((1,), dtype=mx.uint8), codec, True)
    if g is not None:
        y = g.astype(out) * y
    return r.astype(out) + y


def _check(x, w, wd, r, g, codec="q4_k"):
    """R1, and R2 on the kernel. Returns (R1 error, R2 error or None)."""
    n, k = wd.shape
    dtype = mx.bfloat16 if x.dtype == mx.float32 else x.dtype
    y = kq.quantized_matmul_residual(x, w, codec, r, gate=g)
    mx.eval(y)
    assert y.dtype == dtype
    assert y.shape == tuple(x.shape[:-1]) + (n,)
    got = _f64(y).reshape(-1, n)
    assert np.isfinite(got).all()
    ref = _ref(x, wd, r, g)
    err = _rel(got, ref)
    if not _fused(x.size // k, codec, dtype, k):
        comp = _composed(x, w, r, g, codec)
        mx.eval(comp)
        assert np.array_equal(_f64(comp).reshape(-1, n), got)
        assert err <= (R1[dtype] if _nax() else CPU_REL), err
        col = _col(got, ref)
        assert col <= (COL[dtype] if _nax() else CPU_COL), col
        return err, None
    assert err <= R1[dtype], (err, R1[dtype])
    assert _col(got, ref) <= COL[dtype]
    ref2 = _ref(x, wd, r, g, model=dtype)
    floor2 = _rel(_rounded(ref2, dtype), ref2)
    err2 = _rel(got, ref2)
    assert err2 <= 1.05 * floor2 + 2e-5, (err2, floor2)
    return err, err2


def _case(shape, n, gate="batch", dtype=mx.bfloat16, codec="q4_k", seed=0):
    k = shape[-1]
    w, wd = _weight(codec, n, k)
    x = _x(shape, dtype, 7 * shape[-2] + n + seed)
    r = _x(tuple(shape[:-1]) + (n,), dtype, 11 * shape[-2] + n + seed, scale=1.0)
    g = _gate(gate, shape, n, dtype, 13 * shape[-2] + n + seed)
    return _check(x, w, wd, r, g, codec)


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("m", [1, 17, 1023, 1024, 1025, 1039, 1100, 1151, 1152])
@pytest.mark.parametrize("n", [64, 96, 200])
@pytest.mark.parametrize("gate", ["none", "n"])
def test_resid_rows_and_columns(gate, n, m, codec):
    # Below MIN_ROWS the op is the composition. From there up: a last row
    # tile of 1 row, 15 rows, 76 rows, 127 rows and none, with N on and
    # off the column tile.
    _case([m, K], n, gate, codec=codec)


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("rows", [500, 1100])
@pytest.mark.parametrize("gate", GATES)
def test_resid_batch(gate, rows, batch, codec):
    # A gate row per batch runs the batch in the grid; a gate [N] or none
    # folds the batch into the rows. 2 x 500 rows passes the row bound.
    _case([batch, rows, K], 96, gate, codec=codec)


def test_resid_batch_dims_of_several_axes():
    _case([3, 2, 200, K], 128, "batch")
    _case([3, 2, 200, K], 128, "n")


@pytest.mark.parametrize("codec", CODECS)
def test_resid_inner_dim_above_the_large_tile_ceiling(codec):
    # q4_k and q8_0 take 64 rows a tile above K 10240, q5_k and q6_k do
    # not: both tiles hold the same sums.
    _case([2, 1100, 10496], 64, "batch", codec=codec)


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("m", [32768, 32769, 33792])
def test_resid_rows_above_32767(m, codec):
    # Row offsets pass a 16-bit index: every row is compared.
    _case([1, m, 256], 64, "batch", codec=codec)


def test_resid_batch_rows_above_32767():
    _case([2, 33000, 256], 64, "batch")


@pytest.mark.parametrize("codec", ["q3_k", "q4_0", "q5_k", "q8_0"])
def test_resid_codec_or_dtype_without_kernel_composes(codec):
    dtype = mx.float16 if codec in KERNEL_BK else mx.bfloat16
    err, err2 = _case([1100, K], 96, "batch", dtype=dtype, codec=codec)
    assert err2 is None


def test_resid_q8_0_inner_dim_off_the_k_tile():
    # q8_0 blocks are 32 weights: K 800 is no whole number of K tiles.
    err, err2 = _case([1100, 800], 96, "n", codec="q8_0")
    assert err2 is None
    _, err2 = _case([1100, 832], 96, "n", codec="q8_0")
    assert (err2 is not None) == _nax()


def test_resid_float32_x_runs_in_bfloat16():
    w, wd = _weight("q4_k", 96, K)
    x = _x([1100, K], mx.float32, 3)
    r = _x([1100, 96], mx.float32, 5)
    g = _gate("n", x.shape, 96, mx.float32, 9)
    y = kq.quantized_matmul_residual(x, w, "q4_k", r, gate=g)
    want = kq.quantized_matmul_residual(
        x.astype(mx.bfloat16),
        w,
        "q4_k",
        r.astype(mx.bfloat16),
        gate=g.astype(mx.bfloat16),
    )
    mx.eval(y, want)
    assert y.dtype == mx.bfloat16
    assert mx.array_equal(y, want).item()


def test_resid_one_dim_x():
    w, wd = _weight("q4_k", 96, K)
    x = _x([K], mx.bfloat16, 3)
    r = _x([96], mx.bfloat16, 5)
    g = _gate("n", x.shape, 96, mx.bfloat16, 9)
    _check(x, w, wd, r, g)


def test_resid_empty_inputs():
    w, _ = _weight("q4_k", 96, K)
    y = kq.quantized_matmul_residual(
        mx.zeros((0, K), dtype=mx.bfloat16),
        w,
        "q4_k",
        mx.zeros((0, 96), dtype=mx.bfloat16),
    )
    mx.eval(y)
    assert y.shape == (0, 96)


def _check_views(views, call, state):
    """call(view) against call(dense copy of the view): the same bits on the
    GPU, one rounding apart on the CPU stream (see test_matmul_lora)."""
    for name, make in views.items():
        want = call(_dense(make()))
        view = make()
        if state == "evaluated":
            mx.eval(view)
        got = call(view)
        mx.eval(got, want)
        if force_cpu():
            assert _rel(_f64(got), _f64(want)) <= R1[mx.bfloat16], (name, state)
        else:
            assert mx.array_equal(got, want).item(), (name, state)


def _weight_views(w):
    n, kb = w.shape
    other = mx.array(np.random.default_rng(3).integers(0, 255, (n, kb), dtype=np.uint8))
    rows = mx.stack([w, other], axis=1).reshape(2 * n, kb)
    cols = mx.concatenate([other, w], axis=1)
    rev = mx.array(np.ascontiguousarray(np.array(w)[::-1]))
    mx.eval(rows, cols, rev)
    return {
        "even_rows": lambda: rows[0::2],
        "col_band": lambda: cols[:, kb:],
        "reversed_rows": lambda: rev[::-1],
    }


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("m", [1023, 1100])
@pytest.mark.parametrize("state", ["lazy", "evaluated"])
def test_resid_operand_views(state, m, codec):
    """A view of any operand gives the bits of its dense copy, lazy or
    evaluated, on both sides of the row bound."""
    n = 96
    w, wd = _weight(codec, n, K)
    x = _x([m, K], mx.bfloat16, 47)
    r = _x([m, n], mx.bfloat16, 49)
    g = _x([n], mx.bfloat16, 51)

    def call(x_=x, w_=w, r_=r, g_=g):
        return kq.quantized_matmul_residual(x_, w_, codec, r_, gate=g_)

    _check_views(
        _strided(_x([2 * m, 2 * K], mx.bfloat16, 53)), lambda v: call(x_=v), state
    )
    _check_views(
        _strided(_x([2 * m, 2 * n], mx.bfloat16, 59)), lambda v: call(r_=v), state
    )
    _check_views(_weight_views(w), lambda v: call(w_=v), state)
    g2 = _x([2 * n], mx.bfloat16, 67)
    views = {"even": lambda: g2[0::2], "reversed": lambda: g2[::-1][:n]}
    _check_views(views, lambda v: call(g_=v), state)
    # A transposed x keeps the result right, not only equal to the copy.
    xv = _strided(_x([2 * m, 2 * K], mx.bfloat16, 53))["transposed"]()
    _check(xv, w, wd, r, g, codec)


@pytest.mark.parametrize("m", [1023, 1100])
@pytest.mark.parametrize("gate", GATES)
def test_resid_gradient_is_the_gradient_of_the_composition(gate, m):
    """mx.grad wrt x, residual and gate on both sides of the row bound."""
    n = 96
    w, _ = _weight("q4_k", n, K)
    x = _x([2, m // 2 + 1, K], mx.bfloat16, 71)
    r = _x([2, m // 2 + 1, n], mx.bfloat16, 73)
    g = _gate(gate, x.shape, n, mx.bfloat16, 79)
    cot = _x([2, m // 2 + 1, n], mx.float32, 83)

    def op(x_, r_, *g_):
        y = kq.quantized_matmul_residual(x_, w, "q4_k", r_, gate=g_[0] if g_ else None)
        return (y.astype(mx.float32) * cot).sum()

    def comp(x_, r_, *g_):
        y = _composed(x_, w, r_, g_[0] if g_ else None)
        return (y.astype(mx.float32) * cot).sum()

    args = (x, r) if g is None else (x, r, g)
    nums = tuple(range(len(args)))
    got = mx.grad(op, argnums=nums)(*args)
    want = mx.grad(comp, argnums=nums)(*args)
    mx.eval(got, want)
    for a, b in zip(got, want, strict=True):
        assert mx.array_equal(a, b).item()


def test_resid_jvp_and_vmap_over_residual_and_gate():
    n = 96
    w, _ = _weight("q4_k", n, K)
    for m in (1023, 1100):
        x = _x([m, K], mx.bfloat16, 91)
        r = _x([m, n], mx.bfloat16, 93)
        g = _x([n], mx.bfloat16, 95)
        t = _x([m, n], mx.bfloat16, 97)

        def f(r_, g_, x_=x):
            return kq.quantized_matmul_residual(x_, w, "q4_k", r_, gate=g_)

        def fc(r_, g_, x_=x):
            return _composed(x_, w, r_, g_)

        (got,), (want,) = (
            mx.jvp(f, [r, g], [t, mx.zeros_like(g)])[1],
            mx.jvp(fc, [r, g], [t, mx.zeros_like(g)])[1],
        )
        mx.eval(got, want)
        assert mx.array_equal(got, want).item()
        rs = mx.stack([r, r * 0.5])
        got = mx.vmap(lambda r_, g_=g: f(r_, g_), in_axes=0)(rs)
        want = mx.stack([f(r, g), f(r * 0.5, g)])
        mx.eval(got, want)
        assert _rel(_f64(got), _f64(want)) <= R1[mx.bfloat16]


def test_resid_weight_is_frozen():
    n = 96
    w, _ = _weight("q4_k", n, K)
    for m in (1023, 1100):
        x = _x([m, K], mx.bfloat16, 107)
        r = _x([m, n], mx.bfloat16, 109)

        def loss(w_, x_=x, r_=r):
            return (
                kq.quantized_matmul_residual(x_, w_, "q4_k", r_)
                .astype(mx.float32)
                .sum()
            )

        op = "quantized_matmul_residual" if _fused(m) else "quantized_matmul"
        text = rf"\[mlx_kquant\] {op} vjp: no gradient wrt the quantized weight"
        with pytest.raises(ValueError, match=text):
            mx.eval(mx.grad(loss)(w))


def test_resid_compiles_and_repeats():
    n = 96
    w, _ = _weight("q5_k", n, K)
    x = _x([1100, K], mx.bfloat16, 111)
    r = _x([1100, n], mx.bfloat16, 113)
    g = _x([n], mx.bfloat16, 117)
    f = mx.compile(
        lambda x_, r_, g_: kq.quantized_matmul_residual(x_, w, "q5_k", r_, gate=g_)
    )
    want = kq.quantized_matmul_residual(x, w, "q5_k", r, gate=g)
    for _ in range(2):
        got = f(x, r, g)
        mx.eval(got, want)
        assert mx.array_equal(got, want).item()


@pytest.mark.parametrize("codec", CODECS)
def test_resid_weight_off_its_byte_boundary_raises(codec):
    if force_cpu() or mx.default_device() != mx.gpu:
        pytest.skip("the start check guards the Metal kernels")
    n = 96
    w, _ = _weight(codec, n, K)
    align = BASE_ALIGN.get(codec, 2)
    for m in (1023, 1100):
        x = _x([m, K], mx.bfloat16, 191)
        r = _x([m, n], mx.bfloat16, 193)
        op = "quantized_matmul_residual" if _fused(m, codec) else "quantized_matmul"
        text = (
            rf"\[mlx_kquant\] {op}: the {codec} weight starts 1 byte past a "
            rf"{align}-byte boundary"
        )
        with pytest.raises(ValueError, match=text):
            kq.quantized_matmul_residual(x, _shift(w, 1), codec, r)
        y = kq.quantized_matmul_residual(x, _shift(w, 1, evaluate=False), codec, r)
        with pytest.raises(ValueError, match=text):
            mx.eval(y)
        good = kq.quantized_matmul_residual(x, _shift(w, 16), codec, r)
        ref = kq.quantized_matmul_residual(x, w, codec, r)
        mx.eval(good, ref)
        assert mx.array_equal(good, ref).item()


def test_resid_argument_errors():
    n = 96
    w, _ = _weight("q4_k", n, K)
    x = _x([2, 4, K], mx.bfloat16, 89)
    r = _x([2, 4, n], mx.bfloat16, 90)

    def bad(match, x_=x, w_=w, r_=r, codec="q4_k", **kw):
        with pytest.raises(ValueError, match=r"quantized_matmul_residual\] .*" + match):
            kq.quantized_matmul_residual(x_, w_, codec, r_, **kw)

    bad("Unknown kquant_type", codec="q9_z")
    bad("2-D uint8", w_=w.astype(mx.float32))
    bad("whole number", w_=w[:, :-1])
    bad("x last dim", x_=x[..., :-1])
    bad("floating x", x_=x.astype(mx.int32))
    bad("residual must be", r_=r[..., :-1])
    bad("residual must be", r_=r[:1])
    bad("residual must be", r_=r.reshape(8, n))
    bad("residual must be", r_=r.astype(mx.int32))
    for shape in ((n + 1,), (1, n), (2, 4, n), (2, 1, 1, n), (1, 1, n), (2, 1, n - 1)):
        bad("gate must be", gate=mx.ones(shape))
    bad("gate must be", gate=mx.ones((n,), dtype=mx.int32))
    for shape in ((n,), (2, 1, n)):
        kq.quantized_matmul_residual(x, w, "q4_k", r, gate=mx.ones(shape))


def test_resid_size_limits():
    """K or N of 2^25 and a row count of 2^31 raise by the name of the op.
    The arrays are lazy and stay so."""
    cols, rows = 1 << 25, 1 << 31

    def lazy(*shape, dtype=mx.bfloat16):
        return mx.zeros(shape, dtype=dtype)

    def w8(n, k):
        return lazy(n, k // 32 * 34, dtype=mx.uint8)

    cases = {
        "K": (lazy(cols), w8(1, cols), lazy(1)),
        "N": (lazy(4, 32), w8(cols, 32), lazy(4, cols)),
    }
    for name, (x, w, r) in cases.items():
        text = (
            rf"quantized_matmul_residual\] K and N must each be below {cols} "
            rf".*{name} {cols}"
        )
        with pytest.raises(ValueError, match=text):
            kq.quantized_matmul_residual(x, w, "q8_0", r)
    x = lazy(2, rows // 2, 32)
    text = rf"quantized_matmul_residual\] x must hold fewer than {rows} \(2\^31\) rows"
    with pytest.raises(ValueError, match=text):
        kq.quantized_matmul_residual(x, w8(64, 32), "q8_0", lazy(2, rows // 2, 64))


def _name(codec, n, k, gated):
    max_k = MAX_K[codec]
    bm = 64 if max_k and k > max_k else 128
    al = "true" if n % 64 == 0 else "false"
    gt = "true" if gated else "false"
    return (
        f"kquant_{codec}_qmm_t_nax_resid_bfloat16_t_bm{bm}_bn64_bk{KERNEL_BK[codec]}"
        f"_wm2_wn2_alN_{al}_gate_{gt}"
    )


def test_resid_kernels_in_metallib():
    if force_cpu():
        pytest.skip("Metal kernel names only")
    names = kq.metallib_kernels()
    if not names:
        pytest.skip("no metallib loaded")
    got = {n for n in names if "_qmm_t_nax_resid_" in n}
    want = {
        _name(c, n, k, gt)
        for c in CODECS
        for n in (64, 96)
        for k in (256, 10496)
        for gt in (True, False)
    }
    assert got == want, sorted(got ^ want)


_PROBE = r"""
import json, os, sys
spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["tests_dir"])
import mlx.core as mx
import mlx_kquant as kq
import test_matmul_resid as t

out = []
for i, c in enumerate(spec["calls"]):
    codec = c.get("codec", "q4_k")
    shape = c["shape"]
    n = c["n"]
    w, wd = t._weight(codec, n, shape[-1])
    x = t._x(shape, mx.bfloat16, 7 * shape[-2] + n)
    r = t._x(shape[:-1] + [n], mx.bfloat16, 11 * shape[-2] + n, scale=1.0)
    g = t._gate(c.get("gate", "batch"), shape, n, mx.bfloat16, 13 * shape[-2] + n)
    os.environ.update(c.get("env", {}))
    print(f"[probe] {i}", file=sys.stderr, flush=True)
    res = {"error": None}
    try:
        y = kq.quantized_matmul_residual(x, w, codec, r, gate=g)
        mx.eval(y)
        got = t._f64(y).reshape(-1, n)
        ref = t._ref(x, wd, r, g)
        res["rel"] = t._rel(got, ref)
        res["col"] = t._col(got, ref)
        ref2 = t._ref(x, wd, r, g, model=mx.bfloat16)
        res["rel2"] = t._rel(got, ref2)
        res["floor2"] = t._rel(t._rounded(ref2, mx.bfloat16), ref2)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    out.append(res)
print("RESULT " + json.dumps(out))
"""


def _probe(calls, env=None):
    """Run calls in a fresh interpreter that keeps no KQ_ variable but
    those of env and of each call. Per call: resid (the residual kernels
    that KQ_QMM_LOG named), plain (the other kernels), error and errors."""
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
            cur.update(resid=[], plain=[])
        elif cur is not None and ln.startswith("[kq qmm] "):
            name = ln.split()[2]
            cur["resid" if "_resid_" in name else "plain"].append(name)
    return results


def _expect(calls, results, kernel):
    for c, r, k in zip(calls, results, kernel, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] <= R1[mx.bfloat16], (c, r["rel"])
        assert r["col"] <= COL[mx.bfloat16], (c, r["col"])
        if k:
            want = _name(
                c.get("codec", "q4_k"), c["n"], c["shape"][-1], c.get("gate") != "none"
            )
            assert r["resid"] == [want], (c, r["resid"])
            assert r["plain"] == [], (c, r["plain"])
            assert r["rel2"] <= 1.05 * r["floor2"] + 2e-5, (c, r["rel2"], r["floor2"])
        else:
            assert r["resid"] == [], (c, r["resid"])
            assert len(r["plain"]) == 1, (c, r["plain"])


def test_resid_known_answer_of_every_kernel():
    """Each kernel of the metallib runs once, named by KQ_QMM_LOG, against
    the float64 sum and the one-rounding model: every codec, both row
    tiles, N on and off the column tile, with and without a gate."""
    if not _nax():
        pytest.skip("the kernel needs a NAX GPU")
    calls = []
    for codec in CODECS:
        # K above the ceiling takes the BM 64 kernels, where a codec has one.
        for k in (256, 10496) if MAX_K[codec] else (256,):
            for n in (64, 96):
                for gate in ("batch", "none"):
                    calls.append(
                        {"codec": codec, "shape": [1, 1100, k], "n": n, "gate": gate}
                    )
    res = _probe(calls)
    _expect(calls, res, [True] * len(calls))
    names = {r["resid"][0] for r in res}
    assert len(names) == len(calls)
    assert names == {n for n in kq.metallib_kernels() if "_qmm_t_nax_resid_" in n}


def test_resid_kernel_that_runs_and_levers():
    """KQ_QMM_LOG names the kernel of each call: from 1024 rows by default,
    at every row count under KQ_QMM_RESID=2, never under 0. Batch dims fold
    into rows for the row bound."""
    if not _nax():
        pytest.skip("the kernel needs a NAX GPU")
    calls = [
        ({"shape": [1024, K], "n": 128, "gate": "n"}, True),
        ({"shape": [1023, K], "n": 128, "gate": "n"}, False),
        ({"shape": [2, 512, K], "n": 128}, True),
        ({"shape": [2, 511, K], "n": 128}, False),
        ({"shape": [1100, 800], "n": 128, "codec": "q8_0", "gate": "n"}, False),
        ({"shape": [17, K], "n": 96, "gate": "n", "env": {"KQ_QMM_RESID": "2"}}, True),
        (
            {"shape": [1100, K], "n": 96, "gate": "n", "env": {"KQ_QMM_RESID": "0"}},
            False,
        ),
        (
            {"shape": [1100, K], "n": 96, "gate": "n", "env": {"KQ_QMM_RESID": "1"}},
            True,
        ),
    ]
    res = _probe([c for c, _ in calls], env={"KQ_TUNE_LIVE": "1"})
    _expect([c for c, _ in calls], res, [k for _, k in calls])
    # Without KQ_TUNE_LIVE the lever is read once, at the first call.
    calls = [
        (
            {"shape": [1100, K], "n": 96, "gate": "n", "env": {"KQ_QMM_RESID": "0"}},
            False,
        ),
        (
            {"shape": [1100, K], "n": 96, "gate": "n", "env": {"KQ_QMM_RESID": "1"}},
            False,
        ),
    ]
    res = _probe([c for c, _ in calls])
    _expect([c for c, _ in calls], res, [k for _, k in calls])
