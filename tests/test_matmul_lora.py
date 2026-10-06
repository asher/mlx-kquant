"""quantized_matmul_lora: y = x @ W^T + (strength * (x @ A)) @ B.

On a NAX GPU the op runs one kernel for q4_k, q5_k, q6_k and q8_0 weights
on bfloat16 from 1024 rows: stock ops compute z = strength * (x @ A), and
the kernel walks the rank for z and B and then K for the weight in the same
f32 sums. Everywhere else it builds the sum from quantized_matmul and stock
ops. By itself the op gives the kernel only the calls of its shape rule
(K up to 4096, N of at least K and, below 4096 rows, N of at least 8192).
The weights here are narrow, so the tests of this module run under
KQ_QMM_LORA=2, which runs the kernel wherever it can run, and the rule is
tested in fresh interpreters.

- R1: relative Frobenius error against the float64 sum, at most 6e-3 on
  bfloat16 and 1e-3 on float16.
- R2: against the model of the kernel, with the weight and z rounded to
  the activation dtype, exact sums and one rounding of y. The kernel sits
  at the error of that one rounding.
- Where the op has no kernel, y is the composition a caller writes, bit
  for bit.

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

import mlx_kquant as kq

R1 = {mx.bfloat16: 6e-3, mx.float16: 1e-3}
DTYPES = [mx.bfloat16, mx.float16]
# Gate of the composition on the CPU stream, as in test_cpu_neon.
CPU_REL = 5e-2
K = 768
# Rows from which the op runs the kernel (kLoraMinRows).
MIN_ROWS = 1024
# Column tile: a band starts on a multiple of it. A band ends on the column
# tile of a simdgroup, or at N.
BN = 64
SN = 32
# Largest rank of a band, and most bands, the kernel runs.
MAX_RANK = 256
MAX_BANDS = 16
# The shape rule: by itself the op gives the kernel a weight with K up to
# RULE_K and N of at least K, from RULE_ROWS rows, and below that row count
# only with N of at least RULE_WIDE_N.
RULE_K = 4096
RULE_ROWS = 4096
RULE_WIDE_N = 8192
# Codecs with a kernel on bfloat16, and the K tile of each kernel. K is a
# whole number of K tiles where the kernel runs.
KERNEL_BK = {"q4_k": 128, "q5_k": 128, "q6_k": 128, "q8_0": 64}
CODECS = list(KERNEL_BK)
KERNELS = {(c, mx.bfloat16) for c in CODECS}
# Every kernel, and float16 once: float16 has no kernel.
PAIRS = [(c, mx.bfloat16) for c in CODECS] + [("q4_k", mx.float16)]
PAIR_IDS = [f"{c}-bf16" for c in CODECS] + ["q4_k-f16"]
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

_W = {}


@pytest.fixture(autouse=True, scope="module")
def _kernel_wherever_it_can_run():
    """KQ_QMM_LORA=2 for the calls of this module. The op reads the lever
    at its first call in the process and keeps it, and no other module
    calls the op."""
    old = os.environ.get("KQ_QMM_LORA")
    os.environ["KQ_QMM_LORA"] = "2"
    yield
    if old is None:
        del os.environ["KQ_QMM_LORA"]
    else:
        os.environ["KQ_QMM_LORA"] = old


def _nax():
    return not force_cpu() and kq.nax_available()


def _shape_rule(k, n, rows):
    """The calls on which the op picks the kernel by itself."""
    return k <= RULE_K and n >= k and (rows >= RULE_ROWS or n >= RULE_WIDE_N)


def _fused(rows, rank, n, codec="q4_k", dtype=mx.bfloat16, bands=None, k=K):
    """Where the kernel can run: the dispatch rule of the op under
    KQ_QMM_LORA=2."""
    if not _nax() or (codec, dtype) not in KERNELS:
        return False
    if k % KERNEL_BK[codec]:
        return False
    if rows < MIN_ROWS or rank == 0:
        return False
    if bands is None:
        return rank <= MAX_RANK
    return len(bands) <= MAX_BANDS and all(
        cs % BN == 0 and (ce % SN == 0 or ce == n) and rb <= MAX_RANK
        for cs, ce, _, rb in bands
    )


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


def _ab(k, n, r, dtype, seed=0):
    """A [k, r] and B [r, n], sized so that the low-rank term is near the
    size of the product."""
    a = mx.random.normal((k, r), key=mx.random.key(31 * seed + r)) / np.sqrt(k)
    b = mx.random.normal((r, n), key=mx.random.key(37 * seed + r + n)) * 0.3
    a, b = a.astype(dtype), b.astype(dtype)
    mx.eval(a, b)
    return a, b


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _rounded(v, dtype):
    return _f64(mx.array(v.astype(np.float32)).astype(dtype))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _delta(z, b, n, bands):
    if bands is None:
        return z @ b
    d = np.zeros((z.shape[0], n))
    for cs, ce, z0, rb in bands:
        d[:, cs:ce] += z[:, z0 : z0 + rb] @ b[z0 : z0 + rb, cs:ce]
    return d


def _ref(x, wd, a, b, strength=None, bands=None, model=None):
    """The float64 sum, rows flattened. model=dtype rounds the weight and z
    to that dtype, as the kernel reads them."""
    out = mx.bfloat16 if x.dtype == mx.float32 else x.dtype
    x64 = _f64(x.astype(out)).reshape(-1, x.shape[-1])
    z = x64 @ _f64(a.astype(out))
    if model is not None:
        z = _rounded(z, model)
        wd = _rounded(wd, model)
    if strength is not None:
        z = z * _f64(strength.astype(out)).reshape(-1)
        if model is not None:
            z = _rounded(z, model)
    with np.errstate(invalid="ignore", over="ignore"):
        return x64 @ wd.T + _delta(z, _f64(b.astype(out)), wd.shape[0], bands)


def _composed(x, w, a, b, codec="q4_k", strength=None, bands=None):
    """The form a caller writes without the op. Where the op has no kernel
    it builds this, bit for bit."""
    out = mx.bfloat16 if x.dtype == mx.float32 else x.dtype
    base = kq.quantized_matmul(x, w, mx.zeros((1,), dtype=mx.uint8), codec, True)
    z = x.astype(out) @ a.astype(out)
    if strength is not None:
        z = z * strength.astype(out)
    b = b.astype(out)
    if bands is None:
        return mx.addmm(base, z, b)
    parts, col = [], 0
    for cs, ce, z0, rb in bands:
        if cs > col:
            parts.append(base[..., col:cs])
        y = base[..., cs:ce]
        if rb > 0:
            y = mx.addmm(y, z[..., z0 : z0 + rb], b[z0 : z0 + rb, cs:ce])
        parts.append(y)
        col = ce
    if col < base.shape[-1]:
        parts.append(base[..., col:])
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)


def _check(x, w, wd, a, b, codec="q4_k", strength=None, bands=None, cols=None):
    """R1 and, on the kernel, R2. cols limits the comparison to a column
    range. Returns (R1 error, R2 error or None)."""
    n, k = wd.shape
    dtype = mx.bfloat16 if x.dtype == mx.float32 else x.dtype
    y = kq.quantized_matmul_lora(x, w, codec, a, b, strength=strength, bands=bands)
    mx.eval(y)
    assert y.dtype == dtype
    assert y.shape == tuple(x.shape[:-1]) + (n,)
    sel = slice(*cols) if cols else slice(None)
    got = _f64(y).reshape(-1, n)[:, sel]
    assert np.isfinite(got).all()
    ref = _ref(x, wd, a, b, strength, bands)[:, sel]
    err = _rel(got, ref)
    if not _fused(x.size // k, a.shape[1], n, codec, dtype, bands, k):
        comp = _composed(x, w, a, b, codec, strength, bands)
        mx.eval(comp)
        assert np.array_equal(_f64(comp).reshape(-1, n)[:, sel], got)
        assert err <= (R1[dtype] if _nax() else CPU_REL), err
        return err, None
    assert err <= R1[dtype], (err, R1[dtype])
    ref2 = _ref(x, wd, a, b, strength, bands, model=dtype)[:, sel]
    floor2 = _rel(_rounded(ref2, dtype), ref2)
    err2 = _rel(got, ref2)
    assert err2 <= 1.05 * floor2 + 2e-5, (err2, floor2)
    return err, err2


def _case(m, n, r, dtype=mx.bfloat16, codec="q4_k", k=K, seed=0, **kw):
    w, wd = _weight(codec, n, k)
    a, b = _ab(k, n, r, dtype, seed)
    return _check(_x([m, k], dtype, 7 * m + n + r), w, wd, a, b, codec, **kw)


@pytest.mark.parametrize(("codec", "dtype"), PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize(
    "m", [1, 17, 577, 1023, 1024, 1025, 1039, 1040, 1100, 1151, 1152]
)
@pytest.mark.parametrize("n", [64, 96, 200])
def test_lora_rows_and_columns(m, n, codec, dtype):
    # Below MIN_ROWS the op is the composition. From there up: a last row
    # tile of 1 row, 15 and 16 rows, 76 rows, 127 rows and none, with N on
    # and off the column tile.
    _case(m, n, 16, dtype, codec)


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("r", [1, 4, 16, 31, 32, 33, 64, 80, 96, 130, 192, 256, 257])
@pytest.mark.parametrize("n", [128, 200])
def test_lora_ranks(r, n, codec):
    # A rank below, at and above the step of 32, no multiple of 64, several
    # steps, the largest rank of the kernel and one above it.
    _case(1100, n, r, codec=codec)


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("m", [4127, 4128, 4608])
def test_lora_image_rows(m, codec):
    n = 192
    bands = [(0, 64, 0, 64), (64, 128, 64, 64), (128, 192, 128, 64)]
    _case(m, n, 192, codec=codec, bands=bands)


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("m", [32768, 32769, 33792])
def test_lora_rows_above_32767(m, codec):
    # Row offsets pass a 16-bit index: every row is compared.
    _case(m, 64, 16, codec=codec, k=256)


@pytest.mark.parametrize("k", [832, 800, 96, 32])
def test_lora_q8_0_inner_dim_off_the_k_tile(k):
    # q8_0 holds K in blocks of 32 and its kernel walks K in tiles of 64: a
    # K off the tile is the composition.
    err, err2 = _case(1100, 96, 16, codec="q8_0", k=k)
    assert (err2 is not None) == (_nax() and k % 64 == 0)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("batch", [1, 2])
def test_lora_batch(batch, dtype):
    n, r, m = 96, 16, 1100
    w, wd = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, dtype)
    x = _x([batch, m, K], dtype, 11 + batch)
    _check(x, w, wd, a, b)
    y = kq.quantized_matmul_lora(x, w, "q4_k", a, b)
    flat = kq.quantized_matmul_lora(x.reshape(batch * m, K), w, "q4_k", a, b)
    mx.eval(y, flat)
    if _fused(batch * m, r, n, dtype=dtype) == _fused(m, r, n, dtype=dtype):
        assert mx.array_equal(y.reshape(batch * m, n), flat).item()


def test_lora_one_dim_x():
    n, r = 96, 16
    w, wd = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    x = _x([K], mx.bfloat16, 5)
    y = kq.quantized_matmul_lora(x, w, "q4_k", a, b)
    two = kq.quantized_matmul_lora(x.reshape(1, K), w, "q4_k", a, b)
    mx.eval(y, two)
    assert y.shape == (n,)
    assert mx.array_equal(y, two[0]).item()


def test_lora_float32_x_runs_in_bfloat16():
    n, r = 96, 16
    w, wd = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, mx.float32)
    x = _x([1100, K], mx.float32, 13)
    _check(x, w, wd, a, b)
    y = kq.quantized_matmul_lora(x, w, "q4_k", a, b)
    y16 = kq.quantized_matmul_lora(
        x.astype(mx.bfloat16),
        w,
        "q4_k",
        a.astype(mx.bfloat16),
        b.astype(mx.bfloat16),
    )
    mx.eval(y, y16)
    assert y.dtype == mx.bfloat16
    assert mx.array_equal(y, y16).item()


BANDS = {
    # Three bands over every column, as a fused q, k and v weight.
    "three": (192, 48, [(0, 64, 0, 16), (64, 128, 16, 16), (128, 192, 32, 16)]),
    # Columns before, between and after the bands take no low-rank term.
    "gaps": (320, 40, [(64, 128, 0, 24), (192, 256, 24, 16)]),
    # A band that ends on the column tile of a simdgroup, and a band that
    # ends at N off the tile.
    "end_on_half_tile": (300, 40, [(0, 96, 0, 24), (128, 300, 24, 16)]),
    # A band of the last 44 columns, after columns in no band.
    "end_at_n": (300, 40, [(64, 128, 0, 24), (256, 300, 24, 16)]),
    # A band that ends inside the columns of a simdgroup: the composition.
    "end_in_tile": (300, 40, [(0, 100, 0, 24), (128, 250, 24, 16)]),
    # A band of rank 0, and a band that takes no rank from the first columns
    # of z.
    "rank_zero": (192, 32, [(0, 64, 5, 0), (64, 192, 12, 20)]),
    # Two bands that share z columns, and z ranges out of column order.
    "shared_z": (192, 40, [(0, 64, 20, 20), (64, 128, 0, 40), (128, 192, 10, 7)]),
    # Ranks that are no multiple of the step of 32, over several steps.
    "odd_ranks": (192, 133, [(0, 64, 0, 33), (64, 192, 33, 100)]),
    # A start off the column tile: the op builds the composition.
    "start_off_tile": (192, 32, [(0, 32, 0, 16), (32, 192, 16, 16)]),
    # A band above the largest rank of the kernel: the composition.
    "rank_above_max": (128, 300, [(0, 64, 0, 257), (64, 128, 257, 43)]),
    # More bands than the kernel looks through: the composition.
    "seventeen": (17 * 64, 34, [(64 * i, 64 * i + 64, 2 * i, 2) for i in range(17)]),
    "sixteen": (16 * 64, 32, [(64 * i, 64 * i + 64, 2 * i, 2) for i in range(16)]),
}


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("m", [1023, 1100])
@pytest.mark.parametrize("name", list(BANDS))
def test_lora_bands(name, m, codec):
    n, r, bands = BANDS[name]
    _case(m, n, r, codec=codec, bands=bands)


@pytest.mark.parametrize("end", [96, 100])
@pytest.mark.parametrize("codec", CODECS)
def test_lora_band_columns_are_isolated(codec, end):
    """A band reads its own ranks of z and its own block of B, whatever the
    rest holds: an infinite strength on the ranks of the other band and
    infinite values in B outside the two blocks leave the columns of the
    first band finite and right. The first band ends on the column tile of
    a simdgroup (the kernel) and inside one (the composition)."""
    n, r, m = 192, 32, 1100
    bands = [(0, end, 0, 20), (128, 192, 20, 12)]
    w, wd = _weight(codec, n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    mask = np.ones((r, n), dtype=bool)
    for cs, ce, z0, rb in bands:
        mask[z0 : z0 + rb, cs:ce] = False
    b = mx.where(mx.array(mask), mx.array(float("inf")), b).astype(mx.bfloat16)
    strength = mx.array([1.0] * 20 + [float("inf")] * 12)
    x = _x([m, K], mx.bfloat16, 23)
    # The columns of the first band, then the columns outside every band.
    kw = dict(codec=codec, strength=strength, bands=bands)
    _check(x, w, wd, a, b, cols=(0, end), **kw)
    _check(x, w, wd, a, b, cols=(end, 128), **kw)


@pytest.mark.parametrize("end", [96, 100])
@pytest.mark.parametrize("m", [1023, 1100])
@pytest.mark.parametrize("codec", CODECS)
def test_lora_z_that_is_not_finite_stays_in_its_band(codec, m, end):
    """A column outside every band takes no term of z. With an infinite
    strength every column of a band is not finite, and every other column
    is finite and right: the columns before the first band, the columns
    after it in its column tile, a column tile between two bands, and the
    columns after the last band. The first band ends on the column tile of
    a simdgroup (the kernel) and inside one (the composition)."""
    n, r = 448, 32
    bands = [(64, end, 0, 20), (192, 256, 20, 12)]
    w, wd = _weight(codec, n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    strength = mx.array(float("inf"))
    x = _x([m, K], mx.bfloat16, 113)
    kw = dict(codec=codec, strength=strength, bands=bands)
    for cols in ((0, 64), (end, 128), (128, 192), (256, 448)):
        _check(x, w, wd, a, b, cols=cols, **kw)
    y = _f64(
        kq.quantized_matmul_lora(x, w, codec, a, b, strength=strength, bands=bands)
    )
    for cs, ce, _, _ in bands:
        assert not np.isfinite(y[:, cs:ce]).any()


@pytest.mark.parametrize("m", [1023, 1100])
def test_lora_strength(m):
    n, r = 128, 24
    w, wd = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    x = _x([m, K], mx.bfloat16, 29)
    per_col = mx.array(np.linspace(-1.5, 1.5, r).astype(np.float32))
    for strength in (mx.array([0.5]), mx.array(-2.0), per_col):
        _check(x, w, wd, a, b, strength=strength)
    # A strength of one is no strength, and a strength of zero is the
    # product alone.
    y = kq.quantized_matmul_lora(x, w, "q4_k", a, b)
    one = kq.quantized_matmul_lora(x, w, "q4_k", a, b, strength=mx.ones((r,)))
    zero = kq.quantized_matmul_lora(x, w, "q4_k", a, b, strength=mx.zeros((1,)))
    base = kq.quantized_matmul(x, w, mx.zeros((1,), dtype=mx.uint8), "q4_k", True)
    mx.eval(y, one, zero, base)
    assert mx.array_equal(y, one).item()
    if _fused(m, r, n):
        # The K walk of the kernel may not be the tile of quantized_matmul.
        assert _rel(_f64(zero), _f64(base)) <= 1e-3
    else:
        assert mx.array_equal(zero, base).item()


def test_lora_strength_change_needs_no_new_trace():
    n, r, m = 128, 16, 1100
    w, wd = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    x = _x([m, K], mx.bfloat16, 31)
    traces = []

    def f(t, s):
        traces.append(1)
        return kq.quantized_matmul_lora(t, w, "q4_k", a, b, strength=s)

    cf = mx.compile(f)
    for value in (1.0, -0.25, 0.0, 3.0):
        s = mx.array([value])
        got = cf(x, s)
        want = kq.quantized_matmul_lora(x, w, "q4_k", a, b, strength=s)
        mx.eval(got, want)
        assert mx.array_equal(got, want).item(), value
    assert len(traces) == 1


@pytest.mark.parametrize("m", [1023, 1100])
def test_lora_stacked_pair(m):
    """Two adapters in one call: their A columns and B rows joined, one
    strength per column. Equal to the sum of the two terms."""
    n = 128
    w, wd = _weight("q4_k", n, K)
    a1, b1 = _ab(K, n, 16, mx.bfloat16, seed=1)
    a2, b2 = _ab(K, n, 64, mx.bfloat16, seed=2)
    a = mx.concatenate([a1, a2], axis=1)
    b = mx.concatenate([b1, b2], axis=0)
    strength = mx.array([0.75] * 16 + [-0.5] * 64)
    x = _x([m, K], mx.bfloat16, 37)
    _check(x, w, wd, a, b, strength=strength)
    y = kq.quantized_matmul_lora(x, w, "q4_k", a, b, strength=strength)
    x64 = _f64(x)
    want = (
        x64 @ wd.T
        + 0.75 * (x64 @ _f64(a1)) @ _f64(b1)
        - 0.5 * (x64 @ _f64(a2)) @ _f64(b2)
    )
    assert _rel(_f64(y), want) <= (R1[mx.bfloat16] if _nax() else CPU_REL)


def test_lora_rank_zero_and_empty_inputs():
    n = 96
    w, wd = _weight("q4_k", n, K)
    s8 = mx.zeros((1,), dtype=mx.uint8)
    for m in (1023, 1100):
        x = _x([m, K], mx.bfloat16, 41)
        # Rank 0: the product alone, bit for bit.
        y = kq.quantized_matmul_lora(x, w, "q4_k", mx.zeros((K, 0)), mx.zeros((0, n)))
        base = kq.quantized_matmul(x, w, s8, "q4_k", True)
        mx.eval(y, base)
        assert y.dtype == base.dtype and mx.array_equal(y, base).item()
    a, b = _ab(K, n, 16, mx.bfloat16)
    for shape in ([0, K], [2, 0, K], [0, 3, K]):
        y = kq.quantized_matmul_lora(
            mx.zeros(shape, dtype=mx.bfloat16), w, "q4_k", a, b
        )
        mx.eval(y)
        assert y.shape == tuple(shape[:-1]) + (n,) and y.dtype == mx.bfloat16
    # No output column.
    w0 = w[:0]
    y = kq.quantized_matmul_lora(
        _x([1100, K], mx.bfloat16, 43), w0, "q4_k", a, mx.zeros((16, 0))
    )
    mx.eval(y)
    assert y.shape == (1100, 0)
    # No inner dim: every sum is empty, so y is zero.
    y = kq.quantized_matmul_lora(
        mx.zeros((5, 0), dtype=mx.bfloat16),
        mx.zeros((n, 0), dtype=mx.uint8),
        "q4_k",
        mx.zeros((0, 16)),
        b,
    )
    mx.eval(y)
    assert y.shape == (5, n) and y.dtype == mx.bfloat16
    assert not _f64(y).any()


def _check_views(views, call, state):
    """call(view) against call(dense copy of the view). On the GPU the two
    have the same bits. On the CPU stream the bound is one rounding of the
    activation dtype (R1), not bit equality: the stock matmul there sums a
    product with a transposed second operand and a product of dense
    operands in different orders. Both are right to about one rounding
    against the exact sum, the dense order with about 1.3 times the error
    of the other. That is the accuracy of the stock op on bfloat16 and
    float16, so the two calls may differ by a rounding. A view read with
    the wrong strides differs by its whole size."""
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
    """Views of one weight whose rows are not dense."""
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
def test_lora_operand_views(state, m, codec):
    """A view of any operand gives the bits of its dense copy, lazy or
    evaluated, on both sides of the row bound."""
    n, r = 96, 16
    w, wd = _weight(codec, n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    x = _x([m, K], mx.bfloat16, 47)
    s = mx.array(np.linspace(0.5, 1.5, r).astype(np.float32)).astype(mx.bfloat16)
    bands = [(0, 64, 0, 16), (64, 96, 4, 12)]

    def call(x_=x, w_=w, a_=a, b_=b, s_=s):
        return kq.quantized_matmul_lora(x_, w_, codec, a_, b_, strength=s_, bands=bands)

    _check_views(
        _strided(_x([2 * m, 2 * K], mx.bfloat16, 53)), lambda v: call(x_=v), state
    )
    _check_views(
        _strided(_x([2 * K, 2 * r], mx.bfloat16, 59)), lambda v: call(a_=v), state
    )
    _check_views(
        _strided(_x([2 * r, 2 * n], mx.bfloat16, 61)), lambda v: call(b_=v), state
    )
    _check_views(_weight_views(w), lambda v: call(w_=v), state)
    s2 = _x([2 * r], mx.bfloat16, 67)
    views = {"even": lambda: s2[0::2], "reversed": lambda: s2[::-1][:r]}
    _check_views(views, lambda v: call(s_=v), state)
    # The views of x keep the result right, not only equal to the copy.
    xv = _strided(_x([2 * m, 2 * K], mx.bfloat16, 53))["odd_cols"]()
    _check(xv, w, wd, a, b, codec, strength=s, bands=bands)


@pytest.mark.parametrize(
    ("codec", "dtype"),
    [
        ("q3_k", mx.bfloat16),
        ("q4_0", mx.bfloat16),
        ("q5_k", mx.float16),
        ("q6_k", mx.float16),
        ("q8_0", mx.float16),
    ],
    ids=["q3_k-bf16", "q4_0-bf16", "q5_k-f16", "q6_k-f16", "q8_0-f16"],
)
def test_lora_codec_without_kernel_composes(codec, dtype):
    n, r = 96, 16
    w, wd = _weight(codec, n, K)
    a, b = _ab(K, n, r, dtype)
    err, err2 = _check(_x([1100, K], dtype, 71), w, wd, a, b, codec)
    assert err2 is None


@pytest.mark.parametrize("m", [1023, 1100])
@pytest.mark.parametrize(
    "dtype", [mx.bfloat16, mx.float16, mx.float32], ids=["bf16", "f16", "f32"]
)
@pytest.mark.parametrize("banded", [False, True])
def test_lora_gradient_is_the_gradient_of_the_composition(banded, dtype, m):
    """mx.grad wrt x, A, B and the strength on both sides of the row bound.
    The kernel has the vjp of the unfused ops."""
    n, r = 128, 16
    w, _ = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, dtype)
    x = _x([m, K], dtype, 73)
    s = mx.array(np.linspace(0.5, 1.5, r).astype(np.float32)).astype(dtype)
    c = mx.random.normal((m, n), key=mx.random.key(79))
    mx.eval(c)
    bands = [(0, 64, 0, 10), (64, 128, 10, 6)] if banded else None

    def loss(f):
        def run(x_, a_, b_, s_):
            y = f(x_, w, a_, b_, "q4_k", strength=s_, bands=bands)
            return (y.astype(mx.float32) * c).sum()

        return run

    def op(x_, w_, a_, b_, codec, **kw):
        return kq.quantized_matmul_lora(x_, w_, codec, a_, b_, **kw)

    g_op = mx.grad(loss(op), argnums=[0, 1, 2, 3])(x, a, b, s)
    g_ref = mx.grad(loss(_composed), argnums=[0, 1, 2, 3])(x, a, b, s)
    mx.eval(g_op, g_ref)
    for got, want, arg in zip(g_op, g_ref, (x, a, b, s), strict=True):
        assert got.shape == arg.shape and got.dtype == arg.dtype
        ref = _f64(want)
        assert np.isfinite(ref).all() and np.abs(ref).max() > 0
        # bfloat16 keeps 8 significant bits, and the terms of a gradient
        # may be summed in another order than in the composition.
        tol = 1e-3 if dtype == mx.float16 else 6e-3
        assert _rel(_f64(got), ref) <= tol


def test_lora_compiles_and_repeats():
    n, r, m = 128, 16, 1100
    w, wd = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    x = _x([m, K], mx.bfloat16, 83)
    f = mx.compile(lambda t: kq.quantized_matmul_lora(t, w, "q4_k", a, b))
    first = f(x)
    second = f(x)
    plain = kq.quantized_matmul_lora(x, w, "q4_k", a, b)
    mx.eval(first, second, plain)
    assert mx.array_equal(first, second).item()
    assert mx.array_equal(first, plain).item()


def test_lora_compiled_function_with_two_band_tables():
    """Two calls that differ in the band table alone stay two nodes when a
    compiled function holds both."""
    n, r, m = 128, 32, 1100
    w, _ = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    x = _x([m, K], mx.bfloat16, 103)
    one = [(0, 64, 0, 16), (64, 128, 16, 16)]
    two = [(0, 64, 16, 16), (64, 128, 0, 16)]

    def both(t):
        p = kq.quantized_matmul_lora(t, w, "q4_k", a, b, bands=one)
        q = kq.quantized_matmul_lora(t, w, "q4_k", a, b, bands=two)
        # mx.compile merges equivalent nodes but never an output of the
        # function, so the two calls feed the outputs. The casts are exact.
        return p.astype(mx.float32), q.astype(mx.float32)

    p, q = mx.compile(both)(x)
    rp, rq = both(x)
    mx.eval(p, q, rp, rq)
    assert mx.array_equal(p, rp).item()
    assert mx.array_equal(q, rq).item()
    assert not mx.array_equal(rp, rq).item()


def test_lora_runs_on_the_stream_of_the_call():
    """A call on a stream that is not the default one builds the
    composition on that stream, at a row count that takes the kernel on
    the default one."""
    if force_cpu() or mx.default_device() != mx.gpu:
        pytest.skip("the CPU is the default device")
    n, r, m = 96, 16, 1100
    w, wd = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    x = _x([m, K], mx.bfloat16, 97)
    y = kq.quantized_matmul_lora(x, w, "q4_k", a, b, stream=mx.cpu)
    with mx.stream(mx.cpu):
        ref = _composed(x, w, a, b)
    mx.eval(y, ref)
    assert mx.array_equal(y, ref).item()
    assert _rel(_f64(y), _ref(x, wd, a, b)) <= CPU_REL


def test_lora_jvp_and_vmap_do_not_depend_on_the_row_count():
    """quantized_matmul has no jvp and no vmap, so both raise for the
    composition. The kernel raises the same way."""
    n, r = 96, 16
    w, _ = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    seen = {"jvp": [], "vmap": []}
    for m in (1023, 1100):
        x = _x([2, m, K], mx.bfloat16, 101)

        def f(t):
            return kq.quantized_matmul_lora(t, w, "q4_k", a, b)

        with pytest.raises(ValueError) as e:
            mx.eval(mx.jvp(f, [x[0]], [x[0]])[1])
        seen["jvp"].append(str(e.value))
        with pytest.raises(ValueError) as e:
            mx.eval(mx.vmap(f)(x))
        seen["vmap"].append(type(e.value))
    assert seen["jvp"][0] == seen["jvp"][1], seen["jvp"]
    assert "KQuantMatmul" in seen["jvp"][0]


def test_lora_weight_is_frozen():
    """A gradient with respect to the weight raises at every row count."""
    n, r = 96, 16
    w, _ = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    for m in (1023, 1100):
        x = _x([m, K], mx.bfloat16, 107)

        def loss(w_, x_=x):
            y = kq.quantized_matmul_lora(x_, w_, "q4_k", a, b)
            return y.astype(mx.float32).sum()

        with pytest.raises(ValueError):
            mx.eval(mx.grad(loss)(w))


def test_lora_argument_errors():
    n, r = 96, 16
    w, _ = _weight("q4_k", n, K)
    a, b = _ab(K, n, r, mx.bfloat16)
    x = _x([4, K], mx.bfloat16, 89)

    def bad(match, x_=x, w_=w, a_=a, b_=b, codec="q4_k", **kw):
        # The error names this op, not an op it builds from.
        with pytest.raises(ValueError, match=r"quantized_matmul_lora\] .*" + match):
            kq.quantized_matmul_lora(x_, w_, codec, a_, b_, **kw)

    bad("Unknown kquant_type", codec="q9_z")
    bad("2-D uint8", w_=w.astype(mx.float32))
    bad("2-D uint8", w_=w.reshape(1, n, -1))
    bad("whole number", w_=w[:, :-1])
    bad("x last dim", x_=mx.array(1.0, dtype=mx.bfloat16))
    bad("x last dim", x_=x[:, :-1])
    bad("floating x", x_=x.astype(mx.int32))
    bad("lora_a must be", a_=a[:-1])
    bad("lora_a must be", a_=a.astype(mx.int32))
    bad("lora_a must be", a_=a.reshape(K, r, 1))
    bad("lora_b must be", b_=b[:-1])
    bad("lora_b must be", b_=b[:, :-1])
    bad("lora_b must be", b_=b.astype(mx.int32))
    bad("lora_b must be", b_=b.reshape(r, n, 1))
    bad("strength must be", strength=mx.ones((r + 1,)))
    bad("strength must be", strength=mx.ones((1, r)))
    bad("strength must be", strength=mx.ones((r, 1)))
    bad("strength must be", strength=mx.ones((r,), dtype=mx.int32))
    bad("rows of", bands=[(0, 64, 0)])
    bad("rows of", bands=[(0, 64, 0), (64, 96, 0, 8, 8)])
    bad("column order", bands=[(64, 96, 0, 8), (0, 64, 8, 8)])
    bad("column order", bands=[(0, 64, 0, 8), (63, 96, 8, 8)])
    bad("column order", bands=[(32, 32, 0, 8)])
    bad("column order", bands=[(0, n + 1, 0, 8)])
    bad("column order", bands=[(-1, 64, 0, 8)])
    bad("takes ranks", bands=[(0, 64, r - 7, 8)])
    bad("takes ranks", bands=[(0, 64, -1, 8)])
    bad("takes ranks", bands=[(0, 64, 0, -1)])


def test_lora_kernels_in_metallib():
    if force_cpu():
        pytest.skip("Metal kernel names only")
    names = kq.metallib_kernels()
    if not names:
        pytest.skip("no metallib loaded")
    lora = {n for n in names if "_qmm_t_nax_lora_" in n}
    want = {_name(n, codec) for n in (64, 96) for codec in CODECS}
    assert lora == want, sorted(lora ^ want)


_PROBE = r"""
import json, os, sys
spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["tests_dir"])
import mlx.core as mx
import mlx_kquant as kq
import test_matmul_lora as t

out = []
for i, c in enumerate(spec["calls"]):
    dtype = getattr(mx, c.get("dtype", "bfloat16"))
    codec = c.get("codec", "q4_k")
    k = c.get("k", t.K)
    if c.get("tile"):
        # A wide weight as copies of a block of rows.
        w, wd = t._weight(codec, c["tile"], k)
        w = mx.tile(w, (c["n"] // c["tile"], 1))
        wd = t.np.tile(wd, (c["n"] // c["tile"], 1))
    else:
        w, wd = t._weight(codec, c["n"], k)
    a, b = t._ab(k, c["n"], c["r"], dtype)
    x = t._x(c.get("shape", [c["m"], k]), dtype, 7 * c["m"] + c["n"])
    bands = c.get("bands")
    os.environ.update(c.get("env", {}))
    print(f"[probe] {i}", file=sys.stderr, flush=True)
    res = {"error": None}
    try:
        y = kq.quantized_matmul_lora(x, w, codec, a, b, bands=bands)
        mx.eval(y)
        got = t._f64(y).reshape(-1, c["n"])
        res["rel"] = t._rel(got, t._ref(x, wd, a, b, None, bands))
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    out.append(res)
print("RESULT " + json.dumps(out))
"""


def _probe(calls, env=None):
    """Run calls in a fresh interpreter that keeps no KQ_ variable but
    those of env and of each call. Per call: lora (the low-rank kernels
    that KQ_QMM_LOG named), plain (the other kernels), error and rel."""
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
            cur.update(lora=[], plain=[])
        elif cur is not None and ln.startswith("[kq qmm] "):
            name = ln.split()[2]
            cur["lora" if "_lora_" in name else "plain"].append(name)
    return results


def _name(n, codec="q4_k"):
    aligned = "true" if n % 64 == 0 else "false"
    return (
        f"kquant_{codec}_qmm_t_nax_lora_bfloat16_t_bm128_bn64_bk{KERNEL_BK[codec]}"
        f"_wm2_wn2_alN_{aligned}"
    )


def _expect(calls, results, kernel):
    for c, r, k in zip(calls, results, kernel, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] <= R1[getattr(mx, c.get("dtype", "bfloat16"))], (c, r["rel"])
        if k:
            assert r["lora"] == [_name(c["n"], c.get("codec", "q4_k"))], (c, r["lora"])
            assert r["plain"] == [], (c, r["plain"])
        else:
            assert r["lora"] == [], (c, r["lora"])
            assert len(r["plain"]) == 1, (c, r["plain"])


def test_lora_kernel_that_runs():
    """KQ_QMM_LOG names the kernel of each call. With KQ_QMM_LORA=2, where
    the kernel can run it runs, once, and no matmul tile. Everywhere else
    one matmul tile runs and no low-rank kernel."""
    if not _nax():
        pytest.skip("the kernel needs a NAX GPU")
    sixteen = BANDS["sixteen"]
    seventeen = BANDS["seventeen"]
    calls = [
        ({"m": 1024, "n": 128, "r": 16}, True),
        ({"m": 1023, "n": 128, "r": 16}, False),
        ({"m": 1100, "n": 96, "r": 16}, True),
        ({"m": 1100, "n": 128, "r": 256}, True),
        ({"m": 1100, "n": 128, "r": 257}, False),
        ({"m": 1100, "n": sixteen[0], "r": sixteen[1], "bands": sixteen[2]}, True),
        (
            {"m": 1100, "n": seventeen[0], "r": seventeen[1], "bands": seventeen[2]},
            False,
        ),
        ({"m": 1100, "n": 192, "r": 32, "bands": BANDS["start_off_tile"][2]}, False),
        ({"m": 1100, "n": 300, "r": 40, "bands": BANDS["end_on_half_tile"][2]}, True),
        ({"m": 1100, "n": 300, "r": 40, "bands": BANDS["end_at_n"][2]}, True),
        ({"m": 1100, "n": 300, "r": 40, "bands": BANDS["end_in_tile"][2]}, False),
        ({"m": 1100, "n": 128, "r": 256, "bands": [(0, 128, 0, 256)]}, True),
        ({"m": 1100, "n": 128, "r": 300, "bands": BANDS["rank_above_max"][2]}, False),
        ({"m": 2200, "n": 128, "r": 16, "shape": [2, 1100, K]}, True),
        ({"m": 1100, "n": 128, "r": 16, "dtype": "float16"}, False),
        ({"m": 1100, "n": 128, "r": 16, "codec": "q5_k"}, True),
        ({"m": 1100, "n": 96, "r": 16, "codec": "q5_k"}, True),
        ({"m": 1100, "n": 128, "r": 16, "codec": "q6_k"}, True),
        ({"m": 1100, "n": 96, "r": 16, "codec": "q6_k"}, True),
        ({"m": 1100, "n": 128, "r": 16, "codec": "q8_0"}, True),
        ({"m": 1100, "n": 96, "r": 16, "codec": "q8_0"}, True),
        ({"m": 1100, "n": 128, "r": 16, "codec": "q8_0", "k": 832}, True),
        ({"m": 1100, "n": 128, "r": 16, "codec": "q8_0", "k": 800}, False),
        ({"m": 1100, "n": 128, "r": 16, "codec": "q6_k", "dtype": "float16"}, False),
        ({"m": 1100, "n": 128, "r": 16, "codec": "q3_k"}, False),
        ({"m": 1100, "n": 128, "r": 16, "codec": "q4_0"}, False),
    ]
    res = _probe([c for c, _ in calls], env={"KQ_QMM_LORA": "2"})
    _expect([c for c, _ in calls], res, [k for _, k in calls])


def _wide(k, n, **kw):
    """A call on a weight [n, k] made of copies of 64 rows."""
    return {"m": 1100, "n": n, "r": 16, "k": k, "tile": 64, **kw}


# Both sides of each bound of the shape rule. From RULE_ROWS rows: N against
# K, and K against RULE_K. Below RULE_ROWS rows: N against RULE_WIDE_N.
RULE_CASES = [
    (_wide(256, 256, m=RULE_ROWS), True),
    (_wide(256, 256, m=RULE_ROWS - 1), False),
    (_wide(256, 192, m=RULE_ROWS), False),
    (_wide(RULE_K, RULE_K, m=RULE_ROWS), True),
    (_wide(RULE_K + 256, RULE_K + 256, m=RULE_ROWS), False),
    (_wide(64, 64, m=RULE_ROWS, codec="q8_0"), True),
    (_wide(128, 64, m=RULE_ROWS, codec="q8_0"), False),
    (_wide(256, RULE_WIDE_N), True),
    (_wide(256, RULE_WIDE_N - 64), False),
    (_wide(RULE_K, RULE_WIDE_N), True),
    (_wide(RULE_K + 256, RULE_WIDE_N), False),
]


def test_lora_shape_rule():
    """With KQ_QMM_LORA unset the op gives the kernel the calls of the
    shape rule and builds the composition for every other call."""
    if not _nax():
        pytest.skip("the kernel needs a NAX GPU")
    calls = [c for c, _ in RULE_CASES]
    want = [k for _, k in RULE_CASES]
    assert want == [_shape_rule(c["k"], c["n"], c["m"]) for c in calls]
    _expect(calls, _probe(calls), want)


def test_lora_levers():
    """KQ_QMM_LORA=0 (or off) builds the composition on every call. 2 runs
    the kernel outside the shape rule, inside its other bounds. 1, or a
    value that is no mode, is the shape rule. KQ_QMM_LORA_MIN_ROWS moves the
    row bound, down to one row of a 1-D x."""
    if not _nax():
        pytest.skip("the kernel needs a NAX GPU")
    inside, outside = _wide(K, RULE_WIDE_N), _wide(K, 128)
    calls = [inside, outside, {**inside, "m": 4608, "r": 64}]
    for value in ("0", "off"):
        res = _probe(calls, env={"KQ_QMM_LORA": value})
        _expect(calls, res, [False, False, False])
    calls = [
        inside,
        outside,
        _wide(RULE_K + 256, RULE_K + 256),
        {**outside, "m": 1023},
        {**outside, "r": 257},
        {**outside, "dtype": "float16"},
    ]
    res = _probe(calls, env={"KQ_QMM_LORA": "2"})
    _expect(calls, res, [True, True, True, False, False, False])
    calls = [inside, outside]
    for value in ("1", "3", "-1", "on", "x"):
        _expect(calls, _probe(calls, env={"KQ_QMM_LORA": value}), [True, False])
    calls = [
        {"m": 600, "n": 128, "r": 16},
        {"m": 511, "n": 128, "r": 16},
        {"m": 512, "n": 128, "r": 16},
    ]
    res = _probe(calls, env={"KQ_QMM_LORA": "2", "KQ_QMM_LORA_MIN_ROWS": "512"})
    _expect(calls, res, [True, False, True])
    calls = [{"m": 1, "n": 128, "r": 16, "shape": [K]}]
    res = _probe(calls, env={"KQ_QMM_LORA": "2", "KQ_QMM_LORA_MIN_ROWS": "1"})
    _expect(calls, res, [True])
    # The row bound holds inside the shape rule too.
    calls = [{**inside, "m": 1023}, {**inside, "m": 1024}]
    _expect(calls, _probe(calls), [False, True])
