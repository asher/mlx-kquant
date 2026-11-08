"""int8_quantize_rows and int8_matmul: int8 x int8 linears on NAX GPUs.

Every numeric bound below was fixed before the kernel first ran, from a
reference figure times a margin named in the test's docstring. The figures
are those of a reference implementation of the same kernel (per-row int8
activations, per-channel int8 weights, bfloat16 output) at a video DiT's
linear shapes, 2376 rows: against the float32 product of the same codes and
scales, relative L2 1.66e-3 over the output and 1.74e-3 on the worst row,
both from the bfloat16 rounding of the output, and 1.02e-4 against that
product rounded to bfloat16. The quantizer bound is sdpa_qk8_inputs'.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="int8_quantize_rows and int8_matmul run on Metal; no CPU path.",
)

NAX = (
    not os.environ.get("KQUANT_FORCE_CPU")
    and mx.metal.is_available()
    and kq.nax_available()
)

requires_nax = pytest.mark.skipif(
    not NAX, reason="int8_matmul needs tensor-op (NAX) hardware."
)

big = pytest.mark.skipif(
    not os.environ.get("KQUANT_BIG_TESTS"),
    reason="holds a 4.4 GB output; set KQUANT_BIG_TESTS=1 to run",
)

# Reference figures and margins (see the module docstring).
REF_REL = 1.66e-3
REL = 0.002  # margin 1.2x
REF_ROW = 1.74e-3
REL_ROW = 0.0025  # margin 1.44x
REF_BF16 = 1.02e-4
REL_BF16 = 1e-3  # margin about 10x
# float32 output of a grouped call against float64: 16 float32 roundings
# of about 6e-8 each.
REL_F32_GROUPED = 1e-6
# Codes against the MLX-ops formula: within 1, and at most 0.2 percent
# differing (sdpa_qk8_inputs' margin; the reference quantizer differed in
# none against its own rule).
CODE_DIFF_SHARE = 0.002
# The quantizer's floor on a scale, the smallest normal float32.
FLT_MIN = float(np.finfo(np.float32).tiny)


def _rng(seed):
    return np.random.default_rng(seed)


def _x(rng, shape, dtype=mx.bfloat16):
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(dtype)


def _w(rng, n, k):
    """W ~ N(0, 1/K) in float32."""
    return mx.array((rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float32))


def _quant_ref(x, group_size=None):
    """The formula in MLX ops on float32: codes and scales [..., G]."""
    xf = x.astype(mx.float32)
    k = xf.shape[-1]
    gs = k if group_size is None or group_size >= k else group_size
    codes, scales = [], []
    for c0 in range(0, k, gs):
        blk = xf[..., c0 : c0 + gs]
        m = mx.max(mx.abs(blk), axis=-1, keepdims=True)
        s = mx.where(
            m > 0, mx.maximum(m / 127.0, FLT_MIN), mx.array(1.0 / 127.0, mx.float32)
        )
        codes.append(mx.clip(mx.round(blk / s), -127, 127))
        scales.append(s)
    return mx.concatenate(codes, axis=-1), mx.concatenate(scales, axis=-1)


def _operands(rng, m, n, k, group_size=None, dtype=mx.bfloat16):
    x = _x(rng, (m, k), dtype)
    x8, sx = kq.int8_quantize_rows(x, group_size=group_size)
    w8, sw = kq.int8_quantize_rows(_w(rng, n, k))
    sw = sw.reshape(n)
    mx.eval(x8, sx, w8, sw)
    return x8, sx, w8, sw


def _reference64(x8, sx, w8, sw, group_size=None):
    """y[r, c] in float64 from the same codes and scales: the exact integer
    sum of each group times its row scale, summed over groups, times the
    channel scale."""
    a = np.array(x8).astype(np.float64)
    b = np.array(w8).astype(np.float64)
    s = np.array(sx).astype(np.float64)
    k = a.shape[-1]
    gs = k if group_size is None or group_size >= k else group_size
    y = np.zeros(a.shape[:-1] + (b.shape[0],), np.float64)
    for g, c0 in enumerate(range(0, k, gs)):
        y += (a[..., c0 : c0 + gs] @ b[:, c0 : c0 + gs].T) * s[..., g : g + 1]
    return y * np.array(sw).astype(np.float64)


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _rel(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def _rel_rows(x, ref):
    return np.linalg.norm(x - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


def _bf16_rounded(ref):
    return _f64(mx.array(ref.astype(np.float32)).astype(mx.bfloat16))


def _check_bar(y, ref):
    """The plain call's bar on a bfloat16 output y against the float64
    reference of the same codes and scales."""
    got = _f64(y)
    assert np.isfinite(got).all()
    assert _rel(got, ref) <= REL
    assert _rel_rows(got, ref).max() <= REL_ROW
    assert _rel(got, _bf16_rounded(ref)) <= REL_BF16


# The quantizer.


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize("k,group_size", [(5376, None), (14336, 1024)])
def test_quantizer_matches_its_formula(dtype, k, group_size):
    """Rows of a [2, 33, K] input, one of them all zeros and one with a
    single nonzero value. Codes within 1 of the MLX-ops formula and at most
    0.2 percent differing; scales within 1e-6 relative; the zero row has
    scale 1/127 and codes 0."""
    dt = getattr(mx, dtype)
    rng = _rng(3)
    xn = rng.standard_normal((2, 33, k)).astype(np.float32)
    xn[0, 5] = 0.0
    xn[1, 7] = 0.0
    xn[1, 7, 123] = -2.5
    x = mx.array(xn).astype(dt)
    codes, scales = kq.int8_quantize_rows(x, group_size=group_size)
    g = 1 if group_size is None else -(-k // group_size)
    assert codes.shape == x.shape and codes.dtype == mx.int8
    assert scales.shape == (2, 33, g) and scales.dtype == mx.float32
    c_ref, s_ref = _quant_ref(x, group_size)
    diff = np.abs(np.array(codes).astype(np.int32) - np.array(c_ref).astype(np.int32))
    assert diff.max() <= 1
    assert (diff != 0).mean() <= CODE_DIFF_SHARE
    np.testing.assert_allclose(np.array(scales), np.array(s_ref), rtol=1e-6)
    assert np.all(np.array(scales)[0, 5] == np.float32(1.0 / 127.0))
    assert not np.array(codes)[0, 5].any()
    one = np.array(codes)[1, 7]
    assert one[123] == -127 and np.count_nonzero(one) == 1


def test_short_last_group():
    """K 3200 in groups of 1024: four scales a row, the last over columns
    3072 to 3199 only. The large values sit in the first 3072 columns, so
    a last scale over more columns would differ."""
    rng = _rng(5)
    xn = rng.standard_normal((40, 3200)).astype(np.float32)
    xn[:, :3072] *= 100.0
    x = mx.array(xn).astype(mx.bfloat16)
    codes, scales = kq.int8_quantize_rows(x, group_size=1024)
    assert scales.shape == (40, 4)
    tail = np.array(x[:, 3072:].astype(mx.float32))
    want = np.abs(tail).max(axis=1) / np.float32(127.0)
    np.testing.assert_allclose(np.array(scales)[:, 3], want, rtol=1e-6)
    c_ref, _ = _quant_ref(x, 1024)
    diff = np.abs(np.array(codes).astype(np.int32) - np.array(c_ref).astype(np.int32))
    assert diff.max() <= 1


def test_quantizer_reads_a_strided_input_as_its_copy():
    """A transposed view gives the bits of its row-major copy."""
    rng = _rng(7)
    xt = _x(rng, (2048, 96)).T
    a = kq.int8_quantize_rows(xt, group_size=512)
    b = kq.int8_quantize_rows(mx.contiguous(xt), group_size=512)
    mx.eval(a, b)
    for u, v in zip(a, b, strict=True):
        np.testing.assert_array_equal(np.array(u), np.array(v))


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize("group_size", [None, 32])
def test_quantizer_scalar_path(dtype, group_size):
    """K 1001, not a multiple of 4, runs the scalar kernel: the formula
    test's bars (codes within 1 and at most 0.2 percent differing, scales
    within 1e-6), with a short last group of 9 columns under groups of 32."""
    x = _x(_rng(31), (37, 1001), getattr(mx, dtype))
    codes, scales = kq.int8_quantize_rows(x, group_size=group_size)
    c_ref, s_ref = _quant_ref(x, group_size)
    diff = np.abs(np.array(codes).astype(np.int32) - np.array(c_ref).astype(np.int32))
    assert diff.max() <= 1
    assert (diff != 0).mean() <= CODE_DIFF_SHARE
    np.testing.assert_allclose(np.array(scales), np.array(s_ref), rtol=1e-6)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_tiny_rows_take_the_floor_scale(dtype):
    """A row whose max |x| is under 127 x FLT_MIN gets scale FLT_MIN, not a
    subnormal flushed to 0, and its zeros code as 0: codes rint(x / FLT_MIN),
    here 9 (8 in bfloat16, which rounds 1e-37 down), -4 and zeros."""
    xn = np.zeros((2, 64), np.float32)
    xn[0, 0], xn[0, 1] = 1e-37, -5e-38
    x = mx.array(xn).astype(getattr(mx, dtype))
    codes, scales = kq.int8_quantize_rows(x)
    xf = np.array(x.astype(mx.float32))
    want = np.rint(xf[0] / np.float32(FLT_MIN)).astype(np.int32)
    assert want[0] == (9 if dtype == "float32" else 8) and want[1] == -4
    assert not want[2:].any()
    assert np.array(scales)[0, 0] == np.float32(FLT_MIN)
    np.testing.assert_array_equal(np.array(codes)[0].astype(np.int32), want)
    assert not np.array(codes)[1].any()


# The matmul.


@requires_nax
@pytest.mark.parametrize(
    "m,n,k,lim,group_size",
    [
        (200, 256, 256, 127, None),
        (200, 200, 256, 127, None),
        (200, 256, 5376, 15, None),
        (200, 200, 5376, 15, None),
        (200, 200, 5376, 15, 1024),
    ],
)
def test_exact_sums(m, n, k, lim, group_size):
    """float32 output with scales 1 equals the int64 product exactly: every
    sum is an integer under 2^24 (127 x 127 x 256 = 4.1e6, 15 x 15 x 5376 =
    1.2e6), which float32 holds, grouped or not. M 200 leaves a partial row
    block; N 200 a partial column block."""
    rng = _rng(m + n + k)
    a = rng.integers(-lim, lim + 1, (m, k)).astype(np.int8)
    b = rng.integers(-lim, lim + 1, (n, k)).astype(np.int8)
    g = 1 if group_size is None else -(-k // group_size)
    y = kq.int8_matmul(
        mx.array(a),
        mx.ones((m, g), mx.float32),
        mx.array(b),
        mx.ones((n,), mx.float32),
        group_size=group_size,
        out_dtype=mx.float32,
    )
    want = a.astype(np.int64) @ b.astype(np.int64).T
    np.testing.assert_array_equal(np.array(y), want.astype(np.float32))


@requires_nax
@pytest.mark.parametrize("group_size", [None, 133120])
def test_exact_sum_at_the_longest_group(group_size):
    """K 133120 of codes 127 against codes 127 (and -127 on row 1): float32
    output with scales 1 is 127 x 127 x 133120 = 2147092480 exactly, the
    largest int32 sum the op accepts, which float32 holds. With
    group_size 133120 the K 266240 call sums two such groups."""
    k = 133120 if group_size is None else 266240
    a = np.full((2, k), 127, np.int8)
    a[1] = -127
    g = 1 if group_size is None else 2
    y = kq.int8_matmul(
        mx.array(a),
        mx.ones((2, g), mx.float32),
        mx.full((3, k), 127, mx.int8),
        mx.ones((3,), mx.float32),
        group_size=group_size,
        out_dtype=mx.float32,
    )
    want = np.float32(2147092480.0 * g)
    np.testing.assert_array_equal(np.array(y)[0], np.full(3, want))
    np.testing.assert_array_equal(np.array(y)[1], np.full(3, -want))


@requires_nax
def test_copied_operands_give_the_row_major_bits():
    """x8 at a byte offset off a 16-byte boundary and a transposed w8 are
    copied before the kernel: the bits of the row-major call."""
    m, n, k = 130, 136, 256
    x8, sx, w8, sw = _operands(_rng(37), m, n, k)
    flat = mx.concatenate([mx.zeros((8,), mx.int8), x8.reshape(-1)])
    mx.eval(flat)
    x8_off = flat[8:].reshape(m, k)
    w8_t = mx.contiguous(w8.T).T
    mx.eval(w8_t)
    want = kq.int8_matmul(x8, sx, w8, sw)
    got = kq.int8_matmul(x8_off, sx, w8_t, sw)
    mx.eval(want, got)
    np.testing.assert_array_equal(
        np.array(got.view(mx.uint16)), np.array(want.view(mx.uint16))
    )


PLAIN = {"qkv_proj": (5376, 21504), "out_proj": (7168, 5376), "mlp.fc1": (5376, 28672)}


@requires_nax
@pytest.mark.parametrize("name", sorted(PLAIN))
def test_plain_call(name):
    """bfloat16 output at a DiT linear's shape, M 2376 (ragged): relative L2
    against the float64 product of the same codes and scales at most 0.002
    (reference 1.66e-3, margin 1.2x), worst row at most 0.0025 (1.74e-3,
    1.44x), and at most 1e-3 against that product rounded to bfloat16
    (1.02e-4, about 10x)."""
    k, n = PLAIN[name]
    x8, sx, w8, sw = _operands(_rng(k + n), 2376, n, k)
    y = kq.int8_matmul(x8, sx, w8, sw)
    mx.eval(y)
    assert y.shape == (2376, n) and y.dtype == mx.bfloat16
    _check_bar(y, _reference64(x8, sx, w8, sw))


@requires_nax
def test_grouped_call():
    """K 14336 in groups of 1024 at M 2376. bfloat16 output: the plain
    call's bar. float32 output against float64: relative L2 at most 1e-6
    (16 float32 roundings of about 6e-8). A group scaled by the wrong scale
    would show near sqrt(1024 / 14336) = 0.27."""
    x8, sx, w8, sw = _operands(_rng(11), 2376, 5376, 14336, 1024)
    ref = _reference64(x8, sx, w8, sw, 1024)
    y = kq.int8_matmul(x8, sx, w8, sw, group_size=1024)
    y32 = kq.int8_matmul(x8, sx, w8, sw, group_size=1024, out_dtype=mx.float32)
    mx.eval(y, y32)
    _check_bar(y, ref)
    assert y32.dtype == mx.float32
    assert _rel(np.array(y32).astype(np.float64), ref) <= REL_F32_GROUPED


@requires_nax
def test_float16_output():
    """float16 output: the plain call's bar against the float64 product,
    with the bfloat16-rounded comparison replaced by the float16-rounded
    one at the same bound."""
    x8, sx, w8, sw = _operands(_rng(13), 300, 384, 1024)
    y = kq.int8_matmul(x8, sx, w8, sw, out_dtype=mx.float16)
    mx.eval(y)
    assert y.dtype == mx.float16
    ref = _reference64(x8, sx, w8, sw)
    got = _f64(y)
    assert _rel(got, ref) <= REL
    assert _rel(got, ref.astype(np.float16).astype(np.float64)) <= REL_BF16


TAILS = {
    "m1": (1, 256, 5376, None),
    "m37": (37, 256, 5376, None),
    "m127": (127, 256, 5376, None),
    "m129": (129, 256, 5376, None),
    "m1922": (1922, 256, 5376, None),
    "n200": (129, 200, 5376, None),
    "n5416": (129, 5416, 5376, None),
    "k160": (37, 200, 160, None),
    "grouped_short_last": (129, 200, 3200, 1024),
    "grouped_m1922_n5416": (1922, 5416, 14336, 1024),
    "grouped_device_scales": (129, 200, 2048, 32),
}


@requires_nax
@pytest.mark.parametrize("name", sorted(TAILS))
def test_tails(name):
    """Ragged M and N, K off a multiple of 128, a short last group, and a
    group count whose row scales do not fit in threadgroup memory (64
    groups): the plain call's bar each, and a second call gives the first
    call's bits. Sentinel buffers allocated before and after the output
    keep their bytes. They are separate allocations, so they see a stray
    store only if it reaches one: the stores' row bound (r < M) is not
    observable in-process and is checked by reading the kernel."""
    m, n, k, gs = TAILS[name]
    x8, sx, w8, sw = _operands(_rng(m * 7 + n + k), m, n, k, gs)
    before = mx.full((1 << 16,), -7, mx.int32)
    mx.eval(before)
    y = kq.int8_matmul(x8, sx, w8, sw, group_size=gs)
    mx.eval(y)
    after = mx.full((1 << 16,), -7, mx.int32)
    mx.eval(after)
    y2 = kq.int8_matmul(x8, sx, w8, sw, group_size=gs)
    mx.eval(y2)
    assert y.shape == (m, n)
    _check_bar(y, _reference64(x8, sx, w8, sw, gs))
    np.testing.assert_array_equal(
        np.array(y2.view(mx.uint16)), np.array(y.view(mx.uint16))
    )
    for s in (before, after):
        assert (np.array(s) == -7).all()


@requires_nax
@pytest.mark.parametrize("group_size", [None, 1024])
@pytest.mark.parametrize("chunk", [8192, 64])
def test_chunked_rows_have_the_whole_calls_bits(group_size, chunk):
    """M 9000 in row chunks, the last ragged, each passed as the row slices
    x8[a:b] and x_scales[a:b]: each equals its rows of the whole call bit
    for bit."""
    m, n, k = 9000, 512, 3200
    x8, sx, w8, sw = _operands(_rng(17), m, n, k, group_size)
    whole = kq.int8_matmul(x8, sx, w8, sw, group_size=group_size)
    mx.eval(whole)
    whole = np.array(whole.view(mx.uint16))
    for r0 in range(0, m, chunk):
        r1 = min(r0 + chunk, m)
        part = kq.int8_matmul(x8[r0:r1], sx[r0:r1], w8, sw, group_size=group_size)
        np.testing.assert_array_equal(np.array(part.view(mx.uint16)), whole[r0:r1])


@requires_nax
def test_leading_dims():
    """x8 [3, 50, K] gives the bits of its [150, K] reshape, shaped [3, 50, N]."""
    x8, sx, w8, sw = _operands(_rng(19), 150, 136, 256)
    y = kq.int8_matmul(x8.reshape(3, 50, 256), sx.reshape(3, 50, 1), w8, sw)
    flat = kq.int8_matmul(x8, sx, w8, sw)
    mx.eval(y, flat)
    assert y.shape == (3, 50, 136)
    np.testing.assert_array_equal(
        np.array(y.view(mx.uint16)).reshape(150, 136), np.array(flat.view(mx.uint16))
    )


@requires_nax
def test_zero_rows_give_an_empty_output():
    x8, sx, w8, sw = _operands(_rng(23), 0, 128, 256)
    y = kq.int8_matmul(x8, sx, w8, sw)
    mx.eval(y)
    assert y.shape == (0, 128)


# Outputs past 2 GiB and 4 GiB.


def big_case(rows, k=5376, n=28672, chunk=8192, seed=29):
    """fc1 at `rows` rows, bfloat16 then float32 output, the second made
    after the first is freed. Checks the first 256 rows, the last 200 and
    256 seeded random rows against the reference with the plain call's bar,
    that the whole call equals its `chunk`-row chunks on those rows bit for
    bit, and that a sentinel allocated before each call keeps its bytes.
    Returns the peak memory in bytes."""
    rng = _rng(seed)
    x8, sx, w8, sw = _operands(rng, rows, n, k)
    pick = np.unique(
        np.concatenate(
            [
                np.arange(min(256, rows)),
                np.arange(max(0, rows - 200), rows),
                rng.choice(rows, size=min(256, rows), replace=False),
            ]
        )
    )
    idx = mx.array(pick.astype(np.int32))
    ref = _reference64(x8[idx], sx[idx], w8, sw)
    for dt in (mx.bfloat16, mx.float32):
        sentinel = mx.full((1 << 20,), -7, mx.int32)
        mx.eval(sentinel)
        y = kq.int8_matmul(x8, sx, w8, sw, out_dtype=dt)
        mx.eval(y)
        rows_got = y[idx]
        mx.eval(rows_got)
        got = _f64(rows_got)
        assert _rel(got, ref) <= REL
        assert _rel_rows(got, ref).max() <= REL_ROW
        if dt == mx.bfloat16:
            assert _rel(got, _bf16_rounded(ref)) <= REL_BF16
        bits = np.array(rows_got.view(mx.uint16 if dt == mx.bfloat16 else mx.uint32))
        del y, rows_got
        mx.clear_cache()
        chunk_rows = {}
        for r0 in range(0, rows, chunk):
            r1 = min(r0 + chunk, rows)
            sel = pick[(pick >= r0) & (pick < r1)]
            if sel.size == 0:
                continue
            part = kq.int8_matmul(x8[r0:r1], sx[r0:r1], w8, sw, out_dtype=dt)
            part = part[mx.array((sel - r0).astype(np.int32))]
            mx.eval(part)
            view = part.view(mx.uint16 if dt == mx.bfloat16 else mx.uint32)
            for i, r in enumerate(sel):
                chunk_rows[int(r)] = np.array(view[i])
            del part
            mx.clear_cache()
        for i, r in enumerate(pick):
            np.testing.assert_array_equal(chunk_rows[int(r)], bits[i])
        assert (np.array(sentinel) == -7).all()
        del sentinel
        mx.clear_cache()
    return mx.get_peak_memory()


@big
@requires_nax
def test_outputs_past_2_and_4_gib():
    """fc1 at 38010 rows: the bfloat16 output is 2.18 GB and the float32
    output 4.36 GB, past 2 GiB and 4 GiB of 32-bit offsets."""
    big_case(38010)


# Guards. Each input fails only the named guard, at build time, so the
# tests run on any Metal GPU and evaluate nothing.


def _quant_args():
    return dict(x=mx.zeros((4, 64), mx.bfloat16), group_size=None)


def _mm_args():
    return dict(
        x8=mx.zeros((4, 64), mx.int8),
        x_scales=mx.ones((4, 1), mx.float32),
        w8=mx.zeros((8, 64), mx.int8),
        w_scales=mx.ones((8,), mx.float32),
        group_size=None,
        out_dtype=mx.bfloat16,
    )


QUANT_GUARDS = {
    "scalar": ("at least one dim", {"x": mx.zeros((), mx.bfloat16)}),
    "empty_k": ("at least one dim", {"x": mx.zeros((4, 0), mx.bfloat16)}),
    "int8_x": (
        "must be bfloat16, float16 or float32",
        {"x": mx.zeros((4, 64), mx.int8)},
    ),
    "group_small": ("group_size must be", {"group_size": 0}),
    "group_odd": ("group_size must be", {"group_size": 48}),
    # Lazy inputs one past 2^27 groups; nothing is evaluated.
    "groups_rows": ("at most 2", {"x": mx.zeros(((1 << 27) + 1, 32), mx.bfloat16)}),
    "groups_grouped": (
        "at most 2",
        {"x": mx.zeros(((1 << 22) + 1, 1024), mx.bfloat16), "group_size": 32},
    ),
}

# M is held in 32 bits with a block of headroom.
BIG_ROWS = (1 << 31) - 256

MM_GUARDS = {
    "x8_dtype": ("x8 must be int8", {"x8": mx.zeros((4, 64), mx.uint8)}),
    "x8_scalar": ("x8 must be int8", {"x8": mx.zeros((), mx.int8)}),
    "k_odd": (
        "multiple of 32",
        {
            "x8": mx.zeros((4, 48), mx.int8),
            "w8": mx.zeros((8, 48), mx.int8),
        },
    ),
    "k_small": (
        "multiple of 32",
        {"x8": mx.zeros((4, 0), mx.int8), "w8": mx.zeros((8, 0), mx.int8)},
    ),
    "group_small": ("group_size must be", {"group_size": -32}),
    "group_odd": ("group_size must be", {"group_size": 40}),
    "w8_ndim": ("w8 must be int8", {"w8": mx.zeros((2, 8, 64), mx.int8)}),
    "w8_dtype": ("w8 must be int8", {"w8": mx.zeros((8, 64), mx.uint8)}),
    "w8_k": ("w8's last dim", {"w8": mx.zeros((8, 96), mx.int8)}),
    "x_scales_dtype": ("x_scales must be", {"x_scales": mx.ones((4, 1), mx.bfloat16)}),
    "x_scales_shape": ("x_scales must be", {"x_scales": mx.ones((4, 2), mx.float32)}),
    "x_scales_groups": ("x_scales must be", {"group_size": 32}),
    "w_scales_dtype": ("w_scales must be", {"w_scales": mx.ones((8,), mx.float16)}),
    "w_scales_ndim": ("w_scales must be", {"w_scales": mx.ones((8, 1), mx.float32)}),
    "w_scales_len": ("w_scales must be", {"w_scales": mx.ones((7,), mx.float32)}),
    "out_dtype": ("out_dtype must be", {"out_dtype": mx.int32}),
    # Lazy operands at the first refused row count; nothing is evaluated.
    "m_rows": (
        "fewer than 2",
        {
            "x8": mx.zeros((BIG_ROWS, 64), mx.int8),
            "x_scales": mx.ones((BIG_ROWS, 1), mx.float32),
        },
    ),
    "n_rows": (
        "at most 2",
        {
            "w8": mx.zeros(((1 << 29) + 1, 64), mx.int8),
            "w_scales": mx.ones(((1 << 29) + 1,), mx.float32),
        },
    ),
    "group_len": (
        "at most 133120",
        {"x8": mx.zeros((4, 133152), mx.int8), "w8": mx.zeros((8, 133152), mx.int8)},
    ),
    "group_len_grouped": (
        "at most 133120",
        {
            "x8": mx.zeros((4, 266304), mx.int8),
            "x_scales": mx.ones((4, 2), mx.float32),
            "w8": mx.zeros((8, 266304), mx.int8),
            "group_size": 133152,
        },
    ),
}


@pytest.mark.parametrize("name", sorted(QUANT_GUARDS))
def test_quantize_guard_refuses(name):
    match, change = QUANT_GUARDS[name]
    with pytest.raises(ValueError, match=match):
        kq.int8_quantize_rows(**{**_quant_args(), **change})


@pytest.mark.parametrize("name", sorted(MM_GUARDS))
def test_matmul_guard_refuses(name):
    match, change = MM_GUARDS[name]
    with pytest.raises(ValueError, match=match):
        kq.int8_matmul(**{**_mm_args(), **change})


@requires_nax
def test_limits_accept_their_bound():
    """Each size refusal starts one past its bound: 2^27 groups, N 2^29,
    and a 133120-column group build without error. Nothing is evaluated."""
    kq.int8_quantize_rows(mx.zeros((1 << 27, 32), mx.bfloat16))
    kq.int8_quantize_rows(mx.zeros((1 << 22, 1024), mx.bfloat16), group_size=32)
    n = 1 << 29
    kq.int8_matmul(
        **{**_mm_args(), "w8": mx.zeros((n, 64), mx.int8), "w_scales": mx.ones((n,))}
    )
    k = 133120
    kq.int8_matmul(
        **{
            **_mm_args(),
            "x8": mx.zeros((4, k), mx.int8),
            "w8": mx.zeros((8, k), mx.int8),
        }
    )


# On an M5 this runs with MLX_METAL_GPU_ARCH=applegpu_g15s, which names an
# M3-class GPU to MLX.
@pytest.mark.skipif(NAX, reason="refusal path of GPUs without NAX units.")
def test_refuses_without_nax():
    with pytest.raises(RuntimeError, match="NAX"):
        kq.int8_matmul(**_mm_args())
