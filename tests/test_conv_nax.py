"""conv2d_3x3: y = conv(x, w) + bias, 3x3, stride 1, padding 1, channels last.

On a NAX GPU the op runs one implicit GEMM kernel for bfloat16 with C a
multiple of 16 from 1024 output pixels: f32 sums over the 9 taps and C
channels, the bias added in f32, one rounding. Everywhere else it builds
MLX's convolution on the flipped kernel (its implicit GEMM, no Winograd
transform) and a stock add of the bias.

- R1: relative Frobenius error against the float64 convolution, at most
  6e-3 on bfloat16 and 1e-3 on float16.
- R2: on the kernel, the error sits at the one rounding of y: at most 1.05
  times the error of the float64 result rounded to bfloat16, and each
  element within ULPS ulp of bfloat16 at the float64 result plus 2^-20
  times the sum of |x w| over its taps and channels.
- Where the op has no kernel, y is the composition a caller writes, bit for
  bit.

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
from test_operand_views import _dense

import mlx_kquant as kq

R1 = {mx.bfloat16: 6e-3, mx.float16: 1e-3}
CPU_REL = 5e-2
MIN_ROWS = 1024
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
# Tiles of the metallib: the two the op picks, and two for KQ_CONV_TILE.
TILES = [(128, 128, 4, 4), (128, 64, 4, 2), (64, 128, 2, 4), (64, 64, 2, 2)]
# R2 per element: the ulp count and the allowance for the f32 sums per unit
# of sum |x w|. Bfloat16 holds 8 significant bits, its smallest normal has
# np.frexp exponent -125.
ULPS = 2
SUM_TOL = 2.0**-20


def _nax():
    return not force_cpu() and kq.nax_available()


def _fused(pixels, c, dtype=mx.bfloat16):
    return _nax() and dtype == mx.bfloat16 and c % 16 == 0 and pixels >= MIN_ROWS


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _inputs(b, h, w, c, o, dtype=mx.bfloat16, bias=True, seed=0):
    x = mx.random.normal((b, h, w, c), key=mx.random.key(seed + 1)).astype(dtype)
    wt = mx.random.normal((o, 3, 3, c), key=mx.random.key(seed + 2)) / np.sqrt(9 * c)
    wt = wt.astype(dtype)
    bs = (
        mx.random.normal((o,), key=mx.random.key(seed + 3)).astype(dtype)
        if bias
        else None
    )
    mx.eval(x, wt, *(() if bs is None else (bs,)))
    return x, wt, bs


def _ref(x, w, bias, mag=False):
    """The float64 convolution, pixels flattened. mag=True gives sum |x w|
    per element, without the bias."""
    xd, wd = _f64(x), _f64(w)
    if mag:
        xd, wd, bias = np.abs(xd), np.abs(wd), None
    b, h, wid, c = xd.shape
    o = wd.shape[0]
    xp = np.pad(xd, ((0, 0), (1, 1), (1, 1), (0, 0)))
    taps = [xp[:, dy : dy + h, dx : dx + wid, :] for dy in range(3) for dx in range(3)]
    y = np.concatenate(taps, axis=-1).reshape(-1, 9 * c) @ wd.reshape(o, 9 * c).T
    if bias is not None:
        y = y + _f64(bias)
    return y


def _ulps(got, ref, mag):
    """The largest error in ulp of bfloat16 at |ref|, after the allowance of
    SUM_TOL * mag for the f32 sums, and its pixel and column."""
    e = np.frexp(np.abs(ref))[1]
    e = np.where(ref == 0, -125, np.maximum(e, -125))
    u = (np.abs(got - ref) - SUM_TOL * mag) / np.ldexp(1.0, e - 8)
    i = np.unravel_index(int(np.argmax(u)), u.shape)
    return float(u[i]), tuple(int(v) for v in i)


def _composed(x, w, bias):
    rev = mx.array([2, 1, 0])
    wf = mx.take(mx.take(w.astype(x.dtype), rev, axis=1), rev, axis=2)
    y = mx.conv_general(x, wf, stride=1, padding=1, flip=True)
    if bias is not None:
        y = y + bias.astype(x.dtype)
    return y


def _check(x, w, bias):
    """R1, and R2 on the kernel. Returns (R1 error, R2 ratio or None)."""
    b, h, wid, c = x.shape
    o = w.shape[0]
    y = kq.conv2d_3x3(x, w, bias=bias)
    mx.eval(y)
    assert y.shape == (b, h, wid, o) and y.dtype == x.dtype
    got = _f64(y).reshape(-1, o)
    assert np.isfinite(got).all()
    ref = _ref(x, w, bias)
    err = _rel(got, ref)
    if not _fused(b * h * wid, c, x.dtype):
        comp = _composed(x, w, bias)
        mx.eval(comp)
        assert np.array_equal(_f64(comp).reshape(-1, o), got)
        assert err <= (R1[x.dtype] if _nax() else CPU_REL), err
        return err, None
    assert err <= R1[x.dtype], err
    floor = _rel(_f64(mx.array(ref.astype(np.float32)).astype(x.dtype)), ref)
    assert err <= 1.05 * floor + 1e-5, (err, floor)
    worst = _ulps(got, ref, _ref(x, w, bias, mag=True))
    assert worst[0] <= ULPS, worst
    return err, err / floor


# The channel counts of the Z-Image (128, 256, 512) and Qwen-Image 2.1
# (144, 288, 576, 1152) VAE decoders, and the small and ragged ones.
SHAPES = [
    (1, 37, 53, 128, 128),
    (1, 32, 40, 256, 256),
    (1, 24, 24, 512, 512),
    (1, 33, 29, 144, 144),
    (1, 30, 31, 288, 144),
    (1, 16, 18, 576, 288),
    (1, 12, 12, 1152, 1152),
    (1, 40, 26, 32, 64),
    (1, 64, 64, 128, 3),
    (1, 33, 31, 16, 100),
]


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_conv_shapes(shape, bias):
    _check(*_inputs(*shape, bias=bias))


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("hw", [(31, 33), (16, 32), (1, 1100), (1100, 1)])
def test_conv_batch_and_thin_images(hw, batch):
    # One row or one column of pixels: every tap but the middle one falls
    # outside the image on one axis.
    _check(*_inputs(batch, *hw, 64, 96, seed=batch))


@pytest.mark.parametrize("pixels", [1023, 1024, 1025])
def test_conv_pixel_bound(pixels):
    _check(*_inputs(1, 1, pixels, 32, 64))


def test_conv_pixels_above_32767():
    # Row offsets pass a 16-bit index: every pixel is compared.
    _check(*_inputs(1, 200, 180, 32, 64))
    _check(*_inputs(2, 129, 130, 16, 32))


@pytest.mark.parametrize("c", [8, 24, 40])
def test_conv_channels_off_16_compose(c):
    err, r2 = _check(*_inputs(1, 40, 40, c, 64))
    assert r2 is None


def test_conv_float16_and_float32_compose():
    for dt in (mx.float16, mx.float32):
        x, w, b = _inputs(1, 40, 40, 64, 64, dtype=dt)
        y = kq.conv2d_3x3(x, w, bias=b)
        want = _composed(x, w, b)
        mx.eval(y, want)
        assert y.dtype == dt
        assert mx.array_equal(y, want).item()


def test_conv_weight_and_bias_run_in_the_dtype_of_x():
    x, w, b = _inputs(1, 40, 40, 64, 64)
    y = kq.conv2d_3x3(x, w.astype(mx.float32), bias=b.astype(mx.float32))
    want = kq.conv2d_3x3(x, w, bias=b)
    mx.eval(y, want)
    assert y.dtype == mx.bfloat16
    assert mx.array_equal(y, want).item()


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
def test_conv_operand_views(state):
    """A view of x, w or bias gives the bits of its dense copy, lazy or
    evaluated (one rounding apart on the CPU stream)."""
    x, w, b = _inputs(1, 40, 36, 64, 96)
    x2 = mx.random.normal((1, 36, 40, 128), key=mx.random.key(9)).astype(mx.bfloat16)
    w2 = (
        mx.random.normal((96, 3, 3, 128), key=mx.random.key(10)).astype(mx.bfloat16)
        * 0.05
    )
    b2 = mx.random.normal((192,), key=mx.random.key(11)).astype(mx.bfloat16)
    mx.eval(x2, w2, b2)
    views = {
        "x_transposed": lambda: (mx.swapaxes(x2[..., :64], 1, 2), w, b),
        "x_strided": lambda: (mx.swapaxes(x2, 1, 2)[..., 1::2], w, b),
        "w_strided": lambda: (x, w2[..., 0::2], b),
        "bias_strided": lambda: (x, w, b2[1::2]),
    }
    for name, make in views.items():
        xv, wv, bv = make()
        want = kq.conv2d_3x3(_dense(xv), _dense(wv), bias=_dense(bv))
        if state == "evaluated":
            mx.eval(xv, wv, bv)
        got = kq.conv2d_3x3(xv, wv, bias=bv)
        mx.eval(got, want)
        if force_cpu():
            assert _rel(_f64(got), _f64(want)) <= R1[mx.bfloat16], name
        else:
            assert mx.array_equal(got, want).item(), name
    xv = mx.swapaxes(x2[..., :64], 1, 2)
    _check(xv, w, b)


@pytest.mark.parametrize("bias", [True, False])
def test_conv_gradient_is_the_gradient_of_the_composition(bias):
    x, w, b = _inputs(1, 40, 36, 32, 48, bias=bias)
    cot = mx.random.normal((1, 40, 36, 48), key=mx.random.key(5))

    def op(*a):
        return (
            kq.conv2d_3x3(a[0], a[1], bias=a[2] if bias else None).astype(mx.float32)
            * cot
        ).sum()

    def comp(*a):
        return (
            _composed(a[0], a[1], a[2] if bias else None).astype(mx.float32) * cot
        ).sum()

    args = (x, w, b) if bias else (x, w)
    nums = tuple(range(len(args)))
    got = mx.grad(op, argnums=nums)(*args)
    want = mx.grad(comp, argnums=nums)(*args)
    mx.eval(got, want)
    for g, v in zip(got, want, strict=True):
        assert mx.array_equal(g, v).item()


def _outcome(fn):
    try:
        out = fn()
        mx.eval(out)
        return out, None
    except Exception as e:  # noqa: BLE001
        return None, type(e)


def test_conv_jvp_and_vmap_follow_the_composition():
    """jvp and vmap give the composition's result, or raise where it does."""
    x, w, b = _inputs(1, 40, 36, 32, 48)
    tx = mx.random.normal(x.shape, key=mx.random.key(6)).astype(x.dtype)
    got, gerr = _outcome(
        lambda: mx.jvp(lambda x_: kq.conv2d_3x3(x_, w, bias=b), [x], [tx])[1]
    )
    want, werr = _outcome(lambda: mx.jvp(lambda x_: _composed(x_, w, b), [x], [tx])[1])
    assert gerr == werr
    if got is not None:
        assert mx.array_equal(got[0], want[0]).item()
    xs = mx.random.normal((2, 1, 40, 36, 32), key=mx.random.key(7)).astype(x.dtype)
    got, gerr = _outcome(lambda: mx.vmap(lambda x_: kq.conv2d_3x3(x_, w, bias=b))(xs))
    want, werr = _outcome(lambda: mx.vmap(lambda x_: _composed(x_, w, b))(xs))
    assert gerr == werr
    if got is not None:
        assert mx.array_equal(got, want).item()


def test_conv_compiles_and_repeats():
    x, w, b = _inputs(1, 40, 36, 64, 64)
    f = mx.compile(lambda x_, w_, b_: kq.conv2d_3x3(x_, w_, bias=b_))
    want = kq.conv2d_3x3(x, w, bias=b)
    for _ in range(2):
        got = f(x, w, b)
        mx.eval(got, want)
        assert mx.array_equal(got, want).item()


def test_conv_empty_batch():
    x, w, b = _inputs(1, 4, 4, 32, 32)
    y = kq.conv2d_3x3(mx.zeros((0, 4, 4, 32), dtype=mx.bfloat16), w, bias=b)
    mx.eval(y)
    assert y.shape == (0, 4, 4, 32)


def test_conv_argument_errors():
    x, w, b = _inputs(1, 8, 8, 32, 16)

    def bad(match, x_=x, w_=w, **kw):
        with pytest.raises(ValueError, match=r"conv2d_3x3\] .*" + match):
            kq.conv2d_3x3(x_, w_, **kw)

    bad("x must be", x_=x[0])
    bad("x must be", x_=x.astype(mx.int32))
    bad("w must be", w_=w[..., :-1])
    bad("w must be", w_=w[:, :2])
    bad("w must be", w_=w.reshape(16, 9, 32))
    bad("w must be", w_=w.astype(mx.int32))
    bad("bias must be", bias=b[:-1])
    bad("bias must be", bias=b.reshape(1, 16))
    bad("bias must be", bias=b.astype(mx.int32))


_PROBE = r"""
import json, os, sys
spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["tests_dir"])
import mlx.core as mx
import mlx_kquant as kq
import test_conv_nax as t

out = []
for i, c in enumerate(spec["calls"]):
    os.environ.update(c.get("env", {}))
    x, w, b = t._inputs(*c["shape"], bias=c.get("bias", True))
    print(f"[probe] {i}", file=sys.stderr, flush=True)
    res = {"error": None}
    try:
        y = kq.conv2d_3x3(x, w, bias=b)
        mx.eval(y)
        got = t._f64(y).reshape(-1, c["shape"][-1])
        ref = t._ref(x, w, b)
        res["rel"] = t._rel(got, ref)
        res["floor"] = t._rel(
            t._f64(mx.array(ref.astype(t.np.float32)).astype(mx.bfloat16)), ref)
        res["ulps"] = t._ulps(got, ref, t._ref(x, w, b, mag=True))
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    out.append(res)
print("RESULT " + json.dumps(out))
"""


def _probe(calls, env=None):
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    child_env.update({"KQ_QMM_LOG": "1", **(env or {})})
    p = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            json.dumps({"tests_dir": TESTS_DIR, "calls": calls}),
        ],
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
            cur["kernels"] = []
        elif cur is not None and ln.startswith("[kq qmm] "):
            cur["kernels"].append(ln.split()[2])
    return results


def _name(tile):
    bm, bn, wm, wn = tile
    return f"kq_conv3x3_nax_bfloat16_t_bm{bm}_bn{bn}_wm{wm}_wn{wn}"


def test_conv_kernels_in_metallib():
    if force_cpu():
        pytest.skip("Metal kernel names only")
    names = kq.metallib_kernels()
    if not names:
        pytest.skip("no metallib loaded")
    got = {n for n in names if n.startswith("kq_conv3x3_nax_")}
    assert got == {_name(t) for t in TILES}, sorted(got)


def test_conv_known_answer_of_every_kernel():
    """Each kernel of the metallib, under KQ_CONV_TILE, against the float64
    convolution and the one-rounding floor, with O on and off its column
    tile and C off a multiple of 32."""
    if not _nax():
        pytest.skip("the kernel needs a NAX GPU")
    calls = []
    for tile in TILES:
        env = {"KQ_CONV_TILE": ",".join(map(str, tile))}
        for shape in ((1, 37, 41, 144, 200), (2, 24, 26, 64, 128)):
            calls.append({"shape": list(shape), "env": env})
    res = _probe(calls, env={"KQ_TUNE_LIVE": "1"})
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        tile = tuple(int(v) for v in c["env"]["KQ_CONV_TILE"].split(","))
        assert r["kernels"] == [_name(tile)], (c, r["kernels"])
        assert r["rel"] <= 1.05 * r["floor"] + 1e-5, (c, r["rel"], r["floor"])
        assert r["ulps"][0] <= ULPS, (c, r["ulps"])


def test_conv_tile_choice_and_levers():
    """By default the 128 column tile runs, or the 64 one where it pads O
    less; KQ_CONV_NAX=0 composes, 2 runs the kernel below 1024 pixels, and
    a tile the metallib lacks raises with its name."""
    if not _nax():
        pytest.skip("the kernel needs a NAX GPU")
    calls = [
        ({"shape": [1, 40, 40, 64, 128]}, _name((128, 128, 4, 4))),
        ({"shape": [1, 40, 40, 64, 144]}, _name((128, 64, 4, 2))),
        ({"shape": [1, 40, 40, 64, 256]}, _name((128, 128, 4, 4))),
        ({"shape": [1, 40, 40, 64, 288]}, _name((128, 64, 4, 2))),
        ({"shape": [1, 10, 10, 64, 128]}, None),
        ({"shape": [1, 10, 10, 64, 128], "env": {"KQ_CONV_NAX": "2"}}, _name(TILES[0])),
        ({"shape": [1, 40, 40, 64, 128], "env": {"KQ_CONV_NAX": "0"}}, None),
        ({"shape": [1, 40, 40, 64, 128], "env": {"KQ_CONV_NAX": "1"}}, _name(TILES[0])),
        (
            {"shape": [1, 40, 40, 64, 128], "env": {"KQ_CONV_TILE": "32,32,1,1"}},
            "raise",
        ),
    ]
    res = _probe([c for c, _ in calls], env={"KQ_TUNE_LIVE": "1"})
    for (c, want), r in zip(calls, res, strict=True):
        if want == "raise":
            assert (
                r["error"] is not None
                and "kq_conv3x3_nax_bfloat16_t_bm32" in r["error"]
            )
            continue
        assert r["error"] is None, (c, r["error"])
        assert r["kernels"] == ([] if want is None else [want]), (c, r["kernels"])
