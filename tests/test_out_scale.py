"""out_scale on quantized_matmul: the per-tensor scale of an NVFP4 GGUF.

The result is out_scale * (x @ W^T), with a scale of shape (1,) or (N,)
applied to the float32 sum before the one store, or in the split-K fold.
So on every route:

- a scale of ones gives the bits of no scale;
- a power-of-two scale (per column, 2^-10 to 2^-13 at random) gives the
  unscaled bits times the scale wherever the result is a normal number;
- real scales (5e-5 to 4.4e-4, the range of an Nvidia NVFP4 file) keep the
  float64 bound of test_fp4_scale_range, times the scale;
- float16 activations whose unscaled product passes 65504 give finite,
  in-bound results, split-K included under KQ_SPLITK_F32=0.

The LoRA delta is added after the scale, the gradient wrt x is
W^T (s * dy), KQuantLinear adds its bias after the scale, and the
dequantize route of GPUs before M5 scales in float32 after a bfloat16
GEMM. N 200 puts a partial tile at the end of every route, and N 2048
at K 4096 takes the default per-row qmv and verify_nax of NAX GPUs.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq
from mlx_kquant.nn import KQuantLinear

K = 512
FORCE_CPU = bool(os.environ.get("KQUANT_FORCE_CPU"))
TOL = {"bfloat16": 2.0**-7, "float16": 2.0**-10}
FLOOR = {"bfloat16": 2.0**-126, "float16": 2.0**-14}
ROUTES = ("qmv", "mv_ext", "verify_nax", "splitk", "nax", "nax_splitk")
MS = (1, 4, 8, 16, 64, 256)
Z = mx.zeros((1,), mx.uint8)


def _arms(bound, forced=True):
    """(name, env) per route arm. The CPU stream has one route; its NEON
    GEMV rounds activations to int8 by design, so only the bit checks run
    it. verify_nax_1 runs verify_nax in one split, which scales in the
    kernel; the default route does so at N 17408, K 5120."""
    if FORCE_CPU:
        neon = [] if bound else [("cpu_neon", {})]
        return [("cpu", {"KQ_CPU_NEON": "0"})] + neon
    strict = {"KQ_QMM_ROUTE_STRICT": "1"}
    arms = [("default", {}), ("alu", {"KQ_DISABLE_NAX": "1"})]
    if not forced:
        return arms
    arms += [(r, {"KQ_QMM_ROUTE": r, **strict}) for r in ROUTES]
    one = {"KQ_QMM_ROUTE": "verify_nax", "KQ_VERIFY_NAX_SPLITS": "1", **strict}
    return arms + [("verify_nax_1", one)]


@contextlib.contextmanager
def _env(env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _wire(n, lo, hi, seed=7, k=K):
    """nvfp4 rows of random codes with scale bytes lo to hi - 1."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, (n, k // 64, 36), dtype=np.uint8)
    w[:, :, :4] = rng.integers(lo, hi, (n, k // 64, 4), dtype=np.uint8)
    return w.reshape(n, -1)


def _w64(wire):
    w = kq.dequantize(mx.array(wire), Z, "nvfp4", dtype=mx.float32)
    return np.asarray(w).astype(np.float64)


def _x(M, dtn, mult=1.0, k=K):
    x = mx.random.normal((M, k), key=mx.random.key(M)) * mult
    x = x.astype(getattr(mx, dtn))
    mx.eval(x)
    return x, np.asarray(x.astype(mx.float32)).astype(np.float64)


def _col_scales(n, seed=3):
    rng = np.random.default_rng(seed)
    return np.exp2(rng.uniform(np.log2(5e-5), np.log2(4.4e-4), n)).astype(np.float32)


def _mm(x, w, s=None, **kw):
    out = kq.quantized_matmul(x, w, Z, "nvfp4", out_scale=s, **kw)
    mx.eval(out)
    return out


def _bits(a):
    return np.asarray(a.view(mx.uint16))


def _f64(a):
    return np.asarray(a.astype(mx.float32)).astype(np.float64)


def _routes(x, w, scales, check, bound=False, forced=True):
    """Runs each arm unscaled, then with each scale, and calls
    check(name, unscaled, [scaled...]). An arm that does not serve the
    unscaled call is skipped; one that serves it must serve the scaled ones."""
    ran = []
    for name, env in _arms(bound, forced):
        with _env(env):
            try:
                y0 = _mm(x, w)
            except (RuntimeError, ValueError):
                continue  # strict mode: the route does not serve this cell
            ys = [_mm(x, w, mx.array(s)) for s in scales]
        check(name, y0, ys)
        ran.append(name)
    return ran


def _ones_and_powers(dtn, N, M, k=K, forced=True):
    w = mx.array(_wire(N, 0x08, 0x48, k=k))
    x, _ = _x(M, dtn, k=k)
    # Random exponents, so an index off by a tile width reads a wrong value.
    pow2_col = np.exp2(-np.random.default_rng(5).integers(10, 14, N))
    scales = [
        np.ones(1, np.float32),
        np.ones(N, np.float32),
        np.full(1, 2.0**-12, np.float32),
        pow2_col.astype(np.float32),
    ]

    def check(name, y0, ys):
        for y in ys[:2]:
            assert np.array_equal(_bits(y), _bits(y0)), name
        u = np.asarray(y0.astype(mx.float32))
        for s, y in zip(scales[2:], ys[2:], strict=True):
            want = (mx.array(u * s)).astype(getattr(mx, dtn))
            normal = np.abs(np.asarray(want.astype(mx.float32))) >= 2 * FLOOR[dtn]
            got, exp = _bits(y), _bits(want)
            assert np.array_equal(got[normal], exp[normal]), name

    return _routes(x, w, scales, check, forced=forced)


@pytest.mark.parametrize("dtn", ("bfloat16", "float16"))
@pytest.mark.parametrize("N", (256, 200))
@pytest.mark.parametrize("M", MS)
def test_ones_and_powers_of_two(dtn, N, M):
    ran = _ones_and_powers(dtn, N, M)
    assert "default" in ran or "cpu" in ran
    # The forced routes are the small-M band.
    assert FORCE_CPU or M > 32 or len(ran) > 2, ran


@pytest.mark.parametrize("dtn", ("bfloat16", "float16"))
@pytest.mark.parametrize("M", (2, 4, 8))
def test_default_routes_on_wide_rows(dtn, M):
    ran = _ones_and_powers(dtn, 2048, M, k=4096, forced=False)
    assert "default" in ran or "cpu" in ran


@pytest.mark.parametrize("M", (1, 8, 256))
def test_scale_views_read_as_their_values(M):
    """A strided or lazy view of the scale gives the bits of the dense one."""
    N = 256
    w = mx.array(_wire(N, 0x08, 0x48))
    x, _ = _x(M, "bfloat16")
    col = (2.0 ** -(11 + np.arange(N) % 3)).astype(np.float32)
    want = _bits(_mm(x, w, mx.array(col)))
    strided = mx.array(np.repeat(col, 2))[::2]
    mx.eval(strided)
    reversed_ = mx.array(col[::-1].copy())[::-1]
    for view in (strided, reversed_, mx.array(np.repeat(col, 2))[1::2]):
        assert np.array_equal(_bits(_mm(x, w, view)), want)


def _bound_check(y, ref, bound, dtn, tol=None):
    got = _f64(y)
    assert np.isfinite(got).all()
    bad = np.abs(got - ref) > (tol or TOL[dtn]) * bound + FLOOR[dtn]
    assert not bad.any(), f"{int(bad.sum())} outputs past the bound"


@pytest.mark.parametrize("dtn", ("bfloat16", "float16"))
@pytest.mark.parametrize("N", (256, 200))
@pytest.mark.parametrize("M", MS)
def test_real_scales_against_float64(dtn, N, M):
    # Every scale byte in bfloat16. In float16 the bytes from 0x60 (32) to
    # 0x7E (448) and activations times 32 put most unscaled sums past
    # 65504, while the scaled ones stay near 100.
    f16 = dtn == "float16"
    wire = _wire(N, 0x60, 0x7F) if f16 else _wire(N, 0, 256)
    w64 = _w64(wire)
    w = mx.array(wire)
    x, x64 = _x(M, dtn, 32.0 if f16 else 1.0)
    ref, bound = x64 @ w64.T, np.abs(x64) @ np.abs(w64).T
    col = _col_scales(N)
    scales = [col[:1].copy(), col]

    def check(name, y0, ys):
        if f16 and M >= 4:
            assert np.isinf(_f64(y0)).any(), "the design no longer overflows"
        for s, y in zip(scales, ys, strict=True):
            _bound_check(y, ref * s, bound * s, dtn)

    _routes(x, w, scales, check, bound=True)


@pytest.mark.parametrize("M", (1, 8, 256))
def test_lora_delta_is_not_scaled(M):
    N, r = 256, 8
    wire = _wire(N, 0x30, 0x48)
    w64 = _w64(wire)
    x, x64 = _x(M, "bfloat16")
    a = (mx.random.normal((K, r), key=mx.random.key(11)) * 0.1).astype(mx.bfloat16)
    b = (mx.random.normal((r, N), key=mx.random.key(12)) * 0.1).astype(mx.bfloat16)
    s = 2.0**-12
    y = _mm(x, mx.array(wire), mx.full((1,), s), lora_a=a, lora_b=b)
    a64, b64 = _f64(a), _f64(b)
    ref = s * (x64 @ w64.T) + (x64 @ a64) @ b64
    bound = s * (np.abs(x64) @ np.abs(w64).T) + (np.abs(x64) @ np.abs(a64)) @ np.abs(
        b64
    )
    # A scaled delta would be 4096 times too small, an unscaled base 4096
    # times too large.
    _bound_check(y, ref, bound, "bfloat16", tol=2.0**-5)


@pytest.mark.parametrize("dtn", ("bfloat16", "float16"))
def test_gradient_wrt_x(dtn):
    # A cotangent of a training step's size: s * dy is about 1e-7, below the
    # float16 normal range, so a float16 vjp forms it in bfloat16.
    M, N = 4, 256
    wire = _wire(N, 0x30, 0x48)
    w, w64 = mx.array(wire), _w64(wire)
    x, _ = _x(M, dtn)
    s = _col_scales(N)
    c = (mx.random.normal((M, N), key=mx.random.key(5)) * 1e-3).astype(x.dtype)
    _, (g,) = mx.vjp(lambda x: _mm(x, w, mx.array(s)), [x], [c])
    assert g.dtype == x.dtype
    cs = _f64(c) * s
    _bound_check(g, cs @ w64, np.abs(cs) @ np.abs(w64), "bfloat16", tol=2.0**-6)
    with pytest.raises(ValueError, match="out_scale"):
        mx.vjp(lambda s: _mm(x, w, s), [mx.array(s)], [c])


def test_linear_adds_the_bias_after_the_scale():
    N = 256
    wire = _wire(N, 0x30, 0x48)
    w64 = _w64(wire)
    lin = KQuantLinear(K, N, bias=True, codec="nvfp4")
    lin.weight = mx.array(wire)
    lin.bias = mx.random.normal((N,), key=mx.random.key(9)).astype(mx.bfloat16)
    s = _col_scales(N)
    lin.out_scale = mx.array(s)
    x, x64 = _x(8, "bfloat16")
    b64 = _f64(lin.bias)
    ref = s * (x64 @ w64.T) + b64
    bound = s * (np.abs(x64) @ np.abs(w64).T) + np.abs(b64)
    with _env({"KQ_CPU_NEON": "0"}):
        _bound_check(lin(x), ref, bound, "bfloat16")


def test_rejects():
    w = mx.array(_wire(256, 0x30, 0x48))
    x, _ = _x(2, "bfloat16")
    q4 = kq.quantize(mx.random.normal((256, K), key=mx.random.key(1)), "q4_0")[0]
    with pytest.raises(ValueError, match="nvfp4"):
        kq.quantized_matmul(x, q4, Z, "q4_0", out_scale=mx.ones((1,)))
    with pytest.raises(ValueError, match="transpose"):
        kq.quantized_matmul(
            mx.zeros((2, 256), mx.bfloat16),
            w,
            Z,
            "nvfp4",
            transpose=False,
            out_scale=mx.ones((1,)),
        )
    for bad in (mx.ones((255,)), mx.ones((1, 256)), mx.ones((256,), mx.bfloat16)):
        with pytest.raises(ValueError, match="out_scale must be float32"):
            kq.quantized_matmul(x, w, Z, "nvfp4", out_scale=bad)


# The split-K partial dtype and the dequantize route are read once per
# process, so these run in a child with the lever set from the start.
_CHILD = r"""
import json, os, sys
import mlx.core as mx
import numpy as np
import mlx_kquant as kq
spec = json.loads(sys.argv[1])
Z = mx.zeros((1,), mx.uint8)
rng = np.random.default_rng(7)
N, K = spec["N"], spec["K"]
wire = rng.integers(0, 256, (N, K // 64, 36), dtype=np.uint8)
wire[:, :, :4] = rng.integers(0x60, 0x7F, (N, K // 64, 4), dtype=np.uint8)
w = mx.array(wire.reshape(N, -1))
w64 = np.asarray(kq.dequantize(w, Z, "nvfp4", dtype=mx.float32)).astype(np.float64)
s = np.exp2(rng.uniform(np.log2(5e-5), np.log2(4.4e-4), N)).astype(np.float32)
out = []
for i, c in enumerate(spec["calls"]):
    os.environ.update(c["env"])
    dt = getattr(mx, c["dtype"])
    x = (mx.random.normal((c["M"], K), key=mx.random.key(c["M"])) * 32).astype(dt)
    x64 = np.asarray(x.astype(mx.float32)).astype(np.float64)
    os.write(2, f"[call] {i}\n".encode())
    y0 = kq.quantized_matmul(x, w, Z, "nvfp4")
    mx.eval(y0)
    y = kq.quantized_matmul(x, w, Z, "nvfp4", out_scale=mx.array(s))
    mx.eval(y)
    os.write(2, b"[end]\n")
    got = np.asarray(y.astype(mx.float32)).astype(np.float64)
    ref = (x64 @ w64.T) * s
    bound = (np.abs(x64) @ np.abs(w64).T) * s
    err = np.abs(got - ref) / (bound + 1e-30)
    r = {"finite": bool(np.isfinite(got).all()), "err": float(np.nanmax(err)),
         "unscaled_finite": bool(np.isfinite(np.asarray(y0.astype(mx.float32))).all())}
    if c.get("grad"):
        cot = (mx.random.normal(y.shape, key=mx.random.key(5)) * 1e-4).astype(dt)
        f = lambda x: kq.quantized_matmul(x, w, Z, "nvfp4", out_scale=mx.array(s))
        _, (g,) = mx.vjp(f, [x], [cot])
        cs = np.asarray(cot.astype(mx.float32)).astype(np.float64) * s
        gg = np.asarray(g.astype(mx.float32)).astype(np.float64)
        bound = np.abs(cs) @ np.abs(w64) + 1e-30
        r["grad_err"] = float(np.max(np.abs(gg - cs @ w64) / bound))
        r["grad_dtype"] = str(g.dtype).split(".")[-1]
    out.append(r)
    for k in c["env"]:
        os.environ.pop(k, None)
print(json.dumps(out))
"""


def _child(spec, env):
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    child_env.update(env, KQ_QMM_LOG="1")
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(spec)],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    logs, cur = {}, None
    for line in r.stderr.splitlines():
        if line.startswith("[call] "):
            cur = int(line.split()[1])
            logs[cur] = []
        elif line == "[end]":
            cur = None
        elif cur is not None and line.startswith("[kq "):
            logs[cur].append(line)
    for i, r_ in enumerate(res):
        r_["log"] = logs.get(i, [])
    return res


@pytest.mark.skipif(FORCE_CPU, reason="GPU split-K routes")
def test_float16_split_partials_ignore_the_lever():
    # KQ_SPLITK_F32=0 stores float16 partials, which the unscaled sums
    # overflow to inf or, where the fold meets both signs, NaN. A call with
    # a scale stores float partials anyway.
    strict = {"KQ_QMM_ROUTE_STRICT": "1"}
    calls = []
    for route, ms in (
        ("splitk", (4, 16)),
        ("nax_splitk", (4, 16)),
        ("verify_nax", (2, 8)),
    ):
        for m in ms:
            calls.append(
                {"M": m, "dtype": "float16", "env": {"KQ_QMM_ROUTE": route, **strict}}
            )
    res = _child({"N": 256, "K": 4096, "calls": calls}, {"KQ_SPLITK_F32": "0"})
    for c, r in zip(calls, res, strict=True):
        what = (c["env"]["KQ_QMM_ROUTE"], c["M"])
        assert not r["unscaled_finite"], what
        assert r["finite"] and r["err"] <= TOL["float16"], (what, r["err"])
        splits = [
            x
            for x in r["log"]
            if "grid=" in x and not x.split("grid=")[1].endswith(",1")
        ]
        names = [x.split()[2] for x in r["log"] if x.startswith("[kq qmm]")]
        # The unscaled call, then the scaled one.
        assert len(names) == 2, r["log"]
        assert not names[0].endswith("_pf32") and names[1].endswith("_pf32"), names
        assert splits, r["log"]


@pytest.mark.skipif(FORCE_CPU, reason="GPU route")
def test_dequantize_route():
    # The route's floor for nvfp4: M 2048, K and N 3840.
    calls = [
        {"M": 2048, "dtype": "bfloat16", "env": {}, "grad": True},
        {"M": 2048, "dtype": "float16", "env": {}, "grad": True},
    ]
    res = _child({"N": 3840, "K": 3840, "calls": calls}, {"KQ_DEQUANT_ROUTE": "force"})
    for c, r in zip(calls, res, strict=True):
        assert sum("[kq dequant]" in x for x in r["log"]) == 2, r["log"]
        # float16 runs the GEMM in bfloat16 and rounds to float16 after.
        tol = TOL["bfloat16"] + (TOL["float16"] if c["dtype"] == "float16" else 0)
        assert r["finite"] and r["err"] <= tol, (c["dtype"], r["err"])
    assert not res[1]["unscaled_finite"]
    for c, r in zip(calls, res, strict=True):
        assert r["grad_dtype"] == c["dtype"], r["grad_dtype"]
        assert r["grad_err"] <= 2.0**-6, r["grad_err"]
