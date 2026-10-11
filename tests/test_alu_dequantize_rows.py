"""The route that runs a plain quantized_matmul as dequantize then the stock
matmul on GPUs without NAX before M5.

``_alu_dequantize_row`` answers for any architecture string, so the policy
cases run on every machine. The GPU cases run in a fresh interpreter
(tests/qmm_probe.py and the children below) under ``KQ_QMM_LOG``: a routed
call logs a ``[kq dequant]`` line and no qmm kernel. Where this GPU's class
has no row, ``KQ_DEQUANT_ROUTE=force`` runs the route, so every machine runs
its code. Each numeric pair holds the route against the qmm_t kernel under
``KQ_DEQUANT_ROUTE=off`` and ``KQ_DISABLE_NAX=1``, and both against float64
with exact ``kq.dequantize`` weights. Under ``KQUANT_FORCE_CPU`` the GPU
cases skip.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import mlx.core as mx
import pytest
from kqref import is_synth
from mlx_kquant._ext import _alu_dequantize_row, _dequantize_route_takes
from qmm_probe import TESTS_DIR, force_cpu, run_probe

import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

R1 = {"bfloat16": 6e-3, "float16": 1e-3, "float32": 6e-3}
# Largest ulp distance to the kernel on the elements above 1% of the
# output's root mean square, by output dtype.
ULP = {"bfloat16": 1, "float16": 2}
LIVE = {"KQ_TUNE_LIVE": "1", "KQ_QMM_LOG": "1"}

# Codec -> floor in the rows of x (measured on M3 Max).
ROWS = {
    "nvfp4": 2048,
    "ptq1_0": 2048,
    "q3_k": 2048,
    "q4_k": 16384,
    "q5_k": 2048,
    "q6_k": 2048,
}
MIN_K, MAX_K = 3840, 12288
MIN_N, MAX_N = 3840, 24576

# Architecture strings with a row (generation 13 to 16, any suffix letter
# but p) and without one (phones, M5 and later, earlier generations,
# malformed).
ROW_CLASS = [f"applegpu_g{g}{c}" for g in (13, 14, 15, 16) for c in "gsdc"]
NO_ROW_CLASS = [f"applegpu_g{g}p" for g in (13, 14, 15, 16, 17, 18)] + [
    "applegpu_g17g",
    "applegpu_g17s",
    "applegpu_g17d",
    "applegpu_g18s",
    "applegpu_g12s",
    "applegpu_g99s",
    "applegpu_g15",
    "applegpu_g15sx",
    "applegpu_g15S",
    "applegpu_g015s",
    "applegpu_g5s",
    "applegpu_gxxs",
    "applegpu_15s",
    "",
    "unknown",
]

MATMUL_CODECS = sorted(c for c in CODEC_GEOMETRY if kq.codec_has_matmul(c))


def _metal():
    if force_cpu():
        pytest.skip("Metal kernel logs only")


def _device_class():
    """(architecture string with its suffix, nax) of this GPU."""
    return mx.device_info().get("architecture", ""), kq.nax_available()


def _row_here(codec):
    arch, nax = _device_class()
    return bool(_alu_dequantize_row(arch, nax, codec, ROWS[codec], MIN_N, MIN_K))


def _route_env(codec):
    # The route: by default where this GPU's class has the row, else forced.
    return {
        "KQ_QMM_TILE": None,
        "KQ_DISABLE_NAX": None,
        "KQ_DEQUANT_ROUTE": None if _row_here(codec) else "force",
    }


# The qmm_t kernel without NAX.
KERNEL_ENV = {"KQ_QMM_TILE": None, "KQ_DISABLE_NAX": "1", "KQ_DEQUANT_ROUTE": "off"}


def _routed(r):
    lines = [x for x in r["stderr"] if x.startswith("[kq dequant] ")]
    assert len(lines) <= 1, r["stderr"]
    assert bool(lines) != bool(r["kernels"]), (lines, r["kernels"])
    return bool(lines)


@pytest.mark.parametrize("codec", MATMUL_CODECS)
def test_rows_by_chip_class(codec):
    # A row runs on the classes with one, from its floor in M, inside the K
    # and N range, and nowhere else: not on NAX GPUs, phones, M5 and later,
    # earlier or malformed strings, or for a codec without a row.
    if codec in ROWS:
        min_m = ROWS[codec]
        row = [min_m, MIN_K, MAX_K, MIN_N, MAX_N]
        outside = [
            (1, MIN_N, MIN_K),
            (512, MIN_N, MIN_K),
            (min_m - 1, MIN_N, MIN_K),
            (min_m, MIN_N - 64, MIN_K),
            (min_m, MAX_N + 64, MIN_K),
            (min_m, 248320, MIN_K),
            (min_m, MIN_N, MIN_K - 256),
            (min_m, MIN_N, MAX_K + 256),
            (min_m, MIN_N, 256),
        ]
        for arch in ROW_CLASS:
            for m in (min_m, min_m + 1, 32769):
                for n in (MIN_N, 4096, MAX_N):
                    for k in (MIN_K, 4096, MAX_K):
                        got = _alu_dequantize_row(arch, False, codec, m, n, k)
                        assert got == row, (arch, m, n, k, got)
            assert _alu_dequantize_row(arch, True, codec, min_m, MIN_N, MIN_K) == []
            for m, n, k in outside:
                got = _alu_dequantize_row(arch, False, codec, m, n, k)
                assert got == [], (arch, m, n, k, got)
    else:
        for arch in ROW_CLASS:
            for m in (2048, 16384, 32769):
                got = _alu_dequantize_row(arch, False, codec, m, MIN_N, MIN_K)
                assert got == [], (arch, m, got)
    for arch in NO_ROW_CLASS:
        for nax in (False, True):
            for m in (2048, 16384, 32769):
                got = _alu_dequantize_row(arch, nax, codec, m, MIN_N, MIN_K)
                assert got == [], (arch, nax, got)


MODES = ["auto", "off", "force"]


@pytest.mark.parametrize("codec", sorted(ROWS) + ["q8_0"])
def test_route_decision_by_mode(codec):
    # The op's decision: off and a pinned KQ_QMM_TILE never route, force
    # routes on any GPU, auto only on the classes with a row, and each only
    # for a codec row at a shape inside its floor and range.
    min_m = ROWS.get(codec, 2048)
    inside = (min_m, MIN_N, MIN_K)
    outside = [
        (min_m - 1, MIN_N, MIN_K),
        (min_m, MAX_N + 64, MIN_K),
        (min_m, MIN_N, MAX_K + 256),
    ]
    classes = [(a, False, True) for a in ROW_CLASS] + [
        ("applegpu_g15s", True, False),
        ("applegpu_g17s", True, False),
        ("applegpu_g17s", False, False),
        ("applegpu_g16p", False, False),
        ("unknown", False, False),
    ]
    for arch, nax, on_class in classes:
        assert bool(_alu_dequantize_row(arch, nax, codec, *inside)) == (
            on_class and codec in ROWS
        ), arch
        for mode in MODES:
            for pinned in (False, True):
                want = (
                    codec in ROWS
                    and not pinned
                    and (mode == "force" or (mode == "auto" and on_class))
                )
                got = _dequantize_route_takes(arch, nax, mode, pinned, codec, *inside)
                assert got == want, (arch, nax, mode, pinned, got)
                for shape in outside:
                    got = _dequantize_route_takes(
                        arch, nax, mode, pinned, codec, *shape
                    )
                    assert not got, (arch, nax, mode, pinned, shape)
    with pytest.raises(ValueError, match="auto, off or force"):
        _dequantize_route_takes("applegpu_g15s", False, "on", False, codec, *inside)


def _call(codec, shape, n, env, **kw):
    return dict({"codec": codec, "shape": shape, "n": n, "env": env}, **kw)


@pytest.mark.parametrize("codec", sorted(ROWS))
def test_dispatch_on_this_gpu(codec):
    # Under the route's env a call takes the route from the floor and inside
    # the K and N range, and a batch of two counts its rows together. Below
    # the floor, outside the range, under off and with KQ_QMM_TILE set the
    # kernel runs. By default the route runs where this GPU's class has the
    # row. A value the lever does not take warns and reads as auto.
    _metal()
    min_m = ROWS[codec]
    here = _row_here(codec)
    route = _route_env(codec)
    pin = {"KQ_QMM_TILE": "64,64,2,2"}
    cases = [
        ([1, min_m - 32, MIN_K], MIN_N, route, False),
        ([1, min_m, MIN_K], MIN_N, route, True),
        ([2, min_m // 2, MIN_K], MIN_N, route, True),
        ([1, min_m, MIN_K], MIN_N - 64, route, False),
        ([1, min_m, MIN_K], MAX_N, route, True),
        ([1, min_m, MIN_K], MAX_N + 64, route, False),
        ([1, min_m, MIN_K - 256], MIN_N, route, False),
        ([1, min_m, MAX_K], MIN_N, route, True),
        ([1, min_m, MAX_K + 256], MIN_N, route, False),
        ([1, min_m, MIN_K], MIN_N, pin, False),
        ([1, min_m, MIN_K], MIN_N, dict(route, KQ_DEQUANT_ROUTE="off"), False),
        ([1, min_m, MIN_K], MIN_N, dict(route, KQ_DEQUANT_ROUTE="Force"), True),
        ([1, min_m, MIN_K], MIN_N, dict(route, KQ_DEQUANT_ROUTE=None), here),
        ([1, min_m, MIN_K], MIN_N, dict(route, KQ_DEQUANT_ROUTE="auto"), here),
        ([1, min_m, MIN_K], MIN_N, dict(route, KQ_DEQUANT_ROUTE="on"), here),
    ]
    # The calls check no values, so the quantized codecs take 64 quantized
    # rows tiled to N, which spares quantizing the widest weights.
    design = "normal" if is_synth(codec) else "tiled"
    calls = [_call(codec, shape, n, env, design=design) for shape, n, env, _ in cases]
    res = run_probe(calls, env=LIVE, check=False)
    for c, (*_, want), r in zip(calls, cases, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert _routed(r) == want, (c, r["kernels"], r["stderr"])
        for name, _ in r["kernels"]:
            assert "_qmm_t_" in name, (c, name)
    warned = [x for x in res[-1]["stderr"] if "KQ_DEQUANT_ROUTE=on is not" in x]
    assert len(warned) == 1, res[-1]["stderr"]


@pytest.mark.parametrize("codec", sorted(ROWS))
def test_nax_route_unchanged(codec):
    # On NAX GPUs a call at the route's sizes runs the kernel it runs under
    # KQ_DEQUANT_ROUTE=off, bit for bit, by default and under
    # KQ_DISABLE_NAX=1.
    _metal()
    if not kq.nax_available():
        pytest.skip("NAX GPU")
    calls = []
    for rows in (ROWS[codec], 32769):
        for nax_off in (None, "1"):
            for mode in (None, "off"):
                env = {"KQ_DISABLE_NAX": nax_off, "KQ_DEQUANT_ROUTE": mode}
                calls.append(_call(codec, [1, rows, MIN_K], MIN_N, env, seed=rows))
    res = run_probe(calls, env=LIVE, check=False)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert not _routed(r), (c, r["stderr"])
        ((name, _),) = r["kernels"]
        assert "_qmm_t_" in name, name
        if c["env"]["KQ_DISABLE_NAX"]:
            assert "_nax_" not in name, name
    for i in range(0, len(res), 2):
        assert res[i]["kernels"] == res[i + 1]["kernels"], (calls[i], calls[i + 1])
        assert res[i]["digest"] == res[i + 1]["digest"], (calls[i], calls[i + 1])


# The float64 and ulp checks read every row, 1024 rows at a time, so a
# check holds float64 copies of 1024 rows.
CHECK_BLOCK = 1024


def _pair(codec, shape, dtype, layout, seed):
    # The route, then the kernel on the same inputs, its ulp held to the
    # route's.
    route = _call(codec, shape, MIN_N, _route_env(codec), dtype=dtype, layout=layout)
    route.update(seed=seed, check_block=CHECK_BLOCK)
    return [route, dict(route, env=KERNEL_ENV)]


def _with_ulp(calls):
    for i in range(1, len(calls), 2):
        calls[i]["ulp_vs"] = i - 1
    return calls


def _check_pairs(calls, res, dtype):
    out_dtype = "bfloat16" if dtype == "float32" else dtype
    for i in range(0, len(calls), 2):
        c, r, k = calls[i], res[i], res[i + 1]
        assert r["error"] is None and k["error"] is None, (c, r["error"], k["error"])
        assert _routed(r), (c, r["kernels"])
        assert not _routed(k), (c, k["stderr"])
        ((name, _),) = k["kernels"]
        assert "_qmm_t_" in name and "_nax_" not in name, name
        for x in (r, k):
            assert x["checked"] == c["shape"][0] * c["shape"][1], (c, x["checked"])
            assert x["rel"] < R1[dtype], (c, x["rel"])
            assert x["nonfinite"] == 0, c
        # The route's error against float64 is the kernel's.
        assert abs(r["rel"] - k["rel"]) <= 0.01 * k["rel"], (c, r["rel"], k["rel"])
        assert k["ulp_big"] <= ULP[out_dtype], (c, k["ulp_big"])
        assert k["rdiff"] < 1e-4, (c, k["rdiff"])


@pytest.mark.parametrize("layout", ["dense", "transposed", "strided"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("codec", sorted(ROWS))
def test_route_numerics(codec, dtype, layout):
    # The floor in one batch and in two, every row checked: for q4_k all
    # 16384 rows, row 8192 where the two batches meet among them.
    _metal()
    min_m = ROWS[codec]
    calls = _with_ulp(
        _pair(codec, [1, min_m, MIN_K], dtype, layout, 0)
        + _pair(codec, [2, min_m // 2, MIN_K], dtype, layout, 1)
    )
    _check_pairs(calls, run_probe(calls, env=LIVE), dtype)


@pytest.mark.parametrize("codec", sorted(ROWS))
def test_route_float32_x(codec):
    # float32 x runs as bfloat16, on the route as on the kernel.
    _metal()
    calls = _with_ulp(_pair(codec, [1, ROWS[codec], MIN_K], "float32", "dense", 0))
    _check_pairs(calls, run_probe(calls, env=LIVE), "float32")


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("codec", sorted(ROWS))
def test_route_k_max(codec, dtype):
    _metal()
    calls = _with_ulp(_pair(codec, [1, ROWS[codec], MAX_K], dtype, "dense", 0))
    _check_pairs(calls, run_probe(calls, env=LIVE), dtype)


@pytest.mark.parametrize("codec", sorted(ROWS))
def test_route_rows_above_int16(codec):
    # 32769 rows on the route against the kernel and float64, every row
    # checked, so rows 32767 and 32768 past the int16 range among them.
    _metal()
    calls = _with_ulp(_pair(codec, [1, 32769, MIN_K], "bfloat16", "dense", 0))
    _check_pairs(calls, run_probe(calls, env=LIVE), "bfloat16")


# Children for the calls run_probe does not make. Each prints one JSON line;
# stderr carries the KQ_QMM_LOG lines after a "[case] <name>" marker.
_PRELUDE = r"""
import json, os, sys
import numpy as np
spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["tests_dir"])
import mlx.core as mx
import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY
from kqref import is_synth, synth_wire


def weights(codec, n, k, seed=0):
    _, _, bpb, wpb = CODEC_GEOMETRY[codec]
    if is_synth(codec):
        rng = np.random.default_rng(n * 131 + k + seed)
        wire = synth_wire(rng, codec, bpb, n * (k // wpb))
        w = mx.array(wire.reshape(n, (k // wpb) * bpb))
        s = mx.zeros((1,), dtype=mx.uint8)
    else:
        wf = mx.random.normal((n, k), key=mx.random.key(n * 7 + k + seed)) * 0.1
        w, s = kq.quantize(wf, codec)
    mx.eval(w, s)
    return w, s


def case(name, env):
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    os.write(2, f"[case] {name}\n".encode())


def rel(a, b):
    a = np.array(a.astype(mx.float32)).astype(np.float64)
    b = np.array(b.astype(mx.float32)).astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
"""

_GRAD_CHILD = (
    _PRELUDE
    + r"""
import time

codec, m, n, k = spec["codec"], spec["m"], spec["n"], spec["k"]
w, s = weights(codec, n, k)
dt = getattr(mx, spec["dtype"])
key = mx.random.key(1)
layout = spec["layout"]
if layout == "batched":
    x = (mx.random.normal((2, m // 2, k), key=key) * 0.5).astype(dt)
elif layout == "transposed":
    # A lazy transposed view.
    x = mx.swapaxes((mx.random.normal((k, m), key=key) * 0.5).astype(dt), 0, 1)
elif layout == "strided":
    # Every other column of a wider array.
    x = (mx.random.normal((m, 2 * k), key=key) * 0.5).astype(dt)[:, ::2]
else:
    x = (mx.random.normal((m, k), key=key) * 0.5).astype(dt)
mx.eval(x)


def f(x):
    y = kq.quantized_matmul(x, w, s, codec, transpose=True)
    return y.astype(mx.float32).sum()


def composed(x):
    # Dequantize then matmul in stock ops, whose vjp keeps the dense copy.
    wf = kq.dequantize(w, s, codec, dtype=mx.bfloat16)
    return (x.astype(mx.bfloat16) @ wf.T).astype(mx.float32).sum()


def active(window):
    # The buffers a finished command buffer frees go back in its completion
    # handler, after the wait returns. Nothing allocates meanwhile, so the
    # least of the reads over the window is what stays held.
    mx.synchronize()
    least = mx.get_active_memory()
    end = time.monotonic() + window
    while time.monotonic() < end:
        time.sleep(0.01)
        least = min(least, mx.get_active_memory())
    return least


def settled():
    # Reads until two windows agree, so the buffers an earlier arm freed
    # are back before the base is taken.
    prev = active(0.2)
    for _ in range(5):
        cur = active(0.2)
        if cur == prev:
            break
        prev = cur
    return cur


def forward(fn):
    # What the pending backward graph holds after the forward.
    mx.clear_cache()
    base = settled()
    loss, g = mx.value_and_grad(fn)(x)
    mx.eval(loss)
    return base, loss, g, active(0.2) - base


out = {}
grads = {}
for arm, fn in (("route", f), ("kernel", f), ("composed", composed)):
    case(arm, spec.get(arm + "_env", {}))
    base, loss, g, held = forward(fn)
    # A late completion handler moves a reading. On the route it leaves
    # buffers not yet freed in the reading, which then gets up to five
    # longer windows. On the stock product it leaves them in the base,
    # and the arm runs again, up to five times.
    for _ in range(5):
        if arm == "route" and held >= spec["held_below"]:
            held = active(1.0) - base
        elif arm == "composed" and held < spec["dense"]:
            mx.eval(g)
            del loss, g
            base, loss, g, held = forward(fn)
        else:
            break
    mx.eval(g)
    grads[arm] = g
    out[arm] = {"held": held, "loss": loss.item(), "dtype": str(g.dtype)}
gr, gk = grads["route"], grads["kernel"]
out["grad_equal"] = bool(mx.array_equal(gr, gk).item())
out["dense"] = n * k * 2
print(json.dumps(out))
"""
)

_WGRAD_CHILD = (
    _PRELUDE
    + r"""
codec, m, n, k = spec["codec"], spec["m"], spec["n"], spec["k"]
w, s = weights(codec, n, k)
x = (mx.random.normal((m, k), key=mx.random.key(1)) * 0.5).astype(mx.bfloat16)
mx.eval(x)


def loss(x, w, s):
    y = kq.quantized_matmul(x, w, s, codec, transpose=True)
    return y.astype(mx.float32).sum()


out = {}
for arm in ("route", "kernel"):
    for argnum in (1, 2):
        name = f"{arm}_{argnum}"
        case(name, spec[arm + "_env"])
        try:
            mx.eval(mx.grad(loss, argnums=argnum)(x, w, s))
            out[name] = None
        except Exception as e:
            out[name] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
"""
)

_TRANSFORM_CHILD = (
    _PRELUDE
    + r"""
codec, m, n, k = spec["codec"], spec["m"], spec["n"], spec["k"]
w, s = weights(codec, n, k)
w2, _ = weights(codec, n, k, seed=1)
x = (mx.random.normal((m, k), key=mx.random.key(1)) * 0.5).astype(mx.bfloat16)
a = (mx.random.normal((k, 16), key=mx.random.key(5)) * 0.05).astype(mx.bfloat16)
b = (mx.random.normal((16, n), key=mx.random.key(6)) * 0.05).astype(mx.bfloat16)
r = (mx.random.normal((m, n), key=mx.random.key(7)) * 0.5).astype(mx.bfloat16)
mx.eval(x, a, b, r)
ops = {
    "matmul": lambda t: kq.quantized_matmul(t, w, s, codec, transpose=True),
    "glu": lambda t: kq.quantized_matmul_glu(t, w, w2, codec),
    "lora": lambda t: kq.quantized_matmul_lora(t, w, codec, a, b),
    "residual": lambda t: kq.quantized_matmul_residual(t, w, codec, r),
}
transforms = {
    "jvp": lambda f: mx.jvp(f, [x], [x])[1],
    "vmap": lambda f: mx.vmap(f)(mx.stack([x, x * 0.5])),
}
out = {"nax": kq.nax_available()}
for arm in ("route", "kernel"):
    # A forward glu call first, so the log shows the node this env builds.
    case(f"{arm}_glu_forward", spec[arm + "_env"])
    mx.eval(ops["glu"](x))
    for op, f in ops.items():
        for tr, run in transforms.items():
            name = f"{arm}_{op}_{tr}"
            case(name, spec[arm + "_env"])
            try:
                mx.eval(run(f))
                out[name] = None
            except Exception as e:
                out[name] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
"""
)

_EXCLUDE_CHILD = (
    _PRELUDE
    + r"""
codec, m, n, k = spec["codec"], spec["m"], spec["n"], spec["k"]
env = spec["env"]
w, s = weights(codec, n, k)
wd = kq.dequantize(w, s, codec, dtype=mx.float32)
x = (mx.random.normal((m, k), key=mx.random.key(1)) * 0.5).astype(mx.bfloat16)
mx.eval(x, wd)
xf = x.astype(mx.float32)
ref = xf @ wd.T
out = {}


def run(name, fn, want):
    case(name, env)
    try:
        y = fn()
        mx.eval(y)
        out[name] = {"rel": rel(y, want), "error": None}
    except Exception as e:
        out[name] = {"rel": None, "error": f"{type(e).__name__}: {e}"}


# The plain call takes the route under this env.
run("plain", lambda: kq.quantized_matmul(x, w, s, codec, transpose=True), ref)
# LoRA arguments keep the fused kernel.
a = (mx.random.normal((k, 16), key=mx.random.key(5)) * 0.05).astype(mx.bfloat16)
b = (mx.random.normal((16, n), key=mx.random.key(6)) * 0.05).astype(mx.bfloat16)
mx.eval(a, b)
lora_ref = ref + (xf @ a.astype(mx.float32)) @ b.astype(mx.float32)
run(
    "lora",
    lambda: kq.quantized_matmul(x, w, s, codec, transpose=True, lora_a=a, lora_b=b),
    lora_ref,
)
# A CPU stream under the GPU default device.
run(
    "cpu_stream",
    lambda: kq.quantized_matmul(x, w, s, codec, transpose=True, stream=mx.cpu),
    ref,
)
# transpose=False: w holds [K, N] rows.
wt, st = weights(codec, k, n, seed=1)
wtd = kq.dequantize(wt, st, codec, dtype=mx.float32)
mx.eval(wtd)
run(
    "no_transpose",
    lambda: kq.quantized_matmul(x, wt, st, codec, transpose=False),
    xf @ wtd,
)
# A 3-D w, from quantized_matmul and from gather_qmm without indices.
w3 = mx.stack([w, w])
s3 = s if s.size == 1 else mx.stack([s, s])
mx.eval(w3, s3)
ref3 = mx.stack([ref, ref])
run("w3d", lambda: kq.quantized_matmul(x, w3, s3, codec, transpose=True), ref3)
run("gather_w3d", lambda: kq.gather_qmm(x, w3, s3, codec, transpose=True), ref3)
# gather_qmm without indices on a 2-D w is the plain call.
run("gather_w2d", lambda: kq.gather_qmm(x, w, s, codec, transpose=True), ref)
print(json.dumps(out))
"""
)


def _run_child(code, spec, env):
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    child_env.update(env)
    spec = dict(spec, tests_dir=TESTS_DIR)
    r = subprocess.run(
        [sys.executable, "-c", code, json.dumps(spec)],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stderr[-6000:]
    logs = {}
    cur = None
    for line in r.stderr.splitlines():
        if line.startswith("[case] "):
            cur = line.split()[1]
            logs[cur] = []
        elif cur is not None and line.startswith("[kq "):
            logs[cur].append(line)
    return json.loads(r.stdout.strip().splitlines()[-1]), logs


def _child_routed(lines):
    return any(x.startswith("[kq dequant] ") for x in lines)


VJP_CASES = [(c, "bfloat16", "dense") for c in sorted(ROWS)] + [
    ("q6_k", "float32", "dense"),
    ("q6_k", "float16", "dense"),
    ("nvfp4", "float16", "dense"),
    ("q6_k", "bfloat16", "batched"),
    ("q6_k", "bfloat16", "transposed"),
    ("q6_k", "float16", "strided"),
]


@pytest.mark.parametrize("codec,dtype,layout", VJP_CASES)
def test_route_vjp(codec, dtype, layout):
    # value_and_grad through a routed call: after the forward the pending
    # backward holds no dense copy of the weight, where the same product in
    # stock ops holds one, and the gradient wrt x is the kernel path's bit
    # for bit (both run the quantized matmul the other way).
    _metal()
    spec = {
        "codec": codec,
        "m": ROWS[codec],
        "n": MIN_N,
        "k": MAX_K,
        "dtype": dtype,
        "layout": layout,
        "route_env": _route_env(codec),
        "kernel_env": KERNEL_ENV,
        "held_below": MIN_N * MAX_K * 2 // 4,
        "dense": MIN_N * MAX_K * 2,
    }
    out, logs = _run_child(_GRAD_CHILD, spec, LIVE)
    assert _child_routed(logs["route"]), logs["route"]
    assert not _child_routed(logs["kernel"]), logs["kernel"]
    assert out["route"]["held"] < out["dense"] // 4, out
    # The reading sees a dense copy where one is held.
    assert out["composed"]["held"] >= out["dense"], out
    assert out["route"]["dtype"] == out["kernel"]["dtype"] == f"mlx.core.{dtype}", out
    assert out["grad_equal"], out
    lr, lk = out["route"]["loss"], out["kernel"]["loss"]
    assert abs(lr - lk) <= 1e-3 * abs(lk), out


@pytest.mark.parametrize("codec", ["nvfp4", "q6_k"])
def test_route_weight_grad_raises(codec):
    # The gradient wrt the wire bytes or the scales raises on the route as
    # on the kernel, with the same message.
    _metal()
    spec = {"codec": codec, "m": ROWS[codec], "n": MIN_N, "k": MIN_K}
    spec.update(route_env=_route_env(codec), kernel_env=KERNEL_ENV)
    out, logs = _run_child(_WGRAD_CHILD, spec, LIVE)
    for argnum, msg in ((1, "no gradient wrt the quantized weights"), (2, "scales")):
        r, k = out[f"route_{argnum}"], out[f"kernel_{argnum}"]
        assert r is not None and msg in r, out
        assert r == k, out
        assert _child_routed(logs[f"route_{argnum}"]), logs
        assert not _child_routed(logs[f"kernel_{argnum}"]), logs


@pytest.mark.parametrize("fused_ops", ["default", "off"])
def test_route_jvp_vmap_refuse_as_kernel(fused_ops):
    # jvp and vmap wrt x of quantized_matmul, glu, lora and residual at a
    # routed shape raise by the name KQuantMatmul, as on the kernel, so the
    # error text does not change at the row floor. Where glu builds its
    # fused node (a NAX GPU with KQ_GLU on, KQ_DISABLE_NAX or not), vmap of
    # glu refuses on that node, before any quantized_matmul is built.
    # Elsewhere glu builds two quantized_matmul calls, which route and
    # refuse as KQuantMatmul. off turns the fused glu, lora and residual
    # nodes off, so any GPU builds what a GPU without NAX builds.
    _metal()
    codec = "q6_k"
    spec = {"codec": codec, "m": ROWS[codec], "n": MIN_N, "k": MIN_K}
    off = "0" if fused_ops == "off" else None
    lever = {"KQ_GLU": off, "KQ_QMM_LORA": off, "KQ_QMM_RESID": off}
    force = {"KQ_QMM_TILE": None, "KQ_DISABLE_NAX": None, "KQ_DEQUANT_ROUTE": "force"}
    spec.update(route_env=dict(force, **lever), kernel_env=dict(KERNEL_ENV, **lever))
    out, logs = _run_child(_TRANSFORM_CHILD, spec, LIVE)
    fused = out["nax"] and fused_ops == "default"
    for arm in ("route", "kernel"):
        # The forward call runs the fused kernel exactly where the node is.
        lines = logs[f"{arm}_glu_forward"]
        assert any("_glu_" in x for x in lines) == fused, (arm, out["nax"], lines)
    for op in ("matmul", "glu", "lora", "residual"):
        for tr in ("jvp", "vmap"):
            r, k = out[f"route_{op}_{tr}"], out[f"kernel_{op}_{tr}"]
            on_node = fused and (op, tr) == ("glu", "vmap")
            node = "KQuantMatmulFused" if on_node else "KQuantMatmul"
            want = f"ValueError: [Primitive::{tr}] Not implemented for {node}."
            assert r == want, (op, tr, out)
            assert k == want, (op, tr, out)
            routed = _child_routed(logs[f"route_{op}_{tr}"])
            assert routed != on_node, (op, tr, logs)
            assert not _child_routed(logs[f"kernel_{op}_{tr}"]), (op, tr, logs)


@pytest.mark.parametrize("codec", ["nvfp4", "q6_k"])
def test_route_exclusions(codec):
    # Under the route's env the plain call routes; LoRA arguments, a CPU
    # stream under the GPU default device, transpose=False and a 3-D w
    # (quantized_matmul or gather_qmm) keep the kernel. gather_qmm without
    # indices on a 2-D w is the plain call and routes.
    _metal()
    spec = {"codec": codec, "m": ROWS[codec], "n": MIN_N, "k": MIN_K}
    spec["env"] = _route_env(codec)
    out, logs = _run_child(_EXCLUDE_CHILD, spec, LIVE)
    want = {
        "plain": True,
        "lora": False,
        "cpu_stream": False,
        "no_transpose": False,
        "w3d": False,
        "gather_w3d": False,
        "gather_w2d": True,
    }
    for name, routed in want.items():
        assert out[name]["error"] is None, (name, out[name])
        assert out[name]["rel"] < R1["bfloat16"], (name, out[name])
        assert _child_routed(logs[name]) == routed, (name, logs[name])
