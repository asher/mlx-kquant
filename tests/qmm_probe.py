"""Child-process probe for the quantized matmul tile levers.

The tile levers are read once per process unless ``KQ_TUNE_LIVE=1``, so a
test that sets them needs a fresh interpreter. ``run_probe`` starts one with
a clean ``KQ_*`` environment, runs a list of ``quantized_matmul`` calls,
and returns per call the ``KQ_QMM_LOG`` lines it printed (kernel name and
grid), its error if it raised, and its relative Frobenius error against a
float64 reference built from ``kq.dequantize`` weights. A call with
``"r2": true`` also gets ``rel2`` against those weights rounded to the
activation dtype, and ``floor1``/``floor2``, the error of rounding each
reference to the output dtype. A call with ``"ulp_vs": i`` gets ``ulp``,
the largest ulp distance to call ``i``'s output, and ``rdiff``, the
relative Frobenius distance to it. Every call gets ``digest``, a hash of
its output bits, to compare calls across probe processes.

``"design": "lead"`` draws nonnegative weights and activations whose first
K column is 1024. ``"design": "big"`` draws small weights and dense
activations from 2e4 up, with bfloat16 column 0 at 1e5, past the half
range.

Under ``KQUANT_FORCE_CPU`` the child runs on the CPU device and never
touches Metal, so the same probe covers both streams.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

_CHILD = r"""
import hashlib, json, os, sys
import numpy as np
spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["tests_dir"])
import mlx.core as mx
if os.environ.get("KQUANT_FORCE_CPU"):
    mx.set_default_device(mx.cpu)
import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY, DECODE_ONLY_CODECS
from kqref import is_synth, synth_wire

DT = {"bfloat16": mx.bfloat16, "float16": mx.float16}
_weights = {}
_dequant = {}


def weights(codec, n, k, design):
    # design "lead": nonnegative weights, so every partial sum only grows.
    # design "tiled": 64 quantized rows repeated to n, for wide N at little
    # quantize cost. design "big": small weights, nonnegative where the
    # codec quantizes, so large activations give a large sum per block and
    # a moderate output. design "cancel": weights near +0.05 on the first
    # half of K and -0.05 on the second, so the halves cancel.
    key = (codec, n, k, design)
    if key not in _weights:
        _, _, bpb, wpb = CODEC_GEOMETRY[codec]
        if design == "lead":
            wf = mx.abs(mx.random.normal((n, k), key=mx.random.key(2))) * 0.05
            w, s = kq.quantize(wf, codec)
        elif design == "big" and is_synth(codec):
            rng = np.random.default_rng(n * 131 + k)
            wire = synth_wire(rng, codec, bpb, n * (k // wpb), d_range=(1e-4, 4e-4))
            w = mx.array(wire.reshape(n, (k // wpb) * bpb))
            s = mx.zeros((1,), dtype=mx.uint8)
        elif design == "big":
            wf = mx.abs(mx.random.normal((n, k), key=mx.random.key(2))) * 1e-3
            w, s = kq.quantize(wf, codec)
        elif design == "cancel":
            sign = mx.where(mx.arange(k) < k // 2, 1.0, -1.0)
            noise = mx.random.normal((n, k), key=mx.random.key(3)) * 0.002
            w, s = kq.quantize(0.05 * sign[None, :] + noise, codec)
        elif design == "tiled":
            wf = mx.random.normal((64, k), key=mx.random.key(4)) * 0.1
            w1, s = kq.quantize(wf, codec)
            w = mx.contiguous(mx.tile(w1, (n // 64, 1)))
        elif is_synth(codec) or codec in DECODE_ONLY_CODECS:
            rng = np.random.default_rng(n * 131 + k)
            wire = synth_wire(rng, codec, bpb, n * (k // wpb))
            w = mx.array(wire.reshape(n, (k // wpb) * bpb))
            s = mx.zeros((1,), dtype=mx.uint8)
        else:
            wf = mx.random.normal((n, k), key=mx.random.key(n * 7 + k)) * 0.1
            w, s = kq.quantize(wf, codec)
        mx.eval(w, s)
        _weights[key] = (w, s)
    return _weights[key]


def dequantized(codec, n, k, design):
    # The float64 weights of weights(), built on the first check.
    key = (codec, n, k, design)
    if key not in _dequant:
        w, s = weights(codec, n, k, design)
        wd = kq.dequantize(w, s, codec, dtype=mx.float32)
        _dequant[key] = np.array(wd).astype(np.float64)
    return _dequant[key]


def activations(shape, dtype, layout, seed, design):
    # design "lead": nonnegative activations whose first K column is 1024,
    # so the first K slice of a split holds most of each output. design
    # "cancel": activations near 3e4.
    key = mx.random.key(seed)
    lead = design == "lead"

    def draw(s):
        if lead:
            return mx.abs(mx.random.normal(s, key=key)) * 0.01
        if design == "big":
            # 2e4 and up, inside the float16 range.
            return mx.minimum(mx.abs(mx.random.normal(s, key=key)) * 1e4 + 2e4, 6e4)
        if design == "cancel":
            return mx.random.uniform(-100.0, 100.0, s, key=key) + 3e4
        return mx.random.normal(s, key=key) * 0.5

    if layout == "dense":
        x = draw(shape)
        if lead:
            x[..., 0] = 1024.0
        if design == "big" and dtype == mx.bfloat16:
            # One column past the half range.
            x[..., 0] = 1e5
        x = x.astype(dtype)
        mx.eval(x)
        return x
    if layout == "transposed":
        # A lazy transposed view: the op sees an unevaluated swapaxes.
        t = shape[:-2] + [shape[-1], shape[-2]]
        xt = draw(t)
        if lead:
            xt[..., 0, :] = 1024.0
        xt = xt.astype(dtype)
        mx.eval(xt)
        return mx.swapaxes(xt, -1, -2)
    if layout == "strided":
        # Every other column of a wider array.
        wide = shape[:-1] + [2 * shape[-1]]
        xw = draw(wide)
        if lead:
            xw[..., 0] = 1024.0
        xw = xw.astype(dtype)
        mx.eval(xw)
        return xw[..., ::2]
    raise ValueError(layout)


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def rounded(a, dtype):
    # a rounded to dtype, back in float64.
    r = mx.array(a.astype(np.float32)).astype(dtype).astype(mx.float32)
    return np.array(r).astype(np.float64)


def ordered(bits):
    u = bits.astype(np.int64)
    return np.where(u & 0x8000, 0x8000 - (u & 0x7FFF), u + 0x8000)


outs = []
results = []
_refs = {}
for i, c in enumerate(spec["calls"]):
    for name, value in c.get("env", {}).items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    dtype = DT[c.get("dtype", "bfloat16")]
    design = c.get("design", "normal")
    w, s = weights(c["codec"], c["n"], c["shape"][-1], design)
    x = activations(
        list(c["shape"]), dtype, c.get("layout", "dense"), c.get("seed", i), design
    )
    os.write(2, f"[probe] {i}\n".encode())
    res = {"i": i, "error": None, "rel": None, "ulp": None}
    try:
        y = kq.quantized_matmul(x, w, s, c["codec"], transpose=True)
        mx.eval(y)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        outs.append(None)
        results.append(res)
        continue
    outs.append(y)
    # Digest of the output bits, to compare calls across probe processes.
    bits = y.view(mx.uint16) if y.dtype in DT.values() else y
    res["digest"] = hashlib.sha1(np.array(bits).tobytes()).hexdigest()
    if spec.get("check", True):
        # Calls that differ only in env share their references.
        rkey = (
            c["codec"], c["n"], tuple(c["shape"]), str(dtype),
            c.get("layout", "dense"), c.get("seed", i), design,
        )
        if rkey not in _refs:
            wd = dequantized(c["codec"], c["n"], c["shape"][-1], design)
            xn = np.array(x.astype(mx.float32)).astype(np.float64)
            _refs[rkey] = {"xn": xn, "ref": xn @ wd.T}
        refs = _refs[rkey]
        ref = refs["ref"]
        got = np.array(y.astype(mx.float32)).astype(np.float64)
        res["rel"] = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
        res["nonfinite"] = int((~np.isfinite(got)).sum())
        if c.get("r2"):
            # R2: weights rounded to the activation dtype. floor1 and floor2:
            # the error of rounding each reference to the output dtype.
            if "ref2" not in refs:
                wd = dequantized(c["codec"], c["n"], c["shape"][-1], design)
                refs["ref2"] = refs["xn"] @ rounded(wd, dtype).T
                refs["floor1"] = rel(rounded(ref, dtype), ref)
                refs["floor2"] = rel(rounded(refs["ref2"], dtype), refs["ref2"])
            res["rel2"] = rel(got, refs["ref2"])
            res["floor1"] = refs["floor1"]
            res["floor2"] = refs["floor2"]
    base = c.get("ulp_vs")
    if base is not None and outs[base] is not None:
        a = np.array(y.view(mx.uint16))
        b = np.array(outs[base].view(mx.uint16))
        res["ulp"] = int(np.abs(ordered(a) - ordered(b)).max())
        # Relative Frobenius distance to that call's output.
        yb = np.array(outs[base].astype(mx.float32)).astype(np.float64)
        ya = np.array(y.astype(mx.float32)).astype(np.float64)
        res["rdiff"] = rel(ya, yb)
    results.append(res)
print(json.dumps(results))
"""

_LOG = re.compile(r"^\[kq qmm\] (\S+) M=(\d+) N=(\d+) K=(\d+) grid=(\d+),(\d+),(\d+)$")


def run_probe(calls, env=None, check=True, timeout=900):
    """Run ``calls`` in a fresh interpreter whose environment keeps no
    ``KQ_*`` variable but those in ``env``. Returns one dict per call with
    ``kernels`` (list of (name, grid) from KQ_QMM_LOG), ``error``, ``rel``
    and ``ulp``."""
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    child_env.update(env or {})
    spec = {"tests_dir": TESTS_DIR, "calls": calls, "check": check}
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(spec)],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert r.returncode == 0, r.stderr[-6000:]
    results = json.loads(r.stdout.strip().splitlines()[-1])
    logs = {}
    cur = None
    for line in r.stderr.splitlines():
        if line.startswith("[probe] "):
            cur = int(line.split()[1])
            logs[cur] = []
            continue
        m = _LOG.match(line)
        if m and cur is not None:
            grid = tuple(int(g) for g in m.group(5, 6, 7))
            logs[cur].append((m.group(1), grid))
    for res in results:
        res["kernels"] = logs.get(res["i"], [])
    return results


def tile_of(kname):
    """(bm, bn, wm, wn) parsed from a NAX qmm_t kernel name."""
    m = re.search(r"_bm(\d+)_bn(\d+)_bk\d+_wm(\d+)_wn(\d+)", kname)
    assert m, kname
    return tuple(int(g) for g in m.groups())


def force_cpu() -> bool:
    return bool(os.environ.get("KQUANT_FORCE_CPU"))
