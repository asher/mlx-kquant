"""Child-process probe for the quantized matmul tile levers.

The tile levers are read once per process unless ``KQ_TUNE_LIVE=1``, so a
test that sets them needs a fresh interpreter. ``run_probe`` starts one with
a clean ``KQ_*`` environment, runs a list of ``quantized_matmul`` calls,
and returns per call the ``KQ_QMM_LOG`` lines it printed (kernel name and
grid), its error if it raised, and its relative Frobenius error against a
float64 reference built from ``kq.dequantize`` weights.

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
import json, os, sys
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


def weights(codec, n, k):
    key = (codec, n, k)
    if key not in _weights:
        _, _, bpb, wpb = CODEC_GEOMETRY[codec]
        if is_synth(codec) or codec in DECODE_ONLY_CODECS:
            rng = np.random.default_rng(n * 131 + k)
            wire = synth_wire(rng, codec, bpb, n * (k // wpb))
            w = mx.array(wire.reshape(n, (k // wpb) * bpb))
            s = mx.zeros((1,), dtype=mx.uint8)
        else:
            wf = mx.random.normal((n, k), key=mx.random.key(n * 7 + k)) * 0.1
            w, s = kq.quantize(wf, codec)
        wd = kq.dequantize(w, s, codec, dtype=mx.float32)
        mx.eval(w, s, wd)
        _weights[key] = (w, s, np.array(wd).astype(np.float64))
    return _weights[key]


def activations(shape, dtype, layout, seed):
    key = mx.random.key(seed)
    if layout == "dense":
        x = (mx.random.normal(shape, key=key) * 0.5).astype(dtype)
        mx.eval(x)
        return x
    if layout == "transposed":
        # A lazy transposed view: the op sees an unevaluated swapaxes.
        t = shape[:-2] + [shape[-1], shape[-2]]
        xt = (mx.random.normal(t, key=key) * 0.5).astype(dtype)
        mx.eval(xt)
        return mx.swapaxes(xt, -1, -2)
    if layout == "strided":
        # Every other column of a wider array.
        wide = shape[:-1] + [2 * shape[-1]]
        xw = (mx.random.normal(wide, key=key) * 0.5).astype(dtype)
        mx.eval(xw)
        return xw[..., ::2]
    raise ValueError(layout)


def ordered(bits):
    u = bits.astype(np.int64)
    return np.where(u & 0x8000, 0x8000 - (u & 0x7FFF), u + 0x8000)


outs = []
results = []
for i, c in enumerate(spec["calls"]):
    for name, value in c.get("env", {}).items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    dtype = DT[c.get("dtype", "bfloat16")]
    w, s, wd = weights(c["codec"], c["n"], c["shape"][-1])
    x = activations(list(c["shape"]), dtype, c.get("layout", "dense"), c.get("seed", i))
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
    if spec.get("check", True):
        xn = np.array(x.astype(mx.float32)).astype(np.float64)
        ref = xn @ wd.T
        got = np.array(y.astype(mx.float32)).astype(np.float64)
        res["rel"] = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
    base = c.get("ulp_vs")
    if base is not None and outs[base] is not None:
        a = np.array(y.view(mx.uint16))
        b = np.array(outs[base].view(mx.uint16))
        res["ulp"] = int(np.abs(ordered(a) - ordered(b)).max())
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
