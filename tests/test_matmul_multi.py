"""quantized_matmul_multi against quantized_matmul per weight.

On a NAX GPU the op runs one verify_nax dispatch per four weights at 2 to
8 rows, where quantized_matmul would run verify_nax on the widest weight.
With the split count forced to one value and the verify_nax entry forced
at M 2, quantized_matmul per weight runs the same kernel body and fold, so
the outputs match bit for bit. At the default
levers the joint grid can take another split count than each weight
alone, so the outputs are checked
against a float64 reference built from kq.dequantize weights. Every other
case runs quantized_matmul per weight, which the CPU stream checks.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest
from kqref import synth_wire
from qmm_probe import force_cpu, run_probe

import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

DTYPES = [mx.bfloat16, mx.float16]
R1 = {mx.bfloat16: 6e-3, mx.float16: 1e-3}
# Gate of quantized_matmul on the CPU stream, as in test_cpu_neon.
CPU_REL = 5e-2
# Codecs with a verify_nax kernel, and an inner dim each serves.
CODECS = {
    "q4_0": 512,
    "q4_1": 512,
    "q8_0": 512,
    "iq4_nl": 512,
    "q4_k": 512,
    "q5_k": 512,
    "q6_k": 512,
    "q3_k": 512,
    "q2_k": 512,
    "iq4_xs": 512,
}
# Row counts of the joint grid's weights: tiny, partial row tiles, a
# 32-row remainder and a weight wider than the rest.
NS = [48, 160, 1000, 300]

_W = {}


def _weight(codec, n, k, seed):
    key = (codec, n, k, seed)
    if key not in _W:
        wf = mx.random.normal((n, k), key=mx.random.key(97 * seed + n + k)) * 0.1
        w, s = kq.quantize(wf, codec)
        mx.eval(w, s)
        wd = np.array(kq.dequantize(w, s, codec, dtype=mx.float32)).astype(np.float64)
        _W[key] = (w, wd)
    return _W[key]


def _weights(codec, ns, k):
    return [_weight(codec, n, k, i) for i, n in enumerate(ns)]


def _x(m, k, dtype, seed=0):
    return mx.random.normal((m, k), key=mx.random.key(seed + 7 * m)).astype(dtype)


def _per_weight(x, ws, codec):
    sc = mx.zeros((1,), mx.uint8)
    return [kq.quantized_matmul(x, w, sc, codec, transpose=True) for w in ws]


@pytest.fixture
def exact_levers(monkeypatch):
    """One split count and verify_nax forced from M 2, which also turns the
    default per-row qmv route off, so quantized_matmul runs verify_nax on
    every weight the op joins."""

    def set_splits(n):
        monkeypatch.setenv("KQ_VERIFY_NAX_SPLITS", str(n))

    monkeypatch.setenv("KQ_VERIFY_NAX", "2")
    return set_splits


def _synth(codec, n, k, seed):
    # Codecs kq.quantize does not encode: random valid wire.
    _, _, bpb, wpb = CODEC_GEOMETRY[codec]
    rng = np.random.default_rng(seed)
    wire = synth_wire(rng, codec, bpb, n * (k // wpb))
    return mx.array(wire.reshape(n, (k // wpb) * bpb))


def _equal(a, b):
    return np.array_equal(
        np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("codec", sorted(CODECS))
@pytest.mark.parametrize("splits", [1, 2])
def test_matches_per_weight_bits(codec, dtype, splits, exact_levers):
    exact_levers(splits)
    k = CODECS[codec]
    ws = [w for w, _ in _weights(codec, NS, k)]
    for m in range(1, 9):
        x = _x(m, k, dtype)
        got = kq.quantized_matmul_multi(x, ws, codec)
        want = _per_weight(x, ws, codec)
        mx.eval(got, want)
        assert len(got) == len(ws)
        for g, w in zip(got, want, strict=True):
            assert g.dtype == w.dtype and g.shape == w.shape
            assert np.array_equal(
                np.array(g.astype(mx.float32)), np.array(w.astype(mx.float32))
            ), f"{codec} M={m} N={w.shape[-1]}"


@pytest.mark.parametrize("codec", ["pq2_0", "q2_0"])
@pytest.mark.parametrize("splits", [1, 2, 4])
def test_synth_codecs_match_per_weight_bits(codec, splits, exact_levers):
    exact_levers(splits)
    ws = [_synth(codec, n, 1024, i) for i, n in enumerate(NS)]
    for dtype in DTYPES:
        for m in range(1, 9):
            x = _x(m, 1024, dtype)
            got = kq.quantized_matmul_multi(x, ws, codec)
            want = _per_weight(x, ws, codec)
            mx.eval(got, want)
            for g, w in zip(got, want, strict=True):
                assert _equal(g, w), f"{codec} M={m} N={w.shape[-1]}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("codec", ["q4_0", "q4_1", "q4_k", "q6_k"])
def test_reference_default_levers(codec, dtype):
    k = CODECS[codec]
    pairs = _weights(codec, NS, k)
    ws = [w for w, _ in pairs]
    for m in (1, 2, 3, 4, 6, 8, 9, 16):
        x = _x(m, k, dtype, seed=3)
        got = kq.quantized_matmul_multi(x, ws, codec)
        mx.eval(got)
        xd = np.array(x.astype(mx.float32)).astype(np.float64)
        for g, (_, wd) in zip(got, pairs, strict=True):
            ref = xd @ wd.T
            err = np.linalg.norm(np.array(g.astype(mx.float32)) - ref)
            rel = CPU_REL if force_cpu() else R1[dtype]
            assert err <= rel * np.linalg.norm(ref), f"{codec} M={m}"


@pytest.mark.parametrize("codec", ["q4_0", "q4_1"])
def test_two_block_inner_dim(codec, exact_levers):
    # K not a multiple of 256 runs the two-block q4_0 or q4_1 kernel.
    exact_levers(1)
    ws = [w for w, _ in _weights(codec, [64, 200], 320)]
    for m in (3, 5, 8):
        x = _x(m, 320, mx.bfloat16)
        got = kq.quantized_matmul_multi(x, ws, codec)
        want = _per_weight(x, ws, codec)
        mx.eval(got, want)
        for g, w in zip(got, want, strict=True):
            assert np.array_equal(
                np.array(g.astype(mx.float32)), np.array(w.astype(mx.float32))
            )


@pytest.mark.parametrize("count", [1, 4, 5, 9])
def test_weight_counts(count, exact_levers):
    # More than four weights run one dispatch per four.
    exact_levers(1)
    ns = [64 + 32 * i for i in range(count)]
    ws = [w for w, _ in _weights("q4_0", ns, 512)]
    x = _x(6, 512, mx.bfloat16)
    got = kq.quantized_matmul_multi(x, ws, "q4_0")
    want = _per_weight(x, ws, "q4_0")
    mx.eval(got, want)
    for g, w in zip(got, want, strict=True):
        assert np.array_equal(
            np.array(g.astype(mx.float32)), np.array(w.astype(mx.float32))
        )


def test_batch_dims_views_and_float32(exact_levers):
    exact_levers(1)
    big, _ = _weight("q4_0", 512, 512, 11)
    ws = [big[100:228], big[300:348], _weight("q4_0", 96, 512, 12)[0]]
    x = _x(6, 512, mx.float32).reshape(2, 3, 512)
    got = kq.quantized_matmul_multi(x, ws, "q4_0")
    want = _per_weight(x, ws, "q4_0")
    mx.eval(got, want)
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape and g.dtype == mx.bfloat16
        assert np.array_equal(
            np.array(g.astype(mx.float32)), np.array(w.astype(mx.float32))
        )


def test_empty_weight(exact_levers):
    exact_levers(1)
    ws = [_weight("q4_0", 128, 512, 0)[0], mx.zeros((0, 288), mx.uint8)]
    x = _x(4, 512, mx.bfloat16)
    got = kq.quantized_matmul_multi(x, ws, "q4_0")
    mx.eval(got)
    assert got[1].shape == (4, 0)
    want = _per_weight(x, ws[:1], "q4_0")[0]
    assert np.array_equal(
        np.array(got[0].astype(mx.float32)), np.array(want.astype(mx.float32))
    )


def test_misaligned_base_falls_back(exact_levers):
    # verify_nax reads q4_0 in 4-byte words: a weight 2 bytes past a 4-byte
    # boundary sends the whole group per weight, off verify_nax.
    exact_levers(1)
    w = _weight("q4_0", 64, 512, 3)[0]
    flat = mx.concatenate([mx.zeros((2,), mx.uint8), w.reshape(-1)])
    mx.eval(flat)
    wv = flat[2:].reshape(w.shape)
    mx.eval(wv)
    ws = [_weight("q4_0", 128, 512, 4)[0], wv]
    x = _x(6, 512, mx.bfloat16)
    got = kq.quantized_matmul_multi(x, ws, "q4_0")
    want = _per_weight(x, ws, "q4_0")
    mx.eval(got, want)
    for g, w in zip(got, want, strict=True):
        assert np.isfinite(np.array(g.astype(mx.float32))).all()
        assert _equal(g, w)


def test_grad_matches_per_weight():
    ws = [w for w, _ in _weights("q4_0", NS, 512)]
    x = _x(6, 512, mx.float32)
    cot = [mx.random.normal((6, n), key=mx.random.key(n)) for n in NS]

    def f_multi(x):
        return sum(
            (y.astype(mx.float32) * c).sum()
            for y, c in zip(kq.quantized_matmul_multi(x, ws, "q4_0"), cot, strict=True)
        )

    def f_each(x):
        return sum(
            (y.astype(mx.float32) * c).sum()
            for y, c in zip(_per_weight(x, ws, "q4_0"), cot, strict=True)
        )

    g1 = mx.grad(f_multi)(x)
    g2 = mx.grad(f_each)(x)
    mx.eval(g1, g2)
    assert np.allclose(np.array(g1), np.array(g2), rtol=2e-2, atol=2e-2)


def test_argument_errors():
    w, _ = _weight("q4_0", 64, 512, 0)
    x = _x(4, 512, mx.bfloat16)
    with pytest.raises(ValueError):
        kq.quantized_matmul_multi(x, [], "q4_0")
    with pytest.raises(ValueError):
        kq.quantized_matmul_multi(x, [w, _weight("q4_0", 64, 256, 0)[0]], "q4_0")
    with pytest.raises(ValueError):
        kq.quantized_matmul_multi(_x(4, 256, mx.bfloat16), [w], "q4_0")
    with pytest.raises(ValueError):
        kq.quantized_matmul_multi(x, [w], "no_such_codec")


_LOG_CHILD = r"""
import json, os, sys
import mlx.core as mx
import mlx_kquant as kq
codec, m, k, ns, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
    json.loads(sys.argv[4]), json.loads(sys.argv[5])
w = [kq.quantize(mx.random.normal((n, k), key=mx.random.key(i)) * 0.1, codec)[0]
     for i, n in enumerate(ns)]
x = mx.random.normal((m, k), key=mx.random.key(9)).astype(mx.bfloat16)
sc = mx.zeros((1,), mx.uint8)
mx.eval(w, x)
for i, (op, env) in enumerate(steps):
    for name, value in env.items():
        os.environ[name] = value
    sys.stderr.write(f"=== {i}\n")
    sys.stderr.flush()
    if op == "multi":
        out = kq.quantized_matmul_multi(x, w, codec)
    else:
        out = [kq.quantized_matmul(x, wi, sc, codec, transpose=True) for wi in w]
    mx.eval(out)
"""


def _census(steps, codec="q4_0", m=6, k=512, ns=(4096, 640, 48, 48), env=None):
    """KQ_QMM_LOG kernel names per step of a child process. Each step is
    (op, env): the env is set before the step's call, which is
    quantized_matmul_multi ("multi") or quantized_matmul per weight."""
    child = {kk: v for kk, v in os.environ.items() if not kk.startswith("KQ_")}
    child["KQ_QMM_LOG"] = "1"
    child.update(env or {})
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            _LOG_CHILD,
            codec,
            str(m),
            str(k),
            json.dumps(list(ns)),
            json.dumps(steps),
        ],
        env=child,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr
    out = []
    for part in r.stderr.split("=== ")[1:]:
        lines = part.splitlines()[1:]
        out.append([ln.split()[2] for ln in lines if ln.startswith("[kq qmm]")])
    return out


NAX_ONLY = pytest.mark.skipif(
    force_cpu() or not kq.nax_available(), reason="NAX GPU route"
)


@NAX_ONLY
def test_one_dispatch_on_nax():
    (names,) = _census([["multi", {}]], env={"KQ_VERIFY_NAX_SPLITS": "1"})
    assert len(names) == 1 and "verify_nax_multi" in names[0], names


@NAX_ONLY
@pytest.mark.parametrize(
    "lever",
    [
        {"KQ_VERIFY_MMA": "2"},
        {"KQ_QMM_SPLITK_NAX": "1"},
        {"KQ_QMM_ROUTE": "qmv"},
        {"KQ_QMM_ROUTE": "verify_mma"},
        {"KQ_FORCE_QMM_MIN_M": "2"},
        {"KQ_VERIFY_NAX": "0"},
        {"KQ_NAX_QMV": "8"},
    ],
    ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items()),
)
def test_route_levers_are_followed(lever):
    # A lever that sends quantized_matmul off verify_nax sends the op to
    # quantized_matmul per weight, which then runs the same kernels.
    ns = (4096, 4096)
    per, multi = _census([["per", {}], ["multi", {}]], ns=ns, env=lever)
    assert not any("verify_nax" in n for n in per), per
    assert multi == per


@NAX_ONLY
def test_forced_verify_nax_fuses_narrow_weights():
    # KQ_VERIFY_NAX=2 alone takes quantized_matmul to verify_nax at widths
    # whose default is per-row qmv; the op follows and joins the weights.
    ns = (512, 256)
    per, multi = _census(
        [["per", {}], ["multi", {}]], m=4, ns=ns, env={"KQ_VERIFY_NAX": "2"}
    )
    assert len(per) == 2 and all("_verify_nax_" in n for n in per), per
    assert len(multi) == 1 and "verify_nax_multi" in multi[0], multi


@NAX_ONLY
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("codec", [*CODECS, "pq2_0", "q2_0"])
def test_x16_load_bits(codec, dtype):
    # x at offset 0 (paired 16-byte activation loads), at offset 0 with
    # KQ_VNAX_X16=0, at every 2-byte offset from 2 to 14 bytes (8 bytes: one
    # load per step; the others: an aligned copy, paired) and at 16 bytes:
    # the same bits from one verify_nax_multi dispatch.
    arms = [(0, None), (0, "0"), *[(e, None) for e in range(1, 9)]]
    cases = [(m, sp) for m in range(2, 9) for sp in ("1", "2")]
    calls = [
        {
            "codec": codec,
            "shape": [m, 1024],
            "n": 1000,
            "multi": [48, 300],
            "dtype": dtype,
            "seed": m,
            "x_offset": off,
            "env": {"KQ_VNAX_X16": lever, "KQ_VERIFY_NAX_SPLITS": sp},
        }
        for m, sp in cases
        for off, lever in arms
    ]
    env = {"KQ_QMM_LOG": "1", "KQ_TUNE_LIVE": "1", "KQ_VERIFY_NAX": "2"}
    res = run_probe(calls, env=env, check=False)
    for i, (m, sp) in enumerate(cases):
        group = res[len(arms) * i : len(arms) * (i + 1)]
        for (off, lever), r in zip(arms, group, strict=True):
            tag = (codec, m, sp, off, lever)
            assert r["error"] is None, (tag, r["error"])
            b = 2 * off
            x8 = lever is not None or b % 16 == 8
            end = "_x8(_pf32)?$" if x8 else "_x16(_pf32)?$"
            names = [n for n, _ in r["kernels"]]
            assert names and all(
                "verify_nax_multi" in n and re.search(end, n) for n in names
            ), (tag, names)
            copied = [("quantized_matmul_multi x", b % 8)] if b % 8 else []
            assert r["copies"] == copied, (tag, r["copies"])
            assert set(r["xa"]) == {0 if b % 8 else b % 16}, (tag, r["xa"])
            assert r["digest"] == group[0]["digest"], tag


@NAX_ONLY
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "lever",
    [{"KQ_VERIFY_NAX": "0"}, {"KQ_QMM_ROUTE": "qmv"}],
    ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items()),
)
def test_per_weight_copies_x_once(lever, dtype):
    # Off verify_nax on a NAX GPU, the op runs quantized_matmul per weight
    # on one aligned copy of x: one copy for the three weights, every kernel
    # reads x on a 16-byte boundary, and the bits match x at offset 0.
    calls = [
        {
            "codec": "q4_k",
            "shape": [4, 1024],
            "n": 1024,
            "multi": [256, 512],
            "dtype": dtype,
            "seed": 4,
            "x_offset": off,
        }
        for off in range(9)
    ]
    res = run_probe(calls, env={"KQ_QMM_LOG": "1", **lever}, check=False)
    for off, r in enumerate(res):
        b = 2 * off
        assert r["error"] is None, (off, r["error"])
        names = [n for n, _ in r["kernels"]]
        assert names and not any("multi" in n for n in names), (off, names)
        copied = [("quantized_matmul_multi x", b % 16)] if b % 16 else []
        assert r["copies"] == copied, (off, r["copies"])
        assert set(r["xa"]) == {0}, (off, r["xa"])
        assert r["digest"] == res[0]["digest"], off


@NAX_ONLY
def test_lever_read_once_per_process():
    # KQ_MATMUL_MULTI keeps its first read, as KQ_GLU, unless KQ_TUNE_LIVE=1.
    steps = [["multi", {}], ["multi", {"KQ_MATMUL_MULTI": "0"}]]
    first, second = _census(steps)
    assert first == second and "verify_nax_multi" in first[0], (first, second)
    first, second = _census(steps, env={"KQ_TUNE_LIVE": "1"})
    assert "verify_nax_multi" in first[0], first
    assert not any("verify_nax_multi" in n for n in second), second


def test_lever_off():
    # KQ_MATMUL_MULTI=0 builds quantized_matmul per weight: the same bits.
    r = subprocess.run(
        [sys.executable, "-c", _LEVER_OFF_CHILD],
        env={**os.environ, "KQ_MATMUL_MULTI": "0"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-1] == "EQUAL", r.stdout


_LEVER_OFF_CHILD = r"""
import os
import re
import mlx.core as mx
import numpy as np
if os.environ.get("KQUANT_FORCE_CPU"):
    mx.set_default_device(mx.cpu)
import mlx_kquant as kq
ns = (48, 160, 1000, 300)
w = [kq.quantize(mx.random.normal((n, 512), key=mx.random.key(n)) * 0.1, "q4_0")[0]
     for n in ns]
x = mx.random.normal((6, 512), key=mx.random.key(1)).astype(mx.bfloat16)
sc = mx.zeros((1,), mx.uint8)
got = kq.quantized_matmul_multi(x, w, "q4_0")
want = [kq.quantized_matmul(x, wi, sc, "q4_0", transpose=True) for wi in w]
mx.eval(got, want)
f32 = [np.array(a.astype(mx.float32)) for a in got + want]
same = all(np.array_equal(a, b) for a, b in zip(f32[: len(w)], f32[len(w) :]))
print("EQUAL" if same else "DIFFER")
"""
