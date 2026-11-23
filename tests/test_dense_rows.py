"""Dense row-class kernel (kq_dense_rows.h): quantized_matmul at 2 to 16 rows.

On a NAX GPU, transposed quantized_matmul calls from the codec's entry (ENTRY
below, or KQ_DENSE_ROWS_MIN_M on every codec) to 16 rows run one kernel that
decodes each weight once and applies it to every row. Every codec
quantized_matmul serves has a kernel, in the kq_verify_nax.h decode form where
the codec has one and fits the weight, and in the mat-vec chunk decode form
otherwise (KQ_DENSE_ROWS_FORM pins one form).

Checked here: the kernel each call dispatches, each codec on both forms
and both activation types against a float64 reference from kq.dequantize
weights, and a row's bits across every split of 16 rows (an M-row call
equals the same rows inside any other call of the kernel), on the default
entry and with KQ_DENSE_ROWS_MIN_M at 2 and 1. The output scale, the LoRA
epilogue and quantized_matmul_multi ride the same kernel, the last in one
dispatch under KQ_DENSE_ROWS_MULTI=1.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest
from kqref import is_synth, synth_wire
from qmm_probe import run_probe

import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY, DECODE_ONLY_CODECS

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not kq.nax_available(),
    reason="NAX GPU route",
)

CODECS = [c for c in kq.codecs() if kq.codec_has_matmul(c)]
DTYPES = [mx.bfloat16, mx.float16]
R1 = {mx.bfloat16: 6e-3, mx.float16: 1e-3}
# Codecs with a kq_verify_nax.h decode form, and the K step it needs.
NAX_FORM = {
    "pq2_0": 128,
    "q2_0": 128,
    "nvfp4": 128,
    "q4_0": 64,
    "q4_1": 64,
    "q8_0": 128,
    "q4_k": 256,
    "q5_k": 256,
    "q6_k": 256,
    "q3_k": 256,
    "q2_k": 256,
    "iq4_xs": 256,
    "iq4_nl": 256,
}
N = 200
# The rows each codec's entry takes by default: from the first value in the
# class of 2 to 8 rows and from the second in the class of 9 to 16, 0 for
# none (kq_dense_rows_entry in src/kquant_matmul.cpp).
ENTRY = {
    "q8_0": (2, 9),
    "q5_k": (2, 9),
    "q3_k": (2, 9),
    "iq4_xs": (2, 9),
    "iq4_nl": (2, 9),
    "q6_k": (3, 9),
    "q4_k": (3, 9),
    "q2_k": (3, 9),
    "q4_0": (3, 9),
    "q4_1": (3, 9),
    "q2_0": (3, 9),
    "nvfp4": (3, 9),
    "iq1_s": (3, 9),
    "q5_0": (4, 9),
    "q5_1": (4, 9),
    "iq1_m": (4, 9),
    "iq2_xxs": (5, 9),
    "iq2_s": (6, 9),
    "iq3_s": (8, 9),
    "pq2_0": (3, 0),
    "iq3_xxs": (0, 12),
}


def served(codec, m):
    lo, hi = ENTRY.get(codec, (0, 0))
    first = lo if m <= 8 else hi
    return 2 <= m <= 16 and first > 0 and m >= first


LEVERS = (
    "KQ_DENSE_ROWS",
    "KQ_DENSE_ROWS_MIN_M",
    "KQ_DENSE_ROWS_FORM",
    "KQ_DENSE_ROWS_MULTI",
)


# The forms a test runs: (KQ_DENSE_ROWS_FORM, K). 512 takes the decode form
# where the codec has one; the chunk form takes units of 128 weights or more
# at 512 and, on blocks under 128 weights, one block at 320.
FORMS = {"nax": ("1", 512), "chunk": ("2", 512), "chunk1": ("2", 320)}


_W = {}


def _weight(codec, n, k):
    key = (codec, n, k)
    if key not in _W:
        _, _, bpb, wpb = CODEC_GEOMETRY[codec]
        if is_synth(codec) or codec in DECODE_ONLY_CODECS:
            rng = np.random.default_rng(n * 131 + k)
            wire = synth_wire(rng, codec, bpb, n * (k // wpb))
            w = mx.array(wire.reshape(n, (k // wpb) * bpb))
        else:
            wf = mx.random.normal((n, k), key=mx.random.key(n * 7 + k)) * 0.1
            w, _ = kq.quantize(wf, codec)
        mx.eval(w)
        wd = np.array(
            kq.dequantize(w, mx.zeros((1,), mx.uint8), codec, dtype=mx.float32)
        )
        _W[key] = (w, wd.astype(np.float64))
    return _W[key]


def _mm(x, w, codec, **kw):
    return kq.quantized_matmul(
        x, w, mx.zeros((1,), mx.uint8), codec, transpose=True, **kw
    )


@pytest.fixture
def levers(monkeypatch):
    for k in LEVERS:
        monkeypatch.delenv(k, raising=False)

    def set_(**kv):
        for k, v in kv.items():
            monkeypatch.setenv(k, str(v))

    return set_


def test_rows_dispatch():
    """The kernel of each width: the row kernel on the rows the codec's
    entry takes and under KQ_DENSE_ROWS_MIN_M, the routes before it on the
    other rows, under KQ_DENSE_ROWS=0 and under a lever that forces one."""
    clear = {
        "KQ_DENSE_ROWS": None,
        "KQ_DENSE_ROWS_MIN_M": None,
        "KQ_VERIFY_NAX": None,
        "KQ_NAX_QMV": None,
        "KQ_QMM_SPLITK_NAX": None,
    }
    calls, want = [], []

    def add(codec, m, rows, **env):
        calls.append(
            {"codec": codec, "n": N, "shape": [m, 512], "env": {**clear, **env}}
        )
        want.append((codec, m, rows))

    for codec in CODECS:
        for m in (1, 2, 3, 4, 5, 6, 8, 9, 12, 16):
            add(codec, m, served(codec, m))
        add(codec, 8, False, KQ_DENSE_ROWS="0")
        add(codec, 2, True, KQ_DENSE_ROWS_MIN_M="2")
        add(codec, 1, True, KQ_DENSE_ROWS_MIN_M="1")
        add(codec, 16, True, KQ_DENSE_ROWS_MIN_M="2")
    # A lever that forces another route at the call's M takes precedence.
    add("q6_k", 8, False, KQ_VERIFY_NAX="2")
    add("q6_k", 8, False, KQ_NAX_QMV="8")
    add("q6_k", 8, False, KQ_QMM_SPLITK_NAX="1")
    res = run_probe(calls, env={"KQ_QMM_LOG": "1"}, check=False)
    for r, (codec, m, rows) in zip(res, want, strict=True):
        names = [name for name, _ in r["kernels"]]
        assert r["error"] is None, (codec, m, r["error"])
        assert any("_rows_" in n for n in names) == rows, (codec, m, names)
        if rows:
            gen = f"kquant_{codec}_gen_rows_" in names[0]
            assert gen == (codec not in NAX_FORM), (codec, names)
            assert len(names) == 1, (codec, m, names)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("codec", CODECS)
def test_rows_reference_and_splits(codec, form, dtype, levers):
    """Against the float64 reference at 16 rows, and every row's bits
    across the splits of 16 rows from one row up (KQ_DENSE_ROWS_MIN_M=1)."""
    if form == "nax" and codec not in NAX_FORM:
        pytest.skip("no kq_verify_nax.h decode form")
    if codec == "mxfp4" and dtype == mx.float16:
        pytest.skip("mxfp4 weights are held in x's type: bfloat16 x only")
    if form == "chunk1" and CODEC_GEOMETRY[codec][3] >= 128:
        pytest.skip("units of one block are the 128-weight form")
    form, k = FORMS[form]
    w, wd = _weight(codec, N, k)
    x = mx.random.normal((16, k), key=mx.random.key(5)).astype(dtype)
    levers(KQ_DENSE_ROWS_MIN_M=1, KQ_DENSE_ROWS_FORM=form)
    full = _mm(x, w, codec)
    mx.eval(full)
    ref = np.array(x.astype(mx.float32)).astype(np.float64) @ wd.T
    got = np.array(full.astype(mx.float32)).astype(np.float64)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < R1[dtype], (codec, form, rel)
    for m in range(1, 16):
        for lo in sorted({0, 16 - m}):
            part = _mm(x[lo : lo + m], w, codec)
            assert mx.array_equal(part, full[lo : lo + m]).item(), (codec, form, m, lo)


@pytest.mark.parametrize("entry", [None, 2])
@pytest.mark.parametrize("codec", ["q6_k", "q8_0", "q4_k", "iq4_xs", "q5_0", "iq2_xxs"])
def test_rows_splits_from_entry(codec, entry, levers):
    """From the entry (the codec's default, then KQ_DENSE_ROWS_MIN_M=2) to
    16 rows, a row gives the same bits whatever rows ran with it."""
    if entry is not None:
        levers(KQ_DENSE_ROWS_MIN_M=entry)
    assert ENTRY[codec][1] == 9
    first = entry or ENTRY[codec][0]
    w, _ = _weight(codec, N, 512)
    x = mx.random.normal((16, 512), key=mx.random.key(9)).astype(mx.bfloat16)
    full = _mm(x, w, codec)
    for m in range(first, 16):
        for lo in range(0, 16 - m + 1, 3):
            part = _mm(x[lo : lo + m], w, codec)
            assert mx.array_equal(part, full[lo : lo + m]).item(), (codec, m, lo)


@pytest.mark.parametrize("form", ["1", "2"])
def test_rows_out_scale(form, levers):
    """nvfp4's output scale, per column, on both forms."""
    levers(KQ_DENSE_ROWS_FORM=form)
    k = 512
    w, wd = _weight("nvfp4", N, k)
    s = mx.random.uniform(0.5, 2.0, (N,), key=mx.random.key(3)).astype(mx.float32)
    x = mx.random.normal((8, k), key=mx.random.key(4)).astype(mx.bfloat16)
    y = _mm(x, w, "nvfp4", out_scale=s)
    ref = (np.array(x.astype(mx.float32)).astype(np.float64) @ wd.T) * np.array(s)
    got = np.array(y.astype(mx.float32)).astype(np.float64)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < R1[mx.bfloat16]


@pytest.mark.parametrize("codec", ["q6_k", "q8_0", "iq3_xxs"])
def test_rows_lora(codec, levers):
    """The LoRA epilogue adds rows * (x @ A) @ B to the row kernel's output."""
    levers(KQ_DENSE_ROWS_MIN_M=2)
    k, r = 512, 8
    w, _ = _weight(codec, N, k)
    x = mx.random.normal((8, k), key=mx.random.key(6)).astype(mx.bfloat16)
    a = (mx.random.normal((k, r), key=mx.random.key(7)) * 0.05).astype(mx.bfloat16)
    b = (mx.random.normal((r, N), key=mx.random.key(8)) * 0.05).astype(mx.bfloat16)
    rows = mx.ones((8,), mx.float32)
    base = _mm(x, w, codec)
    y = _mm(x, w, codec, lora_a=a, lora_b=b, lora_rows=rows)
    want = base.astype(mx.float32) + (x @ a).astype(mx.float32) @ b.astype(mx.float32)
    err = mx.abs(y.astype(mx.float32) - want).max().item()
    assert err < 0.05 * mx.abs(want).max().item(), (codec, err)


@pytest.mark.parametrize("codec", ["q6_k", "q8_0", "q4_k", "q5_0", "iq3_xxs"])
@pytest.mark.parametrize("m", [3, 6, 16])
def test_rows_multi(codec, m, levers):
    """quantized_matmul_multi on weights the row kernel serves gives each
    output the bits of quantized_matmul on its weight: two to four weights
    in one dispatch and more in groups of four under KQ_DENSE_ROWS_MULTI=1,
    one dispatch a weight by default. The widths give the weights of a
    dispatch different simdgroup counts."""
    levers(KQ_DENSE_ROWS_MIN_M=2)
    k = 512
    ws = [_weight(codec, n, k)[0] for n in (48, 160, 1000, 200, 33, 2048)]
    x = mx.random.normal((m, k), key=mx.random.key(10)).astype(mx.bfloat16)
    single = [_mm(x, w, codec) for w in ws]
    mx.eval(single)
    for count in (2, 3, 4, 6):
        for multi in ("1", "0"):
            levers(KQ_DENSE_ROWS_MULTI=multi)
            ys = kq.quantized_matmul_multi(x, ws[:count], codec)
            for y, want in zip(ys, single, strict=False):
                assert mx.array_equal(y, want).item(), (codec, m, count, multi)


def test_rows_multi_dispatch():
    """The dispatches of a multi call under KQ_DENSE_ROWS_MULTI=1: one for up
    to four weights the row kernel serves, and one a weight with it at 0."""
    off = {"KQ_DENSE_ROWS_MULTI": "1"}
    calls = [
        {"codec": "q6_k", "n": 48, "shape": [6, 512], "multi": [160, 1000], "env": off},
        {"codec": "q5_0", "n": 48, "shape": [16, 512], "multi": [160], "env": off},
        {
            "codec": "q6_k",
            "n": 48,
            "shape": [6, 512],
            "multi": [160, 1000],
            "env": {"KQ_DENSE_ROWS_MULTI": "0"},
        },
    ]
    env = {"KQ_QMM_LOG": "1", "KQ_DENSE_ROWS_MIN_M": "2"}
    res = run_probe(calls, env=env, check=False)
    for r, (count, one) in zip(res, [(3, True), (2, True), (3, False)], strict=True):
        names = [name for name, _ in r["kernels"]]
        assert r["error"] is None, r["error"]
        assert all("_rows_" in n for n in names), names
        if one:
            assert len(names) == 1 and f"_multi{count}" in names[0], names
        else:
            assert len(names) == count and "_multi" not in "".join(names), names
