"""The rows that run a plain quantized_matmul as dequantize then the stock
matmul on GPUs without NAX.

``_alu_dequantize_row`` answers for any chip class, so the policy cases run
on every machine. The dispatch cases read ``KQ_QMM_LOG`` in a fresh
interpreter (tests/qmm_probe.py): a call on a row logs no qmm kernel. The
numeric cases hold the default route against float64 with exact
``kq.dequantize`` weights, and against the qmm_t kernel that
``KQ_QMM_TILE=64,64,2,2`` pins within 1 ulp, on batch 1 and 2, lazy
transposed and strided inputs and past 32767 rows. Under
``KQUANT_FORCE_CPU`` they run on the CPU stream, where no row applies and
the bounds still hold.
"""

from __future__ import annotations

import re

import mlx.core as mx
import pytest
from mlx_kquant._ext import _alu_dequantize_row
from qmm_probe import force_cpu, run_probe

import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

R1 = {"bfloat16": 6e-3, "float16": 1e-3}
K = 256
LIVE = {"KQ_TUNE_LIVE": "1", "KQ_QMM_LOG": "1"}
PIN = {"KQ_QMM_TILE": "64,64,2,2"}

# M3 Max with 40 GPU cores (applegpu_g15s): codec -> (min_m, inherited).
ROW_CLASS = (15, 40)
ROWS = {
    "nvfp4": (2048, 0),
    "ptq1_0": (2048, 0),
    "q3_k": (2048, 0),
    "q4_k": (16384, 0),
    "q5_k": (2048, 0),
    "q6_k": (2048, 0),
}
MIN_N = 3840

MATMUL_CODECS = sorted(c for c in CODEC_GEOMETRY if kq.codec_has_matmul(c))


def _metal():
    if force_cpu():
        pytest.skip("Metal kernel logs only")


def _device_class():
    """(arch_gen, cores, nax) of this GPU."""
    arch = mx.device_info().get("architecture", "")
    m = re.match(r"applegpu_g(\d+)", arch)
    return (int(m[1]) if m else 0, kq.gpu_core_count(), kq.nax_available())


@pytest.mark.parametrize("codec", MATMUL_CODECS)
def test_rows_by_chip_class(codec):
    # A row runs on its chip class from its floors in M and N, and nowhere
    # else: not on NAX GPUs, other generations or core counts, and not for a
    # codec without a row.
    gen, cores = ROW_CLASS
    if codec in ROWS:
        min_m, inherited = ROWS[codec]
        for m in (min_m, min_m + 1, 16416, 33792):
            for n in (MIN_N, 4096, 24576):
                got = _alu_dequantize_row(gen, cores, False, codec, m, n)
                assert got == [min_m, MIN_N, inherited], (m, n, got)
        for m in (1, 32, 512, min_m - 1):
            assert _alu_dequantize_row(gen, cores, False, codec, m, 4096) == []
        for n in (64, 3072, MIN_N - 1):
            assert _alu_dequantize_row(gen, cores, False, codec, 33792, n) == []
    else:
        for m in (2048, 4128, 16416, 33792):
            assert _alu_dequantize_row(gen, cores, False, codec, m, 4096) == []
    for g, c, nax in [
        (gen, cores, True),
        (17, 40, True),
        (17, 40, False),
        (18, 10, True),
        (16, 40, False),
        (14, 38, False),
        (gen, 30, False),
        (gen, 18, False),
        (gen, 0, False),
    ]:
        for m in (2048, 16416, 33792):
            assert _alu_dequantize_row(g, c, nax, codec, m, 4096) == [], (g, c, nax)


@pytest.mark.parametrize("codec", sorted(ROWS))
def test_dispatch_on_this_gpu(codec):
    # Below the floors the qmm_t kernel runs. From them the call logs no qmm
    # kernel where this GPU's class has the row, a batch of two counts its
    # rows together, and KQ_QMM_TILE keeps the kernel. On NAX GPUs
    # KQ_DISABLE_NAX=1 reaches the plain path, which keeps the kernel.
    _metal()
    gen, cores, nax = _device_class()
    min_m = ROWS[codec][0]
    want = bool(_alu_dequantize_row(gen, cores, nax, codec, min_m, MIN_N))
    calls = [
        {"codec": codec, "shape": [1, min_m - 32, K], "n": MIN_N},
        {"codec": codec, "shape": [1, min_m, K], "n": MIN_N - 64},
        {"codec": codec, "shape": [1, min_m, K], "n": MIN_N},
        {"codec": codec, "shape": [2, min_m // 2, K], "n": MIN_N},
        {"codec": codec, "shape": [1, min_m, K], "n": MIN_N, "env": PIN},
    ]
    res = run_probe(calls, env=dict(LIVE, KQ_DISABLE_NAX="1"))
    routed = []
    for r in res:
        assert r["error"] is None, r["error"]
        assert r["rel"] < R1["bfloat16"], r["rel"]
        assert len(r["kernels"]) <= 1, r["kernels"]
        if r["kernels"]:
            assert "_qmm_t_" in r["kernels"][0][0], r["kernels"]
        routed.append(not r["kernels"])
    assert routed == [False, False, want, want, False], routed


@pytest.mark.parametrize("codec", sorted(ROWS))
def test_nax_route_unchanged(codec):
    # On NAX GPUs a call at the rows' sizes runs the NAX tile by default and
    # the plain qmm_t kernel under KQ_DISABLE_NAX=1, bit for bit as on a
    # build without the rows.
    _metal()
    if not kq.nax_available():
        pytest.skip("NAX GPU")
    calls = []
    for rows in (4128, 16416):
        calls.append({"codec": codec, "shape": [1, rows, 4096], "n": MIN_N})
        calls.append(
            {
                "codec": codec,
                "shape": [1, rows, 4096],
                "n": MIN_N,
                "env": {"KQ_DISABLE_NAX": "1"},
            }
        )
    res = run_probe(calls, env=LIVE)
    for i, r in enumerate(res):
        assert r["error"] is None, r["error"]
        assert len(r["kernels"]) == 1, r["kernels"]
        name = r["kernels"][0][0]
        assert ("_qmm_t_nax_" in name) == (i % 2 == 0), name
        assert "_qmm_t_" in name, name
        assert r["rel"] < R1["bfloat16"], r["rel"]


def _numeric_calls(codec, dtype, layout):
    min_m = ROWS[codec][0]
    calls = []
    for shape in ([1, min_m, K], [2, min_m // 2, K]):
        base = len(calls)
        for env in (PIN, {}):
            calls.append(
                {
                    "codec": codec,
                    "shape": shape,
                    "n": MIN_N,
                    "dtype": dtype,
                    "layout": layout,
                    "seed": base,
                    "env": env,
                    "ulp_vs": base if not env else None,
                }
            )
    return calls


@pytest.mark.parametrize("layout", ["dense", "transposed", "strided"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("codec", sorted(ROWS))
def test_route_numerics(codec, dtype, layout):
    calls = _numeric_calls(codec, dtype, layout)
    res = run_probe(calls, env=LIVE)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] < R1[dtype], (c, r["rel"])
        if r["ulp"] is not None:
            assert r["ulp"] <= 1, (c, r["ulp"])


@pytest.mark.parametrize("codec", sorted(ROWS))
def test_route_rows_above_int16(codec):
    # 32769 rows on the default route against the kernel.
    calls = []
    for shape in ([1, 32769, K],):
        base = len(calls)
        for env in (PIN, {}):
            calls.append(
                {
                    "codec": codec,
                    "shape": shape,
                    "n": MIN_N,
                    "seed": base,
                    "env": env,
                    "ulp_vs": base if not env else None,
                }
            )
    res = run_probe(calls, env=LIVE, check=False)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        if r["ulp"] is not None:
            assert r["ulp"] <= 1, (c, r["ulp"])
