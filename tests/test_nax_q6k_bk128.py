"""q6_k NAX qmm_t on a K tile of 128 (KqNaxQ6_KHalfLoader).

On NAX GPUs the q6_k BM=64 and BM=128 tiles dispatch the ``_bk128``
kernels, ``KQ_NAX_Q6K_BK128=0`` keeps the ``_bk64`` ones, and the two give
bit-identical outputs. The BM=32 tile and the ``_db`` variant keep BK=64.
Each case runs in a fresh interpreter (tests/qmm_probe.py) and checks the
result against float64 with exact kq.dequantize weights, on the CPU stream
too, where the lever leaves the result alone.
"""

from __future__ import annotations

import pytest
from qmm_probe import force_cpu, run_probe, tile_of

import mlx_kquant as kq

R1 = {"bfloat16": 6e-3, "float16": 1e-3}
LIVE = {"KQ_TUNE_LIVE": "1", "KQ_QMM_LOG": "1"}
# Three super-blocks, so the loader walks two halves per block across
# block boundaries.
K = 768


def _nax_gpu():
    return not force_cpu() and kq.nax_available()


def _kernel(res):
    assert res["error"] is None, res["error"]
    assert len(res["kernels"]) == 1, res["kernels"]
    return res["kernels"][0]


def _bk(name):
    return 128 if "_bk128_" in name else 64


def test_bk128_kernels_in_metallib():
    if force_cpu():
        pytest.skip("Metal kernel names only")
    names = kq.metallib_kernels()
    if not names:
        pytest.skip("no metallib loaded")
    # Every qmm_t tile kernel on a K tile of 128, of any codec.
    got = sorted(
        n
        for n in names
        if n.startswith("kquant_") and "_qmm_t_nax_" in n and "_bk128_" in n
    )
    want = sorted(
        f"kquant_q6_k_qmm_t_nax_{t}_gs_256_b_6_bm{bm}_bn64_bk128_wm2_wn2"
        f"_alN_{al}_batch_{b}"
        for t in ("float16_t", "bfloat16_t")
        for bm in (64, 128)
        for al in ("true", "false")
        for b in (0, 1)
    )
    assert got == want


def test_bk128_dispatch():
    if not _nax_gpu():
        pytest.skip("NAX tile policy")
    # KQ_QMM_MIDM=0 keeps M 48 and M 80 on the qmm tile.
    base = {"codec": "q6_k", "n": 128, "env": {"KQ_QMM_MIDM": "0"}}
    cases = [
        # (call, bm, bk, db)
        (dict(base, shape=[4096, K]), 128, 128, False),
        (dict(base, shape=[4128, K]), 128, 128, False),
        (dict(base, shape=[160, K]), 64, 128, False),
        (dict(base, shape=[80, K]), 64, 128, False),
        (dict(base, shape=[48, K]), 64, 128, False),
        # The M 33-64 _db band from N 16384.
        (dict(base, shape=[48, 256], n=16384), 64, 64, True),
        # KQ_QMM_ROUTE=nax sends M 16 to the BM=32 tile.
        (dict(base, shape=[16, K], env={"KQ_QMM_ROUTE": "nax"}), 32, 64, False),
        (
            dict(
                base,
                shape=[4096, K],
                env={"KQ_QMM_ROUTE": None, "KQ_NAX_TILE": "64,64,2,2"},
            ),
            64,
            128,
            False,
        ),
        (
            dict(
                base,
                shape=[4096, K],
                env={"KQ_NAX_TILE": None, "KQ_NAX_Q6K_BK128": "0"},
            ),
            128,
            64,
            False,
        ),
        (dict(base, shape=[160, K]), 64, 64, False),
        (dict(base, shape=[4096, K], env={"KQ_NAX_Q6K_BK128": None}), 128, 128, False),
    ]
    calls = [c for c, *_ in cases]
    res = run_probe(calls, env=LIVE, check=False)
    for (c, bm, bk, db), r in zip(cases, res, strict=True):
        name, _ = _kernel(r)
        assert tile_of(name)[0] == bm, (c, name)
        assert _bk(name) == bk, (c, name)
        assert name.endswith("_db") == db, (c, name)


def test_bk128_lever_keeps_first_read():
    if not _nax_gpu():
        pytest.skip("NAX tile policy")
    call = {"codec": "q6_k", "shape": [4096, K], "n": 128}
    calls = [dict(call), dict(call, env={"KQ_NAX_Q6K_BK128": None})]
    res = run_probe(
        calls, env={"KQ_QMM_LOG": "1", "KQ_NAX_Q6K_BK128": "0"}, check=False
    )
    assert [_bk(_kernel(r)[0]) for r in res] == [64, 64]


# (shape, n): BM=128 at an even and an odd row-tile count, BM=64, N not a
# multiple of 64, and batches of two flattened into 512 rows (BM=128) and
# 544 rows (BM=64).
SHAPES = [
    ([4096, K], 128),
    ([4128, K], 96),
    ([160, K], 128),
    ([160, K], 96),
    ([2, 256, K], 128),
    ([2, 272, K], 96),
]


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_bk128_bit_identical(dtype):
    nax = _nax_gpu()
    calls = []
    for shape, n in SHAPES:
        for design in ("normal", "lead"):
            base = len(calls)
            call = {
                "codec": "q6_k",
                "shape": shape,
                "n": n,
                "dtype": dtype,
                "seed": base,
                "design": design,
            }
            calls.append(dict(call, env={"KQ_NAX_Q6K_BK128": None}))
            calls.append(dict(call, env={"KQ_NAX_Q6K_BK128": "0"}, ulp_vs=base))
    res = run_probe(calls, env=LIVE)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] < R1[dtype], (c, r["rel"])
        assert r["nonfinite"] == 0, c
        if not nax:
            continue
        name, _ = _kernel(r)
        want = 64 if c["env"]["KQ_NAX_Q6K_BK128"] == "0" else 128
        assert _bk(name) == want, (c, name)
        if c.get("ulp_vs") is not None:
            assert r["ulp"] == 0, (c, r["ulp"])
            assert r["digest"] == res[c["ulp_vs"]]["digest"], c
