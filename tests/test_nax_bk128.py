"""K-quant NAX qmm_t on a K tile of 128 (the half super-block loaders).

On NAX GPUs the BM=64 and BM=128 tiles of q6_k, q5_k and q3_k dispatch the
``_bk128`` kernels by default, the codec's ``KQ_NAX_<codec>_BK128=0`` keeps
the ``_bk64`` ones, and the two give bit-identical outputs. q4_k carries the
``_bk128`` kernels behind ``KQ_NAX_Q4K_BK128=1``. The BM=32 tile
and the ``_db`` variant keep BK=64. Each case runs in a fresh interpreter
(tests/qmm_probe.py) and checks the result against float64 with exact
kq.dequantize weights, on the CPU stream too, where the lever leaves the
result alone.
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
CODECS = {"q6_k": 6, "q5_k": 5, "q3_k": 3, "q4_k": 4}
# Codecs whose BM=64 and BM=128 tiles take the K tile of 128 unless the
# lever is 0.
DEFAULT_ON = {"q6_k", "q5_k", "q3_k"}


def _lever(codec):
    return "KQ_NAX_" + codec.replace("_", "").upper() + "_BK128"


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
    # Every qmm_t tile kernel on a K tile of 128, of any codec, so the other
    # K-quants carry none.
    got = sorted(
        n
        for n in names
        if n.startswith("kquant_") and "_qmm_t_nax_" in n and "_bk128_" in n
    )
    want = sorted(
        f"kquant_{c}_qmm_t_nax_{t}_gs_256_b_{b}_bm{bm}_bn64_bk128_wm2_wn2"
        f"_alN_{al}_batch_{bt}"
        for c, b in CODECS.items()
        for t in ("float16_t", "bfloat16_t")
        for bm in (64, 128)
        for al in ("true", "false")
        for bt in (0, 1)
    )
    assert got == want


@pytest.mark.parametrize("codec", sorted(CODECS))
def test_bk128_dispatch(codec):
    if not _nax_gpu():
        pytest.skip("NAX tile policy")
    lever = _lever(codec)
    on = "1"
    # The probe keeps each call's env for the calls after it.
    # KQ_QMM_MIDM=0 keeps M 48 and M 80 on the qmm tile.
    base = {"codec": codec, "n": 128, "env": {"KQ_QMM_MIDM": "0"}}
    cases = [
        # (call, bm, bk, db)
        (
            dict(base, shape=[4096, K], env={"KQ_QMM_MIDM": "0", lever: on}),
            128,
            128,
            False,
        ),
        (dict(base, shape=[4128, K]), 128, 128, False),
        (dict(base, shape=[160, K]), 64, 128, False),
        (dict(base, shape=[80, K]), 64, 128, False),
        (dict(base, shape=[48, K]), 64, 128, False),
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
            dict(base, shape=[4096, K], env={"KQ_NAX_TILE": None, lever: "0"}),
            128,
            64,
            False,
        ),
        (dict(base, shape=[160, K]), 64, 64, False),
        (
            dict(base, shape=[4096, K], env={lever: None}),
            128,
            128 if codec in DEFAULT_ON else 64,
            False,
        ),
    ]
    if codec == "q6_k":
        # The M 33-64 _db band from N 16384.
        cases.append(
            (dict(base, shape=[48, 256], n=16384, env={lever: on}), 64, 64, True)
        )
    calls = [c for c, *_ in cases]
    res = run_probe(calls, env=LIVE, check=False)
    for (c, bm, bk, db), r in zip(cases, res, strict=True):
        name, _ = _kernel(r)
        assert tile_of(name)[0] == bm, (c, name)
        assert _bk(name) == bk, (c, name)
        assert name.endswith("_db") == db, (c, name)


@pytest.mark.parametrize("codec", sorted(CODECS))
def test_bk128_lever_keeps_first_read(codec):
    if not _nax_gpu():
        pytest.skip("NAX tile policy")
    lever = _lever(codec)
    call = {"codec": codec, "shape": [4096, K], "n": 128}
    for first, want in (("0", 64), ("1", 128)):
        flip = "1" if first == "0" else "0"
        calls = [dict(call), dict(call, env={lever: flip})]
        res = run_probe(calls, env={"KQ_QMM_LOG": "1", lever: first}, check=False)
        assert [_bk(_kernel(r)[0]) for r in res] == [want, want]


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
@pytest.mark.parametrize("codec", sorted(CODECS))
def test_bk128_bit_identical(codec, dtype):
    nax = _nax_gpu()
    lever = _lever(codec)
    calls = []
    for shape, n in SHAPES:
        for design in ("normal", "lead"):
            base = len(calls)
            call = {
                "codec": codec,
                "shape": shape,
                "n": n,
                "dtype": dtype,
                "seed": base,
                "design": design,
            }
            calls.append(dict(call, env={lever: "1"}))
            calls.append(dict(call, env={lever: "0"}, ulp_vs=base))
    res = run_probe(calls, env=LIVE)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] < R1[dtype], (c, r["rel"])
        assert r["nonfinite"] == 0, c
        if not nax:
            continue
        name, _ = _kernel(r)
        want = 64 if c["env"][lever] == "0" else 128
        assert _bk(name) == want, (c, name)
        if c.get("ulp_vs") is not None:
            assert r["ulp"] == 0, (c, r["ulp"])
            assert r["digest"] == res[c["ulp_vs"]]["digest"], c


@pytest.mark.parametrize("codec", sorted(CODECS))
def test_bk128_rows_above_int16(codec):
    # 32768 rows, and a batch of two at 16896 rows flattened to 33792, past
    # the int16 row span the tile clamps before narrowing, on the BM=64 and
    # BM=128 tiles.
    nax = _nax_gpu()
    lever = _lever(codec)
    calls = []
    for shape in ([1, 32768, K], [2, 16896, K]):
        for n in (96, 128):
            for tile in ("64,64,2,2", "128,64,2,2"):
                base = len(calls)
                call = {"codec": codec, "shape": shape, "n": n, "seed": base}
                calls.append(dict(call, env={lever: "1", "KQ_NAX_TILE": tile}))
                calls.append(
                    dict(call, env={lever: "0", "KQ_NAX_TILE": tile}, ulp_vs=base)
                )
    res = run_probe(calls, env=LIVE)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] < R1["bfloat16"], (c, r["rel"])
        if not nax:
            continue
        name, _ = _kernel(r)
        assert _bk(name) == (64 if c["env"][lever] == "0" else 128), (c, name)
        assert tile_of(name)[0] == int(c["env"]["KQ_NAX_TILE"].split(",")[0])
        if c.get("ulp_vs") is not None:
            assert r["ulp"] == 0, (c, r["ulp"])
            assert r["digest"] == res[c["ulp_vs"]]["digest"], c
