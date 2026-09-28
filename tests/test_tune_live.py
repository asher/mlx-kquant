"""KQ_TUNE_LIVE: the NAX tile levers read per dispatch, or once per process.

``KQ_NAX_SMALL_BM``, ``KQ_NAX_BM128``, ``KQ_NAX_DB64`` and ``KQ_NAX_SWIZZLE``
keep their first read for the life of the process, and ``KQ_TUNE_LIVE=1``
makes them read their variables on every dispatch. Each case runs in a fresh
interpreter (tests/qmm_probe.py) and reads the dispatched kernel and grid from
``KQ_QMM_LOG``. The numeric case also runs on the CPU stream under
``KQUANT_FORCE_CPU``, where the levers must leave the result alone.
"""

from __future__ import annotations

import pytest
from qmm_probe import force_cpu, run_probe, tile_of

import mlx_kquant as kq

# R1 bounds: relative Frobenius error against float64 with exact
# kq.dequantize weights.
R1 = {"bfloat16": 6e-3, "float16": 1e-3}

K = 256
N = 128

# q6_k takes BM=128 from M 193 and the BM=32 tile at M <= 32; its _db
# band needs N >= 16384, which KQ_NAX_DB64=1 drops. KQ_QMM_MIDM=0 keeps
# M 48 on the qmm tile and KQ_QMM_ROUTE=nax sends M 16 to it; both are
# read per call.
LEVER_CALLS = [
    # 0: the process starts with KQ_NAX_BM128=0.
    {"codec": "q6_k", "shape": [256, K], "n": N},
    # 1: BM128 unset.
    {"codec": "q6_k", "shape": [256, K], "n": N, "env": {"KQ_NAX_BM128": None}},
    # 2: BM128 off again, swizzle on.
    {
        "codec": "q6_k",
        "shape": [256, K],
        "n": N,
        "env": {"KQ_NAX_BM128": "0", "KQ_NAX_SWIZZLE": "1"},
    },
    # 3: the M 33-64 band with the N floor dropped.
    {
        "codec": "q6_k",
        "shape": [48, K],
        "n": N,
        "env": {"KQ_NAX_SWIZZLE": None, "KQ_NAX_DB64": "1", "KQ_QMM_MIDM": "0"},
    },
    # 4: M 16 on the qmm tile with the BM=32 pick off.
    {
        "codec": "q6_k",
        "shape": [16, K],
        "n": N,
        "env": {"KQ_NAX_SMALL_BM": "0", "KQ_QMM_ROUTE": "nax"},
    },
]


def _gpu_nax():
    if force_cpu():
        pytest.skip("Metal kernel names only")
    if not kq.nax_available():
        pytest.skip("NAX tile levers")


def _only_kernel(res):
    assert res["error"] is None, res["error"]
    assert len(res["kernels"]) == 1, res["kernels"]
    return res["kernels"][0]


def test_levers_keep_first_read():
    _gpu_nax()
    res = run_probe(LEVER_CALLS, env={"KQ_QMM_LOG": "1", "KQ_NAX_BM128": "0"})
    first = _only_kernel(res[0])
    assert tile_of(first[0])[0] == 64
    # Every later change is ignored: the first read holds.
    for r in res[1:3]:
        assert _only_kernel(r) == first
    db = _only_kernel(res[3])[0]
    assert tile_of(db)[0] == 64 and not db.endswith("_db")
    assert tile_of(_only_kernel(res[4])[0])[0] == 32


def test_levers_live_under_tune_live():
    _gpu_nax()
    res = run_probe(
        LEVER_CALLS,
        env={"KQ_QMM_LOG": "1", "KQ_NAX_BM128": "0", "KQ_TUNE_LIVE": "1"},
    )
    name, grid = _only_kernel(res[0])
    assert tile_of(name)[0] == 64 and grid == (2, 4, 1)
    name, grid = _only_kernel(res[1])
    assert tile_of(name)[0] == 128 and grid == (2, 2, 1)
    # Four row tiles fold into grid.x with log 2.
    name, grid = _only_kernel(res[2])
    assert tile_of(name)[0] == 64 and grid == (8, 1, 1)
    assert _only_kernel(res[3])[0].endswith("_db")
    assert tile_of(_only_kernel(res[4])[0])[0] == 64


def test_default_levers_unchanged_by_tune_live():
    # KQ_TUNE_LIVE with no lever set dispatches what the cached read does.
    _gpu_nax()
    calls = [dict(c, env={}) for c in LEVER_CALLS]
    calls[3]["env"] = {"KQ_QMM_MIDM": "0"}
    calls[4]["env"] = {"KQ_QMM_ROUTE": "nax"}
    base = run_probe(calls, env={"KQ_QMM_LOG": "1"})
    live = run_probe(calls, env={"KQ_QMM_LOG": "1", "KQ_TUNE_LIVE": "1"})
    for b, v in zip(base, live, strict=True):
        assert _only_kernel(b) == _only_kernel(v)


# The routing at DiT row counts: per image, and with a batch of two
# flattened into M. From 4096 rows the default runs BM=128 at either
# parity; under KQ_NAX_BM128_ODD_M=0 BM=128 runs where ceil(M/64) is even.
DIT_TILES = {
    4096: 128,
    4128: 64,
    4192: 128,
    4256: 64,
    4608: 128,
    4640: 64,
    16384: 128,
    16896: 128,
}
DIT_TILES_B2 = {
    4096: 128,
    4128: 64,
    4192: 64,
    4256: 64,
    4608: 128,
    4640: 64,
    16384: 128,
    16896: 128,
}


@pytest.mark.parametrize("parity", [False, True])
@pytest.mark.parametrize("codec", ["q4_k", "q5_k", "q6_k", "q8_0"])
def test_default_tile_at_dit_rows(codec, parity):
    _gpu_nax()
    calls = []
    expect = []
    for rows, bm in DIT_TILES.items():
        calls.append({"codec": codec, "shape": [1, rows, K], "n": 64})
        expect.append((rows, bm if parity else 128))
        calls.append({"codec": codec, "shape": [2, rows, K], "n": 64})
        expect.append((2 * rows, DIT_TILES_B2[rows] if parity else 128))
    env = {"KQ_QMM_LOG": "1"}
    if parity:
        env["KQ_NAX_BM128_ODD_M"] = "0"
    res = run_probe(calls, env=env, check=False)
    for r, (m, bm) in zip(res, expect, strict=True):
        name, grid = _only_kernel(r)
        assert tile_of(name)[0] == bm, (m, name)
        assert grid == (1, -(-m // bm), 1), (m, grid)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_live_levers_numerics(dtype):
    # Every lever setting gives the same numbers within the R1 bound, on
    # batch 1, batch 2, lazy transposed and strided inputs. On the CPU
    # stream the levers do nothing and the bound still holds.
    settings = [
        {},
        {"KQ_NAX_BM128": "0"},
        {"KQ_NAX_BM128": "2"},
        {"KQ_NAX_BM128": "0", "KQ_NAX_SWIZZLE": "1"},
    ]
    calls = []
    for layout in ("dense", "transposed", "strided"):
        for shape in ([1, 260, K], [2, 130, K], [320, K]):
            for env in settings:
                full = {
                    "KQ_NAX_BM128": None,
                    "KQ_NAX_SWIZZLE": None,
                    **env,
                }
                calls.append(
                    {
                        "codec": "q6_k",
                        "shape": shape,
                        "n": 200,
                        "dtype": dtype,
                        "layout": layout,
                        "seed": len(calls) // len(settings),
                        "env": full,
                    }
                )
    res = run_probe(calls, env={"KQ_TUNE_LIVE": "1"})
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] < R1[dtype], (c, r["rel"])
