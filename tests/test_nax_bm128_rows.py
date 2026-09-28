"""BM=128 at large M with an odd ceil(M/64): KQ_NAX_BM128_ODD_M and the
row-live guard of the NAX qmm_t tile.

Row counts 4128, 4256 and 4640, and a batch of two at 4192 rows flattened
to 8384, leave the last BM=128 row tile at most half live, so the guard
skips the dead simdgroup row there. 32768 rows and a batch of two at 16896
rows (33792) run past the int16 row span. Each case checks the result
against float64 with exact kq.dequantize weights, on the CPU stream too,
and on NAX GPUs checks the dispatched tile and that BM=128 agrees with the
BM=64 tile within 1 ulp.
"""

from __future__ import annotations

import pytest
from qmm_probe import force_cpu, run_probe, tile_of

import mlx_kquant as kq

R1 = {"bfloat16": 6e-3, "float16": 1e-3}
K = 256
LIVE = {"KQ_TUNE_LIVE": "1", "KQ_QMM_LOG": "1"}
BM64 = "64,64,2,2"
MAIN_CODECS = ["q4_k", "q5_k", "q6_k", "q8_0"]
# (shape, rows after the batch flattens); every ceil(rows/64) here is odd
# except the last two, which run past 32767 rows.
SHAPES = [
    ([1, 4128, K], 4128),
    ([1, 4256, K], 4256),
    ([1, 4640, K], 4640),
    ([2, 4192, K], 8384),
    ([1, 32768, K], 32768),
    ([2, 16896, K], 33792),
]


def _bm128_codecs():
    names = kq.metallib_kernels()
    return sorted(
        {
            n[len("kquant_") : n.index("_qmm_t_nax_")]
            for n in names
            if n.startswith("kquant_")
            and "_qmm_t_nax_bfloat16_t_" in n
            and "_bm128_bn64_" in n
            and "_gather_" not in n
            and "_splitk_" not in n
        }
    )


BM128_CODECS = _bm128_codecs()


def _nax_gpu():
    return not force_cpu() and kq.nax_available()


def _kernel(res):
    assert res["error"] is None, res["error"]
    assert len(res["kernels"]) == 1, res["kernels"]
    return res["kernels"][0]


def test_bm128_codecs_found():
    assert set(MAIN_CODECS) <= set(BM128_CODECS), BM128_CODECS


def test_odd_rows_policy():
    if not _nax_gpu():
        pytest.skip("NAX tile policy")
    rows = {"codec": "q4_k", "shape": [1, 4128, K], "n": 128}
    below = {"codec": "q4_k", "shape": [1, 4000, K], "n": 128}
    batched = {"codec": "q4_k", "shape": [2, 4192, K], "n": 128}
    calls = [
        dict(rows),
        dict(rows, env={"KQ_NAX_BM128_ODD_M": "0"}),
        dict(rows, env={"KQ_NAX_BM128_ODD_M": "8192"}),
        dict(batched),
        dict(rows, env={"KQ_NAX_BM128_ODD_M": None, "KQ_NAX_BM128": "0"}),
        dict(rows, env={"KQ_NAX_BM128": None}),
        dict(below),
    ]
    res = run_probe(calls, env=LIVE)
    got = [_kernel(r) for r in res]
    assert [tile_of(n)[0] for n, _ in got] == [128, 64, 64, 128, 64, 128, 64]
    assert [g for _, g in got] == [
        (2, 33, 1),
        (2, 65, 1),
        (2, 65, 1),
        (2, 66, 1),
        (2, 65, 1),
        (2, 33, 1),
        (2, 63, 1),
    ]
    for r in res:
        assert r["rel"] < R1["bfloat16"], r


def test_odd_rows_lever_keeps_first_read():
    if not _nax_gpu():
        pytest.skip("NAX tile policy")
    call = {"codec": "q4_k", "shape": [1, 4128, K], "n": 128}
    calls = [dict(call), dict(call, env={"KQ_NAX_BM128_ODD_M": None})]
    res = run_probe(
        calls, env={"KQ_QMM_LOG": "1", "KQ_NAX_BM128_ODD_M": "0"}, check=False
    )
    assert [tile_of(_kernel(r)[0])[0] for r in res] == [64, 64]


@pytest.mark.parametrize("codec", BM128_CODECS)
def test_bm128_rows_known_answer(codec):
    nax = _nax_gpu()
    dtypes = ["bfloat16", "float16"] if codec in MAIN_CODECS else ["bfloat16"]
    calls = []
    for dtype in dtypes:
        for shape, rows in SHAPES:
            for n in (128, 96) if rows == 4128 else (128,):
                base = len(calls)
                call = {
                    "codec": codec,
                    "shape": shape,
                    "n": n,
                    "dtype": dtype,
                    "seed": base,
                    "rows": rows,
                }
                calls.append(dict(call, env={"KQ_NAX_TILE": None}))
                if nax:
                    calls.append(dict(call, env={"KQ_NAX_TILE": BM64}, ulp_vs=base))
    res = run_probe(calls, env=LIVE)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] < R1[c["dtype"]], (c, r["rel"])
        if not nax:
            continue
        want = 64 if c["env"]["KQ_NAX_TILE"] else 128
        name, grid = _kernel(r)
        assert tile_of(name)[0] == want, (c, name)
        assert grid[1] == -(-c["rows"] // want), (c, grid)
        if r["ulp"] is not None:
            assert r["ulp"] <= 1, (c, r["ulp"])
