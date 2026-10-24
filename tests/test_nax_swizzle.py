"""KQ_NAX_SWIZZLE and the large-M rows of the NAX qmm_t tile policy.

The swizzle folds 2^log row tiles into grid.x, so consecutive threadgroups
share one weight column band. The lever sets the fold log on the BM=64 and
BM=128 tiles, capped at the row-tile count; unset, the codec's large-M row
sets it on the row's own tile. Each case runs in a fresh interpreter
(tests/qmm_probe.py), reads the dispatched kernel and grid from
KQ_QMM_LOG, and checks the result against float64 with exact
kq.dequantize weights, on the CPU stream too, where the lever leaves the
result alone. A fold changes only the order of the row tiles, so its
output is bit-identical to the unfolded grid.
"""

from __future__ import annotations

import pytest
from qmm_probe import force_cpu, run_probe, tile_of

import mlx_kquant as kq

R1 = {"bfloat16": 6e-3, "float16": 1e-3}
K = 512
LIVE = {"KQ_TUNE_LIVE": "1", "KQ_QMM_LOG": "1"}
TILES = {64: "64,64,2,2", 128: "128,64,2,2"}
# The fold log of each codec's large-M row (src/kquant_matmul.cpp,
# kq_largem_row); codecs not listed fold nothing.
POLICY_LOG: dict[str, int] = {}
# Codecs whose large-M row is measured; the rest take their loader class's
# row, the same tile and floor today.
MEASURED = {"q4_k", "q5_k", "q6_k", "q8_0"}
# K ceiling of each row (kq_largem_max_k); codecs not listed have none.
MAX_K = {
    "q2_k": 4096, "pq2_0": 4096, "q2_0": 4096,
    "q4_k": 10240, "q8_0": 10240, "q4_0": 10240, "q4_1": 10240,
    "q5_0": 10240, "q5_1": 10240,
    "iq4_nl": 12288, "mxfp4": 12288, "stq1_0": 12288,
}  # fmt: skip
# A k-quant on K tiles of 64 and of 128, a flat block codec, IQ grid codecs
# and a ternary codec.
CODECS = ["q4_k", "q6_k", "q8_0", "q4_0", "iq2_xxs", "iq4_xs", "stq1_0"]


def _nax_gpu():
    return not force_cpu() and kq.nax_available()


def _kernel(res):
    assert res["error"] is None, res["error"]
    assert len(res["kernels"]) == 1, res["kernels"]
    return res["kernels"][0]


def _fold(rows, bm, n, log):
    """Grid of a fold of 2^log row tiles, capped at the row-tile count."""
    y = -(-rows // bm)
    while log > 0 and (1 << log) > y:
        log -= 1
    x = -(-n // 64)
    return (x << log, -(-y // (1 << log)), 1)


@pytest.mark.parametrize("codec", CODECS)
def test_swizzle_grid(codec):
    if not _nax_gpu():
        pytest.skip("NAX grid")
    cases = []
    # 4128 rows: 33 row tiles of 128 and 65 of 64.
    for bm in (64, 128):
        for log in (0, 1, 2, 3):
            env = {"KQ_NAX_TILE": TILES[bm], "KQ_NAX_SWIZZLE": str(log)}
            cases.append((dict(shape=[4128, K], n=192, env=env), bm, 4128, log))
    # One row tile of 128: nothing to fold.
    env = {"KQ_NAX_TILE": TILES[128], "KQ_NAX_SWIZZLE": "2"}
    cases.append((dict(shape=[100, K], n=192, env=env), 128, 100, 2))
    # Two row tiles take a fold of 2 at most.
    env = {"KQ_NAX_TILE": TILES[64], "KQ_NAX_SWIZZLE": "3"}
    cases.append((dict(shape=[2, 64, K], n=192, env=env), 64, 128, 3))
    # Unset: the large-M row's fold on the row's own tile, none under
    # KQ_NAX_TILE, and none below the row's floor.
    env = {"KQ_NAX_TILE": None, "KQ_NAX_SWIZZLE": None}
    log = POLICY_LOG.get(codec, 0)
    cases.append((dict(shape=[4128, K], n=192, env=env), 128, 4128, log))
    env = {"KQ_NAX_TILE": TILES[128]}
    cases.append((dict(shape=[4128, K], n=192, env=env), 128, 4128, 0))
    env = {"KQ_NAX_TILE": None}
    cases.append((dict(shape=[1050, K], n=192, env=env), 64, 1050, 0))
    calls = [dict(c, codec=codec) for c, *_ in cases]
    res = run_probe(calls, env=dict(LIVE, KQ_QMM_MIDM="0"))
    for (c, bm, rows, log), r in zip(cases, res, strict=True):
        name, grid = _kernel(r)
        assert tile_of(name)[0] == bm, (c, name)
        assert grid == _fold(rows, bm, c["n"], log), (c, grid)
        assert r["rel"] < R1["bfloat16"], (c, r["rel"])


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("codec", CODECS)
def test_swizzle_bit_identical(codec, dtype):
    # An odd row-tile count with N a multiple of 64, and a batch of two
    # flattened into 4160 rows with N not a multiple of 64.
    nax = _nax_gpu()
    calls = []
    for shape, n in (([4128, K], 192), ([2, 2080, K], 96)):
        for bm in (64, 128):
            base = len(calls)
            for log in (0, 1, 2, 3):
                env = {"KQ_NAX_TILE": TILES[bm], "KQ_NAX_SWIZZLE": str(log)}
                calls.append(
                    {
                        "codec": codec,
                        "shape": shape,
                        "n": n,
                        "dtype": dtype,
                        "seed": base,
                        "env": env,
                        "ulp_vs": base if log else None,
                    }
                )
    res = run_probe(calls, env=LIVE)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] < R1[dtype], (c, r["rel"])
        assert r["nonfinite"] == 0, c
        if nax and c["ulp_vs"] is not None:
            assert r["ulp"] == 0, (c, r["ulp"])
            assert r["digest"] == res[c["ulp_vs"]]["digest"], c


def test_swizzle_lever_keeps_first_read():
    if not _nax_gpu():
        pytest.skip("NAX grid")
    call = {"codec": "q4_k", "shape": [4128, K], "n": 192}
    env = {"KQ_QMM_LOG": "1", "KQ_NAX_TILE": TILES[128], "KQ_NAX_SWIZZLE": "2"}
    calls = [dict(call), dict(call, env={"KQ_NAX_SWIZZLE": "0"})]
    res = run_probe(calls, env=env, check=False)
    want = _fold(4128, 128, 192, 2)
    assert [_kernel(r)[1] for r in res] == [want, want]


@pytest.mark.parametrize("codec", kq.codecs())
def test_large_row(codec):
    from mlx_kquant._ext import _nax_large_row

    row = _nax_large_row(codec)
    if not row:
        pytest.skip("no Metal")
    want = [128, 64, 2, 2, POLICY_LOG.get(codec, 0), 4096, 0, MAX_K.get(codec, 0)]
    assert row == want + [int(codec not in MEASURED)]
