"""KQ_SPLITK_CAP, the grid cap on the small-M split-K slice target.

The cap holds the split grid of the NAX split-K route and the ALU split
route (M <= 32) to n threadgroups per GPU core. With 40 cores pinned and
n = 1, N 4096 takes one slice at every M 9 to 32, and the call falls
through to the route the split displaced: the BM=32 tile at M 13 to 32,
and at M 9 to 12 the small-M qmm route or mv_ext, by codec and shape, or
per-row qmv under KQ_VERIFY_EXT=0. verify_qmv serves M 8 at most, so it
is no fallthrough here. Each capped call must match the call with the
split route off bit for bit, and float64 within the R1 bound. A partial
cap cuts the slice count, a forced target is not capped, and 0 keeps the
fixed target. The logged kernel shows which route ran.

Unset, the cap keeps the fixed target on the ALU route, and the NAX route
takes its default: one slice at a grid of 16 threadgroups per core or more
where the call without the split runs the same BM=32 tile, except q6_k and
iq2_xxs below M 32, and q4_0 and q8_0 on float partials held to 16
threadgroups per core below a full grid. q8_0 below M 13 keeps the target
on every grid.
"""

from __future__ import annotations

import functools
import subprocess

import pytest
from qmm_probe import force_cpu, run_probe, tile_of

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(force_cpu(), reason="GPU routing lever")

R1 = 6e-3
N = 4096
K = 5120
CORES = {"KQ_GPU_CORES": "40", "KQ_QMM_LOG": "1"}
CODECS = ["q4_k", "q6_k", "q8_0"]
MS = list(range(9, 33))
# The M 9 to 12 calls whose fallthrough is mv_ext at N 4096, K 5120: q8_0
# takes no small-M qmm route below N 8192, and q6_k enters it at M 11.
MV_EXT = {("q6_k", 9), ("q6_k", 10)} | {("q8_0", m) for m in range(9, 13)}
# Entry M of the ALU split route on NAX GPUs, at or above the entry on
# other GPUs, so the route runs by default from here on either.
ALU_ENTRY = {"q4_k": 8, "q6_k": 12, "q8_0": 16}


def _probe(calls, env):
    """run_probe results, or one failure per call if the probe fails, so the
    tests that share a probe fail on their own entries."""
    try:
        return run_probe(calls, env=env)
    except (AssertionError, subprocess.TimeoutExpired) as e:
        return [{"error": f"probe failed: {e}", "kernels": []}] * len(calls)


def _nax_gpu():
    return kq.nax_available()


def _split(res, route):
    """(name, grid) of a split-K dispatch of ``route`` with more than one
    slice."""
    assert res["error"] is None, res["error"]
    assert len(res["kernels"]) == 1, res["kernels"]
    name, grid = res["kernels"][0]
    assert route in name and grid[2] > 1, (name, grid)
    return name, grid


def _same(capped, off):
    """The capped call ran what the call with the split off ran."""
    for r in (capped, off):
        assert r["error"] is None, r["error"]
        assert r["rel"] < R1, r["rel"]
    assert all("splitk" not in n for n, _ in capped["kernels"]), capped["kernels"]
    assert capped["kernels"] == off["kernels"], (capped["kernels"], off["kernels"])
    assert capped["digest"] == off["digest"]


def _call(codec, m, env, n=N):
    return {"codec": codec, "shape": [m, K], "n": n, "seed": m, "env": env}


@functools.cache
def _nax_fallthrough():
    """Default, KQ_SPLITK_CAP=1 and KQ_QMM_SPLITK_NAX=0 results per (codec,
    M)."""
    arms = [
        {"KQ_SPLITK_CAP": None, "KQ_QMM_SPLITK_NAX": None},
        {"KQ_SPLITK_CAP": "1"},
        {"KQ_SPLITK_CAP": None, "KQ_QMM_SPLITK_NAX": "0"},
    ]
    cases = [(c, m) for c in CODECS for m in MS]
    calls = [_call(c, m, env) for c, m in cases for env in arms]
    res = _probe(calls, CORES)
    return {case: res[3 * i : 3 * i + 3] for i, case in enumerate(cases)}


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("codec", CODECS)
def test_nax_cap_falls_through(codec, m):
    if not _nax_gpu():
        pytest.skip("NAX split-K route")
    default, capped, off = _nax_fallthrough()[(codec, m)]
    _split(default, "_qmm_t_nax_splitk_")
    assert default["rel"] < R1, default["rel"]
    _same(capped, off)
    ((name, grid),) = capped["kernels"]
    if (codec, m) in MV_EXT:
        assert "_mv_ext_" in name, name
    else:
        assert tile_of(name)[0] == 32 and grid[2] == 1, (name, grid)


# KQ_VERIFY_EXT is read once, so the per-row qmv fallthrough runs in its own
# process. KQ_QMM_SPLITK_NAX=1 forces the split route with the default
# target, which the cap still holds.
QMV_MS = [9, 10, 11, 12]


@functools.cache
def _nax_qmv_fallthrough():
    arms = [
        {"KQ_SPLITK_CAP": None, "KQ_QMM_SPLITK_NAX": "1"},
        {"KQ_SPLITK_CAP": "1", "KQ_QMM_SPLITK_NAX": "1"},
        {"KQ_SPLITK_CAP": None, "KQ_QMM_SPLITK_NAX": "0"},
    ]
    calls = [_call("q8_0", m, env) for m in QMV_MS for env in arms]
    res = _probe(calls, dict(CORES, KQ_VERIFY_EXT="0"))
    return {m: res[3 * i : 3 * i + 3] for i, m in enumerate(QMV_MS)}


@pytest.mark.parametrize("m", QMV_MS)
def test_nax_cap_falls_through_to_qmv(m):
    if not _nax_gpu():
        pytest.skip("NAX split-K route")
    forced, capped, off = _nax_qmv_fallthrough()[m]
    _split(forced, "_qmm_t_nax_splitk_")
    _same(capped, off)
    ((name, _),) = capped["kernels"]
    assert "_qmv_" in name, name


# N 1024 at M 16 is 16 BM=32 tiles, and K 5120 is 20 units of 256 weights,
# which the default target of 16 cuts into 10 slices. A cap of 2 per core
# over 40 cores leaves room for 5. A forced target ignores the cap, and 0
# keeps the default.
def test_nax_cap_partial():
    if not _nax_gpu():
        pytest.skip("NAX split-K route")
    arms = [
        {"KQ_SPLITK_CAP": None},
        {"KQ_SPLITK_CAP": "2"},
        {"KQ_SPLITK_CAP": "1", "KQ_QMM_SPLITK_NAX": "10"},
        {"KQ_SPLITK_CAP": "0", "KQ_QMM_SPLITK_NAX": None},
    ]
    res = _probe([_call("q4_k", 16, env, n=1024) for env in arms], CORES)
    depth = [_split(r, "_qmm_t_nax_splitk_")[1][2] for r in res]
    assert depth == [10, 5, 10, 10], depth
    for r in res:
        assert r["rel"] < R1, r["rel"]


def _alu_cases():
    return [(c, m) for c in CODECS for m in MS if m >= max(ALU_ENTRY[c], 9)]


@functools.cache
def _alu_fallthrough():
    """Default and KQ_SPLITK_CAP=1 results per (codec, M) with NAX off, and
    the same calls with the ALU split route off in a second process
    (KQ_QMM_SPLITK is read once)."""
    env = dict(CORES, KQ_DISABLE_NAX="1")
    cases = _alu_cases()
    calls = [_call(c, m, {"KQ_SPLITK_CAP": v}) for c, m in cases for v in (None, "1")]
    res = _probe(calls, env)
    off = _probe([_call(c, m, {}) for c, m in cases], dict(env, KQ_QMM_SPLITK="0"))
    return {case: (res[2 * i], res[2 * i + 1], off[i]) for i, case in enumerate(cases)}


@pytest.mark.parametrize("codec,m", _alu_cases())
def test_alu_cap_falls_through(codec, m):
    default, capped, off = _alu_fallthrough()[(codec, m)]
    _split(default, "_qmm_t_splitk_")
    assert default["rel"] < R1, default["rel"]
    _same(capped, off)


# N 1024 at M 24 is 32 BM=32 by BN=32 tiles, and K 5120 is 20 blocks, which
# the default target of 16 cuts into 10 slices. A cap of 2 per core over 40
# cores leaves room for 2. A forced target (KQ_QMM_SPLITK, read once)
# ignores the cap.
def test_alu_cap_partial():
    env = dict(CORES, KQ_DISABLE_NAX="1")
    res = _probe(
        [_call("q4_k", 24, {"KQ_SPLITK_CAP": v}, n=1024) for v in (None, "2")], env
    )
    forced = _probe(
        [_call("q4_k", 24, {"KQ_SPLITK_CAP": "1"}, n=1024)],
        dict(env, KQ_QMM_SPLITK="4"),
    )
    depth = [_split(r, "_qmm_t_splitk_")[1][2] for r in res + forced]
    assert depth == [10, 2, 4], depth
    for r in res + forced:
        assert r["rel"] < R1, r["rel"]


# Unset, the NAX route takes a slice target by grid, codec and partials. 4
# cores pinned make N 4096 a full grid at every M through 32: 64 BM=32
# tiles, 16 per core. A full grid takes one slice and runs what the call
# with the split route off runs where that call runs the BM=32 tile.
# The split stays where it would run an mv route (iq4_xs at M 9, below its
# small-M qmm route at N 4096) or the BM=64 tile (iq2_xs), for q6_k and
# iq2_xxs below M 32, and for q8_0 below M 13. KQ_SPLITK_CAP=0 keeps the
# fixed target.
HEAD_ENV = {"KQ_GPU_CORES": "4", "KQ_QMM_LOG": "1"}
HEAD_CODECS = ["q4_0", "q4_k", "q6_k", "q8_0", "iq4_xs", "iq2_xs", "iq2_xxs"]
HEAD_MS = [9, 15, 16, 32]
HEAD_DTYPES = ["bfloat16", "float16"]
# iq2_xxs enters the split route at M 13.
HEAD_CASES = [
    (c, m) for c in HEAD_CODECS for m in HEAD_MS if not (c == "iq2_xxs" and m < 13)
]


def _head_keeps_split(codec, m):
    if codec == "iq2_xs":
        return True
    if codec in ("q6_k", "iq2_xxs"):
        return m < 32
    return codec in ("q8_0", "iq4_xs") and m <= 12


@functools.cache
def _nax_head():
    arms = [
        {"KQ_SPLITK_CAP": None, "KQ_QMM_SPLITK_NAX": None},
        {"KQ_SPLITK_CAP": "0", "KQ_QMM_SPLITK_NAX": None},
        {"KQ_SPLITK_CAP": None, "KQ_QMM_SPLITK_NAX": "0"},
    ]
    cases = [(c, m, dt) for c, m in HEAD_CASES for dt in HEAD_DTYPES]
    calls = [dict(_call(c, m, env), dtype=dt) for c, m, dt in cases for env in arms]
    res = _probe(calls, HEAD_ENV)
    return {case: res[3 * i : 3 * i + 3] for i, case in enumerate(cases)}


@pytest.mark.parametrize("dtype", HEAD_DTYPES)
@pytest.mark.parametrize(("codec", "m"), HEAD_CASES)
def test_nax_default_full_grid_takes_one_slice(codec, m, dtype):
    if not _nax_gpu():
        pytest.skip("NAX split-K route")
    default, fixed, off = _nax_head()[(codec, m, dtype)]
    _split(fixed, "_qmm_t_nax_splitk_")
    assert fixed["rel"] < R1, fixed["rel"]
    if _head_keeps_split(codec, m):
        assert default["kernels"] == fixed["kernels"], default["kernels"]
        assert default["digest"] == fixed["digest"]
    else:
        _same(default, off)
        assert tile_of(off["kernels"][0][0])[0] == 32, off["kernels"]


# q8_0 below M 13 keeps the split on a full grid also where the call
# without it runs the BM=32 tile: at N 8192 its small-M qmm route starts
# at M 8.
@pytest.mark.parametrize("dtype", HEAD_DTYPES)
@pytest.mark.parametrize("m", [9, 12])
def test_nax_default_q8_0_small_m_full_grid_keeps_split(m, dtype):
    if not _nax_gpu():
        pytest.skip("NAX split-K route")
    arms = [{"KQ_SPLITK_CAP": None}, {"KQ_SPLITK_CAP": "0"}, {"KQ_QMM_SPLITK_NAX": "0"}]
    default, fixed, off = _probe(
        [dict(_call("q8_0", m, env, n=8192), dtype=dtype) for env in arms], HEAD_ENV
    )
    _split(fixed, "_qmm_t_nax_splitk_")
    assert tile_of(off["kernels"][0][0])[0] == 32, off["kernels"]
    assert default["kernels"] == fixed["kernels"], default["kernels"]
    assert default["digest"] == fixed["digest"]
    assert default["rel"] < R1, default["rel"]


# Below a full grid the target stays 16, except q4_0 and q8_0 on float
# partials, which hold 16 threadgroups per core. With 40 cores at N 4096 and
# M 16 that leaves room for 10 slices. Float partials are the default on
# float16 and KQ_SPLITK_F32=1 sets them on bfloat16.
@pytest.mark.parametrize("codec", ["q4_0", "q8_0", "q4_k"])
def test_nax_default_float_partials_cap(codec):
    if not _nax_gpu():
        pytest.skip("NAX split-K route")
    arms = [
        ("float16", {"KQ_SPLITK_CAP": None}),
        ("float16", {"KQ_SPLITK_CAP": "0"}),
        ("bfloat16", {"KQ_SPLITK_CAP": None}),
    ]
    calls = [dict(_call(codec, 16, env), dtype=dt) for dt, env in arms]
    res = _probe(calls, CORES)
    depth = [_split(r, "_qmm_t_nax_splitk_")[1][2] for r in res]
    want = {"q4_0": [10, 16, 16], "q8_0": [10, 16, 16], "q4_k": [10, 10, 10]}
    assert depth == want[codec], depth
    for r in res:
        assert r["rel"] < R1, r["rel"]


# q8_0 on float partials keeps the target of 16 below M 13.
def test_nax_default_float_partials_q8_0_small_m():
    if not _nax_gpu():
        pytest.skip("NAX split-K route")
    res = _probe(
        [
            dict(_call("q8_0", m, {"KQ_SPLITK_CAP": None}), dtype="float16")
            for m in (9, 12)
        ],
        CORES,
    )
    depth = [_split(r, "_qmm_t_nax_splitk_")[1][2] for r in res]
    assert depth == [16, 16], depth
    for r in res:
        assert r["rel"] < R1, r["rel"]


# KQ_SPLITK_F32 is read once, so float partials on bfloat16 run in their own
# process.
@pytest.mark.parametrize("codec", ["q4_0", "q8_0"])
def test_nax_default_float_partials_cap_bfloat16(codec):
    if not _nax_gpu():
        pytest.skip("NAX split-K route")
    res = _probe(
        [_call(codec, 16, {"KQ_SPLITK_CAP": None})], dict(CORES, KQ_SPLITK_F32="1")
    )
    depth = [_split(r, "_qmm_t_nax_splitk_")[1][2] for r in res]
    assert depth == [10], depth
    assert res[0]["rel"] < R1, res[0]["rel"]
