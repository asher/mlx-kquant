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
fixed target. The mat-vec routes do not log, so the output digest shows
which of them ran.
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
    if (codec, m) in MV_EXT:
        assert capped["kernels"] == [], capped["kernels"]
    else:
        ((name, grid),) = capped["kernels"]
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
    assert capped["kernels"] == []


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
