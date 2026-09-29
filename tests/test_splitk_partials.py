"""Split-K partials: the activation dtype by default, float under
KQ_SPLITK_F32=1.

The split routes (qmm_splitk, verify_mma, verify_nax, qmm_nax_splitk) fold
their K slices in f32. With float partials the fold rounds the result once,
so each route gives the rounding of the same product run unsplit; with
partials in the activation dtype each slice rounds once more.

The inputs follow the lead design: nonnegative weights and activations
whose first K column is 1024, so the first slice carries most of every
output and its rounding shows fully. At K 12288 and N 256:

- qmm_nax_splitk (the default at M 64 on NAX GPUs, and forced at M 8) and
  qmm_splitk (forced at M 64 and M 8) keep the tile's R2 floor, R2 matching
  the tile run within 2 percent of that floor.
- Every split route stays within 1 ulp and half an output rounding of its
  reference: the tile, or the same verify route run as one slice.
- KQ_SPLITK_F32=0, the default, dispatches the T-partial kernels, which sit
  more than one output rounding from that reference, with an R2 excess
  above half the floor on the qmm routes.

Dense, lazy transposed, strided and batch-of-two inputs all take the
checks. On the CPU stream the levers do nothing: every pair of calls
matches bit for bit and stays within the rounding floor of its reference.
The CPU calls run with KQ_CPU_NEON=0, since the int8 small-M kernels
quantize the activations by design.
"""

from __future__ import annotations

import math

import pytest
from qmm_probe import force_cpu, run_probe

import mlx_kquant as kq

K, N = 12288, 256
CODECS = ["q4_k", "q5_k", "q6_k", "q8_0"]
DTYPES = ["bfloat16", "float16"]
# (layout, batch): batch 2 splits the M rows over two leading entries.
LAYOUTS = [("dense", 1), ("transposed", 1), ("strided", 1), ("dense", 2)]
CLEAR = {
    "KQ_QMM_MIDM": None,
    "KQ_DISABLE_NAX": None,
    "KQ_QMM_ROUTE": None,
    "KQ_QMM_ROUTE_STRICT": None,
    "KQ_VERIFY_NAX_SPLITS": None,
    "KQ_VERIFY_MMA_SPLITS": None,
}
FORCE = {"KQ_QMM_ROUTE_STRICT": "1"}


def _shape(m, batch):
    return [batch, m // batch, K] if batch > 1 else [m, K]


def _excess(r, floor):
    return math.sqrt(max(0.0, r * r - floor * floor)) / floor


def _cells(codec, nax):
    """(name, M, split env, reference env, family) per route cell. family
    is the kernel name fragment of the split kernel."""
    cells = []
    if nax:
        cells += [
            ("nax mid-M", 64, {}, {"KQ_QMM_MIDM": "0"}, "_qmm_t_nax_splitk_"),
            (
                "nax_splitk",
                8,
                {"KQ_QMM_ROUTE": "nax_splitk", **FORCE},
                {"KQ_QMM_ROUTE": "nax", **FORCE},
                "_qmm_t_nax_splitk_",
            ),
            (
                "verify_nax",
                8,
                {"KQ_QMM_ROUTE": "verify_nax", **FORCE},
                {"KQ_QMM_ROUTE": "verify_nax", "KQ_VERIFY_NAX_SPLITS": "1", **FORCE},
                "_verify_nax_",
            ),
        ]
    alu = {"KQ_DISABLE_NAX": "1"} if nax else {}
    cells += [
        (
            "splitk mid-M",
            64,
            {**alu, "KQ_QMM_MIDM": "1"},
            {**alu, "KQ_QMM_MIDM": "0"},
            "_qmm_t_splitk_",
        ),
        (
            "splitk",
            8,
            {"KQ_QMM_ROUTE": "splitk", **FORCE},
            {"KQ_QMM_ROUTE": "nax", **FORCE} if nax else None,
            "_qmm_t_splitk_bm8_",
        ),
    ]
    if codec == "q8_0":
        cells.append(
            (
                "verify_mma",
                8,
                {"KQ_QMM_ROUTE": "verify_mma", **FORCE},
                {"KQ_QMM_ROUTE": "verify_mma", "KQ_VERIFY_MMA_SPLITS": "1", **FORCE},
                "_verify_mma_",
            )
        )
    return cells


def _run(codec, dtype, cpu):
    nax = not cpu and kq.nax_available()
    calls, meta = [], []

    def add(tag, cell, m, layout, batch, env, lever, ulp_vs=None):
        calls.append(
            {
                "codec": codec,
                "shape": _shape(m, batch),
                "n": N,
                "dtype": dtype,
                "layout": layout,
                "seed": 7 + m,
                "design": "lead",
                "r2": True,
                "env": {**CLEAR, **env, "KQ_SPLITK_F32": lever},
                "ulp_vs": ulp_vs,
            }
        )
        meta.append((tag, cell, m, layout, batch))
        return len(calls) - 1

    for name, m, split_env, ref_env, _ in _cells(codec, nax):
        for layout, batch in LAYOUTS:
            ref = None
            if ref_env is not None:
                ref = add("ref", name, m, layout, batch, ref_env, "1")
            split = add("split", name, m, layout, batch, split_env, "1", ref)
            if layout == "dense" and batch == 1:
                # Without a reference the old arm compares to the split arm.
                base = ref if ref is not None else split
                add("old", name, m, layout, batch, split_env, "0", base)
    env = {"KQ_TUNE_LIVE": "1", "KQ_QMM_LOG": "1"}
    if cpu:
        env["KQ_CPU_NEON"] = "0"
    res = run_probe(calls, env=env)
    return calls, meta, res, nax


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("codec", CODECS)
def test_splitk_partials(codec, dtype):
    cpu = force_cpu()
    calls, meta, res, nax = _run(codec, dtype, cpu)
    cells = {c[0]: c for c in _cells(codec, nax)}
    refs = {}
    for (tag, cell, _, layout, batch), r in zip(meta, res, strict=True):
        assert r["error"] is None, (tag, cell, layout, batch, r["error"])
        if tag == "ref":
            refs[(cell, layout, batch)] = r
    for c, (tag, cell, _, layout, batch), r in zip(calls, meta, res, strict=True):
        where = (codec, dtype, tag, cell, layout, batch)
        e1 = _excess(r["rel"], r["floor1"])
        e2 = _excess(r["rel2"], r["floor2"])
        if cpu:
            # The CPU matmul rounds once, on one of the two models.
            assert min(e1, e2) < 0.1, (where, e1, e2)
            assert r["kernels"] == [], where
            if c["ulp_vs"] is not None:
                assert r["ulp"] == 0, (where, r["ulp"])
            continue
        (name, grid) = r["kernels"][0]
        family = cells[cell][4]
        verify = cell.startswith("verify")
        if tag == "ref":
            if verify:
                assert family in name and grid[2] == 1, (where, name, grid)
                assert not name.endswith("_pf32"), (where, name)
            else:
                assert "splitk" not in name, (where, name)
            continue
        assert family in name and grid[2] > 1, (where, name, grid)
        assert name.endswith("_pf32") == (tag == "split"), (where, name)
        ref = refs.get((cell, layout, batch))
        # Distance to the reference in output roundings.
        dist = r["rdiff"] / r["floor2"] if ref is not None else None
        if tag == "split":
            if not verify:
                # The tile's R2 model: T weights, f32 sum, one rounding.
                assert e2 < 0.1, (where, e2)
                if ref is not None:
                    gap = abs(r["rel2"] - ref["rel2"]) / r["floor2"]
                    assert gap < 0.02, (where, gap)
            if ref is not None:
                assert r["ulp"] <= 1 and dist < 0.5, (where, r["ulp"], dist)
        else:
            if not verify:
                assert e2 > 0.5, (where, e2)
            if ref is not None:
                assert dist > 1.0, (where, dist)


# With no KQ_SPLITK_F32 set, a split call dispatches the T-partial kernel,
# and KQ_SPLITK_F32=1 its float twin.
def test_splitk_f32_default_off():
    if force_cpu():
        pytest.skip("GPU split-K kernels")
    call = {"codec": "q4_k", "shape": [8, K], "n": N, "dtype": "bfloat16"}
    env = {"KQ_QMM_LOG": "1", "KQ_QMM_ROUTE": "splitk", **FORCE}
    names = []
    for extra in ({}, {"KQ_SPLITK_F32": "1"}):
        (r,) = run_probe([call], env={**env, **extra})
        assert r["error"] is None, r["error"]
        ((name, grid),) = r["kernels"]
        assert "_qmm_t_splitk_" in name and grid[2] > 1, (name, grid)
        names.append(name)
    assert not names[0].endswith("_pf32"), names
    assert names[1] == names[0] + "_pf32", names


# Without KQ_TUNE_LIVE, KQ_SPLITK_F32 is read once per process: a change
# after the first split call does not reach the later ones.
def test_splitk_f32_keeps_first_read():
    if force_cpu():
        pytest.skip("GPU split-K kernels")
    call = {"codec": "q4_k", "shape": [8, K], "n": N, "dtype": "bfloat16"}
    env = {"KQ_QMM_LOG": "1", "KQ_QMM_ROUTE": "splitk", **FORCE}
    for first, later in (("1", [None, "0"]), ("0", [None, "1"])):
        calls = [call] + [dict(call, env={"KQ_SPLITK_F32": v}) for v in later]
        res = run_probe(calls, env={**env, "KQ_SPLITK_F32": first})
        names = []
        for r in res:
            assert r["error"] is None, r["error"]
            ((name, _),) = r["kernels"]
            names.append(name)
        pf32 = first == "1"
        assert all(n.endswith("_pf32") == pf32 for n in names), names
