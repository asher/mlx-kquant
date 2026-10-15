"""Every split-K kernel name the host can build, dispatched.

For each codec with split-K kernels, the calls below force each split
route at the row counts that select each of its tiles, on both activation
dtypes, on an N that is and is not a multiple of the column tile, with
partials in the activation dtype and in float (KQ_SPLITK_F32), and with the
half verify_mma kernels (KQ_VERIFY_MMA_HALF), and run quantized_matmul_multi
for its joined verify_nax kernels. Every verify_nax call also runs with x
8 bytes into its buffer, so each verify_nax kernel runs both its 16-byte
(_x16) and 8-byte (_x8) x load pipelines. KQ_QMM_LOG names the pipeline
each call ran. Together the calls must run every name the host enumerator
lists for the codec, each verify_nax name in both pipelines, and each
result must meet the R1 bound against float64 with exact kq.dequantize
weights, so a kernel that compiles to nothing, or that no call reaches,
fails here.
"""

from __future__ import annotations

import re

import pytest
from mlx_kquant._ext import _split_kernel_names
from qmm_probe import force_cpu, run_probe

import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY


def _no_nax():
    if force_cpu():
        return True
    try:
        return not (kq.metallib_loads() and kq.nax_available())
    except Exception:
        return True


pytestmark = pytest.mark.skipif(_no_nax(), reason="NAX split-K kernel names")

R1 = {"bfloat16": 6e-3, "float16": 1e-3}
# The half verify_mma kernels sum each block in half (test_verify_mma_range).
HALF = 1e-2
K = 4096
CODEC = re.compile(
    r"^kquant_(.+?)_(?:sb_)?(?:qmm_t_splitk|qmm_t_nax_splitk|verify_mma|verify_nax)_"
)
STRICT = {"KQ_QMM_ROUTE_STRICT": "1"}
CLEAR = {
    "KQ_QMM_MIDM": None,
    "KQ_DISABLE_NAX": None,
    "KQ_QMM_ROUTE": None,
    "KQ_QMM_ROUTE_STRICT": None,
    "KQ_VERIFY_MMA_HALF": None,
    "KQ_NAX_QMV": None,
    "KQ_VERIFY_NAX": None,
    "KQ_VERIFY_NAX_SPLITS": None,
    "KQ_VNAX_X16": None,
}
# (family the arm may be declined for, rows, K, env). The split-K tile is
# BM=8 through M 8, BM=16 through M 16 and the plain tile above; the NAX
# split-K tile is BM=32 below the mid-M band and BM=64 in it.
ARMS = [
    (None, 8, K, {"KQ_QMM_ROUTE": "splitk", **STRICT}),
    (None, 16, K, {"KQ_QMM_ROUTE": "splitk", **STRICT}),
    (None, 32, K, {"KQ_QMM_ROUTE": "splitk", **STRICT}),
    (None, 8, K, {"KQ_QMM_ROUTE": "nax_splitk", **STRICT}),
    (None, 64, K, {"KQ_QMM_MIDM": "1"}),
    ("_verify_mma_", 8, K, {"KQ_QMM_ROUTE": "verify_mma", **STRICT}),
    (
        "_verify_mma_",
        8,
        K,
        {"KQ_QMM_ROUTE": "verify_mma", "KQ_VERIFY_MMA_HALF": "1", **STRICT},
    ),
    ("_verify_nax_", 8, K, {"KQ_QMM_ROUTE": "verify_nax", **STRICT}),
]
# K off the 256 step, for the codecs whose block divides it: q4_0 runs its
# 64-step verify_nax kernel there.
OFF_STEP = ("_verify_nax_", 8, K + 64, {"KQ_QMM_ROUTE": "verify_nax", **STRICT})
# quantized_matmul_multi runs the verify_nax_multi kernels: verify_nax from
# M 2 and more than one split, so the partials dtype shows in the name. The
# split count divides the K steps: 16 of 256 at K, 65 of 64 at K + 64.
MULTI = {"KQ_NAX_QMV": "0", "KQ_VERIFY_NAX": "2"}
MULTI_ARMS = [(8, K, "2"), (8, K + 64, "5")]


def _host_names():
    """Host split-K names per codec, without the shared accumulate kernels."""
    out = {}
    for n in _split_kernel_names():
        m = CODEC.match(n)
        if m:
            out.setdefault(m[1], set()).add(n)
    return out


def _pipelines(name):
    """The pipeline names KQ_QMM_LOG gives a host kernel name: a verify_nax
    kernel has an _x16 and an _x8 pipeline, named ahead of its partials
    suffix."""
    if "_verify_nax_" not in name:
        return {name}
    base, suffix = (name[:-5], "_pf32") if name.endswith("_pf32") else (name, "")
    return {base + "_x16" + suffix, base + "_x8" + suffix}


HOST = {} if _no_nax() else _host_names()


def test_split_codecs_found():
    assert {"q4_k", "q8_0", "q2_0", "pq2_0", "iq4_xs", "iq4_nl"} <= set(HOST)
    # Every codec has the three split-K tiles and both NAX split-K tiles.
    for codec, names in HOST.items():
        for frag in (
            "_qmm_t_splitk_bm8_",
            "_qmm_t_splitk_bm16_",
            "_qmm_t_nax_splitk_",
        ):
            assert any(frag in n for n in names), (codec, frag)


@pytest.mark.parametrize("codec", sorted(HOST))
def test_every_split_kernel_runs(codec):
    host = HOST[codec]
    arms = list(ARMS)
    if (K + 64) % CODEC_GEOMETRY[codec][3] == 0:
        arms.append(OFF_STEP)
    calls, families = [], []
    for dtype in ("bfloat16", "float16"):
        for f32 in ("0", "1"):
            for n in (256, 250):
                for family, m, k, env in arms:
                    for off in (0, 4) if family == "_verify_nax_" else (0,):
                        calls.append(
                            {
                                "codec": codec,
                                "shape": [m, k],
                                "n": n,
                                "dtype": dtype,
                                "seed": 1,
                                "x_offset": off,
                                "env": {**CLEAR, **env, "KQ_SPLITK_F32": f32},
                            }
                        )
                        families.append(family)
                for m, k, splits in MULTI_ARMS:
                    if k % CODEC_GEOMETRY[codec][3]:
                        continue
                    for off in (0, 4):
                        calls.append(
                            {
                                "codec": codec,
                                "shape": [m, k],
                                "n": n,
                                "multi": [n + 64],
                                "dtype": dtype,
                                "seed": 1,
                                "x_offset": off,
                                "env": {
                                    **CLEAR,
                                    **MULTI,
                                    "KQ_VERIFY_NAX_SPLITS": splits,
                                    "KQ_SPLITK_F32": f32,
                                },
                            }
                        )
                        families.append(None)
    res = run_probe(calls, env={"KQ_TUNE_LIVE": "1", "KQ_QMM_LOG": "1"})
    ran = set()
    for c, family, r in zip(calls, families, res, strict=True):
        if r["error"] is not None:
            # A forced verify route declines a codec or a K it has no
            # kernel for.
            assert family is not None and "does not serve" in r["error"], (c, r)
            continue
        half = c["env"].get("KQ_VERIFY_MMA_HALF") == "1"
        assert r["rel"] < (HALF if half else R1[c["dtype"]]), (c, r["rel"])
        ran |= {name for name, _ in r["kernels"]}
    want = set().union(*(_pipelines(n) for n in host))
    assert sorted(want - ran) == []
