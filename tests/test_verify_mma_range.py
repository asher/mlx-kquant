"""verify_mma past the half range.

The "big" design draws activations from 2e4 up, with one bfloat16 column
at 1e5, against small weights, so each block's sum runs past the half
range while the output stays inside it. The "normal" design draws
ordinary activations. verify_mma stages the activations in their own
dtype and sums each block in float. With float partials
(KQ_SPLITK_F32=1) it keeps the R1 model on both designs: the result sits
within a tenth of an output rounding of float64 over the exact weights.
The tile rounds the weights to the activation dtype (R2), so the two sit
within two output roundings of each other. With partials in the
activation dtype (KQ_SPLITK_F32=0) each K slice rounds once more, which
adds about one output rounding: the result stays finite and within one and
a half output roundings of the R1 model. KQ_VERIFY_MMA_HALF=1 selects the
half kernels, whose range stops at 65504, on every codec but ptq1_0, which
has none. With the same partials on ordinary float16 activations, their
error beyond the R1 model is at least three times that of the float
kernels.
"""

from __future__ import annotations

import functools
import math
import re
import subprocess

import pytest
from qmm_probe import force_cpu, run_probe

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(force_cpu(), reason="GPU verify route")

CODECS = ["pq2_0", "ptq1_0", "q4_0", "q8_0"]
DTYPES = ["bfloat16", "float16"]
# design: (K, N)
DESIGNS = {"normal": (4096, 512), "big": (1024, 256)}
MS = [8, 3]
FORCE = {"KQ_QMM_ROUTE_STRICT": "1", "KQ_QMM_LOG": "1"}


def _half(name):
    """Whether a verify_mma kernel name is a KQ_VERIFY_MMA_HALF kernel."""
    return re.search(r"_b_\d+_h(_pf32)?$", name) is not None


def _excess(r, floor):
    return math.sqrt(max(0.0, r * r - floor * floor)) / floor


@functools.cache
def _probe():
    """Tile, verify_mma with float and with T partials, and
    KQ_VERIFY_MMA_HALF=1 results per case."""
    tile = "nax" if kq.nax_available() else "splitk"
    arms = [
        {"KQ_QMM_ROUTE": tile, "KQ_VERIFY_MMA_HALF": None, "KQ_SPLITK_F32": "1"},
        {"KQ_QMM_ROUTE": "verify_mma"},
        {"KQ_SPLITK_F32": "0"},
        {"KQ_VERIFY_MMA_HALF": "1"},
    ]
    cases = [(d, c, t, m) for d in DESIGNS for c in CODECS for t in DTYPES for m in MS]
    calls = []
    for design, codec, dtype, m in cases:
        k, n = DESIGNS[design]
        base = len(calls)
        for i, env in enumerate(arms):
            calls.append(
                {
                    "codec": codec,
                    "shape": [m, k],
                    "n": n,
                    "dtype": dtype,
                    "seed": 5 + m,
                    "design": design,
                    "r2": True,
                    "env": env,
                    "ulp_vs": base if i else None,
                }
            )
    try:
        res = run_probe(calls, env={"KQ_TUNE_LIVE": "1", **FORCE})
    except (AssertionError, subprocess.TimeoutExpired) as e:
        res = [{"error": f"probe failed: {e}", "kernels": []}] * len(calls)
    w = len(arms)
    return {case: res[w * i : w * i + w] for i, case in enumerate(cases)}


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("design", list(DESIGNS))
def test_verify_mma_range(design, codec, dtype, m):
    tile, verify, verify_t, _ = _probe()[(design, codec, dtype, m)]
    for r in (tile, verify, verify_t):
        assert r["error"] is None, r["error"]
        assert len(r["kernels"]) == 1, r["kernels"]
    for r, pf32 in ((verify, True), (verify_t, False)):
        name = r["kernels"][0][0]
        assert "_verify_mma_" in name and not _half(name), name
        assert name.endswith("_pf32") == pf32, name
    e1 = _excess(verify["rel"], verify["floor1"])
    assert e1 < 0.1, e1
    # Distance to the tile in output roundings.
    dist = verify["rdiff"] / verify["floor2"]
    assert dist < 2.0, dist
    assert math.isfinite(verify_t["rel"]), verify_t["rel"]
    assert _excess(verify_t["rel"], verify_t["floor1"]) < 1.5


# The half kernels behind KQ_VERIFY_MMA_HALF=1, on ordinary activations.
# ptq1_0 has none, so the lever leaves it on its float kernels.
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("codec", CODECS)
def test_verify_mma_half_lever(codec, dtype):
    for m in MS:
        _, _, verify_t, half = _probe()[("normal", codec, dtype, m)]
        assert half["error"] is None, half["error"]
        ((name, _),) = half["kernels"]
        if codec == "ptq1_0":
            assert "_verify_mma_" in name and not _half(name), name
            assert half["digest"] == verify_t["digest"]
            continue
        assert "_verify_mma_" in name and _half(name), name
        assert math.isfinite(half["rel"]) and half["rel"] < 1e-2, half["rel"]
        if dtype == "float16":
            e_half = _excess(half["rel"], half["floor1"])
            e_float = _excess(verify_t["rel"], verify_t["floor1"])
            assert e_half > 3 * e_float, (e_half, e_float)
