"""PQ2_0 and Q2_0 mat-vec past the half range.

The "big" design draws activations from 2e4 up, with one bfloat16 column
at 1e5, against small weights, so the output stays inside the bfloat16
range. The M=1 mat-vec and the verify-shaped mat-vec stage bfloat16
activations in float, so the result is finite and sits within a tenth of
an output rounding of float64 over the exact weights. float16 activations
keep the half pipe, whose codes are pre-scaled by 1/16 so a lane's sum
stays below the largest activation; its half sums add about two output
roundings. q2_0 runs the same kernels on a 64-weight block.
"""

from __future__ import annotations

import functools
import math
import subprocess

import pytest
from qmm_probe import force_cpu, run_probe

pytestmark = pytest.mark.skipif(force_cpu(), reason="GPU mat-vec route")

# (M, forced route): M=1 takes the mat-vec by default.
ROUTES = [(1, None), (4, "qmv"), (4, "verify_qmv")]
DTYPES = ["bfloat16", "float16"]
DESIGNS = ["big", "normal"]
# Error beyond the rounding of the float64 result, in output roundings.
EXCESS = {"bfloat16": 0.1, "float16": 2.5}


def _excess(r, floor):
    return math.sqrt(max(0.0, r * r - floor * floor)) / floor


@functools.cache
def _probe(codec="pq2_0"):
    cases = [(d, t, m, r) for d in DESIGNS for t in DTYPES for m, r in ROUTES]
    calls = [
        {
            "codec": codec,
            "shape": [m, 1024],
            "n": 256,
            "dtype": dtype,
            "seed": 3,
            "design": design,
            "r2": True,
            "env": {
                "KQ_QMM_ROUTE": route,
                "KQ_QMM_ROUTE_STRICT": "1" if route else None,
            },
        }
        for design, dtype, m, route in cases
    ]
    try:
        res = run_probe(calls, env={"KQ_TUNE_LIVE": "1", "KQ_QMM_LOG": "1"})
    except (AssertionError, subprocess.TimeoutExpired) as e:
        res = [{"error": f"probe failed: {e}", "kernels": []}] * len(calls)
    return dict(zip(cases, res, strict=True))


@pytest.mark.parametrize("m,route", ROUTES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("design", DESIGNS)
def test_pq2_0_mv_range(design, dtype, m, route):
    _check(_probe()[(design, dtype, m, route)], dtype)


@pytest.mark.parametrize("m,route", ROUTES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("design", DESIGNS)
def test_q2_0_mv_range(design, dtype, m, route):
    _check(_probe("q2_0")[(design, dtype, m, route)], dtype)


def _check(r, dtype):
    assert r["error"] is None, r["error"]
    # One mat-vec kernel, no tile.
    ((name, _),) = r["kernels"]
    assert "_qmv_" in name, name
    assert r["nonfinite"] == 0, r["nonfinite"]
    assert _excess(r["rel"], r["floor1"]) < EXCESS[dtype], (r["rel"], r["floor1"])
