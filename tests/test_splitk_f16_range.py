"""Split-K range on float16 activations.

The "cancel" design runs float16 activations near 3e4 against weights near
+0.05 on the first half of K and -0.05 on the second, at K 4096 and N 512.
The output peaks near 1.5e4, well inside the float16 range, but the partial
sum of a K slice runs past 65504. Float16 split calls store float partials
by default, so every split route returns a finite result that matches its
KQ_SPLITK_F32=1 run bit for bit. KQ_SPLITK_F32=0 stores the partials in
float16 and returns inf, so the design reaches the overflow. The unsplit
reference is finite: the tile, with the mid-M split off above M 32, and at
M 32 or less the NAX tile, or per-row qmv on GPUs without NAX.
"""

from __future__ import annotations

import functools
import math
import subprocess

import pytest
from qmm_probe import force_cpu, run_probe

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(force_cpu(), reason="GPU split-K routes")

K, N = 4096, 512
CODECS = ["q4_0", "q8_0", "q4_k"]
STRICT = {"KQ_QMM_ROUTE_STRICT": "1"}
CLEAR = {"KQ_QMM_ROUTE": None, "KQ_QMM_MIDM": None, "KQ_DISABLE_NAX": None}


def _cells(codec, nax):
    """(name, M, env, kernel name fragment) per split route cell."""
    alu = {"KQ_DISABLE_NAX": "1"} if nax else {}
    cells = [
        ("splitk", 16, {"KQ_QMM_ROUTE": "splitk", **STRICT}, "_qmm_t_splitk_"),
        ("splitk mid-M", 64, {**alu, "KQ_QMM_MIDM": "1"}, "_qmm_t_splitk_"),
    ]
    if nax:
        cells += [
            (
                "nax_splitk",
                16,
                {"KQ_QMM_ROUTE": "nax_splitk", **STRICT},
                "_qmm_t_nax_splitk_",
            ),
            ("nax mid-M", 64, {"KQ_QMM_MIDM": "1"}, "_qmm_t_nax_splitk_"),
        ]
        for m in (2, 8):
            cells.append(
                (
                    f"verify_nax M{m}",
                    m,
                    {"KQ_QMM_ROUTE": "verify_nax", **STRICT},
                    "_verify_nax_",
                )
            )
    if codec != "q4_k":
        for m in (2, 8):
            cells.append(
                (
                    f"verify_mma M{m}",
                    m,
                    {"KQ_QMM_ROUTE": "verify_mma", **STRICT},
                    "_verify_mma_",
                )
            )
    return cells


@functools.cache
def _probe(codec):
    """Per cell: the default, KQ_SPLITK_F32=1 and KQ_SPLITK_F32=0 results,
    and the unsplit reference at the cell's M."""
    nax = kq.nax_available()
    calls, index = [], {}
    for name, m, env, _ in _cells(codec, nax):
        base = {
            "codec": codec,
            "shape": [m, K],
            "n": N,
            "dtype": "float16",
            "design": "cancel",
            "seed": m,
        }
        start = len(calls)
        for lever in (None, "1", "0"):
            calls.append(dict(base, env={**CLEAR, **env, "KQ_SPLITK_F32": lever}))
        if m > 32:
            ref_env = {"KQ_QMM_MIDM": "0"}
        else:
            ref_env = {"KQ_QMM_ROUTE": "nax" if nax else "qmv", **STRICT}
        calls.append(dict(base, env={**CLEAR, **ref_env, "KQ_SPLITK_F32": None}))
        index[name] = start
    try:
        res = run_probe(calls, env={"KQ_TUNE_LIVE": "1", "KQ_QMM_LOG": "1"})
    except (AssertionError, subprocess.TimeoutExpired) as e:
        res = [{"error": f"probe failed: {e}", "kernels": []}] * len(calls)
    return {name: res[i : i + 4] for name, i in index.items()}


def _cell_names():
    nax = kq.nax_available()
    return [(c, name) for c in CODECS for name, *_ in _cells(c, nax)]


@pytest.mark.parametrize("codec,cell", _cell_names())
def test_splitk_f16_range(codec, cell):
    frag = {name: f for name, _, _, f in _cells(codec, kq.nax_available())}[cell]
    default, f32, half, ref = _probe(codec)[cell]
    for r in (default, f32, half, ref):
        assert r["error"] is None, r["error"]
    for r, pf32 in ((default, True), (f32, True), (half, False)):
        ((name, grid),) = r["kernels"]
        assert frag in name and grid[2] > 1, (name, grid)
        assert name.endswith("_pf32") == pf32, name
    assert ref["nonfinite"] == 0, ref["nonfinite"]
    assert half["nonfinite"] > 0, "the design no longer overflows float16 partials"
    assert default["nonfinite"] == 0, default["nonfinite"]
    assert default["digest"] == f32["digest"]
    # The halves cancel, so the tile's float16 rounding of the weights shows
    # as 0.7 to 1.9 percent of the output. A lost or doubled K slice moves
    # it by more than the output itself.
    assert math.isfinite(default["rel"]) and default["rel"] < 0.05, default["rel"]
