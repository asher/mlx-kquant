"""Environment lever values.

An empty value reads as unset. 0, false, off and no read as 0, and 1, true,
on and yes as 1, in any case. Any other value reads as unset, with one
warning on stderr the first time the lever holds it. So on a lever whose
unset default differs from 0, an empty or mistyped value keeps the default.
The kernel names of KQ_QMM_LOG show which value each call read.
"""

from __future__ import annotations

import re

import pytest
from qmm_probe import force_cpu, run_probe

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(force_cpu(), reason="GPU kernel names")

K, N = 12288, 256
LOG = {
    "KQ_QMM_LOG": "1",
    "KQ_TUNE_LIVE": "1",
    "KQ_QMM_ROUTE_STRICT": "1",
}

# KQ_SPLITK_F32 on float16, where unset means float partials: (value, _pf32
# kernel, warning). Its set values are 0 and 1, so 2 warns too. The second
# "x" call finds the warning already given.
F32_VALUES = [
    ("", True, False),
    ("true", True, False),
    ("On", True, False),
    ("yes", True, False),
    ("1", True, False),
    ("false", False, False),
    ("off", False, False),
    ("NO", False, False),
    ("0", False, False),
    ("2", True, True),
    ("x", True, True),
    ("x", True, False),
]


def _warnings(r, lever, value):
    return [line for line in r["stderr"] if f"{lever}={value} is not" in line]


def test_splitk_f32_values():
    call = {"codec": "q4_k", "shape": [8, K], "n": N, "dtype": "float16"}
    calls = [dict(call, env={"KQ_SPLITK_F32": v}) for v, _, _ in F32_VALUES]
    res = run_probe(calls, env={**LOG, "KQ_QMM_ROUTE": "splitk"})
    for (v, pf32, warns), r in zip(F32_VALUES, res, strict=True):
        assert r["error"] is None, (v, r["error"])
        ((name, _),) = r["kernels"]
        assert "_qmm_t_splitk_" in name and name.endswith("_pf32") == pf32, (v, name)
        assert len(_warnings(r, "KQ_SPLITK_F32", v)) == warns, (v, r["stderr"])


# KQ_QMM_SPLITK_BM8 is on unless 0 and read once per process, so each value
# runs in its own probe.
@pytest.mark.parametrize(
    "value,bm8", [(None, True), ("", True), ("off", False), ("0", False)]
)
def test_splitk_bm8_values(value, bm8):
    call = {"codec": "q4_k", "shape": [8, K], "n": N, "dtype": "bfloat16"}
    env = {**LOG, "KQ_QMM_ROUTE": "splitk"}
    if value is not None:
        env["KQ_QMM_SPLITK_BM8"] = value
    (r,) = run_probe([call], env=env)
    assert r["error"] is None, r["error"]
    ((name, _),) = r["kernels"]
    assert ("_qmm_t_splitk_bm8_" in name) == bm8, (value, name)


# KQ_VERIFY_NAX_Q4_0_SB and KQ_VERIFY_NAX_Q4_1_SB are on unless 0 and read
# live: q4_0 and q4_1 at a K that is a multiple of 256 take the eight-block
# kernel unless the codec's lever is off.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
@pytest.mark.parametrize("codec", ["q4_0", "q4_1"])
def test_verify_nax_sb_values(codec):
    lever = f"KQ_VERIFY_NAX_{codec.upper()}_SB"
    values = [(None, True), ("", True), ("typo", True), ("no", False)]
    call = {"codec": codec, "shape": [8, 3840], "n": 1000, "dtype": "bfloat16"}
    calls = [dict(call, env={lever: v}) for v, _ in values]
    res = run_probe(calls, env={**LOG, "KQ_QMM_ROUTE": "verify_nax"})
    for (v, sb), r in zip(values, res, strict=True):
        assert r["error"] is None, (v, r["error"])
        ((name, _),) = r["kernels"]
        assert name.startswith(f"kquant_{codec}_sb_verify_nax_") == sb, (v, name)
    assert _warnings(res[2], lever, "typo"), res[2]["stderr"]


# KQ_VERIFY_NAX_NVFP4_SB is a flag read live: unset, nvfp4 at a K that is
# a multiple of 256 takes its kernel by shape (the two-block one at N 1000,
# K 3840), 1 takes the four-block kernel and 0 the two-block one. Other
# values warn and keep the choice by shape.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
def test_verify_nax_nvfp4_sb_values():
    lever = "KQ_VERIFY_NAX_NVFP4_SB"
    values = [
        (None, False),
        ("", False),
        ("1", True),
        ("on", True),
        ("0", False),
        ("2", False),
        ("typo", False),
    ]
    call = {"codec": "nvfp4", "shape": [8, 3840], "n": 1000, "dtype": "bfloat16"}
    calls = [dict(call, env={lever: v}) for v, _ in values]
    res = run_probe(calls, env={**LOG, "KQ_QMM_ROUTE": "verify_nax"})
    for (v, sb), r in zip(values, res, strict=True):
        assert r["error"] is None, (v, r["error"])
        ((name, _),) = r["kernels"]
        assert name.startswith("kquant_nvfp4_sb_verify_nax_") == sb, (v, name)
    assert _warnings(res[5], lever, "2"), res[5]["stderr"]
    assert _warnings(res[6], lever, "typo"), res[6]["stderr"]


# KQ_VNAX_X16 is on unless 0, read live under KQ_TUNE_LIVE: x at a 16-byte
# boundary takes the _x16 pipeline unless it is off. Values outside 0 and 1
# warn and keep it on.
@pytest.mark.skipif(not kq.nax_available(), reason="NAX verify only")
def test_vnax_x16_values():
    values = [
        (None, True, False),
        ("", True, False),
        ("on", True, False),
        ("1", True, False),
        ("off", False, False),
        ("0", False, False),
        ("2", True, True),
        ("x", True, True),
    ]
    call = {"codec": "q4_k", "shape": [8, 3840], "n": 1000, "dtype": "bfloat16"}
    calls = [dict(call, env={"KQ_VNAX_X16": v}) for v, _, _ in values]
    res = run_probe(calls, env={**LOG, "KQ_QMM_ROUTE": "verify_nax"})
    for (v, x16, warns), r in zip(values, res, strict=True):
        assert r["error"] is None, (v, r["error"])
        ((name, _),) = r["kernels"]
        assert "_verify_nax_" in name, (v, name)
        assert re.search("_x16(_pf32)?$" if x16 else "_x8(_pf32)?$", name), (v, name)
        assert bool(_warnings(r, "KQ_VNAX_X16", v)) == warns, (v, r["stderr"])
