"""Split-K kernel names, with no dispatch.

Each split-K kernel has a _pf32 twin that stores float partials (float16
by default, bfloat16 under KQ_SPLITK_F32=1), and each twin has its base
kernel. The ops cast float32 activations to bfloat16, so no split-K kernel
takes float activations. Every name the split-K dispatchers can build must
be in the metallib.
"""

from __future__ import annotations

import re

import pytest
from mlx_kquant._ext import _split_kernel_names
from qmm_probe import force_cpu

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(force_cpu(), reason="Metal kernel names")

FAMILY = re.compile(
    r"_(qmm_t_splitk|qmm_t_nax_splitk|verify_mma|verify_nax)_|^kquant_qmm_splitk_accum_"
)
TYPE = re.compile(r"_(bfloat16_t|float16_t|float)(?=_gs_|$)")
BASE = "kquant_q4_k_qmm_t_splitk_bm8_bfloat16_t_gs_256_b_4_alN_true"
FLOAT = "kquant_q4_k_qmm_t_splitk_bm8_float_gs_256_b_4_alN_true"


def _pairing_errors(names):
    """(reason, name) per break: a split-K kernel on float activations, a
    kernel without its _pf32 twin, a twin without its base."""
    names = set(names)
    errors = []
    for n in sorted(names):
        if not FAMILY.search(n):
            continue
        base = n.removesuffix("_pf32")
        m = TYPE.search(base)
        if m and m[1] == "float":
            errors.append(("float", n))
        elif n != base:
            if base not in names:
                errors.append(("no base", n))
        elif n + "_pf32" not in names:
            errors.append(("no twin", n))
    return errors


def test_split_kernels_pair():
    names = kq.metallib_kernels()
    split = [n for n in names if FAMILY.search(n)]
    assert any(n.endswith("_pf32") for n in split), len(split)
    assert _pairing_errors(names) == []


def test_pairing_check_flags_breaks():
    names = set(kq.metallib_kernels())
    twin = BASE + "_pf32"
    assert {BASE, twin} <= names and FLOAT not in names
    assert _pairing_errors(names - {BASE}) == [("no base", twin)]
    assert _pairing_errors(names - {twin}) == [("no twin", BASE)]
    assert _pairing_errors(names | {FLOAT}) == [("float", FLOAT)]
    bad = FLOAT + "_pf32"
    assert _pairing_errors(names | {bad}) == [("float", bad)]


def test_host_split_names_exist():
    host = set(_split_kernel_names())
    lib = set(kq.metallib_kernels())
    assert sorted(host - lib)[:10] == []
    assert host - (lib - {BASE}) == {BASE}
    for frag in (
        "_qmm_t_splitk_bm8_",
        "_qmm_t_splitk_bm16_",
        "_qmm_t_nax_splitk_",
        "_verify_mma_",
        "_verify_nax_",
        "kquant_q4_0_sb_verify_nax_",
        "kquant_qmm_splitk_accum_",
    ):
        assert any(frag in n for n in host), frag
    for end in ("_pf32", "_h", "_h_pf32"):
        assert any(n.endswith(end) for n in host), end
    assert not any(TYPE.search(n.removesuffix("_pf32"))[1] == "float" for n in host)
