"""Split-K kernel names, with no dispatch.

Each split-K kernel has a _pf32 twin that stores float partials (float16
by default, bfloat16 under KQ_SPLITK_F32=1), and each twin has its base
kernel. The ops cast float32 activations to bfloat16, so no split-K kernel
takes float activations, and a codec outside the split-K codecs has no
split-K kernel. Every name the split-K dispatchers can build must be in the
metallib.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest
from mlx_kquant._ext import _split_kernel_names
from qmm_probe import force_cpu

import mlx_kquant as kq


def _no_metallib():
    """True under KQUANT_FORCE_CPU, on a build without Metal, or when the
    bundled metallib does not load."""
    if force_cpu():
        return True
    try:
        return not kq.metallib_loads()
    except Exception:
        return True


pytestmark = pytest.mark.skipif(_no_metallib(), reason="Metal kernel names")

FAMILY = re.compile(
    r"_(qmm_t_splitk|qmm_t_nax_splitk|verify_mma|verify_nax)_|^kquant_qmm_splitk_accum_"
)
TYPE = re.compile(r"_(bfloat16_t|float16_t|float)(?=_gs_|$)")
CODEC = re.compile(
    r"^kquant_(.+?)_(?:sb_)?(?:qmm_t_splitk|qmm_t_nax_splitk|verify_mma|verify_nax)_"
)
BASE = "kquant_q4_k_qmm_t_splitk_bm8_bfloat16_t_gs_256_b_4_alN_true"
FLOAT = "kquant_q4_k_qmm_t_splitk_bm8_float_gs_256_b_4_alN_true"
FP8 = "kquant_mxfp8_qmm_t_splitk_bfloat16_t_gs_32_b_8_alN_true"


def _split_codecs():
    """The codecs kq_splitk_codec admits: those the host enumerator names a
    qmm_t_splitk kernel for."""
    return {CODEC.match(n)[1] for n in _split_kernel_names() if "_qmm_t_splitk_" in n}


def _pairing_errors(names, codecs):
    """(reason, name) per break: a split-K kernel on float activations, one
    for a codec outside codecs, a kernel without its _pf32 twin, a twin
    without its base."""
    names = set(names)
    errors = []
    for n in sorted(names):
        if not FAMILY.search(n):
            continue
        base = n.removesuffix("_pf32")
        m = TYPE.search(base)
        c = CODEC.match(n)
        if m and m[1] == "float":
            errors.append(("float", n))
        elif c and c[1] not in codecs:
            errors.append(("codec", n))
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
    assert _pairing_errors(names, _split_codecs()) == []


def test_pairing_check_flags_breaks():
    names = set(kq.metallib_kernels())
    codecs = _split_codecs()
    assert "q4_k" in codecs and "mxfp8" not in codecs, sorted(codecs)
    twin = BASE + "_pf32"
    assert {BASE, twin} <= names and not {FLOAT, FP8} & names
    assert _pairing_errors(names - {BASE}, codecs) == [("no base", twin)]
    assert _pairing_errors(names - {twin}, codecs) == [("no twin", BASE)]
    assert _pairing_errors(names | {FLOAT}, codecs) == [("float", FLOAT)]
    bad = FLOAT + "_pf32"
    assert _pairing_errors(names | {bad}, codecs) == [("float", bad)]
    assert _pairing_errors(names | {FP8}, codecs) == [("codec", FP8)]


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
        "kquant_q4_1_sb_verify_nax_",
        "kquant_q4_1_verify_mma_",
        "kquant_qmm_splitk_accum_",
    ):
        assert any(frag in n for n in host), frag
    for end in ("_pf32", "_h", "_h_pf32"):
        assert any(n.endswith(end) for n in host), end
    assert not any(TYPE.search(n.removesuffix("_pf32"))[1] == "float" for n in host)


def test_host_names_follow_the_dispatch_levers():
    # The enumerator takes its tiles and verify_nax steps from the dispatch
    # helpers, so the levers that retire a tile or a step retire its names.
    env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    env.update(
        KQ_QMM_SPLITK_BM8="0", KQ_VERIFY_NAX_Q4_0_SB="0", KQ_VERIFY_NAX_Q4_1_SB="0"
    )
    code = (
        "import json; from mlx_kquant._ext import _split_kernel_names as f; "
        "print(json.dumps(f()))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    names = set(json.loads(out.stdout.strip().splitlines()[-1]))
    host = set(_split_kernel_names())
    assert names < host
    retired = {
        n
        for n in host
        if "_splitk_bm8_" in n or n.startswith(("kquant_q4_0_sb_", "kquant_q4_1_sb_"))
    }
    assert retired and host - names == retired
    assert any(n.startswith("kquant_q4_1_sb_") for n in retired)
