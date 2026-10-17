"""The c32 kernel defaults by GPU class: which codecs take them and the
K-lanes per row of the GLU gathers.

The class functions answer for any architecture string, so these cases run
on every machine, CPU builds included.
"""

import pytest
from mlx_kquant._ext import _moe_c32_nx, _moe_c32_on, _qmv_c32_on

WIDE = [f"applegpu_g{g}{c}" for g in (17, 18, 19) for c in "gsdc"]
NARROW = [f"applegpu_g{g}{c}" for g in (13, 14, 15, 16) for c in "gsdc"] + [
    "applegpu_g17p",
    "applegpu_g18p",
    "applegpu_g12s",
    "applegpu_g17",
    "applegpu_g17S",
    "applegpu_g1xs",
    "applegpu_g170s",
    "gpu_g17s",
    "",
]
GRID = ["iq2_xxs", "iq2_xs", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m"]
OTHER = ["q4_k", "q6_k", "q8_0", "iq4_xs", "iq4_nl", "mxfp4", "q2_0"]


@pytest.mark.parametrize("arch", WIDE)
def test_m5_and_later_take_nx16(arch):
    assert _moe_c32_nx(arch) == 16


@pytest.mark.parametrize("arch", NARROW)
def test_earlier_phones_and_malformed_keep_nx8(arch):
    assert _moe_c32_nx(arch) == 8


@pytest.mark.parametrize("arch", WIDE)
def test_m5_and_later_take_c32_for_every_grid_codec(arch):
    for codec in ["iq2_s"] + GRID:
        assert _moe_c32_on(arch, codec), codec
        assert _qmv_c32_on(arch, codec), codec


@pytest.mark.parametrize("arch", NARROW)
def test_earlier_keep_c32_for_iq2_s_gathers_only(arch):
    assert _moe_c32_on(arch, "iq2_s")
    for codec in GRID:
        assert not _moe_c32_on(arch, codec), codec
    for codec in ["iq2_s"] + GRID:
        assert not _qmv_c32_on(arch, codec), codec


@pytest.mark.parametrize("arch", WIDE + NARROW)
def test_other_codecs_never_take_c32(arch):
    for codec in OTHER:
        assert not _moe_c32_on(arch, codec), codec
        assert not _qmv_c32_on(arch, codec), codec
