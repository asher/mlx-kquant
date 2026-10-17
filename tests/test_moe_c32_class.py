"""The K-lanes per row of the iq2_s c32 MoE gathers by GPU class.

``_moe_c32_nx`` answers for any architecture string, so these cases run on
every machine, CPU builds included.
"""

import pytest
from mlx_kquant._ext import _moe_c32_nx

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


@pytest.mark.parametrize("arch", WIDE)
def test_m5_and_later_take_nx16(arch):
    assert _moe_c32_nx(arch) == 16


@pytest.mark.parametrize("arch", NARROW)
def test_earlier_phones_and_malformed_keep_nx8(arch):
    assert _moe_c32_nx(arch) == 8
