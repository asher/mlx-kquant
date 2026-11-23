"""moe_store_warm builds every pipeline the store ops reach.

In a fresh process: warm the kernels of one codec pair, then run the store
ops (the test_moe_store snippet) and the wide forms (the test_moe_store_seg
cases, the pair op too for a paired codec). No store kernel may be built
after the warm; kernel_keys lists what the process built. A store kernel
renamed or added without its line in moe_store_warm fails here. A second
case warms the down kernels alone (a store of the down stacks only) and
checks no GLU store kernel was built.
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="expert store ops are Metal-only kernels; no CPU path.",
)

HERE = os.path.dirname(os.path.abspath(__file__))

_DRIVER = r"""
import json
import sys
sys.path.insert(0, sys.argv[4])
import mlx.core as mx
import mlx_kquant as kq
import test_moe_store
import test_moe_store_seg

codec, shexp, out_path, here = sys.argv[1:5]
STORE = ("kq_store_", "kq_moe_store_plan", "kq_moe_store_combine",
         "kq_moe_tile_plan", "kq_moe_tile_combine")
res = {"supported": bool(kq.moe_store_supported(codec, shexp))}
if res["supported"]:
    n = kq.moe_store_warm(codec, shexp, codec, shexp,
                          [mx.bfloat16, mx.float16],
                          k_glu=test_moe_store_seg.K,
                          k_down=test_moe_store_seg.K)
    warmed = list(kq.kernel_keys())
    res["fetched"] = n
    res["warmed"] = [k for k in warmed if k.startswith(STORE)]
    # A second warm fetches the same pipelines and builds none.
    res["again"] = kq.moe_store_warm(codec, shexp, codec, shexp,
                                     [mx.bfloat16, mx.float16],
                                     k_glu=test_moe_store_seg.K,
                                     k_down=test_moe_store_seg.K)
    res["again_built"] = len(kq.kernel_keys()) - len(warmed)
    ns = {"__name__": "snippet"}
    argv = sys.argv
    sys.argv = ["-c", codec, shexp, out_path + ".ops", here]
    exec(compile(test_moe_store._SNIPPET, "snippet", "exec"), ns)
    sys.argv = argv
    res["seg"] = bool(kq.moe_store_seg_supported(codec, test_moe_store_seg.K))
    if res["seg"]:
        test_moe_store_seg.test_store_seg_equals_gather_qmm_seg(codec)
        if codec in test_moe_store_seg.PAIRED:
            test_moe_store_seg.test_store_seg_pair_equals_two_calls_and_swiglu(codec)
    res["late"] = [k for k in kq.kernel_keys()[len(warmed):]
                   if k.startswith(STORE)]
json.dump(res, open(out_path, "w"))
"""


@pytest.mark.parametrize(
    "codec,shexp",
    (
        ("iq3_xxs", "q8_0"),
        ("q5_0", "q8_0"),
        ("iq4_xs", "iq4_xs"),
        ("iq4_nl", "q6_k"),
        ("iq2_s", "q8_0"),
        ("q2_0", "q2_0"),
    ),
)
def test_warm_builds_every_store_pipeline(tmp_path, codec, shexp):
    out = tmp_path / f"{codec}-{shexp}.json"
    # The warm builds the kernels of the levers as they stand: the store
    # snippet runs its default arm only (widths 1 to 16: the pair form, the
    # pair grid, then the plan and tile kernels).
    env = dict(
        os.environ,
        KQ_MOE_GROUP="1",
        KQ_MOE_TILE="1",
        KQ_MOE_TILE_MIN_T="2",
        KQ_MOE_SHEXP_DENSE="0",
        KQ_TEST_STORE_ARMS="default",
    )
    env.pop("KQ_MOE_MIX_FORM", None)
    for v in list(env):
        if v.startswith("KQ_MOE_STORE_"):
            env.pop(v)
    r = subprocess.run(
        [sys.executable, "-c", _DRIVER, codec, shexp, str(out), HERE],
        env=env,
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    res = json.loads(out.read_text())
    if not res["supported"]:
        pytest.skip("no tensor-op units on this GPU")
    assert res["fetched"] > 0 and res["warmed"]
    assert res["again"] == res["fetched"] and res["again_built"] == 0
    assert res["late"] == [], res["late"]


_DOWN_ONLY = r"""
import json
import sys
import mlx.core as mx
import mlx_kquant as kq

codec, shexp, out_path = sys.argv[1:4]
res = {"supported": bool(kq.moe_store_supported(codec, shexp))}
if res["supported"]:
    res["fetched"] = kq.moe_store_warm("", "", codec, shexp, [mx.bfloat16],
                                       k_glu=0, k_down=256)
    res["keys"] = list(kq.kernel_keys())
json.dump(res, open(out_path, "w"))
"""


def test_warm_down_only_builds_no_glu_kernel(tmp_path):
    out = tmp_path / "down-only.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_MOE_STORE_")}
    r = subprocess.run(
        [sys.executable, "-c", _DOWN_ONLY, "q5_0", "q8_0", str(out)],
        env=env,
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    res = json.loads(out.read_text())
    if not res["supported"]:
        pytest.skip("no tensor-op units on this GPU")
    keys = res["keys"]
    assert res["fetched"] > 0
    assert any(k.startswith("kq_store_down_pg_q5_0_sx_q8_0") for k in keys)
    assert "kq_moe_store_combine_bfloat16_t" in keys
    assert not [k for k in keys if k.startswith(("kq_store_glu", "kq_moe_store_plan"))]
