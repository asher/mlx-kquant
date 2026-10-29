"""The qf fused MoE gathers: the tuned qmv_fast loops of q2_k, q3_k, q4_k,
iq4_xs, q4_0, q4_1, q5_0, q5_1, iq4_nl, mxfp4, nvfp4 and q2_0 under the
fused GLU, down and score-mixed down gathers (kq_moe_qf.h).

Every form is checked against the f32 reference of test_moe_glu for each
qf codec, dtype and activation, with the shared expert in the routed codec
or a q5_k, q6_k or q8_0 upcast. A row gets the same bits at every call
width and in both score-mixed forms (loop and slot-parallel), so a decode
step and a prefill chunk agree. The shared expert keeps its bits only with
KQ_MOE_SHEXP_DENSE=0, which the width test sets: from 4 rows the dense
route computes it (test_moe_shexp_dense). At the MTP verify widths (2 to 8
rows) the GLU of q5_0, q5_1, iq4_nl, iq4_xs, mxfp4, nvfp4 and q2_0 takes its
pair form (two rows on one expert read its weights once), which must give
the same bits as the plain form at every other width. The Ext kernels serve
the call where qf does not apply: KQ_MOE_QF=0, a forced KQ_MOE_NX, a shared
expert outside the upcast set, a q8_0 shared expert at K % 256 != 0, and a
shared expert wider than one routed expert on the score-mixed down.

The levers are read live only once set, so each arm runs in a child.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="fused MoE gathers are Metal-only kernels; no CPU path.",
)

HERE = os.path.dirname(os.path.abspath(__file__))
QF_CODECS = (
    "q2_k",
    "q3_k",
    "q4_k",
    "iq4_xs",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "iq4_nl",
    "mxfp4",
    "nvfp4",
    "q2_0",
)
PAIR_CODECS = ("q5_0", "q5_1", "iq4_nl", "iq4_xs", "mxfp4", "nvfp4", "q2_0")
UPCASTS = ("q5_k", "q6_k", "q8_0")
# bf16 output rounding alone puts the f32-reference error near 2e-3, the
# f16 bound of test_moe_glu, for the Ext and qf kernels alike.
REL_BOUND = {"float16": 2e-3, "bfloat16": 3e-3}


def _child(snippet, args, env):
    r = subprocess.run(
        [sys.executable, "-c", snippet, HERE, *args],
        capture_output=True,
        text=True,
        env={**os.environ, "KQ_MOE_NX_LOG": "1", **env},
    )
    assert r.returncode == 0, f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, r.stdout + r.stderr
    log = [ln for ln in r.stderr.splitlines() if ln.startswith("[kq_moe] ")]
    names = {ln.split()[1] for ln in log}
    return json.loads(line[-1][len("RESULT ") :]), names


_REF_SNIPPET = r"""
import json
import sys
sys.path.insert(0, sys.argv[1])
import mlx.core as mx
import test_moe_glu as tm

codec, dtype = sys.argv[2], getattr(mx, sys.argv[3])
out = {}
for sx, act in [("", "silu"), ("", "gelu"), ("", "silu_limit"),
                ("q5_k", "silu"), ("q6_k", "silu"), ("q8_0", "silu")]:
    limit = 0.3 if act == "silu_limit" else 0.0
    res = tm._check_codec(codec, sx=sx or None, act=act, dtype=dtype, limit=limit)
    out[f"{sx}/{act}"] = (
        None if res is None else [(n, float(r), bool(o)) for n, r, o in res])
print("RESULT " + json.dumps(out))
"""


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("codec", QF_CODECS)
def test_qf_matches_reference(codec, dtype):
    """Each form of each qf codec against the f32 reference, with every
    shared-expert codec qf takes, and the qf kernels serve every call."""
    out, names = _child(_REF_SNIPPET, [codec, dtype], {})
    for combo, res in out.items():
        assert res is not None, (combo, "fixture missing")
        for name, rel, _ in res:
            assert rel < REL_BOUND[dtype], (codec, combo, name, rel)
    mine = {n for n in names if n.startswith(f"kq_{codec}_")}
    assert mine and all("_qf_" in n for n in mine), sorted(mine)
    for sx in UPCASTS:
        assert {n for n in mine if f"_qf_sx_{sx}_" in n}, (sx, sorted(mine))


_ROWS_SNIPPET = r"""
import json
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

codec, sx, dtype = sys.argv[2], sys.argv[3], getattr(mx, sys.argv[4])
rng = np.random.default_rng(17)
E, N, K, R, TMAX = 8, 256, 512, 4, 12
D_OFF = {"q2_k": (80, 82), "q3_k": (108,), "q4_k": (0, 2), "q5_k": (0, 2),
         "q6_k": (208,), "q4_1": (0, 2), "q5_1": (0, 2)}

def wire(c, lead, n, k):
    bpb, wpb = CODEC_GEOMETRY[c][2:]
    w = rng.integers(0, 256, size=(*lead, n, k // wpb, bpb), dtype=np.uint8)
    if c == "nvfp4":
        w[..., :4] = rng.integers(0x20, 0x30, w[..., :4].shape)
    elif c == "mxfp4":
        w[..., 0] = rng.integers(118, 122, w[..., 0].shape)
    else:
        for off in D_OFF.get(c, (0,)):
            d = rng.uniform(0.002, 0.006, w.shape[:-1]).astype(np.float16)
            w[..., off:off + 2] = d.view(np.uint8).reshape(*w.shape[:-1], 2)
    return mx.array(w.reshape(*lead, n, -1))

gw, uw, dw = wire(codec, (E,), N, K), wire(codec, (E,), N, K), wire(codec, (E,), K, N)
sg, su, sd = wire(sx, (), N, K), wire(sx, (), N, K), wire(sx, (), K, N)
def act(shape):
    return mx.array((rng.standard_normal(shape) * 0.1).astype(np.float32)).astype(dtype)
x, h, hs = act((TMAX, K)), act((TMAX, R, N)), act((TMAX, R + 1, N))
inds = mx.array(rng.integers(0, E, size=(TMAX, R)).astype(np.uint32))
sc = mx.array(rng.uniform(0.05, 0.9, size=(TMAX, R)).astype(np.float32))
scs = mx.array(rng.uniform(0.05, 0.9, size=(TMAX, R + 1)).astype(np.float32))
st = "" if sx == codec else sx

def ops(a, b):
    return {
        "glu": kq.moe_glu_gather_kq(x[a:b], gw, uw, codec, inds[a:b], act="silu"),
        "oai": kq.moe_glu_gather_kq(x[a:b], gw, uw, codec, inds[a:b],
                                    act="swiglu_oai", limit=0.4, alpha=1.702),
        "qmv": kq.gather_qmv_kq(h[a:b], dw, codec, inds[a:b]),
        "mix_ns": kq.gather_qmv_mix_ns_kq(h[a:b], dw, codec, inds[a:b], sc[a:b]),
        "shexp": kq.moe_glu_gather_shexp_kq(
            x[a:b], gw, uw, sg, su, codec, inds[a:b], act="silu",
            shexp_kquant_type=st),
        "mix": kq.gather_qmv_mix_kq(
            hs[a:b], dw, sd, codec, inds[a:b], scs[a:b], shexp_kquant_type=st),
        "mix_unit": kq.gather_qmv_mix_kq(
            hs[a:b], dw, sd, codec, inds[a:b], scs[a:b, :R],
            shexp_kquant_type=st),
    }

out = {}
for a, b in [(0, TMAX), (0, 5), (5, 6), (6, 8), (11, 12)]:
    o = ops(a, b)
    mx.eval(o)
    out[f"{a}:{b}"] = {k: np.array(v.view(mx.uint16)).tolist() for k, v in o.items()}
print("RESULT " + json.dumps(out))
"""


def _rows(codec, sx, dtype, env):
    res, names = _child(
        _ROWS_SNIPPET, [codec, sx, dtype], {"KQ_MOE_SHEXP_DENSE": "0", **env}
    )
    t = "bfloat16_t" if dtype == "bfloat16" else "float16_t"
    pair = f"kq_{codec}_qf_moe_glu_gather_dd_silu_{t}"
    assert (pair in names) == (codec in PAIR_CODECS), (codec, sorted(names))
    full = {k: np.array(v) for k, v in res["0:12"].items()}
    for span, part in res.items():
        a, b = map(int, span.split(":"))
        for k, v in part.items():
            assert np.array_equal(np.array(v), full[k][a:b]), (codec, sx, span, k)
    return full, names


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("codec", QF_CODECS)
def test_qf_rows_match_at_every_width(codec, dtype):
    """A row's bits do not depend on the call width (1, 2, 5 and 12 rows:
    the pair GLU at 2 and 5 on its codecs, with experts shared across rows),
    and the loop and slot-parallel score-mixed downs give the same bits."""
    sx = "q5_k" if codec in ("q2_k", "q3_k", "q4_k", "iq4_xs", "q2_0") else "q8_0"
    sp, names_sp = _rows(codec, sx, dtype, {})
    loop, names_loop = _rows(codec, sx, dtype, {"KQ_MOE_MIX_FORM": "loop"})
    for k in sp:
        assert np.array_equal(sp[k], loop[k]), (codec, k)
    t = "bfloat16_t" if dtype == "bfloat16" else "float16_t"
    qf = f"kq_{codec}_qf_"
    assert f"{qf}gather_qmv_mix_ns_sp_{t}" in names_sp, sorted(names_sp)
    assert f"{qf}sx_{sx}_gather_qmv_mix_sp_{t}" in names_sp, sorted(names_sp)
    assert f"{qf}gather_qmv_mix_ns_{t}" in names_loop, sorted(names_loop)
    assert f"{qf}sx_{sx}_gather_qmv_mix_{t}" in names_loop, sorted(names_loop)
    mine = {n for n in names_sp | names_loop if n.startswith(f"kq_{codec}_")}
    assert all("_qf_" in n for n in mine), sorted(mine)


_STAND_SNIPPET = r"""
import json
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

codec, sx, K, NS = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
want = sys.argv[6].split(",")
rng = np.random.default_rng(3)
# N = K: the GLU input and the down input share the width under test
E, N, R, T = 4, K, 2, 2
def wire(c, lead, n, k):
    bpb, wpb = CODEC_GEOMETRY[c][2:]
    w = np.zeros((*lead, n, (k // wpb) * bpb), np.uint8)
    return mx.array(w)
gw, dw = wire(codec, (E,), N, K), wire(codec, (E,), K, N)
sg, sd = wire(sx, (), NS * N, K), wire(sx, (), K, NS * N)
x = mx.zeros((T, K), mx.float16)
hs = mx.zeros((T, R + NS, N), mx.float16)
inds = mx.zeros((T, R), mx.uint32)
st = "" if sx == codec else sx
ops = {
    "glu": lambda: kq.moe_glu_gather_kq(x, gw, gw, codec, inds),
    "shexp": lambda: kq.moe_glu_gather_shexp_kq(
        x, gw, gw, sg, sg, codec, inds, shexp_kquant_type=st),
    "mix": lambda: kq.gather_qmv_mix_kq(
        hs, dw, sd, codec, inds, mx.ones((T, R + NS)), shexp_kquant_type=st),
}
mx.eval(*[ops[k]() for k in want])
print("RESULT 0")
"""


def _stand(codec, sx, k, ns, env, ops="glu,shexp,mix"):
    _, names = _child(_STAND_SNIPPET, [codec, sx, str(k), str(ns), ops], env)
    return names


@pytest.mark.parametrize(
    "env", [{"KQ_MOE_QF": "0"}, {"KQ_MOE_NX": "8"}], ids=["qf_off", "nx_forced"]
)
def test_qf_stands_down_on_levers(env):
    """KQ_MOE_QF=0 and a forced KQ_MOE_NX keep the Ext kernels."""
    on = _stand("q4_k", "q4_k", 512, 1, {})
    assert {n for n in on if "_qf_" in n} and all("_qf_" in n for n in on), on
    off = _stand("q4_k", "q4_k", 512, 1, env)
    assert off and not {n for n in off if "_qf_" in n}, off


def test_qf_stands_down_on_shapes():
    """The shared-expert gathers keep Ext for a shared codec qf does not
    take, a q8_0 shared expert off K % 256, and (the mixed down only) a
    shared expert two routed experts wide; the routed GLU stays qf."""
    t = "float16_t"
    names = _stand("q2_0", "iq4_nl", 512, 1, {}, "glu,mix")
    assert {n for n in names if n.startswith("kq_q2_0_qf_moe_glu_gather_")}, names
    assert f"kq_q2_0_sx_iq4_nl_gather_qmv_mix_sp_{t}" in names, names
    names = _stand("q4_0", "q8_0", 352, 1, {})
    assert {n for n in names if n.startswith("kq_q4_0_qf_moe_glu_gather_")}, names
    assert not {n for n in names if "_sx_q8_0" in n and "_qf_" in n}, names
    names = _stand("q4_0", "q8_0", 512, 1, {})
    assert f"kq_q4_0_qf_sx_q8_0_gather_qmv_mix_sp_{t}" in names, names
    names = _stand("q4_0", "q8_0", 512, 2, {})
    assert f"kq_q4_0_qf_sx_q8_0_moe_glu_gather_shexp_silu_{t}" in names, names
    assert not {n for n in names if "gather_qmv_mix" in n and "_qf_" in n}, names


def test_moe_qf_kernels_query():
    """moe_qf_kernels names the qf codecs and follows KQ_MOE_QF."""
    snippet = r"""
import json
import os
import sys
import mlx_kquant as kq
os.environ["KQ_MOE_QF"] = "1"
on = {c: kq.moe_qf_kernels(c) for c in sys.argv[2:]}
os.environ["KQ_MOE_QF"] = "0"
off = {c: kq.moe_qf_kernels(c) for c in sys.argv[2:]}
print("RESULT " + json.dumps([on, off]))
"""
    others = ("q5_k", "q6_k", "q8_0", "iq2_xs", "iq3_xxs")
    (on, off), _ = _child(snippet, [*QF_CODECS, *others], {})
    assert on == {c: c in QF_CODECS for c in (*QF_CODECS, *others)}, on
    assert not any(off.values()), off


def test_qf_reads_live():
    """KQ_MOE_QF takes effect on the next call."""
    snippet = r"""
import os
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
w = mx.zeros((4, 64, 16 * 144), mx.uint8)
x = mx.zeros((1, 4096), mx.float16)
inds = mx.zeros((1, 2), mx.uint32)
os.environ["KQ_MOE_QF"] = "1"
mx.eval(kq.moe_glu_gather_kq(x, w, w, "q4_k", inds))
os.environ["KQ_MOE_QF"] = "0"
mx.eval(kq.moe_glu_gather_kq(x, w, w, "q4_k", inds))
print("RESULT 0")
"""
    _, names = _child(snippet, [], {})
    assert "kq_q4_k_qf_moe_glu_gather_silu_float16_t" in names, names
    assert {n for n in names if n.startswith("kq_q4_k_moe_glu_gather")}, names


_WIDTH_SNIPPET = r"""
import sys
import mlx.core as mx
import mlx_kquant as kq
T = int(sys.argv[2])
w = mx.zeros((4, 64, 16 * 22), mx.uint8)
x = mx.zeros((T, 512), mx.float16)
inds = mx.zeros((T, 2), mx.uint32)
mx.eval(kq.moe_glu_gather_kq(x, w, w, "q5_0", inds))
print("RESULT 0")
"""


@pytest.mark.parametrize(
    "T, env, want",
    [
        (1, {}, False),
        (2, {}, True),
        (8, {}, True),
        (9, {}, False),
        (4, {"KQ_MOE_DEDUP": "0"}, False),
    ],
    ids=["t1", "t2", "t8", "t9", "dedup_off"],
)
def test_qf_pair_widths(T, env, want):
    """The pair GLU serves 2 to 8 rows, and KQ_MOE_DEDUP=0 turns it off.
    Codecs without it keep the plain qf GLU at every width."""
    _, names = _child(_WIDTH_SNIPPET, [str(T)], env)
    glu = {n for n in names if n.startswith("kq_q5_0_qf_moe_glu_gather_")}
    pick = "dd_silu" if want else "silu"
    assert glu == {f"kq_q5_0_qf_moe_glu_gather_{pick}_float16_t"}, (T, env, glu)
