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
    the pair GLU at 2 on its codecs and the grouped kernels at 5 and 12,
    with experts shared across rows), and the loop, slot-parallel and
    grouped score-mixed downs give the same bits. KQ_MOE_TILE=0 keeps
    the tiled down off: from 8 rows it is the default, and its bits depend
    on the call width (test_moe_tile.py). KQ_MOE_SP=1 runs the slot-parallel
    mix_ns, which q4_0, q4_1 and nvfp4 skip by default at this K."""
    sx = "q5_k" if codec in ("q2_k", "q3_k", "q4_k", "iq4_xs", "q2_0") else "q8_0"
    grp = {"KQ_MOE_GROUP": "1"}
    sp, names_sp = _rows(
        codec, sx, dtype, {"KQ_MOE_TILE": "0", "KQ_MOE_SP": "1", **grp}
    )
    loop, names_loop = _rows(codec, sx, dtype, {"KQ_MOE_MIX_FORM": "loop", **grp})
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
    # the 5 and 12 row calls run the grouped kernels
    for names in (names_sp, names_loop):
        assert f"{qf}sx_{codec}_moe_glu_gather_shexp_grp_silu_{t}" in names, sorted(
            names
        )
        assert f"{qf}sx_{sx}_moe_glu_gather_shexp_grp_silu_{t}" in names, sorted(names)
    assert f"{qf}sx_{codec}_gather_qmv_mix_grp_{t}" in names_sp, sorted(names_sp)
    assert f"{qf}sx_{sx}_gather_qmv_mix_grp_{t}" in names_sp, sorted(names_sp)
    assert not [n for n in names_loop if "_mix_grp" in n], sorted(names_loop)


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


_MIX_NS_SNIPPET = r"""
import json
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq

codec, K = sys.argv[2], int(sys.argv[3])
E, N, T, R = 16, 512, 4, 8
rng = np.random.default_rng(5)
w = mx.array((rng.standard_normal((E, N, K)) * 0.1).astype(np.float32))
wq, _ = kq.quantize(w, codec)
x = rng.standard_normal((T, R, K)) * 0.5
x = mx.array(x.astype(np.float32)).astype(mx.float16)
inds = mx.array(np.stack([rng.choice(E, R, False) for _ in range(T)]).astype(np.uint32))
sc = mx.array(rng.uniform(0.05, 0.3, size=(T, R)).astype(np.float32))
y = kq.gather_qmv_mix_ns_kq(x, wq, codec, inds, sc)
mx.eval(y)
print("RESULT " + json.dumps(np.array(y.astype(mx.float32)).ravel().tolist()))
"""


@pytest.mark.parametrize(
    "codec, K, sp",
    [
        ("q4_0", 1152, False),
        ("q4_1", 1152, False),
        ("q4_0", 2048, True),
        ("q2_0", 1152, True),
        ("q4_k", 1280, True),
    ],
)
def test_mix_ns_short_rows_take_the_loop(codec, K, sp):
    """q4_0, q4_1 and nvfp4 run the loop mix_ns below K 2048, where it
    beats the slot-parallel kernel. Other codecs and longer rows keep
    slot-parallel, and KQ_MOE_SP=1 forces it. Both give the same bits."""
    t = "float16_t"
    stem = f"kq_{codec}_qf_gather_qmv_mix_ns_"
    y, names = _child(_MIX_NS_SNIPPET, [codec, str(K)], {})
    y_sp, names_sp = _child(_MIX_NS_SNIPPET, [codec, str(K)], {"KQ_MOE_SP": "1"})
    assert (stem + ("sp_" if sp else "") + t) in names, sorted(names)
    assert stem + "sp_" + t in names_sp, sorted(names_sp)
    assert np.array_equal(np.array(y), np.array(y_sp)), codec


_ABSENT_SNIPPET = r"""
import json
import os
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
from kqref import is_synth, synth_wire
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

# "gate+up" names a split pair; the down takes the gate's codec.
codec, _, up = sys.argv[2].partition("+")
rng = np.random.default_rng(29)
E, N, K, R = 8, 256, 512, 4

def wire(lead, n, k, c=codec):
    bpb, wpb = CODEC_GEOMETRY[c][2:]
    if is_synth(c) or c == "mxfp4":
        blocks = int(np.prod(lead, dtype=np.int64)) * n * (k // wpb)
        w = synth_wire(rng, c, bpb, blocks, d_range=(0.004, 0.01),
                       nvfp4_scales=(0x08, 0x19))
        return mx.array(w.reshape(*lead, n, (k // wpb) * bpb))
    w = mx.array((rng.standard_normal((*lead, n, k)) * 0.1).astype(np.float32))
    wq, _ = kq.quantize(w, c)
    mx.eval(wq)
    return wq

gw, uw, dw = wire((E,), N, K), wire((E,), N, K, up or codec), wire((E,), K, N)
args = dict(up_kquant_type=up) if up else dict(act="situ", limit=4.0, alpha=25.0)
bad = []
for T, keep in ((1, 0), (4, 1), (6, 2), (8, 0), (12, 3)):
    x = mx.array((rng.standard_normal((T, K)) * 0.1).astype(np.float16))
    h = mx.array((rng.standard_normal((T, R, N)) * 0.1).astype(np.float16))
    ids = np.stack([rng.choice(E, R, False) for _ in range(T)]).astype(np.uint32)
    sc = rng.uniform(0.05, 0.9, size=(T, R)).astype(np.float32)
    gone = np.zeros((T, R), bool)
    gone[keep + 1:] = True
    gone[0, 1] = True
    marked = mx.array(np.where(gone, np.uint32(0x80000000), ids))
    live_sc = mx.array(np.where(gone, 0, sc))
    live = ~gone.all(axis=1)
    for arm in ({}, {"KQ_MOE_NX": "8"}, {"KQ_MOE_MIX_FORM": "loop"},
                {"KQ_MOE_MIX_FORM": "sp"}, {"KQ_MOE_MIX_FORM": "dd"}):
        os.environ.update({"KQ_MOE_NX": "", "KQ_MOE_MIX_FORM": "", **arm})
        ref_h = kq.moe_glu_gather_kq(x, gw, uw, codec, mx.array(ids), **args)
        got_h = kq.moe_glu_gather_kq(x, gw, uw, codec, marked, **args, absent=True)
        ref_y = kq.gather_qmv_mix_ns_kq(h, dw, codec, mx.array(ids), live_sc)
        got_y = kq.gather_qmv_mix_ns_kq(h, dw, codec, marked, mx.array(sc), absent=True)
        mx.eval(ref_h, got_h, ref_y, got_y)
        rh, gh = np.array(ref_h.view(mx.uint16)), np.array(got_h.view(mx.uint16))
        ry, gy = np.array(ref_y.view(mx.uint16)), np.array(got_y.view(mx.uint16))
        if not (np.array_equal(gh[~gone], rh[~gone]) and not gh[gone].any()):
            bad.append([T, arm, "glu"])
        if not (np.array_equal(gy[live], ry[live]) and not gy[~live].any()):
            bad.append([T, arm, "mix"])
print("RESULT " + json.dumps(bad))
"""


# Every expert codec of the fused gathers.
ALL_CODECS = (
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "q8_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "iq4_nl",
    "iq4_xs",
    "iq3_s",
    "iq3_xxs",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq1_s",
    "iq1_m",
    "stq1_0",
    "pq2_0",
    "q2_0",
    "ptq1_0",
    "mxfp4",
    "nvfp4",
    "iq2_xxs+stq1_0",
    "stq1_0+iq2_xxs",
)


@pytest.mark.parametrize("codec", ALL_CODECS)
def test_absent_ids_read_no_expert(codec):
    """With absent=True an id with bit 31 set names no expert: the GLU gives
    zeros for its pair and the mix adds nothing for its slot, and every
    other output keeps the bits of the call without the marks, at widths 1
    to 12 and in each mix form. The marked call skips the grouped and tile
    forms, so the reference does too."""
    env = {
        "KQ_MOE_TILE": "0",
        "KQ_MOE_GROUP": "0",
        "KQ_MOE_NX": "",
        "KQ_MOE_MIX_FORM": "",
    }
    bad, names = _child(_ABSENT_SNIPPET, [codec], env)
    assert not bad, (codec, bad)
    if codec not in QF_CODECS and "stq1_0" not in codec:
        stem = codec + "_ext" if codec in ("q6_k", "q8_0") else codec
        assert f"kq_{stem}_moe_glu_gather_dd_silu_float16_t" in names, sorted(names)
        assert f"kq_{codec}_gather_qmv_mix_ns_dd_float16_t" in names, sorted(names)


def test_absent_ids_take_no_biases():
    """The biased GLU would read an absent id's bias row."""
    import mlx.core as mx

    import mlx_kquant as kq

    x = mx.zeros((2, 512), mx.float16)
    w = mx.zeros((4, 256, 512 // 32 * 17), mx.uint8)
    b = mx.zeros((4, 256), mx.float16)
    ids = mx.zeros((2, 2), mx.uint32)
    with pytest.raises(ValueError, match="absent=True takes no biases"):
        kq.moe_glu_gather_kq(
            x,
            w,
            w,
            "mxfp4",
            ids,
            act="swiglu_clamp",
            limit=7.0,
            gate_bias=b,
            up_bias=b,
            alpha=1.702,
            absent=True,
        )


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


_PLAIN = "moe_glu_gather_silu"
_PAIR = "moe_glu_gather_dd_silu"
_GROUPED = "sx_q5_0_moe_glu_gather_shexp_grp_silu"


@pytest.mark.parametrize(
    "T, env, want",
    [
        (1, {}, _PLAIN),
        (2, {}, _PAIR),
        (4, {}, _PAIR),
        (5, {"KQ_MOE_GROUP": "1"}, _GROUPED),
        (16, {"KQ_MOE_GROUP": "1"}, _GROUPED),
        (17, {}, _PLAIN),
        (8, {"KQ_MOE_GROUP": "0"}, _PAIR),
        (9, {"KQ_MOE_GROUP": "0"}, _PLAIN),
        (4, {"KQ_MOE_DEDUP": "0"}, _PLAIN),
    ],
    ids=["t1", "t2", "t4", "t5", "t16", "t17", "group_off", "t9", "dedup_off"],
)
def test_qf_pair_widths(T, env, want):
    """The pair GLU serves 2 to 4 rows and the grouped GLU 5 to 16 (by
    default on GPUs with the tensor-op units; KQ_MOE_GROUP=1 here).
    KQ_MOE_GROUP=0 gives 5 to 8 rows back to the pair GLU and KQ_MOE_DEDUP=0
    turns that off. Codecs with neither keep the plain qf GLU at every
    width."""
    _, names = _child(_WIDTH_SNIPPET, [str(T)], env)
    glu = {n for n in names if n.startswith("kq_q5_0_qf_")}
    assert glu == {f"kq_q5_0_qf_{want}_float16_t"}, (T, env, glu)


@pytest.mark.parametrize("k", [1536, 1280])
def test_q3_k_rows_whose_superblocks_four_lanes_do_not_split_evenly(k):
    """A Q3_K row of 6 or 5 superblocks: the lanes of a simdgroup share its
    (row, superblock) pairs, so a lane covers parts of both its rows. The
    matvec and the fused GLU gather match the f32 reference there."""
    import mlx.core as mx

    import mlx_kquant as kq

    sys.path.insert(0, HERE)
    from kqref import GT, quants

    rng = np.random.default_rng(5)
    E, N, T, R = 4, 64, 2, 2
    wires, refs = [], []
    for _ in range(2 * E):
        w, scales = kq.quantize(
            mx.array(rng.standard_normal((N, k)).astype(np.float32) * 0.05),
            "q3_k",
            None,
        )
        wires.append((w, scales))
        refs.append(quants.dequantize(np.array(w), GT.Q3_K).astype(np.float32))
    x = mx.array(rng.standard_normal((T, k)).astype(np.float16) * 0.5)
    xf = np.array(x.astype(mx.float32))

    def rel(got, want):
        return float(np.abs(got - want).max() / np.abs(want).max())

    y = kq.quantized_matmul(x, *wires[0], "q3_k")
    assert rel(np.array(y.astype(mx.float32)), xf @ refs[0].T) < 2e-3

    gate = mx.stack([w for w, _ in wires[:E]])
    up = mx.stack([w for w, _ in wires[E:]])
    idx = np.array([[0, 3], [2, 3]], dtype=np.uint32)
    h = kq.moe_glu_gather_kq(x, gate, up, "q3_k", mx.array(idx), act="silu")
    want = np.empty((T, R, N), np.float32)
    for t in range(T):
        for r in range(R):
            g = xf[t] @ refs[idx[t, r]].T
            u = xf[t] @ refs[E + idx[t, r]].T
            want[t, r] = g / (1 + np.exp(-g)) * u
    assert rel(np.array(h.astype(mx.float32)).reshape(T, R, N), want) < 2e-3
