"""Grouped fused-MoE gathers (KQ_MOE_GROUP) at the verify widths.

The grouped kernels read each routed expert once for up to eight rows that
route to it and dot its weights with four of them a pass. By default, on GPUs
with the tensor-op units, the GLU and the score-mixed down run grouped from 5
rows, up to 16 rows; KQ_MOE_GROUP=1 turns them on on any GPU, the grouped
arm here.
KQ_MOE_GROUP_MIN_T and KQ_MOE_GROUP_MIX_MIN_T move the two floors and
KQ_MOE_GROUP=0 keeps the per-row, pair and qf kernels, the base arm here.
Every arm runs with KQ_MOE_TILE=0: from 8 rows the tiled down
(test_moe_tile.py) takes the score-mixed calls ahead of the grouped one.
Each arm runs in a subprocess so the levers are read from the first
dispatch.

Every codec of the fused gathers runs both arms with each shared-expert
codec it pairs with, with no shared expert, with a split shared expert and
with a shared expert several slots wide, in float16 and bfloat16. A codec
on the qf kernels keeps its bits under grouping. The others match within
one output step.
"""

import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="fused MoE gathers are Metal-only kernels; no CPU path.",
)
NAX = (
    not os.environ.get("KQUANT_FORCE_CPU")
    and mx.metal.is_available()
    and kq.nax_available()
)

HERE = os.path.dirname(os.path.abspath(__file__))
# Every expert codec of the fused gathers (kq_moe_glu_kq.metal and its
# output-scale build).
CODECS = (
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
)
# The codecs the qf kernels serve at these shapes.
QF = (
    "q2_k",
    "q3_k",
    "q4_k",
    "iq4_xs",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "iq4_nl",
    "q2_0",
    "mxfp4",
    "nvfp4",
)
UPCASTS = ("q5_k", "q6_k", "q8_0")
TYPES = {"float16": "float16_t", "bfloat16": "bfloat16_t"}
# One output step at the largest value.
STEP = {"float16": 2e-3, "bfloat16": 2.0**-7}

_SNIPPET = r"""
import json
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
from kqref import is_synth, synth_wire
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

codec, cases, out_path = sys.argv[2], json.loads(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(31)
E, N, K, R = 12, 256, 512, 5

def wire(lead, n, k, c):
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

gw, uw, dw = wire((E,), N, K, codec), wire((E,), N, K, codec), wire((E,), K, N, codec)
shared = [
    (c, wire((), c["k"] * N, K, c["sg"]), wire((), c["k"] * N, K, c["su"]),
     wire((), K, c["k"] * N, c["sd"]))
    for c in cases
]
res = {}
for T, picks in ((8, "topk"), (13, "few")):
    if picks == "topk":
        # a router's picks: distinct within a row, some shared between rows
        # and some routed by one row only
        inds = np.stack([rng.permutation(E)[:R] for _ in range(T)])
    else:
        # three experts for every pair: an expert repeats within a row and
        # its pairs fill more than one group
        inds = rng.integers(0, 3, size=(T, R))
    inds = mx.array(inds.astype(np.uint32))
    for dname in ("float16", "bfloat16"):
        dt = getattr(mx, dname)
        def act(shape):
            a = (rng.standard_normal(shape) * 0.02).astype(np.float32)
            return mx.array(a).astype(dt)
        x, h = act((T, K)), act((T, R, N))
        sc = mx.array(rng.uniform(0.05, 0.9, size=(T, R)).astype(np.float32))
        tag = f"{T}/{dname}"
        res[f"{tag}/glu"] = kq.moe_glu_gather_kq(x, gw, uw, codec, inds, act="silu")
        res[f"{tag}/mix_ns"] = kq.gather_qmv_mix_ns_kq(h, dw, codec, inds, sc)
        for i, (c, sgw, suw, sdw) in enumerate(shared):
            k = c["k"]
            hs = act((T, R + k, N))
            scs = mx.array(
                rng.uniform(0.05, 0.9, size=(T, R + k)).astype(np.float32))
            res[f"{tag}/shexp{i}"] = kq.moe_glu_gather_shexp_kq(
                x, gw, uw, sgw, suw, codec, inds, act="silu",
                shexp_kquant_type=c["sg"], shexp_up_kquant_type=c["su"])
            res[f"{tag}/mix{i}"] = kq.gather_qmv_mix_kq(
                hs, dw, sdw, codec, inds, scs, shexp_kquant_type=c["sd"])
            res[f"{tag}/unit{i}"] = kq.gather_qmv_mix_kq(
                hs, dw, sdw, codec, inds, scs[:, :R], shexp_kquant_type=c["sd"])
mx.eval(res)
np.savez(out_path, **{k: np.array(v.astype(mx.float32)) for k, v in res.items()})
"""


def _one(s):
    return {"sg": s, "su": s, "sd": s, "k": 1}


def _run(codec, cases, arm, tmp_path, env):
    out = tmp_path / f"grp{arm}.npz"
    r = subprocess.run(
        [sys.executable, "-c", _SNIPPET, HERE, codec, json.dumps(cases), str(out)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "KQ_MOE_GROUP": arm,
            "KQ_MOE_TILE": "0",
            "KQ_MOE_NX_LOG": "1",
            **env,
        },
    )
    assert r.returncode == 0, (
        f"{codec} KQ_MOE_GROUP={arm} exited {r.returncode}\n"
        f"--- child stdout ---\n{r.stdout}\n--- child stderr ---\n{r.stderr}"
    )
    names = {ln.split()[1] for ln in r.stderr.splitlines() if ln[:9] == "[kq_moe] "}
    return np.load(out), names


def _both(codec, cases, tmp_path, env, exact):
    """Both arms of one codec. exact says the grouped kernels keep the bits
    of the base arm."""
    ref, ref_names = _run(codec, cases, "0", tmp_path, env)
    got, names = _run(codec, cases, "1", tmp_path, env)
    assert not [n for n in ref_names if "_grp" in n], sorted(ref_names)
    for key in ref.files:
        a, b = ref[key], got[key]
        assert np.isfinite(b).all() and np.abs(a).max() > 0, (codec, key)
        if exact:
            assert np.array_equal(a, b), (codec, key, np.abs(a - b).max())
        else:
            rel = np.abs(a - b).max() / np.abs(a).max()
            assert rel <= STEP[key.split("/")[1]], (codec, key, rel)
    return names


def _stem(codec, sx):
    if codec in QF:
        return f"kq_{codec}_qf_sx_{sx}"
    if sx == codec and codec in ("q6_k", "q8_0"):
        return f"kq_{codec}_ext_sx_{codec}_ext"
    return f"kq_{codec}_sx_{sx}"


@pytest.mark.parametrize("codec", CODECS)
def test_group_every_codec(codec, tmp_path):
    """Each expert codec at 8 and 13 rows, with picks that share some experts
    and not others: both GLUs, both score-mixed downs with and without a
    score for the shared slot, and each shared codec the codec pairs with.
    The dispatch log names the grouped kernel of each."""
    shared = list(dict.fromkeys((codec, *UPCASTS)))
    names = _both(
        codec,
        [_one(s) for s in shared],
        tmp_path,
        {"KQ_MOE_SHEXP_DENSE": "0"},
        codec in QF,
    )
    sum_stem = "kq_moe_mix_grp_sum_os_" if codec == "nvfp4" else "kq_moe_mix_grp_sum_"
    for t in TYPES.values():
        for sx in shared:
            stem = _stem(codec, sx)
            for want in (
                f"{stem}_moe_glu_gather_shexp_grp_silu",
                f"{stem}_gather_qmv_mix_grp",
            ):
                hit = [n for n in names if n.startswith(want) and n.endswith(t)]
                assert hit, (want, t, sorted(names))
        assert sum_stem + t in names, sorted(names)
    mine = {n for n in names if n.startswith(f"kq_{codec}_")}
    assert all("_grp" in n for n in mine), sorted(mine)


@pytest.mark.parametrize(
    "codec,sg,su,sd",
    [
        ("iq2_s", "iq4_xs", "q6_k", "q8_0"),
        ("iq2_s", "iq4_xs", "iq4_xs", "iq2_s"),
        ("iq2_xxs", "iq3_s", "iq4_xs", "q6_k"),
        ("iq1_m", "q6_k", "iq3_s", "q5_k"),
    ],
)
@pytest.mark.parametrize("k", [1, 2])
def test_group_split_shared(codec, sg, su, sd, k, tmp_path):
    """A shared expert whose gate and up each have their own codec, one and
    two slots wide."""
    names = _both(
        codec,
        [{"sg": sg, "su": su, "sd": sd, "k": k}],
        tmp_path,
        {"KQ_MOE_SHEXP_DENSE": "0"},
        False,
    )
    want = f"kq_{codec}_sx2_{sg}_{su}_moe_glu_gather_shexp_grp_silu"
    assert [n for n in names if n.startswith(want)], (want, sorted(names))
    want = f"kq_{codec}_sx_{sd}_gather_qmv_mix_grp"
    assert [n for n in names if n.startswith(want)], (want, sorted(names))


@pytest.mark.parametrize(
    "codec,sx,sd,k",
    [
        ("q4_1", "q4_1", "q4_1", 2),
        ("q4_1", "q4_1", "q4_1", 4),
        ("q4_1", "q6_k", "q6_k", 2),
        ("q4_k", "q4_k", "q6_k", 2),
        ("q6_k", "q6_k", "q6_k", 2),
        ("q8_0", "q8_0", "q8_0", 2),
        ("iq2_xs", "q6_k", "q6_k", 2),
        ("q2_0", "q2_0", "iq4_nl", 1),
        ("q2_0", "q2_0", "iq4_nl", 2),
        ("nvfp4", "q8_0", "q8_0", 2),
    ],
)
@pytest.mark.parametrize("dense", ["0", "4"])
def test_group_shared_slots(codec, sx, sd, k, dense, tmp_path):
    """A shared expert k routed experts wide, and the q2_0 stack with an
    iq4_nl shared down, with the shared expert in the kernels and on the
    dense route."""
    names = _both(
        codec,
        [{"sg": sx, "su": sx, "sd": sd, "k": k}],
        tmp_path,
        {"KQ_MOE_SHEXP_DENSE": dense},
        False,
    )
    for want in ("_moe_glu_gather_shexp_grp_silu", "_gather_qmv_mix_grp"):
        hit = [n for n in names if n.startswith(f"kq_{codec}_") and want in n]
        assert hit, (want, sorted(names))
    assert any("shexp_glu_pre" in n for n in names) == (dense == "4"), sorted(names)


_ACT_SNIPPET = r"""
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq

codec, scodec, T, out_path, act = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
act_kw = {"silu_limit": {"limit": 0.1}, "swiglu_oai": {"limit": 0.1, "alpha": 1.702}}
rng = np.random.default_rng(31)
E, N, K, S = 16, 256, 512, 6
def wire(e, n, k, c):
    w = mx.array((rng.standard_normal((e, n, k)) * 0.1).astype(np.float32))
    im = mx.ones((k,), dtype=mx.float32) if c.startswith("iq") else None
    wq, _ = kq.quantize(w, c, im)
    mx.eval(wq)
    return wq
gw, uw = wire(E, N, K, codec), wire(E, N, K, codec)
dw = wire(E, K, N, codec)
sgw, suw = wire(1, N, K, scodec)[0], wire(1, N, K, scodec)[0]
sdw = wire(1, K, N, scodec)[0]
inds = np.stack([rng.permutation(E)[:S] for _ in range(T)])
inds = mx.array(inds.astype(np.uint32))
x = mx.array((rng.standard_normal((T, K)) * 0.05).astype(np.float16))
hs = mx.array((rng.standard_normal((T, S + 1, N)) * 0.05).astype(np.float16))
scs = mx.array(rng.uniform(0.05, 0.9, size=(T, S + 1)).astype(np.float32))
outs = {
    "mix": kq.gather_qmv_mix_kq(
        hs, dw, sdw, codec, inds, scs, shexp_kquant_type=scodec),
    "glu": kq.moe_glu_gather_shexp_kq(
        x, gw, uw, sgw, suw, codec, inds, act=act, shexp_kquant_type=scodec,
        **act_kw.get(act, {})),
    "glu_ns": kq.moe_glu_gather_kq(
        x, gw, uw, codec, inds, act=act, **act_kw.get(act, {})),
}
mx.eval(*outs.values())
np.savez(out_path, **{k: np.array(v.astype(mx.float32)) for k, v in outs.items()})
"""


def _arm(codec, scodec, T, arm, tmp_path, env=None, act="silu"):
    """One arm in a subprocess; arm None leaves KQ_MOE_GROUP unset."""
    out = tmp_path / f"act{arm}.npz"
    cenv = {**os.environ, "KQ_MOE_TILE": "0", "KQ_MOE_NX_LOG": "1", **(env or {})}
    cenv.pop("KQ_MOE_GROUP", None)
    if arm is not None:
        cenv["KQ_MOE_GROUP"] = arm
    r = subprocess.run(
        [sys.executable, "-c", _ACT_SNIPPET, codec, scodec, str(T), str(out), act],
        capture_output=True,
        text=True,
        env=cenv,
    )
    assert r.returncode == 0, (
        f"{codec}/{scodec} T={T} KQ_MOE_GROUP={arm} exited {r.returncode}\n"
        f"--- child stdout ---\n{r.stdout}\n--- child stderr ---\n{r.stderr}"
    )
    return np.load(out), r.stderr


@pytest.mark.parametrize("codec,scodec", [("iq3_xxs", "q6_k"), ("q5_0", "q8_0")])
@pytest.mark.parametrize("act", ["gelu", "silu_limit", "swiglu_oai"])
def test_group_acts_match_per_row(codec, scodec, act, tmp_path):
    """The grouped GLUs apply each activation as the per-row kernels do, on
    the Ext loops and on the qf loops."""
    ref, _ = _arm(codec, scodec, 8, "0", tmp_path, act=act)
    got, log = _arm(codec, scodec, 8, "1", tmp_path, act=act)
    for key in ("glu", "glu_ns"):
        rel = np.abs(got[key] - ref[key]).max() / np.abs(ref[key]).max()
        assert rel < 2e-3, (act, key, rel)
    want = "grp_gelu" if act == "gelu" else "grp_silu"
    assert f"_moe_glu_gather_shexp_{want}_" in log, log


@pytest.mark.parametrize("T", [2, 3, 4, 5, 16])
def test_group_low_floor(T, tmp_path):
    """With both floors at 2 rows the grouped kernels serve every width from
    2 to 16 rows."""
    env = {"KQ_MOE_GROUP_MIN_T": "2", "KQ_MOE_GROUP_MIX_MIN_T": "2"}
    for codec, scodec in (("iq3_xxs", "q6_k"), ("iq4_xs", "q6_k")):
        ref, _ = _arm(codec, scodec, T, "0", tmp_path)
        got, log = _arm(codec, scodec, T, "1", tmp_path, env)
        for key in ref.files:
            rel = np.abs(got[key] - ref[key]).max() / np.abs(ref[key]).max()
            assert rel < 2e-3, (codec, key, rel)
        assert "_moe_glu_gather_shexp_grp_silu" in log, log
        assert "_gather_qmv_mix_grp" in log, log


@pytest.mark.parametrize(
    "T,grouped,dense",
    [
        (1, False, False),
        (4, False, True),
        (5, True, True),
        (8, True, False),
        (16, True, False),
        (17, False, True),
    ],
)
@pytest.mark.parametrize("arm", ["1", None])
def test_group_default_floors(T, grouped, dense, arm, tmp_path):
    """At the default floors both gathers run grouped from 5 to 16 rows, and
    from 7 rows the grouped kernels take the shared expert themselves. With
    KQ_MOE_GROUP unset that holds on GPUs with the tensor-op units; other
    GPUs keep the pair kernels and the dense route from 4 rows."""
    _, log = _arm("iq3_xxs", "q6_k", T, arm, tmp_path)
    if arm is None and not NAX:
        grouped, dense = False, T >= 4
    assert ("_moe_glu_gather_shexp_grp_" in log) == grouped, log
    assert ("_gather_qmv_mix_grp" in log) == grouped, log
    assert ("kq_moe_shexp_glu_pre" in log) == dense, log


@pytest.mark.parametrize("codec,scodec", [("iq3_xxs", "q6_k"), ("q5_0", "q8_0")])
@pytest.mark.parametrize("T", [5, 13])
def test_group_mix_tiles_keep_bits(codec, scodec, T, tmp_path):
    """The output tiles a grouped score-mixed down threadgroup takes
    (KQ_MOE_GROUP_MIX_NT) leave the bits as they are, on the Ext loops and
    on the qf loops, with a last threadgroup short of tiles."""
    outs = []
    for nt in ("1", "3", "8"):
        got, log = _arm(codec, scodec, T, "1", tmp_path, {"KQ_MOE_GROUP_MIX_NT": nt})
        assert "_gather_qmv_mix_grp" in log, log
        outs.append(got["mix"])
    for o in outs[1:]:
        assert np.array_equal(outs[0], o)
