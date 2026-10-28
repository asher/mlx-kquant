#!/usr/bin/env python3
"""Act swiglu_oai on the fused MoE gathers, by launch site.

swiglu_oai (MiniMax-M3, gpt-oss without biases) is
min(g, limit) * sigmoid(alpha * min(g, limit)) * (clip(u, -limit, limit) + 1).
It runs in the silu kernels: bit 4 of function constant 380 switches their
epilogue and binds alpha. Each launch site that can take it is driven at a
shape and lever setting that selects each kernel family it dispatches:

  glu    moe_glu_gather_kq            ext nx8/16/32, dd, c32, half (_h), the
                                       tuned q6_k/q8_0 kernels, nvfp4 with
                                       output scales
  shexp  moe_glu_gather_shexp_kq      the same families with the shared
                                       expert in the routed codec or an
                                       upcast q5_k/q6_k/q8_0 (sx)
  sx2    moe_glu_gather_shexp_kq      shared gate and up in two codecs
  qmv, mix, mix_ns                    the down gathers, which take no act

swiglu_oai is checked against an f32 reference. silu and silu_limit (the
DeepSeek-V4 form), and the down gathers, are checked by value against
outputs saved from mlx-kquant 0.5.7, the release before swiglu_oai. Each
order of the two forms runs in its own process (silu forms first, then
swiglu_oai first), so a pipeline cache entry shared by the two forms fails
one of them.

The saved outputs are exact on the GPU they were written on and within
rounding elsewhere. Write them with a build from before the change:

  python tests/test_moe_swiglu_oai.py --write-ref tests/fixtures/moe_act_ref.npz
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zlib

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "fixtures", "moe_act_ref.npz")

gpu_only = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="fused MoE gathers are Metal-only kernels; no CPU path.",
)

E, N, K, R = 4, 64, 512, 2
ALPHA = 1.702
REL_BOUND = 2e-3

GLU_CODECS = (
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
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
C32_CODECS = ("iq2_s", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m")
HALF_CODECS = ("iq2_xs", "iq2_xxs", "iq3_xxs")
UPCAST = ("q5_k", "q6_k", "q8_0")
SX2 = (("iq4_xs", "iq3_s"), ("q6_k", "iq4_xs"), ("iq3_s", "q6_k"))
BF16_CODECS = ("iq2_xs", "iq3_xxs", "q4_k", "q6_k", "nvfp4")

# The lever settings every arm names; the child sets all of them before its
# first dispatch, so the latched levers read live.
LEVERS = ("KQ_MOE_NX", "KQ_MOE_HALF", "KQ_MOE_C32", "KQ_MOE_DEDUP", "KQ_MOE_MIX_FORM")
BASE_ENV = {
    "KQ_MOE_NX": "",
    "KQ_MOE_HALF": "0",
    "KQ_MOE_C32": "0",
    "KQ_MOE_DEDUP": "1",
    "KQ_MOE_MIX_FORM": "",
}


def _arm(
    site, codec, T, *, sx=None, su=None, env=None, dtype="float16", expect, scales=False
):
    return {
        "site": site,
        "codec": codec,
        "sx": sx,
        "su": su,
        "T": T,
        "env": {**BASE_ENV, **(env or {})},
        "dtype": dtype,
        "expect": expect,
        "scales": scales,
    }


def arms():
    """Every (site, family) arm, with the kernel-name pattern it must
    dispatch. {act} is the instantiation the call names."""
    out = []
    for c in GLU_CODECS:
        ext = c + "_ext" if c in ("q6_k", "q8_0") else c
        out += [
            _arm("glu", c, 1, expect=rf"^kq_{ext}_moe_glu_gather_{{act}}_nx16_"),
            _arm(
                "glu",
                c,
                1,
                env={"KQ_MOE_NX": "32"},
                expect=rf"^kq_{ext}_moe_glu_gather_{{act}}_nx32_",
            ),
            _arm(
                "glu",
                c,
                1,
                env={"KQ_MOE_NX": "8"},
                expect=rf"^kq_{c}_moe_glu_gather_{{act}}_float16_t$",
            ),
            _arm("glu", c, 3, expect=rf"^kq_{ext}_moe_glu_gather_dd_{{act}}_nx16_"),
            _arm("shexp", c, 1, expect=rf"^kq_{ext}_moe_glu_gather_shexp_silu_nx16_"),
            _arm(
                "shexp",
                c,
                1,
                env={"KQ_MOE_NX": "8"},
                expect=rf"^kq_{c}_moe_glu_gather_shexp_silu_float16_t$",
            ),
            _arm(
                "shexp", c, 3, expect=rf"^kq_{ext}_moe_glu_gather_shexp_dd_silu_nx16_"
            ),
        ]
        if c != "q8_0":
            out.append(
                _arm(
                    "shexp",
                    c,
                    1,
                    sx="q8_0",
                    expect=rf"^kq_{c}_sx_q8_0_moe_glu_gather_shexp_silu_nx16_",
                )
            )
            out.append(
                _arm(
                    "shexp",
                    c,
                    3,
                    sx="q8_0",
                    expect=rf"^kq_{c}_sx_q8_0_moe_glu_gather_shexp_dd_silu_nx16_",
                )
            )
    for c in BF16_CODECS:
        ext = c + "_ext" if c in ("q6_k", "q8_0") else c
        out += [
            _arm(
                "glu",
                c,
                1,
                dtype="bfloat16",
                expect=rf"^kq_{ext}_moe_glu_gather_{{act}}_nx16_bfloat16_t$",
            ),
            _arm(
                "shexp",
                c,
                3,
                dtype="bfloat16",
                expect=rf"^kq_{ext}_moe_glu_gather_shexp_dd_silu_nx16_bfloat16_t$",
            ),
        ]
    # the M3.1 file's pairings, the IQ2_XS and IQ3_XXS stacks
    for c, s in (
        ("iq2_xs", "q5_k"),
        ("iq3_xxs", "q6_k"),
        ("q4_k", "q5_k"),
        ("q2_k", "q6_k"),
    ):
        out += [
            _arm(
                "shexp",
                c,
                1,
                sx=s,
                expect=rf"^kq_{c}_sx_{s}_moe_glu_gather_shexp_silu_nx16_",
            ),
            _arm(
                "shexp",
                c,
                3,
                sx=s,
                expect=rf"^kq_{c}_sx_{s}_moe_glu_gather_shexp_dd_silu_nx16_",
            ),
        ]
    for c in C32_CODECS:
        for nx in ("", "8"):
            env = {"KQ_MOE_C32": "1", "KQ_MOE_NX": nx}
            sfx = "(_nx16)?" if nx == "" else ""
            out.append(
                _arm(
                    "glu",
                    c,
                    1,
                    env=env,
                    expect=rf"^kq_{c}_moe_glu_gather_{{act}}_c32{sfx}_float16_t$",
                )
            )
            for s in (None,) + UPCAST:
                stem = c if s is None else f"{c}_sx_{s}"
                out.append(
                    _arm(
                        "shexp",
                        c,
                        1,
                        sx=s,
                        env=env,
                        expect=rf"^kq_{stem}_moe_glu_gather_shexp_silu_c32{sfx}_float16_t$",
                    )
                )
    for c in HALF_CODECS:
        for T in (1, 3):
            env = {"KQ_MOE_HALF": "1"}
            out.append(
                _arm(
                    "glu",
                    c,
                    T,
                    env=env,
                    expect=rf"^kq_{c}_moe_glu_gather_h_{{act}}_float16_t$",
                )
            )
            for s in (None,) + UPCAST:
                stem = c if s is None else f"{c}_sx_{s}"
                out.append(
                    _arm(
                        "shexp",
                        c,
                        T,
                        sx=s,
                        env=env,
                        expect=rf"^kq_{stem}_moe_glu_gather_shexp_h_silu_float16_t$",
                    )
                )
    for T in (1, 3):
        out.append(
            _arm(
                "glu",
                "nvfp4",
                T,
                scales=True,
                expect=r"^kq_nvfp4_moe_glu_gather_(dd_)?{act}_nx16_",
            )
        )
        out.append(
            _arm(
                "shexp",
                "nvfp4",
                T,
                scales=True,
                expect=r"^kq_nvfp4_moe_glu_gather_shexp_(dd_)?silu_nx16_",
            )
        )
    for c in ("iq2_s", "iq2_xxs", "iq1_m"):
        for sg, su in SX2:
            for T in (1, 3):
                out.append(
                    _arm(
                        "sx2",
                        c,
                        T,
                        sx=sg,
                        su=su,
                        expect=rf"^kq_{c}_sx2_{sg}_{su}_moe_glu_gather_shexp_silu_nx16_",
                    )
                )
    # down gathers: no act; checked by value only
    for c in ("iq2_xs", "iq3_xxs", "q5_k", "q6_k", "q8_0", "nvfp4"):
        ext = c + "_ext" if c in ("q6_k", "q8_0") else c
        out.append(_arm("qmv", c, 1, expect=rf"^kq_{c}_gather_qmv_"))
        out.append(_arm("qmv", c, 3, expect=rf"^kq_{c}_gather_qmv_"))
    for c, s in (
        ("iq2_xs", "q5_k"),
        ("iq3_xxs", "q6_k"),
        ("q6_k", None),
        ("q8_0", None),
        ("q4_k", "q8_0"),
    ):
        stem = c if s is None else f"{c}_sx_{s}"
        for form, T, pat in (
            ("sp", 1, "_gather_qmv_mix_sp_"),
            ("loop", 3, "_gather_qmv_mix(_fine)?_"),
            ("dd", 3, "_gather_qmv_mix_dd_"),
        ):
            st = stem + "_ext" if form == "dd" and s is None else stem
            if form == "sp" and s is None:
                # the tuned q6_k/q8_0 mix has no slot-parallel form
                pat = "_gather_qmv_mix(_fine)?_"
            out.append(
                _arm(
                    "mix",
                    c,
                    T,
                    sx=s,
                    env={"KQ_MOE_MIX_FORM": form},
                    expect=rf"^kq_{st}{pat}",
                )
            )
    for c in ("iq2_xs", "q6_k", "q4_k"):
        for form, T, pat in (
            ("sp", 1, "_gather_qmv_mix_ns_sp_"),
            ("loop", 3, "_gather_qmv_mix_ns(_nx16)?_"),
            ("dd", 3, "_gather_qmv_mix_ns_dd"),
        ):
            out.append(
                _arm(
                    "mix_ns",
                    c,
                    T,
                    env={"KQ_MOE_MIX_FORM": form},
                    expect=rf"^kq_{c}{pat}",
                )
            )
    for a in out:
        lev = "".join(
            f"-{k[3:].lower()}{v}" for k, v in a["env"].items() if v != BASE_ENV[k]
        )
        a["id"] = (
            f"{a['site']}-{a['codec']}-{a['sx']}-{a['su']}-t{a['T']}-"
            f"{a['dtype']}{lev}{'-scaled' if a['scales'] else ''}"
        )
    assert len({a["id"] for a in out}) == len(out)
    return out


# acts per site: the ones the 0.5.7 reference covers, and swiglu_oai
OLD_ACTS = {
    "glu": ("silu", "silu_limit"),
    "shexp": ("silu", "silu_limit"),
    "sx2": ("silu",),
    "qmv": ("none",),
    "mix": ("none",),
    "mix_ns": ("none",),
}
OAI_SITES = ("glu", "shexp", "sx2")


# child: build each arm's operands and run it


def _wires():
    sys.path.insert(0, HERE)
    import test_moe_glu as base

    cache = {}

    def get(codec, seed=11):
        key = (codec, seed)
        if key not in cache:
            w, r = base._wire_and_ref(codec, seed=seed)
            if w is None:
                raise RuntimeError(f"fixture for {codec} missing")
            cache[key] = (w[:, :N].copy(), r[:, :N].copy())
        return cache[key]

    return get


def _oai_np(g, u, limit, alpha):
    g = np.minimum(g, limit)
    u = np.clip(u, -limit, limit)
    with np.errstate(over="ignore"):
        return g / (1.0 + np.exp(-alpha * g)) * (u + 1.0)


def _bits(a):
    mx.eval(a)
    return np.array(a.view(mx.uint16))


def _f32(bits, dtype):
    if dtype == "bfloat16":
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return bits.view(np.float16).astype(np.float32)


def _run_arm(a, get, acts):
    """{act: (bits, rel or None)} for one arm; rel is against the f32
    reference for swiglu_oai."""
    dt = getattr(mx, a["dtype"])
    rng = np.random.default_rng(zlib.crc32(a["id"].encode()))
    c, T = a["codec"], a["T"]
    wire, ref = get(c)
    gw, uw = mx.array(wire), mx.array(wire[::-1].copy())
    gref, uref = ref, ref[::-1]
    x = mx.array((rng.standard_normal((T, K)) * 0.1).astype(np.float16)).astype(dt)
    xf = np.array(x.astype(mx.float32))
    inds_np = rng.integers(0, E, size=(T, R)).astype(np.uint32)
    inds = mx.array(inds_np)
    site = a["site"]
    res = {}
    if site in ("qmv", "mix", "mix_ns"):
        S = R + (1 if site == "mix" else 0)
        h = mx.array((rng.standard_normal((T, S, K)) * 0.1).astype(np.float16)).astype(
            dt
        )
        if site == "qmv":
            y = kq.gather_qmv_kq(h[:, :R], gw, c, inds)
        elif site == "mix_ns":
            sc = mx.array(rng.uniform(0.05, 0.9, (T, R)).astype(np.float32))
            y = kq.gather_qmv_mix_ns_kq(h[:, :R], gw, c, inds, sc)
        else:
            sw = get(a["sx"] or c, 13)[0][1] if a["sx"] else wire[1]
            sc = mx.array(rng.uniform(0.05, 0.9, (T, S)).astype(np.float32))
            y = kq.gather_qmv_mix_kq(
                h, gw, mx.array(sw), c, inds, sc, shexp_kquant_type=a["sx"] or ""
            )
        _mark(a["id"], "none")
        res["none"] = (_bits(y), None)
        return res

    g_all = np.einsum("tk,enk->ten", xf, gref)
    limit = float(np.float16(np.quantile(np.abs(g_all), 0.6)))
    kw = {}
    sref = None
    if a["scales"]:
        # per expert and column: each scales its product
        gs = rng.uniform(0.5, 1.5, (E, N)).astype(np.float32)
        us = rng.uniform(0.5, 1.5, (E, N)).astype(np.float32)
        kw = {"gate_scale": mx.array(gs), "up_scale": mx.array(us)}
        gref = gref * gs[:, :, None]
        uref = uref * us[:, :, None]
    if site in ("shexp", "sx2"):
        sg_codec = a["sx"] or c
        su_codec = a["su"] or sg_codec
        sgw, sgref = (
            (wire[2], ref[2])
            if a["sx"] is None
            else (get(sg_codec, 13)[0][2], get(sg_codec, 13)[1][2])
        )
        suw, suref = (
            (wire[3], ref[3])
            if a["sx"] is None
            else (get(su_codec, 17)[0][3], get(su_codec, 17)[1][3])
        )
        if a["scales"]:
            kw["shexp_gate_scale"] = mx.array(np.array([1.25], np.float32))
            kw["shexp_up_scale"] = mx.array(np.array([0.75], np.float32))
            sgref, suref = sgref * 1.25, suref * 0.75
        sref = (sgref, suref)
    for act in acts:
        akw = dict(kw)
        if act == "silu_limit":
            akw["limit"] = limit
        if act == "swiglu_oai":
            akw.update(limit=limit, alpha=ALPHA)
        if site == "glu":
            y = kq.moe_glu_gather_kq(x, gw, uw, c, inds, act=act, **akw)
        else:
            if site == "sx2":
                akw["shexp_up_kquant_type"] = a["su"]
            y = kq.moe_glu_gather_shexp_kq(
                x,
                gw,
                uw,
                mx.array(sgw),
                mx.array(suw),
                c,
                inds,
                act=act,
                shexp_kquant_type=a["sx"] or "",
                **akw,
            )
        mx.eval(x, gw, uw, inds)
        _mark(a["id"], act)
        bits = _bits(y)
        rel = None
        if act == "swiglu_oai":
            rows = [
                [
                    _oai_np(xf[t] @ gref[e].T, xf[t] @ uref[e].T, limit, ALPHA)
                    for e in inds_np[t]
                ]
                for t in range(T)
            ]
            if sref is not None:
                for t in range(T):
                    rows[t].append(
                        _oai_np(xf[t] @ sref[0].T, xf[t] @ sref[1].T, limit, ALPHA)
                    )
            want = np.array(rows, np.float32)
            want = np.array(mx.array(want).astype(dt).astype(mx.float32))
            got = _f32(bits, a["dtype"])
            rel = float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-6))
        res[act] = (bits, rel)
    return res


def _mark(arm_id, act):
    os.write(2, f"@@ {arm_id} {act}\n".encode())


def child_main(mode, order, out_path):
    """mode ref: the 0.5.7-reference acts only; mode run: those and
    swiglu_oai, in the given order."""
    get = _wires()
    data, rels = {}, {}
    for a in arms():
        for k, v in a["env"].items():
            os.environ[k] = v
        old = OLD_ACTS[a["site"]]
        if mode == "ref":
            acts = old
        elif a["site"] in OAI_SITES:
            acts = (
                old + ("swiglu_oai",) if order == "old_first" else ("swiglu_oai",) + old
            )
        else:
            acts = old
        for act, (bits, rel) in _run_arm(a, get, acts).items():
            data[f"{a['id']}|{act}"] = bits
            if rel is not None:
                rels[f"{a['id']}|{act}"] = rel
    if mode == "ref":
        data["__arch__"] = np.array(mx.device_info().get("architecture", ""))
        np.savez_compressed(out_path, **data)
    else:
        np.savez(out_path, **data)
        with open(out_path + ".rel.json", "w") as f:
            json.dump(rels, f)


# parent


def _child(mode, order, out_path):
    env = {**os.environ, **BASE_ENV, "KQ_QMM_LOG": "1"}
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", mode, order, out_path],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, (
        f"child {mode} {order} exited {r.returncode}\n"
        f"--- stdout ---\n{r.stdout[-4000:]}\n--- stderr ---\n{r.stderr[-8000:]}"
    )
    names, key = {}, None
    for line in r.stderr.splitlines():
        if line.startswith("@@ "):
            _, arm_id, act = line.split()
            key = f"{arm_id}|{act}"
            names[key] = []
        elif line.startswith("[kq qmm] ") and key is not None:
            names[key].append(line.split()[2])
    return names


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for order in ("old_first", "oai_first"):
        p = str(tmp_path_factory.mktemp(order) / "out.npz")
        names = _child("run", order, p)
        with open(p + ".rel.json") as f:
            rels = json.load(f)
        out[order] = (dict(np.load(p)), names, rels)
    return out


def _arm_by_id():
    return {a["id"]: a for a in arms()}


@gpu_only
def test_every_site_and_family_dispatches(runs):
    """Each arm runs the kernel family it names, for every act: a lever or
    shape that stopped selecting its family fails here."""
    by_id = _arm_by_id()
    for order, (_, names, _) in runs.items():
        for key, ks in names.items():
            arm_id, act = key.split("|")
            a = by_id[arm_id]
            inst = "silu" if act in ("swiglu_oai", "none") else act
            if a["site"] != "glu":
                inst = "silu"
            pat = a["expect"].replace("{act}", inst)
            moe = [k for k in ks if "moe_glu" in k or "gather_qmv" in k]
            assert moe and re.search(pat, moe[0]), (order, key, pat, ks)


@gpu_only
def test_launch_sites_covered():
    sites = {a["site"] for a in arms()}
    assert sites == {"glu", "shexp", "sx2", "qmv", "mix", "mix_ns"}
    fams = {
        (a["site"], f)
        for a in arms()
        for f in ("_c32", "_dd_", "_h_", "_nx32", "_sx_", "_sx2_", "_float16_t$")
        if f.strip("$") in a["expect"]
    }
    for need in (
        ("glu", "_c32"),
        ("glu", "_dd_"),
        ("glu", "_h_"),
        ("glu", "_nx32"),
        ("glu", "_float16_t$"),
        ("shexp", "_c32"),
        ("shexp", "_dd_"),
        ("shexp", "_h_"),
        ("shexp", "_sx_"),
        ("shexp", "_float16_t$"),
        ("sx2", "_sx2_"),
    ):
        assert need in fams, need


@gpu_only
@pytest.mark.parametrize("order", ["old_first", "oai_first"])
def test_swiglu_oai_matches_reference(runs, order):
    _, _, rels = runs[order]
    n_oai = sum(1 for a in arms() if a["site"] in OAI_SITES)
    assert len(rels) == n_oai
    bad = {k: v for k, v in rels.items() if not v < REL_BOUND}
    assert not bad, bad


@gpu_only
@pytest.mark.parametrize("order", ["old_first", "oai_first"])
def test_older_forms_match_the_release(runs, order):
    """silu, silu_limit and the down gathers give the 0.5.7 values."""
    if not os.path.exists(REF):
        pytest.skip("reference outputs missing")
    ref = np.load(REF)
    exact = str(ref["__arch__"]) == mx.device_info().get("architecture", "")
    data, _, _ = runs[order]
    by_id = _arm_by_id()
    n = 0
    for key in ref.files:
        if key == "__arch__":
            continue
        a = by_id[key.split("|")[0]]
        want = _f32(ref[key], a["dtype"])
        got = _f32(data[key], a["dtype"])
        if exact:
            assert np.array_equal(got, want), key
        else:
            rel = np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-6)
            assert rel < REL_BOUND, (key, rel)
        n += 1
    assert n == sum(len(OLD_ACTS[a["site"]]) for a in arms())


@gpu_only
def test_orders_agree(runs):
    a, _, _ = runs["old_first"]
    b, _, _ = runs["oai_first"]
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key], b[key]), key


# host-side contract (no dispatch)


def _q8_wire():
    return mx.zeros((4, 8, 512 // 32 * 34), dtype=mx.uint8)


@pytest.mark.parametrize("op", ["glu", "shexp"])
@pytest.mark.parametrize("limit,alpha", [(0.0, 1.702), (7.0, 0.0), (-1.0, 1.0)])
def test_swiglu_oai_requires_limit_and_alpha(op, limit, alpha):
    x = mx.zeros((2, 512), dtype=mx.float16)
    inds = mx.zeros((2, 2), dtype=mx.uint32)
    w = _q8_wire()
    with pytest.raises(ValueError, match="swiglu_oai"):
        if op == "glu":
            kq.moe_glu_gather_kq(
                x, w, w, "q8_0", inds, act="swiglu_oai", limit=limit, alpha=alpha
            )
        else:
            kq.moe_glu_gather_shexp_kq(
                x,
                w,
                w,
                w[0],
                w[0],
                "q8_0",
                inds,
                act="swiglu_oai",
                limit=limit,
                alpha=alpha,
            )


def test_swiglu_oai_takes_no_biases():
    """The biased kernels have no swiglu_oai switch; biases stay with
    swiglu_clamp."""
    x = mx.zeros((2, 512), dtype=mx.float16)
    inds = mx.zeros((2, 2), dtype=mx.uint32)
    w = mx.zeros((4, 8, 512 // 32 * 17), dtype=mx.uint8)  # mxfp4 geometry
    b = mx.zeros((4, 8), dtype=mx.float32)
    with pytest.raises(ValueError, match="swiglu_clamp"):
        kq.moe_glu_gather_kq(
            x,
            w,
            w,
            "mxfp4",
            inds,
            act="swiglu_oai",
            limit=7.0,
            alpha=1.702,
            gate_bias=b,
            up_bias=b,
        )


def test_swiglu_oai_named_in_docstrings():
    """Callers find the act by its name in the docstrings."""
    assert "swiglu_oai" in kq.moe_glu_gather_kq.__doc__
    assert "swiglu_oai" in kq.moe_glu_gather_shexp_kq.__doc__


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child_main(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "--write-ref":
        for k, v in BASE_ENV.items():
            os.environ[k] = v
        child_main("ref", "old_first", sys.argv[2])
    else:
        raise SystemExit(__doc__)
