"""The dense shared-expert route of the fused MoE ops.

From KQ_MOE_SHEXP_DENSE tokens, moe_glu_gather_shexp_kq takes the shared gate
and up products from quantized_matmul_multi and gather_qmv_mix_kq the shared
down product, and the fused kernels skip the shared slots. One child per
kernel form runs each case with the route on from 2 tokens and off (the lever
is read live) and saves both outputs and the dense products. The parent
checks:

- the routed GLU slots match bit for bit between the two arms;
- with the route on, the shared GLU slots match the activation of the dense
  products, and the mix output matches the routed float sum plus the dense
  down product, within REL_BOUND of the largest output, and both differ
  from the route off, so the route ran;
- both arms match the float reference; on the route the GLU is held to
  REL_BOUND of the largest gate or up product, since its products round
  through the dense matmul and the clamps keep outputs well below them;
- one token, a shared gate and up in two codecs, and a mix whose shared
  slots carry their own scores keep the fused kernels: both arms match bit
  for bit;
- the kernel log shows each form and, where the route runs, its epilogue.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="fused MoE gathers are Metal-only kernels; no CPU path.",
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
E, H, N, R = 8, 512, 512, 4  # experts, hidden size, expert width, routed slots
ROWS = (1, 3, 8, 15)
REL_BOUND = 2e-3
# limits small enough that the clamps engage on these inputs
LIMIT = {"silu_limit": {"limit": 0.2}, "swiglu_oai": {"limit": 0.2, "alpha": 1.702}}
# (name, gate/up stack, shexp gate, shexp up, down stack, shexp down, k)
CASES = [
    ("q4_k_q5_k", "q4_k", "q5_k", "q5_k", "q4_k", "q5_k", 1),
    ("iq2_xs_q5_k", "iq2_xs", "q5_k", "q5_k", "iq2_xs", "q6_k", 1),
    ("q6_k", "q6_k", "q6_k", "q6_k", "q6_k", "q6_k", 1),
    ("q8_0", "q8_0", "q8_0", "q8_0", "q8_0", "q8_0", 1),
    ("q4_1_k2", "q4_1", "q4_1", "q4_1", "q4_1", "q4_1", 2),
    ("split", "iq2_s", "iq4_xs", "q6_k", "iq2_s", "q8_0", 1),
]
ACTS = ("silu", "gelu", "silu_limit", "swiglu_oai")
# kernel name parts each arm must have run
ARMS = {
    "default": ({}, ("kq_moe_shexp_glu_pre_silu_", "kq_moe_shexp_glu_pre_gelu_")),
    "loop": (
        {"KQ_MOE_MIX_FORM": "loop", "KQ_MOE_DEDUP": "0"},
        (
            "_qf_sx_q5_k_gather_qmv_mix_float16_t",
            "kq_iq2_xs_sx_q6_k_gather_qmv_mix_float16_t",
            "kq_q6_k_gather_qmv_mix_fine_",
        ),
    ),
    "nx8": (
        {"KQ_MOE_NX": "8", "KQ_MOE_DEDUP": "0"},
        ("kq_q6_k_moe_glu_gather_shexp_", "kq_q8_0_moe_glu_gather_shexp_"),
    ),
    "sp": ({"KQ_MOE_MIX_FORM": "sp"}, ("_gather_qmv_mix_sp_",)),
    "dd": (
        {"KQ_MOE_MIX_FORM": "dd", "KQ_MOE_QF": "0"},
        ("_gather_qmv_mix_dd", "_moe_glu_gather_shexp_dd_"),
    ),
    "half": (
        {"KQ_MOE_HALF": "1"},
        ("_moe_glu_gather_shexp_h_", "_gather_qmv_mix_h_"),
    ),
    "ext_loop": (
        {"KQ_MOE_QF": "0", "KQ_MOE_MIX_FORM": "loop", "KQ_MOE_DEDUP": "0"},
        ("kq_q4_k_sx_q5_k_gather_qmv_mix_", "kq_q4_k_sx_q5_k_moe_glu_gather_shexp_"),
    ),
}


def _quantize(rng, shape, codec):
    import mlx.core as mx

    import mlx_kquant as kq

    w = mx.array((rng.standard_normal(shape) * 0.1).astype(np.float32))
    im = mx.ones((shape[-1],), dtype=mx.float32) if codec.startswith("iq") else None
    wq, _ = kq.quantize(w, codec, im)
    mx.eval(wq)
    return np.array(wq)


@pytest.fixture(scope="module")
def wires(tmp_path_factory):
    rng = np.random.default_rng(29)
    out = {}
    for name, cg, sg, su, cd, sd, k in CASES:
        out[f"{name}/gw"] = _quantize(rng, (E, N, H), cg)
        out[f"{name}/uw"] = _quantize(rng, (E, N, H), cg)
        out[f"{name}/dw"] = _quantize(rng, (E, H, N), cd)
        out[f"{name}/sg"] = _quantize(rng, (k * N, H), sg)
        out[f"{name}/su"] = _quantize(rng, (k * N, H), su)
        out[f"{name}/sd"] = _quantize(rng, (H, k * N), sd)
        for t in ROWS:
            out[f"{name}/x{t}"] = (rng.standard_normal((t, H)) * 0.3).astype(np.float16)
            out[f"{name}/h{t}"] = (rng.standard_normal((t, R + k, N)) * 0.3).astype(
                np.float16
            )
            ind = rng.integers(0, E, size=(t, R)).astype(np.uint32)
            ind[:, 0] = 3  # rows share an expert, which the dedupe forms pair
            out[f"{name}/i{t}"] = ind
            out[f"{name}/s{t}"] = rng.uniform(0.05, 0.9, size=(t, R + k)).astype(
                np.float32
            )
    path = os.path.join(str(tmp_path_factory.mktemp("shexp_dense")), "wires.npz")
    np.savez(path, **out)
    return path


def child(wire_path, out_path, rows):
    """Both ops per case, row count and act, with the route on and off."""
    import mlx.core as mx

    import mlx_kquant as kq

    z = np.load(wire_path)
    res = {}
    for name, cg, sg, su, cd, sd, k in CASES:
        a = {
            key: mx.array(z[f"{name}/{key}"])
            for key in ("gw", "uw", "dw", "sg", "su", "sd")
        }
        for t in rows:
            x = mx.array(z[f"{name}/x{t}"])
            h = mx.array(z[f"{name}/h{t}"])
            ind = mx.array(z[f"{name}/i{t}"])
            sc = mx.array(z[f"{name}/s{t}"])
            for arm, lever in (("on", "2"), ("off", "0")):
                os.environ["KQ_MOE_SHEXP_DENSE"] = lever
                for act in ACTS if name != "split" else ("silu",):
                    res[f"{name}/{act}{t}/{arm}"] = kq.moe_glu_gather_shexp_kq(
                        x,
                        a["gw"],
                        a["uw"],
                        a["sg"],
                        a["su"],
                        cg,
                        ind,
                        act=act,
                        shexp_kquant_type=sg,
                        shexp_up_kquant_type=su,
                        **LIMIT.get(act, {}),
                    )
                for s in (R + k, R):
                    res[f"{name}/mix{s}_{t}/{arm}"] = kq.gather_qmv_mix_kq(
                        h, a["dw"], a["sd"], cd, ind, sc[:, :s], shexp_kquant_type=sd
                    )
            if sg == su:
                res[f"{name}/pg{t}"] = kq.quantized_matmul(
                    x, a["sg"], mx.array([0.0]), sg
                )
                res[f"{name}/pu{t}"] = kq.quantized_matmul(
                    x, a["su"], mx.array([0.0]), su
                )
            res[f"{name}/pd{t}"] = kq.quantized_matmul(
                h[:, R:].reshape(t, k * N), a["sd"], mx.array([0.0]), sd
            )
    mx.eval(res)
    np.savez(
        out_path, **{key: np.array(v.astype(mx.float32)) for key, v in res.items()}
    )


_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
import json
import test_moe_shexp_dense as m
m.child(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))
"""


def _run(wire_path, out_path, rows, env):
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, TESTS_DIR, wire_path, out_path, str(list(rows))],
        capture_output=True,
        text=True,
        env={**os.environ, "KQ_MOE_NX_LOG": "1", **env},
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    return np.load(out_path), r.stderr


def _deq(z, name, key, codec, shape):
    import mlx.core as mx

    import mlx_kquant as kq

    w = mx.array(z[f"{name}/{key}"])
    d = kq.dequantize(
        w.reshape(-1, w.shape[-1]), mx.array([0.0]), codec, dtype=mx.float32
    )
    return np.array(d).reshape(shape)


def _act(act, g, u):
    if act == "gelu":
        return 0.5 * g * (1 + np.tanh(0.7978845608028654 * (g + 0.044715 * g**3))) * u
    if act in LIMIT:
        lim = LIMIT[act]["limit"]
        g = np.minimum(g, lim)
        u = np.clip(u, -lim, lim)
    if act == "swiglu_oai":
        a = LIMIT[act]["alpha"]
        return g / (1 + np.exp(-a * g)) * (u + 1)
    return g / (1 + np.exp(-g)) * u


def _rel(a, b):
    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-6))


def _check(z, o, rows):
    for name, cg, sg, su, cd, sd, k in CASES:
        gw = _deq(z, name, "gw", cg, (E, N, H))
        uw = _deq(z, name, "uw", cg, (E, N, H))
        sgw = _deq(z, name, "sg", sg, (k * N, H))
        suw = _deq(z, name, "su", su, (k * N, H))
        dw = _deq(z, name, "dw", cd, (E, H, N))
        sdw = _deq(z, name, "sd", sd, (H, k * N))
        for t in rows:
            x = z[f"{name}/x{t}"].astype(np.float32)
            h = z[f"{name}/h{t}"].astype(np.float32)
            ind = z[f"{name}/i{t}"]
            sc = z[f"{name}/s{t}"]
            dense = t >= 2
            for act in ACTS if name != "split" else ("silu",):
                on, off = o[f"{name}/{act}{t}/on"], o[f"{name}/{act}{t}/off"]
                ref = np.zeros((t, R + k, N), np.float32)
                for i in range(t):
                    for r in range(R):
                        ref[i, r] = _act(
                            act, gw[ind[i, r]] @ x[i], uw[ind[i, r]] @ x[i]
                        )
                    for j in range(k):
                        rows_j = slice(j * N, (j + 1) * N)
                        ref[i, R + j] = _act(
                            act, sgw[rows_j] @ x[i], suw[rows_j] @ x[i]
                        )
                tag = f"{name} {act} T={t}"
                assert _rel(off, ref) <= REL_BOUND, tag
                if not (dense and sg == su):
                    assert np.array_equal(on, off), tag
                    continue
                assert np.array_equal(on[:, :R], off[:, :R]), tag
                assert not np.array_equal(on[:, R:], off[:, R:]), tag
                prod = max(np.abs(o[f"{name}/{p}{t}"]).max() for p in ("pg", "pu"))
                assert np.abs(on - ref).max() <= REL_BOUND * prod, tag
                pre = _act(act, o[f"{name}/pg{t}"], o[f"{name}/pu{t}"]).reshape(t, k, N)
                assert _rel(on[:, R:], pre) <= REL_BOUND, tag
            for s in (R + k, R):
                on, off = o[f"{name}/mix{s}_{t}/on"], o[f"{name}/mix{s}_{t}/off"]
                routed = np.zeros((t, H), np.float32)
                ref = np.zeros((t, H), np.float32)
                for i in range(t):
                    for r in range(R):
                        routed[i] += sc[i, r] * (dw[ind[i, r]] @ h[i, r])
                    ref[i] = routed[i]
                    for j in range(k):
                        wj = sc[i, R + j] if s == R + k else 1.0
                        ref[i] += wj * (sdw[:, j * N : (j + 1) * N] @ h[i, R + j])
                tag = f"{name} mix SC={s} T={t}"
                assert _rel(off, ref) <= REL_BOUND, tag
                if not (dense and (k == 1 or s == R)):
                    assert np.array_equal(on, off), tag
                    continue
                assert not np.array_equal(on, off), tag
                assert _rel(on, ref) <= REL_BOUND, tag
                w0 = sc[:, R : R + 1] if s == R + k else 1.0
                pre = routed + w0 * o[f"{name}/pd{t}"]
                assert _rel(on, pre) <= REL_BOUND, tag


@pytest.mark.parametrize("arm", list(ARMS))
def test_dense_shexp_forms(wires, tmp_path, arm):
    env, parts = ARMS[arm]
    o, log = _run(wires, str(tmp_path / "out.npz"), ROWS[1:], env)
    for p in parts:
        assert p in log, (arm, p)
    _check(np.load(wires), o, ROWS[1:])


def test_one_token_keeps_fused_kernels(wires, tmp_path):
    o, log = _run(wires, str(tmp_path / "out.npz"), (1,), {})
    assert "kq_moe_shexp_glu_pre" not in log
    _check(np.load(wires), o, (1,))
