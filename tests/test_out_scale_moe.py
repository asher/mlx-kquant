"""Output scales on the fused MoE ops: the per-tensor or per-expert
.scale of an NVFP4 GGUF, multiplied into the float32 sums in the kernel.

moe_glu_gather_kq and moe_glu_gather_shexp_kq scale gate and up before the
biases and the activation, gather_qmv_kq scales before its bias and adds
the LoRA delta unscaled, and the mixes scale each slot's weight.

A power-of-two scale 2^k on output row n of expert e equals no scale on a
wire whose group scales in that row have an exponent k higher, since every
float32 sum is then exactly 2^k times larger. So each form is checked bit
for bit against itself unscaled on the shifted wire, for every scale shape,
which catches a scale on the wrong row, expert, projection or side of a
bias. A scale of ones gives the bits of no scale. Real scales (5e-5 to
4.4e-4) on activations whose unscaled sums pass the float16 range give
finite results within the float64 bound.

The fused-MoE levers are read once per process, so each form runs in a
child process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

FORCE_CPU = bool(os.environ.get("KQUANT_FORCE_CPU"))
E, N, K, R = 8, 64, 512, 3
TOL = {"bfloat16": 2.0**-7, "float16": 2.0**-10}
FLOOR = {"bfloat16": 2.0**-126, "float16": 2.0**-14}
SHAPES = ((1,), (N,), (E, 1), (E, N))
KW = {
    "g": "gate_scale",
    "u": "up_scale",
    "d": "out_scale",
    "sg": "shexp_gate_scale",
    "su": "shexp_up_scale",
    "sd": "shexp_scale",
}


def _nv(rows, seed):
    """nvfp4 rows whose group scales have exponents 4 to 6, so a shift of
    -3 to 3 stays normal and below the NaN code."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, (*rows, K // 64, 36), dtype=np.uint8)
    w[..., :4] = rng.integers(0x20, 0x38, (*rows, K // 64, 4))
    return w.reshape(*rows, -1)


def _q8(rows, seed):
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, (*rows, K // 32, 34), dtype=np.uint8)
    d = rng.uniform(0.01, 0.02, (*rows, K // 32)).astype(np.float16)
    w[..., :2] = d.view(np.uint8).reshape(*rows, K // 32, 2)
    return w.reshape(*rows, -1)


def _shift(wire, k):
    """wire with the group scales of each row times 2^k (k per row)."""
    w = wire.reshape(*wire.shape[:-1], -1, 36).astype(np.int16)
    k = np.broadcast_to(np.asarray(k, np.int16), wire.shape[:-1])
    w[..., :4] += k[..., None, None] * 8
    assert (w[..., :4] >= 0x08).all() and (w[..., :4] < 0x50).all()
    return w.astype(np.uint8).reshape(wire.shape)


def _exps(rng, shape):
    if shape == (E, 1):  # one exponent per expert, nearly all distinct
        return rng.permutation(np.array([-3, -2, -1, 0, 1, 2, 3, 1]))[:, None]
    return rng.integers(-3, 4, shape)


def _child(spec):
    import mlx.core as mx

    import mlx_kquant as kq

    T = spec["T"]
    rng = np.random.default_rng(T)
    z = mx.zeros((1,), mx.uint8)
    routed = {"g": _nv((E, N), 1), "u": _nv((E, N), 2), "d": _nv((E, N), 3)}
    shared = {
        "nvfp4": {"sg": _nv((N,), 4), "su": _nv((N,), 5), "sd": _nv((N,), 6)},
        "q8_0": {"sg": _q8((N,), 4), "su": _q8((N,), 5), "sd": _q8((N,), 6)},
    }
    idx = rng.integers(0, E, (T, R)).astype(np.uint32)
    scores = rng.uniform(0.05, 0.9, (T, R + 1)).astype(np.float32)
    bias = rng.normal(0, 0.5, (2, E, N)).astype(np.float32)
    lora = [rng.normal(0, 0.05, s).astype(np.float32) for s in ((E, K, 4), (E, 4, N))]
    fails, ran, skipped = [], [], []

    def deq(wire, codec):
        w = kq.dequantize(
            mx.array(wire.reshape(-1, wire.shape[-1])), z, codec, dtype=mx.float32
        )
        return np.asarray(w).astype(np.float64).reshape(*wire.shape[:-1], K)

    w64 = {k: deq(v, "nvfp4") for k, v in routed.items()}
    s64 = {st: {k: deq(v, st) for k, v in d.items()} for st, d in shared.items()}

    def bits(y):
        return np.asarray(y.view(mx.uint16))

    for dtn in ("bfloat16", "float16"):
        dt = getattr(mx, dtn)
        a = [mx.array(v).astype(dt) for v in lora]

        def ops(x, h, a):
            """name -> (f(wires, scales), routed keys, shared keys, shared codec)."""
            ix, sc = mx.array(idx), mx.array(scores)
            gb, ub = mx.array(bias[0]), mx.array(bias[1])

            def glu(act, lim=0.0, **kw):
                return lambda w, s: kq.moe_glu_gather_kq(
                    x, w["g"], w["u"], "nvfp4", ix, act=act, limit=lim, **kw, **s
                )

            out = {
                "glu": (glu("silu"), "gu", "", None),
                "glu_gelu": (glu("gelu"), "gu", "", None),
                "glu_limit": (glu("silu_limit", 4.0), "gu", "", None),
                "glu_bias": (
                    glu("swiglu_clamp", 7.0, gate_bias=gb, up_bias=ub, alpha=1.702),
                    "gu",
                    "",
                    None,
                ),
                "qmv": (
                    lambda w, s: kq.gather_qmv_kq(h[:, :R], w["d"], "nvfp4", ix, **s),
                    "d",
                    "",
                    None,
                ),
                "qmv_bias": (
                    lambda w, s: kq.gather_qmv_kq(
                        h[:, :R], w["d"], "nvfp4", ix, bias=gb, **s
                    ),
                    "d",
                    "",
                    None,
                ),
                "qmv_lora": (
                    lambda w, s: kq.gather_qmv_kq(
                        h[:, :R], w["d"], "nvfp4", ix, lora_a=a[0], lora_b=a[1], **s
                    ),
                    "d",
                    "",
                    None,
                ),
                "mix_ns": (
                    lambda w, s: kq.gather_qmv_mix_ns_kq(
                        h[:, :R], w["d"], "nvfp4", ix, sc[:, :R], **s
                    ),
                    "d",
                    "",
                    None,
                ),
                "mix_ns_lora": (
                    lambda w, s: kq.gather_qmv_mix_ns_kq(
                        h[:, :R],
                        w["d"],
                        "nvfp4",
                        ix,
                        sc[:, :R],
                        lora_a=a[0],
                        lora_b=a[1],
                        **s,
                    ),
                    "d",
                    "",
                    None,
                ),
            }
            for st in shared:
                sk = ("sg", "su") if st == "nvfp4" else ()
                out[f"shexp_{st}"] = (
                    lambda w, s, st=st: kq.moe_glu_gather_shexp_kq(
                        x,
                        w["g"],
                        w["u"],
                        w["sg"],
                        w["su"],
                        "nvfp4",
                        ix,
                        shexp_kquant_type=st,
                        **s,
                    ),
                    "gu",
                    sk,
                    st,
                )
                out[f"mix_{st}"] = (
                    lambda w, s, st=st: kq.gather_qmv_mix_kq(
                        h, w["d"], w["sd"], "nvfp4", ix, sc, shexp_kquant_type=st, **s
                    ),
                    "d",
                    ("sd",) if st == "nvfp4" else (),
                    st,
                )
            return out

        def call(f, wires, scales):
            y = f(
                {k: mx.array(v) for k, v in wires.items()},
                {KW[k]: mx.array(v) for k, v in scales.items()},
            )
            mx.eval(y)
            return y

        # Exactness: ones and powers of two, every scale shape and pairing.
        x = (mx.random.normal((T, K), key=mx.random.key(1)) * 0.25).astype(dt)
        h = (mx.random.normal((T, R + 1, K), key=mx.random.key(2)) * 0.25).astype(dt)
        for name, (f, rk, sk, st) in ops(x, h, a).items():
            wires = {
                **{k: routed[k] for k in ("g", "u", "d")},
                **(shared[st] if st else {}),
            }
            try:
                y0 = call(f, wires, {})
            except (RuntimeError, ValueError) as e:
                skipped.append(f"{name}: {str(e)[:120]}")
                continue
            ran.append(f"{dtn}/{name}")
            combos = [(m, False) for m in range(4)]
            if sk:
                combos += [(m, True) for m in range(-1, 4)]
            for m, sh in combos:
                keys = ([k for k in rk] if m >= 0 else []) + (list(sk) if sh else [])
                shape = {k: (SHAPES[m] if k in rk else (1,)) for k in keys}
                what = f"{dtn}/{name}/mode{m + 1}{'+shared' if sh else ''}"
                y1 = call(f, wires, {k: np.ones(shape[k], np.float32) for k in keys})
                if not np.array_equal(bits(y1), bits(y0)):
                    fails.append(f"{what}: ones differ from no scale")
                ks = {k: _exps(rng, shape[k]) for k in keys}
                p2 = {k: np.exp2(ks[k]).astype(np.float32) for k in keys}
                shifted = dict(wires)
                for k in keys:
                    shifted[k] = _shift(
                        wires[k],
                        np.broadcast_to(ks[k], (E, N)) if k in rk else ks[k][0],
                    )
                y, want = call(f, wires, p2), call(f, shifted, {})
                if not np.array_equal(bits(y), bits(want)):
                    bad = int((bits(y) != bits(want)).sum())
                    fails.append(f"{what}: {bad} outputs differ from the shifted wire")

        # Real scales on sums past the float16 range.
        x = (mx.random.normal((T, K), key=mx.random.key(3)) * 3000).astype(dt)
        h = (mx.random.normal((T, R + 1, K), key=mx.random.key(4)) * 3000).astype(dt)
        x64 = np.asarray(x.astype(mx.float32)).astype(np.float64)
        h64 = np.asarray(h.astype(mx.float32)).astype(np.float64)
        sr = {
            k: np.exp2(rng.uniform(np.log2(5e-5), np.log2(4.4e-4), (E, N)))
            for k in "gud"
        }
        ss = {
            k: np.exp2(rng.uniform(np.log2(5e-5), np.log2(4.4e-4), (1,)))
            for k in ("sg", "su", "sd")
        }

        def prod(xr, w, k, sr=sr):
            """xr [T, R, K] against expert idx: s * x @ W^T and s * |x| @ |W|^T."""
            p = np.einsum("trk,trnk->trn", xr, w[idx])
            b = np.einsum("trk,trnk->trn", np.abs(xr), np.abs(w[idx]))
            s = sr[k][idx]
            return s * p, s * b

        def sprod(xr, w, k, ss=ss):
            return ss[k] * (xr @ w.T), ss[k] * (np.abs(xr) @ np.abs(w).T)

        def silu_glu(g, gb, u, ub):
            sg = g / (1 + np.exp(-g))
            out = sg * u
            return out, np.abs(out) + 1.1 * gb * np.abs(u) + np.abs(sg) * ub

        xr = np.repeat(x64[:, None], R, axis=1)
        g, gbd = prod(xr, w64["g"], "g")
        u, ubd = prod(xr, w64["u"], "u")
        d, dbd = prod(h64[:, :R], w64["d"], "d")
        refs = {"glu": silu_glu(g, gbd, u, ubd)}
        gc = np.minimum(g + bias[0][idx], 7.0)
        uc = np.clip(u + bias[1][idx], -7.0, 7.0)
        gs = gc / (1 + np.exp(-1.702 * gc))
        refs["glu_bias"] = (
            gs * (uc + 1),
            np.abs(gs * (uc + 1)) + 1.1 * gbd * np.abs(uc + 1) + np.abs(gs) * ubd,
        )
        refs["qmv"] = (d, dbd)
        refs["qmv_bias"] = (d + bias[0][idx], dbd + np.abs(bias[0][idx]))
        sc3 = scores[:, :R, None]
        refs["mix_ns"] = ((sc3 * d).sum(1), (sc3 * dbd).sum(1))
        sg_, sgb = sprod(x64, s64["nvfp4"]["sg"], "sg")
        su_, sub = sprod(x64, s64["nvfp4"]["su"], "su")
        so, sob = silu_glu(sg_, sgb, su_, sub)
        refs["shexp_nvfp4"] = (
            np.concatenate([refs["glu"][0], so[:, None]], 1),
            np.concatenate([refs["glu"][1], sob[:, None]], 1),
        )
        sd_, sdb = sprod(h64[:, R], s64["nvfp4"]["sd"], "sd")
        refs["mix_nvfp4"] = (
            refs["mix_ns"][0] + scores[:, R, None] * sd_,
            refs["mix_ns"][1] + scores[:, R, None] * sdb,
        )
        allops = ops(x, h, a)
        for name, (want, bound) in refs.items():
            f, rk, sk, st = allops[name]
            wires = {**routed, **(shared[st] if st else {})}
            scales = {**{k: sr[k] for k in rk}, **{k: ss[k] for k in sk}}
            try:
                y = call(f, wires, {k: v.astype(np.float32) for k, v in scales.items()})
            except (RuntimeError, ValueError):
                continue
            got = np.asarray(y.astype(mx.float32)).astype(np.float64)
            what = f"{dtn}/{name}/real"
            if not np.isfinite(got).all():
                fails.append(
                    f"{what}: {int((~np.isfinite(got)).sum())} outputs not finite"
                )
                continue
            bad = np.abs(got - want) > TOL[dtn] * bound + FLOOR[dtn]
            if bad.any():
                fails.append(f"{what}: {int(bad.sum())} outputs past the bound")
            if dtn == "float16" and name in ("qmv", "glu", "mix_ns"):
                y0 = call(f, wires, {})
                if np.isfinite(np.asarray(y0.astype(mx.float32))).all():
                    fails.append(
                        f"{what}: the unscaled sums stay in range, nothing tested"
                    )
    print(json.dumps({"fails": fails, "ran": ran, "skipped": skipped}))


def _run(T, env):
    cenv = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    cenv.update({"KQ_MOE_QF": "0", **env}, KQ_MOE_NX_LOG="1")
    r = subprocess.run(
        [sys.executable, __file__, json.dumps({"T": T})],
        env=cenv,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    res["kernels"] = sorted(
        {ln.split()[1] for ln in r.stderr.splitlines() if ln.startswith("[kq_moe] ")}
    )
    return res


# (T, levers, kernel name parts that must have run). Each part names a form:
# the GLU gathers at NX 8, 16 and 32 and their dedupe form, the down gather,
# the slot-parallel, dedupe pair and loop mixes, the shared q8_0 stems, and
# the shared-expert epilogue of the dense route (parts starting kq_moe_).
# The arms run the Ext kernels unless they set KQ_MOE_QF=1.
ARMS = {
    "t1": (
        1,
        {},
        (
            "_moe_glu_gather_silu_nx16_",
            "_gather_qmv_mix_sp_",
            "_gather_qmv_mix_ns_sp_",
            "_moe_glu_gather_bias_swiglu_clamp",
            "nvfp4_sx_q8_0_gather_qmv_mix_sp_",
            "nvfp4_sx_q8_0_moe_glu_gather_shexp_",
        ),
    ),
    "t3": (3, {}, ("_moe_glu_gather_dd_silu", "_moe_glu_gather_shexp_dd_silu")),
    "t4": (
        4,
        {},
        (
            "_moe_glu_gather_dd_silu",
            "_moe_glu_gather_shexp_dd_silu",
            "kq_moe_shexp_glu_pre_silu_",
        ),
    ),
    "t4_fold": (
        4,
        {"KQ_MOE_SHEXP_DENSE": "0"},
        ("_moe_glu_gather_shexp_dd_silu", "nvfp4_sx_q8_0_moe_glu_gather_shexp_"),
    ),
    "t64": (64, {}, ("_moe_glu_gather_silu_nx16_", "_moe_glu_gather_shexp_silu_nx16_")),
    "nx8": (
        1,
        {"KQ_MOE_NX": "8"},
        ("_moe_glu_gather_silu_bfloat16_t", "_gather_qmv_float16_t"),
    ),
    "nx32": (4, {"KQ_MOE_NX": "32"}, ("_nx32_",)),
    "pair": (
        4,
        {"KQ_MOE_MIX_FORM": "dd"},
        ("_gather_qmv_mix_dd", "_gather_qmv_mix_ns_dd"),
    ),
    "loop": (
        4,
        {"KQ_MOE_MIX_FORM": "loop", "KQ_MOE_DEDUP": "0"},
        (
            "_gather_qmv_mix_bfloat16_t",
            "_gather_qmv_mix_ns_bfloat16_t",
            "_moe_glu_gather_silu_nx16_",
        ),
    ),
    "qf1": (
        1,
        {"KQ_MOE_QF": "1"},
        (
            "_qf_moe_glu_gather_silu_float16_t",
            "_qf_moe_glu_gather_bias_swiglu_clamp",
            "_qf_sx_nvfp4_gather_qmv_mix_sp_",
            "_qf_sx_q8_0_gather_qmv_mix_sp_",
            "_qf_sx_q8_0_moe_glu_gather_shexp_",
        ),
    ),
    "qf4": (
        4,
        {"KQ_MOE_QF": "1"},
        (
            "_qf_moe_glu_gather_dd_silu",
            "_qf_gather_qmv_float16_t",
            "_qf_gather_qmv_bias_",
            "_qf_gather_qmv_mix_ns_sp_",
            "_qf_sx_nvfp4_moe_glu_gather_shexp_silu",
        ),
    ),
    "qf_loop": (
        4,
        {"KQ_MOE_QF": "1", "KQ_MOE_MIX_FORM": "loop", "KQ_MOE_DEDUP": "0"},
        (
            "_qf_sx_nvfp4_gather_qmv_mix_bfloat16_t",
            "_qf_gather_qmv_mix_ns_bfloat16_t",
            "_qf_moe_glu_gather_silu_bfloat16_t",
        ),
    ),
}


@pytest.mark.skipif(FORCE_CPU, reason="the fused MoE ops are Metal-only")
@pytest.mark.parametrize("arm", list(ARMS))
def test_scaled_forms(arm):
    T, env, parts = ARMS[arm]
    res = _run(T, env)
    assert not res["fails"], res["fails"]
    assert not res["skipped"], res["skipped"]
    nv = [k for k in res["kernels"] if k.startswith("kq_nvfp4")]
    for p in parts:
        pool = res["kernels"] if p.startswith("kq_moe_") else nv
        assert any(p in k for k in pool), (p, pool)
    # the dense shared-expert route runs from 4 rows by default
    if env.get("KQ_MOE_SHEXP_DENSE") == "0" or T < 4:
        assert not any("shexp_glu_pre" in k for k in res["kernels"]), res["kernels"]


def test_rejects():
    import mlx.core as mx

    import mlx_kquant as kq

    w = mx.array(_nv((E, N), 1))
    sw = mx.array(_nv((N,), 4))
    q8 = mx.array(_q8((N,), 4))
    x = mx.zeros((2, K), mx.bfloat16)
    h = mx.zeros((2, R + 1, K), mx.bfloat16)
    ix = mx.zeros((2, R), mx.uint32)
    sc = mx.ones((2, R + 1))
    one, col = mx.ones((1,)), mx.ones((N,))
    qk = mx.zeros((E, N, (K // 256) * 144), mx.uint8)
    cases = [
        (
            lambda: kq.moe_glu_gather_kq(x, w, w, "nvfp4", ix, gate_scale=one),
            "go together",
        ),
        (
            lambda: kq.moe_glu_gather_kq(
                x, w, w, "nvfp4", ix, gate_scale=one, up_scale=col
            ),
            "go together",
        ),
        (
            lambda: kq.moe_glu_gather_kq(
                x, qk, qk, "q4_k", ix, gate_scale=one, up_scale=one
            ),
            "for nvfp4",
        ),
        (
            lambda: kq.gather_qmv_kq(
                h[:, :R], w, "nvfp4", ix, out_scale=one.astype(mx.float16)
            ),
            "float32",
        ),
        (
            lambda: kq.gather_qmv_kq(
                h[:, :R], w, "nvfp4", ix, out_scale=mx.ones((E + 1, 1))
            ),
            "(8, 64)",
        ),
        (
            lambda: kq.gather_qmv_mix_ns_kq(
                h[:, :R], w, "nvfp4", ix, sc[:, :R], out_scale=mx.ones((N + 1,))
            ),
            "shape",
        ),
        (
            lambda: kq.gather_qmv_mix_kq(
                h, w, q8, "nvfp4", ix, sc, shexp_kquant_type="q8_0", shexp_scale=one
            ),
            "nvfp4 shared-expert",
        ),
        (
            lambda: kq.gather_qmv_mix_kq(h, w, sw, "nvfp4", ix, sc, shexp_scale=col),
            "shape (1,)",
        ),
        (
            lambda: kq.moe_glu_gather_shexp_kq(
                x, w, w, sw, sw, "nvfp4", ix, shexp_gate_scale=one
            ),
            "go together",
        ),
    ]
    for f, msg in cases:
        with pytest.raises(
            ValueError, match=msg.replace("(", r"\(").replace(")", r"\)")
        ):
            f()


if __name__ == "__main__":
    _child(json.loads(sys.argv[1]))
