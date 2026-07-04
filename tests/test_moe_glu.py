#!/usr/bin/env python3
"""Fused MoE gather validation across the full codec matrix.

For every codec with fused MoE kernels, builds E expert stacks (K-quants from
tests/fixtures/<codec>_moe.npz, flat codecs synthesized via
gguf.quants.quantize, IQ codecs as structurally-valid random wire) and checks
each fused op against an independent f32 reference computed from
gguf.quants.dequantize:

    moe_glu_gather_kq        act(x @ G_e.T) * (x @ U_e.T)
    gather_qmv_kq            h_s @ D_e.T
    moe_glu_gather_shexp_kq  routed slots + shared-expert slot (last)
    gather_qmv_mix_kq        sum_s score_s * (h_s @ D_e.T)
    gather_qmv_mix_ns_kq     same, no shared expert (all S slots routed)
    moe_router_topk          f32 softmax + top-k + norm + shared-gate sigmoid
                             (+ shared_gate=False / per_expert_scale arms)

The q8_0 row also runs at K=352 (K % 256 != 0) to cover the generic-uniform
q8_0_ext dispatch stem (gemma-4-a4b down-proj geometry class).

Mixed-codec shared experts (the UD-style upcast) are covered by pairing every
expert codec with a q8_0 shexp, plus the q8_0-experts/q6_k-shexp reverse.

The kernels are Metal-only (eval_cpu throws), so the module is skipped under
KQUANT_FORCE_CPU.

Usage: test_moe_glu.py [--codecs q4_k,q8_0,...]
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType as GT
from gguf import quants

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="fused MoE gathers are Metal-only kernels; no CPU path.",
)

# codec -> (gguf type, weights_per_block, bytes_per_block, is_kquant)
CODECS = {
    "q4_0": (GT.Q4_0, 32, 18, False),
    "q4_1": (GT.Q4_1, 32, 20, False),
    "q5_0": (GT.Q5_0, 32, 22, False),
    "q5_1": (GT.Q5_1, 32, 24, False),
    "q8_0": (GT.Q8_0, 32, 34, False),
    "q2_k": (GT.Q2_K, 256, 84, True),
    "q3_k": (GT.Q3_K, 256, 110, True),
    "q4_k": (GT.Q4_K, 256, 144, True),
    "q5_k": (GT.Q5_K, 256, 176, True),
    "q6_k": (GT.Q6_K, 256, 210, True),
    "iq4_nl": (GT.IQ4_NL, 32, 18, False),
    "iq4_xs": (GT.IQ4_XS, 256, 136, False),
    "iq3_s": (GT.IQ3_S, 256, 110, False),
    "iq3_xxs": (GT.IQ3_XXS, 256, 98, False),
    "iq2_xxs": (GT.IQ2_XXS, 256, 66, False),
    "iq2_xs": (GT.IQ2_XS, 256, 74, False),
    "iq2_s": (GT.IQ2_S, 256, 82, False),
    "iq1_s": (GT.IQ1_S, 256, 50, False),
    "iq1_m": (GT.IQ1_M, 256, 56, False),
}

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
E, N, K = 4, 128, 512  # matches the committed *_moe.npz fixtures
T, R = 3, 2  # tokens, routed slots (S = R + 1 with the shared expert)
REL_BOUND = 2e-3  # dequant is bit-exact; residual is f32 reorder + f16 x


def _synth_iq_wire(rng, bpb, n_blocks):
    """Structurally-valid random IQ wire (gguf-py is decode-only for IQ):
    random bytes with a sane fp16 d at block offset 0 (IQ1_M: scale nibbles)
    so dequant can't hit Inf/NaN."""
    wire = rng.integers(0, 256, size=(n_blocks, bpb), dtype=np.uint8)
    # Keep d small: random scale/grid bits already reach the codec's max
    # magnitude, and the GLU product act(g) * u must stay inside f16 range
    # (iq4_xs random wire hits |w| ~ 80 at d = 0.02).
    d = rng.uniform(0.004, 0.01, n_blocks).astype(np.float16)
    if bpb == 56:
        dbits = d.view(np.uint16)
        for k, byteidx in enumerate((49, 51, 53, 55)):
            nib = ((dbits >> (4 * k)) & 0xF).astype(np.uint8)
            wire[:, byteidx] = (wire[:, byteidx] & 0x0F) | (nib << 4)
    else:
        wire[:, 0:2] = d.view(np.uint8).reshape(n_blocks, 2)
    return wire


def _wire_and_ref(codec, seed=11, k=K):
    """Return (wire uint8[E, N, packed], ref float32[E, N, k]) or (None, None)
    when a K-quant fixture is missing. k only varies for synthesized codecs."""
    gtype, wpb, bpb, is_kq = CODECS[codec]
    if codec.startswith("iq"):
        rng = np.random.default_rng(seed)
        wires = [
            _synth_iq_wire(rng, bpb, N * (k // wpb)).reshape(N, (k // wpb) * bpb)
            for _ in range(E)
        ]
        refs = [quants.dequantize(np.ascontiguousarray(w), gtype) for w in wires]
        return np.stack(wires, 0), np.stack(refs, 0).astype(np.float32)
    if is_kq:
        path = os.path.join(FIX, f"{codec}_moe.npz")
        if not os.path.exists(path):
            return None, None
        z = np.load(path)
        wire = z["wire"].astype(np.uint8)
        refs = [
            quants.dequantize(np.ascontiguousarray(wire[e]), gtype)
            for e in range(wire.shape[0])
        ]
        return wire, np.stack(refs, 0).astype(np.float32)
    rng = np.random.default_rng(seed)
    wires, refs = [], []
    for _ in range(E):
        we = rng.standard_normal((N, k)).astype(np.float32) * 0.1
        wq = quants.quantize(we, gtype).astype(np.uint8)
        wires.append(wq)
        refs.append(quants.dequantize(np.ascontiguousarray(wq), gtype))
    return np.stack(wires, 0), np.stack(refs, 0).astype(np.float32)


def _act_np(g, act):
    if act == "gelu":
        return 0.5 * g * (1.0 + np.tanh(0.7978845608028654 * (g + 0.044715 * g**3)))
    with np.errstate(over="ignore"):  # exp(-g) -> inf for very negative g is fine
        return g / (1.0 + np.exp(-g))


def _rel(got, ref):
    g = np.array(got.astype(mx.float32))
    return float(np.linalg.norm(g - ref) / (np.linalg.norm(ref) + 1e-6))


def _check_codec(codec, sx=None, act="silu", dtype=mx.float16, k=K):
    """Run all five gather ops for one (expert codec, shexp codec) pair.
    Returns list of (name, rel, ok)."""
    scodec = sx or codec
    wire, ref = _wire_and_ref(codec, k=k)
    if wire is None:
        return None
    if scodec == codec:
        swire, sref = wire, ref
    else:
        swire, sref = _wire_and_ref(scodec, seed=13, k=k)
        if swire is None:
            return None
    rng = np.random.default_rng(0)
    dw = mx.array(wire)  # full [E, N, packed] stack (gate / down / qmv)
    # shared-expert tensors: single-expert 2-D [N, packed] matrices
    sgw, suw, sdw = mx.array(swire[2]), mx.array(swire[3]), mx.array(swire[1])
    sg_ref, su_ref, sd_ref = sref[2], sref[3], sref[1]

    x_np = (rng.standard_normal((T, k)) * 0.1).astype(np.float16)
    x = mx.array(x_np).astype(dtype)
    xf = np.array(x.astype(mx.float32))  # exactly what the kernel reads
    inds_np = rng.integers(0, E, size=(T, R)).astype(np.uint32)
    inds = mx.array(inds_np)
    out = []

    # moe_glu_gather_kq: gate/up must be stacks -> reuse dw as both (gate) and
    # a second stack (up) built from the same wires reversed for asymmetry.
    up_stack_np = wire[::-1].copy()
    up_stack = mx.array(up_stack_np)
    up_ref = ref[::-1].copy()
    got = kq.moe_glu_gather_kq(x, dw, up_stack, codec, inds, act=act)
    mx.eval(got)
    r = np.stack(
        [
            np.stack(
                [
                    _act_np(xf[t] @ ref[e].T, act) * (xf[t] @ up_ref[e].T)
                    for e in inds_np[t]
                ],
                0,
            )
            for t in range(T)
        ],
        0,
    )
    rel = _rel(got, r)
    out.append(("glu", rel, rel < REL_BOUND))

    # gather_qmv_kq
    h_np = (rng.standard_normal((T, R, k)) * 0.1).astype(np.float16)
    h = mx.array(h_np).astype(dtype)
    hf = np.array(h.astype(mx.float32))
    got = kq.gather_qmv_kq(h, dw, codec, inds)
    mx.eval(got)
    r = np.stack(
        [
            np.stack([hf[t, s] @ ref[inds_np[t, s]].T for s in range(R)], 0)
            for t in range(T)
        ],
        0,
    )
    rel = _rel(got, r)
    out.append(("qmv", rel, rel < REL_BOUND))

    # moe_glu_gather_shexp_kq (shexp codec = scodec)
    got = kq.moe_glu_gather_shexp_kq(
        x,
        dw,
        up_stack,
        sgw,
        suw,
        codec,
        inds,
        act=act,
        shexp_kquant_type=("" if scodec == codec else scodec),
    )
    mx.eval(got)
    routed = np.stack(
        [
            np.stack(
                [
                    _act_np(xf[t] @ ref[e].T, act) * (xf[t] @ up_ref[e].T)
                    for e in inds_np[t]
                ],
                0,
            )
            for t in range(T)
        ],
        0,
    )
    shared = np.stack(
        [_act_np(xf[t] @ sg_ref.T, act) * (xf[t] @ su_ref.T) for t in range(T)], 0
    )
    r = np.concatenate([routed, shared[:, None, :]], axis=1)
    rel = _rel(got, r)
    out.append(("shexp", rel, rel < REL_BOUND))

    # gather_qmv_mix_kq (shexp codec = scodec)
    S = R + 1
    hs_np = (rng.standard_normal((T, S, k)) * 0.1).astype(np.float16)
    hs = mx.array(hs_np).astype(dtype)
    hsf = np.array(hs.astype(mx.float32))
    sc_np = rng.uniform(0.05, 0.9, size=(T, S)).astype(np.float32)
    sc = mx.array(sc_np)
    got = kq.gather_qmv_mix_kq(
        hs,
        dw,
        sdw,
        codec,
        inds,
        sc,
        shexp_kquant_type=("" if scodec == codec else scodec),
    )
    mx.eval(got)
    r = np.stack(
        [
            sum(sc_np[t, s] * (hsf[t, s] @ ref[inds_np[t, s]].T) for s in range(R))
            + sc_np[t, R] * (hsf[t, R] @ sd_ref.T)
            for t in range(T)
        ],
        0,
    )
    rel = _rel(got, r)
    out.append(("mix", rel, rel < REL_BOUND))

    # gather_qmv_mix_ns_kq: all S slots routed (gemma-style, no shared expert)
    inds_ns_np = rng.integers(0, E, size=(T, S)).astype(np.uint32)
    got = kq.gather_qmv_mix_ns_kq(hs, dw, codec, mx.array(inds_ns_np), sc)
    mx.eval(got)
    r = np.stack(
        [
            sum(sc_np[t, s] * (hsf[t, s] @ ref[inds_ns_np[t, s]].T) for s in range(S))
            for t in range(T)
        ],
        0,
    )
    rel = _rel(got, r)
    out.append(("mix_ns", rel, rel < REL_BOUND))
    return out


def _check_router():
    rng = np.random.default_rng(5)
    fails = []
    for e, r, norm in ((256, 8, True), (256, 8, False), (128, 4, True), (16, 2, True)):
        logits_np = rng.standard_normal((5, e + 1)).astype(np.float32)
        inds, sc = kq.moe_router_topk(mx.array(logits_np), r, norm)
        mx.eval(inds, sc)
        p = np.exp(logits_np[:, :e] - logits_np[:, :e].max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        top = np.argsort(-p, axis=-1, kind="stable")[:, :r]
        got_i = np.array(inds)
        for t in range(5):
            if set(got_i[t]) != set(top[t]):
                fails.append(f"E={e} R={r} t={t} indices")
                continue
            pk = p[t, got_i[t]]
            want = pk / (pk.sum() if norm else 1.0)
            if not np.allclose(np.array(sc)[t, :r], want, rtol=1e-5, atol=1e-6):
                fails.append(f"E={e} R={r} t={t} scores")
            sig = 1.0 / (1.0 + np.exp(-logits_np[t, e]))
            if abs(float(np.array(sc)[t, r]) - sig) > 1e-5:
                fails.append(f"E={e} R={r} t={t} shared gate")
    # shared_gate=False + per_expert_scale (gemma routing epilogue)
    for e, r in ((128, 8), (16, 4)):
        logits_np = rng.standard_normal((5, e)).astype(np.float32)
        pes_np = rng.uniform(0.5, 1.5, e).astype(np.float32)
        inds, sc = kq.moe_router_topk(
            mx.array(logits_np),
            r,
            True,
            shared_gate=False,
            per_expert_scale=mx.array(pes_np),
        )
        mx.eval(inds, sc)
        if sc.shape != (5, r):
            fails.append(f"E={e} R={r} no-shared scores shape {sc.shape}")
            continue
        p = np.exp(logits_np - logits_np.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        top = np.argsort(-p, axis=-1, kind="stable")[:, :r]
        got_i = np.array(inds)
        for t in range(5):
            if set(got_i[t]) != set(top[t]):
                fails.append(f"E={e} R={r} t={t} pes indices")
                continue
            pk = p[t, got_i[t]]
            want = pk / pk.sum() * pes_np[got_i[t]]
            if not np.allclose(np.array(sc)[t], want, rtol=1e-5, atol=1e-6):
                fails.append(f"E={e} R={r} t={t} pes scores")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", default="")
    args = ap.parse_args(argv)
    allow = {c.strip() for c in args.codecs.split(",") if c.strip()}

    print(f"=== test_moe_glu  E={E} N={N} K={K} T={T} R={R} ===")
    print(
        f"  {'codec':<10} {'shexp':<8} {'act':<5} {'glu':>10} {'qmv':>10} "
        f"{'shexp':>10} {'mix':>10} {'mix_ns':>10} {'verdict':>8}"
    )
    fails, missing = 0, []

    def run(codec, sx=None, act="silu", dtype=mx.float16, k=K):
        nonlocal fails
        res = _check_codec(codec, sx=sx, act=act, dtype=dtype, k=k)
        if res is None:
            missing.append(codec if sx is None else f"{codec}+{sx}")
            fails += 1
            print(f"  {codec:<10} {'MISSING fixture - run gen_fixtures.py':>50}")
            return
        bad = not all(ok for _n, _r, ok in res)
        fails += bad
        cells = " ".join(f"{r:>10.3e}" for _n, r, _ok in res)
        label = codec if k == K else f"{codec}/{k}"
        print(
            f"  {label:<10} {(sx or '-'):<8} {act:<5} {cells} "
            f"{'FAIL' if bad else 'ok':>8}"
        )

    for codec in CODECS:
        if allow and codec not in allow:
            continue
        run(codec)  # uniform, silu, f16
        if codec != "q8_0":
            run(codec, sx="q8_0")  # UD-style upcast shexp
    if not allow or "q8_0" in allow:
        run("q8_0", sx="q6_k")  # reverse mixed combo
        run("q8_0", k=352)  # K % 256 != 0 -> q8_0_ext generic-uniform stem
    if not allow or "q4_k" in allow:
        run("q4_k", act="gelu")  # gelu epilogue (gemma GeGLU)
        run("q4_k", dtype=mx.bfloat16)  # bf16 activations

    router_fails = _check_router()
    for f in router_fails:
        print(f"  router FAIL: {f}")
    fails += len(router_fails)
    print("  router ok" if not router_fails else "")

    if missing:
        print(f"\nMISSING fixtures (FAIL): {', '.join(missing)}")
    print(f"{'FAILURES: ' + str(fails) if fails else 'ALL OK'}")
    return 1 if fails else 0


def test_moe_glu():
    """pytest entry: runs the full codec-matrix sweep."""
    assert main([]) == 0


if __name__ == "__main__":
    sys.exit(main())
