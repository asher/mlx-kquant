"""Operands at an element offset into their buffer.

A row-contiguous view can start at any element of its buffer, and an op's
contiguous node passes it on in place. Kernels load activations, caches and
some float operands as vectors, and Metal defines a vector load only at an
address aligned to its size. So each op binds such an operand that starts
off the boundary of its kernel's widest load from a copy, and a strided
operand whose rows step off it too; the matmul ops bind x from a 16-byte
boundary. Each case runs an op on views at offsets of 1 to 8 elements and
requires the output bits of the same values in their own buffer.
KQ_QMM_LOG prints a "[kq align] <op> <operand> xin=<n>" line for each copy,
and every matmul dispatch logs x on a 16-byte boundary (xa=0), but
verify_nax's: it reads x where it lands on an 8-byte boundary (xa=8) and
from the copy otherwise. The cases require all three: each view that
starts off the boundary reaches its op there and is copied or read on its
kernel's boundary, each view on the boundary is read in place, and no
matmul kernel reads x off it.

- quantized_matmul for every codec with a matmul kernel, at 1 to 64 rows,
  on the default routes, with the NAX mat-vec off (KQ_NAX_QMV=0), with NAX
  off (KQ_DISABLE_NAX=1, the routes of a GPU without NAX), and on each
  forced route (KQ_QMM_ROUTE);
- the transpose=False, GLU, LoRA (also with the fused LoRA kernel,
  KQ_QMM_LORA=2), residual and bias matmuls, skinny_matmul, gather_qmm,
  gather_qmm_seg, the fused MoE gathers and gather_mix, in a child per route
  environment;
- rmsnorm_gate, latent_fp4_pack, dsa_indexer_score_decode, the hc glue ops,
  the attention ops and the sparse attention ops, in a child with NAX on and
  off, each operand read as vectors at the offsets and, for the keys and
  values the attention ops read through their strides, as rows that step
  one element past their length.

Float32 activations of the matmul ops are cast to bfloat16 into a new
buffer, so their views reach the kernels aligned. Under KQUANT_FORCE_CPU
the quantized_matmul cases run on the CPU stream, which needs no alignment;
the other ops are Metal-only kernels with no CPU eval, and are skipped.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest
from kqref import is_synth, synth_wire
from qmm_probe import TESTS_DIR, force_cpu, run_probe

import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

OFFSETS = tuple(range(1, 9))
DTYPES = ("float16", "bfloat16", "float32")
ROWS = (1, 2, 4, 8, 12, 16, 32, 64)
STRICT = {"KQ_QMM_ROUTE_STRICT": "1"}
# (name, env, rows). The forced routes run at the rows they serve; a route
# that declines a codec or shape raises, on the offset views too.
ROUTES = [
    ("default", {}, ROWS),
    ("nax_qmv_off", {"KQ_NAX_QMV": "0"}, ROWS),
    ("nax_off", {"KQ_DISABLE_NAX": "1"}, ROWS),
    ("qmv", {"KQ_QMM_ROUTE": "qmv", **STRICT}, (1, 2, 8)),
    ("verify_qmv", {"KQ_QMM_ROUTE": "verify_qmv", **STRICT}, (2, 8)),
    ("mv_ext", {"KQ_QMM_ROUTE": "mv_ext", **STRICT}, (2, 8, 12)),
    ("verify_mma", {"KQ_QMM_ROUTE": "verify_mma", **STRICT}, (2, 8)),
    ("verify_nax", {"KQ_QMM_ROUTE": "verify_nax", **STRICT}, (2, 8)),
    ("splitk", {"KQ_QMM_ROUTE": "splitk", **STRICT}, (8, 32)),
    ("nax", {"KQ_QMM_ROUTE": "nax", **STRICT}, (8, 32)),
    ("nax_splitk", {"KQ_QMM_ROUTE": "nax_splitk", **STRICT}, (8, 32)),
]
CPU_ROUTES = [("default", {}, ROWS)]
MATMUL_CODECS = [c for c in kq.codecs() if kq.codec_has_matmul(c)]
ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4}


def _unaligned_x(name, xa):
    """Whether a kernel read x off its boundary: 16 bytes, or 8 for
    verify_nax, which reads x where it lands on an 8-byte boundary."""
    return xa % (8 if "_verify_nax_" in name else 16) != 0


metal_only = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="Metal-only kernel; no CPU path.",
)


def quantized_matmul_offsets(codec):
    """(bad, unaligned, unreached, copied) over the routes: the offset calls
    whose output or error differs from the call on x in its own buffer, the
    kernels that read x off a 16-byte boundary with the byte offset they
    logged, the calls on a view off the boundary that log neither a copy of
    x nor a kernel reading x there, and the calls on a view on the boundary
    that log a copy."""
    bad, unaligned, unreached, copied = [], set(), [], []
    for route, env, rows in CPU_ROUTES if force_cpu() else ROUTES:
        calls, cells = [], []
        for dtype in DTYPES:
            # float32 activations are cast to bfloat16 into a new buffer:
            # the default route covers them.
            if dtype == "float32" and route != "default":
                continue
            for m in rows:
                for off in (0, *OFFSETS):
                    calls.append(
                        {
                            "codec": codec,
                            "shape": [m, 4096],
                            "n": 256,
                            "dtype": dtype,
                            "seed": m,
                            "x_offset": off,
                        }
                    )
                    cells.append((dtype, m, off))
        res = run_probe(calls, env={"KQ_QMM_LOG": "1", **env}, check=False)
        base = {}
        for (dtype, m, off), r in zip(cells, res, strict=True):
            got = r["error"] or r["digest"]
            names = [k for k, _ in r["kernels"]]
            if off == 0:
                base[(dtype, m)] = got
            elif got != base[(dtype, m)]:
                bad.append((route, dtype, m, off, names, r["xa"]))
            for name, xa in zip(names, r["xa"], strict=True):
                if _unaligned_x(name, xa):
                    unaligned.add((name, xa))
            off_boundary = (off * ITEMSIZE[dtype]) % 16 != 0
            if not off_boundary and r["copies"]:
                copied.append((route, dtype, m, off, r["copies"]))
            if (
                off_boundary
                and dtype != "float32"
                and not force_cpu()
                and not r["error"]
                and not any(xin for _, xin in r["copies"])
                and not any(r["xa"])
            ):
                unreached.append((route, dtype, m, off, names))
    return bad, sorted(unaligned), unreached, copied


@pytest.mark.parametrize("codec", MATMUL_CODECS)
def test_quantized_matmul_x_offsets(codec):
    """Bit-exact output at every offset, every kernel reads x from a
    16-byte boundary, every view off it reaches the op there, and every
    view on it is read in place."""
    bad, unaligned, unreached, copied = quantized_matmul_offsets(codec)
    assert bad == [], bad
    assert unaligned == [], unaligned
    assert unreached == [], unreached
    assert copied == [], copied


# -------------------------------- other ops --------------------------------
# Each group runs in a child interpreter per environment, since the route
# levers are read once per process. The child logs its dispatches and
# copies, and marks each offset call on stderr with the copies it needs.

ENVS = {
    "default": {},
    "nax_qmv_off": {"KQ_NAX_QMV": "0"},
    "nax_off": {"KQ_DISABLE_NAX": "1"},
    "lora2": {"KQ_QMM_LORA": "2"},
}
K, N = 768, 256
OP_CODECS = ("q4_0", "q4_k", "q6_k", "q8_0", "iq4_nl", "iq2_xxs")
MOE_CODECS = ("q4_k", "q8_0", "iq4_nl", "iq2_xxs")
MX_DT = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
HALF = ("float16", "bfloat16")
_W = {}


def _own_buffer(make):
    """make() evaluated into a buffer of its own size. A buffer reused from
    the cache can run more than 16 KiB past the view, and the op's
    contiguous node then copies the view before the eval sees it; buffers
    freed late by earlier work could refill an emptied cache, so the cache
    stays off while the buffer is allocated."""
    limit = mx.set_cache_limit(0)
    mx.clear_cache()
    try:
        a = make()
        mx.eval(a)
    finally:
        mx.set_cache_limit(limit)
    return a


def _at_offset(x, off):
    """The values of x as a row-contiguous view off elements into a buffer
    of their own."""
    flat = _own_buffer(
        lambda: mx.concatenate([mx.zeros((off,), x.dtype), x.reshape(-1)])
    )
    return flat[off:].reshape(x.shape)


def _rows_view(x):
    """The values of x in rows that step one element past their length, so
    that every row after the first starts off a 16-byte boundary."""
    pad = mx.zeros((*x.shape[:-1], 1), x.dtype)
    return _own_buffer(lambda: mx.concatenate([x, pad], axis=-1))[..., :-1]


def _mark(label, need, names=()):
    """need: "off" when the call must copy each operand in names (or, with
    no names, log a copy or a kernel reading x) at a nonzero offset, "copy"
    when it must copy each of them at any offset, "none" when it must copy
    no operand, "any" for no requirement."""
    print(f"[arm] {label} | {need} | {','.join(names)}", file=sys.stderr, flush=True)


def _listify(out):
    return list(out) if isinstance(out, (list, tuple)) else [out]


def _same(got, want):
    return all(
        g.dtype == w.dtype and mx.array_equal(g, w).item()
        for g, w in zip(got, want, strict=True)
    )


def _x(shape, dtype, seed):
    x = (mx.random.normal(shape, key=mx.random.key(seed)) * 0.5).astype(MX_DT[dtype])
    mx.eval(x)
    return x


def _weight(codec, n, k, seed=0, experts=None):
    """Weights [n, k], or [experts, n, k], and scales. The codecs that
    quantize only with an importance matrix get random valid wire."""
    key = (codec, n, k, seed, experts)
    if key not in _W:
        e = experts or 1
        if is_synth(codec):
            _, _, bpb, wpb = CODEC_GEOMETRY[codec]
            rng = np.random.default_rng(seed * 131 + n + k)
            wire = synth_wire(rng, codec, bpb, e * n * (k // wpb))
            w = mx.array(wire.reshape(e, n, (k // wpb) * bpb))
            w = w if experts else w[0]
            s = mx.zeros((1,), dtype=mx.uint8)
        else:
            shape = (n, k) if experts is None else (experts, n, k)
            wf = mx.random.normal(shape, key=mx.random.key(seed)) * 0.1
            w, s = kq.quantize(wf, codec)
        w = mx.contiguous(w)
        mx.eval(w, s)
        _W[key] = (w, s)
    return _W[key]


def _offset_failures(call, shape, dtype, seed=0, label="", names=()):
    """The offsets at which call(view) differs from call(x) in its own
    buffer. A call the op refuses is skipped. A float32 view is cast into a
    new buffer, so only the 16-bit views must reach the op off the
    boundary; every view on it must reach the op uncopied."""
    x = _x(shape, dtype, seed)
    try:
        want = _listify(call(x))
        mx.eval(*want)
    except (ValueError, RuntimeError):
        return []
    bad = []
    for off in OFFSETS:
        view = _at_offset(x, off)
        need = "none"
        if (off * ITEMSIZE[dtype]) % 16:
            need = "off" if dtype in HALF else "any"
        _mark(f"{label} {dtype} {list(shape)} off{off}", need, names)
        got = _listify(call(view))
        mx.eval(*got)
        if not _same(got, want):
            bad.append(off)
    return bad


def _operand_failures(label, call, ops, cases):
    """call(**ops) with one operand at a time replaced by its views. cases:
    (operand, the copies it needs, vec, strided), vec the elements of the
    kernel's widest vector load of the operand. Each view at an offset off
    a boundary of vec elements must be copied, and each view on one must
    not; a strided operand also runs as rows that step one element past
    that boundary, which must be copied too."""
    want = _listify(call(**ops))
    mx.eval(*want)
    bad = []
    for name, names, vec, strided in cases:
        a = ops[name]
        arms = [(f"off{off}", off, "off" if off % vec else "none") for off in OFFSETS]
        if strided:
            arms.append(("rows", None, "copy"))
        for arm, off, need in arms:
            view = _rows_view(a) if off is None else _at_offset(a, off)
            _mark(f"{label} {a.dtype} {name} {arm}", need, names)
            got = _listify(call(**{**ops, name: view}))
            mx.eval(*got)
            if not _same(got, want):
                bad.append([label, str(a.dtype), name, arm])
    return bad


def _matmul_ops():
    """transpose=False, GLU, LoRA and residual matmuls."""
    bad = []
    a = (mx.random.normal((K, 8), key=mx.random.key(5)) / np.sqrt(K)).astype(
        mx.bfloat16
    )
    b = (mx.random.normal((8, N), key=mx.random.key(6)) * 0.3).astype(mx.bfloat16)
    mx.eval(a, b)
    for codec in OP_CODECS:
        w, _ = _weight(codec, N, K)
        wg, _ = _weight(codec, N, K, 1)
        wt, st = _weight(codec, K, N, 2)

        def resid(x, w=w, codec=codec):
            out = mx.bfloat16 if x.dtype == mx.float32 else x.dtype
            r = (mx.random.normal((x.shape[0], N), key=mx.random.key(7)) * 0.5).astype(
                out
            )
            return kq.quantized_matmul_residual(x, w, codec, r)

        def lora(x, w=w, codec=codec):
            dt = mx.bfloat16 if x.dtype == mx.float32 else x.dtype
            return kq.quantized_matmul_lora(x, w, codec, a.astype(dt), b.astype(dt))

        def glu(x, w=w, wg=wg, codec=codec):
            return kq.quantized_matmul_glu(x, wg, w, codec)

        def matmul_n(x, wt=wt, st=st, codec=codec):
            return kq.quantized_matmul(x, wt, st, codec, False)

        ops = [
            ("matmul_n", (1, 8, 64), matmul_n),
            ("glu", (1, 8, 64, 1024), glu),
            ("lora", (1, 8, 64, 1024), lora),
            ("residual", (1, 8, 64, 1024), resid),
        ]
        for name, rows, call in ops:
            for dtype in DTYPES:
                for m in rows:
                    offs = _offset_failures(call, (m, K), dtype, m, f"{name} {codec}")
                    if offs:
                        bad.append([name, codec, dtype, m, offs])
    return bad


def _bias_and_skinny():
    """The q8_0 bias mat-vec and skinny_matmul, whose dense weight is read
    as vectors too."""
    bad = []
    w, s = _weight("q8_0", N, K)
    bias = mx.random.normal((N,), key=mx.random.key(3)).astype(mx.float32)
    for dtype in DTYPES:
        offs = _offset_failures(
            lambda x: kq.quantized_matmul_qmv_bias(x, w, s, bias, "q8_0"),
            (1, K),
            dtype,
            label="qmv_bias",
        )
        if offs:
            bad.append(["qmv_bias", dtype, 1, offs])
        wd = (mx.random.normal((64, K), key=mx.random.key(4)) * 0.5).astype(
            MX_DT[dtype]
        )
        mx.eval(wd)
        for m in (1, 4, 16):
            offs = _offset_failures(
                lambda x, wd=wd: kq.skinny_matmul(x, wd), (m, K), dtype, label="skinny"
            )
            if offs:
                bad.append(["skinny", dtype, m, offs])
            if dtype in HALF:
                x = _x((m, K), dtype, m)
                bad += _operand_failures(
                    f"skinny w m{m}",
                    lambda x, w: kq.skinny_matmul(x, w),
                    {"x": x, "w": wd},
                    [("w", ("skinny_matmul w",), 8, False)],
                )
    return bad


def _gathers():
    """gather_qmm: the mat-vec, the tiled gather, the sorted per-expert
    GEMM and transpose=False; and gather_qmm_seg."""
    experts, bad = 4, []
    rng = np.random.default_rng(4)
    for codec in MOE_CODECS:
        wt, st = _weight(codec, N, K, 3, experts)
        wn, sn = _weight(codec, K, N, 4, experts)
        for rows, m, transpose, sort in [
            (8, 1, True, False),
            (4, 32, True, False),
            (256, 1, True, True),
            (8, 1, False, False),
        ]:
            idx = rng.integers(0, experts, rows).astype(np.uint32)
            rhs = mx.array(np.sort(idx) if sort else idx)
            w, s = (wt, st) if transpose else (wn, sn)

            def call(x, w=w, s=s, rhs=rhs, transpose=transpose, sort=sort, c=codec):
                return kq.gather_qmm(
                    x,
                    w,
                    s,
                    c,
                    rhs_indices=rhs,
                    transpose=transpose,
                    sorted_indices=sort,
                )

            for dtype in DTYPES:
                offs = _offset_failures(
                    call, (rows, m, K), dtype, label=f"gather_qmm {codec}"
                )
                if offs:
                    bad.append(["gather_qmm", codec, dtype, rows, m, transpose, offs])
        idx = np.sort(rng.integers(0, experts, 200).astype(np.uint32))
        maps = kq.expert_tile_map(mx.array(idx), experts)
        for dtype in DTYPES:
            offs = _offset_failures(
                lambda x, wt=wt, st=st, c=codec, mp=maps: kq.gather_qmm_seg(
                    x, wt, st, c, *mp
                ),
                (len(idx), K),
                dtype,
                label=f"gather_qmm_seg {codec}",
            )
            if offs:
                bad.append(["gather_qmm_seg", codec, dtype, len(idx), offs])
    return bad


T, R = 3, 2
S = R + 1
ME, MN, MK = 4, 64, 512


def _routing(seed, slots):
    rng = np.random.default_rng(seed)
    inds = mx.array(rng.integers(0, ME, size=(T, slots)).astype(np.uint32))
    sc = mx.array(rng.uniform(0.05, 0.9, size=(T, slots)).astype(np.float32))
    mx.eval(inds, sc)
    return inds, sc


def _kq_moe_ops(codec):
    """name -> (shape of x, call(x)) for the K-quant fused MoE ops."""
    gw, uw = _weight(codec, MN, MK, 1, ME)[0], _weight(codec, MN, MK, 2, ME)[0]
    shexp = _weight(codec, MN, MK, 3, ME)[0]
    sgw, suw = mx.contiguous(shexp[0]), mx.contiguous(shexp[1])
    inds, _ = _routing(3, R)
    inds_s, sc_s = _routing(4, S)
    la = (mx.random.normal((ME, MK, 4), key=mx.random.key(8)) * 0.05).astype(mx.float16)
    lb = (mx.random.normal((ME, 4, MN), key=mx.random.key(9)) * 0.05).astype(mx.float16)
    mx.eval(sgw, suw, la, lb)
    return {
        "moe_glu_gather_kq": (
            (T, MK),
            lambda x: kq.moe_glu_gather_kq(x, gw, uw, codec, inds),
        ),
        "gather_qmv_kq": ((T, R, MK), lambda x: kq.gather_qmv_kq(x, gw, codec, inds)),
        "gather_qmv_kq_lora": (
            (T, R, MK),
            lambda x: kq.gather_qmv_kq(x, gw, codec, inds, lora_a=la, lora_b=lb),
        ),
        "moe_glu_gather_shexp_kq": (
            (T, MK),
            lambda x: kq.moe_glu_gather_shexp_kq(x, gw, uw, sgw, suw, codec, inds),
        ),
        "gather_qmv_mix_kq": (
            (T, S, MK),
            lambda x: kq.gather_qmv_mix_kq(x, gw, sgw, codec, inds, sc_s),
        ),
        "gather_qmv_mix_ns_kq": (
            (T, S, MK),
            lambda x: kq.gather_qmv_mix_ns_kq(x, gw, codec, inds_s, sc_s),
        ),
        "gather_qmv_mix_ns_kq_lora": (
            (T, S, MK),
            lambda x: kq.gather_qmv_mix_ns_kq(
                x, gw, codec, inds_s, sc_s, lora_a=la, lora_b=lb
            ),
        ),
    }


def _mxfp4_moe_ops():
    """name -> (shape of x, call(x)) for the mxfp4 fused MoE ops."""
    rng = np.random.default_rng(1)

    def stack():
        w = mx.array(rng.integers(0, 2**32, size=(ME, MN, MK // 8), dtype=np.uint32))
        s = mx.array(rng.integers(118, 124, size=(ME, MN, MK // 32), dtype=np.uint8))
        b = mx.array(rng.uniform(-0.2, 0.2, (ME, MN)).astype(np.float32))
        mx.eval(w, s, b)
        return w, s, b

    gw, gs, gb = stack()
    uw, us, ub = stack()
    inds, sc = _routing(3, R)
    return {
        "moe_glu_gather": (
            (T, MK),
            lambda x: kq.moe_glu_gather(x, gw, gs, gb, uw, us, ub, inds),
        ),
        "gather_qmv_bias": (
            (T, R, MK),
            lambda x: kq.gather_qmv_bias(x, gw, gs, gb, inds),
        ),
        "gather_qmv_mix_bias": (
            (T, R, MK),
            lambda x: kq.gather_qmv_mix_bias(x, gw, gs, gb, inds, sc),
        ),
    }


def _moe():
    """The fused MoE gathers, K-quant and mxfp4."""
    bad = []
    for codec in (*MOE_CODECS, "mxfp4"):
        ops = _mxfp4_moe_ops() if codec == "mxfp4" else _kq_moe_ops(codec)
        for name, (shape, call) in ops.items():
            for dtype in DTYPES:
                offs = _offset_failures(call, shape, dtype, label=f"{name} {codec}")
                if offs:
                    bad.append([name, codec, dtype, offs])
    return bad


def _gather_mix():
    """gather_mix reads the expert outputs y, its activation, as vectors."""
    tokens, k, n = 5, 4, 256
    order = np.random.default_rng(6).permutation(tokens * k).astype(np.uint32)
    inv = mx.array(order)
    sc = mx.array(
        np.random.default_rng(7).uniform(0.1, 0.9, (tokens, k)).astype(np.float32)
    )
    mx.eval(inv, sc)
    bad = []
    for dtype in HALF:
        offs = _offset_failures(
            lambda y: kq.gather_mix(y, inv, sc),
            (tokens * k, n),
            dtype,
            label="gather_mix",
        )
        if offs:
            bad.append(["gather_mix", dtype, offs])
    return bad


def _rand(shape, dtype, seed, scale=0.5):
    a = (mx.random.normal(shape, key=mx.random.key(seed)) * scale).astype(dtype)
    mx.eval(a)
    return a


def _norm_latent_indexer():
    """rmsnorm_gate, latent_fp4_pack and dsa_indexer_score_decode."""
    bad = []
    for dt in (mx.float16, mx.bfloat16):
        for d in (64, 256):
            ops = {
                "x": _rand((6, d), dt, 1),
                "w": _rand((d,), dt, 2) + 1.0,
                "gate": _rand((6, d), dt, 3),
            }
            bad += _operand_failures(
                f"rmsnorm_gate d{d}",
                lambda x, w, gate: kq.rmsnorm_gate(x, w, gate, 1e-6),
                ops,
                [(n, (f"rmsnorm_gate {n}",), 4, False) for n in ("x", "w", "gate")],
            )
        rng = np.random.default_rng(11)
        for heads, ql, rows, qo, ratio in ((4, 1, 64, 63, 1), (64, 2, 1027, 4106, 4)):
            ops = {
                "q": mx.array(rng.standard_normal((1, heads, ql, 128)) * 0.4).astype(
                    dt
                ),
                "k": mx.array(rng.standard_normal((1, rows, 128)) * 0.4).astype(dt),
                "w": mx.array(rng.standard_normal((1, ql, heads)) * 0.2).astype(dt),
            }
            bad += _operand_failures(
                f"dsa_indexer_score_decode h{heads}",
                lambda q, k, w, qo=qo, ratio=ratio: kq.dsa_indexer_score_decode(
                    q, k, w, qo, ratio
                ),
                ops,
                [(n, (f"dsa_indexer_score_decode {n}",), 8, False) for n in ("q", "k")],
            )
    for dt in (mx.float16, mx.bfloat16, mx.float32):
        bad += _operand_failures(
            "latent_fp4_pack",
            lambda x: kq.latent_fp4_pack(x),
            {"x": _rand((8, 64), dt, 4)},
            [("x", ("latent_fp4_pack x",), 8, False)],
        )
    return bad


def _hc():
    """The hc glue ops: the stream, residual and collapsed rows, the
    projection fn, the mixes and the base biases."""
    return [b for dt in (mx.float16, mx.bfloat16) for b in _hc_dtype(dt)]


def _hc_dtype(dt):
    from test_hc_glue import HC, HC_EPS, ITERS, MIX, NORM_EPS

    d = 1024
    sk = {"iters": ITERS, "hc_eps": HC_EPS, "norm_eps": NORM_EPS}
    bad = []
    x = _rand((1, 1, HC, d), dt, 1, 0.05)
    fn = _rand((MIX, HC * d), mx.float32, 2, 0.02)
    scale = mx.array([1.1, 0.9, 1.3], dtype=mx.float32)
    base = _rand((MIX,), mx.float32, 3, 0.1)
    w = _rand((d,), dt, 4, 0.1) + 1.0
    mr, ssq = kq.hc_front_reduce(x, fn)
    col, post, comb = kq.hc_sinkhorn_collapse(x, mr, ssq, scale, base, w, **sk)
    mx.eval(scale, w, mr, ssq, col, post, comb)

    def case(op, *names):
        return [(n, (f"{op} {n}",), 4, False) for n in names]

    bad += _operand_failures(
        "hc_front_reduce",
        lambda x, fn: kq.hc_front_reduce(x, fn),
        {"x": x, "fn": fn},
        case("hc_front_reduce", "x", "fn"),
    )
    bad += _operand_failures(
        "hc_sinkhorn_collapse",
        lambda x, mixes, base: kq.hc_sinkhorn_collapse(
            x, mixes, ssq, scale, base, w, **sk
        ),
        {"x": x, "mixes": mr, "base": base},
        case("hc_sinkhorn_collapse", "x", "mixes", "base"),
    )
    bad += _operand_failures(
        "hc_expand",
        lambda x, resid: kq.hc_expand(x, resid, post, comb),
        {"x": col, "resid": x},
        case("hc_expand", "x", "resid"),
    )
    bad += _operand_failures(
        "hc_front_expand_reduce",
        lambda x_sub, resid, fn: kq.hc_front_expand_reduce(
            x_sub, resid, post, comb, fn
        ),
        {"x_sub": col, "resid": x, "fn": fn},
        case("hc_front_expand_reduce", "x_sub", "resid", "fn"),
    )
    bad += _operand_failures(
        "hc_front_expand_collapse",
        lambda x_sub, resid, fn, base: kq.hc_front_expand_collapse(
            x_sub, resid, post, comb, fn, scale, base, w, **sk
        ),
        {"x_sub": col, "resid": x, "fn": fn, "base": base},
        case("hc_front_expand_collapse", "x_sub", "resid", "fn", "base"),
    )
    return bad


def _qkv(op, strided=("k", "v"), q="q"):
    """Cases for an attention op's query and its keys and values."""
    return [(q, (f"{op} {q}",), 4, False)] + [
        (n, (f"{op} {n}",), 4, True) for n in strided
    ]


def _sdpa():
    """The attention ops: q, k and v, the q8 cache words and the KVarN
    stages."""
    bad = [b for dt in (mx.float16, mx.bfloat16) for b in _sdpa_dtype(dt)]
    return bad + _sdpa_kvarn()


def _sdpa_dtype(dt):
    from test_sdpa import _bs_prefill_lists, _make_indexed

    bad = []
    q = _rand((1, 8, 1, 128), dt, 1)
    k = _rand((1, 2, 512, 128), dt, 2)
    v = _rand((1, 2, 512, 128), dt, 3)
    qkv = {"q": q, "k": k, "v": v}
    bad += _operand_failures(
        "sdpa_vector",
        lambda q, k, v: kq.sdpa_vector(q, k, v, 0.0625),
        {
            "q": _rand((1, 8, 2, 256), dt, 15),
            "k": _rand((1, 2, 300, 256), dt, 16),
            "v": _rand((1, 2, 300, 256), dt, 17),
        },
        _qkv("sdpa_vector"),
    )
    bad += _operand_failures(
        "sdpa_decode_gqa",
        lambda q, k, v: kq.sdpa_decode_gqa(q, k, v, 0.0884),
        qkv,
        _qkv("sdpa_decode_gqa"),
    )
    pages = mx.array(np.broadcast_to(np.arange(16, dtype=np.int32), (1, 2, 16)))
    bad += _operand_failures(
        "sdpa_decode_gqa_paged",
        lambda q, k, v: kq.sdpa_decode_gqa_paged(q, k, v, 0.0884, pages),
        qkv,
        _qkv("sdpa_decode_gqa"),
    )
    # Head dim 512 at the depth that runs pass 1 on NAX.
    wide = {
        "q": _rand((1, 8, 1, 512), dt, 4),
        "k": _rand((1, 1, 4096, 512), dt, 5),
        "v": _rand((1, 1, 4096, 512), dt, 6),
    }
    bad += _operand_failures(
        "sdpa_decode_gqa d512",
        lambda q, k, v: kq.sdpa_decode_gqa(q, k, v, 0.0442),
        wide,
        _qkv("sdpa_decode_gqa"),
    )
    # A q8 cache: k and v are the packed words.
    kw, ks, kb = mx.quantize(k, group_size=64, bits=8)
    vw, vs, vb = mx.quantize(v, group_size=64, bits=8)
    mx.eval(kw, ks, kb, vw, vs, vb)
    bad += _operand_failures(
        "sdpa_decode_gqa q8",
        lambda k, v: kq.sdpa_decode_gqa(
            q, k, v, 0.0884, k_scales=ks, k_biases=kb, v_scales=vs, v_biases=vb
        ),
        {"k": kw, "v": vw},
        _qkv("sdpa_decode_gqa")[1:],
    )
    qf = _rand((1, 2, 12, 128), dt, 7)
    bad += _operand_failures(
        "sdpa_fa_verify",
        lambda q, k, v: kq.sdpa_fa_verify(q, k, v, 0.0884, q_len=3),
        {"q": qf, "k": k, "v": v},
        _qkv("sdpa_fa_verify"),
    )
    qi, kv, idx = _make_indexed(8, 1, 100, 64, 0, dt, seed=9)
    bad += _operand_failures(
        "sdpa_fa_indexed",
        lambda q, kv: kq.sdpa_fa_indexed(q, kv, idx, 0.0442),
        {"q": qi, "kv": kv},
        _qkv("sdpa_fa_indexed", strided=("kv",)),
    )
    # One sequence: the shared pass's folded query is q itself.
    bad += _operand_failures(
        "sdpa_decode_gqa_cascade",
        lambda q, k_shared, v_shared, k, v: kq.sdpa_decode_gqa_cascade(
            q, k_shared, v_shared, k, v, 0.0884
        ),
        {
            "q": q,
            "k_shared": k,
            "v_shared": v,
            "k": _rand((1, 2, 64, 128), dt, 10),
            "v": _rand((1, 2, 64, 128), dt, 11),
        },
        [
            (
                "q",
                ("sdpa_decode_gqa_cascade q", "sdpa_decode_gqa_cascade q_folded"),
                4,
                False,
            )
        ]
        + _qkv("sdpa_decode_gqa_cascade", ("k_shared", "v_shared", "k", "v"))[1:],
    )
    # A q8 cache: the shared pass stages the words in the fa_verify kernel.
    kp, kps, kpb = mx.quantize(_rand((1, 2, 64, 128), dt, 12), group_size=64, bits=8)
    vp, vps, vpb = mx.quantize(_rand((1, 2, 64, 128), dt, 13), group_size=64, bits=8)
    mx.eval(kp, kps, kpb, vp, vps, vpb)
    bad += _operand_failures(
        "sdpa_decode_gqa_cascade q8",
        lambda k_shared, v_shared, k, v: kq.sdpa_decode_gqa_cascade(
            q,
            k_shared,
            v_shared,
            k,
            v,
            0.0884,
            k_shared_scales=ks,
            k_shared_biases=kb,
            v_shared_scales=vs,
            v_shared_biases=vb,
            k_priv_scales=kps,
            k_priv_biases=kpb,
            v_priv_scales=vps,
            v_priv_biases=vpb,
        ),
        {"k_shared": kw, "v_shared": vw, "k": kp, "v": vp},
        _qkv("sdpa_decode_gqa_cascade", ("k_shared", "v_shared", "k", "v"))[1:],
    )
    # Four queries: the folded query is a view of q.
    length, s, topk = 4, 1024, 16
    offset = s - length
    sel = np.zeros((length, topk), dtype=np.int64)
    rng = np.random.default_rng(3)
    for t in range(length):
        sel[t] = rng.choice((offset + t + 1) // 4, size=topk, replace=False)
    lists = [mx.array(a) for a in _bs_prefill_lists(sel, offset, s)]
    bad += _operand_failures(
        "sdpa_prefill_block_sparse",
        lambda q, k, v: kq.sdpa_prefill_block_sparse(q, k, v, 0.0625, *lists, offset),
        {
            "q": _rand((1, 24, length, 256), dt, 12),
            "k": _rand((1, 2, s, 256), dt, 13),
            "v": _rand((1, 2, s, 256), dt, 14),
        },
        _qkv("sdpa_prefill_block_sparse"),
    )
    return bad


def _sdpa_kvarn():
    """The KVarN stages, float16."""
    from test_kvarn_sdpa import build_state, make_q

    bad = []
    st = build_state(1, 2, 300, 6, 6, seed=21)
    codes = {n: st[n] for n in ("codes_k", "axes_k", "codes_v", "axes_v")}
    qk = mx.array(make_q(1, 8, 1)).astype(mx.float16)
    qv = mx.array(make_q(1, 8, 4)).astype(mx.float16).reshape(1, 2, 16, 128)
    stages = {"stage_k": st["stage_k"], "stage_v": st["stage_v"]}
    bad += _operand_failures(
        "sdpa_decode_gqa_kvarn",
        lambda stage_k, stage_v: kq.sdpa_decode_gqa_kvarn(
            qk,
            codes["codes_k"],
            codes["axes_k"],
            codes["codes_v"],
            codes["axes_v"],
            stage_k,
            stage_v,
            300,
            0.0884,
            6,
            6,
        ),
        stages,
        [(n, (f"sdpa_decode_gqa {n}",), 4, False) for n in stages],
    )
    bad += _operand_failures(
        "sdpa_fa_verify_kvarn",
        lambda stage_k, stage_v: kq.sdpa_fa_verify_kvarn(
            qv,
            codes["codes_k"],
            codes["axes_k"],
            codes["codes_v"],
            codes["axes_v"],
            stage_k,
            stage_v,
            300,
            0.0884,
            6,
            6,
            4,
        ),
        stages,
        [(n, (f"sdpa_fa_verify {n}",), 4, False) for n in stages],
    )
    return bad


def _sparse():
    """The sparse attention ops: q and the window, pool and key rows."""
    return [b for dt in (mx.float16, mx.bfloat16) for b in _sparse_dtype(dt)]


def _sparse_dtype(dt):
    from test_dsa_sparse_attn import _build as dsa_build
    from test_sdpa_sparse_decode import _case as decode_case
    from test_sdpa_sparse_decode import _on_grid_pool
    from test_sdpa_sparse_prefill import _case as prefill_case

    bad = []
    (q, win, pool, idx, scale, kw), _ = decode_case(
        1, 16, 2, 128, 24, 300, 40, dt, np.int32, True, True, True
    )
    mx.eval(q, win, pool, idx)
    names = [(n, (f"sdpa_sparse_decode {n}",), 8, False) for n in ("q", "win", "pool")]
    bad += _operand_failures(
        "sdpa_sparse_decode",
        lambda q, win, pool: kq.sdpa_sparse_decode(q, win, pool, idx, scale, **kw),
        {"q": q, "win": win, "pool": pool},
        names,
    )
    if dt == mx.float16:
        codes, scales = kq.latent_fp4_pack(_on_grid_pool(pool))
        mx.eval(codes, scales)
        bad += _operand_failures(
            "sdpa_sparse_decode packed",
            lambda pool: kq.sdpa_sparse_decode(
                q, win, pool, idx, scale, pool_scales=scales, **kw
            ),
            {"pool": codes},
            [("pool", ("sdpa_sparse_decode pool",), 16, False)],
        )
    (q, win, pool, idx, scale, band, kw), _ = prefill_case(
        1, 12, 9, 128, 30, 96, 24, 8, dt, np.int32, True, True, True
    )
    mx.eval(q, win, pool, idx)
    bad += _operand_failures(
        "sdpa_sparse_prefill",
        lambda q, win, pool: kq.sdpa_sparse_prefill(
            q, win, pool, idx, scale, band, **kw
        ),
        {"q": q, "win": win, "pool": pool},
        [(n, (f"sdpa_sparse_prefill {n}",), 8, False) for n in ("q", "win", "pool")],
    )
    q, local_kv, pooled, topk, sinks, scale = dsa_build(
        1, 128, 640, 512, 2560, 4, 128, dt, 5, "random"
    )
    mx.eval(q, local_kv, pooled, topk, sinks)
    bad += _operand_failures(
        "dsa_sparse_attention",
        lambda q, local_kv, pooled: kq.dsa_sparse_attention(
            q, local_kv, pooled, topk, sinks, scale, 2560, 4, 128
        ),
        {"q": q, "local_kv": local_kv, "pooled": pooled},
        [
            (n, (f"dsa_sparse_attention {n}",), 8, False)
            for n in ("q", "local_kv", "pooled")
        ],
    )
    return bad


GROUPS = {
    "matmul_ops": _matmul_ops,
    "bias_skinny": _bias_and_skinny,
    "gathers": _gathers,
    "moe": _moe,
    "gather_mix": _gather_mix,
    "norm_latent_indexer": _norm_latent_indexer,
    "hc": _hc,
    "sdpa": _sdpa,
    "sparse": _sparse,
}
# The matmul groups run under each route environment, the fused LoRA kernel
# (KQ_QMM_LORA=2) with the LoRA matmul; the other ops with NAX on and off.
MATMUL_ENVS = ("default", "nax_qmv_off", "nax_off")
GROUP_ENVS = [
    *((g, e) for g in list(GROUPS)[:5] for e in MATMUL_ENVS),
    ("matmul_ops", "lora2"),
    *((g, e) for g in list(GROUPS)[5:] for e in ("default", "nax_off")),
]

_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import test_x_offset_views as t
print("RESULT " + json.dumps(t.GROUPS[sys.argv[2]]()))
"""
_XA = re.compile(r"^\[kq qmm\] (\S+) .* xa=(\d+)$")
_ALIGN = re.compile(r"^\[kq align\] (.+) xin=(\d+)$")
_ARM = re.compile(r"^\[arm\] (.*) \| (\S+) \| (.*)$")


def _unreached(arm, need, names, copies, xa):
    """True when an arm that needs copies has not logged them."""
    if need in ("none", "any"):
        return False
    want = [n for n in names.split(",") if n]
    if need == "copy":
        return any(n not in copies for n in want) if want else not copies
    off = {n for n, xin in copies.items() if xin}
    if want:
        return any(n not in off for n in want)
    return not off and not any(xa)


def ops_offsets(group, env):
    """(bad, unaligned, unreached, copied) of an op group in a child under
    one environment, as quantized_matmul_offsets gives them."""
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    child_env.update({"KQ_QMM_LOG": "1", **ENVS[env]})
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, TESTS_DIR, group],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stderr[-6000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    bad = json.loads(line[len("RESULT ") :])
    unaligned, unreached, copied = set(), [], []
    arm = None
    for ln in [*r.stderr.splitlines(), "[arm] end | any | "]:
        m = _ARM.match(ln)
        if m:
            if arm and _unreached(*arm):
                unreached.append(arm[0])
            if arm and arm[1] == "none" and arm[3]:
                copied.append((arm[0], sorted(arm[3])))
            arm = (m.group(1), m.group(2), m.group(3), {}, [])
            continue
        m = _XA.match(ln)
        if m:
            if _unaligned_x(m.group(1), int(m.group(2))):
                unaligned.add((m.group(1), int(m.group(2))))
            if arm:
                arm[4].append(int(m.group(2)))
            continue
        m = _ALIGN.match(ln)
        if m and arm:
            arm[3][m.group(1)] = max(arm[3].get(m.group(1), 0), int(m.group(2)))
    return bad, sorted(unaligned), unreached, copied


@metal_only
@pytest.mark.parametrize("group,env", GROUP_ENVS)
def test_ops_x_offsets(group, env):
    """Bit-exact output at every offset, every matmul kernel reads x from a
    16-byte boundary, every view off the boundary reaches its op there and
    is copied, and every view on it is read in place."""
    bad, unaligned, unreached, copied = ops_offsets(group, env)
    assert bad == [], bad
    assert unaligned == [], unaligned
    assert unreached == [], unreached[:40]
    assert copied == [], copied[:40]
