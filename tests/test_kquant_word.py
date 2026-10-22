"""KQ_KQUANT_WORD: the q2_k, q4_k and q5_k chunk decodes of the Ext
mat-vecs and MoE gathers form four codes per 32-bit word, and
KQ_KQUANT_WORD=0 restores the per-byte loop.

The two forms give the same weights, so each output must be bit-identical
across them: the verify mat-vec (mv_ext), the gate/up and score-mixed down
gathers of each codec's expert stacks, and the q5_k shared expert over
iq2_xxs experts, both on the 32-weight kernels (the M5 default) and with
KQ_MOE_C32=0. The weight bytes are random, so every nibble, high bit and
scale code occurs. Each arm runs in a subprocess, since the lever is read
once per process, and with the lever off KQ_QMM_LOG must show that every
kernel whose name carries one of these codecs was fetched with the byte
loop. q3_k keeps the per-byte loop, so its kernels are not specialized.
"""

import os
import re
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="Metal kernel decode; the CPU kernels have their own.",
)

CODECS = ["q2_k", "q4_k", "q5_k"]

_SNIPPET = r"""
import os
import sys
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

out_path = sys.argv[1]
codecs = sys.argv[2].split(",")
rng = np.random.default_rng(7)
E, N, K = 16, 64, 1024
# fp16 d and dmin offsets; q3_k has no dmin.
D_OFF = {"q2_k": (80, 82), "q3_k": (108, None), "q4_k": (0, 2),
         "q5_k": (0, 2), "iq2_xxs": (0, None)}


def wire(codec, lead):
    bpb, wpb = CODEC_GEOMETRY[codec][2:]
    nb = int(np.prod(lead)) * (K // wpb)
    w = rng.integers(0, 256, size=(nb, bpb), dtype=np.uint8)
    d_off, m_off = D_OFF[codec]
    d = rng.uniform(0.004, 0.01, nb).astype(np.float16)
    w[:, d_off:d_off + 2] = d.view(np.uint8).reshape(nb, 2)
    if m_off is not None:
        m = rng.uniform(0.001, 0.004, nb).astype(np.float16)
        w[:, m_off:m_off + 2] = m.view(np.uint8).reshape(nb, 2)
    return mx.array(w.reshape(*lead, (K // wpb) * bpb))


def act(shape):
    return mx.array((rng.standard_normal(shape) * 0.05).astype(np.float16))


outs = {}
scales = mx.zeros((1,), dtype=mx.uint8)
for c in codecs:
    w2 = wire(c, (N,))
    # The route is read when the graph is evaluated.
    os.environ["KQ_QMM_ROUTE"] = "mv_ext"
    for m in (4, 9):
        outs[f"{c}/mv_ext{m}"] = kq.quantized_matmul(
            act((m, K)), w2, scales, c, transpose=True)
        mx.eval(outs[f"{c}/mv_ext{m}"])
    del os.environ["KQ_QMM_ROUTE"]
    gw, uw = wire(c, (E, N)), wire(c, (E, N))
    for T, R in ((1, 6), (3, 6)):
        inds = mx.array(rng.integers(0, E, size=(T, R)).astype(np.uint32))
        sc = mx.array(rng.uniform(0.05, 0.9, size=(T, R)).astype(np.float32))
        outs[f"{c}/glu{T}"] = kq.moe_glu_gather_kq(act((T, K)), gw, uw, c, inds)
        outs[f"{c}/mix_ns{T}"] = kq.gather_qmv_mix_ns_kq(
            act((T, R, K)), gw, c, inds, sc)
    if c == "q5_k":
        rg, ru = wire("iq2_xxs", (E, N)), wire("iq2_xxs", (E, N))
        sg, su = wire(c, (N,)), wire(c, (N,))
        for T, R in ((1, 6), (3, 6)):
            inds = mx.array(rng.integers(0, E, size=(T, R)).astype(np.uint32))
            sc = mx.array(rng.uniform(0.05, 0.9, size=(T, R + 1)).astype(np.float32))
            outs[f"sx/glu{T}"] = kq.moe_glu_gather_shexp_kq(
                act((T, K)), rg, ru, sg, su, "iq2_xxs", inds,
                shexp_kquant_type=c)
            outs[f"sx/mix{T}"] = kq.gather_qmv_mix_kq(
                act((T, R + 1, K)), rg, sg, "iq2_xxs", inds, sc,
                shexp_kquant_type=c)
mx.eval(list(outs.values()))
np.savez(out_path, **{k: np.array(v.astype(mx.float32)) for k, v in outs.items()})
"""


def _run(tmp_path, arm, env):
    f = tmp_path / f"{arm}.npz"
    base = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    r = subprocess.run(
        [sys.executable, "-c", _SNIPPET, str(f), ",".join(CODECS + ["q3_k"])],
        capture_output=True,
        text=True,
        env={**base, "KQ_QMM_LOG": "1", "KQ_MOE_NX_LOG": "1", **env},
    )
    assert r.returncode == 0, f"{arm} exited {r.returncode}\n{r.stderr[-4000:]}"
    return np.load(f), r.stderr


def _kernels(log):
    """Kernel names of the dispatch log lines (KQ_QMM_LOG, KQ_MOE_NX_LOG)
    and the keys of the byte-loop fetch lines."""
    names = set(re.findall(r"^\[kq(?: qmm|_moe)\] ((?:kq|kquant)_\S+)", log, re.M))
    byte = set(re.findall(r"^\[kq decode\] byte (\S+)", log, re.M))
    return names, byte


@pytest.mark.parametrize("c32", ["", "0"], ids=["default", "c32_off"])
def test_word_decode_bit_identical(tmp_path, c32):
    env = {"KQ_MOE_C32": c32} if c32 else {}
    word, wlog = _run(tmp_path, "word", env)
    byte, blog = _run(tmp_path, "byte", {**env, "KQ_KQUANT_WORD": "0"})
    assert sorted(word.files) == sorted(byte.files)
    bad = [k for k in word.files if not np.array_equal(word[k], byte[k])]
    assert not bad, f"word and byte decodes differ: {bad}"
    for k in word.files:
        assert np.isfinite(word[k]).all() and np.abs(word[k]).max() > 0, k
    # The word arm fetches nothing with the byte loop; the byte arm fetches
    # every kernel it dispatches that names one of the codecs with it.
    assert "[kq decode]" not in wlog
    names, keys = _kernels(blog)
    for c in CODECS:
        assert any(c in n and "mv_ext" in n for n in names), (c, sorted(names))
    assert any("_sx_q5_k" in n for n in names), sorted(names)
    assert keys and all(k.endswith("_kw0") for k in keys), sorted(keys)
    missing = [
        n
        for n in names
        if any(c in n for c in CODECS) and not any(k.startswith(n) for k in keys)
    ]
    assert not missing, f"dispatched without a byte-loop fetch: {missing}"
    assert any("q3_k" in n and "mv_ext" in n for n in names), sorted(names)
    q3 = [k for k in keys if "q3_k" in k and not any(c in k for c in CODECS)]
    assert not q3, f"q3_k fetched with the decode constant: {q3}"
