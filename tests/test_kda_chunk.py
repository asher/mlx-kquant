"""kda_chunk: chunked KDA (per-key-channel gated delta rule) prefill against
the token-sequential recurrence in fp32.

The Metal path needs tensor-op (NAX) hardware; under KQUANT_FORCE_CPU=1 the
op runs its CPU recurrence, which the same tests cover.
"""

import os

import mlx.core as mx
import pytest

import mlx_kquant as kq

FORCE_CPU = bool(os.environ.get("KQUANT_FORCE_CPU"))

requires_nax_or_cpu = pytest.mark.skipif(
    not FORCE_CPU and not kq.nax_available(),
    reason="kda_chunk's Metal kernel requires tensor-op (NAX) hardware.",
)

D = 128

# Scaled key/query operands round to the activation dtype inside the
# kernel; fp32 inputs run the CPU path only.
REL_BOUND = {mx.bfloat16: 2e-2, mx.float16: 1e-2, mx.float32: 1e-4}


def _rel(a, b):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    return float(mx.linalg.norm(af - bf) / (mx.linalg.norm(bf) + 1e-9))


def _ref(q, k, v, log_g, beta, state):
    """Sequential fp32 recurrence, the mlx-lm gated_delta_ops form."""
    B, T, H, Dk = k.shape
    S = state.astype(mx.float32)
    g = mx.exp(log_g.astype(mx.float32))
    outs = []
    for t in range(T):
        qt = q[:, t].astype(mx.float32)
        kt = k[:, t].astype(mx.float32)
        vt = v[:, t].astype(mx.float32)
        S = S * g[:, t][:, :, None, :]
        kv = (S * kt[..., None, :]).sum(axis=-1)
        delta = (vt - kv) * beta[:, t].astype(mx.float32)[..., None]
        S = S + kt[..., None, :] * delta[..., None]
        outs.append((S * qt[..., None, :]).sum(axis=-1))
    return mx.stack(outs, axis=1), S


def _make(B, T, H, dtype, lb, seed):
    key = mx.random.key(seed)
    k0, k1, k2, k3, k4, k5 = mx.random.split(key, 6)
    scale = D**-0.5
    q = (mx.random.normal((B, T, H, D), key=k0) * scale).astype(dtype)
    k = mx.random.normal((B, T, H, D), key=k1)
    k = (k / mx.sqrt((k * k).sum(-1, keepdims=True))).astype(dtype)
    v = mx.random.normal((B, T, H, D), key=k2).astype(dtype)
    # GLM form: log g = lb * sigmoid(a), in (lb, 0).
    log_g = lb * mx.sigmoid(mx.random.normal((B, T, H, D), key=k3) * 2)
    beta = mx.sigmoid(mx.random.normal((B, T, H), key=k4))
    state = mx.random.normal((B, H, D, D), key=k5) * 0.1
    mx.eval(q, k, v, log_g, beta, state)
    return q, k, v, log_g, beta, state


@requires_nax_or_cpu
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 1, -0.5),  # one chunk, mild decay
        (1, 64, 2, -5.0),  # two chunks at the GLM gate floor
        (2, 96, 3, -5.0),  # batched
        (1, 200, 2, -3.0),  # tail padded to 224
        (1, 7, 1, -1.0),  # shorter than a chunk
    ],
)
def test_kda_chunk_matches_recurrence(shape, dtype):
    B, T, H, lb = shape
    q, k, v, log_g, beta, state = _make(B, T, H, dtype, lb, seed=T + H)
    y, s_out = kq.kda_chunk(q, k, v, log_g, beta, state)
    y_ref, s_ref = _ref(q, k, v, log_g, beta, state)
    mx.eval(y, s_out, y_ref, s_ref)
    assert y.shape == (B, T, H, D) and y.dtype == dtype
    assert s_out.shape == (B, H, D, D) and s_out.dtype == mx.float32
    assert _rel(y, y_ref) < REL_BOUND[dtype], f"{shape} y rel {_rel(y, y_ref):.3e}"
    assert _rel(s_out, s_ref) < REL_BOUND[dtype], (
        f"{shape} state rel {_rel(s_out, s_ref):.3e}"
    )


@requires_nax_or_cpu
def test_kda_chunk_zero_state_long():
    # A long sequence from an empty state (the first prefill chunk).
    B, T, H = 1, 1024 if not FORCE_CPU else 96, 2
    q, k, v, log_g, beta, _ = _make(B, T, H, mx.bfloat16, -5.0, seed=11)
    state = mx.zeros((B, H, D, D), dtype=mx.float32)
    y, s_out = kq.kda_chunk(q, k, v, log_g, beta, state)
    y_ref, s_ref = _ref(q, k, v, log_g, beta, state)
    mx.eval(y, s_out, y_ref, s_ref)
    assert _rel(y, y_ref) < REL_BOUND[mx.bfloat16]
    assert _rel(s_out, s_ref) < REL_BOUND[mx.bfloat16]


@requires_nax_or_cpu
def test_kda_chunk_chunking_is_transparent():
    # Running one call over 96 tokens equals three calls of 32 with the
    # state threaded through, in the kernel's own numerics.
    q, k, v, log_g, beta, state = _make(1, 96, 2, mx.bfloat16, -2.0, seed=5)
    y_all, s_all = kq.kda_chunk(q, k, v, log_g, beta, state)
    ys = []
    s = state
    for c in range(3):
        sl = slice(32 * c, 32 * (c + 1))
        y_c, s = kq.kda_chunk(
            q[:, sl], k[:, sl], v[:, sl], log_g[:, sl], beta[:, sl], s
        )
        ys.append(y_c)
    y_split = mx.concatenate(ys, axis=1)
    mx.eval(y_all, s_all, y_split, s)
    assert _rel(y_split, y_all) < 1e-5
    assert _rel(s, s_all) < 1e-5


@requires_nax_or_cpu
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("T", [64, 40, 3])
def test_kda_chunk_gated_matches_log_g_form(dtype, T):
    # The in-kernel gate equals kda_chunk on the same log gate formed in
    # fp32, and both track the recurrence; padded lengths carry no decay.
    B, H, lb = 1, 2, -5.0
    q, k, v, _, beta, state = _make(B, T, H, dtype, lb, seed=T)
    key = mx.random.key(T + 1)
    k0, k1, k2 = mx.random.split(key, 3)
    a = (mx.random.normal((B, T, H, D), key=k0) * 2).astype(dtype)
    a_scale = mx.exp(mx.random.normal((H,), key=k1) * 0.3)
    dt_bias = mx.random.normal((H, D), key=k2) * 0.5
    log_g = lb * mx.sigmoid(a_scale[:, None] * (a.astype(mx.float32) + dt_bias))
    mx.eval(a, a_scale, dt_bias, log_g)
    y_g, s_g = kq.kda_chunk_gated(q, k, v, a, a_scale, dt_bias, beta, state, lb)
    y_l, s_l = kq.kda_chunk(q, k, v, log_g, beta, state)
    y_ref, s_ref = _ref(q, k, v, log_g, beta, state)
    mx.eval(y_g, s_g, y_l, s_l, y_ref, s_ref)
    assert y_g.shape == (B, T, H, D) and y_g.dtype == dtype
    assert _rel(y_g, y_l) < 2e-3
    assert _rel(s_g, s_l) < 2e-3
    assert _rel(y_g, y_ref) < REL_BOUND[dtype]
    assert _rel(s_g, s_ref) < REL_BOUND[dtype]


def test_kda_chunk_rejects_bad_shapes():
    q, k, v, log_g, beta, state = _make(1, 32, 1, mx.bfloat16, -1.0, seed=1)
    with pytest.raises(ValueError):
        kq.kda_chunk(q, k, v, log_g, beta[:, :16], state)
    with pytest.raises(ValueError):
        kq.kda_chunk(q, k, v[..., :64], log_g, beta, state)
    with pytest.raises(ValueError):
        kq.kda_chunk(q, k, v, log_g, beta, state[:, :, :64])
