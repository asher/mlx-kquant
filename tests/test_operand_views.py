"""Views of the operands that the MoE, routing and attention kernels read by
layout.

An op builder sees the strides of an operand only when the operand has
memory: an unevaluated view reports dense strides whatever it becomes. The
ops here decide the layout of an operand in their eval, where every input
has memory, so a view gives the same result whether it is evaluated before
the call or not:

- an activation, router logits and the three operands of route_shed are
  copied when they are not row-major;
- an expert stack, a shared-expert tensor and the packed mxfp4 weights and
  scales are refused when they are not row-major, with the message the op
  builder gives for a view that has memory;
- the keys and values of sdpa_prefill_block_sparse are refused when their
  head dim is not dense, as in the other attention ops;
- the window, pool and pool scales of the sparse attention ops and the
  keys of dsa_indexer_score_decode are copied when a row of theirs is not
  dense, and a packed pool is copied when its codes and scales step
  batches differently;
- the scales and biases of a quantized cache are refused when the two
  differ in layout or their last dim is not dense.

An operand that is dense adds no node to the graph.

route_shed runs on the default device, so the suite covers the GPU and,
under KQUANT_FORCE_CPU, the CPU stream. The other ops are Metal-only
kernels with no CPU eval, and their cases are skipped there.
"""

from __future__ import annotations

import os
import re

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

metal_only = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="Metal-only kernel; no CPU path.",
)

E, N, K = 4, 64, 512
T, R = 3, 2
S = R + 1
_W = {}


def _dense(view):
    """A copy of a view in its own row-major buffer. numpy has no bfloat16
    and float32 holds it exactly."""
    if view.dtype == mx.bfloat16:
        wide = np.ascontiguousarray(np.array(view.astype(mx.float32)))
        return mx.array(wide).astype(mx.bfloat16)
    return mx.array(np.ascontiguousarray(np.array(view)))


def _nodes(*outputs):
    """Names of the primitives in the graph of the outputs."""
    import tempfile

    with tempfile.NamedTemporaryFile("r", suffix=".dot") as f:
        mx.export_to_dot(f.name, *outputs)
        text = f.read()
    return sorted(re.findall(r'label ="([A-Za-z0-9_]+)"', text))


def _same(got, want):
    got = got if isinstance(got, (list, tuple)) else [got]
    want = want if isinstance(want, (list, tuple)) else [want]
    mx.eval(*got, *want)
    return all(mx.array_equal(g, w).item() for g, w in zip(got, want, strict=True))


def _check_views(views, call, state):
    """call(view) equals call(dense copy of the view) for every view."""
    for name, make in views.items():
        want = call(_dense(make()))
        view = make()
        if state == "evaluated":
            mx.eval(view)
        assert _same(call(view), want), (name, state)


def _strided(base):
    """Views of one [rows, cols] array that are not row-major, as builders
    of a fresh view. base is [2 * rows, 2 * cols] random values."""
    rows, cols = base.shape[0] // 2, base.shape[1] // 2
    turned = _dense(mx.swapaxes(base[:rows, :cols], 0, 1))
    mx.eval(turned)
    return {
        "even_cols": lambda: base[:rows, 0::2],
        "odd_cols": lambda: base[:rows, 1::2],
        "even_rows": lambda: base[0::2, :cols],
        "reversed_rows": lambda: base[::-1][:rows, :cols],
        "col_band": lambda: base[:rows, cols // 2 : cols // 2 + cols],
        "transposed": lambda: mx.swapaxes(turned, 0, 1),
    }


# ----------------------------- route_shed -----------------------------


def _route_case(seed=0, n_experts=256):
    rng = np.random.default_rng(seed)
    idx = mx.array(rng.integers(0, n_experts, size=(2 * T, 2 * R)).astype(np.uint32))
    sc = mx.array(rng.uniform(0.05, 0.9, size=(2 * T, 2 * R)).astype(np.float32))
    slots = np.where(
        rng.random(2 * n_experts) < 0.7, rng.integers(0, 64, 2 * n_experts), -1
    )
    slot = mx.array(slots.astype(np.int32))
    mx.eval(idx, sc, slot)
    return idx, sc, slot


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
def test_route_shed_operand_views(state):
    idx2, sc2, slot2 = _route_case()
    n_experts = slot2.shape[0] // 2
    idx = _dense(idx2[:T, :R])
    sc = _dense(sc2[:T, :R])
    slot = _dense(slot2[:n_experts])
    _check_views(_strided(idx2), lambda v: kq.route_shed(v, sc, slot), state)
    _check_views(_strided(sc2), lambda v: kq.route_shed(idx, v, slot), state)
    _check_views(
        {
            "every_other": lambda: slot2[0::2],
            "reversed": lambda: slot2[::-1][:n_experts],
            "band": lambda: slot2[n_experts // 2 : n_experts // 2 + n_experts],
        },
        lambda v: kq.route_shed(idx, sc, v),
        state,
    )


def test_route_shed_adds_no_node():
    """The call a streamed decode makes: unevaluated router outputs. The
    graph holds the op and the ops that feed it, and no layout node."""
    idx2, sc2, slot2 = _route_case(1)
    idx = _dense(idx2[:T, :R])
    sc = _dense(sc2[:T, :R])
    feed = set(_nodes(idx + 0, sc * 1.0))
    assert "Add" in feed and "Multiply" in feed
    outs = kq.route_shed(idx + 0, sc * 1.0, slot2)
    assert set(_nodes(*outs)) == feed | {"KQuantRouteShed"}
    outs = kq.route_shed(idx2[:T, 0::2], sc, slot2[0::2])
    assert set(_nodes(*outs)) == {"KQuantRouteShed", "Slice"}


# ------------------------- fused MoE gathers --------------------------


def _stack(codec, seed, n=N, experts=E):
    """A K-quant expert stack uint8 [experts, n, bytes_per_row]."""
    key = (codec, seed, n, experts)
    if key not in _W:
        wf = mx.random.normal((experts * n, K), key=mx.random.key(seed)) * 0.1
        w, _ = kq.quantize(wf, codec)
        w = _dense(w.reshape(experts, n, -1))
        mx.eval(w)
        _W[key] = w
    return _W[key]


def _shexp(codec, e):
    """A shared-expert tensor uint8 [N, bytes_per_row] in its own buffer."""
    key = (codec, "shexp", e)
    if key not in _W:
        _W[key] = _dense(_stack(codec, 3)[e])
        mx.eval(_W[key])
    return _W[key]


def _acts(shape, dtype, seed):
    """Activations of twice the row and column count, for _strided."""
    full = (*shape[:-2], 2 * shape[-2], 2 * shape[-1])
    x = (mx.random.normal(full, key=mx.random.key(seed)) * 0.1).astype(dtype)
    mx.eval(x)
    return x


def _x_views(x2):
    """_strided over the last two dims of a 2-D or 3-D activation base."""
    if x2.ndim == 2:
        return _strided(x2)
    t, s2, k2 = x2.shape
    s, k = s2 // 2, k2 // 2
    return {
        "even_cols": lambda: x2[:, :s, 0::2],
        "even_slots": lambda: x2[:, 0::2, :k],
        "reversed": lambda: x2[::-1][:, :s, :k],
        "col_band": lambda: x2[:, :s, k // 2 : k // 2 + k],
    }


def _routing(seed=3, slots=R):
    rng = np.random.default_rng(seed)
    inds = mx.array(rng.integers(0, E, size=(T, slots)).astype(np.uint32))
    sc = mx.array(rng.uniform(0.05, 0.9, size=(T, slots)).astype(np.float32))
    mx.eval(inds, sc)
    return inds, sc


def _kq_ops(codec):
    """name -> (shape of x, call(x)) for the K-quant fused MoE ops."""
    gw, uw = _stack(codec, 1), _stack(codec, 2)
    sgw, suw = _shexp(codec, 0), _shexp(codec, 1)
    inds, _ = _routing()
    inds_s, sc_s = _routing(4, S)
    r = 4
    la = (mx.random.normal((E, K, r), key=mx.random.key(8)) * 0.05).astype(mx.float16)
    lb = (mx.random.normal((E, r, N), key=mx.random.key(9)) * 0.05).astype(mx.float16)
    mx.eval(la, lb)
    return {
        "moe_glu_gather_kq": (
            (T, K),
            lambda x: kq.moe_glu_gather_kq(x, gw, uw, codec, inds),
        ),
        "gather_qmv_kq": (
            (T, R, K),
            lambda x: kq.gather_qmv_kq(x, gw, codec, inds),
        ),
        "gather_qmv_kq_lora": (
            (T, R, K),
            lambda x: kq.gather_qmv_kq(x, gw, codec, inds, lora_a=la, lora_b=lb),
        ),
        "moe_glu_gather_shexp_kq": (
            (T, K),
            lambda x: kq.moe_glu_gather_shexp_kq(x, gw, uw, sgw, suw, codec, inds),
        ),
        "gather_qmv_mix_kq": (
            (T, S, K),
            lambda x: kq.gather_qmv_mix_kq(x, gw, sgw, codec, inds, sc_s),
        ),
        "gather_qmv_mix_ns_kq": (
            (T, S, K),
            lambda x: kq.gather_qmv_mix_ns_kq(x, gw, codec, inds_s, sc_s),
        ),
        "gather_qmv_mix_ns_kq_lora": (
            (T, S, K),
            lambda x: kq.gather_qmv_mix_ns_kq(
                x, gw, codec, inds_s, sc_s, lora_a=la, lora_b=lb
            ),
        ),
    }


KQ_OPS = [
    "moe_glu_gather_kq",
    "gather_qmv_kq",
    "gather_qmv_kq_lora",
    "moe_glu_gather_shexp_kq",
    "gather_qmv_mix_kq",
    "gather_qmv_mix_ns_kq",
    "gather_qmv_mix_ns_kq_lora",
]


@metal_only
@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("codec", ["q4_k", "q8_0"])
@pytest.mark.parametrize("op", KQ_OPS)
def test_moe_activation_views(op, codec, state):
    shape, call = _kq_ops(codec)[op]
    _check_views(_x_views(_acts(shape, mx.float16, 21)), call, state)


def _mxfp4_stack(seed):
    """A packed mxfp4 stack: uint32 [E, N, K / 8] codes, uint8 [E, N, K / 32]
    block exponents near 2^-6, float32 [E, N] bias."""
    rng = np.random.default_rng(seed)
    w = mx.array(rng.integers(0, 2**32, size=(E, N, K // 8), dtype=np.uint32))
    sc = mx.array(rng.integers(118, 124, size=(E, N, K // 32), dtype=np.uint8))
    b = mx.array(rng.uniform(-0.2, 0.2, (E, N)).astype(np.float32))
    mx.eval(w, sc, b)
    return w, sc, b


def _mxfp4_ops():
    gw, gs, gb = _mxfp4_stack(1)
    uw, us, ub = _mxfp4_stack(2)
    inds, sc = _routing()
    return {
        "moe_glu_gather": (
            (T, K),
            lambda x: kq.moe_glu_gather(x, gw, gs, gb, uw, us, ub, inds),
        ),
        "gather_qmv_bias": (
            (T, R, K),
            lambda x: kq.gather_qmv_bias(x, gw, gs, gb, inds),
        ),
        "gather_qmv_mix_bias": (
            (T, R, K),
            lambda x: kq.gather_qmv_mix_bias(x, gw, gs, gb, inds, sc),
        ),
    }


@metal_only
@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize(
    "op", ["moe_glu_gather", "gather_qmv_bias", "gather_qmv_mix_bias"]
)
def test_moe_mxfp4_activation_views(op, state):
    shape, call = _mxfp4_ops()[op]
    _check_views(_x_views(_acts(shape, mx.float16, 23)), call, state)


@metal_only
@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize(
    "dtype", [mx.float32, mx.float16, mx.bfloat16], ids=["f32", "f16", "bf16"]
)
def test_moe_router_logit_views(dtype, state):
    n_experts = 128
    lw = (mx.random.normal((2 * 7, 2 * n_experts), key=mx.random.key(5)) * 2).astype(
        dtype
    )
    mx.eval(lw)
    for shared in (False, True):
        _check_views(
            _strided(lw),
            lambda v, shared=shared: kq.moe_router_topk(v, 4, True, shared_gate=shared),
            state,
        )


@metal_only
def test_moe_dense_activation_adds_no_node():
    """The call a decode step makes: an unevaluated activation that is
    row-major. Its graph is the op, the ops that feed x and the layout node
    of the indices."""
    gw, uw = _stack("q4_k", 1), _stack("q4_k", 2)
    inds, _ = _routing()
    x = _dense(_acts((T, K), mx.float16, 21)[:T, :K])
    feed = _nodes(x * 1.0)
    assert "Multiply" in feed and "Contiguous" not in feed
    y = kq.moe_glu_gather_kq(x * 1.0, gw, uw, "q4_k", inds)
    assert _nodes(y) == sorted([*feed, "Contiguous", "KQuantMoEGLUKQ"])
    lw = mx.random.normal((7, 128), key=mx.random.key(5))
    mx.eval(lw)
    outs = kq.moe_router_topk(lw * 1.0, 4, True, shared_gate=False)
    assert set(_nodes(*outs)) == {*_nodes(lw * 1.0), "KQuantMoERouterTopK"}


def _refused(call, kind, match):
    with pytest.raises(kind, match=match):
        mx.eval(call())


@metal_only
@pytest.mark.parametrize("codec", ["q4_k", "q8_0"])
def test_moe_strided_stack_is_refused(codec):
    """The gate and up halves of one fused stack are not row-major: the
    expert stride is the fused row count. The op refuses them, evaluated
    (at the call) or not (at eval), and never reads them as dense."""
    gu = _stack(codec, 5, n=2 * N)
    gw, uw = _stack(codec, 1), _stack(codec, 2)
    sg2 = _dense(_stack(codec, 3, n=2 * N)[0])
    sgw = _shexp(codec, 0)
    mx.eval(sg2)
    inds, _ = _routing()
    inds_s, sc_s = _routing(4, S)
    x = _dense(_acts((T, K), mx.float16, 21)[:T, :K])
    h = _dense(_acts((T, R, K), mx.float16, 22)[:, :R, :K])
    hs = _dense(_acts((T, S, K), mx.float16, 22)[:, :S, :K])
    msg = "weights must be row-contiguous"
    calls = {
        "glu_gate": lambda g: kq.moe_glu_gather_kq(x, g, uw, codec, inds),
        "glu_up": lambda g: kq.moe_glu_gather_kq(x, gw, g, codec, inds),
        "qmv": lambda g: kq.gather_qmv_kq(h, g, codec, inds),
        "shexp_gate": lambda g: kq.moe_glu_gather_shexp_kq(
            x, g, uw, sgw, sgw, codec, inds
        ),
        "mix": lambda g: kq.gather_qmv_mix_kq(hs, g, sgw, codec, inds, sc_s),
        "mix_ns": lambda g: kq.gather_qmv_mix_ns_kq(hs, g, codec, inds_s, sc_s),
    }
    for name, call in calls.items():
        for half in (lambda: gu[:, :N], lambda: gu[:, N:]):
            _refused(lambda call=call, half=half: call(half()), ValueError, msg)
            view = half()
            mx.eval(view)
            with pytest.raises(ValueError, match=msg):
                call(view)
        # The same rows in their own buffer run.
        assert _same(call(_dense(gu[:, :N])), call(_dense(gu[:, :N]))), name
    # A shared-expert tensor of every other row.
    shexp = {
        "shexp_gate": lambda v: kq.moe_glu_gather_shexp_kq(
            x, gw, uw, v, sgw, codec, inds
        ),
        "shexp_up": lambda v: kq.moe_glu_gather_shexp_kq(
            x, gw, uw, sgw, v, codec, inds
        ),
        "mix": lambda v: kq.gather_qmv_mix_kq(hs, gw, v, codec, inds, sc_s),
    }
    for call in shexp.values():
        _refused(lambda call=call: call(sg2[0::2]), ValueError, "shared-expert " + msg)
        view = sg2[0::2]
        mx.eval(view)
        with pytest.raises(ValueError, match="shared-expert " + msg):
            call(view)


@metal_only
@pytest.mark.parametrize("state", ["lazy", "evaluated"])
def test_moe_row_major_stack_views_run(state):
    """A band of experts from a larger stack is row-major at an offset. It
    is read in place, evaluated or not."""
    codec = "q4_k"
    big = _stack(codec, 6, experts=2 * E)
    uw = _stack(codec, 2)
    inds, _ = _routing()
    x = _dense(_acts((T, K), mx.float16, 21)[:T, :K])
    want = kq.moe_glu_gather_kq(x, _dense(big[2 : 2 + E]), uw, codec, inds)
    band = big[2 : 2 + E]
    if state == "evaluated":
        mx.eval(band)
    assert _same(kq.moe_glu_gather_kq(x, band, uw, codec, inds), want)


@metal_only
def test_moe_mxfp4_strided_stack_is_refused():
    gw, gs, gb = _mxfp4_stack(1)
    uw, us, ub = _mxfp4_stack(2)
    w2 = mx.concatenate([gw, uw], axis=1)
    s2 = mx.concatenate([gs, us], axis=1)
    mx.eval(w2, s2)
    inds, sc = _routing()
    x = _dense(_acts((T, K), mx.float16, 21)[:T, :K])
    h = _dense(_acts((T, R, K), mx.float16, 22)[:, :R, :K])
    msg = "weights/scales must be row-contiguous"
    calls = [
        lambda w, s: kq.moe_glu_gather(x, w, s, gb, uw, us, ub, inds),
        lambda w, s: kq.moe_glu_gather(x, gw, gs, gb, w, s, ub, inds),
        lambda w, s: kq.gather_qmv_bias(h, w, s, gb, inds),
        lambda w, s: kq.gather_qmv_mix_bias(h, w, s, gb, inds, sc),
    ]
    for call in calls:
        # A strided weight with dense scales, and dense weights with
        # strided scales.
        _refused(lambda call=call: call(w2[:, :N], gs), ValueError, msg)
        _refused(lambda call=call: call(gw, s2[:, N:]), ValueError, msg)
        wv, sv = w2[:, :N], s2[:, N:]
        mx.eval(wv, sv)
        with pytest.raises(ValueError, match=msg):
            call(wv, gs)
        with pytest.raises(ValueError, match=msg):
            call(gw, sv)


# ------------------- sdpa_prefill_block_sparse k and v -------------------


def _bs_case():
    """The block sparse prefill at head dim 256, with keys and values twice
    as wide as the op reads."""
    from test_sdpa import _bs_prefill_lists

    hq, hkv, d, s, length, topk, r = 24, 2, 256, 4096, 64, 16, 4
    offset = s - length
    q = mx.random.normal((1, hq, length, d), key=mx.random.key(1)).astype(mx.bfloat16)
    k2 = mx.random.normal((1, hkv, s, 2 * d), key=mx.random.key(2)).astype(mx.bfloat16)
    v2 = mx.random.normal((1, hkv, s, 2 * d), key=mx.random.key(3)).astype(mx.bfloat16)
    mx.eval(q, k2, v2)
    rng = np.random.default_rng(3)
    sel = np.zeros((length, topk), dtype=np.int64)
    for t in range(length):
        sel[t] = rng.choice((offset + t + 1) // r, size=topk, replace=False)
    pages, pmask, counts = (mx.array(a) for a in _bs_prefill_lists(sel, offset, s))
    return q, k2, v2, pages, pmask, counts, offset


@metal_only
def test_bs_prefill_strided_head_dim():
    """Keys and values whose head dim is every other element of a wider
    buffer. Evaluated, the op copies them. Unevaluated, it cannot see the
    strides when it is built, and the eval refuses them as the other
    attention ops do."""
    q, k2, v2, pages, pmask, counts, offset = _bs_case()

    def call(k, v):
        return kq.sdpa_prefill_block_sparse(
            q, k, v, 0.0625, pages, pmask, counts, offset
        )

    want = call(_dense(k2[..., 0::2]), _dense(v2[..., 0::2]))
    mx.eval(want)
    assert mx.isfinite(want.astype(mx.float32)).all().item()
    kv, vv = k2[..., 0::2], v2[..., 0::2]
    mx.eval(kv, vv)
    assert _same(call(kv, vv), want)
    msg = "k/v head_dim must be contiguous"
    _refused(lambda: call(k2[..., 0::2], _dense(v2[..., 0::2])), RuntimeError, msg)
    _refused(lambda: call(_dense(k2[..., 0::2]), v2[..., 0::2]), RuntimeError, msg)


# ----------------- sparse attention window and pool rows -----------------


def _embedded(a, rows_axis):
    """Views that hold the values of a and are not row-major, as builders of
    a fresh view. The first three have dims that fold into one run, which a
    reshape keeps as a view with an element stride other than 1."""
    junk = (a.astype(mx.float32) * 3 + 1).astype(a.dtype)
    cols = a.shape[-1]
    even = mx.stack([a, junk], axis=-1).reshape(*a.shape[:-1], 2 * cols)
    odd = mx.stack([junk, a], axis=-1).reshape(*a.shape[:-1], 2 * cols)
    flip = _dense(a[..., ::-1, ::-1])
    longer = mx.concatenate([a, junk], axis=rows_axis)
    turned = _dense(mx.swapaxes(a, -1, -2))
    mx.eval(even, odd, flip, longer, turned)
    rows = a.shape[rows_axis]
    pre = (slice(None),) * rows_axis
    return {
        "even_cols": lambda: even[..., 0::2],
        "odd_cols": lambda: odd[..., 1::2],
        "reversed": lambda: flip[..., ::-1, ::-1],
        "prefix": lambda: longer[(*pre, slice(0, rows))],
        "transposed": lambda: mx.swapaxes(turned, -1, -2),
    }


def _check_embedded(views, call, want, state):
    for name, make in views.items():
        view = make()
        if state == "evaluated":
            mx.eval(view)
        assert _same(call(view), want), (name, state)


@metal_only
@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("batch", [1, 2])
def test_sparse_decode_row_views(batch, state):
    from test_sdpa_sparse_decode import _case

    args, _ = _case(
        batch, 16, 2, 128, 24, 300, 40, mx.float16, np.int32, True, True, True
    )
    q, win, pool, idx, scale, kw = args
    mx.eval(q, win, pool, idx)
    want = kq.sdpa_sparse_decode(q, win, pool, idx, scale, **kw)
    _check_embedded(
        _embedded(win, 2),
        lambda v: kq.sdpa_sparse_decode(q, v, pool, idx, scale, **kw),
        want,
        state,
    )
    _check_embedded(
        _embedded(pool, 1),
        lambda v: kq.sdpa_sparse_decode(q, win, v, idx, scale, **kw),
        want,
        state,
    )


@metal_only
@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("batch", [1, 2])
def test_sparse_prefill_row_views(batch, state):
    from test_sdpa_sparse_prefill import _case

    args, _ = _case(
        batch, 16, 20, 128, 40, 300, 48, 16, mx.float16, np.int32, True, True, True
    )
    q, win, pool, idx, scale, band, kw = args
    mx.eval(q, win, pool, idx)
    want = kq.sdpa_sparse_prefill(q, win, pool, idx, scale, band, **kw)
    _check_embedded(
        _embedded(win, 2),
        lambda v: kq.sdpa_sparse_prefill(q, v, pool, idx, scale, band, **kw),
        want,
        state,
    )
    _check_embedded(
        _embedded(pool, 1),
        lambda v: kq.sdpa_sparse_prefill(q, win, v, idx, scale, band, **kw),
        want,
        state,
    )


@metal_only
@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("op", ["decode", "prefill"])
def test_sparse_packed_pool_views(op, batch, state):
    """A packed pool is codes and scales. Each is read with a unit stride in
    a row, and a batch is stepped by the stride of the codes, with an eighth
    of it for the scales. A view of either that steps differently gives the
    values of its copy."""
    from test_sdpa_sparse_decode import _on_grid_pool

    if op == "decode":
        from test_sdpa_sparse_decode import _case

        args, _ = _case(
            batch, 16, 2, 128, 24, 300, 40, mx.float16, np.int32, True, True, True
        )
        q, win, pool, idx, scale, kw = args

        def run(codes, scales):
            return kq.sdpa_sparse_decode(
                q, win, codes, idx, scale, pool_scales=scales, **kw
            )
    else:
        from test_sdpa_sparse_prefill import _case

        args, _ = _case(
            batch, 16, 20, 128, 40, 300, 48, 16, mx.float16, np.int32, True, True, True
        )
        q, win, pool, idx, scale, band, kw = args

        def run(codes, scales):
            return kq.sdpa_sparse_prefill(
                q, win, codes, idx, scale, band, pool_scales=scales, **kw
            )

    codes, scales = kq.latent_fp4_pack(_on_grid_pool(pool))
    mx.eval(q, win, idx, codes, scales)
    want = run(codes, scales)
    # The prefix view of the codes alone steps batches by the longer
    # buffer, and the dense scales do not.
    _check_embedded(_embedded(codes, 1), lambda v: run(v, scales), want, state)
    _check_embedded(_embedded(scales, 1), lambda v: run(codes, v), want, state)


@metal_only
@pytest.mark.parametrize("state", ["lazy", "evaluated"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["f16", "bf16"])
def test_indexer_score_decode_key_views(dtype, state):
    """One sequence passes its keys through a reshape that keeps a view."""
    rng = np.random.default_rng(11)
    heads, ql, rows, d = 64, 2, 1027, 128
    q = mx.array(rng.standard_normal((1, heads, ql, d)) * 0.4).astype(dtype)
    k = mx.array(rng.standard_normal((1, rows, d)) * 0.4).astype(dtype)
    w = mx.array(rng.standard_normal((1, ql, heads)) * 0.2).astype(dtype)
    mx.eval(q, k, w)
    want = kq.dsa_indexer_score_decode(q, k, w, 4106, 4)
    _check_embedded(
        _embedded(k, 1),
        lambda v: kq.dsa_indexer_score_decode(q, v, w, 4106, 4),
        want,
        state,
    )


# ---------------- scales and biases of a quantized cache ----------------


def _q8_case(batch=1, hq=16, hkv=4, rows=256, d=128):
    q = mx.random.normal((batch, hq, 1, d), key=mx.random.key(64)).astype(mx.float16)
    k = mx.random.normal((batch, hkv, rows, d), key=mx.random.key(65))
    v = mx.random.normal((batch, hkv, rows, d), key=mx.random.key(66))
    kw, ks, kb = mx.quantize(k.astype(mx.float16), group_size=64, bits=8)
    vw, vs, vb = mx.quantize(v.astype(mx.float16), group_size=64, bits=8)
    mx.eval(q, kw, ks, kb, vw, vs, vb)
    return q, (kw, ks, kb), (vw, vs, vb)


def _q8_pair_cells(call, ks, kb):
    """call(k_scales, k_biases) with one pair of a quantized cache."""
    msg = "scales and biases need one layout"
    want = call(ks, kb)
    mx.eval(want)
    views_s = _embedded(ks, 2)
    views_b = _embedded(kb, 2)
    # Both as prefix slices of longer buffers, the layout of a cache with
    # spare rows: read in place.
    for state in ("lazy", "evaluated"):
        sv, bv = views_s["prefix"](), views_b["prefix"]()
        if state == "evaluated":
            mx.eval(sv, bv)
        assert _same(call(sv, bv), want), state
    # One of the two as a prefix slice: the head strides differ.
    for state in ("lazy", "evaluated"):
        bv = views_b["prefix"]()
        sv = views_s["prefix"]()
        if state == "evaluated":
            mx.eval(sv, bv)
        _refused(lambda bv=bv: call(ks, bv), RuntimeError, msg)
        _refused(lambda sv=sv: call(sv, kb), RuntimeError, msg)
    # A last dim that is every other element. With memory the builder
    # copies it, without memory the eval refuses it.
    for views, pick in ((views_s, 0), (views_b, 1)):
        for name in ("even_cols", "odd_cols", "reversed"):
            view = views[name]()
            pair = [ks, kb]
            pair[pick] = view
            _refused(lambda pair=pair: call(*pair), RuntimeError, msg)
        view = views["even_cols"]()
        mx.eval(view)
        pair = [ks, kb]
        pair[pick] = view
        assert _same(call(*pair), want)


@metal_only
def test_q8_scale_and_bias_layout_decode_gqa():
    q, (kw, ks, kb), (vw, vs, vb) = _q8_case()
    scale = 128**-0.5

    def call_k(s, b):
        return kq.sdpa_decode_gqa(
            q, kw, vw, scale, k_scales=s, k_biases=b, v_scales=vs, v_biases=vb
        )

    def call_v(s, b):
        return kq.sdpa_decode_gqa(
            q, kw, vw, scale, k_scales=ks, k_biases=kb, v_scales=s, v_biases=b
        )

    _q8_pair_cells(call_k, ks, kb)
    _q8_pair_cells(call_v, vs, vb)


@metal_only
def test_q8_scale_and_bias_layout_cascade():
    q, (kw, ks, kb), (vw, vs, vb) = _q8_case(batch=2, rows=64)
    _, (sw, ss, sb), (tw, ts, tb) = _q8_case(batch=1, rows=511)
    scale = 128**-0.5

    def call(shared_k, priv_k):
        return kq.sdpa_decode_gqa_cascade(
            q,
            sw,
            tw,
            kw,
            vw,
            scale,
            k_shared_scales=shared_k[0],
            k_shared_biases=shared_k[1],
            v_shared_scales=ts,
            v_shared_biases=tb,
            k_priv_scales=priv_k[0],
            k_priv_biases=priv_k[1],
            v_priv_scales=vs,
            v_priv_biases=vb,
        )

    _q8_pair_cells(lambda s, b: call((s, b), (ks, kb)), ss, sb)
    _q8_pair_cells(lambda s, b: call((ss, sb), (s, b)), ks, kb)
