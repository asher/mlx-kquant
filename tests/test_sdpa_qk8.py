"""sdpa_qk8_inputs and sdpa_qk8: int8 Q K^T attention at head dim 128.

Every numeric bound below was fixed before the kernel first ran, from a
reference figure times a margin named in the test's docstring. The figures
are the relative L2 distances of a reference implementation of the same
scheme on random normal inputs at 1024 rows: 1.97e-3 against its float
formula, 9.40e-3 against MLX's attention.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="sdpa_qk8 runs on Metal; no CPU path.",
)

NAX = (
    not os.environ.get("KQUANT_FORCE_CPU")
    and mx.metal.is_available()
    and kq.nax_available()
)

requires_nax = pytest.mark.skipif(
    not NAX, reason="sdpa_qk8 needs tensor-op (NAX) hardware."
)

big = pytest.mark.skipif(
    not os.environ.get("KQUANT_BIG_TESTS"),
    reason="holds a 4.7 GB array; set KQUANT_BIG_TESTS=1 to run",
)

D = 128
SCALE = D**-0.5

# Reference figures (relative L2 on random normal q, k, v) and margins.
REF_VS_FORMULA = 1.97e-3
REL_FORMULA = 2 * REF_VS_FORMULA  # margin 2x
REF_VS_DENSE = 9.40e-3
REL_DENSE_MAX = 2e-2  # margin about 2.1x
REL_DENSE_MIN = REF_VS_DENSE / 2  # margin 2x below: the int8 path ran


def _bf16(rng, shape):
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)


def _qkv(seed, b, h, hk, ql, kl):
    rng = np.random.default_rng(seed)
    return (
        _bf16(rng, (b, h, ql, D)),
        _bf16(rng, (b, hk, kl, D)),
        _bf16(rng, (b, hk, kl, D)),
    )


def _rel(x, ref):
    x = np.asarray(x, np.float64)
    ref = np.asarray(ref, np.float64)
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _formula64(q8, qs, k8, ks, v):
    """softmax((q8 qs)(k8 ks)^T) v in float64, k and v repeated over GQA."""
    qd = np.array(q8).astype(np.float64) * np.array(qs)[..., None]
    kd = np.array(k8).astype(np.float64) * np.array(ks)[..., None]
    vd = _f64(v)
    rep = qd.shape[1] // kd.shape[1]
    out = np.empty(qd.shape, np.float64)
    for b in range(qd.shape[0]):
        for h in range(qd.shape[1]):
            s = qd[b, h] @ kd[b, h // rep].T
            p = np.exp(s - s.max(axis=-1, keepdims=True))
            out[b, h] = (p @ vd[b, h // rep]) / p.sum(axis=-1, keepdims=True)
    return out


# The inputs op against its formula in MLX ops.


def _codes_ref(f):
    m = mx.maximum(mx.max(mx.abs(f), axis=-1, keepdims=True), 1e-30)
    codes = mx.clip(mx.round(f * (127.0 / m)), -127, 127)
    return codes, (m / 127.0)[..., 0]


def test_inputs_match_their_formula_on_strided_views():
    """q and k as the [B, L, H, D] transposes callers pass. Codes within 1
    of the formula and at most 0.2 percent differing (a reference kernel of
    the same arithmetic differed in 0.08 to 0.16 percent: the GPU's
    reciprocal rounds differently from MLX's division); scales within 1e-6
    relative."""
    rng = np.random.default_rng(11)
    rows, heads = 1922, 4
    q = _bf16(rng, (1, rows, heads, D)).transpose(0, 2, 1, 3)
    k = (_bf16(rng, (1, rows, heads, D)) + 0.5).transpose(0, 2, 1, 3)
    q8, qs, k8, ks = kq.sdpa_qk8_inputs(q, k, SCALE)
    assert q8.shape == (1, heads, rows, D) and q8.dtype == mx.int8
    assert k8.shape == (1, heads, rows, D) and k8.dtype == mx.int8
    assert qs.shape == (1, heads, rows) and qs.dtype == mx.float32
    assert ks.shape == (1, heads, rows) and ks.dtype == mx.float32

    mean = mx.mean(k, axis=2, keepdims=True).astype(mx.float32)
    q_ref, qs_ref = _codes_ref(q.astype(mx.float32) * SCALE)
    k_ref, ks_ref = _codes_ref(k.astype(mx.float32) - mean)
    for got, want, s_got, s_want in (
        (q8, q_ref, qs, qs_ref),
        (k8, k_ref, ks, ks_ref),
    ):
        diff = np.abs(np.array(got).astype(np.int32) - np.array(want))
        assert diff.max() <= 1
        assert (diff != 0).mean() <= 0.002
        np.testing.assert_allclose(np.array(s_got), np.array(s_want), rtol=1e-6)


# The attention call.

ROWS = {
    "aligned_4096": (1, 4, 4, 4096, 4096),
    "ragged_1922": (1, 4, 4, 1922, 1922),
    "short_37": (1, 4, 4, 37, 37),
    "gqa_batch2_512": (2, 4, 2, 512, 512),
    "ragged_keys_300": (1, 4, 4, 512, 300),
}


@requires_nax
@pytest.mark.parametrize("name", sorted(ROWS))
def test_call_matches_its_formula(name):
    """Relative L2 against the float64 formula on the op's own int8 inputs
    and scales, within 2x the reference's 1.97e-3; against MLX's attention
    on the bfloat16 inputs within 2e-2 (reference 9.40e-3, about 2.1x), and
    at least 9.40e-3 / 2 from it, so the int8 path ran."""
    b, h, hk, ql, kl = ROWS[name]
    q, k, v = _qkv(sum(ROWS[name]), b, h, hk, ql, kl)
    q8, qs, k8, ks = kq.sdpa_qk8_inputs(q, k, SCALE)
    out = kq.sdpa_qk8(q8, qs, k8, ks, v)
    dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
    mx.eval(out, dense, q8, qs, k8, ks)
    assert out.shape == (b, h, ql, D) and out.dtype == mx.bfloat16
    got = _f64(out)
    assert np.isfinite(got).all()
    assert _rel(got, _formula64(q8, qs, k8, ks, v)) < REL_FORMULA
    rel_dense = _rel(got, _f64(dense))
    assert REL_DENSE_MIN < rel_dense < REL_DENSE_MAX


@requires_nax
def test_chunked_rows_have_the_whole_calls_bits():
    """The query rows in 4096-row chunks, the last ragged, as callers that
    pause between chunks run them, and one slice off the 64-row blocks:
    each equals its rows of the whole call bit for bit."""
    rows = 9000
    q, k, v = _qkv(7, 1, 4, 4, rows, rows)
    q8, qs, k8, ks = kq.sdpa_qk8_inputs(q, k, SCALE)
    whole = kq.sdpa_qk8(q8, qs, k8, ks, v)
    mx.eval(whole)
    whole = np.array(whole.view(mx.uint16))
    spans = [(r, min(r + 4096, rows)) for r in range(0, rows, 4096)]
    for r0, r1 in spans + [(1000, 3001)]:
        part = kq.sdpa_qk8(q8[:, :, r0:r1], qs[:, :, r0:r1], k8, ks, v)
        np.testing.assert_array_equal(
            np.array(part.view(mx.uint16)), whole[:, :, r0:r1]
        )


@requires_nax
def test_zero_query_rows_give_an_empty_output():
    q, k, v = _qkv(9, 1, 4, 4, 0, 64)
    out = kq.sdpa_qk8(*kq.sdpa_qk8_inputs(q, k, SCALE), v)
    mx.eval(out)
    assert out.shape == (1, 4, 0, D)


# Guards. Each input fails only the named guard, at build time, so the
# tests run on any Metal GPU and evaluate nothing.


def _inputs_args():
    q, k, _ = _qkv(21, 1, 4, 2, 64, 96)
    return dict(q=q, k=k, scale=SCALE)


def _call_args():
    rng = np.random.default_rng(23)
    b, h, hk, ql, kl = 1, 4, 2, 64, 96
    return dict(
        q8=mx.zeros((b, h, ql, D), mx.int8),
        q_scale=mx.ones((b, h, ql), mx.float32),
        k8=mx.zeros((b, hk, kl, D), mx.int8),
        k_scale=mx.ones((b, hk, kl), mx.float32),
        v=_bf16(rng, (b, hk, kl, D)),
    )


def _zeros_like_shape(x, shape):
    return mx.zeros(shape, x.dtype)


INPUTS_GUARDS = {
    "ndim": ("must be 4-D", {"q": lambda x: x[0]}),
    "head_dim": (
        "only head dim 128",
        {"q": lambda x: _zeros_like_shape(x, x.shape[:3] + (64,))},
    ),
    "batch": ("share a batch size", {"k": lambda x: mx.concatenate([x, x])}),
    "heads": ("must be a multiple of", {"k": lambda x: mx.repeat(x[:, :1], 3, 1)}),
    "float16": ("must be bfloat16", {"q": lambda x: x.astype(mx.float16)}),
    "float32": ("must be bfloat16", {"k": lambda x: x.astype(mx.float32)}),
    "scale": ("scale must be finite", {"scale": float("nan")}),
}

CALL_GUARDS = {
    "ndim": ("must be 4-D", {"k8": lambda x: x[0]}),
    "head_dim": (
        "only head dim 128",
        {
            "k8": lambda x: _zeros_like_shape(x, x.shape[:3] + (64,)),
            "v": lambda x: _zeros_like_shape(x, x.shape[:3] + (64,)),
        },
    ),
    "batch": (
        "share a batch size",
        {
            "q8": lambda x: mx.concatenate([x, x]),
            "q_scale": lambda x: mx.concatenate([x, x]),
        },
    ),
    "heads": (
        "must be a multiple of",
        {"q8": lambda x: x[:, :3], "q_scale": lambda x: x[:, :3]},
    ),
    "codes_dtype": ("must be int8", {"q8": lambda x: x.astype(mx.uint8)}),
    "no_keys": ("at least one key row", None),
    "q_scale": ("q_scale must be", {"q_scale": lambda x: x[..., :-1]}),
    "k_scale": ("k_scale must be", {"k_scale": lambda x: x[:, :1]}),
    "v_rows": ("v must have the shape of k8", {"v": lambda x: x[:, :, :-1]}),
    "v_dtype": ("v must be bfloat16", {"v": lambda x: x.astype(mx.float16)}),
}


def _apply(args, change):
    for name, fn in change.items():
        args[name] = fn(args[name]) if callable(fn) else fn
    return args


@pytest.mark.parametrize("name", sorted(INPUTS_GUARDS))
def test_inputs_guard_refuses(name):
    match, change = INPUTS_GUARDS[name]
    with pytest.raises(ValueError, match=match):
        kq.sdpa_qk8_inputs(**_apply(_inputs_args(), change))


@pytest.mark.parametrize("name", sorted(CALL_GUARDS))
def test_call_guard_refuses(name):
    match, change = CALL_GUARDS[name]
    args = _call_args()
    if change is None:  # no keys: k8, k_scale and v with zero rows
        args["k8"] = args["k8"][:, :, :0]
        args["k_scale"] = args["k_scale"][:, :, :0]
        args["v"] = args["v"][:, :, :0]
    else:
        args = _apply(args, change)
    with pytest.raises(ValueError, match=match):
        kq.sdpa_qk8(**args)


@pytest.mark.skipif(NAX, reason="refusal path of GPUs without NAX units.")
def test_refuses_without_nax():
    with pytest.raises(RuntimeError, match="NAX"):
        kq.sdpa_qk8(**_call_args())


# Strided operands past 2^31 elements of extent.


def strided_v_case(rows, q_rows=128, heads=56):
    """q, k and v as views of one fused [1, rows, 3 * heads * D] product,
    as a video model's attention hands them over. Returns the call on the
    views and on contiguous copies, both evaluated."""
    inner = heads * D
    block = 1031  # prime, so the tiled rows do not line up with the tiles
    rng = np.random.default_rng(31)
    base = _bf16(rng, (1, block, 3 * inner))
    qkv = mx.tile(base, (1, -(-rows // block), 1))[:, :rows]
    mx.eval(qkv)

    def view(qkv, i, n):
        return (
            qkv[:, :n, i * inner : (i + 1) * inner]
            .reshape(1, n, heads, D)
            .transpose(0, 2, 1, 3)
        )

    q, k, v = view(qkv, 0, q_rows), view(qkv, 1, rows), view(qkv, 2, rows)
    q8, qs, k8, ks = kq.sdpa_qk8_inputs(q, k, SCALE)
    mx.eval(q8, qs, k8, ks)
    got = kq.sdpa_qk8(q8, qs, k8, ks, v)
    mx.eval(got)
    got = np.array(got.view(mx.uint16))
    vc = mx.contiguous(v)
    del qkv, q, k, v
    mx.eval(vc)
    want = kq.sdpa_qk8(q8, qs, k8, ks, vc)
    mx.eval(want)
    return got, np.array(want.view(mx.uint16))


@big
@requires_nax
def test_strided_v_past_2_31_elements_matches_a_contiguous_copy():
    """109087 rows at 56 heads: the v view spans 2.35e9 elements."""
    got, want = strided_v_case(109087)
    np.testing.assert_array_equal(got, want)
