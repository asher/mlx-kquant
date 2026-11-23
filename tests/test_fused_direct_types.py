"""Direct replay of the listed primitive types, one type a case.

Each case records a small graph around one type on the direct list
(fuse_graph's direct_listed) with direct=True, runs it on fresh inputs, the
first call recording and the rest replaying, and compares every output bit
for bit with the plain graph on the same inputs. A second record of the
same key, learned on other inputs, must show the same Metal calls
(fused_direct_calls): a call list that moves with input values fails.
Inputs that vary per call are record inputs (activations, expert ids,
scores, slot tables, key ends); weights, stores and tables are held.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from qmm_probe import force_cpu

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(force_cpu(), reason="direct replay is a Metal path")

NAX = not force_cpu() and kq.nax_available()
nax_only = pytest.mark.skipif(not NAX, reason="NAX GPU route")

E, R = 12, 5
ABSENT = 0x80000000

_W = {}


@pytest.fixture
def lever_cache():
    was = kq.env_cache(True)
    yield
    kq.env_cache(was)


def _bits(a):
    mx.eval(a)
    if a.dtype == mx.bool_:
        return a.astype(mx.uint8)
    return a.view({1: mx.uint8, 2: mx.uint16, 4: mx.uint32}[a.dtype.size])


def _same(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and bool(mx.array_equal(_bits(a), _bits(b)).item())
    )


def _check(build, inputs_of, name, calls=4):
    """Record build(*inputs_of(0)), run it on inputs_of(1..calls) against
    the plain graph, then compare the call list with a second record of
    the same key learned on other inputs. Returns the first record's info."""
    ins = inputs_of(0)
    handle = kq.fuse_graph(ins, list(build(*ins)), direct=True)
    try:
        names = kq.fused_info(handle)[0]
        assert name in names, names
        assert kq.fused_direct_info(handle)["state"] == "new"
        for seed in range(1, calls + 1):
            again = inputs_of(seed)
            got = kq.fused_call(handle, again)
            want = list(build(*again))
            for i, (a, b) in enumerate(zip(got, want, strict=True)):
                assert _same(a, b), (seed, i)
        info = kq.fused_direct_info(handle)
        assert info["state"] == "replay", info
        assert info["replays"] == calls - 1
        assert info["dispatches"] >= 1
        first = kq.fused_direct_calls(handle)
    finally:
        kq.fuse_release(handle)
    ins = inputs_of(100)
    handle = kq.fuse_graph(ins, list(build(*ins)), direct=True)
    try:
        for seed in (101, 102):
            again = inputs_of(seed)
            got = kq.fused_call(handle, again)
            for a, b in zip(got, build(*again), strict=True):
                assert _same(a, b), seed
        assert kq.fused_direct_info(handle)["state"] == "replay"
        second = kq.fused_direct_calls(handle)
    finally:
        kq.fuse_release(handle)
    assert any("groups" in c or "threads" in c for c in first)
    assert second == first
    return info


def _wire(shape, codec, seed):
    """Wire bytes of a normal weight of `shape` (lead dims, N, K)."""
    key = (shape, codec, seed)
    if key not in _W:
        rng = np.random.default_rng(seed)
        w = mx.array((rng.standard_normal(shape) * 0.1).astype(np.float32))
        wq, _ = kq.quantize(w, codec)
        mx.eval(wq)
        _W[key] = wq
    return _W[key]


def _normal(rng, shape, dtype, scale=0.5):
    return mx.array((rng.standard_normal(shape) * scale).astype(np.float32)).astype(
        dtype
    )


def _routes(rng, T):
    """[T, R] distinct expert ids a row; from 8 rows expert 2 takes every
    row and expert 9 three, so tiles of more than two rows form."""
    rows = []
    for t in range(T):
        fixed = ([2, 9] if t < 3 else [2]) if T >= 8 else []
        rest = [e for e in range(E) if e not in fixed]
        rows.append(fixed + [int(e) for e in rng.choice(rest, R - len(fixed), False)])
    return mx.array(np.array(rows, np.uint32))


def _evaled(ins):
    mx.eval(ins)
    return ins


# Expert stores: the router, the store GLU and the store down in one record.

STORE_K, STORE_N = 512, 256  # GLU K inputs to N outputs; the down N to K
STORE_IDS = np.array([1, 3, 4, 8, 9, 11], np.uint32)


def _slots(rng):
    """A slot table over the half store: each packed expert present or
    read in place, by a coin flip."""
    s = np.full((E,), ABSENT, np.uint32)
    for i, e in enumerate(STORE_IDS):
        if rng.random() < 0.7:
            s[e] = i
    return mx.array(s)


@pytest.mark.parametrize("slots_in", [False, True], ids=["held table", "table input"])
@pytest.mark.parametrize("T", [1, 3, 8, 16])
@pytest.mark.parametrize("codec", ["q5_0", "q4_k"])
def test_store_block(codec, T, slots_in, lever_cache):
    if not kq.moe_store_supported(codec, codec):
        pytest.skip(f"no store kernels for {codec} on this device")
    K, N = STORE_K, STORE_N
    gw, uw = _wire((E, N, K), codec, 1), _wire((E, N, K), codec, 2)
    sg, su = _wire((N, K), codec, 3), _wire((N, K), codec, 4)
    dw, sd = _wire((E, K, N), codec, 5), _wire((K, N), codec, 6)
    ids = mx.array(STORE_IDS)
    gp = kq.moe_store_pack(gw, codec, K, ids)
    up = kq.moe_store_pack(uw, codec, K, ids)
    dp = kq.moe_store_pack(dw, codec, N, ids)
    held = _slots(np.random.default_rng(7))
    mx.eval(gp, up, dp, held)

    def build(x, logits, *table):
        slots = table[0] if slots_in else held
        inds, sc = kq.moe_router_topk(logits, R)
        h = kq.moe_store_glu(x, gw, uw, sg, su, codec, inds, slots, gp, up)
        y = kq.moe_store_down(h, dw, sd, codec, inds, sc, slots, dp)
        return [y, h, inds, sc]

    def inputs_of(seed):
        rng = np.random.default_rng(seed)
        x = _normal(rng, (T, K), mx.bfloat16)
        logits = mx.array(rng.standard_normal((T, E + 1)).astype(np.float32))
        return _evaled([x, logits] + ([_slots(rng)] if slots_in else []))

    ins = inputs_of(0)
    probe = kq.fuse_graph(ins, build(*ins))
    names = kq.fused_info(probe)[0]
    kq.fuse_release(probe)
    assert {"KQuantMoERouterTopK", "KQuantMoeStoreGLU"} <= set(names), names
    _check(build, inputs_of, "KQuantMoeStoreDown")


# Fused MoE gathers.

MIX_IN, MIX_OUT = 256, 512


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
@pytest.mark.parametrize("T", [1, 3, 6, 8, 16])
@pytest.mark.parametrize("codec", ["q4_k", "iq4_xs"])
def test_gather_qmv_mix(codec, T, dtype, lever_cache):
    w = _wire((E, MIX_OUT, MIX_IN), codec, 11)
    sw = _wire((MIX_OUT, MIX_IN), codec, 12)

    def build(x, inds, sc):
        return [kq.gather_qmv_mix_kq(x, w, sw, codec, inds, sc)]

    def inputs_of(seed):
        rng = np.random.default_rng(seed)
        x = _normal(rng, (T, R + 1, MIX_IN), dtype)
        sc = mx.array(rng.uniform(0.05, 0.3, size=(T, R + 1)).astype(np.float32))
        return _evaled([x, _routes(rng, T), sc])

    info = _check(build, inputs_of, "KQuantGatherQMVMixKQ")
    keys = kq.kernel_keys()
    if NAX and T >= 8:
        # The tiled down: plan, tile and combine, with per-call temporaries.
        assert any("_gather_qmv_mix_tl_" in k for k in keys)
        assert info["dispatches"] >= 3
    elif NAX and T >= 5:
        assert any("_gather_qmv_mix_grp" in k for k in keys)


@pytest.mark.parametrize(
    "act", [("silu", {}), ("swiglu_oai", {"limit": 7.0, "alpha": 1.702})], ids=str
)
@pytest.mark.parametrize("T", [1, 3, 6, 16])
def test_moe_glu_gather_shexp(T, act, lever_cache):
    codec, K, N = "q4_k", 512, 256
    gw, uw = _wire((E, N, K), codec, 21), _wire((E, N, K), codec, 22)
    sg, su = _wire((N, K), codec, 23), _wire((N, K), codec, 24)
    name, kw = act

    def build(x, inds):
        return [
            kq.moe_glu_gather_shexp_kq(x, gw, uw, sg, su, codec, inds, act=name, **kw)
        ]

    def inputs_of(seed):
        rng = np.random.default_rng(seed)
        return _evaled([_normal(rng, (T, K), mx.bfloat16), _routes(rng, T)])

    _check(build, inputs_of, "KQuantMoEGLUShexpKQ")


@pytest.mark.parametrize(
    "scoring,dtype",
    [("softmax", mx.float32), ("sigmoid", mx.bfloat16)],
    ids=["softmax", "sigmoid"],
)
@pytest.mark.parametrize("T", [1, 8])
def test_router_topk(T, scoring, dtype, lever_cache):
    n, k = 64, 8
    rng = np.random.default_rng(3)
    bias = mx.array((rng.standard_normal(n) * 0.1).astype(np.float32))
    mx.eval(bias)
    kw = {}
    if scoring == "sigmoid":
        kw = {"bias": bias, "scoring": "sigmoid", "scale": 2.5}

    def build(logits):
        return list(kq.moe_router_topk(logits, k, **kw))

    def inputs_of(seed):
        rng = np.random.default_rng(seed)
        return _evaled([_normal(rng, (T, n + 1), dtype, 1.0)])

    _check(build, inputs_of, "KQuantMoERouterTopK")


# Attention.


@pytest.mark.parametrize("ends", [False, True], ids=["full", "ends input"])
@pytest.mark.parametrize("kL", [512, 2048])
@pytest.mark.parametrize("qL", [1, 4])
def test_sdpa_decode_gqa(qL, kL, ends, lever_cache):
    Hq, Hkv, D = 8, 2, 128

    def build(q, k, v, *e):
        kw = {"ends": e[0]} if ends else {}
        return [kq.sdpa_decode_gqa(q, k, v, D**-0.5, **kw)]

    def inputs_of(seed):
        rng = np.random.default_rng(seed)
        ins = [
            _normal(rng, (1, Hq, qL, D), mx.bfloat16, 1.0),
            _normal(rng, (1, Hkv, kL, D), mx.bfloat16, 1.0),
            _normal(rng, (1, Hkv, kL, D), mx.bfloat16, 1.0),
        ]
        if ends:
            ins.append(mx.array([int(rng.integers(kL // 2, kL + 1))], mx.int32))
        return _evaled(ins)

    _check(build, inputs_of, "KQuantSDPAGQA")


@pytest.mark.parametrize("mask", [None, "causal"], ids=["no mask", "causal"])
@pytest.mark.parametrize("kL", [256, 1536])
@pytest.mark.parametrize("L", [1, 4, 16])
def test_scaled_dot_product_attention(L, kL, mask, lever_cache):
    Hq, Hkv, D = 8, 2, 128

    def build(q, k, v):
        return [mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)]

    def inputs_of(seed):
        rng = np.random.default_rng(seed)
        return _evaled(
            [
                _normal(rng, (1, Hq, L, D), mx.bfloat16, 1.0),
                _normal(rng, (1, Hkv, kL, D), mx.bfloat16, 1.0),
                _normal(rng, (1, Hkv, kL, D), mx.bfloat16, 1.0),
            ]
        )

    _check(build, inputs_of, "ScaledDotProductAttention")


@pytest.mark.parametrize("prefix", [False, True], ids=["full rows", "causal prefix"])
@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("L", [1, 4])
def test_dsa_topk_indices(L, n, prefix, lever_cache):
    def build(scores):
        return [kq.dsa_topk_indices(scores, 512, True, prefix)]

    def inputs_of(seed):
        rng = np.random.default_rng(seed)
        return _evaled([_normal(rng, (1, 1, L, n), mx.bfloat16, 1.0)])

    _check(build, inputs_of, "KQDsaTopKIndices")


# Dense kernels.


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=str)
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_hc_lowrank_mix(rows, dtype, lever_cache):
    # float32 streams with q8_0 wire and a float32 inject take the folded
    # form where the epilogue runs its simdgroup kernel (NAX by default).
    D, LR, HC = 256, 64, 4
    rng = np.random.default_rng(41)
    gamma = mx.array((rng.standard_normal(HC * D) * 0.3 + 1.0).astype(np.float32))
    gamma = gamma.astype(mx.bfloat16)
    wd = _wire((LR, HC * D), "q8_0", 42)
    wu = _wire((HC * D, LR), "q8_0", 43)
    wi = mx.array((rng.standard_normal((HC, HC * D)) * 0.02).astype(np.float32))
    mx.eval(gamma, wi)

    def build(h):
        return list(kq.hc_lowrank_mix(h, gamma, wd, wi, wu, 1e-6))

    def inputs_of(seed):
        rng = np.random.default_rng(seed)
        return _evaled([_normal(rng, (rows, HC, D), dtype, 1.5)])

    _check(build, inputs_of, "KQuantHcLowrankMix")


@pytest.mark.parametrize(
    "xt,wt",
    [(mx.bfloat16, mx.bfloat16), (mx.float32, mx.bfloat16), (mx.float16, mx.float32)],
    ids=["bf16", "f32 x", "f32 w"],
)
@pytest.mark.parametrize("M", [1, 4, 16])
def test_skinny_matmul(M, xt, wt, lever_cache):
    K, N = 1024, 64
    w = _normal(np.random.default_rng(51), (N, K), wt)
    mx.eval(w)

    def build(x):
        return [kq.skinny_matmul(x, w)]

    def inputs_of(seed):
        return _evaled([_normal(np.random.default_rng(seed), (M, K), xt)])

    _check(build, inputs_of, "KQuantSkinnyMV")


@nax_only
@pytest.mark.parametrize("M", [2, 6, 16])
def test_quantized_matmul_multi(M, lever_cache):
    codec, K = "q4_k", 512
    ws = [_wire((n, K), codec, 60 + i) for i, n in enumerate((1024, 640, 48, 48))]

    def build(x):
        return kq.quantized_matmul_multi(x, ws, codec)

    def inputs_of(seed):
        return _evaled([_normal(np.random.default_rng(seed), (M, K), mx.bfloat16)])

    _check(build, inputs_of, "KQuantMatmulMulti")


# mlx types the direct list names that the other cases do not reach.

_AXPB = mx.fast.metal_kernel(
    name="kq_test_direct_axpb",
    input_names=["a", "b"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        if (HALF == 0 || i % 2 == 0) {
            out[i] = a[i] * T(2) + b[i % NB];
        }
    """,
)


@pytest.mark.parametrize("init", [None, 0.25], ids=["no init", "init value"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16], ids=str)
def test_custom_kernel(dtype, init, lever_cache):
    # With an init value the kernel writes every other output; the rest
    # keep the value mlx fills before the dispatch.
    n, nb = 4096, 64

    def build(a, b):
        return _AXPB(
            inputs=[a, b],
            template=[("T", dtype), ("NB", nb), ("HALF", int(init is not None))],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n,)],
            output_dtypes=[dtype],
            init_value=init,
        )

    def inputs_of(seed):
        rng = np.random.default_rng(seed)
        return _evaled([_normal(rng, (n,), dtype), _normal(rng, (nb,), dtype)])

    _check(build, inputs_of, "CustomKernel")


def test_less_equal(lever_cache):
    m, n, kL, L = 4, 512, 64, 5
    cols = mx.arange(kL, dtype=mx.int32)[None]
    mx.eval(cols)

    def build(x, t, pos):
        le = x <= t
        return [le.astype(x.dtype) * x + 1.0, le, cols <= pos]

    def inputs_of(seed):
        rng = np.random.default_rng(seed)
        pos = rng.integers(0, kL, size=(L, 1)).astype(np.int32)
        return _evaled(
            [
                _normal(rng, (m, n), mx.bfloat16, 1.0),
                _normal(rng, (m, 1), mx.bfloat16, 0.5),
                mx.array(pos),
            ]
        )

    _check(build, inputs_of, "LessEqual")
