"""gather_mix: fused unsort and score mix of sorted-prefill MoE outputs
against the eager gather, multiply and sum in fp32. Plain kernel (any GPU);
under KQUANT_FORCE_CPU=1 the same cases run the CPU path.
"""

import mlx.core as mx
import pytest

import mlx_kquant as kq

REL_BOUND = {mx.bfloat16: 1e-2, mx.float16: 3e-3}


def _rel(a, b):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    return float(mx.linalg.norm(af - bf) / (mx.linalg.norm(bf) + 1e-9))


def _make(T, k, N, E, dtype, seed):
    key = mx.random.key(seed)
    k0, k1, k2 = mx.random.split(key, 3)
    inds = mx.random.randint(0, E, (T, k), key=k0)
    order = mx.argsort(inds.flatten())
    inv_order = mx.argsort(order)
    y = mx.random.normal((T * k, N), key=k1).astype(dtype)
    scores = mx.softmax(mx.random.normal((T, k), key=k2), axis=-1)
    mx.eval(inv_order, y, scores)
    return y, inv_order, scores


def _ref(y, inv_order, scores):
    T, k = scores.shape
    yt = y.astype(mx.float32)[inv_order].reshape(T, k, -1)
    return (yt * scores[..., None]).sum(-2)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (64, 8, 4096, 288),  # glm5 shape class
        (37, 4, 256, 8),  # odd T, small N
        (5, 2, 1020, 4),  # N not a multiple of the thread span
        (1, 8, 128, 16),  # one token
    ],
)
def test_gather_mix_matches_eager(shape, dtype):
    T, k, N, E = shape
    y, inv_order, scores = _make(T, k, N, E, dtype, seed=T + N)
    out = kq.gather_mix(y, inv_order, scores)
    ref = _ref(y, inv_order, scores)
    mx.eval(out, ref)
    assert out.shape == (T, N) and out.dtype == dtype
    assert _rel(out, ref) < REL_BOUND[dtype], f"{shape} rel {_rel(out, ref):.3e}"
    # int32 indices and a bf16 score tensor are accepted.
    out2 = kq.gather_mix(y, inv_order.astype(mx.int32), scores.astype(mx.bfloat16))
    mx.eval(out2)
    assert _rel(out2, ref) < 2e-2


def test_gather_mix_rejects_bad_args():
    y, inv_order, scores = _make(8, 4, 256, 8, mx.bfloat16, seed=1)
    with pytest.raises(ValueError):
        kq.gather_mix(y, inv_order[:-1], scores)
    with pytest.raises(ValueError):
        kq.gather_mix(y[:, :254], inv_order, scores)
    with pytest.raises(ValueError):
        kq.gather_mix(y, inv_order, scores[:, :2])
