"""route_shed: slot remap + residency shed for streamed MoE decode.

The op contract (shed all non-resident, mass-preserving renorm over kept,
misses front-packed descending-score) is pinned by a pure-python reference;
the CPU eval must match it bit-for-bit and the GPU kernel must match the CPU
eval bit-for-bit (same f32 accumulation order).
"""

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

gpu = pytest.mark.skipif(
    not mx.metal.is_available(), reason="GPU parity requires Metal"
)


def ref_route_shed(indices, scores, slot_table):
    """Pure-python reference, mirroring the kernel's arithmetic order."""
    T, R = indices.shape
    E = slot_table.shape[0]
    slots = np.zeros((T, R), np.uint32)
    mix = np.zeros((T, R), np.float32)
    miss_ids = np.full((T, R), -1, np.int32)
    miss_scores = np.zeros((T, R), np.float32)
    for t in range(T):
        slot_of = [int(slot_table[e]) if 0 <= e < E else -1 for e in indices[t]]
        s_all = np.float32(0.0)
        s_kept = np.float32(0.0)
        first_kept = 0
        have_kept = False
        for r in range(R):
            sv = np.float32(scores[t, r])
            s_all = np.float32(s_all + sv)
            if slot_of[r] >= 0:
                s_kept = np.float32(s_kept + sv)
                if not have_kept:
                    first_kept = slot_of[r]
                    have_kept = True
        renorm = np.float32(s_all / s_kept) if s_kept > 0 else np.float32(0.0)
        misses = []
        for r in range(R):
            if slot_of[r] >= 0:
                slots[t, r] = slot_of[r]
                mix[t, r] = np.float32(np.float32(scores[t, r]) * renorm)
            else:
                slots[t, r] = first_kept
                misses.append((int(indices[t, r]), np.float32(scores[t, r])))
        # Stable descending-score order.
        misses.sort(key=lambda m: -m[1])
        for i, (e, sv) in enumerate(misses):
            miss_ids[t, i] = e
            miss_scores[t, i] = sv
    return slots, mix, miss_ids, miss_scores


def make_case(T=4, R=8, E=256, resident_frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    indices = np.stack([rng.choice(E, size=R, replace=False) for _ in range(T)]).astype(
        np.uint32
    )
    scores = rng.uniform(0.01, 1.0, size=(T, R)).astype(np.float32)
    slot_table = np.full(E, -1, np.int32)
    resident = rng.choice(E, size=int(E * resident_frac), replace=False)
    slot_table[resident] = np.arange(len(resident), dtype=np.int32)
    return indices, scores, slot_table


def run_op(indices, scores, slot_table, stream=None):
    kwargs = {} if stream is None else {"stream": stream}
    outs = kq.route_shed(
        mx.array(indices), mx.array(scores), mx.array(slot_table), **kwargs
    )
    mx.eval(*outs)
    return [np.array(o) for o in outs]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("resident_frac", [0.0, 0.3, 0.7, 1.0])
def test_cpu_matches_reference(seed, resident_frac):
    indices, scores, slot_table = make_case(resident_frac=resident_frac, seed=seed)
    got = run_op(indices, scores, slot_table, stream=mx.cpu)
    want = ref_route_shed(indices, scores, slot_table)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, w)


def test_all_resident_is_identity():
    indices, scores, slot_table = make_case(resident_frac=1.0, seed=1)
    slots, mix, miss_ids, miss_scores = run_op(
        indices, scores, slot_table, stream=mx.cpu
    )
    np.testing.assert_array_equal(slots, slot_table[indices.astype(np.int64)])
    np.testing.assert_allclose(mix, scores, rtol=1e-6)
    assert (miss_ids == -1).all()
    assert (miss_scores == 0).all()


def test_all_miss_row_zero_mix():
    indices, scores, slot_table = make_case(resident_frac=0.0, seed=2)
    slots, mix, miss_ids, miss_scores = run_op(
        indices, scores, slot_table, stream=mx.cpu
    )
    assert (mix == 0).all()
    assert (slots == 0).all()
    assert (miss_ids >= 0).all()
    # Descending score, all entries reported.
    assert (np.diff(miss_scores, axis=-1) <= 0).all()
    np.testing.assert_allclose(
        np.sort(miss_scores, axis=-1), np.sort(scores, axis=-1), rtol=0
    )


def test_mass_preserving_renorm():
    indices, scores, slot_table = make_case(resident_frac=0.5, seed=3)
    _, mix, _, _ = run_op(indices, scores, slot_table, stream=mx.cpu)
    np.testing.assert_allclose(mix.sum(axis=-1), scores.sum(axis=-1), rtol=1e-5)


def test_shed_entries_reuse_first_kept_slot():
    E = 16
    indices = np.array([[3, 5, 7, 9]], np.uint32)
    scores = np.array([[0.4, 0.3, 0.2, 0.1]], np.float32)
    slot_table = np.full(E, -1, np.int32)
    slot_table[5] = 11  # only expert 5 resident
    slots, mix, miss_ids, miss_scores = run_op(
        indices, scores, slot_table, stream=mx.cpu
    )
    np.testing.assert_array_equal(slots, [[11, 11, 11, 11]])
    np.testing.assert_allclose(mix[0, 1], 1.0, rtol=1e-6)  # 0.3 * (1.0/0.3)
    assert mix[0, 0] == mix[0, 2] == mix[0, 3] == 0
    np.testing.assert_array_equal(miss_ids[0], [3, 7, 9, -1])
    np.testing.assert_allclose(miss_scores[0], [0.4, 0.2, 0.1, 0.0])


def test_input_validation():
    indices, scores, slot_table = make_case()
    with pytest.raises(ValueError):
        kq.route_shed(
            mx.array(indices.astype(np.int32)),
            mx.array(scores),
            mx.array(slot_table),
        )
    with pytest.raises(ValueError):
        kq.route_shed(
            mx.array(indices),
            mx.array(scores.astype(np.float16)),
            mx.array(slot_table),
        )
    with pytest.raises(ValueError):
        kq.route_shed(
            mx.array(indices),
            mx.array(scores),
            mx.array(slot_table.astype(np.int16)),
        )
    with pytest.raises(ValueError):
        kq.route_shed(
            mx.array(np.zeros((1, 65), np.uint32)),
            mx.array(np.zeros((1, 65), np.float32)),
            mx.array(slot_table),
        )


@gpu
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("shape", [(1, 8), (4, 8), (33, 4), (2, 64)])
def test_gpu_matches_cpu_bitwise(seed, shape):
    T, R = shape
    indices, scores, slot_table = make_case(T=T, R=R, resident_frac=0.6, seed=seed)
    cpu = run_op(indices, scores, slot_table, stream=mx.cpu)
    dev = run_op(indices, scores, slot_table, stream=mx.gpu)
    for c, g in zip(cpu, dev, strict=True):
        np.testing.assert_array_equal(c, g)
