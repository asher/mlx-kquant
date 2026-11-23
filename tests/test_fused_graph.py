"""fuse_graph and fused_call against the lazy graph they were recorded from.

A recorded graph runs the primitives of the graph it was built from, in
their order, on the same operands, so its outputs carry the bits of the
plain graph on any input of the recorded shapes. The cases here record a
small layer of two quantized matmuls of different codecs with shape glue,
a dtype round trip and a residual between them, at 1 and 8 rows.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from qmm_probe import force_cpu

import mlx_kquant as kq

K = 512
# (codec of the first weight, codec of the second): the quant mixes.
MIXES = [
    ("q4_k", "q8_0"),
    ("iq4_xs", "q6_k"),
    ("q4_0", "q3_k"),
    ("q5_k", "q4_1"),
    ("q2_k", "iq4_nl"),
]
DTYPES = [mx.bfloat16, mx.float16]
ROWS = [1, 8]

_W = {}


def _weight(codec, seed):
    key = (codec, seed)
    if key not in _W:
        wf = mx.random.normal((K, K), key=mx.random.key(31 * seed + 5)) * 0.1
        w, s = kq.quantize(wf, codec)
        mx.eval(w, s)
        _W[key] = w
    return _W[key]


def _x(m, dtype, seed):
    return mx.random.normal((m, K), key=mx.random.key(seed + 11 * m)).astype(dtype)


def _layer(x, w1, c1, w2, c2):
    """(output, an inner array): two matmuls with glue between them."""
    m = x.shape[0]
    sc = mx.zeros((1,), mx.uint8)
    a = kq.quantized_matmul(x, w1, sc, c1, transpose=True)
    b = a.astype(mx.float32).reshape(m, 2, K // 2).transpose(0, 2, 1)
    b = mx.contiguous(b).reshape(m, K).astype(x.dtype)
    c = kq.quantized_matmul(b, w2, sc, c2, transpose=True)
    return c + x, a


def _bits(a):
    mx.eval(a)
    return a.view({2: mx.uint16, 4: mx.uint32}[a.dtype.size])


def _same(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and bool(mx.array_equal(_bits(a), _bits(b)).item())
    )


def _record(c1, c2, m, dtype, **kw):
    w1, w2 = _weight(c1, 0), _weight(c2, 1)
    x0 = _x(m, dtype, 0)
    handle = kq.fuse_graph([x0], list(_layer(x0, w1, c1, w2, c2)), **kw)
    return handle, w1, w2


@pytest.mark.parametrize("mix", MIXES, ids=lambda m: "+".join(m))
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("m", ROWS)
def test_bits_match_the_plain_graph(mix, dtype, m):
    c1, c2 = mix
    handle, w1, w2 = _record(c1, c2, m, dtype)
    try:
        for seed in (1, 2):
            x = _x(m, dtype, seed)
            want = _layer(x, w1, c1, w2, c2)
            got = kq.fused_call(handle, [x])
            assert len(got) == 2
            assert _same(got[0], want[0])
            assert _same(got[1], want[1])
    finally:
        kq.fuse_release(handle)


@pytest.mark.parametrize("m", ROWS)
def test_calls_chained_in_one_graph(m):
    # Each call reads the one before it, all in one eval.
    c1, c2 = "q4_k", "q8_0"
    handle, w1, w2 = _record(c1, c2, m, mx.bfloat16)
    try:
        x = _x(m, mx.bfloat16, 3)
        want = got = x
        for _ in range(3):
            want = _layer(want, w1, c1, w2, c2)[0]
            got = kq.fused_call(handle, [got])[0]
        assert _same(got, want)
    finally:
        kq.fuse_release(handle)


def test_weights_are_held_not_passed():
    handle, _, _ = _record("q4_k", "q8_0", 8, mx.bfloat16)
    try:
        names, constants, used = kq.fused_info(handle)
        assert names.count("KQuantMatmul") == 2
        # The two weights at least; the caller passes the input alone.
        assert constants >= 2
        assert list(used) == [True]
    finally:
        kq.fuse_release(handle)


def test_an_input_the_graph_does_not_read_is_reported():
    c1, c2 = "q4_k", "q8_0"
    w1, w2 = _weight(c1, 0), _weight(c2, 1)
    x0 = _x(8, mx.bfloat16, 0)
    spare = _x(8, mx.bfloat16, 9)
    handle = kq.fuse_graph([x0, spare], list(_layer(x0, w1, c1, w2, c2)))
    try:
        assert list(kq.fused_info(handle)[2]) == [True, False]
    finally:
        kq.fuse_release(handle)


def test_an_input_of_another_shape_or_dtype_is_refused():
    handle, _, _ = _record("q4_k", "q8_0", 8, mx.bfloat16)
    try:
        with pytest.raises(ValueError, match="shape or dtype"):
            kq.fused_call(handle, [_x(4, mx.bfloat16, 1)])
        with pytest.raises(ValueError, match="shape or dtype"):
            kq.fused_call(handle, [_x(8, mx.float16, 1)])
        with pytest.raises(ValueError, match="input count"):
            kq.fused_call(handle, [])
    finally:
        kq.fuse_release(handle)


def test_a_graph_with_no_work_is_refused():
    x = _x(8, mx.bfloat16, 0)
    y = x + 1
    mx.eval(y)
    with pytest.raises(ValueError, match="no output has work"):
        kq.fuse_graph([x], [y])
    with pytest.raises(ValueError, match="needs an output"):
        kq.fuse_graph([x], [])


def test_release_drops_the_record_and_spares_arrays_already_made():
    c1, c2 = "q4_k", "q8_0"
    handle, w1, w2 = _record(c1, c2, 8, mx.bfloat16)
    x = _x(8, mx.bfloat16, 4)
    got = kq.fused_call(handle, [x])
    kq.fuse_release(handle)
    # A released handle serves no further call, and a second release is quiet.
    with pytest.raises(ValueError, match="bad handle"):
        kq.fused_call(handle, [x])
    with pytest.raises(ValueError, match="bad handle"):
        kq.fused_info(handle)
    kq.fuse_release(handle)
    # The array made before the release still evaluates: it holds the record.
    assert _same(got[0], _layer(x, w1, c1, w2, c2)[0])


@pytest.mark.parametrize("mix", MIXES[:2], ids=lambda m: "+".join(m))
@pytest.mark.parametrize("m", ROWS)
def test_cpu_stream(mix, m):
    c1, c2 = mix
    with mx.stream(mx.cpu):
        handle, w1, w2 = _record(c1, c2, m, mx.bfloat16)
        try:
            x = _x(m, mx.bfloat16, 5)
            want = _layer(x, w1, c1, w2, c2)
            got = kq.fused_call(handle, [x])
            assert _same(got[0], want[0])
            assert _same(got[1], want[1])
        finally:
            kq.fuse_release(handle)


@pytest.mark.skipif(force_cpu(), reason="the serial encoder is a Metal lever")
@pytest.mark.parametrize("m", ROWS)
def test_serial_encoder_bits(m):
    c1, c2 = "iq4_xs", "q6_k"
    handle, w1, w2 = _record(c1, c2, m, mx.bfloat16, serial=True)
    try:
        x = _x(m, mx.bfloat16, 6)
        want = got = x
        for _ in range(2):
            want = _layer(want, w1, c1, w2, c2)[0]
            got = kq.fused_call(handle, [got])[0]
        # Ops before and after the serial encoder share its command buffer.
        assert _same(got + 1, want + 1)
    finally:
        kq.fuse_release(handle)


@pytest.mark.skipif(force_cpu(), reason="needs two devices")
def test_a_graph_on_two_streams_is_refused():
    x = _x(8, mx.float32, 0)
    a = mx.add(x, x, stream=mx.cpu)
    b = mx.multiply(a, a, stream=mx.gpu)
    with pytest.raises(ValueError, match="more than one stream"):
        kq.fuse_graph([x], [b])


# Direct replay: a record made with direct=True writes down the Metal calls
# of its primitives on its first eval and replays them from the second.

metal = pytest.mark.skipif(force_cpu(), reason="direct replay is a Metal path")


@pytest.fixture
def lever_cache():
    was = kq.env_cache(True)
    yield
    kq.env_cache(was)


def _learn(handle, w1, c1, w2, c2, m, dtype):
    """Two evals on different inputs: the recording and one replay."""
    for seed in (20, 21):
        x = _x(m, dtype, seed)
        got = kq.fused_call(handle, [x])
        assert _same(got[0], _layer(x, w1, c1, w2, c2)[0])
    assert kq.fused_direct_info(handle)["state"] == "replay"


@metal
@pytest.mark.parametrize("mix", MIXES, ids=lambda m: "+".join(m))
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("m", ROWS)
def test_direct_bits_match_the_plain_graph(mix, dtype, m, lever_cache):
    c1, c2 = mix
    handle, w1, w2 = _record(c1, c2, m, dtype, direct=True)
    try:
        assert kq.fused_direct_info(handle)["state"] == "new"
        for seed in range(1, 6):
            x = _x(m, dtype, seed)
            want = _layer(x, w1, c1, w2, c2)
            got = kq.fused_call(handle, [x])
            assert _same(got[0], want[0])
            assert _same(got[1], want[1])
        info = kq.fused_direct_info(handle)
        assert info["state"] == "replay"
        assert info["replays"] == 4
        assert info["dispatches"] >= 2
        assert 0 < info["calls"] <= info["calls_recorded"] + 1
    finally:
        kq.fuse_release(handle)


@metal
@pytest.mark.parametrize("m", ROWS)
def test_direct_outputs_of_two_calls_are_both_read_after_the_second(m, lever_cache):
    # A caller keeps outputs across calls (cache and state arrays): a later
    # call must not write into what an earlier one returned.
    c1, c2 = "q4_k", "q8_0"
    handle, w1, w2 = _record(c1, c2, m, mx.bfloat16, direct=True)
    try:
        _learn(handle, w1, c1, w2, c2, m, mx.bfloat16)
        xa, xb, xc = (_x(m, mx.bfloat16, s) for s in (30, 31, 32))
        ga = kq.fused_call(handle, [xa])
        mx.eval(ga)
        gb = kq.fused_call(handle, [xb])
        mx.eval(gb)
        # A third call, left lazy and evaluated with the reads below.
        gc = kq.fused_call(handle, [xc])
        for g, x in ((gc, xc), (ga, xa), (gb, xb)):
            want = _layer(x, w1, c1, w2, c2)
            assert _same(g[0], want[0])
            assert _same(g[1], want[1])
        assert kq.fused_direct_info(handle)["replays"] == 4
    finally:
        kq.fuse_release(handle)


@metal
@pytest.mark.parametrize(
    "caps", [(2, 100000), (1000000, 1000000)], ids=["2 ops", "one"]
)
def test_direct_among_other_ops_and_across_command_buffers(caps, lever_cache):
    # Replays chained through plain mlx ops in one eval. At 2 ops a buffer
    # every replay lands in its own command buffer, so the order between
    # them is mlx's order between encoders.
    c1, c2, m = "iq4_xs", "q6_k", 8
    handle, w1, w2 = _record(c1, c2, m, mx.bfloat16, direct=True)
    held = kq.set_cb_caps(*caps)
    try:
        _learn(handle, w1, c1, w2, c2, m, mx.bfloat16)
        x = _x(m, mx.bfloat16, 40)
        want = got = x * 0.5
        for i in range(6):
            want = mx.tanh(_layer(want, w1, c1, w2, c2)[0]) + i
            got = mx.tanh(kq.fused_call(handle, [got])[0]) + i
        assert _same(mx.sum(got, axis=0), mx.sum(want, axis=0))
        assert kq.fused_direct_info(handle)["replays"] == 7
    finally:
        kq.set_cb_caps(int(held[0]), int(held[1]))
        kq.fuse_release(handle)


@metal
def test_direct_input_views_and_layouts(lever_cache):
    c1, c2, m = "q4_k", "q8_0", 8
    handle, w1, w2 = _record(c1, c2, m, mx.bfloat16, direct=True)
    try:
        _learn(handle, w1, c1, w2, c2, m, mx.bfloat16)
        # A row block of a larger buffer: the same layout at an offset.
        big = mx.stack([_x(m, mx.bfloat16, s) for s in (50, 51, 52)])
        mx.eval(big)
        # (A slice shares the buffer; an integer index would copy the row.)
        before = kq.fused_direct_info(handle)
        x = big[1:2].reshape(m, K)
        assert _same(kq.fused_call(handle, [x])[0], _layer(x, w1, c1, w2, c2)[0])
        assert kq.fused_direct_info(handle)["replays"] == before["replays"] + 1
        # Other strides: the call runs the primitives.
        before = kq.fused_direct_info(handle)
        wide = mx.random.normal((m, 2 * K), key=mx.random.key(53)).astype(mx.bfloat16)
        mx.eval(wide)
        x = wide[:, ::2]
        assert _same(kq.fused_call(handle, [x])[0], _layer(x, w1, c1, w2, c2)[0])
        after = kq.fused_direct_info(handle)
        assert after["replays"] == before["replays"]
        assert after["plain_runs"] == before["plain_runs"] + 1
        assert after["state"] == "replay"
    finally:
        kq.fuse_release(handle)


def _layer2(x, s, w1, c1, w2, c2):
    return _layer(x, w1, c1, w2, c2)[0] + s.astype(x.dtype)


@metal
@pytest.mark.parametrize("learn_shared", [False, True], ids=["apart", "shared"])
def test_direct_inputs_on_one_buffer(learn_shared, lever_cache):
    c1, c2, m = "q4_0", "q3_k", 8
    w1, w2 = _weight(c1, 0), _weight(c2, 1)

    def pair(seed, shared):
        if shared:
            both = mx.stack([_x(m, mx.bfloat16, seed), _x(m, mx.bfloat16, seed + 1)])
            mx.eval(both)
            return both[0:1].reshape(m, K), both[1:2].reshape(m, K)
        return _x(m, mx.bfloat16, seed), _x(m, mx.bfloat16, seed + 1)

    x0, s0 = pair(60, False)
    handle = kq.fuse_graph([x0, s0], [_layer2(x0, s0, w1, c1, w2, c2)], direct=True)
    try:
        for seed in (62, 64):
            x, s = pair(seed, learn_shared)
            assert _same(
                kq.fused_call(handle, [x, s])[0], _layer2(x, s, w1, c1, w2, c2)
            )
        assert kq.fused_direct_info(handle)["state"] == "replay"
        for seed, shared in ((66, False), (68, True), (70, False)):
            x, s = pair(seed, shared)
            assert _same(
                kq.fused_call(handle, [x, s])[0], _layer2(x, s, w1, c1, w2, c2)
            )
        info = kq.fused_direct_info(handle)
        # Learned apart, inputs on one buffer still replay. Learned on one
        # buffer, inputs apart run the primitives.
        assert info["plain_runs"] == (2 if learn_shared else 0)
        assert info["replays"] == (2 if learn_shared else 4)
    finally:
        kq.fuse_release(handle)


@metal
def test_direct_learns_again_after_a_lever_cache_call(lever_cache):
    c1, c2, m = "q5_k", "q4_1", 8
    handle, w1, w2 = _record(c1, c2, m, mx.bfloat16, direct=True)
    try:
        _learn(handle, w1, c1, w2, c2, m, mx.bfloat16)
        kq.env_cache(True)
        assert kq.fused_direct_info(handle)["state"] == "new"
        _learn(handle, w1, c1, w2, c2, m, mx.bfloat16)
        # With the cache off a lever may change between any two calls, so
        # every call runs the primitives.
        kq.env_cache(False)
        assert kq.fused_direct_info(handle)["state"] == "waiting"
        x = _x(m, mx.bfloat16, 71)
        assert _same(kq.fused_call(handle, [x])[0], _layer(x, w1, c1, w2, c2)[0])
        info = kq.fused_direct_info(handle)
        assert info["state"] == "waiting"
        assert info["plain_runs"] == 1
    finally:
        kq.fuse_release(handle)


@metal
def test_direct_check_lever_compares_two_recordings(monkeypatch):
    monkeypatch.setenv("KQ_FUSED_DIRECT_CHECK", "1")
    was = kq.env_cache(True)
    c1, c2, m = "q4_k", "q8_0", 8
    handle, w1, w2 = _record(c1, c2, m, mx.bfloat16, direct=True)
    try:
        states = []
        for seed in (76, 77, 78):
            x = _x(m, mx.bfloat16, seed)
            assert _same(kq.fused_call(handle, [x])[0], _layer(x, w1, c1, w2, c2)[0])
            states.append(kq.fused_direct_info(handle)["state"])
        assert states == ["recorded", "replay", "replay"]
    finally:
        kq.fuse_release(handle)
        kq.env_cache(was)


@metal
def test_direct_check_lever_refuses_recordings_that_differ(monkeypatch):
    # The self-test lever stands in for a primitive that passes a value of
    # its input data to a kernel: the bytes of two recordings differ.
    monkeypatch.setenv("KQ_FUSED_DIRECT_CHECK", "1")
    monkeypatch.setenv("KQ_FUSED_DIRECT_SELFTEST", "1")
    was = kq.env_cache(True)
    c1, c2, m = "q2_k", "iq4_nl", 8
    handle, w1, w2 = _record(c1, c2, m, mx.bfloat16, direct=True)
    try:
        for seed in (80, 81, 82, 83):
            x = _x(m, mx.bfloat16, seed)
            assert _same(kq.fused_call(handle, [x])[0], _layer(x, w1, c1, w2, c2)[0])
        info = kq.fused_direct_info(handle)
        assert info["state"] == "refused"
        assert "Metal calls" in info["reason"]
        assert info["primitives"]
        assert info["replays"] == 0
        assert info["plain_runs"] == 2
    finally:
        kq.fuse_release(handle)
        kq.env_cache(was)


@metal
def test_direct_runs_the_primitives_of_a_type_off_the_list(monkeypatch):
    c1, c2, m = "q4_k", "q8_0", 8
    w1, w2 = _weight(c1, 0), _weight(c2, 1)

    def layer(x):
        return mx.erf(_layer(x, w1, c1, w2, c2)[0])

    def run(handle):
        for seed in (84, 85, 86):
            x = _x(m, mx.bfloat16, seed)
            assert _same(kq.fused_call(handle, [x])[0], layer(x))
        return kq.fused_direct_info(handle)

    was = kq.env_cache(True)
    x0 = _x(m, mx.bfloat16, 0)
    handle = kq.fuse_graph([x0], [layer(x0)], direct=True)
    try:
        info = run(handle)
        assert info["state"] == "refused"
        assert "not on the direct list" in info["reason"]
        assert info["primitives"] == ["Erf"]
        assert info["replays"] == 0
        # The probing lever lets such a record learn.
        monkeypatch.setenv("KQ_FUSED_DIRECT_ANY", "1")
        kq.env_cache(True)
        info = run(handle)
        assert info["state"] == "replay"
        assert info["replays"] == 2
    finally:
        kq.fuse_release(handle)
        kq.env_cache(was)


def _calls_of(build, inputs_of, seeds):
    """The call list of one record per seed, each made and learned on its
    own inputs."""
    lists = []
    for seed in seeds:
        ins = inputs_of(seed)
        handle = kq.fuse_graph(ins, build(*ins), direct=True)
        try:
            for k in (0, 1):
                again = inputs_of(1000 + 7 * seed + k)
                got = kq.fused_call(handle, again)
                for a, b in zip(got, build(*again), strict=True):
                    assert _same(a, b)
            assert kq.fused_direct_info(handle)["state"] == "replay"
            lists.append(kq.fused_direct_calls(handle))
        finally:
            kq.fuse_release(handle)
    return lists


@metal
@pytest.mark.parametrize("mix", MIXES, ids=lambda m: "+".join(m))
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("view", [False, True], ids=["own buffer", "row block"])
def test_direct_call_list_depends_on_the_key_alone(mix, dtype, m, view, lever_cache):
    # One key (graph, shapes, dtype, input layout, levers) and many input
    # values and buffers: every record must show the same Metal calls,
    # buffers by role. A primitive that let anything else reach its calls
    # fails here.
    c1, c2 = mix
    w1, w2 = _weight(c1, 0), _weight(c2, 1)

    def inputs_of(seed):
        if not view:
            return [_x(m, dtype, seed) * (1.0 + seed % 5)]
        # The middle row block of a larger buffer.
        big = mx.stack([_x(m, dtype, seed + i) for i in range(3)]).reshape(3 * m, K)
        mx.eval(big)
        return [big[m : 2 * m]]

    lists = _calls_of(lambda x: list(_layer(x, w1, c1, w2, c2)), inputs_of, range(6))
    assert lists[0]
    assert any("groups" in c or "threads" in c for c in lists[0])
    for other in lists[1:]:
        assert other == lists[0]


def _glue(x, s):
    """The listed mlx types around one kernel op: views, copies,
    elementwise ops, a norm, a matmul and a compiled body."""
    m = x.shape[0]
    a, b = mx.split(x.astype(mx.float32), 2, axis=-1)
    g = mx.fast.rms_norm(a, None, 1e-5) * mx.sigmoid(b)
    g = mx.concatenate([g, b[:, : K // 4], mx.full((m, K // 4), 0.5)], axis=-1)
    g = mx.contiguous(g.reshape(m, 2, K // 2).transpose(0, 2, 1))
    g = mx.flatten(g, 1, 2) + mx.broadcast_to(s[:, :1].astype(mx.float32), (m, K))
    g = mx.squeeze(mx.expand_dims(g, 1) @ _GLUE_W, axis=1)
    return _GLUE_F(g.astype(x.dtype), x), mx.unflatten(g, 1, (2, K // 2))


_GLUE_W = mx.random.normal((K, K), key=mx.random.key(404)) * 0.05
_GLUE_F = mx.compile(lambda a, b: a * 0.5 + b)


@metal
@pytest.mark.parametrize("m", [1, 8])
def test_direct_call_list_of_the_listed_mlx_types(m, lever_cache, compile_on):
    mx.eval(_GLUE_W)

    def inputs_of(seed):
        return [_x(m, mx.bfloat16, seed), _x(m, mx.bfloat16, seed + 50) * 3.0]

    x0, s0 = inputs_of(0)
    probe = kq.fuse_graph([x0, s0], list(_glue(x0, s0)))
    names = set(kq.fused_info(probe)[0])
    kq.fuse_release(probe)
    assert {
        "Split",
        "RMSNorm",
        "Sigmoid",
        "Multiply",
        "Concatenate",
        "Slice",
        "Full",
        "Reshape",
        "Transpose",
        "Contiguous",
        "Flatten",
        "Add",
        "Broadcast",
        "ExpandDims",
        "Matmul",
        "Squeeze",
        "AsType",
        "Unflatten",
    } <= names
    assert any(n.startswith("Compiled") for n in names)
    lists = _calls_of(lambda x, s: list(_glue(x, s)), inputs_of, range(4))
    assert lists[0]
    for other in lists[1:]:
        assert other == lists[0]


@metal
def test_direct_release_with_a_call_pending(lever_cache):
    c1, c2, m = "q4_k", "q8_0", 8
    handle, w1, w2 = _record(c1, c2, m, mx.bfloat16, direct=True)
    _learn(handle, w1, c1, w2, c2, m, mx.bfloat16)
    x = _x(m, mx.bfloat16, 90)
    got = kq.fused_call(handle, [x])
    mx.async_eval(got)
    more = kq.fused_call(handle, [got[0]])
    kq.fuse_release(handle)
    want = _layer(x, w1, c1, w2, c2)
    assert _same(got[1], want[1])
    assert _same(more[0], _layer(want[0], w1, c1, w2, c2)[0])


@metal
def test_direct_records_share_the_buffers_between_nodes(lever_cache):
    c1, c2, m = "q4_k", "q8_0", 8
    first, w1, w2 = _record(c1, c2, m, mx.bfloat16, direct=True)
    second, _, _ = _record(c1, c2, m, mx.bfloat16, direct=True)
    try:
        _learn(first, w1, c1, w2, c2, m, mx.bfloat16)
        x = _x(m, mx.bfloat16, 95)
        assert _same(kq.fused_call(first, [x])[0], _layer(x, w1, c1, w2, c2)[0])
        one = kq.fused_direct_info(first)
        assert one["scratch_bytes"] > 0
        _learn(second, w1, c1, w2, c2, m, mx.bfloat16)
        # Calls of the two records, interleaved in one eval.
        want = got = x
        for h in (first, second, second, first):
            want = _layer(want, w1, c1, w2, c2)[0]
            got = kq.fused_call(h, [got])[0]
        assert _same(got, want)
        two = kq.fused_direct_info(second)
        # The second record needs the same buffers and adds none.
        assert two["scratch_bytes"] == one["scratch_bytes"]
        assert two["pool_bytes"] == one["pool_bytes"]
    finally:
        kq.fuse_release(first)
        kq.fuse_release(second)


def test_direct_on_a_cpu_stream_runs_the_primitives():
    c1, c2, m = "q4_k", "q8_0", 8
    with mx.stream(mx.cpu):
        handle, w1, w2 = _record(c1, c2, m, mx.bfloat16, direct=True)
        try:
            for seed in (1, 2, 3):
                x = _x(m, mx.bfloat16, seed)
                assert _same(
                    kq.fused_call(handle, [x])[0], _layer(x, w1, c1, w2, c2)[0]
                )
            assert kq.fused_direct_info(handle)["state"] == ""
        finally:
            kq.fuse_release(handle)


def test_serial_and_direct_together_are_refused():
    x = _x(8, mx.bfloat16, 0)
    with pytest.raises(ValueError, match="serial or direct"):
        kq.fuse_graph([x], [x + 1], serial=True, direct=True)


def _branches(x, w1, c1, w2, c2):
    """Two independent branches of two kernels each off one input, then
    their sum: a walk from the output runs one branch to its end before
    the other starts."""
    sc = mx.zeros((1,), mx.uint8)
    a = mx.sigmoid(kq.quantized_matmul(x, w1, sc, c1, transpose=True))
    b = mx.sigmoid(kq.quantized_matmul(x, w2, sc, c2, transpose=True))
    return a * x + b


@metal
@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("direct", [False, True])
def test_level_order_runs_the_same_kernels_with_fewer_barriers(m, direct, lever_cache):
    c1, c2 = MIXES[0]
    w1, w2 = _weight(c1, 1), _weight(c2, 2)
    x0 = _x(m, mx.bfloat16, 0)
    y0 = _branches(x0, w1, c1, w2, c2)
    walk = kq.fuse_graph([x0], [y0], direct=direct)
    level = kq.fuse_graph([x0], [y0], direct=direct, level=True)
    group = kq.fuse_graph([x0], [y0], direct=direct, level=2)
    try:
        assert sorted(kq.fused_info(walk)[0]) == sorted(kq.fused_info(level)[0])
        assert sorted(kq.fused_info(walk)[0]) == sorted(kq.fused_info(group)[0])
        for seed in range(1, 5):
            x = _x(m, mx.bfloat16, seed)
            want = _branches(x, w1, c1, w2, c2)
            assert _same(kq.fused_call(walk, [x])[0], want)
            assert _same(kq.fused_call(level, [x])[0], want)
            assert _same(kq.fused_call(group, [x])[0], want)
        if direct:
            a, b = kq.fused_direct_info(walk), kq.fused_direct_info(level)
            g = kq.fused_direct_info(group)
            assert a["state"] == b["state"] == g["state"] == "replay"
            assert a["dispatches"] == b["dispatches"] == g["dispatches"]
            assert b["barriers"] < a["barriers"]
            assert g["barriers"] <= b["barriers"]
    finally:
        kq.fuse_release(walk)
        kq.fuse_release(level)
        kq.fuse_release(group)


@metal
def test_the_encoder_trace_lists_the_calls_of_an_eval(lever_cache):
    c1, c2 = MIXES[0]
    w1, w2 = _weight(c1, 1), _weight(c2, 2)
    x = _x(ROWS[0], mx.bfloat16, 0)
    mx.eval(_branches(x, w1, c1, w2, c2))
    y = _branches(x, w1, c1, w2, c2)
    mx.synchronize()
    kq.encoder_trace_begin()
    mx.async_eval(y)
    lines = kq.encoder_trace_end()
    mx.eval(y)
    kinds = [line.split()[0] for line in lines]
    assert kinds.count("groups") + kinds.count("threads") >= 2
    assert "pso" in kinds and "buf" in kinds
    assert kq.encoder_trace_end() == []
    assert _same(y, _branches(x, w1, c1, w2, c2))


# Parameter blocks: values that change between replays of one record.

CAP = 24


def _p(*vals):
    p = mx.array(list(vals), dtype=mx.int32)
    mx.eval(p)
    return p


def _append_layer(x, buf, p, off, w, codec):
    """(the buffer with the new rows, a read of it after the write): a
    matmul whose rows go into buf at off, the offset taken from slot 1 of
    p when p is given."""
    m = x.shape[0]
    sc = mx.zeros((1,), mx.uint8)
    y = kq.quantized_matmul(x, w, sc, codec, transpose=True).reshape(1, 1, m, K)
    if p is None:
        out = kq.kv_append(buf, y, off)
    else:
        out = kq.kv_append(buf, y, off, params=p, offset_slot=1)
    return out, out[:, :, : CAP // 2, :] + x[:1]


def _buf(dtype):
    b = mx.zeros((1, 1, CAP, K), dtype)
    mx.eval(b)
    return b


@metal
@pytest.mark.parametrize("m", [1, 3, 8])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_param_block_moves_an_append_under_one_record(m, dtype, lever_cache):
    w = _weight("q4_k", 0)
    buf, ref = _buf(dtype), _buf(dtype)
    x0, p0 = _x(m, dtype, 0), _p(7, 0)
    out = _append_layer(x0, buf, p0, 0, w, "q4_k")
    handle = kq.fuse_graph(
        [x0, buf, p0], list(out), direct=True, params=2, param_ranges=[(1, 0, CAP - m)]
    )
    try:
        mx.eval(out)
        buf = out[0]
        ref = _append_layer(x0, ref, None, 0, w, "q4_k")[0]
        mx.eval(ref)
        offs = [m, 2 * m, 0, CAP - m, 1, 5, CAP - m]
        for seed, off in enumerate(offs):
            x = _x(m, dtype, seed + 1)
            got = kq.fused_call(handle, [x, buf, _p(7, off)])
            want = _append_layer(x, ref, None, off, w, "q4_k")
            assert _same(got[0], want[0]), off
            assert _same(got[1], want[1]), off
            buf, ref = got[0], want[0]
        info = kq.fused_direct_info(handle)
        assert info["state"] == "replay"
        assert info["replays"] == len(offs) - 1
        # A value the record does not cover, and a slot nobody declared.
        with pytest.raises(ValueError, match="outside the range"):
            kq.fused_call(handle, [x0, buf, _p(7, CAP - m + 1)])
        with pytest.raises(ValueError, match="outside the range"):
            kq.fused_call(handle, [x0, buf, _p(8, 0)])
        assert kq.fused_direct_info(handle)["replays"] == len(offs) - 1
    finally:
        kq.fuse_release(handle)


@metal
def test_param_block_ranges_are_checked_at_record():
    w = _weight("q4_k", 0)
    buf, x0, p0 = _buf(mx.bfloat16), _x(2, mx.bfloat16, 0), _p(7, 4)
    out = list(_append_layer(x0, buf, p0, 4, w, "q4_k"))
    with pytest.raises(ValueError, match="leaves out the recorded value"):
        kq.fuse_graph(
            [x0, buf, p0], out, direct=True, params=2, param_ranges=[(1, 5, 9)]
        )
    with pytest.raises(ValueError, match="names no input"):
        kq.fuse_graph([x0, buf, p0], out, direct=True, params=3)
    with pytest.raises(ValueError, match="needs params"):
        kq.fuse_graph([x0, buf, p0], out, direct=True, param_ranges=[(1, 0, 9)])
    with pytest.raises(ValueError, match="int32 vector"):
        kq.fuse_graph([x0, buf, p0], out, direct=True, params=0)


@metal
def test_params_read_from_an_undeclared_input_keep_the_primitives(lever_cache):
    # The block is an input the record does not name: a program would hold
    # the first offset for good, so the record runs its primitives.
    w = _weight("q4_k", 0)
    dtype, m = mx.bfloat16, 2
    buf, ref = _buf(dtype), _buf(dtype)
    x0, p0 = _x(m, dtype, 0), _p(7, 0)
    out = _append_layer(x0, buf, p0, 0, w, "q4_k")
    handle = kq.fuse_graph([x0, buf, p0], list(out), direct=True)
    try:
        mx.eval(out)
        buf = out[0]
        ref = _append_layer(x0, ref, None, 0, w, "q4_k")[0]
        for seed, off in enumerate([2, 9, 4]):
            x = _x(m, dtype, seed + 1)
            got = kq.fused_call(handle, [x, buf, _p(7, off)])
            want = _append_layer(x, ref, None, off, w, "q4_k")
            assert _same(got[0], want[0])
            assert _same(got[1], want[1])
            buf, ref = got[0], want[0]
        info = kq.fused_direct_info(handle)
        assert info["state"] == "refused"
        assert "parameter block" in info["reason"]
    finally:
        kq.fuse_release(handle)


@metal
def test_param_block_under_the_check_lever(monkeypatch):
    # Two recordings at different offsets are the same record.
    monkeypatch.setenv("KQ_FUSED_DIRECT_CHECK", "1")
    prev = kq.env_cache(True)
    w = _weight("q4_k", 0)
    dtype, m = mx.bfloat16, 2
    buf, ref = _buf(dtype), _buf(dtype)
    x0, p0 = _x(m, dtype, 0), _p(7, 0)
    out = _append_layer(x0, buf, p0, 0, w, "q4_k")
    handle = kq.fuse_graph(
        [x0, buf, p0], list(out), direct=True, params=2, param_ranges=[(1, 0, CAP - m)]
    )
    try:
        mx.eval(out)
        buf = out[0]
        ref = _append_layer(x0, ref, None, 0, w, "q4_k")[0]
        for seed, off in enumerate([3, 11, 6, 20]):
            x = _x(m, dtype, seed + 1)
            got = kq.fused_call(handle, [x, buf, _p(7, off)])
            want = _append_layer(x, ref, None, off, w, "q4_k")
            assert _same(got[0], want[0])
            buf, ref = got[0], want[0]
        info = kq.fused_direct_info(handle)
        assert info["state"] == "replay", info
        assert info["replays"] == 2
    finally:
        kq.fuse_release(handle)
        kq.env_cache(prev)


@metal
def test_bound_replay_from_a_thread_with_no_eval(lever_cache):
    # The record replayed by fused_commit_detached: a thread of its own, no
    # GIL, a command buffer of its own, patched offsets, outputs read with
    # no eval.
    w = _weight("q4_k", 0)
    dtype, m = mx.bfloat16, 2
    buf, ref = _buf(dtype), _buf(dtype)
    x0, p0 = _x(m, dtype, 0), _p(7, 0)
    out = _append_layer(x0, buf, p0, 0, w, "q4_k")
    handle = kq.fuse_graph(
        [x0, buf, p0], list(out), direct=True, params=2, param_ranges=[(1, 0, CAP - m)]
    )
    try:
        mx.eval(out)
        buf = out[0]
        ref = _append_layer(x0, ref, None, 0, w, "q4_k")[0]
        mx.eval(ref)
        with pytest.raises(RuntimeError, match="no program yet"):
            kq.fused_bind(handle, [x0, buf, p0])
        x = _x(m, dtype, 1)
        got = kq.fused_call(handle, [x, buf, _p(7, 2)])
        ref = _append_layer(x, ref, None, 2, w, "q4_k")[0]
        mx.eval(got, ref)
        buf = got[0]
        assert kq.fused_direct_info(handle)["state"] == "replay"
        # Two stages in flight, each with its own block and its own input.
        xa, xb = _x(m, dtype, 2), _x(m, dtype, 3)
        pa, pb = _p(7, 0), _p(7, 0)
        mx.eval(xa, xb)
        ba, outs_a = kq.fused_bind(handle, [xa, buf, pa])
        bb, outs_b = kq.fused_bind(handle, [xb, buf, pb])
        assert kq.fused_patch(ba, 1, 4) and kq.fused_patch(bb, 1, 6)
        assert not kq.fused_patch(ba, 1, CAP)
        assert not kq.fused_patch(ba, 0, 8)
        assert not kq.fused_patch(ba, 5, 0)
        mx.synchronize()
        assert kq.fused_commit_detached([ba, bb]) == [True, True]
        wa = _append_layer(xa, ref, None, 4, w, "q4_k")
        mx.eval(wa)
        wb = _append_layer(xb, wa[0], None, 6, w, "q4_k")
        mx.eval(wb)
        # outs_a[1] read the buffer before stage b wrote its rows.
        assert _same(outs_a[1], wa[1])
        assert _same(outs_b[1], wb[1])
        assert _same(outs_b[0], wb[0])
        # The same bindings again, patched from here.
        assert kq.fused_patch(ba, 1, 10) and kq.fused_patch(bb, 1, 12)
        assert kq.fused_commit_detached([ba, bb]) == [True, True]
        wa = _append_layer(xa, wb[0], None, 10, w, "q4_k")
        mx.eval(wa)
        wb = _append_layer(xb, wa[0], None, 12, w, "q4_k")
        mx.eval(wb)
        assert _same(outs_b[0], wb[0])
        assert _same(outs_b[1], wb[1])
        info = kq.fused_binding_info(ba)
        assert info["param_bytes"] == 8
        assert info["output_bytes"] > 0
        kq.fused_unbind(ba)
        kq.fused_unbind(bb)
    finally:
        kq.fuse_release(handle)


def _score_layer(q, keys, wts, tail, p, off):
    if p is None:
        return kq.dsa_indexer_score_decode(q, keys, wts, off, 4, keys_tail=tail)
    return kq.dsa_indexer_score_decode(
        q, keys, wts, off, 4, keys_tail=tail, params=p, q_offset_slot=0
    )


@metal
@pytest.mark.parametrize("L", [2, 5, 8])
def test_param_block_offset_of_the_indexer_scores(L, lever_cache):
    dtype, P, PT = mx.bfloat16, 640, 64
    key = mx.random.key(3)
    keys = mx.random.normal((1, P, 128), key=key).astype(dtype)
    tail = mx.random.normal((1, PT, 128), key=mx.random.key(4)).astype(dtype)
    wts = mx.ones((1, L, 4), dtype) * 0.09
    mx.eval(keys, tail, wts)

    def q_of(seed):
        q = mx.random.normal((1, 4, L, 128), key=mx.random.key(seed)).astype(dtype)
        mx.eval(q)
        return q

    off0 = 4 * P + 3
    q0, p0 = q_of(0), _p(off0)
    out = _score_layer(q0, keys, wts, tail, p0, off0)
    handle = kq.fuse_graph(
        [q0, p0],
        [out],
        direct=True,
        params=1,
        param_ranges=[(0, 4 * 520, 4 * (P + PT))],
    )
    try:
        mx.eval(out)
        for seed, off in enumerate(
            [off0 + 1, 4 * 600, 4 * 520, 4 * (P + PT) - L, 4 * P + 17, 4 * P + 18]
        ):
            q = q_of(seed + 1)
            got = kq.fused_call(handle, [q, _p(off)])[0]
            assert _same(got, _score_layer(q, keys, wts, tail, None, off)), off
        assert kq.fused_direct_info(handle)["state"] == "replay"
        assert kq.fused_direct_info(handle)["replays"] == 5
    finally:
        kq.fuse_release(handle)


@metal
@pytest.mark.parametrize("L", [2, 8])
def test_param_block_base_length_of_paged_rows(L, lever_cache):
    dtype, D, Hq, Hkv, PR = mx.bfloat16, 256, 4, 2, 4
    nb, nt, n_sel = 512, 64, 16
    kb = mx.random.normal((1, Hkv, nb, D), key=mx.random.key(1)).astype(dtype)
    vb = mx.random.normal((1, Hkv, nb, D), key=mx.random.key(2)).astype(dtype)
    kt = mx.random.normal((1, Hkv, nt, D), key=mx.random.key(3)).astype(dtype)
    vt = mx.random.normal((1, Hkv, nt, D), key=mx.random.key(4)).astype(dtype)
    mx.eval(kb, vb, kt, vt)

    def call(q, pages, ends, base_len, p):
        kw = {} if p is None else {"params": p, "base_len_slot": 2}
        return kq.sdpa_paged_rows(
            q,
            kb,
            vb,
            0.0625,
            pages,
            ends,
            kt,
            vt,
            base_len,
            page_rows=PR,
            tail_page=True,
            **kw,
        )

    def inputs(seed, base_len, used):
        q = mx.random.normal((1, Hq, L, D), key=mx.random.key(10 + seed)).astype(dtype)
        end0 = base_len + used - L
        pages = mx.random.randint(
            0, end0 // PR, (1, L, n_sel), key=mx.random.key(20 + seed)
        ).astype(mx.int32)
        ends = (end0 + 1 + mx.arange(L, dtype=mx.int32))[None]
        mx.eval(q, pages, ends)
        return q, pages, ends

    q0, pg0, e0 = inputs(0, 400, 20)
    p0 = _p(0, 0, 400)
    out = call(q0, pg0, e0, 400, p0)
    handle = kq.fuse_graph(
        [q0, pg0, e0, p0], [out], direct=True, params=3, param_ranges=[(2, 0, nb)]
    )
    try:
        mx.eval(out)
        for seed, (bl, used) in enumerate([(400, 30), (512, 9), (37, 64), (300, 12)]):
            q, pg, e = inputs(seed + 1, bl, used)
            got = kq.fused_call(handle, [q, pg, e, _p(0, 0, bl)])[0]
            assert _same(got, call(q, pg, e, bl, None)), bl
        assert kq.fused_direct_info(handle)["state"] == "replay"
        assert kq.fused_direct_info(handle)["replays"] == 3
    finally:
        kq.fuse_release(handle)


@metal
@pytest.mark.parametrize("L", [2, 8])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_a_key_tail_at_capacity_selects_the_same_blocks(L, dtype):
    # Block keys past the live count are not visible to the scorer, and the
    # top-k over the longer row gives the same ids in the same order: what
    # lets a record keep its block key tail at a fixed capacity.
    P, cap = 640, 66
    keys = mx.random.normal((1, P, 128), key=mx.random.key(3)).astype(dtype)
    full = mx.random.normal((1, cap, 128), key=mx.random.key(4)).astype(dtype)
    q = mx.random.normal((1, 4, L, 128), key=mx.random.key(5)).astype(dtype)
    wts = mx.ones((1, L, 4), dtype) * 0.09
    for live in (0, 1, 7, 40, 64):
        # the rows end where block P + live ends, as on the real path
        off = 4 * (P + live) - L
        if live:
            exact = kq.dsa_indexer_score_decode(
                q, keys, wts, off, 4, keys_tail=full[:, :live]
            )
        else:
            exact = kq.dsa_indexer_score_decode(q, keys, wts, off, 4)
        padded = kq.dsa_indexer_score_decode(q, keys, wts, off, 4, keys_tail=full)
        a = kq.dsa_topk_indices(exact, 512, True)
        b = kq.dsa_topk_indices(padded, 512, True)
        assert bool(mx.array_equal(a, b).item()), live
        assert _same(exact, padded[..., : P + live])


def _rope_layer(x, p, off):
    y = mx.fast.rope(
        x,
        64,
        traditional=False,
        base=10000.0,
        scale=1.0,
        offset=off if p is None else p[1:2].squeeze(),
    )
    return y * 2.0


@metal
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_param_block_offset_of_rope(dtype, monkeypatch):
    # mx.fast.rope reads an array offset on the GPU: under the check lever
    # two recordings at different offsets are one record, and replays at
    # other offsets carry the bits of the int offset.
    monkeypatch.setenv("KQ_FUSED_DIRECT_CHECK", "1")
    prev = kq.env_cache(True)

    def x_of(seed):
        x = mx.random.normal((1, 4, 5, 128), key=mx.random.key(seed)).astype(dtype)
        mx.eval(x)
        return x

    x0, p0 = x_of(0), _p(3, 100)
    out = _rope_layer(x0, p0, None)
    handle = kq.fuse_graph(
        [x0, p0], [out], direct=True, params=1, param_ranges=[(1, 0, 1 << 20)]
    )
    try:
        mx.eval(out)
        for seed, off in enumerate([101, 0, 77777, 2052, 2053]):
            x = x_of(seed + 1)
            got = kq.fused_call(handle, [x, _p(3, off)])[0]
            assert _same(got, _rope_layer(x, None, off)), off
        info = kq.fused_direct_info(handle)
        assert info["state"] == "replay", info
        assert info["replays"] == 3
    finally:
        kq.fuse_release(handle)
        kq.env_cache(prev)
