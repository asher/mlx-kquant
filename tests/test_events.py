"""Shared-event stream primitives (event_signal / event_wait + host API).

These are the feeder-loop handoff edges (docs/feeder/DESIGN.md): identity ops
that encode an MTLSharedEvent signal/wait at their position in evaluation
order, plus a host-side registry for the CPU feeder. GPU-encoding tests are
skipped off-Metal (the host registry itself is Metal-only too: there is no
event object to create without a device).
"""

import threading
import time

import mlx.core as mx
import pytest

import mlx_kquant as kq

POISON = 2**64 - 1

metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="shared events require Metal"
)


@metal
def test_host_registry_roundtrip():
    h = kq.shared_event_create()
    assert h > 0
    assert kq.shared_event_read(h) == 0
    kq.shared_event_set(h, 7)
    assert kq.shared_event_read(h) == 7
    assert kq.shared_event_wait(h, 7, 100)
    assert not kq.shared_event_wait(h, 8, 50)
    kq.shared_event_destroy(h)
    with pytest.raises(ValueError):
        kq.shared_event_read(h)


@metal
def test_signal_is_identity_and_reaches_host():
    h = kq.shared_event_create()
    x = mx.arange(16, dtype=mx.float32)
    with mx.stream(mx.gpu):
        y = kq.event_signal(x, h, 1)
    mx.eval(y)
    assert mx.all(y == x)
    assert kq.shared_event_read(h) >= 1
    # CPU stream: plain identity, encodes nothing
    with mx.stream(mx.cpu):
        z = kq.event_signal(x, h, 2)
    mx.eval(z)
    assert mx.all(z == x)
    assert kq.shared_event_read(h) == 1
    kq.shared_event_destroy(h)


@metal
def test_pre_satisfied_wait():
    h = kq.shared_event_create()
    kq.shared_event_set(h, 5)
    x = mx.arange(16, dtype=mx.float32)
    with mx.stream(mx.gpu):
        w = kq.event_wait(x * 2, h, 5)
    out = w + 1
    mx.eval(out)
    assert mx.all(out == x * 2 + 1)
    kq.shared_event_destroy(h)


@metal
def test_feeder_handoff_blocks_and_resumes():
    r_done = kq.shared_event_create()
    w_ready = kq.shared_event_create()
    seen = []

    def feeder():
        if kq.shared_event_wait(r_done, 1, 5000):
            seen.append(kq.shared_event_read(r_done))
            time.sleep(0.01)
            kq.shared_event_set(w_ready, 1)

    t = threading.Thread(target=feeder)
    t.start()
    a = mx.ones((256, 256))
    router = a @ a
    with mx.stream(mx.gpu):
        sig = kq.event_signal(router, r_done, 1)
        gated = kq.event_wait(sig, w_ready, 1)
    expert = gated @ a
    t0 = time.time()
    mx.eval(expert)
    blocked = time.time() - t0
    t.join()
    assert seen == [1]
    assert mx.allclose(expert, (a @ a) @ a)
    assert blocked >= 0.010  # eval stalled until the feeder answered
    kq.shared_event_destroy(r_done)
    kq.shared_event_destroy(w_ready)


@metal
def test_handoff_survives_command_buffer_splits():
    r = kq.shared_event_create()
    w = kq.shared_event_create()

    def feeder():
        if kq.shared_event_wait(r, 1, 5000):
            kq.shared_event_set(w, 1)

    t = threading.Thread(target=feeder)
    t.start()
    b = mx.ones((64, 64))
    with mx.stream(mx.gpu):
        mid = kq.event_signal(b, r, 1)
        for _ in range(200):  # far beyond the default ops-per-buffer limit
            mid = mid + 1
        gated = kq.event_wait(mid, w, 1)
    final = gated * 2
    mx.eval(final)
    t.join()
    assert mx.all(final == (b + 200) * 2)
    kq.shared_event_destroy(r)
    kq.shared_event_destroy(w)


@metal
def test_poison_drains_wedged_wait():
    h = kq.shared_event_create()
    with mx.stream(mx.gpu):
        wedged = kq.event_wait(mx.ones(8), h, 42)  # 42 is never signaled

    def watchdog():
        time.sleep(0.2)
        kq.shared_event_set(h, POISON)

    t = threading.Thread(target=watchdog)
    t.start()
    t0 = time.time()
    mx.eval(wedged * 3)
    waited = time.time() - t0
    t.join()
    assert 0.15 <= waited < 3.0
    assert mx.all(wedged * 3 == 3)
    # device stays healthy after the poison recovery
    c = mx.ones((128, 128))
    assert mx.allclose(c @ c, mx.full((128, 128), 128.0))
    kq.shared_event_destroy(h)


@metal
def test_bad_handle_throws():
    with pytest.raises(ValueError):
        kq.event_signal(mx.ones(4), 0, 1)
    x = mx.ones(4)
    with mx.stream(mx.gpu):
        y = kq.event_signal(x, 10**9, 1)  # unknown handle: throws at eval
    with pytest.raises(Exception):
        mx.eval(y)
