"""Writable zero-copy host buffers (arena_alloc).

The feeder's staging memory (docs/feeder/DESIGN.md): one allocation seen as a
writable memoryview by the CPU feeder and as a uint8 array by kernels. Tests
cover both sides observing the same bytes, disk reads landing directly in the
buffer via os.preadv, a kquant kernel consuming arena-backed wire bytes
identically to a normal array, and the full staged handoff ordered by the
shared-event pair.
"""

import os
import tempfile
import threading

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="arena buffers require the Metal allocator"
)


@metal
def test_cpu_write_gpu_read():
    arr, mv = kq.arena_alloc([4, 16])
    assert arr.shape == (4, 16) and arr.dtype == mx.uint8
    assert len(mv) == 64 and not mv.readonly
    mv[:] = bytes(range(64))
    assert mx.all(arr.reshape(-1) == mx.arange(64, dtype=mx.uint8))
    with mx.stream(mx.gpu):
        total = arr.astype(mx.uint32).sum()
    mx.eval(total)
    assert total.item() == sum(range(64))


@metal
def test_preadv_lands_in_buffer():
    arr, mv = kq.arena_alloc([256])
    payload = bytes(reversed(range(256)))
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(payload)
        path = f.name
    fd = os.open(path, os.O_RDONLY)
    try:
        assert os.preadv(fd, [mv], 0) == 256
    finally:
        os.close(fd)
        os.unlink(path)
    assert bytes(mv) == payload
    with mx.stream(mx.gpu):
        got = arr + 0
    mx.eval(got)
    assert np.array_equal(np.array(got), np.frombuffer(payload, np.uint8))


@metal
def test_kernel_consumes_arena_wire_bytes():
    # quantize a weight, stage its wire bytes through an arena buffer, and
    # check the GPU dequantize matches the normal-array path bit-for-bit
    w = mx.random.normal((8, 256))
    wq, sc = kq.quantize(w, "q5_k")
    mx.eval(wq, sc)
    ref = kq.dequantize(wq, sc, "q5_k")
    arr, mv = kq.arena_alloc(list(wq.shape))
    mv[:] = bytes(np.array(wq).reshape(-1))
    with mx.stream(mx.gpu):
        got = kq.dequantize(arr, sc, "q5_k")
    mx.eval(ref, got)
    assert np.array_equal(np.array(ref), np.array(got))


@metal
def test_staged_handoff_through_event_pair():
    # feeder writes the buffer THEN signals; compute is gated on the wait
    arr, mv = kq.arena_alloc([1024])
    evt = kq.shared_event_create()

    def feeder():
        mv[:] = bytes([7]) * 1024
        kq.shared_event_set(evt, 1)

    with mx.stream(mx.gpu):
        gated = kq.event_wait(arr, evt, 1)
        total = gated.astype(mx.uint32).sum()
    t = threading.Thread(target=feeder)
    t.start()
    mx.eval(total)
    t.join()
    assert total.item() == 7 * 1024
    kq.shared_event_destroy(evt)


@metal
def test_alloc_validation():
    with pytest.raises(ValueError):
        kq.arena_alloc([0, 4])
    with pytest.raises(ValueError):
        kq.arena_alloc([-1])
