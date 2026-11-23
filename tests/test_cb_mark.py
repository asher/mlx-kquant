"""cb_commit_mark: identity on values, a command buffer boundary on Metal."""

import mlx.core as mx
import pytest

import mlx_kquant as kq


def _metal() -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu


def test_mark_is_identity():
    x = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    y = kq.cb_commit_mark(x * 2 + 1)
    assert y.shape == (3, 4) and y.dtype == mx.float32
    assert mx.array_equal(y, x * 2 + 1).item()
    assert mx.array_equal(kq.cb_commit_mark(y)[1:], (x * 2 + 1)[1:]).item()


def test_mark_on_an_empty_buffer_is_no_boundary():
    if not _metal():
        pytest.skip("Metal only")
    x = mx.ones((4,))
    mx.eval(x)
    mx.synchronize()
    n = kq.cb_commit_marks()
    y = kq.cb_commit_mark(x)
    mx.eval(y)
    assert kq.cb_commit_marks() == n


def test_marks_are_the_boundaries_under_high_caps():
    if not _metal():
        pytest.skip("Metal only")
    ops, mb = kq.get_cb_caps()
    x = mx.random.normal((64, 64))
    mx.eval(x)

    def chain(marks):
        h = x
        for i in range(12):
            h = mx.tanh(h @ x) * 0.01 + x
            if i in marks:
                h = kq.cb_commit_mark(h)
        return h

    want = chain(())
    mx.eval(want)
    kq.set_cb_caps(1 << 20, 1 << 20)
    try:
        mx.synchronize()
        n = kq.cb_commit_marks()
        got = chain((0, 5))
        mx.eval(got)
        made = kq.cb_commit_marks() - n
    finally:
        kq.set_cb_caps(ops, mb)
    assert made == 2
    assert mx.array_equal(got, want).item()
    # a later eval at the stock caps is not disturbed
    again = chain((3,))
    mx.eval(again)
    assert mx.array_equal(again, want).item()
