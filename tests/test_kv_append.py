"""kv_append: KV cache rows written into the cache's own buffer.

The op shares the cache buffer, so the tests pin both sides of that: the
rows land where a slice update puts them, nothing is allocated while another
array holds the buffer, and a view of the rows below the write still reads
its old rows. The memory tests need the Metal device and skip under
KQUANT_FORCE_CPU.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

metal = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="allocation is measured on the Metal device",
)

B, H, CAP, D = 2, 3, 40, 64


def _rand(shape, dtype):
    if dtype == mx.uint32:
        return mx.random.randint(0, 2**31, shape).astype(mx.uint32)
    return mx.random.normal(shape).astype(dtype)


def _bits(a):
    a = a.view(mx.uint16) if a.dtype == mx.bfloat16 else a
    return np.array(a)


def _reference(buf, x, offset):
    ref = mx.slice_update(buf, x, mx.array([0, 0, offset, 0]), (0, 1, 2, 3))
    mx.eval(ref)
    return _bits(ref)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32, mx.uint32])
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("where", ["first", "mid", "last"])
@pytest.mark.parametrize("transposed", [False, True])
def test_rows_land(dtype, n, where, transposed):
    offset = {"first": 0, "mid": 17, "last": CAP - n}[where]
    buf = _rand((B, H, CAP, D), dtype)
    if transposed:
        # Values usually arrive as a [B, n, H, D] projection seen head-first.
        x = _rand((B, n, H, D), dtype).transpose(0, 2, 1, 3)
    else:
        x = _rand((B, H, n, D), dtype)
    mx.eval(buf, x)
    ref = _reference(buf, x, offset)
    out = kq.kv_append(buf, x, offset)
    mx.eval(out)
    np.testing.assert_array_equal(_bits(out), ref)


def test_rows_land_cpu():
    # The CPU eval on purpose, also on a box with a GPU.
    buf = _rand((B, H, CAP, D), mx.uint32)
    x = _rand((B, 3, H, D), mx.uint32).transpose(0, 2, 1, 3)
    mx.eval(buf, x)
    ref = _reference(buf, x, 9)
    out = kq.kv_append(buf, x, 9, stream=mx.cpu)
    mx.eval(out)
    np.testing.assert_array_equal(_bits(out), ref)


@metal
def test_no_allocation_while_held():
    # 256 MiB held by `buf`: a slice update copies all of it, the append
    # writes one row in place.
    buf = mx.zeros((1, 8, 131072, 128), mx.float16)
    x = mx.ones((1, 8, 1, 128), mx.float16)
    mx.eval(buf, x)

    def grows(fn):
        mx.synchronize()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        return mx.get_peak_memory() - base, out

    copied, ref = grows(
        lambda: mx.slice_update(buf, x, mx.array([0, 0, 5000, 0]), (0, 1, 2, 3))
    )
    assert copied >= buf.nbytes
    del ref
    mx.clear_cache()

    grown, out = grows(lambda: kq.kv_append(buf, x, 5000))
    assert grown < 1 << 20
    assert out[0, 0, 5000, 0].item() == 1.0


def test_aliasing_visible_and_bounded():
    buf = _rand((B, H, CAP, D), mx.float16)
    x = _rand((B, H, 2, D), mx.float16)
    mx.eval(buf, x)
    below = buf[..., :20, :]
    mx.eval(below)
    old = _bits(below).copy()
    out = kq.kv_append(buf, x, 20)
    mx.eval(out)
    # A view of the rows below the write reads its old rows.
    np.testing.assert_array_equal(_bits(below), old)
    # The full-buffer holder sees the new rows. That is the hazard the
    # caller keeps away from.
    np.testing.assert_array_equal(_bits(buf)[..., 20:22, :], _bits(x))


def test_pending_reader_of_lower_rows():
    buf = _rand((B, H, CAP, D), mx.float32)
    x = _rand((B, H, 4, D), mx.float32)
    mx.eval(buf, x)
    want = buf[..., :24, :].sum(axis=2)
    mx.eval(want)
    reader = buf[..., :24, :].sum(axis=2)
    out = kq.kv_append(buf, x, 24)  # built before the reader evaluates
    mx.eval(reader, out)
    np.testing.assert_array_equal(np.array(reader), np.array(want))
    np.testing.assert_array_equal(np.array(out)[..., 24:28, :], np.array(x))


def test_no_later_donation_into_cache():
    # An op on another holder of the buffer must not reuse it for its
    # output: the cache's rows would be overwritten.
    buf = _rand((B, H, CAP, D), mx.float32)
    x = _rand((B, H, 1, D), mx.float32)
    mx.eval(buf, x)
    out = kq.kv_append(buf, x, 7)
    mx.eval(out)
    kept = np.array(out).copy()
    y = mx.multiply(buf, 2.0)
    del buf
    mx.eval(y)
    np.testing.assert_array_equal(np.array(out), kept)


@pytest.mark.parametrize("view", ["strided", "offset"])
def test_view_is_copied_first(view):
    base = _rand((B, H, 2 * CAP, D), mx.float16)
    mx.eval(base)
    buf = base[:, :, ::2, :] if view == "strided" else base[:, :, 5 : 5 + CAP, :]
    x = _rand((B, H, 2, D), mx.float16)
    mx.eval(buf, x)
    before = _bits(base).copy()
    ref = _reference(buf, x, 11)
    out = kq.kv_append(buf, x, 11)
    mx.eval(out)
    np.testing.assert_array_equal(_bits(out), ref)
    np.testing.assert_array_equal(_bits(base), before)


@pytest.mark.parametrize(
    "x_shape, x_dtype, offset, match",
    [
        ((B, H, 1, D), mx.float32, 0, "dtype"),
        ((B, H + 1, 1, D), mx.float16, 0, "axes 0, 1 and 3"),
        ((B, H, 1, D + 1), mx.float16, 0, "axes 0, 1 and 3"),
        ((B, H, 2, D), mx.float16, CAP - 1, "within buf"),
        ((B, H, 1, D), mx.float16, -1, "within buf"),
        ((H, 1, D), mx.float16, 0, r"\[B, H, rows, D\]"),
    ],
)
def test_guards_raise(x_shape, x_dtype, offset, match):
    buf = mx.zeros((B, H, CAP, D), mx.float16)
    with pytest.raises(ValueError, match=match):
        kq.kv_append(buf, mx.zeros(x_shape, x_dtype), offset)
