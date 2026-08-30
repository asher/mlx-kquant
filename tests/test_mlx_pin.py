"""Pinned-mlx behavior this package relies on.

mlx < 0.32.2 encoded a strided quantized operand's contiguity copy after
binding earlier operands (ml-explore/mlx#4381): dequantize read the copy
kernel's buffers whenever biases arrived strided. Fixed at the 0.32.2 pin;
these tests hold the fix so a venv drifting off the pin fails loudly.
"""

import itertools

import mlx.core as mx
import pytest

GROUP, BITS = 64, 8


def _strided(t):
    # row-prefix slice of a taller buffer: lazy, strided when B*heads > 1
    pad = mx.concatenate([t, mx.zeros_like(t)], axis=-2)
    return pad[..., : t.shape[-2], :]


def _quantized(shape):
    mx.random.seed(3)
    x = mx.random.normal(shape, dtype=mx.float16)
    return x, mx.quantize(x, group_size=GROUP, bits=BITS)


def test_mlx_pin_version():
    assert mx.__version__ == "0.32.2", (
        f"mlx {mx.__version__}: package is certified against the 0.32.2 pin"
    )


@pytest.mark.parametrize(
    "sw,ss,sb",
    [c for c in itertools.product([False, True], repeat=3) if any(c)],
)
def test_dequantize_strided_operands(sw, ss, sb):
    x, (w, scales, biases) = _quantized((2, 4, 96, 128))
    ref = mx.dequantize(w, scales, biases, group_size=GROUP, bits=BITS)
    got = mx.dequantize(
        _strided(w) if sw else w,
        _strided(scales) if ss else scales,
        _strided(biases) if sb else biases,
        group_size=GROUP,
        bits=BITS,
    )
    err = float(mx.abs(got - ref).max().item())
    # fp16 dequant is exact vs itself; any operand misread lands ~1e3 off
    assert err == 0.0, f"strided (w={sw},s={ss},b={sb}) diverged: {err}"


def test_quantize_strided_input():
    x, (w, scales, biases) = _quantized((2, 4, 96, 128))
    w2, s2, b2 = mx.quantize(_strided(x), group_size=GROUP, bits=BITS)
    assert bool(mx.all(w2 == w).item())
    assert bool(mx.all(s2 == scales).item())
    assert bool(mx.all(b2 == biases).item())
