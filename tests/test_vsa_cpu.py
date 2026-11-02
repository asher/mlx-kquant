"""vsa_attention's tile sums on a CPU stream. A dense call with a gate inside
mx.stream(mx.cpu) pools through this branch, MLX's float32 copy and sum of
the tiles. tests/test_vsa_attention.py runs on Metal and skips under
KQUANT_FORCE_CPU; this test runs on every device.
"""

import mlx.core as mx
import numpy as np

from mlx_kquant import vsa

TILE = 64
D = 128


def test_tile_sums_on_a_cpu_stream_have_the_bits_of_mlx_float32_sums():
    rng = np.random.default_rng(53)
    n, heads = 7, 3
    x = rng.standard_normal((n * TILE, heads, D)).astype(np.float32)
    x = mx.array(x).astype(mx.bfloat16)
    with mx.stream(mx.cpu):
        sums = vsa._vsa_tile_sums(x)
        ref = x.astype(mx.float32).reshape(n, TILE, heads, D).sum(axis=1)
    np.testing.assert_array_equal(
        np.array(sums.view(mx.uint32)), np.array(ref.view(mx.uint32))
    )
