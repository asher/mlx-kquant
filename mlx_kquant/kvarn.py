"""KVarN rotation glue.

The KVarN pipeline (kvarn_quantize / kvarn_dequant) operates on WHT-rotated
head vectors. The rotation is mlx's own hadamard butterfly, so no custom
kernel is needed; this wrapper pins the numerics that make it byte-compatible
with BeeLlama's composite WHT (per-slice WHT-128 plus cross-slice butterfly,
H_s x H_128): the butterfly runs in float32 whatever the input dtype, with a
single rounding on output. mlx's fp16 hadamard rounds between stages and
deviates by up to 2^-7, so the upcast is load-bearing.

At head_dim 128 the fp32 butterfly and the fp16 staging path are bit-exact
against the BeeLlama CPU reference. At 256/512 mlx's stage order differs in
last fp32 ulps; after fp16 rounding a fraction well under 0.1% of cells moves
by one ulp (near-zero cells by a few subnormal steps). tests/test_kvarn.py
pins both statements.
"""

from __future__ import annotations

import mlx.core as mx

_HEAD_DIMS = (128, 256, 512)


def kvarn_rotate(
    x: mx.array, *, dtype: mx.Dtype | None = None, stream=None
) -> mx.array:
    """Normalized WHT over the trailing head dim; self-inverse.

    Args:
        x: [..., D] with D in {128, 256, 512}, any float dtype.
        dtype: output dtype. Default ``x.dtype``.

    Returns:
        The rotated array. Applying the same call again recovers the input
        up to output-dtype rounding.
    """
    d = x.shape[-1]
    if d not in _HEAD_DIMS:
        raise ValueError(
            f"[mlx_kquant.kvarn_rotate] head dim must be one of {_HEAD_DIMS}, got {d}."
        )
    out = mx.hadamard_transform(
        x.astype(mx.float32, stream=stream), scale=d**-0.5, stream=stream
    )
    out_dtype = x.dtype if dtype is None else dtype
    return out.astype(out_dtype, stream=stream)
