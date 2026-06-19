"""Single source of truth for GGUF K-quant / legacy codec geometry.

Every other Python module (``nn``, ``recipes``, ``quantize``, the loader, and the
``scripts/check-codecs.py`` doc-lint) imports its codec facts from here so the
numbers can never drift apart. The authoritative values are the block layouts
the ``mlx_kquant`` extension's kernels implement (``kq.codecs()`` enumerates the
same set at runtime).
"""

from __future__ import annotations

# codec -> (group_size, bits, bytes_per_block, weights_per_block)
#
# K-quant superblocks pack 256 weights per block; the legacy block codecs pack
# 32. ``bytes_per_block`` is the on-disk GGUF block size (scales live inside it).
CODEC_GEOMETRY: dict[str, tuple[int, int, int, int]] = {
    "q8_0": (32, 8, 34, 32),
    "q4_0": (32, 4, 18, 32),
    "q4_1": (32, 4, 20, 32),
    "q5_0": (32, 5, 22, 32),
    "q5_1": (32, 5, 24, 32),
    "q4_k": (256, 4, 144, 256),
    "q5_k": (256, 5, 176, 256),
    "q6_k": (256, 6, 210, 256),
    "q3_k": (256, 3, 110, 256),
    "q2_k": (256, 2, 84, 256),
    # IQ codecs: grid/LUT decode (see DECODE_ONLY_CODECS for which still lack a
    # CPU encoder). iq4_nl is flat-32 like the legacy codecs; the rest are
    # 256-weight superblocks.
    "iq4_nl": (32, 4, 18, 32),
    "iq4_xs": (256, 4, 136, 256),
    "iq3_s": (256, 3, 110, 256),
    "iq3_xxs": (256, 3, 98, 256),
    "iq2_xxs": (256, 2, 66, 256),
    "iq2_xs": (256, 2, 74, 256),
    "iq2_s": (256, 2, 82, 256),
    # IQ1 (1.56 / 1.75 bpw): grid + a ±0.125 delta on each grid value; iq1_m
    # has no super-block d (reconstructed from scattered scale nibbles).
    "iq1_s": (256, 1, 50, 256),
    "iq1_m": (256, 1, 56, 256),
}

# IQ codecs that still lack a CPU encoder. Encode lands incrementally: iq4_nl
# and iq4_xs already encode (so they are no longer here); the rest load community
# GGUFs but cannot yet be produced by kq.quantize.
DECODE_ONLY_CODECS: frozenset[str] = frozenset(
    {
        "iq1_s",
        "iq1_m",
    }
)

# Every codec the ``kq.quantize`` encoder can produce: the ten K-quant/legacy
# codecs on CPU or Metal (the four legacy block codecs + q8_0 ignore an imatrix).
ENCODER_CODECS: frozenset[str] = frozenset(CODEC_GEOMETRY) - DECODE_ONLY_CODECS

# The five K-quant superblocks whose committed fixtures are minted with an
# imatrix (gen_fixtures.py). Pinned to an explicit literal, NOT derived from
# wpb == 256: the IQ superblocks are wpb == 256 too and several are imatrix-
# sensitive, but their encode is CPU-only with no committed GPU fixtures, so they
# must stay out of this fixture-minting set. IQ imatrix behavior is asserted
# separately (registry-driven) in test_encode.py.
IMATRIX_CODECS: frozenset[str] = frozenset({"q2_k", "q3_k", "q4_k", "q5_k", "q6_k"})


def geometry(codec: str) -> tuple[int, int, int, int]:
    """Return ``(group_size, bits, bytes_per_block, weights_per_block)``."""
    try:
        return CODEC_GEOMETRY[codec]
    except KeyError:
        raise ValueError(
            f"unknown codec {codec!r}; known codecs: "
            f"{', '.join(sorted(CODEC_GEOMETRY))}"
        ) from None


def bytes_per_row(codec: str, in_dims: int) -> int:
    """uint8 wire-byte width of one quantized row of ``in_dims`` weights."""
    gs, bits, bpb, wpb = geometry(codec)
    if in_dims % wpb != 0:
        raise ValueError(
            f"codec {codec!r} packs {wpb} weights/block; row width {in_dims} "
            f"is not a multiple of {wpb}"
        )
    return (in_dims // wpb) * bpb


def in_features(codec: str, row_bytes: int) -> int:
    """Inverse of :func:`bytes_per_row`: the logical input width of a quantized
    row given its uint8 wire-byte length. The kquant modules store only the wire
    width, so this recovers the in-features the LoRA wrappers need."""
    gs, bits, bpb, wpb = geometry(codec)
    if row_bytes % bpb != 0:
        raise ValueError(
            f"codec {codec!r} uses {bpb}-byte blocks; row width {row_bytes} "
            f"is not a multiple of {bpb}"
        )
    return (row_bytes // bpb) * wpb
