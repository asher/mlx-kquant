"""KQ_NAX_TILE and KQ_QMM_TILE: pinning the quantized matmul tile.

Each case runs in a fresh interpreter (tests/qmm_probe.py) with
``KQ_TUNE_LIVE=1`` unless it tests the cached read, and reads the dispatched
kernel from ``KQ_QMM_LOG``. The numeric cases also run on the CPU stream
under ``KQUANT_FORCE_CPU``, where the overrides must leave the result alone.
The enumerating case dispatches every kernel name in the metallib that an
override can select and checks each against float64, which catches a
silently empty kernel.
"""

from __future__ import annotations

import re

import pytest
from qmm_probe import force_cpu, run_probe, tile_of

import mlx_kquant as kq
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

# R1 bounds: relative Frobenius error against float64 with exact
# kq.dequantize weights.
R1 = {"bfloat16": 6e-3, "float16": 1e-3}
TYPE_DTYPE = {"bfloat16_t": "bfloat16", "float16_t": "float16"}

K = 256
LIVE = {"KQ_TUNE_LIVE": "1", "KQ_QMM_LOG": "1"}
TILES = {32: "32,64,2,2", 64: "64,64,2,2", 128: "128,64,2,2"}
MAIN_CODECS = ["q4_k", "q5_k", "q6_k", "q8_0"]


def _metal():
    if force_cpu():
        pytest.skip("Metal kernel names only")


def _nax():
    _metal()
    if not kq.nax_available():
        pytest.skip("NAX tile")


def _only_kernel(res):
    assert res["error"] is None, res["error"]
    assert len(res["kernels"]) == 1, res["kernels"]
    return res["kernels"][0]


def test_nax_tile_selects_kernel():
    _nax()
    # 4128 rows under the parity rule: the policy runs BM=64 (ceil(4128/64)
    # is odd).
    call = {"codec": "q4_k", "shape": [1, 4128, K], "n": 128}
    calls = [
        dict(call),
        dict(call, env={"KQ_NAX_TILE": TILES[128]}),
        dict(call, env={"KQ_NAX_TILE": TILES[32]}),
        dict(call, env={"KQ_NAX_TILE": None}),
    ]
    res = run_probe(calls, env=dict(LIVE, KQ_NAX_BM128_ODD_M="0"), check=False)
    got = [_only_kernel(r) for r in res]
    assert [tile_of(n)[0] for n, _ in got] == [64, 128, 32, 64]
    assert [g for _, g in got] == [(2, 65, 1), (2, 33, 1), (2, 129, 1), (2, 65, 1)]


def test_nax_tile_keeps_first_read():
    _nax()
    call = {"codec": "q4_k", "shape": [1, 4128, K], "n": 128}
    calls = [dict(call), dict(call, env={"KQ_NAX_TILE": None})]
    res = run_probe(
        calls, env={"KQ_QMM_LOG": "1", "KQ_NAX_TILE": TILES[128]}, check=False
    )
    assert [tile_of(_only_kernel(r)[0])[0] for r in res] == [128, 128]


def test_nax_tile_uninstantiated_raises():
    _nax()
    call = {"codec": "q4_k", "shape": [300, K], "n": 128}
    calls = [
        dict(call, env={"KQ_NAX_TILE": "128,128,2,4"}),
        # The process keeps working after the raise.
        dict(call, env={"KQ_NAX_TILE": TILES[64]}),
    ]
    res = run_probe(calls, env=LIVE)
    name = (
        "kquant_q4_k_qmm_t_nax_bfloat16_t_gs_256_b_4_bm128_bn128_bk64_wm2_wn4"
        "_alN_true_batch_0"
    )
    assert res[0]["error"].startswith("ValueError"), res[0]["error"]
    assert name in res[0]["error"] and "KQ_NAX_TILE=128,128,2,4" in res[0]["error"]
    assert tile_of(_only_kernel(res[1])[0])[0] == 64
    assert res[1]["rel"] < R1["bfloat16"]


def test_qmm_tile_selects_and_raises():
    # The path without NAX, reached with KQ_DISABLE_NAX=1 on NAX GPUs.
    _metal()
    call = {"codec": "q4_k", "shape": [300, K], "n": 128}
    calls = [
        dict(call, env={"KQ_QMM_TILE": TILES[64]}),
        dict(call, env={"KQ_QMM_TILE": TILES[128]}),
        dict(call, env={"KQ_QMM_TILE": None}),
    ]
    res = run_probe(calls, env={**LIVE, "KQ_DISABLE_NAX": "1"})
    plain = "kquant_q4_k_qmm_t_bfloat16_t_gs_256_b_4_alN_true_batch_0"
    assert _only_kernel(res[0])[0] == plain
    assert res[0]["rel"] < R1["bfloat16"]
    tagged = (
        "kquant_q4_k_qmm_t_bfloat16_t_gs_256_b_4_bm128_bn64_wm2_wn2_alN_true_batch_0"
    )
    assert res[1]["error"].startswith("ValueError"), res[1]["error"]
    assert tagged in res[1]["error"] and "KQ_QMM_TILE=128,64,2,2" in res[1]["error"]
    assert _only_kernel(res[2])[0] == plain


@pytest.mark.parametrize(
    "lever,value",
    [
        ("KQ_NAX_TILE", "128,64"),
        ("KQ_NAX_TILE", "0,64,2,2"),
        ("KQ_QMM_TILE", "a,b,c,d"),
    ],
)
def test_tile_malformed_raises(lever, value):
    _metal()
    if lever == "KQ_NAX_TILE" and not kq.nax_available():
        pytest.skip("NAX tile")
    env = {**LIVE, lever: value}
    if lever == "KQ_QMM_TILE":
        env["KQ_DISABLE_NAX"] = "1"
    res = run_probe([{"codec": "q4_k", "shape": [300, K], "n": 128}], env=env)
    err = res[0]["error"]
    assert err is not None and err.startswith("ValueError"), err
    assert f"{lever}={value}" in err


def _numeric_calls(codec, dtype, tiles):
    calls = []
    for layout in ("dense", "transposed", "strided"):
        for shape in ([1, 260, K], [2, 130, K]):
            n = 256 if layout == "dense" else 200
            base = len(calls)
            for j, bm in enumerate(tiles):
                calls.append(
                    {
                        "codec": codec,
                        "shape": shape,
                        "n": n,
                        "dtype": dtype,
                        "layout": layout,
                        "seed": base,
                        "env": {"KQ_NAX_TILE": TILES.get(bm)},
                        "ulp_vs": base if j else None,
                    }
                )
    return calls


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("codec", MAIN_CODECS)
def test_tile_numerics(codec, dtype):
    # Every tile meets the R1 bound on batch 1, batch 2, lazy transposed and
    # strided inputs, and agrees with the BM=64 tile within 1 ulp. On the
    # CPU stream the override does nothing and the bound still holds.
    tiles = [64, None, 32, 128]
    calls = _numeric_calls(codec, dtype, tiles)
    res = run_probe(calls, env=LIVE)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] < R1[dtype], (c, r["rel"])
        if r["ulp"] is not None:
            assert r["ulp"] <= 1, (c, r["ulp"])
        if not force_cpu() and kq.nax_available() and c["env"]["KQ_NAX_TILE"]:
            want = int(c["env"]["KQ_NAX_TILE"].split(",")[0])
            assert tile_of(_only_kernel(r)[0])[0] == want


def _qmm_codecs():
    return sorted(c for c in CODEC_GEOMETRY if kq.codec_has_matmul(c))


def _nax_tiles(codec):
    """BM of every unsplit qmm_t NAX tile of `codec` in the metallib."""
    out = set()
    for name in kq.metallib_kernels():
        m = _NAX_NAME.match(name)
        if m and m["codec"] == codec and m["db"] is None and m["bn"] == "64":
            out.add(int(m["bm"]))
    return sorted(out)


@pytest.mark.parametrize("codec", _qmm_codecs())
def test_tile_rows_above_int16(codec):
    # 32768 and 32769 rows (an even and an odd ceil(M/64)), and a batch of
    # two at 16896 rows flattened to 33792, past the int16 row span the tile
    # clamps before narrowing, on every tile the codec carries and on folded
    # row-tile orders (KQ_NAX_SWIZZLE).
    nax = not force_cpu() and kq.nax_available()
    envs = [{}]
    tiles = _nax_tiles(codec) if nax else []
    if tiles:
        # BM=64 first: every other tile agrees with it within 1 ulp.
        tiles.sort(key=lambda bm: bm != 64)
        envs = [{"KQ_NAX_TILE": TILES[bm]} for bm in tiles]
        envs += [
            {"KQ_NAX_TILE": TILES[64], "KQ_NAX_SWIZZLE": "2"},
            {"KQ_NAX_TILE": TILES[128], "KQ_NAX_SWIZZLE": "1"},
            {"KQ_NAX_TILE": TILES[128], "KQ_NAX_SWIZZLE": "2"},
        ]
    calls = []
    for shape in ([1, 32768, K], [1, 32769, K], [2, 16896, K]):
        for n in (96, 128):
            base = len(calls)
            for j, env in enumerate(envs):
                full = {"KQ_NAX_TILE": None, "KQ_NAX_SWIZZLE": None, **env}
                calls.append(
                    {
                        "codec": codec,
                        "shape": shape,
                        "n": n,
                        "seed": base,
                        "env": full,
                        "ulp_vs": base if j else None,
                    }
                )
    res = run_probe(calls, env=LIVE)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c, r["error"])
        assert r["rel"] < R1["bfloat16"], (c, r["rel"])
        if r["ulp"] is not None:
            assert r["ulp"] <= 1, (c, r["ulp"])
        if tiles and c["env"]["KQ_NAX_TILE"]:
            want = int(c["env"]["KQ_NAX_TILE"].split(",")[0])
            assert tile_of(_only_kernel(r)[0])[0] == want, c


_NAX_NAME = re.compile(
    r"^kquant_(?P<codec>.+?)_qmm_t_nax_(?P<type>bfloat16_t|float16_t|float)"
    r"_gs_\d+_b_\d+_bm(?P<bm>\d+)_bn(?P<bn>\d+)_bk(?P<bk>64|128)_wm(?P<wm>\d+)"
    r"_wn(?P<wn>\d+)"
    r"_alN_(?P<al>true|false)_batch_(?P<batch>[01])(?P<db>_db)?$"
)
_ALU_NAME = re.compile(
    r"^kquant_(?P<codec>.+?)_qmm_t_(?P<type>bfloat16_t|float16_t|float)"
    r"_gs_\d+_b_\d+(?:_bm(?P<bm>\d+)_bn(?P<bn>\d+)_wm(?P<wm>\d+)_wn(?P<wn>\d+))?"
    r"_alN_(?P<al>true|false)_batch_(?P<batch>[01])$"
)


def test_tile_names_match_known_patterns():
    # Every qmm_t tile kernel outside the split-K, gather, GLU, low-rank and
    # residual families must match a pattern above, so a new tile family cannot
    # escape the enumerating test below. test_matmul_glu.py holds the GLU names,
    # test_matmul_lora.py the low-rank ones and test_matmul_resid.py the
    # residual ones.
    _metal()
    names = kq.metallib_kernels()
    other = (
        "_gather_",
        "_qmm_t_nax_splitk_",
        "_qmm_t_nax_glu_",
        "_qmm_t_nax_lora_",
        "_qmm_t_nax_resid_",
    )
    nax = [n for n in names if "_qmm_t_nax_" in n and not any(o in n for o in other)]
    alu = [
        n
        for n in names
        if re.search(r"_qmm_t_(bfloat16_t|float16_t|float)_", n) and "_gather_" not in n
    ]
    assert alu, "no qmm_t kernels in the metallib"
    for pat, group in ((_NAX_NAME, nax), (_ALU_NAME, alu)):
        bad = [
            n
            for n in group
            if not (m := pat.match(n)) or m["codec"] not in CODEC_GEOMETRY
        ]
        assert not bad, bad[:5]


def _bk128_lever(codec):
    if codec.endswith("_k"):
        return "KQ_NAX_" + codec.replace("_", "").upper() + "_BK128"
    return "KQ_NAX_" + codec.upper() + "_BK128"


def _selectable(codec):
    """Calls that dispatch every qmm_t tile kernel of `codec` an override can
    select: half and bfloat activations, unbatched (the op flattens a batch
    into rows and never builds a float activation for the tile)."""
    calls, other = [], []
    for name in kq.metallib_kernels():
        for kind, pat in (("nax", _NAX_NAME), ("alu", _ALU_NAME)):
            m = pat.match(name)
            if not m or m["codec"] != codec:
                continue
            if m["type"] == "float" or m["batch"] == "1":
                other.append(name)
                continue
            if kind == "nax" and not kq.nax_available():
                continue
            tile = ",".join(
                m[g] or d
                for g, d in zip("bm bn wm wn".split(), "64 64 2 2".split(), strict=True)
            )
            if kind == "nax":
                db = m["db"] is not None
                rows = 48 if db else max(133, 2 * int(m["bm"]) + 5)
                env = {
                    "KQ_NAX_TILE": tile,
                    "KQ_NAX_DB64": "1" if db else "0",
                    _bk128_lever(codec): "1" if m["bk"] == "128" else "0",
                    "KQ_QMM_MIDM": "0",
                    "KQ_DISABLE_NAX": None,
                    "KQ_QMM_TILE": None,
                }
            else:
                rows = 133
                env = {
                    "KQ_QMM_TILE": tile,
                    "KQ_DISABLE_NAX": "1",
                    "KQ_NAX_TILE": None,
                }
            calls.append(
                {
                    "codec": codec,
                    "shape": [rows, K],
                    "n": 128 if m["al"] == "true" else 96,
                    "dtype": TYPE_DTYPE[m["type"]],
                    "env": env,
                    "name": name,
                }
            )
    return calls, other


@pytest.mark.parametrize("codec", sorted(CODEC_GEOMETRY))
def test_enumerate_tile_kernels(codec):
    _metal()
    calls, _ = _selectable(codec)
    if not kq.codec_has_matmul(codec):
        assert not calls
        return
    assert calls, f"no qmm_t tile kernels found for {codec}"
    res = run_probe(calls, env=LIVE)
    for c, r in zip(calls, res, strict=True):
        assert r["error"] is None, (c["name"], r["error"])
        assert _only_kernel(r)[0] == c["name"]
        assert r["rel"] < R1[c["dtype"]], (c["name"], r["rel"])
