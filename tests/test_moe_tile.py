"""Tiled score-mixed down of the fused MoE ops (KQ_MOE_TILE) at the verify
widths.

From KQ_MOE_TILE_MIN_T rows (8) to 16 the score-mixed downs sort their
(row, slot) pairs by expert into tiles of up to eight and read each tile's
expert once (kq_moe_tile.h). It needs the tensor-op units; elsewhere, for
nvfp4, with output scales and with KQ_MOE_TILE=0 the grouped kernels run.
Each arm runs in a subprocess so the levers are read from the first
dispatch. The base arm is the per-row kernels (KQ_MOE_GROUP=0); the tiled
arm sets KQ_MOE_TILE_MIN_T=4 unless a test passes its own. A tiled
row rounds each pair's sums to the output type before the mix and its
weights to half, so the arms match within two output steps.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="fused MoE gathers are Metal-only kernels; no CPU path.",
)

HERE = os.path.dirname(os.path.abspath(__file__))
# Every expert codec of the fused gathers.
CODECS = (
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "q8_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "iq4_nl",
    "iq4_xs",
    "iq3_s",
    "iq3_xxs",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq1_s",
    "iq1_m",
    "stq1_0",
    "pq2_0",
    "q2_0",
    "ptq1_0",
    "mxfp4",
    "nvfp4",
)
UPCASTS = ("q5_k", "q6_k", "q8_0")
TYPES = {"float16": "float16_t", "bfloat16": "bfloat16_t"}
# Two output steps at the largest value.
STEP = {"float16": 4e-3, "bfloat16": 2.0**-6}

_SNIPPET = r"""
import json
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
from kqref import is_synth, synth_wire
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

codec, cfg, out_path = sys.argv[2], json.loads(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(31)
E, N, K, R = 12, 256, cfg.get("kout", 512), 5

SCALES = [None]

def wire(lead, n, k, c):
    bpb, wpb = CODEC_GEOMETRY[c][2:]
    if is_synth(c) or c == "mxfp4":
        blocks = int(np.prod(lead, dtype=np.int64)) * n * (k // wpb)
        w = synth_wire(rng, c, bpb, blocks, d_range=(0.004, 0.01),
                       nvfp4_scales=(0x08, 0x19))
        return mx.array(w.reshape(*lead, n, (k // wpb) * bpb))
    w = mx.array((rng.standard_normal((*lead, n, k)) * 0.1).astype(np.float32))
    wq, SCALES[0] = kq.quantize(w, c)
    mx.eval(wq)
    return wq

# The down projection reads h [T, S, N] and writes [T, K].
dw = wire((E,), K, N, codec)
dws = SCALES[0]
shared = [(c, wire((), K, c["k"] * N, c["sd"])) for c in cfg["cases"]]
res = {}
for T, picks in cfg["widths"]:
    if picks == "topk":
        # a router's picks: distinct within a row, some shared between rows
        inds = np.stack([rng.permutation(E)[:R] for _ in range(T)])
    else:
        # three experts for every pair: an expert's pairs fill more than
        # one tile
        inds = rng.integers(0, 3, size=(T, R))
    inds = mx.array(inds.astype(np.uint32))
    for dname in ("float16", "bfloat16"):
        dt = getattr(mx, dname)
        def act(shape):
            a = (rng.standard_normal(shape) * 0.02).astype(np.float32)
            return mx.array(a).astype(dt)
        h = act((T, R, N))
        sc = mx.array(rng.uniform(0.05, 0.9, size=(T, R)).astype(np.float32))
        tag = f"{T}{picks}/{dname}"
        res[f"{tag}/mix_ns"] = kq.gather_qmv_mix_ns_kq(h, dw, codec, inds, sc)
        if cfg.get("ref"):
            # the same sum from the dequantized weights
            wd = kq.dequantize(dw, dws, codec).astype(mx.float32)
            wd = wd.reshape(E, K, N)[inds]
            y = (wd @ h.astype(mx.float32)[..., None])[..., 0]
            res[f"{tag}/ref"] = (y * sc[..., None]).sum(axis=1)
        for i, (c, sdw) in enumerate(shared):
            k = c["k"]
            hs = act((T, R + k, N))
            scs = mx.array(
                rng.uniform(0.05, 0.9, size=(T, R + k)).astype(np.float32))
            res[f"{tag}/mix{i}"] = kq.gather_qmv_mix_kq(
                hs, dw, sdw, codec, inds, scs, shexp_kquant_type=c["sd"])
            res[f"{tag}/unit{i}"] = kq.gather_qmv_mix_kq(
                hs, dw, sdw, codec, inds, scs[:, :R], shexp_kquant_type=c["sd"])
mx.eval(res)
np.savez(out_path, **{k: np.array(v.astype(mx.float32)) for k, v in res.items()})
"""

WIDTHS = [[8, "topk"], [13, "few"], [4, "topk"], [16, "topk"]]


def _run(codec, cfg, arm, tmp_path, env):
    out = tmp_path / f"tl{arm}.npz"
    lever = {
        "base": {"KQ_MOE_GROUP": "0"},
        "tile": {"KQ_MOE_GROUP": "1", "KQ_MOE_TILE": "1", "KQ_MOE_TILE_MIN_T": "4"},
    }[arm]
    r = subprocess.run(
        [sys.executable, "-c", _SNIPPET, HERE, codec, json.dumps(cfg), str(out)],
        capture_output=True,
        text=True,
        env={**os.environ, "KQ_MOE_NX_LOG": "1", **lever, **env},
    )
    assert r.returncode == 0, (
        f"{codec} arm {arm} exited {r.returncode}\n"
        f"--- child stdout ---\n{r.stdout}\n--- child stderr ---\n{r.stderr}"
    )
    names = {ln.split()[1] for ln in r.stderr.splitlines() if ln[:9] == "[kq_moe] "}
    return np.load(out), names


def _both(codec, cfg, tmp_path, env=None):
    """The base and tiled arms of one codec; the tiled arm's kernel names."""
    ref, ref_names = _run(codec, cfg, "base", tmp_path, {})
    got, names = _run(codec, cfg, "tile", tmp_path, env or {})
    assert not [n for n in ref_names if "_tl_" in n or "tile" in n], sorted(ref_names)
    for key in ref.files:
        a, b = ref[key], got[key]
        assert np.isfinite(b).all() and np.abs(a).max() > 0, (codec, key)
        rel = np.abs(a - b).max() / np.abs(a).max()
        assert rel <= STEP[key.split("/")[1]], (codec, key, rel)
    return got, names


def _tiled(names):
    return sorted(n for n in names if "_tl_" in n or "kq_moe_tile" in n)


def _stem(codec, sx):
    if sx == codec and codec in ("q6_k", "q8_0"):
        return f"kq_{codec}_ext_sx_{codec}_ext"
    return f"kq_{codec}_sx_{sx}"


def _one(s):
    return {"sd": s, "k": 1}


@pytest.fixture(scope="module")
def nax():
    """Whether this GPU has the tensor-op units the tiled kernels need."""
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, numpy as np, mlx.core as mx, mlx_kquant as kq\n"
            "rng = np.random.default_rng(3)\n"
            "w, _ = kq.quantize(mx.array(rng.standard_normal((4, 64, 64))"
            ".astype(np.float32)), 'q4_0')\n"
            "h = mx.zeros((4, 2, 64), dtype=mx.float16)\n"
            "i = mx.zeros((4, 2), dtype=mx.uint32)\n"
            "mx.eval(kq.gather_qmv_mix_ns_kq(h, w, 'q4_0', i, mx.ones((4, 2))))\n",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "KQ_MOE_NX_LOG": "1",
            "KQ_MOE_GROUP": "1",
            "KQ_MOE_TILE": "1",
        }
        | {"KQ_MOE_TILE_MIN_T": "4"},
    )
    assert r.returncode == 0, r.stderr
    return "kq_moe_tile_plan" in r.stderr


@pytest.mark.parametrize("codec", CODECS)
def test_tile_every_codec(codec, tmp_path, nax):
    """Each expert codec at 4, 8, 13 and 16 rows, with picks that share some
    experts and picks that fill several tiles of one: the score-mixed down
    with no shared expert and with each shared codec the codec pairs with,
    with and without a score for the shared slot. The dispatch log names the
    tiled kernel of each; nvfp4 keeps the grouped kernels."""
    shared = list(dict.fromkeys((codec, *UPCASTS)))
    cfg = {"cases": [_one(s) for s in shared], "widths": WIDTHS}
    _, names = _both(codec, cfg, tmp_path)
    if codec == "nvfp4" or not nax:
        assert not _tiled(names), _tiled(names)
        return
    for t in TYPES.values():
        for sx in shared:
            want = f"{_stem(codec, sx)}_gather_qmv_mix_tl_{t}"
            assert want in names, (want, sorted(names))
        assert f"kq_moe_tile_plan_{t}" in names, sorted(names)
        assert f"kq_moe_tile_combine_{t}" in names, sorted(names)
    mine = {n for n in names if n.startswith(f"kq_{codec}_")}
    assert all("_tl_" in n for n in mine), sorted(mine)


@pytest.mark.parametrize("rows", ["0", "1"])
def test_tile_column_rows_lever(rows, tmp_path, nax):
    """KQ_MOE_TILE_COL_ROWS moves tiles of one and two rows between the
    column kernel and the staged one; q5_0 and iq4_nl have their own block
    decode in both."""
    for codec in ("q5_0", "iq4_nl", "iq3_xxs"):
        cfg = {"cases": [_one("q8_0")], "widths": WIDTHS}
        _, names = _both(codec, cfg, tmp_path, {"KQ_MOE_TILE_COL_ROWS": rows})
        assert bool(_tiled(names)) == nax


@pytest.mark.parametrize("codec", ["q5_0", "q4_k", "q8_0"])
def test_tile_matches_dequantized(codec, tmp_path, nax):
    """The tiled down against the same sum from the dequantized weights."""
    cfg = {"cases": [], "widths": [[8, "topk"], [13, "few"]], "ref": True}
    got, names = _both(codec, cfg, tmp_path)
    assert bool(_tiled(names)) == nax
    for key in got.files:
        if not key.endswith("/mix_ns"):
            continue
        a, b = got[key[: -len("mix_ns")] + "ref"], got[key]
        rel = np.abs(a - b).max() / np.abs(a).max()
        assert rel <= STEP[key.split("/")[1]], (codec, key, rel)


@pytest.mark.parametrize(
    "codec,sd,k",
    [
        ("q4_1", "q4_1", 2),
        ("q4_1", "q4_1", 4),
        ("q4_k", "q6_k", 2),
        ("q8_0", "q8_0", 2),
        ("iq2_xs", "q6_k", 2),
        ("q2_0", "iq4_nl", 1),
        ("q2_0", "iq4_nl", 2),
    ],
)
def test_tile_shared_slots(codec, sd, k, tmp_path, nax):
    """A shared expert k routed experts wide, and the q2_0 stack with an
    iq4_nl shared down."""
    cfg = {"cases": [{"sd": sd, "k": k}], "widths": WIDTHS}
    _, names = _both(codec, cfg, tmp_path)
    assert bool(_tiled(names)) == nax


def test_tile_ragged_columns(tmp_path, nax):
    """An output width that is not a multiple of the 64 columns a
    threadgroup takes."""
    cfg = {"cases": [_one("q8_0")], "widths": WIDTHS, "kout": 328}
    _, names = _both("q5_0", cfg, tmp_path)
    assert bool(_tiled(names)) == nax


@pytest.mark.parametrize(
    "env",
    [
        {"KQ_MOE_TILE": "0"},
        {"KQ_MOE_TILE_MIN_T": "17"},
        {"KQ_MOE_MIX_FORM": "loop"},
    ],
)
def test_tile_levers_keep_other_kernels(env, tmp_path):
    """KQ_MOE_TILE=0, a floor above the widths and a forced mix form each
    keep the other kernels."""
    cfg = {"cases": [_one("q8_0")], "widths": WIDTHS}
    _, names = _both("q5_0", cfg, tmp_path, env)
    assert not _tiled(names), _tiled(names)


def test_tile_min_rows_lever(tmp_path, nax):
    """The default floor is 8 rows: a 7-row call keeps the grouped
    kernels and an 8-row call is tiled. KQ_MOE_TILE_MIN_T=2 tiles a
    two-row call."""
    cfg = {"cases": [_one("q8_0")], "widths": [[7, "topk"], [2, "topk"], [3, "few"]]}
    _, names = _both("q5_0", cfg, tmp_path, {"KQ_MOE_TILE_MIN_T": ""})
    assert not _tiled(names), _tiled(names)
    cfg8 = {"cases": [_one("q8_0")], "widths": [[8, "topk"]]}
    _, names = _both("q5_0", cfg8, tmp_path, {"KQ_MOE_TILE_MIN_T": ""})
    assert bool(_tiled(names)) == nax
    _, names = _both("q5_0", cfg, tmp_path, {"KQ_MOE_TILE_MIN_T": "2"})
    assert bool(_tiled(names)) == nax
