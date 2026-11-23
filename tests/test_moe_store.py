"""Expert store ops of the fused MoE block (kq_moe_store.h).

moe_store_pack copies routed experts into planes, moe_store_glu and
moe_store_down read them. Checked per store codec (the four with a plane a
block field and every codec of the generic format), with a shared expert in
the routed codec and in q8_0, at 1 to 16 rows, in every kernel form:
  - with no K split (the pair-grid kernels at one simdgroup a weight row,
    and the plan and tile kernels) the down equals gather_qmv_mix_kq's
    tiled kernel bit for bit;
  - a store holding half the experts (the rest read in place) equals the
    full store bit for bit, on the GLU and the down, in every form and
    under the per-width defaults;
  - the GLU and the K-split down match the in-place ops within the tile
    kernels' rounding (weights rounded to half, parts summed in order);
  - a q5_0 plane holds the bytes the layout comment says.
The arms run in a subprocess so the tile levers are read from the first
dispatch; the store levers are read per call.
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="expert store ops are Metal-only kernels; no CPU path.",
)

HERE = os.path.dirname(os.path.abspath(__file__))
# Formats with a plane a block field, then the codecs of the generic format
# (one plane of whole blocks).
CODECS = ("q5_0", "iq4_nl", "iq3_xxs", "iq4_xs")
GENERIC = (
    "iq2_s",
    "q2_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "q8_0",
    "q4_0",
    "q4_1",
    "q5_1",
    "iq3_s",
    "iq2_xxs",
    "iq2_xs",
    "iq1_s",
    "iq1_m",
    "stq1_0",
    "pq2_0",
    "ptq1_0",
    "mxfp4",
)

_SNIPPET = r"""
import json
import os
import sys
sys.path.insert(0, sys.argv[4])
import numpy as np
import mlx.core as mx
import mlx_kquant as kq
from kqref import is_synth, synth_wire
from mlx_kquant.codec_geometry import CODEC_GEOMETRY

# Kernel forms by their levers: the per-width defaults, the pair-grid form
# with no K split, with K split 2 and 4 ways, with the shared expert on the
# qf row loops, the plan and tile kernels, and the in-place pair form on
# the planes (iq3_xxs, up to 3 rows).
LEVERS = ("KQ_MOE_STORE_FORM", "KQ_MOE_STORE_GLU_QR", "KQ_MOE_STORE_GLU_QH",
          "KQ_MOE_STORE_DOWN_QR", "KQ_MOE_STORE_DOWN_QH",
          "KQ_MOE_STORE_SHEXP_QF", "KQ_MOE_STORE_GLU_MIN_T",
          "KQ_MOE_STORE_DOWN_MIN_T", "KQ_MOE_STORE_MAX_T",
          "KQ_MOE_STORE_GLU_DD", "KQ_MOE_STORE_PG_MAX_T")


def form(pg, q, qf, dd="0"):
    return {"KQ_MOE_STORE_FORM": pg, "KQ_MOE_STORE_GLU_QR": q,
            "KQ_MOE_STORE_GLU_QH": q, "KQ_MOE_STORE_DOWN_QR": q,
            "KQ_MOE_STORE_DOWN_QH": q, "KQ_MOE_STORE_SHEXP_QF": qf,
            "KQ_MOE_STORE_GLU_MIN_T": "1", "KQ_MOE_STORE_DOWN_MIN_T": "1",
            "KQ_MOE_STORE_MAX_T": "64", "KQ_MOE_STORE_GLU_DD": dd,
            "KQ_MOE_STORE_PG_MAX_T": "64"}


ARMS = {"default": {}, "q1": form("1", "1", "0"), "q2": form("1", "2", "0"),
        "q4": form("1", "4", "0"), "qf": form("1", "2", "16"),
        "plan": form("0", "1", "0"), "dd": form("1", "2", "0", "3")}


def set_arm(arm):
    for v in LEVERS:
        os.environ.pop(v, None)
    os.environ.update(ARMS[arm])

codec, shexp, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"supported": bool(kq.moe_store_supported(codec, shexp))}
if not res["supported"]:
    json.dump(res, open(out_path, "w"))
    raise SystemExit(0)
rng = np.random.default_rng(31)
E, R = 12, 5
ABSENT = 0x80000000
# Synthetic wires hold large weights: small rows keep the float16 GLU's
# gate times up product finite.
XS = 0.05 if (is_synth(codec) or codec == "mxfp4") else 0.5


def wire(lead, n, k, c):
    bpb, wpb = CODEC_GEOMETRY[c][2:]
    if is_synth(c) or c == "mxfp4":
        blocks = int(np.prod(lead, dtype=np.int64)) * n * (k // wpb)
        w = synth_wire(rng, c, bpb, blocks, d_range=(0.004, 0.01),
                       nvfp4_scales=(0x08, 0x19))
        return mx.array(w.reshape(*lead, n, (k // wpb) * bpb))
    w = mx.array((rng.standard_normal((*lead, n, k)) * 0.1).astype(np.float32))
    wq, _ = kq.quantize(w, c)
    mx.eval(wq)
    return wq


def stores(w, k):
    full = kq.moe_store_pack(w, codec, k, mx.arange(E).astype(mx.uint32))
    ids = np.array([1, 3, 4, 8, 9, 11], np.uint32)
    half = kq.moe_store_pack(w, codec, k, mx.array(ids))
    slots = np.full((E,), ABSENT, np.uint32)
    slots[ids] = np.arange(len(ids), dtype=np.uint32)
    return (mx.arange(E).astype(mx.uint32), full), (mx.array(slots), half)


def bits(a):
    return np.array(a.astype(mx.float32)).tobytes()


for dt_name in ("bfloat16", "float16"):
    dt = getattr(mx, dt_name)
    # GLU: K inputs to N outputs. Down: KD inputs to ND outputs.
    K, N = 512, 128
    gw, uw = wire((E,), N, K, codec), wire((E,), N, K, codec)
    sg, su = wire((), N, K, shexp), wire((), N, K, shexp)
    KD, ND = 256, 256
    dw = wire((E,), ND, KD, codec)
    sd = wire((), ND, KD, shexp)
    (g_slots, g_full), (g_hslots, g_half) = stores(gw, K)
    (_, u_full), (_, u_half) = stores(uw, K)
    (d_slots, d_full), (d_hslots, d_half) = stores(dw, KD)
    skw = {} if shexp == codec else {"shexp_kquant_type": shexp}
    for T in (1, 2, 3, 8, 9, 16):
        x = mx.array((rng.standard_normal((T, K)) * XS).astype(np.float32))
        x = x.astype(dt)
        picks = [rng.choice(E, R, replace=False) for _ in range(T)]
        inds = mx.array(np.stack(picks).astype(np.uint32))
        if T >= 8:
            # a tile of more than two rows, and one of eight
            ii = np.array(inds)
            ii[:, 0] = 2
            ii[: min(T, 3), 1] = 9
            inds = mx.array(ii.astype(np.uint32))
        hd = mx.array((rng.standard_normal((T, R + 1, KD)) * 0.5).astype(np.float32))
        hd = hd.astype(dt)
        sc = mx.array(rng.uniform(0.05, 0.3, size=(T, R + 1)).astype(np.float32))
        h_ref = kq.moe_glu_gather_shexp_kq(
            x, gw, uw, sg, su, codec, inds, act="silu", **skw)
        y_ref = kq.gather_qmv_mix_kq(hd, dw, sd, codec, inds, sc, **skw)
        hr, yr = np.array(h_ref.astype(mx.float32)), np.array(y_ref.astype(mx.float32))
        for arm in ARMS:
            if arm not in os.environ.get("KQ_TEST_STORE_ARMS", arm):
                continue
            set_arm(arm)
            h_full = kq.moe_store_glu(
                x, gw, uw, sg, su, codec, inds, g_slots, g_full, u_full, **skw)
            h_half = kq.moe_store_glu(
                x, gw, uw, sg, su, codec, inds, g_hslots, g_half, u_half, **skw)
            hf = np.array(h_full.astype(mx.float32))
            key = f"{arm}/{dt_name}/{T}"
            res[key + "/glu_shape"] = list(h_full.shape) == list(h_ref.shape)
            res[key + "/glu_err"] = float(
                np.abs(hr - hf).max() / max(np.abs(hr).max(), 1e-9))
            res[key + "/glu_half_equal"] = bits(h_full) == bits(h_half)
            res[key + "/glu_equal"] = bits(h_ref) == bits(h_full)
            y_full = kq.moe_store_down(
                hd, dw, sd, codec, inds, sc, d_slots, d_full, **skw)
            y_half = kq.moe_store_down(
                hd, dw, sd, codec, inds, sc, d_hslots, d_half, **skw)
            yf = np.array(y_full.astype(mx.float32))
            res[key + "/down_err"] = float(
                np.abs(yr - yf).max() / max(np.abs(yr).max(), 1e-9))
            res[key + "/down_equal"] = bits(y_ref) == bits(y_full)
            res[key + "/down_half_equal"] = bits(y_full) == bits(y_half)
        set_arm("default")
if codec == "q5_0":
    # plane 0 [slot][N / 64][group][64][16] holds bytes 6 to 21 of each block
    w = np.array(dw).reshape(E, ND, KD // 32, 22)
    want = w[..., 6:22].reshape(E, ND // 64, 64, KD // 32, 16)
    want = np.ascontiguousarray(want.transpose(0, 1, 3, 2, 4))
    res["plane0"] = np.array(d_full[0]).tobytes() == want.tobytes()
    want = w[..., 0:2].reshape(E, ND // 64, 64, KD // 32, 2)
    want = np.ascontiguousarray(want.transpose(0, 1, 3, 2, 4))
    res["plane2"] = np.array(d_full[2]).tobytes() == want.tobytes()
    res["bytes"] = sum(p.nbytes for p in d_full) == dw.nbytes
json.dump(res, open(out_path, "w"))
"""


def _run(tmp_path, codec, shexp, tile_min_t):
    out = tmp_path / f"{codec}-{shexp}-{tile_min_t}.json"
    env = dict(
        os.environ,
        KQ_MOE_GROUP="1",
        KQ_MOE_TILE="1",
        KQ_MOE_TILE_MIN_T=str(tile_min_t),
        KQ_MOE_SHEXP_DENSE="0",
    )
    env.pop("KQ_MOE_MIX_FORM", None)
    r = subprocess.run(
        [sys.executable, "-c", _SNIPPET, codec, shexp, str(out), HERE],
        env=env,
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(out.read_text())


@pytest.mark.parametrize("codec", CODECS + GENERIC)
@pytest.mark.parametrize("shexp_same", (True, False))
def test_store_ops(tmp_path, codec, shexp_same):
    shexp = codec if shexp_same else "q8_0"
    res = _run(tmp_path, codec, shexp, 2)
    if not res["supported"]:
        pytest.skip("no tensor-op units on this GPU")
    for key, val in res.items():
        if (
            key.endswith("_equal")
            or key.endswith("_shape")
            or key in ("plane0", "plane2", "bytes")
        ):
            # The in-place tiled down runs from KQ_MOE_TILE_MIN_T rows (2
            # here), and only the forms with no K split keep its sums.
            if key.endswith("down_equal") and (
                key.split("/")[2] == "1" or key.split("/")[0] not in ("q1", "plan")
            ):
                continue
            # The pair form on the planes keeps the in-place GLU's sums.
            if key.endswith("glu_equal") and not (
                codec == "iq3_xxs"
                and key.split("/")[0] == "dd"
                and key.split("/")[2] in ("1", "2", "3")
            ):
                continue
            assert val, key
        elif key.endswith("_err"):
            assert val < 2e-2, (key, val)


# Operand checks, in process.

E_IN, R_IN, K_IN, N_IN = 12, 5, 512, 128


def _inproc(codec, shexp):
    import mlx.core as mx
    import numpy as np

    import mlx_kquant as kq

    if not kq.moe_store_supported(codec, shexp):
        pytest.skip("no tensor-op units on this GPU")
    rng = np.random.default_rng(11)

    def wire(shape, c=codec):
        w = mx.array((rng.standard_normal(shape) * 0.1).astype(np.float32))
        wq, _ = kq.quantize(w, c)
        mx.eval(wq)
        return wq

    T = 3
    x = mx.array((rng.standard_normal((T, K_IN)) * 0.5).astype(np.float32))
    x = x.astype(mx.bfloat16)
    picks = [rng.choice(E_IN, R_IN, replace=False) for _ in range(T)]
    inds = mx.array(np.stack(picks).astype(np.uint32))
    hd = mx.array((rng.standard_normal((T, R_IN + 1, N_IN)) * 0.5).astype(np.float32))
    hd = hd.astype(mx.bfloat16)
    sc = mx.array(rng.uniform(0.05, 0.3, size=(T, R_IN + 1)).astype(np.float32))
    mx.eval(x, inds, hd, sc)
    return mx, np, kq, wire, x, inds, hd, sc


def test_store_ops_read_a_slot_table_view_and_refuse_strided_stacks():
    codec = "q5_0"
    mx, np, kq, wire, x, inds, hd, sc = _inproc(codec, codec)
    gw, uw = wire((E_IN, N_IN, K_IN)), wire((E_IN, N_IN, K_IN))
    sg, su = wire((N_IN, K_IN)), wire((N_IN, K_IN))
    dw, sd = wire((E_IN, K_IN, N_IN)), wire((K_IN, N_IN))
    ids = mx.arange(E_IN).astype(mx.uint32)
    gp = kq.moe_store_pack(gw, codec, K_IN, ids)
    up = kq.moe_store_pack(uw, codec, K_IN, ids)
    dp = kq.moe_store_pack(dw, codec, N_IN, ids)
    mx.eval(gp, up, dp)
    # Column 1 of a wider table is the slot table; the other columns map
    # each expert to another expert's slot.
    e = np.arange(E_IN, dtype=np.uint32)
    table = mx.array(np.stack([(e + 5) % E_IN, e, (e + 7) % E_IN], axis=1))
    mx.eval(table)
    want_h = kq.moe_store_glu(x, gw, uw, sg, su, codec, inds, ids, gp, up)
    want_y = kq.moe_store_down(hd, dw, sd, codec, inds, sc, ids, dp)
    mx.eval(want_h, want_y)
    view = table[:, 1]
    mx.eval(view)
    for slots in (table[:, 1], view):
        h = kq.moe_store_glu(x, gw, uw, sg, su, codec, inds, slots, gp, up)
        y = kq.moe_store_down(hd, dw, sd, codec, inds, sc, slots, dp)
        assert mx.array_equal(h, want_h).item()
        assert mx.array_equal(y, want_y).item()
    # A stack that is a column block of a wider one: refused at the call
    # when it has memory, at eval when it does not.
    wide = wire((E_IN, 2 * N_IN, K_IN))
    with pytest.raises(ValueError, match="row-contiguous"):
        mx.eval(kq.moe_store_pack(wide[:, :N_IN], codec, K_IN, ids))
    block = wide[:, :N_IN]
    mx.eval(block)
    with pytest.raises(ValueError, match="row-contiguous"):
        kq.moe_store_pack(block, codec, K_IN, ids)
    with pytest.raises(ValueError, match="row-contiguous"):
        kq.moe_store_glu(x, block, uw, sg, su, codec, inds, ids, gp, up)


def test_store_glu_checks_the_shared_rows_against_their_codec():
    codec = "q5_0"
    mx, np, kq, wire, x, inds, hd, sc = _inproc(codec, "q8_0")
    gw, uw = wire((E_IN, N_IN, K_IN)), wire((E_IN, N_IN, K_IN))
    ids = mx.arange(E_IN).astype(mx.uint32)
    gp = kq.moe_store_pack(gw, codec, K_IN, ids)
    up = kq.moe_store_pack(uw, codec, K_IN, ids)
    # q6_k rows named q8_0: shorter than a q8_0 row of K weights.
    sg, su = wire((N_IN, K_IN), "q6_k"), wire((N_IN, K_IN), "q6_k")
    with pytest.raises(ValueError, match="K weights of 'q8_0'"):
        kq.moe_store_glu(
            x, gw, uw, sg, su, codec, inds, ids, gp, up, shexp_kquant_type="q8_0"
        )


@pytest.mark.parametrize(
    "lever,value",
    [("MIN_T", "99"), ("KQ_MOE_STORE_MAX_T", "1")],
    ids=["min rows", "max rows"],
)
def test_a_stub_stack_has_no_in_place_fallback(monkeypatch, lever, value):
    # A one-expert stub stands for a stack whose experts all have a slot;
    # a row count the width levers send to the in-place op would read
    # experts the stub does not hold.
    codec = "q5_0"
    mx, np, kq, wire, x, inds, hd, sc = _inproc(codec, codec)
    gw, uw = wire((E_IN, N_IN, K_IN)), wire((E_IN, N_IN, K_IN))
    dw = wire((E_IN, K_IN, N_IN))
    sg, su, sd = wire((N_IN, K_IN)), wire((N_IN, K_IN)), wire((K_IN, N_IN))
    ids = mx.arange(E_IN).astype(mx.uint32)
    gp = kq.moe_store_pack(gw, codec, K_IN, ids)
    up = kq.moe_store_pack(uw, codec, K_IN, ids)
    dp = kq.moe_store_pack(dw, codec, N_IN, ids)
    mx.eval(gp, up, dp)
    stub_g, stub_u, stub_d = gw[:1], uw[:1], dw[:1]
    mx.eval(stub_g, stub_u, stub_d)
    # Inside the width range the stub reads the store alone.
    h = kq.moe_store_glu(x, stub_g, stub_u, sg, su, codec, inds, ids, gp, up)
    want = kq.moe_store_glu(x, gw, uw, sg, su, codec, inds, ids, gp, up)
    assert mx.array_equal(h, want).item()
    names = (
        ["KQ_MOE_STORE_GLU_MIN_T", "KQ_MOE_STORE_DOWN_MIN_T"]
        if lever == "MIN_T"
        else [lever]
    )
    for name in names:
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="stub"):
        kq.moe_store_glu(x, stub_g, stub_u, sg, su, codec, inds, ids, gp, up)
    with pytest.raises(ValueError, match="stub"):
        kq.moe_store_down(hd, stub_d, sd, codec, inds, sc, ids, dp)
