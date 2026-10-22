"""Shared rows in pieces in the score-mixed down with a shared expert.

gather_qmv_mix_kq can run each shared row as SP pieces of whole blocks.
The slot-parallel kernel gives each piece its own simdgroup pair, and the
loop and dedupe kernels sum the same pieces in the same order. By default
a one-row call runs two pieces and a wider call one. KQ_MOE_SX_SPLIT=<n>
forces n pieces at any row count (read per call). Each case checks, for
each forced count, every kernel form against a float reference, the forms
against each other bit for bit, the default pick against the count the
rule gives at its row count, and the slot-parallel launch: one simdgroup
pair per routed slot and per piece, wherever the pipeline takes that many
threads, and the loop kernel elsewhere. A forced count past the shared
row's block count, or past 16 simdgroup pairs, runs as the bound. A shared
expert in the routed codec keeps one piece. Under a lower thread limit
(KQ_MOE_SP_THREAD_CAP) that fits one piece but not two, the default pick
at one row runs its two pieces on the loop kernel.

Pieces reorder a sum, which a 16-bit output seldom shows, so one case
makes the order visible: a shared row whose second half repeats its first
half, against an activation whose second half is the first half negated,
with the routed activations zero. Two pieces sum to exactly zero there, and
one piece leaves the rounding of the first half's partial sums, so the
default pick shows its count at one row and at two.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np
import pytest
from kqref import GT, quants

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="fused MoE gathers are Metal-only kernels; no CPU path.",
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
GTYPE = {
    "iq2_s": GT.IQ2_S,
    "iq3_xxs": GT.IQ3_XXS,
    "q4_1": GT.Q4_1,
    "q4_k": GT.Q4_K,
    "q5_k": GT.Q5_K,
    "q6_k": GT.Q6_K,
    "q8_0": GT.Q8_0,
}
E, H, R = 8, 512, 6  # experts, output rows, routed slots
ROWS = (1, 2, 3)
REL_BOUND = 2e-3
# (name, routed down codec, shared down codec, shared slots, K, forced
# counts, the count each forced count runs as)
CASES = [
    ("iq2_s_q8_0", "iq2_s", "q8_0", 1, 768, (2, 3, 5, 12), (2, 3, 5, 10)),
    ("iq3_xxs_q5_k", "iq3_xxs", "q5_k", 1, 1024, (2, 3, 4, 6), (2, 3, 4, 4)),
    ("q4_1_q6_k_k2", "q4_1", "q6_k", 2, 512, (2, 3), (2, 2)),
    ("q4_k_q4_k", "q4_k", "q4_k", 1, 512, (4,), (1,)),
]
# The count the default pick runs at one row: two pieces, or one for a
# shared expert in the routed codec; one at every wider call.
ONE_ROW = {"iq2_s_q8_0": 2, "iq3_xxs_q5_k": 2, "q4_1_q6_k_k2": 2, "q4_k_q4_k": 1}
# (name, routed down codec, shared down codec, K)
CANCEL = [
    ("iq2_s_q8_0", "iq2_s", "q8_0", 768),
    ("iq3_xxs_q5_k", "iq3_xxs", "q5_k", 1024),
]
FORMS = ("loop", "sp", "dd")


def _quantize(rng, shape, codec):
    import mlx.core as mx

    import mlx_kquant as kq

    w = mx.array((rng.standard_normal(shape) * 0.1).astype(np.float32))
    im = mx.ones((shape[-1],), dtype=mx.float32) if codec.startswith("iq") else None
    wq, _ = kq.quantize(w, codec, im)
    mx.eval(wq)
    return np.array(wq)


@pytest.fixture(scope="module")
def wires(tmp_path_factory):
    path = os.path.join(str(tmp_path_factory.mktemp("sx_split")), "wires.npz")
    rng = np.random.default_rng(23)
    out = {}
    for name, cd, sd, k, n, _f, _b in CASES:
        out[f"{name}/dw"] = _quantize(rng, (E, H, n), cd)
        out[f"{name}/sd"] = _quantize(rng, (H, k * n), sd)
        for t in ROWS:
            out[f"{name}/h{t}"] = (rng.standard_normal((t, R + k, n)) * 0.1).astype(
                np.float16
            )
            ind = rng.integers(0, E, size=(t, R)).astype(np.uint32)
            ind[:, 0] = 3  # rows share an expert, which the dedupe form pairs
            out[f"{name}/i{t}"] = ind
            out[f"{name}/s{t}"] = rng.uniform(0.05, 0.9, size=(t, R + k)).astype(
                np.float32
            )
    for name, cd, sd, n in CANCEL:
        half = _quantize(rng, (H, n // 2), sd)
        out[f"cancel/{name}/dw"] = _quantize(rng, (E, H, n), cd)
        out[f"cancel/{name}/sd"] = np.concatenate([half, half], axis=1)
        for t in ROWS[:2]:
            h = np.zeros((t, R + 1, n), dtype=np.float16)
            # partial sums near 100, whose rounding a 16-bit output keeps
            x = (rng.standard_normal((t, n // 2)) * 64).astype(np.float16)
            h[:, R] = np.concatenate([x, -x], axis=1)
            out[f"cancel/{name}/h{t}"] = h
            out[f"cancel/{name}/i{t}"] = rng.integers(0, E, size=(t, R)).astype(
                np.uint32
            )
            out[f"cancel/{name}/s{t}"] = rng.uniform(0.05, 0.9, size=(t, R + 1)).astype(
                np.float32
            )
    np.savez(path, **out)
    return path


# The thread limit of the capped call: 64 threads for each of the 6 routed
# slots and one shared piece, so two pieces (512 threads) do not fit.
CAP_THREADS = 64 * (R + 1)


def _calls():
    """(key, prefix, down codec, shared codec, row count, form, count, env)
    for every call the child makes; form "" with count None is the default,
    and env holds the call's other levers."""
    out = []
    for name, cd, sd, _k, _n, forced, _b in CASES:
        for t in ROWS:
            out.append((f"{name}/default/{t}", name, cd, sd, t, "", None, {}))
            for f in FORMS:
                for sp in (1, *forced):
                    out.append((f"{name}/{f}{sp}/{t}", name, cd, sd, t, f, sp, {}))
    cap = {"KQ_MOE_SP_THREAD_CAP": str(CAP_THREADS)}
    out.append(("iq2_s_q8_0/capped/1", "iq2_s_q8_0", "iq2_s", "q8_0", 1, "", None, cap))
    for name, cd, sd, _n in CANCEL:
        for t in ROWS[:2]:
            out.append(
                (
                    f"cancel/{name}/default/{t}",
                    f"cancel/{name}",
                    cd,
                    sd,
                    t,
                    "",
                    None,
                    {},
                )
            )
            for f in FORMS:
                for sp in (1, 2):
                    out.append(
                        (
                            f"cancel/{name}/{f}{sp}/{t}",
                            f"cancel/{name}",
                            cd,
                            sd,
                            t,
                            f,
                            sp,
                            {},
                        )
                    )
    return out


def _rule(name, t):
    return ONE_ROW[name] if t == 1 else 1


def child(wire_path, out_path):
    """Run every call of _calls, each evaluated while its levers are set.
    KQ_MOE_MIX_FORM is read live once it exists, so it stays set (empty
    for the default pick). A marker line on stderr precedes each call."""
    import mlx.core as mx

    import mlx_kquant as kq

    z = np.load(wire_path)
    res = {}
    for key, pre, cd, sd, t, form, sp, env in _calls():
        os.environ["KQ_MOE_MIX_FORM"] = form
        os.environ.pop("KQ_MOE_SX_SPLIT", None)
        os.environ.pop("KQ_MOE_SP_THREAD_CAP", None)
        if sp is not None:
            os.environ["KQ_MOE_SX_SPLIT"] = str(sp)
        os.environ.update(env)
        sys.stderr.write(f"== {key}\n")
        sys.stderr.flush()
        y = kq.gather_qmv_mix_kq(
            mx.array(z[f"{pre}/h{t}"]),
            mx.array(z[f"{pre}/dw"]),
            mx.array(z[f"{pre}/sd"]),
            cd,
            mx.array(z[f"{pre}/i{t}"]),
            mx.array(z[f"{pre}/s{t}"]),
            shexp_kquant_type=sd,
        )
        mx.eval(y)
        res[key] = np.array(y.astype(mx.float32))
    np.savez(out_path, **res)


_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
import test_moe_sx_split as m
m.child(sys.argv[2], sys.argv[3])
"""


def _launches(log):
    """Per call key, the slot-parallel kernel it launched and the
    threadgroup height; and the thread limit the log gives for each kernel
    (KQ_MOE_NX_LOG names a kernel once, at its first launch)."""
    out, caps, key = {}, {}, None
    for ln in log.splitlines():
        m = re.match(r"\[kq_moe\] (\S+) \(pipeline max (\d+)\)", ln)
        if ln.startswith("== "):
            key = ln[3:]
        elif m:
            caps[m.group(1)] = int(m.group(2))
        elif key is not None and "_gather_qmv_mix_sp_" in ln and "tg=" in ln:
            height = int(re.search(r"tg=\d+,(\d+),", ln).group(1))
            out[key] = (ln.split()[2], height)
    return out, caps


@pytest.fixture(scope="module", params=["c32", "c32off"])
def runs(request, wires, tmp_path_factory):
    env = {"KQ_MOE_NX_LOG": "1", "KQ_QMM_LOG": "1", "KQ_MOE_MIX_FORM": ""}
    if request.param == "c32off":
        env["KQ_MOE_C32"] = "0"
    out = os.path.join(str(tmp_path_factory.mktemp(request.param)), "out.npz")
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, TESTS_DIR, wires, out],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    assert r.returncode == 0, f"child exited {r.returncode}\n{r.stdout}\n{r.stderr}"
    return np.load(out), r.stderr


def _deq(wire, codec):
    w = np.ascontiguousarray(wire)
    lead = w.shape[:-1]
    flat = w.reshape(-1, w.shape[-1])
    return quants.dequantize(flat, GTYPE[codec]).astype(np.float32).reshape(*lead, -1)


def _rel(got, ref):
    return float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-6))


def test_sx_split_against_reference(wires, runs):
    """Every form at every count against the float reference."""
    outs, _ = runs
    z = np.load(wires)
    for name, cd, sd, k, n, forced, _b in CASES:
        dw, sdw = _deq(z[f"{name}/dw"], cd), _deq(z[f"{name}/sd"], sd)
        for t in ROWS:
            h = z[f"{name}/h{t}"].astype(np.float32)
            ind, sc = z[f"{name}/i{t}"], z[f"{name}/s{t}"]
            ref = np.stack(
                [
                    sum(sc[r, s] * (h[r, s] @ dw[ind[r, s]].T) for s in range(R))
                    for r in range(t)
                ]
            ) + sum(
                sc[:, R + j, None] * (h[:, R + j] @ sdw[:, j * n : (j + 1) * n].T)
                for j in range(k)
            )
            for f in FORMS:
                for sp in (1, *forced):
                    got = outs[f"{name}/{f}{sp}/{t}"]
                    assert np.isfinite(got).all(), (name, f, sp, t)
                    rel = _rel(got, ref)
                    assert rel < REL_BOUND, (name, f, sp, t, rel)


def test_sx_split_forms_bit_identical(runs):
    """At each count the slot-parallel and dedupe kernels give the loop
    kernel's bytes, and the default pick gives the bytes of the count the
    rule gives at its row count."""
    outs, log = runs
    assert "_gather_qmv_mix_sp_" in log and "_gather_qmv_mix_dd" in log, log
    for name, _cd, _sd, _k, _n, forced, _b in CASES:
        for t in ROWS:
            a = outs[f"{name}/loop{_rule(name, t)}/{t}"]
            assert np.array_equal(a, outs[f"{name}/default/{t}"]), (name, t)
            for sp in (1, *forced):
                a = outs[f"{name}/loop{sp}/{t}"]
                for f in ("sp", "dd"):
                    b = outs[f"{name}/{f}{sp}/{t}"]
                    assert np.array_equal(a, b), (name, f, sp, t, np.abs(a - b).max())


def test_sx_split_slot_parallel_launch(runs):
    """The slot-parallel launch holds one simdgroup pair per routed slot and
    per piece of each shared slot, at the bound past it, and one piece for
    a shared expert in the routed codec. Every count whose pairs fit the
    pipeline's thread limit runs the slot-parallel kernel, so a clamped
    count shows its bound, and a count past the limit runs the loop kernel.
    The limit is the one the log gives for the kernel the one-piece call
    launched; when that call ran the loop kernel, no count fits."""
    _, log = runs
    tg, caps = _launches(log)
    for name, _cd, _sd, k, _n, forced, bound in CASES:
        for t in ROWS:
            first = tg.get(f"{name}/sp1/{t}")
            cap = caps[first[0]] if first else 64 * (R + k) - 1
            arms = (
                ("sp1", 1),
                *((f"sp{f}", b) for f, b in zip(forced, bound, strict=True)),
            )
            for arm, b in (*arms, ("default", _rule(name, t))):
                pairs = R + k * b
                key = f"{name}/{arm}/{t}"
                if 64 * pairs <= cap:
                    assert key in tg, (key, pairs, cap)
                    assert tg[key][1] == 2 * pairs, (key, tg[key], pairs)
                else:
                    assert key not in tg, (key, pairs, cap)


def test_sx_split_thread_cap_takes_loop(runs):
    """Under a thread limit that fits one shared piece but not two, the
    default call at one row launches no slot-parallel kernel, and the loop
    kernel gives the bytes of two forced pieces."""
    outs, log = runs
    tg, _ = _launches(log)
    assert "iq2_s_q8_0/capped/1" not in tg, tg.get("iq2_s_q8_0/capped/1")
    assert "iq2_s_q8_0/default/1" in tg, sorted(tg)
    a, b = outs["iq2_s_q8_0/capped/1"], outs["iq2_s_q8_0/loop2/1"]
    assert np.array_equal(a, b), np.abs(a - b).max()


def test_sx_split_order_visible(runs):
    """Two pieces of a shared row that cancels half against half sum to
    exactly zero in every form; one piece leaves a rounding residue."""
    outs, _ = runs
    for name, *_ in CANCEL:
        for t in ROWS[:2]:
            for f in FORMS:
                one = outs[f"cancel/{name}/{f}1/{t}"]
                two = outs[f"cancel/{name}/{f}2/{t}"]
                assert np.all(two == 0), (name, f, t, np.abs(two).max())
                assert np.any(one != 0), (name, f, t)


def test_sx_split_default_rule(runs):
    """The default pick runs two pieces at one row and one at two rows, so
    the cancelling shared row sums to zero at one row and leaves a residue
    at two."""
    outs, _ = runs
    for name, *_ in CANCEL:
        one_row = outs[f"cancel/{name}/default/1"]
        two_rows = outs[f"cancel/{name}/default/2"]
        assert np.all(one_row == 0), (name, np.abs(one_row).max())
        assert np.any(two_rows != 0), name
