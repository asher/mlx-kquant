"""Views of the operands that the MoE, routing and attention kernels read by
layout.

An op builder sees the strides of an operand only when the operand has
memory: an unevaluated view reports dense strides whatever it becomes. The
ops here decide the layout of an operand in their eval, where every input
has memory, so a view gives the same result whether it is evaluated before
the call or not:

- the three operands of route_shed are copied when they are not
  row-major.

An operand that is dense adds no node to the graph.

route_shed runs on the default device, so the suite covers the GPU and,
under KQUANT_FORCE_CPU, the CPU stream.
"""

from __future__ import annotations

import os
import re

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

metal_only = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="Metal-only kernel; no CPU path.",
)

E, N, K = 4, 64, 512
T, R = 3, 2
S = R + 1
_W = {}


def _dense(view):
    """A copy of a view in its own row-major buffer. numpy has no bfloat16
    and float32 holds it exactly."""
    if view.dtype == mx.bfloat16:
        wide = np.ascontiguousarray(np.array(view.astype(mx.float32)))
        return mx.array(wide).astype(mx.bfloat16)
    return mx.array(np.ascontiguousarray(np.array(view)))


def _nodes(*outputs):
    """Names of the primitives in the graph of the outputs."""
    import tempfile

    with tempfile.NamedTemporaryFile("r", suffix=".dot") as f:
        mx.export_to_dot(f.name, *outputs)
        text = f.read()
    return sorted(re.findall(r'label ="([A-Za-z0-9_]+)"', text))


def _same(got, want):
    got = got if isinstance(got, (list, tuple)) else [got]
    want = want if isinstance(want, (list, tuple)) else [want]
    mx.eval(*got, *want)
    return all(mx.array_equal(g, w).item() for g, w in zip(got, want, strict=True))


def _check_views(views, call, state):
    """call(view) equals call(dense copy of the view) for every view."""
    for name, make in views.items():
        want = call(_dense(make()))
        view = make()
        if state == "evaluated":
            mx.eval(view)
        assert _same(call(view), want), (name, state)


def _strided(base):
    """Views of one [rows, cols] array that are not row-major, as builders
    of a fresh view. base is [2 * rows, 2 * cols] random values."""
    rows, cols = base.shape[0] // 2, base.shape[1] // 2
    turned = _dense(mx.swapaxes(base[:rows, :cols], 0, 1))
    mx.eval(turned)
    return {
        "even_cols": lambda: base[:rows, 0::2],
        "odd_cols": lambda: base[:rows, 1::2],
        "even_rows": lambda: base[0::2, :cols],
        "reversed_rows": lambda: base[::-1][:rows, :cols],
        "col_band": lambda: base[:rows, cols // 2 : cols // 2 + cols],
        "transposed": lambda: mx.swapaxes(turned, 0, 1),
    }


# ----------------------------- route_shed -----------------------------


def _route_case(seed=0, n_experts=256):
    rng = np.random.default_rng(seed)
    idx = mx.array(rng.integers(0, n_experts, size=(2 * T, 2 * R)).astype(np.uint32))
    sc = mx.array(rng.uniform(0.05, 0.9, size=(2 * T, 2 * R)).astype(np.float32))
    slots = np.where(
        rng.random(2 * n_experts) < 0.7, rng.integers(0, 64, 2 * n_experts), -1
    )
    slot = mx.array(slots.astype(np.int32))
    mx.eval(idx, sc, slot)
    return idx, sc, slot


@pytest.mark.parametrize("state", ["lazy", "evaluated"])
def test_route_shed_operand_views(state):
    idx2, sc2, slot2 = _route_case()
    n_experts = slot2.shape[0] // 2
    idx = _dense(idx2[:T, :R])
    sc = _dense(sc2[:T, :R])
    slot = _dense(slot2[:n_experts])
    _check_views(_strided(idx2), lambda v: kq.route_shed(v, sc, slot), state)
    _check_views(_strided(sc2), lambda v: kq.route_shed(idx, v, slot), state)
    _check_views(
        {
            "every_other": lambda: slot2[0::2],
            "reversed": lambda: slot2[::-1][:n_experts],
            "band": lambda: slot2[n_experts // 2 : n_experts // 2 + n_experts],
        },
        lambda v: kq.route_shed(idx, sc, v),
        state,
    )


def test_route_shed_adds_no_node():
    """The call a streamed decode makes: unevaluated router outputs. The
    graph holds the op and the ops that feed it, and no layout node."""
    idx2, sc2, slot2 = _route_case(1)
    idx = _dense(idx2[:T, :R])
    sc = _dense(sc2[:T, :R])
    feed = set(_nodes(idx + 0, sc * 1.0))
    assert "Add" in feed and "Multiply" in feed
    outs = kq.route_shed(idx + 0, sc * 1.0, slot2)
    assert set(_nodes(*outs)) == feed | {"KQuantRouteShed"}
    outs = kq.route_shed(idx2[:T, 0::2], sc, slot2[0::2])
    assert set(_nodes(*outs)) == {"KQuantRouteShed", "Slice"}
