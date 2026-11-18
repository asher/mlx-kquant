"""Row-per-thread gathered mat-vec at one activation row.

kq.gather_qmm with transpose at M 1 gives each output row its own thread
when the codec is q8_0 and K is a multiple of 32 under 256: the per-head
projections of an absorbed MLA, 128 wide. KQ_GATHER_ROWS=0 keeps
gather_qmv. The two differ only in float32 summation order: by one unit in
the last place, or near zero where the sum cancels.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or mx.default_device() != mx.gpu,
    reason="Metal kernel routing",
)

LEVER = "KQ_GATHER_ROWS"
MANTISSA = {mx.float32: 7, mx.float16: 10, mx.bfloat16: 7}
# Each case: N, K, x shape, lhs (None takes the default), rhs, experts. The
# first is an absorbed MLA's query projection, a row of x per head. The
# second has one shared x, repeated experts and an N that leaves the last
# threadgroup part empty. The third has two batch dimensions and a permuted
# lhs with a repeat. The fourth is one block wide and a few rows tall.
CASES = {
    "per_head": (512, 128, (1, 8, 1, 128), [list(range(8))], [list(range(8))], 8),
    "shared_x": (70, 96, (1, 1, 1, 96), None, [[3, 0, 3, 5, 1, 2]], 6),
    "batch_2d": (
        64,
        224,
        (2, 3, 1, 224),
        [[5, 0, 0], [2, 4, 1]],
        [[1, 2, 0], [2, 2, 1]],
        3,
    ),
    "one_block": (5, 32, (1, 2, 1, 32), [[1, 0]], [[0, 1]], 2),
}


def _experts(e, n, k):
    rng = np.random.default_rng(n + k)
    ws = []
    for _ in range(e):
        wf = (rng.standard_normal((n, k)) * 0.1).astype(np.float32)
        wq, _ = kq.quantize(mx.array(wf), "q8_0")
        ws.append(wq)
    w = mx.stack(ws)
    mx.eval(w)
    return w


def _gather(x, w, lhs, rhs, lever=None):
    if lever is None:
        os.environ.pop(LEVER, None)
    else:
        os.environ[LEVER] = lever
    y = kq.gather_qmm(
        x,
        w,
        mx.zeros((1,), dtype=mx.uint8),
        "q8_0",
        lhs_indices=lhs,
        rhs_indices=rhs,
        transpose=True,
    )
    mx.eval(y)
    return y


@pytest.fixture(autouse=True)
def _lever(monkeypatch):
    monkeypatch.delenv(LEVER, raising=False)
    yield
    os.environ.pop(LEVER, None)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("case", list(CASES))
def test_matches_gather_qmv_and_the_decoded_product(case, dtype):
    n, k, xs, lhs_l, rhs_l, e = CASES[case]
    w = _experts(e, n, k)
    rng = np.random.default_rng(3)
    x = mx.array((rng.standard_normal(xs) * 0.5).astype(np.float32)).astype(dtype)
    lhs = None if lhs_l is None else mx.array(np.array(lhs_l, np.uint32))
    rhs = mx.array(np.array(rhs_l, np.uint32))
    got = _gather(x, w, lhs, rhs)
    base = _gather(x, w, lhs, rhs, lever="0")
    assert got.dtype == base.dtype and got.shape == base.shape
    a = np.array(got.astype(mx.float32))
    b = np.array(base.astype(mx.float32))
    ulp = np.maximum(np.abs(a), np.abs(b)) * 2.0 ** -MANTISSA[dtype]
    assert np.all(np.abs(a - b) <= ulp + 1e-5 * float(np.abs(b).max()))

    # The product on the decoded weights, in float64. The op reads a
    # float32 x as bfloat16.
    xr = x.astype(mx.bfloat16) if dtype == mx.float32 else x
    rows = np.array(xr.astype(mx.float32)).astype(np.float64).reshape(-1, k)
    rhs_f = np.array(rhs_l, np.int64).ravel()
    if lhs_l is None:
        lhs_f = np.broadcast_to(
            np.arange(rows.shape[0]).reshape(xs[:-2]), np.array(rhs_l).shape
        ).ravel()
    else:
        lhs_f = np.array(lhs_l, np.int64).ravel()
    wd = np.array(
        kq.dequantize(w, mx.zeros((1,), dtype=mx.uint8), "q8_0", mx.float32)
    ).astype(np.float64)
    ref = np.stack(
        [wd[int(j)] @ rows[int(i)] for i, j in zip(lhs_f, rhs_f, strict=True)]
    ).reshape(a.shape)
    tol = np.abs(ref) * 2.0 ** -MANTISSA[dtype] + 1e-4 * float(np.abs(ref).max())
    assert np.all(np.abs(a - ref) <= tol)


_CHILD = r"""
import json, os, sys
import mlx.core as mx
import mlx_kquant as kq
for i, (codec, k, m, lever) in enumerate(json.loads(sys.argv[1])):
    os.environ.pop("KQ_GATHER_ROWS", None)
    if lever is not None:
        os.environ["KQ_GATHER_ROWS"] = lever
    wf = mx.random.normal((2, 16, k), key=mx.random.key(k)) * 0.1
    w = mx.stack([kq.quantize(wf[e], codec)[0] for e in range(2)])
    x = mx.random.normal((1, 2, m, k), key=mx.random.key(1)).astype(mx.bfloat16)
    idx = mx.array([[0, 1]], dtype=mx.uint32)
    mx.eval(w, x)
    print(f"[case] {i}", file=sys.stderr, flush=True)
    y = kq.gather_qmm(x, w, mx.zeros((1,), dtype=mx.uint8), codec,
                      lhs_indices=idx, rhs_indices=idx, transpose=True)
    mx.eval(y)
"""
_LOG = re.compile(r"^\[kq qmm\] (\S+) ")
# codec, K, activation rows, lever, whether the row kernel runs.
ROUTES = [
    ("q8_0", 128, 1, None, True),
    ("q8_0", 32, 1, None, True),
    ("q8_0", 224, 1, None, True),
    ("q8_0", 128, 1, "0", False),
    # 256 divides K: the fast tile.
    ("q8_0", 256, 1, None, False),
    ("q8_0", 512, 1, None, False),
    # Past 256 with a tail: the general tile has most lanes at work.
    ("q8_0", 288, 1, None, False),
    ("q8_0", 128, 2, None, False),
    ("q4_0", 128, 1, None, False),
]


# The outputs of the two kernels agree to the last place, so the test reads
# the kernel name from KQ_QMM_LOG in a child process.
def test_routing():
    env = {k: v for k, v in os.environ.items() if not k.startswith("KQ_")}
    env["KQ_QMM_LOG"] = "1"
    spec = json.dumps([[c, k, m, lever] for c, k, m, lever, _ in ROUTES])
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, spec],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    got = {}
    cur = None
    for line in r.stderr.splitlines():
        if line.startswith("[case] "):
            cur = int(line.split()[1])
            continue
        m = _LOG.match(line)
        if m and cur is not None:
            got.setdefault(cur, []).append(m.group(1))
    for i, (codec, k, rows, lever, want) in enumerate(ROUTES):
        names = got.get(i, [])
        assert names, f"no dispatch logged for {ROUTES[i]}"
        took = any("gather_qmv_rows" in n for n in names)
        assert took == want, f"{codec} K={k} M={rows} lever={lever}: {names}"
