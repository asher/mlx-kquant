"""The DiT benchmark harness: the median CI at any sample count, and
rebuilding reports from a raw log. Pure Python, no GPU."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from fractions import Fraction

import pytest

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks")
sys.path.insert(0, BENCH_DIR)

import bench_stats as bs  # noqa: E402


def _tail(n, k):
    """P(X < k) for X ~ Binomial(n, 1/2), exact."""
    return Fraction(sum(math.comb(n, i) for i in range(k)), 2**n)


@pytest.mark.parametrize("n", [3, 24, 1100, 5000])
def test_median_ci_order_statistics(n):
    s = list(range(n))
    ci = bs.median_ci(s)
    if n == 3:
        # Even the widest interval [s(1), s(n)] covers under 95 percent.
        assert ci is None
        return
    lo, hi = ci
    k = lo + 1
    assert hi == n - k
    # Largest k with P(X < k) <= 0.025.
    assert _tail(n, k) <= Fraction(1, 40) < _tail(n, k + 1)
    if n == 24:
        # Standard table value: the 95 percent CI of the median of 24 is
        # [x(7), x(18)].
        assert (lo, hi) == (6, 17)
    else:
        assert abs(k - (n - 1.96 * math.sqrt(n)) / 2) < 1.5


def test_summarize_ci_finite_for_thousands_of_pairs():
    s = [1.0 + 0.001 * math.sin(i) for i in range(5000)]
    out = bs.summarize(s)
    assert out["n"] == 5000
    assert out["ci95"] is not None
    assert 0 < out["ci95_half"] < 0.001


_WRITER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import bench_dit_shapes as b
    raw, json_out, md_out = sys.argv[2:5]
    g = {"family": "zimage", "op": "qkv", "K": 256, "N": 704, "L": 64, "B": 1,
         "M": 64, "L_model": 4096, "vector": False, "heavy": False,
         "codecs": ["q4_k"]}
    args = {"rounds": 3, "slots": 8, "inner": 3, "max_inner": 64,
            "min_sample_ms": 10.0, "shrink": 16, "arms": ["kq", "kq_aa"]}
    machine = {"device": "gpu", "architecture": "applegpu_g17s", "arch_gen": 17,
               "arch_letter": "s", "gpu_cores": 40, "nax": True,
               "max_threadgroup_memory": 32768, "os": "26", "mlx": "0.32.2",
               "mlx_kquant": "0.5.0", "metallib_sha256": None, "tf32": True}
    log = b.RawLog(raw)
    log.write({"kind": "meta", "machine": machine, "args": args,
               "heavy_codecs": [], "groups": [g], "mlx_kquant_file": "kq"})
    log.write({"kind": "warmup", "dtype": "bfloat16", "inner": [3],
               "kernels": [[0, "kq/q4_k", "k"], [0, "kq_aa/q4_k", "k"]],
               "errors": []})
    visit = 0
    for r in range(2):
        for s in range(8):
            ms = 1.0 + 0.01 * s
            rows = [[0, "kq/q4_k", visit, r, s, 1.0, ms],
                    [0, "kq_aa/q4_k", visit, r, s, 1.0, ms * 1.01]]
            log.write({"kind": "slot", "dtype": "bfloat16", "round": r,
                       "slot": s, "samples": rows})
            visit += 1
    log.close()
    with open(raw, "a") as f:
        f.write('{"kind": "slot", "dtype": "bfl')
    sys.exit(b.main(["--reaggregate", raw, "--json-out", json_out,
                     "--md-out", md_out]))
    """
)


def test_reaggregate_rebuilds_a_stopped_run(tmp_path):
    raw = tmp_path / "run.json.raw.jsonl"
    json_out = tmp_path / "run.json"
    md_out = tmp_path / "run.md"
    script = tmp_path / "writer.py"
    script.write_text(_WRITER)
    # A child process, since dit_common sets KQ_TUNE_LIVE on import.
    subprocess.run(
        [sys.executable, str(script), BENCH_DIR, str(raw), str(json_out), str(md_out)],
        check=True,
        timeout=120,
    )
    result = json.loads(json_out.read_text())
    assert result["complete"] is False
    (p,) = result["passes"]
    assert (p["slots_done"], p["slots_planned"]) == (16, 24)
    (ag,) = [a for a in p["aggregate"] if a["ratio"] == "aa/kq"]
    assert ag["op"] == "zimage.qkv"
    # Round 0 is left out of the pooled figure.
    assert ag["pooled"]["n"] == 8
    assert ag["pooled"]["median"] == pytest.approx(1.01)
    md = md_out.read_text()
    assert "Partial run" in md
    assert "16 of 24 slots ran." in md
