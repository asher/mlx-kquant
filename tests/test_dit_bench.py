"""The DiT benchmark harness: the median CI at any sample count, rebuilding
reports from a raw log, and, at tiny shapes in a child process, the cache
limit, the per-visit memory records and resuming a stopped run."""

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
    # A log without a sample mode is from the independent-call bench.
    assert result["sample_mode"] == "independent"
    md = md_out.read_text()
    assert "do not compare with chained runs" in md
    assert "Partial run" in md
    assert "16 of 24 slots ran." in md


_ORDER_WRITER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import bench_dit_shapes as b
    import dit_common as dc
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
               "heavy_codecs": [], "groups": [g], "mlx_kquant_file": "kq",
               "sample_mode": "chain"})
    log.write({"kind": "warmup", "dtype": "bfloat16", "inner": [3],
               "kernels": [[0, "kq/q4_k", "k"], [0, "kq_aa/q4_k", "k"]],
               "errors": []})
    # kq_aa takes 1.02x kq, and the second call of a visit 1.1x the first.
    visit, t = 0, 0.0
    for r in range(3):
        for s in range(8):
            arms = [("kq/q4_k", 1.0), ("kq_aa/q4_k", 1.02)]
            if dc.THUE_MORSE[s]:
                arms.reverse()
            rows = []
            for i, (key, ms) in enumerate(arms):
                t += 1.0
                rows.append([0, key, visit, r, s, t, ms * (1.1 if i else 1.0)])
            log.write({"kind": "slot", "dtype": "bfloat16", "round": r,
                       "slot": s, "samples": rows})
            visit += 1
    log.write({"kind": "end", "wall_s": 1.0})
    log.close()
    sys.exit(b.main(["--reaggregate", raw, "--json-out", json_out,
                     "--md-out", md_out]))
    """
)


def test_ratio_balances_call_order(tmp_path):
    raw = tmp_path / "run.json.raw.jsonl"
    json_out = tmp_path / "run.json"
    script = tmp_path / "writer.py"
    script.write_text(_ORDER_WRITER)
    subprocess.run(
        [
            sys.executable,
            str(script),
            BENCH_DIR,
            str(raw),
            str(json_out),
            str(tmp_path / "run.md"),
        ],
        check=True,
        timeout=120,
    )
    result = json.loads(json_out.read_text())
    (p,) = result["passes"]
    (g,) = p["groups"]
    (c,) = g["cells"]
    rt = c["ratios"]["aa/kq"]
    # The pairs of the two orders, 1.122 and 0.927, meet at 1.02.
    assert rt["order"] == pytest.approx(1.1**2)
    assert sorted({round(v, 9) for _r, v, _f in rt["pairs_raw"]}) == [
        round(1.02 / 1.1, 9),
        round(1.02 * 1.1, 9),
    ]
    st = rt["settled"]
    assert st["median"] == pytest.approx(1.02)
    assert st["p10"] == pytest.approx(st["p90"])
    (ag,) = [a for a in p["aggregate"] if a["ratio"] == "aa/kq"]
    assert ag["pooled"]["median"] == pytest.approx(1.02)
    assert ag["cell_order"]["median"] == pytest.approx(1.1**2)


# Two tiny shapes, two rounds of two slots, one call per sample.
_TINY = [
    "--families",
    "zimage",
    "--ops",
    "qkv,out",
    "--rows",
    "64",
    "--batches",
    "1",
    "--codecs",
    "q4_k",
    "--heavy-codecs",
    "none",
    "--arms",
    "kq,kq_aa",
    "--dtypes",
    "bfloat16",
    "--rounds",
    "2",
    "--slots",
    "2",
    "--inner",
    "1",
    "--max-inner",
    "1",
    "--min-sample-ms",
    "0.001",
    "--shrink",
    "16",
]


def _bench(tmp_path, *extra):
    json_out = tmp_path / "run.json"
    out = subprocess.run(
        [
            sys.executable,
            os.path.join(BENCH_DIR, "bench_dit_shapes.py"),
            *_TINY,
            "--json-out",
            str(json_out),
            "--md-out",
            str(tmp_path / "run.md"),
            *extra,
        ],
        check=True,
        timeout=300,
        capture_output=True,
        text=True,
    )
    raw = tmp_path / "run.json.raw.jsonl"
    recs = [json.loads(line) for line in raw.read_text().splitlines()]
    return json.loads(json_out.read_text()), raw, recs, out.stdout


def test_cache_limit_and_memory_per_visit(tmp_path):
    limit = 1 << 20
    result, _, recs, out = _bench(
        tmp_path,
        "--cache-limit-gib",
        str(limit / 2**30),
        "--mem-trace",
        "--visit-warmup-calls",
        "1,full",
    )
    assert result["complete"]
    assert result["sample_mode"] == recs[0]["sample_mode"] == "chain"
    assert "dependent call chains" in (tmp_path / "run.md").read_text()
    assert recs[0]["args"]["cache_limit_gib"] == limit / 2**30
    slots = [r for r in recs if r["kind"] == "slot"]
    (warm,) = [r for r in recs if r["kind"] == "warmup"]
    assert len(slots) == 4
    assert [r["visit_warmup_calls"] for r in slots] == ["1", "full"] * 2
    for rec in slots:
        # One glue sample per visit, beside each arm's.
        keys = [row[1] for row in rec["samples"]]
        assert keys.count("glue") == 2
        assert len(keys) == 6
    (p,) = result["passes"]
    assert all("glue" in g["shared"] for g in p["groups"])
    for rec in [warm, *slots]:
        assert sorted(m[0] for m in rec["mem"]) == [0, 1]
        for _, active, cache, fp in rec["mem"]:
            assert active > 0
            assert cache <= limit
            assert (fp is None) == (sys.platform != "darwin")
    # The trace prints every visit of the warmup and the four slots.
    traced = [line.split()[1] for line in out.splitlines() if line.startswith("mem ")]
    assert sorted(traced) == ["0"] * 5 + ["1"] * 5


def test_resume_runs_only_the_slots_left(tmp_path):
    _, raw, recs, _ = _bench(tmp_path)
    lines = raw.read_text().splitlines()
    # Meta, warmup and two slots, then a record cut short.
    assert [r["kind"] for r in recs[:4]] == ["meta", "warmup", "slot", "slot"]
    raw.write_text("\n".join(lines[:4]) + "\n" + lines[4][:20])
    result, _, after, _ = _bench(tmp_path, "--resume")
    assert after[:4] == recs[:4]
    assert [r["kind"] for r in after] == ["meta", "warmup"] + ["slot"] * 4 + ["end"]
    done = [(r["round"], r["slot"]) for r in after if r["kind"] == "slot"]
    assert sorted(done) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    visits = [row[2] for r in after if r["kind"] == "slot" for row in r["samples"]]
    assert len(set(visits)) == 8
    assert result["complete"]
    (p,) = result["passes"]
    assert p["slots_done"] == p["slots_planned"] == 4
    # A complete log only rebuilds the reports.
    before = raw.read_text()
    _bench(tmp_path, "--resume")
    assert raw.read_text() == before


_CHAIN = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import mlx.core as mx
    import dit_common as dc

    seen = []

    def f(a):
        seen.append(a)
        # The second call's output trips the tie, so the third call is fed
        # zeros: only a chained call sees it.
        return a + (1e31 if len(seen) == 2 else 0.0)

    x0 = mx.ones((2, 3), dtype=mx.float32)
    y = dc.chain(f, x0, 4)
    mx.eval(y)
    b = dc.tie(mx.ones((4,), dtype=mx.bfloat16), x0.astype(mx.bfloat16))
    ms = dc.chain_ms(lambda a: a * 2, x0, 3)
    print(json.dumps({
        "first": [float(a[0, 0]) for a in seen],
        "out": float(y[0, 0]),
        "tie_dtype": str(b.dtype),
        "ms": ms,
    }))
    """
)


def test_sample_is_one_dependent_chain(tmp_path):
    script = tmp_path / "chain.py"
    script.write_text(_CHAIN)
    # A child process, since dit_common sets KQ_TUNE_LIVE on import.
    out = subprocess.run(
        [sys.executable, str(script), BENCH_DIR],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )
    r = json.loads(out.stdout.splitlines()[-1])
    assert r["first"] == [1.0, 1.0, 0.0, 1.0]
    assert r["out"] == 1.0
    assert r["tie_dtype"] == "mlx.core.bfloat16"
    assert r["ms"] > 0


def test_lever_arm_pairs_with_kq(tmp_path):
    result, _, recs, _ = _bench(
        tmp_path, "--env-arm", "sw1:KQ_NAX_SWIZZLE=1,KQ_QMM_MIDM=0"
    )
    assert recs[0]["args"]["arms"] == ["kq", "kq_aa", "sw1"]
    (p,) = result["passes"]
    for g in p["groups"]:
        (c,) = g["cells"]
        assert set(c["arms"]) == {"kq", "kq_aa", "sw1"}
        assert c["ratios"]["sw1/kq"]["pairs"]
    assert "sw1/kq" in (tmp_path / "run.md").read_text()


@pytest.mark.parametrize("spec", ["sw1", "sw1:", ":KQ_NAX_SWIZZLE=1", "kq:A=1"])
def test_lever_arm_malformed(tmp_path, spec):
    with pytest.raises(subprocess.CalledProcessError):
        _bench(tmp_path, "--env-arm", spec)


def test_time_budget_splits_a_run(tmp_path):
    # A budget that every slot passes: each process runs one slot and stops,
    # and --resume continues until the run is complete.
    runs = 0
    while True:
        result, _, recs, out = _bench(tmp_path, "--time-budget-min", "1e-9", "--resume")
        runs += 1
        if result["complete"]:
            break
        assert "--resume continues" in out
        assert runs < 6
    assert runs == 4
    assert [r["kind"] for r in recs].count("slot") == 4
    assert recs[-1]["kind"] == "end"
    (p,) = result["passes"]
    assert p["slots_done"] == p["slots_planned"] == 4
