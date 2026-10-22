#!/usr/bin/env python3
"""Per-file and per-test times from a pytest JUnit report, as Markdown.

  python scripts/test-timing.py junit.xml [--top N] >> "$GITHUB_STEP_SUMMARY"

A test's time covers its setup, call and teardown. Without a report, as
after a crashed run, it prints a note and exits 0.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def split(classname: str) -> tuple[str, str]:
    """A JUnit classname as its test file and the class path inside it."""
    parts = classname.split(".")
    for n in range(len(parts), 0, -1):
        path = "/".join(parts[:n]) + ".py"
        if Path(path).is_file():
            return path, "::".join(parts[n:])
    return classname, ""


def read(report: Path) -> list[tuple[str, str, float]]:
    """(file, test name, seconds) for each test case."""
    cases = []
    for case in ET.parse(report).iter("testcase"):
        path, cls = split(case.get("classname", ""))
        name = case.get("name", "")
        cases.append(
            (path, f"{cls}::{name}" if cls else name, float(case.get("time") or 0))
        )
    return cases


def cell(text: str) -> str:
    return text.replace("|", "\\|")


def render(cases: list[tuple[str, str, float]], top: int) -> str:
    total = sum(t for _, _, t in cases)
    files: dict[str, list[float]] = defaultdict(list)
    for f, _, t in cases:
        files[f].append(t)
    lines = [
        "## Test timing",
        "",
        f"{len(cases)} tests, {total / 60:.1f} min of test time.",
        "",
        "### Slowest files",
        "",
        "| File | Tests | Minutes | Share |",
        "|---|---:|---:|---:|",
    ]
    ranked = sorted(files.items(), key=lambda kv: -sum(kv[1]))
    for f, times in ranked[:top]:
        s = sum(times)
        share = 100 * s / total if total else 0.0
        lines.append(f"| {cell(f)} | {len(times)} | {s / 60:.1f} | {share:.0f}% |")
    lines += ["", "### Slowest tests", "", "| Test | Seconds |", "|---|---:|"]
    for f, name, t in sorted(cases, key=lambda c: -c[2])[:top]:
        lines.append(f"| {cell(f'{f}::{name}')} | {t:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", type=Path, help="pytest --junitxml output")
    ap.add_argument("--top", type=int, default=25, help="rows per table")
    args = ap.parse_args()
    if not args.report.is_file():
        print(f"No test report at {args.report}; no timing table.")
        return 0
    sys.stdout.write(render(read(args.report), args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
