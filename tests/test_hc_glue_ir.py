"""The handoff buffers of the fused hyper-connection cycle in the compiled IR.

kq_hc_front_expand_collapse hands h, the mix dots and the sum of squares
from every threadgroup of a row to the last one to arrive, inside one
dispatch. By default device memory has threadgroup coherence (MSL spec
4.8, Memory Coherency), so those accesses must be coherent(device), which
the compiler lowers to air.store.device_coherent and
air.load.device_coherent. The split kernels hand off at a dispatch
boundary and keep plain accesses. Compiles metal/kq_hc_glue.metal to LLVM
IR with xcrun metal, as the build does; skips where that is missing.

Usage: test_hc_glue_ir.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import mlx.core as mx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "metal", "kq_hc_glue.metal")
MLX_INCLUDE = os.path.join(os.path.dirname(mx.__file__), "include")

_DEFINE = re.compile(r"^define [^@]*@([\w.$]+)\(")
_CALL = re.compile(r"@([\w.$]+)\(")


def _metal():
    if shutil.which("xcrun") is None or not os.path.exists(SOURCE):
        return None
    r = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else None


@pytest.fixture(scope="module")
def defines(tmp_path_factory):
    """Function name -> body lines of the compiled kq_hc_glue.metal."""
    metal = _metal()
    if metal is None:
        pytest.skip("needs xcrun metal and the metal/ sources")
    out = tmp_path_factory.mktemp("ir") / "kq_hc_glue.ll"
    # The build's Metal deployment target (setup.py).
    env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET="26.2")
    cmd = [metal, "-S", "-emit-llvm", "-I", os.path.join(ROOT, "metal")]
    cmd += ["-I", MLX_INCLUDE, "-I", os.path.join(MLX_INCLUDE, "metal_cpp")]
    cmd += ["-Wno-c++17-extensions", SOURCE, "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-4000:]
    funcs, name = {}, None
    for line in out.read_text().splitlines():
        m = _DEFINE.match(line)
        if m:
            name = m.group(1)
            funcs[name] = []
        elif line == "}":
            name = None
        elif name is not None:
            funcs[name].append(line)
    return funcs


def _reach(funcs, kernel):
    # The kernel's body and the bodies of the defined functions it calls.
    seen, todo, lines = set(), [kernel], []
    while todo:
        f = todo.pop()
        if f in seen or f not in funcs:
            continue
        seen.add(f)
        lines += funcs[f]
        todo += [c for line in funcs[f] for c in _CALL.findall(line)]
    return "\n".join(lines)


def _kernels(funcs, prefix):
    names = [f for f in funcs if re.fullmatch(rf"{prefix}_(bfloat16|float16)_t", f)]
    assert len(names) == 2, (prefix, names)
    return names


def test_fused_handoff_is_device_coherent(defines):
    # Both the stores that publish and the loads of the collapse.
    for name in _kernels(defines, "kq_hc_front_expand_collapse"):
        body = _reach(defines, name)
        assert "@air.store.device_coherent." in body, name
        assert "@air.load.device_coherent." in body, name


@pytest.mark.parametrize(
    "prefix",
    ["kq_hc_front_reduce", "kq_hc_front_expand_reduce", "kq_hc_sinkhorn_collapse"],
)
def test_split_kernels_stay_plain(defines, prefix):
    for name in _kernels(defines, prefix):
        assert "device_coherent" not in _reach(defines, name), name
