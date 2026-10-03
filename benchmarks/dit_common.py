"""Shared pieces of the DiT benchmarks: shape matrix, weights, arms, kernel
log capture, Thue-Morse order and the machine descriptor.

Imported by the DiT benchmarks in this directory, which run from the
repository root or from here.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import sys
import tempfile
import time
import zlib

# Tile levers must be live before mlx_kquant reads them on its first call.
os.environ["KQ_TUNE_LIVE"] = "1"

TESTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"
)

# (K, N) per linear. Rows per item are image tokens plus padded caption
# tokens at 1024 and 2048 pixels, each also run as a batch of two; Qwen-Image
# adds two text lengths that are not multiples of 32. The modulation linears
# take one row per image (the timestep embedding), so they run at M 1 and 2.
IMAGE_ROWS = [4096, 4128, 4192, 4256, 4608, 4640, 16384, 16896]
FAMILIES = {
    "zimage": {
        "ops": {
            "qkv": (3840, 11520),
            "out": (3840, 3840),
            "w13": (3840, 10240),
            "w2": (10240, 3840),
            "adaln": (256, 15360),
        },
        "rows": IMAGE_ROWS,
        "op_rows": {"adaln": [1]},
    },
    "qwen_image": {
        "ops": {
            "qkvo": (4096, 4096),
            "gate_up": (4096, 24576),
            "mlp_out": (12288, 4096),
            "mod": (4096, 16384),
        },
        "rows": IMAGE_ROWS + [4096 + 77, 4096 + 213],
        "op_rows": {"mod": [1]},
    },
}
# The two heaviest cells by FLOPs per step, where every NAX codec runs.
HEAVY = [("zimage", "w13", 4608, 1), ("qwen_image", "gate_up", 4096, 1)]
MAIN_CODECS = ["q4_k", "q5_k", "q6_k", "q8_0"]

# t[i] = parity of popcount(i): balances linear drift without the quadratic
# aliasing of ABBA blocks.
THUE_MORSE = [bin(i).count("1") & 1 for i in range(8)]

TILE_BM64 = "64,64,2,2"
TILE_BM128 = "128,64,2,2"
ARM_TILE_FORCED = ("kq_bm64", "kq_bm128")

DTYPES = ("bfloat16", "float16")


def mx_dtype(name):
    import mlx.core as mx

    return {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}[name]


def shrink_dim(v, f, mult):
    """v / f rounded up to a multiple of `mult`, at least `mult`."""
    if f <= 1:
        return v
    return max(mult, -(-(v // f) // mult) * mult)


def nax_codecs():
    """Codecs with NAX qmm_t kernels in the bundled metallib, independent of
    whether this GPU has NAX, so every machine runs the same list."""
    import mlx_kquant as kq
    from mlx_kquant.codec_geometry import CODEC_GEOMETRY

    pat = re.compile(r"^kquant_(.+?)_qmm_t_nax_")
    found = {m[1] for n in kq.metallib_kernels() if (m := pat.match(n))}
    return sorted(found & set(CODEC_GEOMETRY))


def kq_weight(codec, n, k, seed):
    """Quantized (N, K) weight. Codecs the tests cannot encode get random
    structurally valid wire with a sane scale; the rest quantize a normal
    matrix, so the decode sees realistic scales."""
    import mlx.core as mx
    import numpy as np

    import mlx_kquant as kq
    from mlx_kquant.codec_geometry import CODEC_GEOMETRY, DECODE_ONLY_CODECS

    if TESTS_DIR not in sys.path:
        sys.path.insert(0, TESTS_DIR)
    from kqref import is_synth, synth_wire

    _, _, bpb, wpb = CODEC_GEOMETRY[codec]
    if is_synth(codec) or codec in DECODE_ONLY_CODECS:
        rng = np.random.default_rng(seed)
        wire = synth_wire(rng, codec, bpb, n * (k // wpb))
        w = mx.array(wire.reshape(n, (k // wpb) * bpb))
        s = mx.zeros((1,), dtype=mx.uint8)
    else:
        wf = mx.random.normal((n, k), key=mx.random.key(seed)) * 0.02
        w, s = kq.quantize(wf, codec)
    mx.eval(w, s)
    return w, s


def weight_bits(codec):
    """Stored bits per weight from the codec geometry."""
    from mlx_kquant.codec_geometry import CODEC_GEOMETRY

    _, _, bpb, wpb = CODEC_GEOMETRY[codec]
    return 8.0 * bpb / wpb


def affine_bits(codec):
    return 8 if codec == "q8_0" else 4


class KernelLog:
    """Capture the KQ_QMM_LOG lines (kernel name, M, N, K, grid) printed
    while a block runs, by pointing fd 2 at a temporary file."""

    LINE = re.compile(
        r"^\[kq qmm\] (\S+) M=(\d+) N=(\d+) K=(\d+) grid=(\d+),(\d+),(\d+)$"
    )

    def __enter__(self):
        sys.stderr.flush()
        self._tmp = tempfile.TemporaryFile(mode="w+b")
        self._saved = os.dup(2)
        os.dup2(self._tmp.fileno(), 2)
        self._prev = os.environ.get("KQ_QMM_LOG")
        os.environ["KQ_QMM_LOG"] = "1"
        self.kernels = []
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop("KQ_QMM_LOG", None)
        else:
            os.environ["KQ_QMM_LOG"] = self._prev
        sys.stderr.flush()
        os.dup2(self._saved, 2)
        os.close(self._saved)
        self._tmp.seek(0)
        other = []
        for line in self._tmp.read().decode(errors="replace").splitlines():
            m = self.LINE.match(line)
            if m:
                self.kernels.append(m[1])
            elif line.strip():
                other.append(line)
        self._tmp.close()
        if other:
            sys.stderr.write("\n".join(other) + "\n")
        return False


def tile_of(kname):
    """'bmX bnY wmA wnB' from a kq qmm kernel name; the default ALU tile has
    no tag in its name."""
    if kname is None:
        return None
    m = re.search(r"_bm(\d+)_bn(\d+)(?:_bk\d+)?_wm(\d+)_wn(\d+)", kname)
    db = "_db" if kname.endswith("_db") else ""
    if m:
        return f"bm{m[1]} bn{m[2]} wm{m[3]} wn{m[4]}{db}"
    if "_qmm_t_" in kname:
        return "bm64 bn64 wm2 wn2"
    return None


def tile_bm_bn(tile):
    m = re.match(r"bm(\d+) bn(\d+)", tile or "")
    return (int(m[1]), int(m[2])) if m else (None, None)


def mlx_dense_tile(M, N, K, arch, nax):
    """Tile of MLX's GEMM for a half or bfloat x @ w.T, derived from its
    dispatch rule; MLX does not log it."""
    letter = arch[-1:] if arch else ""
    if nax:
        if letter in ("s", "c", "d"):
            bk = 64 if (K >= 8192 and K > M + N) else 256
            return f"bm64 bn128 bk{bk} wm2 wn4 (rule)"
        return "bm128 bn128 bk512 wm4 wn4 (rule)"
    if letter in ("g", "p"):
        return "bm64 bn32 bk32 wm2 wn2 (rule)"
    if letter == "d":
        if M * N >= 1 << 20 and 2 * max(M, N) > K:
            return "bm64 bn64 bk16 wm1 wn2 (rule)"
        return "bm64 bn32 bk32 wm2 wn2 (rule)"
    return "bm64 bn64 bk16 wm2 wn2 (rule)"


def mlx_affine_tile(nax):
    """Tile of MLX's affine quantized matmul, from its dispatch rule."""
    if nax:
        return "bm64 bn64 bk64 wm2 wn2 (rule)"
    return "bm32 bn32 bk32 wm2 wn2 (rule)"


ARM_TILE = {"kq": None, "kq_aa": None, "kq_bm64": TILE_BM64}
ARM_TILE["kq_bm128"] = TILE_BM128


def set_tile(tile, nax):
    os.environ.pop("KQ_NAX_TILE", None)
    os.environ.pop("KQ_QMM_TILE", None)
    if tile:
        os.environ["KQ_NAX_TILE" if nax else "KQ_QMM_TILE"] = tile


class Store:
    """Weights and activations, built on first use."""

    def __init__(self, seed):
        self.seed = seed
        self.kqw = {}
        self.dense = {}
        self.affine = {}
        self.x = {}

    def clear_dtype(self):
        self.dense.clear()
        self.affine.clear()
        self.x.clear()

    def kq(self, g, codec):
        key = (g["family"], g["op"], codec)
        if key not in self.kqw:
            h = zlib.crc32("/".join(key).encode()) % 100000
            self.kqw[key] = kq_weight(codec, g["N"], g["K"], self.seed + h)
        return self.kqw[key]

    def dense_w(self, g, dt):
        import mlx.core as mx

        key = (g["family"], g["op"])
        if key not in self.dense:
            w = mx.random.normal((g["N"], g["K"]), key=mx.random.key(self.seed))
            self.dense[key] = (w * 0.02).astype(dt)
            mx.eval(self.dense[key])
        return self.dense[key]

    def affine_w(self, g, dt, bits):
        import mlx.core as mx

        key = (g["family"], g["op"], bits)
        if key not in self.affine:
            q = mx.quantize(self.dense_w(g, dt), group_size=64, bits=bits)
            mx.eval(*q)
            self.affine[key] = q
        return self.affine[key]

    def act(self, g, dt):
        import mlx.core as mx

        key = (g["K"], g["L"], g["B"])
        if key not in self.x:
            shape = (g["B"], g["L"], g["K"])
            x = mx.random.normal(shape, key=mx.random.key(self.seed + 1)) * 0.5
            self.x[key] = x.astype(dt)
            mx.eval(self.x[key])
        return self.x[key]


def make_call(arm, codec, g, dt, store):
    """(x0, f): the arm's op as a function of its activation, and the
    activation of the shape."""
    import mlx.core as mx

    import mlx_kquant as kq

    x = store.act(g, dt)
    if arm == "dense":
        wd = store.dense_w(g, dt)
        return x, lambda a: a @ wd.T
    if arm.startswith("affine"):
        bits = int(arm[len("affine") :])
        wq, sc, bi = store.affine_w(g, dt, bits)
        return x, lambda a: mx.quantized_matmul(
            a, wq, sc, bi, transpose=True, group_size=64, bits=bits
        )
    w, s = store.kq(g, codec)
    if arm == "deq":
        return x, lambda a: a @ kq.dequantize(w, s, codec, dtype=dt).T
    return x, lambda a: kq.quantized_matmul(a, w, s, codec, transpose=True)


def make_fn(arm, codec, g, dt, store):
    x, f = make_call(arm, codec, g, dt, store)
    return lambda: f(x)


def tie(o, x0):
    """x0 again, computed from the first element of `o`, so a call fed the
    result starts after the call that made `o` has finished."""
    import mlx.core as mx

    return mx.where(o.reshape(-1)[:1] > 1e30, 0.0, x0)


def chain(f, x0, n):
    """n calls of f, each fed the tie of the previous output: one lazy
    dependent chain."""
    x = x0
    for _ in range(n):
        x = tie(f(x), x0)
    return x


def chain_ms(f, x0, n):
    """ms per call over a chain of n calls in one evaluation. The GPU runs
    the calls one after another, as in a model forward; independent calls in
    one evaluation would overlap."""
    import mlx.core as mx

    x = chain(f, x0, n)
    t0 = time.perf_counter()
    mx.eval(x)
    return (time.perf_counter() - t0) * 1e3 / n


def time_fn(fn, inner):
    """ms per call over `inner` calls evaluated together, so one sync (about
    0.17 ms) is spread over all of them."""
    import mlx.core as mx

    t0 = time.perf_counter()
    ys = [fn() for _ in range(inner)]
    mx.eval(*ys)
    ms = (time.perf_counter() - t0) * 1e3 / inner
    del ys
    return ms


def op_rows(family, op, rows=None):
    """Rows per item for an op: its fixed rows if it has them, else `rows`
    or the family's image rows."""
    spec = FAMILIES[family]
    return spec["op_rows"].get(op) or rows or spec["rows"]


def cell(family, op, rows, batch, shrink=1):
    """One shape visit: family op at `rows` per item and `batch` items."""
    K, N = FAMILIES[family]["ops"][op]
    L = max(1, rows // shrink) if shrink > 1 else rows
    return {
        "family": family,
        "op": op,
        "K": shrink_dim(K, shrink, 256),
        "N": shrink_dim(N, shrink, 64),
        "L": L,
        "B": batch,
        "M": batch * L,
        "L_model": rows,
        "vector": op in FAMILIES[family]["op_rows"],
    }


def metallib_sha256():
    import mlx_kquant as kq

    path = os.path.join(kq.metallib_dir(), "mlx_kquant.metallib")
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def machine_descriptor():
    """Chip, arch generation and letter, GPU cores, NAX, threadgroup memory,
    OS, versions, metallib hash and the tuning environment."""
    import mlx.core as mx

    import mlx_kquant as kq

    info = mx.device_info()
    arch = info.get("architecture", "")
    m = re.match(r"applegpu_g(\d+)(\w)", arch)
    env = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("KQ_", "MLX_", "KQUANT_"))
    }
    return {
        "device": info.get("device_name"),
        "architecture": arch,
        "arch_gen": int(m[1]) if m else None,
        "arch_letter": m[2] if m else None,
        "gpu_cores": kq.gpu_core_count(),
        "nax": kq.nax_available(),
        "max_threadgroup_memory": kq.max_threadgroup_memory(),
        "memory_bytes": info.get("memory_size"),
        "os": platform.mac_ver()[0] or platform.platform(),
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "mlx_kquant": kq.__version__,
        "metallib_sha256": metallib_sha256(),
        "tf32": os.environ.get("MLX_ENABLE_TF32", "default"),
        "env": env,
    }


def descriptor_md(d):
    return (
        f"Device: {d['device']} ({d['architecture']}, gen {d['arch_gen']}, "
        f"letter {d['arch_letter']}), {d['gpu_cores']} GPU cores, "
        f"NAX {d['nax']}, threadgroup memory {d['max_threadgroup_memory']} B. "
        f"macOS {d['os']}, mlx {d['mlx']}, mlx-kquant {d['mlx_kquant']}, "
        f"metallib {str(d['metallib_sha256'])[:12]}, TF32 {d['tf32']}."
    )
