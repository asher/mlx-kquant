#!/usr/bin/env python3
"""Generate KVarN golden fixtures from the BeeLlama CPU reference.

Extracts the kvarn_cpu_* functions from a beellama.cpp checkout at generation
time (nothing is vendored), compiles a standalone driver, and emits records,
reconstructions, Sinkhorn scales, and WHT outputs for fixed random tiles:

    python tests/fixtures/kvarn/gen_kvarn_fixtures.py
    shasum -a 256 -c tests/fixtures/kvarn/SHA256SUMS   # from the repo root

The BeeLlama CPU backend accumulates the Sinkhorn stds in double while the
GPU path (and our Metal port and tests/kvarn_ref.py) uses fp32, so consumers
compare against these fixtures with tolerances, not bit-exactness.

Checkout location: KVARN_BEELLAMA_DIR (default ~/src/beellama.cpp).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

PINNED_COMMIT = "d71585e3fa6fd30b55fadb0528170edb5de17126"

FUNCS = [
    "kvarn_cpu_hadamard",
    "kvarn_cpu_hadamard_head",
    "kvarn_cpu_sample_std",
    "kvarn_cpu_imbalance",
    "kvarn_cpu_variance_normalize",
    "kvarn_cpu_pack",
    "kvarn_cpu_unpack",
    "kvarn_cpu_record_value",
    "kvarn_cpu_quantize_stage",
]

BITS = (2, 3, 4, 5, 6, 8)
RECON_BITS = (2, 6)
N_TILES = 4

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

PRELUDE = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

typedef __fp16 ggml_fp16_t;
static inline float ggml_fp16_to_fp32(ggml_fp16_t x) { return (float) x; }
static inline ggml_fp16_t ggml_fp32_to_fp16(float x) { return (ggml_fp16_t) x; }
#define GGML_ASSERT(x) assert(x)
struct ggml_tensor { void * data; size_t nb[4]; };
"""

DRIVER = r"""
static size_t record_nbytes(int bits) {
    return size_t(KVAR_N_GROUP) * KVAR_N_GROUP * bits / 8 + 3 * KVAR_N_GROUP * 2;
}

static void dump(const std::string & dir, const std::string & name, const void * p, size_t n) {
    const std::string path = dir + "/" + name + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    assert(f && fwrite(p, 1, n, f) == n);
    fclose(f);
}

int main(int argc, char ** argv) {
    assert(argc == 2);
    const std::string dir = argv[1];
    std::mt19937 rng(20260806);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    for (int tix = 0; tix < 4; ++tix) {
        // Input group, token-major [128 tokens][128 dims], stored fp16.
        std::vector<float> raw(128 * 128);
        if (tix == 0) {
            for (auto & v : raw) v = normal(rng);
        } else if (tix == 1) {
            std::vector<float> dim_scale(128);
            for (auto & s : dim_scale) s = std::exp(1.5f * normal(rng));
            for (int t = 0; t < 128; ++t)
                for (int d = 0; d < 128; ++d)
                    raw[t * 128 + d] = normal(rng) * dim_scale[d];
            for (int i = 0; i < 8; ++i)
                raw[(rng() % 128) * 128 + (rng() % 128)] = 40.0f * normal(rng);
        } else if (tix == 2) {
            for (int t = 0; t < 128; ++t)
                for (int d = 0; d < 128; ++d)
                    raw[t * 128 + d] = 4.0f * std::sin(0.11f * t) * std::cos(0.07f * d)
                                       + 0.1f * normal(rng);
        } else {
            for (int t = 0; t < 128; ++t)
                for (int d = 0; d < 128; ++d)
                    raw[t * 128 + d] = t < 64 ? 1.0f : (t < 96 ? 0.0f : 1e-6f * normal(rng));
        }
        std::vector<ggml_fp16_t> input(128 * 128);
        for (int i = 0; i < 128 * 128; ++i) input[i] = ggml_fp32_to_fp16(raw[i]);
        dump(dir, "t" + std::to_string(tix) + "_input", input.data(), input.size() * 2);

        // WHT outputs for head_dim 128/256/512 built from the same fp16 data.
        for (int slices : {1, 2, 4}) {
            const int heads = 128 / slices;
            std::vector<float> out(128 * 128);
            for (int h = 0; h < heads; ++h) {
                std::array<std::array<float, KVAR_N_GROUP>, 4> vals = {};
                for (int s = 0; s < slices; ++s)
                    for (int d = 0; d < 128; ++d)
                        vals[s][d] = ggml_fp16_to_fp32(input[(h * slices + s) * 128 + d]);
                kvarn_cpu_hadamard_head(vals, slices);
                for (int s = 0; s < slices; ++s)
                    for (int d = 0; d < 128; ++d)
                        out[(h * slices + s) * 128 + d] = vals[s][d];
            }
            dump(dir, "t" + std::to_string(tix) + "_wht" + std::to_string(slices * 128),
                 out.data(), out.size() * 4);
        }

        // Stage tensor: ptr = data + d*nb[0] + head*nb[1] + pos*nb[2].
        ggml_tensor stage = {};
        stage.data = input.data();
        stage.nb[0] = 2;
        stage.nb[1] = 0;
        stage.nb[2] = 256;

        for (int value = 0; value < 2; ++value) {
            const std::string side = value ? "v" : "k";
            // Sinkhorn scales (bits-independent).
            std::vector<float> tile(128 * 128);
            for (int t = 0; t < 128; ++t)
                for (int d = 0; d < 128; ++d)
                    tile[value ? t * 128 + d : d * 128 + t] = ggml_fp16_to_fp32(input[t * 128 + d]);
            std::vector<float> balanced;
            std::array<float, 128> s_col;
            std::array<float, 128> s_row;
            kvarn_cpu_variance_normalize(tile, 16, balanced, s_col, s_row);
            dump(dir, "t" + std::to_string(tix) + "_" + side + "_scol", s_col.data(), 128 * 4);
            dump(dir, "t" + std::to_string(tix) + "_" + side + "_srow", s_row.data(), 128 * 4);

            for (int bits : {2, 3, 4, 5, 6, 8}) {
                std::vector<uint8_t> record(record_nbytes(bits));
                kvarn_cpu_quantize_stage(&stage, 0, 0, 0, bits, 16, value != 0, record.data());
                dump(dir, "t" + std::to_string(tix) + "_" + side + "_b" + std::to_string(bits)
                          + "_record", record.data(), record.size());
                if (bits == 2 || bits == 6) {
                    std::vector<float> recon(128 * 128);
                    for (int t = 0; t < 128; ++t)
                        for (int d = 0; d < 128; ++d)
                            recon[t * 128 + d] = kvarn_cpu_record_value(
                                record.data(), bits, value != 0, t, d);
                    dump(dir, "t" + std::to_string(tix) + "_" + side + "_b"
                              + std::to_string(bits) + "_recon", recon.data(), recon.size() * 4);
                }
            }
        }
    }
    return 0;
}
"""


def _extract_functions(ops_cpp: str) -> str:
    """Pull the kvarn_cpu_* static functions out of ops.cpp by brace matching."""
    out = []
    for name in FUNCS:
        ix = ops_cpp.find(f" {name}(")
        if ix < 0:
            raise SystemExit(f"function {name} not found in ops.cpp")
        start = ops_cpp.rfind("static ", 0, ix)
        if start < 0:
            raise SystemExit(f"no 'static' prefix found for {name}")
        brace = ops_cpp.index("{", ix)
        depth = 0
        end = brace
        for end in range(brace, len(ops_cpp)):
            if ops_cpp[end] == "{":
                depth += 1
            elif ops_cpp[end] == "}":
                depth -= 1
                if depth == 0:
                    break
        out.append(ops_cpp[start : end + 1])
    return "\n\n".join(out)


def main() -> None:
    bee = os.path.expanduser(os.environ.get("KVARN_BEELLAMA_DIR", "~/src/beellama.cpp"))
    ops_path = os.path.join(bee, "ggml", "src", "ggml-cpu", "ops.cpp")
    if not os.path.exists(ops_path):
        raise SystemExit(f"beellama.cpp checkout not found at {bee} (set KVARN_BEELLAMA_DIR)")
    head = subprocess.run(
        ["git", "-C", bee, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if head != PINNED_COMMIT:
        print(f"warning: beellama.cpp HEAD {head[:12]} != pinned {PINNED_COMMIT[:12]}")

    with open(ops_path) as f:
        text = f.read()
    ix = text.find("static constexpr int KVAR_N_GROUP")
    group_decl = text[ix : text.index("\n", ix) + 1] if ix >= 0 else "static constexpr int KVAR_N_GROUP = 128;\n"
    source = PRELUDE + group_decl + _extract_functions(text) + DRIVER

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "gen.cpp")
        exe = os.path.join(tmp, "gen")
        with open(src, "w") as f:
            f.write(source)
        subprocess.run(["clang++", "-O2", "-std=c++17", "-o", exe, src], check=True)
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)
        subprocess.run([exe, raw], check=True)

        arrays: dict[str, np.ndarray] = {}
        for fn in sorted(os.listdir(raw)):
            name = fn[: -len(".bin")]
            data = np.fromfile(os.path.join(raw, fn), np.uint8)
            if name.endswith("_input"):
                arrays[name] = data.view(np.float16).reshape(128, 128)
            elif "_wht" in name or name.endswith(("_scol", "_srow")) or name.endswith("_recon"):
                arr = data.view(np.float32)
                arrays[name] = arr.reshape(128, 128) if arr.size == 128 * 128 else arr
            elif name.endswith("_record"):
                arrays[name] = data
            else:
                raise SystemExit(f"unrecognized fixture blob {fn}")

    out = os.path.join(HERE, "kvarn_cpu.npz")
    np.savez_compressed(out, **arrays)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    rel = os.path.relpath(out, REPO_ROOT)
    with open(os.path.join(HERE, "SHA256SUMS"), "w") as f:
        f.write(f"{digest}  {rel}\n")
    print(f"wrote {rel} ({os.path.getsize(out)} bytes, {len(arrays)} arrays)")


if __name__ == "__main__":
    main()
