# Contributing to mlx-kquant

Thanks for your interest! This is a C++/Metal MLX extension with a Python
front-end. A few notes to get productive quickly.

## Prerequisites

- **macOS on Apple Silicon** with the Xcode command-line tools (`xcrun metal`
  must work - the kernels compile to a `.metallib`).
- The exact pinned MLX wheel. The kernels include MLX's steel headers and the
  extension links `libmlx`, so the ABI must match:

  ```sh
  python -m venv .venv && source .venv/bin/activate
  pip install "mlx==0.32.3" "nanobind==3.0.1" "cmake>=3.27"
  pip install -e ".[dev,tools]"   # builds _ext + mlx_kquant.metallib
  ```

## Running the checks

```sh
ruff check . && ruff format --check .      # Python lint + format
clang-format --dry-run --Werror \
  bindings.cpp src/*.cpp src/*.h \
  metal/*.metal metal/mlx/backend/metal/kernels/*.h   # C++/Metal format check
python scripts/check-codecs.py --check     # codec geometry doc-lint (no GPU)
pytest -q                                  # op tests
```

The op tests run on a Metal GPU or, under `KQUANT_FORCE_CPU=1`, on CPU (the decode
and encode paths both have a scalar CPU implementation). Tests gated on `KQUANT_TEST_*`
environment variables need external assets (real GGUFs / HF models) and are
skipped by default.

## Regenerating the test fixtures

`tests/fixtures/*.npz` are reproducible from this repo with a built extension
(encode runs on CPU or Metal):

```sh
python tests/gen_fixtures.py            # re-encodes via kq.quantize
shasum -a 256 -c tests/fixtures/SHA256SUMS
```

Commit both the regenerated `.npz` files and the updated `SHA256SUMS` if you
change codec geometry or the encoder.

## Adding a new codec

Codec facts live in **one** place: `mlx_kquant/codec_geometry.py`. Update it, the
README "Codec reference" table, and the kernel, then run
`python scripts/check-codecs.py --check` - it fails on drift between the three.

## Kernels and the wheel size

PyPI refuses any file of this project over 100 MB, and the metallib is most
of the wheel. Every kernel instantiation adds to it, so check the cost of a
change before you open a pull request:

```sh
python scripts/check-wheel-size.py --check   # wheel estimate, largest kernel families
```

`tests/test_wheel_size.py` fails when the estimate passes 95 MB. If your
change takes it past 90 MB, fold another family in the same change, largest
first.

### One instantiation per code shape

Instantiate a kernel once for each thing that changes the shape of its code:
the codec traits, tile sizes, `NX` lanes, anything that sizes an array or
picks a matrix type. Everything else is a function constant, which Metal
resolves when it builds the pipeline, so the specialized kernel runs at the
same speed.

- Activation dtype. A kernel whose math runs in float32 takes `kq_x16`
  activations (`kq_x16.h`) and is instantiated once, named `..._kq_x16`.
  Load with `kq_ld4` or `kq_ld8` and store with `static_cast<T>`.
- GLU activation. Instantiate with `KQ_GLU_ACT_FC` and put `act` in the name
  where the activation was.
- Host code keeps asking for the typed name, such as
  `kq_q4_k_moe_glu_gather_silu_nx16_bfloat16_t`. `kq_get_kernel` finds the
  folded function and sets function constants 390 (dtype) and 391
  (activation).
- Kernels that load activations into matrix-unit tiles (`tensor<...>`,
  `simdgroup_matrix`: the NAX matmuls and the expert stores) need the real
  type and keep one instantiation per type.

When you fold an existing family, compare `kq.metallib_kernels()` before
and after: every old name must still resolve under the rule above, and every
new function must be reached by one.

### Iterate on a kernel

Compile the one source you changed, in seconds, before a full build:

```sh
MLX_INC=$(python -c 'import mlx, os; print(os.path.join(os.path.dirname(mlx.__file__), "include"))')
xcrun -sdk macosx metal -Imetal -I"$MLX_INC" -Wall -Wextra -fno-fast-math \
  -Wno-c++17-extensions -Wmetal-addr-spaces \
  -c metal/kq_moe_glu_kq.metal -o /tmp/kq_moe_glu_kq.air
```

Then rebuild in place, run the test files of the family you touched, and run
the whole suite once before the pull request:

```sh
python setup.py build_ext --inplace
pytest -q tests/test_moe_glu.py tests/test_moe_glu_variants.py
python scripts/check-wheel-size.py --check
pytest -q
```

## Style

- **Python:** `ruff` is the linter/formatter (config in `pyproject.toml`).
- **C++ / Metal:** `clang-format` (config in `.clang-format`) covers all
  first-party sources - the host code (`src/*.{cpp,h}`, `bindings.cpp`) and the
  Metal kernels (`metal/`). The kernels' hand-aligned macro-instantiation
  tables are exempted with inline `// clang-format off` / `// clang-format on`
  regions in the `.metal` files; everything else is enforced.
- `.pre-commit-config.yaml` runs both on commit (`pre-commit install`); the CI
  `lint` job enforces them.
- Keep the raw `kq.*` op layer dependency-free; model-level helpers belong behind
  the `[tools]` extra.
