# Contributing to mlx-kquant

Thanks for your interest! This is a C++/Metal MLX extension with a Python
front-end. A few notes to get productive quickly.

## Prerequisites

- **macOS on Apple Silicon** with the Xcode command-line tools (`xcrun metal`
  must work — the kernels compile to a `.metallib`).
- The exact pinned MLX wheel. The kernels include MLX's steel headers and the
  extension links `libmlx`, so the ABI must match:

  ```sh
  python -m venv .venv && source .venv/bin/activate
  pip install "mlx==0.31.2" "nanobind==2.12.0" "cmake>=3.27"
  pip install -e ".[dev,tools]"   # builds _ext + mlx_kquant.metallib
  ```

## Running the checks

```sh
ruff check . && ruff format --check .      # lint + format
python scripts/check-codecs.py --check     # codec geometry doc-lint (no GPU)
pytest -q                                  # op tests
```

The op tests need a Metal GPU (until the CPU decode path is in place, after which
they also run under `KQUANT_FORCE_CPU=1`). Tests gated on `KQUANT_TEST_*`
environment variables need external assets (real GGUFs / HF models) and are
skipped by default.

## Regenerating the test fixtures

`tests/fixtures/*.npz` are reproducible from this repo with a built extension and
a GPU:

```sh
python tests/gen_fixtures.py            # re-encodes via kq.quantize
shasum -a 256 -c tests/fixtures/SHA256SUMS
```

Commit both the regenerated `.npz` files and the updated `SHA256SUMS` if you
change codec geometry or the encoder.

## Adding a new codec

Codec facts live in **one** place: `mlx_kquant/codec_geometry.py`. Update it, the
README "Codec reference" table, and the kernel, then run
`python scripts/check-codecs.py --check` — it fails on drift between the three.

## Style

- `ruff` is the linter/formatter (config in `pyproject.toml`); a
  `.pre-commit-config.yaml` runs it on commit (`pre-commit install`).
- Keep the raw `kq.*` op layer dependency-free; model-level helpers belong behind
  the `[tools]` extra.
