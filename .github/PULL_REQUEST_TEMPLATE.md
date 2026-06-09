## What

<!-- A short description of the change and why. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `python scripts/check-codecs.py --check` passes (codec geometry in sync)
- [ ] `pytest -q` passes (note if any tests need a GPU / external assets)
- [ ] Updated `CHANGELOG.md` under `[Unreleased]` if user-facing
- [ ] If codec geometry / encoder changed: regenerated `tests/fixtures/` +
      `SHA256SUMS`

## Notes

<!-- Anything reviewers should know: ABI/pin implications, GPU-only paths, etc. -->
