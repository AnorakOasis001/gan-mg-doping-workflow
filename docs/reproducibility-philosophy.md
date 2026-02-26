# Reproducibility philosophy

This project treats reproducibility as a first-class software requirement.

## Core principles

1. **Run isolation**: every execution is scoped to `runs/<run-id>/`.
2. **Determinism**: synthetic generation is controlled by explicit seeds.
3. **Stable schema**: thermodynamic outputs use consistent field names.
4. **Side-effect boundaries**: pure API functions are separated from CLI file I/O.
5. **Cross-platform behavior**: smoke scripts and CLI are usable on Linux and Windows.

## Why this matters

Scientific claims are only as strong as their ability to be reproduced. By organizing outputs per run, fixing inputs and seeds, and keeping thermodynamic logic testable, the workflow supports:

- repeatable experiments
- easier debugging
- robust collaboration
- CI automation

## Provenance and schema versioning metadata

`metrics.json` and `diagnostics_T{T}.json` now include a `provenance` block that captures lightweight runtime metadata for auditability while preserving backward compatibility of existing fields.

- `schema_version`: explicit metadata schema version (`"1.1"`) for forward-compatible consumers.
- `git_commit`: best-effort git commit hash when available; `null` when git is unavailable or the working directory is not a git checkout.
- `python_version` and `platform`: runtime environment fingerprint for cross-platform traceability.
- `cli_args`: CLI arguments used for the run.
- `input_hash`: reproducibility hash tying outputs to input content and analysis settings.

These fields strengthen reproducibility guarantees by making each artifact self-describing and easier to audit in CI, local development, and published result bundles.

Example:

```json
{
  "provenance": {
    "schema_version": "1.1",
    "git_commit": "4b6f9f1b8e0c3a1d9d9b8a0f9b8b12d1fbb2e6aa",
    "python_version": "3.11.9",
    "platform": "Linux-6.8.0-x86_64-with-glibc2.39",
    "cli_args": {
      "run_dir": "runs",
      "run_id": "example-run",
      "T": 500.0
    },
    "input_hash": "sha256:..."
  }
}
```
