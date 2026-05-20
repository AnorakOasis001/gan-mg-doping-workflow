# CI Strategy

This repository uses a three-tier CI strategy to balance iteration speed with scientific rigor and reproducibility.

## Tier 1: Fast PR CI

**Workflow:** `.github/workflows/ci.yml`

Runs on pull requests and non-`main` branch pushes for Python/code-relevant changes.

Includes:
- Ubuntu-only execution
- Editable install (`-e ".[dev]" --no-build-isolation`)
- Optional ruff lint
- mypy
- smoke tests
- `pytest -m "not slow"`

Goal: fast, high-signal feedback (target 2-4 minutes).

## Tier 2: Full Validation CI

**Workflow:** `.github/workflows/full-validation.yml`

Runs on `push` to `main`, `workflow_dispatch`, and nightly schedule.

Includes:
- Ubuntu + Windows matrix
- Python 3.11/3.12 coverage
- Core + plot dependency modes
- Full pytest suite (including slow scientific tests)
- Artifact build and install validation
- Benchmark/reproducibility checks
- Plotting/scientific integration paths

Goal: preserve strict scientific confidence and reproducibility guarantees on protected branches.

## Tier 3: Local Developer Quick Checks

Use one of:
- `scripts/quick_check.sh`
- `scripts/quick_check.ps1`

These run fast pre-push checks:
1. `ruff` (if installed)
2. `mypy src`
3. smoke tests (or `tests/smoke` if present)

## Slow Tests

Heavy scientific tests are marked with `@pytest.mark.slow` and excluded from fast PR CI via:

```bash
pytest -m "not slow"
```

Full validation runs all tests, including slow markers.

## Concurrency Cancellation

All workflows use:

- `cancel-in-progress: true`
- group by workflow + PR/ref

This cancels stale runs on new commits to reduce CI waste and improve feedback latency.

## Contributor Workflow

Recommended flow:
1. Activate virtual environment.
2. Run local quick checks (`scripts/quick_check.sh` or `.ps1`).
3. Push branch.
4. Wait for Fast PR CI to pass.
5. Merge only after Full Validation CI on `main` is green.
