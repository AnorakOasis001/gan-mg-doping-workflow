$ErrorActionPreference = "Stop"

Write-Host "[quick-check] ruff"
python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('ruff') else 1)"
if ($LASTEXITCODE -eq 0) {
  python -m ruff check src tests
} else {
  Write-Host "ruff not installed; skipping"
}

Write-Host "[quick-check] mypy"
python -m mypy src

Write-Host "[quick-check] smoke tests"
python -m pytest -q -m "smoke and not slow"
