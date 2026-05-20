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
if (Test-Path "tests/smoke") {
  python -m pytest tests/smoke -q -m "not slow"
} else {
  powershell -ExecutionPolicy Bypass -File scripts\00_smoke_test.ps1
}
