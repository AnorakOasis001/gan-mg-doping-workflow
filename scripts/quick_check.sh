#!/usr/bin/env bash
set -euo pipefail

echo "[quick-check] ruff"
if python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('ruff') else 1)"; then
  python -m ruff check src tests
else
  echo "ruff not installed; skipping"
fi

echo "[quick-check] mypy"
python -m mypy src

echo "[quick-check] smoke tests"
if [ -d tests/smoke ]; then
  python -m pytest tests/smoke -q -m "not slow"
else
  bash scripts/00_smoke_test.sh
fi
