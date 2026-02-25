#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# -------------------------
# Choose Python interpreter
# -------------------------
if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "./.venv/Scripts/python.exe" ]]; then
  PYTHON="./.venv/Scripts/python.exe"      # Windows venv
elif [[ -x "./.venv/bin/python" ]]; then
  PYTHON="./.venv/bin/python"              # Linux/macOS venv
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "[smoke] ERROR: No Python found."
  exit 1
fi

echo "[smoke] repo: $REPO_ROOT"
echo "[smoke] python: $PYTHON"
"$PYTHON" --version
"$PYTHON" -m pip --version

echo "[smoke] editable install (dev only)"
"$PYTHON" -m pip install -U pip
"$PYTHON" -m pip install -e ".[dev]" --no-build-isolation

# --------------------------------------------------
# Detect whether matplotlib (plot extra) is present
# --------------------------------------------------
if "$PYTHON" -c "import matplotlib" >/dev/null 2>&1; then
  HAS_PLOT="yes"
else
  HAS_PLOT="no"
fi

echo "[smoke] matplotlib available: $HAS_PLOT"

# Prefer module invocation (robust)
GANMG=( "$PYTHON" -m gan_mg.cli )

RUN_ID="smoke"
SEED="123"
T="1000"
TMIN="300"
TMAX="1200"
NT="10"

# -------------------------
# Clean previous run
# -------------------------
if [ -d "runs/$RUN_ID" ]; then
  echo "[smoke] removing existing runs/$RUN_ID"
  rm -rf "runs/$RUN_ID"
fi

echo "[smoke] doctor"
"${GANMG[@]}" doctor --run-dir "runs"

echo "[smoke] generate"
"${GANMG[@]}" generate --run-id "$RUN_ID" --seed "$SEED"

echo "[smoke] analyze"
"${GANMG[@]}" analyze --run-id "$RUN_ID" --T "$T"

echo "[smoke] sweep"
"${GANMG[@]}" sweep \
  --run-id "$RUN_ID" \
  --T-min "$TMIN" \
  --T-max "$TMAX" \
  --nT "$NT"

echo "[smoke] assert CSV outputs"
test -f "runs/$RUN_ID/inputs/results.csv"
test -f "runs/$RUN_ID/outputs/thermo_vs_T.csv"

# --------------------------------------------------
# Only require PNG if matplotlib is installed
# --------------------------------------------------
if [ "$HAS_PLOT" = "yes" ]; then
  echo "[smoke] checking plot output"
  test -f "runs/$RUN_ID/outputs/thermo_vs_T.png"
  echo "[smoke] plot file exists"
else
  echo "[smoke] matplotlib not installed — skipping PNG assertion"
fi

echo "[smoke] OK"