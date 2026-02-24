#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# -------------------------
# Choose a Python interpreter that works on:
# - Linux/macOS
# - Windows Git Bash (MINGW)
# Prefers the repo-local .venv if present.
# You can also override by running:
#   PYTHON=/path/to/python bash scripts/00_smoke_test.sh
# -------------------------
if [[ -n "${PYTHON:-}" ]]; then
  : # use user-provided PYTHON
elif [[ -x "./.venv/Scripts/python.exe" ]]; then
  PYTHON="./.venv/Scripts/python.exe"          # Windows venv
elif [[ -x "./.venv/bin/python" ]]; then
  PYTHON="./.venv/bin/python"                  # Linux/macOS venv
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "[smoke] ERROR: No Python found."
  echo "        If you're on Windows Git Bash, make sure you created/activated .venv,"
  echo "        or run with: PYTHON=./.venv/Scripts/python.exe bash scripts/00_smoke_test.sh"
  exit 1
fi

echo "[smoke] repo: $REPO_ROOT"
echo "[smoke] python: $PYTHON"
"$PYTHON" --version
"$PYTHON" -m pip --version

echo "[smoke] editable install"
"$PYTHON" -m pip install -U pip
"$PYTHON" -m pip install -e ".[dev]" --no-build-isolation

# Prefer module invocation (more reliable than relying on ganmg being on PATH in Git Bash)
GANMG=( "$PYTHON" -m gan_mg.cli )

RUN_ID="smoke"
SEED="123"
T="1000"
TMIN="300"
TMAX="1200"
NT="10"

# Clean previous smoke run (idempotent)
if [ -d "runs/$RUN_ID" ]; then
  echo "[smoke] removing existing runs/$RUN_ID"
  rm -rf "runs/$RUN_ID"
fi

echo "[smoke] generate"
"${GANMG[@]}" generate --run-id "$RUN_ID" --seed "$SEED"

echo "[smoke] analyze"
"${GANMG[@]}" analyze --run-id "$RUN_ID" --T "$T"

echo "[smoke] sweep"
"${GANMG[@]}" sweep --run-id "$RUN_ID" --T-min "$TMIN" --T-max "$TMAX" --nT "$NT"

echo "[smoke] assert outputs"
test -f "runs/$RUN_ID/inputs/results.csv"
test -f "runs/$RUN_ID/outputs/thermo_vs_T.csv"

if test -f "runs/$RUN_ID/outputs/thermo_vs_T.png"; then
  echo "[smoke] found thermo_vs_T.png"
else
  echo "[smoke] thermo_vs_T.png not found (ok if plotting not implemented)"
fi

echo "[smoke] OK"