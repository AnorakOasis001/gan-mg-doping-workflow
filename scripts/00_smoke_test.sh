#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip setuptools wheel
python -m pip install -e . --no-build-isolation

ganmg --help
ganmg generate --run-id smoke --n 5 --seed 7
ganmg analyze  --run-id smoke --T 1000
ganmg sweep    --run-id smoke --nT 5