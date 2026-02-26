# Reproducibility tutorial

This tutorial gives a copy-paste path from fresh install to reproducible outputs.

## Fresh install

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install .
```

## Minimal end-to-end CLI run

```bash
ganmg generate --run-dir runs --run-id repro-demo --n 8 --seed 11 --model demo
ganmg analyze --run-dir runs --run-id repro-demo --T 1000 --validate-output
```

Outputs are written under `runs/repro-demo/`.

## Config-driven run (PR10)

```bash
cat > run_config.toml <<'TOML'
schema_version = 1

[analyze]
csv = "data/golden/v1/inputs/realistic_small.csv"
T = 300.0
diagnostics = true
output_root = "repro_out"
TOML

ganmg --config run_config.toml
```

Outputs are written to `repro_out/results/tables/`.

## Run manifest

`run_manifest.json` records the reproducibility contract for config-driven runs.

- Path: `<output_root>/results/tables/run_manifest.json`
- Includes config hash (`config_sha256`), input hash (`input_csv_sha256`), git commit, package version, and Python/platform metadata.
- For papers/reports, cite: `config_sha256`, `input_csv_sha256`, `git_commit`, and `package_version` from that file.

## Output contract validation

CLI path during analyze:

```bash
ganmg analyze --run-dir runs --run-id repro-demo --T 1000 --diagnostics --validate-output
```

Python/module path:

```bash
python -c "from pathlib import Path; from gan_mg.validation import validate_output_file; validate_output_file(Path('runs/repro-demo/outputs/metrics.json'), kind='metrics')"
```

## Streaming parity

Run parity tests to verify chunked/streaming mode matches non-streaming mode within tolerances:

```bash
python -m pytest -q tests/test_streaming_parity.py
```

Interpretation: pass means streaming and non-streaming outputs are numerically aligned within the test tolerances.

## Golden fixtures

Regenerate deterministic fixtures:

```bash
python scripts/generate_golden.py
```

Verify changes are expected:

```bash
git diff -- data/golden/v1
python -m pytest -q tests/test_golden_regression.py tests/test_output_validation.py
```
