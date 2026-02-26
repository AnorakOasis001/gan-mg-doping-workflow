# Minimal end-to-end workflow (golden fixture, 2-minute demo)

## What this demonstrates

This workflow runs the repository's real `ganmg analyze` CLI command on a committed golden fixture (`data/golden/v1/inputs/realistic_small.csv`) at **T = 300 K** and writes reproducible outputs without any external dataset downloads.

It is designed as a fast portfolio demo: a hiring manager can run one command and inspect concrete artifacts (`metrics.json`, `diagnostics_T300.json`, and `demo_thermo.txt`).

## Prerequisites

- Python 3.10+
- Installed project dependencies (at minimum: `pandas`)
- Run commands from the repository root

## Quick run (Linux/macOS)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python examples/minimal_end_to_end/run_example.py
```

## Quick run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
python .\examples\minimal_end_to_end\run_example.py
```

## Inputs used

- `data/golden/v1/inputs/realistic_small.csv`

This file is committed to the repository and is part of the golden-fixture set.

## Produced files and locations

The example script runs CLI analysis from inside `examples/minimal_end_to_end/_outputs/`, so CLI default output locations resolve under that folder:

- `examples/minimal_end_to_end/_outputs/results/tables/metrics.json`
- `examples/minimal_end_to_end/_outputs/results/tables/diagnostics_T300.json` (because `--diagnostics` is enabled)
- `examples/minimal_end_to_end/_outputs/results/tables/demo_thermo.txt`

These paths are printed by the script after the run completes.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'gan_mg'`**
  - Ensure your virtual environment is active and run `python -m pip install -e .` from repo root.
- **`ImportError` / missing dependency errors (for example `pandas`)**
  - Reinstall dependencies: `python -m pip install -e .`.
- **Windows path/activation issues**
  - Use PowerShell command `\.venv\Scripts\Activate.ps1` (or Command Prompt `\.venv\Scripts\activate.bat`).
  - Use backslashes in local paths when invoking commands manually in `cmd.exe`/PowerShell.
