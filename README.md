---

# gan-mg-doping-workflow

Reproducible Scientific ML workflow for thermodynamic analysis of Mg-doped GaN using structured, CLI-driven pipelines.

This repository demonstrates the evolution of a research codebase into a reproducible, testable, production-style scientific software project suitable for AI-for-Science and computational materials workflows.

---

## Overview

`gan-mg-doping-workflow` provides:

* Deterministic dataset generation
* Canonical Boltzmann thermodynamic analysis
* Temperature sweep utilities
* Run-folder reproducibility architecture
* Cross-platform CLI workflow (Windows + Linux)
* Editable install (no `PYTHONPATH` hacks)
* Structured repository layout (`src/` pattern)

The project is intentionally structured to reflect research software engineering best practices.

---

## Thermodynamics

The workflow computes canonical Boltzmann statistics over mixing energies:

\[
Z = \sum_i \exp\left(-\frac{\Delta E_{\mathrm{mix},i}}{k_B T}\right)
\]

with exported naming:

* `temperature_K`
* `num_configurations`
* `mixing_energy_min_eV`
* `mixing_energy_avg_eV`
* `partition_function`
* `free_energy_mix_eV`

The free energy is reported as:

\[
\mathrm{free\_energy\_mix\_eV} = -k_B T \ln Z
\]

This is the canonical Helmholtz free energy of mixing; for solids, the `PV` term is negligible, so it closely approximates Gibbs free energy.

---

## Installation

Python ≥ 3.10 required.

We strongly recommend using a virtual environment.

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]" --no-build-isolation
```

Editable install is required.
Do **not** run modules directly from `src/`.

---

## Quickstart (Codex-Friendly Smoke Test)

This reproduces a deterministic workflow end-to-end.

```bash
# Install (editable)
python -m pip install -U pip
python -m pip install -e ".[dev]" --no-build-isolation

# Generate synthetic dataset
ganmg generate --run-id smoke --seed 123

# Analyze at single temperature
ganmg analyze --run-id smoke --temperature 1000

# Sweep across temperature grid
ganmg sweep --run-id smoke --tmin 300 --tmax 1200 --tstep 100 --plot
```

Expected output structure:

```
runs/smoke/
  inputs/results.csv
  outputs/
    thermo_T1000.txt
    thermo_vs_T.csv
    thermo_vs_T.png
    metrics.json
```

If matplotlib is not installed, omit `--plot`.

---

## Models

Generation now supports pluggable energy backends via `--model`:

* `demo` (default): seeded pseudo-random demo energies (current baseline behavior).
* `toy`: deterministic toy pair-potential penalty model for backend-swappable integration tests.

Example commands:

```bash
# Default demo backend
ganmg generate --run-id demo-backend --seed 11 --model demo

# Deterministic toy backend
ganmg generate --run-id toy-backend --seed 11 --model toy

# Thermodynamics pipeline works with either model
ganmg analyze --run-id toy-backend --T 1000
ganmg sweep --run-id toy-backend --nT 7
```

---


## Importing external results (HPC)

Use the import command to bring externally computed energies (for example from HPC relaxations) into a run directory:

```bash
ganmg import --run-id <id> --run-dir <dir> --results <path>
```

Supported input formats:

* `results.csv`: must include `structure_id`, `mechanism`, and `energy_eV` columns.
* `extxyz`/`xyz`: parsed in best-effort mode from per-structure comment-line fields such as `energy=...` or `energy_eV=...`.

What gets written:

* Canonical analysis input at `runs/<id>/inputs/results.csv`
* A stored source copy at `runs/<id>/inputs/external_results.<ext>`
* Import metadata at `runs/<id>/inputs/import_metadata.json` with source path and UTC timestamp

If schema validation fails, the CLI exits with a clear `Input validation error:` message describing the issue.

---

## CLI Commands

### Generate

Creates a deterministic synthetic dataset.

```bash
ganmg generate --run-id <id> --seed <int>
```

### Analyze

Computes Boltzmann thermodynamics at a single temperature.

```bash
ganmg analyze --run-id <id> --temperature <K>
```

### Sweep

Evaluates thermodynamic quantities across a temperature grid.

```bash
ganmg sweep \
  --run-id <id> \
  --tmin 300 \
  --tmax 1200 \
  --tstep 100 \
  [--plot]
```


### Bench

Run a lightweight synthetic performance check for thermodynamics sweeps.

```bash
ganmg bench thermo --n 1000 --nT 50
```

Writes `outputs/bench.json` with timings and environment metadata.

### Runs

Inspect run history and show metadata + latest standardized metrics summary.

```bash
ganmg runs list
ganmg runs latest
ganmg runs show --run-id <id>
```

Example `runs/<id>/outputs/metrics.json`:

```json
{
  "command": "analyze",
  "temperature_K": 1000.0,
  "num_configurations": 10,
  "mixing_energy_min_eV": -2.053,
  "mixing_energy_avg_eV": -1.998,
  "partition_function": 2.041,
  "free_energy_mix_eV": -2.113,
  "reproducibility_hash": "<sha256>",
  "timings": {
    "runtime_s": 0.012
  }
}
```

---

## Python API

You can use the thermodynamics workflow directly from Python without invoking the CLI.

```python
from gan_mg import analyze_from_csv, sweep_from_csv

single = analyze_from_csv("runs/demo/inputs/results.csv", temperature_K=1000.0)
print(single.result.free_energy_mix_eV)

sweep = sweep_from_csv(
    "runs/demo/inputs/results.csv",
    temperatures_K=[300.0, 600.0, 900.0, 1200.0],
)
for row in sweep.results:
    print(row.temperature_K, row.free_energy_mix_eV)
```

API return types are structured dataclasses:

* `AnalyzeResponse(csv_path, energy_col, result)`
* `SweepResponse(csv_path, energy_col, results)`

Each `result`/`results` item is a `ThermoResult` dataclass containing:

* `temperature_K`
* `num_configurations`
* `mixing_energy_min_eV`
* `mixing_energy_avg_eV`
* `partition_function`
* `free_energy_mix_eV`

The API functions are side-effect free: they do not print/log or write output files.

---

## Architecture

High-level execution flow:

```
ganmg (CLI)
  ├─ generate → runs/<id>/inputs/results.csv
  ├─ analyze  → runs/<id>/outputs/thermo_T*.txt
  └─ sweep    → runs/<id>/outputs/thermo_vs_T.(csv/png)
```

Repository layout:

```
gan-mg-doping-workflow/
│
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
│
├── scripts/
│   ├── 00_smoke_test.sh
│   └── 00_smoke_test.ps1
│
├── src/
│   └── gan_mg/
│       ├── __init__.py
│       ├── cli.py
│       ├── run.py
│       └── analysis/
│           ├── thermo.py
│           ├── types.py
│           └── io.py
│
└── runs/
    └── <run-id>/
```

---

## Reproducibility Guarantees

Each workflow is isolated inside:

```
runs/<run-id>/
```

Key properties:

* Deterministic dataset generation via `--seed`
* No global state
* Stable thermodynamic result schema
* Numerically stable shifted partition function
* Cross-platform execution scripts
* CI-ready structure

---

## Cross-Platform Smoke Scripts

Linux / macOS:

```bash
bash scripts/00_smoke_test.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/00_smoke_test.ps1
```

Both scripts execute the same deterministic workflow.

---


## Documentation

The project includes an MkDocs documentation site under `docs/` with pages for installation, CLI usage, API usage, thermodynamics math, and reproducibility philosophy.

Build locally:

```bash
python -m pip install -e ".[docs]" --no-build-isolation
mkdocs serve
```

GitHub Pages deployment is automated through `.github/workflows/docs.yml` on pushes to `main`.

---

## Development

Install with development extras:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Continuous Integration runs automatically on push and pull requests.

---

## Release process

This project uses a lightweight semantic-ish versioning flow.

Version is defined in exactly one place:

* `pyproject.toml` → `[project].version`

Version bump guidance:

* `PATCH` (`x.y.Z`): bug fixes, docs-only updates, refactors that do not change public behavior.
* `MINOR` (`x.Y.z`): new backwards-compatible CLI options, analyses, or workflow features.
* `MAJOR` (`X.y.z`): breaking changes to CLI contract, output schema, or package API.

Release checklist:

1. Update `pyproject.toml` version.
2. Add a new dated section in `CHANGELOG.md` summarizing Added/Changed/Fixed.
3. Run tests (`pytest`) and smoke workflow checks.
4. Create a release commit: `chore(release): v<version>`.
5. Tag the commit: `git tag v<version>` and push branch + tags.

For pre-1.0 development, breaking changes may still occur, but we still follow the same bump intent to keep release history clear.

---

## Project Goals

This repository demonstrates:

* Clean scientific workflow engineering
* Reproducible CLI-driven computational pipelines
* Numerically stable thermodynamic modeling
* Proper Git discipline and PR workflow
* Transition from research scripts to structured software

Target audience:

* Scientific ML / AI-for-Science teams
* Computational materials science groups
* Research software engineering roles
* Applied ML engineering (physics-informed systems)

---

## Future Roadmap

* Expanded unit test coverage
* CI hardening
* Validation against real GaN datasets
* Config-driven workflows (TOML/YAML)
* Structured logging
* Static type enforcement

---

## License

See `LICENSE`.

---
