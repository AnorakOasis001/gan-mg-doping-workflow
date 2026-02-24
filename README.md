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
```

If matplotlib is not installed, omit `--plot`.

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
