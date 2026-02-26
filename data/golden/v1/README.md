# Golden Thermodynamics Fixtures (v1)

This directory contains deterministic CSV fixtures for thermodynamic analysis tests.

## Input format

All CSV files in `inputs/` use the exact required columns:

- `structure_id`
- `mechanism`
- `energy_eV`

Energies are expressed in **electronvolts (eV)**.

## Directory layout

- `inputs/`: tiny deterministic CSV fixtures intended for CI and contract-style tests.
- `expected/`: placeholder for generated expected outputs.

Expected outputs are generated later by `scripts/generate_golden.py` (to be added in a follow-up change).
