# CLI usage

The project exposes the `ganmg` command-line interface.

## Generate synthetic structures

```bash
ganmg generate --run-id demo --seed 123
```

Or select an energy backend explicitly:

```bash
# Seeded pseudo-random baseline
ganmg generate --run-id demo-backend --seed 11 --model demo

# Deterministic toy pair-energy backend
ganmg generate --run-id toy-backend --seed 11 --model toy
```

Writes deterministic synthetic input data to:

- `runs/<run-id>/inputs/results.csv`
- `runs/<run-id>/run.json` (includes the selected `model`)

## Analyze at one temperature

```bash
ganmg analyze --run-id demo --T 1000
```

Writes a thermodynamic report to:

- `runs/demo/outputs/thermo_T1000.txt`

## Sweep over a temperature grid

```bash
ganmg sweep --run-id demo --tmin 300 --tmax 1200 --tstep 100 --plot
```

Writes:

- `runs/demo/outputs/thermo_vs_T.csv`
- `runs/demo/outputs/thermo_vs_T.png` (when plotting is enabled and matplotlib is installed)

## Typical smoke flow

```bash
ganmg generate --run-id smoke --seed 123 --model toy
ganmg analyze --run-id smoke --T 1000
ganmg sweep --run-id smoke --tmin 300 --tmax 1200 --tstep 100
```

## Benchmark thermodynamics sweep

```bash
ganmg bench thermo --n 1000 --nT 50
```

Writes:

- `outputs/bench.json`

## Regenerate run figures

```bash
ganmg plot --run-id demo --kind thermo
```

Writes:

- `runs/demo/figures/thermo_vs_T.png`

## Inspect run inventory

List discovered runs and compact artifact availability:

```bash
ganmg runs list --runs-dir runs --limit 5
```

Print the latest run id (or structured metadata with `--json`):

```bash
ganmg runs latest --runs-dir runs
```
