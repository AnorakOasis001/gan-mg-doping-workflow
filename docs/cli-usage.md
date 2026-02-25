# CLI usage

The project exposes the `ganmg` command-line interface.

## Generate synthetic structures

```bash
ganmg generate --run-id demo --seed 123
```

Writes deterministic synthetic input data to:

- `runs/demo/inputs/results.csv`

## Analyze at one temperature

```bash
ganmg analyze --run-id demo --temperature 1000
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
ganmg generate --run-id smoke --seed 123
ganmg analyze --run-id smoke --temperature 1000
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
