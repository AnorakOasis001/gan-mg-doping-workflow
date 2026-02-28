# Figure gallery

## Quickstart demo

Run a deterministic, fast end-to-end showcase with one command:

```bash
pip install -e ".[dev]"
ganmg demo
```

This creates:

- `runs/demo/derived/gibbs_summary.csv`
- `runs/demo/derived/mechanism_crossover.csv`
- `runs/demo/derived/phase_map.csv`
- `runs/demo/derived/phase_boundary.csv`
- `reports/demo/` (shareable report bundle + README summary)

Use `ganmg demo --plot` to also emit figure assets (for example `runs/demo/figures/phase_map.png`) when matplotlib is installed.

## Regenerate figure assets from an existing run

Use the plotting CLI to regenerate key figures from an existing run:

```bash
ganmg plot --run-id demo --kind thermo
```

The command writes figures into the run-local gallery directory:

- `runs/demo/figures/thermo_vs_T.png`

Example thermodynamic figure:

![Example thermo_vs_T figure](assets/thermo-example.svg)

For a fully self-contained, synthetic-data walkthrough (public API only), see:

- `docs/examples/thermo_analysis_synthetic.ipynb`
