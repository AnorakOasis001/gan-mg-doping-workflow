# API usage

You can run the thermodynamic workflow directly from Python.

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

## Return types

- `AnalyzeResponse(csv_path, energy_col, result)`
- `SweepResponse(csv_path, energy_col, results)`

Each result item is a `ThermoResult` with:

- `temperature_K`
- `num_configurations`
- `mixing_energy_min_eV`
- `mixing_energy_avg_eV`
- `partition_function`
- `free_energy_mix_eV`

## Design property

The API is side-effect free: it does not print/log and does not write files.
