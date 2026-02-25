# API usage

You can run the thermodynamic workflow directly from Python without using the CLI.

```python
from pathlib import Path

from gan_mg import analyze_from_csv, sweep_from_csv

single = analyze_from_csv(Path("runs/demo/inputs/results.csv"), temperature_K=1000.0)
print(single.free_energy_mix_eV)

sweep = sweep_from_csv(
    Path("runs/demo/inputs/results.csv"),
    temperatures=[300.0, 600.0, 900.0, 1200.0],
)
for row in sweep:
    print(row.temperature_K, row.free_energy_mix_eV)
```

## Return types

- `analyze_from_csv(...)` returns a single `ThermoResult`
- `sweep_from_csv(...)` returns `list[ThermoResult]` sorted by temperature

`ThermoResult` includes:

- `temperature_K`
- `num_configurations`
- `mixing_energy_min_eV`
- `mixing_energy_avg_eV`
- `partition_function`
- `free_energy_mix_eV`

## Design property

The API layer is side-effect free: it does not print/log and does not write files.
