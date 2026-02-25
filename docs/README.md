# Docs README

## Python API

Use the library API directly when integrating thermodynamics into notebooks or scripts:

```python
from pathlib import Path

from gan_mg import analyze_from_csv, analyze_run, sweep_from_csv, sweep_run

single = analyze_from_csv(Path("runs/demo/inputs/results.csv"), temperature_K=900.0)
series = sweep_from_csv(Path("runs/demo/inputs/results.csv"), temperatures_K=[300.0, 600.0, 900.0])

run_single = analyze_run(Path("runs"), run_id="demo", temperature_K=900.0)
out_csv = sweep_run(Path("runs"), run_id="demo", temperatures_K=[300.0, 600.0, 900.0])
```
