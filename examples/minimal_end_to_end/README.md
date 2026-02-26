# Minimal end-to-end example

This example runs the production CLI (`ganmg analyze`) against a committed golden fixture at **300 K** and writes outputs under this folder.

## Run

From repository root:

```bash
python examples/minimal_end_to_end/run_example.py
```

## Expected outputs

After a successful run, the following files exist:

- `examples/minimal_end_to_end/_outputs/results/tables/metrics.json`
- `examples/minimal_end_to_end/_outputs/results/tables/diagnostics_T300.json`
- `examples/minimal_end_to_end/_outputs/results/tables/demo_thermo.txt`

The script also prints absolute paths for `metrics.json` and diagnostics JSON.

## Notes

- Uses only repository fixture data: `data/golden/v1/inputs/realistic_small.csv`
- No external datasets or network access required for the analysis itself
