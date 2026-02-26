# Thermodynamics benchmark harness

This benchmark compares two implementations of thermodynamic analysis from CSV input:

- **In-memory:** `boltzmann_thermo_from_csv(...)`
- **Streaming:** `thermo_from_csv_streaming(...)`

It is intended for lightweight, reproducible scaling checks (not performance gating).

## What is benchmarked

For each requested dataset size (`--rows`), the runner:

1. Generates a synthetic CSV in a temporary directory with required schema:
   `structure_id, mechanism, energy_eV`
2. Times in-memory thermodynamics.
3. Times streaming thermodynamics.
4. Verifies numerical parity for stable fields:
   - `temperature_K`
   - `num_configurations`
   - `mixing_energy_min_eV`
   - `mixing_energy_avg_eV`
   - `partition_function`
   - `free_energy_mix_eV`

Parity uses `rtol=1e-12` and `atol=1e-15`.

## Methodology

- **Synthetic data generation:** deterministic pseudo-random energies from a Gaussian distribution.
- **Energy distribution:** `N(loc=-1.0 eV, scale=0.25 eV)`.
- **Temperature:** configurable via `--temperature` (default: `300 K`).
- **Chunk size:** configurable via `--chunksize` (default: `200000`).
- **Seed:** configurable via `--seed` (default: `0`) to keep data reproducible.

Optional memory tracking can be enabled via `--measure-memory` using stdlib `tracemalloc`.
This is useful for relative comparisons, but absolute values may differ across Python versions and platforms.

## Run locally

```bash
python scripts/benchmark_thermo.py \
  --rows 10000 100000 1000000 \
  --temperature 300 \
  --seed 0 \
  --outdir results/benchmarks \
  --chunksize 200000
```

Generate a runtime plot at the same time:

```bash
python scripts/benchmark_thermo.py \
  --rows 10000 100000 1000000 \
  --outdir results/benchmarks \
  --plot
```

## Outputs

- Summary (JSONL): `results/benchmarks/thermo_benchmarks.jsonl`
- Optional plot: `results/benchmarks/runtime_vs_rows.png`

Each JSONL record includes:

- `rows`
- `temperature_K`
- `chunksize`
- `time_in_memory_s`
- `time_streaming_s`
- `speedup` (`time_in_memory_s / time_streaming_s`; values >1 mean streaming was faster)
- `python_version`
- `platform`
- `parity_passed`
- optional memory fields when `--measure-memory` is used

## Interpreting results

- **Speedup** indicates runtime ratio of in-memory to streaming.
- Streaming is expected to have better memory behavior for large inputs, especially when
  in-memory loading becomes a bottleneck.
- Very small datasets may show little difference due to fixed overheads.

## CI policy

CI includes a non-gating smoke benchmark run to ensure the harness remains functional and parity checks pass.
Benchmark *numbers* are not used as a pass/fail threshold to avoid flaky CI behavior.
