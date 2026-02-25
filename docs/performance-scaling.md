# Performance & scaling

## Benchmark command

Use the built-in benchmark command to time thermodynamic sweeps on synthetic energies:

```bash
ganmg bench thermo --n 1000 --nT 50
```

This writes JSON output to:

- `outputs/bench.json`

The benchmark captures:

- benchmark parameters (`n`, `nT`, temperature range, seed)
- runtime timings
- a small result snapshot for sanity checks
- environment metadata (Python, platform, processor, package version)

## Expected runtime

Default settings are intentionally lightweight (`n=1000`, `nT=50`) and are expected to complete in under 5 seconds on typical laptops and CI runners.

## CI policy

Benchmarking is a manual performance check and is **not** part of the default CI test path.
