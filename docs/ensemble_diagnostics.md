# Canonical ensemble diagnostics

The `analyze` command can optionally emit a diagnostics JSON with additional canonical-ensemble observables computed from Boltzmann weights.

## Observables

For energies \(E_i\), inverse temperature \(\beta = 1/(k_B T)\), and normalized weights
\(w_i = \exp(-\beta E_i) / Z\):

- Expected energy: \(\langle E \rangle = \sum_i E_i w_i\)
- Energy variance: \(\mathrm{Var}(E) = \langle E^2 \rangle - \langle E \rangle^2\)
- Energy standard deviation: \(\sqrt{\mathrm{Var}(E)}\)
- Minimum-state probability: `p_min`
- Effective sample size (ESS):
  \[
  \mathrm{ESS} = \frac{1}{\sum_i w_i^2}
  \]

The implementation uses numerically stable log-sum-exp expressions and a shifted-energy formulation.

## CLI usage

```bash
ganmg analyze --run-dir runs --run-id <RUN_ID> --T 1000 --diagnostics
```

Streaming mode is also supported:

```bash
ganmg analyze --csv path/to/results.csv --T 1000 --chunksize 200000 --diagnostics
```

## Output file

The diagnostics file is additional output and does **not** change existing output formats.

- Run mode: `<run>/outputs/diagnostics_T{int(T)}.json`
- CSV mode: `results/tables/diagnostics_T{int(T)}.json`

Existing `metrics.json` and `thermo_T*.txt` files remain unchanged.
