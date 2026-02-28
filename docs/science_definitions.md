# Science definitions and units

This page defines the key thermodynamics quantities used in `ganmg` outputs and figures.

## Energy definitions

- `E_total` (eV): total supercell energy from the source calculation.
- `E_ref` (eV): reference energy predicted by the selected reference model for the same composition.
- `E_mix` (eV): mixing energy,
  \[
  E_{mix} = E_{total} - E_{ref}
  \]
- `dE` (eV): relative mixing energy within a `(mechanism_code, x_mg_cation)` group,
  \[
  dE_i = E_{mix,i} - \min_j E_{mix,j}
  \]

## Thermodynamic ensemble quantities

For each mechanism/doping group and temperature `T`:

- `Z(T)` (dimensionless): partition function built from `dE` values.
- `logZ` (dimensionless): natural log of `Z(T)` (reported directly for numerical stability).
- `Gmix` (eV): free energy of mixing,
  \[
  G_{mix}(T) = E_{mix,min} - k_B T \log Z
  \]
- `<E>` (eV): Boltzmann-weighted average mixing energy (`mixing_energy_avg_eV`).
- `Smix` (eV/K): optional derived entropy of mixing from
  \[
  S_{mix} = (\langle E \rangle - G_{mix})/T
  \]
  if computed in downstream analysis.

## Crossover definition

`mechanism_crossover.csv` compares `vn` and `mgi` at equal `(x_mg_cation, T_K)`:

- `delta_free_energy_eV = Gmix(vn) - Gmix(mgi)`.
- Preferred mechanism is:
  - `vn` when `delta_free_energy_eV < 0`
  - `mgi` otherwise.

The zero contour (`ΔG = 0`) is the mechanism crossover boundary.

## Normalization conventions

Unless noted otherwise, base quantities are per supercell:

- `*_eV`: per supercell.
- `*_eV_per_atom`: per total atom count (`site_count_total`).
- `*_eV_per_cation`: per cation count (`mg_count + ga_count`).

`x_mg_cation` is the cation-site fraction, and `doping_level_percent = 100 * x_mg_cation`.
