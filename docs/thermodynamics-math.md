# Thermodynamics math

The workflow computes canonical Boltzmann statistics for mixing energies.

## Partition function

\[
Z = \sum_i \exp\left(-\frac{\Delta E_{\mathrm{mix}, i}}{k_B T}\right)
\]

where:

- \(\Delta E_{\mathrm{mix}, i}\) is the mixing energy of configuration \(i\)
- \(k_B\) is the Boltzmann constant
- \(T\) is temperature in Kelvin

## Free energy of mixing

\[
F_{\mathrm{mix}} = -k_B T \ln Z
\]

In exported outputs this is recorded as `free_energy_mix_eV`.

## Numerical stability

The implementation uses a shifted exponentiation strategy to keep the partition function stable for realistic temperature ranges and energy spreads.
