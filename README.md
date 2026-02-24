# gan-mg-doping-workflow
A reproducible, step-by-step walkthrough of Mg-doped GaN simulations using Janus–MACE ML potentials — from environment setup to data, workflows, and results.

## Thermodynamics
For a set of mixing energies \(\Delta E_{mix}\), the canonical partition function is:

\[
Z = \sum_i \exp\left(-\frac{\Delta E_{mix,i}}{k_B T}\right)
\]

We report:

- `partition_function` from the Boltzmann sum above.
- `free_energy_mix_eV = -k_B T \ln Z`.

This is the canonical Helmholtz free energy form; for condensed solids where \(PV\) contributions are typically negligible, it is commonly used as a practical approximation to Gibbs free energy.
