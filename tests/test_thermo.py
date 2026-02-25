import math

import pytest

from gan_mg.analysis.thermo import K_B_EV_PER_K, boltzmann_thermo_from_energies


def _probabilities(energies: list[float], temperature: float) -> list[float]:
    beta = 1.0 / (K_B_EV_PER_K * temperature)
    e_min = min(energies)
    weights = [math.exp(-beta * (e - e_min)) for e in energies]
    partition_function = sum(weights)
    return [w / partition_function for w in weights]


def test_energy_shift_invariance_of_probabilities_and_relative_mean_energy() -> None:
    energies = [-3.2, -3.0, -2.7, -2.4]
    shifted_energies = [e + 7.5 for e in energies]
    temperature = 700.0

    base = boltzmann_thermo_from_energies(energies, T=temperature)
    shifted = boltzmann_thermo_from_energies(shifted_energies, T=temperature)

    assert _probabilities(energies, temperature) == pytest.approx(
        _probabilities(shifted_energies, temperature)
    )
    assert (base.mixing_energy_avg_eV - base.mixing_energy_min_eV) == pytest.approx(shifted.mixing_energy_avg_eV - shifted.mixing_energy_min_eV)


@pytest.mark.parametrize("temperature", [298.15, 600.0, 1200.0])
def test_partition_function_is_positive_and_finite(temperature: float) -> None:
    energies = [-1.4, -1.2, -0.9, -0.2]

    result = boltzmann_thermo_from_energies(energies, T=temperature)

    assert result.partition_function > 0.0
    assert math.isfinite(result.partition_function)


def test_free_energy_matches_analytic_formula() -> None:
    energies = [-4.0, -3.6, -3.55, -3.1]
    temperature = 450.0

    result = boltzmann_thermo_from_energies(energies, T=temperature)
    expected = -K_B_EV_PER_K * temperature * math.log(result.partition_function) + result.mixing_energy_min_eV

    assert result.free_energy_mix_eV == pytest.approx(expected)
