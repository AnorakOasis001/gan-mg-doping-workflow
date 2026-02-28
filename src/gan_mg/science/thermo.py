from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from gan_mg.science.constants import K_B_EV_PER_K
from gan_mg.science.streaming import LogSumExpAccumulator

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThermoResult:
    temperature_K: float
    num_configurations: int
    mixing_energy_min_eV: float
    mixing_energy_avg_eV: float
    partition_function: float
    free_energy_mix_eV: float


class ThermoSweepRow(TypedDict):
    temperature_K: float
    num_configurations: int
    mixing_energy_min_eV: float
    mixing_energy_avg_eV: float
    partition_function: float
    free_energy_mix_eV: float


@dataclass(frozen=True)
class ThermoDiagnostics:
    temperature_K: float
    num_configurations: int
    expected_energy_eV: float
    energy_variance_eV2: float
    energy_std_eV: float
    p_min: float
    effective_sample_size: float
    logZ_shifted: float
    logZ_absolute: float
    notes: list[str]


def log_partition_function(delta_e_eV: NDArray[np.float64], temperature_K: float) -> float:
    """
    Compute log(partition function) using a numerically stable log-sum-exp evaluation.

    The partition function is defined for delta energies as:
        Z = sum_i exp(-delta_e_i / (kB * T))
    and this function returns log(Z).
    """
    if temperature_K <= 0:
        raise ValueError("temperature_K must be > 0.")

    if delta_e_eV.size == 0:
        raise ValueError("delta_e_eV must be non-empty.")

    if not np.all(np.isfinite(delta_e_eV)):
        raise ValueError("delta_e_eV must contain only finite values.")

    x = -delta_e_eV / (K_B_EV_PER_K * temperature_K)
    m = float(np.max(x))
    log_z = m + float(np.log(np.sum(np.exp(x - m))))
    return log_z


def free_energy_from_logZ(logZ: float, temperature_K: float) -> float:
    if temperature_K <= 0:
        raise ValueError("temperature_K must be > 0.")
    return -K_B_EV_PER_K * temperature_K * logZ


def boltzmann_thermo_from_energies(energies: list[float], T: float) -> ThermoResult:
    """
    Compute basic canonical thermodynamics from a list of energies (eV) at temperature T (K).
    Uses a numerically-stable energy shift by Emin.

    Returns:
      - partition_function = sum_i exp(-beta*(Ei - Emin))
      - mixing_energy_avg_eV in eV
      - free_energy_mix_eV in eV, where
        free_energy_mix_eV = Emin - (1/beta)*ln(partition_function)
    """
    if T <= 0:
        raise ValueError("Temperature T must be > 0.")

    if not energies:
        raise ValueError("energies must be non-empty.")

    emin = min(energies)

    delta_e = np.asarray([e - emin for e in energies], dtype=float)
    log_z = log_partition_function(delta_e_eV=delta_e, temperature_K=T)

    max_log_float = math.log(float(np.finfo(float).max))
    if log_z <= max_log_float:
        partition_function = math.exp(log_z)
    else:
        partition_function = float("inf")
        LOGGER.warning(
            "Partition function overflow for T=%s K (logZ=%s); storing partition_function=inf.",
            T,
            log_z,
        )

    energies_np = np.asarray(energies, dtype=float)
    mixing_energy_avg_eV = float(np.mean(energies_np))

    free_energy_mix_eV = emin + free_energy_from_logZ(logZ=log_z, temperature_K=T)

    return ThermoResult(
        temperature_K=T,
        num_configurations=len(energies),
        mixing_energy_min_eV=emin,
        mixing_energy_avg_eV=mixing_energy_avg_eV,
        partition_function=partition_function,
        free_energy_mix_eV=free_energy_mix_eV,
    )


def boltzmann_diagnostics_from_energies(energies: list[float], T: float) -> ThermoDiagnostics:
    if T <= 0:
        raise ValueError("Temperature T must be > 0.")
    if not energies:
        raise ValueError("energies must be non-empty.")

    energies_np = np.asarray(energies, dtype=float)
    if not np.all(np.isfinite(energies_np)):
        raise ValueError("energies must contain only finite values.")

    beta = 1.0 / (K_B_EV_PER_K * T)
    emin = float(np.min(energies_np))
    delta = energies_np - emin
    x = -beta * delta

    log_z_shifted = log_partition_function(delta_e_eV=delta, temperature_K=T)
    log_z_absolute = log_z_shifted - beta * emin

    lse_w2 = LogSumExpAccumulator()
    lse_w2.update(2.0 * x)
    log_sum_w2 = lse_w2.logsumexp()

    p_min = math.exp(-log_z_shifted)
    effective_sample_size = math.exp(2.0 * log_z_shifted - log_sum_w2)

    if np.any(delta > 0.0):
        m = float(np.max(x))
        exp_shift = np.exp(x - m)
        sum_delta_exp = float(np.sum(delta * exp_shift))
        sum_delta2_exp = float(np.sum((delta**2) * exp_shift))

        expected_delta = 0.0 if sum_delta_exp == 0.0 else math.exp(m + math.log(sum_delta_exp) - log_z_shifted)
        expected_delta2 = (
            0.0
            if sum_delta2_exp == 0.0
            else math.exp(m + math.log(sum_delta2_exp) - log_z_shifted)
        )
    else:
        expected_delta = 0.0
        expected_delta2 = 0.0

    variance_e = max(expected_delta2 - expected_delta**2, 0.0)
    std_e = math.sqrt(variance_e)

    notes: list[str] = []
    max_log_float = math.log(float(np.finfo(float).max))
    if log_z_shifted > max_log_float:
        notes.append("partition_function_overflow")

    return ThermoDiagnostics(
        temperature_K=T,
        num_configurations=int(energies_np.size),
        expected_energy_eV=emin + expected_delta,
        energy_variance_eV2=variance_e,
        energy_std_eV=std_e,
        p_min=p_min,
        effective_sample_size=effective_sample_size,
        logZ_shifted=log_z_shifted,
        logZ_absolute=log_z_absolute,
        notes=notes,
    )
