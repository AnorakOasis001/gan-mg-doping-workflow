from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

K_B_EV_PER_K = 8.617333262e-5  # Boltzmann constant in eV/K


@dataclass(frozen=True)
class ThermoResult:
    T: float
    n: int
    emin: float
    Z_tilde: float
    E_avg: float
    F: float


def read_energies_csv(csv_path: Path, energy_col: str = "energy_eV") -> list[float]:
    """
    Read energies (eV) from a CSV file.
    Expects a column named `energy_col` (default: energy_eV).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    energies: list[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or energy_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain column '{energy_col}'. Found: {reader.fieldnames}")

        for row in reader:
            energies.append(float(row[energy_col]))

    if not energies:
        raise ValueError("No energies found in CSV.")
    return energies


def boltzmann_thermo_from_energies(energies: list[float], T: float) -> ThermoResult:
    """
    Compute basic canonical thermodynamics from a list of energies (eV) at temperature T (K).
    Uses a numerically-stable energy shift by Emin.

    Returns:
      - Z_tilde = sum_i exp(-beta*(Ei - Emin))
      - <E> in eV
      - F in eV, where F = Emin - (1/beta)*ln(Z_tilde)
    """
    if T <= 0:
        raise ValueError("Temperature T must be > 0.")

    beta = 1.0 / (K_B_EV_PER_K * T)
    emin = min(energies)

    shifted = [e - emin for e in energies]
    weights = [math.exp(-beta * e) for e in shifted]
    Z_tilde = sum(weights)

    probs = [w / Z_tilde for w in weights]
    E_avg = sum(p * e for p, e in zip(probs, energies))

    F = emin - (1.0 / beta) * math.log(Z_tilde)

    return ThermoResult(T=T, n=len(energies), emin=emin, Z_tilde=Z_tilde, E_avg=E_avg, F=F)


def boltzmann_thermo_from_csv(csv_path: Path, T: float, energy_col: str = "energy_eV") -> ThermoResult:
    energies = read_energies_csv(csv_path, energy_col=energy_col)
    return boltzmann_thermo_from_energies(energies, T=T)


def write_thermo_txt(result: ThermoResult, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(
        f"T(K) = {result.T}\n"
        f"N    = {result.n}\n"
        f"Emin (eV) = {result.emin:.6f}\n"
        f"Z_tilde   = {result.Z_tilde:.6e}\n"
        f"<E> (eV)  = {result.E_avg:.6f}\n"
        f"F   (eV)  = {result.F:.6f}\n",
        encoding="utf-8",
    )