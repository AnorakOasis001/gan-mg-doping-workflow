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

def _get_attr(obj, names, default=None):
    """Try multiple attribute names; return the first that exists."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def sweep_thermo_from_csv(
    csv_path: Path,
    T_values: list[float],
    energy_col: str = "energy_eV",
) -> list[dict]:
    """
    Run boltzmann_thermo_from_csv for each temperature and return list of dict rows.
    This wrapper is robust to different ThermoResult field names.
    """
    csv_path = Path(csv_path)
    rows: list[dict] = []

    for T in T_values:
        res = boltzmann_thermo_from_csv(csv_path, T=T, energy_col=energy_col)

        N = _get_attr(res, ["N", "n", "num_states"])
        Emin = _get_attr(res, ["Emin_eV", "Emin", "emin_eV", "emin"])
        Zt = _get_attr(res, ["Z_tilde", "Ztilde", "Z", "z_tilde"])
        Eavg = _get_attr(res, ["Eavg_eV", "Eavg", "E_avg"])   # <-- add E_avg
        F = _get_attr(res, ["F_eV", "F"])                      # <-- F is your field

        rows.append(
            {
                "T_K": float(T),
                "N": N,
                "Emin_eV": Emin,
                "Z_tilde": Zt,
                "Eavg_eV": Eavg,
                "F_eV": F,
            }
        )

    rows.sort(key=lambda r: r["T_K"])
    return rows


def write_thermo_vs_T_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["T_K", "N", "Emin_eV", "Z_tilde", "Eavg_eV", "F_eV"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_thermo_vs_T(rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Debug: validate keys once
    required = {"T_K", "F_eV", "Eavg_eV"}
    missing = [required - set(r.keys()) for r in rows[:1]]
    if missing and missing[0]:
        raise KeyError(f"Row is missing keys {missing[0]}. Row keys are: {list(rows[0].keys())}")

    T = [r["T_K"] for r in rows]
    F = [r["F_eV"] for r in rows]
    E = [r["Eavg_eV"] for r in rows]

    plt.figure()
    plt.plot(T, F, label="F (eV)")
    plt.plot(T, E, label="<E> (eV)")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Energy (eV)")
    plt.legend()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()