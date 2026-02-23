import argparse
import csv
import math
from pathlib import Path

K_B_EV_PER_K = 8.617333262e-5  # Boltzmann constant in eV/K


def read_energies(csv_path: Path) -> list[float]:
    energies: list[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            energies.append(float(row["energy_eV"]))
    return energies


def main() -> None:
    """
    Demo Boltzmann analysis:
    - reads energies (eV) from CSV
    - computes partition function Z, mean energy <E>, and Helmholtz free energy F
    - writes a small text summary into results/tables/
    """
    parser = argparse.ArgumentParser(description="Boltzmann thermodynamics demo.")
    parser.add_argument("--csv", required=True, help="Path to CSV containing energy_eV column.")
    parser.add_argument("--T", type=float, default=1000.0, help="Temperature in K.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    energies = read_energies(csv_path)
    if len(energies) == 0:
        raise ValueError("No energies found in CSV.")

    beta = 1.0 / (K_B_EV_PER_K * args.T)

    # Use a numerically-stable shift: exp(-beta*(E - Emin))
    emin = min(energies)
    shifted = [e - emin for e in energies]
    weights = [math.exp(-beta * e) for e in shifted]
    Z_tilde = sum(weights)  # Z * exp(+beta*emin)
    Z = Z_tilde * math.exp(-beta * 0.0)  # keep symbolically; Z_tilde is enough for probs

    probs = [w / Z_tilde for w in weights]
    E_avg = sum(p * e for p, e in zip(probs, energies))

    # F = -1/beta * ln(Z) ; since Z = exp(-beta*emin) * Z_tilde
    F = emin - (1.0 / beta) * math.log(Z_tilde)

    out_dir = Path("results") / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / "demo_thermo.txt"
    out_txt.write_text(
        f"T(K) = {args.T}\n"
        f"N    = {len(energies)}\n"
        f"Emin (eV) = {emin:.6f}\n"
        f"Z_tilde   = {Z_tilde:.6e}\n"
        f"<E> (eV)  = {E_avg:.6f}\n"
        f"F   (eV)  = {F:.6f}\n",
        encoding="utf-8",
    )

    print(out_txt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()