import argparse
from pathlib import Path

from gan_mg.analysis.thermo import boltzmann_thermo_from_csv, write_thermo_txt


def main() -> None:
    parser = argparse.ArgumentParser(description="Boltzmann thermodynamics demo (wrapper).")
    parser.add_argument("--csv", required=True, help="CSV path containing energy_eV column.")
    parser.add_argument("--T", type=float, default=1000.0, help="Temperature in K.")
    parser.add_argument("--energy-col", type=str, default="energy_eV", help="Energy column name.")
    args = parser.parse_args()

    result = boltzmann_thermo_from_csv(Path(args.csv), T=args.T, energy_col=args.energy_col)

    out_path = Path("results") / "tables" / "demo_thermo.txt"
    write_thermo_txt(result, out_path)

    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()