import argparse
from pathlib import Path

from gan_mg.analysis.thermo import boltzmann_thermo_from_csv, write_thermo_txt


def main() -> None:
    parser = argparse.ArgumentParser(prog="ganmg")
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--csv", required=True)
    analyze_parser.add_argument("--T", type=float, default=1000.0)
    analyze_parser.add_argument("--energy-col", default="energy_eV")

    args = parser.parse_args()

    if args.command == "analyze":
        result = boltzmann_thermo_from_csv(
            Path(args.csv), T=args.T, energy_col=args.energy_col
        )

        out_path = Path("results") / "tables" / "demo_thermo.txt"
        write_thermo_txt(result, out_path)
        print(out_path.read_text(encoding="utf-8"))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()