import argparse
from pathlib import Path

from gan_mg.analysis.thermo import boltzmann_thermo_from_csv, write_thermo_txt
from gan_mg.demo.generate import generate_demo_csv


def main() -> None:
    parser = argparse.ArgumentParser(prog="ganmg")
    subparsers = parser.add_subparsers(dest="command")

    # ---- generate ----
    gen_parser = subparsers.add_parser("generate", help="Generate a demo dataset (CSV).")
    gen_parser.add_argument("--n", type=int, default=10, help="Number of demo structures.")
    gen_parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    gen_parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "sample_outputs" / "results.csv"),
        help="Output CSV path.",
    )

    # ---- analyze ----
    analyze_parser = subparsers.add_parser("analyze", help="Run Boltzmann analysis on CSV energies.")
    analyze_parser.add_argument("--csv", required=True, help="CSV path containing energy_eV column.")
    analyze_parser.add_argument("--T", type=float, default=1000.0, help="Temperature in K.")
    analyze_parser.add_argument("--energy-col", default="energy_eV", help="Energy column name.")
    analyze_parser.add_argument(
        "--out",
        type=str,
        default=str(Path("results") / "tables" / "demo_thermo.txt"),
        help="Output text file path.",
    )

    args = parser.parse_args()

    if args.command == "generate":
        out_csv = generate_demo_csv(n=args.n, seed=args.seed, out_csv=Path(args.out))
        print(f"Wrote {args.n} rows -> {out_csv}")

    elif args.command == "analyze":
        result = boltzmann_thermo_from_csv(Path(args.csv), T=args.T, energy_col=args.energy_col)
        out_path = Path(args.out)
        write_thermo_txt(result, out_path)
        print(out_path.read_text(encoding="utf-8"))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()