import argparse
from pathlib import Path

from gan_mg.analysis.thermo import boltzmann_thermo_from_csv, write_thermo_txt
from gan_mg.demo.generate import generate_demo_csv
from gan_mg.run import init_run, make_run_id, write_run_meta


def main() -> None:
    parser = argparse.ArgumentParser(prog="ganmg")
    subparsers = parser.add_subparsers(dest="command")

    # Common run args helper
    def add_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-dir", default="runs", help="Root directory to store runs.")
        p.add_argument("--run-id", default=None, help="Optional run id. If not provided, auto-generated.")

    # ---- generate ----
    gen = subparsers.add_parser("generate", help="Generate a demo dataset (CSV) into a run folder.")
    add_run_args(gen)
    gen.add_argument("--n", type=int, default=10, help="Number of demo structures.")
    gen.add_argument("--seed", type=int, default=7, help="Random seed.")

    # ---- analyze ----
    ana = subparsers.add_parser("analyze", help="Run Boltzmann analysis; defaults to latest run if not specified.")
    add_run_args(ana)
    ana.add_argument("--csv", default=None, help="CSV path. If omitted, uses <run>/inputs/results.csv.")
    ana.add_argument("--T", type=float, default=1000.0, help="Temperature in K.")
    ana.add_argument("--energy-col", default="energy_eV", help="Energy column name.")

    args = parser.parse_args()

    if args.command == "generate":
        run_id = args.run_id or make_run_id(seed=args.seed, n=args.n)
        paths = init_run(Path(args.run_dir), run_id)

        out_csv = paths.inputs_dir / "results.csv"
        generate_demo_csv(n=args.n, seed=args.seed, out_csv=out_csv)

        write_run_meta(
            paths.meta_path,
            {
                "command": "generate",
                "run_id": run_id,
                "n": args.n,
                "seed": args.seed,
                "inputs_csv": str(out_csv),
            },
        )

        print(f"Run created: {paths.run_dir}")
        print(f"Wrote {args.n} rows -> {out_csv}")

    elif args.command == "analyze":
        # Analyze either a provided csv, or the run folder's default csv
        if args.run_id is None and args.csv is None:
            raise SystemExit("Provide --run-id or --csv (for now). Next step we can add 'latest run' lookup.")

        run_id = args.run_id
        if run_id is not None:
            run_dir = Path(args.run_dir) / run_id
            csv_path = Path(args.csv) if args.csv else (run_dir / "inputs" / "results.csv")
            out_txt = run_dir / "outputs" / f"thermo_T{int(args.T)}.txt"
        else:
            csv_path = Path(args.csv)
            out_txt = Path("results") / "tables" / "demo_thermo.txt"  # fallback

        result = boltzmann_thermo_from_csv(csv_path, T=args.T, energy_col=args.energy_col)
        write_thermo_txt(result, out_txt)

        print(out_txt.read_text(encoding="utf-8"))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()