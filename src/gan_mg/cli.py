import argparse
from pathlib import Path

from gan_mg.analysis.thermo import boltzmann_thermo_from_csv, write_thermo_txt, sweep_thermo_from_csv, write_thermo_vs_T_csv, plot_thermo_vs_T
from gan_mg.demo.generate import generate_demo_csv
from gan_mg.run import (
    init_run,
    make_run_id,
    write_run_meta,
    latest_run_id,
    list_runs,
    load_run_meta,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ganmg")
    subparsers = parser.add_subparsers(dest="command")

    # Common run args helper
    def add_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--run-dir", default="runs", help="Root directory to store runs."
        )
        p.add_argument(
            "--run-id",
            default=None,
            help="Optional run id. If not provided, auto-generated.",
        )

    # ---- generate ----
    gen = subparsers.add_parser(
        "generate", help="Generate a demo dataset (CSV) into a run folder."
    )
    add_run_args(gen)
    gen.add_argument("--n", type=int, default=10, help="Number of demo structures.")
    gen.add_argument("--seed", type=int, default=7, help="Random seed.")

    # ---- analyze ----
    ana = subparsers.add_parser(
        "analyze",
        help="Run Boltzmann analysis; defaults to latest run if not specified.",
    )
    add_run_args(ana)
    ana.add_argument(
        "--csv",
        default=None,
        help="CSV path. If omitted, uses <run>/inputs/results.csv.",
    )
    ana.add_argument("--T", type=float, default=1000.0, help="Temperature in K.")
    ana.add_argument("--energy-col", default="energy_eV", help="Energy column name.")

    # ---- runs ----
    runs_parser = subparsers.add_parser("runs", help="Inspect available runs.")
    runs_sub = runs_parser.add_subparsers(dest="runs_command")

    runs_sub.add_parser("list", help="List all runs with metadata.")
    runs_sub.add_parser("latest", help="Print latest run id.")

    # ---- sweep ----
    sw = subparsers.add_parser("sweep", help="Compute thermo across a temperature range and save CSV/plot.")
    add_run_args(sw)
    sw.add_argument("--csv", default=None, help="CSV path. If omitted, uses <run>/inputs/results.csv.")
    sw.add_argument("--T-min", type=float, default=300.0, dest="T_min")
    sw.add_argument("--T-max", type=float, default=1500.0, dest="T_max")
    sw.add_argument("--nT", type=int, default=25)
    sw.add_argument("--energy-col", default="energy_eV")

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
        # If neither --run-id nor --csv is provided, analyze the latest run under --run-dir
        if args.run_id is None and args.csv is None:
            args.run_id = latest_run_id(Path(args.run_dir))

        # Analyze either a provided csv, or the run folder's default csv
        if args.run_id is not None:
            run_dir = Path(args.run_dir) / args.run_id
            csv_path = Path(args.csv) if args.csv else (run_dir / "inputs" / "results.csv")
            out_txt = run_dir / "outputs" / f"thermo_T{int(args.T)}.txt"
        else:
            # fallback: direct csv path (no run folder)
            csv_path = Path(args.csv)
            out_txt = Path("results") / "tables" / "demo_thermo.txt"

        result = boltzmann_thermo_from_csv(csv_path, T=args.T, energy_col=args.energy_col)
        write_thermo_txt(result, out_txt)
        print(out_txt.read_text(encoding="utf-8"))

    elif args.command == "sweep":

        if args.csv is None:
            if args.run_id is None:
                args.run_id = latest_run_id(Path(args.run_dir))

            run_dir = Path(args.run_dir) / args.run_id
            csv_path = run_dir / "inputs" / "results.csv"
            out_csv = run_dir / "outputs" / "thermo_vs_T.csv"
            out_png = run_dir / "outputs" / "thermo_vs_T.png"
        else:
            csv_path = Path(args.csv)
            out_csv = Path("results") / "tables" / "thermo_vs_T.csv"
            out_png = Path("results") / "figures" / "thermo_vs_T.png"

        if args.nT < 2:
            raise SystemExit("--nT must be >= 2")

        T_values = [
            args.T_min + i * (args.T_max - args.T_min) / (args.nT - 1)
            for i in range(args.nT)
        ]

        rows = sweep_thermo_from_csv(csv_path, T_values, energy_col=args.energy_col)

        write_thermo_vs_T_csv(rows, out_csv)
        plot_thermo_vs_T(rows, out_png)

        print(f"Wrote: {out_csv}")
        print(f"Wrote: {out_png}")

    elif args.command == "runs":
        run_root = Path(args.run_dir) if hasattr(args, "run_dir") else Path("runs")

        if args.runs_command == "list":
            runs = list_runs(run_root)
            if not runs:
                print("No runs found.")
                return

            print(f"{'Run ID':<35} {'n':<5} {'seed':<5}")
            print("-" * 50)

            # Sort newest first
            runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)

            for run_dir in runs:
                meta = load_run_meta(run_dir)
                n = meta.get("n", "-")
                seed = meta.get("seed", "-")
                print(f"{run_dir.name:<35} {n:<5} {seed:<5}")

        elif args.runs_command == "latest":
            try:
                latest = latest_run_id(run_root)
                print(latest)
            except FileNotFoundError:
                print("No runs found.")

        else:
            runs_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
