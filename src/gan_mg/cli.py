import argparse
import importlib.util
import platform
import sys
from pathlib import Path

from gan_mg.analysis.thermo import (
    boltzmann_thermo_from_csv,
    plot_thermo_vs_T,
    sweep_thermo_from_csv,
    write_thermo_txt,
    write_thermo_vs_T_csv,
)
from gan_mg.demo.generate import generate_demo_csv
from gan_mg.run import (
    init_run,
    latest_run_id,
    list_runs,
    load_run_meta,
    make_run_id,
    write_run_meta,
)


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir", default="runs", help="Root directory to store runs."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. If not provided, auto-generated.",
    )


def build_generate_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "generate", help="Generate a demo dataset (CSV) into a run folder."
    )
    add_run_args(parser)
    parser.add_argument("--n", type=int, default=10, help="Number of demo structures.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser


def build_analyze_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "analyze",
        help="Run Boltzmann analysis; defaults to latest run if not specified.",
    )
    add_run_args(parser)
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV path. If omitted, uses <run>/inputs/results.csv.",
    )
    parser.add_argument("--T", type=float, default=1000.0, help="Temperature in K.")
    parser.add_argument("--energy-col", default="energy_eV", help="Energy column name.")
    return parser


def build_runs_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("runs", help="Inspect available runs.")
    runs_sub = parser.add_subparsers(dest="runs_command")
    runs_sub.add_parser("list", help="List all runs with metadata.")
    runs_sub.add_parser("latest", help="Print latest run id.")
    return parser


def build_sweep_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "sweep", help="Compute thermo across a temperature range and save CSV/plot."
    )
    add_run_args(parser)
    parser.add_argument(
        "--csv", default=None, help="CSV path. If omitted, uses <run>/inputs/results.csv."
    )
    parser.add_argument("--T-min", type=float, default=300.0, dest="T_min")
    parser.add_argument("--T-max", type=float, default=1500.0, dest="T_max")
    parser.add_argument("--nT", type=int, default=25)
    parser.add_argument("--energy-col", default="energy_eV")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write a PNG plot (requires optional dependency: gan-mg-doping-workflow[plot]).",
    )
    return parser


def build_doctor_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "doctor", help="Print environment diagnostics for reproducibility."
    )
    parser.add_argument("--run-dir", default="runs", help="Root directory to store runs.")
    return parser


def handle_generate(args: argparse.Namespace) -> None:
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


def handle_analyze(args: argparse.Namespace) -> None:
    if args.run_id is None and args.csv is None:
        args.run_id = latest_run_id(Path(args.run_dir))

    if args.run_id is not None:
        run_dir = Path(args.run_dir) / args.run_id
        csv_path = Path(args.csv) if args.csv else (run_dir / "inputs" / "results.csv")
        out_txt = run_dir / "outputs" / f"thermo_T{int(args.T)}.txt"
    else:
        csv_path = Path(args.csv)
        out_txt = Path("results") / "tables" / "demo_thermo.txt"

    result = boltzmann_thermo_from_csv(csv_path, T=args.T, energy_col=args.energy_col)
    write_thermo_txt(result, out_txt)
    print(out_txt.read_text(encoding="utf-8"))


def handle_sweep(args: argparse.Namespace) -> None:
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

    t_values = [
        args.T_min + i * (args.T_max - args.T_min) / (args.nT - 1)
        for i in range(args.nT)
    ]

    rows = sweep_thermo_from_csv(csv_path, t_values, energy_col=args.energy_col)

    write_thermo_vs_T_csv(rows, out_csv)
    print(f"Wrote: {out_csv}")

    if args.plot:
        try:
            plot_thermo_vs_T(rows, out_png)
            print(f"Wrote: {out_png}")
        except ModuleNotFoundError as e:
            raise SystemExit(
                "Plotting requires matplotlib. Install with:\n"
                "  python -m pip install -e '.[plot]'\n"
                "or\n"
                "  python -m pip install -e '.[dev,plot]'\n"
            ) from e


def handle_doctor(args: argparse.Namespace) -> None:
    print("ganmg doctor")
    print("-" * 60)

    print(f"python_executable : {sys.executable}")
    print(f"python_version    : {sys.version.split()[0]}")
    print(f"platform          : {platform.platform()}")
    print(f"cwd               : {Path.cwd()}")

    try:
        import gan_mg

        pkg_path = Path(gan_mg.__file__).resolve()
        print(f"gan_mg_package    : {pkg_path}")
    except Exception as e:
        print(f"gan_mg_package    : <ERROR> {e}")

    mpl = importlib.util.find_spec("matplotlib")
    print(f"matplotlib        : {'available' if mpl is not None else 'missing'}")

    print(f"default_run_dir   : {Path(args.run_dir).resolve()}")

    print("-" * 60)


def handle_runs(args: argparse.Namespace, runs_parser: argparse.ArgumentParser) -> None:
    run_root = Path(args.run_dir) if hasattr(args, "run_dir") else Path("runs")

    if args.runs_command == "list":
        runs = list_runs(run_root)
        if not runs:
            print("No runs found.")
            return

        print(f"{'Run ID':<35} {'n':<5} {'seed':<5}")
        print("-" * 50)

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


def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(prog="ganmg")
    subparsers = parser.add_subparsers(dest="command")

    build_generate_parser(subparsers)
    build_analyze_parser(subparsers)
    runs_parser = build_runs_parser(subparsers)
    build_sweep_parser(subparsers)
    build_doctor_parser(subparsers)

    return parser, runs_parser


def main() -> None:
    parser, runs_parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        handle_generate(args)
    elif args.command == "analyze":
        handle_analyze(args)
    elif args.command == "sweep":
        handle_sweep(args)
    elif args.command == "doctor":
        handle_doctor(args)
    elif args.command == "runs":
        handle_runs(args, runs_parser)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
