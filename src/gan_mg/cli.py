from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gan_mg import __version__
from gan_mg.analysis.thermo import (
    boltzmann_diagnostics_from_energies,
    boltzmann_thermo_from_energies,
    boltzmann_thermo_from_csv,
    diagnostics_from_csv_streaming,
    plot_thermo_vs_T,
    read_energies_csv,
    sweep_thermo_from_csv,
    thermo_from_csv_streaming,
    write_thermo_txt,
    write_thermo_vs_T_csv,
)
from gan_mg.demo.generate import generate_demo_csv
from gan_mg.analysis.figures import regenerate_thermo_figure
from gan_mg.import_results import import_results_to_run
from gan_mg.payloads import build_diagnostics_payload, build_metrics_sweep_payload
from gan_mg.validation import ValidationError, validate_output_file
from gan_mg.run import (
    init_run,
    compute_reproducibility_hash,
    latest_run_id,
    list_runs,
    load_run_meta,
    make_run_id,
    write_run_meta,
)


LOG_FORMAT = "%(message)s"
LOG_LEVELS = {
    "quiet": logging.WARNING,
    "default": logging.INFO,
    "verbose": logging.DEBUG,
}
logger = logging.getLogger(__name__)
SCHEMA_VERSION = "1.1"

COMMON_FLOW_EXAMPLES = """Examples:
  # Minimal end-to-end run
  ganmg generate --run-id demo --n 8 --seed 11
  ganmg analyze --run-id demo --T 1000 --validate-output

  # Config-driven run (PR10)
  ganmg --config configs/run_config.toml

  # Validate outputs from an existing run (PR11)
  ganmg analyze --run-id demo --T 1000 --diagnostics --validate-output

  # Regenerate deterministic golden fixtures
  python scripts/generate_golden.py
"""


def get_git_commit() -> str | None:
    """
    Return current git commit hash if available.

    Return None when git is unavailable, the working directory is not a git repo,
    or the command fails for any reason.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None

    commit = completed.stdout.strip()
    return commit or None


def get_runtime_provenance(
    cli_args: dict[str, Any],
    input_hash: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cli_args": dict(cli_args),
        "input_hash": input_hash,
    }


def log_profile(stage: str, start_time: float, num_configurations: int) -> None:
    elapsed_s = time.perf_counter() - start_time
    logger.info(
        "[profile] %s runtime_s=%.6f num_configurations=%d",
        stage,
        elapsed_s,
        num_configurations,
    )


def write_metrics_json(metrics_path: Path, metrics: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _runtime_seconds(start_time: float) -> float:
    return time.perf_counter() - start_time


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()



def configure_logging(verbose: bool, quiet: bool) -> int:
    if verbose and quiet:
        raise SystemExit("--verbose and --quiet cannot be used together")

    if verbose:
        level = LOG_LEVELS["verbose"]
    elif quiet:
        level = LOG_LEVELS["quiet"]
    else:
        level = LOG_LEVELS["default"]

    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stdout, force=True)
    return level


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir", default="runs", help="Root directory to store runs."
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. If not provided, auto-generated.",
    )


def build_generate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "generate", help="Generate a demo dataset (CSV) into a run folder."
    )
    add_run_args(parser)
    parser.add_argument("--n", type=int, default=10, help="Number of demo structures.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--model",
        choices=["demo", "toy"],
        default="demo",
        help="Energy model backend used during generation.",
    )
    return parser


def build_analyze_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Optional CSV chunk size for streaming analysis.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write additional canonical ensemble diagnostics JSON (does not change existing outputs).",
    )
    parser.add_argument(
        "--validate-output",
        action="store_true",
        help="Validate JSON outputs against the output contract (or set GAN_MG_VALIDATE=1).",
    )
    return parser


def build_runs_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("runs", help="Inspect available runs.")
    parser.add_argument("--run-dir", default="runs", help="Root directory to store runs.")
    runs_sub = parser.add_subparsers(dest="runs_command")
    runs_sub.add_parser("list", help="List all runs with metadata.")
    runs_sub.add_parser("latest", help="Print latest run id.")
    show_parser = runs_sub.add_parser("show", help="Print run metadata and latest metrics.json.")
    show_parser.add_argument("--run-id", required=True, help="Run id to inspect.")
    return parser


def build_sweep_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
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




def build_plot_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "plot",
        help="Regenerate key figures from a run.",
    )
    add_run_args(parser)
    parser.add_argument(
        "--kind",
        choices=["thermo"],
        default="thermo",
        help="Figure kind to regenerate.",
    )
    parser.add_argument("--energy-col", default="energy_eV")
    parser.add_argument("--T-min", type=float, default=300.0, dest="T_min")
    parser.add_argument("--T-max", type=float, default=1500.0, dest="T_max")
    parser.add_argument("--nT", type=int, default=25)
    return parser

def build_doctor_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "doctor", help="Print environment diagnostics for reproducibility."
    )
    parser.add_argument("--run-dir", default="runs", help="Root directory to store runs.")
    return parser


def build_bench_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "bench", help="Run lightweight performance benchmarks."
    )
    bench_sub = parser.add_subparsers(dest="bench_command")

    thermo = bench_sub.add_parser(
        "thermo",
        help="Benchmark thermodynamic sweep on synthetic energies.",
    )
    thermo.add_argument("--n", type=int, default=1000, help="Number of synthetic energies.")
    thermo.add_argument("--nT", type=int, default=50, help="Number of temperatures in the sweep.")
    thermo.add_argument("--T-min", type=float, default=300.0, dest="T_min")
    thermo.add_argument("--T-max", type=float, default=1500.0, dest="T_max")
    thermo.add_argument("--seed", type=int, default=7, help="Random seed for synthetic energies.")
    thermo.add_argument(
        "--out",
        type=Path,
        default=Path("outputs") / "bench.json",
        help="Path to write benchmark JSON output (default: outputs/bench.json).",
    )

    return parser


def build_import_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "import",
        help="Import external results file (.csv or .extxyz) into a run directory.",
    )
    parser.add_argument("--run-id", required=True, help="Run id to import data into.")
    parser.add_argument("--run-dir", required=True, help="Root directory containing run folders.")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to external results input (.csv or .extxyz/.xyz).",
    )
    return parser


def handle_generate(args: argparse.Namespace) -> None:
    stage_start = time.perf_counter()
    run_id = args.run_id or make_run_id(seed=args.seed, n=args.n)
    paths = init_run(Path(args.run_dir), run_id)

    out_csv = paths.inputs_dir / "results.csv"
    generate_demo_csv(n=args.n, seed=args.seed, out_csv=out_csv, model_name=args.model)

    write_run_meta(
        paths.meta_path,
        {
            "command": "generate",
            "run_id": run_id,
            "n": args.n,
            "seed": args.seed,
            "model": args.model,
            "inputs_csv": str(out_csv),
        },
    )

    logger.info("Run created: %s", paths.run_dir)
    logger.info("Wrote %s rows -> %s", args.n, out_csv)
    if args.profile:
        log_profile("generate", stage_start, num_configurations=args.n)


def _should_validate_output(args: argparse.Namespace) -> bool:
    env_value = os.getenv("GAN_MG_VALIDATE", "").strip().lower()
    env_enabled = env_value in {"1", "true", "yes", "on"}
    return bool(getattr(args, "validate_output", False) or env_enabled)


def _validate_output_or_exit(path: Path, *, kind: str) -> None:
    try:
        validate_output_file(path, kind=kind)
    except ValidationError as e:
        raise SystemExit(
            f"Output validation failed: file={path}, first_error_path={e.path}, detail={e.message}"
        ) from e


def handle_analyze(args: argparse.Namespace) -> None:
    stage_start = time.perf_counter()
    if args.run_id is None and args.csv is None:
        args.run_id = latest_run_id(Path(args.run_dir))

    if args.run_id is not None:
        run_dir = Path(args.run_dir) / args.run_id
        csv_path = Path(args.csv) if args.csv else (run_dir / "inputs" / "results.csv")
        out_txt = run_dir / "outputs" / f"thermo_T{int(args.T)}.txt"
        diagnostics_path = run_dir / "outputs" / f"diagnostics_T{int(args.T)}.json"
    else:
        csv_path = Path(args.csv)
        out_txt = Path("results") / "tables" / "demo_thermo.txt"
        diagnostics_path = Path("results") / "tables" / f"diagnostics_T{int(args.T)}.json"

    try:
        if args.chunksize is None:
            result = boltzmann_thermo_from_csv(csv_path, T=args.T, energy_col=args.energy_col)
        else:
            result = thermo_from_csv_streaming(
                csv_path=csv_path,
                temperature_K=args.T,
                energy_column=args.energy_col,
                chunksize=args.chunksize,
            )
    except FileNotFoundError as e:
        raise SystemExit(f"Input file not found: {csv_path} ({e})") from e
    except ValueError as e:
        raise SystemExit(f"Input validation error: {e}") from e

    reproducibility_hash = compute_reproducibility_hash(
        input_csv=csv_path,
        temperature_grid=[args.T],
        code_version=__version__,
    )

    timings: dict[str, float] = {}
    if args.profile:
        timings["runtime_s"] = _runtime_seconds(stage_start)

    metrics: dict[str, Any] = {
        "temperature_K": result.temperature_K,
        "num_configurations": result.num_configurations,
        "mixing_energy_min_eV": result.mixing_energy_min_eV,
        "mixing_energy_avg_eV": result.mixing_energy_avg_eV,
        "partition_function": result.partition_function,
        "free_energy_mix_eV": result.free_energy_mix_eV,
        "created_at": _iso_utc_now(),
    }
    if reproducibility_hash:
        metrics["reproducibility_hash"] = reproducibility_hash
    if timings:
        metrics["timings"] = timings
    metrics["provenance"] = get_runtime_provenance(vars(args), reproducibility_hash)

    if args.run_id is not None:
        metrics_path = run_dir / "outputs" / "metrics.json"
        write_metrics_json(metrics_path, metrics)

        meta = load_run_meta(run_dir)
        meta["reproducibility_hash"] = reproducibility_hash
        write_run_meta(run_dir / "run.json", meta)
    else:
        metrics_path = Path("results") / "tables" / "metrics.json"
        write_metrics_json(metrics_path, metrics)

    write_thermo_txt(result, out_txt)
    if args.diagnostics:
        if args.chunksize is None:
            energies = read_energies_csv(csv_path, energy_col=args.energy_col)
            diagnostics = boltzmann_diagnostics_from_energies(energies, T=args.T)
        else:
            diagnostics = diagnostics_from_csv_streaming(
                csv_path=csv_path,
                temperature_K=args.T,
                energy_column=args.energy_col,
                chunksize=args.chunksize,
            )
        diagnostics_payload = build_diagnostics_payload(
            diagnostics,
            get_runtime_provenance(vars(args), reproducibility_hash),
        )
        write_metrics_json(diagnostics_path, diagnostics_payload)
        logger.info("Wrote: %s", diagnostics_path)

    if _should_validate_output(args):
        _validate_output_or_exit(metrics_path, kind="metrics")
        if args.diagnostics:
            _validate_output_or_exit(diagnostics_path, kind="diagnostics")

    logger.info("Wrote: %s", metrics_path)
    logger.info("%s", out_txt.read_text(encoding="utf-8"))
    if args.profile:
        # Keep profile logging format stable for existing consumers/tests.
        log_profile("analyze", stage_start, num_configurations=result.num_configurations)


def handle_sweep(args: argparse.Namespace) -> None:
    stage_start = time.perf_counter()
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

    reproducibility_hash = compute_reproducibility_hash(
        input_csv=csv_path,
        temperature_grid=t_values,
        code_version=__version__,
    )

    try:
        rows = sweep_thermo_from_csv(csv_path, t_values, energy_col=args.energy_col)
    except ValueError as e:
        raise SystemExit(f"Input validation error: {e}") from e

    write_thermo_vs_T_csv(rows, out_csv)
    logger.info("Wrote: %s", out_csv)

    num_configurations = 0 if not rows else int(rows[0]["num_configurations"])
    timings: dict[str, float] = {}
    if args.profile:
        timings["runtime_s"] = _runtime_seconds(stage_start)

    metrics = build_metrics_sweep_payload(
        rows,
        reproducibility_hash=reproducibility_hash,
        created_at=_iso_utc_now(),
        timings=timings if timings else None,
    )

    if args.csv is None:
        metrics_path = run_dir / "outputs" / "metrics_sweep.json"
    else:
        metrics_path = Path("results") / "tables" / "metrics_sweep.json"
    write_metrics_json(metrics_path, metrics)
    logger.info("Wrote: %s", metrics_path)

    if args.csv is None:
        meta = load_run_meta(run_dir)
        meta["reproducibility_hash"] = reproducibility_hash
        write_run_meta(run_dir / "run.json", meta)

    if args.plot:
        try:
            plot_thermo_vs_T(rows, out_png)
            logger.info("Wrote: %s", out_png)
        except ModuleNotFoundError as e:
            raise SystemExit(
                "Plotting requires matplotlib. Install with:\n"
                "  python -m pip install -e '.[plot]'\n"
                "or\n"
                "  python -m pip install -e '.[dev,plot]'\n"
            ) from e

    if args.profile:
        log_profile("sweep", stage_start, num_configurations=num_configurations)




def handle_plot(args: argparse.Namespace) -> None:
    if args.run_id is None:
        args.run_id = latest_run_id(Path(args.run_dir))

    run_dir = Path(args.run_dir) / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")

    if args.kind == "thermo":
        try:
            out_png = regenerate_thermo_figure(
                run_dir=run_dir,
                energy_col=args.energy_col,
                t_min=args.T_min,
                t_max=args.T_max,
                n_t=args.nT,
            )
        except ModuleNotFoundError as e:
            raise SystemExit(
                "Plotting requires matplotlib. Install with:\n"
                "  python -m pip install -e '.[plot]'\n"
                "or\n"
                "  python -m pip install -e '.[dev,plot]'\n"
            ) from e
        except ValueError as e:
            raise SystemExit(f"Input validation error: {e}") from e

        logger.info("Wrote: %s", out_png)
        return

    raise SystemExit(f"Unsupported --kind: {args.kind}")

def handle_doctor(args: argparse.Namespace) -> None:
    logger.info("ganmg doctor")
    logger.info("%s", "-" * 60)

    logger.info("python_executable : %s", sys.executable)
    logger.info("python_version    : %s", sys.version.split()[0])
    logger.info("platform          : %s", platform.platform())
    logger.info("cwd               : %s", Path.cwd())

    try:
        import gan_mg

        pkg_path = Path(gan_mg.__file__).resolve()
        logger.info("gan_mg_package    : %s", pkg_path)
    except Exception as e:
        logger.info("gan_mg_package    : <ERROR> %s", e)

    mpl = importlib.util.find_spec("matplotlib")
    logger.info("matplotlib        : %s", "available" if mpl is not None else "missing")

    logger.info("default_run_dir   : %s", Path(args.run_dir).resolve())
    logger.info("logging_level     : %s", logging.getLevelName(logging.getLogger().getEffectiveLevel()))

    logger.info("%s", "-" * 60)


def handle_runs(args: argparse.Namespace, runs_parser: argparse.ArgumentParser) -> None:
    run_root = Path(args.run_dir) if hasattr(args, "run_dir") else Path("runs")

    if args.runs_command == "list":
        runs = list_runs(run_root)
        if not runs:
            logger.info("No runs found.")
            return

        logger.info("%s", f"{'Run ID':<35} {'n':<5} {'seed':<5}")
        logger.info("%s", "-" * 50)

        runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)

        for run_dir in runs:
            meta = load_run_meta(run_dir)
            n = meta.get("n", "-")
            seed = meta.get("seed", "-")
            logger.info("%s", f"{run_dir.name:<35} {n:<5} {seed:<5}")

    elif args.runs_command == "latest":
        try:
            latest = latest_run_id(run_root)
            logger.info("%s", latest)
        except FileNotFoundError:
            logger.info("No runs found.")

    elif args.runs_command == "show":
        run_dir = run_root / args.run_id
        if not run_dir.exists():
            raise SystemExit(f"Run not found: {run_dir}")

        meta = load_run_meta(run_dir)
        metrics_path = run_dir / "outputs" / "metrics.json"
        metrics_sweep_path = run_dir / "outputs" / "metrics_sweep.json"
        logger.info("run_id            : %s", args.run_id)
        logger.info("run_dir           : %s", run_dir)

        for key in ("command", "n", "seed", "model", "inputs_csv", "reproducibility_hash"):
            if key in meta:
                logger.info("%s%s", f"{key:<18}: ", meta[key])

        if metrics_path.exists() or metrics_sweep_path.exists():
            logger.info("latest_metrics    :")
            if metrics_path.exists():
                latest_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                logger.info("metrics.json      :")
                logger.info("%s", json.dumps(latest_metrics, indent=2))
            if metrics_sweep_path.exists():
                latest_sweep_metrics = json.loads(metrics_sweep_path.read_text(encoding="utf-8"))
                logger.info("metrics_sweep.json:")
                logger.info("%s", json.dumps(latest_sweep_metrics, indent=2))
        else:
            logger.info("latest_metrics    : <missing> %s", run_dir / "outputs")

    else:
        runs_parser.print_help()


def handle_import(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        metadata = import_results_to_run(run_dir=run_dir, source_path=Path(args.results))
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"Input validation error: {e}") from e

    meta = load_run_meta(run_dir)
    meta["command"] = "import"
    meta["run_id"] = args.run_id
    meta["inputs_csv"] = metadata["results_csv"]
    write_run_meta(run_dir / "run.json", meta)

    logger.info("Run import complete: %s", run_dir)
    logger.info("Imported source: %s", metadata["source_path"])
    logger.info("Wrote canonical CSV: %s", metadata["results_csv"])
    logger.info("Wrote import metadata: %s", metadata["metadata_path"])


def _synthetic_energies(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [
        -1.8
        + 0.0007 * i
        + 0.025 * math.sin(i / 17.0)
        + 0.01 * rng.uniform(-1.0, 1.0)
        for i in range(n)
    ]


def handle_bench(args: argparse.Namespace, bench_parser: argparse.ArgumentParser) -> None:
    if args.bench_command != "thermo":
        bench_parser.print_help()
        return

    if args.n <= 0:
        raise SystemExit("--n must be > 0")
    if args.nT < 2:
        raise SystemExit("--nT must be >= 2")

    energies = _synthetic_energies(n=args.n, seed=args.seed)
    temperatures = [
        args.T_min + i * (args.T_max - args.T_min) / (args.nT - 1)
        for i in range(args.nT)
    ]

    sweep_start = time.perf_counter()
    results = [boltzmann_thermo_from_energies(energies, T=t) for t in temperatures]
    sweep_runtime_s = _runtime_seconds(sweep_start)
    time_per_temperature_ms = (sweep_runtime_s / args.nT) * 1000.0

    payload = {
        "command": "bench thermo",
        "params": {
            "n": args.n,
            "nT": args.nT,
            "T_min": args.T_min,
            "T_max": args.T_max,
            "seed": args.seed,
        },
        "timings": {
            "sweep_runtime_s": sweep_runtime_s,
            "time_per_temperature_ms": time_per_temperature_ms,
        },
        "result_snapshot": {
            "num_configurations": results[0].num_configurations,
            "first_temperature_K": results[0].temperature_K,
            "last_temperature_K": results[-1].temperature_K,
            "first_free_energy_mix_eV": results[0].free_energy_mix_eV,
            "last_free_energy_mix_eV": results[-1].free_energy_mix_eV,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "gan_mg_version": __version__,
        },
    }

    write_metrics_json(args.out, payload)
    logger.info("Wrote: %s", args.out)
    logger.info("Benchmark timing summary")
    logger.info("  configurations (n): %d", args.n)
    logger.info("  temperatures (nT) : %d", args.nT)
    logger.info("  sweep runtime (s) : %.6f", sweep_runtime_s)
    logger.info("  time / temperature : %.3f ms", time_per_temperature_ms)


def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        prog="ganmg",
        description="Reproducible CLI workflow for Mg-doped GaN thermodynamics.",
        epilog=COMMON_FLOW_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Run an analysis from a TOML config file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print runtime and configuration count for generate/analyze/sweep.",
    )
    subparsers = parser.add_subparsers(dest="command")

    build_generate_parser(subparsers)
    build_analyze_parser(subparsers)
    runs_parser = build_runs_parser(subparsers)
    build_sweep_parser(subparsers)
    build_plot_parser(subparsers)
    build_doctor_parser(subparsers)
    build_import_parser(subparsers)
    bench_parser = build_bench_parser(subparsers)

    return parser, runs_parser, bench_parser


def main() -> None:
    parser, runs_parser, bench_parser = build_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    if args.config is not None:
        from gan_mg.run_config import run_from_config

        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"Config file not found: {config_path}")
        try:
            run_from_config(config_path)
        except ValidationError as e:
            raise SystemExit(
                f"Output validation failed: first_error_path={e.path}, detail={e.message}"
            ) from e
        except ValueError as e:
            raise SystemExit(f"Config validation error: {e}") from e
        return

    if args.command == "generate":
        handle_generate(args)
    elif args.command == "analyze":
        handle_analyze(args)
    elif args.command == "sweep":
        handle_sweep(args)
    elif args.command == "plot":
        handle_plot(args)
    elif args.command == "doctor":
        handle_doctor(args)
    elif args.command == "runs":
        handle_runs(args, runs_parser)
    elif args.command == "import":
        handle_import(args)
    elif args.command == "bench":
        handle_bench(args, bench_parser)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
