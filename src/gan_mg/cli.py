from __future__ import annotations

import argparse
import importlib.util
from importlib import metadata as importlib_metadata
import json
import logging
import hashlib
import math
import os
import platform
import random
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any

from gan_mg import __version__
from gan_mg._mpl import ensure_agg
from gan_mg.artifacts import write_json
from gan_mg.analysis.thermo import (
    boltzmann_thermo_from_energies,
    plot_thermo_vs_T,
)
from gan_mg.demo.generate import generate_demo_csv
from gan_mg.analysis.figures import regenerate_thermo_figure
from gan_mg.analysis.crossover import derive_mechanism_crossover_dataset
from gan_mg.analysis.crossover_uncertainty import derive_crossover_uncertainty_dataset
from gan_mg.analysis.phase_map import derive_phase_map_dataset
from gan_mg.import_results import import_results_to_run
from gan_mg.science.mixing import derive_mixing_dataset
from gan_mg.science.gibbs import derive_gibbs_summary_dataset
from gan_mg.science.uncertainty import derive_gibbs_uncertainty_dataset
from gan_mg.science.per_structure import derive_per_structure_dataset
from gan_mg.science.reference import load_reference_config
from gan_mg.services import analyze_run, sweep_run
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
    write_json(metrics_path, metrics)


def _runtime_seconds(start_time: float) -> float:
    return time.perf_counter() - start_time


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest_entry(path: Path) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": _sha256_hex(path)}


def _gan_mg_package_version() -> str | None:
    for dist_name in ("gan-mg-doping-workflow", "gan_mg"):
        try:
            return importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


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




def build_derive_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "derive",
        help="Build runs/<id>/derived/per_structure.csv from results + structure artifacts.",
    )
    add_run_args(parser)
    return parser


def build_mix_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "mix",
        help="Compute reference energies, mixing energies, and athermal summary CSVs.",
    )
    add_run_args(parser)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Path to reference config (.json or .toml). Defaults to runs/<id>/inputs/reference.json.",
    )
    return parser


def build_gibbs_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "gibbs",
        help="Compute Boltzmann Gibbs summaries from per_structure_mixing.csv and generate overlay plots.",
    )
    add_run_args(parser)
    parser.add_argument(
        "--temps",
        default=None,
        help="Comma-separated temperature list in K, e.g. 300,500,700,1000",
    )
    parser.add_argument("--T-min", type=float, default=None, dest="T_min")
    parser.add_argument("--T-max", type=float, default=None, dest="T_max")
    parser.add_argument("--nT", type=int, default=None)
    return parser


def build_uncertainty_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "uncertainty",
        help="Compute bootstrap uncertainty bands for Gibbs free energy and mixing-energy averages.",
    )
    add_run_args(parser)
    parser.add_argument(
        "--temps",
        default=None,
        help="Comma-separated temperature list in K, e.g. 300,500,700,1000",
    )
    parser.add_argument("--T-min", type=float, default=None, dest="T_min")
    parser.add_argument("--T-max", type=float, default=None, dest="T_max")
    parser.add_argument("--nT", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser

def build_phase_map_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "phase-map",
        help="Build phase_map.csv and phase-map visualizations from crossover_uncertainty.csv.",
    )
    add_run_args(parser)
    return parser

def build_reproduce_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "reproduce",
        help="One-command reproduction workflows.",
    )
    reproduce_sub = parser.add_subparsers(dest="reproduce_command")

    overlay = reproduce_sub.add_parser(
        "overlay",
        help="Ensure derive/mix/gibbs outputs exist, regenerate overlay, and write a reproducibility manifest.",
    )
    add_run_args(overlay)
    overlay.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to reference config (.json or .toml).",
    )
    overlay.add_argument(
        "--temps",
        required=True,
        help="Comma-separated temperature list in K, e.g. 300,500,700,1000",
    )

    return parser


def build_repropack_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "repropack",
        help="Export a portable reproducibility pack for a run.",
    )
    add_run_args(parser)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("repropack"),
        help="Destination directory for exported repro packs (default: repropack).",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also write <out>/<run-id>.zip containing the exported repro pack.",
    )
    parser.add_argument(
        "--zip-only",
        action="store_true",
        help="With --zip, remove the unpacked folder after creating the zip archive.",
    )
    return parser
def _parse_temperatures(args: argparse.Namespace) -> list[float]:
    temps_arg = getattr(args, "temps", None)
    t_min = getattr(args, "T_min", None)
    t_max = getattr(args, "T_max", None)
    n_t = getattr(args, "nT", None)

    if temps_arg:
        values = [token.strip() for token in str(temps_arg).split(",") if token.strip()]
        if not values:
            raise SystemExit("--temps provided but empty")
        temps = [float(v) for v in values]
    elif t_min is not None or t_max is not None or n_t is not None:
        if t_min is None or t_max is None or n_t is None:
            raise SystemExit("Sweep form requires --T-min, --T-max, and --nT together")
        if n_t < 1:
            raise SystemExit("--nT must be >= 1")
        if n_t == 1:
            temps = [float(t_min)]
        else:
            temps = [
                t_min + i * (t_max - t_min) / (n_t - 1)
                for i in range(n_t)
            ]
    else:
        raise SystemExit("Provide either --temps or sweep form --T-min/--T-max/--nT")

    if any(t <= 0 for t in temps):
        raise SystemExit("All temperatures must be > 0 K")

    return sorted(float(t) for t in temps)

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

    timings: dict[str, float] = {}
    if args.profile:
        timings["runtime_s"] = _runtime_seconds(stage_start)

    if args.run_id is not None:
        metrics_path = run_dir / "outputs" / "metrics.json"
    else:
        metrics_path = Path("results") / "tables" / "metrics.json"

    try:
        reproducibility_hash = compute_reproducibility_hash(
            input_csv=csv_path,
            temperature_grid=[args.T],
            code_version=__version__,
        )
        provenance = get_runtime_provenance(vars(args), reproducibility_hash)

        artifacts = analyze_run(
            csv_path=csv_path,
            metrics_path=metrics_path,
            thermo_path=out_txt,
            temperature_K=args.T,
            energy_col=args.energy_col,
            chunksize=args.chunksize,
            diagnostics=args.diagnostics,
            diagnostics_path=diagnostics_path,
            reproducibility_hash=reproducibility_hash,
            created_at=_iso_utc_now(),
            timings=timings if timings else None,
            provenance=provenance,
            include_reproducibility_hash=True,
        )
    except FileNotFoundError as e:
        raise SystemExit(f"Input file not found: {csv_path} ({e})") from e
    except ValueError as e:
        raise SystemExit(f"Input validation error: {e}") from e

    if args.run_id is not None:
        meta = load_run_meta(run_dir)
        meta["reproducibility_hash"] = reproducibility_hash
        write_run_meta(run_dir / "run.json", meta)

    if args.diagnostics and artifacts.diagnostics_path is not None:
        logger.info("Wrote: %s", artifacts.diagnostics_path)

    if _should_validate_output(args):
        _validate_output_or_exit(metrics_path, kind="metrics")
        if args.diagnostics:
            _validate_output_or_exit(diagnostics_path, kind="diagnostics")

    logger.info("Wrote: %s", metrics_path)
    logger.info("%s", artifacts.thermo_path.read_text(encoding="utf-8"))
    if args.profile:
        # Keep profile logging format stable for existing consumers/tests.
        log_profile("analyze", stage_start, num_configurations=artifacts.thermo_result.num_configurations)


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

    if args.csv is None:
        metrics_path = run_dir / "outputs" / "metrics_sweep.json"
    else:
        metrics_path = Path("results") / "tables" / "metrics_sweep.json"

    timings: dict[str, float] = {}
    if args.profile:
        timings["runtime_s"] = _runtime_seconds(stage_start)

    try:
        artifacts = sweep_run(
            csv_path=csv_path,
            t_values=t_values,
            energy_col=args.energy_col,
            thermo_vs_t_path=out_csv,
            metrics_sweep_path=metrics_path,
            reproducibility_hash=reproducibility_hash,
            created_at=_iso_utc_now(),
            timings=timings if timings else None,
        )
    except ValueError as e:
        raise SystemExit(f"Input validation error: {e}") from e

    logger.info("Wrote: %s", artifacts.thermo_vs_t_path)
    logger.info("Wrote: %s", artifacts.metrics_sweep_path)

    num_configurations = 0 if not artifacts.rows else int(artifacts.rows[0]["num_configurations"])

    if args.csv is None:
        meta = load_run_meta(run_dir)
        meta["reproducibility_hash"] = reproducibility_hash
        write_run_meta(run_dir / "run.json", meta)

    if args.plot:
        try:
            ensure_agg()
            plot_thermo_vs_T(artifacts.rows, out_png)
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
            ensure_agg()
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




def handle_derive(args: argparse.Namespace) -> None:
    if args.run_id is None:
        args.run_id = latest_run_id(Path(args.run_dir))

    run_dir = Path(args.run_dir) / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")

    try:
        out_path = derive_per_structure_dataset(run_dir)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"Derive error: {e}") from e

    logger.info("Derived per-structure dataset: %s", out_path)


def handle_mix(args: argparse.Namespace) -> None:
    if args.run_id is None:
        args.run_id = latest_run_id(Path(args.run_dir))

    run_dir = Path(args.run_dir) / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")

    try:
        mixing_path, summary_path = derive_mixing_dataset(run_dir, reference_path=args.reference)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"Mix error: {e}") from e

    logger.info("Derived mixing dataset: %s", mixing_path)
    logger.info("Derived athermal summary: %s", summary_path)


def _write_overlay_related_figures(run_dir: Path, all_mech_path: Path) -> tuple[Path, Path, Path | None]:
    try:
        ensure_agg()
        from gan_mg.viz.overlay import (
            plot_athermal_emin_vs_doping,
            plot_overlay_dgmix_vs_doping_multi_t,
        )
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Plotting requires matplotlib. Install with\n"
            "  python -m pip install -e \'.[plot]\'\n"
            "or\n"
            "  python -m pip install -e \'.[dev,plot]\'\n"
        ) from e


    overlay_path = run_dir / "figures" / "overlay_dGmix_vs_doping_multiT.png"
    plot_overlay_dgmix_vs_doping_multi_t(all_mech_path, overlay_path)

    athermal_png: Path | None = None
    athermal_csv = run_dir / "derived" / "mixing_athermal_summary.csv"
    if athermal_csv.exists():
        athermal_png = run_dir / "figures" / "athermal_Emixmin_vs_doping.png"
        plot_athermal_emin_vs_doping(athermal_csv, athermal_png)

    crossover_csv, crossover_png = derive_mechanism_crossover_dataset(run_dir)
    logger.info("Wrote: %s", crossover_csv)
    return overlay_path, crossover_png, athermal_png


def handle_gibbs(args: argparse.Namespace) -> None:
    if args.run_id is None:
        args.run_id = latest_run_id(Path(args.run_dir))

    run_dir = Path(args.run_dir) / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")

    temperatures = _parse_temperatures(args)

    try:
        gibbs_path, all_mech_path = derive_gibbs_summary_dataset(run_dir, temperatures)
    except FileNotFoundError as e:
        raise SystemExit(
            f"Gibbs error: {e} Please run `ganmg mix --run-id {args.run_id}` first."
        ) from e
    except ValueError as e:
        raise SystemExit(f"Gibbs error: {e}") from e

    logger.info("Derived Gibbs summary: %s", gibbs_path)
    logger.info("Derived all-mechanisms Gibbs summary: %s", all_mech_path)

    overlay_path, crossover_png, athermal_png = _write_overlay_related_figures(run_dir, all_mech_path)
    logger.info("Wrote: %s", overlay_path)
    logger.info("Wrote: %s", crossover_png)

    if athermal_png is not None:
        logger.info("Wrote: %s", athermal_png)


def handle_uncertainty(args: argparse.Namespace) -> None:
    if args.run_id is None:
        args.run_id = latest_run_id(Path(args.run_dir))

    run_dir = Path(args.run_dir) / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")

    temperatures = _parse_temperatures(args)

    try:
        uncertainty_csv = derive_gibbs_uncertainty_dataset(
            run_dir,
            temperatures,
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.seed),
        )
    except FileNotFoundError as e:
        raise SystemExit(
            f"Uncertainty error: {e} Please run `ganmg mix --run-id {args.run_id}` first."
        ) from e
    except ValueError as e:
        raise SystemExit(f"Uncertainty error: {e}") from e

    logger.info("Derived Gibbs uncertainty summary: %s", uncertainty_csv)

    try:
        ensure_agg()
        from gan_mg.viz.overlay import plot_overlay_dgmix_vs_doping_multi_t_ci
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Plotting requires matplotlib. Install with\n"
            "  python -m pip install -e '.[plot]'\n"
            "or\n"
            "  python -m pip install -e '.[dev,plot]'\n"
        ) from e

    overlay_ci = run_dir / "figures" / "overlay_dGmix_vs_doping_multiT_ci.png"
    plot_overlay_dgmix_vs_doping_multi_t_ci(uncertainty_csv, overlay_ci)
    logger.info("Wrote: %s", overlay_ci)

    crossover_uncertainty_csv = derive_crossover_uncertainty_dataset(run_dir)
    logger.info("Wrote: %s", crossover_uncertainty_csv)
    logger.info("Wrote: %s", run_dir / "figures" / "crossover_map_uncertainty.png")

def handle_phase_map(args: argparse.Namespace) -> None:
    if args.run_id is None:
        args.run_id = latest_run_id(Path(args.run_dir))

    run_dir = Path(args.run_dir) / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")

    try:
        phase_map_csv = derive_phase_map_dataset(run_dir)
    except FileNotFoundError as e:
        raise SystemExit(
            f"Phase-map error: {e} Please run `ganmg uncertainty --run-id {args.run_id} ...` first."
        ) from e
    except ValueError as e:
        raise SystemExit(f"Phase-map error: {e}") from e

    logger.info("Wrote: %s", phase_map_csv)

    try:
        ensure_agg()
        from gan_mg.viz.phase_map import plot_phase_map_delta_g, plot_phase_map_preference
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Plotting requires matplotlib. Install with\n"
            "  python -m pip install -e '.[plot]'\n"
            "or\n"
            "  python -m pip install -e '.[dev,plot]'\n"
        ) from e

    pref_png = run_dir / "figures" / "phase_map_preference.png"
    plot_phase_map_preference(phase_map_csv, pref_png)
    logger.info("Wrote: %s", pref_png)

    delta_png = run_dir / "figures" / "phase_map_deltaG.png"
    plot_phase_map_delta_g(phase_map_csv, delta_png)
    logger.info("Wrote: %s", delta_png)

def handle_reproduce_overlay(args: argparse.Namespace) -> None:
    if args.run_id is None:
        args.run_id = latest_run_id(Path(args.run_dir))

    run_dir = Path(args.run_dir) / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")

    per_structure_csv = run_dir / "derived" / "per_structure.csv"
    if not per_structure_csv.exists():
        per_structure_csv = derive_per_structure_dataset(run_dir)
        logger.info("Derived per-structure dataset: %s", per_structure_csv)

    mixing_csv = run_dir / "derived" / "per_structure_mixing.csv"
    if not mixing_csv.exists():
        mixing_csv, summary_path = derive_mixing_dataset(run_dir, reference_path=args.reference)
        logger.info("Derived mixing dataset: %s", mixing_csv)
        logger.info("Derived athermal summary: %s", summary_path)

    temperatures = _parse_temperatures(args)
    gibbs_path, all_mech_path = derive_gibbs_summary_dataset(run_dir, temperatures)
    overlay_path, crossover_png, athermal_png = _write_overlay_related_figures(run_dir, all_mech_path)

    reference_model, reference_energies = load_reference_config(Path(args.reference))

    output_files = [
        gibbs_path,
        all_mech_path,
        overlay_path,
        run_dir / "derived" / "mechanism_crossover.csv",
        crossover_png,
    ]
    if athermal_png is not None:
        output_files.append(athermal_png)

    outputs = sorted(str(path.resolve()) for path in output_files)
    file_hashes = {
        str(path.resolve()): _sha256_hex(path)
        for path in sorted(
            [run_dir / "inputs" / "results.csv", per_structure_csv, mixing_csv, Path(args.reference)] + output_files
        )
    }

    manifest = {
        "git_commit": get_git_commit(),
        "timestamp_utc": _iso_utc_now(),
        "run_id": args.run_id,
        "run_dir": str(run_dir.resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gan_mg_version": _gan_mg_package_version() or __version__,
        "inputs": {
            "inputs/results.csv": _file_manifest_entry(run_dir / "inputs" / "results.csv"),
            "derived/per_structure.csv": _file_manifest_entry(per_structure_csv),
            "derived/per_structure_mixing.csv": _file_manifest_entry(mixing_csv),
            f"inputs/{Path(args.reference).name}": _file_manifest_entry(Path(args.reference)),
        },
        "reference": {
            "model": reference_model,
            "energies": {k: v for k, v in reference_energies.__dict__.items() if v is not None},
        },
        "temperatures_K": [float(t) for t in temperatures],
        "outputs": sorted(outputs),
        "file_hashes": file_hashes,
    }

    manifest_path = run_dir / "derived" / "repro_manifest.json"
    write_json(manifest_path, manifest)

    logger.info("Final outputs:")
    for output_path in sorted(outputs + [str(manifest_path.resolve())]):
        logger.info("%s", output_path)


def _resolve_reference_input(run_dir: Path) -> Path:
    ref_json = run_dir / "inputs" / "reference.json"
    ref_toml = run_dir / "inputs" / "reference.toml"
    if ref_json.exists():
        return ref_json
    if ref_toml.exists():
        return ref_toml
    raise SystemExit(
        "Core input missing: expected runs/<id>/inputs/reference.json or runs/<id>/inputs/reference.toml"
    )


def _write_repropack_index(index_path: Path, run_id: str, copied_files: list[Path], git_sha: str | None) -> None:
    temps_hint = "<temps>"
    descriptions = {
        "inputs/results.csv": "Core generated/imported per-structure energies used as analysis input.",
        "inputs/reference.json": "Reference energy model/config used for mixing-energy calculation.",
        "inputs/reference.toml": "Reference energy model/config used for mixing-energy calculation.",
        "derived/per_structure.csv": "Derived per-structure table with mechanism labels and composition metadata.",
        "derived/per_structure_mixing.csv": "Derived mixing-energy table including E_mix and dE columns.",
        "derived/gibbs_summary.csv": "Boltzmann Gibbs summary by mechanism, composition, and temperature.",
        "derived/all_mechanisms_gibbs_summary.csv": "Gibbs summary across all mechanisms used for overlays/crossover.",
        "derived/mechanism_crossover.csv": "Mechanism crossover table with ΔG = Gmix(vn) - Gmix(mgi).",
        "derived/repro_manifest.json": "Reproducibility manifest with environment metadata and file hashes.",
    }

    lines = [
        f"# Repro Pack: {run_id}",
        "",
        "This pack contains core inputs and available derived artifacts for reproducibility.",
        "",
        "## Included files",
        "",
    ]
    for rel_path in sorted(str(path.as_posix()) for path in copied_files):
        description = descriptions.get(rel_path, "Generated figure artifact.")
        lines.append(f"- `{rel_path}`: {description}")
    lines.extend(
        [
            "",
            "## Rerun",
            "",
            "```bash",
            f"ganmg reproduce overlay --run-id {run_id} --reference inputs/reference.json --temps {temps_hint}",
            "```",
            "",
            "Adjust `--reference` to `inputs/reference.toml` if your pack uses TOML.",
            "",
            "## Provenance",
            "",
            f"- Git SHA: `{git_sha if git_sha else 'unknown'}`",
        ]
    )
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def handle_repropack(args: argparse.Namespace) -> None:
    if args.run_id is None:
        args.run_id = latest_run_id(Path(args.run_dir))

    run_dir = Path(args.run_dir) / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")
    if args.zip_only and not args.zip:
        raise SystemExit("--zip-only requires --zip")

    core_results = run_dir / "inputs" / "results.csv"
    if not core_results.exists():
        raise SystemExit("Core input missing: expected runs/<id>/inputs/results.csv")
    reference_path = _resolve_reference_input(run_dir)

    destination_root = Path(args.out) / args.run_id
    destination_root.mkdir(parents=True, exist_ok=True)

    candidate_files = [
        core_results,
        reference_path,
        run_dir / "derived" / "per_structure.csv",
        run_dir / "derived" / "per_structure_mixing.csv",
        run_dir / "derived" / "gibbs_summary.csv",
        run_dir / "derived" / "all_mechanisms_gibbs_summary.csv",
        run_dir / "derived" / "mechanism_crossover.csv",
        run_dir / "derived" / "repro_manifest.json",
    ]
    candidate_files.extend(sorted((run_dir / "figures").glob("*.png")))

    copied_relpaths: list[Path] = []
    for source in candidate_files:
        if not source.exists():
            continue
        rel = source.relative_to(run_dir)
        dest = destination_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied_relpaths.append(rel)

    git_sha = None
    manifest_path = run_dir / "derived" / "repro_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            git_sha = payload.get("git_commit")
        except json.JSONDecodeError:
            git_sha = None

    _write_repropack_index(destination_root / "index.md", args.run_id, copied_relpaths, git_sha)
    logger.info("Wrote: %s", destination_root / "index.md")

    if args.zip:
        zip_path = Path(args.out) / f"{args.run_id}.zip"
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(destination_root.rglob("*")):
                if item.is_dir():
                    continue
                archive.write(item, item.relative_to(Path(args.out)))
        logger.info("Wrote: %s", zip_path)
        if args.zip_only:
            shutil.rmtree(destination_root)
            logger.info("Removed unpacked repro pack directory: %s", destination_root)


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
    build_derive_parser(subparsers)
    build_mix_parser(subparsers)
    build_gibbs_parser(subparsers)
    build_uncertainty_parser(subparsers)
    build_phase_map_parser(subparsers)
    build_reproduce_parser(subparsers)
    build_repropack_parser(subparsers)
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
    elif args.command == "derive":
        handle_derive(args)
    elif args.command == "mix":
        handle_mix(args)
    elif args.command == "gibbs":
        handle_gibbs(args)
    elif args.command == "uncertainty":
        handle_uncertainty(args)
    elif args.command == "phase-map":
        handle_phase_map(args)
    elif args.command == "reproduce":
        if args.reproduce_command == "overlay":
            handle_reproduce_overlay(args)
        else:
            parser.print_help()
    elif args.command == "repropack":
        handle_repropack(args)
    elif args.command == "bench":
        handle_bench(args, bench_parser)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
