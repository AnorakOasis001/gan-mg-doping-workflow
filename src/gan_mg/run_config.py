from __future__ import annotations

import platform
import subprocess
from hashlib import sha256
from pathlib import Path

from gan_mg import __version__
from gan_mg._toml import load_toml
from gan_mg.artifacts import write_json
from gan_mg.services import analyze_run
from gan_mg.validation import validate_output_file

SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _get_git_commit() -> str | None:
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


def run_from_config(config_path: Path) -> None:
    config_path = Path(config_path)
    config = load_toml(config_path)

    version = int(config.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported config schema_version={version}; expected {SCHEMA_VERSION}.")

    analyze = config.get("analyze")
    if not isinstance(analyze, dict):
        raise ValueError("Config must contain an [analyze] table.")

    csv_path = Path(str(analyze.get("csv", "")))
    if not csv_path.is_absolute():
        csv_path = (config_path.parent / csv_path).resolve()
    if not csv_path.exists():
        raise ValueError(f"CSV not found: {csv_path}")

    temperature = float(analyze.get("T", 300.0))
    energy_col = str(analyze.get("energy_col", "energy_eV"))
    chunksize_raw = analyze.get("chunksize")
    chunksize = None if chunksize_raw is None else int(chunksize_raw)
    diagnostics = bool(analyze.get("diagnostics", False))
    output_root = Path(str(analyze.get("output_root", ".")))
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()

    out_tables = output_root / "results" / "tables"
    artifacts = analyze_run(
        csv_path=csv_path,
        metrics_path=out_tables / "metrics.json",
        thermo_path=out_tables / "demo_thermo.txt",
        temperature_K=temperature,
        energy_col=energy_col,
        chunksize=chunksize,
        diagnostics=diagnostics,
        diagnostics_path=out_tables / f"diagnostics_T{int(temperature)}.json",
        reproducibility_hash=None,
        created_at=None,
        timings=None,
        provenance=None,
        include_reproducibility_hash=False,
    )

    validate_output_file(artifacts.metrics_path, kind="metrics")

    produced_outputs = [str(artifacts.metrics_path), str(artifacts.thermo_path)]
    if artifacts.diagnostics_path is not None:
        validate_output_file(artifacts.diagnostics_path, kind="diagnostics")
        produced_outputs.append(str(artifacts.diagnostics_path))

    manifest = {
        "config_sha256": _file_sha256(config_path),
        "config_schema_version": SCHEMA_VERSION,
        "git_commit": _get_git_commit(),
        "input_csv_sha256": _file_sha256(csv_path),
        "package_version": __version__,
        "platform": platform.platform(),
        "produced_outputs": sorted(produced_outputs),
        "python_version": platform.python_version(),
    }
    manifest_path = out_tables / "run_manifest.json"
    write_json(manifest_path, manifest)
    validate_output_file(manifest_path, kind="run_manifest")
