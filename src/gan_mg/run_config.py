from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from gan_mg import __version__
from gan_mg._toml import load_toml
from gan_mg.analysis.thermo import (
    boltzmann_diagnostics_from_energies,
    boltzmann_thermo_from_csv,
    diagnostics_from_csv_streaming,
    read_energies_csv,
    thermo_from_csv_streaming,
    write_thermo_txt,
)
from gan_mg.payloads import (
    build_diagnostics_payload,
    build_metrics_payload,
    build_provenance,
)
from gan_mg.run import compute_reproducibility_hash
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    thermo_path = out_tables / "demo_thermo.txt"
    metrics_path = out_tables / "metrics.json"
    diagnostics_path = out_tables / f"diagnostics_T{int(temperature)}.json"

    reproducibility_hash = compute_reproducibility_hash(
        input_csv=csv_path,
        temperature_grid=[temperature],
        code_version=__version__,
    )
    cli_args = {
        "config_path": str(config_path),
        "analyze": {
            "csv": str(csv_path),
            "T": temperature,
            "energy_col": energy_col,
            "chunksize": chunksize,
            "diagnostics": diagnostics,
            "output_root": str(output_root),
        },
    }
    provenance = build_provenance(
        cli_args=cli_args,
        input_hash=reproducibility_hash,
        git_commit=_get_git_commit(),
    )

    if chunksize is None:
        result = boltzmann_thermo_from_csv(csv_path, T=temperature, energy_col=energy_col)
    else:
        result = thermo_from_csv_streaming(
            csv_path=csv_path,
            temperature_K=temperature,
            energy_column=energy_col,
            chunksize=chunksize,
        )

    metrics_payload = build_metrics_payload(
        result,
        reproducibility_hash=reproducibility_hash,
        created_at=_iso_utc_now(),
        provenance=provenance,
        timings=None,
    )
    _write_json(metrics_path, metrics_payload)
    validate_output_file(metrics_path, kind="metrics")
    write_thermo_txt(result, thermo_path)

    produced_outputs = [str(metrics_path), str(thermo_path)]

    if diagnostics:
        if chunksize is None:
            energies = read_energies_csv(csv_path, energy_col=energy_col)
            diagnostics_result = boltzmann_diagnostics_from_energies(energies, T=temperature)
        else:
            diagnostics_result = diagnostics_from_csv_streaming(
                csv_path=csv_path,
                temperature_K=temperature,
                energy_column=energy_col,
                chunksize=chunksize,
            )

        diagnostics_payload = build_diagnostics_payload(
            diagnostics_result,
            reproducibility_hash=reproducibility_hash,
            provenance=provenance,
            notes=None,
        )
        _write_json(diagnostics_path, diagnostics_payload)
        validate_output_file(diagnostics_path, kind="diagnostics")
        produced_outputs.append(str(diagnostics_path))

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
    _write_json(manifest_path, manifest)
    validate_output_file(manifest_path, kind="run_manifest")
