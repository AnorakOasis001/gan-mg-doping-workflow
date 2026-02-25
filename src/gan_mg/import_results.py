from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gan_mg.analysis.thermo import REQUIRED_RESULTS_COLUMNS


def _timestamp_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_csv_results_schema(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = [] if reader.fieldnames is None else list(reader.fieldnames)

    if not fieldnames:
        raise ValueError("CSV schema error: file is missing a header row.")

    missing_columns = [column for column in REQUIRED_RESULTS_COLUMNS if column not in fieldnames]
    if missing_columns:
        raise ValueError(
            "CSV schema error: missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    if not rows:
        raise ValueError("CSV schema error: file must contain at least one data row.")

    for i, row in enumerate(rows, start=2):
        for column in REQUIRED_RESULTS_COLUMNS:
            raw = row.get(column)
            if raw is None or str(raw).strip() == "":
                raise ValueError(f"CSV schema error: row {i} has empty '{column}'.")
        try:
            float(row["energy_eV"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"CSV schema error: row {i} has non-numeric energy_eV='{row['energy_eV']}'."
            ) from exc

    return rows, fieldnames


_ENERGY_PATTERNS = (
    re.compile(r"\benergy_eV\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b"),
    re.compile(r"\benergy\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b"),
    re.compile(r"\bE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b"),
)


def _extract_energy_from_comment(comment: str) -> float | None:
    for pattern in _ENERGY_PATTERNS:
        match = pattern.search(comment)
        if match:
            return float(match.group(1))
    return None


def extxyz_to_results_rows(extxyz_path: Path) -> list[dict[str, str | float]]:
    lines = extxyz_path.read_text(encoding="utf-8").splitlines()
    idx = 0
    frame = 0
    rows: list[dict[str, str | float]] = []

    while idx < len(lines):
        natoms_line = lines[idx].strip()
        if not natoms_line:
            idx += 1
            continue
        try:
            natoms = int(natoms_line)
        except ValueError as exc:
            raise ValueError(
                f"extxyz parse error: expected integer atom-count at line {idx + 1}, got '{natoms_line}'."
            ) from exc

        if idx + 1 >= len(lines):
            raise ValueError(f"extxyz parse error: missing comment line after atom-count at line {idx + 1}.")

        comment = lines[idx + 1].strip()
        energy = _extract_energy_from_comment(comment)
        if energy is None:
            raise ValueError(
                "extxyz parse error: missing per-structure energy in comment line "
                f"for frame {frame + 1}. Expected keys like energy=... or energy_eV=...."
            )

        frame += 1
        rows.append(
            {
                "structure_id": f"extxyz_{frame:06d}",
                "mechanism": "external",
                "energy_eV": energy,
            }
        )

        idx += 2 + natoms
        if idx > len(lines):
            raise ValueError(
                f"extxyz parse error: frame {frame} declares {natoms} atoms but file ends early."
            )

    if not rows:
        raise ValueError("extxyz parse error: no structures found.")

    return rows


def write_results_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(REQUIRED_RESULTS_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in REQUIRED_RESULTS_COLUMNS})


def import_results_to_run(run_dir: Path, source_path: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Import path not found or not a file: {source_path}")

    inputs_dir = run_dir / "inputs"
    outputs_dir = run_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ext = source_path.suffix.lower()
    if ext == ".csv":
        _validate_csv_results_schema(source_path)
        results_csv = inputs_dir / "results.csv"
        shutil.copy2(source_path, results_csv)
    elif ext in {".extxyz", ".xyz"}:
        rows = extxyz_to_results_rows(source_path)
        results_csv = inputs_dir / "results.csv"
        write_results_csv(rows, results_csv)
    else:
        raise ValueError(
            "Unsupported import format. Supported files: .csv, .extxyz, .xyz"
        )

    stored_source = inputs_dir / f"external_results{source_path.suffix.lower()}"
    shutil.copy2(source_path, stored_source)

    metadata = {
        "imported_at_utc": _timestamp_utc_iso(),
        "source_path": str(source_path),
        "source_file": source_path.name,
        "stored_source": str(stored_source),
        "results_csv": str(results_csv),
        "format": ext,
    }
    metadata_path = inputs_dir / "import_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
