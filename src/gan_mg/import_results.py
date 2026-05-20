from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias

from gan_mg.analysis.thermo import REQUIRED_RESULTS_COLUMNS
from gan_mg.analysis.thermo import read_energies_csv

RowValue: TypeAlias = str | float
CanonicalRow: TypeAlias = dict[str, RowValue]

_RELAXED_CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "structure_id": ("structure_id", "config_id", "configuration_id", "relaxed_configuration_id", "id"),
    "mechanism": ("mechanism", "mechanism_label", "defect_mechanism", "channel"),
    "energy_eV": ("energy_eV", "total_energy_eV", "relaxed_energy_eV", "energy"),
}
_OPTIONAL_RAW_COLUMNS = (
    "mg_count",
    "ga_count",
    "n_count",
    "relaxed_structure_ref",
    "input_structure_name",
    "relaxed_structure_name",
    "n_atoms",
)


def _timestamp_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_csv_results_schema(csv_path: Path) -> tuple[list[CanonicalRow], list[str]]:
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


def _canonicalize_relaxed_configuration_rows(rows: list[CanonicalRow], fieldnames: list[str]) -> list[CanonicalRow]:
    normalized_to_original = {name.strip().lower(): name for name in fieldnames}
    resolved_cols: dict[str, str] = {}

    for target, aliases in _RELAXED_CONFIG_ALIASES.items():
        for alias in aliases:
            candidate = normalized_to_original.get(alias.strip().lower())
            if candidate is not None:
                resolved_cols[target] = candidate
                break

    missing = [key for key in REQUIRED_RESULTS_COLUMNS if key not in resolved_cols]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"CSV schema error: missing required columns: {missing_str}")

    canonical_rows: list[CanonicalRow] = []
    for i, row in enumerate(rows, start=2):
        canonical: CanonicalRow = {}
        for key in REQUIRED_RESULTS_COLUMNS:
            raw = row.get(resolved_cols[key], "")
            value = "" if raw is None else str(raw).strip()
            if value == "":
                raise ValueError(f"CSV schema error: row {i} has empty '{key}'.")
            canonical[key] = value
        for optional_key in _OPTIONAL_RAW_COLUMNS:
            source_col = normalized_to_original.get(optional_key)
            if source_col is None:
                continue
            raw_optional = row.get(source_col, "")
            if raw_optional is not None and str(raw_optional).strip() != "":
                canonical[optional_key] = str(raw_optional).strip()
        try:
            float(canonical["energy_eV"])
        except ValueError as exc:
            raise ValueError(
                f"CSV schema error: row {i} has non-numeric energy_eV='{canonical['energy_eV']}'."
            ) from exc
        canonical_rows.append(canonical)

    return canonical_rows


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def extxyz_to_results_rows(extxyz_path: Path) -> list[CanonicalRow]:
    lines = extxyz_path.read_text(encoding="utf-8").splitlines()
    idx = 0
    frame = 0
    rows: list[CanonicalRow] = []

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


def write_results_csv(
    rows: Sequence[Mapping[str, RowValue]],
    out_csv: Path,
    columns: tuple[str, ...] | None = None,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns) if columns is not None else list(REQUIRED_RESULTS_COLUMNS)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def import_results_to_run(run_dir: Path, source_path: Path) -> dict[str, str]:
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
        try:
            rows, fieldnames = _validate_csv_results_schema(source_path)
        except ValueError as exc:
            if "missing required columns" not in str(exc):
                raise
            with source_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = [dict(row) for row in reader]
                fieldnames = [] if reader.fieldnames is None else list(reader.fieldnames)
            rows = _canonicalize_relaxed_configuration_rows(rows, fieldnames)
        else:
            if tuple(REQUIRED_RESULTS_COLUMNS) != tuple(fieldnames[: len(REQUIRED_RESULTS_COLUMNS)]):
                rows = _canonicalize_relaxed_configuration_rows(rows, fieldnames)
        canonical_csv = inputs_dir / "results.csv"
        if canonical_csv.exists():
            results_csv = inputs_dir / "imported_results.csv"
        else:
            results_csv = canonical_csv
        passthrough_columns = tuple(column for column in _OPTIONAL_RAW_COLUMNS if any(column in row for row in rows))
        write_results_csv(rows, results_csv, columns=tuple(REQUIRED_RESULTS_COLUMNS) + passthrough_columns)
        # Reuse canonical CSV validation logic used by thermo analysis.
        read_energies_csv(results_csv, energy_col="energy_eV")
    elif ext in {".extxyz", ".xyz"}:
        rows = extxyz_to_results_rows(source_path)
        canonical_csv = inputs_dir / "results.csv"
        if canonical_csv.exists():
            results_csv = inputs_dir / "imported_results.csv"
        else:
            results_csv = canonical_csv
        write_results_csv(rows, results_csv)
    else:
        raise ValueError(
            "Unsupported import format. Supported files: .csv, .extxyz, .xyz"
        )

    metadata = {
        "source_path": str(source_path),
        "imported_at": _timestamp_utc_iso(),
        "sha256": _sha256_hex(source_path),
        "results_csv": str(results_csv),
        "format": ext,
    }
    metadata_path = inputs_dir / "import.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        **metadata,
        "metadata_path": str(metadata_path),
    }
