from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PER_STRUCTURE_COLUMNS = (
    "structure_id",
    "mechanism_code",
    "mechanism_label",
    "energy_total_eV",
    "mg_count",
    "ga_count",
    "n_count",
    "site_count_total",
    "x_mg_cation",
    "doping_level_percent",
    "relaxed_structure_ref",
)


# Units:
# - energy_total_eV: eV per relaxed supercell
# - mg_count/ga_count/n_count/site_count_total: atom counts per supercell
# - x_mg_cation: unitless fraction (Mg / (Mg + Ga))
# - doping_level_percent: percent = 100 * x_mg_cation

def canonicalize_mechanism(mechanism_label: str) -> str:
    lowered = mechanism_label.strip().lower()
    if "vn" in lowered or "mgga+vn" in lowered:
        return "vn"
    if "mgi" in lowered:
        return "mgi"
    LOGGER.warning("Unknown mechanism label '%s'; setting mechanism_code=unknown", mechanism_label)
    return "unknown"


def _load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        headers = [] if reader.fieldnames is None else list(reader.fieldnames)
    return rows, headers


def _require_nonempty(value: str | None, *, field: str, row_index: int) -> str:
    if value is None or value.strip() == "":
        raise ValueError(f"row {row_index} has empty '{field}'.")
    return value.strip()


def _parse_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)


def _parse_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _normalize_element_symbol(token: str) -> str:
    cleaned = token.strip().strip("\"'")
    if not cleaned:
        return ""
    if len(cleaned) >= 2 and cleaned[1].islower():
        return cleaned[0].upper() + cleaned[1].lower()
    return cleaned[0].upper()


def _count_elements_in_extxyz(path: Path) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise ValueError(f"extxyz parse error: empty file: {path}")

    natoms = int(lines[idx].strip())
    comment_idx = idx + 1
    atom_start = comment_idx + 1
    atom_end = atom_start + natoms
    if atom_end > len(lines):
        raise ValueError(f"extxyz parse error: truncated atoms block: {path}")

    counts: dict[str, int] = {}
    for line in lines[atom_start:atom_end]:
        parts = line.split()
        if not parts:
            continue
        symbol = _normalize_element_symbol(parts[0])
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _count_elements_in_cif(path: Path) -> dict[str, int]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    counts: dict[str, int] = {}
    i = 0
    while i < len(lines):
        if lines[i] != "loop_":
            i += 1
            continue

        i += 1
        headers: list[str] = []
        while i < len(lines) and lines[i].startswith("_"):
            headers.append(lines[i])
            i += 1

        if not headers:
            continue

        symbol_idx = None
        for idx_header, header in enumerate(headers):
            if header in {"_atom_site_type_symbol", "_atom_site_symbol", "_atom_site_label"}:
                symbol_idx = idx_header
                break

        while i < len(lines):
            row = lines[i]
            if not row or row.startswith("#"):
                i += 1
                continue
            if row == "loop_" or row.startswith("_"):
                break
            if symbol_idx is not None:
                parts = row.split()
                if symbol_idx < len(parts):
                    symbol = _normalize_element_symbol(parts[symbol_idx])
                    counts[symbol] = counts.get(symbol, 0) + 1
            i += 1

    if not counts:
        raise ValueError(f"cif parse error: unable to infer atom species counts: {path}")
    return counts


def count_composition_from_structure(path: Path) -> tuple[int, int, int, int]:
    ext = path.suffix.lower()
    if ext == ".extxyz":
        counts = _count_elements_in_extxyz(path)
    elif ext == ".cif":
        counts = _count_elements_in_cif(path)
    else:
        raise ValueError(f"Unsupported structure format '{path.suffix}' for {path}")

    mg_count = counts.get("Mg", 0)
    ga_count = counts.get("Ga", 0)
    n_count = counts.get("N", 0)
    total = sum(counts.values())
    return mg_count, ga_count, n_count, total


def _resolve_structure_path(
    run_dir: Path,
    structure_id: str,
    explicit_ref: str | None,
    manifest_row: dict[str, str] | None,
) -> Path | None:
    if explicit_ref and explicit_ref.strip():
        ref_path = Path(explicit_ref)
        if not ref_path.is_absolute():
            ref_path = (run_dir / ref_path).resolve()
        if ref_path.exists():
            return ref_path

    if manifest_row is not None:
        manifest_path = manifest_row.get("path", "").strip()
        if manifest_path:
            ref_path = Path(manifest_path)
            if not ref_path.is_absolute():
                ref_path = (run_dir / manifest_path).resolve()
            if ref_path.exists():
                return ref_path

    structures_dir = run_dir / "structures"
    for suffix in (".extxyz", ".cif"):
        candidate = structures_dir / f"{structure_id}{suffix}"
        if candidate.exists():
            return candidate

    return None


def build_per_structure_rows(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    results_path = run_dir / "inputs" / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"results.csv not found: {results_path}")

    results_rows, headers = _load_csv_rows(results_path)
    if not results_rows:
        raise ValueError("results.csv must contain at least one row")

    manifest_path = run_dir / "inputs" / "structures.csv"
    manifest_by_id: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        manifest_rows, _ = _load_csv_rows(manifest_path)
        manifest_by_id = {
            row["structure_id"].strip(): row
            for row in manifest_rows
            if row.get("structure_id") and row["structure_id"].strip()
        }

    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(results_rows, start=2):
        structure_id = _require_nonempty(row.get("structure_id"), field="structure_id", row_index=idx)
        mechanism_label = _require_nonempty(row.get("mechanism"), field="mechanism", row_index=idx)
        energy_total_eV = _parse_float(row.get("energy_eV"))
        if energy_total_eV is None:
            raise ValueError(f"row {idx} has empty 'energy_eV'.")

        manifest_row = manifest_by_id.get(structure_id)

        mg_count = _parse_int(row.get("mg_count"))
        ga_count = _parse_int(row.get("ga_count"))
        n_count = _parse_int(row.get("n_count"))

        if mg_count is None or ga_count is None or n_count is None:
            if manifest_row is not None:
                mg_count = mg_count if mg_count is not None else _parse_int(manifest_row.get("mg_count"))
                ga_count = ga_count if ga_count is not None else _parse_int(manifest_row.get("ga_count"))
                n_count = n_count if n_count is not None else _parse_int(manifest_row.get("n_count"))

        explicit_ref = row.get("relaxed_structure_ref")
        structure_path = _resolve_structure_path(run_dir, structure_id, explicit_ref, manifest_row)

        if mg_count is None or ga_count is None or n_count is None:
            if structure_path is None:
                raise ValueError(
                    f"Unable to determine composition for structure_id='{structure_id}': "
                    "provide mg_count/ga_count/n_count or a structure artifact path."
                )
            mg_count, ga_count, n_count, total_atoms = count_composition_from_structure(structure_path)
        else:
            total_atoms = mg_count + ga_count + n_count

        cation_total = mg_count + ga_count
        x_mg_cation = (mg_count / cation_total) if cation_total > 0 else 0.0
        doping_level_percent = 100.0 * x_mg_cation

        out_rows.append(
            {
                "structure_id": structure_id,
                "mechanism_code": canonicalize_mechanism(mechanism_label),
                "mechanism_label": mechanism_label,
                "energy_total_eV": energy_total_eV,
                "mg_count": mg_count,
                "ga_count": ga_count,
                "n_count": n_count,
                "site_count_total": total_atoms,
                "x_mg_cation": x_mg_cation,
                "doping_level_percent": doping_level_percent,
                "relaxed_structure_ref": "" if structure_path is None else str(structure_path),
            }
        )

    out_rows.sort(key=lambda row: str(row["structure_id"]))
    return out_rows


def validate_per_structure_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("per_structure.csv must contain at least one row")

    for i, row in enumerate(rows, start=2):
        for key in PER_STRUCTURE_COLUMNS:
            if key not in row:
                raise ValueError(f"row {i} is missing required column '{key}'.")

        if row["mechanism_code"] not in {"vn", "mgi", "unknown"}:
            raise ValueError(f"row {i} has invalid mechanism_code='{row['mechanism_code']}'.")

        for key in ("energy_total_eV", "x_mg_cation", "doping_level_percent"):
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"row {i} has non-finite '{key}'.")

        for key in ("mg_count", "ga_count", "n_count", "site_count_total"):
            value = int(row[key])
            if value < 0:
                raise ValueError(f"row {i} has negative '{key}'.")


def write_per_structure_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    validate_per_structure_rows(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PER_STRUCTURE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in PER_STRUCTURE_COLUMNS})


def derive_per_structure_dataset(run_dir: Path) -> Path:
    rows = build_per_structure_rows(run_dir)
    out_path = Path(run_dir) / "derived" / "per_structure.csv"
    write_per_structure_csv(rows, out_path)
    return out_path
