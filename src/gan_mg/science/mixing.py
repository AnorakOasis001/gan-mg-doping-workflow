from __future__ import annotations

import csv
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from gan_mg.science.reference import compute_reference_energy, load_reference_config

LOGGER = logging.getLogger(__name__)

PER_STRUCTURE_MIXING_COLUMNS = (
    "structure_id",
    "mechanism_code",
    "mechanism_label",
    "energy_total_eV",
    "energy_reference_eV",
    "energy_mixing_eV",
    "energy_mixing_eV_per_atom",
    "energy_mixing_eV_per_cation",
    "dE_eV",
    "dE_eV_per_atom",
    "dE_eV_per_cation",
    "mg_count",
    "ga_count",
    "n_count",
    "site_count_total",
    "x_mg_cation",
    "doping_level_percent",
    "n_mismatch",
    "relaxed_structure_ref",
)

MIXING_ATHERMAL_SUMMARY_COLUMNS = (
    "mechanism_code",
    "x_mg_cation",
    "doping_level_percent",
    "num_structures",
    "energy_mixing_min_eV",
    "energy_mixing_min_eV_per_atom",
    "energy_mixing_min_eV_per_cation",
    "energy_mixing_mean_eV",
    "energy_mixing_std_eV",
    "dE_max_eV",
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_int(row: dict[str, Any], key: str) -> int:
    return int(str(row[key]))


def _to_float(row: dict[str, Any], key: str) -> float:
    return float(str(row[key]))


def build_per_structure_mixing_rows(run_dir: Path, reference_path: Path | None = None) -> list[dict[str, Any]]:
    run_path = Path(run_dir)
    per_structure_path = run_path / "derived" / "per_structure.csv"
    if not per_structure_path.exists():
        raise FileNotFoundError(
            f"per_structure.csv not found: {per_structure_path}. Run `ganmg derive --run-id <id>` first."
        )

    if reference_path is None:
        reference_path = run_path / "inputs" / "reference.json"

    ref_path = Path(reference_path)
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference config not found: {ref_path}. Please provide runs/<id>/inputs/reference.json (or .toml)."
        )

    model, energies = load_reference_config(ref_path)

    rows = _load_rows(per_structure_path)
    if not rows:
        raise ValueError("per_structure.csv must contain at least one row")

    out_rows: list[dict[str, Any]] = []
    mismatch_warned = False
    for row in rows:
        mg_count = _to_int(row, "mg_count")
        ga_count = _to_int(row, "ga_count")
        n_count = _to_int(row, "n_count")
        site_count_total = _to_int(row, "site_count_total")
        cation_total = mg_count + ga_count

        ref = compute_reference_energy(
            model=model,
            energies=energies,
            ga_count=ga_count,
            mg_count=mg_count,
            n_count=n_count,
            site_count_total=site_count_total,
        )

        if model == "gan_mg3n2" and abs(ref.n_mismatch) > 1e-9 and not mismatch_warned:
            LOGGER.warning(
                "gan_mg3n2 reference has nonzero N mismatch for some structures (first mismatch=%+.6f atoms); "
                "see per_structure_mixing.csv column n_mismatch",
                ref.n_mismatch,
            )
            mismatch_warned = True

        if model == "linear_endmember" and energies.E_Mg3N2_fu is None and not mismatch_warned:
            LOGGER.warning(
                "linear_endmember requested without substituted endmember energy; "
                "falling back to GaN-only baseline"
            )
            mismatch_warned = True

        energy_total = _to_float(row, "energy_total_eV")
        energy_mix = energy_total - ref.energy_reference_eV
        per_atom = energy_mix / site_count_total if site_count_total > 0 else float("nan")
        per_cation = energy_mix / cation_total if cation_total > 0 else float("nan")

        out_rows.append(
            {
                **row,
                "energy_reference_eV": ref.energy_reference_eV,
                "energy_mixing_eV": energy_mix,
                "energy_mixing_eV_per_atom": per_atom,
                "energy_mixing_eV_per_cation": per_cation,
                "dE_eV": 0.0,
                "dE_eV_per_atom": 0.0,
                "dE_eV_per_cation": 0.0,
                "n_mismatch": ref.n_mismatch,
            }
        )

    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in out_rows:
        groups[(str(row["mechanism_code"]), float(row["x_mg_cation"]))].append(row)

    for grouped_rows in groups.values():
        min_mix = min(float(r["energy_mixing_eV"]) for r in grouped_rows)
        for row in grouped_rows:
            d_e = float(row["energy_mixing_eV"]) - min_mix
            site_count_total = _to_int(row, "site_count_total")
            cation_total = _to_int(row, "mg_count") + _to_int(row, "ga_count")
            row["dE_eV"] = d_e
            row["dE_eV_per_atom"] = d_e / site_count_total if site_count_total > 0 else float("nan")
            row["dE_eV_per_cation"] = d_e / cation_total if cation_total > 0 else float("nan")

    out_rows.sort(key=lambda r: (str(r["mechanism_code"]), float(r["x_mg_cation"]), str(r["structure_id"])))
    return out_rows


def build_athermal_summary_rows(per_structure_mixing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in per_structure_mixing_rows:
        grouped[(str(row["mechanism_code"]), float(row["x_mg_cation"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (mechanism_code, x_mg), rows in grouped.items():
        energies = [float(r["energy_mixing_eV"]) for r in rows]
        min_energy = min(energies)
        representative = rows[0]
        site_count = _to_int(representative, "site_count_total")
        cation_count = _to_int(representative, "mg_count") + _to_int(representative, "ga_count")
        min_per_atom = min_energy / site_count if site_count > 0 else float("nan")
        min_per_cation = min_energy / cation_count if cation_count > 0 else float("nan")
        std_energy = statistics.pstdev(energies) if len(energies) > 1 else 0.0

        summary_rows.append(
            {
                "mechanism_code": mechanism_code,
                "x_mg_cation": x_mg,
                "doping_level_percent": float(representative["doping_level_percent"]),
                "num_structures": len(rows),
                "energy_mixing_min_eV": min_energy,
                "energy_mixing_min_eV_per_atom": min_per_atom,
                "energy_mixing_min_eV_per_cation": min_per_cation,
                "energy_mixing_mean_eV": statistics.mean(energies),
                "energy_mixing_std_eV": std_energy,
                "dE_max_eV": max(float(r["dE_eV"]) for r in rows),
            }
        )

    summary_rows.sort(key=lambda r: (str(r["mechanism_code"]), float(r["x_mg_cation"])))
    return summary_rows


def validate_per_structure_mixing_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("per_structure_mixing.csv must contain at least one row")
    for i, row in enumerate(rows, start=2):
        for key in PER_STRUCTURE_MIXING_COLUMNS:
            if key not in row:
                raise ValueError(f"row {i} is missing required column '{key}'.")
        for key in (
            "energy_total_eV",
            "energy_reference_eV",
            "energy_mixing_eV",
            "energy_mixing_eV_per_atom",
            "energy_mixing_eV_per_cation",
            "dE_eV",
            "dE_eV_per_atom",
            "dE_eV_per_cation",
            "x_mg_cation",
            "doping_level_percent",
            "n_mismatch",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"row {i} has non-finite '{key}'.")


def validate_mixing_athermal_summary_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("mixing_athermal_summary.csv must contain at least one row")
    for i, row in enumerate(rows, start=2):
        for key in MIXING_ATHERMAL_SUMMARY_COLUMNS:
            if key not in row:
                raise ValueError(f"row {i} is missing required column '{key}'.")
        if int(row["num_structures"]) < 1:
            raise ValueError(f"row {i} has invalid num_structures")
        for key in (
            "x_mg_cation",
            "doping_level_percent",
            "energy_mixing_min_eV",
            "energy_mixing_min_eV_per_atom",
            "energy_mixing_min_eV_per_cation",
            "energy_mixing_mean_eV",
            "energy_mixing_std_eV",
            "dE_max_eV",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"row {i} has non-finite '{key}'.")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in columns})


def derive_mixing_dataset(run_dir: Path, reference_path: Path | None = None) -> tuple[Path, Path]:
    run_path = Path(run_dir)
    mixing_rows = build_per_structure_mixing_rows(run_path, reference_path=reference_path)
    validate_per_structure_mixing_rows(mixing_rows)

    summary_rows = build_athermal_summary_rows(mixing_rows)
    validate_mixing_athermal_summary_rows(summary_rows)

    mixing_path = run_path / "derived" / "per_structure_mixing.csv"
    summary_path = run_path / "derived" / "mixing_athermal_summary.csv"
    _write_csv(mixing_path, mixing_rows, PER_STRUCTURE_MIXING_COLUMNS)
    _write_csv(summary_path, summary_rows, MIXING_ATHERMAL_SUMMARY_COLUMNS)
    return mixing_path, summary_path
