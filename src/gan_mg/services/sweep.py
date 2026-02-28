from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gan_mg.artifacts import write_json, write_thermo_vs_T_csv
from gan_mg.analysis.thermo import ThermoSweepRow, sweep_thermo_from_csv
from gan_mg.payloads import build_metrics_sweep_payload

REQUIRED_SWEEP_KEYS = (
    "temperature_K",
    "num_configurations",
    "mixing_energy_min_eV",
    "mixing_energy_avg_eV",
    "partition_function",
    "free_energy_mix_eV",
)

RowLike = Mapping[str, Any] | ThermoSweepRow


@dataclass(frozen=True)
class SweepArtifacts:
    rows: list[ThermoSweepRow]
    thermo_vs_t_path: Path
    metrics_sweep_path: Path


def sweep_rows_to_dicts(rows: Sequence[RowLike]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            normalized.append(dict(row))
            continue
        if is_dataclass(row):
            normalized.append(asdict(row))
            continue

        normalized.append({key: getattr(row, key) for key in REQUIRED_SWEEP_KEYS})

    return normalized


def sweep_run(
    *,
    csv_path: Path,
    t_values: list[float],
    energy_col: str,
    thermo_vs_t_path: Path,
    metrics_sweep_path: Path,
    reproducibility_hash: str | None,
    created_at: str,
    timings: dict[str, float] | None,
) -> SweepArtifacts:
    rows = sweep_thermo_from_csv(csv_path, t_values, energy_col=energy_col)
    row_dicts = sweep_rows_to_dicts(rows)

    write_thermo_vs_T_csv(row_dicts, thermo_vs_t_path)

    metrics_payload = build_metrics_sweep_payload(
        row_dicts,
        reproducibility_hash=reproducibility_hash,
        created_at=created_at,
        timings=timings,
    )
    write_json(metrics_sweep_path, metrics_payload)

    return SweepArtifacts(
        rows=rows,
        thermo_vs_t_path=thermo_vs_t_path,
        metrics_sweep_path=metrics_sweep_path,
    )
