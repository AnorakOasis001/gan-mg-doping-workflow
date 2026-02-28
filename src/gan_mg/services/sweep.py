from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gan_mg.artifacts import write_json
from gan_mg.analysis.thermo import sweep_thermo_from_csv, write_thermo_vs_T_csv
from gan_mg.payloads import build_metrics_sweep_payload


@dataclass(frozen=True)
class SweepArtifacts:
    rows: list[dict[str, Any]]
    thermo_vs_t_path: Path
    metrics_sweep_path: Path


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
    write_thermo_vs_T_csv(rows, thermo_vs_t_path)

    metrics_payload = build_metrics_sweep_payload(
        rows,
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
