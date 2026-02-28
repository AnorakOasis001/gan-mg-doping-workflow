from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from gan_mg.analysis.thermo import ThermoDiagnostics, ThermoSweepRow


def build_diagnostics_payload(
    diagnostics: ThermoDiagnostics,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "temperature_K": diagnostics.temperature_K,
        "num_configurations": diagnostics.num_configurations,
        "expected_energy_eV": diagnostics.expected_energy_eV,
        "energy_variance_eV2": diagnostics.energy_variance_eV2,
        "energy_std_eV": diagnostics.energy_std_eV,
        "p_min": diagnostics.p_min,
        "effective_sample_size": diagnostics.effective_sample_size,
        "logZ_shifted": diagnostics.logZ_shifted,
        "logZ_absolute": diagnostics.logZ_absolute,
        "notes": diagnostics.notes,
    }
    if provenance is not None:
        payload["provenance"] = provenance
    return payload


def _normalize_sweep_row(row: dict[str, Any] | ThermoSweepRow) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if is_dataclass(row):
        return asdict(row)
    row_dict = getattr(row, "__dict__", None)
    if isinstance(row_dict, dict):
        return row_dict
    return {
        "temperature_K": row.temperature_K,
        "num_configurations": row.num_configurations,
        "mixing_energy_min_eV": row.mixing_energy_min_eV,
        "mixing_energy_avg_eV": row.mixing_energy_avg_eV,
        "partition_function": row.partition_function,
        "free_energy_mix_eV": row.free_energy_mix_eV,
    }


def build_metrics_sweep_payload(
    rows: list[dict[str, Any]] | list[ThermoSweepRow],
    reproducibility_hash: str | None,
    created_at: str,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    normalized_rows = [_normalize_sweep_row(row) for row in rows]

    payload: dict[str, Any] = {
        "temperature_grid_K": [row["temperature_K"] for row in normalized_rows],
        "num_temperatures": len(normalized_rows),
        "reproducibility_hash": reproducibility_hash,
        "created_at": created_at,
    }
    if timings:
        payload["timings"] = timings
    return payload
