from __future__ import annotations

import platform
from dataclasses import asdict, is_dataclass
from typing import Any

SCHEMA_VERSION = "1.1"


def _to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        data = asdict(obj)
        if not isinstance(data, dict):
            raise TypeError("Dataclass payload must serialize to a dictionary.")
        return data
    raise TypeError(f"Expected dict or dataclass instance, got {type(obj).__name__}.")


def build_provenance(*, cli_args: dict[str, Any] | list[Any] | None, input_hash: str, git_commit: str | None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cli_args": {} if cli_args is None else cli_args,
        "input_hash": input_hash,
        "git_commit": git_commit,
    }


def build_metrics_payload(
    thermo_result: Any,
    *,
    reproducibility_hash: str,
    created_at: str,
    provenance: dict[str, Any],
    timings: dict[str, float] | None,
) -> dict[str, Any]:
    thermo = _to_dict(thermo_result)
    payload: dict[str, Any] = {
        "temperature_K": thermo["temperature_K"],
        "num_configurations": thermo["num_configurations"],
        "mixing_energy_min_eV": thermo["mixing_energy_min_eV"],
        "mixing_energy_avg_eV": thermo["mixing_energy_avg_eV"],
        "partition_function": thermo["partition_function"],
        "free_energy_mix_eV": thermo["free_energy_mix_eV"],
        "reproducibility_hash": reproducibility_hash,
        "created_at": created_at,
        "provenance": dict(provenance),
    }
    if timings:
        payload["timings"] = dict(timings)
    return payload


def build_diagnostics_payload(
    diag: Any,
    *,
    reproducibility_hash: str | None,
    provenance: dict[str, Any],
    notes: list[str] | None,
) -> dict[str, Any]:
    _ = reproducibility_hash
    diagnostics = _to_dict(diag)
    payload: dict[str, Any] = {
        "temperature_K": diagnostics["temperature_K"],
        "num_configurations": diagnostics["num_configurations"],
        "expected_energy_eV": diagnostics["expected_energy_eV"],
        "energy_variance_eV2": diagnostics["energy_variance_eV2"],
        "energy_std_eV": diagnostics["energy_std_eV"],
        "p_min": diagnostics["p_min"],
        "effective_sample_size": diagnostics["effective_sample_size"],
        "logZ_shifted": diagnostics["logZ_shifted"],
        "logZ_absolute": diagnostics["logZ_absolute"],
        "notes": diagnostics["notes"] if notes is None else notes,
        "provenance": dict(provenance),
    }
    return payload


def build_metrics_sweep_payload(
    rows: list[dict[str, Any]],
    *,
    reproducibility_hash: str,
    created_at: str,
    timings: dict[str, float] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "temperature_grid_K": [row["temperature_K"] for row in rows],
        "num_temperatures": len(rows),
        "reproducibility_hash": reproducibility_hash,
        "created_at": created_at,
    }
    if timings:
        payload["timings"] = dict(timings)
    return payload
