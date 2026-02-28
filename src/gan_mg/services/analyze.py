from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gan_mg.artifacts import write_json
from gan_mg.analysis.thermo import write_thermo_txt
from gan_mg.io.results_csv import (
    diagnostics_from_csv_streaming,
    read_energies_csv,
    thermo_from_csv_streaming,
)
from gan_mg.payloads import build_diagnostics_payload
from gan_mg.science.thermo import (
    ThermoResult,
    boltzmann_diagnostics_from_energies,
    boltzmann_thermo_from_energies,
)


@dataclass(frozen=True)
class AnalyzeArtifacts:
    thermo_result: ThermoResult
    metrics_path: Path
    thermo_path: Path
    diagnostics_path: Path | None


def analyze_run(
    *,
    csv_path: Path,
    metrics_path: Path,
    thermo_path: Path,
    temperature_K: float,
    energy_col: str,
    chunksize: int | None,
    diagnostics: bool,
    diagnostics_path: Path | None,
    reproducibility_hash: str | None,
    created_at: str | None,
    timings: dict[str, float] | None,
    provenance: dict[str, Any] | None,
    include_reproducibility_hash: bool,
) -> AnalyzeArtifacts:
    if chunksize is None:
        energies = read_energies_csv(csv_path, energy_col=energy_col)
        result = boltzmann_thermo_from_energies(energies, T=temperature_K)
    else:
        result = thermo_from_csv_streaming(
            csv_path=csv_path,
            temperature_K=temperature_K,
            energy_column=energy_col,
            chunksize=chunksize,
        )

    metrics: dict[str, Any] = {
        "temperature_K": result.temperature_K,
        "num_configurations": result.num_configurations,
        "mixing_energy_min_eV": result.mixing_energy_min_eV,
        "mixing_energy_avg_eV": result.mixing_energy_avg_eV,
        "partition_function": result.partition_function,
        "free_energy_mix_eV": result.free_energy_mix_eV,
    }
    if created_at is not None:
        metrics["created_at"] = created_at
    if include_reproducibility_hash and reproducibility_hash:
        metrics["reproducibility_hash"] = reproducibility_hash
    if timings:
        metrics["timings"] = timings
    if provenance is not None:
        metrics["provenance"] = provenance

    write_json(metrics_path, metrics)
    write_thermo_txt(result, thermo_path)

    written_diagnostics_path: Path | None = None
    if diagnostics:
        if diagnostics_path is None:
            raise ValueError("diagnostics_path must be provided when diagnostics=True")

        if chunksize is None:
            energies = read_energies_csv(csv_path, energy_col=energy_col)
            diagnostics_result = boltzmann_diagnostics_from_energies(energies, T=temperature_K)
        else:
            diagnostics_result = diagnostics_from_csv_streaming(
                csv_path=csv_path,
                temperature_K=temperature_K,
                energy_column=energy_col,
                chunksize=chunksize,
            )

        diagnostics_payload = build_diagnostics_payload(diagnostics_result, provenance=provenance)
        write_json(diagnostics_path, diagnostics_payload)
        written_diagnostics_path = diagnostics_path

    return AnalyzeArtifacts(
        thermo_result=result,
        metrics_path=metrics_path,
        thermo_path=thermo_path,
        diagnostics_path=written_diagnostics_path,
    )
