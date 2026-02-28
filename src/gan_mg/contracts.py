from __future__ import annotations

from dataclasses import dataclass

from gan_mg.analysis.crossover import CROSSOVER_COLUMNS
from gan_mg.analysis.crossover_uncertainty import CROSSOVER_UNCERTAINTY_COLUMNS
from gan_mg.analysis.phase_boundary import PHASE_BOUNDARY_COLUMNS
from gan_mg.analysis.phase_map import PHASE_MAP_COLUMNS
from gan_mg.science.gibbs import GIBBS_SUMMARY_COLUMNS
from gan_mg.science.uncertainty import UNCERTAINTY_COLUMNS


@dataclass(frozen=True)
class CsvContract:
    name: str
    schema_version: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class JsonContract:
    name: str
    schema_version: str
    required_keys: tuple[str, ...]
    optional_keys: tuple[str, ...] = ()


GIBBS_SUMMARY_CSV_CONTRACT = CsvContract(
    name="derived/gibbs_summary.csv",
    schema_version="1.0",
    required_columns=GIBBS_SUMMARY_COLUMNS,
)

MECHANISM_CROSSOVER_CSV_CONTRACT = CsvContract(
    name="derived/mechanism_crossover.csv",
    schema_version="1.0",
    required_columns=CROSSOVER_COLUMNS,
)

GIBBS_UNCERTAINTY_CSV_CONTRACT = CsvContract(
    name="derived/gibbs_uncertainty.csv",
    schema_version="1.0",
    required_columns=UNCERTAINTY_COLUMNS,
)

CROSSOVER_UNCERTAINTY_CSV_CONTRACT = CsvContract(
    name="derived/crossover_uncertainty.csv",
    schema_version="1.0",
    required_columns=CROSSOVER_UNCERTAINTY_COLUMNS,
)

PHASE_MAP_CSV_CONTRACT = CsvContract(
    name="derived/phase_map.csv",
    schema_version="1.0",
    required_columns=("T_K", "x_mg_cation", "doping_level_percent", "winner_mechanism", "delta_G_eV"),
    optional_columns=tuple(c for c in PHASE_MAP_COLUMNS if c not in {"T_K", "x_mg_cation", "doping_level_percent", "winner_mechanism", "delta_G_eV"}),
)

PHASE_BOUNDARY_CSV_CONTRACT = CsvContract(
    name="derived/phase_boundary.csv",
    schema_version="1.0",
    required_columns=PHASE_BOUNDARY_COLUMNS,
)

METRICS_JSON_CONTRACT = JsonContract(
    name="outputs/metrics.json",
    schema_version="1.0",
    required_keys=(
        "temperature_K",
        "num_configurations",
        "partition_function",
        "free_energy_mix_eV",
    ),
    optional_keys=(
        "mixing_energy_min_eV",
        "mixing_energy_avg_eV",
        "created_at",
        "reproducibility_hash",
        "provenance",
    ),
)

DIAGNOSTICS_JSON_CONTRACT = JsonContract(
    name="outputs/diagnostics_T*.json",
    schema_version="1.0",
    required_keys=(
        "temperature_K",
        "num_configurations",
        "expected_energy_eV",
        "energy_variance_eV2",
        "energy_std_eV",
        "p_min",
        "effective_sample_size",
        "logZ_shifted",
        "logZ_absolute",
        "notes",
    ),
    optional_keys=("provenance",),
)

REPRO_MANIFEST_JSON_CONTRACT = JsonContract(
    name="derived/repro_manifest.json",
    schema_version="1.0",
    required_keys=(),
    optional_keys=(
        "created_at",
        "reproducibility_hash",
        "run_id",
        "python_version",
        "platform",
        "git_commit",
        "input_hash",
    ),
)

REPORT_MANIFEST_JSON_CONTRACT = JsonContract(
    name="reports/<run-id>/manifest.json",
    schema_version="1.0",
    required_keys=("run_id", "generated_at", "included_files", "sha256"),
    optional_keys=("source_run_dir", "files", "file_hashes"),
)


def all_csv_contracts() -> tuple[CsvContract, ...]:
    return (
        GIBBS_SUMMARY_CSV_CONTRACT,
        MECHANISM_CROSSOVER_CSV_CONTRACT,
        GIBBS_UNCERTAINTY_CSV_CONTRACT,
        CROSSOVER_UNCERTAINTY_CSV_CONTRACT,
        PHASE_MAP_CSV_CONTRACT,
        PHASE_BOUNDARY_CSV_CONTRACT,
    )


def all_json_contracts() -> tuple[JsonContract, ...]:
    return (
        METRICS_JSON_CONTRACT,
        DIAGNOSTICS_JSON_CONTRACT,
        REPRO_MANIFEST_JSON_CONTRACT,
        REPORT_MANIFEST_JSON_CONTRACT,
    )
