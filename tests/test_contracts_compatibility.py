from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gan_mg.contracts import (
    CsvContract,
    JsonContract,
    all_csv_contracts,
    all_json_contracts,
)
from gan_mg.validation import validate_csv_contract, validate_json_contract


def _csv_fixture_for_contract(contract: CsvContract, tmp_path: Path) -> Path:
    fixture_map = {
        "derived/gibbs_summary.csv": Path("tests/data/notebook_parity/expected_gibbs_summary.csv"),
        "derived/mechanism_crossover.csv": Path("tests/data/notebook_parity/expected_mechanism_crossover.csv"),
    }
    fixture = fixture_map.get(contract.name)
    if fixture is not None:
        return fixture

    out = tmp_path / contract.name.replace("/", "_")
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(contract.required_columns + contract.optional_columns)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({column: "1" for column in fieldnames})
    return out


@pytest.mark.parametrize("contract", all_csv_contracts(), ids=lambda c: c.name)
def test_csv_contracts_match_representative_artifacts(contract: CsvContract, tmp_path: Path) -> None:
    csv_path = _csv_fixture_for_contract(contract, tmp_path)
    validate_csv_contract(csv_path, contract)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames

    assert fieldnames is not None
    assert set(contract.required_columns).issubset(set(fieldnames))
    assert all(column.strip() != "" for column in contract.required_columns)


_MINIMAL_JSON_BY_CONTRACT: dict[str, dict[str, object]] = {
    "outputs/metrics.json": {
        "temperature_K": 300.0,
        "num_configurations": 5,
        "partition_function": 1.2,
        "free_energy_mix_eV": -0.2,
    },
    "outputs/diagnostics_T*.json": {
        "temperature_K": 300.0,
        "num_configurations": 5,
        "expected_energy_eV": -0.2,
        "energy_variance_eV2": 0.1,
        "energy_std_eV": 0.1,
        "p_min": 0.2,
        "effective_sample_size": 4.8,
        "logZ_shifted": 0.3,
        "logZ_absolute": -1.2,
        "notes": [],
    },
    "derived/repro_manifest.json": {},
    "reports/<run-id>/manifest.json": {
        "run_id": "run-1",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "included_files": ["README.md"],
        "sha256": {"README.md": "a" * 64},
    },
}


@pytest.mark.parametrize("contract", all_json_contracts(), ids=lambda c: c.name)
def test_json_contracts_accept_required_and_extra_keys(contract: JsonContract, tmp_path: Path) -> None:
    payload = dict(_MINIMAL_JSON_BY_CONTRACT[contract.name])

    json_path = tmp_path / contract.name.replace("/", "_").replace("<", "").replace(">", "")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    validate_json_contract(json_path, contract)

    payload_with_extra = dict(payload)
    payload_with_extra["extra_key"] = {"ok": True}
    json_path.write_text(json.dumps(payload_with_extra), encoding="utf-8")
    validate_json_contract(json_path, contract)
