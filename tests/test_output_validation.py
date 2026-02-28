from __future__ import annotations

import json
from pathlib import Path

import pytest

from gan_mg.validation import ValidationError, validate_output, validate_output_file


def test_validate_all_golden_outputs() -> None:
    golden_dir = Path("data/golden/v1/expected")
    json_paths = sorted(golden_dir.glob("*.json"))
    assert json_paths

    for json_path in json_paths:
        validate_output_file(json_path)


def test_validation_error_includes_json_path() -> None:
    payload = {
        "temperature_K": 300.0,
        "num_configurations": 10,
        "mixing_energy_min_eV": -0.2,
        "mixing_energy_avg_eV": -0.1,
        "partition_function": 1.0,
    }

    with pytest.raises(ValidationError) as exc:
        validate_output(payload, kind="thermo_summary")

    assert "root.free_energy_mix_eV" in str(exc.value)


def test_validate_run_config_outputs(tmp_path: Path) -> None:
    fixture = Path("data/golden/v1/inputs/realistic_small.csv").resolve()
    config = tmp_path / "run_config.toml"
    output_root = tmp_path / "config_output"
    config.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[analyze]",
                f"csv = {json.dumps(str(fixture))}",
                "T = 300.0",
                "diagnostics = true",
                f"output_root = {json.dumps(str(output_root))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from gan_mg.run_config import run_from_config

    run_from_config(config)

    out_dir = output_root / "results" / "tables"
    validate_output_file(out_dir / "metrics.json")
    validate_output_file(out_dir / "diagnostics_T300.json")
    validate_output_file(out_dir / "run_manifest.json")


@pytest.mark.parametrize(
    ("kind", "payload", "missing_key", "expected_path"),
    [
        (
            "metrics",
            {
                "temperature_K": 300.0,
                "num_configurations": 3,
                "mixing_energy_min_eV": -0.2,
                "mixing_energy_avg_eV": -0.1,
                "partition_function": 1.1,
                "free_energy_mix_eV": -0.11,
                "created_at": "2025-01-01T00:00:00+00:00",
                "reproducibility_hash": "abc123",
                "provenance": {
                    "schema_version": "1.1",
                    "git_commit": None,
                    "python_version": "3.12.0",
                    "platform": "linux",
                    "cli_args": {"T": 300.0},
                    "input_hash": "deadbeef",
                },
            },
            "input_hash",
            "root.provenance.input_hash",
        ),
        (
            "diagnostics",
            {
                "temperature_K": 300.0,
                "num_configurations": 3,
                "expected_energy_eV": -0.2,
                "energy_variance_eV2": 0.01,
                "energy_std_eV": 0.1,
                "p_min": 0.2,
                "effective_sample_size": 2.7,
                "logZ_shifted": 0.5,
                "logZ_absolute": -1.2,
                "notes": [],
                "provenance": {
                    "schema_version": "1.1",
                    "git_commit": None,
                    "python_version": "3.12.0",
                    "platform": "linux",
                    "cli_args": {"T": 300.0},
                    "input_hash": "deadbeef",
                },
            },
            "git_commit",
            "root.provenance.git_commit",
        ),
        (
            "run_manifest",
            {
                "config_sha256": "a" * 64,
                "config_schema_version": 1,
                "git_commit": None,
                "input_csv_sha256": "b" * 64,
                "package_version": "0.1.0",
                "platform": "linux",
                "python_version": "3.12.0",
                "produced_outputs": ["/tmp/metrics.json"],
            },
            "git_commit",
            "root.git_commit",
        ),
    ],
)
def test_output_contract_rejects_missing_required_keys(
    kind: str,
    payload: dict[str, object],
    missing_key: str,
    expected_path: str,
) -> None:
    if kind in {"metrics", "diagnostics"}:
        provenance = dict(payload["provenance"])
        provenance.pop(missing_key)
        payload = dict(payload)
        payload["provenance"] = provenance
    else:
        payload = dict(payload)
        payload.pop(missing_key)

    with pytest.raises(ValidationError, match=expected_path):
        validate_output(payload, kind=kind)
