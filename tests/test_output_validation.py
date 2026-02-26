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
