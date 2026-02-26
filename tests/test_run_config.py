from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gan_mg.run_config import run_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    return subprocess.run(
        [sys.executable, "-m", "gan_mg.cli", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_run_from_config_matches_cli_minimal_example_and_writes_manifest(tmp_path: Path) -> None:
    pytest.importorskip("pandas")

    fixture_csv = REPO_ROOT / "data" / "golden" / "v1" / "inputs" / "realistic_small.csv"
    cli_root = tmp_path / "cli_run"
    config_root = tmp_path / "config_run"
    cli_root.mkdir()
    config_root.mkdir()

    _run_cli(
        "analyze",
        "--csv",
        str(fixture_csv),
        "--T",
        "300",
        "--diagnostics",
        cwd=cli_root,
    )

    config_path = tmp_path / "run_config.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[analyze]",
                f"csv = {json.dumps(str(fixture_csv))}",
                "T = 300.0",
                'energy_col = "energy_eV"',
                "diagnostics = true",
                f"output_root = {json.dumps(str(config_root))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_from_config(config_path)

    cli_tables = cli_root / "results" / "tables"
    cfg_tables = config_root / "results" / "tables"

    assert (cfg_tables / "run_manifest.json").exists()
    assert (cfg_tables / "metrics.json").exists()
    assert (cfg_tables / "demo_thermo.txt").exists()
    assert (cfg_tables / "diagnostics_T300.json").exists()

    assert (cfg_tables / "demo_thermo.txt").read_text(encoding="utf-8") == (
        cli_tables / "demo_thermo.txt"
    ).read_text(encoding="utf-8")

    cli_metrics = json.loads((cli_tables / "metrics.json").read_text(encoding="utf-8"))
    cfg_metrics = json.loads((cfg_tables / "metrics.json").read_text(encoding="utf-8"))
    for field in (
        "temperature_K",
        "num_configurations",
        "mixing_energy_min_eV",
        "mixing_energy_avg_eV",
        "partition_function",
        "free_energy_mix_eV",
    ):
        assert cfg_metrics[field] == pytest.approx(cli_metrics[field])

    cli_diag = json.loads((cli_tables / "diagnostics_T300.json").read_text(encoding="utf-8"))
    cfg_diag = json.loads((cfg_tables / "diagnostics_T300.json").read_text(encoding="utf-8"))
    for field in (
        "temperature_K",
        "num_configurations",
        "expected_energy_eV",
        "energy_variance_eV2",
        "energy_std_eV",
        "p_min",
        "effective_sample_size",
    ):
        assert cfg_diag[field] == pytest.approx(cli_diag[field])

    manifest = json.loads((cfg_tables / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_schema_version"] == 1
    assert manifest["package_version"]
    assert manifest["python_version"]
    assert manifest["platform"]
    assert manifest["config_sha256"]
    assert manifest["input_csv_sha256"]
    assert sorted(manifest["produced_outputs"]) == [
        str(cfg_tables / "demo_thermo.txt"),
        str(cfg_tables / "diagnostics_T300.json"),
        str(cfg_tables / "metrics.json"),
    ]


def test_cli_config_option_executes_config_run(tmp_path: Path) -> None:
    pytest.importorskip("pandas")

    fixture_csv = REPO_ROOT / "data" / "golden" / "v1" / "inputs" / "realistic_small.csv"
    output_root = tmp_path / "config_cli"
    config_path = tmp_path / "run_config.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "",
                "[analyze]",
                f"csv = {json.dumps(str(fixture_csv))}",
                "T = 300.0",
                f"output_root = {json.dumps(str(output_root))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _run_cli("--config", str(config_path), cwd=tmp_path)

    assert (output_root / "results" / "tables" / "metrics.json").exists()
    assert (output_root / "results" / "tables" / "run_manifest.json").exists()
