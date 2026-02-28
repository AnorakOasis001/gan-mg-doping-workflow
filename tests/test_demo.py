from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _assert_rows_match(actual: list[dict[str, str]], expected: list[dict[str, str]]) -> None:
    assert len(actual) == len(expected)
    assert list(actual[0].keys()) == list(expected[0].keys())

    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert set(actual_row) == set(expected_row)
        for key, expected_value in expected_row.items():
            actual_value = actual_row[key]
            try:
                assert float(actual_value) == pytest.approx(float(expected_value), abs=1e-12)
            except ValueError:
                assert actual_value == expected_value


def test_demo_command_creates_expected_artifacts_and_matches_golden(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_id = "demo-golden-check"

    _run_cli("demo", "--runs-dir", str(runs_dir), "--out-run-id", run_id, cwd=tmp_path)

    run_dir = runs_dir / run_id
    report_dir = tmp_path / "reports" / run_id

    assert (run_dir / "derived" / "gibbs_summary.csv").exists()
    assert (run_dir / "derived" / "mechanism_crossover.csv").exists()
    assert (run_dir / "derived" / "phase_map.csv").exists()
    assert (run_dir / "derived" / "phase_boundary.csv").exists()
    assert (report_dir / "README.md").exists()

    fixture_dir = REPO_ROOT / "tests" / "data" / "demo"
    expected_gibbs = _read_csv(fixture_dir / "expected_gibbs_summary.csv")
    expected_phase_map = _read_csv(fixture_dir / "expected_phase_map.csv")

    actual_gibbs = _read_csv(run_dir / "derived" / "gibbs_summary.csv")
    actual_phase_map = _read_csv(run_dir / "derived" / "phase_map.csv")

    _assert_rows_match(actual_gibbs, expected_gibbs)
    _assert_rows_match(actual_phase_map, expected_phase_map)
