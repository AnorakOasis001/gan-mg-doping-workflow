from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gan_mg import ThermoResult
from gan_mg.api import analyze_from_csv, analyze_run, sweep_from_csv, sweep_run
from gan_mg.run import init_run, write_run_meta


def _write_results_csv(csv_path: Path) -> None:
    csv_path.write_text(
        "structure_id,mechanism,energy_eV\n"
        "demo_0001,MgGa+VN,-1.50\n"
        "demo_0002,MgGa+VN,-1.20\n"
        "demo_0003,MgGa+VN,-1.35\n",
        encoding="utf-8",
    )


@pytest.fixture
def results_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "results.csv"
    _write_results_csv(csv_path)
    return csv_path


def test_api_analyze_and_sweep_from_csv(
    results_csv: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("pandas")

    result = analyze_from_csv(results_csv, temperature_K=600.0)
    sweep = sweep_from_csv(results_csv, temperatures_K=[900.0, 300.0, 600.0])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    assert isinstance(result, ThermoResult)
    assert result.temperature_K == pytest.approx(600.0)
    assert result.num_configurations == 3
    assert result.mixing_energy_min_eV == pytest.approx(-1.50)

    assert isinstance(sweep, list)
    assert [row.temperature_K for row in sweep] == [300.0, 600.0, 900.0]
    assert all(row.num_configurations == 3 for row in sweep)


def test_api_run_helpers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("pandas")

    run_root = tmp_path / "runs"
    run_id = "demo_run"
    run_paths = init_run(run_root, run_id)
    results_csv = run_paths.inputs_dir / "results.csv"
    _write_results_csv(results_csv)
    write_run_meta(
        run_paths.meta_path,
        {
            "command": "generate",
            "run_id": run_id,
            "inputs_csv": str(results_csv),
        },
    )

    single = analyze_run(run_dir=run_root, run_id=run_id, temperature_K=1000.0)
    out_csv = sweep_run(
        run_dir=run_root,
        run_id=run_id,
        temperatures_K=[300.0, 600.0, 900.0],
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    assert isinstance(single, ThermoResult)
    assert single.temperature_K == pytest.approx(1000.0)

    assert out_csv == run_paths.outputs_dir / "thermo_vs_T.csv"
    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["temperature_K"]) for row in rows] == [300.0, 600.0, 900.0]


def test_public_package_exports_api_symbols() -> None:
    from gan_mg import __all__

    assert "analyze_from_csv" in __all__
    assert "sweep_from_csv" in __all__
    assert "analyze_run" in __all__
    assert "sweep_run" in __all__
    assert "ThermoResult" in __all__
