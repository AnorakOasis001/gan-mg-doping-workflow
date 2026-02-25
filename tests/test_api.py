from __future__ import annotations

from pathlib import Path

import pytest

from gan_mg import ThermoResult
from gan_mg.api import analyze_from_csv, sweep_from_csv


@pytest.fixture
def results_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "structure_id,mechanism,energy_eV\n"
        "demo_0001,MgGa+VN,-1.50\n"
        "demo_0002,MgGa+VN,-1.20\n"
        "demo_0003,MgGa+VN,-1.35\n",
        encoding="utf-8",
    )
    return csv_path


def test_analyze_from_csv_returns_thermo_result(results_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("pandas")
    result = analyze_from_csv(results_csv, temperature_K=600.0)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    assert isinstance(result, ThermoResult)
    assert result.temperature_K == pytest.approx(600.0)
    assert result.num_configurations == 3
    assert result.mixing_energy_min_eV == pytest.approx(-1.50)


def test_sweep_from_csv_returns_sorted_results(results_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("pandas")
    sweep = sweep_from_csv(results_csv, temperatures=[900.0, 300.0, 600.0])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    assert isinstance(sweep, list)
    assert [row.temperature_K for row in sweep] == [300.0, 600.0, 900.0]
    assert all(row.num_configurations == 3 for row in sweep)


def test_public_package_exports_api_symbols() -> None:
    from gan_mg import __all__

    assert "analyze_from_csv" in __all__
    assert "sweep_from_csv" in __all__
    assert "ThermoResult" in __all__
