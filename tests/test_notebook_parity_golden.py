from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gan_mg.analysis.crossover import CROSSOVER_COLUMNS, build_mechanism_crossover_rows
from gan_mg.science.gibbs import GIBBS_SUMMARY_COLUMNS, build_gibbs_summary_rows
from gan_mg.science.reference import load_reference_config


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _assert_float_columns_close(
    actual_rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...],
    tol: float,
) -> None:
    assert len(actual_rows) == len(expected_rows)
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        for column in columns:
            assert float(actual[column]) == pytest.approx(float(expected[column]), abs=tol)


def test_notebook_parity_golden_pipeline(tmp_path: Path) -> None:
    fixture_dir = Path(__file__).parent / "data" / "notebook_parity"

    per_structure_mixing_rows = _read_csv(fixture_dir / "per_structure_mixing.csv")

    reference_model, _ = load_reference_config(fixture_dir / "reference.json")
    assert reference_model == "gan_mg3n2"

    gibbs_rows = build_gibbs_summary_rows(per_structure_mixing_rows, [1.0, 300.0])
    crossover_rows = build_mechanism_crossover_rows([{k: str(v) for k, v in row.items()} for row in gibbs_rows])

    out_gibbs = tmp_path / "gibbs_summary.csv"
    out_cross = tmp_path / "mechanism_crossover.csv"

    with out_gibbs.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(GIBBS_SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(gibbs_rows)

    with out_cross.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CROSSOVER_COLUMNS))
        writer.writeheader()
        writer.writerows(crossover_rows)

    expected_gibbs_rows = _read_csv(fixture_dir / "expected_gibbs_summary.csv")
    expected_crossover_rows = _read_csv(fixture_dir / "expected_mechanism_crossover.csv")

    actual_gibbs_rows = _read_csv(out_gibbs)
    actual_crossover_rows = _read_csv(out_cross)

    assert list(actual_gibbs_rows[0].keys()) == list(GIBBS_SUMMARY_COLUMNS)
    assert list(actual_crossover_rows[0].keys()) == list(CROSSOVER_COLUMNS)

    assert [
        (row["mechanism_code"], float(row["x_mg_cation"]), float(row["T_K"]))
        for row in actual_gibbs_rows
    ] == [
        (row["mechanism_code"], float(row["x_mg_cation"]), float(row["T_K"]))
        for row in expected_gibbs_rows
    ]

    _assert_float_columns_close(
        actual_gibbs_rows,
        expected_gibbs_rows,
        columns=(
            "energy_mixing_min_eV",
            "logZ",
            "free_energy_mixing_eV",
            "mixing_energy_avg_eV",
            "energy_mixing_min_eV_per_atom",
            "free_energy_mixing_eV_per_atom",
            "mixing_energy_avg_eV_per_atom",
            "energy_mixing_min_eV_per_cation",
            "free_energy_mixing_eV_per_cation",
            "mixing_energy_avg_eV_per_cation",
        ),
        tol=1e-12,
    )

    _assert_float_columns_close(
        actual_crossover_rows,
        expected_crossover_rows,
        columns=("delta_free_energy_eV", "delta_free_energy_eV_per_cation"),
        tol=1e-12,
    )
