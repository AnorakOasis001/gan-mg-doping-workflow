from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gan_mg.analysis.thermo import REQUIRED_RESULTS_COLUMNS, read_energies_csv

GOLDEN_INPUT_DIR = Path("data/golden/v1/inputs")


@pytest.mark.parametrize("csv_path", sorted(GOLDEN_INPUT_DIR.glob("*.csv")), ids=lambda p: p.name)
def test_golden_input_csvs_parse_successfully(csv_path: Path) -> None:
    energies = read_energies_csv(csv_path, energy_col="energy_eV")

    assert len(energies) >= 1
    assert all(isinstance(energy, float) for energy in energies)


@pytest.mark.parametrize("csv_path", sorted(GOLDEN_INPUT_DIR.glob("*.csv")), ids=lambda p: p.name)
def test_golden_input_csvs_match_required_schema(csv_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert tuple(reader.fieldnames) == REQUIRED_RESULTS_COLUMNS
        assert any(reader)
