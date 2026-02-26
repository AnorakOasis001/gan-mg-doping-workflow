import csv
import math
from pathlib import Path

import pytest

from gan_mg.analysis.thermo import boltzmann_thermo_from_csv

GOLDEN_INPUT_DIR = Path("data/golden/v1/inputs")
EXPECTED_ROW_COUNTS = {
    "dominant_minimum.csv": 6,
    "equal_energies.csv": 5,
    "huge_magnitudes.csv": 6,
    "mixed_sign.csv": 6,
    "near_degenerate.csv": 6,
    "realistic_small.csv": 10,
    "repeated_ids.csv": 6,
    "tiny_magnitudes.csv": 6,
}


@pytest.mark.parametrize("csv_name", sorted(EXPECTED_ROW_COUNTS))
def test_golden_input_produces_thermo_result(csv_name: str) -> None:
    csv_path = GOLDEN_INPUT_DIR / csv_name

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))

    result = boltzmann_thermo_from_csv(csv_path, T=300.0, energy_col="energy_eV")

    assert row_count == EXPECTED_ROW_COUNTS[csv_name]
    assert result.num_configurations == row_count
    assert result.partition_function > 0.0
    assert math.isfinite(result.partition_function)
    assert math.isfinite(result.free_energy_mix_eV)
