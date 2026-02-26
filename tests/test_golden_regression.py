import json
import math
from pathlib import Path
from typing import Any

from gan_mg.analysis.thermo import ThermoResult, boltzmann_thermo_from_csv

GOLDEN_INPUT_DIR = Path("data/golden/v1/inputs")
GOLDEN_EXPECTED_DIR = Path("data/golden/v1/expected")
STABLE_KEYS = (
    "temperature_K",
    "num_configurations",
    "mixing_energy_min_eV",
    "mixing_energy_avg_eV",
    "partition_function",
    "free_energy_mix_eV",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def thermo_result_to_dict(result: ThermoResult) -> dict[str, float | int]:
    return {
        "temperature_K": result.temperature_K,
        "num_configurations": result.num_configurations,
        "mixing_energy_min_eV": result.mixing_energy_min_eV,
        "mixing_energy_avg_eV": result.mixing_energy_avg_eV,
        "partition_function": result.partition_function,
        "free_energy_mix_eV": result.free_energy_mix_eV,
    }


def assert_dict_close(
    expected: dict[str, Any],
    actual: dict[str, float | int],
    *,
    rtol: float,
    atol: float,
    context: str,
) -> None:
    expected_keys = set(expected)
    actual_keys = set(actual)
    assert expected_keys == actual_keys, (
        f"{context}: key mismatch; expected keys {sorted(expected_keys)}, "
        f"got {sorted(actual_keys)}"
    )

    for field in sorted(expected_keys):
        expected_value = expected[field]
        actual_value = actual[field]

        if isinstance(expected_value, bool) or isinstance(actual_value, bool):
            raise AssertionError(f"{context}: field '{field}' must be numeric, got booleans")

        if isinstance(expected_value, int):
            assert isinstance(actual_value, int), (
                f"{context}: field '{field}' expected int, got {type(actual_value).__name__}"
            )
            assert actual_value == expected_value, (
                f"{context}: field '{field}' mismatch; expected {expected_value}, got {actual_value}"
            )
            continue

        if isinstance(expected_value, (int, float)):
            assert isinstance(actual_value, (int, float)), (
                f"{context}: field '{field}' expected numeric, got {type(actual_value).__name__}"
            )
            assert math.isfinite(float(expected_value)), (
                f"{context}: field '{field}' expected non-finite value {expected_value}"
            )
            assert math.isfinite(float(actual_value)), (
                f"{context}: field '{field}' actual non-finite value {actual_value}"
            )
            assert math.isclose(float(actual_value), float(expected_value), rel_tol=rtol, abs_tol=atol), (
                f"{context}: field '{field}' mismatch; expected {expected_value}, got {actual_value} "
                f"(rtol={rtol}, atol={atol})"
            )
            continue

        raise AssertionError(
            f"{context}: field '{field}' expected unsupported type {type(expected_value).__name__}"
        )


def test_golden_regression_thermo_outputs_match_expected() -> None:
    csv_paths = sorted(GOLDEN_INPUT_DIR.glob("*.csv"))
    assert csv_paths, f"No golden CSV files found in {GOLDEN_INPUT_DIR}"

    for csv_path in csv_paths:
        expected_path = GOLDEN_EXPECTED_DIR / f"{csv_path.stem}.json"
        assert expected_path.exists(), (
            f"Missing expected golden JSON for '{csv_path.name}': {expected_path}. "
            "Regenerate expectations with: python scripts/generate_golden.py --overwrite"
        )

        expected_raw = load_json(expected_path)
        assert set(expected_raw) == set(STABLE_KEYS), (
            f"{csv_path.name}: expected JSON must contain exactly stable keys {list(STABLE_KEYS)}, "
            f"got {sorted(expected_raw)}"
        )
        expected = {key: expected_raw[key] for key in STABLE_KEYS}
        actual_result = boltzmann_thermo_from_csv(
            csv_path,
            T=float(expected["temperature_K"]),
            energy_col="energy_eV",
        )
        actual = thermo_result_to_dict(actual_result)

        assert set(actual) == set(STABLE_KEYS), (
            f"{csv_path.name}: actual result keys mismatch stable key set {list(STABLE_KEYS)}"
        )

        assert_dict_close(
            expected,
            actual,
            rtol=1e-12,
            atol=1e-15,
            context=f"{csv_path.name} vs {expected_path.name}",
        )
