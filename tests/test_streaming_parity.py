import json
import math
from pathlib import Path
from typing import Any

from gan_mg.analysis.thermo import (
    ThermoResult,
    boltzmann_thermo_from_csv,
    thermo_from_csv_streaming,
)

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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stable_dict(result: ThermoResult) -> dict[str, float | int]:
    return {
        "temperature_K": result.temperature_K,
        "num_configurations": result.num_configurations,
        "mixing_energy_min_eV": result.mixing_energy_min_eV,
        "mixing_energy_avg_eV": result.mixing_energy_avg_eV,
        "partition_function": result.partition_function,
        "free_energy_mix_eV": result.free_energy_mix_eV,
    }


def _assert_close(
    expected: dict[str, float | int],
    actual: dict[str, float | int],
    *,
    context: str,
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> None:
    assert set(expected) == set(STABLE_KEYS), (
        f"{context}: expected keys mismatch; expected stable keys {sorted(STABLE_KEYS)}, got {sorted(expected)}"
    )
    assert set(actual) == set(STABLE_KEYS), (
        f"{context}: actual keys mismatch; expected stable keys {sorted(STABLE_KEYS)}, got {sorted(actual)}"
    )

    for field in STABLE_KEYS:
        expected_value = expected[field]
        actual_value = actual[field]

        if isinstance(expected_value, int):
            assert isinstance(actual_value, int), (
                f"{context}: field '{field}' expected int, got {type(actual_value).__name__}"
            )
            assert actual_value == expected_value, (
                f"{context}: field '{field}' mismatch; expected {expected_value}, got {actual_value}"
            )
            continue

        assert isinstance(expected_value, (int, float)), (
            f"{context}: field '{field}' expected numeric type, got {type(expected_value).__name__}"
        )
        assert isinstance(actual_value, (int, float)), (
            f"{context}: field '{field}' expected numeric type, got {type(actual_value).__name__}"
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


def test_streaming_thermo_matches_in_memory_and_golden() -> None:
    csv_paths = sorted(GOLDEN_INPUT_DIR.glob("*.csv"))
    assert csv_paths, f"No golden CSV files found in {GOLDEN_INPUT_DIR}"

    for csv_path in csv_paths:
        expected_path = GOLDEN_EXPECTED_DIR / f"{csv_path.stem}.json"
        assert expected_path.exists(), f"Missing expected golden JSON for {csv_path.name}: {expected_path}"

        expected_raw = _load_json(expected_path)
        assert set(expected_raw) == set(STABLE_KEYS), (
            f"{csv_path.name}: expected JSON must contain exactly stable keys {list(STABLE_KEYS)}, "
            f"got {sorted(expected_raw)}"
        )
        expected = {key: expected_raw[key] for key in STABLE_KEYS}
        temperature = float(expected["temperature_K"])

        in_mem = boltzmann_thermo_from_csv(csv_path, T=temperature, energy_col="energy_eV")
        stream = thermo_from_csv_streaming(
            csv_path,
            temperature_K=temperature,
            energy_column="energy_eV",
            chunksize=2,
        )

        in_mem_dict = _stable_dict(in_mem)
        stream_dict = _stable_dict(stream)

        _assert_close(
            in_mem_dict,
            stream_dict,
            context=f"{csv_path.name}: in-memory vs streaming",
        )
        _assert_close(
            expected,
            in_mem_dict,
            context=f"{csv_path.name}: golden vs in-memory",
        )
        _assert_close(
            expected,
            stream_dict,
            context=f"{csv_path.name}: golden vs streaming",
        )
