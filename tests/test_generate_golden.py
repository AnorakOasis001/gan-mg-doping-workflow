import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest

MODULE_PATH = Path("scripts/generate_golden.py")
SPEC = importlib.util.spec_from_file_location("generate_golden", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

STABLE_THERMO_FIELDS = MODULE.STABLE_THERMO_FIELDS
generate_golden_outputs = MODULE.generate_golden_outputs


def test_generate_golden_outputs_writes_expected_json(tmp_path: Path) -> None:
    source_csv = Path("data/golden/v1/inputs/equal_energies.csv")
    input_dir = tmp_path / "inputs"
    expected_dir = tmp_path / "expected"
    input_dir.mkdir()

    copied_csv = input_dir / source_csv.name
    shutil.copy2(source_csv, copied_csv)

    written_files = generate_golden_outputs(
        input_dir=input_dir,
        expected_dir=expected_dir,
        temperature=300.0,
        overwrite=False,
    )

    output_path = expected_dir / "equal_energies.json"
    assert output_path in written_files
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert set(payload.keys()) == set(STABLE_THERMO_FIELDS)
    assert payload["num_configurations"] > 0

    for key in STABLE_THERMO_FIELDS:
        value = payload[key]
        if isinstance(value, float):
            assert math.isfinite(value), f"Field {key} must be finite"


def test_generate_golden_outputs_raises_without_overwrite(tmp_path: Path) -> None:
    source_csv = Path("data/golden/v1/inputs/equal_energies.csv")
    input_dir = tmp_path / "inputs"
    expected_dir = tmp_path / "expected"
    input_dir.mkdir()

    shutil.copy2(source_csv, input_dir / source_csv.name)

    generate_golden_outputs(input_dir=input_dir, expected_dir=expected_dir)

    with pytest.raises(FileExistsError):
        generate_golden_outputs(input_dir=input_dir, expected_dir=expected_dir, overwrite=False)
