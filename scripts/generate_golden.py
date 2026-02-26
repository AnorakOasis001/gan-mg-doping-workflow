from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gan_mg.analysis.thermo import ThermoResult, boltzmann_thermo_from_csv
from gan_mg.validation import validate_output

STABLE_THERMO_FIELDS = (
    "temperature_K",
    "num_configurations",
    "mixing_energy_min_eV",
    "mixing_energy_avg_eV",
    "partition_function",
    "free_energy_mix_eV",
)


def thermo_result_to_stable_dict(result: ThermoResult) -> dict[str, float | int]:
    result_dict = asdict(result)
    return {field: result_dict[field] for field in STABLE_THERMO_FIELDS}


def generate_golden_outputs(
    input_dir: Path,
    expected_dir: Path,
    temperature: float = 300.0,
    overwrite: bool = False,
) -> list[Path]:
    input_dir = Path(input_dir)
    expected_dir = Path(expected_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    expected_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        result = boltzmann_thermo_from_csv(csv_path, T=temperature, energy_col="energy_eV")
        stable_result = thermo_result_to_stable_dict(result)
        validate_output(stable_result, kind="thermo_summary")

        output_path = expected_dir / f"{csv_path.stem}.json"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Expected output already exists: {output_path}. "
                "Pass --overwrite to replace existing files."
            )

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(stable_result, handle, indent=2, sort_keys=True)
            handle.write("\n")

        written_files.append(output_path)

    return written_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic golden thermodynamic JSON outputs from "
            "CSV inputs in data/golden/v1/inputs/."
        )
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in Kelvin used for boltzmann_thermo_from_csv (default: 300.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in data/golden/v1/expected/.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path("data/golden/v1/inputs")
    expected_dir = Path("data/golden/v1/expected")

    written_files = generate_golden_outputs(
        input_dir=input_dir,
        expected_dir=expected_dir,
        temperature=args.temperature,
        overwrite=args.overwrite,
    )
    print(f"Generated {len(written_files)} golden JSON files in {expected_dir}.")


if __name__ == "__main__":
    main()
