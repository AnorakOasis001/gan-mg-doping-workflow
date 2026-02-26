from __future__ import annotations

import csv
from pathlib import Path

from gan_mg.demo.generate import generate_demo_csv


def _read_energy_column(csv_path: Path) -> list[float]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return [float(row["energy_eV"]) for row in csv.DictReader(f)]


def test_generate_is_deterministic_for_fixed_seed_demo_and_toy(tmp_path: Path) -> None:
    demo_1 = tmp_path / "demo_1.csv"
    demo_2 = tmp_path / "demo_2.csv"
    toy_1 = tmp_path / "toy_1.csv"
    toy_2 = tmp_path / "toy_2.csv"

    generate_demo_csv(n=8, seed=17, out_csv=demo_1, model_name="demo")
    generate_demo_csv(n=8, seed=17, out_csv=demo_2, model_name="demo")
    generate_demo_csv(n=8, seed=17, out_csv=toy_1, model_name="toy")
    generate_demo_csv(n=8, seed=17, out_csv=toy_2, model_name="toy")

    assert demo_1.read_text(encoding="utf-8") == demo_2.read_text(encoding="utf-8")
    assert toy_1.read_text(encoding="utf-8") == toy_2.read_text(encoding="utf-8")


def test_models_produce_different_energy_sequences(tmp_path: Path) -> None:
    demo_csv = tmp_path / "demo.csv"
    toy_csv = tmp_path / "toy.csv"

    generate_demo_csv(n=8, seed=5, out_csv=demo_csv, model_name="demo")
    generate_demo_csv(n=8, seed=5, out_csv=toy_csv, model_name="toy")

    assert _read_energy_column(demo_csv) != _read_energy_column(toy_csv)
