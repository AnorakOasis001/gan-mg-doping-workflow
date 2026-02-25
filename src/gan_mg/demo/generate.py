from __future__ import annotations

import csv
import random
from pathlib import Path

from gan_mg.models import DemoModel, EnergyModel, StructureConfig, ToyPairPotentialModel


def build_demo_configs(n: int, seed: int) -> list[StructureConfig]:
    rng = random.Random(seed)
    return [
        StructureConfig(
            structure_id=f"demo_{i:04d}",
            mechanism=rng.choice(["MgGa+VN", "Mgi+2MgGa"]),
            descriptor=round(rng.random(), 6),
        )
        for i in range(n)
    ]


def resolve_model(model_name: str, seed: int) -> EnergyModel:
    if model_name == "demo":
        return DemoModel(seed=seed)
    if model_name == "toy":
        return ToyPairPotentialModel()
    raise ValueError(f"unsupported model: {model_name}")


def generate_demo_csv(n: int, seed: int, out_csv: Path, model_name: str = "demo") -> Path:
    """Generate deterministic demo CSV rows with energies from a pluggable model."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    configs = build_demo_configs(n=n, seed=seed)
    model = resolve_model(model_name=model_name, seed=seed)
    energies_eV = model.evaluate(configs)

    rows = [
        {
            "structure_id": cfg.structure_id,
            "mechanism": cfg.mechanism,
            "energy_eV": energy,
        }
        for cfg, energy in zip(configs, energies_eV, strict=True)
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return out_csv
