from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from gan_mg.models.base import EnergyModel, StructureConfig


@dataclass(frozen=True)
class DemoEnergyModel(EnergyModel):
    """Existing demo behavior: seeded pseudo-random energies in [-0.8, 0.2] eV."""

    seed: int

    def evaluate(self, configs: Sequence[StructureConfig]) -> list[float]:
        rng = random.Random(self.seed)
        return [round(rng.uniform(-0.8, 0.2), 6) for _ in configs]


@dataclass(frozen=True)
class ToyPairEnergyModel(EnergyModel):
    """Deterministic toy model based on mechanism offsets + pairwise penalty."""

    pair_strength_eV: float = 0.35

    def evaluate(self, configs: Sequence[StructureConfig]) -> list[float]:
        mechanism_offset = {
            "MgGa+VN": -0.45,
            "Mgi+2MgGa": -0.25,
        }
        energies: list[float] = []
        for cfg in configs:
            base = mechanism_offset[cfg.mechanism]
            pair_penalty = self.pair_strength_eV * (cfg.descriptor - 0.5) ** 2
            structure_term = ((sum(ord(ch) for ch in cfg.structure_id) % 17) - 8) * 0.003
            energies.append(round(base + pair_penalty + structure_term, 6))
        return energies


# Backward-compatible aliases
DemoModel = DemoEnergyModel
ToyPairPotentialModel = ToyPairEnergyModel
