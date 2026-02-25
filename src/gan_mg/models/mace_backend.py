from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from gan_mg.models.base import EnergyModel, StructureConfig


@dataclass(frozen=True)
class MACEBackend(EnergyModel):
    """Placeholder backend for Janus-core/MACE-powered energy evaluation."""

    dependency_hint: str = "janus-core with MACE-enabled calculator dependencies"

    def evaluate(self, configs: Sequence[StructureConfig]) -> list[float]:
        raise NotImplementedError(
            "The 'mace' backend is a placeholder and is not wired into this local demo "
            f"workflow yet. Install and configure {self.dependency_hint} on HPC, then "
            "import generated energies with `ganmg import`."
        )
