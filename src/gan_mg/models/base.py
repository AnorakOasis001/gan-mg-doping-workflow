from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class StructureConfig:
    """Minimal synthetic configuration metadata used for model evaluation."""

    structure_id: str
    mechanism: str
    descriptor: float


class EnergyModel(Protocol):
    """Interface for energy evaluators that return mixing energies in eV."""

    def evaluate(self, configs: Sequence[Any]) -> list[float]:
        """Return one mixing energy (eV) per input configuration."""
