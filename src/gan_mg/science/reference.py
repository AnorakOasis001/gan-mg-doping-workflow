from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gan_mg._toml import load_toml

LOGGER = logging.getLogger(__name__)

SUPPORTED_REFERENCE_MODELS = {"linear_endmember", "gan_mg3n2", "chemical_potentials"}


@dataclass(frozen=True)
class ReferenceEnergies:
    """Reservoir energies for reference baselines.

    Units:
    - E_GaN_fu: eV per GaN formula unit (Ga1N1).
    - E_Mg3N2_fu: eV per Mg3N2 formula unit (Mg3N2).
    - E_Mg_metal_atom: eV per atom for Mg reservoir (optional).
    - E_Ga_metal_atom: eV per atom for Ga reservoir (optional).
    - mu_Ga/mu_Mg/mu_N: eV per atom chemical potentials (optional).
    """

    E_GaN_fu: float
    E_Mg3N2_fu: float | None = None
    E_Mg_metal_atom: float | None = None
    E_Ga_metal_atom: float | None = None
    mu_Ga: float | None = None
    mu_Mg: float | None = None
    mu_N: float | None = None


@dataclass(frozen=True)
class ReferenceComputation:
    energy_reference_eV: float
    n_mismatch: float


def _load_reference_payload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        payload = load_toml(path)
    else:
        raise ValueError(f"Unsupported reference config format '{path.suffix}'. Use .json or .toml")

    if not isinstance(payload, dict):
        raise ValueError("Reference config must be a top-level object/table")
    return payload


def load_reference_config(path: Path) -> tuple[str, ReferenceEnergies]:
    cfg = _load_reference_payload(Path(path))
    model = cfg.get("model")
    if not isinstance(model, str) or model not in SUPPORTED_REFERENCE_MODELS:
        supported = ", ".join(sorted(SUPPORTED_REFERENCE_MODELS))
        raise ValueError(f"reference config must define model in [{supported}]")

    energies = cfg.get("energies")
    if not isinstance(energies, dict):
        raise ValueError("reference config must define an object/table 'energies'")

    def _num(name: str, *, required: bool = False) -> float | None:
        value = energies.get(name)
        if value is None:
            if required:
                raise ValueError(f"reference energies is missing required value '{name}'")
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"reference energy '{name}' must be numeric")
        return float(value)

    reference = ReferenceEnergies(
        E_GaN_fu=_num("E_GaN_fu", required=True),
        E_Mg3N2_fu=_num("E_Mg3N2_fu"),
        E_Mg_metal_atom=_num("E_Mg_metal_atom"),
        E_Ga_metal_atom=_num("E_Ga_metal_atom"),
        mu_Ga=_num("mu_Ga"),
        mu_Mg=_num("mu_Mg"),
        mu_N=_num("mu_N"),
    )
    return model, reference


def compute_reference_energy(
    *,
    model: str,
    energies: ReferenceEnergies,
    ga_count: int,
    mg_count: int,
    n_count: int,
    site_count_total: int,
) -> ReferenceComputation:
    """Compute total reference energy (eV per structure/supercell).

    Returns total reference energy and nitrogen-count mismatch diagnostics (atoms).
    """

    cation_total = ga_count + mg_count
    x_mg = (mg_count / cation_total) if cation_total > 0 else 0.0

    if model == "linear_endmember":
        gan_scaled = float(cation_total) * energies.E_GaN_fu
        if energies.E_Mg3N2_fu is None:
            return ReferenceComputation(energy_reference_eV=gan_scaled, n_mismatch=float(n_count - cation_total))

        mg3n2_scaled = (cation_total / 3.0) * energies.E_Mg3N2_fu
        e_ref = (1.0 - x_mg) * gan_scaled + x_mg * mg3n2_scaled
        return ReferenceComputation(energy_reference_eV=e_ref, n_mismatch=float(n_count - cation_total))

    if model == "gan_mg3n2":
        if energies.E_Mg3N2_fu is None:
            raise ValueError("model=gan_mg3n2 requires energies.E_Mg3N2_fu")
        a = float(ga_count)
        b = float(mg_count) / 3.0
        expected_n = a + 2.0 * b
        n_mismatch = float(n_count) - expected_n
        e_ref = a * energies.E_GaN_fu + b * energies.E_Mg3N2_fu
        return ReferenceComputation(energy_reference_eV=e_ref, n_mismatch=n_mismatch)

    if model == "chemical_potentials":
        if energies.mu_Ga is None or energies.mu_Mg is None or energies.mu_N is None:
            raise ValueError("model=chemical_potentials requires mu_Ga, mu_Mg, mu_N in energies")
        e_ref = ga_count * energies.mu_Ga + mg_count * energies.mu_Mg + n_count * energies.mu_N
        return ReferenceComputation(energy_reference_eV=e_ref, n_mismatch=0.0)

    raise ValueError(f"Unsupported reference model '{model}'")
