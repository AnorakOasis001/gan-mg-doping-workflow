from __future__ import annotations

from gan_mg.analysis.thermo import ThermoResult
from gan_mg.api import analyze_from_csv, analyze_run, sweep_from_csv, sweep_run

__version__ = "0.1.0"

__all__ = [
    "ThermoResult",
    "analyze_from_csv",
    "sweep_from_csv",
    "analyze_run",
    "sweep_run",
    "__version__",
]
