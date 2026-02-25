from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from gan_mg.analysis.thermo import ThermoResult
from gan_mg.api import analyze_from_csv, sweep_from_csv


try:
    __version__ = version("gan-mg-doping-workflow")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "ThermoResult",
    "analyze_from_csv",
    "sweep_from_csv",
    "__version__",
]
