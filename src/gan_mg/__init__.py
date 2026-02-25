from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from gan_mg.api import AnalyzeResponse, SweepResponse, analyze_from_csv, sweep_from_csv


try:
    __version__ = version("gan-mg-doping-workflow")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "AnalyzeResponse",
    "SweepResponse",
    "analyze_from_csv",
    "sweep_from_csv",
    "__version__",
]
