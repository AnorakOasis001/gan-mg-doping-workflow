from __future__ import annotations

import os
from importlib.util import find_spec


def ensure_agg() -> None:
    """Select a deterministic headless backend before importing pyplot."""
    if not os.environ.get("MPLBACKEND"):
        os.environ["MPLBACKEND"] = "Agg"

    if find_spec("matplotlib") is None:
        return

    import matplotlib

    matplotlib.use("Agg", force=True)
