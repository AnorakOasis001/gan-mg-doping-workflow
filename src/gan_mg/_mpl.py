from __future__ import annotations


def ensure_agg_backend() -> None:
    """Select a deterministic headless backend before importing pyplot."""
    import matplotlib

    matplotlib.use("Agg", force=True)
