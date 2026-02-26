from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, cast

if TYPE_CHECKING:
    class _TomlModule(Protocol):
        def loads(self, s: str, /) -> Mapping[str, Any]: ...


def _import_toml_module() -> "_TomlModule":
    try:
        module = importlib.import_module("tomllib")
    except ModuleNotFoundError:
        module = importlib.import_module("tomli")
    return cast("_TomlModule", module)


def load_toml(path: Path) -> dict[str, Any]:
    parser = _import_toml_module()
    parsed = parser.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Top-level TOML document must be a table.")
    return cast(dict[str, Any], parsed)
