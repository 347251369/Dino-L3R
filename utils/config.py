"""Load and validate the TOML configuration used by the unified runner."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def require(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing required configuration key: {dotted_key}")
        value = value[key]
    return value
