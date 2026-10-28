"""Load the TOML config. Each section is returned as a plain dict, and paths are resolved from the repo root."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "grace_v7_parts.toml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    for key, value in cfg.get("data", {}).items():
        if key.endswith(("_dir", "_path", "_csv")):
            cfg["data"][key] = ROOT / value
    return cfg
