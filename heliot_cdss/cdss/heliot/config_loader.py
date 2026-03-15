from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.yml"


def load_config() -> dict[str, Any]:
    """
    Load the application configuration from config.yml.
    """
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise RuntimeError("config.yml must contain a top-level mapping")

    return data