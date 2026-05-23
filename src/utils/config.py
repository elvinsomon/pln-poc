"""Loader YAML con herencia simple vía `extends:`.

Permite que cada experimento defina su propio archivo y comparta defaults
en `base.yaml`. El hijo sobrescribe al padre clave a clave (nivel superior).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(name: str | Path) -> dict[str, Any]:
    path = Path(name)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    base = load_config(parent)
    base.update(cfg)   # override superficial; los dicts anidados se reemplazan completos
    return base
