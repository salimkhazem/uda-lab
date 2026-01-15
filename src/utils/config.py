"""Configuration utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_by_path(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _parse_value(raw: str) -> Any:
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.lower() in {"null", "none"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_config(
    config_path: str,
    dataset: str | None = None,
    model: str | None = None,
    opts: List[str] | None = None,
) -> Dict[str, Any]:
    """Load and merge configuration files with overrides.

    Args:
        config_path: Base experiment config path.
        dataset: Dataset config name or path.
        model: Model config name or path.
        opts: List of key=value overrides. Supports @file for YAML overrides.

    Returns:
        Merged configuration dict.
    """
    base = _load_yaml(Path(__file__).resolve().parents[2] / "configs" / "default.yaml")
    exp_cfg = _load_yaml(config_path)
    cfg = _deep_update(copy.deepcopy(base), exp_cfg)

    if dataset:
        ds_path = dataset
        if not dataset.endswith(".yaml"):
            ds_path = Path(__file__).resolve().parents[2] / "configs" / "datasets" / f"{dataset}.yaml"
        cfg = _deep_update(cfg, _load_yaml(ds_path))

    if model:
        model_path = model
        if not model.endswith(".yaml"):
            model_path = Path(__file__).resolve().parents[2] / "configs" / "models" / f"{model}.yaml"
        cfg = _deep_update(cfg, _load_yaml(model_path))

    if opts:
        for item in opts:
            if item.startswith("@"):
                cfg = _deep_update(cfg, _load_yaml(item[1:]))
                continue
            if "=" not in item:
                continue
            key, raw = item.split("=", 1)
            _set_by_path(cfg, key, _parse_value(raw))

    return cfg
