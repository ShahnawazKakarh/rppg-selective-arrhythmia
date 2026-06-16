"""YAML config loading with optional base-config inheritance."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`. `override` wins on conflicts."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving an optional `experiment.base_config` chain.

    A child config may set `experiment.base_config: path/to/parent.yaml`. The
    parent is loaded first (recursively), then the child is deep-merged on top.
    """
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}

    base_path = cfg.get("experiment", {}).get("base_config")
    if base_path:
        base_full = (path.parent / base_path).resolve() if not Path(base_path).is_absolute() else Path(base_path)
        # Fall back to repo-root-relative if the relative-to-parent path is missing.
        if not base_full.exists():
            base_full = Path(base_path)
        base_cfg = load_config(base_full)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg
