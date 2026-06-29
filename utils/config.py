"""YAML configuration loading with inheritance and dotted overrides.

A config file may declare ``base: <path>`` (or a list of paths, relative to the
file) whose contents are deep-merged underneath it. This lets every method /
ablation config inherit from ``configs/base.yaml`` and override only what differs
-- which is exactly how the 18 ablations are expressed (see ``ablations/``).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins). Pure."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _resolve_bases(cfg: Dict[str, Any], cfg_dir: Path) -> Dict[str, Any]:
    """Resolve and merge any ``base:`` references declared in ``cfg``."""
    bases = cfg.pop("base", None)
    if bases is None:
        return cfg
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for b in bases:
        parent = load_config(cfg_dir / b)
        merged = deep_merge(merged, parent)
    return deep_merge(merged, cfg)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config, resolving ``base:`` inheritance relative to the file."""
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping.")
    return _resolve_bases(cfg, path.parent)


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a flat ``{"a.b.c": value}`` mapping onto a nested config (returns a copy)."""
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node = out
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise KeyError(f"Cannot descend into non-dict at '{k}' for '{dotted}'.")
        node[keys[-1]] = value
    return out


def flatten(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested config to dotted keys (handy for logging / manifests)."""
    flat: Dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, prefix=f"{key}."))
        else:
            flat[key] = v
    return flat


def merge_all(*cfgs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Left-to-right deep merge of several configs."""
    out: Dict[str, Any] = {}
    for c in cfgs:
        out = deep_merge(out, c)  # type: ignore[arg-type]
    return out
