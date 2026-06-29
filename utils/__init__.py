"""Shared utilities: seeding, device, config loading, logging."""
from .seeding import set_seed
from .device import get_device
from .config import load_config, deep_merge, apply_overrides, flatten
from .logging import get_logger, git_hash

__all__ = [
    "set_seed",
    "get_device",
    "load_config",
    "deep_merge",
    "apply_overrides",
    "flatten",
    "get_logger",
    "git_hash",
]
