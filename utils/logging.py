"""Logging + small reproducibility helpers."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str = "pnn", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide configured logger (idempotent)."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def git_hash(cwd: str | Path | None = None) -> str:
    """Best-effort short git hash of the working tree, or 'nogit'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "nogit"
