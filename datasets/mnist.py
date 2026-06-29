"""MNIST single-task benchmark (thin wrapper over single_task + idx source)."""
from __future__ import annotations

CLASS_NAMES = [str(d) for d in range(10)]
SOURCE = "mnist"
