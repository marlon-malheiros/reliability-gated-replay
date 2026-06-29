"""Baseline continual-learning methods (spec list of 10)."""
from .ewc import EWC
from .finetune import EarlyStopping, Finetune
from .l2 import L2
from .lwf import LwF
from .prog_freeze import ProgressiveFreezing
from .replay import Replay
from .si import SI

__all__ = [
    "Finetune",
    "EarlyStopping",
    "L2",
    "EWC",
    "SI",
    "Replay",
    "LwF",
    "ProgressiveFreezing",
]
