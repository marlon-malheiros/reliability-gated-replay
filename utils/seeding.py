"""Reproducible seeding across random / numpy / torch."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we rely on.

    With ``deterministic=True`` we also pin cuDNN to deterministic kernels so
    repeated runs of the same config reproduce bit-for-bit (at a small speed
    cost). This is the right default for a reproducibility-focused study.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding helper (keeps multi-worker loading reproducible)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
