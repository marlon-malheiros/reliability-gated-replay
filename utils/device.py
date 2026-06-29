"""Device selection."""
from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve a torch device.

    ``prefer`` may be ``"auto"`` (CUDA if available else CPU), ``"cuda"`` or
    ``"cpu"``. Falls back to CPU with no error if CUDA is requested but absent,
    so configs are portable across machines (e.g. the 4 GB RTX 2050 here vs CI).
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
