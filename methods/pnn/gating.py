"""Plasticity gating: how consolidation state P throttles weight updates.

Both spec variants are implemented as a per-element multiplicative factor on the
gradient (applied before the optimizer step):

* ``grad_scale`` : factor = exp(-beta * P)   -- gradient suppression
* ``lr_scale``   : factor = (1 - P)          -- effective per-parameter LR

For SGD, ``lr_scale`` reproduces ``lr_i = lr * (1 - P_i)`` exactly; for Adam it is
a monotone throttle (moment normalization makes it approximate, which we note in
the report). ``none`` disables gating (used by ablations that keep P for analysis
only).
"""
from __future__ import annotations

import torch


def gating_factor(P: torch.Tensor, mode: str, beta: float) -> torch.Tensor:
    if mode == "grad_scale":
        return torch.exp(-beta * P)
    if mode == "lr_scale":
        return (1.0 - P).clamp(0.0, 1.0)
    if mode == "none":
        return torch.ones_like(P)
    raise ValueError(f"Unknown gating mode '{mode}'. Use grad_scale|lr_scale|none")
