"""Parameter-importance estimation for PNN maturation.

Implements the spec's five importance methods (magnitude, gradient magnitude,
gradient variance, approximate Fisher, hybrid). Gradient-based statistics are
tracked as exponential moving averages updated each batch from ``p.grad``; the
final importance per layer is min-max normalized to ``[0,1]`` so it composes
cleanly with the maturation step ``P += alpha * importance * closure_signal``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

_EPS = 1e-12


def _normalize(t: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a tensor to [0,1] (per layer)."""
    lo = t.min()
    hi = t.max()
    return (t - lo) / (hi - lo + _EPS)


class ImportanceEstimator:
    METHODS = ("magnitude", "grad", "grad_var", "fisher", "hybrid")

    def __init__(
        self,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        method: str = "hybrid",
        ema_decay: float = 0.9,
        hybrid_weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ):
        if method not in self.METHODS:
            raise ValueError(f"Unknown importance method '{method}'. {self.METHODS}")
        self.method = method
        self.beta = ema_decay
        self.w = hybrid_weights
        self.grad_ema = {n: torch.zeros_like(p) for n, p in named_params}
        self.grad2_ema = {n: torch.zeros_like(p) for n, p in named_params}
        self.steps = 0

    def update(self, named_params: List[Tuple[str, torch.nn.Parameter]]) -> None:
        """EMA-update gradient statistics (call after backward, before step)."""
        b = self.beta
        for n, p in named_params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            self.grad_ema[n].mul_(b).add_(g, alpha=1 - b)
            self.grad2_ema[n].mul_(b).add_(g * g, alpha=1 - b)
        self.steps += 1

    def _raw(self, name: str, p: torch.Tensor) -> torch.Tensor:
        if self.method == "magnitude":
            return p.detach().abs()
        if self.method == "grad":
            return self.grad_ema[name].abs()
        if self.method == "grad_var":
            return (self.grad2_ema[name] - self.grad_ema[name] ** 2).clamp_min(0.0)
        if self.method == "fisher":
            return self.grad2_ema[name]
        # hybrid: weighted sum of normalized components
        mag = _normalize(p.detach().abs())
        grad = _normalize(self.grad_ema[name].abs())
        fisher = _normalize(self.grad2_ema[name])
        w1, w2, w3 = self.w
        return (w1 * mag + w2 * grad + w3 * fisher)

    def importance(
        self, named_params: List[Tuple[str, torch.nn.Parameter]]
    ) -> Dict[str, torch.Tensor]:
        """Return per-parameter importance tensors normalized to [0,1]."""
        out: Dict[str, torch.Tensor] = {}
        for n, p in named_params:
            out[n] = _normalize(self._raw(n, p)).detach()
        return out
