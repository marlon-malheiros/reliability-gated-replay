"""L2 regularization toward the previous task's parameters (uniform importance).

This is the standard continual-learning L2 baseline: a quadratic anchor to the
weights at the end of the previous task, with every parameter equally important
-- i.e. EWC with the Fisher matrix replaced by the identity. Distinct from plain
weight decay (which anchors to zero).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from methods.base import ContinualMethod
from methods.utils import protected_named_parameters


class L2(ContinualMethod):
    name = "l2"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        self.lam = float(self.cfg.get("lambda", 1.0))
        self._named: List[Tuple[str, torch.nn.Parameter]] = []
        self.star: Dict[str, torch.Tensor] = {}

    def on_task_start(self, task_id, model):
        if not self._named:
            self._named = list(protected_named_parameters(model))

    def extra_loss(self, model, x, y, task_id):
        if not self.star:
            return x.new_zeros(())
        loss = x.new_zeros(())
        for n, p in self._named:
            loss = loss + (p - self.star[n]).pow(2).sum()
        return self.lam * loss

    def on_task_end(self, task_id, model, train_loader):
        self.star = {n: p.detach().clone() for n, p in self._named}
