"""Learning without Forgetting (Li & Hoiem, 2017).

Before each new task we snapshot the current model; during training we distill the
snapshot's predictions on the *current* inputs through the old task heads,
penalizing divergence of the new model from the old (knowledge distillation).
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn.functional as F

from methods.base import ContinualMethod


class LwF(ContinualMethod):
    name = "lwf"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        self.lam = float(self.cfg.get("lambda", 1.0))
        self.T = float(self.cfg.get("temperature", 2.0))
        self.old_model = None
        self.n_old = 0

    def on_task_start(self, task_id, model):
        if task_id > 0:
            self.old_model = copy.deepcopy(model).eval()
            for p in self.old_model.parameters():
                p.requires_grad_(False)
            self.n_old = task_id

    def extra_loss(self, model, x, y, task_id):
        if self.old_model is None or self.n_old == 0:
            return x.new_zeros(())
        loss = x.new_zeros(())
        with torch.no_grad():
            old_feat = self.old_model.features(x)
        new_feat = model.features(x)
        for t in range(self.n_old):
            with torch.no_grad():
                old_logits = self.old_model.head(t)(old_feat)
            new_logits = model.head(t)(new_feat)
            soft_old = F.softmax(old_logits / self.T, dim=1)
            log_new = F.log_softmax(new_logits / self.T, dim=1)
            loss = loss + F.kl_div(log_new, soft_old, reduction="batchmean") * (self.T**2)
        return self.lam * loss / max(self.n_old, 1)
