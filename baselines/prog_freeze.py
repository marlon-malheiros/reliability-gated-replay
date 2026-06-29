"""Progressive Layer Freezing.

After finishing task t we freeze the backbone's weight layers up to index t
(cumulative), always leaving the final weight layer and the task heads trainable.
A simple structural baseline against the learned, graded PNN protection.
"""
from __future__ import annotations

from typing import Any, Dict

from methods.base import ContinualMethod


class ProgressiveFreezing(ContinualMethod):
    name = "prog_freeze"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        self.keep_last = bool(self.cfg.get("keep_last_trainable", True))

    def on_task_end(self, task_id, model, train_loader):
        layers = model.weight_layers()
        last_idx = len(layers) - 1
        for i, (_, module) in enumerate(layers):
            if i <= task_id and not (self.keep_last and i == last_idx):
                for p in module.parameters():
                    p.requires_grad_(False)
