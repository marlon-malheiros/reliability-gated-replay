"""Plain fine-tuning baselines.

``Finetune`` (no continual mechanism) backs the Standard-Adam, Standard-SGD and
Dropout baselines -- those differ only in the optimizer / model config, not in
the training logic. ``EarlyStopping`` adds per-task early stopping on validation
loss.
"""
from __future__ import annotations

from typing import Any, Dict

from methods.base import ContinualMethod


class Finetune(ContinualMethod):
    name = "finetune"


class EarlyStopping(ContinualMethod):
    name = "early_stopping"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        self.patience = int(self.cfg.get("patience", 2))
        self.min_delta = float(self.cfg.get("min_delta", 1e-3))
        self._best = float("inf")
        self._wait = 0

    def on_task_start(self, task_id, model):
        self._best = float("inf")
        self._wait = 0

    def on_epoch_end(self, model, val_metrics, epoch) -> bool:
        loss = val_metrics["loss"]
        if loss < self._best - self.min_delta:
            self._best = loss
            self._wait = 0
            return False
        self._wait += 1
        return self._wait >= self.patience
