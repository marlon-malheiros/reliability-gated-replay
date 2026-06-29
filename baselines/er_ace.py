"""ER-ACE (Caccia et al., ICLR 2022): Experience Replay with Asymmetric Cross-Entropy.

The forgetting in class-incremental ER is partly driven by the *current* task's
cross-entropy suppressing the logits of past classes. ER-ACE fixes this with an
asymmetric update: the incoming (stream) batch is scored against **only the classes
present in that batch** (its logits cannot push down unseen/old classes), while the
replayed buffer batch uses the standard full-class cross-entropy.

Implemented via the ``main_loss`` hook (masks the trainer's existing forward output,
so no second forward / no BatchNorm double-update) plus a buffer-replay ``extra_loss``.
In a multi-head / task-IL setup the head already contains only the task's classes, so
the mask is a no-op and ER-ACE gracefully reduces to ER.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from methods.base import ContinualMethod
from methods.replay_buffer import ReplayBuffer, replay_ce_loss


class ERACE(ContinualMethod):
    name = "er_ace"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        seed = int(self.cfg.get("seed", 0))
        self.buf = ReplayBuffer(int(self.cfg.get("buffer_size", 500)), seed=seed)
        self.replay_n = int(self.cfg.get("replay_n", self.cfg.get("batch_size", 32)))
        self._device = None

    def on_task_start(self, task_id, model):
        self._device = next(model.parameters()).device

    def main_loss(self, model, out, x, y, task_id):
        # asymmetric CE: mask the current batch to the classes present in it
        present = torch.unique(y)
        mask = torch.full_like(out, float("-inf"))
        mask[:, present] = 0.0
        return F.cross_entropy(out + mask, y)

    def extra_loss(self, model, x, y, task_id):
        if len(self.buf) == 0:
            return x.new_zeros(())
        return replay_ce_loss(model, self.buf.sample(self.replay_n), self._device)

    def on_batch_end(self, model, x, y, task_id):
        # admit everything (reservoir); audit purity against the clean label
        self.buf.add(x, y, task_id, y_clean=self._batch_y_clean)

    def consolidation_state(self, model) -> Dict[str, Any]:
        return {"buffer_size": len(self.buf), "buffer_purity": self.buf.purity()}
