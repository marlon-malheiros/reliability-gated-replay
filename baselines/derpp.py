"""DER++: replay with stored-logit distillation."""
from __future__ import annotations

from typing import Any, Dict

import torch

from methods.base import ContinualMethod
from methods.replay_buffer import LogitReplayBuffer, derpp_loss


class DERPP(ContinualMethod):
    name = "derpp"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        self.buf = LogitReplayBuffer(int(self.cfg.get("buffer_size", 500)))
        self.replay_n = int(self.cfg.get("batch_size", 32))
        self.alpha = float(self.cfg.get("alpha", 0.5))
        self.beta = float(self.cfg.get("beta", 1.0))
        self._device = None

    def on_task_start(self, task_id, model):
        self._device = next(model.parameters()).device

    def extra_loss(self, model, x, y, task_id):
        if len(self.buf) == 0:
            return x.new_zeros(())
        return derpp_loss(
            model,
            self.buf.sample(self.replay_n),
            self._device,
            alpha=self.alpha,
            beta=self.beta,
        )

    @torch.no_grad()
    def on_batch_end(self, model, x, y, task_id):
        logits = model(x, task_id)
        # admit everything (standard DER++); track the clean label for the purity audit
        self.buf.add(x, y, task_id, logits, y_clean=self._batch_y_clean)

    def consolidation_state(self, model):
        return {
            "buffer_size": len(self.buf),
            "buffer_purity": self.buf.purity(),
            "derpp_alpha": self.alpha,
            "derpp_beta": self.beta,
        }
