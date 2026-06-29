"""The :class:`ContinualMethod` interface shared by PNN + every baseline.

The trainer (``methods/trainer.py``) calls these hooks at fixed points in the
loop; a plain fine-tuning method implements none of them (all are no-ops). Each
method may append per-epoch diagnostics to ``self.history`` and expose
consolidation state via ``consolidation_state`` -- the analysis layer reads both.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ContinualMethod:
    name: str = "base"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.history: List[Dict[str, Any]] = []   # one record per epoch (method signals)
        # per-batch context set by the trainer before each batch (used by
        # reliability-gated methods): stable example uids and the *clean* labels.
        self._batch_uids: "torch.Tensor | None" = None
        self._batch_y_clean: "torch.Tensor | None" = None

    # --- lifecycle hooks (all optional) ---
    def observe_batch(
        self,
        task_id: int,
        uids: "torch.Tensor | None",
        y_clean: "torch.Tensor | None",
    ) -> None:
        """Trainer-supplied per-batch context (stable uids + clean labels).

        Called immediately after a batch is loaded, before forward/backward.
        Reliability-gated methods read it to key per-example stability EMAs and to
        audit buffer purity against the true labels; other methods ignore it.
        """
        self._batch_uids = uids
        self._batch_y_clean = y_clean

    def on_task_start(self, task_id: int, model: nn.Module) -> None:
        ...

    def main_loss(
        self,
        model: nn.Module,
        out: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> "torch.Tensor | None":
        """Optionally REPLACE the default current-batch CE, reusing the trainer's
        forward output ``out`` (so no extra forward pass / no BatchNorm double-update).

        Return ``None`` (default) to keep the standard ``F.cross_entropy(out, y)``.
        ER-ACE overrides this to mask the current batch to its present classes.
        """
        return None

    def extra_loss(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        """Extra loss added to the task CE each batch (penalties / replay / KD)."""
        return x.new_zeros(())

    def modify_gradients(self, model: nn.Module) -> None:
        """Runs after ``backward()`` and before ``optimizer.step()``."""
        ...

    def on_batch_end(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, task_id: int
    ) -> None:
        """Runs after ``optimizer.step()`` (path-integral / buffer updates)."""
        ...

    def on_epoch_end(
        self, model: nn.Module, val_metrics: Dict[str, float], epoch: int
    ) -> bool:
        """Return True to stop the current task early (e.g. early stopping)."""
        return False

    def on_stream_step(
        self,
        model: nn.Module,
        step: int,
        x: torch.Tensor,
        y: torch.Tensor,
        loss: float,
        n_steps: int = 0,
        at_boundary: bool = False,
    ) -> None:
        """Task-free (StreamTrainer) per-step hook, called after ``on_batch_end``.

        Used by online consolidation: drive the plateau detector from the running
        training ``loss`` and trigger schedule-based theta* snapshots. ``loss`` is
        the task CE only (no anchor/replay); ``at_boundary`` marks nominal region
        boundaries (used solely by the ``task_boundary`` snapshot schedule).
        """
        ...

    def on_task_end(self, task_id: int, model: nn.Module, train_loader: DataLoader) -> None:
        ...

    # --- diagnostics the trainer/analysis may read ---
    def consolidation_state(self, model: nn.Module) -> Dict[str, Any]:
        """Method-specific arrays/scalars for analysis (P values, importance...)."""
        return {}

    def offline_phase(self, model: nn.Module, task_id: int) -> None:
        """Optional post-task offline consolidation (PNN+Sleep uses this)."""
        ...
