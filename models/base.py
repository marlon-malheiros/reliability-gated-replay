"""Base continual-learning model: shared backbone + per-task heads.

Two design hooks the rest of the framework relies on:

* ``features(x)`` returns the penultimate representation -- consumed by the CKA /
  representational-drift analysis.
* ``weight_layers()`` returns the backbone's weight-bearing modules in forward
  order -- consumed by progressive-layer-freezing (freeze first k) and by the
  layer-wise PNN variant (ablation A17).

``forward(x, task_id)`` selects the head; with ``multihead=False`` a single head
is shared across tasks (domain-incremental).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class ContinualModel(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes_per_task: int,
        n_tasks: int,
        multihead: bool = True,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.n_classes_per_task = n_classes_per_task
        self.n_tasks = n_tasks
        self.multihead = multihead
        self.feature_dim: int = 0  # set by subclass after building backbone

    def _build_heads(self) -> None:
        n_heads = self.n_tasks if self.multihead else 1
        self.heads = nn.ModuleList(
            [nn.Linear(self.feature_dim, self.n_classes_per_task) for _ in range(n_heads)]
        )

    # --- to implement in subclasses ---
    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def weight_layers(self) -> List[Tuple[str, nn.Module]]:
        raise NotImplementedError

    # --- shared ---
    def head(self, task_id: int) -> nn.Linear:
        return self.heads[task_id if self.multihead else 0]

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        return self.head(task_id)(self.features(x))

    def backbone_parameters(self):
        head_ids = {id(p) for p in self.heads.parameters()}
        return [p for p in self.parameters() if id(p) not in head_ids]
