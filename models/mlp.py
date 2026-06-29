"""Model A: MLP  784 -> 512 -> 256 -> head, ReLU (spec)."""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from .base import ContinualModel


class MLP(ContinualModel):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes_per_task: int,
        n_tasks: int,
        multihead: bool = True,
        hidden: Tuple[int, ...] = (512, 256),
        dropout: float = 0.0,
    ):
        super().__init__(input_shape, n_classes_per_task, n_tasks, multihead)
        in_dim = int(math.prod(input_shape))
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.feature_dim = hidden[1]
        self._build_heads()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        return x

    def weight_layers(self) -> List[Tuple[str, nn.Module]]:
        return [("fc1", self.fc1), ("fc2", self.fc2)]


class LinearProbe(ContinualModel):
    """Linear classifier on top of fixed inputs (e.g. frozen features). Limited
    capacity -> cannot memorize random labels, so the small-loss gate stays valid.
    ``features(x)`` is the (flattened) input itself; the head is the only weights."""

    def __init__(self, input_shape, n_classes_per_task, n_tasks, multihead=True, dropout=0.0):
        super().__init__(input_shape, n_classes_per_task, n_tasks, multihead)
        self.feature_dim = int(math.prod(input_shape))
        self.drop = nn.Dropout(dropout)
        self._build_heads()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x.flatten(1))

    def weight_layers(self) -> List[Tuple[str, nn.Module]]:
        return []
