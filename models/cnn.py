"""Model B: small CNN  [Conv-ReLU-Pool] x2 -> Dense -> head (spec)."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .base import ContinualModel


class SmallCNN(ContinualModel):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes_per_task: int,
        n_tasks: int,
        multihead: bool = True,
        channels: Tuple[int, int] = (32, 64),
        dense: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__(input_shape, n_classes_per_task, n_tasks, multihead)
        c_in, h, w = input_shape
        self.conv1 = nn.Conv2d(c_in, channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        flat_dim = channels[1] * (h // 4) * (w // 4)
        self.fc = nn.Linear(flat_dim, dense)
        self.feature_dim = dense
        self._build_heads()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        x = self.drop(self.act(self.fc(x)))
        return x

    def weight_layers(self) -> List[Tuple[str, nn.Module]]:
        return [("conv1", self.conv1), ("conv2", self.conv2), ("fc", self.fc)]
