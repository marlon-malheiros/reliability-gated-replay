"""ResNet-18 for CIFAR (the standard continual-learning backbone).

CIFAR variant: a 3x3 stride-1 stem and no initial max-pool (so 32x32 inputs are
not downsampled too early), then the usual four BasicBlock stages
[64, 128, 256, 512] with [2, 2, 2, 2] blocks. ``features(x)`` returns the
512-dim globally-pooled representation; a single shared head (``multihead=False``)
makes this a domain-incremental classifier.

Implements the :class:`ContinualModel` interface so it drops into the existing
trainer / EWC / gate machinery unchanged.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ContinualModel


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(ContinualModel):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes_per_task: int,
        n_tasks: int,
        multihead: bool = True,
        width: int = 64,
    ):
        super().__init__(input_shape, n_classes_per_task, n_tasks, multihead)
        c_in = input_shape[0]
        self.in_planes = width
        self.conv1 = nn.Conv2d(c_in, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, 2, stride=1)
        self.layer2 = self._make_layer(width * 2, 2, stride=2)
        self.layer3 = self._make_layer(width * 4, 2, stride=2)
        self.layer4 = self._make_layer(width * 8, 2, stride=2)
        self.feature_dim = width * 8 * BasicBlock.expansion
        self._build_heads()

    def _make_layer(self, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return out

    def weight_layers(self) -> List[Tuple[str, nn.Module]]:
        return [
            ("conv1", self.conv1),
            ("layer1", self.layer1),
            ("layer2", self.layer2),
            ("layer3", self.layer3),
            ("layer4", self.layer4),
        ]
