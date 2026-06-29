"""Model registry: config + benchmark -> ContinualModel."""
from __future__ import annotations

from typing import Any, Dict

from datasets.base import ContinualBenchmark

from .base import ContinualModel
from .cnn import SmallCNN
from .mlp import MLP, LinearProbe
from .resnet import ResNet18


def build_model(cfg: Dict[str, Any], benchmark: ContinualBenchmark) -> ContinualModel:
    name = cfg.get("name", "mlp")
    multihead = cfg.get("multihead", benchmark.multihead)
    common = dict(
        input_shape=benchmark.input_shape,
        n_classes_per_task=benchmark.n_classes_per_task,
        n_tasks=len(benchmark.tasks),
        multihead=multihead,
    )
    if name == "mlp":
        return MLP(
            **common,
            hidden=tuple(cfg.get("hidden", (512, 256))),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if name == "cnn":
        return SmallCNN(
            **common,
            channels=tuple(cfg.get("channels", (32, 64))),
            dense=int(cfg.get("dense", 128)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if name in ("resnet18", "resnet"):
        return ResNet18(**common, width=int(cfg.get("width", 64)))
    if name == "linear":
        return LinearProbe(**common, dropout=float(cfg.get("dropout", 0.0)))
    raise KeyError(f"Unknown model '{name}'. Known: mlp, cnn, resnet18, linear")
