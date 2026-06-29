"""Models: shared backbone + per-task heads."""
from .base import ContinualModel
from .cnn import SmallCNN
from .mlp import MLP
from .registry import build_model

__all__ = ["ContinualModel", "MLP", "SmallCNN", "build_model"]
