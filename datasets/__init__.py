"""Datasets: MNIST-family sources + continual-learning protocols."""
from .base import ContinualBenchmark, Task
from .registry import build_benchmark, load_source_images

__all__ = ["ContinualBenchmark", "Task", "build_benchmark", "load_source_images"]
