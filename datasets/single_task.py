"""Single-task (standard classification) benchmark builder.

Used for MNIST and Fashion-MNIST, which the spec lists as mandatory non-continual
datasets. Wrapped as a one-task :class:`ContinualBenchmark` so they flow through
the exact same trainer / metrics code as the continual benchmarks.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import (
    ContinualBenchmark,
    Task,
    normalize_images,
    subsample,
    to_tensor_dataset,
    train_val_split,
)


def build_single_task(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    cfg: Dict[str, Any],
) -> ContinualBenchmark:
    seed = int(cfg.get("seed", 0))
    val_fraction = float(cfg.get("val_fraction", 0.1))
    max_train = cfg.get("max_train_per_task")
    n_classes = int(train_y.max()) + 1

    if cfg.get("normalize", True):
        train_x = normalize_images(train_x)
        test_x = normalize_images(test_x)
    else:
        train_x = train_x.astype(np.float32) / 255.0
        test_x = test_x.astype(np.float32) / 255.0

    train_x, train_y = subsample(train_x, train_y, max_train, seed)
    tr_x, tr_y, va_x, va_y = train_val_split(train_x, train_y, val_fraction, seed)

    task = Task(
        task_id=0,
        name=name,
        train=to_tensor_dataset(tr_x, tr_y),
        val=to_tensor_dataset(va_x, va_y),
        test=to_tensor_dataset(test_x, test_y),
        global_classes=list(range(n_classes)),
        n_classes=n_classes,
    )
    return ContinualBenchmark(
        name=name,
        tasks=[task],
        input_shape=train_x.shape[1:],
        n_classes_per_task=n_classes,
        multihead=False,
    )
