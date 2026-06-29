"""Permuted-MNIST: each task applies a fixed random pixel permutation.

Included as a cheap second continual benchmark (the spec lists it as optional).
All tasks are 10-way digit classification, so by default this is a single-head
(domain-incremental) benchmark; set ``multihead: true`` for the task-IL variant.
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


def _apply_perm(x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    flat = x.reshape(n, c, h * w)[:, :, perm]
    return flat.reshape(n, c, h, w)


def build_permuted_mnist(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    cfg: Dict[str, Any],
) -> ContinualBenchmark:
    seed = int(cfg.get("seed", 0))
    n_tasks = int(cfg.get("n_tasks", 5))
    val_fraction = float(cfg.get("val_fraction", 0.1))
    max_train_per_task = cfg.get("max_train_per_task")
    n_classes = int(train_y.max()) + 1

    if cfg.get("normalize", True):
        train_x = normalize_images(train_x)
        test_x = normalize_images(test_x)
    else:
        train_x = train_x.astype(np.float32) / 255.0
        test_x = test_x.astype(np.float32) / 255.0

    n_pixels = int(np.prod(train_x.shape[2:]))
    rng = np.random.RandomState(seed)
    tasks = []
    for tid in range(n_tasks):
        perm = np.arange(n_pixels) if tid == 0 else rng.permutation(n_pixels)
        tr_x = _apply_perm(train_x, perm)
        te_x = _apply_perm(test_x, perm)
        tr_x_s, tr_y_s = subsample(tr_x, train_y, max_train_per_task, seed + tid)
        a, b, va_x, va_y = train_val_split(tr_x_s, tr_y_s, val_fraction, seed + tid)
        tasks.append(
            Task(
                task_id=tid,
                name=f"perm{tid}",
                train=to_tensor_dataset(a, b),
                val=to_tensor_dataset(va_x, va_y),
                test=to_tensor_dataset(te_x, test_y),
                global_classes=list(range(n_classes)),
                n_classes=n_classes,
            )
        )
    return ContinualBenchmark(
        name=cfg.get("name", "permuted_mnist"),
        tasks=tasks,
        input_shape=train_x.shape[1:],
        n_classes_per_task=n_classes,
        multihead=bool(cfg.get("multihead", False)),
    )
