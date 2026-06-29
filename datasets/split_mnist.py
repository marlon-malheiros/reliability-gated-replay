"""Split-MNIST (the primary benchmark): 5 binary tasks 0v1, 2v3, 4v5, 6v7, 8v9.

Works on any MNIST-shaped source (real MNIST, Fashion-MNIST, or the synthetic
fallback), so the same protocol drives both real and offline-smoke runs.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import ContinualBenchmark, make_binary_tasks, normalize_images, subsample

DEFAULT_PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def build_split_mnist(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    cfg: Dict[str, Any],
) -> ContinualBenchmark:
    seed = int(cfg.get("seed", 0))
    class_pairs = cfg.get("class_pairs") or DEFAULT_PAIRS
    val_fraction = float(cfg.get("val_fraction", 0.1))
    max_train_per_task = cfg.get("max_train_per_task")
    max_test_per_task = cfg.get("max_test_per_task")

    if cfg.get("normalize", True):
        train_x = normalize_images(train_x)
        test_x = normalize_images(test_x)
    else:
        train_x = train_x.astype(np.float32) / 255.0
        test_x = test_x.astype(np.float32) / 255.0

    if max_test_per_task is not None:
        # cap test set globally before per-task selection (smoke speed)
        test_x, test_y = subsample(test_x, test_y, max_test_per_task * len(class_pairs), seed)

    tasks = make_binary_tasks(
        train_x, train_y, test_x, test_y,
        class_pairs=class_pairs,
        val_fraction=val_fraction,
        seed=seed,
        max_train_per_task=max_train_per_task,
    )
    return ContinualBenchmark(
        name=cfg.get("name", "split_mnist"),
        tasks=tasks,
        input_shape=train_x.shape[1:],
        n_classes_per_task=len(class_pairs[0]),
        multihead=bool(cfg.get("multihead", True)),
    )
