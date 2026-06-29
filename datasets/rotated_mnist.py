"""Rotated-MNIST: each task applies a fixed image rotation.

All tasks are 10-way classification over the same source images, so this is a
domain-incremental benchmark by default. Angles are evenly spaced between
``min_angle`` and ``max_angle`` unless explicitly supplied as ``angles``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
from scipy import ndimage

from .base import (
    ContinualBenchmark,
    Task,
    normalize_images,
    subsample,
    to_tensor_dataset,
    train_val_split,
)


def _angles(cfg: Dict[str, Any], n_tasks: int) -> List[float]:
    explicit = cfg.get("angles")
    if explicit is not None:
        vals = [float(a) for a in explicit]
        if len(vals) != n_tasks:
            raise ValueError(f"Expected {n_tasks} rotation angles, got {len(vals)}")
        return vals
    lo = float(cfg.get("min_angle", 0.0))
    hi = float(cfg.get("max_angle", 180.0))
    return np.linspace(lo, hi, n_tasks).astype(float).tolist()


def _rotate_one(img: np.ndarray, angle: float) -> np.ndarray:
    return ndimage.rotate(
        img,
        angle=angle,
        axes=(-2, -1),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    )


def _rotate_images(x: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-12:
        return x.copy()
    rotated = np.empty_like(x)
    for i in range(x.shape[0]):
        rotated[i] = _rotate_one(x[i], angle)
    return rotated


def build_rotated_mnist(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    cfg: Dict[str, Any],
) -> ContinualBenchmark:
    seed = int(cfg.get("seed", 0))
    n_tasks = int(cfg.get("n_tasks", 10))
    val_fraction = float(cfg.get("val_fraction", 0.1))
    max_train_per_task = cfg.get("max_train_per_task")
    max_test_per_task = cfg.get("max_test_per_task")
    n_classes = int(train_y.max()) + 1
    angles = _angles(cfg, n_tasks)

    # Rotate in [0, 1] image space, then apply the usual source normalization.
    train_x = train_x.astype(np.float32) / 255.0
    test_x = test_x.astype(np.float32) / 255.0

    tasks = []
    for tid, angle in enumerate(angles):
        tr_base_x, tr_base_y = subsample(train_x, train_y, max_train_per_task, seed + tid)
        te_base_x, te_base_y = subsample(test_x, test_y, max_test_per_task, seed + 10_000 + tid)

        tr_x = _rotate_images(tr_base_x, angle)
        te_x = _rotate_images(te_base_x, angle)
        if cfg.get("normalize", True):
            tr_x = normalize_images((tr_x * 255.0).astype(np.float32))
            te_x = normalize_images((te_x * 255.0).astype(np.float32))

        a, b, va_x, va_y = train_val_split(tr_x, tr_base_y, val_fraction, seed + tid)
        tasks.append(
            Task(
                task_id=tid,
                name=f"rot{tid}_{angle:.1f}",
                train=to_tensor_dataset(a, b),
                val=to_tensor_dataset(va_x, va_y),
                test=to_tensor_dataset(te_x, te_base_y),
                global_classes=list(range(n_classes)),
                n_classes=n_classes,
            )
        )
    return ContinualBenchmark(
        name=cfg.get("name", "rotated_mnist"),
        tasks=tasks,
        input_shape=train_x.shape[1:],
        n_classes_per_task=n_classes,
        multihead=bool(cfg.get("multihead", False)),
    )
