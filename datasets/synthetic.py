"""Synthetic MNIST-shaped dataset -- the offline fallback for smoke tests / CI.

Generates ``(N, 1, 28, 28)`` images for 10 classes, each class defined by a
smooth random prototype plus per-sample noise. Classes are linearly separable
enough that even a 1-epoch MLP reaches well above chance, so the smoke pipeline
produces non-trivial accuracy with zero network access.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _smooth_prototype(rng: np.random.RandomState, size: int = 28) -> np.ndarray:
    """A low-frequency random image in [0,255], distinct per call."""
    low = rng.randn(7, 7)
    # bilinear upsample 7x7 -> 28x28 via repeat + cheap blur
    up = np.kron(low, np.ones((4, 4)))
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    padded = np.pad(up, 1, mode="edge")
    blurred = sum(
        k[i, j] * padded[i : i + size, j : j + size]
        for i in range(3)
        for j in range(3)
    )
    blurred -= blurred.min()
    if blurred.max() > 0:
        blurred /= blurred.max()
    return (blurred * 255.0).astype(np.float32)


def make_synthetic(
    n_classes: int = 10,
    n_train_per_class: int = 600,
    n_test_per_class: int = 100,
    noise: float = 40.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_x, train_y, test_x, test_y), images uint8 (N,1,28,28)."""
    rng = np.random.RandomState(seed)
    prototypes = [_smooth_prototype(rng) for _ in range(n_classes)]

    def gen(n_per_class: int, rng_: np.random.RandomState):
        xs, ys = [], []
        for c in range(n_classes):
            base = prototypes[c][None, :, :]
            imgs = base + rng_.randn(n_per_class, 28, 28) * noise
            imgs = np.clip(imgs, 0, 255)
            xs.append(imgs)
            ys.append(np.full(n_per_class, c, dtype=np.int64))
        x = np.concatenate(xs, 0).astype(np.uint8)[:, None, :, :]
        y = np.concatenate(ys, 0)
        perm = rng_.permutation(len(x))
        return x[perm], y[perm]

    train_x, train_y = gen(n_train_per_class, np.random.RandomState(seed + 1))
    test_x, test_y = gen(n_test_per_class, np.random.RandomState(seed + 2))
    return train_x, train_y, test_x, test_y
