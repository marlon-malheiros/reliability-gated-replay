"""Linear CKA and cosine similarity between representation matrices.

Used for the representational-drift / feature-stability metrics: both inputs are
``(n_probe, d)`` activation matrices evaluated on the *same* fixed probe set, so
rows are aligned across task snapshots.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered linear CKA in [0,1] (1 = identical representation geometry)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic_xy = np.linalg.norm(Y.T @ X) ** 2
    hsic_xx = np.linalg.norm(X.T @ X)
    hsic_yy = np.linalg.norm(Y.T @ Y)
    denom = hsic_xx * hsic_yy
    return float(hsic_xy / (denom + _EPS))


def mean_cosine(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean row-wise cosine similarity between aligned representations."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    num = (X * Y).sum(1)
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + _EPS
    return float(np.mean(num / den))


def cka_matrix(reps) -> np.ndarray:
    """Pairwise CKA across a list of representation snapshots."""
    n = len(reps)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = linear_cka(np.array(reps[i]), np.array(reps[j]))
    return M
