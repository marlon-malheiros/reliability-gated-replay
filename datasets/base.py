"""Core data abstractions shared by every benchmark.

A :class:`ContinualBenchmark` is an ordered list of :class:`Task` objects. Each
task carries train/val/test ``TensorDataset``s whose labels are **local** to the
task (remapped to ``0..n_classes_per_task-1``) -- this is what the default
multi-head, task-incremental protocol needs. The original global class ids are
kept on the task for reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass
class Task:
    task_id: int
    name: str
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    global_classes: List[int]   # original labels, e.g. [2, 3] for Split-MNIST task 1
    n_classes: int              # outputs of this task's head
    # label-noise bookkeeping (set by datasets/noise.py; aligned to train order):
    #   train_y_clean  -- the *true* labels (train.tensors[1] may be corrupted)
    #   train_is_noisy -- bool mask, True where the training label was flipped
    train_y_clean: "np.ndarray | None" = None
    train_is_noisy: "np.ndarray | None" = None
    noise_rate: float = 0.0     # fraction of this task's train labels corrupted


@dataclass
class ContinualBenchmark:
    name: str
    tasks: List[Task]
    input_shape: Tuple[int, ...]   # (C, H, W)
    n_classes_per_task: int
    multihead: bool                # True => task-incremental (one head per task)

    def __len__(self) -> int:
        return len(self.tasks)


def normalize_images(x: np.ndarray, mean: float = 0.1307, std: float = 0.3081) -> np.ndarray:
    """uint8 [0,255] -> float32 standardized (MNIST statistics by default)."""
    x = x.astype(np.float32) / 255.0
    return (x - mean) / std


def to_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    xt = torch.from_numpy(np.ascontiguousarray(x)).float()
    yt = torch.from_numpy(np.ascontiguousarray(y)).long()
    return TensorDataset(xt, yt)


def train_val_split(
    x: np.ndarray, y: np.ndarray, val_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic stratified-ish shuffle split into (train, val)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x))
    n_val = int(round(val_fraction * len(x)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def subsample(x: np.ndarray, y: np.ndarray, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Take at most ``n`` samples (used by the smoke config to keep runs tiny)."""
    if n is None or n >= len(x):
        return x, y
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x))[:n]
    return x[idx], y[idx]


def make_binary_tasks(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    class_pairs: Sequence[Sequence[int]],
    val_fraction: float,
    seed: int,
    max_train_per_task: int | None = None,
) -> List[Task]:
    """Build the per-task datasets for a class-split benchmark (e.g. Split-MNIST).

    Labels are remapped to be local to each task (the i-th class in the pair maps
    to local index i).
    """
    tasks: List[Task] = []
    for tid, classes in enumerate(class_pairs):
        local = {g: i for i, g in enumerate(classes)}

        def select(x, y):
            mask = np.isin(y, classes)
            xs, ys = x[mask], y[mask]
            ys = np.vectorize(local.get)(ys).astype(np.int64)
            return xs, ys

        tr_x, tr_y = select(train_x, train_y)
        te_x, te_y = select(test_x, test_y)
        tr_x, tr_y = subsample(tr_x, tr_y, max_train_per_task, seed + tid)
        tr_x, tr_y, va_x, va_y = train_val_split(tr_x, tr_y, val_fraction, seed + tid)
        tasks.append(
            Task(
                task_id=tid,
                name=f"task{tid}_{'v'.join(map(str, classes))}",
                train=to_tensor_dataset(tr_x, tr_y),
                val=to_tensor_dataset(va_x, va_y),
                test=to_tensor_dataset(te_x, te_y),
                global_classes=list(classes),
                n_classes=len(classes),
            )
        )
    return tasks
