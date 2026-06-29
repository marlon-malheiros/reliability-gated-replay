"""Split-CIFAR-100 class-incremental benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .base import ContinualBenchmark, Task, subsample, to_tensor_dataset, train_val_split


def _load_cifar100(data_root: str | Path, download: bool = True):
    try:
        from torchvision.datasets import CIFAR100
    except Exception as e:  # pragma: no cover - depends on optional install
        raise RuntimeError("Split-CIFAR-100 requires torchvision.") from e

    root = Path(data_root)
    train = CIFAR100(root=str(root), train=True, download=download)
    test = CIFAR100(root=str(root), train=False, download=download)
    train_x = np.asarray(train.data).transpose(0, 3, 1, 2)
    test_x = np.asarray(test.data).transpose(0, 3, 1, 2)
    train_y = np.asarray(train.targets, dtype=np.int64)
    test_y = np.asarray(test.targets, dtype=np.int64)
    return train_x, train_y, test_x, test_y


def _normalize_cifar(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    mean = np.asarray([0.5071, 0.4867, 0.4408], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray([0.2675, 0.2565, 0.2761], dtype=np.float32).reshape(1, 3, 1, 1)
    return (x - mean) / std


def _class_groups(n_classes: int, classes_per_task: int, seed: int, shuffle: bool) -> list[list[int]]:
    classes = np.arange(n_classes)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(classes)
    return [
        classes[i : i + classes_per_task].astype(int).tolist()
        for i in range(0, n_classes, classes_per_task)
    ]


def _make_class_incremental_tasks(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    class_groups: Sequence[Sequence[int]],
    val_fraction: float,
    seed: int,
    max_train_per_task: int | None,
    max_test_per_task: int | None,
) -> list[Task]:
    tasks: list[Task] = []
    for tid, classes in enumerate(class_groups):
        def select(x, y):
            mask = np.isin(y, classes)
            return x[mask], y[mask].astype(np.int64)

        tr_x, tr_y = select(train_x, train_y)
        te_x, te_y = select(test_x, test_y)
        tr_x, tr_y = subsample(tr_x, tr_y, max_train_per_task, seed + tid)
        te_x, te_y = subsample(te_x, te_y, max_test_per_task, seed + 1000 + tid)
        tr_x, tr_y, va_x, va_y = train_val_split(tr_x, tr_y, val_fraction, seed + tid)
        tasks.append(
            Task(
                task_id=tid,
                name=f"cifar100_task{tid}_{classes[0]}to{classes[-1]}",
                train=to_tensor_dataset(tr_x, tr_y),
                val=to_tensor_dataset(va_x, va_y),
                test=to_tensor_dataset(te_x, te_y),
                global_classes=list(classes),
                n_classes=100,
            )
        )
    return tasks


def build_split_cifar100(cfg: Dict[str, Any], data_root: str | Path = "data") -> ContinualBenchmark:
    seed = int(cfg.get("seed", 0))
    n_classes = int(cfg.get("n_classes", 100))
    classes_per_task = int(cfg.get("classes_per_task", 10))
    val_fraction = float(cfg.get("val_fraction", 0.1))
    max_train_per_task = cfg.get("max_train_per_task")
    max_test_per_task = cfg.get("max_test_per_task")
    shuffle_classes = bool(cfg.get("shuffle_classes", False))

    train_x, train_y, test_x, test_y = _load_cifar100(
        data_root, download=bool(cfg.get("allow_download", True))
    )
    if cfg.get("normalize", True):
        train_x = _normalize_cifar(train_x)
        test_x = _normalize_cifar(test_x)
    else:
        train_x = train_x.astype(np.float32) / 255.0
        test_x = test_x.astype(np.float32) / 255.0

    class_groups = cfg.get("class_groups") or _class_groups(
        n_classes=n_classes,
        classes_per_task=classes_per_task,
        seed=seed,
        shuffle=shuffle_classes,
    )
    tasks = _make_class_incremental_tasks(
        train_x,
        train_y,
        test_x,
        test_y,
        class_groups=class_groups,
        val_fraction=val_fraction,
        seed=seed,
        max_train_per_task=max_train_per_task,
        max_test_per_task=max_test_per_task,
    )
    return ContinualBenchmark(
        name="split_cifar100",
        tasks=tasks,
        input_shape=train_x.shape[1:],
        n_classes_per_task=n_classes,
        multihead=False,
    )
