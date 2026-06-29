from __future__ import annotations

import numpy as np
import pytest

import datasets.cifar10n as cifar10n


def fake_cifar10(*_args, **_kwargs):
    train_x = np.zeros((4, 3, 2, 2), dtype=np.uint8)
    train_y = np.asarray([0, 1, 2, 3], dtype=np.int64)
    test_x = np.zeros((4, 3, 2, 2), dtype=np.uint8)
    test_y = train_y.copy()
    return train_x, train_y, test_x, test_y


def test_projected_taskil_preserves_error_mask(monkeypatch):
    monkeypatch.setattr(cifar10n, "_load_cifar10", fake_cifar10)
    human = np.asarray([0, 2, 2, 0], dtype=np.int64)
    monkeypatch.setattr(cifar10n, "_human_labels", lambda *_args: human)
    base = {
        "n_classes": 4,
        "classes_per_task": 2,
        "val_fraction": 0.0,
        "normalize": False,
        "allow_download": False,
    }

    class_il = cifar10n.build_cifar10n({**base, "task_il": False})
    task_il = cifar10n.build_cifar10n({**base, "task_il": True})

    assert not class_il.multihead
    assert task_il.multihead
    assert task_il.n_classes_per_task == 2
    assert [task.global_classes for task in task_il.tasks] == [[0, 1], [2, 3]]
    for original, projected in zip(class_il.tasks, task_il.tasks):
        assert np.array_equal(original.train_is_noisy, projected.train_is_noisy)
        noisy_local = projected.train.tensors[1].numpy()
        clean_local = projected.train_y_clean
        mask = projected.train_is_noisy
        assert set(noisy_local.tolist()) <= {0, 1}
        assert np.all(noisy_local[mask] == 1 - clean_local[mask])
        assert np.all(noisy_local[~mask] == clean_local[~mask])


def test_projected_taskil_rejects_nonbinary_tasks():
    with pytest.raises(ValueError, match="classes_per_task=2"):
        cifar10n.build_cifar10n({"task_il": True, "classes_per_task": 3})
