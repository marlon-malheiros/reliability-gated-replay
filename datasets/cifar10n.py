"""CIFAR-10N: real human-annotation label noise on CIFAR-10 (Wei et al., ICLR 2022).

Replaces the clean CIFAR-10 training labels with the crowdsourced *noisy* human
labels from ``CIFAR-10_human.pt`` while **preserving the clean/reference labels**, so
buffer purity and gate-vs-correctness remain computable (the central requirement of
this study). Built as a Seq-CIFAR-10 class-incremental benchmark (5 tasks x 2 classes,
single head) -- tasks are formed by the *clean* class, and each example carries its
human-noisy training label.

Two protocols (``task_il`` cfg):

* ``task_il=False`` (default): single-head class-IL with global labels 0--9.
* ``task_il=True``: the **projected task-IL CIFAR-10N** bridge, matching native
  Split-CIFAR-10 with binary task-local heads. Tasks are formed from the clean
  class, the CIFAR-10N per-example correctness mask is retained, and every
  incorrect annotation is mapped to the other local class (``1 - clean``).

Approximately 88% of the human errors name a class outside the example's
two-class task and therefore have no representable binary target. Projection
does not preserve that external human confusion target. It preserves the
correctness mask used to compute buffer purity, gate--correctness separation,
and inversion; clean test labels are unchanged. The default class-IL protocol
continues to use the original global human labels and preserves the complete
confusion structure. See ``scripts/sanity_check_cifar10n_taskil.py``.

Label variants (``variant`` cfg): ``aggre`` (~9% noise, aggregated vote), ``worse``
(~40%, worst single annotator), ``random1``/``random2``/``random3`` (~18% each). Test
labels are clean (CIFAR-10N adds no test noise).

If the label file is absent the caller is expected to fall back to documented
synthetic noise; this builder raises ``FileNotFoundError`` so the fallback is explicit
and never silent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .base import ContinualBenchmark, Task, subsample, to_tensor_dataset, train_val_split
from .split_cifar10 import _class_groups, _load_cifar10, _normalize_cifar

_VARIANT_KEY = {
    "aggre": "aggre_label",
    "worse": "worse_label",
    "random1": "random_label1",
    "random2": "random_label2",
    "random3": "random_label3",
    "clean": "clean_label",
}


def _human_labels(data_root: str | Path, variant: str) -> np.ndarray:
    path = Path(data_root) / "cifar-10n" / "CIFAR-10_human.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"CIFAR-10N labels not found at {path}. Run "
            "`bash scripts/setup_neural_networks_submission.sh --with-cifar10n`, "
            "or the runner will use the documented synthetic-noise fallback."
        )
    key = _VARIANT_KEY.get(variant)
    if key is None:
        raise ValueError(f"Unknown CIFAR-10N variant '{variant}'. Known: {sorted(_VARIANT_KEY)}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(blob[key], dtype=np.int64).reshape(-1)


def build_cifar10n(cfg: Dict[str, Any], data_root: str | Path = "data") -> ContinualBenchmark:
    seed = int(cfg.get("seed", 0))
    variant = str(cfg.get("variant", "aggre"))
    n_classes = int(cfg.get("n_classes", 10))
    classes_per_task = int(cfg.get("classes_per_task", 2))
    val_fraction = float(cfg.get("val_fraction", 0.05))
    max_train = cfg.get("max_train_per_task")
    max_test = cfg.get("max_test_per_task")
    task_il = bool(cfg.get("task_il", False))

    if task_il and classes_per_task != 2:
        raise ValueError(
            "Projected task-IL CIFAR-10N requires classes_per_task=2 because "
            "incorrect labels are mapped to the other binary class."
        )

    train_x, train_y_clean, test_x, test_y = _load_cifar10(
        data_root, download=bool(cfg.get("allow_download", True))
    )
    noisy_all = _human_labels(data_root, variant)  # aligned to torchvision train order
    if noisy_all.shape[0] != train_y_clean.shape[0]:
        raise RuntimeError("CIFAR-10N label length mismatch with CIFAR-10 train set.")

    if cfg.get("normalize", True):
        train_x = _normalize_cifar(train_x)
        test_x = _normalize_cifar(test_x)
    else:
        train_x = train_x.astype(np.float32) / 255.0
        test_x = test_x.astype(np.float32) / 255.0

    groups = cfg.get("class_groups") or _class_groups(n_classes, classes_per_task, seed, False)
    tasks: list[Task] = []
    for tid, classes in enumerate(groups):
        tr_mask = np.isin(train_y_clean, classes)
        te_mask = np.isin(test_y, classes)
        idx = np.where(tr_mask)[0]
        # subsample by index so noisy labels stay aligned to images
        if max_train is not None and idx.size > max_train:
            rng = np.random.RandomState(seed + tid)
            idx = rng.permutation(idx)[:max_train]
        tr_x = train_x[idx]
        tr_clean_g = train_y_clean[idx].astype(np.int64)
        tr_noisy_g = noisy_all[idx].astype(np.int64)
        is_noisy = tr_noisy_g != tr_clean_g
        te_x = test_x[te_mask]
        te_y_g = test_y[te_mask].astype(np.int64)
        te_x, te_y_g = subsample(te_x, te_y_g, max_test, seed + 1000 + tid)

        if task_il:
            if len(classes) != 2:
                raise ValueError(
                    "Every projected task-IL CIFAR-10N class group must contain "
                    "exactly two classes."
                )
            local = {int(global_class): i for i, global_class in enumerate(classes)}
            clean_local = np.asarray(
                [local[int(label)] for label in tr_clean_g], dtype=np.int64
            )
            train_label = np.where(is_noisy, 1 - clean_local, clean_local).astype(
                np.int64
            )
            clean_label = clean_local
            te_y = np.asarray([local[int(label)] for label in te_y_g], dtype=np.int64)
            task_n_classes = 2
        else:
            train_label = tr_noisy_g
            clean_label = tr_clean_g
            te_y = te_y_g
            task_n_classes = n_classes

        # Split aligned image, training-label, clean-label, and correctness arrays.
        n = tr_x.shape[0]
        order = np.random.RandomState(seed + tid).permutation(n)
        n_val = int(round(val_fraction * n))
        v_idx, t_idx = order[:n_val], order[n_val:]
        train_ds = to_tensor_dataset(tr_x[t_idx], train_label[t_idx])
        val_ds = to_tensor_dataset(tr_x[v_idx], clean_label[v_idx])
        suffix = "_taskil" if task_il else ""
        task = Task(
            task_id=tid,
            name=f"cifar10n_{variant}{suffix}_t{tid}_{classes[0]}to{classes[-1]}",
            train=train_ds,
            val=val_ds,
            test=to_tensor_dataset(te_x, te_y),
            global_classes=list(classes),
            n_classes=task_n_classes,
        )
        task.train_y_clean = clean_label[t_idx].copy()
        task.train_is_noisy = is_noisy[t_idx].copy()
        task.noise_rate = float(task.train_is_noisy.mean())
        tasks.append(task)

    suffix = "_taskil" if task_il else ""
    return ContinualBenchmark(
        name=f"cifar10n_{variant}{suffix}",
        tasks=tasks,
        input_shape=train_x.shape[1:],
        n_classes_per_task=2 if task_il else n_classes,
        multihead=task_il,
    )
