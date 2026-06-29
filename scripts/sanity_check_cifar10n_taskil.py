#!/usr/bin/env python3
"""Validate the projected task-IL CIFAR-10N construction without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.cifar10n import build_cifar10n  # noqa: E402


EXPECTED_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
PUBLISHED_RATE = {"aggre": 0.090, "worse": 0.402}


def labels(dataset) -> np.ndarray:
    return dataset.tensors[1].numpy().astype(np.int64)


def check_variant(
    variant: str, seed: int, max_train: int | None, data_root: Path
) -> bool:
    base = {
        "variant": variant,
        "seed": seed,
        "n_classes": 10,
        "classes_per_task": 2,
        "val_fraction": 0.05,
        "max_train_per_task": max_train,
        "allow_download": True,
    }
    class_il = build_cifar10n({**base, "task_il": False}, data_root=data_root)
    task_il = build_cifar10n({**base, "task_il": True}, data_root=data_root)

    ok = True

    def require(condition: bool, message: str) -> None:
        nonlocal ok
        print(f"    [{'OK' if condition else 'FAIL'}] {message}")
        ok &= bool(condition)

    print(f"\n=== CIFAR-10N {variant}, seed {seed} ===")
    require(task_il.multihead and not class_il.multihead, "task-IL uses multiple heads")
    require(task_il.n_classes_per_task == 2, "task-IL heads are binary")
    require(
        [task.global_classes for task in task_il.tasks] == EXPECTED_GROUPS,
        "clean-class pairs match native Split-CIFAR-10",
    )

    wrong_total = total = outside_total = 0
    class_rates: list[float] = []
    task_rates: list[float] = []
    for class_task, projected_task in zip(class_il.tasks, task_il.tasks):
        projected = labels(projected_task.train)
        clean_local = np.asarray(projected_task.train_y_clean, dtype=np.int64)
        projected_mask = np.asarray(projected_task.train_is_noisy, dtype=bool)
        class_mask = np.asarray(class_task.train_is_noisy, dtype=bool)

        require(
            np.array_equal(projected_mask, class_mask),
            f"task {projected_task.task_id}: correctness mask is identical",
        )
        require(
            set(np.unique(projected)).issubset({0, 1})
            and set(np.unique(labels(projected_task.test))).issubset({0, 1}),
            f"task {projected_task.task_id}: train and test labels are binary",
        )
        require(
            np.all(projected[projected_mask] == 1 - clean_local[projected_mask])
            and np.all(projected[~projected_mask] == clean_local[~projected_mask]),
            f"task {projected_task.task_id}: projection rule is exact",
        )

        class_rates.append(class_task.noise_rate)
        task_rates.append(projected_task.noise_rate)
        clean_global = np.asarray(class_task.train_y_clean, dtype=np.int64)
        noisy_global = labels(class_task.train)
        wrong = noisy_global != clean_global
        outside = np.asarray(
            [value not in class_task.global_classes for value in noisy_global[wrong]],
            dtype=bool,
        )
        wrong_total += int(wrong.sum())
        total += int(wrong.size)
        outside_total += int(outside.sum())

    require(
        np.allclose(class_rates, task_rates, atol=0.0, rtol=0.0),
        "per-task noise incidence is preserved",
    )
    overall = wrong_total / total
    require(
        abs(overall - PUBLISHED_RATE[variant]) < 0.03,
        f"overall rate {overall:.3f} matches approximately {PUBLISHED_RATE[variant]:.3f}",
    )
    print(
        f"    overall={overall:.3f}; outside-task errors="
        f"{100 * outside_total / wrong_total:.1f}%"
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="aggre,worse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    ok = all(
        check_variant(variant.strip(), args.seed, args.max_train, args.data_root)
        for variant in args.variants.split(",")
    )
    print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
