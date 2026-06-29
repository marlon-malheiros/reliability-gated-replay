"""Label-noise injection for continual-learning benchmarks.

The central experimental axis of the reliability-gate pivot: when training labels
are unreliable, does an *internal-model* reliability gate behave differently from
a *supervised-error* gate? This module corrupts a fraction of each task's
**training** labels (val/test stay clean -- "final clean accuracy" is measured on
clean test data) and records, per training example, the true label and a
corruption mask so the analysis can compute buffer purity and the correlation
between a gate's value and true label correctness.

A noise *schedule* decides which tasks are corrupted:

* ``clean``  -- no corruption (rate forced to 0).
* ``all``    -- every task corrupted at ``rate``.
* ``early``  -- only the first ``n_noisy`` tasks (default 1) corrupted.
* ``late``   -- only the last ``n_noisy`` tasks corrupted.
* ``mixed``  -- alternating tasks corrupted (even task ids).

Corruption is symmetric label flipping: a selected label is reassigned uniformly
at random to one of the *other* classes of that task's head.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .base import ContinualBenchmark, Task


def _noisy_task_ids(schedule: str, n_tasks: int, n_noisy: int) -> set:
    schedule = (schedule or "clean").lower()
    if schedule in ("clean", "none"):
        return set()
    if schedule == "all":
        return set(range(n_tasks))
    if schedule == "early":
        return set(range(min(n_noisy, n_tasks)))
    if schedule == "late":
        return set(range(max(0, n_tasks - n_noisy), n_tasks))
    if schedule == "mixed":
        return {t for t in range(n_tasks) if t % 2 == 0}
    raise ValueError(f"Unknown noise schedule '{schedule}'")


def _flip(
    y: np.ndarray,
    rate: float,
    n_classes: int,
    rng: np.random.RandomState,
    noise_type: str = "symmetric",
):
    """Label flip; returns (y_noisy, is_noisy_mask).

    ``symmetric``  -- selected labels go to a uniformly-random *different* class.
    ``asymmetric`` -- class-conditional: selected labels of class c become
                      ``(c+1) mod K`` (the standard generic asymmetric scheme;
                      noise is concentrated on a structured confusion, which makes
                      the corrupted *majority* far easier to form -> a sharper test
                      of the memorization-inversion).
    """
    y = y.copy()
    is_noisy = np.zeros(len(y), dtype=bool)
    if rate <= 0.0 or n_classes < 2:
        return y, is_noisy
    sel = rng.rand(len(y)) < rate
    for i in np.where(sel)[0]:
        if noise_type == "asymmetric":
            y[i] = (y[i] + 1) % n_classes
        else:
            choices = [c for c in range(n_classes) if c != y[i]]
            y[i] = rng.choice(choices)
        is_noisy[i] = True
    return y, is_noisy


def apply_label_noise(
    bench: ContinualBenchmark, cfg: Dict[str, Any], seed: int = 0
) -> ContinualBenchmark:
    """Corrupt training labels in-place per the ``label_noise`` config block.

    Always records ``train_y_clean`` / ``train_is_noisy`` on every task (even at
    rate 0, so downstream code can rely on the fields existing).
    """
    nc = cfg or {}
    rate = float(nc.get("rate", 0.0))
    schedule = str(nc.get("schedule", "clean" if rate <= 0 else "all"))
    noise_type = str(nc.get("noise_type", "symmetric"))
    n_noisy = int(nc.get("n_noisy", 1))
    noisy_ids = _noisy_task_ids(schedule, len(bench.tasks), n_noisy)

    for tid, task in enumerate(bench.tasks):
        x, y = task.train.tensors
        y_np = y.numpy().astype(np.int64)
        task.train_y_clean = y_np.copy()
        if tid in noisy_ids and rate > 0.0:
            rng = np.random.RandomState(seed * 100003 + tid)
            y_noisy, is_noisy = _flip(y_np, rate, task.n_classes, rng, noise_type)
            task.train = TensorDataset(x, torch.from_numpy(y_noisy).long())
            task.train_is_noisy = is_noisy
            task.noise_rate = float(is_noisy.mean())
        else:
            task.train_is_noisy = np.zeros(len(y_np), dtype=bool)
            task.noise_rate = 0.0
    return bench
