"""Task-free (blurry-boundary) stream construction.

Turns a task-structured :class:`ContinualBenchmark` into a single ordered batch
stream with **no task-boundary signals**: the method only ever sees ``(x, y)``
batches and a single shared head over global labels. This is the data side of the
task-free pivot; per-task *test* sets are still used (by the trainer) to measure
per-task accuracy / forgetting, but they are never shown to the method.

Convention -- blurry class-incremental (cf. Bang et al. 2021 *Rainbow Memory*;
Koh et al. 2022 *i-Blurry*); the disjoint limit recovers Aljundi et al. 2019
*Task-Free Continual Learning*:

* Each task owns a contiguous *region* of the stream.
* Region ``t`` contains all of task ``t``'s training samples (traversed
  ``local_epochs`` times, shuffled within the region), plus a ``blur_ratio``
  fraction of task ``t+1``'s samples sprinkled in -- a gradual onset of the next
  distribution.
* ``blur_ratio == 0`` -> disjoint task-free stream (clean control for ablations).

Memory note: we keep a single image pool and only an integer ``order`` array, so
``local_epochs > 1`` and ``blur_ratio > 0`` do not duplicate image data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch

from .base import ContinualBenchmark


@dataclass
class TaskFreeStream:
    x: torch.Tensor              # image pool (concatenated tasks, unordered)
    y: torch.Tensor              # global labels for the pool
    order: np.ndarray            # stream order: indices into the pool
    region_of: np.ndarray        # dominant-task id at each stream position
    source_task_of: np.ndarray   # true source-task id at each stream position
    boundaries: List[int]        # stream positions (sample index) where each region starts
    n_tasks: int
    blur_ratio: float = 0.0
    local_epochs: int = 1

    def __len__(self) -> int:
        return int(self.order.shape[0])

    def batches(self, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
        """Yield ``(x, y, region)`` mini-batches in stream order (CPU tensors)."""
        n = len(self)
        for start in range(0, n, batch_size):
            idx = self.order[start : start + batch_size]
            yield self.x[idx], self.y[idx], self.region_of[start : start + batch_size]


def build_task_free_stream(
    benchmark: ContinualBenchmark, cfg: Dict[str, Any] | None = None
) -> TaskFreeStream:
    cfg = cfg or {}
    blur_ratio = float(cfg.get("blur_ratio", 0.0))
    local_epochs = int(cfg.get("local_epochs", cfg.get("stream_passes", 1)))
    seed = int(cfg.get("stream_seed", cfg.get("seed", 0)))
    rng = np.random.RandomState(seed)
    # irregular: variable, UNANNOUNCED region lengths (dwell time ~ U[min,max] x base).
    # This is the setting where a fixed/random snapshot cadence cannot align with the
    # (unknown) shift points, so an adaptive error-gated detector can earn its keep.
    irregular = bool(cfg.get("irregular", False))
    irr_min = float(cfg.get("irregular_min", 0.4))
    irr_max = float(cfg.get("irregular_max", 1.6))

    # concatenate per-task training tensors into one pool (global labels)
    xs, ys, owner = [], [], []
    for t, task in enumerate(benchmark.tasks):
        tx, ty = task.train.tensors
        xs.append(tx)
        ys.append(ty)
        owner.append(np.full(len(ty), t, dtype=np.int64))
    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    owner = np.concatenate(owner)
    n_tasks = len(benchmark.tasks)
    task_indices = [np.where(owner == t)[0] for t in range(n_tasks)]

    order: List[int] = []
    region_ids: List[int] = []
    boundaries: List[int] = []
    for t in range(n_tasks):
        boundaries.append(len(order))
        pool = task_indices[t]
        region: List[int] = []
        if irregular:
            # variable dwell time: U[min,max] x (one pass x local_epochs), avg = base
            target = max(int(round(len(pool) * rng.uniform(irr_min, irr_max) * max(local_epochs, 1))), 1)
            region = rng.choice(pool, size=target, replace=target > len(pool)).tolist()
        else:
            for _ in range(max(local_epochs, 1)):
                region.extend(pool.tolist())
        # blur: sprinkle a preview of the next task's data into this region
        if blur_ratio > 0 and t + 1 < n_tasks:
            nxt = task_indices[t + 1]
            k = int(round(blur_ratio * len(region)))
            if k > 0 and len(nxt) > 0:
                borrow = rng.choice(nxt, size=k, replace=k > len(nxt))
                region.extend(borrow.tolist())
        region_arr = np.asarray(region, dtype=np.int64)
        rng.shuffle(region_arr)
        order.extend(region_arr.tolist())
        region_ids.extend([t] * len(region_arr))

    order_arr = np.asarray(order, dtype=np.int64)
    return TaskFreeStream(
        x=X,
        y=Y,
        order=order_arr,
        region_of=np.asarray(region_ids, dtype=np.int64),
        source_task_of=owner[order_arr],
        boundaries=boundaries,
        n_tasks=n_tasks,
        blur_ratio=blur_ratio,
        local_epochs=max(local_epochs, 1),
    )
