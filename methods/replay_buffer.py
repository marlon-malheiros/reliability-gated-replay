"""Reservoir buffers shared by Replay, PNN+Replay, and the Sleep condition.

``ReplayBuffer`` stores raw inputs (experience replay). ``FeatureBuffer`` stores
penultimate *activations* -- the substrate for the Sleep/Reactivation condition's
generator-free pseudo-rehearsal, which deliberately keeps no raw inputs.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity: int = 500, seed: int = 0):
        self.capacity = capacity
        self.x: List[torch.Tensor] = []
        self.y: List[int] = []           # stored (possibly noisy) label -- rehearsed
        self.yc: List[int] = []          # true label (for purity audit only)
        self.t: List[int] = []
        self._seen = 0
        self._rng = torch.Generator().manual_seed(seed)

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        y_clean: torch.Tensor | None = None,
    ) -> None:
        x = x.detach().cpu()
        y = y.detach().cpu()
        yc = y if y_clean is None else y_clean.detach().cpu()
        for i in range(x.shape[0]):
            self._seen += 1
            if len(self.x) < self.capacity:
                self.x.append(x[i])
                self.y.append(int(y[i]))
                self.yc.append(int(yc[i]))
                self.t.append(task_id)
            else:
                j = int(torch.randint(0, self._seen, (1,), generator=self._rng).item())
                if j < self.capacity:
                    self.x[j], self.y[j], self.t[j] = x[i], int(y[i]), task_id
                    self.yc[j] = int(yc[i])

    def __len__(self) -> int:
        return len(self.x)

    def purity(self) -> float:
        """Fraction of buffered examples whose stored label is the true label."""
        if not self.y:
            return 1.0
        return float(np.mean([int(a == b) for a, b in zip(self.y, self.yc)]))

    def sample(self, n: int) -> List[Tuple[torch.Tensor, int, int]]:
        if len(self.x) == 0:
            return []
        idx = torch.randint(0, len(self.x), (n,), generator=self._rng).tolist()
        return [(self.x[i], self.y[i], self.t[i]) for i in idx]


def replay_ce_loss(model, samples, device) -> torch.Tensor:
    """Mean cross-entropy over replayed (x, y, task) samples, grouped by head."""
    if not samples:
        return next(model.parameters()).new_zeros(())
    by_task: dict = {}
    for x, y, t in samples:
        by_task.setdefault(t, ([], []))
        by_task[t][0].append(x)
        by_task[t][1].append(y)
    total = next(model.parameters()).new_zeros(())
    count = 0
    for t, (xs, ys) in by_task.items():
        xb = torch.stack(xs).to(device)
        yb = torch.tensor(ys, dtype=torch.long, device=device)
        out = model(xb, t)
        total = total + F.cross_entropy(out, yb, reduction="sum")
        count += yb.numel()
    return total / max(count, 1)


class LogitReplayBuffer(ReplayBuffer):
    """Reservoir buffer that also stores logits for DER++ distillation."""

    def __init__(self, capacity: int = 500, seed: int = 0):
        super().__init__(capacity=capacity, seed=seed)
        self.logits: List[torch.Tensor] = []

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        logits: torch.Tensor,
        y_clean: torch.Tensor | None = None,
    ) -> None:
        x = x.detach().cpu()
        y = y.detach().cpu()
        yc = y if y_clean is None else y_clean.detach().cpu()
        logits = logits.detach().cpu()
        for i in range(x.shape[0]):
            self._seen += 1
            if len(self.x) < self.capacity:
                self.x.append(x[i])
                self.y.append(int(y[i]))
                self.yc.append(int(yc[i]))
                self.t.append(task_id)
                self.logits.append(logits[i])
            else:
                j = int(torch.randint(0, self._seen, (1,), generator=self._rng).item())
                if j < self.capacity:
                    self.x[j] = x[i]
                    self.y[j] = int(y[i])
                    self.yc[j] = int(yc[i])
                    self.t[j] = task_id
                    self.logits[j] = logits[i]

    def sample(self, n: int) -> List[Tuple[torch.Tensor, int, int, torch.Tensor]]:
        if len(self.x) == 0:
            return []
        idx = torch.randint(0, len(self.x), (n,), generator=self._rng).tolist()
        return [(self.x[i], self.y[i], self.t[i], self.logits[i]) for i in idx]


def derpp_loss(
    model,
    samples,
    device,
    alpha: float = 0.5,
    beta: float = 1.0,
) -> torch.Tensor:
    """DER++ replay loss: stored-logit distillation plus replay CE."""
    if not samples:
        return next(model.parameters()).new_zeros(())

    by_task: dict = {}
    for x, y, t, logits in samples:
        by_task.setdefault(t, ([], [], []))
        by_task[t][0].append(x)
        by_task[t][1].append(y)
        by_task[t][2].append(logits)

    total_ce = next(model.parameters()).new_zeros(())
    total_mse = next(model.parameters()).new_zeros(())
    n_ce = 0
    n_logits = 0
    for t, (xs, ys, logits) in by_task.items():
        xb = torch.stack(xs).to(device)
        yb = torch.tensor(ys, dtype=torch.long, device=device)
        target_logits = torch.stack(logits).to(device)
        out = model(xb, t)
        total_ce = total_ce + F.cross_entropy(out, yb, reduction="sum")
        total_mse = total_mse + F.mse_loss(out, target_logits, reduction="sum")
        n_ce += yb.numel()
        n_logits += target_logits.numel()

    ce = total_ce / max(n_ce, 1)
    mse = total_mse / max(n_logits, 1)
    return beta * ce + alpha * mse


class FeatureBuffer:
    """Stores (feature, label, task_id) for feature-space reactivation."""

    def __init__(self):
        self.f: List[torch.Tensor] = []
        self.y: List[int] = []
        self.t: List[int] = []

    def add(self, feats: torch.Tensor, y: torch.Tensor, task_id: int) -> None:
        feats = feats.detach().cpu()
        y = y.detach().cpu()
        for i in range(feats.shape[0]):
            self.f.append(feats[i])
            self.y.append(int(y[i]))
            self.t.append(task_id)

    def __len__(self) -> int:
        return len(self.f)

    def batches(self, batch_size: int, shuffle: bool = True):
        n = len(self.f)
        order = torch.randperm(n).tolist() if shuffle else list(range(n))
        for i in range(0, n, batch_size):
            sel = order[i : i + batch_size]
            feats = torch.stack([self.f[j] for j in sel])
            ys = torch.tensor([self.y[j] for j in sel], dtype=torch.long)
            ts = torch.tensor([self.t[j] for j in sel], dtype=torch.long)
            yield feats, ys, ts
