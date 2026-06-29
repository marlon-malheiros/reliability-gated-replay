"""When to snapshot theta* in the online (task-free) path.

This is the *ablated variable* of the task-free pivot. Maturation, gating and
reopening are the shared error-gated substrate (unchanged across schedules); only
the anchor-snapshot timing differs here:

    none           -> never (anchor effectively off)
    task_boundary  -> at nominal region boundaries (oracle that knows task changes)
    fixed_interval -> every ``interval_steps`` steps
    random         -> Bernoulli(1/interval_steps) per step (event count matched to
                      fixed_interval -- a fair random control)
    error_gated    -> on the closure detector's re-armable closure events (main)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

MODES = ("none", "task_boundary", "fixed_interval", "random", "error_gated")


class SnapshotSchedule:
    def __init__(self, cfg: Dict[str, Any] | None = None, seed: int = 0):
        cfg = cfg or {}
        self.mode = cfg.get("schedule", "error_gated")
        if self.mode not in MODES:
            raise ValueError(f"Unknown snapshot schedule '{self.mode}'. Use one of {MODES}.")
        self.interval_steps = max(int(cfg.get("interval_steps", 200)), 1)
        self.random_p = 1.0 / self.interval_steps
        self._rng = np.random.RandomState(seed + 12345)
        self.n_events = 0

    def fire(self, step: int, closure_event: bool, at_boundary: bool) -> bool:
        m = self.mode
        if m == "none":
            fired = False
        elif m == "task_boundary":
            fired = bool(at_boundary)
        elif m == "fixed_interval":
            fired = step > 0 and (step % self.interval_steps == 0)
        elif m == "random":
            fired = bool(self._rng.rand() < self.random_p)
        else:  # error_gated
            fired = bool(closure_event)
        if fired:
            self.n_events += 1
        return fired
