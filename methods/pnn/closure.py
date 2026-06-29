"""Error-gated critical-period closure detector.

No fixed closure epoch: the network stays plastic while validation error is high,
improving fast, or unstable. We track an EMA of validation loss ``E_t``, its
improvement over a window ``dE_t``, and its variance ``Var_E_t``, then emit a
continuous ``closure_signal`` in [0,1] that drives gradual PNN maturation:

    closure_signal = sigmoid( gamma * [ (thr_E  - E_t)
                                      + (thr_dE - dE_t)
                                      + (thr_var - Var_E_t) ] )

A discrete *closure epoch* is recorded when all three conditions hold for N
consecutive epochs. Reopening fires when ``E_t`` exceeds a reopening threshold for
M consecutive epochs (e.g. when a new task arrives) -- adult-style plasticity.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict


class ClosureDetector:
    def __init__(self, cfg: Dict[str, Any] | None = None):
        cfg = cfg or {}
        self.thr_E = float(cfg.get("error_threshold", 0.15))
        self.thr_dE = float(cfg.get("improvement_threshold", 0.02))
        self.thr_var = float(cfg.get("stability_threshold", 0.01))
        self.gamma = float(cfg.get("gamma", 5.0))
        self.N = int(cfg.get("consecutive_epochs", 3))
        self.k = int(cfg.get("slope_window", 3))
        self.ema_decay = float(cfg.get("ema_decay", 0.6))
        # reopening
        self.reopen_enabled = bool(cfg.get("reopen_enabled", True))
        self.reopen_thr = float(cfg.get("reopening_threshold", 0.5))
        self.reopen_M = int(cfg.get("reopening_epochs", 2))

        self.E_t: float | None = None
        self.E_hist: deque = deque(maxlen=self.k + 1)
        self.var_hist: deque = deque(maxlen=self.k + 1)
        self._closed_streak = 0
        self._reopen_streak = 0
        self.closed = False
        self.closure_epoch: int | None = None
        self._global_epoch = -1
        # re-armable discrete event for the online (task-free) path: fires once each
        # time the closed condition is (re)entered, and re-arms after it breaks.
        self._armed = True
        self.n_events = 0

    def update(self, val_loss: float) -> Dict[str, Any]:
        """Advance one epoch; return the closure signals for this step."""
        self._global_epoch += 1
        # EMA of validation loss
        self.E_t = val_loss if self.E_t is None else (
            self.ema_decay * self.E_t + (1 - self.ema_decay) * val_loss
        )
        self.E_hist.append(self.E_t)
        # improvement over window: E_{t-k} - E_t  (positive when still improving)
        dE = (self.E_hist[0] - self.E_t) if len(self.E_hist) > self.k else float("inf")
        # variance of recent EMA values
        self.var_hist.append(self.E_t)
        var = _variance(self.var_hist)

        cond = (self.E_t < self.thr_E) and (dE < self.thr_dE) and (var < self.thr_var)
        if cond:
            self._closed_streak += 1
        else:
            self._closed_streak = 0
            self._armed = True  # condition broke -> re-arm for the next stabilization
        if self._closed_streak >= self.N and not self.closed:
            self.closed = True
            self.closure_epoch = self._global_epoch
        # re-armable discrete event (online): rising edge of a sustained closure
        closure_event = False
        if self._closed_streak >= self.N and self._armed:
            closure_event = True
            self._armed = False
            self.n_events += 1

        # continuous closure signal (gradual maturation)
        z = self.gamma * (
            (self.thr_E - self.E_t)
            + (self.thr_dE - (0.0 if math.isinf(dE) else dE))
            + (self.thr_var - var)
        )
        closure_signal = _sigmoid(z)

        # reopening
        just_reopened = False
        if self.reopen_enabled and self.E_t > self.reopen_thr:
            self._reopen_streak += 1
            if self._reopen_streak == self.reopen_M:
                just_reopened = True
        else:
            self._reopen_streak = 0

        return {
            "E_t": self.E_t,
            "dE_t": None if math.isinf(dE) else dE,
            "var_E_t": var,
            "closure_signal": closure_signal,
            "closed": self.closed,
            "closure_event": closure_event,
            "closure_epoch": self.closure_epoch,
            "just_reopened": just_reopened,
        }


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _variance(values) -> float:
    n = len(values)
    if n < 2:
        return float("inf")
    m = sum(values) / n
    return sum((v - m) ** 2 for v in values) / n
