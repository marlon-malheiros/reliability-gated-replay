"""Error-Gated PNN Maturation -- the contribution, as a ``ContinualMethod``.

Wires together importance (``importance.py``), the error-gated closure detector
(``closure.py``), gating (``gating.py``) and the P-state manager
(``consolidation.py``). Per epoch it derives a ``closure_signal`` (from the
chosen closure mode), deepens consolidation of important parameters, and -- when
error spikes -- reopens plasticity. Optional Replay (A15) and Sleep/Reactivation
offline consolidation are toggled from config.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base import ContinualMethod
from ..replay_buffer import (
    FeatureBuffer,
    LogitReplayBuffer,
    ReplayBuffer,
    derpp_loss,
    replay_ce_loss,
)
from .closure import ClosureDetector
from .consolidation import PNNConsolidation
from .schedule import SnapshotSchedule


class PNNMethod(ContinualMethod):
    name = "pnn"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        self.closure_mode = self.cfg.get("closure_mode", "error_gated")
        self.fixed_closure_epoch = int(self.cfg.get("fixed_closure_epoch", 3))
        self.reopen_factor = float(self.cfg.get("reopening", {}).get("factor", 0.3))
        self.reopen_enabled = bool(self.cfg.get("reopening", {}).get("enabled", True))

        rcfg = self.cfg.get("replay", {})
        self.replay_enabled = bool(rcfg.get("enabled", False))
        self.replay_n = int(rcfg.get("batch_size", 32))
        self.replay_distill = bool(rcfg.get("distill", False))
        self.replay_alpha = float(rcfg.get("alpha", 0.5))
        self.replay_beta = float(rcfg.get("beta", 1.0))
        if self.replay_enabled and self.replay_distill:
            self.replay_buf = LogitReplayBuffer(int(rcfg.get("buffer_size", 500)))
        elif self.replay_enabled:
            self.replay_buf = ReplayBuffer(int(rcfg.get("buffer_size", 500)))
        else:
            self.replay_buf = None

        scfg = self.cfg.get("sleep", {})
        self.sleep_enabled = bool(scfg.get("enabled", False))
        self.sleep_epochs = int(scfg.get("epochs", 2))
        self.sleep_lr = float(scfg.get("lr", 1e-3))
        self.sleep_strength = float(scfg.get("strength", 1.0))
        self.sleep_per_task = int(scfg.get("buffer_per_task", 200))
        self.feature_buf = FeatureBuffer() if self.sleep_enabled else None

        # --- task-free online consolidation (StreamTrainer path) ---
        ccfg = self.cfg.get("consolidation", {})
        self.schedule_cfg = ccfg
        self.detector_period = max(int(self.cfg.get("detector_period", 20)), 1)
        self.snapshot_pulse = float(ccfg.get("pulse", 0.0))

        self.consol: PNNConsolidation | None = None
        self.detector: ClosureDetector | None = None
        self.schedule: SnapshotSchedule | None = None
        self._global_epoch = 0
        self._closure_snapshot: Dict[str, Any] | None = None
        self._device = None
        # online state (re-initialised in on_task_start)
        self.stream_log: list[Dict[str, Any]] = []
        self._loss_window: deque = deque(maxlen=self.detector_period)
        self._theta_star_steps: list[int] = []
        self._closure_event_steps: list[int] = []

    def _closure_cfg(self) -> Dict[str, Any]:
        """Build the ClosureDetector config, with reopening settings correctly wired.

        Reopening threshold/patience are *canonically* configured under the
        ``reopening:`` block (``threshold``/``epochs``), alongside ``enabled`` and
        ``factor``. The detector itself reads ``reopening_threshold`` /
        ``reopening_epochs`` / ``reopen_enabled`` from the dict it is given, so we
        merge the ``reopening:`` block into the ``closure:`` block here. For
        back-compat we also honor legacy ``closure.reopening_{threshold,epochs}``
        (used by the task-free configs): the ``reopening:`` block wins when present.
        """
        ccfg = dict(self.cfg.get("closure", {}))  # may carry legacy reopening_* keys
        reop = self.cfg.get("reopening", {}) or {}
        if "threshold" in reop:
            ccfg["reopening_threshold"] = reop["threshold"]
        if "epochs" in reop:
            ccfg["reopening_epochs"] = reop["epochs"]
        ccfg["reopen_enabled"] = self.reopen_enabled
        return ccfg

    # --- lifecycle ---
    def on_task_start(self, task_id: int, model: nn.Module) -> None:
        self._device = next(model.parameters()).device
        if self.consol is None:
            self.consol = PNNConsolidation(model, self.cfg)
            self.detector = ClosureDetector(self._closure_cfg())
            # online snapshot schedule (seeded off the global RNG -> per-seed runs)
            self.schedule = SnapshotSchedule(
                self.schedule_cfg, seed=int(np.random.randint(0, 2**31 - 1))
            )
            self._loss_window = deque(maxlen=self.detector_period)
            self.stream_log = []
            self._theta_star_steps = []
            self._closure_event_steps = []

    def extra_loss(self, model, x, y, task_id):
        loss = x.new_zeros(())
        if self.replay_enabled and self.replay_buf is not None and len(self.replay_buf) > 0:
            samples = self.replay_buf.sample(self.replay_n)
            if self.replay_distill:
                loss = loss + derpp_loss(
                    model,
                    samples,
                    self._device,
                    alpha=self.replay_alpha,
                    beta=self.replay_beta,
                )
            else:
                loss = loss + replay_ce_loss(model, samples, self._device)
        if self.consol is not None:
            anchor = self.consol.anchor_loss()
            if anchor is not None:
                loss = loss + anchor
        return loss

    def modify_gradients(self, model):
        assert self.consol is not None
        self.consol.update_importance()   # importance from true (pre-gating) grads
        self.consol.apply_gating()        # throttle by exp(-beta P) or (1-P)
        self.consol.apply_freeze()        # hard-freeze consolidated elements (if enabled)

    def on_batch_end(self, model, x, y, task_id):
        if self.replay_enabled and self.replay_buf is not None:
            if self.replay_distill:
                with torch.no_grad():
                    logits = model(x, task_id)
                self.replay_buf.add(x, y, task_id, logits)
            else:
                self.replay_buf.add(x, y, task_id)

    def on_epoch_end(self, model, val_metrics, epoch) -> bool:
        assert self.consol is not None and self.detector is not None
        ge = self._global_epoch
        self._global_epoch += 1

        rec: Dict[str, Any] = {}
        if self.closure_mode == "error_gated":
            sig = self.detector.update(val_metrics["loss"])
            cs = sig["closure_signal"]
            if sig["just_reopened"] and self.reopen_enabled:
                self.consol.reopen(self.reopen_factor)
                rec["reopened"] = True
            if sig["closed"] and self._closure_snapshot is None:
                self._closure_snapshot = {
                    "closure_epoch": ge,
                    "closure_error": val_metrics["loss"],
                    "closure_acc": val_metrics["acc"],
                }
            rec.update({k: sig[k] for k in ("E_t", "dE_t", "var_E_t", "closure_signal", "closed")})
        elif self.closure_mode == "immediate":
            cs = 1.0
        elif self.closure_mode in ("fixed", "delayed"):
            cs = 1.0 if ge >= self.fixed_closure_epoch else 0.0
        else:  # "none"
            cs = 0.0

        self.consol.mature(cs)
        self.consol.adapt_anchor_lambda(float(rec.get("closure_signal", cs)))
        rec["closure_signal"] = rec.get("closure_signal", cs)
        rec["mean_P"] = float(self.consol._flat_P().mean())
        rec["anchor_lambda"] = self.consol.anchor_lambda
        self.history.append(rec)
        return False

    # --- task-free online path (StreamTrainer) ---
    def on_stream_step(self, model, step, x, y, loss, n_steps=0, at_boundary=False) -> None:
        """Online consolidation: drive the plateau detector from the running train
        loss (shared, error-gated maturation), then snapshot theta* per the schedule
        (the ablated variable). Detector + maturation tick every ``detector_period``
        steps; snapshot triggers are checked every step."""
        if self.consol is None or self.schedule is None:
            return
        self._loss_window.append(float(loss))
        is_tick = (step % self.detector_period == 0)

        sig: Dict[str, Any] | None = None
        cs = 0.0
        closure_event = False
        if is_tick:
            running = sum(self._loss_window) / max(len(self._loss_window), 1)
            if self.closure_mode == "error_gated":
                sig = self.detector.update(running)
                cs = sig["closure_signal"]
                closure_event = bool(sig["closure_event"])
                if sig["just_reopened"] and self.reopen_enabled:
                    self.consol.reopen(self.reopen_factor)
            elif self.closure_mode == "immediate":
                cs = 1.0
            elif self.closure_mode in ("fixed", "delayed"):
                cs = 1.0 if step >= self.fixed_closure_epoch * self.detector_period else 0.0
            # shared maturation substrate (same across snapshot schedules)
            self.consol.mature(cs)
            self.consol.adapt_anchor_lambda(cs)
            self._loss_window.clear()

        # theta* snapshot schedule (the ablated variable)
        if self.schedule.fire(step, closure_event, at_boundary):
            self.consol.snapshot_anchor()
            self.consol.freeze_consolidated()  # hard/discrete consolidation (if enabled)
            if self.snapshot_pulse > 0.0:
                self.consol.mature(self.snapshot_pulse)
            if self.consol.anchor_enabled or self.consol.freeze_enabled:
                self._theta_star_steps.append(step)
        if closure_event:
            self._closure_event_steps.append(step)

        if is_tick or (self._theta_star_steps and self._theta_star_steps[-1] == step):
            self._log_stream_tick(model, step, x, y, loss, sig, closure_event, is_tick)

    def _log_stream_tick(self, model, step, x, y, loss, sig, closure_event, heavy) -> None:
        consol = self.consol
        rec: Dict[str, Any] = {
            "step": int(step),
            "running_loss": float(loss),
            "closure_signal": float(sig["closure_signal"]) if sig else 0.0,
            "E_t": (float(sig["E_t"]) if sig and sig["E_t"] is not None else None),
            "dE_t": (float(sig["dE_t"]) if sig and sig["dE_t"] is not None else None),
            "var_E_t": (float(sig["var_E_t"]) if sig and sig["var_E_t"] is not None else None),
            "closure_event": bool(closure_event),
            "theta_star_update": bool(self._theta_star_steps and self._theta_star_steps[-1] == step),
            "mean_P": float(consol._flat_P().mean()) if consol.named else 0.0,
            "anchor_lambda": float(consol.anchor_lambda),
            "anchor_loss": self._anchor_loss_value(),
            "anchor_grad_norm": consol.anchor_grad_norm(),
            "param_displacement": consol.param_displacement(),
            "n_theta_star_updates": int(consol.anchor_updates),
        }
        rec.update(consol.P_percentiles())
        if heavy:  # heavier diagnostics only on detector ticks
            rec["task_grad_norm"] = self._grad_norm_of_task(model, x, y)
            rec["replay_grad_norm"] = self._grad_norm_of_replay(model)
            rec["layerwise_anchor_activity"] = consol.layerwise_anchor_activity()
        self.stream_log.append(rec)

    def _anchor_loss_value(self) -> float:
        a = self.consol.anchor_loss() if self.consol is not None else None
        return float(a.item()) if a is not None else 0.0

    def _grad_norm_of_task(self, model, x, y) -> float:
        try:
            params = [p for _, p in self.consol.named]
            ce = F.cross_entropy(model(x, 0), y)
            g = torch.autograd.grad(ce, params, retain_graph=False, allow_unused=True)
            return float(sum((gi.detach() ** 2).sum() for gi in g if gi is not None) ** 0.5)
        except Exception:
            return 0.0

    def _grad_norm_of_replay(self, model) -> float:
        if not (self.replay_enabled and self.replay_buf is not None and len(self.replay_buf) > 0):
            return 0.0
        try:
            params = [p for _, p in self.consol.named]
            samples = self.replay_buf.sample(self.replay_n)
            if self.replay_distill:
                rloss = derpp_loss(
                    model, samples, self._device, alpha=self.replay_alpha, beta=self.replay_beta
                )
            else:
                rloss = replay_ce_loss(model, samples, self._device)
            g = torch.autograd.grad(rloss, params, retain_graph=False, allow_unused=True)
            return float(sum((gi.detach() ** 2).sum() for gi in g if gi is not None) ** 0.5)
        except Exception:
            return 0.0

    def on_task_end(self, task_id, model, train_loader: DataLoader) -> None:
        if self.sleep_enabled and self.feature_buf is not None:
            self._store_features(model, train_loader, task_id)
        if self.consol is not None:
            self.consol.snapshot_anchor()

    def offline_phase(self, model, task_id) -> None:
        if self.sleep_enabled:
            self._sleep(model)

    # --- sleep / reactivation: generator-free offline consolidation ---
    @torch.no_grad()
    def _store_features(self, model, train_loader, task_id):
        model.eval()
        stored = 0
        for x, y in train_loader:
            x = x.to(self._device)
            feats = model.features(x)
            self.feature_buf.add(feats, y, task_id)
            stored += x.shape[0]
            if stored >= self.sleep_per_task:
                break

    def _sleep(self, model):
        """Offline epochs: reactivate stored features through heads + deepen P.

        No raw inputs are replayed (that is the Replay baseline); only internal
        activations are reactivated, and consolidation continues to mature.
        """
        if len(self.feature_buf) == 0:
            return
        head_params = [p for p in model.heads.parameters() if p.requires_grad]
        opt = torch.optim.Adam(head_params, lr=self.sleep_lr)
        model.train()
        for _ in range(self.sleep_epochs):
            for feats, ys, ts in self.feature_buf.batches(batch_size=64):
                feats, ys = feats.to(self._device), ys.to(self._device)
                opt.zero_grad()
                loss = feats.new_zeros(())
                count = 0
                for t in ts.unique():
                    m = ts == t
                    out = model.head(int(t))(feats[m].to(self._device))
                    loss = loss + F.cross_entropy(out, ys[m], reduction="sum")
                    count += int(m.sum())
                (loss / max(count, 1)).backward()
                opt.step()
            # deepen consolidation offline (closure forced high during "sleep")
            self.consol.mature(self.sleep_strength)

    # --- diagnostics ---
    def consolidation_state(self, model) -> Dict[str, Any]:
        if self.consol is None:
            return {}
        st = self.consol.state()
        if self._closure_snapshot:
            st.update(self._closure_snapshot)
        st["closure_mode"] = self.closure_mode
        st["replay_enabled"] = self.replay_enabled
        st["replay_distill"] = self.replay_distill
        st["replay_alpha"] = self.replay_alpha
        st["replay_beta"] = self.replay_beta
        st["sleep_enabled"] = self.sleep_enabled
        # task-free online consolidation timing
        st["snapshot_schedule"] = self.schedule.mode if self.schedule else None
        st["closure_event_steps"] = list(self._closure_event_steps)
        st["theta_star_update_steps"] = list(self._theta_star_steps)
        st["n_closure_events"] = int(self.detector.n_events) if self.detector else 0
        st["n_theta_star_updates"] = int(self.consol.anchor_updates)
        return st
