"""Synaptic Intelligence (Zenke et al., 2017).

Accumulates a per-parameter path integral of (-gradient * update) during training;
at task end this is normalized by the squared total displacement to give an
importance Omega, which weights a quadratic penalty toward the last consolidated
weights. The pre-step snapshot is taken in ``modify_gradients`` (which runs right
before the optimizer step) and finalized in ``on_batch_end`` (right after it).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from methods.base import ContinualMethod
from methods.utils import protected_named_parameters


class SI(ContinualMethod):
    name = "si"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        self.lam = float(self.cfg.get("lambda", 1.0))
        self.xi = float(self.cfg.get("xi", 0.1))
        # online (task-free): finalize Omega every ``interval_steps`` instead of at
        # task boundaries (which never occur in the stream).
        self.online = bool(self.cfg.get("online", False))
        self.interval = max(int(self.cfg.get("interval_steps", 200)), 1)
        self._step = 0
        self._finalized = False
        # error-gated stabilization (the bio hypothesis): withhold consolidation while
        # the world is adverse (high error). gate g = sigmoid(gamma*(thr - error)).
        self.error_gated = bool(self.cfg.get("error_gated", False))
        self.err_threshold = float(self.cfg.get("error_threshold", 0.3))
        self.err_gamma = float(self.cfg.get("error_gamma", 10.0))
        self.gate_history: List[Dict[str, float]] = []
        self._named: List[Tuple[str, torch.nn.Parameter]] = []
        self.omega: Dict[str, torch.Tensor] = {}
        self.w: Dict[str, torch.Tensor] = {}
        self.star: Dict[str, torch.Tensor] = {}
        self.theta_task_start: Dict[str, torch.Tensor] = {}
        self._theta_before: Dict[str, torch.Tensor] = {}
        self._g: Dict[str, torch.Tensor] = {}

    def on_task_start(self, task_id, model):
        if not self._named:
            self._named = list(protected_named_parameters(model))
            self.omega = {n: torch.zeros_like(p) for n, p in self._named}
            self.w = {n: torch.zeros_like(p) for n, p in self._named}
            self.star = {n: p.detach().clone() for n, p in self._named}
        self.theta_task_start = {n: p.detach().clone() for n, p in self._named}

    def extra_loss(self, model, x, y, task_id):
        if not self.omega:
            return x.new_zeros(())
        if self.online:
            if not self._finalized:
                return x.new_zeros(())
        elif task_id == 0:
            return x.new_zeros(())
        loss = x.new_zeros(())
        for n, p in self._named:
            loss = loss + (self.omega[n] * (p - self.star[n]).pow(2)).sum()
        return self.lam * loss

    def modify_gradients(self, model):
        # snapshot params and gradients just before the optimizer step
        for n, p in self._named:
            self._theta_before[n] = p.detach().clone()
            self._g[n] = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)

    def on_batch_end(self, model, x, y, task_id):
        for n, p in self._named:
            d_theta = p.detach() - self._theta_before[n]
            self.w[n] += -self._g[n] * d_theta
        if self.online:
            self._step += 1
            if self._step % self.interval == 0:  # fixed-interval online consolidation
                self._finalize()
                self._finalized = True

    def _finalize(self, gate: float = 1.0):
        # gate in [0,1] scales how much new protection is committed: ~0 in an adverse
        # (high-error) world => synapses stay plastic ("PNNs stay immature").
        for n, p in self._named:
            d_tot = p.detach() - self.theta_task_start[n]
            self.omega[n] += gate * (self.w[n] / (d_tot.pow(2) + self.xi)).clamp_min(0.0)
            self.w[n].zero_()
            self.star[n] = p.detach().clone()
            self.theta_task_start[n] = p.detach().clone()

    def _error_gate(self, model, train_loader, task_id) -> float:
        """g = sigmoid(gamma*(thr - mean train error)); ~1 favorable, ~0 adverse."""
        device = next(model.parameters()).device
        model.eval()
        ls, n = 0.0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                ls += F.cross_entropy(model(x, task_id), y, reduction="sum").item()
                n += y.numel()
                if i >= 9:
                    break
        model.train()
        e = ls / max(n, 1)
        g = 1.0 / (1.0 + math.exp(-self.err_gamma * (self.err_threshold - e)))
        self.gate_history.append({"task": float(task_id), "error": e, "gate": g})
        return g

    def on_task_end(self, task_id, model, train_loader):
        if self.online:
            return  # online mode consolidates on a step interval, not at boundaries
        gate = self._error_gate(model, train_loader, task_id) if self.error_gated else 1.0
        self._finalize(gate)

    def consolidation_state(self, model) -> Dict[str, Any]:
        if not self.omega:
            return {}
        flat = torch.cat([v.reshape(-1) for v in self.omega.values()])
        last = self.gate_history[-1] if self.gate_history else {}
        return {"mean_omega": float(flat.mean()), "gate": last.get("gate"),
                "train_error": last.get("error")}
