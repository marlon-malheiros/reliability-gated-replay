"""Memory Aware Synapses (Aljundi et al., 2018) -- task-free by construction.

Importance ``omega`` is the (unsupervised) sensitivity of the squared L2 norm of
the network output to each parameter: ``omega_i = E_x | d(||f(x)||^2)/d theta_i |``.
Because it needs no labels and no task boundaries, MAS is a natural task-free
baseline: we accumulate ``omega`` online as an EMA and re-anchor ``theta*`` to the
current weights every ``interval_steps`` (a fixed-interval online consolidation).
The penalty pulls each parameter toward ``theta*`` weighted by ``omega``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from methods.base import ContinualMethod
from methods.utils import protected_named_parameters


class MAS(ContinualMethod):
    name = "mas"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        self.lam = float(self.cfg.get("lambda", 1.0))
        self.decay = float(self.cfg.get("ema_decay", 0.9))
        self.interval = max(int(self.cfg.get("interval_steps", 200)), 1)
        self.omega_period = max(int(self.cfg.get("omega_period", 1)), 1)
        # Error-gated stabilization: withhold the importance accumulated during
        # an adverse/high-error task (mirrors the EWC gate). MAS's omega is
        # label-INDEPENDENT (output sensitivity), so unlike EWC's Fisher it does
        # not inflate under label noise; the gate is therefore expected to be a
        # functional null here -- it cannot suppress what never over-committed.
        self.error_gated = bool(self.cfg.get("error_gated", False))
        self.err_threshold = float(self.cfg.get("error_threshold", 0.3))
        self.err_gamma = float(self.cfg.get("error_gamma", 10.0))
        self.gate_history: List[Dict[str, float]] = []
        self._named: List[Tuple[str, torch.nn.Parameter]] = []
        self.omega: Dict[str, torch.Tensor] = {}
        self.star: Dict[str, torch.Tensor] = {}
        self._omega_at_task_start: Dict[str, torch.Tensor] | None = None
        self._step = 0

    def on_task_start(self, task_id, model):
        if not self._named:
            self._named = list(protected_named_parameters(model))
            self.omega = {n: torch.zeros_like(p) for n, p in self._named}
            self.star = {n: p.detach().clone() for n, p in self._named}
        if self.error_gated:
            self._omega_at_task_start = {n: v.detach().clone() for n, v in self.omega.items()}

    def extra_loss(self, model, x, y, task_id):
        if not self.omega:
            return x.new_zeros(())
        loss = x.new_zeros(())
        for n, p in self._named:
            loss = loss + (self.omega[n] * (p - self.star[n]).pow(2)).sum()
        return 0.5 * self.lam * loss

    def _accumulate_omega(self, model, x, task_id):
        params = [p for _, p in self._named]
        out = model(x, task_id)
        l2 = out.pow(2).sum(dim=1).mean()
        grads = torch.autograd.grad(l2, params, retain_graph=False, allow_unused=True)
        for (n, _), g in zip(self._named, grads):
            if g is None:
                continue
            self.omega[n].mul_(self.decay).add_(g.detach().abs(), alpha=1.0 - self.decay)

    def on_batch_end(self, model, x, y, task_id):
        if not self._named:
            return
        if self._step % self.omega_period == 0:
            self._accumulate_omega(model, x, task_id)
        self._step += 1
        if self._step % self.interval == 0:  # fixed-interval online re-anchor
            self.star = {n: p.detach().clone() for n, p in self._named}

    # classic task-aware use: also re-anchor at a (known) task boundary
    def on_task_end(self, task_id, model, train_loader):
        if not self._named:
            return
        # Error-gated: scale back the importance GAINED during this task by the
        # gate, so an adverse (high-error) task leaves omega "immature".
        if self.error_gated and self._omega_at_task_start is not None:
            gate = self._error_gate(model, train_loader, task_id)
            for n in self.omega:
                base = self._omega_at_task_start[n]
                self.omega[n] = base + gate * (self.omega[n] - base)
        self.star = {n: p.detach().clone() for n, p in self._named}

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
        err = ls / max(n, 1)
        gate = 1.0 / (1.0 + math.exp(-self.err_gamma * (self.err_threshold - err)))
        self.gate_history.append({"task": float(task_id), "error": err, "gate": gate})
        return gate

    def consolidation_state(self, model) -> Dict[str, Any]:
        if not self.omega:
            return {}
        flat = torch.cat([v.reshape(-1) for v in self.omega.values()])
        last = self.gate_history[-1] if self.gate_history else {}
        return {
            "mean_omega": float(flat.mean()),
            "interval_steps": self.interval,
            "n_reanchors": self._step // self.interval,
            "gate": last.get("gate"),
            "train_error": last.get("error"),
        }
