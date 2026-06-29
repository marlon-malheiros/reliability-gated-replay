"""Elastic Weight Consolidation (Kirkpatrick et al., 2017) + online variant.

Classic (``online: false``): after each task we estimate the diagonal (empirical)
Fisher and store a per-task anchor; the penalty pulls parameters toward each
consolidated value weighted by Fisher.

Online (``online: true``, Schwarz et al. 2018 style; the task-free baseline): a
single running Fisher (EMA of the squared CE gradient) and a single ``theta*``
re-anchored every ``interval_steps`` -- no task boundaries required.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from methods.base import ContinualMethod
from methods.gates import ReliabilityGate
from methods.utils import head_param_ids, protected_named_parameters


class EWC(ContinualMethod):
    name = "ewc"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        self.lam = float(self.cfg.get("lambda", 100.0))
        self.fisher_samples = int(self.cfg.get("fisher_samples", 512))
        self.online = bool(self.cfg.get("online", False))
        self.gamma = float(self.cfg.get("gamma", 0.9))            # running-Fisher EMA
        self.interval = max(int(self.cfg.get("interval_steps", 200)), 1)
        self.fisher_period = max(int(self.cfg.get("fisher_period", 1)), 1)
        # Which parameters consolidation acts on:
        #   backbone (default) -> deep shared weights, excludes heads (the standard choice);
        #   head -> only the classifier head (shallow, label-sensitive);
        #   all -> backbone + head.
        self.scope = str(self.cfg.get("scope", "backbone"))
        # Gated stabilization: scale the per-task Fisher by a reliability gate in
        # [0,1] (``F_gated = g * F``). ``signal: error`` is the supervised
        # baseline (suppress when train error is high); ``confidence`` /
        # ``pred_stability`` / ``repr_stability`` are the label-free internal
        # reliability signals (see methods/gates.py). ``error_gated`` is the
        # on/off switch (back-compat: the legacy error gate lived here).
        self.error_gated = bool(self.cfg.get("error_gated", False))
        gate_cfg = dict(self.cfg.get("gate", {}))
        gate_cfg.setdefault("signal", self.cfg.get("gate_signal", "error"))
        gate_cfg.setdefault("error_threshold", self.cfg.get("error_threshold", 0.3))
        gate_cfg.setdefault("gamma", self.cfg.get("error_gamma", gate_cfg.get("gamma", 10.0)))
        self.gate = ReliabilityGate(gate_cfg)
        self.gate_batches = int(self.cfg.get("gate_batches", 10))
        # Depth-aware adaptation (deep nets): a per-block gate driven by each
        # block's gradient stability -- calm/converged blocks consolidate
        # (gate->1), volatile blocks withhold (gate->0); plus skip-connection
        # exclusion and per-layer Fisher normalization. All default OFF.
        self.layerwise = bool(self.cfg.get("layerwise_gate", False))
        self.exclude_skip = bool(self.cfg.get("exclude_skip", False))
        self.fisher_layer_norm = bool(self.cfg.get("fisher_layer_norm", False))
        self.block_prefixes = list(
            self.cfg.get("block_prefixes", ["conv1", "layer1", "layer2", "layer3", "layer4"])
        )
        self.lw_gamma = float(self.cfg.get("layerwise_gamma", 4.0))
        self._block_gradsq: Dict[str, float] = {}
        self.gate_history: List[Dict[str, float]] = []
        self._named: List[Tuple[str, torch.nn.Parameter]] = []
        self.anchors: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = []
        self.fisher_run: Dict[str, torch.Tensor] = {}
        self.star_online: Dict[str, torch.Tensor] = {}
        self._step = 0

    def _select_params(self, model):
        if self.scope == "head":
            hids = head_param_ids(model)
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and id(p) in hids]
        elif self.scope == "all":
            named = list(protected_named_parameters(model, include_heads=True))
        else:
            named = list(protected_named_parameters(model))
        if self.exclude_skip:
            named = [(n, p) for n, p in named if "shortcut" not in n and "downsample" not in n]
        return named

    def _block_of(self, name: str) -> str:
        for pre in self.block_prefixes:
            if name == pre or name.startswith(pre + "."):
                return pre
        return "_other"

    def _accumulate_block_grads(self) -> None:
        """EMA of each block's per-parameter mean squared gradient (stability)."""
        gsq: Dict[str, float] = {}
        cnt: Dict[str, int] = {}
        for n, p in self._named:
            if p.grad is None:
                continue
            b = self._block_of(n)
            gsq[b] = gsq.get(b, 0.0) + float(p.grad.detach().pow(2).sum())
            cnt[b] = cnt.get(b, 0) + p.numel()
        for b in gsq:
            per_param = gsq[b] / max(cnt[b], 1)
            prev = self._block_gradsq.get(b)
            self._block_gradsq[b] = per_param if prev is None else 0.9 * prev + 0.1 * per_param

    def _layerwise_gates(self) -> Dict[str, float]:
        """gate_b = sigmoid(gamma * (1 - E_b / median_E)); calm blocks (E_b below
        the median block gradient) consolidate, volatile blocks withhold. Scale-free."""
        vals = {b: v for b, v in self._block_gradsq.items() if b != "_other"}
        if not vals:
            return {}
        ordered = sorted(vals.values())
        med = ordered[len(ordered) // 2] or 1e-12

        def gate(e: float) -> float:
            arg = self.lw_gamma * (1.0 - e / med)
            arg = max(-60.0, min(60.0, arg))  # numerically stable sigmoid
            return 1.0 / (1.0 + math.exp(-arg))

        return {b: gate(e) for b, e in vals.items()}

    def _normalize_fisher_per_layer(self, fisher: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Equalize each block's mean Fisher (so wide deep layers don't dominate)
        while PRESERVING the global Fisher scale (so lambda stays in the same range)."""
        bsum: Dict[str, float] = {}
        bcnt: Dict[str, int] = {}
        for n, v in fisher.items():
            b = self._block_of(n)
            bsum[b] = bsum.get(b, 0.0) + float(v.sum())
            bcnt[b] = bcnt.get(b, 0) + v.numel()
        bmean = {b: (bsum[b] / max(bcnt[b], 1)) or 1e-12 for b in bsum}
        global_mean = (sum(bsum.values()) / max(sum(bcnt.values()), 1)) or 1e-12
        return {n: v * (global_mean / (bmean[self._block_of(n)] + 1e-12)) for n, v in fisher.items()}

    def on_task_start(self, task_id, model):
        if not self._named:
            self._named = self._select_params(model)
            if self.online:
                self.fisher_run = {n: torch.zeros_like(p) for n, p in self._named}
                self.star_online = {n: p.detach().clone() for n, p in self._named}

    def extra_loss(self, model, x, y, task_id):
        if self.online:
            if not self.fisher_run:
                return x.new_zeros(())
            loss = x.new_zeros(())
            for n, p in self._named:
                loss = loss + (self.fisher_run[n] * (p - self.star_online[n]).pow(2)).sum()
            return 0.5 * self.lam * loss
        if not self.anchors:
            return x.new_zeros(())
        loss = x.new_zeros(())
        for star, fisher in self.anchors:
            for n, p in self._named:
                loss = loss + (fisher[n] * (p - star[n]).pow(2)).sum()
        return 0.5 * self.lam * loss

    def on_batch_end(self, model, x, y, task_id):
        if self.layerwise and self._named:
            self._accumulate_block_grads()
        if not self.online or not self._named:
            return
        if self._step % self.fisher_period == 0:
            params = [p for _, p in self._named]
            ce = F.cross_entropy(model(x, task_id), y)
            grads = torch.autograd.grad(ce, params, retain_graph=False, allow_unused=True)
            for (n, _), g in zip(self._named, grads):
                if g is None:
                    continue
                self.fisher_run[n].mul_(self.gamma).add_(g.detach() ** 2, alpha=1.0 - self.gamma)
        self._step += 1
        if self._step % self.interval == 0:  # fixed-interval online re-anchor
            self.star_online = {n: p.detach().clone() for n, p in self._named}

    def on_task_end(self, task_id, model, train_loader):
        if self.online:
            return  # online mode re-anchors on a step interval, not at boundaries
        fisher = self._compute_fisher(model, train_loader, task_id)
        if self.fisher_layer_norm:
            fisher = self._normalize_fisher_per_layer(fisher)
        if self.layerwise:
            gates = self._layerwise_gates()
            fisher = {n: gates.get(self._block_of(n), 1.0) * v for n, v in fisher.items()}
            self.gate_history.append(
                {"task": float(task_id), **{f"gate[{b}]": g for b, g in gates.items()}}
            )
            self._block_gradsq = {}
        elif self.error_gated:
            gate = self._reliability_gate(model, train_loader, task_id)
            if gate != 1.0:
                fisher = {n: gate * v for n, v in fisher.items()}
        star = {n: p.detach().clone() for n, p in self._named}
        self.anchors.append((star, fisher))

    def _reliability_gate(self, model, train_loader, task_id) -> float:
        """Scalar Fisher gate over a sample of the train set, using the configured
        reliability signal. error/confidence: one pass; the temporal-stability
        gates seed their per-example EMA on a first (deterministic, non-shuffled)
        pass then measure on the second."""
        device = next(model.parameters()).device
        signal = self.gate.signal

        def run_pass(measure: bool):
            gs, pos = [], 0
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                uids = torch.arange(pos, pos + x.shape[0], device=device)
                pos += x.shape[0]
                g, _ = self.gate.score(model, x, y, task_id, uids=uids)
                if measure:
                    gs.append(g)
                if i >= self.gate_batches - 1:
                    break
            return torch.cat(gs) if gs else torch.ones(1)

        if signal in ("pred_stability", "repr_stability"):
            self.gate.reset_task(task_id)
            run_pass(measure=False)  # seed EMA
            g = run_pass(measure=True)
        else:
            g = run_pass(measure=True)
        gate = float(g.mean())
        self.gate_history.append({"task": float(task_id), "signal": signal, "gate": gate})
        return gate

    def _compute_fisher(self, model, train_loader, task_id) -> Dict[str, torch.Tensor]:
        device = next(model.parameters()).device
        fisher = {n: torch.zeros_like(p) for n, p in self._named}
        model.eval()
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            for i in range(x.shape[0]):
                model.zero_grad()
                out = model(x[i : i + 1], task_id)
                logp = F.log_softmax(out, dim=1)
                logp[0, y[i]].backward()
                for n, p in self._named:
                    if p.grad is not None:
                        fisher[n] += p.grad.detach() ** 2
                seen += 1
                if seen >= self.fisher_samples:
                    break
            if seen >= self.fisher_samples:
                break
        for n in fisher:
            fisher[n] /= max(seen, 1)
        model.zero_grad()
        return fisher

    def consolidation_state(self, model) -> Dict[str, Any]:
        if self.online:
            if not self.fisher_run:
                return {}
            flat = torch.cat([v.reshape(-1) for v in self.fisher_run.values()])
            return {"mean_fisher": float(flat.mean())}
        if not self.anchors:
            return {}
        flat = torch.cat([v.reshape(-1) for _, fisher in self.anchors for v in fisher.values()])
        last = self.gate_history[-1] if self.gate_history else {}
        return {
            "mean_fisher": float(flat.mean()),
            "gate": last.get("gate"),
            "gate_signal": last.get("signal", "error" if self.error_gated else "none"),
            "gate_history": list(self.gate_history),
        }
