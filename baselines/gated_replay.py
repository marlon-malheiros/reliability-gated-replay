"""Reliability-gated Experience Replay.

Buffer ADMISSION is gated by a reliability signal: when the gate is low the
example is withheld from long-term memory. The signal is pluggable
(``methods/gates.py``):

* ``signal: error``          -- supervised baseline (uses labels). This is the
                                original Error-Gated Replay.
* ``signal: confidence``     -- label-free predictive confidence.
* ``signal: pred_stability`` -- label-free prediction-stability (EMA / JS).
* ``signal: repr_stability`` -- label-free representation-stability.

Admission is per-example (``level: sample``, default): example ``i`` is admitted
with probability ``g_i``. ``level: batch`` admits the whole batch with the mean
gate (matching the legacy batch-level error gate). ``error_gated: false`` (or
``signal: none``) recovers standard Experience Replay -- a clean paired control.

The point of the pivot is the label-noise regime: a supervised-error gate sees a
*confidently-wrong* mislabeled example as low-error (high gate) and admits the
corrupted label; an internal-reliability gate can behave differently. We log, per
task, the buffer purity, the gate value, and the correlation between the gate and
true label correctness so the central comparison is directly measurable.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from methods.base import ContinualMethod
from methods.gates import ReliabilityGate
from methods.replay_buffer import ReplayBuffer, replay_ce_loss


class GatedReplay(ContinualMethod):
    name = "gated_replay"

    def __init__(self, cfg: Dict[str, Any] | None = None):
        super().__init__(cfg)
        seed = int(self.cfg.get("seed", 0))
        self.buf = ReplayBuffer(int(self.cfg.get("buffer_size", 500)), seed=seed)
        self.replay_n = int(self.cfg.get("replay_n", self.cfg.get("batch_size", 32)))
        # gating on/off. ``error_gated: false`` -> plain replay (admit everything).
        gate_cfg = dict(self.cfg.get("gate", {}))
        # back-compat: the legacy error gate was configured at top level.
        gate_cfg.setdefault("signal", self.cfg.get("gate_signal", "error"))
        gate_cfg.setdefault("error_threshold", self.cfg.get("error_threshold", 1.0))
        gate_cfg.setdefault("gamma", self.cfg.get("error_gamma", gate_cfg.get("gamma", 5.0)))
        self.gated = bool(self.cfg.get("error_gated", True)) and gate_cfg["signal"] != "none"
        self.level = str(self.cfg.get("gate_level", gate_cfg.get("level", "sample")))
        self.gate = ReliabilityGate(gate_cfg) if self.gated else None
        # Reviewer controls. ``fixed_admission_prob`` is a random gate whose
        # admission rate is supplied from a paired realizable-gate run.
        fixed = self.cfg.get("fixed_admission_prob")
        self.fixed_admission_prob = None if fixed is None else float(fixed)
        if self.fixed_admission_prob is not None and not 0.0 <= self.fixed_admission_prob <= 1.0:
            raise ValueError("fixed_admission_prob must be in [0, 1]")
        # Oracle gate (diagnostic, NOT a real method): admit an example only if its
        # stored label is truly correct (uses the ground-truth noise mask) -> 100%
        # buffer purity. Isolates whether a *perfect* buffer helps accuracy at all
        # (i.e. whether the buffer, vs the model, is the accuracy bottleneck).
        self.oracle = bool(self.cfg.get("oracle", False))
        self.oracle_keep_prob = float(self.cfg.get("oracle_keep_prob", 1.0))
        if not 0.0 <= self.oracle_keep_prob <= 1.0:
            raise ValueError("oracle_keep_prob must be in [0, 1]")
        # Optional slow weight-EMA teacher used to *score* the gate. A teacher that
        # changes slowly resists memorizing individual/majority noisy labels, so its
        # agreement with the stored label stays correlated with true correctness
        # even after the student starts memorizing -- the fix for the small-loss
        # inversion seen on few-class high-noise tasks.
        tcfg = dict(self.cfg.get("teacher", {}))
        self.teacher_enabled = bool(tcfg.get("enabled", False))
        self.teacher_m = float(tcfg.get("momentum", 0.99))
        self.teacher = None
        # Adaptive blend: g = (1-d)*g_student + d*g_teacher, where d = EMA of the
        # teacher<->student argmax disagreement -- an observable (label-free) proxy
        # for "the student is memorizing noise" (it diverges from the slow teacher).
        # benign noise: d low -> trust the powerful fast student; inversion regime:
        # d high -> fall back to the memorization-resistant slow teacher.
        acfg = dict(self.cfg.get("adaptive", {}))
        self.adaptive = bool(acfg.get("enabled", False))
        self.adaptive_ema = float(acfg.get("ema", 0.95))
        self._disagree = None
        # Route 1 -- early-snapshot scorer: freeze a copy of the model EARLY in each
        # task (before it memorizes the corrupted majority) and score admission with
        # it. Mechanism-correct fix (clean examples are learned first).
        escfg = dict(self.cfg.get("early_snapshot", {}))
        self.early_enabled = bool(escfg.get("enabled", False))
        self.early_after = int(escfg.get("after_steps", 20))
        self.early_scorer = None
        self._task_step = 0
        # Route 2 -- co-teaching-lite: an independently re-initialized peer net trained
        # ONLY on the main model's low-loss (clean-looking) subset; admission is gated
        # by the peer's agreement (the peer sees a cleaner stream, so its agreement
        # resists the single-model self-confirmation loop).
        ccfg = dict(self.cfg.get("coteach", {}))
        self.coteach_enabled = bool(ccfg.get("enabled", False))
        self.coteach_keep = float(ccfg.get("keep", 0.5))
        self.coteach_lr = float(ccfg.get("lr", 1e-3))
        self.peer = None
        self.peer_opt = None
        self._device = None
        self._rng = torch.Generator().manual_seed(seed + 7)
        self.gate_history: List[Dict[str, float]] = []
        # per-task running tallies for gate-vs-correctness correlation
        self._task_g: List[float] = []
        self._task_correct: List[int] = []
        self._task_candidates = 0
        self._task_admitted = 0
        self._cur_task = 0

    def on_task_start(self, task_id, model):
        self._device = next(model.parameters()).device
        self._cur_task = task_id
        if self.gate is not None:
            self.gate.reset_task(task_id)
        if self.teacher_enabled and self.teacher is None:
            self.teacher = copy.deepcopy(model)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        self._task_step = 0
        if self.early_enabled:
            self.early_scorer = None  # re-snapshot early within each task
        if self.coteach_enabled and self.peer is None:
            self.peer = copy.deepcopy(model)
            for mod in self.peer.modules():
                if hasattr(mod, "reset_parameters"):
                    mod.reset_parameters()  # independent init -> decorrelated from main
            self.peer.train()
            self.peer_opt = torch.optim.Adam(
                [p for p in self.peer.parameters() if p.requires_grad], lr=self.coteach_lr
            )

    def _train_peer(self, model, x, y, task_id):
        """Train the peer net on the main model's lowest-loss (clean-looking) subset."""
        with torch.no_grad():
            was = model.training
            model.eval()
            ce_main = F.cross_entropy(model(x, task_id), y, reduction="none")
            if was:
                model.train()
        k = max(1, int(self.coteach_keep * x.shape[0]))
        idx = torch.topk(ce_main, k, largest=False).indices
        self.peer.train()
        self.peer_opt.zero_grad()
        F.cross_entropy(self.peer(x[idx], task_id), y[idx]).backward()
        self.peer_opt.step()

    @torch.no_grad()
    def _update_teacher(self, model):
        if not (self.teacher_enabled and self.teacher is not None):
            return
        m = self.teacher_m
        for tp, sp in zip(self.teacher.parameters(), model.parameters()):
            tp.mul_(m).add_(sp.detach(), alpha=1.0 - m)
        for tb, sb in zip(self.teacher.buffers(), model.buffers()):
            tb.copy_(sb)  # BN running stats etc. (no-op for the MLP)

    @torch.no_grad()
    def _argmax_disagreement(self, model, x, task_id) -> float:
        """Fraction of the batch where student and teacher predict different classes
        (label-free signal that the student is diverging -> memorizing noise)."""
        was = model.training
        model.eval()
        ps = model(x, task_id).argmax(1)
        pt = self.teacher(x, task_id).argmax(1)
        if was:
            model.train()
        return float((ps != pt).float().mean())

    def extra_loss(self, model, x, y, task_id):
        if len(self.buf) == 0:
            return x.new_zeros(())
        return replay_ce_loss(model, self.buf.sample(self.replay_n), self._device)

    def on_batch_end(self, model, x, y, task_id):
        y_clean = self._batch_y_clean if self._batch_y_clean is not None else y
        correct = (y.detach().cpu() == y_clean.detach().cpu()).long()
        self._task_candidates += int(x.shape[0])
        if self.oracle:  # clean-only oracle, optionally thinned to a matched rate
            clean = (y == y_clean).detach().cpu()
            keep = torch.rand(x.shape[0], generator=self._rng) < self.oracle_keep_prob
            mask = clean & keep
            g = clean.float() * self.oracle_keep_prob
            self._task_g.extend(g.detach().cpu().tolist())
            self._task_correct.extend(correct.tolist())
            if bool(mask.any()):
                md = mask.to(x.device)
                self.buf.add(x[md], y[md], task_id, y_clean=y_clean[md])
                self._task_admitted += int(mask.sum())
            return
        if self.fixed_admission_prob is not None:
            p = self.fixed_admission_prob
            g = torch.full((x.shape[0],), p)
            self._task_g.extend(g.tolist())
            self._task_correct.extend(correct.tolist())
            mask = torch.rand(x.shape[0], generator=self._rng) < p
            if bool(mask.any()):
                md = mask.to(x.device)
                self.buf.add(x[md], y[md], task_id, y_clean=y_clean[md])
                self._task_admitted += int(mask.sum())
            return
        if not self.gated:
            self.buf.add(x, y, task_id, y_clean=y_clean)
            self._task_admitted += int(x.shape[0])
            return
        # advance per-task step; take the early snapshot once (Route 1)
        self._task_step += 1
        if self.early_enabled and self.early_scorer is None and self._task_step >= self.early_after:
            self.early_scorer = copy.deepcopy(model)
            self.early_scorer.eval()
            for p in self.early_scorer.parameters():
                p.requires_grad_(False)
        # co-teaching-lite: train the peer on the main model's low-loss subset (Route 2)
        if self.coteach_enabled and self.peer is not None:
            self._train_peer(model, x, y, task_id)

        has_teacher = self.teacher_enabled and self.teacher is not None
        if self.adaptive and has_teacher:
            g_t, diag = self.gate.score(self.teacher, x, y, task_id, uids=self._batch_uids)
            g_s, _ = self.gate.score(model, x, y, task_id, uids=self._batch_uids)
            d_batch = self._argmax_disagreement(model, x, task_id)
            self._disagree = (
                d_batch if self._disagree is None
                else self.adaptive_ema * self._disagree + (1 - self.adaptive_ema) * d_batch
            )
            d = self._disagree
            g = (1.0 - d) * g_s + d * g_t
        else:
            # scorer priority: co-teach peer > early snapshot > EMA teacher > live model
            if self.coteach_enabled and self.peer is not None:
                scorer = self.peer
            elif self.early_enabled and self.early_scorer is not None:
                scorer = self.early_scorer
            elif has_teacher:
                scorer = self.teacher
            else:
                scorer = model
            g, diag = self.gate.score(scorer, x, y, task_id, uids=self._batch_uids)
        # record gate vs. true-label correctness for this batch (correctness =
        # stored training label equals the true label).
        self._task_g.extend(g.cpu().tolist())
        self._task_correct.extend(correct.tolist())

        if self.level == "batch":
            gmean = float(g.mean())
            mask = torch.rand(x.shape[0], generator=self._rng) < gmean
        else:  # per-sample admission with probability g_i
            mask = torch.rand(x.shape[0], generator=self._rng) < g.cpu()
        if bool(mask.any()):
            self.buf.add(x[mask], y[mask], task_id, y_clean=y_clean[mask])
            self._task_admitted += int(mask.sum())
        self._update_teacher(model)  # EMA teacher tracks the (post-step) student

    def on_task_end(self, task_id, model, train_loader):
        # snapshot per-task gate diagnostics (consumed by analysis)
        g_arr = np.asarray(self._task_g, dtype=float)
        c_arr = np.asarray(self._task_correct, dtype=float)
        corr = (
            float(np.corrcoef(g_arr, c_arr)[0, 1])
            if g_arr.size > 1 and g_arr.std() > 1e-8 and c_arr.std() > 1e-8
            else 0.0
        )
        self.gate_history.append(
            {
                "task": float(task_id),
                "signal": (self.gate.signal if self.gate else "none"),
                "gate_mean": float(g_arr.mean()) if g_arr.size else 1.0,
                "gate_on_correct": float(g_arr[c_arr == 1].mean()) if (c_arr == 1).any() else float("nan"),
                "gate_on_wrong": float(g_arr[c_arr == 0].mean()) if (c_arr == 0).any() else float("nan"),
                "corr_gate_correct": corr,
                "buffer_purity": self.buf.purity(),
                "buffer_size": len(self.buf),
                "admission_rate": (
                    self._task_admitted / self._task_candidates
                    if self._task_candidates else float("nan")
                ),
            }
        )
        self._task_g.clear()
        self._task_correct.clear()
        self._task_candidates = 0
        self._task_admitted = 0

    def consolidation_state(self, model) -> Dict[str, Any]:
        last = self.gate_history[-1] if self.gate_history else {}
        return {
            "buffer_size": len(self.buf),
            "buffer_purity": self.buf.purity(),
            "gate_signal": (self.gate.signal if self.gate else "none"),
            "fixed_admission_prob": self.fixed_admission_prob,
            "oracle_keep_prob": self.oracle_keep_prob if self.oracle else None,
            "teacher_enabled": self.teacher_enabled,
            "teacher_momentum": self.teacher_m if self.teacher_enabled else None,
            "adaptive": self.adaptive,
            "disagree_ema": self._disagree,
            "early_snapshot": self.early_enabled,
            "coteach": self.coteach_enabled,
            "gate_mean": last.get("gate_mean"),
            "gate_on_correct": last.get("gate_on_correct"),
            "gate_on_wrong": last.get("gate_on_wrong"),
            "corr_gate_correct": last.get("corr_gate_correct"),
            "admission_rate": last.get("admission_rate"),
            "gate_history": list(self.gate_history),
            "buffer_task_counts": {t: self.buf.t.count(t) for t in sorted(set(self.buf.t))},
        }
