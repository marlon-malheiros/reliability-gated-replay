"""Internal-model reliability gates -- the pivot's contribution.

The Error-Gated framework consolidated when *supervised error* was low. The
hypothesis here is that PNN-like maturation should instead track when the
network's own *internal model* becomes reliable -- confident, temporally stable
in its predictions, or stable in its representations -- signals that do **not**
require trustworthy labels.

A :class:`ReliabilityGate` maps a batch ``(x, y, task_id)`` to a per-example gate
``g in [0,1]`` used to scale consolidation (Fisher) or buffer admission. Four
signals share one sigmoid form ``g = sigmoid(gamma * (signal - tau))``:

* ``error``           -- supervised baseline. ``signal = (tau_err - CE)`` so low
                         per-example cross-entropy -> high g. **Uses labels.**
* ``confidence``      -- ``C = 1 - H(p)/log K`` (normalized 1 - predictive
                         entropy). Label-free.
* ``pred_stability``  -- per-example EMA of the softmax ``p_ema <- b p_ema +
                         (1-b) p``; ``S = 1 - JS(p || p_ema)/log 2``. Label-free.
* ``repr_stability``  -- per-example EMA of the penultimate feature ``h_ema``;
                         ``D = mean_j (h - h_ema)_j^2``, ``R = exp(-alpha D)``,
                         ``signal = R``. Label-free.

The stability gates need to see the *same example more than once* (across
epochs), so they are keyed by a stable per-(task, example) uid. On an example's
first sighting they return ``cold_value`` (default 0.5, neutral) and seed the EMA.

All signals are computed in ``eval`` mode under ``no_grad`` so the gate never
perturbs BatchNorm running stats or the optimizer path.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

_SIGNALS = ("error", "agreement", "loss_traj", "confidence", "pred_stability", "repr_stability")


def _sigmoid(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z.clamp(-60.0, 60.0))


class ReliabilityGate:
    """Per-example reliability signal -> gate in [0,1]. See module docstring."""

    def __init__(self, cfg: Dict[str, Any] | None = None):
        cfg = cfg or {}
        self.signal = str(cfg.get("signal", "error"))
        if self.signal not in _SIGNALS:
            raise ValueError(f"Unknown gate signal '{self.signal}'. Known: {_SIGNALS}")
        self.gamma = float(cfg.get("gamma", 5.0))
        self.tau = float(cfg.get("tau", 0.5))
        # error gate: threshold on the cross-entropy (signal = tau_err - CE).
        self.tau_err = float(cfg.get("tau_err", cfg.get("error_threshold", 1.0)))
        # stability gates
        self.beta = float(cfg.get("ema_beta", 0.7))         # prediction/feature EMA
        self.alpha = float(cfg.get("repr_alpha", 1.0))      # repr distance -> stability
        self.repr_layer = str(cfg.get("repr_layer", "penultimate"))
        self.cold_value = float(cfg.get("cold_value", 0.5))
        # per-(task,uid) EMA state for the temporal-stability gates
        self._p_ema: Dict[int, torch.Tensor] = {}
        self._h_ema: Dict[int, torch.Tensor] = {}
        # per-(task,uid) running (count, sum) of CE loss for the loss-trajectory gate
        self._loss_stats: Dict[int, list] = {}
        self._task = -1

    @property
    def uses_labels(self) -> bool:
        return self.signal in ("error", "agreement", "loss_traj")

    def reset_task(self, task_id: int) -> None:
        """Drop per-example EMA state at a task boundary (examples change)."""
        self._task = task_id
        self._p_ema.clear()
        self._h_ema.clear()
        self._loss_stats.clear()

    # --- core ---
    @torch.no_grad()
    def score(
        self,
        model,
        x: torch.Tensor,
        y: torch.Tensor | None,
        task_id: int,
        uids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return ``(g[B], diag)``. ``uids`` (long[B]) are required for the
        temporal-stability gates so the same example maps to the same EMA slot."""
        was_training = model.training
        model.eval()
        feats = model.features(x)
        logits = model.head(task_id)(feats)
        p = F.softmax(logits, dim=1)
        K = p.shape[1]

        if self.signal == "error":
            assert y is not None, "error gate needs labels"
            ce = F.cross_entropy(logits, y, reduction="none")
            sig = self.tau_err - ce
            diag_sig = ce
        elif self.signal == "agreement":
            # supervised, but bounded: gate on the scoring model's probability mass
            # on the *stored* label, g = sigmoid(gamma*(p(y|x) - tau)). When scored
            # with a slow EMA teacher (see GatedReplay) this is memorization-robust.
            assert y is not None, "agreement gate needs labels"
            p_label = p.gather(1, y.view(-1, 1)).squeeze(1)
            sig = p_label - self.tau
            diag_sig = p_label
        elif self.signal == "loss_traj":
            # loss-trajectory (DivideMix-style small-loss over time): gate on each
            # example's *running-mean* CE across all its views, not the instantaneous
            # loss. Noisy examples accumulate higher loss early (clean is learned
            # first), so the time-averaged loss separates clean/noisy better than a
            # single snapshot. signal = tau_err - mean_loss.
            assert y is not None, "loss_traj gate needs labels"
            ce = F.cross_entropy(logits, y, reduction="none").detach().cpu()
            means = torch.empty(ce.shape[0])
            if uids is None:
                means = ce
            else:
                for i in range(ce.shape[0]):
                    k = int(uids[i])
                    st = self._loss_stats.get(k)
                    if st is None:
                        st = [0, 0.0]
                        self._loss_stats[k] = st
                    st[0] += 1
                    st[1] += float(ce[i])
                    means[i] = st[1] / st[0]
            means = means.to(logits.device)
            sig = self.tau_err - means
            diag_sig = means
        elif self.signal == "confidence":
            ent = -(p.clamp_min(1e-12).log() * p).sum(dim=1)
            conf = 1.0 - ent / math.log(max(K, 2))
            sig = conf - self.tau
            diag_sig = conf
        elif self.signal == "pred_stability":
            S, warm = self._pred_stability(p, uids)
            sig = torch.where(warm, S - self.tau, torch.zeros_like(S))
            diag_sig = S
        else:  # repr_stability
            R, warm = self._repr_stability(feats, uids)
            sig = torch.where(warm, R - self.tau, torch.zeros_like(R))
            diag_sig = R

        g = _sigmoid(self.gamma * sig)
        if self.signal in ("pred_stability", "repr_stability"):
            # cold (first-sighting) examples get the neutral cold_value
            cold = ~warm
            if cold.any():
                g = torch.where(cold, torch.full_like(g, self.cold_value), g)

        if was_training:
            model.train()
        diag = {
            "signal": self.signal,
            "g_mean": float(g.mean()),
            "raw_mean": float(diag_sig.float().mean()),
        }
        return g.detach(), diag

    # --- temporal-stability internals ---
    def _key(self, task_id_unused: int, uid: int) -> int:
        return int(uid)

    def _pred_stability(
        self, p: torch.Tensor, uids: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = p.shape[0]
        S = torch.ones(B, device=p.device)
        warm = torch.zeros(B, dtype=torch.bool, device=p.device)
        if uids is None:
            return S, warm  # cannot key examples -> all cold
        pc = p.detach().cpu()
        for i in range(B):
            k = int(uids[i])
            cur = pc[i]
            prev = self._p_ema.get(k)
            if prev is None:
                self._p_ema[k] = cur.clone()
                continue  # cold
            js = _js_divergence(cur, prev)
            S[i] = 1.0 - js / math.log(2.0)
            warm[i] = True
            self._p_ema[k] = self.beta * prev + (1.0 - self.beta) * cur
        return S.clamp(0.0, 1.0), warm

    def _repr_stability(
        self, feats: torch.Tensor, uids: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = feats.shape[0]
        R = torch.ones(B, device=feats.device)
        warm = torch.zeros(B, dtype=torch.bool, device=feats.device)
        if uids is None:
            return R, warm
        hc = feats.detach().cpu()
        for i in range(B):
            k = int(uids[i])
            cur = hc[i]
            prev = self._h_ema.get(k)
            if prev is None:
                self._h_ema[k] = cur.clone()
                continue
            d = float((cur - prev).pow(2).mean())  # mean sq diff per feature
            R[i] = math.exp(-self.alpha * d)
            warm[i] = True
            self._h_ema[k] = self.beta * prev + (1.0 - self.beta) * cur
        return R.clamp(0.0, 1.0), warm


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence (nats) between two prob vectors."""
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    return float((p * (p / q).log()).sum())
