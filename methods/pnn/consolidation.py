"""PNN consolidation state manager.

Owns the per-parameter consolidation variable ``P in [0,1]`` and applies:
  * maturation   ``P <- clip(P + alpha * importance * closure_signal, 0, 1)``
  * gating       scales gradients by ``exp(-beta P)`` or ``(1 - P)``
  * anchoring    optional quadratic penalty toward consolidated weights
  * reopening    ``P <- P * (1 - reopening_factor)``

Granularity (param-wise vs layer-wise, ablations A17/A18), the importance method
(A11-A14), uniform/random/zero initialisation and the on/off maturation switch
(A2/A3) are all driven from config, so the 18 ablations are pure config overrides.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..utils import protected_named_parameters
from .gating import gating_factor
from .importance import ImportanceEstimator


class PNNConsolidation:
    def __init__(self, model: torch.nn.Module, cfg: Dict[str, Any]):
        self.named: List[Tuple[str, torch.nn.Parameter]] = list(
            protected_named_parameters(
                model,
                include_bias=cfg.get("include_bias", False),
                include_heads=cfg.get("include_heads", False),
            )
        )
        self.alpha = float(cfg.get("alpha", 0.1))
        self.beta = float(cfg.get("beta", 5.0))
        self.gating_mode = cfg.get("gating", "grad_scale")
        self.granularity = cfg.get("granularity", "paramwise")
        self.maturation_enabled = bool(cfg.get("maturation", True))
        self.consolidated_thr = float(cfg.get("consolidated_threshold", 0.5))
        acfg = cfg.get("anchor", {})
        self.anchor_enabled = bool(acfg.get("enabled", False))
        self.anchor_base_lambda = float(acfg.get("lambda", 1.0))
        self.anchor_lambda = self.anchor_base_lambda
        self.anchor_normalize = bool(acfg.get("normalize", True))
        adcfg = acfg.get("adaptive", {})
        self.anchor_adaptive_enabled = bool(adcfg.get("enabled", False))
        self.anchor_adaptive_gain = float(adcfg.get("gain", 0.0))
        self.anchor_lambda_min = float(adcfg.get("min", self.anchor_base_lambda))
        self.anchor_lambda_max = float(adcfg.get("max", self.anchor_base_lambda))
        self.anchor_star: Dict[str, torch.Tensor] = {}
        self.anchor_updates = 0

        # discrete/hard consolidation: at a snapshot, irreversibly freeze (grad->0)
        # the elements that are consolidated (P >= threshold). Default off.
        ccfg = cfg.get("consolidation", {})
        self.freeze_enabled = bool(ccfg.get("freeze_on_snapshot", False))
        self.freeze_threshold = float(ccfg.get("freeze_threshold", 0.5))
        # if >0: freeze the top fraction of (still-plastic) weights by importance at each
        # snapshot (selective). Else: freeze all elements with P >= freeze_threshold.
        self.freeze_fraction = float(ccfg.get("freeze_fraction", 0.0))
        self.frozen: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, dtype=torch.bool) for n, p in self.named
        }

        icfg = cfg.get("importance", {})
        self.uniform_importance = bool(icfg.get("uniform", False))
        self.importance = ImportanceEstimator(
            self.named,
            method=icfg.get("method", "hybrid"),
            ema_decay=float(icfg.get("ema_decay", 0.9)),
            hybrid_weights=tuple(icfg.get("weights", (1 / 3, 1 / 3, 1 / 3))),
        )

        self.P: Dict[str, torch.Tensor] = {n: torch.zeros_like(p) for n, p in self.named}
        self._init_P(cfg)

    def _init_P(self, cfg: Dict[str, Any]) -> None:
        mode = cfg.get("P_init", "zeros")
        if mode == "random":
            for n, p in self.named:
                self.P[n] = torch.rand_like(p)
        elif mode == "uniform":
            val = float(cfg.get("P_init_value", 0.5))
            for n, p in self.named:
                self.P[n] = torch.full_like(p, val)

    # --- per-batch ---
    def update_importance(self) -> None:
        if self.maturation_enabled and not self.uniform_importance:
            self.importance.update(self.named)

    def apply_gating(self) -> None:
        if self.gating_mode == "none":
            return
        for n, p in self.named:
            if p.grad is not None:
                p.grad.mul_(gating_factor(self.P[n], self.gating_mode, self.beta))

    # --- discrete/hard consolidation (the "lock it in" variant) ---
    def freeze_consolidated(self) -> None:
        """At a snapshot event: irreversibly freeze consolidated elements. Either the
        top ``freeze_fraction`` of still-plastic weights by current importance
        (selective), or all elements with ``P >= freeze_threshold``."""
        if not self.freeze_enabled:
            return
        if self.freeze_fraction > 0.0:
            imp = self.importance.importance(self.named)
            for n, _ in self.named:
                fr = self.frozen[n]
                k = int(round(self.freeze_fraction * fr.numel()))
                if k <= 0:
                    continue
                scores = imp[n].reshape(-1).clone()
                scores[fr.reshape(-1)] = -1.0  # exclude already-frozen
                avail = int((scores >= 0).sum().item())
                if avail <= 0:
                    continue
                idx = torch.topk(scores, min(k, avail)).indices
                fr.reshape(-1)[idx] = True
        else:
            for n, _ in self.named:
                self.frozen[n] |= (self.P[n] >= self.freeze_threshold)

    def apply_freeze(self) -> None:
        """Zero the gradient of frozen elements every step (hard, irreversible)."""
        if not self.freeze_enabled:
            return
        for n, p in self.named:
            if p.grad is not None:
                p.grad[self.frozen[n]] = 0.0

    def frozen_fraction(self) -> float:
        if not self.freeze_enabled or not self.frozen:
            return 0.0
        tot = sum(v.numel() for v in self.frozen.values())
        fz = sum(int(v.sum()) for v in self.frozen.values())
        return fz / max(tot, 1)

    # --- per-epoch ---
    def mature(self, closure_signal: float) -> None:
        if not self.maturation_enabled or closure_signal <= 0.0:
            return
        imp = (
            {n: torch.ones_like(p) for n, p in self.named}
            if self.uniform_importance
            else self.importance.importance(self.named)
        )
        for n, p in self.named:
            im = imp[n]
            if self.granularity == "layerwise":
                im = torch.full_like(im, float(im.mean()))
            self.P[n] = (self.P[n] + self.alpha * im * closure_signal).clamp_(0.0, 1.0)

    def adapt_anchor_lambda(self, closure_signal: float) -> None:
        """Adjust anchor strength from the existing closure signal.

        This is intentionally low-bandwidth: the effective lambda changes at
        epoch boundaries and affects the next optimization epoch/task. It keeps
        adaptive consolidation separate from the per-batch optimizer path.
        """
        if not self.anchor_enabled or not self.anchor_adaptive_enabled:
            return
        target = self.anchor_base_lambda * (1.0 + self.anchor_adaptive_gain * closure_signal)
        self.anchor_lambda = float(np.clip(target, self.anchor_lambda_min, self.anchor_lambda_max))

    def reopen(self, factor: float) -> None:
        for n in self.P:
            self.P[n].mul_(1.0 - factor)

    # --- optional P-weighted parameter anchor ---
    def snapshot_anchor(self) -> None:
        """Store the current consolidated parameter values for future anchoring."""
        if not self.anchor_enabled:
            return
        self.anchor_star = {n: p.detach().clone() for n, p in self.named}
        self.anchor_updates += 1

    def anchor_loss(self) -> torch.Tensor | None:
        """Return ``lambda * P * (theta - theta_star)^2`` for protected params."""
        if not self.anchor_enabled or not self.anchor_star:
            return None
        loss = None
        denom = 0
        for n, p in self.named:
            term = (self.P[n].detach() * (p - self.anchor_star[n]).pow(2)).sum()
            loss = term if loss is None else loss + term
            denom += p.numel()
        if loss is None:
            return None
        if self.anchor_normalize:
            loss = loss / max(denom, 1)
        return 0.5 * self.anchor_lambda * loss

    # --- online diagnostics (cheap, analytic; used by the task-free stream log) ---
    def param_displacement(self) -> float:
        """``||theta - theta*||`` over protected params (0 if no snapshot yet)."""
        if not self.anchor_star:
            return 0.0
        sq = 0.0
        for n, p in self.named:
            if n in self.anchor_star:
                sq += float((p.detach() - self.anchor_star[n]).pow(2).sum().item())
        return sq**0.5

    def anchor_grad_norm(self) -> float:
        """Analytic norm of d(anchor_loss)/d(theta) = lambda * P * (theta - theta*)."""
        if not self.anchor_enabled or not self.anchor_star:
            return 0.0
        denom = sum(p.numel() for _, p in self.named) if self.anchor_normalize else 1
        scale = self.anchor_lambda / max(denom, 1)
        sq = 0.0
        for n, p in self.named:
            if n in self.anchor_star:
                g = scale * self.P[n] * (p.detach() - self.anchor_star[n])
                sq += float(g.pow(2).sum().item())
        return sq**0.5

    def layerwise_anchor_activity(self) -> Dict[str, float]:
        """Per-layer ``sum P * (theta - theta*)^2`` (anchor energy by layer)."""
        if not self.anchor_star:
            return {}
        out: Dict[str, float] = {}
        for n, p in self.named:
            if n in self.anchor_star:
                out[n] = float((self.P[n] * (p.detach() - self.anchor_star[n]).pow(2)).sum().item())
        return out

    def P_percentiles(self) -> Dict[str, float]:
        flat = self._flat_P()
        if not len(flat):
            return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
        q = np.percentile(flat, [10, 50, 90])
        return {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}

    # --- diagnostics ---
    def _flat_P(self) -> np.ndarray:
        return torch.cat([v.reshape(-1) for v in self.P.values()]).detach().cpu().numpy()

    def state(self) -> Dict[str, Any]:
        flat = self._flat_P()
        per_layer = {n: float(v.mean()) for n, v in self.P.items()}
        # gating (effective plasticity) factor averaged over params
        gfac = []
        for n, _ in self.named:
            gfac.append(float(gating_factor(self.P[n], self.gating_mode, self.beta).mean()))
        # correlation(P, importance) on a subsample
        corr = self._corr_P_importance()
        hist, edges = np.histogram(flat, bins=20, range=(0.0, 1.0))
        rng = np.random.RandomState(0)
        sample = flat[rng.permutation(len(flat))[:2000]] if len(flat) else flat
        return {
            "mean_P": float(flat.mean()) if len(flat) else 0.0,
            "var_P": float(flat.var()) if len(flat) else 0.0,
            "frac_consolidated": float((flat > self.consolidated_thr).mean()) if len(flat) else 0.0,
            "mean_gating_factor": float(np.mean(gfac)) if gfac else 1.0,
            "anchor_enabled": self.anchor_enabled,
            "anchor_lambda": self.anchor_lambda,
            "anchor_base_lambda": self.anchor_base_lambda,
            "anchor_adaptive_enabled": self.anchor_adaptive_enabled,
            "anchor_updates": self.anchor_updates,
            "per_layer_mean_P": per_layer,
            "corr_P_importance": corr,
            "P_hist": {"counts": hist.tolist(), "edges": edges.tolist()},
            "P_sample": sample.tolist(),
            "importance_method": self.importance.method,
            "granularity": self.granularity,
        }

    def _corr_P_importance(self) -> float:
        try:
            imp = self.importance.importance(self.named)
            ps, is_ = [], []
            for n, _ in self.named:
                ps.append(self.P[n].reshape(-1))
                is_.append(imp[n].reshape(-1))
            P = torch.cat(ps).cpu().numpy()
            I = torch.cat(is_).cpu().numpy()
            if P.std() < 1e-8 or I.std() < 1e-8:
                return 0.0
            return float(np.corrcoef(P, I)[0, 1])
        except Exception:
            return 0.0
