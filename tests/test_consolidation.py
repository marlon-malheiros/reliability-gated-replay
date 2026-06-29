"""PNN consolidation: P stays in [0,1], grows under closure, reopens; gating."""
import torch

from methods.pnn.consolidation import PNNConsolidation
from methods.pnn.gating import gating_factor


def _consol(model, **over):
    cfg = dict(alpha=0.5, beta=2.0, gating="grad_scale",
               importance={"uniform": True}, maturation=True)
    cfg.update(over)
    return PNNConsolidation(model, cfg)


def test_P_starts_zero_and_grows_clamped(tiny_model):
    c = _consol(tiny_model)
    assert c._flat_P().max() == 0.0
    c.mature(1.0)
    assert abs(c._flat_P().mean() - 0.5) < 1e-6
    c.mature(1.0)
    flat = c._flat_P()
    assert flat.max() <= 1.0 + 1e-6 and flat.min() >= 0.0
    assert abs(flat.mean() - 1.0) < 1e-6   # 0.5 + 0.5 -> clamp at 1


def test_no_maturation_when_signal_zero(tiny_model):
    c = _consol(tiny_model)
    c.mature(0.0)
    assert c._flat_P().mean() == 0.0


def test_reopen_reduces_P(tiny_model):
    c = _consol(tiny_model)
    c.mature(1.0)
    before = c._flat_P().mean()
    c.reopen(0.5)
    assert c._flat_P().mean() < before


def test_gating_factor_monotone():
    P = torch.linspace(0, 1, 5)
    g = gating_factor(P, "grad_scale", beta=2.0)
    assert torch.all(g[:-1] >= g[1:])          # decreasing in P
    assert torch.allclose(gating_factor(torch.zeros(3), "grad_scale", 2.0), torch.ones(3))
    lr = gating_factor(P, "lr_scale", beta=2.0)
    assert torch.allclose(lr, 1 - P)


def test_apply_gating_scales_grads(tiny_model):
    c = _consol(tiny_model)
    c.mature(1.0)  # P=0.5 everywhere
    for _, p in c.named:
        p.grad = torch.ones_like(p)
    c.apply_gating()
    for n, p in c.named:
        expected = torch.exp(-c.beta * c.P[n])
        assert torch.allclose(p.grad, expected, atol=1e-6)
