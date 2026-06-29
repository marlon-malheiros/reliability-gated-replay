"""Unit tests: reliability-gate computation + gate/correctness correlation metric.

Covers (Phase-1 deliverable a, d): the gate's per-example scoring behaves as
specified (low CE -> high error-gate; peaked logits -> high confidence-gate, which
is label-free), and the gate<->correctness correlation / separation metrics are
correct, including the inversion sign.
"""
import numpy as np
import torch
import torch.nn as nn

from analysis.gate_metrics import (
    gate_correctness_correlation,
    gate_means,
    gate_separation,
)
from methods.gates import ReliabilityGate


class _LogitModel(nn.Module):
    """features = identity, head = identity -> the input IS the logits (full control)."""

    def features(self, x):
        return x

    def head(self, task_id):
        return nn.Identity()

    def forward(self, x, task_id=0):
        return x


def test_error_gate_high_for_low_ce_low_for_high_ce():
    gate = ReliabilityGate({"signal": "error", "error_threshold": 1.0, "gamma": 5.0})
    model = _LogitModel()
    # one confidently-correct (logits peaked at the true class) and one
    # confidently-wrong (peaked at a different class than the label).
    logits = torch.tensor([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    y = torch.tensor([0, 1])  # first correct, second wrong
    g, _ = gate.score(model, logits, y, task_id=0)
    assert g[0] > 0.8, "low-CE (correct) example should get a high error gate"
    assert g[1] < 0.2, "high-CE (wrong) example should get a low error gate"


def test_confidence_gate_is_label_free():
    """Confidence depends only on the prediction sharpness, not on the label."""
    gate = ReliabilityGate({"signal": "confidence", "tau": 0.5, "gamma": 6.0})
    model = _LogitModel()
    peaked = torch.tensor([[10.0, 0.0, 0.0]])
    uniform = torch.tensor([[0.0, 0.0, 0.0]])
    g_peaked, _ = gate.score(model, peaked, torch.tensor([0]), task_id=0)
    g_uniform, _ = gate.score(model, uniform, torch.tensor([0]), task_id=0)
    assert g_peaked.item() > g_uniform.item()
    # same logits, different label -> identical gate (label-free)
    g_a, _ = gate.score(model, peaked, torch.tensor([0]), task_id=0)
    g_b, _ = gate.score(model, peaked, torch.tensor([2]), task_id=0)
    assert abs(g_a.item() - g_b.item()) < 1e-6
    assert gate.uses_labels is False


def test_gate_never_leaves_model_in_eval():
    gate = ReliabilityGate({"signal": "confidence"})
    model = _LogitModel()
    model.train()
    gate.score(model, torch.zeros(2, 3), torch.tensor([0, 1]), task_id=0)
    assert model.training is True, "gate must restore the model's training mode"


def test_gate_separation_and_means():
    gates = [0.9, 0.8, 0.2, 0.1]
    correct = [1, 1, 0, 0]
    sep = gate_separation(gates, correct)
    assert abs(sep - 0.7) < 1e-9  # (0.9+0.8)/2 - (0.2+0.1)/2
    m = gate_means(gates, correct)
    assert abs(m["gate_on_correct"] - 0.85) < 1e-9
    assert abs(m["gate_on_wrong"] - 0.15) < 1e-9
    assert abs(m["gate_separation"] - 0.7) < 1e-9


def test_gate_correctness_correlation_positive_and_inverted():
    # aligned: high gate on correct -> positive Pearson & Spearman
    gates = [0.95, 0.9, 0.85, 0.15, 0.1, 0.05]
    correct = [1, 1, 1, 0, 0, 0]
    c = gate_correctness_correlation(gates, correct)
    # Spearman on a binary target has tied ranks (3 vs 3), so a perfectly
    # separating monotone gate maxes out below 1.0 (here ~0.88) -- still strongly +.
    assert c["pearson"] > 0.9 and c["spearman"] > 0.85
    # inverted: gate prefers the WRONG samples -> negative correlation + negative sep
    inv_gates = [0.1, 0.05, 0.15, 0.9, 0.95, 0.85]
    c2 = gate_correctness_correlation(inv_gates, correct)
    assert c2["pearson"] < 0 and c2["spearman"] < 0
    assert gate_separation(inv_gates, correct) < 0


def test_correlation_handles_degenerate_input():
    assert np.isnan(gate_correctness_correlation([0.5, 0.5], [1, 1])["pearson"])
    assert np.isnan(gate_separation([0.5], [1])), "one group empty -> nan"
