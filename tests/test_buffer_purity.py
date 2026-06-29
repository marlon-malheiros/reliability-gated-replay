"""Unit tests: replay-buffer admission + buffer purity / diversity metrics.

Covers (Phase-1 deliverable b, c): admission stores the (possibly noisy) label but
audits against the clean label; ``purity()`` is the fraction of stored==clean; the
oracle gate yields a 100%-pure buffer; class-entropy / balance behave as specified.
"""
import numpy as np
import torch

from analysis.gate_metrics import buffer_class_balance, buffer_class_entropy
from baselines.gated_replay import GatedReplay
from methods.replay_buffer import ReplayBuffer


def test_buffer_purity_counts_stored_vs_clean():
    buf = ReplayBuffer(capacity=10, seed=0)
    x = torch.zeros(4, 3)
    stored = torch.tensor([0, 1, 2, 3])          # what gets rehearsed
    clean = torch.tensor([0, 1, 9, 9])           # truth: last two are mislabeled
    buf.add(x, stored, task_id=0, y_clean=clean)
    assert len(buf) == 4
    assert abs(buf.purity() - 0.5) < 1e-9        # 2 of 4 stored labels are correct


def test_empty_buffer_purity_is_one():
    assert ReplayBuffer(capacity=5).purity() == 1.0


def test_oracle_gate_gives_pure_buffer():
    """The oracle selector admits only truly-clean examples -> purity == 1.0."""
    method = GatedReplay({"buffer_size": 100, "batch_size": 8, "oracle": True, "seed": 0})
    model = _tiny_model()
    method.on_task_start(0, model)
    x = torch.randn(8, 4)
    noisy = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    clean = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0])  # half are flipped
    method.observe_batch(0, uids=torch.arange(8), y_clean=clean)
    method.on_batch_end(model, x, noisy, 0)
    assert method.buf.purity() == 1.0
    # only the truly-clean examples were admitted
    assert len(method.buf) == int((noisy == clean).sum())


def test_plain_replay_admits_noise():
    """error_gated:false (plain ER) admits everything -> buffer purity == data purity."""
    method = GatedReplay({"buffer_size": 100, "batch_size": 8, "error_gated": False, "seed": 0})
    model = _tiny_model()
    method.on_task_start(0, model)
    x = torch.randn(6, 4)
    noisy = torch.tensor([0, 1, 0, 1, 0, 1])
    clean = torch.tensor([0, 1, 0, 1, 1, 0])  # last two flipped -> data purity 4/6
    method.observe_batch(0, uids=torch.arange(6), y_clean=clean)
    method.on_batch_end(model, x, noisy, 0)
    assert len(method.buf) == 6
    assert abs(method.buf.purity() - 4 / 6) < 1e-9


def test_fixed_random_admission_extremes():
    model = _tiny_model()
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    for probability, expected in [(0.0, 0), (1.0, 6)]:
        method = GatedReplay({
            "buffer_size": 100, "fixed_admission_prob": probability, "seed": 0,
        })
        method.on_task_start(0, model)
        method.observe_batch(0, uids=torch.arange(6), y_clean=y)
        method.on_batch_end(model, x, y, 0)
        assert len(method.buf) == expected


def test_oracle_can_be_thinned_without_losing_purity():
    method = GatedReplay({
        "buffer_size": 100, "oracle": True, "oracle_keep_prob": 0.0, "seed": 0,
    })
    model = _tiny_model()
    method.on_task_start(0, model)
    x = torch.randn(4, 4)
    y = torch.tensor([0, 1, 0, 1])
    method.observe_batch(0, uids=torch.arange(4), y_clean=y)
    method.on_batch_end(model, x, y, 0)
    assert len(method.buf) == 0
    assert method.buf.purity() == 1.0


def test_class_entropy_and_balance():
    # perfectly balanced over 2 of 4 classes
    half = buffer_class_entropy([0, 0, 1, 1], n_classes=4)
    assert half["n_classes_represented"] == 2
    assert abs(half["normalized_class_entropy"] - (np.log(2) / np.log(4))) < 1e-9
    # single class -> zero entropy
    single = buffer_class_entropy([3, 3, 3], n_classes=4)
    assert single["n_classes_represented"] == 1
    assert single["normalized_class_entropy"] == 0.0
    # balance ratio
    assert buffer_class_balance([0, 0, 1, 1]) == 1.0
    assert abs(buffer_class_balance([0, 0, 0, 1]) - (1 / 3)) < 1e-9


def _tiny_model():
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def features(self, x):
            return x

        def head(self, t):
            return self.lin

        def forward(self, x, t=0):
            return self.lin(x)

    return M()
