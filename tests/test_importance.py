"""Importance estimators: normalization range, method differences."""
import torch
import torch.nn.functional as F

from methods.pnn.importance import ImportanceEstimator
from methods.utils import protected_named_parameters


def _populate_grads(model, bench):
    x, y = bench.tasks[0].train.tensors
    out = model(x[:32], 0)
    F.cross_entropy(out, y[:32]).backward()


def test_importance_in_unit_range(tiny_model, tiny_bench):
    named = list(protected_named_parameters(tiny_model))
    for method in ImportanceEstimator.METHODS:
        est = ImportanceEstimator(named, method=method)
        tiny_model.zero_grad()
        _populate_grads(tiny_model, tiny_bench)
        est.update(named)
        imp = est.importance(named)
        for n, _ in named:
            t = imp[n]
            assert float(t.min()) >= -1e-6 and float(t.max()) <= 1 + 1e-6
            assert t.shape == dict(named)[n].shape


def test_magnitude_matches_weights(tiny_model):
    named = list(protected_named_parameters(tiny_model))
    est = ImportanceEstimator(named, method="magnitude")
    imp = est.importance(named)
    # argmax of importance should match argmax of |w| within a layer
    for n, p in named:
        assert imp[n].argmax() == p.detach().abs().argmax()
