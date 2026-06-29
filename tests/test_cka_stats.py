"""CKA / cosine and the statistics helpers."""
import numpy as np

from analysis.cka import linear_cka, mean_cosine
from analysis.stats import (
    cohens_d_paired,
    holm_bonferroni,
    mean_std_ci,
    paired_ttest,
)


def test_cka_identity_is_one():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 8)
    assert np.isclose(linear_cka(X, X), 1.0, atol=1e-6)


def test_cka_in_unit_interval():
    rng = np.random.RandomState(1)
    X, Y = rng.randn(50, 8), rng.randn(50, 5)
    c = linear_cka(X, Y)
    assert 0.0 <= c <= 1.0


def test_cosine_self_is_one():
    rng = np.random.RandomState(2)
    X = rng.randn(20, 6)
    assert np.isclose(mean_cosine(X, X), 1.0, atol=1e-6)


def test_ci_and_ttest():
    ci = mean_std_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(ci["mean"], 3.0)
    assert ci["ci_low"] < 3.0 < ci["ci_high"]
    t, p = paired_ttest([0.9, 0.92, 0.88], [0.7, 0.72, 0.68])
    assert p < 0.05
    assert cohens_d_paired([0.9, 0.92, 0.88], [0.7, 0.72, 0.68]) > 0


def test_holm_bonferroni():
    out = holm_bonferroni([0.001, 0.04, 0.5])
    assert out["reject"][0] is True
    assert out["reject"][2] is False
    assert out["corrected"][0] >= 0.001
