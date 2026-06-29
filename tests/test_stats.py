"""Unit tests for the inferential-statistics helpers in scripts/stats_nn_submission.py.

These guard the math the paper's significance claims rest on (roadmap §5): BH-FDR
multiplicity correction, paired effect sizes, rank-biserial, and the bootstrap CI.
Checked against hand-computed / scipy-cross-validated values.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

_SPEC = importlib.util.spec_from_file_location(
    "stats_nn_submission",
    Path(__file__).resolve().parents[1] / "scripts" / "stats_nn_submission.py",
)
S = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(S)


def test_benjamini_hochberg_known_values():
    # Classic BH example: p = [.01,.02,.03,.04,.05] -> adjusted = p * n / rank, monotone.
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    adj = S.benjamini_hochberg(p)
    assert np.allclose(adj, 0.05, atol=1e-12)  # all tie to 0.05 here


def test_benjamini_hochberg_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50)
    adj = S.benjamini_hochberg(p)
    order = np.argsort(p)
    # adjusted p in p-order must be non-decreasing, and within [0,1]
    assert np.all(np.diff(adj[order]) >= -1e-12)
    assert adj.min() >= 0 and adj.max() <= 1


def test_benjamini_hochberg_passes_through_nan():
    adj = S.benjamini_hochberg([0.01, np.nan, 0.5])
    assert np.isnan(adj[1])
    assert not np.isnan(adj[0]) and not np.isnan(adj[2])


def test_cohen_dz_matches_definition():
    delta = np.array([0.2, 0.1, 0.3, 0.0, 0.4])
    expected = delta.mean() / delta.std(ddof=1)
    assert S.cohen_dz(delta) == pytest.approx(expected)


def test_cohen_dz_zero_variance_is_nan():
    assert np.isnan(S.cohen_dz(np.array([0.5, 0.5, 0.5])))


def test_rank_biserial_all_positive_is_one():
    assert S.rank_biserial(np.array([0.1, 0.2, 0.3])) == pytest.approx(1.0)
    assert S.rank_biserial(np.array([-0.1, -0.2, -0.3])) == pytest.approx(-1.0)


def test_rank_biserial_symmetric_is_zero():
    # equal-magnitude opposite-sign pairs -> W+ == W- -> r == 0
    assert S.rank_biserial(np.array([1.0, -1.0, 2.0, -2.0])) == pytest.approx(0.0)


def test_wilcoxon_p_matches_scipy():
    delta = np.array([0.3, 0.1, 0.4, 0.2, 0.5, 0.25])
    expected = sps.wilcoxon(delta, alternative="two-sided", mode="exact").pvalue
    assert S.wilcoxon_p(delta) == pytest.approx(expected)


def test_wilcoxon_p_drops_zeros_and_nans():
    # zeros and nans removed; should equal scipy on the cleaned vector
    delta = np.array([0.3, 0.0, np.nan, 0.1, 0.4, 0.2, 0.5])
    cleaned = np.array([0.3, 0.1, 0.4, 0.2, 0.5])
    expected = sps.wilcoxon(cleaned, alternative="two-sided", mode="exact").pvalue
    assert S.wilcoxon_p(delta) == pytest.approx(expected)


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 0.1, 200)
    lo, hi = S.bootstrap_mean_ci(x)
    assert lo < x.mean() < hi
    assert hi - lo < 0.1  # tight for n=200


def test_bootstrap_ci_is_deterministic():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert S.bootstrap_mean_ci(x) == S.bootstrap_mean_ci(x)  # fixed seed


def test_t_ci95_matches_scipy_interval():
    x = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
    m, sem, lo, hi = S.t_ci95(x)
    exp_lo, exp_hi = sps.t.interval(0.95, len(x) - 1, loc=x.mean(),
                                    scale=sps.sem(x))
    assert m == pytest.approx(x.mean())
    assert lo == pytest.approx(exp_lo)
    assert hi == pytest.approx(exp_hi)


def test_ols_slope_recovers_known_line():
    x = np.linspace(0, 1, 50)
    y = 2.0 * x + 0.5
    slope, intercept, lo, hi = S.ols_slope_ci(x, y)
    assert slope == pytest.approx(2.0, abs=1e-6)
    assert intercept == pytest.approx(0.5, abs=1e-6)
    assert lo <= slope <= hi
