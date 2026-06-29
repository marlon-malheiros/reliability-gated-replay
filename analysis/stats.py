"""Statistical analysis: CIs, paired tests, multiple-comparison correction, effects."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def mean_std_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean()) if n else 0.0
    if n < 2:
        return {"mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean, "n": n}
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(n)
    t = stats.t.ppf(0.5 + confidence / 2, df=n - 1)
    return {
        "mean": mean, "std": sd,
        "ci_low": mean - t * se, "ci_high": mean + t * se, "n": n,
    }


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(a) != len(b):
        return float("nan"), float("nan")
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(a) != len(b) or np.allclose(a, b):
        return float("nan"), float("nan")
    try:
        w, p = stats.wilcoxon(a, b)
        return float(w), float(p)
    except ValueError:
        return float("nan"), float("nan")


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return float("nan")
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    """Matched-pairs rank-biserial effect size for the Wilcoxon test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    r_plus = ranks[diff > 0].sum()
    r_minus = ranks[diff < 0].sum()
    total = ranks.sum()
    return float((r_plus - r_minus) / total)


def holm_bonferroni(pvalues: List[float], alpha: float = 0.05) -> Dict[str, list]:
    """Holm-Bonferroni step-down correction. Returns corrected p and reject flags."""
    p = np.asarray(pvalues, float)
    valid = ~np.isnan(p)
    order = np.argsort(np.where(valid, p, np.inf))
    m = int(valid.sum())
    corrected = np.full_like(p, np.nan)
    reject = np.zeros(len(p), dtype=bool)
    running_max = 0.0
    rank = 0
    for idx in order:
        if not valid[idx]:
            continue
        adj = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, adj)
        corrected[idx] = running_max
        reject[idx] = running_max < alpha
        rank += 1
    return {"corrected": corrected.tolist(), "reject": reject.tolist()}


def compare(
    pnn_values: Sequence[float], baseline_values: Sequence[float]
) -> Dict[str, float]:
    """Full paired comparison of PNN vs a baseline across seeds (higher=better)."""
    t, p_t = paired_ttest(pnn_values, baseline_values)
    w, p_w = wilcoxon(pnn_values, baseline_values)
    return {
        "mean_diff": float(np.mean(pnn_values) - np.mean(baseline_values)),
        "t_stat": t, "p_ttest": p_t,
        "w_stat": w, "p_wilcoxon": p_w,
        "cohens_d": cohens_d_paired(pnn_values, baseline_values),
        "rank_biserial": rank_biserial(pnn_values, baseline_values),
    }
