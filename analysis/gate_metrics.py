"""Gate-diagnostic and calibration metrics for the reliability-gate package.

These are the *standardized* metric definitions for the Neural Networks submission
(see ``results/neural_networks_submission/README_RESULTS.md``). They are pure
functions of arrays / run-records so they can be (a) unit-tested in isolation and
(b) reused by the aggregator that builds the CSVs. Nothing here trains or touches
a GPU.

Central definitions
-------------------
* **gate separation** ``= mean(gate | correct) - mean(gate | wrong)``. Positive when
  the gate scores truly-correct (clean) samples higher than wrong (noisy) ones.
* **inversion** -- a gate is *inverted* when ``gate_separation < 0`` (it prefers the
  mislabeled samples). ``inversion_flag = gate_separation < 0``.
* **time-to-inversion** -- the first evaluation index at which ``gate_separation``
  turns negative *and stays negative for >= 2 consecutive evaluations*; ``NaN`` if it
  never inverts.
* correctness is always defined against the **clean / reference label**, never the
  (possibly noisy) training label.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# gate <-> correctness
# --------------------------------------------------------------------------- #
def gate_separation(gate_values: Sequence[float], correct_mask: Sequence[int]) -> float:
    """``mean(gate|correct) - mean(gate|wrong)``.

    ``correct_mask[i] == 1`` iff sample i's stored label equals its clean label.
    Returns ``nan`` if either group is empty.
    """
    g = np.asarray(gate_values, dtype=float)
    c = np.asarray(correct_mask, dtype=float)
    if g.size == 0 or c.size != g.size:
        return float("nan")
    gc = g[c == 1]
    gw = g[c == 0]
    if gc.size == 0 or gw.size == 0:
        return float("nan")
    return float(gc.mean() - gw.mean())


def gate_means(gate_values: Sequence[float], correct_mask: Sequence[int]) -> Dict[str, float]:
    """Return ``gate_on_correct``, ``gate_on_wrong``, ``gate_mean``, ``gate_separation``."""
    g = np.asarray(gate_values, dtype=float)
    c = np.asarray(correct_mask, dtype=float)
    gc = g[c == 1] if g.size else g
    gw = g[c == 0] if g.size else g
    on_correct = float(gc.mean()) if gc.size else float("nan")
    on_wrong = float(gw.mean()) if gw.size else float("nan")
    sep = (on_correct - on_wrong) if (gc.size and gw.size) else float("nan")
    return {
        "gate_mean": float(g.mean()) if g.size else float("nan"),
        "gate_on_correct": on_correct,
        "gate_on_wrong": on_wrong,
        "gate_separation": sep,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank (ties shared), dependency-free Spearman helper."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def gate_correctness_correlation(
    gate_values: Sequence[float], correct_mask: Sequence[int]
) -> Dict[str, float]:
    """Pearson + Spearman correlation between gate value and true correctness.

    A positive correlation means high gate => correct (clean) sample. A negative
    correlation is the inversion signature.
    """
    g = np.asarray(gate_values, dtype=float)
    c = np.asarray(correct_mask, dtype=float)
    if g.size < 2 or g.size != c.size:
        return {"pearson": float("nan"), "spearman": float("nan")}
    pearson = _pearson(g, c)
    if g.std() < 1e-12 or c.std() < 1e-12:
        spearman = float("nan")
    else:
        spearman = _pearson(_rankdata(g), _rankdata(c))
    return {"pearson": pearson, "spearman": spearman}


# --------------------------------------------------------------------------- #
# inversion over a trajectory of per-evaluation separations
# --------------------------------------------------------------------------- #
def inversion_flag(separation: float) -> bool:
    """A single evaluation is inverted when its gate separation is negative."""
    return bool(np.isfinite(separation) and separation < 0.0)


def inversion_rate(separations: Sequence[float]) -> float:
    """Fraction of (finite) evaluations whose gate separation is negative."""
    s = np.asarray([v for v in separations if np.isfinite(v)], dtype=float)
    if s.size == 0:
        return float("nan")
    return float((s < 0.0).mean())


def time_to_inversion(separations: Sequence[float], min_consecutive: int = 2):
    """First evaluation index where separation goes negative and *stays* negative
    for at least ``min_consecutive`` consecutive evaluations.

    Returns the (0-based) index of the first such evaluation, or ``float('nan')``
    if the trajectory never sustains an inversion. ``separations`` is ordered by
    evaluation step (e.g. one entry per task or per checkpoint). Non-finite entries
    break a run of negatives (treated as "not inverted").
    """
    s = list(separations)
    n = len(s)
    for i in range(n):
        if not (np.isfinite(s[i]) and s[i] < 0.0):
            continue
        run = 0
        j = i
        while j < n and np.isfinite(s[j]) and s[j] < 0.0:
            run += 1
            j += 1
        if run >= min_consecutive:
            return int(i)
    return float("nan")


def run_inversion_summary(separations: Sequence[float], min_consecutive: int = 2) -> Dict[str, float]:
    """Bundle the inversion metrics for one run's per-evaluation separation series."""
    tti = time_to_inversion(separations, min_consecutive)
    final = float(separations[-1]) if len(separations) else float("nan")
    return {
        "final_gate_separation": final,
        "inversion_flag": bool(np.isfinite(tti)),
        "inversion_rate": inversion_rate(separations),
        "time_to_inversion": tti,
    }


# --------------------------------------------------------------------------- #
# buffer diversity / balance
# --------------------------------------------------------------------------- #
def buffer_class_entropy(labels: Sequence[int], n_classes: int | None = None) -> Dict[str, float]:
    """Number of represented classes + normalized class entropy of a buffer.

    Normalized entropy = H(class distribution) / log(n_classes), in [0,1]; 1.0 is a
    perfectly balanced buffer over all classes, 0.0 a single-class buffer.
    """
    y = np.asarray(list(labels), dtype=int)
    if y.size == 0:
        return {"n_classes_represented": 0, "normalized_class_entropy": float("nan")}
    classes, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    h = float(-(p * np.log(p)).sum())
    k = int(n_classes) if n_classes else int(classes.size)
    norm = h / math.log(k) if k > 1 else 0.0
    return {
        "n_classes_represented": int(classes.size),
        "normalized_class_entropy": float(min(max(norm, 0.0), 1.0)),
    }


def buffer_class_balance(labels: Sequence[int]) -> float:
    """min/max class-count ratio in the buffer (1.0 = perfectly balanced)."""
    y = np.asarray(list(labels), dtype=int)
    if y.size == 0:
        return float("nan")
    _, counts = np.unique(y, return_counts=True)
    return float(counts.min() / counts.max())


# --------------------------------------------------------------------------- #
# calibration: ECE / Brier / NLL (shared binning scheme across all methods)
# --------------------------------------------------------------------------- #
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins (the standard scheme)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return float("nan")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.shape[0]
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score = mean squared error of the one-hot vs predicted prob."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return float("nan")
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean NLL (cross-entropy) of the predicted probabilities."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return float("nan")
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def reliability_diagram_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Return (bin_centers, bin_acc, bin_conf, bin_count) for a reliability diagram."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    acc = np.full(n_bins, np.nan)
    cf = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        cnt[b] = int(m.sum())
        if m.sum() > 0:
            acc[b] = float(correct[m].mean())
            cf[b] = float(conf[m].mean())
    return centers, acc, cf, cnt
