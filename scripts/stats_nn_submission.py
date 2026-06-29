#!/usr/bin/env python
"""Inferential statistics over the submission CSVs -- roadmap sections 4 and 5.

Pure post-processing of ``csv/final_metrics.csv`` (no runs, no GPU). Produces the
paired tests, effect sizes, multiplicity correction, bootstrap CIs, and the
mechanism regressions that the Neural Networks reviewer set will look for, plus a
human-readable ``STATS_SUMMARY.md`` that states each named test's verdict.

Design choices (deliberately conservative and honest):

* **Pairing.** Method-vs-ER comparisons are *paired* on ``(benchmark, condition,
  seed)`` -- the same task stream and seed, differing only in the method. Deltas are
  ``method - ER``. This matches the roadmap's "same seeds/task streams" requirement.
* **Two granularities.** Per-``(benchmark, condition)`` tests (few pairs, noise-regime
  specific -- the roadmap's "under moderate/high noise" tests) AND pooled-over-condition
  tests within a benchmark (more pairs, "overall vs ER"). Every row carries ``n_pairs``
  and an ``underpowered`` flag (a two-sided Wilcoxon cannot reach p<0.05 below n=6).
* **Tests.** Wilcoxon signed-rank (exact when possible), paired Cohen's ``dz``, matched-
  pairs rank-biserial ``r``, percentile bootstrap 95% CI on the mean delta.
* **Multiplicity.** Benjamini-Hochberg FDR within each *metric family* (performance /
  mechanism / calibration), across the set of method-vs-ER comparisons.
* **Mechanism regressions.** delta-acc-vs-ER ~ gate_separation, ~ buffer_purity, and
  delta-forgetting-vs-ER ~ gate_separation: Pearson + Spearman + OLS slope with a
  bootstrap CI. This is the "predictive diagnostic" claim.

Outputs (under ``--output``, default ``.../stats``):
  descriptives.csv          mean / sd / sem / t-95%CI per (benchmark, method, condition, metric)
  paired_tests.csv          every method-vs-ER paired comparison, with BH-corrected p
  frontier_correlations.csv the three mechanism regressions, per benchmark and pooled
  oracle_gap.csv            oracle vs each realizable gate (paired), accuracy + purity
  STATS_SUMMARY.md          the named roadmap tests with an explicit verdict line

Usage:
    python scripts/stats_nn_submission.py \
        --metrics results/neural_networks_submission/csv/final_metrics.csv \
        --output  results/neural_networks_submission/stats
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

BASELINE = "er"
N_BOOT = 10000
BOOT_SEED = 12345

# Metric families drive the BH-FDR grouping. "lower_is_better" flips the verdict wording.
METRIC_FAMILIES: Dict[str, List[str]] = {
    "performance": ["average_accuracy", "mean_forgetting", "backward_transfer"],
    "mechanism": ["buffer_purity", "gate_separation", "gate_corr", "inversion_rate"],
    "calibration": ["ece", "brier", "nll"],
}
LOWER_IS_BETTER = {"mean_forgetting", "inversion_rate", "ece", "brier", "nll"}
FAMILY_OF = {m: fam for fam, ms in METRIC_FAMILIES.items() for m in ms}
ALL_METRICS = [m for ms in METRIC_FAMILIES.values() for m in ms]

# Realizable self-scoring gates (exclude the oracle, which uses clean labels).
GATE_METHODS = (
    "gate_loss", "gate_conf", "gate_coteach", "gate_teacher", "gate_spr",
    "gate_predstab", "gate_reprstab", "gate_agree",
    "replay_agree", "replay_adaptive", "replay_teacher_m95",
    "replay_early", "replay_early5",
)


# --------------------------------------------------------------------------- #
# small stats helpers (no statsmodels dependency)
# --------------------------------------------------------------------------- #
def t_ci95(x: np.ndarray) -> Tuple[float, float, float, float]:
    """mean, sem, and t-based 95% CI for a 1-D sample."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    m = x.mean()
    if n == 1:
        return (m, np.nan, np.nan, np.nan)
    sem = x.std(ddof=1) / np.sqrt(n)
    h = sem * stats.t.ppf(0.975, n - 1)
    return (m, sem, m - h, m + h)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = N_BOOT,
                      seed: int = BOOT_SEED) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean of a paired-delta sample."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def cohen_dz(delta: np.ndarray) -> float:
    """Paired effect size: mean(delta) / sd(delta)."""
    delta = np.asarray(delta, float)
    delta = delta[~np.isnan(delta)]
    if delta.size < 2:
        return np.nan
    sd = delta.std(ddof=1)
    return float(delta.mean() / sd) if sd > 0 else np.nan


def rank_biserial(delta: np.ndarray) -> float:
    """Matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-), zeros dropped."""
    d = np.asarray(delta, float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    tot = w_pos + w_neg
    return float((w_pos - w_neg) / tot) if tot > 0 else np.nan


def wilcoxon_p(delta: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p; exact for small n, NaN if degenerate."""
    d = np.asarray(delta, float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    if d.size < 1:
        return np.nan
    try:
        mode = "exact" if d.size <= 25 else "approx"
        return float(stats.wilcoxon(d, alternative="two-sided", mode=mode).pvalue)
    except ValueError:
        return np.nan


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """BH-FDR adjusted p-values; NaNs are passed through unchanged."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    pm = p[mask]
    n = pm.size
    if n == 0:
        return out
    order = np.argsort(pm)
    ranked = pm[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    adj = np.clip(adj, 0, 1)
    res = np.empty(n)
    res[order] = adj
    out[mask] = res
    return out


def ols_slope_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = BOOT_SEED) -> Tuple[float, float, float, float]:
    """OLS slope + intercept with a percentile-bootstrap 95% CI on the slope."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    slope, intercept = np.polyfit(x, y, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    slopes = np.array([np.polyfit(x[i], y[i], 1)[0]
                       if np.ptp(x[i]) > 0 else np.nan for i in idx])
    lo, hi = np.nanpercentile(slopes, [2.5, 97.5])
    return (float(slope), float(intercept), float(lo), float(hi))


# --------------------------------------------------------------------------- #
# core computations
# --------------------------------------------------------------------------- #
def build_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (bench, method, cond), g in df.groupby(["benchmark", "method", "condition"]):
        for metric in ALL_METRICS:
            if metric not in g:
                continue
            vals = g[metric].to_numpy(float)
            if np.all(np.isnan(vals)):
                continue
            m, sem, lo, hi = t_ci95(vals)
            rows.append(dict(benchmark=bench, method=method, condition=cond,
                             metric=metric, n=int(np.sum(~np.isnan(vals))),
                             mean=m, sem=sem, ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


# Baselines we run paired comparisons against. ER -> Table S1, DER++ -> Table S2.
BASELINES = ("er", "derpp")


def _paired_deltas(df: pd.DataFrame, bench: str, method: str, metric: str,
                   conditions: Optional[Sequence[str]] = None,
                   baseline: str = BASELINE
                   ) -> Tuple[np.ndarray, List[str]]:
    """method - baseline deltas paired on (condition, seed); optional condition subset."""
    sub = df[(df.benchmark == bench)]
    if conditions is not None:
        sub = sub[sub.condition.isin(conditions)]
    base = sub[sub.method == baseline]
    cur = sub[sub.method == method]
    keys = ["condition", "seed"]
    merged = cur.merge(base, on=keys, suffixes=("_m", "_b"))
    if merged.empty or f"{metric}_m" not in merged or f"{metric}_b" not in merged:
        return np.array([]), []
    delta = merged[f"{metric}_m"].to_numpy(float) - merged[f"{metric}_b"].to_numpy(float)
    labels = [f"{c}:s{s}" for c, s in zip(merged.condition, merged.seed)]
    keep = ~np.isnan(delta)
    return delta[keep], [l for l, k in zip(labels, keep) if k]


def paired_test_row(df, bench, method, metric, conditions, scope_label,
                    baseline=BASELINE) -> Optional[dict]:
    delta, _ = _paired_deltas(df, bench, method, metric, conditions, baseline=baseline)
    if delta.size == 0:
        return None
    m, sem, t_lo, t_hi = t_ci95(delta)
    b_lo, b_hi = bootstrap_mean_ci(delta)
    return dict(
        benchmark=bench, method=method, baseline=baseline, metric=metric,
        family=FAMILY_OF[metric], scope=scope_label, n_pairs=int(delta.size),
        mean_delta=m, boot_ci_low=b_lo, boot_ci_high=b_hi,
        t_ci_low=t_lo, t_ci_high=t_hi,
        cohen_dz=cohen_dz(delta), rank_biserial=rank_biserial(delta),
        p_wilcoxon=wilcoxon_p(delta),
        underpowered=bool(delta.size < 6),
        lower_is_better=metric in LOWER_IS_BETTER,
    )


def build_paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for baseline in BASELINES:
        if baseline not in set(df.method):
            continue
        methods = [m for m in df.method.unique() if m != baseline]
        for bench in sorted(df.benchmark.unique()):
            bdf = df[df.benchmark == bench]
            if baseline not in set(bdf.method):
                continue
            conds = sorted(bdf.condition.unique())
            for method in sorted(methods):
                if method not in set(bdf.method):
                    continue
                for metric in ALL_METRICS:
                    if metric not in bdf:
                        continue
                    r = paired_test_row(df, bench, method, metric, None, "pooled", baseline)
                    if r:
                        rows.append(r)
                    for c in conds:
                        r = paired_test_row(df, bench, method, metric, [c],
                                            f"cond={c}", baseline)
                        if r:
                            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # BH-FDR within each (baseline, benchmark, scope, family) bundle of comparisons
    out["p_bh"] = np.nan
    for _, idx in out.groupby(["baseline", "benchmark", "scope", "family"]).groups.items():
        out.loc[idx, "p_bh"] = benjamini_hochberg(out.loc[idx, "p_wilcoxon"].to_numpy())
    return out


def write_stat_tables(out_dir: Path, paired: pd.DataFrame) -> None:
    """Emit Table S1 (vs ER) and S2 (vs DER++): pooled per-benchmark method comparisons
    on the key metrics, with mean delta, 95% bootstrap CI, dz, and BH-corrected p."""
    if paired.empty:
        return
    key_metrics = ["average_accuracy", "mean_forgetting", "buffer_purity",
                   "gate_separation", "ece", "brier"]
    label = {"er": "S1", "derpp": "S2"}
    for baseline, sx in label.items():
        sub = paired[(paired.baseline == baseline) & (paired.scope == "pooled")
                     & (paired.metric.isin(key_metrics))].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["benchmark", "metric", "method"])
        cols = ["benchmark", "method", "metric", "n_pairs", "mean_delta",
                "boot_ci_low", "boot_ci_high", "cohen_dz", "p_wilcoxon", "p_bh",
                "underpowered"]
        sub[cols].to_csv(out_dir / f"table_{sx}_vs_{baseline}.csv", index=False)


def build_frontier_correlations(final: pd.DataFrame) -> pd.DataFrame:
    """delta-vs-ER ~ {gate_separation, buffer_purity} across realizable gate runs.

    One observation per (benchmark, condition, seed, gate-method): the gate run's
    own gate_separation / buffer_purity vs its accuracy (or forgetting) delta to the
    matched ER run. This is the "gate diagnostic predicts gated-replay payoff" claim.
    """
    recs = []
    gate_runs = final[final.method.isin(GATE_METHODS)]
    base = final[final.method == BASELINE]
    keys = ["benchmark", "condition", "seed"]
    merged = gate_runs.merge(base, on=keys, suffixes=("_g", "_b"))
    merged["d_acc"] = merged["average_accuracy_g"] - merged["average_accuracy_b"]
    merged["d_forget"] = merged["mean_forgetting_g"] - merged["mean_forgetting_b"]

    specs = [
        ("d_acc", "gate_separation_g", "d_acc ~ gate_separation"),
        ("d_acc", "buffer_purity_g", "d_acc ~ buffer_purity"),
        ("d_forget", "gate_separation_g", "d_forget ~ gate_separation"),
    ]

    def _one(scope, sub):
        for ycol, xcol, name in specs:
            x = sub[xcol].to_numpy(float)
            y = sub[ycol].to_numpy(float)
            m = ~(np.isnan(x) | np.isnan(y))
            x, y = x[m], y[m]
            if x.size < 3 or np.ptp(x) == 0:
                continue
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            slope, intercept, s_lo, s_hi = ols_slope_ci(x, y)
            recs.append(dict(
                scope=scope, relation=name, n=int(x.size),
                pearson_r=pr, pearson_p=pp, spearman_rho=sr, spearman_p=sp,
                ols_slope=slope, ols_intercept=intercept,
                slope_ci_low=s_lo, slope_ci_high=s_hi))

    _one("pooled_all", merged)
    for bench, sub in merged.groupby("benchmark"):
        _one(f"benchmark={bench}", sub)
    return pd.DataFrame(recs)


def build_oracle_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Oracle vs each realizable gate (paired on condition,seed): accuracy + purity gap."""
    if "oracle" not in set(df.method):
        return pd.DataFrame()
    rows = []
    for bench in sorted(df.benchmark.unique()):
        bdf = df[df.benchmark == bench]
        if "oracle" not in set(bdf.method):
            continue
        oracle = bdf[bdf.method == "oracle"]
        for method in sorted(set(bdf.method) & set(GATE_METHODS)):
            cur = bdf[bdf.method == method]
            merged = oracle.merge(cur, on=["condition", "seed"], suffixes=("_o", "_g"))
            if merged.empty:
                continue
            for metric in ("average_accuracy", "buffer_purity"):
                gap = merged[f"{metric}_o"].to_numpy(float) - merged[f"{metric}_g"].to_numpy(float)
                gap = gap[~np.isnan(gap)]
                if gap.size == 0:
                    continue
                m, sem, lo, hi = t_ci95(gap)
                rows.append(dict(benchmark=bench, gate=method, metric=metric,
                                 n_pairs=int(gap.size), oracle_minus_gate=m,
                                 t_ci_low=lo, t_ci_high=hi,
                                 p_wilcoxon=wilcoxon_p(gap), cohen_dz=cohen_dz(gap)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# named-test summary (roadmap section 5)
# --------------------------------------------------------------------------- #
def _two_method_deltas(df: pd.DataFrame, bench: str, method_a: str, method_b: str,
                       metric: str, conditions: Optional[Sequence[str]]
                       ) -> np.ndarray:
    """method_a - method_b deltas, paired on (condition, seed). For gate-vs-gate tests
    on gate-only quantities (e.g. inversion_rate, gate_separation) where ER has no value."""
    sub = df[df.benchmark == bench]
    if conditions is not None:
        sub = sub[sub.condition.isin(conditions)]
    a = sub[sub.method == method_a]
    b = sub[sub.method == method_b]
    merged = a.merge(b, on=["condition", "seed"], suffixes=("_a", "_b"))
    if merged.empty or f"{metric}_a" not in merged:
        return np.array([])
    d = merged[f"{metric}_a"].to_numpy(float) - merged[f"{metric}_b"].to_numpy(float)
    return d[~np.isnan(d)]


def _fmt_delta(label: str, delta: np.ndarray) -> str:
    if delta.size == 0:
        return f"  - {label}: *no matched data.*"
    m = float(np.nanmean(delta))
    p = wilcoxon_p(delta)
    dz = cohen_dz(delta)
    lo, hi = bootstrap_mean_ci(delta)
    flag = " _(underpowered, n<6)_" if delta.size < 6 else ""
    sig = "n/a" if np.isnan(p) else f"{p:.3f}"
    return (f"  - {label}: Δ={m:+.3f} [boot95 {lo:+.3f},{hi:+.3f}], "
            f"dz={dz:+.2f}, Wilcoxon p={sig}, n={delta.size}{flag}")


def _fmt_test(df_paired: pd.DataFrame, bench: str, method: str, metric: str,
              conditions: Optional[Sequence[str]], df_raw: pd.DataFrame) -> str:
    delta, _ = _paired_deltas(df_raw, bench, method, metric, conditions)
    return _fmt_delta(f"**{method}** vs ER ({bench}, {conditions or 'pooled'}, {metric})", delta)


def write_summary(out_dir: Path, df: pd.DataFrame, paired: pd.DataFrame,
                  corr: pd.DataFrame, oracle: pd.DataFrame) -> None:
    L: List[str] = []
    L.append("# Inferential statistics — power–safety frontier\n")
    L.append("_Generated by `scripts/stats_nn_submission.py` from `csv/final_metrics.csv`. "
             "All method-vs-ER comparisons are paired on (condition, seed); deltas are "
             "method−ER. Wilcoxon two-sided, BH-FDR within metric family. "
             "**n<6 pairs cannot reach p<0.05 — flagged.**_\n")

    has = lambda b: b in set(df.benchmark)
    mnist = "split_mnist" if has("split_mnist") else ("permuted_mnist" if has("permuted_mnist") else None)
    moderate = [c for c in ("rate20", "sym20") if c in set(df.condition)]
    high = [c for c in ("rate60", "rate80", "sym60") if c in set(df.condition)]

    L.append("## Named tests (roadmap §5)\n")
    L.append("### Small-loss gate vs ER")
    for bench in [b for b in ("split_mnist", "permuted_mnist", "split_cifar10") if has(b)]:
        L.append(_fmt_test(paired, bench, "gate_loss", "average_accuracy", moderate, df) + "  (moderate noise)")
        L.append(_fmt_test(paired, bench, "gate_loss", "average_accuracy", high, df) + "  (high noise)")
    L.append("\n### Co-teaching gate vs ER under high noise")
    for bench in [b for b in ("split_mnist", "permuted_mnist") if has(b)]:
        L.append(_fmt_test(paired, bench, "gate_coteach", "average_accuracy", high, df))
    L.append("\n### Label-free (confidence) vs supervised (loss) gate — gate-vs-gate")
    L.append("_Inversion_rate is a gate-only quantity and saturates at 1.0 under high noise, "
             "so the discriminating signal is the continuous `gate_separation` "
             "(less-negative = safer/weaker). Paired conf−loss, high-noise conditions._")
    for bench in [b for b in ("split_mnist", "permuted_mnist", "split_cifar10") if has(b)]:
        L.append(_fmt_delta(f"**gate_conf − gate_loss** ({bench}, high noise, gate_separation)",
                            _two_method_deltas(df, bench, "gate_conf", "gate_loss", "gate_separation", high)))
        L.append(_fmt_delta(f"**gate_conf − gate_loss** ({bench}, high noise, inversion_rate)",
                            _two_method_deltas(df, bench, "gate_conf", "gate_loss", "inversion_rate", high)))
    L.append("\n### Oracle vs realizable gates — buffer purity & accuracy gap")
    if not oracle.empty:
        for _, r in oracle.iterrows():
            sig = "n/a" if pd.isna(r.p_wilcoxon) else f"{r.p_wilcoxon:.3f}"
            L.append(f"  - {r.benchmark} oracle−{r.gate} {r.metric}: "
                     f"{r.oracle_minus_gate:+.3f} [{r.t_ci_low:+.3f},{r.t_ci_high:+.3f}], "
                     f"p={sig}, n={int(r.n_pairs)}")
    else:
        L.append("  - _no oracle runs in corpus._")

    L.append("\n## Mechanism regressions (roadmap §4) — does the diagnostic predict the payoff?\n")
    if not corr.empty:
        for _, r in corr[corr.scope == "pooled_all"].iterrows():
            L.append(f"  - **{r.relation}** (pooled, n={int(r.n)}): "
                     f"Pearson r={r.pearson_r:+.2f} (p={r.pearson_p:.3g}), "
                     f"Spearman ρ={r.spearman_rho:+.2f} (p={r.spearman_p:.3g}), "
                     f"OLS slope={r.ols_slope:+.3f} [{r.slope_ci_low:+.3f},{r.slope_ci_high:+.3f}]")
    else:
        L.append("  - _insufficient gate runs for regression._")

    # significant survivors after FDR
    L.append("\n## Comparisons surviving BH-FDR (p_bh < 0.05, n>=6)\n")
    sig = paired[(paired.p_bh < 0.05) & (~paired.underpowered)]
    if sig.empty:
        L.append("  - _none at p_bh<0.05 with n>=6._")
    else:
        for _, r in sig.sort_values("p_bh").head(50).iterrows():
            L.append(f"  - {r.benchmark} {r.scope} {r.metric}: {r.method} vs "
                     f"{r.baseline.upper()} Δ={r.mean_delta:+.3f}, dz={r.cohen_dz:+.2f}, "
                     f"p_bh={r.p_bh:.3f}")

    (out_dir / "STATS_SUMMARY.md").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default="results/neural_networks_submission/csv/final_metrics.csv")
    ap.add_argument("--output", default="results/neural_networks_submission/stats")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    descriptives = build_descriptives(df)
    paired = build_paired_tests(df)
    corr = build_frontier_correlations(df)
    oracle = build_oracle_gap(df)

    descriptives.to_csv(out_dir / "descriptives.csv", index=False)
    paired.to_csv(out_dir / "paired_tests.csv", index=False)
    corr.to_csv(out_dir / "frontier_correlations.csv", index=False)
    oracle.to_csv(out_dir / "oracle_gap.csv", index=False)
    write_stat_tables(out_dir, paired)        # Table S1 (vs ER), S2 (vs DER++)
    write_summary(out_dir, df, paired, corr, oracle)

    n_sig = 0 if paired.empty else int(((paired.p_bh < 0.05) & (~paired.underpowered)).sum())
    print(f"[stats] runs={len(df)}  descriptives={len(descriptives)}  "
          f"paired_tests={len(paired)}  corr_rows={len(corr)}  oracle_rows={len(oracle)}")
    print(f"[stats] paired comparisons surviving BH-FDR (n>=6, p_bh<0.05): {n_sig}")
    print(f"[stats] wrote -> {out_dir}/  (descriptives, paired_tests, "
          f"frontier_correlations, oracle_gap, STATS_SUMMARY.md)")


if __name__ == "__main__":
    main()
