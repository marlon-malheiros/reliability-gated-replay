#!/usr/bin/env python
"""Build the reviewer-requested robustness analyses from existing/new runs.

Outputs are deliberately separate from the frozen main figures until the
manuscript text is updated:

* integrated power--risk frontier across noise rates,
* accuracy-after-task curves for the four key CIFAR cells,
* class-IL regression with purity, forgetting, and class entropy,
* AER summary including runtime overhead,
* matched-control / threshold / buffer-size tables when those runs exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from scripts.aggregate_nn_submission_results import process  # noqa: E402

DISPLAY = {
    "er": "ER", "derpp": "DER++", "gate_loss": "Small-loss",
    "gate_conf": "Confidence", "gate_spr": "SPR gate",
    "gate_coteach": "Co-teaching", "gate_teacher": "Slow teacher",
    "gate_predstab": "Prediction stability", "gate_reprstab": "Representation stability",
    "ewc": "EWC", "si": "SI", "oracle": "Oracle",
}
COLORS = {
    "er": "#7F7F7F", "derpp": "#D98C3F", "gate_loss": "#B33A3A",
    "gate_conf": "#3977A8", "gate_spr": "#A9578E", "gate_coteach": "#B66B23",
    "gate_teacher": "#4E8F8B", "gate_predstab": "#5E8C5A",
    "gate_reprstab": "#7B5A9E", "ewc": "#806252", "si": "#A38C85",
    "oracle": "#111111",
}
AXIS_FONT = 16
TICK_FONT = 14
LEGEND_FONT = 14
TITLE_FONT = 14
NOTE_FONT = 13


def save_figure(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def reviewer_rows(raw: Path):
    final, per_eval = [], []
    if not raw.exists():
        return pd.DataFrame(), pd.DataFrame()
    for fp in sorted(raw.glob("*.json")):
        try:
            result = json.loads(fp.read_text())
            fr, pe, *_ = process(result, fp.stem)
            meta = result.get("reviewer_control") or {}
            fr.update(meta)
            for row in pe:
                row.update(meta)
            final.append(fr)
            per_eval.extend(pe)
        except Exception as exc:
            print(f"skip {fp.name}: {exc}")
    return pd.DataFrame(final), pd.DataFrame(per_eval)


def integrated_frontier(df: pd.DataFrame, out: Path):
    keys = ["benchmark", "condition", "noise_type", "noise_rate", "seed"]
    er = df[df.method == "er"][keys + ["average_accuracy"]].rename(
        columns={"average_accuracy": "er_accuracy"}
    )
    paired = df.merge(er, on=keys, how="inner")
    paired["gain"] = paired.average_accuracy - paired.er_accuracy
    moderate = paired[(paired.noise_type == "symmetric")
                      & paired.noise_rate.between(0.2, 0.4)]
    high = paired[((paired.noise_type == "symmetric") & (paired.noise_rate >= 0.6))
                  | paired.noise_type.astype(str).str.startswith("asymmetric")]
    rows = []
    for method in sorted(set(moderate.method) & set(high.method)):
        if method == "er":
            continue
        mg = moderate[moderate.method == method].gain
        hg = high[high.method == method].gain
        if mg.empty or hg.empty:
            continue
        rows.append({
            "method": method,
            "power_mean_gain": float(mg.mean()),
            "risk_worst_penalty": float(np.maximum(-hg.to_numpy(), 0).max()),
            "negative_gain_area": float(np.maximum(-hg.to_numpy(), 0).mean()),
            "n_moderate_pairs": len(mg), "n_high_pairs": len(hg),
        })
    tab = pd.DataFrame(rows).sort_values("power_mean_gain", ascending=False)
    tab.to_csv(out / "integrated_frontier.csv", index=False)
    if tab.empty:
        return
    keep = [
        "derpp", "ewc", "si", "gate_loss", "gate_spr", "gate_conf",
        "gate_predstab", "gate_reprstab", "gate_teacher", "gate_coteach", "oracle",
    ]
    tab = tab[tab.method.isin(keep)]
    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    handles = []
    for row in tab.itertuples():
        marker = "*" if row.method == "oracle" else ("^" if row.method in {
            "gate_conf", "gate_predstab", "gate_reprstab"} else "o")
        point = ax.scatter(row.power_mean_gain, row.risk_worst_penalty,
                           s=220 if marker == "*" else 95, marker=marker,
                           color=COLORS.get(row.method, "#777777"), edgecolor="white",
                           linewidth=.9, zorder=3,
                           label=DISPLAY.get(row.method, row.method))
        handles.append(point)
    ax.axhline(0, color="0.5", lw=.8); ax.axvline(0, color="0.5", lw=.8)
    ax.grid(color="0.92", linewidth=.7)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel("Power: mean accuracy gain over ER\n(moderate-noise conditions)", fontsize=16)
    ax.set_ylabel("Risk: worst accuracy penalty versus ER\n(high/structured-noise conditions; lower is safer)",
                  fontsize=16)
    if handles:
        ax.legend(handles=handles, title="Method", title_fontsize=12, fontsize=16,
                  frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0)
    save_figure(fig, out, "fig_integrated_power_risk_frontier")


def task_curves(per_eval: pd.DataFrame, out: Path):
    methods = ["er", "derpp", "gate_loss", "gate_conf", "oracle"]
    panels = [
        ("split_cifar10", 0.2), ("split_cifar10", 0.6),
        ("seq_cifar10", 0.2), ("seq_cifar10", 0.6),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), constrained_layout=True)
    drew = False
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, (bench, rate), panel in zip(axes.flat, panels, panel_labels):
        sub = per_eval[(per_eval.benchmark == bench) & np.isclose(per_eval.noise_rate, rate)]
        for method in methods:
            d = sub[sub.method == method]
            if d.empty:
                continue
            g = d.groupby("eval_after_task").avg_acc_so_far.agg(["mean", "std"])
            ax.errorbar(g.index + 1, g["mean"], yerr=g["std"].fillna(0),
                        marker="o", capsize=2, label=DISPLAY.get(method, method),
                        color=COLORS.get(method, "#777777"), linewidth=2.0, markersize=6)
            drew = True
        name = "Split-CIFAR-10" if bench == "split_cifar10" else "Seq-CIFAR-10"
        ax.set_title(f"{panel} {name}, symmetric {int(rate*100)}%", loc="left", fontsize=TITLE_FONT)
        ax.set_xlabel("Tasks learned", fontsize=AXIS_FONT)
        ax.set_ylabel("Average accuracy so far", fontsize=AXIS_FONT)
        ax.set_xticks(range(1, 6))
        ax.tick_params(axis="both", labelsize=TICK_FONT)
        ax.grid(color="0.92", linewidth=.7)
    if drew:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, 1.03), fontsize=LEGEND_FONT)
        save_figure(fig, out, "figS1_taskwise_cifar_trajectories")
    else:
        plt.close(fig)


def class_il_regression(df: pd.DataFrame, out: Path):
    d = df[df.benchmark.isin(["seq_cifar10", "seq_cifar100", "cifar10n"])].copy()
    cols = ["average_accuracy", "buffer_purity", "mean_forgetting", "buffer_diversity"]
    d = d.dropna(subset=cols)
    if len(d) < 5:
        pd.DataFrame().to_csv(out / "class_il_regression.csv", index=False)
        return
    X = d[["buffer_purity", "mean_forgetting", "buffer_diversity"]].to_numpy(float)
    X = np.column_stack([np.ones(len(X)), X])
    y = d.average_accuracy.to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    pd.DataFrame([{
        "n": len(d), "intercept": beta[0], "beta_buffer_purity": beta[1],
        "beta_mean_forgetting": beta[2], "beta_class_entropy": beta[3], "r2": r2,
    }]).to_csv(out / "class_il_regression.csv", index=False)


def aer_summary(path: Path, out: Path):
    if not path.exists():
        return
    d = pd.read_csv(path)
    d = d[d.rc == 0].copy()
    rows = []
    for (model, noise), g in d.groupby(["model", "noise"]):
        rows.append({
            "model": model, "condition": noise, "n": len(g),
            "class_il_mean": g.class_il.mean(), "class_il_std": g.class_il.std(ddof=1),
            "task_il_mean": g.task_il.mean(), "task_il_std": g.task_il.std(ddof=1),
            "runtime_s_mean": g.seconds.mean(),
        })
    tab = pd.DataFrame(rows)
    er_t = d[(d.model == "er") & (d.noise == "sym60")].seconds.mean()
    tab["runtime_overhead_vs_er_sym60_pct"] = np.where(
        np.isfinite(er_t), 100 * (tab.runtime_s_mean / er_t - 1), np.nan
    )
    tab.to_csv(out / "aer_summary.csv", index=False)
    aer_figure(d, out)


def aer_figure(d: pd.DataFrame, out: Path):
    """Direct AER comparison using only completed official-implementation runs."""
    aer = d[d.model == "er_ace_aer_abs"].copy()
    order = ["sym20", "sym60", "asym40"]
    labels = ["Symmetric 20%", "Symmetric 60%", "Asymmetric 40%"]
    stats = aer.groupby("noise")[["class_il", "task_il"]].agg(["mean", "std"]).reindex(order)
    er = d[(d.model == "er") & (d.noise == "sym60")]
    if stats.empty or er.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(order)); width = .32
    ax.bar(x - width/2, stats[("class_il", "mean")], width,
           yerr=stats[("class_il", "std")], capsize=3, color="#356A8A",
           label="Class-IL")
    ax.bar(x + width/2, stats[("task_il", "mean")], width,
           yerr=stats[("task_il", "std")], capsize=3, color="#78A6A3",
           label="Masked Task-IL")
    # Bridge ER is available only for symmetric 60%.
    ax.scatter([1 - width/2], [er.class_il.mean()], marker="D", s=80,
               color="#555555", label="ER bridge (Class-IL)", zorder=4)
    ax.scatter([1 + width/2], [er.task_il.mean()], marker="D", s=80,
               facecolor="white", edgecolor="#555555",
               label="ER bridge (masked Task-IL)", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Final average accuracy (%)", fontsize=AXIS_FONT)
    ax.set_title("(a) Published AER implementation", loc="left", fontsize=TITLE_FONT)
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    ax.grid(axis="y", color="0.92", linewidth=.7)
    handles, legend_labels = ax.get_legend_handles_labels()

    ax = axes[1]
    aer60 = aer[aer.noise == "sym60"]
    means = [er.seconds.mean() / 60, aer60.seconds.mean() / 60]
    errors = [er.seconds.std(ddof=1) / 60, aer60.seconds.std(ddof=1) / 60]
    bars = ax.bar(["ER bridge", "AER"], means, yerr=errors, capsize=4,
                  color=["#777777", "#356A8A"], width=.58)
    overhead = 100 * (means[1] / means[0] - 1)
    ax.text(1, means[1] + errors[1] + 1.2, f"+{overhead:.0f}%",
            ha="center", va="bottom", fontsize=NOTE_FONT)
    ax.set_ylabel("Runtime per run (minutes)", fontsize=AXIS_FONT)
    ax.set_title("(b) Symmetric 60% runtime", loc="left", fontsize=TITLE_FONT)
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    ax.grid(axis="y", color="0.92", linewidth=.7)
    fig.legend(handles, legend_labels, frameon=False, fontsize=LEGEND_FONT, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.03))
    save_figure(fig, out, "figS2_aer_external_baseline")


def control_tables(review: pd.DataFrame, out: Path):
    if review.empty:
        return
    review.to_csv(out / "reviewer_control_runs.csv", index=False)
    if "control" in review:
        matched = review[review.control.isin(["random_matched", "oracle_thinned"])]
        if not matched.empty:
            matched.to_csv(out / "matched_admission_controls.csv", index=False)
        sens = review[review.control == "threshold_sensitivity"]
        if not sens.empty:
            sens.to_csv(out / "threshold_sensitivity.csv", index=False)
        buf = review[review.control == "buffer_sensitivity"]
        if not buf.empty:
            buf.to_csv(out / "buffer_size_sensitivity.csv", index=False)
        erace = review[review.control == "er_ace"]
        if not erace.empty:
            erace.to_csv(out / "erace_results.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(
        ROOT / "results/neural_networks_submission/csv"))
    ap.add_argument("--reviewer-raw", default=str(
        ROOT / "results/neural_networks_submission/reviewer_controls/raw_logs"))
    ap.add_argument("--aer", default=str(
        ROOT / "results/neural_networks_submission/aer/summary.csv"))
    ap.add_argument("--output", default=str(
        ROOT / "results/neural_networks_submission/reviewer_analysis"))
    args = ap.parse_args()
    inp, out = Path(args.csv), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    final = pd.read_csv(inp / "final_metrics.csv")
    per_eval = pd.read_csv(inp / "per_eval_metrics.csv")
    review, review_eval = reviewer_rows(Path(args.reviewer_raw))
    integrated_frontier(final, out)
    task_curves(pd.concat([per_eval, review_eval], ignore_index=True), out)
    class_il_regression(final, out)
    aer_summary(Path(args.aer), out)
    control_tables(review, out)
    print(f"reviewer analyses -> {out}")


if __name__ == "__main__":
    main()
