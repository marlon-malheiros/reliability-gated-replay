"""Figure generation (Figures 1-14 of the spec).

Each ``fig_*`` consumes the analysis *bundle* assembled by ``analyze_results.py``
and writes a PNG + PDF into the figures directory. Figures 1-2 are schematic;
3-14 are data-driven. Everything is defensive: a figure with missing inputs draws
a labelled placeholder rather than aborting the batch.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 10})


# --------------------------------------------------------------------------- #
def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png")
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)


def _methods(bundle, kinds=None) -> List[str]:
    items = bundle["methods"]
    labels = [k for k, v in items.items() if (kinds is None or v["kind"] in kinds)]
    return sorted(labels)


def _agg_value(bundle, label, key, default=0.0):
    a = bundle["methods"][label]["agg"].get(key)
    return (a["mean"] if a else default), (a["std"] if a else 0.0)


def _flatten_history(result) -> List[Dict[str, Any]]:
    rows, boundaries = [], []
    for hist in result.get("task_history", []):
        boundaries.append(len(rows))
        rows.extend(hist)
    return rows, boundaries


def _placeholder(outdir, name, msg):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True)
    ax.axis("off")
    _save(fig, outdir, name)


# --------------------------------------------------------------------------- #
def fig01_method_overview(bundle, outdir):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    steps = [
        "Plastic\nlearning",
        "Error-gated\nclosure detector",
        "Importance\nestimation",
        "PNN maturation\nP ↑",
        "Plasticity gating\ng × exp(-βP)",
        "Adaptive\nreopening",
    ]
    x = np.linspace(0.04, 0.83, len(steps))
    for i, (xi, s) in enumerate(zip(x, steps)):
        box = FancyBboxPatch((xi, 0.35), 0.12, 0.3, boxstyle="round,pad=0.02",
                             fc="#cfe8ff" if i % 2 == 0 else "#d8f5d8", ec="#333")
        ax.add_patch(box)
        ax.text(xi + 0.06, 0.5, s, ha="center", va="center", fontsize=8.5)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((xi + 0.12, 0.5), (x[i + 1], 0.5),
                         arrowstyle="-|>", mutation_scale=12, color="#555"))
    ax.set_title("Figure 1 — Error-Gated PNN Maturation: method overview")
    _save(fig, outdir, "fig01_method_overview")


def fig02_maturation_mechanism(bundle, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    e = np.linspace(0, 1, 200)
    gamma = 5.0
    thr = 0.15
    sig = 1 / (1 + np.exp(-gamma * (thr - e)))
    axes[0].plot(e, sig, color="#1f77b4")
    axes[0].axvline(thr, ls="--", color="gray")
    axes[0].set_xlabel("validation error  E_t")
    axes[0].set_ylabel("closure_signal")
    axes[0].set_title("Closure gate  σ(γ(thr − E))")
    P = np.linspace(0, 1, 200)
    for beta in (2, 5, 10):
        axes[1].plot(P, np.exp(-beta * P), label=f"β={beta}")
    axes[1].plot(P, 1 - P, "k--", label="1 − P (lr scale)")
    axes[1].set_xlabel("consolidation  P")
    axes[1].set_ylabel("gradient gating factor")
    axes[1].set_title("Plasticity gating")
    axes[1].legend(fontsize=8)
    fig.suptitle("Figure 2 — PNN maturation mechanism")
    _save(fig, outdir, "fig02_maturation_mechanism")


def fig03_accuracy_over_tasks(bundle, outdir):
    labels = _methods(bundle, {"baseline", "pnn"})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in labels:
        res = bundle["methods"][label]["sample_result"]
        R = np.array(res["acc_matrix"])
        running = [R[i, : i + 1].mean() for i in range(R.shape[0])]
        ax.plot(range(1, len(running) + 1), running, marker="o",
                label=label, lw=2 if label == bundle.get("pnn_label") else 1)
    ax.set_xlabel("tasks seen")
    ax.set_ylabel("avg accuracy over learned tasks")
    ax.set_title("Figure 3 — Accuracy over the task sequence")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, outdir, "fig03_accuracy_over_tasks")


def fig04_forgetting_comparison(bundle, outdir):
    labels = _methods(bundle, {"baseline", "pnn"})
    means = [_agg_value(bundle, m, "mean_forgetting") for m in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = [m[0] for m in means]
    err = [m[1] for m in means]
    colors = ["#d62728" if m == bundle.get("pnn_label") else "#1f77b4" for m in labels]
    ax.bar(labels, y, yerr=err, color=colors, capsize=3)
    ax.set_ylabel("mean forgetting (lower better)")
    ax.set_title("Figure 4 — Forgetting comparison")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, outdir, "fig04_forgetting_comparison")


def fig05_weight_drift(bundle, outdir):
    labels = _methods(bundle, {"baseline", "pnn"})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in labels:
        res = bundle["methods"][label]["sample_result"]
        d = res.get("weight_drift", [])
        if d:
            ax.plot(range(1, len(d) + 1), d, marker="o", label=label,
                    lw=2 if label == bundle.get("pnn_label") else 1)
    ax.set_xlabel("task")
    ax.set_ylabel("‖θ − θ₀‖")
    ax.set_title("Figure 5 — Cumulative weight drift")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, outdir, "fig05_weight_drift")


def fig06_representation_drift(bundle, outdir):
    labels = _methods(bundle, {"baseline", "pnn"})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for label in labels:
        runs = bundle["methods"][label]["runs"]
        cka = runs[0].get("cka_matrix")
        if cka and len(cka) > 1:
            cka = np.array(cka)
            ax.plot(range(len(cka)), cka[0], marker="o", label=label,
                    lw=2 if label == bundle.get("pnn_label") else 1)
            plotted = True
    if not plotted:
        return _placeholder(outdir, "fig06_representation_drift",
                            "Figure 6 — representation drift\n(needs >1 task)")
    ax.set_xlabel("task snapshot")
    ax.set_ylabel("CKA with task-0 representation")
    ax.set_title("Figure 6 — Representational stability (CKA)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, outdir, "fig06_representation_drift")


def fig07_pi_distribution(bundle, outdir):
    pnn = bundle.get("pnn_label")
    if not pnn:
        return _placeholder(outdir, "fig07_pi_distribution", "Figure 7 — no PNN run")
    sample = bundle["methods"][pnn]["runs"][0].get("P_sample")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if sample:
        ax.hist(sample, bins=25, range=(0, 1), color="#9467bd", alpha=0.85)
    ax.set_xlabel("consolidation value  P_i")
    ax.set_ylabel("count")
    ax.set_title("Figure 7 — Distribution of P_i (final)")
    _save(fig, outdir, "fig07_pi_distribution")


def fig08_closure_dynamics(bundle, outdir):
    pnn = bundle.get("pnn_label")
    res = bundle["methods"][pnn]["sample_result"] if pnn else None
    rows, bounds = _flatten_history(res) if res else ([], [])
    if not rows:
        return _placeholder(outdir, "fig08_closure_dynamics", "Figure 8 — no closure log")
    cs = [r.get("closure_signal", np.nan) for r in rows]
    mp = [r.get("mean_P", np.nan) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cs, label="closure_signal", color="#1f77b4")
    ax.plot(mp, label="mean P", color="#d62728")
    for b in bounds:
        ax.axvline(b, ls=":", color="gray", alpha=0.6)
    ax.set_xlabel("epoch (across tasks; dotted = task start)")
    ax.set_title("Figure 8 — Closure & consolidation dynamics")
    ax.legend(fontsize=8)
    _save(fig, outdir, "fig08_closure_dynamics")


def fig09_error_closure_trigger(bundle, outdir):
    pnn = bundle.get("pnn_label")
    res = bundle["methods"][pnn]["sample_result"] if pnn else None
    rows, bounds = _flatten_history(res) if res else ([], [])
    if not rows:
        return _placeholder(outdir, "fig09_error_closure_trigger", "Figure 9 — no log")
    E = [r.get("E_t", r.get("val_loss", np.nan)) for r in rows]
    closed = [r.get("closed", False) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(E, label="E_t (EMA val loss)", color="#1f77b4")
    first_closed = next((i for i, c in enumerate(closed) if c), None)
    if first_closed is not None:
        ax.axvline(first_closed, color="green", ls="--", label="closure trigger")
    for b in bounds:
        ax.axvline(b, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("error")
    ax.set_title("Figure 9 — Error evolution & closure trigger")
    ax.legend(fontsize=8)
    _save(fig, outdir, "fig09_error_closure_trigger")


def fig10_plasticity_stability(bundle, outdir):
    labels = _methods(bundle, {"baseline", "pnn"})
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label in labels:
        x, _ = _agg_value(bundle, label, "plasticity_index")
        f, _ = _agg_value(bundle, label, "mean_forgetting")
        ax.scatter(x, 1 - f, s=70,
                   color="#d62728" if label == bundle.get("pnn_label") else "#1f77b4")
        ax.annotate(label, (x, 1 - f), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("plasticity (mean learning accuracy)")
    ax.set_ylabel("stability (1 − forgetting)")
    ax.set_title("Figure 10 — Plasticity vs stability")
    _save(fig, outdir, "fig10_plasticity_stability")


def fig11_ablation_heatmap(bundle, outdir):
    labels = _methods(bundle, {"ablation"})
    if not labels:
        return _placeholder(outdir, "fig11_ablation_heatmap",
                            "Figure 11 — ablation heatmap\n(no ablation runs in this set)")
    keys = ["average_accuracy", "mean_forgetting", "backward_transfer",
            "mean_P", "frac_consolidated", "representational_drift"]
    M = np.array([[_agg_value(bundle, lb, k)[0] for k in keys] for lb in labels])
    # column-normalize for visual comparability
    Mn = (M - M.min(0)) / (np.ptp(M, axis=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(labels))))
    im = ax.imshow(Mn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(keys)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="w", fontsize=6)
    fig.colorbar(im, ax=ax, label="column-normalized")
    ax.set_title("Figure 11 — Ablation heatmap")
    _save(fig, outdir, "fig11_ablation_heatmap")


def fig12_baseline_comparison(bundle, outdir):
    labels = _methods(bundle, {"baseline", "pnn"})
    acc = [_agg_value(bundle, m, "average_accuracy") for m in labels]
    forg = [_agg_value(bundle, m, "mean_forgetting") for m in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, [a[0] for a in acc], 0.4, yerr=[a[1] for a in acc],
           label="avg accuracy", capsize=2)
    ax.bar(x + 0.2, [f[0] for f in forg], 0.4, yerr=[f[1] for f in forg],
           label="mean forgetting", capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Figure 12 — Baseline comparison")
    ax.legend(fontsize=8)
    _save(fig, outdir, "fig12_baseline_comparison")


def fig13_reopening(bundle, outdir):
    pnn = bundle.get("pnn_label")
    res = bundle["methods"][pnn]["sample_result"] if pnn else None
    rows, bounds = _flatten_history(res) if res else ([], [])
    if not rows:
        return _placeholder(outdir, "fig13_reopening", "Figure 13 — no reopening log")
    mp = [r.get("mean_P", np.nan) for r in rows]
    reopened = [i for i, r in enumerate(rows) if r.get("reopened")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(mp, color="#d62728", label="mean P")
    for b in bounds:
        ax.axvline(b, ls=":", color="gray", alpha=0.5)
    for r in reopened:
        ax.axvline(r, color="orange", alpha=0.7)
    if reopened:
        ax.plot([], [], color="orange", label="reopening event")
    ax.set_xlabel("epoch")
    ax.set_title("Figure 13 — Adaptive reopening behaviour")
    ax.legend(fontsize=8)
    _save(fig, outdir, "fig13_reopening")


def fig14_dashboard(bundle, outdir):
    labels = _methods(bundle, {"baseline", "pnn"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # (a) accuracy
    for m in labels:
        a, _ = _agg_value(bundle, m, "average_accuracy")
        axes[0, 0].bar(m, a, color="#d62728" if m == bundle.get("pnn_label") else "#1f77b4")
    axes[0, 0].set_title("avg accuracy")
    axes[0, 0].tick_params(axis="x", rotation=45, labelsize=7)
    # (b) forgetting
    for m in labels:
        f, _ = _agg_value(bundle, m, "mean_forgetting")
        axes[0, 1].bar(m, f, color="#d62728" if m == bundle.get("pnn_label") else "#1f77b4")
    axes[0, 1].set_title("mean forgetting")
    axes[0, 1].tick_params(axis="x", rotation=45, labelsize=7)
    # (c) plasticity-stability scatter
    for m in labels:
        x, _ = _agg_value(bundle, m, "plasticity_index")
        f, _ = _agg_value(bundle, m, "mean_forgetting")
        axes[1, 0].scatter(x, 1 - f,
                           color="#d62728" if m == bundle.get("pnn_label") else "#1f77b4")
    axes[1, 0].set_xlabel("plasticity")
    axes[1, 0].set_ylabel("stability")
    axes[1, 0].set_title("plasticity vs stability")
    # (d) P distribution
    pnn = bundle.get("pnn_label")
    sample = bundle["methods"][pnn]["runs"][0].get("P_sample") if pnn else None
    if sample:
        axes[1, 1].hist(sample, bins=20, range=(0, 1), color="#9467bd")
    axes[1, 1].set_title("P_i distribution (PNN)")
    fig.suptitle("Figure 14 — Summary dashboard")
    _save(fig, outdir, "fig14_dashboard")


ALL_FIGURES = [
    fig01_method_overview, fig02_maturation_mechanism, fig03_accuracy_over_tasks,
    fig04_forgetting_comparison, fig05_weight_drift, fig06_representation_drift,
    fig07_pi_distribution, fig08_closure_dynamics, fig09_error_closure_trigger,
    fig10_plasticity_stability, fig11_ablation_heatmap, fig12_baseline_comparison,
    fig13_reopening, fig14_dashboard,
]


def make_all_figures(bundle, outdir) -> List[str]:
    outdir = Path(outdir)
    made = []
    for fn in ALL_FIGURES:
        try:
            fn(bundle, outdir)
            made.append(fn.__name__)
        except Exception as e:  # one bad figure must not kill the batch
            warnings.warn(f"{fn.__name__} failed: {e}", RuntimeWarning)
    return made
