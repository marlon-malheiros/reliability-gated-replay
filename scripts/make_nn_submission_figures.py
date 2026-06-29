#!/usr/bin/env python
"""Generate the submission figures (1-7) from the aggregated CSVs.

Every figure reads only ``--input/*.csv`` (no hidden runtime objects) and is saved
as both ``.png`` and ``.pdf`` WITHOUT a title (titles are added in the manuscript).
Figures degrade gracefully when a benchmark has no rows yet (partial grids).

By default the script also mirrors the outputs into ``paper/rgr/figs`` so the
compiled manuscript picks up refreshed assets immediately.

    python scripts/make_nn_submission_figures.py \
        --input results/neural_networks_submission/csv \
        --output results/neural_networks_submission/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path
from matplotlib.transforms import Bbox

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_FIGS = ROOT / "paper" / "rgr" / "figs"
SAVE_MIRRORS: tuple[Path, ...] = ()
PNG_DPI = 600
BASE_FONT = 13
AXIS_FONT = 14
TICK_FONT = 13
LEGEND_FONT = 13
PANEL_FONT = 14
SMALL_FONT = 12

# canonical display order + colors for the gate family
ORDER = ["er", "derpp", "er_ace", "ewc", "si", "gate_loss", "gate_spr", "gate_conf",
         "gate_predstab", "gate_reprstab", "gate_teacher", "gate_coteach", "oracle"]
COLORS = {
    "er": "#7f7f7f", "derpp": "#ff7f0e", "er_ace": "#bcbd22", "ewc": "#8c564b", "si": "#c49c94",
    "gate_loss": "#d62728", "gate_spr": "#e377c2", "gate_conf": "#1f77b4",
    "gate_predstab": "#2ca02c", "gate_reprstab": "#9467bd", "gate_teacher": "#17becf",
    "gate_coteach": "#aec7e8", "oracle": "#000000",
}
LABELFREE = {"gate_conf", "gate_predstab", "gate_reprstab"}
DISPLAY = {
    "er": "ER",
    "derpp": "DER++",
    "er_ace": "ER-ACE",
    "ewc": "EWC",
    "si": "SI",
    "gate_loss": "Small-loss",
    "gate_spr": "SPR gate",
    "gate_conf": "Confidence",
    "gate_predstab": "Pred. stability",
    "gate_reprstab": "Repr. stability",
    "gate_teacher": "Teacher",
    "gate_coteach": "Co-teach",
    "oracle": "Oracle",
}


def _read(path: Path, cols=None) -> pd.DataFrame:
    """Read a CSV, tolerating absent/empty files (partial pipelines)."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=cols or [])


def _display(method: str) -> str:
    return DISPLAY.get(method, method)


def _display_path(path: Path, root: Path) -> str:
    """Format a path relative to the repository when possible."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _save(fig, out: Path, name: str, dpi: int = 200):
    out = out.resolve()
    targets = []
    for target in (out, *SAVE_MIRRORS):
        resolved = target.resolve()
        if resolved not in targets:
            targets.append(resolved)
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / f"{name}.png", dpi=max(dpi, PNG_DPI), bbox_inches="tight")
        fig.savefig(target / f"{name}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    root = ROOT.resolve()
    extras = [_display_path(target, root) for target in targets[1:] if target != out]
    suffix = f" (mirrored to {', '.join(extras)})" if extras else ""
    print(f"  wrote {name}.png / .pdf -> {_display_path(out, root)}{suffix}")


def _label_bbox(text: str, anchor_px: tuple[float, float], offset_px: tuple[float, float]) -> Bbox:
    width = max(42, 6.4 * len(text))
    height = 14
    x0 = anchor_px[0] + offset_px[0]
    y0 = anchor_px[1] + offset_px[1] - height / 2
    return Bbox.from_bounds(x0, y0, width, height)


def _annotate_non_overlapping(ax, points, fontsize=SMALL_FONT):
    occupied: list[Bbox] = []
    offsets = [(6, 6), (6, 16), (6, -10), (12, 0), (-54, 6), (-54, -10), (12, 16), (-54, 16)]
    for x, y, label in sorted(points, key=lambda p: (p[0], p[1])):
        anchor_px = ax.transData.transform((x, y))
        chosen = offsets[-1]
        for cand in offsets:
            bbox = _label_bbox(label, anchor_px, cand)
            if not any(bbox.overlaps(prev) for prev in occupied):
                chosen = cand
                occupied.append(bbox)
                break
        else:
            occupied.append(_label_bbox(label, anchor_px, chosen))
        ha = "left" if chosen[0] >= 0 else "right"
        ax.annotate(label, (x, y), fontsize=fontsize, xytext=chosen,
                    textcoords="offset points", ha=ha, va="center")


def _journal_axes(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.2")
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONT, width=0.8, colors="0.2")
    ax.grid(color="0.92", linewidth=0.55)


def _panel_label(ax, label: str):
    ax.text(
        0.02, 0.98, label, transform=ax.transAxes, fontsize=PANEL_FONT,
        fontweight="bold", ha="left", va="top", clip_on=False, zorder=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5)
    )


def _mean(df, by, col):
    return df.groupby(by)[col].mean()


def _present(df, methods):
    return [m for m in ORDER if m in methods] + [m for m in methods if m not in ORDER]


# --------------------------------------------------------------------------- #
# Figure 1 -- method schematic (drawn, not data; no biological overclaim)
# --------------------------------------------------------------------------- #
def fig1_schematic(out: Path):
    fig, ax = plt.subplots(figsize=(13.8, 7.2), constrained_layout=True)
    ax.axis("off")
    edge = "#34383E"
    arrow = "#696D72"
    fills = ["#E8EEF3", "#DDE9D5", "#F2E3C6", "#D9E4F0", "#D8E8E1", "#E8D8DD"]

    def box(x, y, w, h, face, linewidth=1.35):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=face, edgecolor=edge, linewidth=linewidth
        )
        ax.add_patch(patch)
        return patch

    def connect(x0, x1, y, style="-|>", ls="-", lw=1.65):
        ax.annotate(
            "", xy=(x1, y), xytext=(x0, y),
            arrowprops=dict(arrowstyle=style, lw=lw, color=arrow, linestyle=ls)
        )

    # Biological row: compact functional analogy, deliberately not anatomical.
    top_y, top_h = 0.68, 0.15
    top_left, top_right, top_gap = 0.035, 0.985, 0.022
    top_labels = [
        ("Experience /\nactivity", fills[0]),
        ("Endogenous\nstabilization signal", fills[1]),
        ("PNN-associated\nconstraint", fills[2]),
        ("Durable\ncircuit state", fills[3]),
        ("Adaptive or maladaptive\nstabilization", fills[5]),
    ]
    top_w = (top_right - top_left - top_gap * 4) / 5
    ax.text(top_left, 0.895, "(a) Biological selective stabilization",
            fontsize=AXIS_FONT, fontweight="bold", ha="left", va="center")
    for i, (label, face) in enumerate(top_labels):
        x = top_left + i * (top_w + top_gap)
        box(x, top_y, top_w, top_h, face)
        ax.text(x + top_w / 2, top_y + top_h / 2, label, ha="center", va="center",
                fontsize=BASE_FONT, linespacing=1.12)
        if i < 4:
            connect(x + top_w + 0.004, x + top_w + top_gap - 0.004,
                    top_y + top_h / 2)
    ax.text(
        0.5, 0.635,
        "Functional correspondence: endogenous control determines which activity-dependent changes become durable.",
        fontsize=SMALL_FONT, color="0.30", ha="center", va="center"
    )

    # Artificial row: retain the mechanistic detail of the original figure.
    ax.text(top_left, 0.585, "(b) Artificial reliability-gated replay",
            fontsize=AXIS_FONT, fontweight="bold", ha="left", va="center")
    y, h, left, right, gap = 0.145, 0.34, 0.025, 0.99, 0.014
    widths = [0.145, 0.145, 0.145, 0.145, 0.175, 0.14]
    scale = (right - left - gap * 5) / sum(widths)
    widths = [w * scale for w in widths]
    xs = [left]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + gap)
    titles = [
        "Noisy continual\nstream",
        "Reliability\nscore",
        "Stochastic\nadmission",
        "Replay\nbuffer",
        "Repeated gradient\ninfluence",
        "Diagnostics",
    ]
    for i, (x, w, title) in enumerate(zip(xs, widths, titles)):
        box(x, y, w, h, fills[i])
        ax.text(x + w / 2, y + h - 0.048, title, ha="center", va="center",
                fontsize=BASE_FONT, fontweight="bold", linespacing=1.12)
        if i < 5:
            connect(x + w + 0.003, x + w + gap - 0.003, y + h / 2)

    # Stream contents.
    x, w = xs[0], widths[0]
    ax.add_patch(plt.Circle((x + 0.032, y + 0.190), 0.014, color="#2B6A23"))
    ax.text(x + 0.055, y + 0.190, "clean label", fontsize=SMALL_FONT, va="center")
    ax.add_patch(plt.Circle((x + 0.032, y + 0.135), 0.014, color="#D62728"))
    ax.text(x + 0.055, y + 0.135, "corrupted label", fontsize=SMALL_FONT, va="center")
    ax.text(x + w / 2, y + 0.055, "non-stationary task stream",
            fontsize=SMALL_FONT, ha="center", va="center", color="0.25")

    # Reliability score.
    x, w = xs[1], widths[1]
    ax.text(x + w / 2, y + 0.195, "learner-derived signal", fontsize=SMALL_FONT,
            ha="center", va="center")
    ax.text(x + w / 2, y + 0.135, r"$s(x,\tilde{y};\,f_\theta)$",
            fontsize=AXIS_FONT, ha="center", va="center")
    ax.text(x + w / 2, y + 0.075, "loss, confidence,\nstability, or agreement",
            fontsize=SMALL_FONT, ha="center", va="center", linespacing=1.15, color="0.25")

    # Stochastic admission and rejection branch.
    x, w = xs[2], widths[2]
    ax.text(x + w / 2, y + 0.190, "admission probability", fontsize=SMALL_FONT,
            ha="center", va="center")
    ax.text(x + w / 2, y + 0.133, r"$g=\sigma\!\left(\gamma(s-\tau)\right)$",
            fontsize=AXIS_FONT, ha="center", va="center")
    ax.text(x + w / 2, y + 0.074, r"$u<g$  $\Rightarrow$  admit",
            fontsize=SMALL_FONT, ha="center", va="center")
    branch_x = x + w / 2
    ax.annotate(
        "", xy=(branch_x, y - 0.050), xytext=(branch_x, y + 0.005),
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color=arrow)
    )
    ax.text(branch_x, y - 0.067, "rejected samples fade",
            fontsize=SMALL_FONT, fontweight="bold", ha="center", va="top")

    # Replay buffer icon and oracle control.
    x, w = xs[3], widths[3]
    for j in range(4):
        px = x + w * 0.29 + j * 0.008
        py = y + 0.085 + j * 0.010
        ax.add_patch(plt.Polygon(
            [[px, py], [px + w * 0.38, py], [px + w * 0.43, py + 0.105],
             [px + w * 0.05, py + 0.105]],
            closed=True, facecolor="white", edgecolor="#17384C", linewidth=1.1
        ))
    ax.text(x + w / 2, y + 0.055, "reservoir memory\nstabilization substrate",
            fontsize=SMALL_FONT, ha="center", va="center", linespacing=1.10)
    oracle_y = y + h + 0.040
    oracle_w, oracle_h = w * 0.94, 0.074
    oracle_x = x + (w - oracle_w) / 2
    oracle = FancyBboxPatch(
        (oracle_x, oracle_y), oracle_w, oracle_h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="white", edgecolor="#6A6A6A", linewidth=1.15, linestyle="--"
    )
    ax.add_patch(oracle)
    ax.text(x + w / 2, oracle_y + oracle_h * 0.64, "Oracle clean-selector",
            fontsize=SMALL_FONT, fontweight="bold", ha="center", va="center")
    ax.text(x + w / 2, oracle_y + oracle_h * 0.25, "clean-only upper bound",
            fontsize=SMALL_FONT, ha="center", va="center")
    ax.annotate(
        "", xy=(x + w / 2, y + h + 0.004),
        xytext=(x + w / 2, oracle_y),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color=arrow, linestyle="--")
    )

    # Repeated update icon.
    x, w = xs[4], widths[4]
    ax.text(x + w / 2, y + 0.195, "current batch + replay batch",
            fontsize=SMALL_FONT, ha="center", va="center")
    layers = [
        [(0.22, 0.09), (0.22, 0.15), (0.22, 0.21)],
        [(0.50, 0.07), (0.50, 0.13), (0.50, 0.19), (0.50, 0.25)],
        [(0.78, 0.10), (0.78, 0.16), (0.78, 0.22)],
    ]
    for a, b in zip(layers[:-1], layers[1:]):
        for xa, ya in a:
            for xb, yb in b:
                ax.plot([x + w * xa, x + w * xb], [y + ya, y + yb],
                        color="0.35", lw=0.55, zorder=1)
    for layer in layers:
        for xn, yn in layer:
            ax.add_patch(plt.Circle(
                (x + w * xn, y + yn), 0.0085, facecolor="white",
                edgecolor="black", linewidth=1.0, zorder=2
            ))
    ax.text(x + w / 2, y + 0.038, "stored samples shape\nmultiple future updates",
            fontsize=SMALL_FONT, ha="center", va="center", linespacing=1.10)

    # Diagnostics.
    x, w = xs[5], widths[5]
    diagnostics = "Buffer purity\nGate separation\nInversion\nAccuracy\nForgetting"
    ax.text(x + 0.025, y + 0.190, diagnostics, fontsize=SMALL_FONT,
            ha="left", va="center", linespacing=1.28)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, out, "fig2_reliability_gated_replay")


# --------------------------------------------------------------------------- #
# Figure 2 -- consolidation sanity: gate helps on MNIST, not decisively on CIFAR
# --------------------------------------------------------------------------- #
def fig2_sanity(df, out: Path):
    methods = ["er", "derpp", "gate_loss", "gate_conf", "oracle"]
    panels = [("permuted_mnist", "Permuted-MNIST (MLP)"),
              ("seq_cifar10", "Seq-CIFAR-10 (ResNet-18)")]
    rate = 0.4
    fig, axes = plt.subplots(1, len(panels), figsize=(9.6, 3.8), sharey=True, constrained_layout=True)
    for ax, (bench, lab) in zip(axes, panels):
        sub = df[(df.benchmark == bench) & (np.isclose(df.noise_rate, rate))]
        ms = [m for m in methods if m in set(sub.method)]
        if not ms:
            ax.text(0.5, 0.5, f"no data\n({bench})", ha="center", va="center")
            ax.set_xlabel(lab); continue
        vals = [sub[sub.method == m].average_accuracy.mean() for m in ms]
        errs = [sub[sub.method == m].average_accuracy.std(ddof=0) for m in ms]
        ax.bar(range(len(ms)), vals, yerr=errs, color=[COLORS.get(m, "#999") for m in ms],
               capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([_display(m) for m in ms], rotation=28, ha="right")
        ax.set_title(lab, fontsize=AXIS_FONT)
        ax.grid(axis="y", color="0.9", linewidth=0.7)
    axes[0].set_ylabel(f"final avg accuracy @ {int(rate*100)}% sym noise")
    _save(fig, out, "fig2_consolidation_sanity")


# --------------------------------------------------------------------------- #
# Figure 3 -- main CIFAR benchmark: accuracy + forgetting vs noise
# --------------------------------------------------------------------------- #
def fig3_main_cifar(df, out: Path):
    benches = [b for b in ["split_cifar10", "seq_cifar10"] if b in set(df.benchmark)]
    if not benches:
        print("  fig3: no CIFAR data yet — skipping"); return
    methods = ["er", "derpp", "gate_loss", "gate_conf", "gate_spr", "oracle"]
    fig, axes = plt.subplots(
        len(benches), 2, figsize=(11.6, 4.7 * len(benches)), squeeze=False, constrained_layout=True
    )
    for r, bench in enumerate(benches):
        sub = df[df.benchmark == bench]
        for c, (col, ylab) in enumerate([("average_accuracy", "final avg accuracy"),
                                         ("mean_forgetting", "mean forgetting")]):
            ax = axes[r][c]
            for m in methods:
                d = sub[sub.method == m]
                if d.empty:
                    continue
                g = d.groupby("noise_rate")[col].agg(["mean", "std"]).sort_index()
                ax.errorbar(
                    g.index, g["mean"], yerr=g["std"].fillna(0), marker="o",
                    color=COLORS.get(m, "#999"), label=_display(m), capsize=3,
                    lw=2.0, ms=6.0
                )
            _journal_axes(ax)
            ax.set_xlabel("Symmetric noise rate", fontsize=16)
            ax.set_ylabel(f"{bench.replace('_', '-')}\n{ylab}", fontsize=16)
            ax.set_xticks(sorted(sub.noise_rate.dropna().unique()))
            _panel_label(ax, f"({chr(97 + r * 2 + c)})")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(6, len(labels)),
            frameon=False, bbox_to_anchor=(0.5, 1.005), fontsize=15
        )
    _save(fig, out, "fig3_main_cifar_benchmark")


# --------------------------------------------------------------------------- #
# Figure 4 -- buffer purity + gate alignment across methods/noise
# --------------------------------------------------------------------------- #
def fig4_buffer_gate(df, out: Path):
    bench = "split_mnist" if "split_mnist" in set(df.benchmark) else df.benchmark.iloc[0]
    sub = df[df.benchmark == bench]
    cols = [("buffer_purity", "buffer purity"),
            ("gate_corr", "corr(gate, correctness)"),
            ("gate_separation", "gate separation")]
    methods = [m for m in ["er", "gate_loss", "gate_spr", "gate_conf", "gate_predstab",
                           "gate_reprstab", "oracle"] if m in set(sub.method)]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), constrained_layout=True)
    for ax, (col, ylab) in zip(axes, cols):
        for m in methods:
            d = sub[sub.method == m]
            g = d.groupby("noise_rate")[col].mean().sort_index()
            if g.notna().any():
                ax.plot(g.index, g.values, marker="o", color=COLORS.get(m, "#999"),
                        label=_display(m), lw=2.1, ms=6.2)
        _journal_axes(ax)
        if col in {"gate_corr", "gate_separation"}:
            ax.axhline(0, color="0.15", lw=0.8)
        ax.set_xlabel("Noise rate", fontsize=16)
        ax.set_ylabel(f"{ylab}\n[{bench.replace('_', '-')}]", fontsize=16)
        ax.set_xticks(sorted(sub.noise_rate.dropna().unique()))
        if col == "gate_separation":
            ymin, ymax = ax.get_ylim()
            lim = max(abs(ymin), abs(ymax), 0.1)
            ax.set_ylim(-lim, lim)
    for label, ax in zip(["(a)", "(b)", "(c)"], axes):
        _panel_label(ax, label)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(7, len(labels)),
            frameon=False, bbox_to_anchor=(0.5, 1.005), fontsize=15
        )
    _save(fig, out, "fig4_buffer_purity_gate_alignment")


# --------------------------------------------------------------------------- #
# Figure 5 -- memorization inversion: gate_on_correct vs gate_on_wrong over time
# --------------------------------------------------------------------------- #
def fig5_inversion(gate_df, out: Path):
    # an inverting cell (few-class high noise) and a safe cell (many-class moderate)
    picks = [("split_mnist", "gate_loss", 0.6, "Split-MNIST · 60% (inverts)"),
             ("permuted_mnist", "gate_loss", 0.4, "Permuted-MNIST · 40% (safe)")]
    avail = [p for p in picks if not gate_df[(gate_df.benchmark == p[0]) &
             (gate_df.method == p[1]) & (np.isclose(gate_df.noise_rate, p[2]))].empty]
    if not avail:
        print("  fig5: no gate-history data — skipping"); return
    fig, axes = plt.subplots(
        1, len(avail), figsize=(6.0 * len(avail), 4.4), squeeze=False, constrained_layout=True
    )
    for ax, (bench, method, rate, lab) in zip(axes[0], avail):
        d = gate_df[(gate_df.benchmark == bench) & (gate_df.method == method) &
                    (np.isclose(gate_df.noise_rate, rate))]
        g = d.groupby("task")[["gate_on_correct", "gate_on_wrong", "gate_separation"]].mean()
        ax.plot(g.index, g["gate_on_correct"], marker="o", color="#2ca02c",
                label="Correct label", lw=2.1, ms=6.5)
        ax.plot(g.index, g["gate_on_wrong"], marker="s", color="#d62728",
                label="Mislabeled", lw=2.1, ms=6.2)
        # highlight tasks where separation < 0 (inversion)
        inv = g[g["gate_separation"] < 0]
        if not inv.empty:
            ax.scatter(inv.index, inv["gate_on_wrong"], s=140, facecolors="none",
                       edgecolors="black", lw=1.5, zorder=5, label="Inverted")
        _journal_axes(ax)
        ax.set_xlabel(f"Task index\n{lab}", fontsize=17)
        ax.set_ylabel("Mean gate value", fontsize=17)
        ax.set_ylim(0, 1)
    for label, ax in zip(["(a)", "(b)"], axes[0]):
        _panel_label(ax, label)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(3, len(labels)),
            frameon=False, bbox_to_anchor=(0.5, 1.005), fontsize=16
        )
    _save(fig, out, "fig5_memorization_inversion")


# --------------------------------------------------------------------------- #
# Figure 6 -- power-safety frontier
# --------------------------------------------------------------------------- #
def fig6_frontier(df, out: Path, benign=("permuted_mnist", 0.4),
                  inversion=("split_mnist", 0.6)):
    bb, br = benign; ib, ir = inversion
    bsub = df[(df.benchmark == bb) & (np.isclose(df.noise_rate, br))]
    isub = df[(df.benchmark == ib) & (np.isclose(df.noise_rate, ir))]
    if bsub.empty or isub.empty:
        print("  fig6: insufficient data for frontier — skipping"); return
    er_b = bsub[bsub.method == "er"].average_accuracy.mean()
    er_i = isub[isub.method == "er"].average_accuracy.mean()
    # canonical methods only (keep the manuscript figure uncluttered)
    methods = [m for m in ORDER if m in set(df.method) and m != "er"]
    fig, ax = plt.subplots(figsize=(9.6, 5.9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    _journal_axes(ax)
    palette = {
        "derpp": "#CC8963",
        "ewc": "#7A8FA3",
        "si": "#9A8C98",
        "gate_loss": "#5B8E7D",
        "gate_spr": "#4F6D7A",
        "gate_conf": "#7B9E87",
        "gate_predstab": "#809BCE",
        "gate_reprstab": "#A67C9B",
        "gate_teacher": "#8D99AE",
        "gate_coteach": "#BC6C25",
        "oracle": "#1B1B1B",
    }
    baseline_methods = {"derpp", "ewc", "si"}
    method_points = []
    for idx, m in enumerate(methods, start=1):
        b = bsub[bsub.method == m].average_accuracy.mean()
        i = isub[isub.method == m].average_accuracy.mean()
        if np.isnan(b) or np.isnan(i):
            continue
        benign_gain = b - er_b
        inversion_penalty = er_i - i
        if m == "oracle":
            marker, size, face, edge = "*", 320, palette[m], "#A37C27"
        elif m in baseline_methods:
            marker, size, face, edge = "s", 220, palette.get(m, "#999999"), "white"
        elif m in LABELFREE:
            marker, size, face, edge = "^", 230, palette.get(m, "#999999"), "white"
        else:
            marker, size, face, edge = "o", 220, palette.get(m, "#999999"), "white"
        ax.scatter(
            benign_gain, inversion_penalty, s=size, marker=marker, color=face,
            edgecolors=edge, linewidth=1.2, zorder=3
        )
        text_color = "white" if m in baseline_methods or m == "oracle" else "black"
        ax.text(
            benign_gain, inversion_penalty, str(idx), ha="center", va="center",
            fontsize=SMALL_FONT, color=text_color, fontweight="bold", zorder=4
        )
        method_points.append((idx, m))
    ax.axhline(0, color="0.35", lw=0.9, ls=":")
    ax.axvline(0, color="0.35", lw=0.9, ls=":")
    ax.set_xlabel("Benign gain over ER (Permuted-MNIST 40%)", fontsize=AXIS_FONT)
    ax.set_ylabel("High-noise penalty vs ER (Split-MNIST 60%; lower is safer)", fontsize=AXIS_FONT)
    ax.text(
        0.02, 0.03, "Safer direction (down)", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=SMALL_FONT, color="0.3"
    )
    category_handles = [
        plt.Line2D([], [], marker="s", markersize=9, linestyle="", markerfacecolor="#7A8FA3",
                   markeredgecolor="white", label="Baselines"),
        plt.Line2D([], [], marker="o", markersize=9, linestyle="", markerfacecolor="#5B8E7D",
                   markeredgecolor="white", label="Gated"),
        plt.Line2D([], [], marker="^", markersize=9, linestyle="", markerfacecolor="#809BCE",
                   markeredgecolor="white", label="Label-free"),
        plt.Line2D([], [], marker="*", markersize=12, linestyle="", markerfacecolor="#1B1B1B",
                   markeredgecolor="#A37C27", label="Oracle"),
    ]
    method_labels = [f"{idx}. {_display(m)}" for idx, m in method_points]
    blank_handles = [
        plt.Line2D([], [], linestyle="", marker=None, label=label)
        for label in method_labels
    ]
    legend_categories = ax.legend(
        handles=category_handles, frameon=False, fontsize=LEGEND_FONT,
        loc="upper left", bbox_to_anchor=(1.02, 1.00), borderaxespad=0.0
    )
    ax.add_artist(legend_categories)
    ax.legend(
        handles=blank_handles, frameon=False, fontsize=LEGEND_FONT, ncol=1,
        loc="upper left", bbox_to_anchor=(1.02, 0.56), borderaxespad=0.0,
        handlelength=0, handletextpad=0.0
    )
    _save(fig, out, "fig6_power_safety_frontier", dpi=600)


# --------------------------------------------------------------------------- #
# Figure 7 -- calibration: reliability diagram + ECE/Brier comparison
# --------------------------------------------------------------------------- #
def fig7_calibration(df, cal_df, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    ax = axes[0]
    # reliability diagram (a main CIFAR benchmark if present, else any)
    bench = next((b for b in ["seq_cifar10", "split_cifar10"] if b in set(cal_df.benchmark)),
                 (cal_df.benchmark.iloc[0] if not cal_df.empty else None))
    drew = False
    if bench is not None:
        sub = cal_df[cal_df.benchmark == bench]
        rate = sorted(sub.noise_rate.unique())[len(sub.noise_rate.unique()) // 2] if not sub.empty else None
        for m in ["er", "gate_loss", "gate_conf", "oracle"]:
            d = sub[(sub.method == m) & (np.isclose(sub.noise_rate, rate))]
            if d.empty:
                continue
            g = d.groupby("bin_center")[["bin_acc", "bin_conf"]].mean().dropna()
            if not g.empty:
                ax.plot(g["bin_conf"], g["bin_acc"], marker="o", color=COLORS.get(m, "#999"),
                        label=_display(m), lw=2.0, ms=5.8); drew = True
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        _journal_axes(ax)
        ax.set_xlabel(f"Confidence [{bench.replace('_', '-')}]", fontsize=AXIS_FONT)
        ax.set_ylabel("Accuracy", fontsize=AXIS_FONT)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if drew:
            ax.legend(fontsize=LEGEND_FONT, frameon=False, loc="upper left")
    if not drew:
        ax.text(0.5, 0.5, "no calibration-bin data\n(legacy MNIST runs omit it)",
                ha="center", va="center")
    # ECE / Brier bars on the same benchmark
    ax2 = axes[1]
    cb = df[df.benchmark == bench] if bench is not None else df
    methods = [m for m in ["er", "derpp", "gate_loss", "gate_conf", "oracle"] if m in set(cb.method)]
    if methods and cb.ece.notna().any():
        x = np.arange(len(methods)); wd = 0.38
        ece = [cb[cb.method == m].ece.mean() for m in methods]
        bri = [cb[cb.method == m].brier.mean() for m in methods]
        ax2.bar(x - wd / 2, ece, wd, label="ECE", color="#1f77b4")
        ax2.bar(x + wd / 2, bri, wd, label="Brier", color="#ff7f0e")
        ax2.set_xticks(x)
        ax2.set_xticklabels([_display(m) for m in methods], rotation=28, ha="right")
        _journal_axes(ax2)
        ax2.set_ylabel("Calibration error", fontsize=AXIS_FONT)
        ax2.legend(fontsize=LEGEND_FONT, frameon=False, loc="upper right")
    else:
        ax2.text(0.5, 0.5, "no ECE/Brier data", ha="center", va="center")
    _panel_label(ax, "(a)")
    _panel_label(ax2, "(b)")
    _save(fig, out, "fig7_calibration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "results/neural_networks_submission/csv"))
    ap.add_argument("--output", default=str(ROOT / "results/neural_networks_submission/figures"))
    ap.add_argument("--manuscript-output", default=str(MANUSCRIPT_FIGS))
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output)
    manuscript_out = Path(args.manuscript_output)
    global SAVE_MIRRORS
    SAVE_MIRRORS = () if manuscript_out == out else (manuscript_out,)
    final = _read(inp / "final_metrics.csv")
    gate = _read(inp / "gate_diagnostics.csv")
    cal = _read(inp / "calibration_bins.csv",
                ["benchmark", "method", "noise_rate", "bin_center", "bin_acc", "bin_conf"])
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": BASE_FONT,
        "axes.labelsize": AXIS_FONT,
        "axes.titlesize": AXIS_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
        "legend.fontsize": LEGEND_FONT,
        "lines.linewidth": 2.1,
        "lines.markersize": 6.2,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    print(f"figures from {len(final)} runs across benchmarks {sorted(set(final.benchmark))}")
    fig1_schematic(out)
    fig2_sanity(final, out)
    fig3_main_cifar(final, out)
    fig4_buffer_gate(final, out)
    fig5_inversion(gate, out)
    fig6_frontier(final, out)
    fig7_calibration(final, cal, out)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
