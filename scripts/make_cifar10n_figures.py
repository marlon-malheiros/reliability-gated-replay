#!/usr/bin/env python
"""CIFAR-10N (real human-label noise) figures A/B/C for the Neural Networks submission.

Reads only ``csv/final_metrics.csv`` (no runtime objects). Saved WITHOUT titles
(added in the manuscript) as both .pdf and .png into ``paper/rgr/figs/``.

  Figure A  Power--safety position on CIFAR-10N: accuracy gain over ER (power, y)
            vs mean gate separation (safety, x). The separation<0 "inversion zone"
            is shaded; on real noise every method sits OUTSIDE it (the key finding).
  Figure B  Gate separation vs noise condition (aggregate ~9% vs worst ~40%).
  Figure C  Buffer purity vs method, grouped by label set.

    python scripts/make_cifar10n_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "rgr" / "figs"

ORDER = ["er", "derpp", "gate_loss", "gate_conf", "gate_coteach", "oracle"]
COLORS = {"er": "#7f7f7f", "derpp": "#ff7f0e", "gate_loss": "#d62728",
          "gate_conf": "#1f77b4", "gate_coteach": "#aec7e8", "oracle": "#000000"}
DISP = {"er": "ER", "derpp": "DER++", "gate_loss": "Small-loss",
        "gate_conf": "Confidence", "gate_coteach": "Co-teach", "oracle": "Oracle"}
# label set -> (column condition, display, approx noise)
VARIANTS = [("c10n_aggre", "aggregate (9%)"), ("c10n_worse", "worst (40%)")]
GATES = ["gate_loss", "gate_conf", "gate_coteach"]
BASE_FONT = 16
AXIS_FONT = 16
TICK_FONT = 14
LEGEND_FONT = 14
PANEL_FONT = 16
NOTE_FONT = 13


def _save(fig, name: str, figs: Path, dpi: int = 150):
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"{name}.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(figs / f"{name}.png", dpi=max(dpi, 600), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def _present(methods):
    return [m for m in ORDER if m in methods]


def _journal_axes(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.2")
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONT, width=0.8, colors="0.2")
    ax.grid(color="0.92", linewidth=0.55)


def _panel_label(ax, label: str):
    ax.text(
        -0.055, 1.025, label, transform=ax.transAxes, fontsize=PANEL_FONT,
        fontweight="bold", ha="left", va="bottom", clip_on=False, zorder=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5)
    )


def fig_a(d, out: Path):  # power-safety position
    fig, ax = plt.subplots(figsize=(14.2, 6.4))
    fig.subplots_adjust(left=0.10, right=0.62, bottom=0.16, top=0.97)
    fig.patch.set_facecolor("white")
    _journal_axes(ax)
    palette = {
        "er": "#7A7A7A",
        "derpp": "#CC8963",
        "gate_loss": "#5B8E7D",
        "gate_conf": "#4C78A8",
        "gate_coteach": "#9C89B8",
        "oracle": "#1B1B1B",
    }
    er = d[d.method == "er"].groupby("condition").average_accuracy.mean()
    method_handles = []
    seen_methods = set()
    for cond, disp in VARIANTS:
        s = d[d.condition == cond]
        for m in _present(s.method.unique()):
            ms = s[s.method == m]
            sep = ms.gate_separation.mean()
            gain = ms.average_accuracy.mean() - er.get(cond, np.nan)
            if np.isnan(sep):           # ER/DER++ have no gate -> plot at sep=0 marker
                sep = 0.0
            marker = "o" if cond == "c10n_aggre" else "^"
            face = palette.get(m, "#666666")
            edge = "white" if cond == "c10n_aggre" else face
            fill = face if cond == "c10n_aggre" else "none"
            size = 220 if m == "oracle" else 165
            ax.scatter(
                sep, gain, s=size, marker=marker, facecolors=fill, edgecolors=edge,
                linewidth=1.6, zorder=4 if m == "oracle" else 3
            )
            if m not in seen_methods:
                method_handles.append(
                    plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                               markerfacecolor=face, markeredgecolor=face,
                               label=DISP.get(m, m))
                )
                seen_methods.add(m)
    ax.margins(x=0.10, y=0.12)
    x_left, _ = ax.get_xlim()
    ax.axvspan(x_left, 0, color="#C44E52", alpha=0.035, zorder=0)
    ax.text(
        0.03, 0.95, "Inversion zone\n(separation < 0)", transform=ax.transAxes,
        fontsize=NOTE_FONT, color="#8C3B3F", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9)
    )
    ax.axhline(0, color="0.35", lw=0.9, ls="--")
    ax.axvline(0, color="0.20", lw=0.9)
    ax.set_xlabel("Mean gate separation (higher is safer)", fontsize=AXIS_FONT)
    ax.set_ylabel("Accuracy gain over ER", fontsize=AXIS_FONT)
    condition_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=10, markerfacecolor="0.45",
                   markeredgecolor="white", label="aggregate (9%)"),
        plt.Line2D([], [], marker="^", linestyle="", markersize=10, markerfacecolor="none",
                   markeredgecolor="0.35", label="worst (40%)"),
    ]
    fig.legend(
        handles=condition_handles, frameon=False, fontsize=LEGEND_FONT,
        loc="upper left", bbox_to_anchor=(0.72, 0.96), borderaxespad=0.0,
        title="Label set", title_fontsize=LEGEND_FONT
    )
    fig.legend(
        handles=method_handles, frameon=False, fontsize=LEGEND_FONT, ncol=1,
        loc="upper left", bbox_to_anchor=(0.72, 0.58), borderaxespad=0.0,
        title="Method", title_fontsize=LEGEND_FONT
    )
    _save(fig, "figA_c10n_frontier", out, dpi=600)


def fig_b(d, out: Path):  # gate separation vs noise condition
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.78)
    gates = [g for g in GATES if g in set(d.method)]
    x = np.arange(len(VARIANTS)); w = 0.8 / max(len(gates), 1)
    for i, g in enumerate(gates):
        vals = [d[(d.condition == c) & (d.method == g)].gate_separation.mean()
                for c, _ in VARIANTS]
        ax.bar(
            x + i * w, vals, w, label=DISP.get(g, g),
            color=COLORS.get(g, "#333"), edgecolor="0.2", linewidth=0.5
        )
    _journal_axes(ax)
    ax.axhline(0, color="0.15", lw=0.9)
    ax.set_xticks(x + w * (len(gates) - 1) / 2)
    ax.set_xticklabels([disp for _, disp in VARIANTS], fontsize=TICK_FONT)
    ax.set_ylabel("Gate separation (clean - mislabeled)", fontsize=AXIS_FONT)
    ax.set_xlabel("CIFAR-10N label set", fontsize=AXIS_FONT)
    ax.legend(fontsize=LEGEND_FONT, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.25), ncol=max(1, len(gates)))
    _panel_label(ax, "(a)")
    _save(fig, "figB_c10n_separation", out, dpi=600)


def fig_c(d, out: Path):  # buffer purity vs method
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.76)
    methods = _present(d.method.unique())
    x = np.arange(len(methods)); w = 0.38
    for j, (cond, disp) in enumerate(VARIANTS):
        vals = [d[(d.condition == cond) & (d.method == m)].buffer_purity.mean() for m in methods]
        ax.bar(x + j * w, vals, w, label=disp,
               color=[COLORS.get(m, "#333") for m in methods],
               alpha=0.65 if j == 0 else 1.0, edgecolor="0.2", linewidth=0.5)
    _journal_axes(ax)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([DISP.get(m, m) for m in methods], rotation=25, ha="right", fontsize=TICK_FONT)
    ax.set_ylabel("Buffer purity (fraction clean)", fontsize=AXIS_FONT)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=LEGEND_FONT, title="Label set", title_fontsize=LEGEND_FONT,
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2)
    _panel_label(ax, "(b)")
    _save(fig, "figC_c10n_purity", out, dpi=600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "results/neural_networks_submission/csv/final_metrics.csv"))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    out = Path(args.output)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": BASE_FONT,
        "axes.labelsize": AXIS_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
        "legend.fontsize": LEGEND_FONT,
        "lines.linewidth": 2.1,
        "lines.markersize": 6.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    d = pd.read_csv(args.input)
    d = d[d.benchmark == "cifar10n"]
    if d.empty:
        print("no CIFAR-10N rows yet — skipping."); return
    n = d.groupby(["method", "condition"]).size().max()
    print(f"CIFAR-10N figures from {len(d)} rows (max {n} seeds/cell), "
          f"methods={sorted(d.method.unique())}")
    fig_a(d, out); fig_b(d, out); fig_c(d, out)


if __name__ == "__main__":
    main()
