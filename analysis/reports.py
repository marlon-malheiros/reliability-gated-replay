"""Generate the Markdown final report and executive summary from the bundle."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _val(bundle, label, key, pct=False, default="-"):
    a = bundle["methods"].get(label, {}).get("agg", {}).get(key)
    if not a:
        return default
    return f"{100*a['mean']:.1f}%" if pct else f"{a['mean']:.3f}"


def _methods(bundle, kinds):
    return sorted([k for k, v in bundle["methods"].items() if v["kind"] in kinds])


def _best_baseline(bundle, key="average_accuracy"):
    best, best_v = None, -1e9
    for m in _methods(bundle, {"baseline"}):
        a = bundle["methods"][m]["agg"].get(key)
        if a and a["mean"] > best_v:
            best, best_v = m, a["mean"]
    return best


def _primary_pnn_name(bundle):
    return bundle.get("pnn_label") or "PNN"


def write_executive_summary(bundle, config, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pnn = bundle.get("pnn_label")
    bb = _best_baseline(bundle)
    note = bundle.get("note", "")
    lines = [
        "# Executive Summary — PNN-Inspired Weighted Consolidation for Continual Learning",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} · dataset: {bundle.get('dataset','?')} · "
        f"seeds: {bundle.get('n_seeds','?')}_",
        "",
        f"> {note}" if note else "",
        "",
        "## Core question",
        "Can a PNN-inspired maturation signal serve as a replay-free parameter-importance "
        "weight for quadratic consolidation, reducing catastrophic forgetting under "
        "sequential task interference?",
        "",
        "## Headline numbers",
        "",
        f"| metric | primary PNN ({_primary_pnn_name(bundle)}) | best baseline |",
        "| --- | --- | --- |",
        f"| avg accuracy | {_val(bundle, pnn,'average_accuracy',pct=True)} | "
        f"{_val(bundle, bb,'average_accuracy',pct=True)} ({bb}) |",
        f"| mean forgetting | {_val(bundle, pnn,'mean_forgetting',pct=True)} | "
        f"{_val(bundle, bb,'mean_forgetting',pct=True)} |",
        f"| backward transfer | {_val(bundle, pnn,'backward_transfer')} | "
        f"{_val(bundle, bb,'backward_transfer')} |",
        f"| representational drift | {_val(bundle, pnn,'representational_drift')} | "
        f"{_val(bundle, bb,'representational_drift')} |",
        "",
        "## Mechanism (sanity)",
        f"- final mean consolidation P̄ = {_val(bundle, pnn,'mean_P')}, "
        f"protected fraction = {_val(bundle, pnn,'frac_consolidated')}",
        f"- corr(P, importance) = {_val(bundle, pnn,'corr_P_importance')} "
        "(consolidation tracks parameter importance)",
        "- PNN-anchor uses P as an importance weight, not as a direct claim that "
        "gradient gating alone solves forgetting.",
        "- cost and persistent-memory estimates: see Tables 9–10.",
        "",
        "## Caveat",
        "These numbers come from the run in `results/`. A short smoke run (few seeds / epochs / "
        "subset, possibly synthetic data) is a **pipeline demonstration**, not a scientific result. "
        "Run `configs/full_suite.yaml` for paper-grade numbers.",
        "",
    ]
    path = outdir / "executive_summary.md"
    path.write_text("\n".join(l for l in lines if l is not None))
    return path


def _method_table(bundle, labels: List[str]) -> List[str]:
    rows = ["| method | avg acc | forgetting | BWT | FWT |",
            "| --- | --- | --- | --- | --- |"]
    for m in labels:
        rows.append(
            f"| {m} | {_val(bundle,m,'average_accuracy',pct=True)} | "
            f"{_val(bundle,m,'mean_forgetting',pct=True)} | "
            f"{_val(bundle,m,'backward_transfer')} | {_val(bundle,m,'forward_transfer')} |"
        )
    return rows


def write_final_report(bundle, config, outdir: Path, figures_rel="../figures",
                       tables_rel="../tables") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pnn = bundle.get("pnn_label")
    bb = _best_baseline(bundle)
    L: List[str] = []
    L += ["# PNN-Inspired Weighted Consolidation for Continual Learning — Final Report", ""]
    L += [f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Dataset: {bundle.get('dataset','?')}; "
          f"seeds: {bundle.get('n_seeds','?')}._", ""]
    if bundle.get("note"):
        L += [f"> **Scope note.** {bundle['note']}", ""]

    L += ["## Abstract",
          "We test whether a biologically inspired perineuronal-net (PNN) maturation signal can "
          "act as a replay-free parameter-importance weight for quadratic consolidation. The "
          "central positive mechanism is P-weighted anchoring to previous task weights; "
          "gradient gating is treated as a control unless it independently improves retention. "
          "This report summarizes the selected continual-learning benchmark and method set from "
          "the run directory.", ""]

    L += ["## Introduction",
          "Continual learning suffers from catastrophic forgetting. Biological brains retain "
          "memories partly through PNNs that mature after critical periods, stabilizing synapses. "
          "We translate this into a maturation-derived consolidation variable P∈[0,1] per "
          "parameter and test whether it can weight a replay-free anchor penalty.", ""]

    L += ["## Biological motivation — PNNs and critical periods",
          "Mature PNNs stabilize circuit state and reduce drift. In this machine-learning "
          "analogue, that biological idea is used as a structured consolidation signal rather "
          "than a direct biological simulation.", ""]

    L += ["## Methods — mathematical formulation",
          "- Maturation: `P ← clip(P + α·importance·closure_signal, 0, 1)`",
          "- Closure gate: `closure_signal = σ(γ[(θ_E−E_t)+(θ_ΔE−ΔE_t)+(θ_Var−Var_E_t)])`",
          "- Optional gradient gating control: `g ← g·exp(−βP)` or effective `lr·(1−P)`",
          "- Primary positive mechanism: `L_anchor = 0.5·λ·Σ_i P_i(θ_i−θ_i*)²`",
          "- Reopening: if `E_t>θ_reopen` for M epochs, `P ← P·(1−ρ)`",
          "Importance ∈ {magnitude, gradient, gradient-variance, Fisher, hybrid}; consolidation "
          "may be parameter- or layer-wise. See Figures 1–2.", ""]

    L += ["## Datasets & experiments",
          f"Dataset in this run: {bundle.get('dataset','?')}. Models and methods are read from "
          "the experiment manifest. See Tables 1–3.", ""]

    L += ["## Baselines", "", *_method_table(bundle, _methods(bundle, {"baseline"})), "",
          "(See Table 4.)", ""]
    L += ["## PNN results", "", *_method_table(bundle, _methods(bundle, {"pnn"})), "",
          f"Final mean P̄ = {_val(bundle,pnn,'mean_P')}, protected fraction "
          f"{_val(bundle,pnn,'frac_consolidated')}, corr(P, importance) "
          f"{_val(bundle,pnn,'corr_P_importance')}. See Table 5 and Figures 7–9, 13.", ""]

    abls = _methods(bundle, {"ablation"})
    L += ["## Ablations", ""]
    if abls:
        L += [*_method_table(bundle, abls), "", "(See Table 6, Figure 11.)", ""]
    else:
        L += ["No ablation runs were included in this result set "
              "(enable them in the experiment config). See `ablations/registry.py`.", ""]

    L += ["## Results summary",
          f"On {bundle.get('dataset','?')}, PNN reaches avg accuracy "
          f"{_val(bundle,pnn,'average_accuracy',pct=True)} with forgetting "
          f"{_val(bundle,pnn,'mean_forgetting',pct=True)}, vs the best baseline {bb} "
          f"({_val(bundle,bb,'average_accuracy',pct=True)} / "
          f"{_val(bundle,bb,'mean_forgetting',pct=True)}). See Figures 3–6, 10, 12, 14.", ""]

    L += ["## Efficiency and storage",
          "Replay stores raw examples; PNN-anchor stores parameter-scale consolidation state "
          "(P, gradient statistics, and an anchor snapshot) without retaining raw training "
          "inputs. Tables 9–10 report runtime, peak memory, inference cost, persistent method "
          "state, and raw-data storage estimates. The efficiency claim should therefore be "
          "phrased as replay-free/no-raw-example consolidation unless the table shows a clear "
          "absolute memory win for the specific benchmark.", ""]

    L += ["## Statistical analysis",
          "Paired t-test and Wilcoxon signed-rank with Holm–Bonferroni correction and effect "
          "sizes (Cohen's d, rank-biserial) over seeds — Tables 7–8. With few seeds these are "
          "indicative only.", ""]

    L += ["## Discussion",
          "If P-weighted anchoring concentrates stability on useful parameters while leaving "
          "enough capacity plastic, PNN-inspired consolidation can trade off stability and "
          "plasticity without storing raw examples. The current evidence supports weighted "
          "anchoring as the operative mechanism; gradient-only PNN variants should be read as "
          "negative controls when they track Adam-like forgetting.", ""]
    L += ["## Limitations",
          "- The method is close to EWC/SI/L2 in mathematical form; novelty depends on the "
          "maturation-derived P weighting and mechanism controls.",
          "- Replay may still win average accuracy on harder settings even when PNN-anchor "
          "matches replay-level forgetting.",
          "- `lr_scale` gating is approximate under Adam (moment normalization).",
          "- Empirical Fisher (EWC) and the SI path integral use the regularized gradient.", ""]
    L += ["## Future work",
          "Permuted-MNIST and Split-CIFAR; PNN+Replay and PNN+Sleep three-way comparison at scale; "
          "per-neuron (rather than per-weight) consolidation closer to the biology.", ""]
    L += ["## Conclusion",
          "PNN-inspired maturation can provide a competitive replay-free importance signal for "
          "quadratic consolidation. The strongest current claim is weighted anchoring, not "
          "gradient gating as a standalone solution.", ""]

    path = outdir / "final_report.md"
    path.write_text("\n".join(L))
    return path
